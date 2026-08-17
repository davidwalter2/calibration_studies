// radharm_g4driver -- ask GEANT4 ITSELF what its RADIATIVE (bremsstrahlung +
// pair production) channel does at a fully specified step, for ANY species.
//
// WHY THIS EXISTS
// ---------------
// Two different questions, one driver.
//
// (1) THE MUON, and whether the analytic radiative CF is HARMONISED with what
//     Geant4 samples.  The offline model's radiative term (cf_brems_exact) is
//     a compound Poisson whose intensity dN/dv is Geant4's own
//     ComputeDMicroscopicCrossSection, tabulated by
//     Geant4ePropagator::fillRadiativeSpectrum on a 48-point grid over
//     v in [1e-6, 1] and renormalized to each process's own
//     ComputeDEDXPerVolume.  It has NEVER been compared against what the
//     simulation actually draws.  It cannot be, from the CF alone: the
//     simulation splits the channel into a CONTINUOUS sub-cut part (no
//     fluctuation at all) and DISCRETE secondaries above the gamma production
//     cut, and the model has no such split.  `--sample` draws exactly what
//     G4VEnergyLossProcess draws -- Poisson(sigma(ekin, gcut, tmax) * L)
//     explicit emissions from SampleSecondaries, plus the restricted mean --
//     so the two distributions can be compared in units of the MC error.
//
// (2) THE HADRONS, which have NO radiative channel in the model AT ALL.
//     Geant4ePropagator::fillRadiativeSpectrum and computeRadiativeDEDX both
//     open with `if (abs(pdg) != 13) return;`, so the exported `radv` record
//     is identically zero for pi/K/p and the model CF has two channels, not
//     three -- while the SIM runs hBrems and hPairProd.  `--spectrum` runs
//     G4hBremsstrahlungModel / G4hPairProductionModel through the SAME
//     tabulation the propagator does for the muon, so the missing channel can
//     be built and sized rather than argued about.  Both hadron models derive
//     from the muon ones and only override the protected
//     ComputeDMicroscopicCrossSection, which is exposed here by the identical
//     trivial-subclass trick Geant4ePropagator.cc itself uses.
//
// Nothing in the CMSSW or Geant4 source tree is written to.
//
// Usage:
//   radharm_g4driver --pdg 13 --Z 8 --A 16 --rho 9 --ekin 3030 --len 1.045 \
//                    --gcut 0.00099 --spectrum --sample --n 2000000 \
//                    --seed 20260817 --out /path/prefix
//
// Writes <prefix>.rec (text: the header numbers and, with --spectrum, the
// 48-point dN/dv for brems and pair) and, with --sample, <prefix>.rad.bin
// (N float64, MeV, the TOTAL radiative energy the primary loses in the step).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#include "G4Material.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4ProductionCuts.hh"
#include "G4DynamicParticle.hh"
#include "G4ParticleDefinition.hh"
#include "G4DataVector.hh"
#include "G4Element.hh"
#include "G4ElementVector.hh"
#include "G4MuonMinus.hh"
#include "G4MuonPlus.hh"
#include "G4PionMinus.hh"
#include "G4PionPlus.hh"
#include "G4KaonMinus.hh"
#include "G4KaonPlus.hh"
#include "G4Proton.hh"
#include "G4AntiProton.hh"
#include "G4Electron.hh"
#include "G4Gamma.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4MuBremsstrahlungModel.hh"
#include "G4MuPairProductionModel.hh"
#include "G4hBremsstrahlungModel.hh"
#include "G4hPairProductionModel.hh"
#include "G4Poisson.hh"
#include "Randomize.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4TransportationManager.hh"

namespace {

// The propagator's own grid, quoted: Geant4ePropagator.h kNRadV / kRadVMin /
// kRadVMax, and Geant4ePropagator::radVGrid.  Kept identical so the driver's
// spectrum is directly comparable with the exported `radspecv`.
constexpr int kNRadV = 48;
constexpr double kRadVMin = 1e-6;
constexpr double kRadVMax = 1.0;

void radVGrid(double* v) {
  const double lo = std::log(kRadVMin), hi = std::log(kRadVMax);
  for (int i = 0; i < kNRadV; ++i)
    v[i] = std::exp(lo + (hi - lo) * double(i) / double(kNRadV - 1));
}

// ComputeDMicroscopicCrossSection is protected in all four models; the same
// trivial derived class Geant4ePropagator.cc uses exposes it.
struct MuBremProbe : public G4MuBremsstrahlungModel {
  explicit MuBremProbe(const G4ParticleDefinition* p) : G4MuBremsstrahlungModel(p) {}
  using G4MuBremsstrahlungModel::ComputeDMicroscopicCrossSection;
};
struct MuPairProbe : public G4MuPairProductionModel {
  explicit MuPairProbe(const G4ParticleDefinition* p) : G4MuPairProductionModel(p) {}
  using G4MuPairProductionModel::ComputeDMicroscopicCrossSection;
};
struct HBremProbe : public G4hBremsstrahlungModel {
  explicit HBremProbe(const G4ParticleDefinition* p) : G4hBremsstrahlungModel(p) {}
  using G4hBremsstrahlungModel::ComputeDMicroscopicCrossSection;
};
struct HPairProbe : public G4hPairProductionModel {
  explicit HPairProbe(const G4ParticleDefinition* p) : G4hPairProductionModel(p) {}
  using G4hPairProductionModel::ComputeDMicroscopicCrossSection;
};

double argd(int argc, char** argv, const char* key, double def) {
  for (int i = 1; i + 1 < argc; ++i)
    if (!strcmp(argv[i], key))
      return atof(argv[i + 1]);
  return def;
}
long argl(int argc, char** argv, const char* key, long def) {
  for (int i = 1; i + 1 < argc; ++i)
    if (!strcmp(argv[i], key))
      return atol(argv[i + 1]);
  return def;
}
std::string args(int argc, char** argv, const char* key, const char* def) {
  for (int i = 1; i + 1 < argc; ++i)
    if (!strcmp(argv[i], key))
      return std::string(argv[i + 1]);
  return std::string(def);
}
bool argf(int argc, char** argv, const char* key) {
  for (int i = 1; i < argc; ++i)
    if (!strcmp(argv[i], key))
      return true;
  return false;
}

G4ParticleDefinition* particleOf(int pdg) {
  switch (pdg) {
    case 13: return G4MuonMinus::MuonMinus();
    case -13: return G4MuonPlus::MuonPlus();
    case -211: return G4PionMinus::PionMinus();
    case 211: return G4PionPlus::PionPlus();
    case -321: return G4KaonMinus::KaonMinus();
    case 321: return G4KaonPlus::KaonPlus();
    case 2212: return G4Proton::Proton();
    case -2212: return G4AntiProton::AntiProton();
  }
  return nullptr;
}
}  // namespace

int main(int argc, char** argv) {
  const int pdg = (int)argl(argc, argv, "--pdg", 13);
  const double Z = argd(argc, argv, "--Z", 8.0);
  const double A = argd(argc, argv, "--A", 16.0);
  const double rho = argd(argc, argv, "--rho", 9.0);
  const double ekin = argd(argc, argv, "--ekin", 3030.0);   // MeV, KINETIC
  const double len = argd(argc, argv, "--len", 1.045);      // mm
  const double gcut = argd(argc, argv, "--gcut", 0.00099);  // MeV, gamma cut
  const long n = argl(argc, argv, "--n", 2000000);
  const long seed = argl(argc, argv, "--seed", 20260817);
  const std::string out = args(argc, argv, "--out", "radharm");
  const bool dospec = argf(argc, argv, "--spectrum");
  const bool dosamp = argf(argc, argv, "--sample");
  const bool doemit = argf(argc, argv, "--emit");

  G4Material* mat = new G4Material("ToyLayerMat", Z, A * g / mole, rho * g / cm3);
  {
    G4Box* wb = new G4Box("World", 1. * m, 1. * m, 1. * m);
    G4LogicalVolume* wl = new G4LogicalVolume(wb, mat, "World");
    G4VPhysicalVolume* wp =
        new G4PVPlacement(nullptr, G4ThreeVector(), wl, "World", nullptr, false, 0);
    G4TransportationManager::GetTransportationManager()->SetWorldForTracking(wp);
  }

  G4ParticleDefinition* part = particleOf(pdg);
  if (part == nullptr) {
    fprintf(stderr, "unknown pdg %d\n", pdg);
    return 2;
  }
  const bool ismu = (abs(pdg) == 13);
  const double mass = part->GetPDGMass();
  const double etot = ekin + mass;

  CLHEP::HepJamesRandom eng(seed);
  G4Random::setTheEngine(&eng);

  FILE* fr = fopen((out + ".rec").c_str(), "w");
  fprintf(fr, "# pdg Z A rho ekin[MeV] len[mm] gcut[MeV] n seed mass[MeV] etot[MeV]\n");
  fprintf(fr, "input %d %g %g %g %.17g %.17g %.17g %ld %ld %.17g %.17g\n",
          pdg, Z, A, rho, ekin, len, gcut, n, seed, mass, etot);

  // ---------------------------------------------------------------- SPECTRUM
  //
  // Byte for byte what Geant4ePropagator::fillRadiativeSpectrum does, except
  // that the species guard is not applied.  UNRESTRICTED cuts (DBL_MAX) --
  // that is what the propagator initialises the probes with, and it is why the
  // exported spectrum is the FULL one rather than a restricted one.
  if (dospec) {
    G4DataVector cuts(std::max<size_t>(G4Material::GetNumberOfMaterials(), 1), DBL_MAX);
    // LEAKED ON PURPOSE: the destructors deregister from G4LossTableManager
    // and race the static teardown at exit (the same abort urban_g4driver.cc
    // documents), which would return nonzero on a run whose output is complete.
    G4VEmModel* bremP = nullptr;
    G4VEmModel* pairP = nullptr;
    const G4ParticleDefinition* proto =
        ismu ? (const G4ParticleDefinition*)G4MuonPlus::MuonPlus() : part;
    if (ismu) {
      auto* b = new MuBremProbe(proto);
      auto* p = new MuPairProbe(proto);
      b->Initialise(proto, cuts);
      p->Initialise(proto, cuts);
      b->SetUseBaseMaterials(false);
      p->SetUseBaseMaterials(false);
      bremP = b;
      pairP = p;
    } else {
      auto* b = new HBremProbe(proto);
      auto* p = new HPairProbe(proto);
      b->Initialise(proto, cuts);
      p->Initialise(proto, cuts);
      b->SetUseBaseMaterials(false);
      p->SetUseBaseMaterials(false);
      bremP = b;
      pairP = p;
    }

    double v[kNRadV];
    radVGrid(v);
    const G4ElementVector* elems = mat->GetElementVector();
    const double* natoms = mat->GetVecNbOfAtomsPerVolume();
    const size_t nel = mat->GetNumberOfElements();

    std::vector<double> db(kNRadV, 0.), dp_(kNRadV, 0.);
    for (int i = 0; i < kNRadV; ++i) {
      const double eps = v[i] * etot;
      double sb = 0., sp = 0.;
      if (eps > 0. && eps < ekin) {
        for (size_t ie = 0; ie < nel; ++ie) {
          const double ZZ = (*elems)[ie]->GetZ();
          const double w = natoms[ie] * len * etot;
          if (ismu) {
            sb += w * ((MuBremProbe*)bremP)->ComputeDMicroscopicCrossSection(ekin, ZZ, eps);
            sp += w * ((MuPairProbe*)pairP)->ComputeDMicroscopicCrossSection(ekin, ZZ, eps);
          } else {
            sb += w * ((HBremProbe*)bremP)->ComputeDMicroscopicCrossSection(ekin, ZZ, eps);
            sp += w * ((HPairProbe*)pairP)->ComputeDMicroscopicCrossSection(ekin, ZZ, eps);
          }
        }
      }
      db[i] = (sb > 0. && std::isfinite(sb)) ? sb : 0.;
      dp_[i] = (sp > 0. && std::isfinite(sp)) ? sp : 0.;
    }
    // the two dE/dx values the offline normalization anchors on, from the SAME
    // model objects (unrestricted, i.e. the whole spectrum)
    const double dedxB = bremP->ComputeDEDXPerVolume(mat, part, ekin, ekin);
    const double dedxP = pairP->ComputeDEDXPerVolume(mat, part, ekin, ekin);
    fprintf(fr, "dedx brem %.17g pair %.17g   [MeV/mm, unrestricted]\n", dedxB, dedxP);
    fprintf(fr, "# i v dNdvBrem dNdvPair\n");
    for (int i = 0; i < kNRadV; ++i)
      fprintf(fr, "spec %d %.17g %.17g %.17g\n", i, v[i], db[i], dp_[i]);
  }

  // ------------------------------------------------------------------ SAMPLE
  //
  // What G4VEnergyLossProcess ACTUALLY does for a radiative process over one
  // step:
  //   * below the secondary production cut -- CONTINUOUS, no fluctuation:
  //     ComputeDEDXPerVolume(ekin, gcut) * L, subtracted deterministically;
  //   * above it -- DISCRETE emissions, Poisson(CrossSectionPerVolume(ekin,
  //     gcut, tmax) * L) of them, each sampled by the model's own
  //     SampleSecondaries.
  // The primary's total radiative loss over the step is the sum, and THAT is
  // the object the analytic compound-Poisson CF has to match.  The analytic CF
  // has no such split: it treats the WHOLE spectrum from v = 1e-6 as discrete
  // and fluctuating, which is the difference this measures.
  if (dosamp) {
    G4ProductionCuts* pc = new G4ProductionCuts();
    pc->SetProductionCut(gcut, G4ProductionCuts::GetIndex("gamma"));
    pc->SetProductionCut(gcut, G4ProductionCuts::GetIndex("e-"));
    pc->SetProductionCut(gcut, G4ProductionCuts::GetIndex("e+"));
    G4MaterialCutsCouple* couple = new G4MaterialCutsCouple(mat, pc);
    couple->SetIndex(0);

    G4DataVector cutv;
    cutv.push_back(gcut);
    G4VEmModel* brem = ismu ? (G4VEmModel*)new G4MuBremsstrahlungModel(part)
                            : (G4VEmModel*)new G4hBremsstrahlungModel(part);
    G4VEmModel* pair = ismu ? (G4VEmModel*)new G4MuPairProductionModel(part)
                            : (G4VEmModel*)new G4hPairProductionModel(part);
    brem->Initialise(part, cutv);
    pair->Initialise(part, cutv);
    brem->SetUseBaseMaterials(false);
    pair->SetUseBaseMaterials(false);

    G4DynamicParticle dp(part, G4ThreeVector(0., 0., 1.), ekin * MeV);

    const double dedxBr = brem->ComputeDEDXPerVolume(mat, part, ekin, gcut);
    const double dedxPr = pair->ComputeDEDXPerVolume(mat, part, ekin, gcut);
    const double sigB = brem->CrossSectionPerVolume(mat, part, ekin, gcut, ekin);
    const double sigP = pair->CrossSectionPerVolume(mat, part, ekin, gcut, ekin);
    const double subMean = (dedxBr + dedxPr) * len;
    const double lamB = sigB * len, lamP = sigP * len;
    fprintf(fr,
            "restricted dedxbrem %.17g dedxpair %.17g submean %.17g "
            "lambdabrem %.17g lambdapair %.17g\n",
            dedxBr, dedxPr, subMean, lamB, lamP);

    std::vector<double> buf(n);
    std::vector<G4DynamicParticle*> vdp;
    vdp.reserve(8);
    double c1 = 0., c2 = 0., nb = 0., np_ = 0., emax = 0.;
    for (long i = 0; i < n; ++i) {
      double hard = 0.;
      const G4int kb = (G4int)G4Poisson(lamB);
      for (G4int k = 0; k < kb; ++k) {
        vdp.clear();
        brem->SampleSecondaries(&vdp, couple, &dp, gcut, ekin);
        for (auto* d : vdp) {
          hard += d->GetKineticEnergy();
          delete d;
        }
      }
      const G4int kp = (G4int)G4Poisson(lamP);
      for (G4int k = 0; k < kp; ++k) {
        vdp.clear();
        pair->SampleSecondaries(&vdp, couple, &dp, gcut, ekin);
        for (auto* d : vdp) {
          hard += d->GetKineticEnergy();
          delete d;
        }
      }
      nb += kb;
      np_ += kp;
      const double e = subMean + hard;
      if (hard > emax)
        emax = hard;
      buf[i] = e;
      c1 += e;
      c2 += e * e;
    }
    const double m1 = c1 / n;
    fprintf(fr,
            "sample mean %.17g var %.17g nbrem %.17g npair %.17g maxhard %.17g n %ld\n",
            m1, c2 / n - m1 * m1, nb / (double)n, np_ / (double)n, emax, n);
    FILE* bf = fopen((out + ".rad.bin").c_str(), "wb");
    fwrite(buf.data(), sizeof(double), n, bf);
    fclose(bf);
  }

  // -------------------------------------------------------------------- EMIT
  //
  // THE STATISTICALLY USABLE FORM OF THE SAME TEST.  A radiative step has
  // lambda ~ 5e-5, so 99.995 % of `--sample` draws return the sub-cut mean and
  // nothing else: two million of them contain about a hundred emissions.  A
  // compound Poisson's centred log-CF factorises,
  //
  //     S(a) = lambda [ <e^{i a eps}> - 1 - i a <eps> ]
  //
  // so the RATE (lambda, from G4's own CrossSectionPerVolume) and the SHAPE
  // (the distribution of ONE emission, from G4's own SampleSecondaries)
  // separate exactly, and every sample is spent on the shape.  This mode draws
  // M emissions per process, forced -- which is precisely what the model's
  // dN/dv is supposed to be, so the two are directly comparable.
  //
  // The energy the PRIMARY loses is the secondary's TOTAL energy, not its
  // kinetic energy: a bremsstrahlung gamma is massless so the two agree, but a
  // produced pair costs 2 m_e of rest mass as well.
  if (doemit) {
    G4ProductionCuts* pc = new G4ProductionCuts();
    pc->SetProductionCut(gcut, G4ProductionCuts::GetIndex("gamma"));
    pc->SetProductionCut(gcut, G4ProductionCuts::GetIndex("e-"));
    pc->SetProductionCut(gcut, G4ProductionCuts::GetIndex("e+"));
    G4MaterialCutsCouple* couple = new G4MaterialCutsCouple(mat, pc);
    couple->SetIndex(0);

    G4DataVector cutv;
    cutv.push_back(gcut);
    G4VEmModel* brem = ismu ? (G4VEmModel*)new G4MuBremsstrahlungModel(part)
                            : (G4VEmModel*)new G4hBremsstrahlungModel(part);
    G4VEmModel* pair = ismu ? (G4VEmModel*)new G4MuPairProductionModel(part)
                            : (G4VEmModel*)new G4hPairProductionModel(part);
    brem->Initialise(part, cutv);
    pair->Initialise(part, cutv);
    brem->SetUseBaseMaterials(false);
    pair->SetUseBaseMaterials(false);

    G4DynamicParticle dp(part, G4ThreeVector(0., 0., 1.), ekin * MeV);

    const double dedxBr = brem->ComputeDEDXPerVolume(mat, part, ekin, gcut);
    const double dedxPr = pair->ComputeDEDXPerVolume(mat, part, ekin, gcut);
    const double sigB = brem->CrossSectionPerVolume(mat, part, ekin, gcut, ekin);
    const double sigP = pair->CrossSectionPerVolume(mat, part, ekin, gcut, ekin);
    fprintf(fr,
            "emitrates lambdabrem %.17g lambdapair %.17g subdedxbrem %.17g "
            "subdedxpair %.17g len %.17g\n",
            sigB * len, sigP * len, dedxBr, dedxPr, len);

    std::vector<G4DynamicParticle*> vdp;
    vdp.reserve(8);
    for (int which = 0; which < 2; ++which) {
      G4VEmModel* m = which ? pair : brem;
      const double lam = which ? sigP : sigB;
      if (lam <= 0.) {
        fprintf(fr, "emit %s n 0 mean 0 var 0   (zero rate)\n",
                which ? "pair" : "brem");
        continue;
      }
      std::vector<double> e(n);
      double s1 = 0., s2 = 0., emx = 0.;
      long nempty = 0;
      for (long i = 0; i < n; ++i) {
        vdp.clear();
        m->SampleSecondaries(&vdp, couple, &dp, gcut, ekin);
        double tot = 0.;
        for (auto* d : vdp) {
          tot += d->GetTotalEnergy();
          delete d;
        }
        if (vdp.empty())
          ++nempty;
        e[i] = tot;
        s1 += tot;
        s2 += tot * tot;
        if (tot > emx)
          emx = tot;
      }
      const double m1 = s1 / n;
      fprintf(fr, "emit %s n %ld mean %.17g var %.17g max %.17g empty %ld\n",
              which ? "pair" : "brem", n, m1, s2 / n - m1 * m1, emx, nempty);
      FILE* bf =
          fopen((out + (which ? ".pair.bin" : ".brem.bin")).c_str(), "wb");
      fwrite(e.data(), sizeof(double), n, bf);
      fclose(bf);
    }
  }

  fclose(fr);
  return 0;
}
