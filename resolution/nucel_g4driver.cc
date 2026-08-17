// nucel_g4driver -- ask GEANT4 ITSELF what its NUCLEAR ELASTIC channel
// (`hadElastic`) does at a fully specified step, per species.
//
// WHY THIS EXISTS
// ---------------
// The clean-propagation model (cf_propagation_test.model_phi) carries three
// noise channels -- ionization, multiple scattering, radiative -- and has NO
// nuclear elastic channel at all, while the simulation runs `hadElastic` for
// every hadron.  NOTES: the elastic/inelastic split (hadron_probe arms
// `elonly` / `inelonly`, 2026-08-17) showed the elastic half is essentially
// the WHOLE nuclear effect and is a pure SHAPE effect -- acceptance in
// `elonly` is 100.0000% for pi/K/p -- and that in `locx` it runs to 38-107
// sigma, roughly 10x the residual the all-corrections closure is left with.
// So this is the dominant missing channel for hadrons, and it has to be built
// from what Geant4 actually does rather than from a textbook elastic form.
//
// "FOLLOW THE G4 IMPLEMENTATION" IS FOUR IMPLEMENTATIONS, NOT ONE
// ---------------------------------------------------------------
// Read statically off the constructor CMS actually registers -- stock
// G4HadronElasticPhysics (SimG4Core/PhysicsLists/plugins/FTFPCMS_BERT_EMM.cc
// registers `new G4HadronElasticPhysics(ver)`), Geant4 v11.2.2:
//
//   proton   G4ChipsElasticModel      xs G4BGGNucleonElasticXS
//   pi+/pi-  G4ElasticHadrNucleusHE   xs G4BGGPionElasticXS
//   K+/K-    G4HadronElastic          xs G4CrossSectionElastic(GGHadronNucleus)
//   pbar     G4HadronElastic <100 MeV / G4AntiNuclElastic >100 MeV
//                                     xs G4CrossSectionElastic(AntiNuclNuclear)
//
// The kaons reach the generic model via G4HadronicBuilder::BuildElastic(
// G4HadParticles::GetKaons()) rather than an explicit per-species block, which
// is WHY they get G4HadronElastic and not something kaon-specific.  A single
// analytic elastic kernel would be wrong for three of the four species.
//
// THE MUON IS THE NULL.  It has no hadElastic at all -- measured, not assumed:
// the step census gives hadElastic == 0 primary steps for mu- in every arm.
// This driver REFUSES to run for a muon rather than silently inventing a
// deflection for a particle the simulation never scatters.
//
// WHAT IT PRODUCES
//   rate   -- the macroscopic elastic cross section Sigma(ekin, material) from
//             the species-correct cross-section dataset, i.e. the Poisson mean
//             per unit length.  Taken from the dataset, never parameterised.
//   kernel -- these models are SAMPLERS, not closed forms, so the angular
//             kernel is built from what they actually draw: ApplyYourself() is
//             called n times at the step conditions and the primary's
//             deflection is recorded.  Same method as the Urban sampler study.
//
// Nothing in the CMSSW or Geant4 source tree is written to.
//
// Usage:
//   nucel_g4driver --pdg -211 --Z 8 --A 16 --rho 9 --ekin 3030 --len 1.045 \
//                  --n 2000000 --seed 20260817 --out /path/prefix
//
// Writes <prefix>.rec (text: header, Sigma, mfp, expected collisions in the
// step, and the sampled angular moments) plus, unless --norays,
// <prefix>.theta.bin (N float64, rad -- the polar deflection of the PRIMARY
// per elastic collision) and <prefix>.eloss.bin (N float64, MeV -- the kinetic
// energy the PRIMARY loses to the nuclear recoil in that same collision).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>

#include "G4Material.hh"
#include "G4Element.hh"
#include "G4ElementVector.hh"
#include "G4DynamicParticle.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4PionMinus.hh"
#include "G4PionPlus.hh"
#include "G4KaonMinus.hh"
#include "G4KaonPlus.hh"
#include "G4Proton.hh"
#include "G4AntiProton.hh"
#include "G4MuonMinus.hh"
#include "G4MuonPlus.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

#include "G4HadronElastic.hh"
#include "G4ChipsElasticModel.hh"
#include "G4ElasticHadrNucleusHE.hh"
#include "G4AntiNuclElastic.hh"
#include "G4HadProjectile.hh"
#include "G4HadFinalState.hh"
#include "G4Nucleus.hh"
#include "G4NucleiProperties.hh"
#include "G4LorentzVector.hh"

#include "G4VCrossSectionDataSet.hh"
#include "G4CrossSectionDataStore.hh"
#include "G4CrossSectionElastic.hh"
#include "G4BGGNucleonElasticXS.hh"
#include "G4BGGPionElasticXS.hh"
#include "G4ComponentGGHadronNucleusXsc.hh"
#include "G4ComponentAntiNuclNuclearXS.hh"
#include "G4HadProcesses.hh"

#include "G4GenericIon.hh"
#include "G4IonTable.hh"
#include "G4ProcessManager.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4Region.hh"
#include "G4ProductionCuts.hh"
#include "G4ProductionCutsTable.hh"
#include "G4TransportationManager.hh"
#include "Randomize.hh"

namespace {

double argd(int argc, char** argv, const char* key, double def) {
  for (int i = 1; i + 1 < argc; ++i)
    if (!strcmp(argv[i], key)) return atof(argv[i + 1]);
  return def;
}
long argl(int argc, char** argv, const char* key, long def) {
  for (int i = 1; i + 1 < argc; ++i)
    if (!strcmp(argv[i], key)) return atol(argv[i + 1]);
  return def;
}
std::string args_(int argc, char** argv, const char* key, const char* def) {
  for (int i = 1; i + 1 < argc; ++i)
    if (!strcmp(argv[i], key)) return std::string(argv[i + 1]);
  return std::string(def);
}
bool argf(int argc, char** argv, const char* key) {
  for (int i = 1; i < argc; ++i)
    if (!strcmp(argv[i], key)) return true;
  return false;
}

G4ParticleDefinition* particleOf(int pdg) {
  switch (pdg) {
    case -211: return G4PionMinus::PionMinus();
    case 211: return G4PionPlus::PionPlus();
    case -321: return G4KaonMinus::KaonMinus();
    case 321: return G4KaonPlus::KaonPlus();
    case 2212: return G4Proton::Proton();
    case -2212: return G4AntiProton::AntiProton();
    case 13: return G4MuonMinus::MuonMinus();
    case -13: return G4MuonPlus::MuonPlus();
  }
  return nullptr;
}

double quantile(std::vector<double>& v, double q) {
  if (v.empty()) return 0.;
  const size_t k = std::min(v.size() - 1,
                            (size_t)std::llround(q * (double)(v.size() - 1)));
  std::nth_element(v.begin(), v.begin() + k, v.end());
  return v[k];
}

}  // namespace

int main(int argc, char** argv) {
  const int pdg = (int)argl(argc, argv, "--pdg", -211);
  const double Z = argd(argc, argv, "--Z", 8.0);
  const double A = argd(argc, argv, "--A", 16.0);
  const double rho = argd(argc, argv, "--rho", 9.0);
  const double ekin = argd(argc, argv, "--ekin", 3030.0);  // MeV, KINETIC
  const double len = argd(argc, argv, "--len", 1.045);     // mm
  const long n = argl(argc, argv, "--n", 2000000);
  const long seed = argl(argc, argv, "--seed", 20260817);
  const std::string out = args_(argc, argv, "--out", "nucel");
  const bool norays = argf(argc, argv, "--norays");

  const G4int iZ = (G4int)std::llround(Z);
  const G4int iA = (G4int)std::llround(A);
  G4ParticleDefinition* part = particleOf(pdg);
  if (part == nullptr) {
    fprintf(stderr, "nucel_g4driver: unknown pdg %d\n", pdg);
    return 2;
  }
  // THE NULL, ENFORCED IN THE TOOL.  G4HadronElasticPhysics registers
  // hadElastic for hadrons only; the muon has no such process, and the step
  // census confirms zero hadElastic steps for mu- in every arm.  Producing a
  // number here would be inventing physics.
  if (abs(pdg) == 13) {
    fprintf(stderr,
            "nucel_g4driver: the muon has NO hadElastic process -- there is "
            "nothing to sample.  This is the physics null; refusing.\n");
    return 3;
  }

  // ---------------------------------------------------------------- geometry
  // Two separate reasons this block exists, both of which cost time when
  // omitted:
  //  (a) EM/hadronic model initialisation reaches G4SafetyHelper and aborts
  //      with GeomNav0003 / "NULL world" unless a world volume is set on the
  //      tracking navigator.  A 1 m box that is never tracked through is
  //      enough (the same trap urban_g4driver.cc documents).
  //  (b) G4ElasticHadrNucleusHE::InitialiseModel() -- the PION model -- builds
  //      its per-Z tables by walking G4ProductionCutsTable, NOT by looking at
  //      the projectile.  A locally-constructed G4MaterialCutsCouple of the
  //      kind the other drivers pass to SampleSecondaries is invisible to it,
  //      so the couple has to be registered GLOBALLY or the pion tables come
  //      out empty and every sampled angle is zero.
  G4Material* mat = new G4Material("ToyLayerMat", Z, A * g / mole, rho * g / cm3);
  {
    G4Box* wb = new G4Box("World", 1. * m, 1. * m, 1. * m);
    G4LogicalVolume* wl = new G4LogicalVolume(wb, mat, "World");
    G4VPhysicalVolume* wp =
        new G4PVPlacement(nullptr, G4ThreeVector(), wl, "World", nullptr, false, 0);
    G4TransportationManager::GetTransportationManager()->SetWorldForTracking(wp);

    G4Region* reg = new G4Region("DefaultRegionForTheWorld");
    G4ProductionCuts* pc = new G4ProductionCuts();
    pc->SetProductionCut(1. * um);
    reg->SetProductionCuts(pc);
    reg->AddRootLogicalVolume(wl);
    G4ProductionCutsTable* tab = G4ProductionCutsTable::GetProductionCutsTable();
    tab->SetEnergyRange(990. * eV, 100. * TeV);
    tab->UpdateCoupleTable(wp);
  }

  // ------------------------------------------------------------- ion table
  // Every elastic model produces a RECOIL NUCLEUS as a secondary, so it calls
  // G4IonTable::CreateIon.  Outside a run manager that refuses with
  // "Can not create ions because GenericIon is not ready" -- issued as a
  // WARNING, after which ApplyYourself dereferences the null ion and the
  // process dumps core.  A warning that is really a fatal is exactly the kind
  // of thing that reads as a physics crash, so it is handled explicitly:
  // declare the particle table ready and give G4GenericIon a process manager,
  // which is all CreateIon actually checks.
  {
    G4ParticleTable* ptab = G4ParticleTable::GetParticleTable();
    ptab->SetReadiness();
    G4GenericIon* gi = G4GenericIon::Definition();
    if (gi->GetProcessManager() == nullptr) gi->SetProcessManager(new G4ProcessManager(gi));
    G4IonTable::GetIonTable()->CreateAllIon();
  }

  const double mass = part->GetPDGMass();
  const double etot = ekin + mass;
  const double plab = std::sqrt(ekin * (ekin + 2. * mass));  // MeV/c

  CLHEP::HepJamesRandom eng(seed);
  G4Random::setTheEngine(&eng);

  // ------------------------------------------------- the SPECIES-CORRECT pair
  // Mirrors G4HadronElasticPhysics::ConstructProcess exactly.  LEAKED ON
  // PURPOSE, as in radharm_g4driver: the destructors race static teardown and
  // would turn a completed run into a nonzero exit.
  G4HadronicInteraction* model = nullptr;
  G4VCrossSectionDataSet* xs = nullptr;
  std::string modelName, xsName;

  const double elimitAntiNuc = 100. * MeV;
  if (pdg == 2212) {
    model = new G4ChipsElasticModel();
    xs = new G4BGGNucleonElasticXS(part);
    modelName = "G4ChipsElasticModel";
    xsName = "G4BGGNucleonElasticXS";
  } else if (abs(pdg) == 211) {
    auto* he = new G4ElasticHadrNucleusHE();
    he->SetMaxEnergy(100. * TeV);
    model = he;
    xs = new G4BGGPionElasticXS(part);
    modelName = "G4ElasticHadrNucleusHE";
    xsName = "G4BGGPionElasticXS";
  } else if (abs(pdg) == 321) {
    auto* le = new G4HadronElastic();
    le->SetMaxEnergy(100. * TeV);
    model = le;
    // Use the registry exactly as G4HadronicBuilder::BuildElastic does, so the
    // model and the cross section share ONE component instance -- see the note
    // on the antiproton branch below.
    xs = G4HadProcesses::ElasticXS("Glauber-Gribov");
    modelName = "G4HadronElastic";
    xsName = "G4CrossSectionElastic(G4ComponentGGHadronNucleusXsc)";
  } else if (pdg == -2212) {
    // The physics list registers BOTH, split at 100 MeV.  Pick the one that
    // actually owns this energy rather than assuming the high-energy branch.
    // `--forcelhep` overrides that choice so the LOW-energy model can be driven
    // at high energy: the sim shows a 16x excess of sub-2-mrad pbar scatters
    // that G4AntiNuclElastic does not produce, and the obvious suspect is the
    // process picking the other registered model.
    const bool forcelhep = argf(argc, argv, "--forcelhep");
    if (ekin >= elimitAntiNuc && !forcelhep) {
      auto* an = new G4AntiNuclElastic();
      an->SetMinEnergy(elimitAntiNuc);
      an->SetMaxEnergy(100. * TeV);
      model = an;
      modelName = "G4AntiNuclElastic";
    } else {
      auto* le = new G4HadronElastic();
      le->SetMaxEnergy(forcelhep ? 100. * TeV : elimitAntiNuc + 0.1 * MeV);
      model = le;
      modelName = forcelhep ? "G4HadronElastic(FORCED at high energy)"
                            : "G4HadronElastic(lowE anti-nucleon branch)";
    }
    // MUST come from the registry, not a fresh instance.  G4AntiNuclElastic's
    // constructor does reg->GetComponentCrossSection("AntiAGlauber") and keeps
    // that pointer as its own `cs`; SampleInvariantT then calls
    // cs->GetAntiHadronNucleonTotCrSc() and drives Ref2/ceff2 -- i.e. the ANGULAR
    // SHAPE -- from it.  The physics list constructs the model first and then
    // takes the SAME component back out of the registry via
    // G4HadProcesses::ElasticXS("AntiAGlauber"), so model and cross section share
    // one initialised instance.  Building a second instance here gave the model
    // one component and the rate another.
    xs = G4HadProcesses::ElasticXS("AntiAGlauber");
    xsName = "G4CrossSectionElastic(G4ComponentAntiNuclNuclearXS)";
  } else {
    fprintf(stderr, "nucel_g4driver: no elastic assignment for pdg %d\n", pdg);
    return 2;
  }
  model->InitialiseModel();

  // ------------------------------------------------------------------- rate
  // Sigma is MACROSCOPIC (1/length): G4CrossSectionDataStore::GetCrossSection
  // (dp, material) is exactly what G4HadronicProcess inverts to get the mean
  // free path, so this is the process's own number, not a reconstruction.
  auto* store = new G4CrossSectionDataStore();
  store->AddDataSet(xs);
  store->BuildPhysicsTable(*part);

  G4DynamicParticle probe(part, G4ThreeVector(0., 0., 1.), ekin);
  const double sigma = store->GetCrossSection(&probe, mat);  // 1/mm
  const double mfp = (sigma > 0.) ? 1. / sigma : -1.;
  const double nexp = sigma * len;

  FILE* fr = fopen((out + ".rec").c_str(), "w");
  if (fr == nullptr) {
    fprintf(stderr, "nucel_g4driver: cannot open %s.rec\n", out.c_str());
    return 4;
  }
  fprintf(fr, "# pdg Z A rho ekin[MeV] len[mm] n seed mass[MeV] etot[MeV] plab[MeV]\n");
  fprintf(fr, "input %d %g %g %g %.17g %.17g %ld %ld %.17g %.17g %.17g\n",
          pdg, Z, A, rho, ekin, len, n, seed, mass, etot, plab);
  fprintf(fr, "model %s\n", modelName.c_str());
  fprintf(fr, "xs %s\n", xsName.c_str());
  // Sigma in 1/mm, mfp in mm, nexp dimensionless = the Poisson mean over --len
  fprintf(fr, "rate %.17g %.17g %.17g\n", sigma * mm, mfp / mm, nexp);

  // ------------------------------------------------- the COULOMB ADMIXTURE
  //
  // G4AntiNuclElastic is the ONLY one of the four elastic models that samples
  // Coulomb scattering inside SampleInvariantT: it draws Rutherford with
  // probability XsCoulomb/(XsCoulomb + XsElastHadronic) and the nuclear form
  // otherwise (G4AntiNuclElastic.cc:163-171).  That matters here because the
  // SIMULATION already carries single EM scattering off the nucleus as a
  // SEPARATE process, `CoulombScat`, which NOTES_HADRONS s0.4 leaves ON in
  // every arm -- including the `off` arm the model is judged against.  So the
  // Coulomb part of a pbar kernel built from ApplyYourself is already in the
  // baseline, and putting it in the channel too DOUBLE COUNTS it.
  //
  // This block recomputes G4's own CoulombProb from its own public helpers so
  // the size of that admixture is measured rather than argued.
  if (argf(argc, argv, "--coulomb")) {
    if (pdg != -2212) {
      fprintf(stderr, "--coulomb is only meaningful for the antiproton\n");
    } else {
      auto* an = static_cast<G4AntiNuclElastic*>(model);
      CLHEP::HepLorentzVector Pproj(0., 0., plab, std::sqrt(plab * plab + mass * mass));
      const G4double ctet1 = an->GetcosTeta1(plab, iA);
      const G4double energy = Pproj.e() - mass;
      const G4double TargMass = G4NucleiProperties::GetNuclearMass(iA, iZ);
      CLHEP::HepLorentzVector lv(0., 0., 0., TargMass);
      lv += Pproj;
      Pproj.boost(-lv.boostVector());
      const G4double ptot = Pproj.vect().mag();
      const G4double beta = an->CalculateParticleBeta(part, ptot);
      const G4double nz = an->CalculateZommerfeld(beta, part->GetPDGCharge(), Z);
      const G4double Am = an->CalculateAm(ptot, nz, Z);
      const G4double mevToBarn = 0.38938e+6;
      const G4double XsCoulomb = mevToBarn * (nz / ptot) * (nz / ptot) * CLHEP::pi
                                 * (1. + ctet1) / (1. + Am) / (1. + 2. * Am - ctet1);
      G4ComponentAntiNuclNuclearXS xsc;
      const G4double XsEl =
          xsc.GetElasticElementCrossSection(part, energy, iZ, (G4double)iA) / CLHEP::millibarn;
      const G4double prob = XsCoulomb / (XsCoulomb + XsEl);
      fprintf(fr, "# XsCoulomb[mb] XsElastHadronic[mb] CoulombProb\n");
      fprintf(fr, "coulomb %.17g %.17g %.17g\n", XsCoulomb, XsEl, prob);
      printf("  COULOMB ADMIXTURE: XsCoul=%.4g mb  XsEl=%.4g mb  "
             "P(coulomb)=%.4f  <-- already in the sim's separate CoulombScat\n",
             XsCoulomb, XsEl, prob);
    }
  }


  // ----------------------------------------------------------------- kernel
  std::vector<double> th, dE;
  if (!norays) {
    th.reserve(n);
    dE.reserve(n);
  }
  double s1 = 0., s2 = 0., sE1 = 0., sE2 = 0.;
  long nbad = 0;

  for (long i = 0; i < n; ++i) {
    // Fresh projectile AND fresh nucleus every call: ApplyYourself may modify
    // the target, and G4HadProjectile caches a to-lab rotation.
    G4DynamicParticle dp(part, G4ThreeVector(0., 0., 1.), ekin);
    G4HadProjectile proj(dp);
    G4Nucleus targ(iA, iZ);

    G4HadFinalState* fs = model->ApplyYourself(proj, targ);
    if (fs == nullptr) {
      ++nbad;
      continue;
    }
    const G4ThreeVector& d = fs->GetMomentumChange();
    double a = d.theta();                       // rad, wrt the incoming +z
    double e = ekin - fs->GetEnergyChange();    // MeV lost to the recoil
    fs->Clear();

    if (!std::isfinite(a) || a < 0.) {
      ++nbad;
      continue;
    }
    if (!std::isfinite(e)) e = 0.;

    s1 += a;
    s2 += a * a;
    sE1 += e;
    sE2 += e * e;
    if (!norays) {
      th.push_back(a);
      dE.push_back(e);
    }
  }

  const long ngood = n - nbad;
  const double inv = (ngood > 0) ? 1. / (double)ngood : 0.;
  const double mth = s1 * inv, mth2 = s2 * inv;
  const double mE = sE1 * inv, mE2 = sE2 * inv;

  fprintf(fr, "# ngood nbad <theta> <theta^2> rms_theta <dE> <dE^2> rms_dE  [rad, MeV]\n");
  fprintf(fr, "moments %ld %ld %.17g %.17g %.17g %.17g %.17g %.17g\n",
          ngood, nbad, mth, mth2, std::sqrt(std::max(0., mth2)),
          mE, mE2, std::sqrt(std::max(0., mE2)));

  // The transport moment the MS channel is normalised on, for a like-for-like
  // comparison against the Moliere/WentzelVI terms already in the model:
  // <1-cos> per collision, and the per-unit-length transport cross section.
  double s1mc = 0.;
  if (!norays)
    for (double a : th) s1mc += 1. - std::cos(a);
  const double m1mc = (!norays && ngood > 0) ? s1mc / (double)ngood : 0.;
  fprintf(fr, "# <1-cos(theta)> per collision, and sigma_tr = Sigma*<1-cos> [1/mm]\n");
  fprintf(fr, "transport %.17g %.17g\n", m1mc, sigma * mm * m1mc);

  if (!norays) {
    std::vector<double> q = th;
    fprintf(fr, "# theta quantiles: 0.5 0.9 0.99 0.999 0.9999 [rad]\n");
    fprintf(fr, "quant %.17g %.17g %.17g %.17g %.17g\n",
            quantile(q, 0.5), quantile(q, 0.9), quantile(q, 0.99),
            quantile(q, 0.999), quantile(q, 0.9999));

    FILE* fb = fopen((out + ".theta.bin").c_str(), "wb");
    if (fb != nullptr) {
      fwrite(th.data(), sizeof(double), th.size(), fb);
      fclose(fb);
    }
    FILE* fe = fopen((out + ".eloss.bin").c_str(), "wb");
    if (fe != nullptr) {
      fwrite(dE.data(), sizeof(double), dE.size(), fe);
      fclose(fe);
    }
  }
  fclose(fr);

  printf("nucel_g4driver: pdg %d  %s  Sigma=%.6g /mm  mfp=%.6g mm  "
         "nexp(%.4g mm)=%.6g  <theta>=%.6g rad  ngood=%ld nbad=%ld\n",
         pdg, modelName.c_str(), sigma * mm, mfp / mm, len, nexp, mth,
         ngood, nbad);
  // Deliberately no cleanup: see the leak note above.
  return 0;
}
