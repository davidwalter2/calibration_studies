// urban_g4driver -- drive Geant4's OWN ionization-straggling samplers at a
// single, fully specified step, and dump the samples.
//
// WHY THIS EXISTS
// ---------------
// The offline resolution model (cgf_saddlepoint.ioni_cgf_derivs) builds an
// analytic compound-Poisson characteristic function out of the per-step Urban
// record `ioniurbanv` = (regime, gsig2, a1, e1, a2, e2, a3, e0, tmax, scaling,
// cs) exported by G4UniversalFluctuationForExtrapolator::SampleFluctuations.
// That CF has only ever been checked NUMERICALLY (1e-14 against mpmath on the
// defining integral) -- i.e. the mathematics of the model was validated, never
// the claim that the model is what Geant4 draws from.
//
// This program closes that gap the only way that is not circular: it calls
// the REAL C++ samplers.
//
//   * G4UniversalFluctuationForExtrapolator::SampleFluctuations  -> the record
//     (the exact numbers the offline model is built from)
//   * G4UniversalFluctuationForExtrapolator::SampleFluctuations2 -> N samples
//     at EXACTLY those numbers (same material / particle / step, so the a1,
//     a2, a3, e1, e2, e0, tmax, scaling recomputed inside are bit-identical)
//   * G4UniversalFluctuation (stock Geant4 11.2.2, Urban 2021)   -> N samples
//     of the model the full CMSSW SIM actually uses, at the same mean loss
//   * --composite: what the SIM ACTUALLY DRAWS for one step, which is neither
//     of the above: stock G4UniversalFluctuation RESTRICTED to the e-
//     production cut, PLUS the explicit delta-ray secondaries above that cut
//     sampled from G4MuBetheBlochModel::SampleSecondaries.  The single-call
//     `--stock` mode drives the straggling at tcut = Tmax, a configuration the
//     simulation never runs; this one reproduces the decomposition
//     G4VEnergyLossProcess uses.  Default off; when off not one line of the
//     pre-existing code path executes.
//
// Nothing in the CMSSW source area is modified; this links against the
// already-built libTrackPropagationGeant4e.so and the Geant4 externals.
//
// The step is specified the same way Geant4ePropagator specifies it:
//     SampleFluctuations(material, dp, Emax[MeV], stepLength[mm], ekin[MeV])
// so the driver's job is only to build the material and the dynamic particle.
// The material is created BEFORE the fluctuation object because
// G4TablesForExtrapolatorForCVH builds its dE/dx tables over the material
// table that exists at construction time.
//
// Usage:
//   urban_g4driver --Z 8 --A 16 --rho 9.0 --ekin 3160 --len 1.0 --tmax 694.2 \
//                  --n 4000000 --seed 12345 --out /path/prefix [--stock] \
//                  [--tcut <MeV>] [--pdg 13]
// Writes <prefix>.rec (text) and <prefix>.bin (N float64, MeV), and with
// --stock also <prefix>.stock.bin.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#include "TrackPropagation/Geant4e/interface/G4UniversalFluctuationForExtrapolator.hh"

#include "G4Material.hh"
#include "G4MaterialCutsCouple.hh"
#include "G4ProductionCuts.hh"
#include "G4DynamicParticle.hh"
#include "G4ParticleDefinition.hh"
#include "G4MuonMinus.hh"
#include "G4MuonPlus.hh"
#include "G4PionMinus.hh"
#include "G4KaonMinus.hh"
#include "G4Proton.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4UniversalFluctuation.hh"
#include "G4MuBetheBlochModel.hh"
#include "G4BetheBlochModel.hh"
#include "G4VEmModel.hh"
#include "G4DataVector.hh"
#include "G4Electron.hh"
#include "Randomize.hh"
#include "G4IonisParamMat.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4TransportationManager.hh"

namespace {
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
}  // namespace

int main(int argc, char** argv) {
  const double Z = argd(argc, argv, "--Z", 8.0);
  const double A = argd(argc, argv, "--A", 16.0);
  const double rho = argd(argc, argv, "--rho", 9.0);
  const double ekin = argd(argc, argv, "--ekin", 3160.0);   // MeV
  const double len = argd(argc, argv, "--len", 1.0);        // mm
  double tmax = argd(argc, argv, "--tmax", -1.0);           // MeV, <0 = kinematic
  double tcut = argd(argc, argv, "--tcut", -1.0);           // MeV, <0 = tmax
  const long n = argl(argc, argv, "--n", 1000000);
  const long seed = argl(argc, argv, "--seed", 12345);
  const int pdg = (int)argl(argc, argv, "--pdg", 13);
  const std::string out = args(argc, argv, "--out", "urban");
  const bool dostock = argf(argc, argv, "--stock");
  const bool docomp = argf(argc, argv, "--composite");

  // ---- material first: the extrapolator tables are built over the material
  //      table that exists when the fluctuation object is constructed.
  G4Material* mat = new G4Material("ToyLayerMat", Z, A * g / mole, rho * g / cm3);

  // Minimal world so that the EM models the dE/dx-table builder initialises
  // (G4SafetyHelper via the msc model) find a non-null tracking navigator.
  // Nothing is ever tracked through it.
  {
    G4Box* wb = new G4Box("World", 1. * m, 1. * m, 1. * m);
    G4LogicalVolume* wl = new G4LogicalVolume(wb, mat, "World");
    G4VPhysicalVolume* wp = new G4PVPlacement(nullptr, G4ThreeVector(), wl, "World", nullptr, false, 0);
    G4TransportationManager::GetTransportationManager()->SetWorldForTracking(wp);
  }

  G4ParticleDefinition* part = nullptr;
  switch (abs(pdg)) {
    case 13:
      part = (pdg > 0) ? (G4ParticleDefinition*)G4MuonMinus::MuonMinus()
                       : (G4ParticleDefinition*)G4MuonPlus::MuonPlus();
      break;
    case 211:
      part = G4PionMinus::PionMinus();
      break;
    case 321:
      part = G4KaonMinus::KaonMinus();
      break;
    case 2212:
      part = G4Proton::Proton();
      break;
    default:
      part = G4MuonMinus::MuonMinus();
  }
  const double mass = part->GetPDGMass();

  if (tmax <= 0.) {
    // exactly Geant4ePropagator's Emax
    const double etot = ekin + mass;
    const double gam = etot / mass;
    const double beta = std::sqrt(1.0 - 1.0 / (gam * gam));
    const double eta = beta * gam;
    const double mr = CLHEP::electron_mass_c2 / mass;
    tmax = 2.0 * CLHEP::electron_mass_c2 * eta * eta / (1.0 + 2.0 * mr * gam + mr * mr);
  }
  if (tcut <= 0.)
    tcut = tmax;

  G4DynamicParticle dp(part, G4ThreeVector(0., 0., 1.), ekin * MeV);

  CLHEP::HepJamesRandom eng(seed);
  G4Random::setTheEngine(&eng);

  G4UniversalFluctuationForExtrapolator fl;
  fl.SetParticleAndCharge(part, 1.0);

  // ---- the record: exactly what the CVH maker exports for this step
  const double var = fl.SampleFluctuations(mat, &dp, tmax, len, ekin);
  const auto& r = fl.lastRecord();

  // the mean loss the sampler will use, recomputed the same way (dE/dx table
  // x length). Reported so the offline side can standardize identically.
  // (regime 1 divides it by `scaling` internally.)
  FILE* fr = fopen((out + ".rec").c_str(), "w");
  fprintf(fr, "# Z A rho ekin len tmax tcut pdg n seed\n");
  fprintf(fr, "input %g %g %g %.17g %.17g %.17g %.17g %d %ld %ld\n", Z, A, rho, ekin, len, tmax, tcut, pdg, n, seed);
  fprintf(fr, "# regime gsig2 a1 e1 a2 e2 a3 e0 tmaxr scaling  (MeV units)\n");
  fprintf(fr,
          "record %d %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
          r.regime,
          r.gsig2,
          r.a1,
          r.e1,
          r.a2,
          r.e2,
          r.a3,
          r.e0r,
          r.tmaxr,
          r.scaling);
  fprintf(fr, "returnedvar %.17g\n", var);
  fprintf(fr,
          "mationi ipot %.17g e1f %.17g e2f %.17g e0f %.17g f1 %.17g f2 %.17g\n",
          mat->GetIonisation()->GetMeanExcitationEnergy(),
          mat->GetIonisation()->GetEnergy1fluct(),
          mat->GetIonisation()->GetEnergy2fluct(),
          mat->GetIonisation()->GetEnergy0fluct(),
          mat->GetIonisation()->GetF1fluct(),
          mat->GetIonisation()->GetF2fluct());
  // Electron density and the Landau/Bethe xi of THIS step. xi is the whole
  // normalization of the exact delta-ray spectrum
  //     dN/dT = (xi/T^2) [1 - beta^2 T/Tmax + T^2/(2 E^2)]
  // (PDG "Passage of particles through matter" eq. 34.7 / G4 PRM MuBetheBloch),
  // and the Urban a3 channel's own normalization is a3*C = rate*meanLoss/
  // ln(tmax/e0).  Both are printed so the offline side can compare them
  // WITHOUT re-deriving Geant4's dE/dx table.  Units: 1/cm3 and MeV.
  {
    const double etot = ekin + mass;
    const double gam = etot / mass;
    const double beta2 = 1.0 - 1.0 / (gam * gam);
    const double nel = mat->GetElectronDensity() * cm3;  // 1/cm3
    const double xi = CLHEP::twopi_mc2_rcl2 * mat->GetElectronDensity() * (len * mm) / beta2;
    fprintf(fr, "matxi nel %.17g xi %.17g beta2 %.17g gamma %.17g\n", nel, xi / MeV, beta2, gam);
  }

  // ---- Tmax and spin, from GEANT4's OWN models rather than from a formula.
  // The record's tmaxr is whatever Geant4ePropagator passed in; the simulation
  // samples its deltas from G4MuBetheBlochModel (muons) or G4BetheBlochModel
  // (hadrons).  Print all of them side by side so the two can be compared
  // numerically instead of by inspecting two expressions that look alike.
  {
    // LEAKED ON PURPOSE. Their destructors deregister from
    // G4LossTableManager, which at process exit races the static Geant4
    // teardown and aborts with `munmap_chunk(): invalid pointer` AFTER all the
    // output has been written -- i.e. a nonzero exit status on a run whose
    // numbers are complete. Leaking them removes the abort.
    auto* mubbp = new G4MuBetheBlochModel();
    auto* bbp = new G4BetheBlochModel();
    G4MuBetheBlochModel& mubb = *mubbp;
    G4BetheBlochModel& bb = *bbp;
    double tmu = -1., tbb = -1.;
    try {
      tmu = mubb.MaxSecondaryKinEnergy(&dp);
    } catch (...) {
    }
    try {
      tbb = bb.MaxSecondaryKinEnergy(&dp);
    } catch (...) {
    }
    // Geant4ePropagator.cc's own Emax, transcribed verbatim (note its
    // hard-coded electron mass, which is NOT CLHEP::electron_mass_c2)
    const double massGeV = mass / CLHEP::GeV;
    const double pGeV = std::sqrt(ekin * (ekin + 2. * mass)) / CLHEP::GeV;
    const double EtotGeV = std::sqrt(pGeV * pGeV + massGeV * massGeV);
    const double betaP = pGeV / EtotGeV, gammaP = EtotGeV / massGeV;
    const double eMass = 0.51099906 / CLHEP::GeV;
    const double massRatio = eMass / massGeV;
    const double etasq = betaP * gammaP * betaP * gammaP;
    const double EmaxProp =
        1.e6 * (2. * eMass * etasq) / (1. + 2. * massRatio * gammaP + massRatio * massRatio) * 1e-3;  // MeV
    fprintf(fr,
            "tmaxcmp pdgmass %.17g pdgspin %.17g leptonnumber %d emassCLHEP %.17g "
            "tmaxrecord %.17g tmaxMuBB %.17g tmaxBB %.17g tmaxProp %.17g\n",
            mass,
            part->GetPDGSpin(),
            part->GetLeptonNumber(),
            CLHEP::electron_mass_c2,
            r.tmaxr,
            tmu,
            tbb,
            EmaxProp);
  }

  // ---- N samples from the SAME object at the SAME arguments
  std::vector<double> buf(n);
  double s1 = 0., s2 = 0.;
  for (long i = 0; i < n; ++i) {
    const double e = fl.SampleFluctuations2(mat, &dp, tmax, len, ekin, 0.);
    buf[i] = e;
    s1 += e;
    s2 += e * e;
  }
  const double m1 = s1 / n;
  fprintf(fr, "sample2 mean %.17g var %.17g n %ld\n", m1, s2 / n - m1 * m1, n);

  FILE* fb = fopen((out + ".bin").c_str(), "wb");
  fwrite(buf.data(), sizeof(double), n, fb);
  fclose(fb);

  // ---- stock Geant4 sampler (the one the full CMSSW SIM uses), driven with
  //      the mean loss the extrapolator computed, so the two are compared at
  //      the same <dE>.
  if (dostock) {
    G4ProductionCuts* pc = new G4ProductionCuts();
    G4MaterialCutsCouple couple(mat, pc);
    couple.SetIndex(0);
    G4UniversalFluctuation stock;
    stock.InitialiseMe(part);
    stock.SetParticleAndCharge(part, 1.0);
    // mean loss: dE/dx table x length is not accessible from the stock class,
    // so reconstruct it from the extrapolator record when possible, else use
    // the value passed on the command line.
    double meanLoss = argd(argc, argv, "--meanloss", -1.0);
    if (meanLoss <= 0.) {
      fprintf(stderr, "--stock needs --meanloss <MeV> (the dE/dx x length of the step)\n");
      return 2;
    }
    std::vector<double> sb(n);
    double t1 = 0., t2 = 0.;
    for (long i = 0; i < n; ++i) {
      const double e = stock.SampleFluctuations(&couple, &dp, tcut, tmax, len, meanLoss);
      sb[i] = e;
      t1 += e;
      t2 += e * e;
    }
    const double u1 = t1 / n;
    fprintf(fr, "stock meanloss %.17g mean %.17g var %.17g tcut %.17g\n", meanLoss, u1, t2 / n - u1 * u1, tcut);
    FILE* sf = fopen((out + ".stock.bin").c_str(), "wb");
    fwrite(sb.data(), sizeof(double), n, sf);
    fclose(sf);
  }

  // ---- THE SIM-FAITHFUL COMPOSITE.
  //
  // G4VEnergyLossProcess does NOT hand the whole step to the fluctuation
  // model.  It splits the loss at the e- production threshold `tcut`:
  //   * below tcut  -- continuous, sampled by G4UniversalFluctuation at the
  //                    RESTRICTED mean loss  dE/dx(ekin, tcut) * length;
  //   * above tcut  -- discrete, explicit G4Electron secondaries whose number
  //                    is Poisson(sigma(ekin, tcut, Tmax) * length) and whose
  //                    energies come from G4MuBetheBlochModel::SampleSecondaries.
  // The muon's total loss over the step is the sum, and THAT is the object the
  // offline analytic CF (built from the exported record, which spans e0 to the
  // full Tmax) has to match.  The `--stock` mode above drives the straggling at
  // tcut = Tmax instead, i.e. it puts the whole delta spectrum inside the Urban
  // 1/E^2 channel -- a configuration the simulation never runs.
  //
  // Everything here comes from Geant4's own models; nothing is reconstructed.
  if (docomp) {
    if (tcut >= tmax) {
      fprintf(stderr, "--composite needs --tcut < tmax (got %g >= %g)\n", tcut, tmax);
      return 3;
    }
    G4ProductionCuts* pcc = new G4ProductionCuts();
    G4MaterialCutsCouple* couple = new G4MaterialCutsCouple(mat, pcc);
    couple->SetIndex(0);

    // LEAKED ON PURPOSE, same reason as the tmaxcmp block above.
    G4VEmModel* ion = nullptr;
    if (abs(pdg) == 13)
      ion = new G4MuBetheBlochModel();
    else
      ion = new G4BetheBlochModel();
    G4DataVector cutv;
    cutv.push_back(tcut);
    ion->Initialise(part, cutv);

    // restricted / unrestricted dE/dx and the hard-delta rate, from the SAME
    // model the simulation uses.  Internal Geant4 units are MeV and mm, so
    // these are already MeV/mm and 1/mm.
    const double dedxR = ion->ComputeDEDXPerVolume(mat, part, ekin, tcut);
    const double dedxU = ion->ComputeDEDXPerVolume(mat, part, ekin, tmax);
    const double sigV = ion->CrossSectionPerVolume(mat, part, ekin, tcut, tmax);
    const double mlR = argd(argc, argv, "--compmeanloss", -1.0) > 0.
                           ? argd(argc, argv, "--compmeanloss", -1.0)
                           : dedxR * len;
    const double lambda = sigV * len;
    fprintf(fr,
            "composite tcut %.17g dedxrestricted %.17g dedxunrestricted %.17g "
            "meanrestricted %.17g lambda %.17g len %.17g\n",
            tcut,
            dedxR,
            dedxU,
            mlR,
            lambda,
            len);

    G4UniversalFluctuation stk;
    stk.InitialiseMe(part);
    stk.SetParticleAndCharge(part, 1.0);

    std::vector<double> cb(n);
    std::vector<double> sb(n);   // the sub-cut (straggling-only) part
    std::vector<G4DynamicParticle*> vdp;
    vdp.reserve(8);
    double c1 = 0., c2 = 0., h1 = 0., hn = 0., hmax = 0.;
    for (long i = 0; i < n; ++i) {
      const double soft = stk.SampleFluctuations(couple, &dp, tcut, tmax, len, mlR);
      double hard = 0.;
      const G4int nd = (G4int)G4Poisson(lambda);
      for (G4int k = 0; k < nd; ++k) {
        vdp.clear();
        ion->SampleSecondaries(&vdp, couple, &dp, tcut, tmax);
        for (auto* d : vdp) {
          const double te = d->GetKineticEnergy();
          hard += te;
          if (te > hmax)
            hmax = te;
          delete d;
        }
      }
      hn += nd;
      h1 += hard;
      const double e = soft + hard;
      sb[i] = soft;
      cb[i] = e;
      c1 += e;
      c2 += e * e;
    }
    const double cm = c1 / n;
    fprintf(fr,
            "compsample mean %.17g var %.17g ndelta %.17g ehard %.17g tmaxseen %.17g n %ld\n",
            cm,
            c2 / n - cm * cm,
            (double)hn / n,
            h1 / n,
            hmax,
            n);
    FILE* cf = fopen((out + ".comp.bin").c_str(), "wb");
    fwrite(cb.data(), sizeof(double), n, cf);
    fclose(cf);
    FILE* sf2 = fopen((out + ".soft.bin").c_str(), "wb");
    fwrite(sb.data(), sizeof(double), n, sf2);
    fclose(sf2);
  }

  fclose(fr);
  return 0;
}
