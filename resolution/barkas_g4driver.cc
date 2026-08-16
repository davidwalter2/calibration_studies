// barkas_g4driver -- ask GEANT4 ITSELF, and the CVH extrapolator, what their
// mean stopping power is for a particle and for its charge conjugate.
//
// WHY THIS EXISTS
// ---------------
// NOTES_HADRONS s6 measured that Geant4's simulated mean ionization loss is
// charge-ODD by 0.17-0.36 % while the CVH reference trajectory is charge-EVEN
// to 1.7e-9, and attributed the former to "the Barkas/Bloch z^3 term".  That
// attribution was an inference from a code path that was never evaluated.
// This driver evaluates it.
//
// Geant4's projectile-charge expansion lives in G4EmCorrections and enters
// G4BetheBlochModel::ComputeDEDXPerVolume (hadrons) and
// G4MuBetheBlochModel::ComputeDEDXPerVolume (muons) through ONE line:
//
//     dedx += corr->HighOrderCorrections(p, material, kineticEnergy, cutEnergy);
//
// with (G4EmCorrections.cc)
//
//     sum  = 2.0*(Barkas + Bloch) + Mott;
//     sum *= material->GetElectronDensity()*q2*twopi_mc2_rcl2/beta2;
//
// so the three terms are separable and only two of them are odd in z:
//     Barkas  ~  1.29 * charge * f(beta)          ODD, falls with beta
//     Bloch   ~ -y2*term,  y2 = q2/ba2            EVEN (function of q^2 only)
//     Mott    =  pi * alpha * beta * charge       ODD, GROWS with beta
//
// The driver prints all three, per species, per SIGN, next to the total dE/dx
// they correct, so the charge-odd fraction is read off rather than assumed.
// It also calls G4EnergyLossForExtrapolatorForCVH::ComputeDEDX for the same
// eight particles -- the table the CVH reference trajectory actually uses --
// so "the reference is charge-blind" is a measurement, not a code reading.
//
// Nothing in the CMSSW source area is written to; this links against the
// already-built libTrackPropagationGeant4e.so and the Geant4 externals, in
// the pattern of urban_g4driver.cc.
//
// Usage:
//   barkas_g4driver --Z 8 --A 16 --rho 9 --p 3136.0 --tcut 0.0094862
//   barkas_g4driver ... --ekinlist <file>     # lines: "<pdg> <ekin/MeV>"
//   barkas_g4driver ... --scan                # beta*gamma scan, per species

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#include "TrackPropagation/Geant4e/interface/G4EnergyLossForExtrapolatorForCVH.h"

#include "G4Material.hh"
#include "G4DataVector.hh"
#include "G4DynamicParticle.hh"
#include "G4ParticleDefinition.hh"
#include "G4MuonMinus.hh"
#include "G4MuonPlus.hh"
#include "G4PionMinus.hh"
#include "G4PionPlus.hh"
#include "G4KaonMinus.hh"
#include "G4KaonPlus.hh"
#include "G4Proton.hh"
#include "G4AntiProton.hh"
#include "G4Electron.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4MuBetheBlochModel.hh"
#include "G4BetheBlochModel.hh"
#include "G4EmCorrections.hh"
#include "G4LossTableManager.hh"
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
    case 13:
      return G4MuonMinus::MuonMinus();
    case -13:
      return G4MuonPlus::MuonPlus();
    case -211:
      return G4PionMinus::PionMinus();
    case 211:
      return G4PionPlus::PionPlus();
    case -321:
      return G4KaonMinus::KaonMinus();
    case 321:
      return G4KaonPlus::KaonPlus();
    case 2212:
      return G4Proton::Proton();
    case -2212:
      return G4AntiProton::AntiProton();
  }
  return nullptr;
}

// The eight species of NOTES_HADRONS, in its order.
const int kPdg[8] = {13, -13, -211, 211, -321, 321, -2212, 2212};
const char* kLab[8] = {"mu-", "mu+", "pi-", "pi+", "K-", "K+", "pbar", "p"};

struct Kin {
  double mass, ekin, etot, gam, beta, beta2, bg, tmax;
};

Kin kinOf(const G4ParticleDefinition* part, double p) {
  Kin k;
  k.mass = part->GetPDGMass();
  k.etot = std::sqrt(p * p + k.mass * k.mass);
  k.ekin = k.etot - k.mass;
  k.gam = k.etot / k.mass;
  k.beta = p / k.etot;
  k.beta2 = k.beta * k.beta;
  k.bg = p / k.mass;
  const double mr = CLHEP::electron_mass_c2 / k.mass;
  k.tmax = 2.0 * CLHEP::electron_mass_c2 * k.bg * k.bg / (1.0 + 2.0 * k.gam * mr + mr * mr);
  return k;
}
}  // namespace

int main(int argc, char** argv) {
  const double Z = argd(argc, argv, "--Z", 8.0);
  const double A = argd(argc, argv, "--A", 16.0);
  const double rho = argd(argc, argv, "--rho", 9.0);
  const double pmom = argd(argc, argv, "--p", 3136.0);     // MeV/c
  double tcut = argd(argc, argv, "--tcut", 0.0094862);     // MeV (cut1e4)
  const std::string ekinlist = args(argc, argv, "--ekinlist", "");
  const bool doscan = argf(argc, argv, "--scan");

  // ---- material first: G4TablesForExtrapolatorForCVH builds its dE/dx tables
  //      over the material table that exists at construction time.
  G4Material* mat = new G4Material("ToyLayerMat", Z, A * g / mole, rho * g / cm3);
  {
    G4Box* wb = new G4Box("World", 1. * m, 1. * m, 1. * m);
    G4LogicalVolume* wl = new G4LogicalVolume(wb, mat, "World");
    G4VPhysicalVolume* wp = new G4PVPlacement(nullptr, G4ThreeVector(), wl, "World", nullptr, false, 0);
    G4TransportationManager::GetTransportationManager()->SetWorldForTracking(wp);
  }
  G4DataVector cuts;
  cuts.resize(G4Material::GetNumberOfMaterials(), DBL_MAX);

  G4EmCorrections* corr = G4LossTableManager::Instance()->EmCorrections();

  const double nel = mat->GetElectronDensity();  // Geant4 internal units
  printf("MAT name %s Z %g A %g rho %g ipot_eV %.17g nel_percm3 %.17g\n",
         mat->GetName().c_str(), Z, A, rho,
         mat->GetIonisation()->GetMeanExcitationEnergy() / eV, nel * cm3);

  // ------------------------------------------------------------------ per-species
  // The models are LEAKED on purpose (urban_g4driver.cc records the reason:
  // their destructors deregister from G4LossTableManager and race the static
  // teardown at exit, aborting AFTER all output is written).
  if (ekinlist.empty() && !doscan) {
    printf("# pdg lab mass ekin beta beta2 bg tmax  model  dedxR dedxU pref hoc "
           "barkas bloch mott  Lrest  xs  extrap\n");
    for (int i = 0; i < 8; ++i) {
      G4ParticleDefinition* part = particleOf(kPdg[i]);
      const Kin k = kinOf(part, pmom);
      const bool ismu = (std::abs(kPdg[i]) == 13);

      // The model the SIMULATION uses for this species at this energy:
      //   G4MuIonisation  -> G4MuBetheBlochModel above 0.2 MeV
      //   G4hIonisation   -> G4BetheBlochModel  above 2 MeV * m/m_p
      G4VEmModel* mdl;
      if (ismu) {
        auto* m = new G4MuBetheBlochModel();
        m->Initialise(part, cuts);
        m->SetUseBaseMaterials(false);
        mdl = m;
      } else {
        auto* m = new G4BetheBlochModel();
        m->Initialise(part, cuts);
        m->SetUseBaseMaterials(false);
        mdl = m;
      }

      const double dedxR = mdl->ComputeDEDXPerVolume(mat, part, k.ekin, tcut);
      const double dedxU = mdl->ComputeDEDXPerVolume(mat, part, k.ekin, k.ekin);

      // the three high-order terms, separately, and the prefactor that turns
      // them into an energy loss per unit length
      const double barkas = corr->BarkasCorrection(part, mat, k.ekin);
      const double bloch = corr->BlochCorrection(part, mat, k.ekin);
      const double mott = corr->MottCorrection(part, mat, k.ekin);
      const double q = part->GetPDGCharge() / CLHEP::eplus;
      const double pref = nel * q * q * CLHEP::twopi_mc2_rcl2 / k.beta2;
      const double hoc = corr->HighOrderCorrections(part, mat, k.ekin, tcut);

      printf("SPECIES %d %s %.10g %.10g %.10g %.12g %.10g %.10g %s "
             "%.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g",
             kPdg[i], kLab[i], k.mass, k.ekin, k.beta, k.beta2, k.bg, k.tmax,
             ismu ? "G4MuBetheBlochModel" : "G4BetheBlochModel",
             dedxR / (MeV / mm), dedxU / (MeV / mm), pref / (MeV / mm),
             hoc / (MeV / mm), barkas, bloch, mott, (dedxR - hoc) / pref);

      // The DELTA-RAY side, for contrast.  The brief's premise is that the
      // knock-on cross section goes as z^2 and so cannot be charge-odd; that
      // is checked here rather than assumed, by asking the same model for the
      // number of secondaries above the production cut.
      const double xs = mdl->CrossSectionPerVolume(mat, part, k.ekin, tcut, k.tmax);
      printf(" %.17g", xs * mm);

      // and what the CVH reference trajectory's OWN table gives for the same
      // particle -- G4EnergyLossForExtrapolatorForCVH::ComputeDEDX
      static G4EnergyLossForExtrapolatorForCVH* eloss = nullptr;
      if (!eloss) {
        eloss = new G4EnergyLossForExtrapolatorForCVH(0);
        eloss->Initialisation();
      }
      const double dex = eloss->ComputeDEDX(k.ekin, part, mat);
      printf(" %.17g\n", dex / (MeV / mm));
      fflush(stdout);
    }
    return 0;
  }

  // ------------------------------------------------------------------ ekin list
  // "<pdg> <ekin/MeV>" per line: the exact per-step kinematics the CVH export
  // recorded, so the prediction needs no interpolation.
  if (!ekinlist.empty()) {
    FILE* f = fopen(ekinlist.c_str(), "r");
    if (!f) {
      fprintf(stderr, "cannot open %s\n", ekinlist.c_str());
      return 1;
    }
    printf("# CORR pdg ekin barkas bloch mott hoc pref dedxR\n");
    int pdg;
    double ekin, mZ, mA, mrho;
    // One G4Material per distinct (Z, A, rho) in the list: the exported record
    // spans three (ToyLayerMat, the Beryllium beam pipe, Air), and Barkas --
    // unlike Mott -- is material-dependent.  It depends only on the ELEMENT
    // COMPOSITION, not on the density: G4EmCorrections::BarkasCorrection
    // divides by GetTotNbOfAtomsPerVolume(), so atomDensity[i] enters only as
    // a number fraction.  A and rho are therefore carried for bookkeeping.
    std::vector<std::pair<double, G4Material*> > cache;
    while (fscanf(f, "%d %lf %lf %lf %lf", &pdg, &ekin, &mZ, &mA, &mrho) == 5) {
      G4ParticleDefinition* part = particleOf(pdg);
      if (!part)
        continue;
      G4Material* smat = nullptr;
      for (auto& c : cache)
        if (c.first == mZ)
          smat = c.second;
      if (!smat) {
        char nm[64];
        snprintf(nm, sizeof(nm), "StepMat_Z%g", mZ);
        smat = new G4Material(nm, mZ, mA * g / mole, mrho * g / cm3);
        cache.push_back(std::make_pair(mZ, smat));
      }
      const double barkas = corr->BarkasCorrection(part, smat, ekin);
      const double bloch = corr->BlochCorrection(part, smat, ekin);
      const double mott = corr->MottCorrection(part, smat, ekin);
      const double hoc = corr->HighOrderCorrections(part, smat, ekin, tcut);
      const double mass = part->GetPDGMass();
      const double gam = 1.0 + ekin / mass;
      const double beta2 = 1.0 - 1.0 / (gam * gam);
      const double q = part->GetPDGCharge() / CLHEP::eplus;
      const double pref = nel * q * q * CLHEP::twopi_mc2_rcl2 / beta2;
      printf("CORR %d %.17g %.17g %.17g %.17g %.17g %.17g\n", pdg, ekin, barkas, bloch, mott,
             hoc / (MeV / mm), pref / (MeV / mm));
    }
    fclose(f);
    return 0;
  }

  // ------------------------------------------------------------------ scan
  // beta*gamma dependence of the two ODD terms, which is the discriminant
  // between them: Barkas falls with beta, Mott grows like beta.
  printf("# SCAN pdg bg ekin barkas bloch mott hoc dedxR oddfrac\n");
  for (int i = 0; i < 8; ++i) {
    G4ParticleDefinition* part = particleOf(kPdg[i]);
    const double mass = part->GetPDGMass();
    G4VEmModel* mdl;
    if (std::abs(kPdg[i]) == 13) {
      auto* m = new G4MuBetheBlochModel();
      m->Initialise(part, cuts);
      m->SetUseBaseMaterials(false);
      mdl = m;
    } else {
      auto* m = new G4BetheBlochModel();
      m->Initialise(part, cuts);
      m->SetUseBaseMaterials(false);
      mdl = m;
    }
    for (int j = 0; j <= 60; ++j) {
      const double bg = std::pow(10.0, -0.5 + 2.5 * j / 60.0);  // 0.316 .. 100
      const double gam = std::sqrt(1.0 + bg * bg);
      const double ekin = mass * (gam - 1.0);
      if (ekin < 2.0)
        continue;  // below G4BetheBlochModel's validity for hadrons
      const double beta2 = bg * bg / (gam * gam);
      const double barkas = corr->BarkasCorrection(part, mat, ekin);
      const double bloch = corr->BlochCorrection(part, mat, ekin);
      const double mott = corr->MottCorrection(part, mat, ekin);
      const double hoc = corr->HighOrderCorrections(part, mat, ekin, tcut);
      const double dedxR = mdl->ComputeDEDXPerVolume(mat, part, ekin, tcut);
      const double q = part->GetPDGCharge() / CLHEP::eplus;
      const double pref = nel * q * q * CLHEP::twopi_mc2_rcl2 / beta2;
      // the charge-odd FRACTION of the restricted dE/dx: the + and - values
      // differ by twice the odd part.
      const double oddfrac = 2.0 * pref * (2.0 * barkas + mott) / dedxR;
      printf("SCAN %d %.10g %.10g %.17g %.17g %.17g %.17g %.17g %.17g\n", kPdg[i], bg, ekin, barkas,
             bloch, mott, hoc / (MeV / mm), dedxR / (MeV / mm), oddfrac);
    }
  }
  return 0;
}
