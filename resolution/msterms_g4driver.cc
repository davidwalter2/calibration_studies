// msterms_g4driver -- ask GEANT4 ITSELF what its Wentzel scattering cross
// section is for the toy material and each species, so that the offline
// re-implementation used to decompose it can be VALIDATED rather than trusted.
//
// WHY THIS EXISTS
// ---------------
// The analytic model's MS channel (cf_ms_exact.moliere_params + gshape) is a
// screened-Rutherford single-scattering density
//
//     n(t) = chi_c^2 / (t + chi_a^2)^2 / (1 + t/theta_FF^2)^2 ,  t = theta^2
//     chi_c^2 = 0.157e-6 * sum_i w_i Z_i(Z_i+1)/A_i * x / (p^2 beta^2)
//
// integrated as ONE continuous law over the whole angular range.  Geant4 runs
// G4WentzelVIModel + G4CoulombScattering, whose cross section
// (G4WentzelOKandVIxSection) is the SAME Wentzel law but with FOUR extra
// pieces of structure:
//
//   (a) the nucleus and the atomic electrons are separate terms with
//       DIFFERENT upper angular limits -- cosTetMaxNuc for the nucleus,
//       cosTetMaxElec for the electrons, the latter set by the DELTA-RAY
//       PRODUCTION CUT (ComputeMaxElectronScattering: 1 - cos = min(cut,Tmax)
//       * m_e / p^2).  The model's Z(Z+1) carries the electron term at the
//       nuclear limit;
//   (b) an angular ceiling 1 - cos = factorA2 * A^(-2/3) / p^2 above which
//       G4WentzelVIModel hands over to the separate CoulombScat process;
//   (c) a spin term (1 - z*factB, factB = spin/invbeta2) and a second-Born
//       McKinley-Feshbach term (factB1*Z*sqrt(z*factB)*(2-z)) in the
//       single-scattering rejection;
//   (d) the form factor applied as |F|^2 with formfactA = FormFactor[Z]*p^2.
//
// This driver prints the G4 members that fix (a)-(d) and the public transport
// cross section at a set of cosTMax, so `msterms.py terms` can reproduce the
// integrand exactly and then decompose it.  Nothing in the CMSSW or Geant4
// source tree is written to.
//
// Usage:
//   msterms_g4driver --Z 8 --A 16 --rho 9 --p 3136.0 --tcut 0.0094862

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#include "G4Material.hh"
#include "G4DataVector.hh"
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
#include "G4IonisParamMat.hh"
#include "G4EmParameters.hh"
#include "G4WentzelOKandVIxSection.hh"
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

const int kPdg[8] = {13, -13, -211, 211, -321, 321, -2212, 2212};
const char* kLab[8] = {"mu-", "mu+", "pi-", "pi+", "K-", "K+", "pbar", "p"};

// All of G4WentzelOKandVIxSection's state is `protected`, so a derived class
// is the supported way to read it.  Nothing is overridden; this only exposes.
class Wokvi : public G4WentzelOKandVIxSection {
public:
  explicit Wokvi(G4bool comb) : G4WentzelOKandVIxSection(comb) {}
  double screenZ_() const { return screenZ; }
  double formfactA_() const { return formfactA; }
  double kinFactor_() const { return kinFactor; }
  double factB_() const { return factB; }
  double factD_() const { return factD; }
  double factorA2_() const { return factorA2; }
  double invbeta2_() const { return invbeta2; }
  double mom2_() const { return mom2; }
  double spin_() const { return spin; }
  double cosTetMaxNuc_() const { return cosTetMaxNuc; }
  double cosTetMaxElec_() const { return cosTetMaxElec; }
  double cosThetaMax_() const { return cosThetaMax; }
  const void* mott_() const { return fMottXSection; }
};
}  // namespace

int main(int argc, char** argv) {
  const double Z = argd(argc, argv, "--Z", 8.0);
  const double A = argd(argc, argv, "--A", 16.0);
  const double rho = argd(argc, argv, "--rho", 9.0);
  const double pmom = argd(argc, argv, "--p", 3136.0);     // MeV/c
  const double tcut = argd(argc, argv, "--tcut", 0.0094862);  // MeV

  G4Material* mat = new G4Material("ToyLayerMat", Z, A * g / mole, rho * g / cm3);
  {
    G4Box* wb = new G4Box("World", 1. * m, 1. * m, 1. * m);
    G4LogicalVolume* wl = new G4LogicalVolume(wb, mat, "World");
    G4VPhysicalVolume* wp = new G4PVPlacement(nullptr, G4ThreeVector(), wl, "World", nullptr, false, 0);
    G4TransportationManager::GetTransportationManager()->SetWorldForTracking(wp);
  }

  G4EmParameters* par = G4EmParameters::Instance();
  printf("PARAM MscThetaLimit %.17g FactorForAngleLimit %.17g ScreeningFactor %.17g "
         "NuclearFormfactorType %d MuHadLateralDisplacement %d UseMottCorrection %d "
         "MscMuHadRangeFactor %.17g\n",
         par->MscThetaLimit(), par->FactorForAngleLimit(), par->ScreeningFactor(),
         (int)par->NuclearFormfactorType(), (int)par->MuHadLateralDisplacement(),
         (int)par->UseMottCorrection(), par->MscMuHadRangeFactor());

  const double invA23 = mat->GetIonisation()->GetInvA23();
  const double natoms = mat->GetTotNbOfAtomsPerVolume();  // 1/mm3 internal
  printf("MAT Z %g A %g rho %g InvA23 %.17g natoms_percm3 %.17g X0_cm %.17g\n",
         Z, A, rho, invA23, natoms * cm3, mat->GetRadlen() / cm);

  // the cosTMax grid at which the public transport cross section is sampled;
  // the offline side reproduces every one of these.
  const double omc[] = {1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 3.4e-4, 1e-3, 1e-2, 1e-1, 1.0, 2.0};
  const int nomc = sizeof(omc) / sizeof(omc[0]);

  for (int i = 0; i < 8; ++i) {
    G4ParticleDefinition* part = particleOf(kPdg[i]);
    const double mass = part->GetPDGMass();
    const double etot = std::sqrt(pmom * pmom + mass * mass);
    const double ekin = etot - mass;

    // combined mode, and the cosThetaLim G4WentzelVIModel passes when
    // MscThetaLimit == pi: the model's own default member value, -1.
    Wokvi w(true);
    w.Initialise(part, -1.0);
    const double ctn = w.SetupKinematic(ekin, mat);
    const double ctn2 = w.SetupTarget((G4int)(Z + 0.5), tcut);

    printf("WOKVI %d %s mass %.17g ekin %.17g mom2 %.17g invbeta2 %.17g spin %.17g "
           "cosThetaMax %.17g omcTetMaxNuc %.17g omcTetMaxElec %.17g "
           "screenZ %.17g formfactA %.17g kinFactor %.17g factB %.17g factD %.17g "
           "factorA2 %.17g mott %d ctn %.17g ctn2 %.17g\n",
           kPdg[i], kLab[i], mass, ekin, w.mom2_(), w.invbeta2_(), w.spin_(),
           w.cosThetaMax_(), 1.0 - w.cosTetMaxNuc_(), 1.0 - w.cosTetMaxElec_(),
           w.screenZ_(), w.formfactA_(), w.kinFactor_(), w.factB_(), w.factD_(),
           w.factorA2_(), w.mott_() != nullptr ? 1 : 0, 1.0 - ctn, 1.0 - ctn2);

    for (int j = 0; j < nomc; ++j) {
      const double c = 1.0 - omc[j];
      const double xs = w.ComputeTransportCrossSectionPerAtom(c);
      printf("SIGTR %s %.17g %.17g\n", kLab[i], omc[j], xs / (cm * cm));
    }
    // the number G4 actually uses for the msc step: cosTMax = cosTetMaxNuc
    printf("SIGTRNUC %s %.17g %.17g\n", kLab[i], 1.0 - w.cosTetMaxNuc_(),
           w.ComputeTransportCrossSectionPerAtom(w.cosTetMaxNuc_()) / (cm * cm));
    // and with the electron term forced to the same limit as the nucleus,
    // which is what Z(Z+1) does: raise cosTetMaxElec above cosTetMaxNuc so
    // std::max(cosTMax, cosTetMaxElec) == cosTMax.
    printf("NUCONLY %s %.17g\n", kLab[i], 0.0);
  }
  fflush(stdout);
  // Leak everything: the destructors race static teardown after all output.
  return 0;
}
