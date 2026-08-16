// speciesdedx_g4driver -- does the RANGE branch of
// G4EnergyLossForExtrapolatorForCVH::EnergyAfterStep lose the same energy as
// the dE/dx branch, with and without CVH_REF_SPECIESDEDX?
//
// WHY THIS EXISTS
// ---------------
// `EnergyAfterStep` picks between
//
//     E - step * ComputeDEDX(E)                      (step < linLossLimit * R)
//     ComputeEnergy(ComputeRange(E) - step)          (otherwise)
//
// so a fix that corrected only `ComputeDEDX` would make the two branches
// disagree by EXACTLY the term it removed (NOTES_PION s9 item 1,
// NOTES_BARKAS s8.1 item 2).  The toy closure cannot see this: at pT = 3 in
// the layered geometry the range is ~1e3 times the step, so the dE/dx branch
// is taken at every step and the range branch is never exercised (measured --
// the exported reference does not move with the quadrature interval count).
//
// This driver calls the three methods DIRECTLY, at momenta and step lengths
// chosen so the range branch is the one a real propagation would take, and
// prints the loss each branch gives.  Three things are then measurable:
//
//   * the ROUND TRIP `ComputeEnergy(ComputeRange(E)) - E`, which says whether
//     ComputeEnergy is still the numerical inverse of ComputeRange after the
//     range is corrected;
//   * the BRANCH DISAGREEMENT with the switch off and on -- it must not grow;
//   * the disagreement a PARTIAL fix would have left, obtained by pairing the
//     corrected dE/dx branch with the uncorrected range branch.  That is the
//     number the two bullets above are being compared against.
//
// Nothing in the CMSSW source area is written to; this links against the
// already-built libTrackPropagationGeant4e.so, in the pattern of
// barkas_g4driver.cc / urban_g4driver.cc.
//
// Usage:
//   speciesdedx_g4driver --Z 8 --A 16 --rho 9
//   (reads CVH_REF_SPECIESDEDX from the environment, like the fit does)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#include "TrackPropagation/Geant4e/interface/G4EnergyLossForExtrapolatorForCVH.h"

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
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
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

const int kPdg[8] = {13, -13, -211, 211, -321, 321, -2212, 2212};
const char* kLab[8] = {"mu-", "mu+", "pi-", "pi+", "K-", "K+", "pbar", "p"};

// momenta, MeV/c.  3136 is the toy's; the rest walk down towards the end of
// range, where a real propagation DOES take the range branch.
const double kP[5] = {3136.0, 1000.0, 400.0, 200.0, 100.0};
}  // namespace

int main(int argc, char** argv) {
  const double Z = argd(argc, argv, "--Z", 8.0);
  const double A = argd(argc, argv, "--A", 16.0);
  const double rho = argd(argc, argv, "--rho", 9.0);

  G4Material* mat = new G4Material("ToyLayerMat", Z, A * g / mole, rho * g / cm3);
  {
    G4Box* wb = new G4Box("World", 1. * m, 1. * m, 1. * m);
    G4LogicalVolume* wl = new G4LogicalVolume(wb, mat, "World");
    G4VPhysicalVolume* wp = new G4PVPlacement(nullptr, G4ThreeVector(), wl, "World", nullptr, false, 0);
    G4TransportationManager::GetTransportationManager()->SetWorldForTracking(wp);
  }

  auto* eloss = new G4EnergyLossForExtrapolatorForCVH(0);

  printf("# BRANCH pdg lab p ekin range_mm step_mm frac lossDedx lossRange roundtrip\n");
  for (int i = 0; i < 8; ++i) {
    G4ParticleDefinition* part = particleOf(kPdg[i]);
    for (double p : kP) {
      const double m = part->GetPDGMass();
      const double ekin = std::sqrt(p * p + m * m) - m;
      const double R = eloss->ComputeRange(ekin, part, mat);
      // Round trip of the corrected pair, at the same energy.
      const double rt = eloss->ComputeEnergy(R, part, mat) - ekin;
      // ABSOLUTE step lengths, not fractions of the range: the range itself
      // MOVES when the switch is on (that is the point), so `frac * R` would
      // compare two different steps and confound the two effects. `frac` is
      // reported for reference -- linLossLimit is 0.01, so anything at or
      // above 1 % is the RANGE branch in EnergyAfterStep.
      for (double stepmm : {1.0, 20.0, 200.0, 500.0}) {
        const double step = stepmm * mm;
        if (step >= 0.9 * R) {
          continue;
        }
        const double frac = step / R;
        const double lossD = step * eloss->ComputeDEDX(ekin, part, mat);
        const double lossR = ekin - eloss->ComputeEnergy(R - step, part, mat);
        printf("BRANCH %d %s %.10g %.10g %.17g %.17g %g %.17g %.17g %.17g\n",
               kPdg[i], kLab[i], p, ekin, R / mm, step / mm, frac, lossD / MeV, lossR / MeV, rt / MeV);
      }
    }
  }
  fflush(stdout);
  // Leak `eloss` deliberately: its destructor tears down the shared static
  // table object and races the Geant4 static teardown at exit, aborting AFTER
  // all output is written (same reason barkas_g4driver leaks its models).
  return 0;
}
