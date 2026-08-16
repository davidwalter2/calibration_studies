// meanloss_g4driver -- does the FLUCTUATION model's `meanLoss` read the same
// dE/dx table as the REFERENCE trajectory's `ComputeDEDX`?
//
// WHY THIS EXISTS
// ---------------
// Two classes form a mean energy loss from the same table set:
//
//   G4EnergyLossForExtrapolatorForCVH::ComputeDEDX   -- the REFERENCE trajectory
//   G4UniversalFluctuationForExtrapolator            -- `meanLoss = length*dedx`,
//                                                       which sets the Urban
//                                                       channel weights and, in
//                                                       the Gaussian regime, the
//                                                       returned loss itself
//
// `CVH_REF_CHARGEAWARE` made the FIRST of them select fDedxMuonMinus /
// fDedxAntiProton for a negative track.  It did not touch the second, whose
// SetParticleAndCharge selected fDedxMuon / fDedxProton unconditionally.  So
// with the switch on, the reference's mean and the noise model's mean
// disagreed by exactly the charge-odd part of Geant4's high-order block --
// 3.2e-3 -- on every negative track (NOTES_SPECIESDEDX s2.1/s8).
//
// This driver measures both, per species, at the same kinematics, so the
// statement is a measurement rather than a reading of the source.
//
// HOW THE FLUCTUATION'S MEAN IS OBTAINED WITHOUT REPLICATING IT
// -------------------------------------------------------------
// `SampleFluctuations2` computes `meanLoss = length * dedx` and, when
// `meanLoss < minLoss` (10 eV), returns it UNCHANGED and without touching the
// random engine.  Calling it at a step short enough to land in that branch
// therefore returns the production code's own `meanLoss`, exactly -- no table
// lookup, no mass scaling and no Tmax correction is re-implemented here.
// dE/dx is then `ret / length`.
//
// THE TWO NUMBERS ARE NOT EXPECTED TO BE EQUAL IN ABSOLUTE VALUE, and that is
// not what is being tested.  The two classes hold SEPARATE table objects built
// on different grids -- the extrapolator's is (nbins, 1 MeV, 100 TeV) and
// carries the radiative mean unless CVH_IONONLY is set, the fluctuation's is
// (70, 1 MeV, 10 TeV) and is always ionization-only.  For a muon that alone is
// ~1e-3.  What must agree is the CHARGE-ODD PART: the conjugate ratio
// dedx(q<0)/dedx(q>0) - 1, which is 0 when the switch is off and -3.2e-3 when
// it is on, in BOTH columns or the mean the noise is built on is not the mean
// the trajectory follows.
//
// Nothing in the CMSSW source area is written to; this links against the
// already-built libTrackPropagationGeant4e.so, in the pattern of
// barkas_g4driver.cc / speciesdedx_g4driver.cc / urban_g4driver.cc.
//
// Usage:
//   meanloss_g4driver --Z 8 --A 16 --rho 9 --p 3136
//   (reads CVH_REF_CHARGEAWARE / CVH_REF_SPECIESDEDX from the environment,
//    exactly as the fit does)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "TrackPropagation/Geant4e/interface/G4EnergyLossForExtrapolatorForCVH.h"
#include "TrackPropagation/Geant4e/interface/G4UniversalFluctuationForExtrapolator.hh"

#include "G4Material.hh"
#include "G4ParticleDefinition.hh"
#include "G4DynamicParticle.hh"
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

// positive first in each pair, so the printed conjugate ratio is
// (negative / positive) - 1 and carries the sign of the correction
const int kPdg[8] = {-13, 13, 211, -211, 321, -321, 2212, -2212};
const char* kLab[8] = {"mu+", "mu-", "pi+", "pi-", "K+", "K-", "p", "pbar"};
}  // namespace

int main(int argc, char** argv) {
  const double Z = argd(argc, argv, "--Z", 8.0);
  const double A = argd(argc, argv, "--A", 16.0);
  const double rho = argd(argc, argv, "--rho", 9.0);
  const double p = argd(argc, argv, "--p", 3136.0);
  // Short enough that meanLoss < minLoss = 10 eV for every species here
  // (dE/dx ~ 1.5-1.8 MeV/mm, so 1e-6 mm gives ~1.7 eV).  Asserted below.
  const double L = argd(argc, argv, "--len", 1e-6);

  G4Material* mat = new G4Material("ToyLayerMat", Z, A * g / mole, rho * g / cm3);
  {
    G4Box* wb = new G4Box("World", 1. * m, 1. * m, 1. * m);
    G4LogicalVolume* wl = new G4LogicalVolume(wb, mat, "World");
    G4VPhysicalVolume* wp = new G4PVPlacement(nullptr, G4ThreeVector(), wl, "World", nullptr, false, 0);
    G4TransportationManager::GetTransportationManager()->SetWorldForTracking(wp);
  }

  auto* eloss = new G4EnergyLossForExtrapolatorForCVH(0);
  auto* fluct = new G4UniversalFluctuationForExtrapolator();

  printf("# MEANLOSS pdg lab ekin dedxRef dedxFluct\n");
  for (int i = 0; i < 8; ++i) {
    G4ParticleDefinition* part = particleOf(kPdg[i]);
    const double m = part->GetPDGMass();
    const double ekin = std::sqrt(p * p + m * m) - m;

    const double dedxRef = eloss->ComputeDEDX(ekin, part, mat);

    // Tmax is irrelevant on this branch (the early return precedes every use
    // of it); pass the physical one anyway so the call is well formed.
    const double gam = ekin / m + 1.0;
    const double bg2 = gam * gam - 1.0;
    const double r = CLHEP::electron_mass_c2 / m;
    const double tmax = 2. * CLHEP::electron_mass_c2 * bg2 / (1. + 2. * gam * r + r * r);

    G4DynamicParticle dp(part, G4ThreeVector(0., 0., 1.), ekin);
    const double step = L * mm;
    const double ml = fluct->SampleFluctuations2(mat, &dp, tmax, step, ekin, 0.);
    const double dedxFluct = ml / step;

    // The early-return branch is the whole method here: if the step were long
    // enough to reach the sampler this would be a RANDOM number and the
    // comparison would be meaningless, so refuse rather than print noise.
    if (ml >= 10. * CLHEP::eV) {
      fprintf(stderr,
              "meanloss_g4driver: step %g mm gives meanLoss %g MeV >= minLoss "
              "(10 eV) for %s -- the sampler ran; shorten --len\n",
              L, ml / MeV, kLab[i]);
      return 2;
    }

    printf("MEANLOSS %d %s %.17g %.17g %.17g\n", kPdg[i], kLab[i], ekin, dedxRef / (MeV / mm), dedxFluct / (MeV / mm));
  }
  fflush(stdout);
  // Leak deliberately: the destructors tear down the shared static table
  // objects and race Geant4's static teardown at exit (same reason
  // barkas_g4driver and speciesdedx_g4driver leak theirs).
  return 0;
}
