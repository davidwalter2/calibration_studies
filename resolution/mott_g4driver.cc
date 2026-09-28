// mott_g4driver -- the single-scattering rejection GEANT4 applies to e+- in
// G4WentzelOKandVIxSection::SampleSingleCosTheta, evaluated with its own
// G4ScreeningMottCrossSection, next to the spin-1/2 term every other
// species gets.
//
// For e+- the explicit single scatters (the WentzelVI tail above theta_min
// and the CoulombScat process above the msc ceiling) are drawn from the
// screened Rutherford law and accepted with
//     grej = RatioMottRutherfordCosT(sqrt(z)) * fm^2 ,  z = 1 - cos(theta)
// against fMottFactor * flat(), the RATE carrying fMottFactor, so the net law
// is Rutherford x RatioMottRutherford x fm^2.  Other species get
//     (1 - z factB + factB1 Z sqrt(z factB) (2 - z)) fm^2 / (1 + z factD).
// The transport cross section (the Gaussian part, and the clean-propagation
// model's rows) carries only the spin term fb.  This driver prints, per Z and
// species, the ratio R(theta) = net / Rutherford on a theta grid, for the
// offline side to fold into the per-step angular law.
//
// Usage:  mott_g4driver --p 3136 [--Z 4,6,14,29]
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "G4Electron.hh"
#include "G4Positron.hh"
#include "G4ScreeningMottCrossSection.hh"
#include "G4SystemOfUnits.hh"

int main(int argc, char** argv) {
  double pmom = 3136.0;  // MeV/c
  std::vector<int> zs = {1, 4, 6, 7, 8, 9, 13, 14, 26, 29, 50, 82};
  for (int i = 1; i + 1 < argc; ++i) {
    if (!strcmp(argv[i], "--p"))
      pmom = atof(argv[i + 1]);
    if (!strcmp(argv[i], "--Z")) {
      zs.clear();
      std::stringstream ss(argv[i + 1]);
      std::string t;
      while (std::getline(ss, t, ','))
        zs.push_back(atoi(t.c_str()));
    }
  }
  const double thetas[] = {1e-4, 3e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.5, 1.0};
  for (G4ParticleDefinition* part : {(G4ParticleDefinition*)G4Electron::Electron(),
                                     (G4ParticleDefinition*)G4Positron::Positron()}) {
    const double mass = part->GetPDGMass();
    const double tkin = std::sqrt(pmom * pmom + mass * mass) - mass;
    G4ScreeningMottCrossSection mott;
    mott.Initialise(part, 1.0);
    for (int Z : zs) {
      mott.SetupKinematic(tkin, Z);
      const double fmott = (part == G4Electron::Electron()) ? 1.0 + 2.0e-4 * Z * Z : 1.0;
      printf("MOTT %s Z %d p %.6g fMottFactor %.8g", part->GetParticleName().c_str(), Z, pmom, fmott);
      for (double th : thetas) {
        const double z1 = 1.0 - std::cos(th);
        printf(" %.4g:%.8g", th, mott.RatioMottRutherfordCosT(std::sqrt(z1)));
      }
      printf("\n");
    }
  }
  fflush(stdout);
  return 0;
}
