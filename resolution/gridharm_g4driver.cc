// Does harmonising the extrapolator's energy grid change what the tables
// return at the energies the CVH fit uses?
//
// The fluctuation model and the reference trajectory each build a
// G4TablesForExtrapolatorForCVH.  The two candidate grids are (70 bins,
// 1 MeV - 10 TeV) and (80 bins, 1 MeV - 100 TeV), and both tables are built on
// the long one.  The grids are ALIGNED -- (Emax/Emin)^(1/bins) = 10^0.1 for
// both -- so every node of the short grid is a node of the long one and the
// only thing that can move is the SPLINE, whose second derivatives come from a
// solve over all nodes.
//
// This driver isolates exactly that.  It fills the two G4PhysicsLogVectors
// with the SAME analytic function, calls Geant4's own FillSecondDerivatives on
// both, and compares Value(E) against each other and against the function.  No
// materials, no models, no run manager: the only Geant4 code exercised is the
// interpolation itself.
//
// build:  ./build_gridharm.sh
// run:    $SCRATCH/gridharm_g4driver.sh

#include <cmath>
#include <cstdio>

#include "G4PhysicsLogVector.hh"

namespace {

  // A Bethe-Bloch-shaped stand-in for dE/dx(T) of a muon in silicon, in
  // MeV/mm.  Only its SHAPE matters here -- the point is to give both vectors
  // identical node values so that any difference is the spline and nothing
  // else.  Relativistic rise plus the 1/beta^2 fall, which is the curvature
  // the spline has to represent.
  double dedx(double T_MeV) {
    const double m = 105.6583745;                  // muon, MeV
    const double E = T_MeV + m;
    const double bg2 = E * E / (m * m) - 1.0;      // (beta*gamma)^2
    const double b2 = bg2 / (1.0 + bg2);
    return 0.0385 * (std::log(1.0e6 * bg2) - b2) / b2 * 0.02;
  }

}  // namespace

int main() {
  // exactly the two constructions, spline flag on
  G4PhysicsLogVector shortv(1.0, 1.0e7, 70, true);   // 1 MeV - 10 TeV, short grid
  G4PhysicsLogVector longv(1.0, 1.0e8, 80, true);    // 1 MeV - 100 TeV, reference

  for (std::size_t j = 0; j <= 70; ++j)
    shortv.PutValue(j, dedx(shortv.Energy(j)));
  for (std::size_t j = 0; j <= 80; ++j)
    longv.PutValue(j, dedx(longv.Energy(j)));
  shortv.FillSecondDerivatives();
  longv.FillSecondDerivatives();

  // the node alignment claim, checked and not assumed
  double maxnode = 0.0;
  for (std::size_t j = 0; j <= 70; ++j) {
    const double a = shortv.Energy(j), b = longv.Energy(j);
    maxnode = std::max(maxnode, std::abs(a - b) / b);
  }
  std::printf("node alignment over the 71 shared nodes: max |dE|/E = %.3e\n\n", maxnode);

  std::printf("%14s %16s %16s %16s %12s %12s\n", "T [MeV]", "f(T)", "short grid",
              "long grid", "long/short-1", "long/f-1");
  const double probes[] = {500., 1000., 3136., 1.0e4, 4.0e4, 1.0e5, 1.0e6,
                           1.0e7 / 10., 1.0e7 / 2., 0.9 * 1.0e7, 1.0e7};
  for (double T : probes) {
    std::size_t ia = 0, ib = 0;
    const double f = dedx(T);
    const double a = shortv.Value(T, ia);
    const double b = longv.Value(T, ib);
    std::printf("%14.4g %16.10f %16.10f %16.10f %12.3e %12.3e\n", T, f, a, b,
                b / a - 1.0, b / f - 1.0);
  }
  return 0;
}
