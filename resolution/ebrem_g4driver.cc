// ebrem_g4driver -- Geant4's OWN e+- bremsstrahlung, tabulated independently
// of the propagator, to validate the e+- radiative spectrum and dE/dx that
// Geant4ePropagator::fillRadiativeSpectrum / computeRadiativeDEDX export.
//
// The simulation runs G4eBremsstrahlung as CMSEmStandardPhysics registers it:
// G4SeltzerBergerModel below 1 GeV and G4eBremsstrahlungRelModel above.  For
// one material and a list of kinetic energies this writes, per model, the
// macroscopic emission rate and energy-loss rate in bins of v = k / T:
//   n_b = CrossSectionPerVolume(mat, e, T, max(v_b T, gcut), v_{b+1} T)   [1/mm]
//   e_b = dEdx(cut = v_{b+1} T) - dEdx(cut = max(v_b T, gcut))            [MeV/mm]
// plus the total, the sub-cut and the unrestricted dE/dx -- the functions
// G4VEnergyLossProcess tabulates for its lambda and dE/dx tables.  The export
// takes dSigma/dk from the same CrossSectionPerVolume over a narrow window, so
// bin integrals here and point values there are two routes to one cross
// section (agreement on realmat_full: 4e-4, the bin-centre approximation).
//
// Nothing in the CMSSW or Geant4 source tree is written to.
//
// Usage:
//   ebrem_g4driver --pdg 11 --mat "Z1:A1:w1,Z2:A2:w2" --rho 2.33 \
//                  --gcut 0.00099 --tkin "100,200,..." --nbin 280 --vmin 1e-8 \
//                  --out /path/prefix
// Writes <prefix>.txt; with --sample N the joint emission law instead
// (see SAMPLE MODE).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "G4Material.hh"
#include "G4Element.hh"
#include "G4ProductionCuts.hh"
#include "G4ProductionCutsTable.hh"
#include "G4Region.hh"
#include "G4DataVector.hh"
#include "G4Electron.hh"
#include "G4Positron.hh"
#include "G4Gamma.hh"
#include "G4SeltzerBergerModel.hh"
#include "G4eBremsstrahlungRelModel.hh"
#include "G4EmParameters.hh"
#include "G4SystemOfUnits.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4TransportationManager.hh"
#include "G4DynamicParticle.hh"
#include "G4ParticleChangeForLoss.hh"
#include "G4VEmAngularDistribution.hh"
#include "G4MaterialCutsCouple.hh"
#include "Randomize.hh"

namespace {
  const char* argv_of(int argc, char** argv, const char* key) {
    for (int i = 1; i + 1 < argc; ++i)
      if (!std::strcmp(argv[i], key)) return argv[i + 1];
    return nullptr;
  }
  double argd(int argc, char** argv, const char* key, double def) {
    const char* s = argv_of(argc, argv, key);
    return s ? std::atof(s) : def;
  }
  std::string args(int argc, char** argv, const char* key, const char* def) {
    const char* s = argv_of(argc, argv, key);
    return s ? std::string(s) : std::string(def);
  }
  struct Comp { double Z, A, w; };
  // the particle change is protected on G4VEmModel; exposed as the propagator's
  // probes expose ComputeDMicroscopicCrossSection
  struct SBSampler : public G4SeltzerBergerModel {
    using G4VEmModel::GetParticleChangeForLoss;
  };
  struct RelSampler : public G4eBremsstrahlungRelModel {
    using G4VEmModel::GetParticleChangeForLoss;
  };
}  // namespace

int main(int argc, char** argv) {
  const int pdg = (int)argd(argc, argv, "--pdg", 11);
  const std::string matspec = args(argc, argv, "--mat", "");
  const double rho = argd(argc, argv, "--rho", 2.33);         // g/cm3
  const double gcut = argd(argc, argv, "--gcut", 0.00099);    // MeV
  const std::string tlist = args(argc, argv, "--tkin", "3000");
  const int nbin = (int)argd(argc, argv, "--nbin", 280);
  const double vmin = argd(argc, argv, "--vmin", 1e-8);
  const std::string out = args(argc, argv, "--out", "ebrem");
  if (std::abs(pdg) != 11 || matspec.empty()) {
    fprintf(stderr, "ebrem_g4driver: --pdg must be +-11 and --mat is required\n");
    return 2;
  }

  std::vector<Comp> comp;
  {
    std::stringstream ss(matspec);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      Comp c{};
      if (std::sscanf(tok.c_str(), "%lf:%lf:%lf", &c.Z, &c.A, &c.w) != 3) {
        fprintf(stderr, "bad --mat token '%s'\n", tok.c_str());
        return 2;
      }
      comp.push_back(c);
    }
  }
  std::vector<double> tkin;
  {
    std::stringstream ss(tlist);
    std::string tok;
    while (std::getline(ss, tok, ',')) tkin.push_back(std::atof(tok.c_str()));
  }

  G4EmParameters::Instance()->SetDefaults();   // as CMSEmStandardPhysics

  double wsum = 0.;
  for (const auto& c : comp) wsum += c.w;
  G4Material* mat = new G4Material("EbremMat", rho * g / cm3, (G4int)comp.size());
  for (size_t i = 0; i < comp.size(); ++i) {
    const std::string nm = "El" + std::to_string(i);
    mat->AddElement(new G4Element(nm, nm, comp[i].Z, comp[i].A * g / mole), comp[i].w / wsum);
  }
  // a world and a registered couple: the SB model's element selectors walk
  // G4ProductionCutsTable, and model initialisation needs a tracking world
  {
    G4Box* wb = new G4Box("World", 1. * m, 1. * m, 1. * m);
    G4LogicalVolume* wl = new G4LogicalVolume(wb, mat, "World");
    G4VPhysicalVolume* wp = new G4PVPlacement(nullptr, G4ThreeVector(), wl, "World", nullptr, false, 0);
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

  G4Gamma::Gamma();
  const G4ParticleDefinition* part =
      pdg == 11 ? (const G4ParticleDefinition*)G4Electron::Electron()
                : (const G4ParticleDefinition*)G4Positron::Positron();
  G4DataVector cuts(std::max<size_t>(G4Material::GetNumberOfMaterials(), 1), gcut * MeV);
  // LEAKED ON PURPOSE (destructors race the static teardown; see radharm_g4driver)
  auto* sb = new SBSampler();
  auto* rel = new RelSampler();
  sb->Initialise(part, cuts);
  rel->Initialise(part, cuts);
  sb->SetUseBaseMaterials(false);
  rel->SetUseBaseMaterials(false);

  // SAMPLE MODE (--sample N): the model's own SampleSecondaries at the first
  // --tkin, primary along +z -- the joint law of the emission: photon energy
  // k, photon polar angle, and the primary's post-emission kinetic energy and
  // polar angle (momentum conservation against the photon, as the model does
  // it).  Writes <prefix>.samp.bin: N x 4 float64 (k, theta_gamma, T', theta_e)
  // [MeV, rad] and prints the angular generator's name.
  const long nsamp = (long)argd(argc, argv, "--sample", 0);
  if (nsamp > 0) {
    CLHEP::HepJamesRandom eng((long)argd(argc, argv, "--seed", 20260927));
    G4Random::setTheEngine(&eng);
    const double T = tkin.at(0) * MeV;
    G4VEmModel* mo = (T < GeV) ? static_cast<G4VEmModel*>(sb) : static_cast<G4VEmModel*>(rel);
    G4ParticleChangeForLoss* pc = (T < GeV) ? sb->GetParticleChangeForLoss() : rel->GetParticleChangeForLoss();
    // a local couple: with no element selectors (the standalone couple table
    // is empty) SelectTargetAtom draws the atom from the material's own
    // per-element cross sections, which is what the selectors tabulate
    G4ProductionCuts* spc = new G4ProductionCuts();
    spc->SetProductionCut(1. * um);
    const G4MaterialCutsCouple* couple = new G4MaterialCutsCouple(mat, spc);
    printf("angular generator: %s\n", mo->GetAngularDistribution() ? mo->GetAngularDistribution()->GetName().c_str() : "none");
    std::vector<double> buf;
    buf.reserve(4 * nsamp);
    std::vector<G4DynamicParticle*> vdp;
    long got = 0;
    while (got < nsamp) {
      G4DynamicParticle dp(part, G4ThreeVector(0., 0., 1.), T);
      pc->SetProposedKineticEnergy(T);
      pc->SetProposedMomentumDirection(G4ThreeVector(0., 0., 1.));
      vdp.clear();
      mo->SampleSecondaries(&vdp, couple, &dp, gcut * MeV, T);
      if (vdp.empty()) continue;
      const G4DynamicParticle* g = vdp[0];
      const double k = g->GetKineticEnergy();
      const double thg = g->GetMomentumDirection().theta();
      const double tp = pc->GetProposedKineticEnergy();
      const double the = pc->GetProposedMomentumDirection().theta();
      buf.push_back(k / MeV);
      buf.push_back(thg);
      buf.push_back(tp / MeV);
      buf.push_back(the);
      for (auto* d : vdp) delete d;
      ++got;
    }
    FILE* fb = fopen((out + ".samp.bin").c_str(), "wb");
    fwrite(buf.data(), sizeof(double), buf.size(), fb);
    fclose(fb);
    printf("wrote %s.samp.bin (%ld samples at T = %g MeV)\n", out.c_str(), got, T / MeV);
    fflush(stdout);
    std::_Exit(0);
  }

  std::vector<double> ve(nbin + 1);
  for (int b = 0; b <= nbin; ++b) ve[b] = vmin * std::pow(1. / vmin, double(b) / nbin);

  FILE* f = fopen((out + ".txt").c_str(), "w");
  fprintf(f, "# ebrem_g4driver pdg rho[g/cm3] gcut[MeV] nbin vmin ncomp\n");
  fprintf(f, "input %d %.10g %.10g %d %.10g %zu\n", pdg, rho, gcut, nbin, vmin, comp.size());
  fprintf(f, "vedges");
  for (double v : ve) fprintf(f, " %.12e", v);
  fprintf(f, "\n");
  for (double T : tkin) {
    const double e = T * MeV;
    const double gc = gcut * MeV;
    G4VEmModel* ms[2] = {sb, rel};
    fprintf(f, "tkin %.12e\n", T);
    for (int im = 0; im < 2; ++im) {
      G4VEmModel* mo = ms[im];
      const double dtot = mo->ComputeDEDXPerVolume(mat, part, e, e) / (MeV / mm);
      const double dsub = mo->ComputeDEDXPerVolume(mat, part, e, std::min(gc, e)) / (MeV / mm);
      std::vector<double> nb(nbin, 0.), eb(nbin, 0.);
      double dlo = dsub;
      for (int b = 0; b < nbin; ++b) {
        const double klo = std::max(ve[b] * e, gc);
        const double khi = std::min(ve[b + 1] * e, e);
        if (khi <= klo) continue;
        nb[b] = mo->CrossSectionPerVolume(mat, part, e, klo, khi) / (1. / mm);
        const double dhi = mo->ComputeDEDXPerVolume(mat, part, e, khi) / (MeV / mm);
        eb[b] = dhi - dlo;
        dlo = dhi;
      }
      fprintf(f, "%s dedx %.12e sub %.12e\n", im == 0 ? "sb" : "rel", dtot, dsub);
      fprintf(f, "%s_n", im == 0 ? "sb" : "rel");
      for (double x : nb) fprintf(f, " %.10e", x);
      fprintf(f, "\n%s_e", im == 0 ? "sb" : "rel");
      for (double x : eb) fprintf(f, " %.10e", x);
      fprintf(f, "\n");
    }
  }
  fclose(f);
  printf("wrote %s.txt (%zu energies, %d bins)\n", out.c_str(), tkin.size(), nbin);
  fflush(stdout);
  std::_Exit(0);
}
