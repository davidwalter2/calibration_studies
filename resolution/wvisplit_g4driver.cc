// wvisplit_g4driver -- reproduce, with GEANT4's OWN cross-section object, what
// `G4WentzelVIModel::SampleScattering` actually draws for one step, so that the
// INTERNAL Gaussian/single-scattering split can be modelled offline and
// validated against the sampler rather than transcribed from the manual.
//
// WHAT THE SPLIT IS  (G4WentzelVIModel.cc, 11.2.p02, lines 397-700)
// -----------------------------------------------------------------
// ComputeTrueStepLength:
//     cosThetaMin  = 1 - ssFactor * tPathLength / lambda_transport    (ssFactor = 1.25)
//     cross        = ComputeTransportXSectionPerVolume(cosThetaMin)   RESTRICTED
//     lambdaeff    = 1/cross
//     xtsec        = n * [ ComputeNuclearCrossSection (cosThetaMin, cosTetMaxNuc)
//                        + ComputeElectronCrossSection(cosThetaMin, cosTetMaxElec) ]
// SampleScattering:
//     z0 = tPathLength/(2 lambdaeff)
//     prob2 = 0   ALWAYS  (useSecondMoment = false AND
//                          G4WentzelOKandVIxSection::ComputeSecondTransportMoment
//                          returns 0.0 unconditionally)  -> the Gamma(2,2)
//                          branch is dead code
//     z ~ Exp(mean z0) truncated at z <= 1 ;  cost = 1 - 2z
//     plus explicit single scatters at rate xtsec, each from
//     wokvi->SampleSingleScattering(cosThetaMin, cosTetM, elecRatio)
//
// So a step's deflection is [an EXACT Gaussian in the projected angle, matched
// to the RESTRICTED first transport moment] (x) [a compound Poisson of explicit
// single scatters above theta_min].  The offline transform is a compound
// Poisson over the WHOLE range.  The two differ only in the sub-theta_min
// SHAPE -- the variance is identical by construction -- and this driver
// measures that difference with G4's own code on both sides.
//
// TWO ARMS, sampled with the SAME G4 objects:
//     split : cosThetaMin as G4 computes it        -- what WentzelVI does
//     full  : cosThetaMin = 1                      -- every scatter explicit,
//                                                     i.e. the pure compound
//                                                     Poisson the model has
// The difference between the two empirical CFs IS the split, measured, with no
// analytic step in between.
//
// OUTPUT (all on stdout, one record per line, parsed by wvisplit.py)
//   PARAM ...            G4EmParameters, as in msterms_g4driver
//   MAT   ...            material constants
//   STATE <lab> ...      the wokvi state + lambda/cosThetaMin/xtsec/z0
//   MOM   <lab> <arm> .. sampled moments (mean 1-cos, <thx^2>, <thx^4>, <nss>)
//   ECF   <lab> <arm> <s> <re> <err>   empirical CF of the PROJECTED angle,
//                                      estimated as <J0(s*theta)> (exact for an
//                                      isotropic 2D deflection)
//
// Usage:
//   wvisplit_g4driver --Z 8 --A 16 --rho 9 --p 3136 --tcut 0.0094862 \
//                     --steplen 0.108854 --n 4000000 --nfull 1000000

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#include "G4Material.hh"
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
#include "G4IonisParamMat.hh"
#include "G4EmParameters.hh"
#include "G4WentzelOKandVIxSection.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4TransportationManager.hh"
#include "Randomize.hh"

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

// Everything in G4WentzelOKandVIxSection that is not part of the public API is
// `protected`, so a derived class is the supported way to read it.  Nothing is
// overridden -- this ONLY exposes.  Identical pattern to msterms_g4driver.cc.
class Wokvi : public G4WentzelOKandVIxSection {
public:
  explicit Wokvi(G4bool comb) : G4WentzelOKandVIxSection(comb) {}
  double screenZ_() const { return screenZ; }
  double formfactA_() const { return formfactA; }
  double kinFactor_() const { return kinFactor; }
  double factB_() const { return factB; }
  double factD_() const { return factD; }
  double invbeta2_() const { return invbeta2; }
  double mom2_() const { return mom2; }
  double spin_() const { return spin; }
  double cosTetMaxNuc_() const { return cosTetMaxNuc; }
  double cosTetMaxElec_() const { return cosTetMaxElec; }
  const void* mott_() const { return fMottXSection; }
};
}  // namespace

int main(int argc, char** argv) {
  const double Z = argd(argc, argv, "--Z", 8.0);
  const double A = argd(argc, argv, "--A", 16.0);
  const double rho = argd(argc, argv, "--rho", 9.0);
  const double pmom = argd(argc, argv, "--p", 3136.0);        // MeV/c
  const double tcut = argd(argc, argv, "--tcut", 0.0094862);  // MeV
  // The PATH is what the model treats as one record (a layer crossing); it is
  // divided into G4 steps by delta-ray emission, `ndelta` of them on average,
  // exponentially distributed and truncated by the layer boundary.  Both
  // numbers are MEASURED (wvisplit.py steps): the census counts the primary's
  // ionization secondaries inside r < 107 cm, 118.2 (mu) to 127.9 (p) per event
  // over 14 layer crossings.  ndelta = 0 collapses to a single step of `pathlen`.
  const double pathcm = argd(argc, argv, "--pathlen", 0.10450);  // cm
  const double ndelta = argd(argc, argv, "--ndelta", 8.444);
  const long nsplit = (long)argd(argc, argv, "--n", 4000000);
  const long nfull = (long)argd(argc, argv, "--nfull", 1000000);
  const long seed = (long)argd(argc, argv, "--seed", 12345);

  G4Random::setTheSeed(seed);

  G4Material* mat = new G4Material("ToyLayerMat", Z, A * g / mole, rho * g / cm3);
  {
    G4Box* wb = new G4Box("World", 1. * m, 1. * m, 1. * m);
    G4LogicalVolume* wl = new G4LogicalVolume(wb, mat, "World");
    G4VPhysicalVolume* wp =
        new G4PVPlacement(nullptr, G4ThreeVector(), wl, "World", nullptr, false, 0);
    G4TransportationManager::GetTransportationManager()->SetWorldForTracking(wp);
  }

  G4EmParameters* par = G4EmParameters::Instance();
  printf("PARAM MscThetaLimit %.17g MuHadLateralDisplacement %d UseMottCorrection %d "
         "MscMuHadRangeFactor %.17g\n",
         par->MscThetaLimit(), (int)par->MuHadLateralDisplacement(),
         (int)par->UseMottCorrection(), par->MscMuHadRangeFactor());

  const double natoms = mat->GetTotNbOfAtomsPerVolume();  // 1/mm3 (G4 internal)
  const double tpath = pathcm * cm;                       // G4 internal length
  const double lmean = (ndelta > 0.0) ? tpath / ndelta : tpath;
  printf("MAT Z %g A %g rho %g natoms_percm3 %.17g X0_cm %.17g pathlen_cm %.17g "
         "ndelta %.17g\n",
         Z, A, rho, natoms * cm3, mat->GetRadlen() / cm, pathcm, ndelta);

  // The sampled deflection is accumulated as a LOG HISTOGRAM of the total polar
  // angle rather than as <J0(s theta)> on a fixed s grid.  Two reasons, both
  // practical: the empirical CF can then be formed offline on ANY s grid (and
  // its multinomial error with it), and the cost per sample drops from 24
  // Bessel evaluations to one log -- which is what makes 1e8 samples per
  // species affordable.  12000 bins over 1e-9..pi is dln(theta) = 1.8e-3, so
  // the quadrature error on J0 is < 1e-5 at every s used here.
  const int NB = 12000;
  const double LB0 = std::log(1e-9), LB1 = std::log(CLHEP::pi);
  const double dLB = (LB1 - LB0) / NB;
  std::vector<double> hist(NB + 2, 0.0);   // [0] underflow, [NB+1] overflow

  for (int i = 0; i < 8; ++i) {
    G4ParticleDefinition* part = particleOf(kPdg[i]);
    const double mass = part->GetPDGMass();
    const double etot = std::sqrt(pmom * pmom + mass * mass);
    const double ekin = etot - mass;

    Wokvi w(true);
    w.Initialise(part, -1.0);                    // cosThetaLim = the model default
    w.SetupKinematic(ekin, mat);
    w.SetupTarget((G4int)(Z + 0.5), tcut);

    const double cosNuc = w.cosTetMaxNuc_();
    const double cosElec = w.cosTetMaxElec_();

    // --- exactly G4WentzelVIModel::ComputeTrueStepLength -------------------
    // lambda_transport at the FULL nuclear range: this is what
    // G4VMscModel::GetTransportMeanFreePath returns, since
    // G4WentzelVIModel::ComputeCrossSectionPerAtom is
    // wokvi->ComputeTransportCrossSectionPerAtom(SetupTarget(...)).
    const double crossFull = natoms * w.ComputeTransportCrossSectionPerAtom(cosNuc);
    const double lambdaFull = 1.0 / crossFull;
    const double ssFactor = 1.25;

    // representative step (the MEAN of the exponential step distribution), for
    // the printed state; the sampler recomputes all of it per step.
    const double cmin0 = 1.0 - ssFactor * lmean / lambdaFull;
    const double lameff0 = 1.0 / (natoms * w.ComputeTransportCrossSectionPerAtom(cmin0));
    const double nsec0 = w.ComputeNuclearCrossSection(cmin0, cosNuc);
    const double esec0 = w.ComputeElectronCrossSection(cmin0, cosElec);

    // the FULL arm: every scatter explicit, cosThetaMin = 1
    const double nucsecF = w.ComputeNuclearCrossSection(1.0, cosNuc);
    const double esecF = w.ComputeElectronCrossSection(1.0, cosElec);
    const double xtsecF = natoms * (nucsecF + esecF);
    const double elecRatioF = (nucsecF + esecF > 0.0) ? esecF / (nucsecF + esecF) : 0.0;

    printf("STATE %s mass %.17g ekin %.17g mom2 %.17g invbeta2 %.17g spin %.17g "
           "screenZ %.17g formfactA %.17g kinFactor %.17g factB %.17g factD %.17g "
           "omcNuc %.17g omcElec %.17g natomsG4 %.17g tpathG4 %.17g lmeanG4 %.17g "
           "lambdaFull %.17g lambdaEff0 %.17g omcThetaMin0 %.17g z00 %.17g "
           "xtsec0 %.17g elecRatio0 %.17g nssFull %.17g elecRatioFull %.17g mott %d\n",
           kLab[i], mass, ekin, w.mom2_(), w.invbeta2_(), w.spin_(), w.screenZ_(),
           w.formfactA_(), w.kinFactor_(), w.factB_(), w.factD_(), 1.0 - cosNuc,
           1.0 - cosElec, natoms, tpath, lmean, lambdaFull, lameff0, 1.0 - cmin0,
           0.5 * lmean / lameff0, natoms * (nsec0 + esec0),
           (nsec0 + esec0 > 0.0) ? esec0 / (nsec0 + esec0) : 0.0, tpath * xtsecF,
           elecRatioF, w.mott_() != nullptr ? 1 : 0);

    for (int arm = 0; arm < 2; ++arm) {
      const bool isSplit = (arm == 0);
      const long nsamp = isSplit ? nsplit : nfull;
      if (nsamp <= 0)
        continue;

      std::fill(hist.begin(), hist.end(), 0.0);
      double s1 = 0.0, s2 = 0.0, s4 = 0.0, snss = 0.0, nstp = 0.0;
      CLHEP::HepRandomEngine* eng = G4Random::getTheEngine();

      for (long ev = 0; ev < nsamp; ++ev) {
        G4ThreeVector dir(0.0, 0.0, 1.0);
        long nss = 0;
        double remain = tpath;
        while (remain > 0.0) {
          // one G4 step: length = min(next delta-ray free flight, path left)
          double L = remain;
          if (isSplit && ndelta > 0.0) {
            const double g = -G4Log(eng->flat()) * lmean;
            if (g < remain) L = g;
          }
          remain -= L;
          nstp += 1.0;

          double ct1 = 1.0, rate = xtsecF, er = elecRatioF, zz0 = 0.0;
          if (isSplit) {
            ct1 = 1.0 - ssFactor * L / lambdaFull;
            if (ct1 > cosNuc) {
              const double lameff =
                  1.0 / (natoms * w.ComputeTransportCrossSectionPerAtom(ct1));
              const double ns = w.ComputeNuclearCrossSection(ct1, cosNuc);
              const double es = w.ComputeElectronCrossSection(ct1, cosElec);
              rate = natoms * (ns + es);
              er = (ns + es > 0.0) ? es / (ns + es) : 0.0;
              zz0 = 0.5 * L / lameff;
            } else {  // singleScatteringMode: no Gaussian at all
              rate = xtsecF;
              er = elecRatioF;
              zz0 = 0.0;
            }
          }
          // explicit single scatters: exactly the x1/x2 bookkeeping of
          // SampleScattering with nMscSteps = 1.
          if (rate > 0.0) {
            double x2 = L;
            double x1 = -G4Log(eng->flat()) / rate;
            while (x1 <= x2) {
              G4ThreeVector t = w.SampleSingleScattering(ct1, cosNuc, er);
              t.rotateUz(dir);
              dir = t;
              ++nss;
              x2 -= x1;
              x1 = -G4Log(eng->flat()) / rate;
            }
          }
          // the msc kick (absent in the `full` arm, where zz0 = 0)
          if (zz0 > 0.0) {
            double z;
            do {
              z = -G4Log(eng->flat()) * zz0;
            } while (z > 1.0);
            double cost = 1.0 - 2.0 * z;
            if (cost > 1.0) cost = 1.0;
            else if (cost < -1.0) cost = -1.0;
            const double sint = std::sqrt((1.0 - cost) * (1.0 + cost));
            const double phi = CLHEP::twopi * eng->flat();
            G4ThreeVector t(sint * std::cos(phi), sint * std::sin(phi), cost);
            t.rotateUz(dir);
            dir = t;
          }
        }
        const double cz = std::min(1.0, std::max(-1.0, dir.z()));
        const double th = std::acos(cz);
        s1 += 1.0 - cz;
        s2 += 0.5 * th * th;               // projected second moment
        s4 += 0.375 * th * th * th * th;   // projected fourth moment
        snss += (double)nss;
        int ib = (th > 0.0) ? (int)((std::log(th) - LB0) / dLB) + 1 : 0;
        if (ib < 0) ib = 0;
        else if (ib > NB + 1) ib = NB + 1;
        hist[ib] += 1.0;
      }
      const double N = (double)nsamp;
      printf("MOM %s %s n %ld omc %.17g thx2 %.17g thx4 %.17g nss %.17g nstep %.17g\n",
             kLab[i], isSplit ? "split" : "full", nsamp, s1 / N, s2 / N, s4 / N,
             snss / N, nstp / N);
      printf("HIST %s %s %d %.17g %.17g", kLab[i], isSplit ? "split" : "full", NB, LB0,
             dLB);
      for (int b = 0; b <= NB + 1; ++b)
        if (hist[b] > 0.0)
          printf(" %d:%.17g", b, hist[b]);
      printf("\n");
      fflush(stdout);
    }
  }
  fflush(stdout);
  // Leak everything: the destructors race static teardown after all output.
  return 0;
}
