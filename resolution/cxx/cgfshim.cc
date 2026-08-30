// C-ABI bridge from the offline CF (python) to the FIT's own CGF block.
//
// WHY THIS EXISTS. `cf_track_resolution.ioni_step_exponent` and
// `cvhcgf::blockExponent` compute the same object -- the centred log-CF
// exponent of one pooled ionization block -- from the same `ioniurbanv`
// records, in two languages. Two implementations of one formula is the exact
// hazard CGFQoPBlock.cc already documents for `toIoniStep` ("Two copies is how
// the two would come to disagree about whether `a3` is a count or an energy,
// or whether `scaling` has been applied -- both silent, both ~1e-5"), and here
// the divergence would sit between the estimator the fit runs and the model
// every closure number is quoted against.
//
// It is also ~10-50x faster: the python path is an 80-term complex series over
// a (nsteps x 448) array with a full np.max reduction inside the loop, and it
// is the measured bottleneck of the extraction stage.
//
// THE CONTRACT is `ioni_step_exponent(steps, wstd, tau)`:
//   steps  (ns, stride) float64, the raw `ioniurbanv` records, stride 11 or 13
//   wstd   scalar, w/sigma (the fit-unit conversion and standardization)
//   tau    (nt,) float64
//   ->     (nt,) complex128
// The column map is the exporter's, and is duplicated in the python docstring:
//   0 regime, 1 gsig2, 2 a1, 3 e1, 4 a2, 5 e2, 6 a3, 7 e0r, 8 tmaxr,
//   9 scaling, 10 g [qop per keV], 11 beta2, 12 etot        (11,12 iff stride>=13)
//
// The per-step scaling below is `toIoniStep`'s, deliberately: e1/e2/e0/tmax
// take `scaling`, and a3 takes it ONLY in regime >= 2 where it is xi (an
// energy) rather than a collision count.
#include "TrackPropagation/Geant4e/interface/CGFQoPBlock.h"

#include <complex>
#include <cstddef>
#include <vector>

extern "C" {

// Returns 0 on success, or a negative code the python side turns into an
// exception. Writes 2*nt doubles (re, im interleaved) into `out`.
int cvhcgf_ioni_step_exponent(const double *steps,
                              int ns,
                              int stride,
                              double wstd,
                              const double *tau,
                              int nt,
                              int kokNbin,
                              double *out) {
  if (steps == nullptr || tau == nullptr || out == nullptr)
    return -1;
  if (ns < 0 || nt <= 0 || (stride != 11 && stride != 13))
    return -2;

  cvhcgf::Block blk;
  blk.ioni.reserve(static_cast<std::size_t>(ns));
  for (int i = 0; i < ns; ++i) {
    const double *r = steps + static_cast<std::size_t>(i) * stride;
    const int regime = static_cast<int>(r[0]);
    const double gam = r[9];

    cvhcgf::IoniStep s;
    s.regime = regime;
    s.gsig2 = r[1];
    s.a1 = r[2];
    s.e1 = r[3] * gam;
    s.a2 = r[4];
    s.e2 = r[5] * gam;
    // count in regime 1, xi (an energy) in 2/3 -- the one asymmetry that must
    // match `toIoniStep` or the two paths differ silently at the 1e-5 level.
    s.a3 = (regime >= 2) ? r[6] * gam : r[6];
    s.e0 = r[7] * gam;
    s.tmax = r[8] * gam;
    // `g` is qop per keV in the record; the block wants per MeV.
    s.gs = wstd * r[10] * 1e-3;
    if (stride >= 13) {
      s.beta2 = r[11];
      s.etot = r[12];
    } else if (regime >= 2) {
      // Same refusal as the python: beta^2 and E are not recoverable from tmax
      // without the particle mass, so a regime-2 record at stride 11 is a
      // mis-exported file, not something to approximate.
      return -3;
    }
    s.kokNbin = (regime >= 2) ? kokNbin : 0;
    blk.ioni.push_back(s);
  }

  for (int j = 0; j < nt; ++j) {
    const std::complex<double> z = cvhcgf::blockExponent(blk, tau[j]);
    out[2 * j] = z.real();
    out[2 * j + 1] = z.imag();
  }
  return 0;
}

// Exposed so the validation can drive the kernel directly, one step at a time,
// across both series branches (|a*w| <= 2 and beyond) without having to
// synthesise whole records.
int cvhcgf_block_exponent_one(int regime,
                              double gsig2,
                              double a1,
                              double e1,
                              double a2,
                              double e2,
                              double a3,
                              double e0,
                              double tmax,
                              double beta2,
                              double etot,
                              double gs,
                              int kokNbin,
                              const double *tau,
                              int nt,
                              double *out) {
  if (tau == nullptr || out == nullptr || nt <= 0)
    return -1;
  cvhcgf::Block blk;
  cvhcgf::IoniStep s;
  s.regime = regime;
  s.gsig2 = gsig2;
  s.a1 = a1;
  s.e1 = e1;
  s.a2 = a2;
  s.e2 = e2;
  s.a3 = a3;
  s.e0 = e0;
  s.tmax = tmax;
  s.beta2 = beta2;
  s.etot = etot;
  s.gs = gs;
  s.kokNbin = kokNbin;
  blk.ioni.push_back(s);
  for (int j = 0; j < nt; ++j) {
    const std::complex<double> z = cvhcgf::blockExponent(blk, tau[j]);
    out[2 * j] = z.real();
    out[2 * j + 1] = z.imag();
  }
  return 0;
}

}  // extern "C"
