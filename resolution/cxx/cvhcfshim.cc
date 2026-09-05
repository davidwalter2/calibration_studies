// C-ABI bridge from python to `cvhcf` -- the IN-MAKER resolution-CF exponents.
//
// WHY THIS EXISTS.  `cvhcf::msBlock` / `ioniBlock` / `radBlock` /
// `trackExponents` are a port of the offline reference
// (cf_track_resolution.ms_step_exponent / ioni_step_exponent,
// cf_brems_exact.rad_exponent, cf_delta_ray.delta_step_exponent, and
// `extract`'s pooling).  A port is only worth anything if it is MEASURED
// against the thing it ports, on the records that actually occur, and doing
// that through cmsRun means a 6-minute build-and-run for every iteration.
// This compiles the maker's OWN translation unit -- unmodified, straight out
// of the CMSSW tree -- into a .so the analysis venv can call directly, exactly
// as `cgfshim` already does for `cvhcgf`.
//
// The arrays crossing the boundary are the FLOATS the tree carries, so the
// shim cannot introduce a conversion the maker does not have.
#include "TrackPropagation/Geant4e/interface/CvhCfExponents.h"

#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

extern "C" {

int cvhcf_ntau() { return cvhcf::kNTau; }

void cvhcf_tau_grid(double *out) { std::memcpy(out, cvhcf::tauGrid(), cvhcf::kNTau * sizeof(double)); }

int cvhcf_model_tag(char *buf, int n) {
  const std::string &t = cvhcf::modelTag();
  const int m = static_cast<int>(t.size());
  if (buf != nullptr && n > 0) {
    const int k = (m < n - 1) ? m : n - 1;
    std::memcpy(buf, t.data(), k);
    buf[k] = '\0';
  }
  return m;
}

// One pooled MS block. `sms` and `sdel` are kNTau doubles each and are
// ACCUMULATED into (pass zeroed arrays for a single block). `sdel` may be
// null.
int cvhcf_ms_block(const float *rows, int stride, int n, double wstd, double *sms, double *sdel) {
  if (rows == nullptr || sms == nullptr || n < 0 || stride < 8)
    return -1;
  cvhcf::msBlock(rows, stride, n, wstd, sms, sdel);
  return 0;
}

double cvhcf_ioni_sq2(const float *rows, int stride, int n, const float *qsc, int nqsc) {
  return cvhcf::ioniSq2(rows, stride, n, qsc, nqsc);
}

int cvhcf_ioni_block(const float *rows, int stride, int n, double wstdSigned, double *sre, double *sim) {
  if (rows == nullptr || sre == nullptr || sim == nullptr)
    return -1;
  if (stride != 11 && stride != 13)
    return -2;
  cvhcf::ioniBlock(rows, stride, n, wstdSigned, sre, sim);
  return 0;
}

int cvhcf_rad_block(const float *rows,
                    int stride,
                    int n,
                    const float *spec,
                    const float *vgrid,
                    int nv,
                    double wstdSigned,
                    double *sre,
                    double *sim) {
  if (rows == nullptr || spec == nullptr || vgrid == nullptr || sre == nullptr || sim == nullptr)
    return -1;
  cvhcf::radBlock(rows, stride, n, spec, vgrid, nv, wstdSigned, sre, sim);
  return 0;
}

// The whole track / candidate, pooling included.  `out` receives
// 6 * kNTau doubles: ms, del, ioRe, ioIm, radRe, radIm.  `aux` receives
// [ok, vgauss, nblockms, nblockioni, npooled].
int cvhcf_track(const unsigned int *resglobidx,
                const int *resfamily,
                const float *resvarv,
                int nres,
                const unsigned int *msidx,
                const float *msv,
                int nms,
                int msstride,
                const unsigned int *ioidx,
                const float *iov,
                int nio,
                int iostride,
                const unsigned int *qsidx,
                const float *qsv,
                int nqs,
                const unsigned int *ridx,
                const float *rv,
                int nrad,
                int radstride,
                const float *rspec,
                const float *rvgrid,
                int radnv,
                double sigma,
                double ioniSign,
                int wantDelta,
                double *out,
                double *aux) {
  cvhcf::TrackInput in;
  in.resglobidx = resglobidx;
  in.resfamily = resfamily;
  in.resvarv = resvarv;
  in.nres = nres;
  in.ms = {msidx, msv, nms, msstride};
  in.ioni = {ioidx, iov, nio, iostride};
  in.qsc = {qsidx, qsv, nqs, 2};
  in.rad = {ridx, rv, nrad, radstride};
  in.radspec = rspec;
  in.radvgrid = rvgrid;
  in.radnv = radnv;
  in.sigma = sigma;
  in.ioniSign = ioniSign;
  in.wantDelta = wantDelta != 0;
  cvhcf::TrackResult r;
  cvhcf::trackExponents(in, r);
  const int nt = cvhcf::kNTau;
  std::memcpy(out + 0 * nt, r.S.ms.data(), nt * sizeof(double));
  std::memcpy(out + 1 * nt, r.S.del.data(), nt * sizeof(double));
  std::memcpy(out + 2 * nt, r.S.ioRe.data(), nt * sizeof(double));
  std::memcpy(out + 3 * nt, r.S.ioIm.data(), nt * sizeof(double));
  std::memcpy(out + 4 * nt, r.S.radRe.data(), nt * sizeof(double));
  std::memcpy(out + 5 * nt, r.S.radIm.data(), nt * sizeof(double));
  aux[0] = r.ok ? 1. : 0.;
  aux[1] = r.vgauss;
  aux[2] = r.nblockms;
  aux[3] = r.nblockioni;
  aux[4] = r.npooled;
  return r.ok ? 0 : 1;
}

}  // extern "C"
