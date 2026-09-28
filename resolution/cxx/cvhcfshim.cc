// C-ABI bridge from python to `cvhcf` -- the IN-MAKER resolution-CF exponents.
//
// WHY THIS EXISTS.  `cvhcf`'s row functions (`ioniRows`, `msRows`,
// `refineSpectra`, `radRows`, `knockonRows`, `nucelRows`) and
// `trackExponents` are a port of the offline reference (`cf_rows` and what it
// calls, `nucel_tables.Table.rows`, and `cf_rows.fit_families` with
// `cf_track_resolution.extract`'s pooling).  A port is only worth anything if
// it is MEASURED against the thing it ports, on the records that actually
// occur, and doing that through cmsRun means a 6-minute build-and-run for
// every iteration; and a closure that evaluates its model through this .so
// tests the model the makers export.
// This compiles the maker's OWN translation unit -- unmodified, straight out
// of the CMSSW tree -- into a .so the analysis venv can call directly, exactly
// as `cgfshim` already does for `cvhcgf`.
//
// The arrays crossing the boundary are the FLOATS the tree carries, so the
// shim cannot introduce a conversion the maker does not have.
#include "TrackPropagation/Geant4e/interface/CvhCfExponents.h"

#include <cstddef>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {
  // The nuclear-elastic family's material table on the python side: the
  // `materials` tree of the file being validated (each job's own table), so
  // the shim builds the SAME mixtures the maker built from G4Material.
  struct NucCtx {
    std::map<int, cvhcf::NucelComposition> mats;
    std::unique_ptr<cvhcf::NucelMixtures> mix;
    NucCtx() {
      mix = std::make_unique<cvhcf::NucelMixtures>([this](int idx, cvhcf::NucelComposition &c) {
        auto it = mats.find(idx);
        if (it == mats.end())
          return false;
        c = it->second;
        return true;
      });
    }
  };
  std::string lastError;
  cvhcf::RowConfig &rowConfig() {
    static cvhcf::RowConfig c = cvhcf::productionRowConfig();
    return c;
  }
}  // namespace

extern "C" {

int cvhcf_ntau() { return cvhcf::kNTau; }

// ---- the row functions (cf_rows) -------------------------------------------
//
// The model's switches are the offline modules' globals; the caller hands
// its module state over with `cvhcf_row_config_set` before evaluating, so a
// closure run with a switch changed evaluates the same model on both sides.
// Every row function ACCUMULATES into its outputs and returns 0, or -3 with
// the exception text in `cvhcf_last_error`.

void cvhcf_row_config_set(int knockonJoint,
                          int qopExact,
                          int knockonNPerDec,
                          int knockonNPerDecThin,
                          double knockonThin,
                          double knockonTcut,
                          double pminFrac,
                          int radNsub,
                          double ioniKokoulin,
                          double ioniKokoulinTcut,
                          double ioniA3Scale,
                          double ioniExcScale,
                          double ioniTmaxScale,
                          int nucelRecoil,
                          int nucelJoint) {
  cvhcf::RowConfig &c = rowConfig();
  c.knockonJoint = knockonJoint != 0;
  c.qopExact = qopExact != 0;
  c.knockonNPerDec = knockonNPerDec;
  c.knockonNPerDecThin = knockonNPerDecThin;
  c.knockonThin = knockonThin;
  c.knockonTcut = knockonTcut;
  c.pminFrac = pminFrac;
  c.radNsub = radNsub;
  c.ioniKokoulin = ioniKokoulin;
  c.ioniKokoulinTcut = ioniKokoulinTcut;
  c.ioniA3Scale = ioniA3Scale;
  c.ioniExcScale = ioniExcScale;
  c.ioniTmaxScale = ioniTmaxScale;
  c.nucelRecoil = nucelRecoil != 0;
  c.nucelJoint = nucelJoint != 0;
}

// scipy's j0 / k1 as the model evaluates them (which = 0 / 1), elementwise
void cvhcf_bessel(int which, const double *x, int n, double *out) {
  for (int i = 0; i < n; ++i)
    out[i] = which ? cvhcf::besselK1(x[i]) : cvhcf::besselJ0(x[i]);
}

// the production defaults back
void cvhcf_row_config_reset() { rowConfig() = cvhcf::productionRowConfig(); }

int cvhcf_ioni_rows(const double *tau, int nt, const double *rows, int stride, int n, const double *wq, double *sre,
                    double *sim) {
  try {
    cvhcf::ioniRows(tau, nt, rows, stride, n, wq, rowConfig(), sre, sim);
  } catch (std::exception &ex) {
    lastError = ex.what();
    return -3;
  }
  return 0;
}

int cvhcf_ms_rows(const double *tau, int nt, const double *rows, int stride, int n, const int *rid, const double *wb,
                  const double *frac, int ne, double scale, double *s) {
  try {
    cvhcf::msRows(tau, nt, rows, stride, n, rid, wb, frac, ne, scale, rowConfig(), s);
  } catch (std::exception &ex) {
    lastError = ex.what();
    return -3;
  }
  return 0;
}

// `vf` must hold (nv - 1) * nsub + 1 doubles and `specf` n * 2 * that.
int cvhcf_refine_spectra(const double *recs, int rstride, int n, const double *spec, const double *vg, int nv, int nsub,
                         double *vf, double *specf) {
  try {
    std::vector<double> v, sp;
    cvhcf::refineSpectra(recs, rstride, n, spec, vg, nv, nsub, v, sp);
    std::memcpy(vf, v.data(), v.size() * sizeof(double));
    std::memcpy(specf, sp.data(), sp.size() * sizeof(double));
  } catch (std::exception &ex) {
    lastError = ex.what();
    return -3;
  }
  return 0;
}

int cvhcf_rad_rows(const double *tau, int nt, const double *recs, int rstride, int n, const double *spec,
                   const double *vg, int nv, const int *rid, const double *wq, const double *wb, const double *frac,
                   int ne, int exactQop, double *sre, double *sim) {
  try {
    cvhcf::radRows(tau, nt, recs, rstride, n, spec, vg, nv, rid, wq, wb, frac, ne, exactQop != 0, sre, sim);
  } catch (std::exception &ex) {
    lastError = ex.what();
    return -3;
  }
  return 0;
}

// part: 0 all, 1 map, 2 joint
int cvhcf_knockon_rows(const double *tau, int nt, const double *rows, int stride, int n, const int *rid,
                       const double *wq, const double *wb, const double *frac, int ne, int part, double *sre,
                       double *sim) {
  try {
    cvhcf::knockonRows(tau, nt, rows, stride, n, rid, wq, wb, frac, ne, static_cast<cvhcf::KnockonPart>(part),
                       rowConfig(), sre, sim);
  } catch (std::exception &ex) {
    lastError = ex.what();
    return -3;
  }
  return 0;
}

void cvhcf_tau_grid(double *out) { std::memcpy(out, cvhcf::tauGrid(), cvhcf::kNTau * sizeof(double)); }

int cvhcf_model_tag(char *buf, int n) {
  const std::string t = cvhcf::modelTag(rowConfig(), false);
  const int m = static_cast<int>(t.size());
  if (buf != nullptr && n > 0) {
    const int k = (m < n - 1) ? m : n - 1;
    std::memcpy(buf, t.data(), k);
    buf[k] = '\0';
  }
  return m;
}

double cvhcf_ioni_sq2(const float *rows, int stride, int n, const float *qsc, int nqsc) {
  return cvhcf::ioniSq2(rows, stride, n, qsc, nqsc);
}

namespace {
  // The families of a result, in the order the readers expect:
  // ms, ioRe, ioIm, radRe, radIm, kxRe, kxIm, kjRe, kjIm (kNTau each).
  constexpr int kNFam = 9;
  void copyFamilies(const cvhcf::Exponents &S, double *out) {
    const int nt = cvhcf::kNTau;
    const std::array<double, cvhcf::kNTau> *fam[kNFam] = {
        &S.ms, &S.ioRe, &S.ioIm, &S.radRe, &S.radIm, &S.kxRe, &S.kxIm, &S.kjRe, &S.kjIm};
    for (int f = 0; f < kNFam; ++f)
      std::memcpy(out + f * nt, fam[f]->data(), nt * sizeof(double));
  }

  void fillInput(cvhcf::TrackInput &in,
                 const unsigned int *resglobidx,
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
                 int wantGroups) {
    in.resglobidx = resglobidx;
    in.resfamily = resfamily;
    in.resvarv = resvarv;
    in.nres = nres;
    in.ms = {msidx, msv, nms, msstride};
    in.ms.groupCol = msstride >= 10 ? 9 : -1;
    in.ioni = {ioidx, iov, nio, iostride};
    in.ioni.groupCol = iostride >= 12 ? iostride - 1 : -1;
    in.qsc = {qsidx, qsv, nqs, 2};
    in.rad = {ridx, rv, nrad, radstride};
    in.rad.groupCol = radstride >= 12 ? radstride - 1 : -1;
    in.radspec = rspec;
    in.radvgrid = rvgrid;
    in.radnv = radnv;
    in.sigma = sigma;
    in.ioniSign = ioniSign;
    in.wantGroups = wantGroups != 0;
    in.rowConfig = &rowConfig();
  }

  // groups: `gid` gets up to `maxg` group ids, `gfam` kNFam * kNTau per group
  // plus [vqms, vqio] in `gvq`; returns the group count
  int copyGroups(const cvhcf::TrackResult &r, int maxg, int *gid, double *gfam, double *gvq) {
    const int ng = static_cast<int>(r.groups.size());
    for (int g = 0; g < ng && g < maxg; ++g) {
      gid[g] = r.groups[g].group;
      copyFamilies(r.groups[g].S, gfam + static_cast<std::size_t>(g) * kNFam * cvhcf::kNTau);
      gvq[2 * g] = r.groups[g].vqms;
      gvq[2 * g + 1] = r.groups[g].vqio;
    }
    return ng;
  }
}  // namespace

// The whole track / candidate: cf_rows.fit_families with the pooling.
// `out` receives 9 * kNTau doubles (ms, ioRe, ioIm, radRe, radIm, kxRe, kxIm,
// kjRe, kjIm), `aux` [ok, vgauss, nblockms, nblockioni, npooled]; with
// `wantGroups`, up to `maxg` groups to `gid` / `gfam` / `gvq` and `*ng`.
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
                int wantKnockon,
                int wantGroups,
                double *out,
                double *aux,
                int maxg,
                int *gid,
                double *gfam,
                double *gvq,
                int *ng) {
  cvhcf::TrackInput in;
  fillInput(in, resglobidx, resfamily, resvarv, nres, msidx, msv, nms, msstride, ioidx, iov, nio, iostride, qsidx,
            qsv, nqs, ridx, rv, nrad, radstride, rspec, rvgrid, radnv, sigma, ioniSign, wantGroups);
  in.wantKnockon = wantKnockon != 0;
  cvhcf::TrackResult r;
  try {
    cvhcf::trackExponents(in, r);
  } catch (std::exception &ex) {
    lastError = ex.what();
    return -3;
  }
  copyFamilies(r.S, out);
  aux[0] = r.ok ? 1. : 0.;
  aux[1] = r.vgauss;
  aux[2] = r.nblockms;
  aux[3] = r.nblockioni;
  aux[4] = r.npooled;
  *ng = copyGroups(r, maxg, gid, gfam, gvq);
  return r.ok ? 0 : 1;
}

// ---- the nuclear-elastic family ------------------------------------------

void *cvhcf_nucel_ctx_new() { return new NucCtx(); }

void cvhcf_nucel_ctx_free(void *ctx) { delete static_cast<NucCtx *>(ctx); }

int cvhcf_nucel_add_material(void *ctx, int index, int n, const double *Z, const double *A, const double *W) {
  if (ctx == nullptr || n < 0)
    return -1;
  cvhcf::NucelComposition c;
  c.Z.assign(Z, Z + n);
  c.A.assign(A, A + n);
  c.W.assign(W, W + n);
  static_cast<NucCtx *>(ctx)->mats[index] = c;
  return 0;
}

int cvhcf_last_error(char *buf, int n) {
  const int m = static_cast<int>(lastError.size());
  if (buf != nullptr && n > 0) {
    const int k = (m < n - 1) ? m : n - 1;
    std::memcpy(buf, lastError.data(), k);
    buf[k] = '\0';
  }
  return m;
}

// `cvhcf::nucelRows` (nucel_tables.Table.rows) on `n` msmoliv rows (double,
// `stride` columns) against the context's material table.  `out` receives
// 5 * nt doubles (ang, rec_re, rec_im, jnt_re, jnt_im) and `N[0]` the
// expected collisions; both ACCUMULATE.
int cvhcf_nucel_rows(void *ctx,
                     const double *tau,
                     int nt,
                     const double *rows,
                     int stride,
                     int n,
                     const int *mat,
                     const int *pdg,
                     const double *mass,
                     const int *rid,
                     const double *wb,
                     const double *frac,
                     int ne,
                     const double *wqRow,
                     const double *wbMid,
                     double *out,
                     double *N) {
  if (ctx == nullptr || out == nullptr || N == nullptr)
    return -1;
  try {
    N[0] += cvhcf::nucelRows(tau, nt, rows, stride, n, mat, pdg, mass, rid, wb, frac, ne, wqRow, wbMid, rowConfig(),
                             *static_cast<NucCtx *>(ctx)->mix, out, out + nt, out + 2 * nt, out + 3 * nt,
                             out + 4 * nt);
  } catch (std::exception &ex) {
    lastError = ex.what();
    return -3;
  }
  return 0;
}

// `cvhcf_track` with the nuclear-elastic family on (`msmat` / `mspdg`
// parallel to the MS rows).  Besides `out` / `aux` as `cvhcf_track`:
// `nuc` receives 5 * kNTau doubles (ang, rec_re, rec_im, jnt_re, jnt_im) +
// N; with `wantGroups`, up to `maxg` groups go to `gid` / `gnuc`
// (5 * kNTau + 1 each) and `*ng` is the group count.
int cvhcf_track_nucel(void *ctx,
                      const unsigned int *resglobidx,
                      const int *resfamily,
                      const float *resvarv,
                      int nres,
                      const unsigned int *msidx,
                      const float *msv,
                      int nms,
                      int msstride,
                      const int *msmat,
                      const int *mspdg,
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
                      int wantGroups,
                      double *out,
                      double *aux,
                      double *nuc,
                      int maxg,
                      int *gid,
                      double *gnuc,
                      int *ng) {
  if (ctx == nullptr)
    return -1;
  cvhcf::TrackInput in;
  fillInput(in, resglobidx, resfamily, resvarv, nres, msidx, msv, nms, msstride, ioidx, iov, nio, iostride, qsidx,
            qsv, nqs, ridx, rv, nrad, radstride, rspec, rvgrid, radnv, sigma, ioniSign, wantGroups);
  in.wantKnockon = false;
  in.wantNucel = true;
  in.msmat = msmat;
  in.mspdg = mspdg;
  in.nucel = static_cast<NucCtx *>(ctx)->mix.get();
  cvhcf::TrackResult r;
  try {
    cvhcf::trackExponents(in, r);
  } catch (std::exception &ex) {
    lastError = ex.what();
    return -3;
  }
  const int nt = cvhcf::kNTau;
  copyFamilies(r.S, out);
  aux[0] = r.ok ? 1. : 0.;
  aux[1] = r.vgauss;
  aux[2] = r.nblockms;
  aux[3] = r.nblockioni;
  aux[4] = r.npooled;
  auto copyNuc = [nt](const cvhcf::NucelExponents &e, double *o) {
    std::memcpy(o + 0 * nt, e.ang.data(), nt * sizeof(double));
    std::memcpy(o + 1 * nt, e.recRe.data(), nt * sizeof(double));
    std::memcpy(o + 2 * nt, e.recIm.data(), nt * sizeof(double));
    std::memcpy(o + 3 * nt, e.jntRe.data(), nt * sizeof(double));
    std::memcpy(o + 4 * nt, e.jntIm.data(), nt * sizeof(double));
    o[5 * nt] = e.N;
  };
  copyNuc(r.nuc, nuc);
  *ng = static_cast<int>(r.nucGroups.size());
  for (int g = 0; g < *ng && g < maxg; ++g) {
    gid[g] = r.nucGroups[g].first;
    copyNuc(r.nucGroups[g].second, gnuc + static_cast<std::size_t>(g) * (5 * nt + 1));
  }
  return r.ok ? 0 : 1;
}

}  // extern "C"
