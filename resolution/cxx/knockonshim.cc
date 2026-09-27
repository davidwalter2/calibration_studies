// C-ABI bridge to the knock-on primitives of `cvhcf` (cvhcf::knockonMapBlock,
// knockonMapRad, knockonJointRows, and trackExponents with wantKnockonMap /
// wantKnockonJoint), for knockon_fit/cxx_gate.py.  Built by
// knockon_fit/build_shim.sh from the UNMODIFIED CMSSW sources, like
// cvhcfshim.cc.
#include "TrackPropagation/Geant4e/interface/CvhCfExponents.h"

#include <cstring>

extern "C" {

int cvhcf_knockon_map_block(const float *rows, int stride, int n, int groupCol, double w, double *sre, double *sim) {
  cvhcf::knockonMapBlock(rows, stride, n, groupCol, w, sre, sim);
  return 0;
}

int cvhcf_knockon_map_rad(const float *rows, int stride, int n, const float *spec, const float *vgrid, int nv,
                          double w, double *sre, double *sim) {
  cvhcf::knockonMapRad(rows, stride, n, spec, vgrid, nv, w, sre, sim);
  return 0;
}

int cvhcf_knockon_joint_rows(const float *rows, int stride, int n, const unsigned int *msidx,
                             const unsigned int *ioniidx, int groupCol, const double *alpha, const double *beta,
                             int exact, double *sre, double *sim) {
  cvhcf::knockonJointRows(rows, stride, n, msidx, ioniidx, groupCol, alpha, beta, exact != 0, sre, sim);
  return 0;
}

// the whole track with the knock-on flags; out = 10*kNTau: ms, del, ioRe,
// ioIm, radRe, radIm, kxRe, kxIm, kjRe, kjIm; groups: sum over the split of
// kx and kj into gsum (4*kNTau) and the group count as return value
int cvhcf_track_knockon(const unsigned int *resglobidx, const int *resfamily, const float *resvarv, int nres,
                        const unsigned int *msidx, const float *msv, int nms, int msstride, int msgcol,
                        const unsigned int *ioidx, const float *iov, int nio, int iostride, int iogcol,
                        const unsigned int *qsidx, const float *qsv, int nqs, const unsigned int *ridx,
                        const float *rv, int nrad, int radstride, int radgcol, const float *rspec,
                        const float *rvgrid, int radnv, double sigma, double ioniSign, int wantMap, int wantJoint,
                        int wantGroups, double *out, double *gsum) {
  cvhcf::TrackInput in;
  in.resglobidx = resglobidx;
  in.resfamily = resfamily;
  in.resvarv = resvarv;
  in.nres = nres;
  in.ms = {msidx, msv, nms, msstride, msgcol};
  in.ioni = {ioidx, iov, nio, iostride, iogcol};
  in.qsc = {qsidx, qsv, nqs, 2};
  in.rad = {ridx, rv, nrad, radstride, radgcol};
  in.radspec = rspec;
  in.radvgrid = rvgrid;
  in.radnv = radnv;
  in.sigma = sigma;
  in.ioniSign = ioniSign;
  in.wantDelta = true;
  in.wantKnockonMap = wantMap != 0;
  in.wantKnockonJoint = wantJoint != 0;
  in.wantGroups = wantGroups != 0;
  cvhcf::TrackResult r;
  cvhcf::trackExponents(in, r);
  const int nt = cvhcf::kNTau;
  const std::array<double, cvhcf::kNTau> *fam[10] = {&r.S.ms, &r.S.del, &r.S.ioRe, &r.S.ioIm, &r.S.radRe,
                                                      &r.S.radIm, &r.S.kxRe, &r.S.kxIm, &r.S.kjRe, &r.S.kjIm};
  for (int f = 0; f < 10; ++f)
    std::memcpy(out + f * nt, fam[f]->data(), nt * sizeof(double));
  std::memset(gsum, 0, 4 * nt * sizeof(double));
  for (const auto &g : r.groups)
    for (int j = 0; j < nt; ++j) {
      gsum[j] += g.S.kxRe[j];
      gsum[nt + j] += g.S.kxIm[j];
      gsum[2 * nt + j] += g.S.kjRe[j];
      gsum[3 * nt + j] += g.S.kjIm[j];
    }
  return r.ok ? static_cast<int>(r.groups.size()) : -1;
}
}
