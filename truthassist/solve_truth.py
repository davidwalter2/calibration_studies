#!/usr/bin/env python3
"""Truth-assisted global corrections from gen-anchored single-track CVH fits.

Input: runCvhTruthAssisted.py mode=derive outputs (tree: gradv, hesspackedv,
globalidxv per track, all marginalised over the free per-track states; runtree:
the parameter catalogue). The fit minimises 0.5*sum(chi2) + priors:

    x = -(0.5 sum H_i + P)^-1 (0.5 sum g_i)

over the FREE parameter types only (default: alignment 0-5 and the pixel
class corrections 16-21). Frozen types (default: per-module Bz 6 and material 7)
are dropped from the system, i.e. held at their current value (zero); free
parameters that no selected track touches are pinned at zero as well.

Accumulation is multithreaded C++ (RDataFrame ForeachSlot, std::atomic_ref) into
the upper triangle of one dense matrix on the free-parameter subspace.

Outputs: <out>.npz (x on the full catalogue, the free mask, per-parameter
information and prior widths) and <out>.root with a corFiles-style parmtree.

usage (cmsenv):
  solve_truth.py -i '/path/task_*/globalcor_truth_*.root' -o truthcor --threads 32
"""
import argparse
import glob
import math
import time

import numpy as np
import ROOT

CPP = r"""
#include <atomic>
#include <vector>
#include <cmath>
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"

// Accumulates 0.5*gradv and the UPPER triangle of 0.5*hesspackedv directly into
// caller-owned dense buffers (numpy, row-major n x n) with std::atomic_ref, so
// no second copy of the matrix is ever held.
struct TruthAgg {
  std::vector<int> remap;                        // catalogue idx -> free idx (-1 frozen)
  unsigned long long n = 0;
  double* H = nullptr;
  double* g = nullptr;
  std::atomic<unsigned long long> ntracks{0};

  TruthAgg(const std::vector<int>& rm, unsigned long long nfree, long long hptr, long long gptr)
      : remap(rm), n(nfree), H(reinterpret_cast<double*>(hptr)), g(reinterpret_cast<double*>(gptr)) {}
  static void addAtomic(double& ref, double v) {
    std::atomic_ref<double>(ref).fetch_add(v, std::memory_order_relaxed);
  }
  void add(const ROOT::RVec<float>& grad, const ROOT::RVec<float>& hess,
           const ROOT::RVec<unsigned int>& idx) {
    const unsigned int np = idx.size();
    std::vector<int> f(np);
    for (unsigned int i = 0; i < np; ++i) f[i] = remap[idx[i]];
    for (unsigned int i = 0; i < np; ++i)
      if (f[i] >= 0) addAtomic(g[f[i]], 0.5 * grad[i]);
    unsigned long long k = 0;
    for (unsigned int i = 0; i < np; ++i) {
      for (unsigned int j = i; j < np; ++j, ++k) {
        const int a = f[i], b = f[j];
        if (a < 0 || b < 0) continue;
        double v = 0.5 * hess[k];
        if (a == b && i != j) v *= 2.;             // duplicate index in one track
        const unsigned long long lo = a < b ? a : b, hi = a < b ? b : a;
        addAtomic(H[lo * n + hi], v);
      }
    }
    ntracks.fetch_add(1, std::memory_order_relaxed);
  }
};

void runTruthAgg(ROOT::RDF::RNode df, TruthAgg& agg) {
  df.ForeachSlot([&agg](unsigned int, const ROOT::RVec<float>& gr, const ROOT::RVec<float>& he,
                        const ROOT::RVec<unsigned int>& ix) { agg.add(gr, he, ix); },
                 {"gradv", "hesspackedv", "globalidxv"});
}
"""

# prior widths as in MuonMomentumScaleCalibration/Analysis/globalfith5pynoreduction.py
def prior_sigma(pt, subdet):
    if pt == 0:
        return 5e-3
    if pt == 1:
        return 5e-3 if subdet < 2 else 2.0
    if pt == 2:
        return 5e-1
    if pt in (3, 4, 5):
        return 5e-3
    if pt in (16, 17, 18, 20, 21):
        return 5e-3
    if pt == 19:
        return 2e-2
    if pt == 22:
        return 0.1
    raise ValueError(pt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, nargs="+", help="glob(s) of derive outputs")
    ap.add_argument("-o", "--out", required=True, help="output prefix")
    ap.add_argument("--free", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20, 21],
                    help="parameter types to float (others frozen at zero)")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--min-valid-hits", type=int, default=9)
    ap.add_argument("--max-grad", type=float, default=1e5, help="drop tracks with max|gradv| above")
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--mask", default="", choices=["", "v718"],
                    help="v718: Josh's truth-assisted mask -- only the in-plane alignment "
                         "dofs, local x/y (types 0, 1) everywhere and the in-plane rotation "
                         "(type 5) only for TID/TEC (subdet 4, 5); types 2-4 frozen")
    ap.add_argument("--prior", action="append", default=[],
                    help="override a prior width, TYPE=SIGMA (cm or rad), repeatable")
    args = ap.parse_args()

    files = sorted(f for g in args.input for f in glob.glob(g))
    if args.max_files:
        files = files[:args.max_files]
    assert files, "no input files"
    print(f"{len(files)} input files")

    rt = ROOT.RDataFrame("runtree", files[0]).AsNumpy(["parmtype", "subdet", "layer", "rawdetid"])
    ptype = rt["parmtype"].astype(int)
    subdet = rt["subdet"].astype(int)
    ncat = len(ptype)
    free = np.isin(ptype, args.free)
    if args.mask == "v718":
        free &= ~np.isin(ptype, [2, 3, 4])
        free &= ~((ptype == 5) & ~np.isin(subdet, [4, 5]))
    overrides = {int(k): float(v) for k, v in (o.split("=") for o in args.prior)}
    if overrides:
        print("prior overrides:", overrides)
    remap = -np.ones(ncat, dtype=np.int64)
    remap[free] = np.arange(free.sum())
    nfree = int(free.sum())
    print(f"catalogue {ncat}; free {nfree} of types {sorted(set(ptype[free]))}; "
          f"frozen types {sorted(set(ptype[~free]))}")

    ROOT.gInterpreter.Declare(CPP)
    ROOT.ROOT.EnableImplicitMT(args.threads)
    rm = ROOT.std.vector("int")(remap.astype(np.int32).tolist())
    t0 = time.time()
    H = np.zeros((nfree, nfree), dtype=np.float64)
    g = np.zeros(nfree, dtype=np.float64)
    agg = ROOT.TruthAgg(rm, nfree, H.ctypes.data, g.ctypes.data)
    print(f"allocated dense matrix ({nfree*nfree*8/1e9:.1f} GB) in {time.time()-t0:.0f} s")

    chain = ROOT.TChain("tree")
    for f in files:
        chain.Add(f)
    df = ROOT.RDataFrame(chain)
    n0 = df.Count()
    df = df.Filter(f"nValidHits >= {args.min_valid_hits} && nValidPixelHits > 0", "hits")
    df = df.Filter(f"Max(abs(gradv)) < {args.max_grad}", "gradsane")
    rep = df.Report()
    t0 = time.time()
    ROOT.runTruthAgg(ROOT.RDF.AsRNode(df), agg)
    print(f"aggregated {agg.ntracks.load()} of {n0.GetValue()} tracks in {time.time()-t0:.0f} s")
    rep.Print()

    # only the UPPER triangle of H is filled and used: LAPACK potrf on the
    # F-ordered transpose with lower=True reads exactly that triangle, in place
    del agg

    fidx = np.flatnonzero(free)
    info = np.diag(H).copy()
    touched = info > 0.
    sig = np.array([overrides.get(int(ptype[k]), prior_sigma(ptype[k], subdet[k])) for k in fidx])
    print(f"touched {touched.sum()} of {nfree} free parameters; pinning {np.sum(~touched)}")
    # pin untouched parameters: unit diagonal, zero gradient -> x = 0 exactly
    for i in np.flatnonzero(~touched):
        H[i, i:] = 0.
        H[:i, i] = 0.
        H[i, i] = 1.
        g[i] = 0.
    H[np.diag_indices(nfree)] += np.where(touched, 1. / sig**2, 0.)

    import scipy.linalg
    t0 = time.time()
    c = scipy.linalg.cho_factor(H.T, lower=True, overwrite_a=True, check_finite=False)
    xf = scipy.linalg.cho_solve(c, -g, check_finite=False)
    print(f"Cholesky solve in {time.time()-t0:.0f} s")

    x = np.zeros(ncat)
    x[fidx] = xf
    np.savez(args.out + ".npz", x=x, free=free, touched_free=touched, info=info, prior=sig,
             fidx=fidx, parmtype=ptype, subdet=subdet, files=np.array(files))
    fo = ROOT.TFile(args.out + ".root", "RECREATE")
    tr = ROOT.TTree("parmtree", "")
    import array
    xv = array.array('f', [0.])
    tr.Branch("x", xv, "x/F")
    for v in x:
        xv[0] = v
        tr.Fill()
    tr.Write()
    fo.Close()
    for t in sorted(set(ptype[free])):
        m = (ptype[fidx] == t) & touched
        if m.any():
            print(f"  type {t:2d}: n={m.sum():6d}  rms {1e4*np.sqrt(np.mean(xf[m]**2)):8.2f}  "
                  f"(x1e4, um for translations / 1e-4 rad for rotations)")
    print("wrote", args.out + ".npz", args.out + ".root")


if __name__ == "__main__":
    main()
