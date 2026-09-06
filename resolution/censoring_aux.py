#!/usr/bin/env python3
"""Track-aligned auxiliary columns for a CANDIDATE (mass) pairs cache.

`cf_mass_likelihood.py --pairs-tt` stores only what the CF model needs, so the
candidate caches carry no chi2, no daughter momentum and no fit-status column
-- exactly the variables a censoring / truncation test needs.  This rebuilds
them from the SAME trees in the SAME order and proves the alignment instead of
assuming it: the per-entry pull z = (Jpsi_mass - Jpsigen_mass)/Jpsi_sigmamass
is recomputed in float64 and matched BIT-FOR-BIT against the cached `z`, the
same contract `cf_skew_closure.load(--ptfrom)` uses.

Columns written (one row per cache candidate, in cache order):
    pmin, pmax   min / max daughter |p| = pT cosh(eta)   [the 2 GeV momentum
                 floor of the Gauss-Newton step clamp lives here]
    ptmin        min daughter pT
    normchi2     chisqval / ndof of the two-track fit
    niter        iterations used (10 = the cap, NOT a failure)
    mkin, mtrk   Jpsikin_mass (vertex/kinematic seed) and Jpsitrk_mass
    frozen       |Jpsi_mass - Jpsikin_mass| < 1e-6, i.e. the refit moved the
                 mass by nothing at all -- the fingerprint of a step scaled to
                 zero by the momentum-floor clamp
    drmax        max of the two gen-match dR (the maker cuts at 0.1)
    event

usage:
  python3 censoring_aux.py --files GLOB --cache runs/cf_masspairs_X.npz \
      --out runs/censoring260904/aux_X.npz [--nproc 24]
"""
import argparse, glob, os, sys
import numpy as np
import uproot
import prodfiles

MJPSI = 3.0969
BR = ["event", "Jpsi_mass", "Jpsigen_mass", "Jpsi_sigmamass", "Jpsikin_mass",
      "Jpsitrk_mass", "chisqval", "ndof", "niter",
      "Muplus_pt", "Muminus_pt", "Muplus_eta", "Muminus_eta",
      "Muplusgen_dr", "Muminusgen_dr"]


def one(fn):
    t = uproot.open(fn)["tree"]
    a = t.arrays(BR, library="np")
    m = a["Jpsi_mass"].astype(np.float64)
    mg = a["Jpsigen_mass"].astype(np.float64)
    s = a["Jpsi_sigmamass"].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (m - mg) / s
    pp = a["Muplus_pt"] * np.cosh(a["Muplus_eta"])
    pm = a["Muminus_pt"] * np.cosh(a["Muminus_eta"])
    return dict(
        z=z, sigma=s,
        pmin=np.minimum(pp, pm).astype(np.float64),
        pmax=np.maximum(pp, pm).astype(np.float64),
        ptmin=np.minimum(a["Muplus_pt"], a["Muminus_pt"]).astype(np.float64),
        normchi2=(a["chisqval"] / np.maximum(a["ndof"], 1)).astype(np.float64),
        niter=a["niter"].astype(np.float64),
        mkin=a["Jpsikin_mass"].astype(np.float64),
        mtrk=a["Jpsitrk_mass"].astype(np.float64),
        frozen=(np.abs(m - a["Jpsikin_mass"].astype(np.float64)) < 1e-6
                ).astype(np.float64),
        drmax=np.maximum(a["Muplusgen_dr"], a["Muminusgen_dr"]).astype(np.float64),
        event=a["event"].astype(np.float64),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--files", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--nproc", type=int, default=24)
    p.add_argument("--ntasks", type=int, default=100000)
    a = p.parse_args()
    files = prodfiles.resolve(a.files, a.ntasks, logger=print)
    print(f"{len(files)} files", flush=True)
    from multiprocessing import Pool
    with Pool(a.nproc) as pool:
        parts = pool.map(one, files)
    A = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    print(f"{len(A['z'])} tree entries", flush=True)

    d = np.load(a.cache)
    zc = d["z"].astype(np.float64)
    n = len(zc)
    print(f"{n} cache candidates", flush=True)
    # sequential greedy match: the cache keeps a SUBSET of the tree rows in
    # tree order, so one forward pass with exact float equality is exact.
    idx = np.empty(n, dtype=np.int64)
    j = 0
    zt = A["z"]
    for i in range(n):
        while j < len(zt) and not (zt[j] == zc[i]):
            j += 1
        if j >= len(zt):
            sys.exit(f"alignment failed at cache row {i} (z={zc[i]!r})")
        idx[i] = j
        j += 1
    out = {k: v[idx] for k, v in A.items()}
    assert np.array_equal(out["z"], zc), "post-check failed"
    assert np.allclose(out["sigma"], d["sigma"], rtol=0, atol=0), "sigma mismatch"
    print(f"ALIGNED: {n} rows, z bit-identical, sigma bit-identical; "
          f"{len(zt)-n} tree rows dropped by the cache", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    # also keep the DROPPED tree rows, for the "what was lost" question
    keep = np.zeros(len(zt), bool); keep[idx] = True
    out.update({("drop_" + k): v[~keep] for k, v in A.items()})
    np.savez_compressed(a.out, **out)
    print(f"-> {a.out}", flush=True)


if __name__ == "__main__":
    main()
