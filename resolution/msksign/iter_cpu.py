#!/usr/bin/env python3
"""Iteration counts and the reference EDM, per arm -- defect (a).

Under `doVtxConstraint` the frozen state index 6 makes the 10x10 reference
covariance singular, `edmvalref` NaN and `edmvalref < edmConvergence` never
true, so every candidate runs to `nIters`.

  usage: iter_cpu.py <dir-or-file> [<dir-or-file> ...]
"""
import glob
import os
import sys

import numpy as np
import uproot


def one(d):
    fs = sorted(glob.glob(os.path.join(d, "*.root"))) if os.path.isdir(d) else [d]
    fs = [f for f in fs if os.path.getsize(f) > 0]
    return fs[0] if fs else None


print(f"{'arm':<18} {'cand':>6} {'<niter>':>8} {'niter=10':>9} {'med':>4} {'max':>4}"
      f" {'edmvalref finite':>17} {'median edmvalref':>17}")
for d in sys.argv[1:]:
    f = one(d)
    if f is None:
        print(f"{os.path.basename(d.rstrip('/')):<18}  (no output)")
        continue
    t = uproot.open(f)["tree"]
    br = [b for b in ("niter", "niter_cons0", "edmvalref", "edmval") if b in t.keys()]
    a = t.arrays(br, library="np")
    n = np.asarray(a["niter"], dtype=float)
    e = np.asarray(a["edmvalref"], dtype=float) if "edmvalref" in a else np.full(n.shape, np.nan)
    fin = np.isfinite(e)
    print(f"{os.path.basename(d.rstrip('/')):<18} {len(n):>6} {n.mean():>8.3f}"
          f" {100*np.mean(n >= 10):>8.1f}% {np.median(n):>4.0f} {n.max():>4.0f}"
          f" {100*fin.mean():>16.1f}% {np.median(e[fin]) if fin.any() else float('nan'):>17.3e}")
