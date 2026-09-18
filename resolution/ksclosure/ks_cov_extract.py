#!/usr/bin/env python3
"""Per-candidate 6x6 reference-momentum covariance and TRUE leg angles.

`ks_pairs.py` writes the mass-CF cache, and reduces `Jpsi_covrefmom` to the
three scalars the Jensen term needs (`sigrelp`, `sigrelm`, `rhomom`, `fang`).
The angular part of the sigma-artefact slope needs the matrix itself -- the
slope is

    a = (grad m)^T Sigma (grad sigma) / (grad m)^T Sigma (grad m)

and the numerator's angular components cannot be rebuilt from the scalars --
together with the mass Jacobian `Jpsi_jacrefmom` in the same (plus, minus) x
(q/p, lambda, phi) order.  It also needs the TRUE per-leg direction, which the
pairs cache drops (it keeps only the true |p| of each daughter).

The selection and the truth join are `ks_pairs.py`'s, imported rather than
copied, so the rows come out in the SAME ORDER as `runs/kspairs_all.npz` and
the two files are joined by position.  `ares_angles.py` asserts that
(run, lumi, event, sigma) agree row by row before using them together.

usage:
  source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
  python3 ks_cov_extract.py --prod <prod dir> --out runs/kscov_all.npz -j 32
"""
import argparse
import glob
import math
import os
import sys
from multiprocessing import Pool

import numpy as np
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ks_pairs  # noqa: E402

# the branches this file adds on top of the pairs cache
_F32 = ("Jpsi_covrefmom", "Jpsi_jacrefmom")
_SCAL = ("Jpsi_qoprefplus", "Jpsi_qoprefminus", "Jpsi_mass", "Jpsi_sigmamass",
         "Muplus_pt", "Muplus_eta", "Muplus_phi",
         "Muminus_pt", "Muminus_eta", "Muminus_phi",
         "Jpsi_x", "Jpsi_y", "Jpsi_z", "cfmass_vgf", "chisqval", "ndof")
_INT = ("run", "lumi", "event", "Muplus_nvalid", "Muminus_nvalid",
        "Muplus_nvalidpixel", "Muminus_nvalidpixel")
# truth columns: the per-leg DIRECTION, which the pairs cache does not keep
_TRUTH = ("lamp", "phip", "pabsp", "lamm", "phim", "pabsm", "mgen",
          "vx", "vy", "vz")


def read_one(fn, t, args):
    try:
        f = uproot.open(fn)
        if "tree" not in f:
            return None
        tr = f["tree"]
    except Exception:
        return None
    have = set(tr.keys())
    need = sorted(set(_F32 + _SCAL + _INT + ("cfmass_ok",)
                      + ("Muplus_phi", "Muminus_phi")))
    miss = [b for b in need if b not in have]
    if miss:
        raise SystemExit(f"{fn} is missing {miss}")
    a = tr.arrays(need, library="np")
    if not len(a["Jpsi_mass"]):
        return None
    tab = {k: np.asarray(a[k], dtype=np.float64) for k in
           ("Muplus_eta", "Muminus_eta", "Muplus_pt", "Muminus_pt",
            "Muplus_phi", "Muminus_phi", "Jpsi_x", "Jpsi_y", "Jpsi_z")}
    for k in ("run", "lumi", "event"):
        tab[k] = np.asarray(a[k], dtype=np.int64)
    jrow, dvtx, namb = ks_pairs.match(tab, t, args)
    ok = np.asarray(a["cfmass_ok"]).astype(bool)
    sig = np.asarray(a["Jpsi_sigmamass"], dtype=np.float64)
    good = ok & np.isfinite(sig) & (sig > 0.0) & (jrow >= 0)
    idx = np.where(good)[0]
    if not len(idx):
        return None
    j = jrow[idx]
    out = {}
    # the two jagged float vectors: 21 and 6 entries, fixed length
    for b, ncol in zip(_F32, (21, 6)):
        v = a[b]
        arr = np.empty((len(idx), ncol), dtype=np.float64)
        for k, i in enumerate(idx):
            row = np.asarray(v[i], dtype=np.float64)
            if len(row) != ncol:
                arr[k] = np.nan
            else:
                arr[k] = row
        out[b] = arr
    for b in _SCAL:
        out[b] = np.asarray(a[b], dtype=np.float64)[idx]
    for b in _INT:
        out[b] = np.asarray(a[b], dtype=np.int64)[idx]
    for b in _TRUTH:
        out["t_" + b] = np.asarray(t[b], dtype=np.float64)[j]
    return out


def _chunk(job):
    cidx, fs, tn, args = job
    t = ks_pairs.load_truth([tn], quiet=True)
    parts = []
    for fn in fs:
        r = read_one(fn, t, args)
        if r is not None:
            parts.append(r)
    if not parts:
        return cidx, None
    keys = parts[0].keys()
    return cidx, {k: np.concatenate([p[k] for p in parts]) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod", required=True)
    ap.add_argument("--truth-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("-j", "--jobs", type=int, default=16)
    ap.add_argument("--maxtasks", type=int, default=0)
    # IDENTICAL defaults to ks_pairs.py: the row order must match
    ap.add_argument("--max-dvtx", type=float, default=2.0)
    ap.add_argument("--max-dlam", type=float, default=0.25)
    ap.add_argument("--max-dphi", type=float, default=0.25)
    ap.add_argument("--max-dp", type=float, default=0.50)
    ap.add_argument("--keep-unmatched", action="store_true")
    args = ap.parse_args()

    tdir = args.truth_dir or os.path.join(args.prod, "truth")
    jobs = []
    for td in sorted(glob.glob(os.path.join(args.prod, "task_[0-9]*"))):
        idx = os.path.basename(td).split("_")[-1]
        if not os.path.exists(os.path.join(td, ".complete")):
            continue
        tn = os.path.join(tdir, f"truth_{idx}.npz")
        if not os.path.exists(tn):
            continue
        fs = sorted(glob.glob(os.path.join(td, "globalcor_ks_*.root")))
        if fs:
            jobs.append((idx, fs, tn, args))
    if args.maxtasks:
        jobs = jobs[:args.maxtasks]
    if not jobs:
        raise SystemExit("no complete (task, truth) chunk pairs")
    print(f"{len(jobs)} chunks, {args.jobs} workers")

    res = {}
    with Pool(args.jobs) as pool:
        for n, (cidx, r) in enumerate(pool.imap_unordered(_chunk, jobs)):
            if r is not None:
                res[cidx] = r
            if (n + 1) % 50 == 0:
                print(f"  {n+1}/{len(jobs)}", flush=True)
    # reassemble in chunk order -- ks_pairs.py iterates the same sorted list
    keys = None
    parts = []
    for cidx, _, _, _ in jobs:
        if cidx in res:
            parts.append(res[cidx])
            keys = res[cidx].keys()
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    n = len(out["Jpsi_mass"])
    print(f"{n} truth-matched candidates")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(args.out, **out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
