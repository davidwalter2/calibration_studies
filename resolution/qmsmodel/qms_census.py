#!/usr/bin/env python3
"""Join the quadratic term's per-candidate material gradient with the per-step
census of the SAME candidate, for one material group.

For every selected candidate this records
  * g_g, K_gg              -- the group's entry in `gradv` and diag(hess)
  * the group's step census from `msmoliv`: n steps, sum x_g [g/cm2],
    sum d/X0, max d/X0, max x_g, x_g-weighted effZ / effA / Z(Z+1)/A / X0_g,
    sum thp2 (Q's projected MS variance)
  * the same for a reference group (default tec_structure) and the whole track
  * kinematics (pt, eta of both muons, the reference dE)
so the question "which tracks carry this group's information, and what material
do they actually see" can be answered without any model assumption.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import uproot

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
import prodfiles  # noqa: E402  (needs resolution/ on sys.path)

C_EFFZ, C_EFFA, C_XG, C_P, C_BETA, C_THP2, C_DOX0, C_ZZP1, C_LNSW, C_GRP = range(10)
_TRIU = {}


def triu_index(n):
    if n not in _TRIU:
        _TRIU[n] = np.triu_indices(n)
    return _TRIU[n]


def build_catalog(fname, parmtypes):
    rt = uproot.open(fname)["runtree"]
    pt = rt["parmtype"].array(library="np")
    raw = rt["rawdetid"].array(library="np")
    fitidx = np.where(np.isin(pt, parmtypes))[0]
    g2f = -np.ones(len(pt), np.int64)
    g2f[fitidx] = np.arange(len(fitidx))
    return pt, raw, g2f, fitidx


def _init(g2f, args, gcol):
    global _G2F, _ARGS, _GCOL
    _G2F, _ARGS, _GCOL = g2f, args, gcol


def _census(rec, m):
    if not m.any():
        return np.zeros(10)
    r = rec[m]
    w = r[:, C_XG]
    sw = max(w.sum(), 1e-300)
    x0g = r[:, C_XG] / np.maximum(r[:, C_DOX0], 1e-300)
    return np.array([len(r), w.sum(), r[:, C_DOX0].sum(), r[:, C_DOX0].max(),
                     w.max(), (w * r[:, C_EFFZ]).sum() / sw,
                     (w * r[:, C_EFFA]).sum() / sw,
                     (w * r[:, C_ZZP1]).sum() / sw,
                     (w * x0g).sum() / sw, r[:, C_THP2].sum()])


def process_file(fname):
    g2f, args, gcol = _G2F, _ARGS, _GCOL
    try:
        f = uproot.open(fname)
        t = f["tree"]
    except Exception:                                        # noqa: BLE001
        return None
    if t.num_entries == 0:
        return None
    keys = set(t.keys())
    want = ["globalidxv", "gradv", "chisqval", "ndof", "msmoliidx", "msmoliv",
            "Muplus_pt", "Muplus_eta", "Muminus_pt", "Muminus_eta",
            "Muplus_dEref", "Muminus_dEref", "Jpsi_mass", "Jpsigen_mass",
            "Jpsi_sigmamass", "run", "lumi", "event"]
    fmt = "factored" if "hessfactorv" in keys else "packed"
    want += (["hessfactorv", "nRank"] if fmt == "factored" else ["hesspackedv"])
    for b in ("gradmax", "hessmax"):
        if b in keys:
            want.append(b)
    want = [w for w in want if w in keys]
    a = t.arrays(want, entry_stop=(args.entries or None), library="np")
    n = len(a["globalidxv"])
    rchi2 = np.asarray(a["chisqval"], np.float64) / np.maximum(
        np.asarray(a["ndof"], np.float64), 1.0)
    ok = rchi2 < args.max_chi2_ndof if args.max_chi2_ndof > 0 else np.ones(n, bool)
    for b, c in (("gradmax", args.max_grad), ("hessmax", args.max_hess)):
        if c > 0 and b in a:
            ok &= np.abs(np.asarray(a[b], np.float64)) < c
    rows = []
    for ic in range(n):
        if not ok[ic]:
            continue
        gidx = np.asarray(a["globalidxv"][ic])
        gv = np.asarray(a["gradv"][ic], np.float64)
        m = len(gidx)
        gA = kA = gB = kB = 0.0
        pa = np.where(gidx == gcol[0])[0]
        pb = np.where(gidx == gcol[1])[0]
        if len(pa) or len(pb):
            if fmt == "packed":
                h = np.asarray(a["hesspackedv"][ic], np.float64)
                H = np.zeros((m, m))
                H[triu_index(m)] = h
                dg = np.diag(H).copy()
            else:
                nr = int(a["nRank"][ic])
                B = np.asarray(a["hessfactorv"][ic], np.float64).reshape(nr, m)
                dg = (B * B).sum(0)
            if len(pa):
                gA, kA = gv[pa[0]], dg[pa[0]]
            if len(pb):
                gB, kB = gv[pb[0]], dg[pb[0]]
        v = np.asarray(a["msmoliv"][ic], np.float64)
        idx = np.asarray(a["msmoliidx"][ic])
        if not len(idx):
            continue
        rec = v.reshape(-1, len(v) // len(idx))
        rec = rec[rec[:, C_THP2] > 0.0]
        if not len(rec):
            continue
        gg = rec[:, C_GRP].astype(np.int64)
        cA = _census(rec, gg == args.group)
        cB = _census(rec, gg == args.refgroup)
        cT = _census(rec, np.ones(len(rec), bool))
        rows.append(np.concatenate([
            [gA, kA, gB, kB, rchi2[ic],
             float(a["Muplus_pt"][ic]), float(a["Muplus_eta"][ic]),
             float(a["Muminus_pt"][ic]), float(a["Muminus_eta"][ic]),
             float(a["Muplus_dEref"][ic]), float(a["Muminus_dEref"][ic]),
             float(a["Jpsi_mass"][ic]), float(a["Jpsigen_mass"][ic]),
             float(a["run"][ic]), float(a["lumi"][ic]), float(a["event"][ic])],
            cA, cB, cT]))
    return np.asarray(rows) if rows else None


COLS = (["g", "k", "gref", "kref", "chi2n", "ptp", "etap", "ptm", "etam",
         "dErefp", "dErefm", "mass", "genmass", "run", "lumi", "event"]
        + [f"{p}_{s}" for p in ("A", "B", "T")
           for s in ("nstep", "xg", "dox0", "maxdox0", "maxxg", "effZ",
                     "effA", "zza", "x0g", "thp2")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-j", "--jobs", type=int, default=16)
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--entries", type=int, default=0)
    ap.add_argument("--group", type=int, default=36)
    ap.add_argument("--refgroup", type=int, default=37)
    ap.add_argument("--parmtypes", type=int, nargs="+", default=[14, 15])
    ap.add_argument("--max-chi2-ndof", type=float, default=3.0)
    ap.add_argument("--max-grad", type=float, default=1e6)
    ap.add_argument("--max-hess", type=float, default=1e8)
    args = ap.parse_args()
    # --max-files caps TASKS (a multi-stream task is N files)
    files = prodfiles.resolve(args.files, args.max_files,
                              logger=lambda m: print(m, flush=True))
    pt, raw, g2f, fitidx = build_catalog(files[0], args.parmtypes)
    ia = int(np.where((pt == 15) & (raw == args.group))[0][0])
    ib = int(np.where((pt == 15) & (raw == args.refgroup))[0][0])
    print(f"{len(files)} files; global idx {ia} (group {args.group}), "
          f"{ib} (group {args.refgroup})", flush=True)
    out, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init,
                             initargs=(g2f, args, (ia, ib))) as ex:
        for i, r in enumerate(ex.map(process_file, files)):
            if r is not None:
                out.append(r)
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(files)} {time.time()-t0:.0f}s", flush=True)
    X = np.concatenate(out)
    np.savez_compressed(args.output, X=X, cols=np.array(COLS),
                        group=args.group, refgroup=args.refgroup)
    print(f"wrote {args.output}  {X.shape}  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
