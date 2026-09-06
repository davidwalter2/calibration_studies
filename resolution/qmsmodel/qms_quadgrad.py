#!/usr/bin/env python3
"""Per-candidate anatomy of the quadratic (hit-chi2) term's material block.

The quadratic term is
    chi2(theta) = chi2_0 + G^T theta + 0.5 theta^T K theta,
    G = sum_i G_i = sum_i gradv_i ,   K = sum_i K_i = sum_i hess_i
(`globalfit/extract.py`, `make_global_term.py`).  Its model covariance is
2 K^-1, which is the EXPECTED-Fisher error and is only correct if the residuals
scatter exactly as the fit's weight matrix R = Q^-1(...) says they do.

Because the candidates are independent, the SANDWICH covariance
    cov_sw = K^-1 [ sum_i (G_i - <G>)(G_i - <G>)^T ] K^-1
is available from the same files and makes NO assumption about R.  Under a
correct R, sum_i G_i G_i^T = 2 K, so
    c_g = [sum_i G_i,g^2] / (2 K_gg)
is a direct, per-parameter measurement of how far the fit's own weighting is
from the scatter of its own gradients.  c_g > 1 means the quoted error on
parameter g is TOO SMALL by sqrt(c_g).

This script accumulates, per selected candidate:  G_i restricted to a chosen
parameter subset, the diagonal of K_i, chi2/ndof, and enough of the sums to
rebuild G, K and the empirical gradient covariance on the subset.  It writes
one npz.  No fitting, no corrections.
"""
import argparse
import glob
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
    return dict(nglobal=len(pt), fitidx=fitidx, parmtype=pt[fitidx],
                subidx=raw[fitidx].astype(np.int64), g2f=g2f)


def _init(g2f, args, keepf):
    global _G2F, _ARGS, _KEEPF
    _G2F, _ARGS, _KEEPF = g2f, args, keepf


def process_file(fname):
    g2f, args, keepf = _G2F, _ARGS, _KEEPF
    nfit = int((g2f >= 0).sum())
    nk = len(keepf)
    try:
        f = uproot.open(fname)
        if "tree" not in f:
            return None
        t = f["tree"]
    except Exception as e:                                   # noqa: BLE001
        print(f"WARNING: cannot open {fname} ({type(e).__name__})")
        return None
    if t.num_entries == 0:
        return None
    keys = set(t.keys())
    want = ["globalidxv", "gradv", "chisqval", "ndof", "Muplus_pt", "Muplus_eta",
            "Muminus_pt", "Muminus_eta", "Muplus_dEref", "Muminus_dEref"]
    fmt = "factored" if "hessfactorv" in keys else "packed"
    want += (["hessfactorv", "nRank"] if fmt == "factored" else ["hesspackedv"])
    for b in ("gradmax", "hessmax"):
        if b in keys:
            want.append(b)
    a = t.arrays(want, entry_stop=(args.entries or None), library="np")
    n = len(a["globalidxv"])
    rchi2 = np.asarray(a["chisqval"], np.float64) / np.maximum(
        np.asarray(a["ndof"], np.float64), 1.0)
    ok = rchi2 < args.max_chi2_ndof if args.max_chi2_ndof > 0 else np.ones(n, bool)
    for b, c in (("gradmax", args.max_grad), ("hessmax", args.max_hess)):
        if c > 0 and b in a:
            ok &= np.abs(np.asarray(a[b], np.float64)) < c

    G = np.zeros(nfit)
    K = np.zeros((nfit, nfit))
    # the SAME accumulation restricted to candidates whose fractional
    # reference energy loss is below each threshold in --fr-cuts, so the
    # "restrict the material information to the regime where the mean-loss
    # model closes" fit is exact rather than a rescaling of the full K
    frc = list(args.fr_cuts)
    Gf = [np.zeros(nfit) for _ in frc]
    Kf = [np.zeros((nfit, nfit)) for _ in frc]
    gi_list, kd_list, c2_list, aux_list = [], [], [], []
    nsel = 0
    for ic in range(n):
        if not ok[ic]:
            continue
        gidx = np.asarray(a["globalidxv"][ic])
        fi = g2f[gidx]
        kk = np.where(fi >= 0)[0]
        if not len(kk):
            continue
        g = np.asarray(a["gradv"][ic], np.float64)
        G[fi[kk]] += g[kk]
        m = len(gidx)
        if fmt == "packed":
            h = np.asarray(a["hesspackedv"][ic], np.float64)
            H = np.zeros((m, m))
            iu = triu_index(m)
            H[iu] = h
            H = H + H.T - np.diag(np.diag(H))
            Hkk = H[np.ix_(kk, kk)]
        else:
            nr = int(a["nRank"][ic])
            B = np.asarray(a["hessfactorv"][ic], np.float64).reshape(nr, m)
            Bk = B[:, kk]
            Hkk = Bk.T @ Bk
        fk = fi[kk]
        K[np.ix_(fk, fk)] += Hkk
        if frc:
            ppp = float(a["Muplus_pt"][ic]) * np.cosh(float(a["Muplus_eta"][ic]))
            pmm = float(a["Muminus_pt"][ic]) * np.cosh(float(a["Muminus_eta"][ic]))
            frv = max(abs(float(a["Muplus_dEref"][ic])) / max(ppp, 1e-9),
                      abs(float(a["Muminus_dEref"][ic])) / max(pmm, 1e-9))
            for jf, cutf in enumerate(frc):
                if frv < cutf:
                    Gf[jf][fi[kk]] += g[kk]
                    Kf[jf][np.ix_(fk, fk)] += Hkk
        # per-candidate record on the watched subset
        gi = np.zeros(nk)
        kd = np.zeros(nk)
        pos = {int(v): j for j, v in enumerate(fk)}
        for j, ff in enumerate(keepf):
            p = pos.get(int(ff))
            if p is not None:
                gi[j] = g[kk][p]
                kd[j] = Hkk[p, p]
        gi_list.append(gi)
        kd_list.append(kd)
        c2_list.append(rchi2[ic])
        aux_list.append([float(a["gradmax"][ic]) if "gradmax" in a else 0.0,
                         float(a["hessmax"][ic]) if "hessmax" in a else 0.0,
                         float(a["Muplus_pt"][ic]), float(a["Muplus_eta"][ic]),
                         float(a["Muminus_pt"][ic]), float(a["Muminus_eta"][ic]),
                         float(a["Muplus_dEref"][ic]), float(a["Muminus_dEref"][ic]),
                         float(a["ndof"][ic]), float(a["chisqval"][ic])])
        nsel += 1
    return dict(G=G, K=K, Gf=np.asarray(Gf), Kf=np.asarray(Kf),
                gi=np.asarray(gi_list), kd=np.asarray(kd_list),
                c2=np.asarray(c2_list), aux=np.asarray(aux_list), nsel=nsel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-j", "--jobs", type=int, default=16)
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--entries", type=int, default=0)
    ap.add_argument("--parmtypes", type=int, nargs="+", default=[14, 15])
    ap.add_argument("--watch-parmtype", type=int, default=15)
    ap.add_argument("--max-chi2-ndof", type=float, default=3.0)
    ap.add_argument("--max-grad", type=float, default=1e6)
    ap.add_argument("--max-hess", type=float, default=1e8)
    ap.add_argument("--fr-cuts", type=float, nargs="*", default=[0.03, 0.01],
                    help="also accumulate G,K over candidates with "
                         "max(dE_ref/p) below each of these")
    args = ap.parse_args()

    # --max-files caps TASKS (a multi-stream task is N files)
    files = prodfiles.resolve(args.files, args.max_files,
                              logger=lambda m: print(m, flush=True))
    cat = build_catalog(files[0], args.parmtypes)
    keepf = np.where(cat["parmtype"] == args.watch_parmtype)[0]
    print(f"{len(files)} files, nfit={len(cat['fitidx'])}, "
          f"watching {len(keepf)} parmtype-{args.watch_parmtype} params",
          flush=True)
    t0 = time.time()
    tot = None
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init,
                             initargs=(cat["g2f"], args, keepf)) as ex:
        for i, r in enumerate(ex.map(process_file, files)):
            if r is None:
                continue
            if tot is None:
                tot = r
            else:
                tot["G"] += r["G"]
                tot["K"] += r["K"]
                tot["Gf"] = tot["Gf"] + r["Gf"]
                tot["Kf"] = tot["Kf"] + r["Kf"]
                tot["nsel"] += r["nsel"]
                for k in ("gi", "kd", "c2", "aux"):
                    tot[k] = np.concatenate([tot[k], r[k]])
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(files)} {time.time()-t0:.0f}s "
                      f"nsel={tot['nsel']}", flush=True)
    np.savez_compressed(args.output, G=tot["G"], K=tot["K"], gi=tot["gi"],
                        kd=tot["kd"], c2=tot["c2"], aux=tot["aux"], nsel=tot["nsel"],
                        parmtype=cat["parmtype"], subidx=cat["subidx"],
                        keepf=keepf, Gf=tot["Gf"], Kf=tot["Kf"],
                        fr_cuts=np.asarray(args.fr_cuts))
    print(f"wrote {args.output}  nsel={tot['nsel']}  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

AUXCOLS = ["gradmax", "hessmax", "ptp", "etap", "ptm", "etam",
           "dErefp", "dErefm", "ndof", "chisqval"]
