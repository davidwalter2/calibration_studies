#!/usr/bin/env python3
"""Maker-level gates of the knock-on export (knockon_fit/run_smoke.sh outputs).

  identity  every branch of the `off` file exists in the `on` file and is
            bit-identical, and the runtree too except `cfmodel`;
            optionally the `off` file against a reference file (--ref)
  branch    <prefix>_kx_re/_im, <prefix>_kj_re/_im against cf_knockon
            recomputed offline from the file's own step records (q/p
            functional: charge sign, sqrt(refCov00); mass functional: -1,
            Jpsi_sigmamass), and the per-group arrays summed against the flat

usage: branch_gate.py OFF.root ON.root [--ref REF.root] [--entries N]
Run with CVH_IONI_KOKOULIN=0."""
import argparse
import os
import sys

import numpy as np
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import cf_knockon as ck                                   # noqa: E402
import cf_track_resolution as ctr                         # noqa: E402
import prodfiles                                          # noqa: E402

NT = 64
TAU = ctr.TG[::4][:NT]


def same_tree(fa, fb, tree, skip=()):
    ta, tb = uproot.open(fa)[tree], uproot.open(fb)[tree]
    bad, missing = [], []
    for k in ta.keys():
        if k in skip:
            continue
        if k not in tb.keys():
            missing.append(k)
            continue
        a = ta[k].array(library="np")
        b = tb[k].array(library="np")
        if a.dtype == object:
            ok = len(a) == len(b) and all(np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True)
                                          for x, y in zip(a, b))
        else:
            ok = np.array_equal(a, b, equal_nan=True) if a.dtype.kind == "f" else np.array_equal(a, b)
        if not ok:
            bad.append(k)
    return bad, missing, len(ta.keys())


def block_weights(gi, fam, vb, M, midx, I, iidx, qsi, qsv, sig, ioni_sign):
    wms, wio = {}, {}
    for g in np.unique(gi[fam == 10]):
        vp = vb[(fam == 10) & (gi == g)].sum()
        st = M[midx == g]
        if vp > 0 and len(st) and st[:, 5].sum() > 0:
            wms[int(g)] = np.sqrt(vp / st[:, 5].sum()) / sig
    for g in np.unique(gi[fam == 11]):
        vp = vb[(fam == 11) & (gi == g)].sum()
        st = I[iidx == g]
        if vp > 0 and len(st):
            sq2 = ctr.ioni_sq2(st, qsv[qsi == g] if qsv is not None else None)
            if sq2 > 0:
                wio[int(g)] = ioni_sign * np.sqrt(vp / sq2) / sig
    return wms, wio


def branch_check(fn, entries):
    f = uproot.open(fn)
    t = f["tree"]
    pt = f["runtree"]["parmtype"].array(library="np")
    twotrack = "Jpsi_sigmamass" in t.keys()
    pre = "cfmass" if twotrack else "cfqop"
    br = ["reseigidx", "resinfvarv", "msmoliidx", "msmoliv", "ioniurbanidx", "ioniurbanv",
          "radstepidx", "radstepv", "radstepspecv", "radvgrid", "ioniqscaleidx", "ioniqscalev",
          pre + "_kx_re", pre + "_kx_im", pre + "_kj_re", pre + "_kj_im", pre + "_ok"]
    grp = pre + "_grp_kx_re" in t.keys()
    if grp:
        br += [pre + "_grp", pre + "_grp_kx_re", pre + "_grp_kx_im", pre + "_grp_kj_re", pre + "_grp_kj_im"]
    br += ["Jpsi_sigmamass"] if twotrack else ["refCov", "refParms"]
    br += prodfiles.stride_keys(t, ("msmoliv", "ioniurbanv", "radstepv", "radstepspecv"))
    a = t.arrays(br, library="np", entry_stop=entries)
    wx = wj = sx = sj = gx = gj = 0.
    n = 0
    for ic in range(len(a["reseigidx"])):
        if not a[pre + "_ok"][ic]:
            continue
        rec = {}
        for nm, ib, vbn, mc in (("M", "msmoliidx", "msmoliv", 8), ("I", "ioniurbanidx", "ioniurbanv", 11),
                                ("R", "radstepidx", "radstepv", 11), ("P", "radstepidx", "radstepspecv", 1)):
            idx = np.asarray(a[ib][ic])
            rec[nm + "idx"] = idx
            rec[nm] = prodfiles.reshape_records(a[vbn][ic], nrec=len(idx),
                                                stride=prodfiles.entry_stride(a, vbn, ic),
                                                branch=vbn, min_cols=mc).astype(np.float64)
        gi = np.asarray(a["reseigidx"][ic])
        vb = np.asarray(a["resinfvarv"][ic], np.float64)
        fam = pt[gi]
        qsi = np.asarray(a["ioniqscaleidx"][ic])
        qsv = np.asarray(a["ioniqscalev"][ic], np.float64).reshape(-1, 2)
        if twotrack:
            sig = float(np.float32(a["Jpsi_sigmamass"][ic]))
            sgn = -1.0
        else:
            sig = float(np.sqrt(np.float32(a["refCov"][ic][0])))
            sgn = 1.0 if a["refParms"][ic][0] >= 0 else -1.0
        wms, wio = block_weights(gi, fam, vb, rec["M"], rec["Midx"], rec["I"], rec["Iidx"], qsi, qsv, sig, sgn)
        vg = np.asarray(a["radvgrid"][ic], np.float64)
        kx = np.zeros(NT, complex)
        for g, w in wio.items():
            kx += ck.fit_map(rec["I"][rec["Iidx"] == g], w, TAU)
            rm = rec["Ridx"] == g
            if rm.any():
                kx += ck.map_rad(TAU, rec["R"][rm], rec["P"][rm], vg, np.full(rm.sum(), w))
        beta, alpha = ck.pair_weights(rec["Midx"], rec["Ridx"], rec["M"], rec["R"], wms, wio)
        kj = ck.fit_joint(rec["M"], alpha, beta, rec["Midx"], rec["Ridx"], TAU,
                          ctr.DELTA_TCUT, ctr.DELTA_TMAXCAP, True)
        ex = np.asarray(a[pre + "_kx_re"][ic], np.float64) + 1j * np.asarray(a[pre + "_kx_im"][ic], np.float64)
        ej = np.asarray(a[pre + "_kj_re"][ic], np.float64) + 1j * np.asarray(a[pre + "_kj_im"][ic], np.float64)
        wx, wj = max(wx, np.abs(ex - kx).max()), max(wj, np.abs(ej - kj).max())
        sx, sj = max(sx, np.abs(kx).max()), max(sj, np.abs(kj).max())
        if grp:
            ng = len(a[pre + "_grp"][ic])
            gsum = lambda k: np.asarray(a[k][ic], np.float64).reshape(ng, NT).sum(0) if ng else np.zeros(NT)
            gx = max(gx, np.abs(gsum(pre + "_grp_kx_re") + 1j * gsum(pre + "_grp_kx_im") - ex).max())
            gj = max(gj, np.abs(gsum(pre + "_grp_kj_re") + 1j * gsum(pre + "_grp_kj_im") - ej).max())
        n += 1
    print(f"[branch] {pre}: {n} entries; kx max|d| {wx:.2e} (max|S| {sx:.2e}), "
          f"kj max|d| {wj:.2e} (max|S| {sj:.2e}); groups-sum vs flat kx {gx:.2e} kj {gj:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("off")
    ap.add_argument("on")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--entries", type=int, default=None)
    args = ap.parse_args()
    assert ctr.IONI_KOKOULIN == 0.0, "run with CVH_IONI_KOKOULIN=0"
    for tr, skip in (("tree", ()), ("runtree", ("cfmodel",))):
        bad, miss, nb = same_tree(args.off, args.on, tr, skip)
        print(f"[identity] off vs on, {tr}: {nb} branches, differing {bad}, missing in on {miss}")
        if args.ref:
            bad, miss, nb = same_tree(args.ref, args.off, tr, skip)
            print(f"[identity] ref vs off, {tr}: {nb} branches, differing {bad}, missing in off {miss}")
    print("[cfmodel] off:", uproot.open(args.off)["runtree"]["cfmodel"].array(library="np")[:1],
          " on:", uproot.open(args.on)["runtree"]["cfmodel"].array(library="np")[:1])
    branch_check(args.on, args.entries)


if __name__ == "__main__":
    main()
