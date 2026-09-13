#!/usr/bin/env python3
"""TAIL study: WHERE the tail lives -- eta, chi2/ndof, and the selection.

The dump of the tail candidates showed two things at once: they sit at
|eta| ~ 2.0-2.4 and they carry chi2/ndof ~ 2.5 against the sample's 0.96,
i.e. right against the extraction's `chi2/ndof < 3`.  Both have to be
quantified, and the second means the published tail is quoted on a sample the
cut has already trimmed -- so the tail is measured here with the cut relaxed
as well.

usage: python3 tail_scan.py --npz <tail/dy_bs_final.npz> [--gun <gun npz>]
"""
import argparse, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import genbkg as GB          # noqa: E402
import tail_common as TC     # noqa: E402


def scan(sel, x, zs, edges, label, chi2n=None):
    print(f"\n  -- {label}")
    hdr = (f"  {'bin':<20s}{'N':>7s}" +
           "".join(f"{'P(|'+nm+'|>3)':>24s}{'P(|'+nm+'|>5)':>24s}{'Var':>9s}"
                   for nm, _ in zs))
    print(hdr)
    for i in range(len(edges) - 1):
        b = sel & (x >= edges[i]) & (x < edges[i + 1])
        if b.sum() < 20:
            continue
        line = f"  {edges[i]:+8.2f}..{edges[i+1]:+8.2f}{int(b.sum()):>7d}"
        for nm, z in zs:
            line += (f"{TC.pm(int((np.abs(z[b])>3).sum()), int(b.sum())):>24s}"
                     f"{TC.pm(int((np.abs(z[b])>5).sum()), int(b.sum())):>24s}"
                     f"{np.var(z[b]):>9.3f}")
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--gun", default="")
    a = ap.parse_args()
    d = dict(np.load(a.npz, allow_pickle=False))
    cls, names, aux = GB.classify(d)
    isig = names.index("signal")
    z1, z2 = d["bsz"][:, 0], d["bsz"][:, 1]
    zv = np.asarray(d["vtxz"], float)
    zs = [("z_1", z1), ("z_2", z2), ("z_v", zv)]
    etamax = np.maximum(np.abs(d["eta_plus"]), np.abs(d["eta_minus"]))

    for lbl, kw in (("WITH chi2/ndof < 3 (the published baseline)",
                     dict(chi2=3.0)),
                    ("WITHOUT the chi2 cut", dict(chi2=0.0))):
        base, chi2n, nl = TC.baseline(d, max_abs_vtxz=0.0, verbose=False, **kw)
        sig = base & (cls == isig)
        print(f"\n=== {lbl}: {int(sig.sum())} gen-signal candidates")
        for nm, z in zs:
            TC.moments(z[sig], nm)
        scan(sig, etamax, zs, np.array([0., .9, 1.4, 1.8, 2.05, 2.2, 2.4]),
             "max |eta| of the two legs")
        scan(sig, chi2n, zs, np.array([0., .8, 1.0, 1.3, 1.8, 2.4, 3.0, 6.0, 1e3]),
             "chi2 / ndof")

    base, chi2n, nl = TC.baseline(d, max_abs_vtxz=0.0, verbose=False)
    sig = base & (cls == isig)
    print("\n=== the two variables are not the same one: chi2/ndof against "
          "max |eta| (gen signal, chi2 cut applied)")
    for lo, hi in ((0., 1.4), (1.4, 1.8), (1.8, 2.05), (2.05, 2.4)):
        b = sig & (etamax >= lo) & (etamax < hi)
        print(f"  |eta| {lo:.2f}-{hi:.2f}  N {int(b.sum()):6d}  "
              f"median chi2/ndof {np.median(chi2n[b]):.4f}  "
              f"p90 {np.percentile(chi2n[b],90):.4f}  "
              f"P(chi2/ndof>2) {TC.pm(int((chi2n[b]>2).sum()), int(b.sum()))}")
    print("\n  -- the tail at FIXED chi2/ndof, in eta bins (is eta anything "
          "beyond chi2?)")
    for lo, hi in ((0., 1.3), (1.3, 3.0)):
        c = sig & (chi2n >= lo) & (chi2n < hi)
        for elo, ehi in ((0., 1.8), (1.8, 2.4)):
            b = c & (etamax >= elo) & (etamax < ehi)
            if b.sum() < 20:
                continue
            print(f"  chi2/ndof {lo:.1f}-{hi:.1f}, |eta| {elo:.1f}-{ehi:.1f}: "
                  f"N {int(b.sum()):6d}  "
                  f"P(|z_1|>3) {TC.pm(int((np.abs(z1[b])>3).sum()), int(b.sum()))}  "
                  f"P(|z_1|>5) {TC.pm(int((np.abs(z1[b])>5).sum()), int(b.sum()))}  "
                  f"Var(z_1) {np.var(z1[b]):.3f}")

    if a.gun and os.path.exists(a.gun):
        g = dict(np.load(a.gun, allow_pickle=False))
        gb, gchi2, _ = TC.baseline(g, need_bs=False, max_abs_vtxz=0.0,
                                   verbose=False)
        gzv = np.asarray(g["vtxz"], float)
        getam = np.maximum(np.abs(g["eta_plus"]), np.abs(g["eta_minus"]))
        print(f"\n=== the J/psi GUN for comparison ({int(gb.sum())} cand)")
        print(f"  median chi2/ndof {np.median(gchi2[gb]):.4f}  "
              f"p90 {np.percentile(gchi2[gb],90):.4f}  "
              f"P(chi2/ndof>2) {TC.pm(int((gchi2[gb]>2).sum()), int(gb.sum()))}")
        scan(gb, getam, [("z_v", gzv)],
             np.array([0., .9, 1.4, 1.8, 2.05, 2.2, 2.4]), "gun: max |eta|")
        scan(gb, gchi2, [("z_v", gzv)],
             np.array([0., .8, 1.0, 1.3, 1.8, 2.4, 3.0]), "gun: chi2/ndof")


if __name__ == "__main__":
    main()
