#!/usr/bin/env python3
"""TAIL study: THE TAIL IS THE chi2, and the chi2 is not self-correlation.

`tail_scan.py` found that every residual's variance tracks the candidate's
`chi2/ndof` -- on DY AND on the J/psi gun, which has no misalignment and no
pileup -- and that what differs between the two samples is the DISTRIBUTION of
`chi2/ndof`, not the ladder.  Two things then have to be checked:

 1. the ladder is not the TRIVIAL self-correlation.  `z_v^2` is literally the
    vertex constraint's contribution to `chisqval` and the two beam pulls are
    three more rows, so a candidate at |z| = 5 raises `chi2/ndof` by ~0.8 all
    by itself.  Removing those rows from both the numerator and `ndof` is the
    test.
 2. what the extra chi2 IS: where it lives in eta, whether it follows pileup,
    the hit count, or the momentum, and how far the DY distribution sits from
    the gun's.

usage: python3 tail_chi2.py --npz <dy npz> [--gun <gun npz>]
"""
import argparse, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import genbkg as GB          # noqa: E402
import tail_common as TC     # noqa: E402


def load(fn, need_bs=True):
    d = dict(np.load(fn, allow_pickle=False))
    base, chi2n, nl = TC.baseline(d, need_bs=need_bs, max_abs_vtxz=0.0,
                                  chi2=0.0, verbose=False)
    if "genidx_plus" in d:
        cls, names, aux = GB.classify(d)
        base = base & (cls == names.index("signal"))
    zv = np.asarray(d["vtxz"], float)
    nd = np.asarray(d["ndof"], float)
    chi = np.asarray(d["chisq"], float)
    # the reduced chi2 with the CONSTRAINT rows taken out of both the sum and
    # the count: 1 row for the vertex constraint, 3 for the beam line
    sub = np.asarray(d.get("vtxdchi2", zv ** 2), float)
    nsub = 1.0
    if need_bs and "bschi2fit" in d:
        sub = sub + np.asarray(d["bschi2fit"], float)
        nsub += 3.0
    chired = (chi - sub) / np.maximum(nd - nsub, 1.0)
    return d, base, chi2n, chired, nl, zv


def ladder(sel, x, zs, edges, label):
    print(f"\n  -- {label}")
    for i in range(len(edges) - 1):
        b = sel & (x >= edges[i]) & (x < edges[i + 1])
        if b.sum() < 20:
            continue
        line = f"  {edges[i]:+7.2f}..{edges[i+1]:+7.2f} N {int(b.sum()):6d}"
        for nm, z in zs:
            line += (f"   Var({nm}) {np.var(z[b]):7.3f}"
                     f"  P>3 {(np.abs(z[b])>3).mean():.5f}"
                     f"  P>5 {(np.abs(z[b])>5).mean():.5f}")
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--gun", default="")
    a = ap.parse_args()

    d, sig, chi2n, chired, nl, zv = load(a.npz)
    z1, z2 = d["bsz"][:, 0], d["bsz"][:, 1]
    zs = [("z_1", z1), ("z_2", z2), ("z_v", zv)]
    etamax = np.maximum(np.abs(d["eta_plus"]), np.abs(d["eta_minus"]))
    print(f"# DY gen signal, NO chi2 cut: {int(sig.sum())} candidates")
    print(f"  chi2/ndof         median {np.median(chi2n[sig]):.4f}"
          f"  p90 {np.percentile(chi2n[sig],90):.4f}"
          f"  p99 {np.percentile(chi2n[sig],99):.4f}"
          f"  P(>2) {TC.pm(int((chi2n[sig]>2).sum()), int(sig.sum()))}")
    print(f"  chi2/ndof REDUCED median {np.median(chired[sig]):.4f}"
          f"  p90 {np.percentile(chired[sig],90):.4f}"
          f"  p99 {np.percentile(chired[sig],99):.4f}"
          f"  P(>2) {TC.pm(int((chired[sig]>2).sum()), int(sig.sum()))}")
    print(f"  corr(chi2/ndof, z_1^2) = "
          f"{np.corrcoef(chi2n[sig], z1[sig]**2)[0,1]:+.4f}   with the "
          f"constraint rows removed: "
          f"{np.corrcoef(chired[sig], z1[sig]**2)[0,1]:+.4f}")

    ladder(sig, chired, zs,
           np.array([0., .7, .85, 1.0, 1.2, 1.5, 2.0, 3.0, 1e3]),
           "the REDUCED chi2/ndof (the constraint rows removed from both)")

    print("\n=== where the extra chi2 lives")
    for nm, q, edges in (("max |eta|", etamax, np.array([0, .9, 1.4, 1.8, 2.05, 2.2, 2.4])),
                         ("nTrueInt", d["ntrueint"], np.array([0, 15, 20, 25, 30, 100])),
                         ("weaker leg nvalid", nl.astype(float), np.array([8, 12, 14, 16, 18, 40])),
                         ("softer pT", np.minimum(d["pt_plus"], d["pt_minus"]),
                          np.array([0, 28, 33, 38, 45, 1e3]))):
        print(f"\n  -- {nm}")
        for i in range(len(edges) - 1):
            b = sig & (q >= edges[i]) & (q < edges[i + 1])
            if b.sum() < 20:
                continue
            print(f"  {edges[i]:+8.2f}..{edges[i+1]:+8.2f} N {int(b.sum()):6d}"
                  f"   median chired {np.median(chired[b]):7.4f}"
                  f"   p90 {np.percentile(chired[b],90):7.4f}"
                  f"   P(chired>1.5) {TC.pm(int((chired[b]>1.5).sum()), int(b.sum()))}")

    if a.gun and os.path.exists(a.gun):
        g, gsig, gchi2n, gchired, gnl, gzv = load(a.gun, need_bs=False)
        print(f"\n=== the J/psi GUN ({int(gsig.sum())} candidates), same "
              "quantities -- ideal geometry, no pileup")
        print(f"  chi2/ndof         median {np.median(gchi2n[gsig]):.4f}"
              f"  p90 {np.percentile(gchi2n[gsig],90):.4f}"
              f"  p99 {np.percentile(gchi2n[gsig],99):.4f}"
              f"  P(>2) {TC.pm(int((gchi2n[gsig]>2).sum()), int(gsig.sum()))}")
        print(f"  chi2/ndof REDUCED median {np.median(gchired[gsig]):.4f}"
              f"  p90 {np.percentile(gchired[gsig],90):.4f}"
              f"  p99 {np.percentile(gchired[gsig],99):.4f}"
              f"  P(>2) {TC.pm(int((gchired[gsig]>2).sum()), int(gsig.sum()))}")
        ladder(gsig, gchired, [("z_v", gzv)],
               np.array([0., .7, .85, 1.0, 1.2, 1.5, 2.0, 3.0, 1e3]),
               "gun: the REDUCED chi2/ndof ladder")
        print("\n  -- THE SAME LADDER, THE DIFFERENT POPULATION: the fraction "
              "of candidates in each reduced-chi2 bin")
        edges = np.array([0., .7, .85, 1.0, 1.2, 1.5, 2.0, 3.0, 1e3])
        print(f"  {'bin':<18s}{'DY':>12s}{'gun':>12s}{'ratio':>9s}")
        for i in range(len(edges) - 1):
            fd = ((chired[sig] >= edges[i]) & (chired[sig] < edges[i+1])).mean()
            fg = ((gchired[gsig] >= edges[i]) & (gchired[gsig] < edges[i+1])).mean()
            print(f"  {edges[i]:6.2f}..{edges[i+1]:<10.2f}{fd:>12.5f}{fg:>12.5f}"
                  f"{fd/max(fg,1e-9):>9.2f}")
        # the prediction: DY's tail from the gun's LADDER and DY's population
        print("\n  -- if the ladder is universal, DY's P(|z_v|>5) is the gun's"
              " ladder folded with DY's chi2 population:")
        pred = 0.0
        for i in range(len(edges) - 1):
            gb = gsig & (gchired >= edges[i]) & (gchired < edges[i+1])
            if gb.sum() < 20:
                continue
            p = (np.abs(gzv[gb]) > 5).mean()
            fd = ((chired[sig] >= edges[i]) & (chired[sig] < edges[i+1])).mean()
            pred += p * fd
        print(f"     predicted {pred:.5f}   measured "
              f"{(np.abs(zv[sig])>5).mean():.5f}   gun inclusive "
              f"{(np.abs(gzv[gsig])>5).mean():.5f}")


if __name__ == "__main__":
    main()
