#!/usr/bin/env python3
"""TAIL study: WHAT MAKES THE EXCESS chi2 -- the quantity the tail is.

`tail_mixture.py` showed the residual tail IS the chi2 excess: `P(chi2 prob <
1e-3)` is 0.051 on DY and 0.016 on the J/psi gun against the 0.001 a correct
model implies, and the pull variance tracks `chi2/ndof` with slope 1.  So the
question becomes what drives the chi2, and the three candidates that differ
between the two samples are the MOMENTUM (40 GeV against 2-10, i.e. hit-model
dominated against multiple-scattering dominated), the PILEUP (the gun has
none) and the ALIGNMENT (the gun is ideal).

The momentum one is testable INSIDE the gun, which spans the whole range.

usage: python3 tail_chi2src.py --npz <dy npz> --gun <gun npz>
"""
import argparse, os, sys
import numpy as np
from scipy import stats

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
        cls, names, _ = GB.classify(d)
        base = base & (cls == names.index("signal"))
    zv = np.asarray(d["vtxz"], float)
    nd = np.asarray(d["ndof"], float)
    sub = np.asarray(d.get("vtxdchi2", zv ** 2), float)
    nsub = 1.0
    if need_bs and "bschi2fit" in d:
        sub = sub + np.asarray(d["bschi2fit"], float)
        nsub += 3.0
    ndr = np.maximum(nd - nsub, 1.0)
    chir = (np.asarray(d["chisq"], float) - sub)
    u = stats.chi2.sf(chir, ndr)          # the chi2 probability
    return d, base, chir / ndr, ndr, u, nl


def tab(tag, sel, x, u, chired, edges, xlabel):
    print(f"\n  -- {tag}: {xlabel}")
    print(f"  {'bin':<20s}{'N':>7s}{'med chi2/n':>12s}{'P(prob<1e-3)':>26s}"
          f"{'P(prob<0.01)':>26s}")
    for i in range(len(edges) - 1):
        b = sel & (x >= edges[i]) & (x < edges[i + 1])
        if b.sum() < 50:
            continue
        print(f"  {edges[i]:+8.2f}..{edges[i+1]:+8.2f}{int(b.sum()):>7d}"
              f"{np.median(chired[b]):>12.4f}"
              f"{TC.pm(int((u[b]<1e-3).sum()), int(b.sum())):>26s}"
              f"{TC.pm(int((u[b]<1e-2).sum()), int(b.sum())):>26s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--gun", required=True)
    a = ap.parse_args()
    d, sig, chired, ndr, u, nl = load(a.npz)
    g, gs, gchired, gndr, gu, gnl = load(a.gun, need_bs=False)

    print(f"=== the chi2 excess, DY ({int(sig.sum())}) against the gun "
          f"({int(gs.sum())})")
    print(f"  DY  P(prob<1e-3) {TC.pm(int((u[sig]<1e-3).sum()), int(sig.sum()))}")
    print(f"  gun P(prob<1e-3) {TC.pm(int((gu[gs]<1e-3).sum()), int(gs.sum()))}"
          f"   expected 0.00100")

    ptsoft = np.minimum(d["pt_plus"], d["pt_minus"])
    gptsoft = np.minimum(g["pt_plus"], g["pt_minus"])
    tab("GUN", gs, gptsoft, gu, gchired,
        np.array([0, 2, 3, 4, 6, 8, 12, 20, 40, 1e3]), "softer leg pT [GeV]")
    tab("DY", sig, ptsoft, u, chired,
        np.array([0, 28, 33, 38, 45, 1e3]), "softer leg pT [GeV]")
    tab("DY", sig, d["ntrueint"], u, chired,
        np.array([0, 12, 17, 22, 27, 32, 100]), "nTrueInt (pileup)")
    tab("DY", sig, np.maximum(np.abs(d["eta_plus"]), np.abs(d["eta_minus"])),
        u, chired, np.array([0, .9, 1.4, 1.8, 2.05, 2.2, 2.4]), "max |eta|")
    tab("GUN", gs, np.maximum(np.abs(g["eta_plus"]), np.abs(g["eta_minus"])),
        gu, gchired, np.array([0, .9, 1.4, 1.8, 2.05, 2.2, 2.4]), "max |eta|")
    tab("DY", sig, np.minimum(d["nvalidpixel_plus"], d["nvalidpixel_minus"]).astype(float),
        u, chired, np.array([0, 2, 3, 4, 9]), "weaker leg pixel hits")

    # the gun at DY-like momentum is the control for the pT explanation
    b = gs & (gptsoft > 12)
    print(f"\n  the GUN above 12 GeV: N {int(b.sum())}, "
          f"P(prob<1e-3) {TC.pm(int((gu[b]<1e-3).sum()), int(b.sum()))}"
          f"   median chi2/ndof {np.median(gchired[b]):.4f}")
    # and DY at the lowest pileup
    b = sig & (d["ntrueint"] < 12)
    print(f"  DY below nTrueInt 12:   N {int(b.sum())}, "
          f"P(prob<1e-3) {TC.pm(int((u[b]<1e-3).sum()), int(b.sum()))}"
          f"   median chi2/ndof {np.median(chired[b]):.4f}")
    b = sig & (d["ntrueint"] < 12) & (np.maximum(np.abs(d["eta_plus"]), np.abs(d["eta_minus"])) < 1.4)
    print(f"  DY, nTrueInt<12 and |eta|<1.4: N {int(b.sum())}, "
          f"P(prob<1e-3) {TC.pm(int((u[b]<1e-3).sum()), int(b.sum()))}")


if __name__ == "__main__":
    main()
