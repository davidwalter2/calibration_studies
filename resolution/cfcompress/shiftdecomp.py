#!/usr/bin/env python3
"""Does the compression-induced shift of the minimum SURVIVE at 1e7 candidates?

The shift is, to first order, dtheta = -H^-1 sum_i dg_i with
dg_i = g_i(compressed) - g_i(full) the per-candidate gradient difference at
the uncompressed minimum.  H grows like n, so

  * the MEAN of dg_i gives a shift that is INDEPENDENT of n  (systematic),
  * the SPREAD of dg_i gives a shift that falls like 1/sqrt(n)  (statistical).

Splitting the measured 20k shift into the two tells whether the number
extrapolates to 1e7 candidates or improves.

Usage
  python3 shiftdecomp.py --tag jpsigun --n 20000 --ranks 4,8,16,32
"""
import argparse
import numpy as np
import massnll_np as MN
import cfbasis as CB
import nll_impact as NI

SCRATCH = NI.SCRATCH
FAMS = NI.FAMS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="jpsigun")
    ap.add_argument("--ntrain", type=int, default=40000)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--nproc", type=int, default=4)
    ap.add_argument("--ranks", default="4,8,16,32,48")
    a = ap.parse_args()
    sub = f"{SCRATCH}/sub_{a.tag}_n60000_s1234.npz"
    tr = NI.load(a.tag, sub, 0, a.ntrain)
    ev = NI.load(a.tag, sub, a.ntrain, a.ntrain + a.n)
    m0 = NI.make_model(ev, ev["M"], a.nproc)
    ref = m0.fit()
    n = a.n
    print(f"reference: " + "  ".join(f"{p}={v:+.6f}+-{e:.6f}" for p, v, e
                                     in zip(MN.PARNAMES, ref["x"], ref["err"])))
    g0 = m0.grad_rows(ref["x"])
    print(f"{'basis':14s} {'B/cand':>7s} {'dalpha(20k)':>13s} "
          f"{'+-stat(20k)':>12s} {'predicted at 1e7':>22s}")
    print("  dalpha(20k) is the best estimate of the n-INDEPENDENT systematic;")
    print("  +-stat is its uncertainty from the finite sample, which falls as "
          "1/sqrt(n).")
    for r in [int(x) for x in a.ranks.split(",")]:
        name, apf, nb = NI.fit_basis(tr["M"], "pca_all", r, ev["TG"])
        mm = NI.make_model(ev, apf(ev["M"]), a.nproc)
        dg = mm.grad_rows(ref["x"]) - g0
        tot = -ref["cov"] @ dg.sum(axis=0)
        # systematic: the mean, scaled to any n (H ~ n cancels the sum ~ n)
        # statistical: the spread contributes sqrt(n)*std, i.e. 1/sqrt(n) of
        # the systematic per unit n -> quote both as an alpha shift at THIS n
        stat = np.sqrt(np.einsum("ij,jk,ik->i", ref["cov"],
                                 (dg - dg.mean(0)).T @ (dg - dg.mean(0)),
                                 ref["cov"]))
        # shift at 1e7: systematic stays, statistical scales by sqrt(n/1e7)
        syst = tot
        print(f"{name:14s} {nb:7d} {tot[0]*1e-3:+13.3e} "
              f"{stat[0]*1e-3:12.3e} "
              f"{syst[0]*1e-3:+11.3e} +- {stat[0]*1e-3*np.sqrt(n/1e7):9.1e}")
        del mm


if __name__ == "__main__":
    main()
