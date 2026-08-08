#!/usr/bin/env python3
"""Is the hadron resolution tail SYMMETRIC?

This decides how much work the nuclear-elastic problem deserves.

The hadron k_ms deficits (K +0.22, pi +0.30, p +0.39) are a missing
large-deflection tail, not a mis-scaled core (rob68 within 4% of the muon's,
std/rob68 2.6-4.5 against 1.18). For the B -> J/psi K mass constraint the
consequence depends entirely on the tail's symmetry:

  * SYMMETRIC -> it inflates the uncertainty on the extracted kaon curvature
    but does not bias it. Vetoing the taggable part and widening the error is
    then sufficient, and no new CF term is needed.
  * ASYMMETRIC -> it pulls the mean, so it biases the B mass constraint and
    must be modelled -- nuclear elastic scattering added to the CF as a second
    compound-Poisson term alongside Moliere.

The statistic is therefore not skewness for its own sake but **how far the tail
drags the mean away from the median**, in units of the residual's own width.
mean - median is the cleanest single number: the median is tail-blind, the mean
is not, so their difference IS the tail's pull.

z here is the standardized curvature residual (refParms[0] - genParms[0])/sigma,
i.e. exactly the quantity the mass constraint uses. It is DATA and does not
depend on the transport model, so any cache generation may be used.

A charge split is included because it separates the two ways a tail can be
asymmetric: a curvature-like (charge-ODD) pull, which would alias directly into
the momentum scale, versus a charge-EVEN pull, which would not.

usage:
  python tail_symmetry.py [--tag ...] [--suffix _fix] [--cuts 2 3 5 10]
"""
import argparse
import os

import numpy as np

TAGS = ("mugun_ul16", "mugun_lowpt", "kaon_ul16", "pion_ul16", "proton_ul16")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", nargs="+", default=list(TAGS))
    p.add_argument("--suffix", default="_fix")
    p.add_argument("--cuts", nargs="+", type=float, default=[2., 3., 5., 10.])
    p.add_argument("--nboot", type=int, default=200,
                   help="bootstrap replicas for the uncertainty on mean-median")
    return p.parse_args()


def robust_width(z):
    """Half the 68% interval -- a width the tail cannot inflate."""
    lo, hi = np.percentile(z, [15.865, 84.135])
    return 0.5 * (hi - lo)


def summarize(z, cuts, nboot, rng, label, width=None):
    n = len(z)
    w = robust_width(z) if width is None else width
    med = float(np.median(z))
    mean = float(np.mean(z))
    pull = (mean - med) / w                      # tail's drag on the mean, in widths
    # bootstrap the pull: with heavy tails its error is NOT sigma/sqrt(n)
    bs = np.empty(nboot)
    for i in range(nboot):
        s = z[rng.integers(0, n, n)]
        bs[i] = (s.mean() - np.median(s)) / w
    err = float(bs.std())
    print(f"  {label:22} n={n:7d}  rob68={w:6.3f}  med={med:+.4f}  mean={mean:+.4f}  "
          f"(mean-med)/rob68 = {pull:+.4f} +- {err:.4f}"
          + ("   <-- SIGNIFICANT" if abs(pull) > 3 * err else ""))
    # tail balance
    row = []
    for c in cuts:
        nlo = int((z < med - c * w).sum())
        nhi = int((z > med + c * w).sum())
        tot = nlo + nhi
        # asymmetry with a binomial error; +1 means all tail on the high side
        asym = (nhi - nlo) / tot if tot else 0.
        aerr = 1. / np.sqrt(tot) if tot else 0.
        row.append(f"{c:g}:{nlo}/{nhi}({asym:+.3f}+-{aerr:.3f})")
    print(f"    tail lo/hi beyond N*rob68:  " + "   ".join(row))
    return pull, err


def main():
    args = parse_args()
    rng = np.random.default_rng(20260808)
    here = os.path.dirname(os.path.abspath(__file__))
    print("(mean-med)/rob68 is the tail's pull on the mean, in robust widths.\n"
          "Tail counts are entries beyond N*rob68 BELOW / ABOVE the median;\n"
          "asymmetry (nhi-nlo)/(nhi+nlo) is 0 for a symmetric tail.\n")
    for tag in args.tag:
        path = os.path.join(here, "runs", f"cf_trackres_{tag}{args.suffix}.npz")
        if not os.path.exists(path):
            print(f"=== {tag}: cache missing"); continue
        src = np.load(path)
        z = np.asarray(src["z"], dtype=np.float64)       # small; do not touch Sms
        print(f"=== {tag}")
        summarize(z, args.cuts, args.nboot, rng, "all")
        if "charge" in src.files:
            q = np.asarray(src["charge"], dtype=np.float64)
            w = robust_width(z)   # common width so the two are comparable
            for sgn, nm in ((+1, "charge +"), (-1, "charge -")):
                sel = q == sgn
                if sel.sum() > 5000:
                    summarize(z[sel], args.cuts, args.nboot, rng, nm, width=w)
        print()


if __name__ == "__main__":
    main()
