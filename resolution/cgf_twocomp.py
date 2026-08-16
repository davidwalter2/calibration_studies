#!/usr/bin/env python3
"""Two-component block prototype: log-concave core + delta-ray remainder.

THE PROPOSAL BEING TESTED. Replace the single Gaussian block (which needs the
alpha truncation to have a finite variance at all) with
    p(z) = w * N(z; z_mode, sigma2@mode)  +  (1-w) * T(z)
where the Gaussian sits on the log-concave core and T carries the delta-ray
remainder. Motivation (NOTES 2026-08-13 XII): the log-concave / non-log-concave
boundary COINCIDES with the physical core / delta-ray split -- a Moyal-like core
is log-concave everywhere, a 1/E^2 channel nowhere -- and the Urban record
already carries the two separately.

THE TEST THAT MATTERS, AND WHY IT IS NOT CIRCULAR. Taking T = p_exact - w*N
would reproduce p_exact by construction and prove nothing. So this asks the
falsifiable question instead:

    inside the log-concave region, is the exact density actually GAUSSIAN?

The core Gaussian is built from the mode and the curvature AT the mode only --
two numbers, no fitting to the shape. If the exact density inside the region
departs from that Gaussian, the proposed decomposition is wrong and the core
needs more than a Gaussian, whatever we do about the tail.

Reported: max |ln p_exact - ln p_gauss| inside the region (a log-density
distance, so it is sensitive in the wings where a linear-scale comparison is
not), and the mass the Gaussian assigns to the region versus the truth.

usage: python cgf_twocomp.py --model model.root [--plane K]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cf_propagation_test import load_model, model_variance, FUNCTIONALS
from cgf_irls_exact import (exact_density, surrogate_from_density,
                            logconcave_interval, ZLO, ZHI, NZ)


def core_gaussian(zg, zm, s2):
    return np.exp(-0.5 * (zg - zm) ** 2 / s2) / np.sqrt(2 * np.pi * s2)


def analyse(legs, k, avec, zg):
    var, _, _ = model_variance(legs, k, avec)
    sigma = float(np.sqrt(var))
    pz = exact_density(legs, k, avec, sigma, zg)
    _, cur, _, _ = surrogate_from_density(zg, pz)
    norm = np.trapezoid(pz, zg)
    pz = pz / norm

    i0 = int(np.nanargmax(pz))
    iv = logconcave_interval(zg, cur, i0)
    if iv is None:
        return None
    lo, hi = iv
    zm = zg[i0]
    s2 = 1.0 / cur[i0]

    g = core_gaussian(zg, zm, s2)
    mass_true = np.trapezoid(pz[lo:hi + 1], zg[lo:hi + 1])
    mass_gauss = np.trapezoid(g[lo:hi + 1], zg[lo:hi + 1])

    # w is fixed by matching the PEAK height, not by fitting the shape:
    # w = p_exact(mode) / N(mode) -- one number, still no shape freedom.
    w = pz[i0] / g[i0]

    sl = slice(lo, hi + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dln = np.log(pz[sl]) - np.log(w * g[sl])
    dln = dln[np.isfinite(dln)]
    return dict(zm=zm, s2=s2, w=w, mass_true=mass_true, mass_gauss=mass_gauss,
                dln_max=float(np.max(np.abs(dln))), dln_rms=float(np.std(dln)),
                lo=zg[lo] - zm, hi=zg[hi] - zm)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--func", default="qop")
    args = p.parse_args()

    legs = load_model(args.model)
    avec = FUNCTIONALS[args.func]
    zg = np.linspace(ZLO, ZHI, NZ)

    print("Is the log-concave core actually Gaussian?")
    print("core built from mode + curvature@mode ONLY (no shape fit); w from")
    print("peak height. dln = ln p_exact - ln(w*N) inside the region.\n")
    print(f"{'plane':>5} {'z_mode':>8} {'sig2':>8} {'w':>6} {'region':>16} "
          f"{'mass_true':>9} {'max|dln|':>9} {'rms dln':>8}")
    print("-" * 78)
    rows = []
    for k in range(len(legs)):
        try:
            r = analyse(legs, k, avec, zg)
        except Exception:
            r = None
        if r is None:
            print(f"{k:5d}   no log-concave region")
            continue
        rows.append(r)
        print(f"{k:5d} {r['zm']:8.3f} {r['s2']:8.4f} {r['w']:6.3f} "
              f"[{r['lo']:+5.2f},{r['hi']:+5.2f}] {r['mass_true']:9.3f} "
              f"{r['dln_max']:9.4f} {r['dln_rms']:8.4f}")

    if rows:
        dm = np.array([r["dln_max"] for r in rows])
        print(f"\nmax|dln| over all planes: min {dm.min():.4f}, "
              f"median {np.median(dm):.4f}, max {dm.max():.4f}")
        print("\nREAD: max|dln| is a LOG-density error inside the core.")
        print("  <~0.05  -> the core is Gaussian to a few %; the two-component")
        print("            decomposition is sound and only T remains to be built.")
        print("  >~0.2   -> the core is NOT Gaussian; a Gaussian surrogate on the")
        print("            log-concave region is the wrong primitive, and no")
        print("            treatment of the tail will rescue it.")


if __name__ == "__main__":
    main()
