#!/usr/bin/env python3
"""mu_eff / sigma2_eff from the EXACT inverted density, not the saddlepoint.

WHY THIS EXISTS. cgf_irls.py built the effective-Gaussian block from the
leading saddlepoint identities sigma2_eff = K''(theta_hat),
mu_eff = r - theta_hat K''. That FAILED validation on 2026-08-13: the score
came out 3-4 orders of magnitude away from theta_hat, because the neglected
prefactor term -0.5 K'''/(K'')^2 DOMINATES rather than corrects. Root cause is
the trap already recorded in Phase 0 -- kappa2 is not a width. Here kappa2 =
5865 in z units against a core of ~1, so solving K'(theta) = r anywhere in the
core gives theta ~ 1e-5, a tilt too small to reshape anything, and all the
r-dependence sits in the prefactor.

So this module drops the saddlepoint entirely and works from the exact density,
inverted from the same characteristic function the closure test already uses
(model_phi -> the FULL block: ionization + multiple scattering + radiative,
which is what a fit block actually contains, not ionization alone).

WHAT IT COMPUTES. The local Gaussian surrogate matching -ln p in gradient and
curvature at r:
    psi(r)      = -dln p/dr
    sigma2_eff  = 1 / (-d2ln p/dr2)
    mu_eff      = r - sigma2_eff * psi(r)
Gaussian check: p = N(mu, s2) gives psi = (r-mu)/s2 and -d2ln p/dr2 = 1/s2, so
mu_eff = mu and sigma2_eff = s2 for every r. That is asserted below.

THE THING TO WATCH. sigma2_eff > 0 requires -d2ln p/dr2 > 0, i.e. p LOG-CONCAVE
at r. The ionization density is NOT log-concave in its power-law tail, so there
is a region where no positive-variance Gaussian surrogate exists at all. That
region is found and reported here rather than clipped away, because it decides
whether plain IRLS can work or whether the scheme needs a trust region / bounded
surrogate instead.

usage: python cgf_irls_exact.py --model model.root [--plane K] [--func qop]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cf_propagation_test import (load_model, model_phi, model_variance,
                                 FUNCTIONALS, TAU)

# Fixed grid, deliberately NOT scaled by sqrt(kappa2) -- see module docstring.
ZLO, ZHI, NZ = -30.0, 30.0, 4801


def exact_density(legs, k, avec, sigma, zg, ntau=20000):
    """p(z) by direct inversion of the model characteristic function."""
    phi = model_phi(legs, k, avec, sigma, TAU)
    tu = np.linspace(0.0, TAU[-1], ntau)
    phiu = np.interp(tu, TAU, phi.real) + 1j * np.interp(tu, TAU, phi.imag)
    return np.array([np.trapezoid((phiu * np.exp(-1j * tu * zz)).real, tu) / np.pi
                     for zz in zg])


def surrogate_from_density(zg, pz, floor=1e-300):
    """(psi, curv, sigma2_eff, mu_eff) on the grid, from finite differences."""
    h = zg[1] - zg[0]
    lp = np.log(np.maximum(pz, floor))
    psi = np.full_like(zg, np.nan)
    cur = np.full_like(zg, np.nan)
    psi[1:-1] = -(lp[2:] - lp[:-2]) / (2 * h)
    cur[1:-1] = -(lp[2:] - 2 * lp[1:-1] + lp[:-2]) / h ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        s2 = np.where(cur > 0, 1.0 / cur, np.nan)
    mu = zg - s2 * psi
    # a density that has underflowed carries no information
    bad = pz <= floor * 10
    psi[bad] = cur[bad] = s2[bad] = mu[bad] = np.nan
    return psi, cur, s2, mu


def logconcave_interval(zg, cur, i0):
    """Largest CONTIGUOUS run of positive curvature containing index i0."""
    if not np.isfinite(cur[i0]) or cur[i0] <= 0:
        return None
    lo = i0
    while lo > 0 and np.isfinite(cur[lo - 1]) and cur[lo - 1] > 0:
        lo -= 1
    hi = i0
    while hi < len(cur) - 1 and np.isfinite(cur[hi + 1]) and cur[hi + 1] > 0:
        hi += 1
    return lo, hi


def mode_centred_report(zg, pz, psi, cur, s2, label=""):
    """Is the log-concave region, centred on the MODE, big enough to work in?

    The decisive number is not the width of the region but the PROBABILITY MASS
    inside it: that is the fraction of real residuals for which a
    positive-variance Gaussian surrogate exists at all.
    """
    h = zg[1] - zg[0]
    norm = np.trapezoid(pz, zg)
    imode = int(np.nanargmax(pz))
    zm = zg[imode]
    iv = logconcave_interval(zg, cur, imode)
    if iv is None:
        print(f"{label}mode at z={zm:+.3f} is NOT in a log-concave region")
        return None
    lo, hi = iv
    mass = np.trapezoid(pz[lo:hi + 1], zg[lo:hi + 1]) / norm
    s2m = 1.0 / cur[imode] if cur[imode] > 0 else np.nan
    print(f"{label}mode z={zm:+.3f}  log-concave (mode-centred) "
          f"[{zg[lo]-zm:+.2f}, {zg[hi]-zm:+.2f}]  mass {100*mass:5.1f} %  "
          f"sigma2@mode {s2m:8.3f}")
    return zm, zg[lo] - zm, zg[hi] - zm, mass, s2m


def _selftest():
    """A unit Gaussian must return mu_eff = 0 and sigma2_eff = 1 everywhere."""
    zg = np.linspace(-6, 6, 4001)
    pz = np.exp(-0.5 * zg ** 2) / np.sqrt(2 * np.pi)
    _, _, s2, mu = surrogate_from_density(zg, pz)
    m = np.isfinite(s2) & (np.abs(zg) < 4)
    assert np.allclose(s2[m], 1.0, atol=2e-3), f"sigma2 {np.nanmax(np.abs(s2[m]-1)):.2e}"
    assert np.allclose(mu[m], 0.0, atol=2e-3), f"mu {np.nanmax(np.abs(mu[m])):.2e}"
    return float(np.nanmax(np.abs(s2[m] - 1))), float(np.nanmax(np.abs(mu[m])))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--plane", type=int, default=None)
    p.add_argument("--func", default="qop")
    args = p.parse_args()

    ds2, dmu = _selftest()
    print(f"selftest (unit Gaussian): max|sigma2-1| = {ds2:.2e}, "
          f"max|mu| = {dmu:.2e}  -> identities and differencing are right\n")

    legs = load_model(args.model)
    k = args.plane if args.plane is not None else len(legs) - 1
    avec = FUNCTIONALS[args.func]
    var, _, _ = model_variance(legs, k, avec)
    sigma = float(np.sqrt(var))

    zg = np.linspace(ZLO, ZHI, NZ)
    pz = exact_density(legs, k, avec, sigma, zg)
    psi, cur, s2, mu = surrogate_from_density(zg, pz)

    imode = int(np.nanargmax(pz))
    print(f"plane {k}, func {args.func}, sigma = {sigma:.4e}")
    print(f"mode of the exact density at z = {zg[imode]:+.3f}  "
          f"(mean is 0 by construction)\n")

    print(f"{'z':>7} {'p(z)':>11} {'psi':>10} {'-d2lnp':>10} {'sig2_eff':>10} {'mu_eff':>10}")
    print("-" * 64)
    for zq in (-4., -2., -1., 0., 1., 2., 3., 4., 6., 8., 12.):
        i = int(np.argmin(np.abs(zg - zq)))
        print(f"{zg[i]:7.2f} {pz[i]:11.4e} {psi[i]:10.4f} {cur[i]:10.4f} "
              f"{s2[i]:10.4f} {mu[i]:10.4f}")

    ok = np.isfinite(s2)
    core = ok & (np.abs(zg) <= 10)
    print(f"\nLOG-CONCAVITY (decides whether plain IRLS is even well posed):")
    if core.any():
        zz = zg[core]
        print(f"  positive curvature on z in [{zz.min():+.2f}, {zz.max():+.2f}] "
              f"({100.*core.sum()/np.sum(np.abs(zg)<=10):.1f} % of |z|<=10)")
    neg = np.isfinite(cur) & (cur <= 0) & (np.abs(zg) <= 10)
    if neg.any():
        zz = zg[neg]
        print(f"  NON-log-concave (no positive-variance surrogate) at "
              f"z in [{zz.min():+.2f}, {zz.max():+.2f}], {neg.sum()} grid points")
    else:
        print("  no non-log-concave region found within |z|<=10")
    fin = np.isfinite(s2) & (np.abs(zg) <= 6)
    if fin.any():
        print(f"\n  sigma2_eff over |z|<=6: min {np.nanmin(s2[fin]):.3f}, "
              f"median {np.nanmedian(s2[fin]):.3f}, max {np.nanmax(s2[fin]):.3f}")
        print("  (z is standardized by the TRUNCATED sigma, so ~1 means the "
              "surrogate\n   reproduces the current block; far from 1 means it does not)")

    print("\nMODE-CENTRED SURROGATE, all planes")
    print("At the mode psi = 0 exactly, so mu_eff = z_mode and sigma2_eff is")
    print("1/curvature there -- the natural block, with no truncation anywhere.")
    print("'mass' = fraction of residuals for which a surrogate EXISTS.\n")
    for kk in range(len(legs)):
        try:
            v, _, _ = model_variance(legs, kk, avec)
            sg = float(np.sqrt(v))
            pk = exact_density(legs, kk, avec, sg, zg)
            ps, cu, ss, _ = surrogate_from_density(zg, pk)
            mode_centred_report(zg, pk, ps, cu, ss, label=f"  plane {kk:2d}: ")
        except Exception as e:
            print(f"  plane {kk:2d}: failed ({type(e).__name__})")


if __name__ == "__main__":
    main()
