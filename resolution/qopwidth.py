#!/usr/bin/env python3
"""Where does the real-geometry q/p closure come from -- width, or shape?

The closure is a single bounded number; it says the data is narrower or wider
than the model but not by how much or in what part of the distribution.  This
turns it into two statements that can be read directly:

    width   a robust CORE width of the data against the SAME statistic computed
            on the model's own predicted density (cgf_channels.exact_density,
            the density fisher_norm inverts for 1/I).  Any common scale cancels
            in the ratio, so it does not matter that the closure standardizes by
            s_F and this by sigma.

    ushape  the closure across the whole u grid, per plane.  u ~ 1 weights the
            core, small u the far tail.  A width error is a core effect and
            shows at large u; a missing tail shows at small u.

WHY NOT BIN BY ENERGY LOSS.  The obvious selector for "did this ray hit extra
material" is its total dE -- and for the q/p functional dE IS the residual under
test, so the split is circular.  That is the trap NOTES_GEOMCLOSURE s9.2
documents for `locx` (the |locx| wander split returns +0.32 / -0.32, the cut
talking).  The non-circular selectors are |locy| wander, which
NOTES_GEOMCLOSURE s10 uses, and this, which selects nothing at all.

usage (from calibration_studies/resolution, after `source ../setup_env.sh`):
    python qopwidth.py --geoms real realmat --func qop
"""
import argparse

import numpy as np

import cgf_channels as cc
import fisher_norm as fn
import geom_closure as gc
import hbasis


def model_quantiles(z, p, qs):
    """Quantiles of the model's predicted density, by trapezoid CDF."""
    p = np.maximum(np.asarray(p, float), 0.0)
    c = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(z))])
    if c[-1] <= 0:
        return np.full(len(qs), np.nan)
    return np.interp(np.asarray(qs) * c[-1], c, z)


def fwhm(z, p):
    p = np.asarray(p, float)
    i = int(np.argmax(p))
    half = 0.5 * p[i]
    lo = np.where(p[:i] <= half)[0]
    hi = np.where(p[i:] <= half)[0]
    if not len(lo) or not len(hi):
        return np.nan
    return float(z[i + hi[0]] - z[lo[-1]])


def sample_fwhm(x, nb=400):
    h, e = np.histogram(x, bins=nb, range=(np.quantile(x, 0.001),
                                           np.quantile(x, 0.999)))
    c = 0.5 * (e[1:] + e[:-1])
    return fwhm(c, h.astype(float))


def cmd_perleg(args):
    """WHERE the reference and the bulk come apart, leg by leg.

    Both the closure and the median-of-z of `width` are properties of the
    ACCUMULATED state, so a defect entering at one leg marks every plane after
    it. This differences them.

    No CF machinery is involved and none is needed: the quantity that diverges
    is a LOCATION, and a location decomposes exactly. Per leg,

        dE_ref  = the reference's own loss, 1/|refqop[k-1]| - 1/|refqop[k]|
        dE_bulk = the MEDIAN over rays of the same difference

    and their gap is the leg's contribution to the mean-vs-mode offset. Summing
    the gap over legs reproduces the accumulated divergence by construction, so
    the per-leg column is the decomposition of exactly the thing measured, not a
    proxy for it. The median is used on the data side because the mean is
    dragged by the Landau tail, which is not what is being localized.
    """
    for g in args.geoms:
        legs, sim = gc.geom_model(g), gc.geom_sim(g)
        n = len(legs)
        print(f"\n=== {g}: per-leg mean-vs-mode gap [MeV]")
        print(f"{'leg':>3} {'r[cm]':>8} {'dE_ref':>9} {'dE_bulk':>9} "
              f"{'gap':>9} {'cum gap':>9} {'gap/dE_ref':>11}")
        cum = 0.0
        for k in range(1, n):
            ok = (sim["valid"][:, k] & sim["valid"][:, k - 1]
                  & np.isfinite(sim["qop"][:, k]) & np.isfinite(sim["qop"][:, k - 1]))
            pk = 1.0 / np.abs(sim["qop"][ok, k])
            pj = 1.0 / np.abs(sim["qop"][ok, k - 1])
            dbulk = 1e3 * float(np.median(pj - pk))
            dref = 1e3 * (1.0 / abs(legs[k - 1]["refqop"]) - 1.0 / abs(legs[k]["refqop"]))
            gap = dref - dbulk
            cum += gap
            frac = gap / dref if dref else np.nan
            print(f"{k:3d} {np.nanmedian(sim['globr'][:, k]):8.2f} {dref:9.4f} "
                  f"{dbulk:9.4f} {gap:+9.4f} {cum:+9.4f} {frac:11.4f}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geoms", nargs="+", default=["real", "realmat"])
    ap.add_argument("--func", default="qop")
    ap.add_argument("--useh", action="store_true", default=True)
    ap.add_argument("--perleg", action="store_true",
                    help="the per-LEG decomposition of the location gap")
    args = ap.parse_args()

    if args.perleg:
        cmd_perleg(args)
        return

    if args.useh:
        hbasis.set_use_h(True)

    for g in args.geoms:
        legs, sim = gc.geom_model(g), gc.geom_sim(g)
        mp = gc.GEOMS[g]["model"]
        import os
        hbasis.bind(legs, mp if os.path.sep in mp
                    else os.path.join(gc.SCRATCH, mp), pdg=13)
        avecs = hbasis.avecs(legs, args.func)
        sb, rb = fn.SIM_BRANCH[args.func], fn.REF_BRANCH[args.func]
        print(f"\n=== {g}: {args.func}, {len(legs)} planes  "
              f"(ratios: data / model, so <1 means the MODEL IS TOO WIDE)")
        print(f"{'k':>3} {'r[cm]':>8} {'w68 ratio':>10} {'FWHM ratio':>11} "
              f"{'med(z) data':>12} {'med(z) model':>13} {'n':>8}")
        for k in range(len(legs)):
            avec = avecs[k]
            sig = float(np.sqrt(fn.model_variance(legs, k, avec)[0]))
            z, p, _ = cc.exact_density(legs, k, avec, sig)
            good = sim["valid"][:, k] & np.isfinite(sim[sb][:, k])
            d = (sim[sb][good, k] - legs[k][rb]) / sig
            mq = model_quantiles(z, p, [0.15865, 0.5, 0.84135])
            dq = np.quantile(d, [0.15865, 0.5, 0.84135])
            w68 = (dq[2] - dq[0]) / (mq[2] - mq[0])
            fr = sample_fwhm(d) / fwhm(z, p)
            print(f"{k:3d} {np.nanmedian(sim['globr'][:, k]):8.2f} {w68:10.4f} "
                  f"{fr:11.4f} {dq[1]:12.4f} {mq[1]:13.4f} {int(good.sum()):8d}")


if __name__ == "__main__":
    main()
