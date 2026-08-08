#!/usr/bin/env python3
"""Momentum SCALE closure of the CVH refit against gen truth.

Everything so far tested the WIDTH (k_ms). This tests the SCALE, which is the
quantity the calibration actually delivers, and it needs no new production: the
nominal refit (fitFromGenParms=False) stores refParms and genParms, and the
_sel caches carry genpt, so

    z      = [(q/p)_reco - (q/p)_gen] / sigma          (already cached)
    Dk/k   = [(q/p)_reco - (q/p)_gen] / (q/p)_gen = q * z * sigma * p_gen

The gun samples are the right place for this: gen truth is unambiguous, and
they were refit with **useIdealGeometry=True and useDefaultField=True**, i.e.
no misalignment and the same field as the simulation. So the test isolates the
field treatment and the ENERGY-LOSS model against Geant4's.

Decomposition. Writing the relative curvature bias against momentum,

    Dk/k = a0 + a1/p + q * a2 * p

    a0  constant           -> field scale error (Dp/p = dB/B is p-independent)
    a1  ~ 1/p              -> unaccounted MEAN ENERGY LOSS, in GeV: if the fit
                              under-corrects by dE then Dp/p = -dE/p
    a2  ~ q*p              -> misalignment (a fixed sagitta error gives
                              D(1/p) = const, hence Dk/k ~ p), and must come out
                              ZERO here because the geometry is ideal -- so a2
                              is the built-in null test of the whole procedure.

Estimator: the MEDIAN per bin, because the hadron samples carry a one-sided
tail that drags the mean by up to 1.6e-3 (see tail_symmetry.py). The mean is
reported alongside precisely so that pull stays visible rather than silently
entering the scale.

usage:
  python scale_closure.py [--tag ...] [--suffix _sel] [--nbins 12]
"""
import argparse
import os

import numpy as np

TAGS = ("mugun_ul16", "mugun_lowpt", "kaon_ul16", "pion_ul16", "proton_ul16")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", nargs="+", default=list(TAGS))
    p.add_argument("--suffix", default="_sel")
    p.add_argument("--nbins", type=int, default=10)
    p.add_argument("--nboot", type=int, default=200)
    p.add_argument("--maxchi2", type=float, default=0.,
                   help="upper cut on chisqval/nValidHits; 0 = no cut. Scanning "
                        "this tests whether the species difference in a0 is "
                        "residual tail leakage into the median (it should shrink) "
                        "or a genuine mass-dependent term (it should not)")
    return p.parse_args()


def med_err(x, nboot, rng):
    """Median and its bootstrap error (sigma/sqrt(n) is wrong for heavy tails)."""
    if len(x) < 50:
        return np.nan, np.nan
    m = float(np.median(x))
    bs = np.array([np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(nboot)])
    return m, float(bs.std())


def main():
    args = parse_args()
    rng = np.random.default_rng(20260808)
    here = os.path.dirname(os.path.abspath(__file__))

    print("Dk/k = [(q/p)_reco - (q/p)_gen]/(q/p)_gen ; NEGATIVE Dp/p of the same size.\n"
          "Fit Dk/k = a0 + a1/p + q*a2*p :  a0 = field scale, a1 = missing dE [GeV],\n"
          "a2 = misalignment (MUST be ~0, the geometry is ideal).\n")

    for tag in args.tag:
        path = os.path.join(here, "runs", f"cf_trackres_{tag}{args.suffix}.npz")
        if not os.path.exists(path):
            print(f"=== {tag}: cache missing"); continue
        src = np.load(path)
        if "genpt" not in src.files:
            print(f"=== {tag}: no genpt, re-extract with _sel"); continue
        z = np.asarray(src["z"], float)
        sig = np.asarray(src["sigma"], float)
        eta = np.asarray(src["eta"], float)
        q = np.asarray(src["charge"], float)
        pgen = np.asarray(src["genpt"], float) * np.cosh(eta)
        good = np.isfinite(z) & np.isfinite(sig) & (pgen > 0) & np.isfinite(pgen)
        if args.maxchi2 > 0.:
            if "chisqval" not in src.files:
                print(f"=== {tag}: --maxchi2 needs the _sel cache"); continue
            r = (np.asarray(src["chisqval"], float)
                 / np.maximum(np.asarray(src["nvalidhits"], float), 1.))
            good &= r < args.maxchi2
        z, sig, q, pgen = z[good], sig[good], q[good], pgen[good]
        dkk = q * z * sig * pgen

        print(f"=== {tag}   n={len(dkk)}"
              + (f"   [chi2/hit < {args.maxchi2:g}]" if args.maxchi2 > 0 else ""))
        gm, ge = med_err(dkk, args.nboot, rng)
        print(f"  ALL      median Dk/k = {gm*1e4:+8.3f} +- {ge*1e4:.3f}  x1e-4"
              f"     (mean {np.mean(dkk)*1e4:+8.3f}, tail pull "
              f"{(np.mean(dkk)-gm)*1e4:+.3f})")

        # per-charge medians: a2 (misalignment) is the charge-ODD part
        mp, ep = med_err(dkk[q > 0], args.nboot, rng)
        mm, em = med_err(dkk[q < 0], args.nboot, rng)
        odd, odde = 0.5 * (mp - mm), 0.5 * np.hypot(ep, em)
        even = 0.5 * (mp + mm)
        print(f"  q=+ {mp*1e4:+8.3f}+-{ep*1e4:.3f}   q=- {mm*1e4:+8.3f}+-{em*1e4:.3f}"
              f"   -> even {even*1e4:+8.3f}   ODD {odd*1e4:+8.3f}+-{odde*1e4:.3f}"
              + ("   <-- charge-odd, but geometry is IDEAL" if abs(odd) > 3 * odde else ""))

        # binned profile in p, median per bin, then a weighted 3-parameter fit
        edges = np.quantile(pgen, np.linspace(0, 1, args.nbins + 1))
        edges = np.unique(edges)
        rows = []
        print(f"  {'<p> GeV':>9} {'n':>7}  {'median Dk/k x1e-4':>20}")
        for i in range(len(edges) - 1):
            s = (pgen >= edges[i]) & (pgen < edges[i + 1])
            if s.sum() < 200:
                continue
            for sgn in (+1, -1):
                ss = s & (q == sgn)
                if ss.sum() < 100:
                    continue
                m, e = med_err(dkk[ss], 60, rng)
                if not np.isfinite(e) or e <= 0:
                    continue
                rows.append((float(np.median(pgen[ss])), sgn, m, e))
            sm, se = med_err(dkk[s], 60, rng)
            print(f"  {np.median(pgen[s]):9.2f} {int(s.sum()):7d}  {sm*1e4:+12.3f} +- {se*1e4:.3f}")
        if len(rows) >= 6:
            P = np.array([r[0] for r in rows]); Q = np.array([r[1] for r in rows])
            Y = np.array([r[2] for r in rows]); E = np.array([r[3] for r in rows])
            A = np.vstack([np.ones_like(P), 1. / P, Q * P]).T
            W = 1. / E
            coef, *_ = np.linalg.lstsq(A * W[:, None], Y * W, rcond=None)
            cov = np.linalg.inv((A * W[:, None]).T @ (A * W[:, None]))
            err = np.sqrt(np.diag(cov))
            chi2 = float(np.sum(((A @ coef - Y) / E) ** 2))
            print(f"  FIT  a0(field) = {coef[0]*1e4:+8.3f} +- {err[0]*1e4:.3f} x1e-4"
                  f"   a1(missing dE) = {coef[1]*1e3:+8.3f} +- {err[1]*1e3:.3f} MeV"
                  f"   a2(misalign) = {coef[2]*1e4:+8.4f} +- {err[2]*1e4:.4f} x1e-4/GeV"
                  f"   chi2/ndf = {chi2:.1f}/{len(rows)-3}")
        print()


if __name__ == "__main__":
    main()
