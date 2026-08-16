#!/usr/bin/env python3
"""Does the _delta_term_2d precision fix move any physics?

_delta_term_2d feeds ioni_step_exponent -> model_phi, which every
clean-propagation closure number in this study runs through. The fix
(2026-08-13 XX) improved its worst error from 7.3e-4 to 5e-16. That is a large
factor, but a large factor on a small number: this script measures what it
actually does to quantities that have already been recorded.

Recomputed old vs new, on the same model and with the SAME procedure that
produced the recorded numbers (deliberately including its imperfections, since
only the DIFFERENCE is being asked for here):

  * the CF itself           -- max |d phi| and max |d S| on the tau grid. This
                               is the "verify the knob moves the output" check:
                               if the CF does not change at all, nothing
                               downstream can, and the patch did not take.
  * ionization-only         -- 1/I (FFT route, cgf_fisher) and the mode of the
                               inverted density
  * FULL block (model_phi)  -- mode, log-concave interval, mass fraction (the
                               "universal 62 %") and sigma2@mode, via
                               cgf_irls_exact verbatim

usage: python cf_delta_term_impact.py --model M.root [--planes 0,9,18]
"""
import argparse
import contextlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cf_track_resolution as CTR
from cf_delta_term_validate import old_delta_term_2d
from cf_propagation_test import load_model, model_variance, FUNCTIONALS, TAU
from cgf_phase0_validate import collect_ioni_steps
from cgf_fisher import phi_uniform, exact_density_fft, fisher_exact
import cgf_irls_exact as IE


@contextlib.contextmanager
def delta_term(which):
    """Swap the module-level _delta_term_2d. ioni_step_exponent resolves it
    from cf_track_resolution's globals, so this reaches model_phi too."""
    keep = CTR._delta_term_2d
    CTR._delta_term_2d = old_delta_term_2d if which == "old" else keep
    try:
        yield
    finally:
        CTR._delta_term_2d = keep


def parabolic_mode(z, p):
    i = int(np.nanargmax(p))
    if 0 < i < len(z) - 1:
        y0, y1, y2 = p[i - 1], p[i], p[i + 1]
        den = y0 - 2 * y1 + y2
        if den != 0:
            return float(z[i] - 0.5 * (z[1] - z[0]) * (y2 - y0) / den)
    return float(z[i])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--planes", default="0,9,18")
    ap.add_argument("--func", default="qop")
    ap.add_argument("--taumax", type=float, default=2000.0)
    ap.add_argument("--logm", type=int, default=19)
    args = ap.parse_args()

    legs = load_model(args.model)
    avec = FUNCTIONALS[args.func]
    planes = [int(x) for x in args.planes.split(",")]
    M = 1 << args.logm

    steps = {}
    for k in planes:
        var, _, _ = model_variance(legs, k, avec)
        sig = float(np.sqrt(var))
        steps[k] = (collect_ioni_steps(legs, k, avec, sig), sig)

    print("=" * 100)
    print("0. Does the CF actually change? (if not, nothing below can, and the "
          "patch did not take)")
    print("=" * 100)
    tu = np.concatenate([[0.0], np.geomspace(1e-4, 100.0, 400)])
    print(f"{'plane':>5} {'max|dS|':>12} {'max|dS|/|S|':>12} "
          f"{'max|dphi|':>12} {'max|dphi|/|phi|':>16} {'t of max':>9}")
    print("-" * 100)
    for k in planes:
        st, sig = steps[k]
        with delta_term("old"):
            So = CTR.ioni_step_exponent(st, 1.0, tu)
        with delta_term("new"):
            Sn = CTR.ioni_step_exponent(st, 1.0, tu)
        dS = np.abs(Sn - So)
        rel = dS / np.maximum(np.abs(So), 1e-300)
        po, pn = np.exp(So), np.exp(Sn)
        dp = np.abs(pn - po)
        rp = dp / np.maximum(np.abs(po), 1e-300)
        i = int(np.argmax(dS))
        print(f"{k:5d} {dS.max():12.3e} {np.nanmax(rel):12.3e} "
              f"{dp.max():12.3e} {np.nanmax(rp):16.3e} {tu[i]:9.3f}")

    print()
    print("=" * 100)
    print("1. IONIZATION-ONLY block: 1/I and the mode of the inverted density "
          "(FFT route)")
    print("=" * 100)
    print(f"{'plane':>5} | {'1/I old':>10} {'1/I new':>10} {'d/old':>10} | "
          f"{'mode old':>10} {'mode new':>10} {'d':>11}")
    print("-" * 100)
    for k in planes:
        st, sig = steps[k]
        res = {}
        for which in ("old", "new"):
            with delta_term(which):
                pre = phi_uniform(st, args.taumax, M)
                z, p, dp, _ = exact_density_fft(st, args.taumax, M, pre=pre)
            I, d = fisher_exact(z, p, dp, floor=1e-8)
            res[which] = (1.0 / I, parabolic_mode(z, p))
        (io, mo), (inw, mn) = res["old"], res["new"]
        print(f"{k:5d} | {io:10.4f} {inw:10.4f} {(inw-io)/io:10.2e} | "
              f"{mo:10.4f} {mn:10.4f} {mn-mo:11.3e}")

    print()
    print("=" * 100)
    print("2. FULL block (model_phi: ionization + MS + radiative), "
          "cgf_irls_exact procedure verbatim")
    print("   fixed grid [-30, 30] x 4801, TAU as shipped -- the same recipe "
          "that produced the")
    print("   recorded mode and 'universal 62 %' numbers, so the columns are "
          "like-for-like.")
    print("=" * 100)
    zg = np.linspace(IE.ZLO, IE.ZHI, IE.NZ)
    print(f"{'plane':>5} | {'mode old':>9} {'mode new':>9} {'d':>10} | "
          f"{'mass% old':>9} {'mass% new':>9} {'d':>10} | "
          f"{'s2@mode old':>11} {'s2@mode new':>11} {'d/old':>9}")
    print("-" * 100)
    for k in planes:
        var, _, _ = model_variance(legs, k, avec)
        sg = float(np.sqrt(var))
        res = {}
        for which in ("old", "new"):
            with delta_term(which):
                pz = IE.exact_density(legs, k, avec, sg, zg)
            _, cur, _, _ = IE.surrogate_from_density(zg, pz)
            norm = np.trapezoid(pz, zg)
            i0 = int(np.nanargmax(pz))
            iv = IE.logconcave_interval(zg, cur, i0)
            lo, hi = iv
            mass = np.trapezoid(pz[lo:hi + 1], zg[lo:hi + 1]) / norm
            res[which] = (parabolic_mode(zg, pz), 100 * mass,
                          1.0 / cur[i0] if cur[i0] > 0 else np.nan)
        (mo, fo, so), (mn, fn, sn) = res["old"], res["new"]
        print(f"{k:5d} | {mo:9.4f} {mn:9.4f} {mn-mo:10.2e} | "
              f"{fo:9.3f} {fn:9.3f} {fn-fo:10.2e} | "
              f"{so:11.5f} {sn:11.5f} {(sn-so)/so:9.2e}")

    print()
    print("Read: a difference at the 1e-12 level or below is round-off, not a "
          "shift. Anything\nabove ~1e-4 relative would mean a recorded number "
          "needs revising.")


if __name__ == "__main__":
    main()
