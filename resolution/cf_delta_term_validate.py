#!/usr/bin/env python3
"""Acceptance test for cf_track_resolution._delta_term_2d.

The imaginary-argument twin of cgf_delta_validate.py. Same integrand, argument
ia instead of b:

    <(e^{iaE} - 1 - iaE)/E^2>  over the 1/E^2 spectrum on [1, w]

This one was never structurally broken -- |e^{iaw}| = 1, so nothing overflowed
and no clip was ever needed -- but both of its branches lost precision, and
worst AT the crossover, which is the regime the muon blocks actually sit in
(w = tmax/e0 ~ 1e8-1e10). Reference is mpmath.quad at 60 dps, which is
independent of the closed form.

Also checks, because they are cheap and would catch a sign or branch error:
  * D(-a) = conj(D(a))   (the integrand is the CF of a real variable)
  * continuity across the series/closed switch at |a w| = 2
  * the a*w-not-a guard: at w = 1e9, a = 1e-6 a series selected on |a| alone
    gives Im = -8.3e-2 against the true -6.5e-6 (the historical bug)

usage: python cf_delta_term_validate.py [--old]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cf_track_resolution import _delta_term_2d

try:
    import mpmath as mp
    HAVE_MP = True
except ImportError:
    HAVE_MP = False

WS = (20.0, 1e4, 1e9, 1e10)
AWS = (1e-6, 1e-3, 1e-2, 3e-2, 5e-2, 7e-2, 1e-1, 1.0, 2.0, 1e1, 1e2)


def ref(a, w, dps=60):
    with mp.workdps(dps):
        a, w = mp.mpf(a), mp.mpf(w)
        N = 1 - 1 / w
        f = lambda E: (mp.e ** (1j * a * E) - 1 - 1j * a * E) / E ** 2
        return complex(mp.quad(f, [1, w]) / N)


def old_delta_term_2d(a, w):
    """The pre-2026-08-13-XX implementation, for the before/after table."""
    from scipy.special import exp1
    a = np.asarray(a, dtype=np.complex128)
    w = np.asarray(w, dtype=np.float64)[:, None]
    ia = 1j * a
    with np.errstate(all="ignore"):
        t1 = (-np.exp(ia * w) / w + np.exp(ia)) + ia * (exp1(-ia) - exp1(-ia * w))
        t1 -= (1.0 - 1.0 / w)
        t1 -= ia * np.log(w)
    norm = 1.0 - 1.0 / w
    out = t1 / norm
    wb = np.broadcast_to(w, a.shape)
    small = np.abs(a) * wb < 5e-2
    if small.any():
        nb = np.broadcast_to(norm, a.shape)
        ar = a.real
        quad = (-0.5 * ar ** 2 * (wb - 1.)
                - 1j * ar ** 3 / 6. * (wb * wb - 1.) / 2.) / nb
        out[small] = quad[small]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    if not HAVE_MP:
        print("mpmath missing -- cannot run the acceptance test")
        return

    print("=" * 92)
    print("1. NEW vs OLD, both against mpmath at 60 dps. rel = |got - ref| / "
          "|ref| on the complex value.")
    print("   branch: S = Taylor series, C = closed form (old switch at "
          "|aw| = 5e-2, new at 2.0)")
    print("=" * 92)
    print(f"{'w':>8} {'a w':>8} {'old br':>7} {'new br':>7} {'rel OLD':>10} "
          f"{'rel NEW':>10} {'gain':>9}")
    print("-" * 92)
    worst_new = worst_old = 0.0
    fails = []
    for w in WS:
        for aw in AWS:
            a = aw / w
            aa, ww = np.array([[a]]), np.array([w])
            new = complex(_delta_term_2d(aa, ww)[0, 0])
            old = complex(old_delta_term_2d(aa, ww)[0, 0])
            r = ref(a, w)
            en = abs(new - r) / abs(r)
            eo = abs(old - r) / abs(r)
            worst_new = max(worst_new, en)
            worst_old = max(worst_old, eo)
            if en > 1e-10:
                fails.append((w, aw, en))
            print(f"{w:8.0e} {aw:8.0e} {'S' if aw < 5e-2 else 'C':>7} "
                  f"{'S' if aw <= 2.0 else 'C':>7} {eo:10.2e} {en:10.2e} "
                  f"{eo/max(en, 1e-300):9.1e}")
    print(f"\nworst OLD {worst_old:.2e}   worst NEW {worst_new:.2e}   "
          f"target <= 1e-10")

    print()
    print("=" * 92)
    print("2. Structural checks")
    print("=" * 92)
    # conjugate symmetry
    emax = 0.0
    for w in WS:
        for aw in AWS:
            a = aw / w
            p = complex(_delta_term_2d(np.array([[a]]), np.array([w]))[0, 0])
            m = complex(_delta_term_2d(np.array([[-a]]), np.array([w]))[0, 0])
            emax = max(emax, abs(m - np.conj(p)) / abs(p))
    print(f"  D(-a) = conj(D(a)):            max rel dev {emax:.2e}")

    # continuity across the new switch. The two samples straddle |a w| = 2 by
    # +-d, so the function itself moves by ~d (dD/dx * x/D = O(1) there); the
    # BRANCH mismatch is what is left once d is small, hence the d scan.
    print("  continuity at |a w| = 2 (jump, which must fall with d and not "
          "floor above ~1e-15):")
    for d in (1e-8, 1e-10, 1e-12):
        cmax = 0.0
        for w in WS:
            a1, a2 = (2.0 - d) / w, (2.0 + d) / w
            v1 = complex(_delta_term_2d(np.array([[a1]]), np.array([w]))[0, 0])
            v2 = complex(_delta_term_2d(np.array([[a2]]), np.array([w]))[0, 0])
            cmax = max(cmax, abs(v2 - v1) / abs(v1))
        print(f"     d = {d:.0e}: max rel jump {cmax:.2e}  "
              f"(expected ~{2*d:.0e} from the function itself)")

    # the historical a*w-vs-a bug
    w, a = 1e9, 1e-6
    got = complex(_delta_term_2d(np.array([[a]]), np.array([w]))[0, 0])
    r = ref(a, w)
    bad_series = complex((-0.5 * a ** 2 * (w - 1.)
                          - 1j * a ** 3 / 6. * (w * w - 1.) / 2.) / (1 - 1 / w))
    print(f"  w=1e9, a=1e-6 (a w = 1e3):     Im true {r.imag:+.3e}, "
          f"got {got.imag:+.3e}, |a|-guarded series would give "
          f"{bad_series.imag:+.3e}")
    print("     -> the switch is on |a|*w; a guard on |a| alone is wrong by "
          f"{abs(bad_series.imag/r.imag):.0f}x here (the 2026-08-08 bug).")

    print()
    if fails:
        print(f"FAIL: {len(fails)} points above 1e-10:")
        for w, aw, e in fails:
            print(f"   w={w:.0e} aw={aw:.0e}: {e:.2e}")
    else:
        print("PASS: every point on the grid agrees with mpmath to better "
              "than 1e-10 relative.")


if __name__ == "__main__":
    main()
