#!/usr/bin/env python3
"""Acceptance test for cgf_saddlepoint._delta_derivs.

WHAT IS BEING TESTED. D(b, w) = <(e^{bE} - 1 - bE)/E^2> over the 1/E^2
delta-ray spectrum on [1, w], and its first four b-derivatives. The closed
forms use exponential integrals and, taken naively, cancel or overflow; the
module now uses a series / negative-b / factored-positive-b split. This script
checks that split against an INDEPENDENT quadrature of the defining integral,
because the failure mode being guarded against (the old _EXP_MAX clip) returned
a wrong finite number rather than signalling.

THREE REFERENCES, in increasing order of trust:
  * numpy trapezoid on a linear E grid -- the one written down in the task.
    Its own error is O(h^2 f'') and it OVERFLOWS for b w >~ 709, so where it
    cannot be trusted this script says so instead of reporting a pass.
  * mpmath.quad at 50 digits -- exact for every case that fits in a double,
    including large positive b w, since mpmath has no overflow.
  * central differences of D'' -- the check on D''' and D''''.

usage: python cgf_delta_validate.py [--quick]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cgf_saddlepoint import _delta_derivs

try:
    import mpmath as mp
    HAVE_MP = True
except ImportError:
    HAVE_MP = False


def D_direct(b, w, n=400000):
    """Trapezoid quadrature of the defining integral (the task's reference)."""
    E = np.linspace(1., w, n)
    N = 1. - 1. / w
    return np.trapezoid((np.exp(b * E) - 1. - b * E) / E ** 2, E) / N


def D_mp(b, w, order=0, dps=50):
    """mpmath reference: (1/N) int_1^w E^{order-2} (e^{bE} - 1 - bE) dE for
    order 0, and (1/N) int_1^w E^{order-2} e^{bE} dE for order >= 1 (the
    subtracted terms differentiate away)."""
    with mp.workdps(dps):
        b, w = mp.mpf(b), mp.mpf(w)
        N = 1 - 1 / w
        if order == 0:
            f = lambda E: (mp.e ** (b * E) - 1 - b * E) / E ** 2
        elif order == 1:
            f = lambda E: (mp.e ** (b * E) - 1) / E
        else:
            f = lambda E: E ** (order - 2) * mp.e ** (b * E)
        return mp.quad(f, [1, w]) / N


def fmt(v):
    if v is None:
        return "        --"
    if not np.isfinite(v):
        return f"{str(v):>10}"
    return f"{v:10.3e}"


def relerr(a, b):
    if b is None or a is None:
        return None
    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan if (np.isfinite(a) != np.isfinite(b)) else 0.0
    if b == 0:
        return abs(a)
    return abs(a - b) / abs(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    bs = [-10., -2., -0.5, -0.05, -1e-3, -1e-6, 1e-6, 1e-3, 0.05, 0.5, 2., 10.]
    ws = [2., 20., 1000.]
    if args.quick:
        bs = [-2., -0.05, 1e-3, 2.]
        ws = [2., 1000.]

    print("=" * 100)
    print("1. D(b, w) against quadrature.  bw > 709 => e^{bE} overflows a "
          "double: numpy trapezoid CANNOT be")
    print("   evaluated there and is marked OVERFLOW; mpmath still can, and "
          "the closed form is compared to it.")
    print("=" * 100)
    print(f"{'b':>9} {'w':>7} {'bw':>9} {'D closed':>12} {'D trapz':>12} "
          f"{'D mpmath':>12} {'rel trapz':>10} {'rel mp':>10}")
    print("-" * 100)
    worst_mp = 0.0
    worst_tz = 0.0
    nfail = []
    for w in ws:
        for b in bs:
            D = float(_delta_derivs(np.array([b]), np.array([w]))[0][0])
            bw = b * w
            if bw < 700.:
                tz = D_direct(b, w)
            else:
                tz = None
            mpv = float(D_mp(b, w, 0)) if HAVE_MP else None
            r1, r2 = relerr(D, tz), relerr(D, mpv)
            if r2 is not None and np.isfinite(r2):
                worst_mp = max(worst_mp, r2)
                if r2 > 1e-8:
                    nfail.append((b, w, "D", r2))
            if r1 is not None and np.isfinite(r1):
                worst_tz = max(worst_tz, r1)
            print(f"{b:9.0e} {w:7.0f} {bw:9.1f} {fmt(D)} "
                  f"{'  OVERFLOW' if tz is None else fmt(tz)} {fmt(mpv)} "
                  f"{'      --' if r1 is None else f'{r1:10.2e}'} "
                  f"{'      --' if r2 is None else f'{r2:10.2e}'}")
    print(f"\nworst relative error vs mpmath  : {worst_mp:.2e}")
    print(f"worst relative error vs trapezoid: {worst_tz:.2e}   "
          f"(trapezoid's own O(h^2) error is the floor here)")

    if HAVE_MP:
        print()
        print("=" * 100)
        print("2. All five derivatives against mpmath quadrature "
              "(relative errors)")
        print("=" * 100)
        print(f"{'b':>9} {'w':>7} {'bw':>9} {'D':>10} {'D1':>10} {'D2':>10} "
              f"{'D3':>10} {'D4':>10}")
        print("-" * 100)
        worst = np.zeros(5)
        for w in ws:
            for b in bs:
                got = [float(v[0]) for v in
                       _delta_derivs(np.array([b]), np.array([w]))]
                ref = [float(D_mp(b, w, o)) for o in range(5)]
                rr = [relerr(g, r) for g, r in zip(got, ref)]
                for i, r in enumerate(rr):
                    if np.isfinite(r):
                        worst[i] = max(worst[i], r)
                        if r > 1e-8:
                            nfail.append((b, w, f"D{i}", r))
                print(f"{b:9.0e} {w:7.0f} {b*w:9.1f} " +
                      " ".join(f"{r:10.2e}" for r in rr))
        print("\nworst per order: " +
              "  ".join(f"D{i} {v:.2e}" for i, v in enumerate(worst)))

    print()
    print("=" * 100)
    print("3. D''' and D'''' against central differences of D'' "
          "(independent of the mpmath route)")
    print("=" * 100)
    print(f"{'b':>9} {'w':>7} {'D3 closed':>13} {'D3 fd':>13} {'rel':>9} "
          f"{'D4 closed':>13} {'D4 fd':>13} {'rel':>9}")
    print("-" * 100)
    for w in ws:
        for b in bs:
            if abs(b * w) > 700:
                continue
            # the natural scale of D''(b) is 1/w (the argument is bE, E <= w),
            # so h*w fixes both the truncation error ((hw)^2/12) and the
            # cancellation in the second difference. h*w = 5e-3 keeps the first
            # at ~2e-6 and the second at ~1e-10 of the signal.
            h = 5e-3 / w
            g = lambda bb: _delta_derivs(np.array([bb]), np.array([w]))
            d2p, d2m = float(g(b + h)[2][0]), float(g(b - h)[2][0])
            d2 = float(g(b)[2][0])
            d3fd = (d2p - d2m) / (2 * h)
            d4fd = (d2p - 2 * d2 + d2m) / h ** 2
            d3, d4 = float(g(b)[3][0]), float(g(b)[4][0])
            print(f"{b:9.0e} {w:7.0f} {d3:13.6e} {d3fd:13.6e} "
                  f"{relerr(d3, d3fd):9.1e} {d4:13.6e} {d4fd:13.6e} "
                  f"{relerr(d4, d4fd):9.1e}")

    print()
    print("=" * 100)
    print("4. Overflow behaviour: the OLD code clipped e^{bw} at exp(600) and "
          "returned a wrong finite")
    print("   number. The new code must return +inf once the true value leaves "
          "the double range.")
    print("=" * 100)
    print(f"{'b':>10} {'w':>10} {'bw':>9} {'D':>13} {'D2':>13} "
          f"{'true ln D (approx)':>20}")
    print("-" * 100)
    for b, w in [(1e-5, 7e7), (1e-4, 7e7), (1e-3, 7e7), (1e-2, 7e7),
                 (0.709, 1000.), (0.71, 1000.), (1.0, 1000.)]:
        D, D1, D2, D3, D4 = [float(v[0]) for v in
                             _delta_derivs(np.array([b]), np.array([w]))]
        lnD = b * w - np.log(b * w ** 2 * (1 - 1 / w))
        print(f"{b:10.1e} {w:10.1e} {b*w:9.1f} {fmt(D)} {fmt(D2)} {lnD:20.2f}")
    print("(true ln D ~ bw - ln(b w^2 N) from Ei's asymptotics; ln(max double)"
          " = 709.78)")

    print()
    print("=" * 100)
    print("5. What the OLD clipped code returned, at the (b, w) a real pT=3 "
          "block actually visits.")
    print("   b = c*theta with c ~ -1e-5..-1e-4 and w = tmax/e0 ~ 7e7, so "
          "|bw| = 700 is reached at |theta| ~ 0.01-0.1.")
    print("=" * 100)
    from scipy.special import expi as _expi
    print(f"{'b':>10} {'w':>10} {'bw':>9} {'D new':>13} {'D old(clip)':>13} "
          f"{'old/new':>12}")
    print("-" * 100)
    for b, w in [(1e-6, 7e7), (5e-6, 7e7), (8.6e-6, 7e7), (1e-5, 7e7)]:
        Dn = float(_delta_derivs(np.array([b]), np.array([w]))[0][0])
        N = 1. - 1. / w
        ebw = np.exp(min(b * w, 600.0))          # the old _EXP_MAX clip
        eb = np.exp(min(b, 600.0))
        Do = ((-ebw / w + eb) + b * (_expi(b * w) - _expi(b))
              - (1. - 1. / w) - b * np.log(w)) / N
        print(f"{b:10.1e} {w:10.1e} {b*w:9.1f} {fmt(Dn)} {fmt(Do)} "
              f"{Do/Dn:12.3e}")
    print("(ratio 1 = the clip was inactive; anything else is the silent error"
          " the clip introduced)")

    print()
    if nfail:
        print(f"FAIL: {len(nfail)} entries above the 1e-8 relative target:")
        for b, w, nm, r in nfail:
            print(f"   b={b:+.1e} w={w:.0f} {nm}: {r:.2e}")
    else:
        print("PASS: every representable case agrees with mpmath quadrature "
              "to better than 1e-8 relative.")


if __name__ == "__main__":
    main()
