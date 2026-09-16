#!/usr/bin/env python3
"""Is the high-u deficit of the measured J/psi FSR kernel acceptance?

Two handles, both acceptance-free at the generator level:

* the LOCAL density ``u dP/dlnu`` compared bin-by-bin with the analytic QED
  kernel -- an overall efficiency cancels in the ratio only if it is flat in
  ``u``, so the SHAPE of the ratio is the acceptance curve;
* the same ratio split by ``Jpsigen_pt``: an acceptance driven by the muon pT
  thresholds and by the ALCARECO dimuon-mass window must depend strongly on
  the J/psi pT, a generator kernel must not.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "/work/submit/david_w/ZMass/calibration_studies/zchannel")
import fsr_analytic as fa                                         # noqa: E402

MJPSI = 3.0969


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--cache", required=True)
    ap.add_argument("--max-chi2-ndof", type=float, default=3.0)
    args = ap.parse_args()

    d = np.load(args.cache)
    mp = d["m_post"].astype(np.float64)
    ml = d["masslep"].astype(np.float64)
    c2 = d["chisqval"].astype(np.float64)
    nd = d["ndof"].astype(np.float64)
    pt = d["pt"].astype(np.float64)
    eta = d["eta"].astype(np.float64)
    ok = (np.isfinite(mp) & (mp > 0) & np.isfinite(ml) & np.isfinite(c2)
          & (nd > 0) & ((c2 / np.maximum(nd, 1.0)) < args.max_chi2_ndof))
    mp, ml, pt, eta = mp[ok], ml[ok], pt[ok], eta[ok]
    u = -np.log(mp / np.where(ml > 0, ml, mp))

    k = fa.FSRKernel(MJPSI, variant="exp2nll", pair=(), mass_exact=True)
    ue = np.geomspace(1e-6, 0.2, 61)
    uc = np.sqrt(ue[:-1] * ue[1:])
    ana = uc * k.pdf_u(uc)

    def dens(sel):
        c, _ = np.histogram(u[sel], bins=ue)
        n = int(sel.sum())
        return c / n / np.diff(np.log(ue)), c, n

    print("=== local density u dP/dlnu vs analytic (inclusive) ===")
    dd, cc, n = dens(np.ones(len(u), bool))
    print(f"n = {n}")
    print(f"{'u':>10} {'meas':>10} {'+-':>9} {'analytic':>10} {'ratio':>8}")
    for c, v, e, a in zip(uc, dd, dd / np.sqrt(np.maximum(cc, 1)), ana):
        print(f"{c:10.3e} {v:10.6f} {e:9.6f} {a:10.6f} {v / a:8.4f}")

    print()
    print("=== ratio meas/analytic in Jpsigen_pt bins ===")
    edges = [0, 8, 10, 12, 15, 20, 30, 1e9]
    ug = np.array([1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 5.66e-2, 0.113])
    hdr = "  ".join(f"{x:9.2e}" for x in ug)
    print(f"{'pt bin':>14} {'n':>9}   {hdr}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (pt >= lo) & (pt < hi)
        if s.sum() < 1000:
            continue
        r = [(u[s] > x).mean() / k.tail(np.array([x]))[0] for x in ug]
        lab = f"{lo:g}-{hi:g}" if hi < 1e8 else f">{lo:g}"
        print(f"{lab:>14} {int(s.sum()):9d}   "
              + "  ".join(f"{x:9.4f}" for x in r))

    print()
    print("=== CONDITIONAL P(u>u0 | u<0.0566), meas/analytic, in pt bins ===")
    print("    (the 2.7 GeV ALCARECO window is at u = 0.1376, far outside;")
    print("     a residual pt dependence here would mean acceptance, none"
          " means kernel)")
    ucut = 0.0566
    pac = float(k.tail(np.array([ucut]))[0])
    ug2 = np.array([1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2])
    anac = [(float(k.tail(np.array([x]))[0]) - pac) / (1.0 - pac) for x in ug2]
    hdr2 = "  ".join(f"{x:8.1e}" for x in ug2)
    print(f"{'pt bin':>14} {'n':>9}   {hdr2}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        s3 = (pt >= lo) & (pt < hi) & (u >= 0) & (u < ucut)
        if s3.sum() < 5000:
            continue
        r = [(u[s3] > x).mean() / a for x, a in zip(ug2, anac)]
        lab = f"{lo:g}-{hi:g}" if hi < 1e8 else f">{lo:g}"
        print(f"{lab:>14} {int(s3.sum()):9d}   "
              + "  ".join(f"{x:8.4f}" for x in r))
    for lo, hi in [(0, 0.8), (0.8, 1.5), (1.5, 2.4), (2.4, 10)]:
        s3 = ((np.abs(eta) >= lo) & (np.abs(eta) < hi) & (u >= 0)
              & (u < ucut))
        if s3.sum() < 5000:
            continue
        r = [(u[s3] > x).mean() / a for x, a in zip(ug2, anac)]
        print(f"|eta| {lo:g}-{hi:g}".rjust(14) + f" {int(s3.sum()):9d}   "
              + "  ".join(f"{x:8.4f}" for x in r))

    print()
    print("=== ratio meas/analytic in |Jpsigen_eta| bins ===")
    ae = np.abs(eta)
    for lo, hi in [(0, 0.8), (0.8, 1.5), (1.5, 2.4), (2.4, 10)]:
        s = (ae >= lo) & (ae < hi)
        if s.sum() < 1000:
            continue
        r = [(u[s] > x).mean() / k.tail(np.array([x]))[0] for x in ug]
        print(f"{lo:g}-{hi:g}".rjust(14) + f" {int(s.sum()):9d}   "
              + "  ".join(f"{x:9.4f}" for x in r))


if __name__ == "__main__":
    main()
