#!/usr/bin/env python3
"""Is the Z legs' eta dependence a MIXTURE of two eta-INDEPENDENT components?

The hit-class agent found exactly that at TRACK level: splitting on
|seed->final dq/p| at its 90th percentile gives +20.59 +- 6.45 e-3 (top 10 %,
chi2 0.0/2 against eta-flat) and -6.33 +- 1.84 e-3 (the rest, chi2 0.4/2),
with a mixing fraction that runs 2.4 % -> 7.0 % -> 21.2 % with |eta| and
reproduces the per-band numbers exactly. Their IN population has 2.1x
sigma_rel, chi2/ndof 1.098 against 0.988, and FEWER pixel hits at the same
total -- the same kind of track the mass-level endcap miss names.

The seed->final step is not in the mass caches, but its signatures are.
`chi2ndof` is used here because it is BOTH the agent's discriminator AND the
safest conditioning variable available on this cache: corr(chi2ndof, z) =
+0.0004, against sigma/m -0.0158, pixel share +0.0141, |eta| lead -0.0081,
maxfraclossp -0.0335, and the agent's own seed->final step +0.113.

If both components come out flat in `eta`, the -14.56 endcap miss of
sec. 0f.43 is a MIXING FRACTION and not a property of the endcap.

    python3 mixture_test.py [--pct 90]
"""
import argparse
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_odd_mass import model_odd, PROBES  # noqa: E402

F = "/work/submit/david_w/ZMass/calibration_studies/fullscale"
BANDS = [(0.0, 0.9, "|eta| 0.0-0.9"), (0.9, 1.6, "0.9-1.6"), (1.6, 3.0, "1.6-3.0")]


def odd(z, u, w):
    return np.average(z * np.exp(-u * z * z), weights=w)


def boot(z, u, w, nb, rng):
    n = len(z)
    return np.std([odd(z[k], u, w[k]) for k in (rng.integers(0, n, n) for _ in range(nb))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=f"{F}/runs/gzpairs_dyv2_n50.npz")
    ap.add_argument("--pct", type=float, nargs="*", default=[80, 90, 95])
    ap.add_argument("--nboot", type=int, default=150)
    a = ap.parse_args()
    rng = np.random.default_rng(20260909)

    d = np.load(a.cache, allow_pickle=True)
    z = d["z"].astype(np.float64); sig = d["sigma"].astype(np.float64)
    mgen = d["mgen"].astype(np.float64); m = z * sig + mgen
    etal = np.maximum(np.abs(d["etap"].astype(np.float64)),
                      np.abs(d["etam"].astype(np.float64)))
    w = d["w"].astype(np.float64) if "w" in d.files else np.ones(len(z))
    c2 = d["chi2ndof"].astype(np.float64)
    a_i = 1.211 * sig / m
    den = 1.0 - a_i * z
    x = np.where(np.abs(den) > 1e-3, z / den, np.nan)
    ok = np.isfinite(x) & (np.abs(x) < 8) & (sig / m < 0.10) & np.isfinite(c2)
    MOD = model_odd(d, None, jensen=True)
    print(f"{int(ok.sum())} candidates. data - model, 1e-3, u = {PROBES[0]}.\n")

    for pct in a.pct:
        thr = np.percentile(c2[ok], pct)
        IN, OUT = c2 >= thr, c2 < thr
        print(f"=== split at chi2/ndof p{pct:.0f} = {thr:.4f}")
        print(f"  {'band':14s} {'f_IN':>7s} {'IN (data-model)':>20s} "
              f"{'OUT (data-model)':>20s}")
        rows = {"IN": [], "OUT": []}
        for lo, hi, lab in BANDS:
            b = ok & (etal >= lo) & (etal < hi)
            f = np.average(IN[b], weights=w[b])
            cell = []
            for nm, sel in (("IN", IN), ("OUT", OUT)):
                mk = b & sel
                dv = odd(x[mk], PROBES[0], w[mk])
                e = boot(x[mk], PROBES[0], w[mk], a.nboot, rng)
                mv = np.average(MOD[mk, 0], weights=w[mk])
                rows[nm].append((1e3 * (dv - mv), 1e3 * e))
                cell.append(f"{1e3*(dv-mv):+9.2f} +-{1e3*e:5.2f}")
            print(f"  {lab:14s} {100*f:6.1f}% {cell[0]:>20s} {cell[1]:>20s}")
        for nm in ("IN", "OUT"):
            v = np.array([r[0] for r in rows[nm]]); e = np.array([r[1] for r in rows[nm]])
            wm = np.sum(v / e**2) / np.sum(1 / e**2)
            chi2 = np.sum((v - wm) ** 2 / e**2)
            print(f"  {nm:3s}: weighted mean {wm:+7.2f}, chi2 vs eta-flat "
                  f"{chi2:.2f} / 2 dof")
        print()
    print("If both components are eta-FLAT, the endcap miss is a MIXING "
          "FRACTION, not a property of the endcap.")


if __name__ == "__main__":
    main()
