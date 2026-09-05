#!/usr/bin/env python3
"""[2026-09-03 `_260903x` variant: the radiative term is in the model, and
`var` is the normalisation on BOTH arms]

k_ms for the muon samples, as a FUNCTION of the hit scale k_hit -- the A/B/C
ladder of kms_solve_260903.py, re-run on the caches extracted from the
productions that carry the `radstepv` / `ioniqscalev` exports.

TWO THINGS CHANGED SINCE kms_solve_260903.py, and they are independent:

1. `--ioni-norm raw` IS GONE FROM ARM C.  `var` recovers the block weight as
   sqrt(v_b/sq2) with sq2 the variance the FIT used; under CgfQoPMode >= 1 the
   fit substitutes the block's Fisher weight, so `var` needed the substitution
   factor, which the 2026-09-03 `ioniqscalev` export now supplies.  Arm C was
   therefore run with `raw` (a one-sided ~10 % deficit in the exponent) and is
   now run with the same exact normalisation as arm B.  B-vs-C is now a clean
   comparison of the two ESTIMATORS and nothing else.

2. THE MODEL HAS A RADIATIVE BLOCK.  `--krad 1` (default) includes it,
   `--krad 0` is the model these ladders were measured with before.  Both are
   reported: the k_hit scan at k_rad = 1, and the published k_hit = 0 row at
   k_rad = 0 for direct comparison with the 2026-09-03 09:50 table.

k_ms > 0 means the DATA is wider than the model; k_ms is ~2x the fractional
width excess.  Each sample costs ~30 min, so this script does ONE sample per
invocation (`--sample`) and `run_kms_260903x.sh` runs the four in parallel.
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "cft", os.path.join(_HERE, "cf_track_resolution.py"))
m = importlib.util.module_from_spec(spec)
sys.modules["cft"] = m
spec.loader.exec_module(m)

PROBES = (0.05, 0.2, 1.0, 2.0)
# 0 = the published configuration; then the two core-prescription values from
# NOTES_HITRES s5 (each sample's own, and the other one, so the scan brackets
# both); then a deliberately large one to show the lever arm.
KHITS = (0.0, 0.0563, 0.0821, 0.15)

SAMPLES = {
    "ul16_C": ("mugun_ul16_260903x_k0",
               "mu pT 20-60  [C: CGF Fisher weight, var norm]", 0.0563),
    "ul16_B": ("mugun_ul16_260903x_m0_k0",
               "mu pT 20-60  [B: legacy truncated Q]", 0.0563),
    "lowpt_C": ("mugun_lowpt_260903x_k0",
                "mu pT 2-20   [C: CGF Fisher weight, var norm]", 0.0821),
    "lowpt_B": ("mugun_lowpt_260903x_m0_k0",
                "mu pT 2-20   [B: legacy truncated Q]", 0.0821),
}
# A: the August caches (old model AND old fit), for the published column.
OLD = {"ul16_C": "runs/cf_trackres_mugun_ul16_fix.npz",
       "lowpt_C": "runs/cf_trackres_mugun_lowpt_fix.npz"}


class A:
    khit = 0.0
    kms = 0.0
    kioni = 0.0
    krad = 1.0
    hitmode = "gauss"


def solve_kms(d, u, khit, krad=1.0, lo=-0.60, hi=0.60, niter=24):
    """Bisection on k_ms at fixed k_hit -- same solve as rerun_all_closures."""
    Ed = np.exp(-u * d["z"] ** 2).mean()
    for _ in range(niter):
        mid = 0.5 * (lo + hi)
        a = A()
        a.khit = khit
        a.kms = mid
        a.krad = krad
        if Ed - m.weier(m.model_phi(d, a), u).mean() > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def solve_kioni(d, u, krad=1.0, lo=-2.0, hi=2.0, niter=24):
    """The same closure carried by the IONIZATION knob instead -- a magnitude
    check, not an alternative answer."""
    Ed = np.exp(-u * d["z"] ** 2).mean()
    for _ in range(niter):
        mid = 0.5 * (lo + hi)
        a = A()
        a.kioni = mid
        a.krad = krad
        if Ed - m.weier(m.model_phi(d, a), u).mean() > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def report(key, krad):
    tag, label, khit_meas = SAMPLES[key]
    f = f"runs/cf_trackres_{tag}.npz"
    if not os.path.exists(f):
        print(f"{label:38} (missing {f})")
        return
    d = np.load(f)
    n = len(d["z"])
    rm = int(d["rad_model"]) if "rad_model" in d.files else 0
    print(f"\n{label}   n = {n}   cache rad_model = {rm}")
    print(f"  {'k_hit':>8} | " + "  ".join(f"u={u:<5g}" for u in PROBES) + "     mean")
    for kh in KHITS:
        ks = [solve_kms(d, u, kh, krad) for u in PROBES]
        mark = "  <- measured (core)" if abs(kh - khit_meas) < 1e-9 else ""
        print(f"  {kh:+8.4f} | " + "  ".join(f"{k:+.4f}" for k in ks)
              + f"   {np.mean(ks):+.4f}{mark}")
    ki = [solve_kioni(d, u, krad) for u in PROBES]
    print(f"  {'k_ioni':>8} | " + "  ".join(f"{k:+.4f}" for k in ki)
          + f"   {np.mean(ki):+.4f}   (same closure, ionization knob)")
    # the SAME row with the radiative term removed: the direct comparison with
    # the 2026-09-03 09:50 ladder, which had no such term
    if rm and krad:
        ks0 = [solve_kms(d, u, 0.0, 0.0) for u in PROBES]
        print(f"  {'krad=0':>8} | " + "  ".join(f"{k:+.4f}" for k in ks0)
              + f"   {np.mean(ks0):+.4f}   (k_hit = 0, radiative term OFF)")
    old = OLD.get(key)
    if old and os.path.exists(old):
        do = np.load(old)
        ks = [solve_kms(do, u, 0.0, 0.0) for u in PROBES]
        print(f"  {'AUGUST':>8} | " + "  ".join(f"{k:+.4f}" for k in ks)
              + f"   {np.mean(ks):+.4f}   (old model AND old fit, k_hit=0)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", choices=sorted(SAMPLES), required=True)
    ap.add_argument("--krad", type=float, default=1.0)
    a = ap.parse_args()
    print(__doc__.split("\n\n")[0])
    print(f"\n[sample {a.sample}, k_rad = {a.krad}]")
    report(a.sample, a.krad)
