#!/usr/bin/env python3
"""k_ms for the muon samples, as a FUNCTION of the hit scale k_hit.

WHY NOT JUST k_ms. `k_ms` is a single knob solved to close <e^{-u z^2}>, so it
absorbs everything in the track's momentum-error model that does not close --
including any mis-assignment of the HIT block. Slide 4 of
260811_trackres_masslik_david.pdf reports k_ms at k_hit = 0, which was the only
option at the time.

It is no longer. NOTES_HITRES s5 measured the hit block directly: applying the
per-class core^2 corrections moves the model's hit share from

    0.1132 -> 0.1217   (mu pT 2-20)
    0.2394 -> 0.2498   (mu pT 20-60)

and since k_hit is a log-scale on the hit VARIANCE (`vg = exp(khit)*vgf` in
cf_track_resolution.model_phi) rather than on the share, the equivalent scales
are obtained by holding the rest fixed:

    c = [f1/(1-f1)] / [f0/(1-f0)]  ->  k_hit = ln c
      low  pT: 0.1217/0.8783 / (0.1132/0.8868) = 1.0855  ->  +0.0821
      high pT: 0.2498/0.7502 / (0.2394/0.7606) = 1.0580  ->  +0.0563

BUT NOT AS A SINGLE NUMBER. Those come from the CORE prescription
(c_b = core^2), and NOTES_HITRES s5 is explicit that the core and the variance
prescriptions move f_hit in OPPOSITE directions (rmsT/core is 1.11 in the
pixels and 0.91 in the strips). The probe here spans both regimes -- small u
weights the tails, u ~ 1 the core -- so a core-derived k_hit is the right
correction at u = 1-2 and the WRONG one at u = 0.05. Quoting one k_ms per
sample would bury that.

So: scan k_hit, report k_ms(k_hit) per probe. k_hit = 0 reproduces the
published column and is the comparison; the measured values say how much of
any residual is really the hit block wearing an MS label.
"""
import importlib.util
import os
import sys

import numpy as np

spec = importlib.util.spec_from_file_location("cft", "cf_track_resolution.py")
m = importlib.util.module_from_spec(spec)
sys.modules["cft"] = m
spec.loader.exec_module(m)

PROBES = (0.05, 0.2, 1.0, 2.0)
# 0 = the published configuration; then the two core-prescription values from
# NOTES_HITRES s5 (each sample's own, and the other one, so the scan brackets
# both); then a deliberately large one to show the lever arm.
KHITS = (0.0, 0.0563, 0.0821, 0.15)

SAMPLES = (
    ("mugun_ul16_260830", "mu pT 20-60", 0.0563),
    ("mugun_lowpt_260830", "mu pT 2-20", 0.0821),
)
# The August caches, for the before/after column.
OLD = {"mugun_ul16_260830": "runs/cf_trackres_mugun_ul16_fix.npz",
       "mugun_lowpt_260830": "runs/cf_trackres_mugun_lowpt_fix.npz"}


class A:
    khit = 0.0
    kms = 0.0
    kioni = 0.0


def solve_kms(d, u, khit, lo=-0.60, hi=0.60, niter=24):
    """Bisection on k_ms at fixed k_hit -- same solve as rerun_all_closures."""
    Ed = np.exp(-u * d["z"] ** 2).mean()
    for _ in range(niter):
        mid = 0.5 * (lo + hi)
        a = A()
        a.khit = khit
        a.kms = mid
        if Ed - m.weier(m.model_phi(d, a), u).mean() > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def solve_kioni(d, u, lo=-2.0, hi=2.0, niter=24):
    """The same closure carried by the IONIZATION knob instead.

    Not an alternative answer -- a magnitude check. The ionization block is a
    far smaller share of Var(q/p) than MS for a muon, so if k_ioni has to move
    by an order of magnitude more than k_ms to close the same discrepancy, the
    discrepancy is not ionization, whatever the CGF changed.
    """
    Ed = np.exp(-u * d["z"] ** 2).mean()
    for _ in range(niter):
        mid = 0.5 * (lo + hi)
        a = A()
        a.kioni = mid
        if Ed - m.weier(m.model_phi(d, a), u).mean() > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def report(tag, label, khit_meas):
    f = f"runs/cf_trackres_{tag}.npz"
    if not os.path.exists(f):
        print(f"{label:14} (missing {f})")
        return
    d = np.load(f)
    n = len(d["z"])
    print(f"\n{label}   n = {n}")
    print(f"  {'k_hit':>8} | " + "  ".join(f"u={u:<5g}" for u in PROBES) + "     mean")
    for kh in KHITS:
        ks = [solve_kms(d, u, kh) for u in PROBES]
        mark = "  <- measured (core)" if abs(kh - khit_meas) < 1e-9 else ""
        print(f"  {kh:+8.4f} | " + "  ".join(f"{k:+.4f}" for k in ks)
              + f"   {np.mean(ks):+.4f}{mark}")
    ki = [solve_kioni(d, u) for u in PROBES]
    print(f"  {'k_ioni':>8} | " + "  ".join(f"{k:+.4f}" for k in ki)
          + f"   {np.mean(ki):+.4f}   (same closure, ionization knob)")

    old = OLD.get(tag)
    if old and os.path.exists(old):
        do = np.load(old)
        ks = [solve_kms(do, u, 0.0) for u in PROBES]
        print(f"  {'AUGUST':>8} | " + "  ".join(f"{k:+.4f}" for k in ks)
              + f"   {np.mean(ks):+.4f}   (old model AND old fit, k_hit=0)")


if __name__ == "__main__":
    print(__doc__.split("\n\n")[0])
    print("\nk_ms > 0 means the DATA is wider than the model; k_ms ~ 2x the "
          "fractional width excess.")
    for tag, label, khm in SAMPLES:
        report(tag, label, khm)
