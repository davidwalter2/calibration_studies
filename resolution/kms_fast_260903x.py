#!/usr/bin/env python3
"""k_MS at k_hit = 0 (the published configuration) for one cache, both k_rad
arms, with the cache held in RAM.

WHY THIS EXISTS. `kms_solve_260903x.py` passes the `np.load` handle straight
into `model_phi`, and a COMPRESSED npz re-inflates the whole array on every
`d["Sms"]` -- ~500 model_phi calls x 6 arrays x 0.9 GB. That is the 4-5 h
runtime, not the arithmetic. Materialising once turns each solve into seconds.
Same bisection, same tolerance, same model: this reproduces the `k_hit = 0`
and `krad = 0` rows of the ladder exactly.

usage: python kms_fast_260903x.py runs/cf_trackres_<tag>.npz
"""
import sys
import numpy as np
import cf_track_resolution as m

PROBES = (0.05, 0.2, 1.0, 2.0)


class A:
    khit = 0.0
    kms = 0.0
    kioni = 0.0
    krad = 1.0
    hitmode = "gauss"


def solve(d, u, khit, krad, lo=-0.60, hi=0.60, niter=24):
    Ed = np.exp(-u * d["z"] ** 2).mean()
    for _ in range(niter):
        mid = 0.5 * (lo + hi)
        a = A(); a.khit = khit; a.kms = mid; a.krad = krad
        if Ed - m.weier(m.model_phi(d, a), u).mean() > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


f = sys.argv[1]
z = np.load(f)
d = {k: z[k] for k in ("z", "vgf", "Sms", "Sio_re", "Sio_im", "Srad_re", "Srad_im")}
print(f"{f}  n = {len(d['z'])}")
for kr in (1.0, 0.0):
    ks = [solve(d, u, 0.0, kr) for u in PROBES]
    print(f"  krad={kr:.0f} k_hit=0 | " + "  ".join(f"{k:+.4f}" for k in ks)
          + f"   mean {np.mean(ks):+.4f}", flush=True)
