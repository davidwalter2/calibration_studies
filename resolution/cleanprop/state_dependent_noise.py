"""Does the material a track meets DEPEND on how far it has already scattered,
and does that change the predicted width?

THE QUESTION (David, 2026-08-08). The CF product assumes the per-step noise is
a fixed function of step index: chi_c^2, chi_a^2 and the step length are all
taken from the REFERENCE path. But a track that has already been deflected is
somewhere else transversely, and the tracker material is NOT transversely
uniform -- measured across 32 sampled rays spanning the region the tracks
occupy, the accumulated material varies by 5-10% RMS and by 25-32% between the
extreme rays. So the PDF of the later kicks genuinely depends on the earlier
ones. The CF cannot represent that: independent steps with fixed parameters is
the assumption that lets the CFs multiply.

This is NOT the same as the transport-nonlinearity check (same kicks, wrong
lever arms; measured at 0.1-0.3% and therefore bounded), and NOT the same as
path-averaging (<CF>_rays, which assumes each track follows ONE ray throughout
and so cannot capture the correlation between a track's own deviation and its
subsequent material; measured at <=0.003).

METHOD. A sequential Monte Carlo over the exported per-step records:
  mode REF   -- material always taken from the reference ray (what the CF does)
  mode STATE -- at each step, material taken from whichever sampled ray lies
                closest to the track's CURRENT transverse deviation
Everything else is identical between the two modes: same kicks drawn from the
same generator seed, same linear transport, same target surfaces. The
difference between them is the effect in question, isolated.

Kicks are drawn GAUSSIAN with variance thp2. That is deliberate: the Moliere
tails dominate the ABSOLUTE closure but cancel in the REF-vs-STATE difference,
which is what this measures. Do not read the absolute widths here as a closure.

usage:
    python state_dependent_noise.py --rays '<dir>/ray_*.root' \
        --rayfile <dir>/rays.txt [--ntracks 200000] [--plane 18]
"""

import argparse
import glob
import os
import sys

import numpy as np
import uproot

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wums import logging  # noqa: E402

logger = logging.child_logger(__name__)


def load_rays(pattern, rayfile):
    """Per-ray: dphi label, per-step thp2, and the leg-boundary radii."""
    dphi = {}
    if rayfile and os.path.exists(rayfile):
        for line in open(rayfile):
            p = line.split()
            if len(p) >= 3:
                dphi[int(p[0])] = float(p[1]) - 0.70   # baseline phi
    rays = []
    for f in sorted(glob.glob(pattern)):
        i = int(os.path.basename(f).split("_")[1].split(".")[0])
        t = uproot.open(f)["propExport/legs"].arrays(
            ["msmoliv", "refglobr", "stepnms"], library="np")
        thp2, legof = [], [0]
        for k in range(len(t["msmoliv"])):
            st = np.asarray(t["msmoliv"][k], dtype=float).reshape(-1, 10)
            thp2.append(st[:, 5])
            legof.append(legof[-1] + len(st))
        rays.append(dict(idx=i, dphi=dphi.get(i, 0.0),
                         thp2=np.concatenate(thp2),
                         legoff=np.array(legof),
                         r=np.asarray(t["refglobr"], dtype=float)))
    return rays


def step_radius(ray, nstep):
    """Radius at each step, interpolated linearly within each leg.

    legoff comes from the reference ray and can run past the COMMON step count
    (rays differ by a step or two), so both ends are clamped to nstep.
    """
    r = np.zeros(nstep)
    for j in range(len(ray["legoff"]) - 1):
        a = int(min(ray["legoff"][j], nstep))
        b = int(min(ray["legoff"][j + 1], nstep))
        if b <= a or j >= len(ray["r"]):
            continue
        r0 = ray["r"][j - 1] if j > 0 else 0.0
        r[a:b] = np.linspace(r0, ray["r"][j], b - a, endpoint=True)
    if r[-1] == 0.0:                      # tail beyond the last full leg
        r[r == 0.0] = ray["r"][min(len(ray["r"]) - 1, len(ray["r"]) - 1)]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rays", required=True)
    ap.add_argument("--rayfile", default="")
    ap.add_argument("--ntracks", type=int, default=200000)
    ap.add_argument("--plane", type=int, default=18)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    logging.setup_logger(__file__, 3, False)

    rays = load_rays(a.rays, a.rayfile)
    if len(rays) < 3:
        raise SystemExit("need several rays to interpolate the material")
    ref = min(rays, key=lambda r: abs(r["dphi"]))
    n = min(len(r["thp2"]) for r in rays)
    logger.info(f"{len(rays)} rays, {n} common steps, reference dphi={ref['dphi']:+.2e}")

    # material table: thp2[ray, step], and each ray's transverse offset per step
    TH = np.array([r["thp2"][:n] for r in rays])
    rad = step_radius(ref, n)
    OFF = np.array([r["dphi"] * rad for r in rays])       # cm, per ray per step
    order = np.argsort([r["dphi"] for r in rays])
    TH, OFF = TH[order], OFF[order]
    iref = int(np.argmin(np.abs([rays[o]["dphi"] for o in order])))

    # lever arm from each step to the target plane (drift approximation is
    # adequate here: both modes use the SAME lever arms, so it cancels)
    lastcut = ref["legoff"][min(a.plane + 1, len(ref["legoff"]) - 1)]
    nst = min(n, lastcut)
    D = np.maximum(rad[nst - 1] - rad[:nst], 0.0)

    rng = np.random.default_rng(a.seed)
    N = a.ntracks
    outs = {}
    for mode in ("REF", "STATE"):
        rng = np.random.default_rng(a.seed)          # identical kicks per mode
        x = np.zeros(N); th = np.zeros(N)
        for s in range(nst):
            if mode == "REF":
                v = np.full(N, TH[iref, s])
            else:
                # nearest sampled ray to the CURRENT deviation
                j = np.argmin(np.abs(OFF[:, s][None, :] - x[:, None]), axis=1)
                v = TH[j, s]
            g = rng.standard_normal(N) * np.sqrt(np.maximum(v, 0.0))
            # accumulate the ANGLE and let it drift the position; adding
            # g*D[s] here as well would count each kick twice (it would get
            # its full lever arm immediately AND again via the drift).
            th += g
            x += th * (D[s] - (D[s + 1] if s + 1 < nst else 0.0))
        outs[mode] = x
    r_, s_ = outs["REF"], outs["STATE"]
    print(f"\nplane {a.plane}, {N} MC tracks, {nst} steps")
    print(f"  REF   (material from the reference path): rms {r_.std():.6e}")
    print(f"  STATE (material at the actual deviation): rms {s_.std():.6e}")
    print(f"  ratio STATE/REF = {s_.std()/r_.std():.4f}   "
          f"variance ratio {(s_.std()/r_.std())**2:.4f}")
    sig = r_.std()
    for u in (0.1, 1.0):
        fr = np.mean(np.exp(-u * (r_ / sig) ** 2))
        fs = np.mean(np.exp(-u * (s_ / sig) ** 2))
        print(f"  u={u:<4g}  E[e^-uz^2]: REF {fr:.5f}  STATE {fs:.5f}  "
              f"difference {fs-fr:+.5f}")


if __name__ == "__main__":
    main()
