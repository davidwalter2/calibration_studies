#!/usr/bin/env python3
"""Is the in-maker's 64-point tau grid fine enough for the Z channel?

The density is the inverse Fourier transform

    L(m) = (1/pi) Int_0^tmax Re[ Phi(t/sigma) e^{-i t delta/sigma} ] dt / (pi sigma)

whose integrand oscillates ``tmax * |delta| / (2 pi sigma)`` times across the
grid -- once per unit of pull, near enough. The in-maker exports 64 points on
``[0, 7.8926]`` (``stride4of448``), so a candidate at ``|delta|/sigma = p``
gets ``64 / (7.8926 p / 2 pi) = 51/p`` points per oscillation. This script
measures what that costs, by comparing the density on the 64-point grid with
the same model on an upsampled grid (the family exponents are smooth in ``t``:
a cubic spline through every other in-maker point reproduces them to ~1e-4).

    python check_tgrid.py --pairs data/zpairs_smoke.npz \\
        --kernel data/zfsr_kernel_reco.npz
"""

import argparse

import numpy as np
import tensorflow as tf
from scipy.interpolate import CubicSpline

import make_z_card


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", required=True)
    p.add_argument("--kernel", required=True)
    p.add_argument("--upsample", type=int, nargs="+", default=[4, 16, 64])
    p.add_argument("--pulls", type=float, nargs="+",
                   default=[0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0])
    return p.parse_args()


def upsampled_term(term, factor):
    """The same term on a `factor`x finer tau grid (splined exponents)."""
    from rabbit import unbinned

    t0 = np.asarray(term.tgrid)
    t1 = np.linspace(t0[0], t0[-1], (len(t0) - 1) * factor + 1)
    fams = []
    for f in term.families:
        e = {"name": f["name"], "param": f["param"], "kind": f["kind"]}
        for c in ("re", "im"):
            if c in f:
                a = np.asarray(f[c], dtype=np.float64)
                e[c] = CubicSpline(t0, a, axis=1)(t1)
        fams.append(e)
    return unbinned.MassCFTerm(
        term.name + f"_up{factor}", sigma=np.asarray(term.sigma),
        mobs=np.asarray(term.mobs), tgrid=t1, families=fams,
        vgf=np.asarray(term.vgf), phik=term.phik_tab, kernel=term.kernel,
        background=term.background, m_ref=term.m_ref, chunk=term.chunk,
        dtype=term.dtype)


def main():
    args = parse_args()
    argv = ["--pairs", args.pairs, "--kernel", args.kernel, "--no-window-norm"]
    term, _, _, info = make_z_card.build(make_z_card.parse_args(argv),
                                         log=lambda *a: None)
    vals = tf.constant(term.param_defaults, tf.float64)
    t0 = np.asarray(term.tgrid)
    print(f"[check_tgrid] {term.n} candidates, in-maker grid {term.nt} points "
          f"on [0, {t0[-1]:.4f}]")

    ref = upsampled_term(term, max(args.upsample))
    d_ref = ref.raw_density(vals).numpy()
    sigma = np.asarray(term.sigma)
    mobs = np.asarray(term.mobs)
    pull = mobs / sigma
    print(f"  candidate pulls: |p| median {np.median(np.abs(pull)):.2f}, "
          f"95 % {np.percentile(np.abs(pull), 95):.2f}, max {np.abs(pull).max():.2f}")

    print(f"\n  density of the actual candidates, grid vs {max(args.upsample)}x "
          f"upsampled:")
    print(f"    {'|pull| bin':>14s} {'n':>5s} {'osc':>7s} {'pts/osc':>8s} "
          f"{'median rel dev':>15s} {'max rel dev':>12s}")
    edges = [0, 1, 2, 3, 5, 8, 12, 30]
    for a, b in zip(edges[:-1], edges[1:]):
        m = (np.abs(pull) >= a) & (np.abs(pull) < b)
        if m.sum() == 0:
            continue
        p = np.abs(pull[m]).mean()
        osc = t0[-1] * p / (2 * np.pi)
        dev = np.abs(term.raw_density(vals).numpy()[m] - d_ref[m]) / np.abs(d_ref[m])
        print(f"    {a:5.0f}-{b:<8.0f} {m.sum():5d} {osc:7.1f} "
              f"{term.nt/max(osc,1e-9):8.1f} {np.median(dev):15.2e} "
              f"{dev.max():12.2e}")

    print(f"\n  convergence of the whole NLL:")
    base = term.nll(vals).numpy()
    prev = base
    for f in args.upsample:
        up = upsampled_term(term, f)
        v = up.nll(vals).numpy()
        print(f"    {f:4d}x ({(term.nt-1)*f+1:6d} points): NLL = {v:.6f}   "
              f"d(64-pt) = {base - v:+.6f}   d(prev) = {prev - v:+.6f}")
        prev = v


if __name__ == "__main__":
    main()
