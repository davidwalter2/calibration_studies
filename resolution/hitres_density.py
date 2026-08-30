#!/usr/bin/env python3
"""The strip hit-residual density, built from the S-curve rather than fitted blind.

Section 10 fitted [uniform(w) (x) Gauss(s)] + tail and got chi2/ndf ~ 1 in the
quantization regime and 5-12 in the interpolating one. The missing ingredient
is the eta / S-curve: `StripCPEfromTrackAngle` returns the cluster BARYCENTRE
plus a rigid shift, and the barycentre is not linear in the impact point, so
the residual depends on where inside the strip the track went.

Measured directly here (hitres_scurve.py, `scurve_mugun_lowpt.png`): with the
true position referred to the CLUSTER's own lattice, psi = hitStripSim -
hitFirstStrip, the profile E[u | psi] is a SAWTOOTH of period one strip -- the
barycentre pulls toward strip centres and resets at each boundary. For N = 1
it is a single ramp; for N = 2, 3, 4 it repeats N times across the cluster.

That fixes the functional form without fitting it blind. With t the position
within one period, uniform on [-1/2, 1/2],

    u = g(t) + noise,     g(t) = -A [ t + c1 sin(2 pi t) + c2 sin(4 pi t) ]

the density is the pushforward of the uniform through g, convolved with the
noise:

    p(u) = sum_{roots} 1/|g'| (x) N(0, s)   +  eps * (the same with k s)

A is the ramp amplitude: A -> pitch is a completely uninformative estimator
(pure quantization) and A -> 0 a perfectly linear one. c1, c2 are the
NON-linearity -- exactly what an eta correction would remove -- and they are
what makes the density depart from a box. Setting c1 = c2 = 0 reproduces
box (x) Gaussian identically, so this is a strict superset of the old family
and the comparison is nested.

usage:
  python hitres_density.py [--tag mugun_lowpt] [--subdir hitres3]
"""
import argparse
import datetime
import glob
import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot
from scipy.optimize import minimize

from wums import logging, output_tools

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)
CEPH = "/ceph/submit/data/user/d/david_w/ZMass/cvh"
BR = ["dxrecsim", "dxerr", "hitDetId", "hitPitch", "hitUProj", "clusterSizeX",
      "hitStripRec", "hitStripSim", "hitFirstStrip"]
NT = 2001          # quadrature points in t


def gfun(t, A, c1, c2):
    return -A * (t + c1 * np.sin(2 * np.pi * t) + c2 * np.sin(4 * np.pi * t))


def swidth(t, s, s1):
    """Position-dependent noise.

    For a two-strip cluster the estimator is the charge RATIO eta, and the
    position error is sigma_eta / |d eta/dt| -- it blows up wherever the
    response flattens, i.e. near the ends of the period where one strip
    carries almost everything. A single width cannot represent that, and it
    is why the constant-s fit stayed at chi2/ndf ~ 8 for N = 2 while it
    already described N = 1 and N = 3.
    """
    return s * (1.0 + s1 * np.cos(2 * np.pi * t))


def tdensity(t, r):
    """The impact point is uniform over the SENSOR, but not within a CLASS.

    Requiring N = 1 selects crossings near a strip centre; requiring N = 2
    selects crossings near a boundary, because that is what splits the charge.
    So t is uniform only before the class cut, and rho(t) = 1 + r cos(2 pi t)
    (r > 0 centre-peaked, r < 0 boundary-peaked) is the leading correction.
    Leaving it out is what kept N = 2 at chi2/ndf ~ 8 while every other class
    was already described.
    """
    w = 1.0 + r * np.cos(2 * np.pi * t)
    return np.maximum(w, 1e-6) / np.sum(np.maximum(w, 1e-6))


def density(x, A, c1, c2, s, eps, k, mu, s1=0.0, r=0.0):
    """p(u): uniform t pushed through g, convolved with the noise.

    Done as a quadrature over t rather than by finding roots of g -- same
    thing, and it stays correct when g is non-monotonic (which is the whole
    point: a fold in g is a caustic, i.e. a peak in p).
    """
    t = np.linspace(-0.5, 0.5, NT)
    g = gfun(t, A, c1, c2) + mu
    w = tdensity(t, r)
    sv = np.maximum(swidth(t, s, s1), 1e-4)[None, :]
    d = x[:, None] - g[None, :]
    core = np.exp(-0.5 * (d / sv) ** 2) / (sv * np.sqrt(2 * np.pi))
    st = k * sv
    tail = np.exp(-0.5 * (d / st) ** 2) / (st * np.sqrt(2 * np.pi))
    return ((1 - eps) * core + eps * tail) @ w


def fit(x, p0, fix_nonlin=False):
    lo, hi = np.percentile(x, [0.05, 99.95])
    x = x[(x > lo) & (x < hi)]
    h, e = np.histogram(x, bins=120, range=(lo, hi))
    c = 0.5 * (e[1:] + e[:-1]); bw = e[1] - e[0]

    def unpack(q):
        A = abs(q[0])
        c1 = 0.0 if fix_nonlin else 0.5 * np.tanh(q[1])
        c2 = 0.0 if fix_nonlin else 0.3 * np.tanh(q[2])
        s = abs(q[3]); eps = 1 / (1 + np.exp(-q[4])); k = 1 + abs(q[5]); mu = q[6]
        s1 = 0.0 if fix_nonlin else np.tanh(q[7]) * 0.95
        r = 0.0 if fix_nonlin else np.tanh(q[8]) * 0.95
        return A, c1, c2, s, eps, k, mu, s1, r

    def nll(q):
        f = density(c, *unpack(q)) * bw * h.sum()
        f = np.maximum(f, 1e-12)
        return np.sum(f - h * np.log(f))

    # multi-start: the pushforward has genuine local minima (a fit can collapse
    # to A -> 0, i.e. a pure Gaussian, and sit there)
    best = None
    starts = [(a, rr) for a in (0.4, 0.7, 1.0, 1.3) for rr in (-1.0, 0.0, 1.0)]
    for Astart, rstart in starts:
        q0 = list(p0); q0[0] = Astart
        if len(q0) > 8:
            q0[8] = rstart
        r = minimize(nll, q0, method="Nelder-Mead",
                     options=dict(maxiter=80000, maxfev=80000, xatol=1e-9, fatol=1e-9))
        if best is None or r.fun < best.fun:
            best = r
    p = unpack(best.x)
    f = density(c, *p) * bw * h.sum()
    g = h > 10
    npar = 5 if fix_nonlin else 9
    chi2 = np.sum((h[g] - f[g]) ** 2 / f[g]); ndf = g.sum() - npar
    return p, chi2 / ndf, (c, h, f), best.x


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="mugun_lowpt")
    p.add_argument("--subdir", default="hitres3")
    p.add_argument("--nfiles", type=int, default=0)
    p.add_argument("--outdir", default="")
    args = p.parse_args()
    today = datetime.date.today().strftime("%y%m%d")
    outdir = output_tools.make_plot_dir(args.outdir or os.path.expanduser(
        f"~/public_html/calibration_studies/{today}_hitres/"))

    fs = [f for f in sorted(glob.glob(
        f"{CEPH}/{args.subdir}_{args.tag}/task_*/globalcor_resclosure_0.root"))
        if os.path.exists(os.path.join(os.path.dirname(f), ".complete"))]
    if args.nfiles:
        fs = fs[:args.nfiles]
    cols = {b: [] for b in BR}
    for fn in fs:
        a = uproot.open(fn)["tree"].arrays(BR, library="np")
        for b in BR:
            cols[b].append(np.concatenate(a[b]))
    d = {b: np.concatenate(v) for b, v in cols.items()}
    sd = (d["hitDetId"].astype(np.uint32) >> 25) & 0x7
    base = ((sd == 3) | (sd == 5)) & (d["hitStripSim"] > -98) & (d["dxerr"] > 0)
    u = d["dxrecsim"] / d["hitPitch"]         # residual in units of the pitch
    logger.info(f"{len(fs)} files, {int(base.sum())} strip hits")

    BINS = [(N, lo, hi) for lo, hi in ((0.0, 0.15), (0.3, 0.6)) for N in (1, 2, 3, 4)]
    fig, axes = plt.subplots(2, 4, figsize=(21, 9.5), constrained_layout=True)
    print(f"\n  {'bin':<22} {'n':>7} | {'BOX (x) G (c=0)':>16} | "
          f"{'S-curve model':>50}")
    print(f"  {'':<22} {'':>7} | {'chi2/ndf':>16} | {'A':>7} {'c1':>7} {'c2':>7} "
          f"{'s':>7} {'s1':>6} {'r':>6} {'tail':>7} {'chi2/ndf':>8}")
    for i, (N, lo, hi) in enumerate(BINS):
        ax = axes[i // 4, i % 4]
        m = base & (d["clusterSizeX"] == N) & (d["hitUProj"] >= lo) & (d["hitUProj"] < hi)
        if m.sum() < 4000:
            ax.set_visible(False); continue
        x = u[m]
        p0 = [0.9, 0.05, 0.02, 0.08, -2.0, 1.0, 0.0, 0.0, 0.0]
        _, c2lin, _, _ = fit(x, p0, fix_nonlin=True)
        par, c2full, (cc, hh, ff), _ = fit(x, p0)
        A, c1, c2, s, eps, k, mu, s1, r = par
        print(f"  N={N} uProj {lo:.2f}-{hi:.2f}  {x.size:7d} | {c2lin:16.2f} | "
              f"{A:7.4f} {c1:7.4f} {c2:7.4f} {s:7.4f} {s1:6.2f} {r:6.2f} {100*eps:6.1f}% {c2full:8.2f}")
        ax.errorbar(cc, hh, np.sqrt(np.maximum(hh, 1)), fmt="ko", ms=2.5, lw=0.8)
        ax.plot(cc, ff, "r-", lw=1.8, label=f"S-curve  $\\chi^2$/ndf {c2full:.2f}")
        _, _, (cl, hl, fl), _ = fit(x, p0, fix_nonlin=True)
        ax.plot(cl, fl, "b--", lw=1.4, label=f"box$\\otimes$G  $\\chi^2$/ndf {c2lin:.2f}")
        ax.set_yscale("log")
        ax.set_ylim(max(0.5, hh.max() * 3e-5), hh.max() * 2)
        ax.set_xlabel("(rec - sim) / pitch")
        ax.set_ylabel("hits")
        ax.set_title(f"N={N},  uProj {lo:.2f}-{hi:.2f}", fontsize=14)
        ax.legend(fontsize=10)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"density_{args.tag}.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {outdir}/density_{args.tag}.png")


if __name__ == "__main__":
    main()
