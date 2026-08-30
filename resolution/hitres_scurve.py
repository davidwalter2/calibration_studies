#!/usr/bin/env python3
"""The eta / S-curve of the strip CPE, and the density it generates.

`StripCPEfromTrackAngle` estimates the position as the cluster BARYCENTRE plus
a rigid Lorentz/backplane shift (`strip = cluster.barycenter() + corr`). The
barycentre is not a linear function of the true impact point -- diffusion,
capacitive coupling and the readout threshold make the charge sharing an
error-function of the distance rather than a straight line -- so the residual
depends on where inside the cluster the track went. That dependence is the
S-curve, and it is the ingredient the box (x) Gaussian family was missing:
section 10 fitted it at chi2/ndf 1.0 in the quantization regime and 5-12 in
the interpolating one.

The variable has to be the true position in the CLUSTER's own lattice frame,

    psi = hitStripSim - hitFirstStrip        in [0, N]

not the absolute strip lattice: near a strip boundary either neighbour can be
the one that fires, and referring the phase to the absolute lattice averages
the two assignments together. Measured on the muon gun that compresses the
N=1 ramp from its geometric +-0.5 to +-0.29 and its slope from -1 to -0.49.

The density then follows without any further assumption. With psi uniform
(impact points are uniform) and

    u = g(psi) + noise,

p(u) is the pushforward of the uniform through g, convolved with the noise:

    p(u) = sum_{roots psi_i of g(psi)=u} 1/|g'(psi_i)|   (x)   N(0, s)  + tail

so the flat tops and the caustic peaks come out of g' and are not fitted
separately. g linear reproduces the box (x) Gaussian exactly, which is why the
old family worked wherever the estimator was quantization-dominated.

usage:
  python hitres_scurve.py [--tag mugun_lowpt] [--subdir hitres3]
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


def load(tag, subdir, nfiles):
    fs = [f for f in sorted(glob.glob(
        f"{CEPH}/{subdir}_{tag}/task_*/globalcor_resclosure_0.root"))
        if os.path.exists(os.path.join(os.path.dirname(f), ".complete"))]
    if nfiles:
        fs = fs[:nfiles]
    if not fs:
        raise SystemExit(f"no complete files in {subdir}_{tag}")
    cols = {b: [] for b in BR}
    for fn in fs:
        a = uproot.open(fn)["tree"].arrays(BR, library="np")
        for b in BR:
            cols[b].append(np.concatenate(a[b]))
    d = {b: np.concatenate(v) for b, v in cols.items()}
    d["subdet"] = (d["hitDetId"].astype(np.uint32) >> 25) & 0x7
    return d, len(fs)


def scurve(psi, N, nb=24):
    """Bin edges spanning the cluster, in its own lattice frame."""
    return np.linspace(0.0, float(N), nb + 1)


def gmodel(psi, N, p):
    """Odd-in-(psi - N/2) Fourier form for the estimator's bias.

    a0 is the LINEAR term: a0 = -1 is a completely uninformative estimator
    (the barycentre does not move with the impact point at all -> pure
    quantization), a0 = 0 is a perfectly linear one. The sine terms are the
    non-linearity that the eta correction would remove.
    """
    x = (psi - 0.5 * N) / N                       # in [-0.5, 0.5]
    g = p[0] * x * N
    for k, a in enumerate(p[1:], start=1):
        g = g + a * np.sin(2 * np.pi * k * x)
    return g


def fit_scurve(psi, u, N, nharm=3):
    c = scurve(psi, N)
    ctr = 0.5 * (c[1:] + c[:-1])
    prof, err = [], []
    for lo, hi in zip(c[:-1], c[1:]):
        m = (psi >= lo) & (psi < hi)
        if m.sum() < 50:
            prof.append(np.nan); err.append(np.nan); continue
        prof.append(np.median(u[m]))
        err.append(1.253 * u[m].std() / np.sqrt(m.sum()))
    prof = np.array(prof); err = np.array(err)
    ok = np.isfinite(prof)

    def chi2(p):
        return np.sum(((gmodel(ctr[ok], N, p) - prof[ok]) / err[ok]) ** 2)

    p0 = np.zeros(1 + nharm); p0[0] = -1.0
    r = minimize(chi2, p0, method="Nelder-Mead",
                 options=dict(maxiter=40000, maxfev=40000, fatol=1e-10, xatol=1e-10))
    return ctr, prof, err, ok, r.x, chi2(r.x) / max(ok.sum() - len(p0), 1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="mugun_lowpt")
    p.add_argument("--subdir", default="hitres3")
    p.add_argument("--nfiles", type=int, default=0)
    p.add_argument("--nharm", type=int, default=3)
    p.add_argument("--outdir", default="")
    args = p.parse_args()
    today = datetime.date.today().strftime("%y%m%d")
    outdir = output_tools.make_plot_dir(args.outdir or os.path.expanduser(
        f"~/public_html/calibration_studies/{today}_hitres/"))

    d, nf = load(args.tag, args.subdir, args.nfiles)
    strip = ((d["subdet"] == 3) | (d["subdet"] == 5)) & (d["hitStripSim"] > -98)
    u = d["hitStripRec"] - d["hitStripSim"]
    psi = d["hitStripSim"] - d["hitFirstStrip"]
    # closure of the new export against the one that already existed
    ok = strip & (d["dxerr"] > 0)
    ratio = (u[ok] * d["hitPitch"][ok]) / d["dxrecsim"][ok]
    logger.info(f"{nf} files, {int(strip.sum())} strip hits with truth")
    print(f"  export closure: (hitStripRec-hitStripSim)*pitch / dxrecsim = "
          f"{np.median(ratio):.6f}  q16/q84 {np.percentile(ratio,16):.6f}"
          f"/{np.percentile(ratio,84):.6f}")

    UB = ((0.0, 0.15), (0.15, 0.3), (0.3, 0.6))
    fig, axes = plt.subplots(len(UB), 4, figsize=(21, 4.6 * len(UB)),
                             constrained_layout=True)
    print(f"\n  {'bin':<24} {'n':>8} {'a0 (linear)':>12} "
          + " ".join(f"{'a'+str(k):>8}" for k in range(1, args.nharm + 1))
          + f" {'chi2/ndf':>9}   {'peak |g|':>8}")
    out = {}
    for iu, (lo, hi) in enumerate(UB):
        for iN, N in enumerate((1, 2, 3, 4)):
            ax = axes[iu, iN]
            m = strip & (d["clusterSizeX"] == N) & (d["hitUProj"] >= lo) & (d["hitUProj"] < hi)
            if m.sum() < 2000:
                ax.set_visible(False)
                continue
            # measurement only. The Fourier form tried here does not describe
            # g well, and it is not needed: hitres_density.py parameterises the
            # SAME physics directly at the level of the density, where it is
            # what the CF consumes and where it fits (chi2/ndf 1.0-3.1).
            keep = (psi[m] >= 0) & (psi[m] <= N)
            ctr, prof, err, good, pars, c2 = fit_scurve(psi[m][keep], u[m][keep], N, args.nharm)
            out[(N, lo, hi)] = (pars, c2, int(m.sum()))
            ax.errorbar(ctr[good], prof[good], err[good], fmt="ko", ms=4)
            xx = np.linspace(0, N, 300)
            ax.plot(xx, -(xx - 0.5 * N), "b--", lw=1.2)
            ax.axhline(0, color="0.7", lw=0.8)
            ax.set_xlabel(r"$\psi$  [strips]")
            ax.set_ylabel(r"$g(\psi)$ [strips]")
            ax.set_title(f"N={N},  uProj {lo:.2f}-{hi:.2f}", fontsize=14)
            print(f"  N={N} uProj {lo:.2f}-{hi:.2f}      {m.sum():8d} "
                  f"{pars[0]:12.4f} "
                  + " ".join(f"{a:8.4f}" for a in pars[1:])
                  + f" {c2:9.2f}   {np.nanmax(np.abs(prof)):8.4f}")
            # the profile itself is the deliverable of this script
    axes[0, 0].plot([], [], "ko", label="measured")
    axes[0, 0].plot([], [], "b--", label=r"pure quantization, $g=-(\psi-N/2)$")
    axes[0, 0].legend(fontsize=10, loc="upper right")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"scurve_{args.tag}.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {outdir}/scurve_{args.tag}.png")
    np.savez(os.path.join(outdir, f"scurve_{args.tag}.npz"),
             **{f"N{N}_u{lo}_{hi}": v[0] for (N, lo, hi), v in out.items()})


if __name__ == "__main__":
    main()
