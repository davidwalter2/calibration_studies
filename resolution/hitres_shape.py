#!/usr/bin/env python3
"""Is the hit residual non-Gaussian INTRINSICALLY, or only by mixing classes?

The two hypotheses make different predictions, and they are separable by
conditioning on every class the CPE itself knows about:

  MIXTURE   each class is Gaussian with its own width; the aggregate is
            heavy-tailed only because the widths differ. Conditioning must
            then produce a Gaussian.
  INTRINSIC quantization gives a UNIFORM (bounded, flat-topped, sub-Gaussian)
            and delta-rays give a POWER-LAW tail (super-Gaussian). Neither
            survives as a Gaussian under any conditioning, because neither is
            one.

Each conditioned distribution is normalised to its OWN 68 % half-width, so the
panels compare SHAPE with width divided out -- which is the whole question.

usage: python hitres_shape.py [--tag mugun_lowpt] [--nfiles 35]
"""
import argparse
import datetime
import glob
import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot

from wums import logging, output_tools

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)
CEPH = "/ceph/submit/data/user/d/david_w/ZMass/cvh"
BR = ["dxrecsim", "dxerr", "hitDetId", "hitPitch", "hitUProj", "clusterSizeX",
      "clusterCharge", "clusterChargeBin", "localdxdz"]


def load(tag, nfiles, subdir="hitres2"):
    fs = [f for f in sorted(glob.glob(
        f"{CEPH}/{subdir}_{tag}/task_*/globalcor_resclosure_0.root"))
        if os.path.exists(os.path.join(os.path.dirname(f), ".complete"))][:nfiles]
    cols = {b: [] for b in BR}
    for fn in fs:
        a = uproot.open(fn)["tree"].arrays(BR, library="np")
        for b in BR:
            cols[b].append(np.concatenate(a[b]))
    d = {b: np.concatenate(v) for b, v in cols.items()}
    d["subdet"] = (d["hitDetId"].astype(np.uint32) >> 25) & 0x7
    d["ok"] = (d["dxrecsim"] > -98) & (d["dxerr"] > 0)
    return d, len(fs)


def w68(x):
    a, b = np.percentile(x, [15.865, 84.135])
    return 0.5 * (b - a)


def draw(ax, x, label, color=None, ls="-"):
    x = x[np.isfinite(x)]
    if x.size < 300:
        return
    z = (x - np.median(x)) / w68(x)
    b = np.linspace(-6, 6, 121)
    c = 0.5 * (b[1:] + b[:-1])
    h, _ = np.histogram(z, bins=b, density=True)
    ax.step(c, h, where="mid", color=color, ls=ls,
            label=f"{label}  (n={x.size/1e3:.0f}k)")


def refs(ax):
    c = np.linspace(-6, 6, 601)
    ax.plot(c, np.exp(-0.5 * c ** 2) / np.sqrt(2 * np.pi), "r--", lw=1.6,
            label="Gaussian")
    # uniform with the same 68 % half-width: half-width a, w68 = 0.6827 a
    a = 1.0 / 0.6827
    u = np.where(np.abs(c) <= a, 0.5 / a, 0.0)
    ax.plot(c, u, color="0.35", ls=":", lw=1.8, label="uniform")
    ax.set_yscale("log")
    ax.set_ylim(2e-5, 1.2)
    ax.set_xlabel(r"residual / its own 68 % half-width")
    ax.set_ylabel("normalised")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="mugun_lowpt")
    p.add_argument("--nfiles", type=int, default=35)
    p.add_argument("--outdir", default="")
    args = p.parse_args()
    today = datetime.date.today().strftime("%y%m%d")
    outdir = output_tools.make_plot_dir(args.outdir or os.path.expanduser(
        f"~/public_html/calibration_studies/{today}_hitres/"))

    d, nf = load(args.tag, args.nfiles)
    logger.info(f"{nf} files, {int(d['ok'].sum())} matched hits")
    r = d["dxrecsim"]
    ad = np.abs(d["localdxdz"])

    fig, ax = plt.subplots(1, 3, figsize=(19, 5.8), constrained_layout=True)

    # (a) STRIPS: the two regimes, each fully conditioned
    tib = d["ok"] & (d["subdet"] == 3)
    draw(ax[0], r[tib & (d["clusterSizeX"] == 1) & (d["hitUProj"] < 0.2)],
         "N=1, uProj<0.2", "tab:blue")
    draw(ax[0], r[tib & (d["clusterSizeX"] == 3) & (d["hitUProj"] < 0.2)],
         "N=3, uProj<0.2", "tab:cyan")
    draw(ax[0], r[tib & (d["clusterSizeX"] == 2) & (d["hitUProj"] > 0.4)],
         "N=2, uProj>0.4", "tab:orange")
    draw(ax[0], r[tib & (d["clusterSizeX"] == 4) & (d["hitUProj"] > 0.4)],
         "N=4, uProj>0.4", "tab:red")
    refs(ax[0])
    ax[0].set_title("TIB strips: (N, uProj)", fontsize=15)
    ax[0].legend(fontsize=10, loc="lower center", ncol=2)

    # (b) PIXELS: conditioned on the template's own class variables
    bp = d["ok"] & (d["subdet"] == 1) & (d["clusterSizeX"] == 2) & (ad < 0.15)
    for q, col in zip((0, 1, 2, 3),
                      ("tab:blue", "tab:orange", "tab:green", "tab:purple")):
        draw(ax[1], r[bp & (d["clusterChargeBin"] == q)], f"qbin {q}", col)
    refs(ax[1])
    ax[1].set_title("BPix x: sizeX=2, |dx/dz|<0.15", fontsize=15)
    ax[1].legend(fontsize=10, loc="lower center", ncol=2)

    # (c) the delta-ray test: tail vs CHARGE at fixed geometry
    edges = np.percentile(d["clusterCharge"][bp], [0, 50, 75, 90, 100])
    for (lo, hi), col in zip(zip(edges[:-1], edges[1:]),
                             ("tab:blue", "tab:green", "tab:orange", "tab:red")):
        m = bp & (d["clusterCharge"] >= lo) & (d["clusterCharge"] < hi)
        draw(ax[2], r[m], f"Q {lo/1e3:.0f}-{hi/1e3:.0f} ke", col)
    refs(ax[2])
    ax[2].set_title("same hits, split by charge", fontsize=15)
    ax[2].legend(fontsize=10, loc="lower center", ncol=2)

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"shape_conditioned.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {outdir}/shape_conditioned.png")


if __name__ == "__main__":
    main()
