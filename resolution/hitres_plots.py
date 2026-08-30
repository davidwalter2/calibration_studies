#!/usr/bin/env python3
"""Figures + cross-species table for the hit-resolution pull study.

Everything here is the same measurement as hitres_pull.py, drawn instead of
tabulated, plus the multi-arm comparison the tables cannot show: the same
CPE covariance is applied to a muon and to a kaon at the SAME momentum, and
the only way the error model can know the difference is through the two
implicit dE/dx dependences (the strip dQdx > maxChgOneMIP branch and the
pixel charge bin), neither of which is given a particle hypothesis.

Core width and tail fraction are always drawn as two separate panels. A
single rms would merge exactly the two things this study has to keep apart.

usage:
  python hitres_plots.py --tags mugun_lowpt piongun_ul16 kaongun_ul16 protongun_ul16
  python hitres_plots.py --tags mugun_lowpt mugun_ul16 --label momentum
"""
import argparse
import datetime
import os
import sys

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hitres_pull as hp                                          # noqa: E402
from wums import logging, output_tools                            # noqa: E402

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

NICE = {"mugun_lowpt": r"$\mu$, $p_T$ 2-20", "mugun_ul16": r"$\mu$, $p_T$ 20-60",
        "piongun_ul16": r"$\pi$, $p_T$ 2-20", "kaongun_ul16": r"$K$, $p_T$ 2-20",
        "protongun_ul16": r"$p$, $p_T$ 2-20"}
COL = {"mugun_lowpt": "k", "mugun_ul16": "0.45", "piongun_ul16": "tab:blue",
       "kaongun_ul16": "tab:red", "protongun_ul16": "tab:green"}


SUFFIX = ""


def save(fig, outdir, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{name}{SUFFIX}.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)


def fig_pull_shape(data, outdir, tags):
    """The distribution itself: is it Gaussian, and where does it stop being one?

    Drawn on a log scale against a unit Gaussian normalised to the same area.
    This is the plot that decides `rescale` vs `wrong likelihood`, and it has
    to be looked at, not summarised.
    """
    groups = [("pixel x", lambda d: d["ispixel"], "pullx"),
              ("pixel y", lambda d: d["ispixel"], "pully"),
              ("strip x / phi", lambda d: ~d["ispixel"], "pullx")]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6), constrained_layout=True)
    b = np.linspace(-8, 8, 161)
    c = 0.5 * (b[1:] + b[:-1])
    for ax, (lab, msk, key) in zip(axes, groups):
        for tag in tags:
            d = data[tag]
            m = d["sel"] & msk(d) & np.isfinite(d[key])
            x = d[key][m]
            if x.size < 100:
                continue
            h, _ = np.histogram(x, bins=b, density=True)
            ax.step(c, h, where="mid", color=COL.get(tag, None),
                    label=f"{NICE.get(tag, tag)}  (n={x.size/1e3:.0f}k)")
        ax.plot(c, np.exp(-0.5 * c ** 2) / np.sqrt(2 * np.pi), "r--", lw=1.5,
                label="unit Gaussian")
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1.0)
        ax.set_xlabel(r"pull  $(\mathrm{rec}-\mathrm{sim})/\sigma_\mathrm{CPE}$")
        ax.set_ylabel("normalised")
        ax.set_title(lab)
        ax.legend(fontsize=11, loc="upper right")
    save(fig, outdir, "pull_shape")


def _profile(d, xvar, edges, key="pullx", extra=None):
    """Core width, tail fraction and median in bins of xvar."""
    core, cerr, tail, med, ns, ctr = [], [], [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = d["sel"] & (d[xvar] >= lo) & (d[xvar] < hi) & np.isfinite(d[key])
        if extra is not None:
            m &= extra
        s = hp.robust_stats(d[key][m])
        if s.get("n", 0) < 200:
            core.append(np.nan); cerr.append(np.nan); tail.append(np.nan)
            med.append(np.nan); ns.append(s.get("n", 0)); ctr.append(0.5 * (lo + hi))
            continue
        core.append(s["core"]); cerr.append(s["core_err"]); tail.append(s["tail3"])
        med.append(s["median"]); ns.append(s["n"]); ctr.append(0.5 * (lo + hi))
    return (np.array(ctr), np.array(core), np.array(cerr),
            np.array(tail), np.array(med), np.array(ns))


def fig_vs_uproj(data, outdir, tags):
    """Against the CPE's own independent variable.

    A flat line at 1 would mean the parametrisation is right and the pull is
    Gaussian. A flat line elsewhere means a pure rescaling would fix it. A
    SLOPE means the functional form is wrong, and that is what a correction
    would have to change.
    """
    edges = np.array([0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 5.0])
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6), constrained_layout=True)
    for tag in tags:
        d = data[tag]
        strip = ~d["ispixel"]
        x, c, ce, t, m, n = _profile(d, "hitUProj", edges, "pullx", extra=strip)
        axes[0].errorbar(x, c, ce, fmt="o-", color=COL.get(tag), ms=4,
                         label=NICE.get(tag, tag))
        axes[1].plot(x, 100 * t, "o-", color=COL.get(tag), ms=4)
        axes[2].plot(x, m, "o-", color=COL.get(tag), ms=4)
    axes[0].axhline(1.0, color="r", ls="--", lw=1.2)
    axes[0].set_ylabel("core width of the pull")
    axes[1].axhline(100 * 0.0027, color="r", ls="--", lw=1.2)
    axes[1].set_ylabel(r"fraction beyond $3\times$core [%]")
    axes[1].set_yscale("log")
    axes[2].axhline(0.0, color="r", ls="--", lw=1.2)
    axes[2].set_ylabel("median pull (bias)")
    for a in axes:
        a.set_xlabel(r"$u_\mathrm{proj}$  (projected path / strip pitch)")
        a.set_xscale("log")
    axes[0].legend(fontsize=11)
    save(fig, outdir, "strip_vs_uproj")


def fig_vs_N(data, outdir, tags):
    edges = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 6.5, 9.5, 20.5])
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6), constrained_layout=True)
    for tag in tags:
        d = data[tag]
        strip = ~d["ispixel"]
        x, c, ce, t, m, n = _profile(d, "clusterSizeX", edges, "pullx", extra=strip)
        axes[0].errorbar(x, c, ce, fmt="o-", color=COL.get(tag), ms=4,
                         label=NICE.get(tag, tag))
        axes[1].plot(x, 100 * t, "o-", color=COL.get(tag), ms=4)
        axes[2].plot(x, m, "o-", color=COL.get(tag), ms=4)
    axes[0].axhline(1.0, color="r", ls="--", lw=1.2)
    axes[0].set_ylabel("core width of the pull")
    axes[1].axhline(100 * 0.0027, color="r", ls="--", lw=1.2)
    axes[1].set_ylabel(r"fraction beyond $3\times$core [%]")
    axes[1].set_yscale("log")
    axes[2].axhline(0.0, color="r", ls="--", lw=1.2)
    axes[2].set_ylabel("median pull (bias)")
    for a in axes:
        a.set_xlabel("strip cluster width N")
        a.axvline(4.5, color="0.6", ls=":", lw=1.0)
    axes[0].legend(fontsize=11, title="dotted line: the CPE\nswitches branch at N=4",
                   title_fontsize=9)
    save(fig, outdir, "strip_vs_N")


def fig_subdet(data, outdir, tags):
    """One bar per subdetector, core and tail side by side."""
    sds = [(1, "BPix x"), (1, "BPix y"), (2, "FPix x"), (2, "FPix y"),
           (3, "TIB"), (4, "TID"), (5, "TOB"), (6, "TEC")]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), constrained_layout=True)
    w = 0.8 / max(len(tags), 1)
    for i, tag in enumerate(tags):
        d = data[tag]
        cs, ce, ts = [], [], []
        for sd, lab in sds:
            key = "pully" if lab.endswith(" y") else "pullx"
            m = d["sel"] & (d["subdet"] == sd) & np.isfinite(d[key])
            s = hp.robust_stats(d[key][m])
            cs.append(s.get("core", np.nan)); ce.append(s.get("core_err", np.nan))
            ts.append(100 * s.get("tail3", np.nan))
        xx = np.arange(len(sds)) + (i - 0.5 * (len(tags) - 1)) * w
        axes[0].bar(xx, cs, w, yerr=ce, color=COL.get(tag), alpha=0.85,
                    label=NICE.get(tag, tag))
        axes[1].bar(xx, ts, w, color=COL.get(tag), alpha=0.85)
    axes[0].axhline(1.0, color="r", ls="--", lw=1.2)
    axes[0].set_ylabel("core width of the pull")
    axes[0].legend(fontsize=11)
    axes[1].axhline(100 * 0.0027, color="r", ls="--", lw=1.2)
    axes[1].set_ylabel(r"fraction beyond $3\times$core [%]")
    axes[1].set_yscale("log")
    for a in axes:
        a.set_xticks(np.arange(len(sds)))
        a.set_xticklabels([l for _, l in sds], rotation=30, ha="right")
    save(fig, outdir, "by_subdet")


def fig_pixel_qbin(data, outdir, tags):
    """The pixel non-Gaussianity is a MIXTURE, and this is the mixing variable.

    qbin 0 is the highest cluster charge (the delta-ray end). If the core
    width ran flat across the bins the aggregate pull could be Gaussian; it
    does not, so the aggregate is a superposition of components with
    different widths, which no per-module scale can represent.
    """
    qs = [0, 1, 2, 3]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6), constrained_layout=True)
    for tag in tags:
        d = data[tag]
        pix = d["sel"] & d["ispixel"]
        for key, ls, mk in (("pullx", "-", "o"), ("pully", "--", "s")):
            cs, ce, ts = [], [], []
            for q in qs:
                st = hp.robust_stats(d[key][pix & (d["clusterChargeBin"] == q)])
                cs.append(st.get("core", np.nan))
                ce.append(st.get("core_err", np.nan))
                ts.append(100 * st.get("tail3", np.nan))
            lab = f"{NICE.get(tag, tag)} {'x' if key == 'pullx' else 'y'}"
            axes[0].errorbar(qs, cs, ce, fmt=mk + ls, color=COL.get(tag), ms=5,
                             label=lab)
            axes[1].plot(qs, ts, mk + ls, color=COL.get(tag), ms=5)
    axes[0].axhline(1.0, color="r", ls="--", lw=1.2)
    axes[0].set_ylabel("core width of the pull")
    axes[0].legend(fontsize=9, ncol=2)
    axes[1].axhline(100 * 0.0027, color="r", ls="--", lw=1.2)
    axes[1].set_ylabel(r"fraction beyond $3\times$core [%]")
    axes[1].set_yscale("log")
    # the mixture itself, drawn for the first arm
    d0 = data[tags[0]]
    pix0 = d0["sel"] & d0["ispixel"]
    b = np.linspace(-8, 8, 161)
    c = 0.5 * (b[1:] + b[:-1])
    for q in qs:
        m = pix0 & (d0["clusterChargeBin"] == q)
        h, _ = np.histogram(d0["pullx"][m], bins=b, density=True)
        axes[2].step(c, h, where="mid", label=f"qbin {q} (n={int(m.sum())/1e3:.0f}k)")
    axes[2].plot(c, np.exp(-0.5 * c ** 2) / np.sqrt(2 * np.pi), "r--", lw=1.5,
                 label="unit Gaussian")
    axes[2].set_yscale("log")
    axes[2].set_ylim(1e-5, 1.0)
    axes[2].set_xlabel(r"pixel local-x pull")
    axes[2].set_ylabel("normalised")
    axes[2].set_title(NICE.get(tags[0], tags[0]))
    axes[2].legend(fontsize=10)
    for a in axes[:2]:
        a.set_xlabel("template charge bin\n(0 = highest charge)")
        a.set_xticks(qs)
    save(fig, outdir, "pixel_vs_qbin")


def table(data, tags, out=print):
    out("")
    out("== cross-arm summary: core width / tail fraction of the hit pull ==")
    out(f"  {'arm':<18} {'sim-matched':>11}  " +
        "  ".join(f"{k:>16}" for k in ("pixel x", "pixel y", "strip x/phi")))
    for tag in tags:
        d = data[tag]
        cells = []
        for key, msk in (("pullx", d["ispixel"]), ("pully", d["ispixel"]),
                         ("pullx", ~d["ispixel"])):
            s = hp.robust_stats(d[key][d["sel"] & msk & np.isfinite(d[key])])
            cells.append(f"{s.get('core', np.nan):.3f} /{100*s.get('tail3', np.nan):5.2f}%"
                         if s.get("n", 0) > 20 else "        --      ")
        out(f"  {NICE.get(tag, tag):<18} {100*d['matchfrac']:10.2f}%  " +
            "  ".join(f"{c:>16}" for c in cells))
    out("  (Gaussian reference: core 1.000, tail 0.27 %)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tags", nargs="+",
                   default=["mugun_lowpt", "piongun_ul16", "kaongun_ul16",
                            "protongun_ul16"])
    p.add_argument("--subdir", default="hitres")
    p.add_argument("--nfiles", type=int, default=0)
    p.add_argument("--maxchi2", type=float, default=0.)
    p.add_argument("--ptmin", type=float, default=0.)
    p.add_argument("--ptmax", type=float, default=0.)
    p.add_argument("--etamax", type=float, default=0.)
    p.add_argument("--label", default="")
    p.add_argument("--outdir", default="")
    args = p.parse_args()
    global SUFFIX
    SUFFIX = f"_{args.label}" if args.label else ""

    today = datetime.date.today().strftime("%y%m%d")
    outdir = args.outdir or os.path.expanduser(
        f"~/public_html/calibration_studies/{today}_hitres/")
    outdir = output_tools.make_plot_dir(outdir)

    data = {}
    for tag in args.tags:
        d = hp.load(tag, args.nfiles, args.subdir)
        d["sel"] = hp.select(d, args)
        d["matchfrac"] = float(np.mean(d["matched"]))
        data[tag] = d
        logger.info(f"{tag}: {int(d['sel'].sum())} hits selected of {d['matched'].size}")

    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    table(data, args.tags, out)
    suff = f"_{args.label}" if args.label else ""
    with open(os.path.join(outdir, f"summary{suff}.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    fig_pull_shape(data, outdir, args.tags)
    fig_vs_uproj(data, outdir, args.tags)
    fig_vs_N(data, outdir, args.tags)
    fig_subdet(data, outdir, args.tags)
    fig_pixel_qbin(data, outdir, args.tags)
    logger.info(f"figures in {outdir}")


if __name__ == "__main__":
    main()
