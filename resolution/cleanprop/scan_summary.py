"""Aggregate the clean-propagation campaign across (pT, eta, species).

Each campaign point writes a cleanprop_rows_<label>.npz (cf_propagation_test
--compare --label). This collects them and answers the two questions the scan
was run for:

  1. does the first-principles model still describe Geant4 as the delta-ray
     tail hardens (pT up), as the material grows (eta up), and for hadrons?
     -> worst |data - model| in the bounded average <exp(-u z^2)>, over layers.

  2. how big is the mean-vs-mode displacement that the track fit mistakes for
     a momentum bias? -> median z at the outermost plane, in units of the
     propagator's own width.

Labels must parse as <species>_pt<pt>_eta<eta>, e.g. K_pt2_eta0.30.

usage:
    python scan_summary.py '~/public_html/calibration_studies/260806_cleanprop_scan/cleanprop_rows_*.npz'
"""

import argparse
import datetime
import glob
import os
import re

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from wums import logging, output_tools, plot_tools

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

LABELRE = re.compile(r"^(?P<sp>[A-Za-z]+)_pt(?P<pt>[\d.]+)_eta(?P<eta>[\d.]+)$")
SPECIES_TEX = {"mu": r"$\mu^-$", "K": r"$K^-$", "pi": r"$\pi^-$", "p": r"$p$"}
SPECIES_COL = {"mu": "#1f77b4", "K": "#d62728", "pi": "#2ca02c", "p": "#9467bd"}
FUNC_TEX = {"qop": r"$q/p$ (ionization)", "locx": r"local $x$ (MS)",
            "dxdz": r"$dx/dz$ (MS)"}


def load(pattern):
    pts = []
    for fn in sorted(glob.glob(os.path.expanduser(pattern))):
        d = np.load(fn, allow_pickle=False)
        label = str(d["label"])
        m = LABELRE.match(label)
        if not m:
            logger.warning(f"label '{label}' does not parse, skipping {fn}")
            continue
        pts.append(dict(
            label=label, sp=m["sp"], pt=float(m["pt"]), eta=float(m["eta"]),
            probes=d["probes"], func=d["func"].astype(str), layer=d["layer"],
            r=d["r"], mean=d["mean"], median=d["median"], rob=d["rob"],
            fdata=d["fdata"], fmodel=d["fmodel"],
            nkept=int(d["nkept"]), ntot=int(d["ntot"])))
    return pts


def worst_dev(p, func):
    """max |data - model| over layers and probes for one functional."""
    m = p["func"] == func
    if not m.any():
        return np.nan
    return float(np.abs(p["fdata"][m] - p["fmodel"][m]).max())


def outer_median(p, func):
    """median z at the outermost plane -- the mean-vs-mode displacement."""
    m = p["func"] == func
    if not m.any():
        return np.nan
    return float(p["median"][m][np.argmax(p["r"][m])])


def table(pts):
    lines = []
    lines.append(f"{'label':>20} {'clean':>8} {'events':>9} "
                 + " ".join(f"{'worst|d-m| ' + f:>18}" for f in ("qop", "locx", "dxdz"))
                 + f" {'med z (qop,outer)':>18}")
    for p in sorted(pts, key=lambda q: (q["sp"], q["pt"], q["eta"])):
        clean = 100. * p["nkept"] / max(p["ntot"], 1)
        lines.append(
            f"{p['label']:>20} {clean:7.2f}% {p['nkept']:9d} "
            + " ".join(f"{worst_dev(p, f):>18.4f}" for f in ("qop", "locx", "dxdz"))
            + f" {outer_median(p, 'qop'):>18.2f}")
    return "\n".join(lines)


def scan_plot(pts, xkey, sel, xlabel, outdir, args, name, logx=True):
    """worst |data-model| vs a scan variable, one panel per functional."""
    sub = [p for p in pts if sel(p)]
    if not sub:
        logger.warning(f"no points for {name}")
        return
    funcs = ("qop", "locx", "dxdz")
    fig, axes = plt.subplots(1, len(funcs), figsize=(16, 5.6), sharey=True)
    for ax, func in zip(axes, funcs):
        for sp in sorted({p["sp"] for p in sub}):
            q = sorted([p for p in sub if p["sp"] == sp], key=lambda z: z[xkey])
            x = [p[xkey] for p in q]
            y = [worst_dev(p, func) for p in q]
            ax.plot(x, y, "o-", color=SPECIES_COL.get(sp, "k"), markersize=8,
                    label=SPECIES_TEX.get(sp, sp), lw=2)
        ax.set_xlabel(xlabel)
        ax.set_title(FUNC_TEX[func], fontsize=19)
        if logx:
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"worst $|$data$-$model$|$")
    axes[0].legend(fontsize=16)
    # figure-level label: add_decor writes inside one axes and collided with
    # both the panel title and the (13 TeV) tag on a 3-panel row
    fig.text(0.09, 0.97, "CMS", fontsize=25, fontweight="bold", va="top")
    fig.text(0.155, 0.965, "Simulation Work in progress",
             fontsize=18, style="italic", va="top")
    fig.text(0.90, 0.965, "(13 TeV)", fontsize=17, va="top", ha="right")
    fig.subplots_adjust(top=0.88)
    fig.savefig(os.path.join(outdir, f"{name}.png"), bbox_inches="tight", dpi=150)
    fig.savefig(os.path.join(outdir, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {outdir}/{name}.png")


def meanmode_plot(pts, outdir, args):
    """median z at the outermost plane vs momentum, per species.

    This is the quantity the track fit mistakes for a momentum bias: the refit
    subtracts the MEAN loss, the typical track loses less, so the mode sits
    high. Its species dependence is the multi-species lever.
    """
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for sp in sorted({p["sp"] for p in pts}):
        q = sorted([p for p in pts if p["sp"] == sp and abs(p["eta"] - 0.30) < 1e-6],
                   key=lambda z: z["pt"])
        if not q:
            continue
        ax.plot([p["pt"] for p in q], [outer_median(p, "qop") for p in q],
                "o-", color=SPECIES_COL.get(sp, "k"), markersize=9,
                label=SPECIES_TEX.get(sp, sp), lw=2)
    ax.axhline(0., color="k", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"$p_T$ [GeV]")
    ax.set_ylabel(r"median $z$, outer plane [$\sigma_{\rm prop}$]")
    ax.legend(fontsize=17)
    ax.grid(alpha=0.3)
    plot_tools.add_decor(ax, "CMS", "Work in progress",
                         data=False, lumi=None, loc=2)
    fig.savefig(os.path.join(outdir, "cleanprop_meanmode.png"),
                bbox_inches="tight", dpi=150)
    fig.savefig(os.path.join(outdir, "cleanprop_meanmode.pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {outdir}/cleanprop_meanmode.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pattern", help="glob of cleanprop_rows_*.npz (quote it)")
    p.add_argument("--outpath", default=None)
    args = p.parse_args()
    logging.setup_logger(__file__, 3, False)

    today = datetime.date.today().strftime("%y%m%d")
    outdir = output_tools.make_plot_dir(
        args.outpath or os.path.expanduser(
            f"~/public_html/calibration_studies/{today}_cleanprop_scan/"))

    pts = load(args.pattern)
    if not pts:
        raise SystemExit(f"no parseable points in {args.pattern}")
    logger.info(f"{len(pts)} campaign points")

    txt = table(pts)
    print("\n" + txt + "\n")
    with open(os.path.join(outdir, "cleanprop_scan_summary.txt"), "w") as fh:
        fh.write(txt + "\n")

    scan_plot(pts, "pt", lambda q: abs(q["eta"] - 0.30) < 1e-6,
              r"$p_T$ [GeV]", outdir, args, "cleanprop_scan_pt")
    scan_plot(pts, "eta", lambda q: q["sp"] == "mu" and abs(q["pt"] - 10.) < 1e-6,
              r"$\eta$", outdir, args, "cleanprop_scan_eta", logx=False)
    meanmode_plot(pts, outdir, args)


if __name__ == "__main__":
    main()
