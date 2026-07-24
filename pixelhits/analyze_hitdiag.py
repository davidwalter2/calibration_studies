#!/usr/bin/env python
"""Per-side bias attribution of pathological pixel hits.

Reads the hitdiag_* branches of a runCvhJpsiGenMC.py run with
fitFromGenParms=True keepPixelEdgeHits=True pixelMinSizeX=1
fillHitDiagnostics=True deweightPathoHits=True: every pixel hit keeps its
surface and trajectory state, pathological hits exert no pull, so their
local residuals dy0 = (hit - predicted) are unbiased w.r.t. the
gen-anchored fit.

Class bits: 0 = -x edge, 1 = +x edge, 2 = -y edge, 3 = +y edge,
4 = sizeX==1, 5 = sizeY==1.
"""

import argparse
import datetime
import glob

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot

from wums import output_tools, plot_tools

hep.style.use(hep.style.ROOT)

CM2UM = 1e4


def load(rundir):
    cols = ["hitdiag_detid", "hitdiag_class", "hitdiag_dx", "hitdiag_dy",
            "hitdiag_exx", "hitdiag_eyy", "hitdiag_lx", "hitdiag_ly"]
    out = {c: [] for c in cols}
    for f in sorted(glob.glob(rundir + "/globalcor_*.root")):
        a = uproot.open(f)["tree"].arrays(cols, library="np")
        for c in cols:
            out[c].append(np.concatenate(a[c]) if len(a[c]) else np.array([]))
    return {c: np.concatenate(v) for c, v in out.items()}


def robust_mean(x):
    """Mean/error of the central 98% (clips pathological tails)."""
    if len(x) < 10:
        return np.nan, np.nan
    lo, hi = np.percentile(x, [1, 99])
    xc = x[(x >= lo) & (x <= hi)]
    return xc.mean(), xc.std() / np.sqrt(len(xc))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rundir", required=True)
    date_tag = datetime.date.today().strftime("%y%m%d")
    p.add_argument("-o", "--outpath",
                   default=f"/home/submit/david_w/public_html/ZMass/{date_tag}_pixelhits_hitdiag")
    args = p.parse_args()
    outdir = output_tools.make_plot_dir(args.outpath, "")

    a = load(args.rundir)
    det, cls = a["hitdiag_detid"], a["hitdiag_class"]
    dx, dy = a["hitdiag_dx"] * CM2UM, a["hitdiag_dy"] * CM2UM
    subdet = (det >> 25) & 0x7          # 1 = BPix, 2 = FPix (phase-0 2016)
    layer = (det >> 16) & 0xF           # BPix layer / FPix disk
    print(f"total pixel hits: {len(det)}   pathological: {(cls != 0).sum()}"
          f" ({100.*(cls != 0).mean():.1f}%)")

    edgeXlo, edgeXhi = (cls & 1) > 0, (cls & 2) > 0
    edgeYlo, edgeYhi = (cls & 4) > 0, (cls & 8) > 0
    sizeX1, sizeY1 = (cls & 16) > 0, (cls & 32) > 0
    anyedge = edgeXlo | edgeXhi | edgeYlo | edgeYhi
    clean = cls == 0

    # (label, residual array, selection, color)
    cats = [
        ("clean (x)",      dx, clean, "gray"),
        ("edge $-x$ (x)",  dx, edgeXlo, "tab:red"),
        ("edge $+x$ (x)",  dx, edgeXhi, "tab:orange"),
        ("sizeX1 (x)",     dx, sizeX1 & ~anyedge, "tab:blue"),
        ("clean (y)",      dy, clean, "darkgray"),
        ("edge $-y$ (y)",  dy, edgeYlo, "tab:pink"),
        ("edge $+y$ (y)",  dy, edgeYhi, "tab:brown"),
        ("sizeY1 (y)",     dy, sizeY1 & ~anyedge, "tab:green"),
    ]

    print("\n=== global per-class/side robust means (um) ===")
    print(f"{'class':>14s} {'subdet':>7s} {'n':>8s} {'<res>':>10s} {'err':>8s} {'RMS':>8s}")
    for label, res, sel, _c in cats:
        for sd, sdname in ((1, "BPix"), (2, "FPix")):
            s = sel & (subdet == sd)
            if s.sum() < 10:
                continue
            m, e = robust_mean(res[s])
            print(f"{label:>14s} {sdname:>7s} {s.sum():8d} {m:+10.2f} {e:8.2f} {res[s].std():8.1f}")

    print("\n=== BPix per-layer (um) ===")
    for label, res, sel, _c in cats:
        for lay in (1, 2, 3):
            s = sel & (subdet == 1) & (layer == lay)
            if s.sum() < 10:
                continue
            m, e = robust_mean(res[s])
            print(f"{label:>14s}  L{lay}  n={s.sum():7d}  <res>={m:+8.2f} +- {e:.2f} um")

    # summary plot: per class/side mean residual, BPix vs FPix
    fig, ax = plt.subplots(figsize=(13, 9))
    xticks, xlabels = [], []
    for i, (label, res, sel, color) in enumerate(cats):
        for j, (sd, marker) in enumerate(((1, "o"), (2, "s"))):
            s = sel & (subdet == sd)
            if s.sum() < 10:
                continue
            m, e = robust_mean(res[s])
            ax.errorbar(i + 0.15 * (j - 0.5), m, yerr=e, fmt=marker, color=color,
                        markersize=9, label="_")
        xticks.append(i)
        xlabels.append(label + f"\n(n={sel.sum()})")
    ax.axhline(0, color="gray", ls="--")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize="x-small")
    ax.set_ylabel(r"robust $\langle$residual$\rangle$ ($\mu$m)")
    ax.plot([], [], "o", color="black", label="BPix")
    ax.plot([], [], "s", color="black", label="FPix")
    ax.legend(fontsize="small")
    plot_tools.add_decor(ax, "CMS", "Simulation, private work", data=False, lumi=None, no_energy=True)
    plot_tools.save_pdf_and_png(outdir, "hitdiag_mean_by_class_side", fig)
    plt.close(fig)

    # residual distributions for the key classes
    for coord, res, pairs in (
            ("x", dx, [("clean", clean, "gray"), ("edge $-x$", edgeXlo, "tab:red"),
                       ("edge $+x$", edgeXhi, "tab:orange"), ("sizeX1", sizeX1 & ~anyedge, "tab:blue")]),
            ("y", dy, [("clean", clean, "darkgray"), ("edge $-y$", edgeYlo, "tab:pink"),
                       ("edge $+y$", edgeYhi, "tab:brown"), ("sizeY1", sizeY1 & ~anyedge, "tab:green")])):
        fig, ax = plt.subplots(figsize=(11, 9))
        bins = np.linspace(-300, 300, 121)
        for label, sel, color in pairs:
            if sel.sum() < 10:
                continue
            m, _ = robust_mean(res[sel])
            ax.hist(np.clip(res[sel], bins[0], bins[-1]), bins=bins, histtype="step",
                    density=True, color=color, label=f"{label} ($\\mu$={m:+.1f} $\\mu$m)")
        ax.set_xlabel(rf"local ${coord}$ residual ($\mu$m)")
        ax.set_ylabel("hits (normalised)")
        ax.legend(fontsize="x-small")
        plot_tools.add_decor(ax, "CMS", "Simulation, private work", data=False, lumi=None, no_energy=True)
        plot_tools.save_pdf_and_png(outdir, f"hitdiag_res{coord}_dist", fig)
        plt.close(fig)

    output_tools.write_index_and_log(outdir, "analyze_hitdiag", args=args)
    print("\nplots in", outdir)


if __name__ == "__main__":
    main()
