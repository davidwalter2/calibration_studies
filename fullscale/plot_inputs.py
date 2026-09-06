#!/usr/bin/env python3
"""Phase-1 input diagnostics: what the selection and the two corrections do.

One panel per file, ratio panel wherever a comparison is being made, into
`~/public_html/cvh/<YYMMDD>_fullscale/`.
"""
import argparse
import datetime
import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.gridspec import GridSpec

from wums import logging, output_tools, plot_tools

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

MZ = 91.1876


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True)
    p.add_argument("--outpath", default=None)
    p.add_argument("--postfix", default="")
    p.add_argument("--title", default="CMS")
    p.add_argument("--subtitle", default="Simulation, work in progress")
    p.add_argument("--titlePos", type=int, default=2)
    p.add_argument("--max-sigma-rel", type=float, default=0.10)
    p.add_argument("--max-chi2-ndof", type=float, default=3.0)
    p.add_argument("--window", type=float, nargs=2, default=[60.0, 120.0])
    return p.parse_args()


def outpath(args):
    if args.outpath:
        return os.path.expanduser(args.outpath)
    today = datetime.date.today().strftime("%y%m%d")
    return os.path.expanduser(f"~/public_html/cvh/{today}_fullscale/")


def save(outdir, name, args):
    name = "_".join(filter(None, [name, args.postfix]))
    plot_tools.save_pdf_and_png(outdir, name)
    output_tools.write_logfile(outdir, name, args=args,
                               wd=os.path.dirname(os.path.abspath(__file__)))
    plt.close("all")


def ratio_axes(figsize=(8, 7)):
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.06)
    ax = fig.add_subplot(gs[0])
    rax = fig.add_subplot(gs[1], sharex=ax)
    plt.setp(ax.get_xticklabels(), visible=False)
    return fig, ax, rax


def main():
    args = parse_args()
    outdir = outpath(args)
    os.makedirs(outdir, exist_ok=True)
    d = np.load(args.pairs, allow_pickle=True)
    z, sig, mgen = d["z"], d["sigma"], d["eta"]
    m = z * sig + mgen
    w = np.asarray(d["w"], np.float64)
    w = w / w.mean()
    chi2n = np.asarray(d["chisqval"], np.float64) / np.maximum(
        np.asarray(d["ndof"], np.float64), 1.0)
    srel = sig / np.maximum(np.abs(m), 1e-9)
    vgf = np.asarray(d["vgf"], np.float64)
    mtrk = np.asarray(d["mtrk"], np.float64)
    lo, hi = args.window
    base = np.isfinite(m) & (sig > 0)
    win = base & (m >= lo) & (m <= hi)
    sel = win & (chi2n < args.max_chi2_ndof) & (srel < args.max_sigma_rel)
    logger.info(f"{len(m)} cached, {int(win.sum())} in window, {int(sel.sum())} selected")

    # ---- 1. the observed mass spectrum, selected vs cached -------------
    edges = np.linspace(lo, hi, 241)
    c = 0.5 * (edges[1:] + edges[:-1])
    h_all, _ = np.histogram(m[win], edges, weights=w[win])
    h_sel, _ = np.histogram(m[sel], edges, weights=w[sel])
    fig, ax, rax = ratio_axes()
    ax.step(c, h_all, where="mid", label="in window", color="k")
    ax.step(c, h_sel, where="mid",
            label=rf"$\chi^2/\mathrm{{ndof}}<{args.max_chi2_ndof:g}$, "
                  rf"$\sigma_m/m<{args.max_sigma_rel:g}$", color="tab:red")
    ax.set_ylabel("weighted candidates / 0.25 GeV")
    ax.set_yscale("log")
    ax.legend(fontsize=13)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = h_sel / h_all
    rax.step(c, r, where="mid", color="tab:red")
    rax.axhline(1.0, color="k", lw=0.8)
    rax.set_ylim(0.9, 1.01)
    rax.set_xlabel(r"$m_{\mu\mu}$ (CVH) [GeV]")
    rax.set_ylabel("selected / all")
    plot_tools.add_decor(ax, args.title, args.subtitle, data=False, lumi=None,
                         loc=args.titlePos)
    save(outdir, "01_mass_selection", args)

    # ---- 2. sigma_m/m and where the cut sits ---------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    b = np.logspace(-2.4, 0.5, 160)
    ax.hist(srel[win], bins=b, weights=w[win], histtype="step", color="k",
            label="in window")
    ax.hist(srel[win & (chi2n < args.max_chi2_ndof)], bins=b,
            weights=w[win & (chi2n < args.max_chi2_ndof)], histtype="step",
            color="tab:blue", label=rf"$\chi^2/\mathrm{{ndof}}<{args.max_chi2_ndof:g}$")
    ax.axvline(args.max_sigma_rel, color="tab:red", ls="--",
               label=rf"cut {args.max_sigma_rel:g}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\sigma_m/m$")
    ax.set_ylabel("weighted candidates")
    ax.legend(fontsize=13)
    plot_tools.add_decor(ax, args.title, args.subtitle, data=False, lumi=None,
                         loc=args.titlePos)
    save(outdir, "02_sigma_rel", args)

    # ---- 3. the two corrections, per candidate -------------------------
    a_res = (1.0 + vgf) * sig / np.maximum(np.abs(m), 1e-9)
    jen = 1.5 * srel ** 2 * MZ * 1e3           # MeV
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(a_res[sel], bins=np.linspace(0, 0.12, 160), weights=w[sel],
            histtype="step", color="tab:blue",
            label=r"$a_i=(1+f_{\mathrm{hit}})\sigma_i/m_i$")
    ax.set_xlabel(r"$a_i$")
    ax.set_ylabel("weighted candidates")
    ax.legend(fontsize=13)
    ax.text(0.55, 0.55, rf"median {np.median(a_res[sel]):.4f}",
            transform=ax.transAxes, fontsize=14)
    plot_tools.add_decor(ax, args.title, args.subtitle, data=False, lumi=None,
                         loc=args.titlePos)
    save(outdir, "03_a_res", args)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(jen[sel], bins=np.linspace(0, 120, 160), weights=w[sel],
            histtype="step", color="tab:orange",
            label=r"$1.5\,(\sigma_m/m)^2\,m_Z$")
    ax.set_xlabel("Jensen shift per candidate [MeV]")
    ax.set_ylabel("weighted candidates")
    ax.legend(fontsize=13)
    ax.text(0.5, 0.6, rf"median {np.median(jen[sel]):.1f} MeV",
            transform=ax.transAxes, fontsize=14)
    plot_tools.add_decor(ax, args.title, args.subtitle, data=False, lumi=None,
                         loc=args.titlePos)
    save(outdir, "04_jensen_shift", args)

    # ---- 4. the selection variable vs the modelled observable ----------
    fig, ax, rax = ratio_axes()
    h_trk, _ = np.histogram(mtrk[base], edges, weights=w[base])
    h_cvh, _ = np.histogram(m[base], edges, weights=w[base])
    ax.step(c, h_trk, where="mid", color="k",
            label=r"$m_{\mu\mu}$ (input tracks) -- the CUT variable")
    ax.step(c, h_cvh, where="mid", color="tab:green",
            label=r"$m_{\mu\mu}$ (CVH refit) -- the MODELLED variable")
    ax.set_ylabel("weighted candidates / 0.25 GeV")
    ax.set_yscale("log")
    ax.legend(fontsize=12)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = h_cvh / h_trk
    rax.step(c, r, where="mid", color="tab:green")
    rax.axhline(1.0, color="k", lw=0.8)
    rax.set_ylim(0.8, 1.2)
    rax.set_xlabel(r"$m_{\mu\mu}$ [GeV]")
    rax.set_ylabel("CVH / input")
    nout = int((base & (mtrk >= lo) & (mtrk <= hi) & ~win).sum())
    ax.text(0.05, 0.15, f"{nout} of {int(base.sum())} "
                        f"({100.0*nout/base.sum():.3f} %) selected on the input\n"
                        f"mass fall outside the window after the refit",
            transform=ax.transAxes, fontsize=12)
    plot_tools.add_decor(ax, args.title, args.subtitle, data=False, lumi=None,
                         loc=args.titlePos)
    save(outdir, "05_selection_variable", args)

    # ---- 5. the MiNNLO weights ----------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(w[sel], bins=np.linspace(-6, 12, 180), histtype="step", color="k")
    ax.set_yscale("log")
    ax.set_xlabel("MiNNLO weight (rescaled to mean 1)")
    ax.set_ylabel("candidates")
    neff = w[sel].sum() ** 2 / (sel.sum() * (w[sel] ** 2).sum())
    ax.text(0.45, 0.75,
            f"{100.0*(w[sel] < 0).mean():.2f} % negative\n"
            rf"$N_{{\rm eff}}/N = {neff:.4f}$" "\n"
            rf"errors $\times {1/np.sqrt(neff):.3f}$",
            transform=ax.transAxes, fontsize=14)
    plot_tools.add_decor(ax, args.title, args.subtitle, data=False, lumi=None,
                         loc=args.titlePos)
    save(outdir, "06_weights", args)
    logger.info(f"figures -> {outdir}")


if __name__ == "__main__":
    logging.setup_logger(__file__, 3, False)
    main()
