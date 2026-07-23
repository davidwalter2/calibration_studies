"""CVH glued-module (TIB-L2 detId 369141860) data/MC efficiency scale factor.

SF = eps_data(buggy) / eps_MC ~ N_bug / N_fix, measured from the A/B CVH
refit on 2016G SingleMuon (with vs without the eb96caef fix). Produces:
  1. SF vs eta   (in the module phi-stripe 0.80 < phi < 1.00), full eta range
  2. SF vs phi   (in the affected eta band 0.1 < eta < 0.65), full phi range
  3. SF 2D (eta, phi), full ranges

Both arms processed identical events, so the per-bin denominator cancels and
SF = N_bug/N_fix directly. Errors are binomial (n = N_fix).

Run with the cms-plots venv:
  ~/.claude/skills/cms-plots/.venv/bin/python plot_cvh_efficiency_sf.py
"""

import argparse
import datetime
import glob
import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot
from matplotlib.ticker import NullLocator

from wums import logging, output_tools, plot_tools

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

BASE = "/ceph/submit/data/user/d/david_w/effstudy_eb96caef"
ORIG_TASKS = ["0000", "0001", "0003", "0004", "0006"]  # matched pairs from the first run
PT_LO, PT_HI = 25.0, 65.0
PHI_LO, PHI_HI = 0.80, 1.00  # module stripe
ETA_LO, ETA_HI = 0.10, 0.65  # affected eta band


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outpath", default=None, help="output base dir")
    p.add_argument("--postfix", default="", help="filename suffix")
    p.add_argument("--title", default="CMS")
    p.add_argument("--subtitle", default="Work in progress")
    p.add_argument("--titlePos", type=int, default=2)
    return p.parse_args()


def default_outpath(script_path):
    stem = os.path.splitext(os.path.basename(script_path))[0]
    root = os.path.dirname(os.path.abspath(script_path))
    while root and root != "/" and not os.path.isdir(os.path.join(root, ".git")):
        root = os.path.dirname(root)
    repo = os.path.basename(root) if root and root != "/" else "plots"
    today = datetime.date.today().strftime("%y%m%d")
    return os.path.expanduser(f"~/public_html/{repo}/{today}_{stem}/")


def load(tag):
    files = sorted(glob.glob(f"{BASE}/night_{tag}_317c988483d/task_*/effstudy_miniaod_{tag}_0.root"))
    files += [f"{BASE}/{tag}_317c988483d/task_{t}/effstudy_miniaod_{tag}_0.root" for t in ORIG_TASKS]
    files = [f for f in files if os.path.exists(f)]
    d = uproot.concatenate(
        [f"{f}:tree" for f in files],
        ["trackPt", "trackEta", "trackPhi"],
        library="np",
    )
    logger.info(f"{tag}: {len(files)} files, {len(d['trackPt'])} tracks")
    return d


def ratio_1d(fix, bug, var, edges, gate):
    """SF and binomial error per bin of `var`, applying the boolean `gate(d)`."""
    xf, xb = fix[var][gate(fix)], bug[var][gate(bug)]
    nf, _ = np.histogram(xf, bins=edges)
    nb, _ = np.histogram(xb, bins=edges)
    sf = np.divide(nb, nf, out=np.ones_like(nf, dtype=float), where=nf > 0)
    err = np.where(nf > 0, np.sqrt(np.clip(sf * (1 - sf), 0, None) / np.maximum(nf, 1)), 0.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, sf, err, nf


def ratio_2d(fix, bug, eta_edges, phi_edges, gate):
    hf, _, _ = np.histogram2d(fix["trackEta"][gate(fix)], fix["trackPhi"][gate(fix)], bins=[eta_edges, phi_edges])
    hb, _, _ = np.histogram2d(bug["trackEta"][gate(bug)], bug["trackPhi"][gate(bug)], bins=[eta_edges, phi_edges])
    sf = np.divide(hb, hf, out=np.full_like(hf, np.nan, dtype=float), where=hf > 20)
    return sf


def base_gate(d):
    return (d["trackPt"] >= PT_LO) & (d["trackPt"] < PT_HI)


def make_plots(outdir, args, fix, bug):
    # ---- 1D eta (phi stripe) ----
    eta_edges = np.arange(-2.4, 2.4001, 0.05)
    gate_eta = lambda d: base_gate(d) & (d["trackPhi"] >= PHI_LO) & (d["trackPhi"] < PHI_HI)
    cx, sf, err, _ = ratio_1d(fix, bug, "trackEta", eta_edges, gate_eta)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(cx, sf, yerr=err, fmt="o", ms=3, color="#1f77b4", label=rf"${PHI_LO} < \phi < {PHI_HI}$")
    ax.axhline(1.0, ls="--", lw=1, color="gray")
    ax.set_xlabel(r"muon $\eta$")
    ax.set_ylabel(r"SF $= \varepsilon_\mathrm{data}/\varepsilon_\mathrm{MC}$")
    ax.set_ylim(0.4, 1.15)
    ax.set_xlim(-2.4, 2.4)
    ax.legend(loc="lower left")
    plot_tools.add_decor(ax, args.title, args.subtitle, data=False, lumi=None, loc=args.titlePos)
    name = "_".join(filter(None, ["cvh_sf_vs_eta", args.postfix]))
    plot_tools.save_pdf_and_png(outdir, name)
    output_tools.write_logfile(outdir, name, args=args, wd=os.path.dirname(__file__))

    # ---- 1D phi (eta band) ----
    phi_edges = np.arange(-np.pi, np.pi + 1e-6, 0.10)
    gate_phi = lambda d: base_gate(d) & (d["trackEta"] >= ETA_LO) & (d["trackEta"] < ETA_HI)
    cx, sf, err, _ = ratio_1d(fix, bug, "trackPhi", phi_edges, gate_phi)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(cx, sf, yerr=err, fmt="o", ms=3, color="#d62728", label=rf"${ETA_LO} < \eta < {ETA_HI}$")
    ax.axhline(1.0, ls="--", lw=1, color="gray")
    ax.axvspan(PHI_LO, PHI_HI, color="orange", alpha=0.15, label="module stripe")
    ax.set_xlabel(r"muon $\phi$ [rad]")
    ax.set_ylabel(r"SF $= \varepsilon_\mathrm{data}/\varepsilon_\mathrm{MC}$")
    ax.set_ylim(0.4, 1.15)
    ax.set_xlim(-np.pi, np.pi)
    ax.legend(loc="lower left")
    plot_tools.add_decor(ax, args.title, args.subtitle, data=False, lumi=None, loc=args.titlePos)
    name = "_".join(filter(None, ["cvh_sf_vs_phi", args.postfix]))
    plot_tools.save_pdf_and_png(outdir, name)
    output_tools.write_logfile(outdir, name, args=args, wd=os.path.dirname(__file__))

    # ---- 2D (eta, phi) ----
    eta2 = np.arange(-2.4, 2.4001, 0.1)
    phi2 = np.arange(-np.pi, np.pi + 1e-6, 0.15)
    sf2 = ratio_2d(fix, bug, eta2, phi2, base_gate)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        sf2.T, origin="lower", aspect="auto",
        extent=[eta2[0], eta2[-1], phi2[0], phi2[-1]],
        cmap="viridis", vmin=0.55, vmax=1.0,
    )
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(r"muon $\eta$")
    ax.set_ylabel(r"muon $\phi$ [rad]")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r"SF $= \varepsilon_\mathrm{data}/\varepsilon_\mathrm{MC}$")
    cb.ax.yaxis.set_minor_locator(NullLocator())
    plot_tools.add_decor(ax, args.title, args.subtitle, data=False, lumi=None, loc=args.titlePos)
    name = "_".join(filter(None, ["cvh_sf_2d_eta_phi", args.postfix]))
    plot_tools.save_pdf_and_png(outdir, name)
    output_tools.write_logfile(outdir, name, args=args, wd=os.path.dirname(__file__))


def main():
    args = parse_args()
    outdir = output_tools.make_plot_dir(args.outpath or default_outpath(__file__))
    logger.info(f"Writing plots to {outdir}")
    fix, bug = load("fix"), load("bug")
    make_plots(outdir, args, fix, bug)
    logger.info("done")


if __name__ == "__main__":
    main()
