"""Momentum spectrum of the CVH-refit cosmic-muon tracks.

Reads the cosmics grads production (Run2016G+H NoBPTX TkAlCosmicsInCollisions,
CVH single-track refit) and plots the seed track momentum distribution:
  - full momentum  p = pT * cosh(eta)
  - transverse pT
both on a log-y axis (cosmic spectrum spans ~10 GeV to >1 TeV), split by charge.

The tracks are the ones that entered the field+material global fit
(cvh/cosmics_calib2016_grads50_260718_*), i.e. after the 3.8T good-run filter.
"""

import argparse
import datetime
import glob
import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from wums import logging, output_tools, plot_tools

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

DEFAULT_GLOB = (
    "/ceph/submit/data/user/d/david_w/ZMass/cvh/"
    "cosmics_calib2016_grads50_260718_357acafa052/task_*/globalcor_cosmics_*.root"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=DEFAULT_GLOB, help="glob of globalcor cosmics files")
    p.add_argument("--cache", default=None,
                   help="npz with pt/eta/q to skip the ROOT read (written on first run)")
    p.add_argument("--outpath", default=None, help="output base dir")
    p.add_argument("--postfix", default="", help="filename suffix")
    p.add_argument("--title", default="CMS")
    p.add_argument("--subtitle", default="Work in progress")
    p.add_argument("--titlePos", type=int, default=2)
    p.add_argument("--rlabel", default="2016 G/H", help="top-right label (era, not beam energy)")
    return p.parse_args()


def default_outpath(script_path):
    stem = os.path.splitext(os.path.basename(script_path))[0]
    root = os.path.dirname(os.path.abspath(script_path))
    while root and root != "/" and not os.path.isdir(os.path.join(root, ".git")):
        root = os.path.dirname(root)
    repo = os.path.basename(root) if root and root != "/" else "plots"
    today = datetime.date.today().strftime("%y%m%d")
    return os.path.expanduser(f"~/public_html/{repo}/{today}_{stem}/")


def load(args):
    keys = ("pt", "eta", "q", "phi", "z0", "d0")
    if args.cache and os.path.exists(args.cache):
        d = np.load(args.cache)
        if all(k in d for k in keys):
            return {k: d[k] for k in keys}
    import uproot
    pt, eta, q, phi, z0, d0 = ([] for _ in range(6))
    files = sorted(glob.glob(args.input))
    logger.info(f"reading {len(files)} files")
    for f in files:
        t = uproot.open(f)["tree"]
        if t.num_entries == 0:
            continue
        pt.append(t["trackPt"].array(library="np"))
        eta.append(t["trackEta"].array(library="np"))
        q.append(t["trackCharge"].array(library="np"))
        phi.append(t["trackPhi"].array(library="np"))
        # trackParms = CVH PCA 5-vector [q/p, lambda, phi, d0, z0] in cm
        tp = np.stack(t["trackParms"].array(library="np"))
        d0.append(tp[:, 3])
        z0.append(tp[:, 4])
    out = {k: np.concatenate(v) for k, v in
           zip(keys, (pt, eta, q, phi, z0, d0))}
    if args.cache:
        np.savez(args.cache, **out)
    return out


def decorate(ax, args):
    """CMS label with the era in the top-right instead of a beam energy
    (cosmics have no collision energy, so '(13 TeV)' is meaningless)."""
    hep.cms.label(ax=ax, data=True, text=args.subtitle,
                  rlabel=args.rlabel, loc=args.titlePos)


def make_plot(outdir, args, data):
    pt, eta, q = data["pt"], data["eta"], data["q"]
    phi, z0 = data["phi"], data["z0"]
    p = pt * np.cosh(eta)
    pos, neg = q > 0, q < 0

    for var, arr, xlabel, stem, hi in [
        (p, p, r"$p$ [GeV]", "cosmic_p", 1000.0),
        (pt, pt, r"$p_{T}$ [GeV]", "cosmic_pt", 1000.0),
    ]:
        bins = np.geomspace(5, hi, 60)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(arr, bins=bins, histtype="step", color="black", label=f"all ({len(arr)})")
        ax.hist(arr[pos], bins=bins, histtype="step", color="#c1272d",
                label=rf"$\mu^{{+}}$ ({pos.sum()})")
        ax.hist(arr[neg], bins=bins, histtype="step", color="#0000a7",
                label=rf"$\mu^{{-}}$ ({neg.sum()})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("cosmic tracks / bin")
        med = np.median(arr)
        ax.axvline(med, ls="--", color="grey", lw=1)
        ax.text(med * 1.05, ax.get_ylim()[1] * 0.5, f"median {med:.0f} GeV",
                rotation=90, va="top", fontsize=12, color="grey")
        ax.legend(loc="upper right", fontsize=13)
        decorate(ax, args)
        name = "_".join(filter(None, [stem, args.postfix]))
        plot_tools.save_pdf_and_png(outdir, name)
        output_tools.write_logfile(outdir, name, args=args, wd=os.path.dirname(__file__))
        plt.close(fig)

    # pseudorapidity: cosmics come from above -> central, linear scale
    bins = np.linspace(-2.0, 2.0, 81)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(eta, bins=bins, histtype="step", color="black", label=f"all ({len(eta)})")
    ax.hist(eta[pos], bins=bins, histtype="step", color="#c1272d",
            label=rf"$\mu^{{+}}$ ({pos.sum()})")
    ax.hist(eta[neg], bins=bins, histtype="step", color="#0000a7",
            label=rf"$\mu^{{-}}$ ({neg.sum()})")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("cosmic tracks / bin")
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=13)
    decorate(ax, args)
    name = "_".join(filter(None, ["cosmic_eta", args.postfix]))
    plot_tools.save_pdf_and_png(outdir, name)
    output_tools.write_logfile(outdir, name, args=args, wd=os.path.dirname(__file__))
    plt.close(fig)

    # azimuth of the track momentum: cosmics travel downward -> phi in (-pi, 0),
    # peaked at -pi/2 (straight down)
    bins = np.linspace(-np.pi, np.pi, 73)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(phi, bins=bins, histtype="step", color="black", label=f"all ({len(phi)})")
    ax.hist(phi[pos], bins=bins, histtype="step", color="#c1272d",
            label=rf"$\mu^{{+}}$ ({pos.sum()})")
    ax.hist(phi[neg], bins=bins, histtype="step", color="#0000a7",
            label=rf"$\mu^{{-}}$ ({neg.sum()})")
    ax.axvline(-np.pi / 2, ls="--", color="grey", lw=1)
    ax.text(-np.pi / 2 + 0.08, ax.get_ylim()[1] * 0.92, r"$-\pi/2$ (down)",
            fontsize=12, color="grey")
    ax.set_xlabel(r"track momentum $\phi$")
    ax.set_ylabel("cosmic tracks / bin")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=13)
    decorate(ax, args)
    name = "_".join(filter(None, ["cosmic_phi", args.postfix]))
    plot_tools.save_pdf_and_png(outdir, name)
    output_tools.write_logfile(outdir, name, args=args, wd=os.path.dirname(__file__))
    plt.close(fig)

    # z of the track PCA w.r.t. the nominal interaction point (z0, cm):
    # cosmics do NOT come from the IP -> broad, unlike the luminous region
    zwin = 150.0
    bins = np.linspace(-zwin, zwin, 76)
    fig, ax = plt.subplots(figsize=(8, 6))
    # no clipping: tracks with |z0| > zwin fall outside the binning and are
    # simply not counted (avoids fake overflow spikes at the edges)
    ax.hist(z0, bins=bins, histtype="step", color="black", label=f"all ({len(z0)})")
    ax.hist(z0[pos], bins=bins, histtype="step", color="#c1272d",
            label=rf"$\mu^{{+}}$ ({pos.sum()})")
    ax.hist(z0[neg], bins=bins, histtype="step", color="#0000a7",
            label=rf"$\mu^{{-}}$ ({neg.sum()})")
    ax.set_xlabel(r"$z_0$ w.r.t. nominal IP [cm]")
    ax.set_ylabel("cosmic tracks / bin")
    ax.set_xlim(-zwin, zwin)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=13)
    decorate(ax, args)
    name = "_".join(filter(None, ["cosmic_z0", args.postfix]))
    plot_tools.save_pdf_and_png(outdir, name)
    output_tools.write_logfile(outdir, name, args=args, wd=os.path.dirname(__file__))
    plt.close(fig)


def main():
    args = parse_args()
    outdir = output_tools.make_plot_dir(args.outpath or default_outpath(__file__))
    logger.info(f"Writing plots to {outdir}")
    data = load(args)
    pt, eta, q = data["pt"], data["eta"], data["q"]
    logger.info(f"{len(pt)} tracks; median p = "
                f"{np.median(pt*np.cosh(eta)):.1f} GeV, charge+/- = "
                f"{(q>0).sum()/(q<0).sum():.3f}; "
                f"median |z0| = {np.median(np.abs(data['z0'])):.1f} cm")
    make_plot(outdir, args, data)


if __name__ == "__main__":
    main()
