"""Kink-finder tag rates by TRUE species on the B -> J/psi + X MC.

Inputs: globalcor_jpsix_{kaon,pi,mu}_*.root trees from
runCvhSingleTrackJpsiX.py — gen-matched kaon / pion / muon tracks from the
SAME inclusive B->J/psi+X ALCARECO production, fitted with the correct mass
hypothesis and doKinkFinder=True.

Physics expectations tested:
  - muons: null (no decays);
  - decay probability P = 1 - exp(-L/lambda), lambda = (p/m) c tau:
    lambda_K/lambda_pi = 0.135 at equal p -> kaon tag excess ~7.4x pion;
  - excess falls ~1/p (lambda grows with p).

Plots: per-species kinkMax survival; tag fraction vs p with decay-probability
expectation bands; prints the excess table and K/pi ratio.
"""

import argparse
import datetime
import os

import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot

from wums import logging, output_tools, plot_tools

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

M_PI, CTAU_PI = 0.13957, 7.804   # GeV, m
M_K, CTAU_K = 0.49368, 3.712

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaons", type=str, nargs="+", required=True)
    parser.add_argument("--pions", type=str, nargs="+", required=True)
    parser.add_argument("--muons", type=str, nargs="+", required=True)
    parser.add_argument("--outpath", type=str, default=None)
    parser.add_argument("--postfix", type=str, default="")
    parser.add_argument("--tagThreshold", type=float, default=50.0,
                        help="kinkMax threshold defining a tagged track")
    parser.add_argument("--pathLength", type=float, default=1.0,
                        help="nominal in-tracker path length [m] for the "
                             "decay-probability expectation bands")
    return parser.parse_args()


def load(files):
    return ak.concatenate([uproot.open(f)["tree"].arrays(
        ["kinkMax", "kinkMaxLayer", "kinkDchisq", "trackPt", "trackEta",
         "trackCharge", "nValidHits", "genPt", "genEta", "chisqval", "ndof"],
        library="ak") for f in files])


def gen_p(arr):
    """Parent (gen) momentum — the physically meaningful binning variable:
    a decayed hadron's RECO momentum follows the daughter, but the decay
    probability depends on the parent p."""
    return (arr.genPt * np.cosh(arr.genEta)).to_numpy()


def decay_prob(p, mass, ctau, L):
    lam = p / mass * ctau
    return 1.0 - np.exp(-L / lam)


def main():
    args = parse_args()
    today = datetime.date.today().strftime("%y%m%d")
    outpath = args.outpath or os.path.expanduser(
        f"~/public_html/ZMass/{today}_kink_species/")
    outdir = output_tools.make_plot_dir(outpath)
    logger.info(f"Writing plots to {outdir}")
    wd = os.path.dirname(os.path.abspath(__file__))

    samples = [("gen kaons", load(args.kaons), "#f89c20"),
               ("gen pions", load(args.pions), "#e42536"),
               ("gen muons", load(args.muons), "black")]

    # ---- kinkMax survival ------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 7))
    grid = np.linspace(0., 200., 201)
    for label, arr, color in samples:
        kmax = arr.kinkMax.to_numpy()
        sf = np.array([(kmax > g).mean() for g in grid])
        ax.plot(grid, sf, label=f"{label} ({len(kmax)} tracks)", color=color, lw=2)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.5)
    ax.set_xlabel(r"per-track $\mathrm{max}_i\,\Delta\chi^2_i$")
    ax.set_ylabel("fraction of tracks above")
    ax.legend()
    plot_tools.add_decor(ax, "CMS", "Work in progress", data=False,
                         lumi=None, loc=2)
    name = "_".join(filter(None, ["kink_species_survival", args.postfix]))
    plot_tools.save_pdf_and_png(outdir, name)
    output_tools.write_logfile(outdir, name, args=args, wd=wd)

    # ---- tag fraction vs p ----------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 7))
    pbins = np.array([3., 4., 5., 6.5, 8., 10., 14., 20.])
    pcent = 0.5 * (pbins[1:] + pbins[:-1])
    mu_frac = None
    for label, arr, color in samples:
        p = gen_p(arr)
        tag = (arr.kinkMax.to_numpy() > args.tagThreshold)
        fr, err = [], []
        for lo, hi in zip(pbins[:-1], pbins[1:]):
            m = (p >= lo) & (p < hi)
            n = m.sum()
            f = tag[m].mean() if n else np.nan
            fr.append(f)
            err.append(np.sqrt(f * (1 - f) / n) if n and not np.isnan(f) else np.nan)
        ax.errorbar(pcent, fr, yerr=err, fmt="o", color=color, label=label)
        if label == "gen muons":
            mu_frac = np.array(fr)
    pfine = np.linspace(3., 20., 100)
    base = np.nanmean(mu_frac) if mu_frac is not None else 0.
    ax.plot(pfine, base + decay_prob(pfine, M_K, CTAU_K, args.pathLength), "--",
            color="#f89c20", alpha=0.7,
            label=r"null + $P_{decay}^{K}$ ($L=%.1f$ m)" % args.pathLength)
    ax.plot(pfine, base + decay_prob(pfine, M_PI, CTAU_PI, args.pathLength), "--",
            color="#e42536", alpha=0.7,
            label=r"null + $P_{decay}^{\pi}$ ($L=%.1f$ m)" % args.pathLength)
    ax.set_xlabel(r"$p^{\mathrm{gen}}$ [GeV]")
    ax.set_ylabel(r"fraction with $\mathrm{max}_i\,\Delta\chi^2_i > %g$"
                  % args.tagThreshold)
    ax.set_ylim(0., None)
    ax.legend(fontsize="small")
    plot_tools.add_decor(ax, "CMS", "Work in progress", data=False,
                         lumi=None, loc=2)
    name = "_".join(filter(None, ["kink_species_tagfrac_p", args.postfix]))
    plot_tools.save_pdf_and_png(outdir, name)
    output_tools.write_logfile(outdir, name, args=args, wd=wd)

    # ---- table -----------------------------------------------------------
    print(f"\ntag fraction (kinkMax > {args.tagThreshold:g}):")
    fr = {}
    for label, arr, color in samples:
        kmax = arr.kinkMax.to_numpy()
        p = gen_p(arr)
        f = (kmax > args.tagThreshold).mean()
        n = len(kmax)
        fr[label] = f
        print(f"  {label:12s}: {f:.4f} +- {np.sqrt(f*(1-f)/n):.4f}  "
              f"({n} tracks, <p> {p.mean():.1f} GeV, "
              f"expected P_decay {decay_prob(p, M_K if 'kaon' in label else M_PI, CTAU_K if 'kaon' in label else CTAU_PI, args.pathLength).mean():.4f})")
    exc_k = fr["gen kaons"] - fr["gen muons"]
    exc_pi = fr["gen pions"] - fr["gen muons"]
    if exc_pi > 0:
        print(f"  kaon excess {exc_k:+.4f}, pion excess {exc_pi:+.4f}, "
              f"ratio {exc_k/exc_pi:.1f} (lifetime prediction at equal p: 7.4)")


if __name__ == "__main__":
    main()
