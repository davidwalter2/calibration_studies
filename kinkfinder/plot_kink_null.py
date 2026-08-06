"""Kink-finder score-test distributions: muon null vs V0 pions (+ MC muons).

Inputs are `tree` TTrees from ResidualGlobalCorrectionMakerG4e with
doKinkFinder=True (branches kinkDchisq*, kinkDqop, kinkMax, kinkMaxLayer).

Muons do not decay in flight (cTau ~ 660 km at these momenta): their
kinkDchisq is the NULL of the score test, chi2(3)-like up to the
multiple-scattering heavy tail. K_S -> pi pi daughter pions DO decay
(cTau 7.8 m -> ~0.5%/track in-tracker): the pion excess over the muon
null at high kinkMax, with q*dqop > 0 (momentum loss), is the
decay-in-flight signal this tagger exists for.

Plots:
  1. per-step kinkDchisq densities + chi2(3) reference (log y)
  2. per-track kinkMax survival curves (the tagging money plot)
  3. signed momentum-step discriminant q*dqophat at tagged steps
Prints per-sample tag fractions above thresholds.
"""

import argparse
import datetime
import os

import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot
from scipy import stats

from wums import logging, output_tools, plot_tools

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--muons", type=str, nargs="+", required=True,
        help="globalcor_single_*.root files: J/psi muon tracks (data null)")
    parser.add_argument(
        "--pions", type=str, nargs="+", required=True,
        help="globalcor_v0single_*.root files: KS->pipi daughter pion tracks")
    parser.add_argument(
        "--muonsMC", type=str, nargs="*", default=[],
        help="optional globalcor_singlemc_*.root files: MC muon tracks")
    parser.add_argument("--outpath", type=str, default=None,
                        help="Output base dir. Default: ~/public_html/ZMass/YYMMDD_kink_null/")
    parser.add_argument("--postfix", type=str, default="")
    parser.add_argument("--tagThreshold", type=float, default=25.0,
                        help="kinkDchisq threshold defining a tagged step for the "
                             "q*dqop discriminant plot")
    return parser.parse_args()


def load(files):
    arrs = []
    for f in files:
        t = uproot.open(f)["tree"]
        arrs.append(t.arrays(
            ["kinkDchisq", "kinkDchisqAngle", "kinkDchisqQop", "kinkDqop",
             "kinkMax", "kinkMaxLayer", "chisqval", "ndof", "trackPt",
             "trackEta", "trackCharge", "nValidHits"],
            library="ak"))
    return ak.concatenate(arrs)


def sf_curve(vals, grid):
    vals = np.asarray(vals)
    return np.array([(vals > g).mean() for g in grid])


def main():
    args = parse_args()
    today = datetime.date.today().strftime("%y%m%d")
    outpath = args.outpath or os.path.expanduser(
        f"~/public_html/ZMass/{today}_kink_null/")
    outdir = output_tools.make_plot_dir(outpath)
    logger.info(f"Writing plots to {outdir}")

    samples = [("Muons (data)", load(args.muons), "black"),
               ("KS pions (data)", load(args.pions), "#e42536")]
    if args.muonsMC:
        samples.append(("Muons (MC)", load(args.muonsMC), "#5790fc"))

    # ---- 1. per-step Dchisq densities -----------------------------------
    fig, ax = plt.subplots(figsize=(8, 7))
    bins = np.linspace(0., 60., 121)
    centers = 0.5 * (bins[1:] + bins[:-1])
    for label, arr, color in samples:
        d3 = ak.flatten(arr.kinkDchisq).to_numpy()
        h, _ = np.histogram(d3, bins=bins, density=True)
        ax.stairs(h, bins, label=f"{label} ({len(d3)} steps)", color=color, lw=2)
    ax.plot(centers, stats.chi2.pdf(centers, 3), "--", color="gray",
            label=r"$\chi^2(3)$")
    ax.set_yscale("log")
    ax.set_ylim(1e-7, 2.)
    ax.set_xlabel(r"per-step kink $\Delta\chi^2$")
    ax.set_ylabel("density")
    ax.legend()
    plot_tools.add_decor(ax, "CMS", "Work in progress", data=True, lumi=None, loc=2)
    name = "_".join(filter(None, ["kink_dchisq_step", args.postfix]))
    plot_tools.save_pdf_and_png(outdir, name)
    output_tools.write_logfile(outdir, name, args=args,
                               wd=os.path.dirname(os.path.abspath(__file__)))

    # ---- 2. per-track kinkMax survival ----------------------------------
    fig, ax = plt.subplots(figsize=(8, 7))
    grid = np.linspace(0., 200., 201)
    for label, arr, color in samples:
        kmax = arr.kinkMax.to_numpy()
        ax.plot(grid, sf_curve(kmax, grid), label=f"{label} ({len(kmax)} tracks)",
                color=color, lw=2)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.5)
    ax.set_xlabel(r"per-track $\mathrm{max}_i\,\Delta\chi^2_i$")
    ax.set_ylabel("fraction of tracks above")
    ax.legend()
    plot_tools.add_decor(ax, "CMS", "Work in progress", data=True, lumi=None, loc=2)
    name = "_".join(filter(None, ["kink_max_survival", args.postfix]))
    plot_tools.save_pdf_and_png(outdir, name)
    output_tools.write_logfile(outdir, name, args=args,
                               wd=os.path.dirname(os.path.abspath(__file__)))

    # ---- 3. signed momentum-step discriminant at tagged steps ------------
    fig, ax = plt.subplots(figsize=(8, 7))
    bins = np.linspace(-0.05, 0.05, 101)
    for label, arr, color in samples:
        d3 = arr.kinkDchisq
        dqop = arr.kinkDqop
        q = arr.trackCharge
        tagged = d3 > args.tagThreshold
        s = ak.flatten(dqop[tagged] * q).to_numpy()
        if len(s) == 0:
            continue
        h, _ = np.histogram(np.clip(s, bins[0], bins[-1]), bins=bins, density=True)
        ax.stairs(h, bins, label=f"{label} ({len(s)} tagged steps)", color=color, lw=2)
    ax.axvline(0., color="gray", ls=":")
    ax.set_xlabel(r"$q\cdot\widehat{\delta}_{q/p}$ at steps with $\Delta\chi^2 > %g$"
                  % args.tagThreshold)
    ax.set_ylabel("density")
    ax.legend()
    plot_tools.add_decor(ax, "CMS", "Work in progress", data=True, lumi=None, loc=2)
    name = "_".join(filter(None, ["kink_qdqop_tagged", args.postfix]))
    plot_tools.save_pdf_and_png(outdir, name)
    output_tools.write_logfile(outdir, name, args=args,
                               wd=os.path.dirname(os.path.abspath(__file__)))

    # ---- summary table ---------------------------------------------------
    print("\ntag fraction per track: kinkMax > threshold")
    thresholds = [10., 25., 50., 100.]
    header = "sample".ljust(18) + "".join(f">{t:<8g}" for t in thresholds) + "tracks"
    print(header)
    ref = None
    for label, arr, color in samples:
        kmax = arr.kinkMax.to_numpy()
        fracs = [(kmax > t).mean() for t in thresholds]
        print(label.ljust(18) + "".join(f"{f:<9.4f}" for f in fracs) + f"{len(kmax)}")
        if ref is None:
            ref = fracs
        else:
            excess = [f - r for f, r in zip(fracs, ref)]
            print("  excess vs null ".ljust(18)
                  + "".join(f"{e:<+9.4f}" for e in excess))
    for label, arr, color in samples:
        pt = arr.trackPt.to_numpy()
        nh = arr.nValidHits.to_numpy()
        print(f"{label}: <pT> = {pt.mean():.2f} GeV, <nValidHits> = {nh.mean():.1f}, "
              f"chisq/ndof median = "
              f"{np.median(arr.chisqval.to_numpy()/arr.ndof.to_numpy()):.2f}")


if __name__ == "__main__":
    main()
