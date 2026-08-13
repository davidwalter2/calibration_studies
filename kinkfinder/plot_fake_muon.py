"""Decay-in-flight hadrons that fake a prompt muon: how many can the kink
finder reject, and at what cost in real muons?

This is a different question from the species ROCs in plot_kink_roc.py. There
the background is truth-clean tracks of the SAME species -- how well do we spot
a damaged kaon among kaons. Here the background is REAL PROMPT MUONS, which is
what matters for the momentum-scale calibration itself: a hadron that decays in
flight and is reconstructed as a muon enters the J/psi (or Z) sample directly.

Inputs are the fitAs=mu doMuons=True passes: hadron tracks fitted with the
MUON mass hypothesis (as they would be in the real analysis, since the analysis
does not know they are hadrons) and matched to the ALCARECO loose-muon
collection so the muon identification can be required.

Only decays *between the first and last measurement* leave any trace in the
tracker fit. Decays past the last hit still fake a muon -- the daughter reaches
the chambers -- but the tracker sees a pristine hadron track, so no kink finder
can ever reject them. That fraction is reported as the ceiling.

usage:
  python plot_fake_muon.py --kaons '<dir>/kaon/chunk_*/globalcor_jpsix_kaon_0.root' \
                           --pions '<dir>/pi/...' --muons '<dir>/mu/...'
"""

import argparse
import datetime
import glob
import os
import sys

import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kink_truth as kt

from wums import logging, output_tools, plot_tools

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

LABEL = {"kaon": "kaon", "pi": "pion", "mu": "muon"}
DISCR = [("D3", r"$\max_i\,\Delta\chi^2_i$ (3 dof)", "#3f90da"),
         ("Dangle", r"angle only ($\Delta\lambda,\Delta\phi$)", "#e42536"),
         ("Dqop", r"$\Delta(q/p)$ only", "#f89c20")]
LOSSES = (0.01, 0.005, 0.001)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--kaons", nargs="+", required=True)
    p.add_argument("--pions", nargs="+", default=[])
    p.add_argument("--muons", nargs="+", required=True)
    p.add_argument("--outpath", default=None)
    p.add_argument("--postfix", default="")
    p.add_argument("--minHits", type=int, default=8)
    p.add_argument("--muonId", nargs="+",
                   default=["any", "muonLoose", "muonMedium", "muonTight"],
                   choices=["muonLoose", "muonMedium", "muonTight",
                            "muonIsGlobal", "any"],
                   help="muon identification(s) a track must pass to count as "
                        "reconstructed-as-a-muon ('any' = matched at all). "
                        "Several may be given; the sample is loaded once and "
                        "the whole ladder is scanned.")
    return p.parse_args()


def expand(patterns):
    out = []
    for pat in patterns:
        out += sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat]
    return out


def load(patterns, minhits):
    files = expand(patterns)
    if not files:
        return None
    logger.info(f"loading {len(files)} files")
    arr = kt.load(files)
    d = kt.label(arr)
    keep = d["sim_found"] & (d["nvalid"] >= minhits) & (d["nsteps"] > 2)
    n0 = d["n"]
    d = {k: (v[keep] if isinstance(v, np.ndarray) and getattr(v, "shape", (0,))[:1] == (n0,) else v)
         for k, v in d.items()}
    d["n"] = int(keep.sum())
    d["arr"] = arr[keep]
    logger.info(f"  {n0} -> {d['n']} tracks after sim-match and quality cuts")
    return d


def isamuon(d, which):
    if "muonPt" not in d:
        raise SystemExit("no muon branches — rerun with doMuons=True")
    matched = d["muonPt"] > 0
    return matched if which == "any" else (matched & d[which].astype(bool))


def main():
    args = parse_args()
    today = datetime.date.today().strftime("%y%m%d")
    outpath = args.outpath or os.path.expanduser(
        f"~/public_html/ZMass/{today}_kink_fakemuon/")
    outdir = output_tools.make_plot_dir(outpath)
    logger.info(f"Writing plots to {outdir}")
    wd = os.path.dirname(os.path.abspath(__file__))

    def save(name):
        name = "_".join(filter(None, [name, args.postfix]))
        plot_tools.save_pdf_and_png(outdir, name)
        output_tools.write_logfile(outdir, name, args=args, wd=wd)
        plt.close("all")

    species = {}
    for key, pats in (("kaon", args.kaons), ("pi", args.pions), ("mu", args.muons)):
        d = load(pats, args.minHits)
        if d is not None:
            species[key] = d
    mu = species["mu"]

    summary = {}
    for muid in args.muonId:
        analyse(args, species, mu, muid, save, summary)

    # ---- how the whole ladder compares -----------------------------------
    print("\n=== rejection at 1% real-muon loss vs muon ID ===")
    print(f"  {'muon ID':12s} {'real mu eff':>11s} " + "".join(
        f"{k:>18s}" for k in ("kaon fakes", "pion fakes")))
    for muid in args.muonId:
        if muid not in summary:
            continue
        s_ = summary[muid]
        print(f"  {muid:12s} {s_['mueff']*100:10.2f}% "
              + "".join(f"{s_.get(k, float('nan')):18.3f}" for k in ("kaon", "pi"))
              + f"   (n = {s_.get('nk', 0)} K, {s_.get('npi', 0)} pi fakes)")


def analyse(args, species, mu, muid, save, summary):
    # ---- how often does each truth class fake a muon? --------------------
    print(f"\n=== reconstructed as a muon ({muid}) ===")
    for key, d in species.items():
        ism = isamuon(d, muid)
        print(f"  {LABEL[key]:5s}: {ism.sum():7d} / {d['n']:8d} tracks "
              f"({ism.mean()*100:6.3f}%)")
        for cls, clab in (("clean", "clean"), ("has_nuclear", "nuclear"),
                          ("has_decay", "decay in flight")):
            m = d[cls]
            if m.sum() < 5:
                continue
            print(f"      {clab:16s} {int((m & ism).sum()):6d} / {int(m.sum()):7d} "
                  f"= {(ism[m].mean()*100):6.3f}%")

    # enrichment: P(in-span decay | fakes a muon) vs P(in-span decay)
    print("\n=== decay enrichment of the fake-muon sample ===")
    for key in ("kaon", "pi"):
        if key not in species:
            continue
        d = species[key]
        ism = isamuon(d, muid)
        if ism.sum() < 5:
            continue
        print(f"  {LABEL[key]}: in-span decay fraction "
              f"{d['has_decay'].mean()*100:.2f}% overall -> "
              f"{d['has_decay'][ism].mean()*100:.2f}% among muon-tagged tracks "
              f"(x{d['has_decay'][ism].mean()/max(d['has_decay'].mean(),1e-12):.1f})")

    # ---- ceiling: where do the decays of muon-faking hadrons happen? -----
    print("\n=== where the decay happens, for hadrons reconstructed as muons ===")
    for key in ("kaon", "pi"):
        if key not in species:
            continue
        d, a = species[key], species[key]["arr"]
        ism = isamuon(d, muid)
        r = np.sqrt(a.simVtxX**2 + a.simVtxY**2)
        s = np.sqrt(r**2 + a.simVtxZ**2)
        isdec = a.simVtxProcType == kt.PROC_DECAY
        sf, sl = d["s_first"], d["s_last"]
        before = ak.to_numpy(ak.any(isdec & (s <= sf[:, None]), axis=1))
        inside = ak.to_numpy(ak.any(isdec & (s > sf[:, None]) & (s < sl[:, None]), axis=1))
        after = ak.to_numpy(ak.any(isdec & (s >= sl[:, None]), axis=1))
        anyd = before | inside | after
        tot = int((anyd & ism).sum())
        if tot < 5:
            continue
        print(f"  {LABEL[key]}s reconstructed as muons with a decay anywhere: {tot}")
        for lab, m in (("before first hit (invisible)", before),
                       ("in-span (taggable)", inside),
                       ("after last hit (no tracker trace)", after)):
            print(f"      {lab:34s} {int((m & ism).sum()):5d} "
                  f"({100*(m & ism).sum()/tot:5.1f}%)")

    # ---- ROC: fakes vs real prompt muons ---------------------------------
    mureal = mu["clean"] & isamuon(mu, muid)
    summary[muid] = {"mueff": float(isamuon(mu, muid)[mu["clean"]].mean())}
    print(f"\n=== rejection of decay-in-flight fakes "
          f"(background = {int(mureal.sum())} real prompt muons passing {muid}) ===")
    fig, ax = plt.subplots(figsize=(8, 7))
    for key, ls in (("kaon", "-"), ("pi", "--")):
        if key not in species:
            continue
        d = species[key]
        sig = d["has_decay"] & isamuon(d, muid)
        if sig.sum() < 10:
            print(f"  {LABEL[key]}: only {int(sig.sum())} fakes — skipping")
            continue
        print(f"  {LABEL[key]} fakes (n={int(sig.sum())}):")
        for name, lab, color in DISCR:
            fpr, tpr, auc = kt.roc(
                np.concatenate([d[name], mu[name]]),
                np.concatenate([sig, np.zeros(mu["n"], bool)]),
                np.concatenate([np.zeros(d["n"], bool), mureal]))
            if len(fpr):
                ax.plot(np.clip(fpr, 1e-5, None), np.clip(tpr, 1e-4, None),
                        color=color, ls=ls, lw=2,
                        label=(lab if key == "kaon" else None))
            rej = []
            for loss in LOSSES:
                cut = kt.threshold_at_fpr(mu[name], mureal, loss)
                rej.append((d[name][sig] > cut).mean())
            print(f"    {name:8s} AUC {auc:.3f}   rejection at real-muon loss "
                  + "   ".join(f"{l*100:g}%: {r:.3f}" for l, r in zip(LOSSES, rej)))
            # headline for the ladder table: best discriminant at 1% loss
            summary[muid][key] = max(summary[muid].get(key, 0.0), rej[0])
            summary[muid]["nk" if key == "kaon" else "npi"] = int(sig.sum())
    diag = np.logspace(-5, 0, 200)
    ax.plot(diag, diag, "k:", lw=1.5)
    hs, _ = ax.get_legend_handles_labels()
    hs += [Line2D([], [], color="k", lw=2, ls="-", label="kaon fakes"),
           Line2D([], [], color="k", lw=2, ls="--", label="pion fakes"),
           Line2D([], [], color="k", lw=1.5, ls=":", label="no discrimination")]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-4, 1)
    ax.set_ylim(3e-3, 1.2)
    ax.set_xlabel("fraction of real prompt muons rejected")
    ax.set_ylabel("fraction of decay-in-flight fakes rejected")
    ax.legend(handles=hs, fontsize="x-small", loc="upper left",
              title=f"muon ID: {muid}", title_fontsize="x-small")
    plot_tools.add_decor(ax, "CMS", "Work in progress", data=False,
                         lumi=None, loc=0, no_energy=True)
    save(f"fakemuon_roc_{muid}")

    # ---- the discriminating variable itself ------------------------------
    fig, ax = plt.subplots(figsize=(8, 7))
    edges = np.linspace(-1., 1., 81)
    ax.hist(np.clip(mu["kink_dpp"][mureal], edges[0], edges[-1]), bins=edges,
            histtype="step", density=True, color="#3f90da", lw=2,
            label=f"real prompt muons ({int(mureal.sum())})")
    for key, color in (("kaon", "#f89c20"), ("pi", "#e42536")):
        if key not in species:
            continue
        d = species[key]
        sig = d["has_decay"] & isamuon(d, muid)
        if sig.sum() < 10:
            continue
        ax.hist(np.clip(d["kink_dpp"][sig], edges[0], edges[-1]), bins=edges,
                histtype="step", density=True, color=color, lw=2,
                label=f"{LABEL[key]} decay fakes ({int(sig.sum())})")
    ax.axvline(0., color="grey", ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\hat{\delta}(q/p)\,/\,(q/p)$ at the candidate step")
    ax.set_ylabel("normalised tracks")
    ax.legend(fontsize="small")
    plot_tools.add_decor(ax, "CMS", "Work in progress", data=False,
                         lumi=None, loc=0, no_energy=True)
    save(f"fakemuon_dpp_{muid}")


if __name__ == "__main__":
    main()
