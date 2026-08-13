"""Gen-truth validation of the CVH kink finder on the v3 B -> J/psi + X MC.

The v3 campaign is the first that keeps the Geant4 SimTracks + SimVertices, so
every gen-matched track can be labelled by what actually happened to it inside
the tracker: decay in flight, nuclear interaction, or nothing. That turns the
kink-finder studies from rate comparisons into an efficiency/purity measurement.

Output is organised as one complete set per species (kaon, pion) plus a few
genuinely cross-species comparisons. Both decays and nuclear interactions are
treated as taggable signal -- both distort the curvature, and both are things a
veto is allowed to remove -- so every ROC shows decay-vs-clean (solid) and
nuclear-vs-clean (dashed) together, with a third ROC against their union.

usage:
  python plot_kink_roc.py --kaons '<dir>/kaon/chunk_*/globalcor_jpsix_kaon_0.root' \
                          --pions '<dir>/pi/chunk_*/...' --muons '<dir>/mu/chunk_*/...'
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
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kink_truth as kt

from wums import logging, output_tools, plot_tools

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

COLORS = {"kaon": "#f89c20", "pi": "#e42536", "mu": "black"}
LABEL = {"kaon": "kaon", "pi": "pion", "mu": "muon"}
PDG = {"kaon": 321, "pi": 211}
M_MU = 0.10566

# truth classes: key, legend label, colour
CLASSES = (("clean", "clean", "#3f90da"),
           ("has_nuclear", "nuclear interaction", "#f89c20"),
           ("has_decay", "decay in flight", "#e42536"))

# discriminants: key, legend label, colour
DISCR = [("D3", r"$\max_i\,\Delta\chi^2_i$ (3 dof)", "#3f90da"),
         ("Dangle", r"angle only ($\Delta\lambda,\Delta\phi$)", "#e42536"),
         ("Dqop", r"$\Delta(q/p)$ only", "#f89c20"),
         ("chi2", r"track $\chi^2/\mathrm{ndof}$", "grey")]
# uniformly worse than the others; kept in the tables, dropped from the plots
DISCR_EXTRA = [("Dloss", r"3 dof, momentum-loss steps only", "#92dadd")]

# Delta-chi2 at the candidate step above which deltahat is not noise-dominated
TAGCUT = 20.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--kaons", nargs="+", required=True)
    p.add_argument("--pions", nargs="+", default=[])
    p.add_argument("--muons", nargs="+", default=[])
    p.add_argument("--outpath", default=None)
    p.add_argument("--postfix", default="")
    p.add_argument("--fpr", type=float, default=0.01,
                   help="mis-tag rate on truth-clean tracks defining the "
                        "working point of the efficiency plots")
    p.add_argument("--minHits", type=int, default=8)
    p.add_argument("--noArbitration", action="store_true",
                   help="skip the cross-species gen-match arbitration "
                        "(shows the mismatch-diluted result)")
    p.add_argument("--biasClip", type=float, default=1.0,
                   help="|p_gen/p_fit - 1| clip for the curvature-bias means")
    return p.parse_args()


def expand(patterns):
    out = []
    for pat in patterns:
        out += sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat]
    return out


def _apply(d, arr, keep):
    n0 = d["n"]
    out = {k: (v[keep] if isinstance(v, np.ndarray) and getattr(v, "shape", (0,))[:1] == (n0,) else v)
           for k, v in d.items()}
    out["n"] = int(keep.sum())
    out["arr"] = arr[keep]
    return out


def load_species(patterns, minhits):
    files = expand(patterns)
    if not files:
        return None
    logger.info(f"loading {len(files)} files")
    arr = kt.load(files)
    d = kt.label(arr)
    keep = d["sim_found"] & (d["nvalid"] >= minhits) & (d["nsteps"] > 2)
    logger.info(f"  {d['n']} tracks -> {int(keep.sum())} after sim-match and quality cuts")
    return _apply(d, arr, keep)


def decor(ax):
    plot_tools.add_decor(ax, "CMS", "Work in progress", data=False,
                         lumi=None, loc=0, no_energy=True)


def main():
    args = parse_args()
    today = datetime.date.today().strftime("%y%m%d")
    outpath = args.outpath or os.path.expanduser(
        f"~/public_html/ZMass/{today}_kink_truth/")
    outdir = output_tools.make_plot_dir(outpath)
    logger.info(f"Writing plots to {outdir}")
    wd = os.path.dirname(os.path.abspath(__file__))

    def save(name):
        name = "_".join(filter(None, [name, args.postfix]))
        plot_tools.save_pdf_and_png(outdir, name)
        output_tools.write_logfile(outdir, name, args=args, wd=wd)
        plt.close("all")

    species, raw = {}, {}
    for key, pats in (("kaon", args.kaons), ("pi", args.pions), ("mu", args.muons)):
        d = load_species(pats, args.minHits)
        if d is not None:
            species[key] = d

    # ---- cross-species gen-match arbitration -----------------------------
    own = None
    if not args.noArbitration and len(species) > 1:
        own = kt.arbitrate(species)
        raw = dict(species)
        print("\n=== cross-species gen-match arbitration ===")
        for key in list(species):
            d, m = species[key], own[key]
            print(f"  {key:5s}: {d['n']:8d} -> {int(m.sum()):8d} kept "
                  f"({m.mean()*100:.1f}%); rejected tracks have a closer gen "
                  f"candidate of another species")
            species[key] = _apply(d, d["arr"], m)

    # ---- composition ------------------------------------------------------
    print("\n=== truth composition (in-span vertices) ===")
    print(f"{'species':8s} {'tracks':>8s} {'decay':>15s} {'nuclear':>15s} {'clean':>15s}")
    for key, d in species.items():
        print(f"{key:8s} {d['n']:8d} "
              f"{d['has_decay'].sum():7d} ({d['has_decay'].mean()*100:5.2f}%) "
              f"{d['has_nuclear'].sum():7d} ({d['has_nuclear'].mean()*100:5.2f}%) "
              f"{d['clean'].sum():7d} ({d['clean'].mean()*100:5.2f}%)")

    # ======================================================================
    #  cross-species plots
    # ======================================================================

    # per-step Delta-chi2 of truth-clean tracks against chi2(3)
    fig, ax = plt.subplots(figsize=(8, 7))
    edges = np.linspace(0, 30, 61)
    cent = 0.5 * (edges[1:] + edges[:-1])
    for key, d in species.items():
        dch = ak.to_numpy(ak.flatten(d["arr"].kinkDchisq[d["clean"]]))
        h, _ = np.histogram(dch, bins=edges, density=True)
        ax.step(cent, h, where="mid", color=COLORS[key], lw=2,
                label=f"truth-clean {LABEL[key]} ({len(dch)} steps)")
    ax.plot(cent, stats.chi2.pdf(cent, 3), "k--", lw=2, label=r"$\chi^2(3)$")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1.0)
    ax.set_xlabel(r"per-step $\Delta\chi^2_i$")
    ax.set_ylabel("normalised steps")
    ax.legend(fontsize="small")
    decor(ax)
    save("kink_null_perstep")

    # survival of in-tracker decays into the reconstructed sample
    fig, ax = plt.subplots(figsize=(8, 7))
    pbins = np.array([3., 4., 5., 6.5, 8., 10., 14., 20., 30.])
    pcent = 0.5 * (pbins[1:] + pbins[:-1])
    print("\n=== in-span decays: observed vs decay probability along the track ===")
    for key in ("kaon", "pi"):
        if key not in species:
            continue
        d = species[key]
        L = (d["s_last"] - d["s_first"]) / 100.0          # cm -> m
        meas, err, pred = [], [], []
        for lo, hi in zip(pbins[:-1], pbins[1:]):
            m = (d["gen_p"] >= lo) & (d["gen_p"] < hi)
            nsel = int(m.sum())
            if nsel < 50:
                meas.append(np.nan); err.append(np.nan); pred.append(np.nan); continue
            f = d["has_decay"][m].mean()
            meas.append(f)
            err.append(np.sqrt(f * (1 - f) / nsel))
            pred.append(np.mean(kt.decay_prob(d["gen_p"][m], PDG[key], L[m])))
        ax.errorbar(pcent, meas, yerr=err, fmt="o", color=COLORS[key],
                    label=f"{LABEL[key]}, Geant4 truth")
        ax.plot(pcent, pred, "--", color=COLORS[key],
                label=rf"{LABEL[key]}, $1-e^{{-L/\lambda}}$ along the track")
        obs, exp = d["has_decay"].mean(), np.mean(kt.decay_prob(d["gen_p"], PDG[key], L))
        print(f"  {LABEL[key]}: observed {obs*100:.3f}%  expected {exp*100:.3f}%  "
              f"survival {obs/exp:.3f}   <L> {L.mean():.2f} m")
    ax.set_yscale("log")
    ax.set_xlabel(r"$p^{\mathrm{gen}}$ [GeV]")
    ax.set_ylabel("in-span decay fraction")
    ax.legend(fontsize="small")
    decor(ax)
    save("kink_decay_survival")

    # ======================================================================
    #  decay-in-flight hadrons faking a prompt muon
    # ======================================================================
    # Different question from the ROCs below: there the background is clean
    # tracks of the SAME species (how well do we spot a damaged kaon among
    # kaons). Here the background is REAL PROMPT MUONS -- the relevant
    # comparison for the momentum-scale calibration itself, where a hadron
    # that decays in flight and is reconstructed as a muon enters the J/psi
    # or Z sample. The figure of merit is the fake rejection at a *tiny* loss
    # of real muons, not at a 1% mis-tag.
    if "mu" in species and len(species) > 1:
        mu = species["mu"]
        muclean = mu["clean"]
        fig, ax = plt.subplots(figsize=(8, 7))
        print("\n=== decay-in-flight hadrons faking a prompt muon ===")
        print("  background = truth-clean gen muons "
              f"({int(muclean.sum())} tracks); "
              "signal = hadrons with an in-span decay")
        for key, ls in (("kaon", "-"), ("pi", "--")):
            if key not in species:
                continue
            d = species[key]
            for name, lab, color in DISCR[:3]:
                fpr, tpr, auc = kt.roc(
                    np.concatenate([d[name], mu[name]]),
                    np.concatenate([d["has_decay"], np.zeros(mu["n"], bool)]),
                    np.concatenate([np.zeros(d["n"], bool), muclean]))
                if len(fpr):
                    ax.plot(np.clip(fpr, 1e-5, None), np.clip(tpr, 1e-4, None),
                            color=color, ls=ls, lw=2,
                            label=(f"{lab}" if key == "kaon" else None))
            print(f"  {LABEL[key]} decays (n={int(d['has_decay'].sum())}):")
            for name, lab, _ in DISCR[:3]:
                row = []
                for loss in (0.01, 0.005, 0.001):
                    cut = kt.threshold_at_fpr(mu[name], muclean, loss)
                    row.append((d[name][d["has_decay"]] > cut).mean())
                print(f"    {name:8s} rejection at real-muon loss "
                      f"1%: {row[0]:.3f}   0.5%: {row[1]:.3f}   0.1%: {row[2]:.3f}")
        diag = np.logspace(-5, 0, 200)
        ax.plot(diag, diag, "k:", lw=1.5)
        hs, _ = ax.get_legend_handles_labels()
        hs += [Line2D([], [], color="k", lw=2, ls="-", label="kaon decays"),
               Line2D([], [], color="k", lw=2, ls="--", label="pion decays"),
               Line2D([], [], color="k", lw=1.5, ls=":", label="no discrimination")]
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-4, 1)
        ax.set_ylim(3e-3, 1.2)
        ax.set_xlabel("fraction of real prompt muons rejected")
        ax.set_ylabel("fraction of decay-in-flight fakes rejected")
        ax.legend(handles=hs, fontsize="x-small", loc="upper left")
        decor(ax)
        save("kink_roc_fakemuon")

    # ======================================================================
    #  one complete set per species
    # ======================================================================
    for key in ("kaon", "pi"):
        if key not in species:
            continue
        d, sl = species[key], LABEL[key]
        sig = {"decay": d["has_decay"], "nuclear": d["has_nuclear"],
               "hard": d["has_decay"] | d["has_nuclear"]}
        wp = {name: kt.threshold_at_fpr(d[name], d["clean"], args.fpr)
              for name, _, _ in DISCR + DISCR_EXTRA}

        # -- arbitration validation ------------------------------------
        if own is not None and key in raw:
            r, m = raw[key], own[key]
            fig, ax = plt.subplots(figsize=(8, 7))
            grid = np.logspace(np.log10(0.5), np.log10(2000), 120)
            for lab, sel, color, ls in (
                    ("kept, decay-labelled", r["has_decay"] & m, "#e42536", "-"),
                    ("rejected, decay-labelled", r["has_decay"] & ~m, "#e42536", "--"),
                    ("kept, clean", r["clean"] & m, "#3f90da", "-"),
                    ("rejected, clean", r["clean"] & ~m, "#3f90da", "--")):
                if sel.sum() < 20:
                    continue
                s = r["D3"][sel]
                ax.plot(grid, [(s > g).mean() for g in grid], color=color, ls=ls,
                        lw=2, label=f"{lab} ({sel.sum()})")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(1e-4, 1.5)
            ax.set_xlabel(r"$\max_i\,\Delta\chi^2_i$")
            ax.set_ylabel(f"fraction of {sl} tracks above")
            ax.legend(fontsize="x-small")
            decor(ax)
            save(f"kink_arbitration_validation_{key}")

        # -- score variants, survival by truth class ---------------------
        for var, vlab, fname in (
                ("D3", r"$\max_i\,\Delta\chi^2_i$ (3 dof)", "score"),
                ("Dqop", r"$\max_i\,\Delta\chi^2_i(q/p)$", "scoreqop"),
                ("Dangle", r"$\max_i\,\Delta\chi^2_i$(angle)", "scoreangle")):
            fig, ax = plt.subplots(figsize=(8, 7))
            grid = np.logspace(np.log10(0.5), np.log10(2000), 120)
            for cls, clab, color in CLASSES:
                s = d[var][d[cls]]
                if len(s) < 20:
                    continue
                ax.plot(grid, [(s > g).mean() for g in grid], color=color, lw=2,
                        label=f"{clab} ({d[cls].sum()})")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(1e-4, 1.5)
            ax.set_xlabel(vlab)
            ax.set_ylabel(f"fraction of {sl} tracks above")
            ax.legend(fontsize="small")
            decor(ax)
            save(f"kink_{fname}_by_truth_{key}")

        # -- estimated momentum step -------------------------------------
        # Kinematic endpoint of P -> mu nu: for a relativistic parent the
        # daughter's lab momentum fraction runs from (m_mu/m_P)^2 to 1, so the
        # step cannot exceed 1 - (m_mu/m_P)^2. Tight (0.43) for pions, nearly
        # unconstraining (0.95) for kaons, which also have K -> pi pi0.
        endpoint = 1.0 - (M_MU / kt.MASS[PDG[key]])**2
        for sel, seltag, rng, fname in (
                (np.ones(d["n"], dtype=bool), "all tracks", 0.3, "dpp"),
                (d["kink_dchisq"] > TAGCUT,
                 rf"step $\Delta\chi^2 > {TAGCUT:g}$", 1.0, "dpp_tagged")):
            fig, ax = plt.subplots(figsize=(8, 7))
            edges = np.linspace(-rng, rng, 81)
            for cls, clab, color in CLASSES:
                x = d["kink_dpp"][d[cls] & sel]
                x = x[np.isfinite(x)]
                if len(x) < 20:
                    continue
                ax.hist(np.clip(x, edges[0], edges[-1]), bins=edges, histtype="step",
                        density=True, color=color, lw=2, label=f"{clab} ({len(x)})")
            ax.axvline(0., color="grey", ls=":", lw=1)
            if endpoint < rng:
                ax.axvline(endpoint, color="#e42536", ls="--", lw=1.5,
                           label=rf"$P\to\mu\nu$ endpoint ({endpoint:.2f})")
            ax.set_yscale("log")
            ax.set_xlabel(r"$\hat{\delta}(q/p)\,/\,(q/p)$ at the candidate step")
            ax.set_ylabel(f"normalised {sl} tracks")
            ax.legend(fontsize="small", title=seltag, title_fontsize="small")
            decor(ax)
            save(f"kink_{fname}_by_truth_{key}")

        # -- estimated angular kink --------------------------------------
        fig, ax = plt.subplots(figsize=(8, 7))
        grid = np.logspace(-4, 0.5, 120)
        for cls, clab, color in CLASSES:
            x = d["kink_angle"][d[cls]]
            x = x[np.isfinite(x)]
            if len(x) < 20:
                continue
            ax.plot(grid, [(x > g).mean() for g in grid], color=color, lw=2,
                    label=f"{clab} ({len(x)})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.5)
        ax.set_xlabel(r"$|\hat{\delta}(\mathrm{d}x/\mathrm{d}z,\,\mathrm{d}y/\mathrm{d}z)|$"
                      " at the candidate step")
        ax.set_ylabel(f"fraction of {sl} tracks above")
        ax.legend(fontsize="small")
        decor(ax)
        save(f"kink_angle_by_truth_{key}")

        # -- ROC: decay (solid) and nuclear (dashed) together ------------
        # Log-log so the tpr = fpr reference is a straight line; it must be
        # drawn from a dense grid, since a two-point plot([lo,hi],[lo,hi]) is
        # rendered straight in DISPLAY space and is not tpr = fpr on a log axis.
        diag = np.logspace(-4, 0, 200)
        fig, ax = plt.subplots(figsize=(8, 7))
        handles = []
        for name, lab, color in DISCR:
            aucs = []
            for cls, ls in (("decay", "-"), ("nuclear", "--")):
                fpr, tpr, auc = kt.roc(d[name], sig[cls], d["clean"])
                aucs.append(auc)
                if len(fpr):
                    ax.plot(np.clip(fpr, 1e-4, None), np.clip(tpr, 1e-4, None),
                            color=color, ls=ls, lw=2)
            handles.append(Line2D([], [], color=color, lw=2,
                                  label=f"{lab}: {aucs[0]:.3f} / {aucs[1]:.3f}"))
        ax.plot(diag, diag, "k:", lw=1.5)
        handles += [Line2D([], [], color="k", lw=2, ls="-", label="decay in flight"),
                    Line2D([], [], color="k", lw=2, ls="--", label="nuclear interaction"),
                    Line2D([], [], color="k", lw=1.5, ls=":", label="no discrimination")]
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-3, 1)
        ax.set_ylim(3e-3, 1.2)
        ax.set_xlabel(f"mis-tag rate on truth-clean {sl} tracks")
        ax.set_ylabel("tag efficiency")
        ax.legend(handles=handles, fontsize="x-small", loc="upper left",
                  title="AUC decay / nuclear", title_fontsize="x-small")
        decor(ax)
        save(f"kink_roc_{key}")

        # -- ROC against the union: what a veto actually removes ---------
        fig, ax = plt.subplots(figsize=(8, 7))
        for name, lab, color in DISCR:
            fpr, tpr, auc = kt.roc(d[name], sig["hard"], d["clean"])
            if len(fpr):
                ax.plot(np.clip(fpr, 1e-4, None), np.clip(tpr, 1e-4, None),
                        color=color, lw=2, label=f"{lab}, AUC {auc:.3f}")
        ax.plot(diag, diag, "k:", lw=1.5, label="no discrimination")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-3, 1)
        ax.set_ylim(3e-3, 1.2)
        ax.set_xlabel(f"mis-tag rate on truth-clean {sl} tracks")
        ax.set_ylabel("tag efficiency for decay OR nuclear interaction")
        ax.legend(fontsize="x-small", loc="upper left")
        decor(ax)
        save(f"kink_roc_hard_{key}")

        # -- efficiency vs truth kinematics ------------------------------
        def eff_profile(ax, x, mask, score, cut, bins, color, ls, label):
            e, ee, c = [], [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                m = mask & (x >= lo) & (x < hi)
                n = int(m.sum())
                if n < 5:
                    e.append(np.nan); ee.append(np.nan); c.append(0.5*(lo+hi)); continue
                p = (score[m] > cut).mean()
                e.append(p); ee.append(np.sqrt(p*(1-p)/n)); c.append(0.5*(lo+hi))
            ax.errorbar(c, e, yerr=ee, fmt="o", ls=ls, color=color, label=label, ms=5)

        for xvar, bins, xlabel, fname in (
                ("vtx_r", np.array([0., 10., 20., 30., 45., 60., 80., 110.]),
                 "true vertex radius [cm]", "eff_vs_radius"),
                ("zfrac", np.linspace(0., 1., 9),
                 r"$z = p_{\mathrm{daughter}}/p_{\mathrm{parent}}$", "eff_vs_zfrac")):
            fig, ax = plt.subplots(figsize=(8, 7))
            for name, lab, color in DISCR[:3]:
                for cls, ls in (("decay", "-"), ("nuclear", "--")):
                    eff_profile(ax, d[xvar], sig[cls], d[name], wp[name], bins,
                                color, ls, lab if cls == "decay" else None)
            hs, ls_ = ax.get_legend_handles_labels()
            hs += [Line2D([], [], color="k", lw=2, ls="-", label="decay"),
                   Line2D([], [], color="k", lw=2, ls="--", label="nuclear")]
            ax.set_xlabel(xlabel)
            ax.set_ylabel(f"tag efficiency at {args.fpr*100:g}% mis-tag")
            ax.set_ylim(0, 1.05)
            ax.legend(handles=hs, fontsize="x-small", ncol=2)
            decor(ax)
            save(f"kink_{fname}_{key}")

        # -- localisation -------------------------------------------------
        m = sig["hard"] & np.isfinite(d["vtx_r"]) & np.isfinite(d["kink_r"]) \
            & (d["D3"] > wp["D3"])
        fig, ax = plt.subplots(figsize=(8, 7))
        if m.sum() > 10:
            ax.hist2d(d["vtx_r"][m], d["kink_r"][m],
                      bins=[np.linspace(0, 110, 23)]*2, cmap="Blues")
            ax.plot([0, 110], [0, 110], "r--", lw=2)
            res = d["kink_r"][m] - d["vtx_r"][m]
            print(f"\nlocalisation, tagged {sl} decays+nuclear: median "
                  f"{np.median(res):+.1f} cm, IQR "
                  f"{np.subtract(*np.percentile(res, [75, 25])):.1f} cm ({m.sum()} tracks)")
        ax.set_xlabel("true vertex radius [cm]")
        ax.set_ylabel("tagged step radius [cm]")
        decor(ax)
        save(f"kink_localisation_{key}")

        # -- curvature bias ------------------------------------------------
        b = d["dqop_rel"]
        fig, ax = plt.subplots(figsize=(8, 7))
        edges = np.linspace(-0.6, 0.6, 81)
        for cls, clab, color in CLASSES:
            ax.hist(np.clip(b[d[cls]], edges[0], edges[-1]), bins=edges,
                    histtype="step", density=True, color=color, lw=2,
                    label=f"{clab} ({d[cls].sum()})")
        ax.set_yscale("log")
        ax.set_xlabel(r"$q/p$ bias  $(q/p)^{\mathrm{fit}}/(q/p)^{\mathrm{gen}} - 1$")
        ax.set_ylabel(f"normalised {sl} tracks")
        ax.legend(fontsize="small")
        decor(ax)
        save(f"kink_curvature_bias_{key}")

        # ================= printed tables for this species ==============
        print(f"\n=== {sl}s: AUC (background = truth-clean) ===")
        for name, lab, _ in DISCR + DISCR_EXTRA:
            a = [kt.roc(d[name], sig[c], d["clean"])[2]
                 for c in ("decay", "nuclear", "hard")]
            print(f"  {name:8s} decay {a[0]:.4f}  nuclear {a[1]:.4f}  "
                  f"either {a[2]:.4f}   ({lab})")
        print(f"  efficiency at {args.fpr*100:g}% mis-tag:")
        for name, _, _ in DISCR + DISCR_EXTRA:
            e = [(d[name][sig[c]] > wp[name]).mean() for c in ("decay", "nuclear", "hard")]
            print(f"    {name:8s} cut {wp[name]:8.2f}  decay {e[0]:.3f}  "
                  f"nuclear {e[1]:.3f}  either {e[2]:.3f}")

        print(f"\n=== {sl}s: momentum step at the candidate kink "
              f"(step Delta-chi2 > {TAGCUT:g}) ===")
        tagged = d["kink_dchisq"] > TAGCUT
        for cls, clab, _ in CLASSES:
            m = d[cls] & tagged & np.isfinite(d["kink_dpp"])
            if m.sum() < 10:
                continue
            x = d["kink_dpp"][m]
            print(f"  {clab:20s} n={m.sum():6d}  median {np.median(x):+.4f}  "
                  f"fraction losing momentum {np.mean(x > 0):.3f}")

        clip, corew = args.biasClip, 0.02
        good = np.isfinite(b) & (np.abs(b) < clip)
        print(f"\n=== {sl}s: curvature bias (clip {clip}; core |b| < {corew}) ===")

        def summarise(lab, m):
            if not m.sum():
                return
            mc = m & (np.abs(b) < corew)
            print(f"  {lab:18s} n={m.sum():7d}  mean {b[m].mean():+.5f} "
                  f"+- {b[m].std()/np.sqrt(m.sum()):.5f}  median {np.median(b[m]):+.5f}  "
                  f"core mean {b[mc].mean():+.6f}  out-of-core {1 - mc.sum()/m.sum():.4f}")

        for lab, mask in (("clean", d["clean"]), ("nuclear", d["has_nuclear"]),
                          ("decay", d["has_decay"]), ("all", np.ones_like(d["clean"]))):
            summarise(lab, mask & good)
        veto = d["D3"] <= wp["D3"]
        summarise("after kink veto", veto & good)
        m_all, m_veto, m_clean = good, veto & good, d["clean"] & good
        cm = lambda m: b[m & (np.abs(b) < corew)].mean()
        print(f"  contamination shift (all - clean): mean "
              f"{b[m_all].mean() - b[m_clean].mean():+.5f}, core {cm(m_all) - cm(m_clean):+.6f}")
        print(f"  veto effect (veto - all):          mean "
              f"{b[m_veto].mean() - b[m_all].mean():+.5f}, core {cm(m_veto) - cm(m_all):+.6f}")


if __name__ == "__main__":
    main()
