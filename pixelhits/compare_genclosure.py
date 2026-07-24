#!/usr/bin/env python
"""Compare CVH gen-closure refits of the B->J/psi+X MC with the legacy
pixel hit-quality veto (baseline) vs the vetoed hits re-included
(keepPixelEdgeHits=True pixelMinSizeX=1), for the pixel edge /
single-column hit study.

Inputs are pairs of run directories produced by runCvhJpsiGenMC.py
(fitFromGenParms=False free fits for the bias plots; the gen-frozen runs
are only used for chi2 comparisons).
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

COLS = [
    "Jpsi_mass", "Jpsigen_mass", "Jpsikin_mass", "Jpsi_eta",
    "Muplus_pt", "Muplus_eta", "Muplusgen_pt",
    "Muminus_pt", "Muminus_eta", "Muminusgen_pt",
    "Muplus_nvalidpixel", "Muminus_nvalidpixel",
    "chisqval", "ndof", "run", "lumi", "event",
]
# 16-bin pathology-class counts of hits used in the fit (may be absent in
# older outputs; loaded separately and tolerated if missing).
PIXCLASS_COLS = ["Muplus_pixClass", "Muminus_pixClass",
                 "Muplus_npixDemoted", "Muminus_npixDemoted"]


def load(rundir):
    files = sorted(glob.glob(rundir + "/globalcor_*.root"))
    assert files, f"no globalcor_*.root in {rundir}"
    trees = [uproot.open(f)["tree"] for f in files]
    parts = [t.arrays(COLS, library="np") for t in trees]
    out = {c: np.concatenate([p[c] for p in parts]) for c in COLS}
    if all(all(c in t.keys() for c in PIXCLASS_COLS) for t in trees):
        for c in PIXCLASS_COLS[:2]:
            out[c] = np.vstack([np.array([list(x) for x in t.arrays([c], library="np")[c]])
                                for t in trees])
        for c in PIXCLASS_COLS[2:]:
            out[c] = np.concatenate([t.arrays([c], library="np")[c] for t in trees])
    return out


# Non-exclusive per-bit counts from the 16-bin class vector:
# bit0=edgeX bit1=edgeY bit2=sizeX1 bit3=sizeY1; bin 0 = clean.
def class_counts(pixclass):
    bins = np.arange(16)
    return {
        "edgeX":  pixclass[:, (bins & 1) > 0].sum(axis=1),
        "edgeY":  pixclass[:, (bins & 2) > 0].sum(axis=1),
        "sizeX1": pixclass[:, (bins & 4) > 0].sum(axis=1),
        "sizeY1": pixclass[:, (bins & 8) > 0].sum(axis=1),
        "singlepix": pixclass[:, (bins & 4) * (bins & 8) > 0].sum(axis=1),
        "patho":  pixclass[:, 1:].sum(axis=1),
    }


def event_key(a):
    # The condor MC re-uses (run=1, lumi=1, event) numbers across jobs, so
    # the EDM event id alone is NOT unique in the merged sample; the gen
    # kinematics of the matched muons disambiguate physically distinct
    # events that share an id.
    return np.stack([a["run"], a["lumi"], a["event"],
                     np.round(a["Jpsigen_mass"] * 1e6),
                     np.round(a["Muplusgen_pt"] * 1e4),
                     np.round(a["Muminusgen_pt"] * 1e4)], axis=1)


def match(a, b):
    """Match candidates between the two runs on (run,lumi,event,genmass)."""
    ka = [tuple(r) for r in event_key(a)]
    kb = [tuple(r) for r in event_key(b)]
    common = set(ka) & set(kb)
    ia = np.array([i for i, k in enumerate(ka) if k in common])
    ib_map = {k: i for i, k in enumerate(kb)}
    ib = np.array([ib_map[ka[i]] for i in ia])
    return ia, ib


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, help="run dir with legacy veto")
    p.add_argument("--keephits", required=True, help="run dir with hits re-included")
    date_tag = datetime.date.today().strftime("%y%m%d")
    p.add_argument("-o", "--outpath",
                   default=f"/home/submit/david_w/public_html/ZMass/{date_tag}_pixelhits_genclosure")
    args = p.parse_args()

    outdir = output_tools.make_plot_dir(args.outpath, "")
    base = load(args.baseline)
    keep = load(args.keephits)
    ia, ib = match(base, keep)
    print(f"baseline {len(base['Jpsi_mass'])}  keephits {len(keep['Jpsi_mass'])}  matched {len(ia)}")

    # --- J/psi mass spectra --------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 9))
    bins = np.linspace(2.95, 3.25, 61)
    for arr, sel, label, color in ((base, ia, "legacy veto", "black"),
                                   (keep, ib, "hits kept, no corrections", "red")):
        m = arr["Jpsi_mass"][sel]
        ax.hist(m, bins=bins, histtype="step", color=color,
                label=f"{label}\n$\\mu$={m.mean():.4f}, $\\sigma$={m.std():.4f}")
    ax.axvline(base["Jpsigen_mass"][ia].mean(), color="gray", ls="--", label="gen mass")
    ax.set_xlabel(r"$m_{\mu\mu}$ (GeV)")
    ax.set_ylabel("candidates / 5 MeV")
    ax.legend(fontsize="small")
    plot_tools.add_decor(ax, "CMS", "Simulation, private work", data=False, lumi=None, no_energy=True)
    plot_tools.save_pdf_and_png(outdir, "jpsi_mass", fig)
    plt.close(fig)

    # --- per-candidate mass shift -------------------------------------------
    dm = keep["Jpsi_mass"][ib] - base["Jpsi_mass"][ia]
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.hist(dm * 1e3, bins=np.linspace(-40, 40, 81), histtype="step", color="black")
    ax.set_xlabel(r"$\Delta m_{\mu\mu}$ (keep $-$ veto) (MeV)")
    ax.set_ylabel("candidates / MeV")
    ax.text(0.05, 0.95, f"mean = {dm.mean()*1e3:.2f} MeV\nRMS = {dm.std()*1e3:.2f} MeV\nfrac($\\Delta m \\neq 0$) = {(np.abs(dm)>1e-6).mean():.2f}",
            transform=ax.transAxes, va="top", fontsize="small")
    plot_tools.add_decor(ax, "CMS", "Simulation, private work", data=False, lumi=None, no_energy=True)
    plot_tools.save_pdf_and_png(outdir, "jpsi_mass_shift", fig)
    plt.close(fig)

    # --- muon curvature bias vs eta -----------------------------------------
    fig, ax = plt.subplots(figsize=(11, 9))
    etabins = np.linspace(-2.4, 2.4, 13)
    for arr, sel, label, color in ((base, ia, "legacy veto", "black"),
                                   (keep, ib, "hits kept, no corrections", "red")):
        eta = np.concatenate([arr["Muplus_eta"][sel], arr["Muminus_eta"][sel]])
        kres = np.concatenate([arr["Muplusgen_pt"][sel] / arr["Muplus_pt"][sel],
                               arr["Muminusgen_pt"][sel] / arr["Muminus_pt"][sel]]) - 1.
        means, errs, centers = [], [], []
        for lo, hi in zip(etabins[:-1], etabins[1:]):
            m = (eta >= lo) & (eta < hi)
            if m.sum() < 5:
                continue
            means.append(np.mean(kres[m]))
            errs.append(np.std(kres[m]) / np.sqrt(m.sum()))
            centers.append(0.5 * (lo + hi))
        ax.errorbar(centers, np.array(means) * 1e3, yerr=np.array(errs) * 1e3,
                    fmt="o", color=color, label=label)
    ax.axhline(0, color="gray", ls="--")
    ax.set_xlabel(r"$\eta^{\mu}$")
    ax.set_ylabel(r"$\langle k_{reco}/k_{gen} - 1 \rangle \times 10^{3}$")
    ax.legend(fontsize="small")
    plot_tools.add_decor(ax, "CMS", "Simulation, private work", data=False, lumi=None, no_energy=True)
    plot_tools.save_pdf_and_png(outdir, "curvature_bias_vs_eta", fig)
    plt.close(fig)

    # --- chi2/ndof ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 9))
    bins = np.logspace(-1, 4, 60)
    for arr, sel, label, color in ((base, ia, "legacy veto", "black"),
                                   (keep, ib, "hits kept, no corrections", "red")):
        c2 = arr["chisqval"][sel] / arr["ndof"][sel]
        ax.hist(c2, bins=bins, histtype="step", color=color,
                label=f"{label} (mean {c2.mean():.1f})")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\chi^{2}$/ndof")
    ax.set_ylabel("candidates")
    ax.legend(fontsize="small")
    plot_tools.add_decor(ax, "CMS", "Simulation, private work", data=False, lumi=None, no_energy=True)
    plot_tools.save_pdf_and_png(outdir, "chisq_ndof", fig)
    plt.close(fig)

    # --- per-class attribution (needs the pixClass branches in keephits) ----
    if "Muplus_pixClass" in keep:
        # Per-muon curvature shift between the two fits of the SAME track
        # (resolution cancels): dk/k = pt_base/pt_keep - 1, tagged by the
        # pathology content of the hits admitted in the keephits fit.
        dk = np.concatenate([base["Muplus_pt"][ia] / keep["Muplus_pt"][ib],
                             base["Muminus_pt"][ia] / keep["Muminus_pt"][ib]]) - 1.
        kres_keep = np.concatenate([keep["Muplusgen_pt"][ib] / keep["Muplus_pt"][ib],
                                    keep["Muminusgen_pt"][ib] / keep["Muminus_pt"][ib]]) - 1.
        kres_base = np.concatenate([base["Muplusgen_pt"][ia] / base["Muplus_pt"][ia],
                                    base["Muminusgen_pt"][ia] / base["Muminus_pt"][ia]]) - 1.
        cc = {k: np.concatenate([v1, v2]) for (k, v1), (_, v2) in
              zip(class_counts(keep["Muplus_pixClass"][ib]).items(),
                  class_counts(keep["Muminus_pixClass"][ib]).items())}

        # Exclusive-ish categories for attribution (a muon can carry several
        # pathological hits; categories tag "has >=1 hit of this class").
        cats = [("clean", cc["patho"] == 0, "black"),
                ("edgeX", cc["edgeX"] > 0, "tab:red"),
                ("edgeY", cc["edgeY"] > 0, "tab:orange"),
                ("sizeX1", cc["sizeX1"] > 0, "tab:blue"),
                ("sizeY1", cc["sizeY1"] > 0, "tab:green"),
                ("single pixel", cc["singlepix"] > 0, "tab:purple")]

        print("\nper-muon curvature shift keep-vs-base and gen bias by class:")
        fig, ax = plt.subplots(figsize=(12, 9))
        labels, means, errs, gmeans, gerrs = [], [], [], [], []
        for label, sel, _c in cats:
            n = sel.sum()
            if n < 3:
                continue
            m, e = dk[sel].mean(), dk[sel].std() / np.sqrt(n)
            gm = kres_keep[sel].mean() - kres_base[sel].mean()
            ge = (kres_keep[sel] - kres_base[sel]).std() / np.sqrt(n)
            print(f"  {label:12s} n={n:6d}  <dk/k>={m:+.2e} +- {e:.2e}"
                  f"   <bias_keep - bias_base>={gm:+.2e} +- {ge:.2e}")
            labels.append(f"{label}\n(n={n})")
            means.append(m * 1e3); errs.append(e * 1e3)
        x = np.arange(len(labels))
        ax.errorbar(x, means, yerr=errs, fmt="o", color="black")
        ax.axhline(0, color="gray", ls="--")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize="small")
        ax.set_ylabel(r"$\langle \Delta k/k \rangle$ (keep $-$ veto) $\times 10^{3}$")
        plot_tools.add_decor(ax, "CMS", "Simulation, private work", data=False, lumi=None, no_energy=True)
        plot_tools.save_pdf_and_png(outdir, "curvature_shift_by_class", fig)
        plt.close(fig)

        # Distribution of per-muon curvature shifts per category.
        fig, ax = plt.subplots(figsize=(12, 9))
        bins = np.linspace(-10, 10, 101)
        for label, sel, color in cats:
            if sel.sum() < 3:
                continue
            ax.hist(np.clip(dk[sel] * 1e3, bins[0], bins[-1]), bins=bins,
                    histtype="step", color=color, density=True,
                    label=f"{label} (n={sel.sum()}, RMS {dk[sel].std()*1e3:.2f})")
        ax.set_xlabel(r"$\Delta k/k$ (keep $-$ veto) $\times 10^{3}$")
        ax.set_ylabel("muons (normalised)")
        ax.set_yscale("log")
        ax.legend(fontsize="x-small")
        plot_tools.add_decor(ax, "CMS", "Simulation, private work", data=False, lumi=None, no_energy=True)
        plot_tools.save_pdf_and_png(outdir, "curvature_shift_dist_by_class", fig)
        plt.close(fig)

    output_tools.write_index_and_log(outdir, "compare_genclosure", args=args)
    print("plots in", outdir)


if __name__ == "__main__":
    main()
