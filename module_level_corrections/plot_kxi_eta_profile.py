"""
plot_kxi_eta_profile.py — kξ (energy-loss) module-level corrections vs |η|
per sub-detector, from a Stage-2b correctionResults_*.root file.

The Stage-2b output stores one float `x` per fitted (reduced) global parameter
in TTree `parmtree`. Mapping each entry back to (parmtype, subdet, layer, eta,
phi, ...) requires the matching producer `runtree` from a globalcor_*.root
ntuple of the same era / code version. Filtering parmtype==7 selects the
energy-loss kξ corrections; ΔE = exp(kξ) · (dE/dx)₀ · Δs, so kξ=0 means no
correction relative to the simulation material model.

Usage
-----
  python plot_kxi_eta_profile.py \
      --corrections correctionResults_v721_recjpsidata.root \
      --runtree    globalcor_data_0_X.root \
      --era        2016postVFP
"""

import argparse
import datetime
import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot
from wums import plot_tools  # noqa: F401  (kept for project-wide style consistency)

hep.style.use(hep.style.ROOT)


PARMTYPE_KXI = 7

SUBDET_NAMES = {
    1: "BPIX",
    2: "FPIX",
    3: "TIB",
    4: "TID",
    5: "TOB",
    6: "TEC",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corrections", required=True,
                   help="correctionResults_*.root (Stage-2b output, has parmtree + idxmaptree)")
    p.add_argument("--runtree", required=True,
                   help="globalcor_*.root with a runtree branch (same era as --corrections)")
    p.add_argument("--era", default="2016postVFP",
                   help="Era label for plot title / filenames")
    p.add_argument("--outdir", default=None,
                   help="Override output directory (default: <yymmdd>_kxi_eta_<era>)")
    p.add_argument("--eta-bin-width", type=float, default=0.1,
                   help="|η| bin width for the profile (default 0.1, matches reduction)")
    return p.parse_args()


def load_runtree(path):
    with uproot.open(path) as f:
        if "runtree" not in f:
            raise RuntimeError(f"{path} has no 'runtree' (was it produced by ResidualGlobalCorrectionMaker*?)")
        rt = f["runtree"]
        return rt.arrays(["parmtype", "subdet", "layer", "eta"], library="np")


def load_correction_parms(path):
    """Return (parm_x, parm_err, idxmap) where idxmap[i] is the parmtree row for runtree row i."""
    with uproot.open(path) as f:
        if "parmtree" not in f or "idxmaptree" not in f:
            raise RuntimeError(f"{path} missing parmtree/idxmaptree (not a Stage-2b output?)")
        parm_x = f["parmtree"]["x"].array(library="np")
        parm_err = f["parmtree"]["err"].array(library="np")
        idxmap = f["idxmaptree"]["idx"].array(library="np")
    return parm_x, parm_err, idxmap


def profile_eta(abseta, values, eta_bins):
    """Mean and stderr-of-mean of `values` in bins of `abseta`."""
    n, _ = np.histogram(abseta, bins=eta_bins)
    s, _ = np.histogram(abseta, bins=eta_bins, weights=values)
    s2, _ = np.histogram(abseta, bins=eta_bins, weights=values**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s / n, np.nan)
        var = np.where(n > 1, (s2 / n - mean**2) * n / (n - 1), 0.0)
        sem = np.where(n > 1, np.sqrt(var / n), 0.0)
    return mean, sem, n


def main():
    args = parse_args()

    runtree = load_runtree(args.runtree)
    parm_x, parm_err, idxmap = load_correction_parms(args.corrections)

    n_run = len(runtree["parmtype"])
    if len(idxmap) != n_run:
        raise RuntimeError(
            f"idxmaptree size {len(idxmap)} != runtree size {n_run} — "
            f"runtree and corrections file are not from the same production"
        )

    mask = runtree["parmtype"] == PARMTYPE_KXI
    n_kxi = int(mask.sum())
    if n_kxi == 0:
        raise RuntimeError("No parmtype==7 (kξ) entries in runtree")
    print(f"[plot_kxi] {n_kxi} kξ entries in runtree across {n_run} total parameters")

    eta = runtree["eta"][mask]
    subdet = runtree["subdet"][mask]
    layer = np.abs(runtree["layer"][mask])
    parm_idx = idxmap[mask]
    kxi = parm_x[parm_idx]

    abseta = np.abs(eta)

    date_prefix = datetime.date.today().strftime("%y%m%d")
    outdir = args.outdir or f"{date_prefix}_kxi_eta_{args.era}"
    os.makedirs(outdir, exist_ok=True)

    eta_bins = np.arange(0.0, 2.6 + args.eta_bin_width / 2, args.eta_bin_width)
    eta_centers = 0.5 * (eta_bins[:-1] + eta_bins[1:])

    for sd_id, sd_name in SUBDET_NAMES.items():
        sd_mask = subdet == sd_id
        if not sd_mask.any():
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        layers_in_sd = np.unique(layer[sd_mask])
        cmap = plt.get_cmap("viridis", max(len(layers_in_sd), 2))

        for i_lay, lay in enumerate(layers_in_sd):
            l_mask = sd_mask & (layer == lay)
            mean, sem, n = profile_eta(abseta[l_mask], kxi[l_mask], eta_bins)
            ok = n > 0
            ax.errorbar(eta_centers[ok], mean[ok], yerr=sem[ok],
                        marker="o", linestyle="-", linewidth=1.2,
                        markersize=3, color=cmap(i_lay),
                        label=f"layer {int(lay)} (n={int(n.sum())})")

        ax.axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
        ax.set_xlabel(r"$|\eta|$")
        ax.set_ylabel(r"$k_{\xi}$")
        ax.set_title(f"{sd_name} — energy-loss correction, {args.era}")
        ax.legend(loc="best", fontsize=9, frameon=False)
        ax.grid(alpha=0.3)

        for ext in ("pdf", "png"):
            fig.savefig(f"{outdir}/kxi_eta_{sd_name}.{ext}", bbox_inches="tight")
        plt.close(fig)
        print(f"[plot_kxi] wrote {outdir}/kxi_eta_{sd_name}.{{pdf,png}}")

    print(f"[plot_kxi] done — {outdir}")


if __name__ == "__main__":
    main()
