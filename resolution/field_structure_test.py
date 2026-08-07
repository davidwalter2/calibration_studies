"""Test whether a SIM-vs-refit magnetic-field mismatch leaks into the
gen-matched pull width, by comparing two refits of the SAME simulated
events that differ ONLY in the field model used by the fit.

Background (2026-08-07). The mu-gun SIM propagates through the default
CMSSW field, which inside the tracker is the OAE_1103l_071212 analytic
parametrization: phi-symmetric, no B_phi, quoted accurate to ~1 per mille.
The resolution-closure refit was being run with useOpera3D=True, i.e. the
full 3D TOSCA volumetric grid, which carries B_phi and phi-asymmetry. That
is a genuine field-model difference between the sample and the fit, and it
injects a coherent, kinematics-dependent shift into (fitted - gen) that has
nothing to do with resolution physics.

The signature is a NON-FLAT pull mean <z> versus eta and versus phi. phi is
the sharper handle: OAE has no phi structure at all by construction, so any
phi dependence of <z> is field-model difference almost by definition.

What this reports, per cache and per axis:
  * the binned <z> profile,
  * its variance in excess of the per-bin statistical floor -- this is the
    contamination of the pull VARIANCE, in units where the total is 1,
  * the implied shift in k_ms, given that MS carries ~77% of that variance.

usage:
    python field_structure_test.py --caches OLD=runs/a.npz NEW=runs/b.npz
"""

import argparse
import datetime
import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from wums import logging, output_tools

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

RED, BLUE = "#A31F34", "#1f4e9c"
MS_SHARE = 0.77          # MS fraction of the q/p variance (measured)


def profile(x, z, nb):
    """Binned <z> on equal-count bins, with the per-bin statistical error.

    Returns (centres, means, err_per_bin, coherent_variance). The coherent
    variance subtracts the statistical floor, because a flat profile
    measured with finite statistics still scatters by rms/sqrt(n_bin) and
    would otherwise be misread as structure.
    """
    qs = np.quantile(x, np.linspace(0., 1., nb + 1))
    ctr, mu, er = [], [], []
    for i in range(nb):
        m = (x >= qs[i]) & (x <= qs[i + 1])
        if m.sum() < 2:
            continue
        ctr.append(0.5 * (qs[i] + qs[i + 1]))
        mu.append(z[m].mean())
        er.append(z[m].std() / np.sqrt(m.sum()))
    ctr, mu, er = np.array(ctr), np.array(mu), np.array(er)
    coherent = max(mu.var() - np.mean(er ** 2), 0.)
    return ctr, mu, er, coherent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--caches", nargs="+", required=True,
                   help="LABEL=path.npz (2+); each needs z, eta, phi")
    p.add_argument("--nbins", type=int, default=16)
    p.add_argument("--outpath", default=None)
    args = p.parse_args()
    logging.setup_logger(__file__, 3, False)

    today = datetime.date.today().strftime("%y%m%d")
    outdir = output_tools.make_plot_dir(
        args.outpath or os.path.expanduser(
            f"~/public_html/calibration_studies/{today}_field_structure/"))

    entries = []
    for spec in args.caches:
        label, path = spec.split("=", 1)
        d = np.load(path)
        if "phi" not in d.files:
            raise SystemExit(
                f"{path} has no 'phi' -- re-extract with the current "
                "cf_track_resolution.py, which stores it.")
        entries.append((label, d))
        logger.info(f"{label}: {len(d['z'])} tracks from {path}")

    fig, axs = plt.subplots(1, 2, figsize=(16, 6.5))
    for ax, axis, xlabel in ((axs[0], "eta", r"$\eta$"),
                             (axs[1], "phi", r"$\phi$")):
        for (label, d), col in zip(entries, (BLUE, RED, "0.4")):
            ctr, mu, er, coh = profile(d[axis], d["z"], args.nbins)
            ax.errorbar(ctr, mu, yerr=er, fmt="o-", color=col, markersize=5,
                        label=f"{label}  (coherent {coh*100:.3f}%)")
            logger.info(
                f"{label:6s} vs {axis:3s}: coherent variance {coh:.5f} "
                f"= {coh*100:.3f}% of unit variance  "
                f"-> implied dk_ms {-coh/MS_SHARE:+.4f}  "
                f"[span {mu.min():+.4f} .. {mu.max():+.4f}]")
        ax.axhline(0., color="k", lw=1, ls=":")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\langle z \rangle$  (pull mean)")
        ax.legend(fontsize=13)
        ax.grid(alpha=0.25)
    fig.text(0.13, 0.96, "CMS", fontsize=22, fontweight="bold", va="top")
    fig.text(0.225, 0.955, "Simulation Work in progress", fontsize=15,
             style="italic", va="top")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"field_structure.{ext}"),
                    bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"wrote {outdir}/field_structure.png")


if __name__ == "__main__":
    main()
