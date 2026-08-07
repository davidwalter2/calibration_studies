"""Decompose the phi-dependent pull bias into charge-EVEN and charge-ODD
parts, to identify what causes it.

Context (2026-08-07). After matching the SIM and refit magnetic field models
(useDefaultField=True), the eta-coherent structure in the pull mean <z>
collapsed from 0.182% to 0.006% of unit variance -- the field mismatch is
closed. But the PHI structure did not move at all: 0.153% -> 0.154%, span
+-0.07 pull units = dp/p ~ 7e-4. So there is a phi-dependent momentum bias
that is NOT the magnetic field model.

Charge parity separates the candidate causes, because the two enter the
curvature with opposite symmetry:

  charge-ODD  (<z>_+ = -<z>_-) : couples to q/p like a CURVATURE offset --
      a residual field/geometry effect, a twist, or anything that biases
      1/pT with a sign. On IDEAL geometry with a phi-symmetric field on
      both sides this would be genuinely surprising and would need its own
      explanation.
  charge-EVEN (<z>_+ = <z>_-)  : couples like ENERGY LOSS / material --
      a phi-modulated material budget (services, cable trays, cooling) that
      the fit's dE/dx or its Geant4e traversal treats differently from the
      simulation.

A mixture is possible and is itself informative: the two components can be
read off independently, since they are orthogonal under q -> -q.

NOTE this is only meaningful on a BOTH-CHARGE sample. The original mu gun
set AddAntiParticle=False and ParticleID=13, giving q+ = 0 and making this
measurement impossible; run_simprod_mugun.sh now defaults to both IDs.

usage:
    python phi_charge_parity.py --cache runs/cf_trackres_mugun2q_phi.npz
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

RED, BLUE, GREEN = "#A31F34", "#1f4e9c", "#2c7a3f"
MS_SHARE = 0.77


def profile(x, z, edges):
    """<z> and its error on fixed bin edges (fixed, not quantile, so the
    two charges share identical bins and can be combined bin by bin)."""
    mu, er = [], []
    for i in range(len(edges) - 1):
        m = (x >= edges[i]) & (x < edges[i + 1])
        if m.sum() < 2:
            mu.append(np.nan)
            er.append(np.nan)
            continue
        mu.append(z[m].mean())
        er.append(z[m].std() / np.sqrt(m.sum()))
    return np.array(mu), np.array(er)


def coherent(mu, er):
    """Variance of the profile in excess of its statistical floor."""
    ok = np.isfinite(mu)
    return max(np.nanvar(mu[ok]) - np.nanmean(er[ok] ** 2), 0.)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--nbins", type=int, default=16)
    ap.add_argument("--outpath", default=None)
    a = ap.parse_args()
    logging.setup_logger(__file__, 3, False)

    today = datetime.date.today().strftime("%y%m%d")
    outdir = output_tools.make_plot_dir(
        a.outpath or os.path.expanduser(
            f"~/public_html/calibration_studies/{today}_phi_charge_parity/"))

    d = np.load(a.cache)
    if "charge" not in d.files:
        raise SystemExit(f"{a.cache} has no 'charge' -- re-extract with the "
                         "current cf_track_resolution.py.")
    z, phi, q = d["z"], d["phi"], d["charge"]
    npos, nneg = int((q > 0).sum()), int((q < 0).sum())
    logger.info(f"{len(z)} tracks: q+ = {npos}, q- = {nneg}")
    if npos == 0 or nneg == 0:
        raise SystemExit("single-charge sample -- parity is not measurable.")

    edges = np.linspace(-np.pi, np.pi, a.nbins + 1)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    mp, ep = profile(phi[q > 0], z[q > 0], edges)
    mm, em = profile(phi[q < 0], z[q < 0], edges)

    even = 0.5 * (mp + mm)
    odd = 0.5 * (mp - mm)
    e_err = 0.5 * np.hypot(ep, em)          # same for both combinations

    c_even, c_odd = coherent(even, e_err), coherent(odd, e_err)
    logger.info(f"charge-EVEN (material/eloss-like): coherent {c_even*100:.3f}% "
                f"of unit variance, span {np.nanmin(even):+.4f}..{np.nanmax(even):+.4f}")
    logger.info(f"charge-ODD  (curvature-like)     : coherent {c_odd*100:.3f}% "
                f"of unit variance, span {np.nanmin(odd):+.4f}..{np.nanmax(odd):+.4f}")
    tot = c_even + c_odd
    if tot > 0:
        logger.info(f"-> EVEN fraction {c_even/tot:.1%}, ODD fraction {c_odd/tot:.1%}")
    verdict = ("CHARGE-EVEN dominates -> material / energy-loss, phi-modulated"
               if c_even > 3. * c_odd else
               "CHARGE-ODD dominates -> curvature-like; needs its own explanation"
               if c_odd > 3. * c_even else "MIXED -- both components present")
    logger.info(f"VERDICT: {verdict}")

    fig, axs = plt.subplots(1, 2, figsize=(16, 6.5))
    ax = axs[0]
    ax.errorbar(ctr, mp, yerr=ep, fmt="o-", color=RED, label=r"$\mu^+$")
    ax.errorbar(ctr, mm, yerr=em, fmt="s-", color=BLUE, label=r"$\mu^-$")
    ax.set_title("by charge", fontsize=15)
    ax = axs[1]
    ax.errorbar(ctr, even, yerr=e_err, fmt="o-", color=GREEN,
                label=f"charge-even ({c_even*100:.3f}%)")
    ax.errorbar(ctr, odd, yerr=e_err, fmt="s-", color="darkorange",
                label=f"charge-odd ({c_odd*100:.3f}%)")
    ax.set_title("parity decomposition", fontsize=15)
    for ax in axs:
        ax.axhline(0., color="k", lw=1, ls=":")
        ax.set_xlabel(r"$\phi$")
        ax.set_ylabel(r"$\langle z \rangle$  (pull mean)")
        ax.legend(fontsize=13)
        ax.grid(alpha=0.25)
    fig.text(0.13, 0.96, "CMS", fontsize=22, fontweight="bold", va="top")
    fig.text(0.225, 0.955, "Simulation Work in progress", fontsize=15,
             style="italic", va="top")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"phi_charge_parity.{ext}"),
                    bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"wrote {outdir}/phi_charge_parity.png")


if __name__ == "__main__":
    main()
