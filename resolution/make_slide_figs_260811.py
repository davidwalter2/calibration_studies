#!/usr/bin/env python3
"""Figures for the 2026-08-11 track-resolution / mass-likelihood deck.

  1. kms_guns.png   -- k_ms per gun sample vs CF probe u, full statistics.
                       The muons close after the four model corrections; the
                       hadrons do not, and their probe dependence is 12x the
                       muon's, i.e. a SHAPE (tail) discrepancy rather than a
                       mis-scaled width.

  2. simhit.png     -- the sim-hit ablation. Left: the hit share of the q/p
                       variance, measured directly by substituting PSimHit
                       truth positions, vs what the CF's own vgf predicts.
                       Right: the paired scale shift, consistent with zero at
                       both momenta.

All numbers measured 2026-08-08/09 on the prompt gun samples; see
Documents/Resolution/NOTES.md.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

hep.style.use(hep.style.ROOT)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "slides_260811", "assets")
MITRED = "#A31F34"
GRAY = "#8A8B8C"


def kms_guns():
    u = np.array([0.05, 0.2, 1.0, 2.0])
    data = {                       # full statistics, 2026-08-08
        r"$\mu$  $p_T$ 20-60": ([+0.0227, +0.0157, +0.0079, +0.0055], GRAY, "o", "-"),
        r"$\mu$  $p_T$ 2-20":  ([+0.0128, +0.0110, +0.0084, +0.0080], "k", "s", "-"),
        r"$K^-$":              ([+0.4340, +0.2246, +0.1213, +0.1036], MITRED, "^", "-"),
        r"$\pi^-$":            ([+0.5573, +0.3147, +0.1840, +0.1618], "#D06A7A", "v", "-"),
        r"$p$":                ([+0.6000, +0.4546, +0.2604, +0.2249], "#6E1020", "D", "-"),
    }
    # constrained_layout, and SHORT axis labels: hep.style.ROOT fonts are large
    # enough that a long xlabel overruns the canvas even with bbox_inches tight
    # (the label is wider than the axes, so the tight bbox clips it), and the
    # right panel's ylabel lands on top of the left panel. The verbal gloss on
    # u lives in the slide caption instead.
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    for lab, (y, c, m, ls) in data.items():
        axs[0].plot(u, y, m + ls, color=c, ms=7, lw=1.7, label=lab)
    axs[0].set_xscale("log"); axs[0].axhline(0, color="k", lw=1, ls=":")
    axs[0].set_xlabel(r"CF probe $u$")
    axs[0].set_ylabel(r"$k_{\rm MS}$")
    axs[0].set_title("all five gun samples", fontsize=15)
    axs[0].legend(fontsize=11, frameon=False, loc="upper right")
    axs[0].grid(alpha=.25)
    ax = axs[1]
    for lab, (y, c, m, ls) in list(data.items())[:2]:
        ax.plot(u, y, m + ls, color=c, ms=7, lw=1.7, label=lab)
    ax.axhline(0, color="k", lw=1, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel(r"CF probe $u$")
    ax.set_ylabel(r"$k_{\rm MS}$")
    ax.set_ylim(-0.06, 0.06)
    ax.set_title("muons only, zoomed  (was $-0.047$ before)", fontsize=15)
    ax.legend(fontsize=12, frameon=False)
    ax.grid(alpha=.25)
    fig.savefig(f"{OUT}/kms_guns.png", dpi=160, bbox_inches="tight")
    print("wrote kms_guns.png")


def simhit():
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.0), constrained_layout=True)
    lab = [r"$p$ 4.6-67 GeV", r"$p$ 29-218 GeV"]
    x = np.arange(2)
    f_obs = np.array([0.1352, 0.2797])      # measured by the ablation
    f_cf = np.array([0.1054, 0.1998])       # the CF's own vgf
    # NOT f_obs/f_cf. Each f is a fraction of ITS OWN total, and the observed
    # total is 0.929 / 0.931 of the modelled one, so the hit-contribution ratio
    # is (f_obs/f_cf) * (total obs/model) = 1.19 / 1.30 (NOTES.md 2026-08-09).
    # The naive bar ratio gives 1.28/1.40 and overstates the gap.
    ratio = np.array([1.191, 1.304])
    w = 0.36
    axs[0].bar(x - w/2, f_obs*100, w, color=MITRED, alpha=.9, label="measured (sim-hit ablation)")
    axs[0].bar(x + w/2, f_cf*100,  w, color=GRAY,   alpha=.9, label="CF model (vgf)")
    for i in range(2):
        axs[0].text(i, max(f_obs[i], f_cf[i])*100 + 1.2,
                    f"×{ratio[i]:.2f}", ha="center", fontsize=13, color=MITRED)
    axs[0].set_xticks(x); axs[0].set_xticklabels(lab)
    axs[0].set_ylabel("hit share of $q/p$ var.  [%]")
    # headroom so the x-factor annotations clear the title
    axs[0].set_ylim(0, max(f_obs.max(), f_cf.max()) * 100 * 1.22)
    axs[0].set_title("hits contribute 19-30% more than modelled", fontsize=15)
    axs[0].legend(fontsize=11, frameon=False); axs[0].grid(alpha=.25, axis="y")
    shift = np.array([-0.048, +0.058]); err = np.array([0.149, 0.177])
    axs[1].errorbar(x, shift, yerr=err, fmt="o", color=MITRED, ms=9, capsize=6, lw=2)
    axs[1].axhline(0, color="k", lw=1.2, ls=":")
    axs[1].fill_between([-.5, 1.5], -0.5, 0.5, color=GRAY, alpha=.15,
                        label=r"$\pm 0.5\times10^{-4}$")
    axs[1].set_xlim(-.5, 1.5); axs[1].set_xticks(x); axs[1].set_xticklabels(lab)
    axs[1].set_ylabel(r"paired scale shift  $[10^{-4}]$")
    axs[1].set_title("and NO scale shift (0.3$\\sigma$ both)", fontsize=15)
    axs[1].legend(fontsize=11, frameon=False); axs[1].grid(alpha=.25, axis="y")
    fig.savefig(f"{OUT}/simhit.png", dpi=160, bbox_inches="tight")
    print("wrote simhit.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    kms_guns()
    simhit()
