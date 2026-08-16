#!/usr/bin/env python3
"""Figures for the 2026-08-11 clean-propagation deck.

Two results, both new since the 2026-08-06 version:

  1. acceptance.png -- the modal-sequence acceptance cut WAS the pT=3
     non-closure. Per-layer closure under the two acceptance modes, at both
     momenta, from the cleanprop_rows npz dumps.

  2. species.png -- for hadrons the CORE closes exactly as well as the muon's
     while the TAIL does not: rob68 vs std/rob68 per species. The excess is a
     large-deflection tail the propagator's physics list cannot contain
     (ionisation + bremsstrahlung + transportation only, no hadronic).

Style per the user's standing preference: hep.style.ROOT, savefig with
bbox_inches='tight'.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

hep.style.use(hep.style.ROOT)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "slides", "assets")
SC = "/tmp/claude-125124/-work-submit-david-w-ZMass/40621a07-b09b-46d9-b7b3-4189252bcf3e/scratchpad"

MITRED = "#A31F34"
GRAY = "#8A8B8C"


def acceptance():
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, (tag, dirn, ttl) in zip(axs, (
            ("pt3", f"{SC}/accfix", r"$p_T = 3$ GeV   (acceptance drops 5.6%)"),
            ("pt40", f"{SC}/accfix40", r"$p_T = 40$ GeV   (drops 0.086%)"))):
        for mode, col, mk, lab in (("modal", GRAY, "o", "modal sequence (all earlier results)"),
                                   ("perplane", MITRED, "s", "per-plane acceptance")):
            f = f"{dirn}/cleanprop_rows_{mode}.npz"
            if not os.path.exists(f):
                continue
            d = np.load(f, allow_pickle=True)
            y = (d["fdata"] - d["fmodel"])[:, -1]     # u = 1 probe
            ax.plot(d["r"], y * 1e4, mk + "-", color=col, ms=6, lw=1.6, label=lab)
        ax.axhline(0., color="k", lw=1, ls=":")
        ax.set_xlabel("layer radius [cm]")
        ax.set_title(ttl, fontsize=15)
        ax.grid(alpha=0.25)
    axs[0].set_ylabel(r"data $-$ model  $[10^{-4}]$   ($u=1$)")
    axs[0].legend(loc="upper left", fontsize=12, frameon=False)
    fig.savefig(f"{OUT}/acceptance.png", dpi=160, bbox_inches="tight")
    print("wrote acceptance.png")


def closure():
    """Main-body result: per-plane closure at both momenta, ONE acceptance.

    The acceptance before/after (acceptance.png) is a piece of this test's own
    history and belongs in backup -- the main body should show the closure that
    stands, not the story of how it was obtained.
    """
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for tag, dirn, col, mk, lab in (
            ("pt3", f"{SC}/accfix", MITRED, "s", r"$p_T = 3$ GeV"),
            ("pt40", f"{SC}/accfix40", "k", "o", r"$p_T = 40$ GeV")):
        f = f"{dirn}/cleanprop_rows_perplane.npz"
        if not os.path.exists(f):
            continue
        d = np.load(f, allow_pickle=True)
        y = (d["fdata"] - d["fmodel"])[:, -1]      # u = 1 probe
        # The headline number is the MEAN over planes; the plane-to-plane
        # scatter is 2-5x larger and must be shown, not hidden behind a band.
        ax.plot(d["r"], y * 1e4, mk + "-", color=col, ms=6, lw=1.7,
                label=f"{lab}:  mean {y.mean()*1e4:+.1f}, rms {y.std()*1e4:.1f}")
        ax.axhline(y.mean() * 1e4, color=col, lw=1.4, ls="--", alpha=.8)
    ax.axhline(0., color="k", lw=1, ls=":")
    ax.set_xlim(0, 115)
    ax.set_xlabel("layer radius [cm]")
    ax.set_ylabel(r"data $-$ model  $[10^{-4}]$   ($u=1$)")
    ax.legend(fontsize=13, frameon=False)
    ax.grid(alpha=0.25)
    fig.savefig(f"{OUT}/closure.png", dpi=160, bbox_inches="tight")
    print("wrote closure.png")


def species():
    # measured 2026-08-08, pT=3, per-plane acceptance, inelastic veto applied
    sp = ["$\\mu$", "$K^-$", "$\\pi^-$", "$p$"]
    rob = [0.881, 0.886, 0.906, 0.919]
    ratio = [1.18, 2.56, 4.11, 4.47]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.0))
    x = np.arange(len(sp))
    cols = [GRAY, MITRED, MITRED, MITRED]
    axs[0].bar(x, rob, color=cols, alpha=.85)
    axs[0].axhline(rob[0], color="k", ls=":", lw=1.2)
    axs[0].set_xticks(x); axs[0].set_xticklabels(sp)
    axs[0].set_ylim(0.6, 1.05)
    axs[0].set_ylabel(r"rob68 of $z$   (the CORE)")
    axs[0].set_title("core width: identical to 4%", fontsize=15)
    axs[1].bar(x, ratio, color=cols, alpha=.85)
    axs[1].axhline(ratio[0], color="k", ls=":", lw=1.2)
    axs[1].set_xticks(x); axs[1].set_xticklabels(sp)
    axs[1].set_ylabel(r"std / rob68   (the TAIL)")
    axs[1].set_title("tail: 2.6 to 4.5 times the muon", fontsize=15)
    for ax in axs:
        ax.grid(alpha=0.25, axis="y")
    fig.savefig(f"{OUT}/species.png", dpi=160, bbox_inches="tight")
    print("wrote species.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    acceptance()
    closure()
    species()
