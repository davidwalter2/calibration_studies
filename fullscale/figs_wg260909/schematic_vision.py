#!/usr/bin/env python3
"""`vision.png` -- the unified likelihood, as a block diagram.

Three columns: the data channels on the left, the likelihood terms they build
in the middle, the parameters those terms constrain on the right.  The point
the picture has to make is the LAST one: every channel's mass term and every
track's hit residual constrain the SAME detector parameters, so there are no
per-channel scale factors anywhere in the chain.

Run it in the plotting venv (matplotlib is not on the el9 head node):

    ssh submit50 'cd .../fullscale/figs_wg260909 && \
      ~/.claude/skills/cms-plots/.venv/bin/python schematic_vision.py'
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wgstyle as W  # noqa: E402

from wums import logging, output_tools, plot_tools  # noqa: E402

logger = logging.child_logger(__name__)

OUTDIR = os.path.expanduser(
    "~/public_html/slides/260909_unbinned_likelihood_walter/assets")

# --------------------------------------------------------------- geometry ---
BUS_X = 1.6                     # the "every track" spine, left of everything
CH = (3.4, 27.5)                # data-channel boxes
MID = (29.8, 66.2)              # the mass-term panel
KER = (31.0, 65.0)              # the per-channel kernel boxes inside it
PAR = (70.0, 99.5)              # the parameter boxes

ROW_TOP = 85.5                  # top of the first channel box
ROW_H = 8.2
ROW_PITCH = 9.4
COSMIC = (28.0, 36.2)           # the cosmic row, set apart: no mass term
HIT = (5.0, 24.0)               # the hit-residual box
MIDPANEL = (38.0, 96.8)
PHYSBOX = (61.5, 96.8)
DETBOX = (5.0, 56.5)

# title, sub-line, kernel line, kernel note
CHANNELS = [
    (r"J/$\psi \to \mu\mu$", "ALCARECO, two-track CVH fit",
     r"$K = \delta(m' - m_{J/\psi})$", "PDG mass, fixed"),
    (r"$\Upsilon$(1S,2S,3S) $\to \mu\mu$", "ALCARECO, two-track CVH fit",
     r"$K = \sum_n \delta(m' - m_{\Upsilon(nS)})$", "PDG masses, fixed"),
    (r"Z/$\gamma^* \to \mu\mu$", "MiniAOD, two-track CVH fit",
     r"$K = d\sigma / (dm'\, dy\, d\cos\theta^*)$",
     r"Breit-Wigner: $m_Z$, $\Gamma_Z$ FLOATING"),
    (r"$K_S \to \pi\pi$,   $\Lambda \to p\pi$", "V0, displaced vertex",
     r"$K = \delta(m' - m_{K_S})$,  $\delta(m' - m_{\Lambda})$",
     "PDG masses, fixed"),
    (r"$D^* \to D^0\pi$,   $B^{\pm} \to J/\psi\, K$",
     "three-track CVH fit",
     r"$K = \delta(m' - m_{D^*})$,  $\delta(m' - m_{B})$",
     "PDG masses, fixed"),
]
COSMIC_CH = ("cosmic muons", "split track -- no vertex, no mass")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outpath", default=OUTDIR)
    p.add_argument("--name", default="vision")
    args = p.parse_args()
    os.makedirs(args.outpath, exist_ok=True)

    W.use_style()
    fig, ax = W.blank_canvas(xlim=(0, 100), ylim=(0, 102))

    # ------------------------------------------------------ column headers --
    W.label(ax, (CH[0] + CH[1]) / 2, 100.0, "data channels", size=21,
            weight="bold", ha="center")
    W.label(ax, (MID[0] + MID[1]) / 2, 100.0, "likelihood terms", size=21,
            weight="bold", ha="center")
    W.label(ax, (PAR[0] + PAR[1]) / 2, 100.0, "parameters", size=21,
            weight="bold", ha="center")

    # ------------------------------------------------- the mass-term panel --
    W.box(ax, MID[0], MIDPANEL[0], MID[1], MIDPANEL[1], fc="#F4F1EE",
          ec=W.MIT_GRAY, lw=1.4, r=1.8, zorder=1)
    W.label(ax, (MID[0] + MID[1]) / 2, 94.4,
            "per-candidate unbinned mass term", size=17.5, weight="bold",
            ha="center")
    W.label(ax, (MID[0] + MID[1]) / 2, 91.0,
            r"$L_i = \int K(m')\, p_i(m_i\,|\,m')\, dm'$", size=20,
            ha="center", color=W.MIT_RED)
    W.label(ax, (MID[0] + MID[1]) / 2, 87.6,
            r"$p_i$ from the CVH track fit, per candidate",
            size=15, ha="center", color=W.MIT_DARK)

    # ------------------------------------- channels + their kernels, row by --
    mids = []
    for i, (title, sub, kern, note) in enumerate(CHANNELS):
        y1 = ROW_TOP - i * ROW_PITCH
        y0 = y1 - ROW_H
        mid = 0.5 * (y0 + y1)
        mids.append(mid)
        col = W.CH_COLORS[i]
        W.box(ax, CH[0], y0, CH[1], y1, fc="white", ec=col, lw=2.2)
        W.label(ax, CH[0] + 1.3, mid + 1.8, title, size=16, weight="bold",
                color=col)
        W.label(ax, CH[0] + 1.3, mid - 2.2, sub, size=14.5, color=W.MIT_GRAY)

        W.box(ax, KER[0], y0, KER[1], y1, fc="white", ec=col, lw=1.6)
        W.label(ax, KER[0] + 1.2, mid + 1.8, kern, size=15, color=W.MIT_DARK)
        W.label(ax, KER[0] + 1.2, mid - 2.2, note, size=14, color=col)

        W.arrow(ax, CH[1], mid, KER[0], mid, color=col, lw=2.2)

    # the cosmic channel: no mass term, so it stops outside the panel
    y0, y1 = COSMIC
    cmid = 0.5 * (y0 + y1)
    col = W.CH_COLORS[5]
    W.box(ax, CH[0], y0, CH[1], y1, fc="white", ec=col, lw=2.2, ls=(0, (5, 2)))
    W.label(ax, CH[0] + 1.3, cmid + 1.8, COSMIC_CH[0], size=16, weight="bold",
            color=col)
    W.label(ax, CH[0] + 1.3, cmid - 2.2, COSMIC_CH[1], size=14.5,
            color=W.MIT_GRAY)
    W.label(ax, KER[0] + 1.2, cmid, "no mass term -- hit residuals only",
            size=15, color=col, style="italic")

    # ------------------------------ the spine: every track -> the hit term --
    hmid = 0.5 * (HIT[0] + HIT[1])
    for m in mids + [cmid]:
        W.lines(ax, [CH[0], BUS_X], [m, m], lw=1.6)
    W.lines(ax, [BUS_X, BUS_X], [mids[0], hmid], lw=2.4)
    W.lines(ax, [BUS_X, MID[0] - 1.2], [hmid, hmid], lw=2.4)
    W.arrow(ax, MID[0] - 1.2, hmid, MID[0], hmid, color=W.MIT_GRAY, lw=2.4)
    W.label(ax, BUS_X + 1.0, hmid + 3.0, "every track of every channel",
            size=14.5, color=W.MIT_GRAY)

    # ------------------------------------------------- the hit-residual box --
    W.box(ax, MID[0], HIT[0], MID[1], HIT[1], fc="#EFEFEF", ec=W.MIT_DARK,
          lw=1.8, r=1.8)
    W.label(ax, (MID[0] + MID[1]) / 2, HIT[1] - 3.0,
            "hit residuals of every track", size=17.5, weight="bold",
            ha="center")
    W.label(ax, (MID[0] + MID[1]) / 2, HIT[1] - 8.0,
            r"$g \cdot \delta\theta + \frac{1}{2}\,"
            r"\delta\theta^{\mathsf{T}} H\, \delta\theta$",
            size=20, ha="center", color=W.MIT_RED)
    W.label(ax, (MID[0] + MID[1]) / 2, HIT[1] - 12.8,
            r"the track-fit $\chi^2$, expanded about $\theta$",
            size=15, ha="center")
    W.label(ax, (MID[0] + MID[1]) / 2, HIT[1] - 16.4,
            r"one $g$, $H$ per candidate -- ALL channels",
            size=15, ha="center")

    # ------------------------------------------------------- the parameters --
    W.box(ax, PAR[0], PHYSBOX[0], PAR[1], PHYSBOX[1], fc="white",
          ec=W.MIT_DARK, lw=2.0, r=1.8)
    W.label(ax, PAR[0] + 1.6, PHYSBOX[1] - 3.4, "physics parameters",
            size=18.5, weight="bold")
    W.label(ax, PAR[0] + 1.6, PHYSBOX[1] - 8.4,
            "fixed to the PDG value --", size=15.5)
    W.label(ax, PAR[0] + 1.6, PHYSBOX[1] - 12.0,
            "these ANCHOR the scale:", size=15.5)
    W.label(ax, PAR[0] + 3.4, PHYSBOX[1] - 16.4,
            r"$m_{J/\psi}$,  $m_{\Upsilon(nS)}$,  $m_{K_S}$,",
            size=16, color=W.MIT_GRAY)
    W.label(ax, PAR[0] + 3.4, PHYSBOX[1] - 20.4,
            r"$m_{\Lambda}$,  $m_{D^*}$,  $m_{B}$",
            size=16, color=W.MIT_GRAY)
    W.label(ax, PAR[0] + 1.6, PHYSBOX[1] - 25.6,
            "floating -- the measurement:", size=15.5)
    W.label(ax, PAR[0] + 3.4, PHYSBOX[1] - 30.4,
            r"$m_Z$,   $\Gamma_Z$,   later $\sin^2\theta_{\rm eff}$",
            size=17, color=W.MIT_RED, weight="bold")

    W.box(ax, PAR[0], DETBOX[0], PAR[1], DETBOX[1], fc="white",
          ec=W.MIT_RED, lw=2.6, r=1.8)
    W.label(ax, PAR[0] + 1.6, DETBOX[1] - 3.6,
            r"shared detector parameters $\theta$", size=17, weight="bold",
            color=W.MIT_RED)
    for k, (nm, det) in enumerate([
            ("alignment", "module positions and rotations"),
            ("magnetic field", "~50 scalar-potential modes"),
            ("material", "amount per material group"),
            ("hit resolutions", "one scale per hit class")]):
        y = DETBOX[1] - 10.4 - 6.6 * k
        W.label(ax, PAR[0] + 2.2, y, nm, size=16, weight="bold")
        W.label(ax, PAR[0] + 2.2, y - 2.9, det, size=14.5, color=W.MIT_GRAY)
    W.lines(ax, [PAR[0] + 1.6, PAR[1] - 1.6], [DETBOX[0] + 13.0] * 2,
            color=W.MIT_GRAY, lw=1.2)
    W.label(ax, PAR[0] + 1.6, DETBOX[0] + 9.4,
            "EVERY term above constrains", size=15.5, weight="bold")
    W.label(ax, PAR[0] + 1.6, DETBOX[0] + 6.0,
            r"these -- one $\theta$ for all channels,", size=15.5)
    W.label(ax, PAR[0] + 1.6, DETBOX[0] + 2.6,
            "no per-channel scale factors", size=15.5)

    # ------------------------------------------- terms -> parameters arrows --
    W.arrow(ax, MID[1], 74.0, PAR[0], 80.0, color=W.MIT_DARK, lw=2.6, rad=0.12)
    W.arrow(ax, MID[1], 50.0, PAR[0], 44.0, color=W.MIT_RED, lw=2.6, rad=-0.12)
    W.arrow(ax, MID[1], hmid, PAR[0], 18.0, color=W.MIT_RED, lw=2.6, rad=-0.1)

    plot_tools.save_pdf_and_png(args.outpath, args.name, fig=fig)
    output_tools.write_logfile(args.outpath, args.name, args=args,
                               wd=os.path.dirname(os.path.abspath(__file__)))
    logger.info(f"-> {args.outpath}/{args.name}.png")
    plt.close(fig)


if __name__ == "__main__":
    logging.setup_logger(__file__, 3, False)
    main()
