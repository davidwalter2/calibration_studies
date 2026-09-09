#!/usr/bin/env python3
"""`workflow.png` -- the first full-scale study, as a left-to-right pipeline.

Every number in the boxes is MEASURED, and the provenance of each is in the
comment next to it.  The two productions are `dymc_8p5M_260906_v2` (condor
cluster 3803254) and `jpsimc_20M_260906_v2` (cluster 3803264); the walls, the
CPU sums and the peak concurrencies come from those clusters' own
`logs/cluster.log`, the volumes from `du` on the output trees, the accumulated
candidate counts from the `runs/*.npz` caches, and the fit costs from
`results/englogs/`.

    ssh submit50 'cd .../fullscale/figs_wg260909 && \
      ~/.claude/skills/cms-plots/.venv/bin/python schematic_workflow.py'
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

# x extent of the four stages; the gaps carry the arrows
COLS = [(0.6, 19.0), (21.9, 47.4), (50.3, 74.6), (77.5, 99.6)]
BOX = (4.0, 95.0)          # y extent of every stage box
BAND = 7.2                 # height of the coloured header band
PITCH_T = 3.2              # line pitch, body
PITCH_H = 3.7              # line pitch, sub-header
PITCH_B = 1.9              # blank
SIZE_T = 14.5
SIZE_H = 15.5
SIZE_TITLE = [17, 17, 15.5, 17]
FILL_MAX = 1.45   # how much a short column may be stretched to fill

# ("h", ...) sub-header, ("t", ...) body, ("b",) spacer
STAGES = [
    ("(a)  MC inputs", [
        # chunk lists: production/chunks_dymc_8p5M_260905.txt (380 lines) and
        # production/condor_jpsimc_v2/chunks_...v2.txt (1645), summed maxEvents
        ("h", r"J/$\psi \to \mu\mu$"),
        ("t", "UL16 ALCARECO,"),
        ("t", "split-1 repack"),
        ("t", "21 750 740 events"),
        ("t", "1645 chunks"),
        ("b",),
        ("h", r"Z $\to \mu\mu$"),
        ("t", "POWHEG MiNNLO,"),
        ("t", "UL16 MiniAOD"),
        ("t", "8 502 597 events"),
        ("t", "380 chunks"),
        ("b",),
        ("t", "both official CMS"),
        ("t", "UL16 SIM/RECO 106X"),
    ]),
    ("(b)  CVH two-track refit", [
        ("h", "HTCondor, CMS global pool"),
        ("t", "4 threads, 5 GB per job"),
        ("t", "3.9x speedup on a real chunk"),
        ("b",),
        ("h", "cost"),
        # submit50/51 benchmarks, STATE_dy_v2.md sec. 2
        ("t", r"1.01 s/event/core  (J/$\psi$)"),
        ("t", "0.385 s/event/core  (Z)"),
        ("t", "6.1 k + 0.9 k core-h at that"),
        # condor "Run Remote Usage", summed over the two main clusters
        ("t", "rate; 9 610 + 1 430 realised"),
        ("b",),
        ("h", "throughput"),
        # submit -> last terminate, main cluster; peak from 001/005 events
        ("t", r"wall 7.7 h (J/$\psi$), 3.8 h (Z)"),
        ("t", "peak 1164 / 378 jobs, 14 sites"),
        ("b",),
        ("h", "per candidate, exported"),
        ("t", "fitted mass + resolution"),
        ("t", "CF exponents / family, 64 pts"),
        ("t", "material exponents, 42 groups"),
        ("t", "hit classes, 6x6 leg cov."),
        ("t", r"g, H of the hit $\chi^2$"),
        ("b",),
        # du on the output trees / candidates in the caches
        ("t", r"88 kB (J/$\psi$), 71 kB (Z)"),
        ("t", "1.58 TB + 0.29 TB"),
    ]),
    ("(c)  accumulation on submit", [
        ("h", "pairs caches"),
        ("t", "3 733 323 Z candidates"),
        ("t", r"7 923 460 J/$\psi$ candidates"),
        ("t", "(600 of 1645 tasks)"),
        ("t", "~25 min each"),
        ("b",),
        ("h", r"hit-$\chi^2$ quadratic"),
        ("t", "G, H over 92 parameters"),
        ("t", "20 706 999 candidates"),
        ("t", r"2.5 min (Z), 10.7 min (J/$\psi$)"),
        ("b",),
        ("h", "cards (hdf5)"),
        ("t", "3.65 GB   Z inclusive"),
        ("t", "10.7 GB   joint"),
        ("t", "28.4 GB   + material"),
        ("t", "10-13 min each"),
    ]),
    ("(d)  fit", [
        ("h", "rabbit on one H200 GPU"),
        ("t", "Z inclusive, 3 682 662"),
        ("t", "candidates, 7 free params"),
        ("t", "v form: 36 Hessians, 1.2 h"),
        ("t", "m form: 38 iter., 4.7 GPU-h"),
        ("t", "EDM 6.4e-12"),
        ("b",),
        ("h", "joint fit"),
        ("t", r"J/$\psi$ 500 k + Z 500 k"),
        ("t", r"+ hit $\chi^2$ over 20.7 M"),
        ("t", "103 parameters"),
        ("t", "3 h 39 min, EDM 6.2e-19"),
        ("b",),
        ("h", "certified on"),
        ("t", "value + NLL + EDM"),
        ("b",),
        ("t", r"$\sigma(m_Z)$ = 2.27 MeV"),
        ("t", r"$\sigma(\Gamma_Z)$ = 4.16 MeV"),
    ]),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outpath", default=OUTDIR)
    p.add_argument("--name", default="workflow")
    args = p.parse_args()
    os.makedirs(args.outpath, exist_ok=True)

    W.use_style()
    fig, ax = W.blank_canvas(xlim=(0, 100), ylim=(6.0, 97.0))

    # each column is stretched to fill the tallest one, capped, and its frame
    # is then cut to the content -- no dead space at the bottom of a short box
    def content(items):
        return sum(PITCH_B if it[0] == "b" else
                   (PITCH_H if it[0] == "h" else PITCH_T) for it in items)

    ytop = BOX[1] - BAND - 3.4
    target = max(content(it) for _, it in STAGES)
    for (x0, x1), (title, items), tsize in zip(COLS, STAGES, SIZE_TITLE):
        scale = min(FILL_MAX, target / content(items))
        ybot = ytop - content(items) * scale - 2.6
        W.box(ax, x0, ybot, x1, BOX[1], fc="white", ec=W.MIT_DARK, lw=2.0,
              r=1.4)
        # the header band: a rounded box plus a square patch that squares off
        # its bottom two corners against the frame
        W.box(ax, x0, BOX[1] - BAND, x1, BOX[1], fc=W.MIT_RED, ec=W.MIT_RED,
              lw=2.0, r=1.4, zorder=2)
        W.box(ax, x0, BOX[1] - BAND, x1, BOX[1] - BAND + 1.6, fc=W.MIT_RED,
              ec=W.MIT_RED, lw=0.0, r=0.0, zorder=2)
        W.label(ax, x0 + 1.2, BOX[1] - BAND / 2, title, size=tsize,
                weight="bold", color="white", zorder=5)

        y = ytop
        for it in items:
            kind = it[0]
            if kind == "b":
                y -= PITCH_B * scale
                continue
            txt = it[1]
            if kind == "h":
                W.label(ax, x0 + 1.2, y, txt, size=SIZE_H, weight="bold",
                        color=W.MIT_RED)
                y -= PITCH_H * scale
            else:
                W.label(ax, x0 + 1.9, y, txt, size=SIZE_T, color=W.MIT_DARK)
                y -= PITCH_T * scale
        if y < ybot:
            logger.warning(f"stage '{title}' overflows its box (y = {y:.1f})")

    # the pipeline arrows
    for (x0, _), (_, x1) in zip(COLS[1:], COLS[:-1]):
        W.arrow(ax, x1 + 0.4, 62.0, x0 - 0.4, 62.0, color=W.MIT_GRAY, lw=4.0,
                ms=26)

    plot_tools.save_pdf_and_png(args.outpath, args.name, fig=fig)
    output_tools.write_logfile(args.outpath, args.name, args=args,
                               wd=os.path.dirname(os.path.abspath(__file__)))
    logger.info(f"-> {args.outpath}/{args.name}.png")
    plt.close(fig)


if __name__ == "__main__":
    logging.setup_logger(__file__, 3, False)
    main()
