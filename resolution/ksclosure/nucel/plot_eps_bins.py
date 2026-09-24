#!/usr/bin/env python3
"""The K_S momentum scale per bin, without / with / first-block-only nuclear
elastic family (same candidates, same card recipe).  eps = alpha_mass / <f>,
<f> the 1/sigma^2-weighted lever of each bin's cache (ks_table.lever); the
ratio-style lower panel is the difference to the family-free fit.

usage: python3 plot_eps_bins.py --runs ../runs/nucel [--outdir ...]
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import mplhep as hep                   # noqa: E402
from wums import logging               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import pubhtml                         # noqa: E402
from ks_table import lever             # noqa: E402

hep.style.use(hep.style.ROOT)
logger = logging.setup_logger(__file__, 3)
BINS = [("all", "all"), ("r0_2", "r < 2"), ("r2_4", "2-4"), ("r4_10", "4-10"),
        ("r10_60", "> 10 cm"), ("plo", "p < 0.8"), ("pmid", "0.8-1.5"),
        ("phi", "> 1.5 GeV"), ("fromb", "from B"), ("prompt", "prompt")]
VAR = [("base", "without nuclear elastic", "crimson", "o"),
       ("nuc", "with nuclear elastic", "royalblue", "s"),
       ("nucfirst", "first block only", "darkgreen", "^")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--outdir", default=os.path.expanduser(
        f"~/public_html/ZMass/cvh/{datetime.date.today().strftime('%y%m%d')}_ks_nucel"))
    args = ap.parse_args()
    x = np.arange(len(BINS))
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                  gridspec_kw=dict(height_ratios=(3, 1), hspace=0.06))
    ref = None
    for k, (v, lab, c, mk) in enumerate(VAR):
        e, s = [], []
        for b, _ in BINS:
            p = json.load(open(os.path.join(args.runs, "results", f"{v}_{b}.json")))
            f = lever(os.path.join(args.runs, f"kspairs_{v}_{b}.npz"))[0]
            e.append(p["fitted"][0] / f)
            s.append(p["err"][0] / f)
        e, s = np.array(e), np.array(s)
        ax.errorbar(x + 0.12 * (k - 1), e, s, fmt=mk, color=c, label=lab)
        if ref is None:
            ref = e
        else:
            rax.plot(x + 0.12 * (k - 1), e - ref, mk, color=c)
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_ylabel(r"$\varepsilon$ [$10^{-3}$]")
    ax.legend(fontsize="small")
    rax.axhline(0.0, color="gray")
    rax.set_ylabel("minus without", fontsize="small")
    rax.set_xticks(x)
    rax.set_xticklabels([b[1] for b in BINS], rotation=35, fontsize="small")
    pubhtml.savefig(fig, os.path.join(args.outdir, "ks_nucel_eps_bins.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
