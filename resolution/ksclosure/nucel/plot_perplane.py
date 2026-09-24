#!/usr/bin/env python3
"""Figures of `perplane_pulls.py`: per-plane tail fractions of a pion's local
residual on the real tracker, data against the model CF with and without the
nuclear-elastic channel.  One figure per (projection, threshold); the ratio
panel is data / model, filled = channel ON, open = OFF.

usage: python3 plot_perplane.py --rows a.npz b.npz [--outdir ...]
"""
import argparse
import datetime
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import mplhep as hep                   # noqa: E402
from wums import logging               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import pubhtml                         # noqa: E402
import ratiopanel                      # noqa: E402

hep.style.use(hep.style.ROOT)
logger = logging.setup_logger(__file__, 3)
PT_ORDER = ["pt0.7", "pt1.5", "pt3", "pt10"]
COL = {"pt0.7": "tab:purple", "pt1.5": "tab:blue", "pt3": "tab:green", "pt10": "tab:orange"}
PROJ = {"locx": "bending (local x)", "locy": "non-bending (local y)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--outdir", default=os.path.expanduser(
        f"~/public_html/ZMass/cvh/{datetime.date.today().strftime('%y%m%d')}_ks_nucel"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    R = {}
    for fn in args.rows:
        d = np.load(fn, allow_pickle=True)
        for k in d.files:
            R.setdefault(k, []).append(d[k])
    R = {k: np.concatenate(v) for k, v in R.items()}
    labels = [p for p in PT_ORDER if p in set(R["label"].tolist())]
    lines = []
    for func in ("locx", "locy"):
        for c in ("3", "5"):
            fig, ax, rax = ratiopanel.make_ratio_fig()
            for lab in labels:
                m = (R["label"] == lab) & (R["func"] == func)
                o = np.argsort(R["r"][m])
                r = R["r"][m][o]
                n = R["n"][m][o]
                dd = R[f"d{c}"][m][o]
                mon = R[f"m{c}_on"][m][o]
                mof = R[f"m{c}_off"][m][o]
                e = np.sqrt(np.maximum(dd * (1 - dd), 1.0 / n) / n)
                p = lab.replace("pt", "")
                ax.errorbar(r, 100 * dd, 100 * e, fmt="o", ms=4, color=COL[lab],
                            label=rf"$\pi^-$, $p_T$ = {p} GeV, data")
                ax.plot(r, 100 * mon, "-", color=COL[lab], lw=1.4)
                ax.plot(r, 100 * mof, "--", color=COL[lab], lw=1.0)
                rax.errorbar(r, dd / mon, e / mon, fmt="o", ms=4, color=COL[lab])
                rax.errorbar(r, dd / mof, e / mof, fmt="o", ms=4, mfc="none", color=COL[lab])
                chi_on = float(np.sum(((dd - mon) / e) ** 2))
                chi_off = float(np.sum(((dd - mof) / e) ** 2))
                lines.append(f"{func} |z|>{c} {lab}: {m.sum()} planes, sum data {100*dd.mean():.3f} % "
                             f"model on {100*mon.mean():.3f} % off {100*mof.mean():.3f} %; "
                             f"chi2 on {chi_on:.1f} off {chi_off:.1f}")
            ax.plot([], [], "k-", lw=1.4, label="model, with nuclear elastic")
            ax.plot([], [], "k--", lw=1.0, label="model, without")
            ax.set_ylabel(rf"$P(|z| > {c})$ [%]")
            ax.set_yscale("log")
            lo_, hi_ = ax.get_ylim()
            ax.set_ylim(lo_, hi_ * 12.0)
            ax.legend(fontsize="x-small", ncol=2, loc="upper left")
            ax.set_title(f"{PROJ[func]}, real tracker, $\\eta$ = 0.3", fontsize="small")
            rax.axhline(1.0, color="gray")
            rax.set_ylim(0.0, 3.0)
            rax.set_ylabel("data / model", fontsize="small")
            rax.set_xlabel("plane radius [cm]")
            pubhtml.savefig(fig, os.path.join(args.outdir, f"perplane_tail{c}_{func}.pdf"))
            plt.close(fig)
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(args.outdir, "perplane_summary.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
