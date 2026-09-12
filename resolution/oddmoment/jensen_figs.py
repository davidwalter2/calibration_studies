#!/usr/bin/env python3
"""Figures for the second-order (Jensen) term of the mass functional."""
import argparse
import datetime
import os
import re
import sys

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
import pubhtml                                                # noqa: E402
from wums import logging, plot_tools                          # noqa: E402

hep.style.use(hep.style.ROOT)
logger = logging.setup_logger(__file__, 3, False)
MJ = 3.0969


def read_fits(path):
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        if "alpha" not in line:
            continue
        lab = line.split("[sigma")[0].strip()
        v = float(line.split("alpha =")[1].split("+-")[0])
        e = float(line.split("+-")[1].split("e-3")[0])
        j = re.search(r"jensen=([\d.]+)", line)
        out[(lab, float(j.group(1)) if j else 0.)] = (v, e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="runs/cf_masspairs_jpsigun_ul16_260905d_m0.npz")
    ap.add_argument("--fits", default="oddmoment/out/jensenfits.txt")
    ap.add_argument("--outdir", default="")
    a = ap.parse_args()
    outdir = a.outdir or os.path.expanduser(
        "~/public_html/ZMass/cvh/%s_oddmoment" % datetime.date.today().strftime("%y%m%d"))
    os.makedirs(outdir, exist_ok=True)
    pubhtml.ensure_index(outdir, logger=logger)
    F = read_fits(a.fits)

    d = np.load(a.cache)
    z = d["z"].astype(np.float64); s = d["sigma"].astype(np.float64)
    mg = d["eta"].astype(np.float64); vgf = d["vgf"].astype(np.float64)
    mobs = z * s + (mg - MJ)
    ai = (1. + vgf) * s / mg
    csrel = (s - ai * mobs) / mg
    good = csrel < 5. * np.median(csrel)
    q = np.quantile(csrel[good], np.linspace(0, 1, 6))
    pred, xs = [], []
    for k in range(5):
        m = (csrel >= q[k]) & (csrel < q[k + 1])
        w = 1. / s[m] ** 2
        pred.append(1e3 * 1.5 * m.sum() / (MJ ** 2 * w.sum()))
        xs.append(float(np.median((s[m] / mg[m])[m[m]] if False else (s / mg)[m])))
    obs = [F.get((f"GUN corr csrel bin {k}/5", 0.)) for k in range(5)]
    ok = [k for k in range(5) if obs[k]]
    if len(ok) >= 3:
        fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                gridspec_kw={"height_ratios": [2.2, 1]})
        xv = np.array([pred[k] for k in ok])
        yv = np.array([obs[k][0] for k in ok])
        ev = np.array([obs[k][1] for k in ok])
        axs[0].errorbar(xv, yv, yerr=ev, marker="o", color="k", ls="none", ms=9,
                        label=r"$\hat\alpha$, resolution-corrected likelihood")
        g = np.linspace(0, 1.05 * xv.max(), 10)
        axs[0].plot(g, g, color="tab:blue", ls="--", lw=2,
                    label=r"$\frac{1}{2}{\rm tr}(H\Sigma)/m = 1.5\,(\sigma_m/m)^2$")
        axs[0].axhline(0, color="grey", lw=1)
        axs[0].set_ylabel(r"$\hat\alpha$  [$10^{-3}$]")
        axs[0].legend(fontsize=12)
        axs[0].set_title(r"J/$\psi$ gun, quintiles of the truth-free "
                         r"$\bar\sigma_m/m$", fontsize=14)
        r = yv / xv
        er = ev / xv
        w = 1. / er ** 2
        rbar = float((r * w).sum() / w.sum())
        erbar = float(1. / np.sqrt(w.sum()))
        axs[1].errorbar(xv, r, yerr=er, marker="o", color="k", ls="none", ms=9)
        axs[1].axhline(1, color="tab:blue", ls="--", lw=2)
        axs[1].axhline(rbar, color="tab:red", lw=2,
                       label=rf"const $= {rbar:.2f}\pm{erbar:.2f}$")
        axs[1].fill_between([0, 1.05 * xv.max()], rbar - erbar, rbar + erbar,
                            color="tab:red", alpha=0.15)
        axs[1].set_ylim(0., 2.2)
        axs[1].set_ylabel(r"$\hat\alpha$ / prediction")
        axs[1].set_xlabel(r"predicted $\frac{1}{2}{\rm tr}(H\Sigma)/m$ in the bin  [$10^{-3}$]")
        axs[1].legend(fontsize=12)
        plot_tools.save_pdf_and_png(outdir, "jensen_differential_gun", fig)
        plt.close(fig)

    # ladder: naive -> corrected -> corrected+Jensen, both samples
    rows = [("gun, naive", F.get(("GUN 260905d NAIVE", 0.)) or (-0.00470, 0.01666)),
            ("gun, +resolution", (0.14102, 0.01665)),
            ("gun, +resolution +Jensen", F.get(("GUN corrected + JENSEN", 1.))),
            ("v3, naive", (-0.04637, 0.02539)),
            ("v3, +resolution", (0.08104, 0.02535)),
            ("v3, +resolution +Jensen", F.get(("V3 corrected + JENSEN", 1.)))]
    rows = [(n, v) for n, v in rows if v]
    fig, ax = plt.subplots(figsize=(8, 6))
    yy = np.arange(len(rows))[::-1]
    for k, (nm, (v, e)) in enumerate(rows):
        c = ("k" if "naive" in nm else
             ("tab:green" if "Jensen" in nm else "tab:orange"))
        ax.errorbar([v], [yy[k]], xerr=[e], marker="o", ms=9, color=c, lw=2)
    ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in rows])
    ax.axvline(0, color="grey", lw=1)
    ax.set_xlabel(r"$\hat\alpha$  [$10^{-3}$]")
    ax.set_title("mass scale: the two first-principles corrections", fontsize=14)
    plot_tools.save_pdf_and_png(outdir, "jensen_ladder", fig)
    plt.close(fig)
    logger.info(f"figures -> {outdir}")


if __name__ == "__main__":
    main()
