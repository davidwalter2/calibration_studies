#!/usr/bin/env python3
"""Figures for the candidate-level (mass) pull-normalisation study."""
import argparse
import datetime
import os
import sys

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.dirname(_HERE)
sys.path.insert(0, _RES)
sys.path.insert(0, _HERE)
import mass_pull as MP                                        # noqa: E402
import pubhtml                                                # noqa: E402
import sigma_pull as SP                                       # noqa: E402
from wums import logging, plot_tools                          # noqa: E402

hep.style.use(hep.style.ROOT)
logger = logging.setup_logger(__file__, 3, False)
MJPSI = 3.0969


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--aux", required=True)
    ap.add_argument("--pull", required=True, help="masspull_*.npz")
    ap.add_argument("--toyfits", default="")
    ap.add_argument("--tag", default="jpsigun")
    ap.add_argument("--label", default="J/$\\psi$ gun candidates")
    ap.add_argument("--outdir", default="")
    a = ap.parse_args()
    outdir = a.outdir or os.path.expanduser(
        "~/public_html/cvh/%s_oddmoment" % datetime.date.today().strftime("%y%m%d"))
    os.makedirs(outdir, exist_ok=True)
    pubhtml.ensure_index(outdir, logger=logger)

    d = np.load(a.cache); x = np.load(a.aux); pl = np.load(a.pull)
    z = d["z"].astype(np.float64); sig = d["sigma"].astype(np.float64)
    mg = x["mgen"]; srel = sig / mg
    zb = pl["zbar"]; sbar = pl["sbar"]
    aclosed = pl["aclosed"]; afl = float(pl["afluct"])
    good = srel < 5. * np.median(srel)
    m5 = (np.abs(z) < 5.) & good

    # -------- panel 1: <z_m> in bins of sigma, fit sigma vs truth sigma_bar
    nb = 8
    e = np.quantile(srel[good], np.linspace(0, 1, nb + 1))
    b = np.clip(np.digitize(srel, e[1:-1]), 0, nb - 1)
    eb = np.quantile((sbar / mg)[good], np.linspace(0, 1, nb + 1))
    bb = np.clip(np.digitize(sbar / mg, eb[1:-1]), 0, nb - 1)
    xs = np.arange(nb) + 0.5
    y1, e1, y2, e2 = [], [], [], []
    for k in range(nb):
        s1 = (b == k) & m5
        y1.append(z[s1].mean()); e1.append(z[s1].std(ddof=1) / np.sqrt(s1.sum()))
        s2 = (bb == k) & (np.abs(zb) < 5.) & good
        y2.append(zb[s2].mean()); e2.append(zb[s2].std(ddof=1) / np.sqrt(s2.sum()))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(xs, y1, yerr=e1, marker="o", color="k", lw=2,
                label=r"$z_m$, bins of $\sigma_m^{\rm fit}$")
    ax.errorbar(xs, y2, yerr=e2, marker="s", color="tab:red", lw=2,
                label=r"$\bar z_m$, bins of $\bar\sigma_m$(gen)")
    ax.axhline(0, color="grey", lw=1)
    ax.set_xlabel(r"$\sigma_m/m$ octile")
    ax.set_ylabel(r"$\langle z_m\rangle_{|z|<5}$")
    ax.legend(fontsize=13); ax.set_title(a.label, fontsize=15)
    plot_tools.save_pdf_and_png(outdir, f"mass_z_vs_sigma_{a.tag}", fig)
    plt.close(fig)

    # -------- panel 2: a_fluct vs the closed form
    nb = 6
    e = np.quantile(srel[good], np.linspace(0, 1, nb + 1))
    b = np.clip(np.digitize(srel, e[1:-1]), 0, nb - 1)
    # in-cell centring, same cells as mass_pull's first definition
    gp1 = x["gpt_p"] * np.cosh(x["geta_p"]); gp2 = x["gpt_m"] * np.cosh(x["geta_m"])
    cid = SP.cellid(MP.qbin(np.log(np.minimum(gp1, gp2)), 8),
                    MP.qbin(np.log(np.maximum(gp1, gp2)), 8),
                    MP.qbin(np.minimum(np.abs(x["geta_p"]), np.abs(x["geta_m"])), 6),
                    MP.qbin(np.maximum(np.abs(x["geta_p"]), np.abs(x["geta_m"])), 6))
    nc = int(cid.max()) + 1
    ls = np.log(np.minimum(sig, 5. * np.median(srel) * mg))
    lsc = ls - SP.cellmean(ls, cid, nc)
    yc = z - SP.cellmean(z, cid, nc)
    xs, am, ac = [], [], []
    for k in range(nb):
        mm = (b == k) & m5
        xs.append(srel[mm].mean())
        am.append(float((lsc[mm] * yc[mm]).sum() / (yc[mm] ** 2).sum()))
        ac.append(float(aclosed[mm].mean()))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(xs, am, "o-", color="k", lw=2, label=r"measured $a_m$ (in-cell slope)")
    ax.plot(xs, ac, "s--", color="tab:blue", lw=2,
            label=r"closed form $(2f_{\rm hit}+f_{\rm ms})\,\sigma_m/m$")
    ax.set_xlabel(r"$\sigma_m/m$")
    ax.set_ylabel(r"$a_m = d\ln\sigma_m/d\ln m \cdot \sigma_m/m$")
    ax.legend(fontsize=13); ax.set_title(a.label, fontsize=15)
    plot_tools.save_pdf_and_png(outdir, f"mass_a_closedform_{a.tag}", fig)
    plt.close(fig)

    # -------- panel 3: the toy alpha response
    if a.toyfits and os.path.exists(a.toyfits):
        rows = {}
        for line in open(a.toyfits):
            if "alpha" not in line:
                continue
            lab = line.split("[sigma")[0].strip()
            val = float(line.split("alpha =")[1].split("+-")[0])
            err = float(line.split("+-")[1].split("e-3")[0])
            rows[lab] = (val, err)
        def get(pat):
            for k, v in rows.items():
                if pat in k:
                    return v
            return None
        base = get("a=0.000 NAIVE")
        pts = [(0.0, base), (0.011, get("a=0.011 NAIVE")),
               (0.05, get("a=0.050 NAIVE"))]
        pts = [(u, v) for u, v in pts if v]
        if len(pts) >= 2 and base:
            fig, ax = plt.subplots(figsize=(8, 6))
            av = np.array([p[0] for p in pts])
            va = np.array([p[1][0] - base[0] for p in pts])
            ea = np.array([np.hypot(p[1][1], base[1]) for p in pts])
            ax.errorbar(av, va, yerr=ea, marker="o", color="k", lw=2,
                        label="toy, naive likelihood")
            for pat, c, mk, nm in (("a=0.011 CORRECTED", "tab:green", "s", None),
                                   ("a=0.050 CORRECTED", "tab:green", "s", None),
                                   ("a=0.011 TRUE", "tab:red", "^", None),
                                   ("a=0.050 TRUE", "tab:red", "^", None)):
                v = get(pat)
                if v:
                    aa = 0.011 if "0.011" in pat else 0.05
                    ax.errorbar([aa], [v[0] - base[0]],
                                yerr=[np.hypot(v[1], base[1])], marker=mk,
                                color=c, ls="none", ms=9,
                                label=("truth-free correction"
                                       if c == "tab:green" and aa == 0.011 else
                                       (r"true $\bar\sigma$" if aa == 0.011 else None)))
            F = float(pl["F"]); sb = 0.00880
            g = np.linspace(0, 0.055, 20)
            ax.plot(g, -F * sb * g * 1e3, color="tab:blue", ls="--", lw=2,
                    label=rf"$-a\,F\,\bar\sigma_{{\rm eff}}/M$, $F={F:.2f}$")
            ax.plot(g, -2. * sb * g * 1e3, color="tab:gray", ls=":", lw=2,
                    label=r"naive Gaussian ($F=2$)")
            ax.axhline(0, color="grey", lw=1)
            ax.set_xlabel(r"injected $a_m$")
            ax.set_ylabel(r"$\hat\alpha(a) - \hat\alpha(0)$  [$10^{-3}$]")
            ax.legend(fontsize=12)
            ax.set_title("toy from the real per-candidate CF models", fontsize=15)
            plot_tools.save_pdf_and_png(outdir, f"mass_toy_alpha_{a.tag}", fig)
            plt.close(fig)
    # -------- panel 4: alpha on the REAL candidates, naive vs corrected
    rf = os.path.join(_HERE, "out", "realfits.txt")
    if os.path.exists(rf):
        rows = []
        for line in open(rf):
            if "alpha" not in line:
                continue
            lab = line.split("[sigma")[0].strip()
            v = float(line.split("alpha =")[1].split("+-")[0])
            e = float(line.split("+-")[1].split("e-3")[0])
            rows.append((lab, v, e))
        order = ["GUN 260905d NAIVE", "GUN 260905d TRUTH sigma_bar",
                 "GUN 260905d CORRECTED a=(1+vgf)sigma/m",
                 "V3 260903x NAIVE",
                 "V3 260903x CORRECTED a=(1+vgf)sigma/m"]
        short = {order[0]: r"gun, naive", order[1]: r"gun, true $\bar\sigma$",
                 order[2]: r"gun, truth-free", order[3]: r"v3, naive",
                 order[4]: r"v3, truth-free"}
        pts = [(short[o], v, e) for o in order
               for (l, v, e) in rows if l == o]
        if not any(l == order[0] for l, _, _ in rows):
            pts = [(r"gun, naive", -0.00470, 0.01666)] + pts
        if pts:
            fig, ax = plt.subplots(figsize=(8, 6))
            yy = np.arange(len(pts))[::-1]
            cols = ["k" if "naive" in p[0] else
                    ("tab:red" if "true" in p[0] else "tab:green") for p in pts]
            for k, (nm, v, e) in enumerate(pts):
                ax.errorbar([v], [yy[k]], xerr=[e], marker="o", ms=9,
                            color=cols[k], lw=2)
            ax.set_yticks(yy); ax.set_yticklabels([p[0] for p in pts])
            ax.axvline(0, color="grey", lw=1)
            ax.set_xlabel(r"$\hat\alpha$  [$10^{-3}$]")
            ax.set_title("mass scale: the naive likelihood vs the corrected one",
                         fontsize=14)
            plot_tools.save_pdf_and_png(outdir, f"mass_alpha_real_{a.tag}", fig)
            plt.close(fig)
    logger.info(f"figures -> {outdir}")


if __name__ == "__main__":
    main()
