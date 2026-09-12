#!/usr/bin/env python3
"""Figures for the hit-class LOCATION study. One file per panel."""
import argparse
import datetime
import os
import sys

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pubhtml                                                    # noqa: E402
import ratiopanel                                                 # noqa: E402
import hitres_classes as HC                                       # noqa: E402
from wums import logging                                          # noqa: E402

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)
SDN = {1: "BPix", 2: "FPix", 3: "TIB", 4: "TID", 5: "TOB", 6: "TEC"}


def save(fig, outdir, name):
    pubhtml.savefig(fig, os.path.join(outdir, f"{name}.pdf"))
    plt.close(fig)
    logger.info(f"  {name}")


def dens(ax, rax, v, ref, lab, col, bins):
    c = 0.5 * (bins[1:] + bins[:-1])
    w = np.diff(bins)
    hh, _ = np.histogram(np.clip(v, bins[0], bins[-1]), bins=bins)
    n = hh / hh.sum() / w
    ax.step(c, n, where="mid", color=col, label=lab)
    hr, _ = np.histogram(np.clip(ref, bins[0], bins[-1]), bins=bins)
    nr = hr / hr.sum() / w
    ok = (nr > 0) & (hh > 20)
    rax.errorbar(c[ok][1:-1], (n / nr)[ok][1:-1],
                 yerr=((n / nr) / np.sqrt(np.maximum(hh, 1)))[ok][1:-1],
                 fmt="o", ms=3, color=col)
    return nr, c, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    day = datetime.date.today().strftime("%y%m%d")
    outdir = a.outdir or os.path.expanduser(
        f"~/public_html/ZMass/cvh/{day}_hitclassbias")
    os.makedirs(outdir, exist_ok=True)
    pubhtml.ensure_index(outdir, logger=logger)
    logger.info(f"writing to {outdir}")

    h = np.load("data/hits_mugun_ul16_h2.npz")
    b = np.load("data/blocks_mugun_ul16_260903x.npz")
    p = np.load("data/pred_track.npz")
    px = h["pullx"].astype(np.float64)
    py = h["pully"].astype(np.float64)
    okx = np.isfinite(px) & (np.abs(px) < 10)
    oky = np.isfinite(py) & (np.abs(py) < 10)
    aw = b["b_s"].astype(np.float64) * np.sqrt(b["b_a2"].astype(np.float64))
    trk = np.repeat(np.arange(len(b["nblk"])), b["nblk"])
    band = np.digitize(np.abs(b["t_eta"]), [0.9, 1.6])
    nb = np.bincount(band, minlength=3)

    # --- 1. location per subdetector ---------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    xs, ys, es, lb = [], [], [], []
    for i, sd in enumerate(range(1, 7)):
        m = okx & (h["sd"] == sd)
        xs.append(i)
        ys.append(px[m].mean())
        es.append(px[m].std() / np.sqrt(m.sum()))
        lb.append(SDN[sd] + r" $x/\varphi$")
    for j, sd in enumerate((1, 2)):
        m = oky & (h["sd"] == sd)
        xs.append(6 + j)
        ys.append(py[m].mean())
        es.append(py[m].std() / np.sqrt(m.sum()))
        lb.append(SDN[sd] + " $y$")
    ax.errorbar(xs, np.array(ys), yerr=es, fmt="o", ms=8, color="k")
    ax.axhline(0, color="0.6", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(lb, rotation=30, ha="right")
    ax.set_ylabel(r"mean $(\mathrm{rec}-\mathrm{sim})/\sigma_{\rm CPE}$")
    ax.set_title(r"per-hit residual LOCATION, ideal geometry, $\mu$ gun 20-60 GeV",
                 fontsize=15)
    save(fig, outdir, "loc_subdet")

    # --- 2. the 18 canonical classes ---------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    for c in range(18):
        isy = 4 <= c < 8
        pp, mm = (py, oky & (h["cls18y"] == c)) if isy else (px, okx & (h["cls18x"] == c))
        if mm.sum() < 300:
            continue
        v = pp[mm]
        ax.errorbar([c], [v.mean()], yerr=[v.std() / np.sqrt(len(v))],
                    fmt="s", ms=7, color="tab:red" if c < 8 else "tab:blue")
    ax.axhline(0, color="0.6", lw=1)
    ax.set_xticks(range(18))
    ax.set_xticklabels(HC.CLASSES, rotation=60, ha="right", fontsize=11)
    ax.set_ylabel(r"mean pull (the location `hitres_classes` discards)")
    ax.set_title("the 18 classes: their LOCATION, not their width", fontsize=15)
    save(fig, outdir, "loc_cls18")

    # --- 3. BPix density vs a unit Gaussian, with ratio ---------------------
    for tag, sel, pp in (("bpix_x", okx & (h["sd"] == 1), px),
                         ("fpix_x", okx & (h["sd"] == 2), px),
                         ("tib_x", okx & (h["sd"] == 3), px),
                         ("tec_x", okx & (h["sd"] == 6), px)):
        v = pp[sel]
        bins = np.linspace(-5, 5, 101)
        c = 0.5 * (bins[1:] + bins[:-1])
        fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(9, 7))
        hh, _ = np.histogram(np.clip(v, bins[0], bins[-1]), bins=bins)
        n = hh / hh.sum() / np.diff(bins)
        g = np.exp(-0.5 * c ** 2) / np.sqrt(2 * np.pi)
        ax.step(c, n, where="mid", color="k", label=f"{tag} data")
        ax.plot(c, g, color="tab:red", label="unit Gaussian at 0")
        ax.axvline(v.mean(), color="tab:blue", ls="--",
                   label=f"mean {v.mean():+.4f}")
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1)
        ax.legend(fontsize=12)
        ax.set_ylabel("density")
        ok = hh > 20
        rax.errorbar(c[ok][1:-1], (n / g)[ok][1:-1],
                     yerr=((n / g) / np.sqrt(hh))[ok][1:-1], fmt="o", ms=3,
                     color="k")
        rax.axhline(1, color="tab:red")
        rax.set_ylim(0.5, 1.6)
        rax.set_ylabel("data / Gauss")
        rax.set_xlabel(r"$(\mathrm{rec}-\mathrm{sim})/\sigma_{\rm CPE}$")
        save(fig, outdir, f"dens_{tag}")

    # --- 4. degraded classes vs the rest ------------------------------------
    fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(9, 7))
    bins = np.linspace(-4, 4, 81)
    base = px[okx & (h["sd"] >= 3) & (h["N"] == 2)]
    for lab, m, col in (
            (r"strip $N=1$", okx & (h["sd"] >= 3) & (h["N"] == 1), "tab:red"),
            (r"strip $N=2$ (ref)", okx & (h["sd"] >= 3) & (h["N"] == 2), "k"),
            (r"strip $N\geq4$", okx & (h["sd"] >= 3) & (h["N"] >= 4), "tab:blue"),
            (r"pixel $q$bin 3", okx & (h["sd"] <= 2) & (h["qbin"] == 3), "tab:green")):
        dens(ax, rax, px[m], base, lab + f"  $\\langle p\\rangle$="
             f"{px[m].mean():+.3f}", col, bins)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)
    ax.legend(fontsize=11)
    ax.set_ylabel("density")
    rax.axhline(1, color="0.5")
    rax.set_ylim(0.4, 2.0)
    rax.set_ylabel(r"/ strip $N=2$")
    rax.set_xlabel(r"$(\mathrm{rec}-\mathrm{sim})/\sigma_{\rm CPE}$")
    save(fig, outdir, "dens_degraded")

    # --- 5. location vs SIGNED local angle ----------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    E = np.array([-0.3, -0.2, -0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1, 0.2, 0.3])
    for sd, col in ((1, "tab:red"), (3, "tab:blue"), (5, "k"), (6, "tab:green")):
        xs, ys, es = [], [], []
        for i in range(len(E) - 1):
            m = okx & (h["sd"] == sd) & (h["dxdz"] >= E[i]) & (h["dxdz"] < E[i + 1])
            if m.sum() < 400:
                continue
            xs.append(0.5 * (E[i] + E[i + 1]))
            ys.append(px[m].mean())
            es.append(px[m].std() / np.sqrt(m.sum()))
        ax.errorbar(xs, ys, yerr=es, fmt="o-", ms=5, color=col, label=SDN[sd])
    ax.axhline(0, color="0.6", lw=1)
    ax.axvline(0, color="0.6", lw=1)
    ax.legend(fontsize=12)
    ax.set_xlabel(r"local $\mathrm{d}x/\mathrm{d}z$ (signed)")
    ax.set_ylabel("mean pull")
    ax.set_title("the location is EVEN in the incidence angle, not odd",
                 fontsize=15)
    save(fig, outdir, "loc_vs_dxdz")

    # --- 6. the lever arm ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    wid = 0.25
    for ib, (nm, col) in enumerate((("barrel", "tab:red"), ("middle", "tab:blue"),
                                    ("endcap", "k"))):
        xs, ys = [], []
        for i, sd in enumerate(range(1, 7)):
            m = (b["b_sd"] == sd) & (b["b_isy"] == 0) & (band[trk] == ib)
            xs.append(i + (ib - 1) * wid)
            ys.append(aw[m].sum() / nb[ib])
        ax.bar(xs, ys, width=wid, color=col, label=nm)
    ax.axhline(0, color="0.4", lw=1)
    ax.set_xticks(range(6))
    ax.set_xticklabels([SDN[s] for s in range(1, 7)])
    ax.set_ylabel(r"$\langle\sum_b s_b a_b\rangle$  (bending-sense lever arm)")
    ax.legend(fontsize=12)
    ax.set_title("what a unit per-hit bias in that subdetector would do",
                 fontsize=15)
    save(fig, outdir, "lever_subdet")

    # --- 7. the headline ----------------------------------------------------
    U = 0.05
    zt = p["zt"].astype(np.float64)
    good = p["good"]
    q = p["q"].astype(np.float64)
    f_ = zt * np.exp(-U * zt * zt)
    pred = 0.8668 * p["dz_mean"] - 0.0394 * p["dz_k3"]
    rng = np.random.default_rng(3)
    fig, ax = plt.subplots(figsize=(9, 6))
    xs = np.arange(3)
    mv, me, pv = [], [], []
    for ib in range(3):
        m = good & (band == ib)
        v = 0.5 * (f_[m & (q > 0)].mean() + f_[m & (q < 0)].mean())
        idx = np.where(m)[0]
        bs = np.empty(200)
        for i in range(200):
            k = idx[rng.integers(0, len(idx), len(idx))]
            bs[i] = 0.5 * (f_[k][q[k] > 0].mean() + f_[k][q[k] < 0].mean())
        mv.append(v * 1e3)
        me.append(bs.std() * 1e3)
        pv.append(pred[m].mean() * 1e3)
    ax.errorbar(xs, mv, yerr=me, fmt="o", ms=10, color="k",
                label="measured (charge-even)")
    ax.plot(xs, pv, "s--", ms=10, color="tab:red",
            label="predicted from the class locations\n(no free parameter)")
    ax.axhline(0, color="0.6", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([r"$|\eta|<0.9$", r"$0.9-1.6$", r"$1.6-2.4$"])
    ax.set_ylabel(r"$\langle z e^{-0.05 z^2}\rangle \times 10^{3}$")
    ax.legend(fontsize=12)
    ax.set_title("the class LOCATION cannot make the observed odd moment",
                 fontsize=15)
    save(fig, outdir, "pred_vs_meas")
    logger.info(f"done -> {outdir}")


if __name__ == "__main__":
    main()
