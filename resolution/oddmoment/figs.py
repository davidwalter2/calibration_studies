#!/usr/bin/env python3
"""Figures for the charge-odd q/p pull asymmetry and its origin.

One file per panel, under ~/public_html/ZMass/cvh/<date>_oddmoment/.
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.dirname(_HERE)
sys.path.insert(0, _RES)
sys.path.insert(0, _HERE)

import decomp as D                                            # noqa: E402
import pubhtml                                                # noqa: E402
from wums import logging, plot_tools                          # noqa: E402

hep.style.use(hep.style.ROOT)
logger = logging.setup_logger(__file__, 3, False)

TG = D.TG
UGRID = np.array([0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0])
TGRID = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0])


def cellsigma(d, npbin=20, netabin=20):
    eta = d["eta"].astype(np.float64)
    pgen = d["genpt"] * np.cosh(eta)
    nvh = d["nvalidhits"]
    ip = np.clip(np.digitize(np.log(pgen),
                 np.quantile(np.log(pgen), np.linspace(0, 1, npbin + 1))[1:-1]),
                 0, npbin - 1).astype(np.int64)
    ie = np.clip(np.digitize(np.abs(eta),
                 np.quantile(np.abs(eta), np.linspace(0, 1, netabin + 1))[1:-1]),
                 0, netabin - 1).astype(np.int64)
    ih = np.clip(nvh.astype(np.int64) - int(nvh.min()), 0, 30)
    import sigma_pull as SP
    cid = SP.cellid(ip, ie, ih)
    nc = int(cid.max()) + 1
    return SP.cellmean(d["sigma"], cid, nc, leave_one_out=True), cid, nc


def AS_split(v, q, sel=None):
    if sel is None:
        sel = np.ones(len(v), bool)
    vp = v[sel & (q > 0)]; vm = v[sel & (q < 0)]
    return 0.5 * (vp.mean() - vm.mean()), 0.5 * (vp.mean() + vm.mean())


def err_split(v, q, sel=None):
    if sel is None:
        sel = np.ones(len(v), bool)
    vp = v[sel & (q > 0)]; vm = v[sel & (q < 0)]
    return 0.5 * np.hypot(vp.std(ddof=1) / np.sqrt(len(vp)),
                          vm.std(ddof=1) / np.sqrt(len(vm)))


def mobius(zg, pmod, a):
    """Density of zh = x/(1+a x) given the density of x (both folded)."""
    x = zg / (1. - a * zg)
    jac = 1. / (1. - a * zg) ** 2
    return np.interp(x, zg, pmod, left=0., right=0.) * jac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--krad", type=float, default=1.0)
    ap.add_argument("--outdir", default="")
    a = ap.parse_args()

    import datetime
    outdir = a.outdir or os.path.expanduser(
        "~/public_html/ZMass/cvh/%s_oddmoment" % datetime.date.today().strftime("%y%m%d"))
    os.makedirs(outdir, exist_ok=True)
    pubhtml.ensure_index(outdir, logger=logger)

    args = D.Args(); args.krad = a.krad
    d = D.load(a.cache)
    z = d["z"]; sig = d["sigma"]
    q = np.sign(d["charge"]).astype(np.float64)
    zh = q * z
    sbar, cid, nc = cellsigma(d)
    zb = z * sig / sbar
    zbh = q * zb
    n = len(z)
    logger.info(f"{n} tracks, {nc} sigma cells")

    # measured self-consistency coefficient a (sigma_pull T3)
    import sigma_pull as SP
    ls = np.log(sig)
    lsc = ls - SP.cellmean(ls, cid, nc)
    yc = zh - SP.cellmean(zh, cid, nc)
    sel5 = np.abs(z) < 5.
    acoef = float((lsc[sel5] * yc[sel5]).sum() / (yc[sel5] ** 2).sum())
    logger.info(f"a = d ln sigma / d(q z) = {acoef:+.5f}")

    # model, per track and pooled per charge
    pool = np.where(q > 0, 0, 1)
    Om, Em, phis, cnt = D.model_per_track(d, args, tuple(UGRID),
                                          pool_id=pool, npool=2)
    phifold = 0.5 * (phis[0] + np.conj(phis[1]))

    # ------------------------------------------------------- panel: vs trim
    fig, ax = plt.subplots(figsize=(8, 6))
    for v, base, nm, c, mk in (
            (zh, z, r"data, $z=\Delta(q/p)/\sigma_{\rm fit}$", "k", "o"),
            (zbh, zb,
             r"data, $\bar z=\Delta(q/p)/\bar\sigma(p_{\rm gen},\eta,n_{\rm hit})$",
             "tab:red", "s")):
        # v is ALREADY folded (v = q z), so A is simply its mean in the window
        y, e = [], []
        for T in TGRID:
            s_ = np.abs(base) < T
            y.append(float(v[s_].mean()))
            e.append(float(v[s_].std(ddof=1) / np.sqrt(s_.sum())))
        ax.errorbar(TGRID, y, yerr=e, marker=mk, color=c, label=nm, lw=2)
    zg = np.linspace(-12, 12, 4801)
    pmod = D.density(phifold, zg)
    ymod = [float(np.trapezoid(zg[np.abs(zg) < T] * pmod[np.abs(zg) < T],
                               zg[np.abs(zg) < T])
                  / np.trapezoid(pmod[np.abs(zg) < T], zg[np.abs(zg) < T]))
            for T in TGRID]
    ax.plot(TGRID, ymod, color="tab:blue", lw=2, ls="--", label="CF model (ioni+rad)")
    pmap = mobius(zg, pmod, acoef)
    pmap /= np.trapezoid(pmap, zg)
    ymap = [float(np.trapezoid(zg[np.abs(zg) < T] * pmap[np.abs(zg) < T],
                               zg[np.abs(zg) < T])
                  / np.trapezoid(pmap[np.abs(zg) < T], zg[np.abs(zg) < T]))
            for T in TGRID]
    ax.plot(TGRID, ymap, color="tab:green", lw=2, ls=":",
            label=rf"model $\otimes$ $\sigma$-map, $a={acoef:.4f}$")
    ax.axhline(0, color="grey", lw=1)
    ax.set_xlabel(r"trim $T$   ($|z| < T$)")
    ax.set_ylabel(r"$A = \langle q\,z\rangle_{|z|<T}$")
    ax.set_xscale("log")
    ax.legend(fontsize=13)
    ax.set_title(a.label, fontsize=15)
    plot_tools.save_pdf_and_png(outdir, f"A_vs_trim_{a.tag}", fig)
    plt.close(fig)

    # ---------------------------------------------------------- panel: vs u
    fig, ax = plt.subplots(figsize=(8, 6))
    yd = [AS_split(z * np.exp(-u * z ** 2), q)[0] for u in UGRID]
    ed = [err_split(z * np.exp(-u * z ** 2), q) for u in UGRID]
    yb = [AS_split(zb * np.exp(-u * zb ** 2), q)[0] for u in UGRID]
    eb = [err_split(zb * np.exp(-u * zb ** 2), q) for u in UGRID]
    ym = [0.5 * (Om[q > 0, i].mean() - Om[q < 0, i].mean()) for i in range(len(UGRID))]
    ax.errorbar(UGRID, yd, yerr=ed, marker="o", color="k", lw=2,
                label=r"data, $z$ (fit $\sigma$)")
    ax.errorbar(UGRID, yb, yerr=eb, marker="s", color="tab:red", lw=2,
                label=r"data, $\bar z$ (truth-only $\bar\sigma$)")
    ax.plot(UGRID, ym, color="tab:blue", ls="--", lw=2, label="CF model")
    ax.axhline(0, color="grey", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"$u$")
    ax.set_ylabel(r"$A = \langle q\,z\,e^{-u z^2}\rangle$")
    ax.legend(fontsize=13)
    ax.set_title(a.label, fontsize=15)
    plot_tools.save_pdf_and_png(outdir, f"A_vs_u_{a.tag}", fig)
    plt.close(fig)

    # -------------------------------------------------- panel: zhat density
    edges = np.linspace(-5, 5, 201)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    h, _ = np.histogram(zh, bins=edges)
    hd = h / (n * (edges[1] - edges[0]))
    hde = np.sqrt(h) / (n * (edges[1] - edges[0]))
    mmod = np.interp(ctr, zg, pmod)
    mmap = np.interp(ctr, zg, pmap)
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                            gridspec_kw={"height_ratios": [2.2, 1]})
    axs[0].errorbar(ctr, hd, yerr=hde, fmt=".", color="k", ms=3, label="data")
    axs[0].plot(ctr, mmod, color="tab:blue", lw=2, ls="--", label="CF model")
    axs[0].plot(ctr, mmap, color="tab:green", lw=2, ls=":",
                label=rf"model $\otimes$ $\sigma$-map ($a={acoef:.4f}$)")
    axs[0].set_yscale("log")
    axs[0].set_ylabel(r"$p(\hat z)$,  $\hat z = q\,z$")
    axs[0].legend(fontsize=13)
    axs[0].set_title(a.label, fontsize=15)
    with np.errstate(divide="ignore", invalid="ignore"):
        axs[1].errorbar(ctr, hd / mmod, yerr=hde / mmod, fmt=".", color="k", ms=3)
        axs[1].plot(ctr, mmap / mmod, color="tab:green", lw=2, ls=":")
    axs[1].axhline(1, color="tab:blue", lw=2, ls="--")
    axs[1].set_ylim(0.9, 1.1)
    axs[1].set_ylabel("data / model")
    axs[1].set_xlabel(r"$\hat z = q\,(q/p_{\rm fit}-q/p_{\rm gen})/\sigma$")
    plot_tools.save_pdf_and_png(outdir, f"zhat_density_{a.tag}", fig)
    plt.close(fig)

    # ---------------------------------------------------- panel: vs sigma
    for var, vname, xlab in ((sig, "sigma", r"$\sigma_{\rm fit}(q/p)$ quantile"),
                             (d["vgf"], "vgf",
                              r"$v_{\rm gauss}/\sigma^2$ quantile")):
        nb = 8
        e = np.quantile(var, np.linspace(0, 1, nb + 1))
        b = np.clip(np.digitize(var, e[1:-1]), 0, nb - 1)
        eb2 = np.quantile(sbar, np.linspace(0, 1, nb + 1))
        bb = np.clip(np.digitize(sbar, eb2[1:-1]), 0, nb - 1)
        xs = np.arange(nb) + 0.5
        yd, ed_, ym2, yb2, eb3 = [], [], [], [], []
        for k in range(nb):
            m1 = (b == k) & sel5
            yd.append(AS_split(z, q, m1)[0]); ed_.append(err_split(z, q, m1))
            ym2.append(0.5 * (Om[(b == k) & (q > 0), 2].mean()
                              - Om[(b == k) & (q < 0), 2].mean()))
            m2 = (bb == k) & (np.abs(zb) < 5.)
            yb2.append(AS_split(zb, q, m2)[0]); eb3.append(err_split(zb, q, m2))
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.errorbar(xs, yd, yerr=ed_, marker="o", color="k", lw=2,
                    label=r"data, $z$, in bins of $\sigma_{\rm fit}$"
                          if vname == "sigma" else r"data, $z$")
        ax.errorbar(xs, yb2, yerr=eb3, marker="s", color="tab:red", lw=2,
                    label=r"data, $\bar z$, in bins of $\bar\sigma$")
        ax.plot(xs, ym2, color="tab:blue", ls="--", lw=2, label=r"CF model ($u=0.05$)")
        ax.axhline(0, color="grey", lw=1)
        ax.set_xlabel(xlab)
        ax.set_ylabel(r"$A = \langle q z\rangle_{|z|<5}$")
        ax.legend(fontsize=13)
        ax.set_title(a.label, fontsize=15)
        plot_tools.save_pdf_and_png(outdir, f"A_vs_{vname}_{a.tag}", fig)
        plt.close(fig)

    # ----------------------------------------- panel: relative momentum bias
    eta = d["eta"].astype(np.float64)
    pgen = d["genpt"] * np.cosh(eta)
    rel = q * (z * sig) * pgen                # = <p_gen/p_fit> - 1
    for var, vname, xlab, ed2 in (
            (eta, "eta", r"$\eta$", np.linspace(-2.4, 2.4, 13)),
            (d["genpt"], "genpt", r"$p_T^{\rm gen}$ [GeV]",
             np.quantile(d["genpt"], np.linspace(0, 1, 9)))):
        nb = len(ed2) - 1
        b = np.clip(np.digitize(var, ed2[1:-1]), 0, nb - 1)
        ctr2 = 0.5 * (ed2[1:] + ed2[:-1])
        yd, ed3, ym2 = [], [], []
        for k in range(nb):
            m1 = (b == k) & sel5
            yd.append(rel[m1].mean())
            ed3.append(rel[m1].std(ddof=1) / np.sqrt(m1.sum()))
            srel_k = (sig * pgen)[m1].mean()
            ym2.append(0.5 * (Om[(b == k) & (q > 0), 2].mean()
                              - Om[(b == k) & (q < 0), 2].mean()) * srel_k)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.errorbar(ctr2, 1e4 * np.array(yd), yerr=1e4 * np.array(ed3),
                    marker="o", color="k", lw=2, label=r"data (raw, no $1/\sigma$)")
        ax.plot(ctr2, 1e4 * np.array(ym2), color="tab:blue", ls="--", lw=2,
                label="CF model")
        ax.axhline(0, color="grey", lw=1)
        ax.set_xlabel(xlab)
        ax.set_ylabel(r"$\langle p_{\rm gen}/p_{\rm fit}\rangle - 1$  [$10^{-4}$]")
        ax.legend(fontsize=13)
        ax.set_title(a.label, fontsize=15)
        plot_tools.save_pdf_and_png(outdir, f"dpp_vs_{vname}_{a.tag}", fig)
        plt.close(fig)

    logger.info(f"figures -> {outdir}")


if __name__ == "__main__":
    main()
