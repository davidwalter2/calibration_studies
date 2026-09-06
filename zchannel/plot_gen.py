#!/usr/bin/env python3
"""Figures for the generator-level closure of the Z/gamma* lineshape kernel.

One file per panel, ratio panel under every density-vs-model plot, mplhep ROOT
style, output under ~/public_html/cvh/<date>_zgen/ with the index.php gallery.
"""
import argparse
import datetime
import json
import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from wums import logging

import pubhtml
import ratiopanel

import fit_gen as FG

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)


def save(fig, outdir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"-> {os.path.join(outdir, name)}.{{pdf,png}}")


def model_density(z, vals, m, shape=None, window=None):
    """The model density (normalised over `window`) evaluated at masses `m`."""
    import tensorflow as tf
    y = z.born_pdf(vals)
    if shape is not None and len(shape):
        uu = (2.0 * (z.m_born - window[0]) / (window[1] - window[0]) - 1.0)
        from numpy.polynomial import legendre
        basis = np.stack([legendre.legval(uu, [0] * k + [1])
                          for k in range(1, len(shape) + 1)])
        y = y * tf.constant(np.exp(np.tensordot(np.asarray(shape), basis, axes=1)),
                            z.dtype)
    p = (z.fold_fsr(y) * z._edge).numpy()
    g = z.m_grid
    sl = (g >= window[0]) & (g <= window[1])
    p = p / (p[sl].sum() * z.dm)
    return np.interp(m, g, p)


def weighted_hist(m, w, edges):
    i = np.clip(np.searchsorted(edges, m, "right") - 1, 0, len(edges) - 2)
    ok = (m >= edges[0]) & (m < edges[-1])
    n = len(edges) - 1
    return (np.bincount(i[ok], w[ok], n), np.bincount(i[ok], w[ok] ** 2, n))


def spectrum_fig(outdir, name, m, w, z, vals, window, title, xlabel,
                 shape=None, nbin=240, logy=True, ratio_clamp=(0.9, 1.1)):
    edges = np.linspace(window[0], window[1], nbin + 1)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    bw = np.diff(edges)
    W, W2 = weighted_hist(np.asarray(m, float), np.asarray(w, float), edges)
    tot = W.sum()
    dens = W / (tot * bw)
    err = np.sqrt(W2) / (tot * bw)
    mod = ratiopanel.bin_average(model_density(z, vals, edges[:-1], shape, window),
                                 model_density(z, vals, ctr, shape, window),
                                 model_density(z, vals, edges[1:], shape, window))
    fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(9.5, 7.4))
    ax.errorbar(ctr, dens, yerr=err, fmt="o", ms=2.6, color="black",
                lw=0, elinewidth=0.9, label="generated")
    ax.plot(ctr, mod, color="C3", lw=1.8, label="ZGammaLineshape")
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(r"$(1/N)\,\mathrm{d}N/\mathrm{d}m$  [GeV$^{-1}$]")
    ax.legend(loc="upper right", fontsize=15, frameon=False)
    ax.set_title(title, fontsize=15, loc="left")
    r = dens / mod
    re = err / mod
    rax.errorbar(ctr, r, yerr=re, fmt="o", ms=2.6, color="black", lw=0,
                 elinewidth=0.9)
    rax.axhline(1.0, color="C3", lw=1.2)
    good = np.isfinite(r) & (re < 0.05)
    if good.sum() > 3:
        half = max(np.max(np.abs(r[good] - 1) + re[good]) * 1.25, 2e-3)
        rax.set_ylim(1 - half, 1 + half)
    else:
        rax.set_ylim(*ratio_clamp)
    rax.set_ylabel("gen / model", fontsize=15)
    rax.set_xlabel(xlabel)
    rax.grid(alpha=0.25)
    save(fig, outdir, name)
    return ctr, r, re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="data/genmerged.npz")
    ap.add_argument("--fits", default="data/fit_prefsr.json")
    ap.add_argument("--fits-post", default=None)
    ap.add_argument("--fits-fid", default=None)
    ap.add_argument("--kernel", default=None)
    ap.add_argument("--kernel-fid", default=None)
    ap.add_argument("--acc", default=None)
    ap.add_argument("--acc-pt", type=float, default=25.0)
    ap.add_argument("--acc-eta", type=float, default=2.4)
    ap.add_argument("--nm", type=int, default=32768)
    ap.add_argument("--outpath", default=None)
    args = ap.parse_args()

    import tensorflow as tf
    outdir = args.outpath or os.path.expanduser(
        "~/public_html/cvh/" + datetime.date.today().strftime("%y%m%d") + "_zgen")
    os.makedirs(outdir, exist_ok=True)
    pubhtml.ensure_index(outdir, logger=logger)

    g = FG.load_gen(args.gen)
    w = FG.clip_weights(g["weight"].astype(np.float64))
    def load_fits(path):
        if path and os.path.exists(path) and os.path.getsize(path) > 2:
            with open(path) as fh:
                return json.load(fh)
        return {}

    fits = load_fits(args.fits)
    fpost = load_fits(args.fits_post)
    ffid = load_fits(args.fits_fid)

    def shape_of(entry, n=5):
        return [entry[f"shape{k}"][0] for k in range(1, n + 1)
                if f"shape{k}" in entry]

    def prov(**kw):
        kw.setdefault("window", (50.0, 130.0))
        kw.setdefault("nm", args.nm)
        return FG.make_provider(
            tf, kw["window"], kw["nm"], kw.get("width_scheme", "fixed"),
            kw.get("lumi", "nnpdf31_nnlo_13tev"),
            kw.get("terms", ("gamma", "int", "z")),
            kw.get("fsr"), kw.get("acc"),
            mz_ref=FG.MZ_FIXED, gz_ref=FG.GZ_FIXED)

    V0 = {"m_Z": tf.constant(0.0, tf.float64), "Gamma_Z": tf.constant(0.0, tf.float64)}

    # -- 1. the Born spectrum at the generator's own parameters -------------
    z = prov(window=(50.0, 160.0))
    spectrum_fig(outdir, "01_born_truth", g["m_pre"], w, z, V0, (55.0, 150.0),
                 r"pre-FSR $m_{\mu\mu}$, model at POWHEG's own $m_Z,\Gamma_Z$ "
                 "(no fit)", r"$m_{\mu\mu}^{\rm pre-FSR}$  [GeV]")

    # -- 2. the same, fitted ------------------------------------------------
    f = fits.get("baseline 60-120")
    if f:
        vf = {"m_Z": tf.constant(f["m_Z"][0], tf.float64),
              "Gamma_Z": tf.constant(f["Gamma_Z"][0], tf.float64)}
        z2 = prov(window=(50.0, 130.0))
        spectrum_fig(outdir, "02_born_fit", g["m_pre"], w, z2, vf, (60.0, 120.0),
                     r"pre-FSR $m_{\mu\mu}$, 2-parameter fit "
                     rf"($\Delta m_Z$ = {f['m_Z'][0]:+.2f} MeV, "
                     rf"$\Delta\Gamma_Z$ = {f['Gamma_Z'][0]:+.2f} MeV)",
                     r"$m_{\mu\mu}^{\rm pre-FSR}$  [GeV]")

    # -- 3. with the smooth shape nuisance ---------------------------------
    for n in (3, 5):
        f = fits.get(f"+ shape nuisance, {n} Legendre")
        if not f:
            continue
        vf = {"m_Z": tf.constant(f["m_Z"][0], tf.float64),
              "Gamma_Z": tf.constant(f["Gamma_Z"][0], tf.float64)}
        sh = [f[f"shape{k}"][0] for k in range(1, n + 1)]
        z3 = prov(window=(50.0, 130.0))
        spectrum_fig(outdir, f"03_born_shape{n}", g["m_pre"], w, z3, vf,
                     (60.0, 120.0),
                     rf"pre-FSR $m_{{\mu\mu}}$ + {n}-term smooth shape "
                     rf"($\Delta m_Z$ = {f['m_Z'][0]:+.2f} MeV, "
                     rf"$\Delta\Gamma_Z$ = {f['Gamma_Z'][0]:+.2f} MeV)",
                     r"$m_{\mu\mu}^{\rm pre-FSR}$  [GeV]", shape=sh)

    # -- 4. the FSR kernel --------------------------------------------------
    if args.kernel and os.path.exists(args.kernel):
        d = np.load(args.kernel)
        r, kw_ = d["r"], d["w"]
        fig, ax = plt.subplots(figsize=(9.0, 6.4))
        cont = r < 1.0 - 1e-9
        u = -np.log(r[cont])
        du = np.gradient(np.sort(u))
        o = np.argsort(u)
        ax.plot(u[o], (kw_[cont][o] / du), lw=1.6, color="C0",
                label=r"continuous part, $u=-\ln(m_{\rm post}/m_{\rm pre})$")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel(r"$u = -\ln(m^{\rm post}_{\mu\mu}/m^{\rm pre}_{\mu\mu})$")
        ax.set_ylabel(r"$\mathrm{d}P/\mathrm{d}u$")
        ax.set_title(f"multiplicative FSR kernel, {len(r)} atoms, "
                     f"P(no radiation) = {kw_[r >= 1 - 1e-9].sum():.4f}",
                     fontsize=15, loc="left")
        ax.legend(fontsize=15, frameon=False)
        save(fig, outdir, "04_fsr_kernel")

    # -- 5. m-dependence of the kernel -------------------------------------
    sel_all = np.isfinite(g["m_pre"])
    fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(9.5, 7.0))
    ref = None
    for lo, hi, c in ((60, 80, "C0"), (86, 96, "C1"), (96, 120, "C2"),
                      (120, 200, "C3")):
        s = (g["m_pre"] >= lo) & (g["m_pre"] < hi)
        u = -np.log(np.clip(g["m_post"][s] / g["m_pre"][s], 1e-6, 1.0))
        e = np.logspace(-5, 0.2, 90)
        h, _ = np.histogram(u, bins=e, weights=w[s])
        h = h / h.sum() / np.diff(e)
        ctr = np.sqrt(e[1:] * e[:-1])
        ax.plot(ctr, h, lw=1.5, color=c, label=rf"$m_{{\rm pre}}\in[{lo},{hi})$")
        if ref is None:
            ref = h
        else:
            rax.plot(ctr, h / ref, lw=1.5, color=c)
    rax.axhline(1.0, color="C0", lw=1.2)
    ax.set_xscale("log"); ax.set_yscale("log")
    rax.set_xscale("log")
    ax.set_ylabel(r"$\mathrm{d}P/\mathrm{d}u$")
    rax.set_ylabel("ratio to 60-80", fontsize=14)
    rax.set_xlabel(r"$u = -\ln(m^{\rm post}/m^{\rm pre})$")
    rax.set_ylim(0.5, 1.5); rax.grid(alpha=0.25)
    ax.legend(fontsize=14, frameon=False)
    ax.set_title("is the FSR kernel multiplicative? (shape of $u$ vs $m_{\\rm pre}$)",
                 fontsize=15, loc="left")
    save(fig, outdir, "05_fsr_kernel_mdep")

    # -- 6. post-FSR spectrum with the folded model ------------------------
    if args.kernel and os.path.exists(args.kernel):
        pf = fpost.get("FSR folded, shape 5")
        vf = V0 if not pf else {
            "m_Z": tf.constant(pf["m_Z"][0], tf.float64),
            "Gamma_Z": tf.constant(pf["Gamma_Z"][0], tf.float64)}
        zf = prov(window=(50.0, 130.0), fsr=args.kernel)
        ttl = ("post-FSR $m_{\\mu\\mu}$, model = Born $\\times K(m)$ "
               "$\\otimes$ FSR fold")
        if pf:
            ttl += (rf" ($\Delta m_Z$ = {pf['m_Z'][0]:+.2f}, "
                    rf"$\Delta\Gamma_Z$ = {pf['Gamma_Z'][0]:+.2f} MeV)")
        spectrum_fig(outdir, "06_postfsr_folded", g["m_post"], w, zf, vf,
                     (60.0, 120.0), ttl, r"$m_{\mu\mu}^{\rm post-FSR}$  [GeV]",
                     shape=shape_of(pf) if pf else None)

    # -- 7. the acceptance --------------------------------------------------
    if args.acc and os.path.exists(args.acc):
        diag = np.load(args.acc.replace(".json", "_diag.npz"))
        with open(args.acc) as fh:
            cfg = json.load(fh)
        fig, ax, rax = ratiopanel.make_ratio_fig(figsize=(9.5, 7.0))
        gd = diag["good"]
        ax.errorbar(diag["m"][gd], diag["a"][gd], yerr=diag["sig"][gd], fmt="o",
                    ms=2.6, color="black", lw=0, elinewidth=0.9, label="generated")
        ax.plot(diag["m"][gd], diag["fit"][gd], color="C3", lw=1.8,
                label=f"Bernstein degree {len(cfg['coef'])-1}")
        ax.set_ylabel(r"$A(m) = P(\mathrm{selected}\,|\,m_{\rm pre})$")
        ax.legend(fontsize=15, frameon=False)
        ax.set_title(rf"acceptance, $p_T > {args.acc_pt:g}$ GeV, "
                     rf"$|\eta| < {args.acc_eta:g}$ on the post-FSR muons",
                     fontsize=15, loc="left")
        rr = diag["a"][gd] / diag["fit"][gd]
        rax.errorbar(diag["m"][gd], rr, yerr=diag["sig"][gd] / diag["fit"][gd],
                     fmt="o", ms=2.6, color="black", lw=0, elinewidth=0.9)
        rax.axhline(1.0, color="C3", lw=1.2)
        rax.set_ylim(0.97, 1.03)
        rax.set_ylabel("gen / fit", fontsize=15)
        rax.set_xlabel(r"$m_{\mu\mu}^{\rm pre-FSR}$  [GeV]")
        rax.grid(alpha=0.25)
        save(fig, outdir, "07_acceptance")

    # -- 8. fiducial post-FSR with A(m) folded ------------------------------
    if args.kernel_fid and args.acc and os.path.exists(args.kernel_fid):
        with open(args.acc) as fh:
            accfg = {k: v for k, v in json.load(fh).items() if not k.startswith("_")}
        sel = FG.fiducial(g, args.acc_pt, args.acc_eta, post=True)
        ff = ffid.get("FSR folded + A(m), shape 5")
        vf = V0 if not ff else {
            "m_Z": tf.constant(ff["m_Z"][0], tf.float64),
            "Gamma_Z": tf.constant(ff["Gamma_Z"][0], tf.float64)}
        za = prov(window=(50.0, 130.0), fsr=args.kernel_fid, acc=accfg)
        ttl = (r"fiducial post-FSR $m_{\mu\mu}$: Born $\times A(m) \times K(m)$ "
               r"$\otimes$ FSR fold")
        if ff:
            ttl += (rf" ($\Delta m_Z$ = {ff['m_Z'][0]:+.2f}, "
                    rf"$\Delta\Gamma_Z$ = {ff['Gamma_Z'][0]:+.2f} MeV)")
        spectrum_fig(outdir, "08_fiducial_folded", g["m_post"][sel], w[sel], za,
                     vf, (60.0, 120.0), ttl,
                     r"$m_{\mu\mu}^{\rm post-FSR}$  [GeV]",
                     shape=shape_of(ff) if ff else None)

    # -- 9. window dependence of the closure --------------------------------
    keys = [(k, v) for k, v in fits.items() if k.startswith("window ")
            or k == "baseline 60-120"]
    if keys:
        fig, ax = plt.subplots(figsize=(9.0, 6.4))
        lbl, mz, emz, gz, egz = [], [], [], [], []
        for k, v in keys:
            lbl.append(k.replace("window ", "").replace("baseline ", ""))
            mz.append(v["m_Z"][0]); emz.append(v["m_Z"][1])
            gz.append(v["Gamma_Z"][0]); egz.append(v["Gamma_Z"][1])
        x = np.arange(len(lbl))
        ax.errorbar(x - 0.08, mz, yerr=emz, fmt="o", color="C0",
                    label=r"$\Delta m_Z$")
        ax.errorbar(x + 0.08, gz, yerr=egz, fmt="s", color="C3",
                    label=r"$\Delta\Gamma_Z$")
        ax.axhline(0.0, color="black", lw=1.0)
        ax.set_xticks(x); ax.set_xticklabels(lbl, rotation=30, ha="right")
        ax.set_ylabel("fitted - generated  [MeV]")
        ax.set_yscale("symlog", linthresh=1.0)
        ax.legend(fontsize=15, frameon=False)
        ax.set_title("closure vs fit window, 2-parameter Born fit",
                     fontsize=15, loc="left")
        ax.grid(alpha=0.25)
        save(fig, outdir, "09_window_dependence")

    # -- 10. the smooth K(m) the fit needs, vs what the luminosity can do ---
    fk = fits.get("+ shape nuisance, 5 Legendre")
    if fk:
        from numpy.polynomial import legendre
        from rabbit.lineshapes.zgamma import load_lumi_table
        mm = np.linspace(60.0, 120.0, 400)
        uu = 2.0 * (mm - 60.0) / 60.0 - 1.0
        c = np.array([fk[f"shape{k}"][0] for k in range(1, 6)])
        basis = np.stack([legendre.legval(uu, [0] * k + [1]) for k in range(1, 6)])
        K = np.exp(np.tensordot(c, basis, axes=1))
        K = K / np.interp(91.1876, mm, K)
        fig, ax = plt.subplots(figsize=(9.2, 6.6))
        ax.plot(mm, K, color="black", lw=2.2,
                label=r"$K(m)$ the fit needs (5 Legendre terms)")
        lm0, ll0, _ = load_lumi_table("nnpdf31_nnlo_13tev")
        for tag, path, col in (
                ("NNPDF3.1 replica 1", "data/lumi/zlumi_nnpdf31_nnlo_rep1_13tev.npz", "C0"),
                ("NNPDF3.1 LO", "data/lumi/zlumi_nnpdf31_lo_13tev.npz", "C1"),
                ("CT18 NNLO", "data/lumi/zlumi_ct18nnlo_13tev.npz", "C2"),
                (r"$\mu_F = Q/2$", "data/lumi/zlumi_nnpdf31_nnlo_muf05_13tev.npz", "C4"),
                (r"$\mu_F = 2Q$", "data/lumi/zlumi_nnpdf31_nnlo_muf20_13tev.npz", "C5")):
            if not os.path.exists(path):
                continue
            lm, ll, _ = load_lumi_table(path)
            # the u-quark luminosity dominates; take the flavour-summed ratio
            r0 = np.exp(np.array([np.interp(np.log(mm), lm0, row) for row in ll0])).sum(0)
            r1 = np.exp(np.array([np.interp(np.log(mm), lm, row) for row in ll])).sum(0)
            rr = r1 / r0
            rr = rr / np.interp(91.1876, mm, rr)
            ax.plot(mm, rr, lw=1.5, color=col, ls="--", label=f"luminosity: {tag}")
        ax.axhline(1.0, color="grey", lw=0.8)
        ax.set_xlabel(r"$m_{\mu\mu}$  [GeV]")
        ax.set_ylabel(r"ratio to nominal, normalised at $m_Z$")
        ax.legend(fontsize=13, frameon=False, ncol=2)
        ax.set_title(r"the LO$\,\to\,$MiNNLO shape correction is 6x the full "
                     "PDF + scale spread", fontsize=15, loc="left")
        ax.grid(alpha=0.25)
        save(fig, outdir, "10_kfactor_vs_lumi")

    # -- 11. closure vs the number of shape terms ---------------------------
    rows = [(0, fits.get("baseline 60-120"))] + [
        (n, fits.get(f"+ shape nuisance, {n} Legendre")) for n in range(1, 7)]
    rows = [(n, v) for n, v in rows if v]
    if len(rows) > 2:
        fig, ax = plt.subplots(figsize=(9.0, 6.4))
        x = [n for n, _ in rows]
        ax.errorbar(x, [v["m_Z"][0] for _, v in rows],
                    yerr=[v["m_Z"][1] for _, v in rows], fmt="o-", color="C0",
                    label=r"$\Delta m_Z$")
        ax.errorbar(x, [v["Gamma_Z"][0] for _, v in rows],
                    yerr=[v["Gamma_Z"][1] for _, v in rows], fmt="s-", color="C3",
                    label=r"$\Delta\Gamma_Z$")
        ax.axhline(0.0, color="black", lw=1.0)
        ax.axhspan(-1.0, 1.0, color="C0", alpha=0.10)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xlabel("Legendre terms in the smooth $K(m)$")
        ax.set_ylabel("fitted - generated  [MeV]")
        ax.legend(fontsize=15, frameon=False)
        ax.set_title("closure vs the flexibility of $K(m)$, window 60-120 GeV",
                     fontsize=15, loc="left")
        ax.grid(alpha=0.25)
        save(fig, outdir, "11_shape_convergence")

    logger.info(f"figures in {outdir}")


if __name__ == "__main__":
    main()
