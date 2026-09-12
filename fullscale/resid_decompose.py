#!/usr/bin/env python3
"""Is the post-fit residual a SHIFT or a WIDTH mismatch?  Decompose it.

The two hypotheses have different, and nearly orthogonal, signatures in
`data/model - 1` around the pole:

* a **shift** of the model by `dm` gives `-dm * dln p/dm`, which is ODD about
  the peak -- deficit on one side, excess on the other, an S;
* a **width** mismatch (the core resolution too narrow or too wide by `ds`)
  gives `ds * dln p/ds`, which is EVEN -- one sign in the core and the other in
  the shoulders.

Both templates are computed from the TERM itself, by re-evaluating the same
per-candidate densities at `m_Z + dm` and at `sigma -> sigma(1 + ds)`, so no
analytic approximation enters.  The residual is then least-squares projected
onto them (plus a constant, which the normalisation absorbs), and what is left
over is the part neither hypothesis explains.

usage:
  ./run_tf.sh python3 resid_decompose.py --card cards/z_full380_fl.hdf5 \
      --fit results/fit_f380fl_base.json --nsub 40000
"""
import argparse
import datetime
import json
import os

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import tensorflow as tf
from matplotlib.gridspec import GridSpec

from wums import logging, output_tools, plot_tools

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)
DT = tf.float64


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--card", required=True)
    p.add_argument("--fit", required=True)
    p.add_argument("--nsub", type=int, default=20000)
    p.add_argument("--nbins", type=int, default=240)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--dm", type=float, default=20.0, help="shift template [MeV]")
    p.add_argument("--ds", type=float, default=0.01, help="width template, relative")
    p.add_argument("--outpath", default=None)
    p.add_argument("--postfix", default="")
    p.add_argument("--title", default="CMS")
    p.add_argument("--subtitle", default="work in progress")
    p.add_argument("--titlePos", type=int, default=2)
    p.add_argument("--name", default="14_residual_decomposition")
    return p.parse_args()


def main():
    import h5py
    from rabbit import unbinned

    args = parse_args()
    outdir = args.outpath or os.path.expanduser(
        f"~/public_html/ZMass/cvh/{datetime.date.today().strftime('%y%m%d')}_fullscale/")
    os.makedirs(outdir, exist_ok=True)

    with h5py.File(args.card, "r") as f:
        term = unbinned.read_unbinned_terms_from_h5(f["unbinned_terms"])[0]
    keys = {}
    for f_ in term.families:
        for comp in ("re", "im"):
            if comp in f_:
                keys[f"S_{comp}_{f_['name']}"] = np.asarray(f_[comp])
    cfg = term.config()
    lo, hi = cfg["norm_window"]
    mref = cfg["m_ref"]
    names = list(term.param_names)
    x = np.array(term.param_defaults, dtype=np.float64, copy=True)
    with open(args.fit) as fh:
        r = json.load(fh)
    for nm, v in zip(r["params"], r.get("fitted", [])):
        if nm in names:
            x[names.index(nm)] = float(v)

    n = term.n
    sigma = term.sigma.numpy()
    mobs = term.mobs.numpy()
    vgf = term.vgf.numpy() if term.vgf is not None else None
    w = term.weights.numpy() if term.weights is not None else np.ones(n)
    m = mobs + mref
    edges = np.linspace(lo, hi, args.nbins + 1)
    c = 0.5 * (edges[1:] + edges[:-1])
    dm_bin = edges[1] - edges[0]
    obs, _ = np.histogram(m, edges, weights=w)
    obs_e = np.sqrt(np.histogram(m, edges, weights=w * w)[0])

    rng = np.random.default_rng(args.seed)
    sub = np.sort(rng.choice(n, min(args.nsub, n), replace=False))
    K = len(sub)
    rep = np.repeat(sub, args.nbins)

    def model(scale_sigma=1.0, dmz=0.0):
        fams = []
        for f_ in term.families:
            e = {"name": f_["name"], "param": f_["param"], "kind": f_["kind"]}
            for comp in ("re", "im"):
                k = f"S_{comp}_{f_['name']}"
                if k in keys:
                    e[comp] = keys[k][rep]
            fams.append(e)
        kw = {}
        if getattr(term, "a_res", None) is not None:
            kw["a_res"] = term.a_res.numpy()[rep]
            kw["self_consistent_sigma"] = term.self_consistent_sigma
        if getattr(term, "jensen_s2", None) is not None:
            kw["jensen_s2"] = term.jensen_s2.numpy()[rep]
            kw["jensen_mode"] = term.jensen_mode
        if getattr(term, "corr_form", "residual") != "residual":
            kw["corr_form"] = term.corr_form
        t2 = unbinned.MassCFTerm(
            "grid", sigma=sigma[rep] * scale_sigma, mobs=np.tile(c, K) - mref,
            tgrid=term.tgrid_stored, families=fams,
            vgf=None if vgf is None else vgf[rep],
            kernel=term.kernel, background=term.background, m_ref=mref,
            scale_param=cfg["scale_param"], bkg_frac=cfg["bkg_frac"],
            upsample=cfg["upsample"], chunk=131072, dtype=DT, **kw)
        if getattr(t2, "_fluct_active", False):
            t2._fl_a = tf.constant(term._fl_a.numpy()[rep], DT)
            t2._fl_g = tf.constant(term._fl_g.numpy()[rep], DT)
            t2._fl_d = (None if term._fl_d is None
                        else tf.constant(term._fl_d.numpy()[rep], DT))
        unbinned.declare_params(t2, {nm: (float(x[i]), np.nan, float(x[i]), 0)
                                     for i, nm in enumerate(names)
                                     if nm in t2.param_names})
        xx = np.array([x[names.index(nm)] for nm in t2.param_names])
        if dmz and "m_Z" in list(t2.param_names):
            xx[list(t2.param_names).index("m_Z")] += dmz
        dens = t2.raw_density(tf.constant(xx, DT)).numpy().reshape(K, args.nbins)
        z = np.sum(dens, axis=1) * dm_bin
        dens = dens / np.maximum(z[:, None], 1e-300)
        ww = w[sub]
        return (ww[:, None] * dens).sum(0) / ww.sum() * w.sum() * dm_bin

    base = model()
    shifted = model(dmz=args.dm)
    widened = model(scale_sigma=1.0 + args.ds)
    print(f"templates: dm = {args.dm} MeV, ds = {args.ds:.3g} relative")

    ok = (base > 0) & (obs > 0)
    res = obs / base - 1.0
    res_e = obs_e / base
    T_shift = shifted / base - 1.0
    T_width = widened / base - 1.0

    # least squares on [1, shift, width], weighted by the data error
    A = np.vstack([np.ones_like(c), T_shift, T_width]).T[ok]
    ycol = res[ok]
    Wd = 1.0 / np.maximum(res_e[ok], 1e-12) ** 2
    ATA = A.T @ (Wd[:, None] * A)
    ATy = A.T @ (Wd * ycol)
    coef = np.linalg.solve(ATA, ATy)
    cov = np.linalg.inv(ATA)
    err = np.sqrt(np.diag(cov))
    fitres = A @ coef
    chi2_0 = float(np.sum(Wd * ycol ** 2))
    chi2_f = float(np.sum(Wd * (ycol - fitres) ** 2))
    ndf = int(ok.sum())
    print(f"  chi2/ndf of the raw residual   {chi2_0:.1f}/{ndf} = {chi2_0/ndf:.2f}")
    print(f"  chi2/ndf after shift+width     {chi2_f:.1f}/{ndf-3} = {chi2_f/(ndf-3):.2f}")
    print(f"  const   {coef[0]:+.5f} +- {err[0]:.5f}")
    print(f"  SHIFT   {coef[1]:+.4f} +- {err[1]:.4f} of a {args.dm:g} MeV template "
                f"-> {coef[1]*args.dm:+.2f} +- {err[1]*args.dm:.2f} MeV")
    print(f"  WIDTH   {coef[2]:+.4f} +- {err[2]:.4f} of a {100*args.ds:g} % template "
                f"-> {100*coef[2]*args.ds:+.3f} +- {100*err[2]*args.ds:.3f} % on sigma")

    fig = plt.figure(figsize=(8, 7))
    gs = GridSpec(2, 1, height_ratios=[3, 2], hspace=0.06)
    ax = fig.add_subplot(gs[0])
    rax = fig.add_subplot(gs[1], sharex=ax)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.errorbar(c[ok], res[ok], yerr=res_e[ok], fmt="ko", ms=2.5, lw=0.8,
                label="data / model $-$ 1")
    ax.plot(c[ok], (coef[0] + coef[1] * T_shift)[ok], color="tab:red", lw=1.6,
            label=rf"SHIFT {coef[1]*args.dm:+.1f} MeV (odd about the pole)")
    ax.plot(c[ok], (coef[0] + coef[2] * T_width)[ok], color="tab:blue", lw=1.6,
            label=rf"WIDTH {100*coef[2]*args.ds:+.2f} % (even)")
    ax.plot(c[ok], fitres, color="tab:green", lw=1.8, ls="--", label="sum")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_ylim(-0.13, 0.19)
    ax.set_ylabel("data / model $-$ 1")
    ax.legend(fontsize=10, loc="lower center", ncol=2)
    rax.errorbar(c[ok], (res - fitres)[ok], yerr=res_e[ok], fmt="ko", ms=2.5, lw=0.8)
    rax.axhline(0.0, color="k", lw=0.8)
    rax.set_ylim(-0.12, 0.12)
    rax.set_xlabel(r"$m_{\mu\mu}$ (CVH) [GeV]")
    rax.set_ylabel("left over")
    plot_tools.add_decor(ax, args.title, args.subtitle, data=False, lumi=None,
                         loc=args.titlePos)
    name = "_".join(filter(None, [args.name, args.postfix]))
    plot_tools.save_pdf_and_png(outdir, name)
    output_tools.write_logfile(outdir, name, args=args,
                               wd=os.path.dirname(os.path.abspath(__file__)))
    print(f"  -> {outdir}{name}")


if __name__ == "__main__":
    main()
