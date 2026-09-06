#!/usr/bin/env python3
"""Post-fit mass spectrum: the model the likelihood actually fitted, on data.

An unbinned per-candidate likelihood has no "the model" curve -- each candidate
has its own density `p_i(m)/Z_i`, and the ensemble prediction is their average,

    P(m) = sum_i w_i p_i(m) / Z_i  /  sum_i w_i .

That is evaluated here EXACTLY, by rebuilding a `MassCFTerm` on the outer
product of a random subsample of candidates with the plotting grid: row (i, j)
carries candidate i's resolution and `mobs = m_j - m_ref`, so `raw_density`
returns `p_i(m_j)` with the term's own arithmetic and no re-derivation.
`--nsub` x `--nbins` rows, so 2000 x 240 is a 480 k-row evaluation.

usage:
  ./run_tf.sh python3 plot_postfit.py --card cards/z_full.hdf5 \
      --fit results/fit_base.json --pairs runs/zpairs_dyv2.npz
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
    p = argparse.ArgumentParser()
    p.add_argument("--card", required=True)
    p.add_argument("--fit", default=None, help="fit.py json; default = the card's start")
    p.add_argument("--outpath", default=None)
    p.add_argument("--postfix", default="")
    p.add_argument("--title", default="CMS")
    p.add_argument("--subtitle", default="Simulation, work in progress")
    p.add_argument("--titlePos", type=int, default=2)
    p.add_argument("--nsub", type=int, default=2000)
    p.add_argument("--nbins", type=int, default=240)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--name", default="10_postfit_mass")
    return p.parse_args()


def main():
    import h5py
    from rabbit import unbinned

    args = parse_args()
    outdir = args.outpath or os.path.expanduser(
        f"~/public_html/cvh/{datetime.date.today().strftime('%y%m%d')}_fullscale/")
    os.makedirs(outdir, exist_ok=True)

    with h5py.File(args.card, "r") as f:
        term = unbinned.read_unbinned_terms_from_h5(f["unbinned_terms"])[0]
        g = f["unbinned_terms"][term.name]
        keys = {k: np.asarray(g[k]) for k in g.keys()
                if k.startswith("S_") and not k.endswith("_norm")}
    cfg = term.config()
    lo, hi = cfg["norm_window"]
    mref = cfg["m_ref"]
    x = np.array(term.param_defaults, dtype=np.float64, copy=True)
    names = list(term.param_names)
    if args.fit:
        with open(args.fit) as fh:
            r = json.load(fh)
        for nm, v in zip(r["params"], r.get("fitted", [])):
            if nm in names:
                x[names.index(nm)] = float(v)
        logger.info(f"parameters from {args.fit}")

    n = term.n
    sigma = term.sigma.numpy()
    mobs = term.mobs.numpy()
    vgf = term.vgf.numpy() if term.vgf is not None else None
    w = term.weights.numpy() if term.weights is not None else np.ones(n)
    m = mobs + mref

    # ---- the observed spectrum -------------------------------------------
    edges = np.linspace(lo, hi, args.nbins + 1)
    c = 0.5 * (edges[1:] + edges[:-1])
    dm = edges[1] - edges[0]
    obs, _ = np.histogram(m, edges, weights=w)
    obs_e = np.sqrt(np.histogram(m, edges, weights=w * w)[0])

    # ---- the model, on the outer product of a subsample with the grid ----
    rng = np.random.default_rng(args.seed)
    sub = np.sort(rng.choice(n, min(args.nsub, n), replace=False))
    K = len(sub)
    rep = np.repeat(sub, args.nbins)
    grid = np.tile(c, K)
    fams = []
    for f in term.families:
        e = {"name": f["name"], "param": f["param"], "kind": f["kind"]}
        for comp in ("re", "im"):
            k = f"S_{comp}_{f['name']}"
            if k in keys:
                e[comp] = keys[k][rep]
        fams.append(e)
    kwargs = {}
    if getattr(term, "a_res", None) is not None:
        kwargs["a_res"] = term.a_res.numpy()[rep]
        kwargs["self_consistent_sigma"] = term.self_consistent_sigma
    if getattr(term, "jensen_s2", None) is not None:
        kwargs["jensen_s2"] = term.jensen_s2.numpy()[rep]
        kwargs["jensen_mode"] = term.jensen_mode
    t2 = unbinned.MassCFTerm(
        "grid", sigma=sigma[rep], mobs=grid - mref, tgrid=term.tgrid_stored,
        families=fams, vgf=None if vgf is None else vgf[rep],
        kernel=term.kernel, background=term.background, m_ref=mref,
        scale_param=cfg["scale_param"], bkg_frac=cfg["bkg_frac"],
        upsample=cfg["upsample"], chunk=131072, dtype=DT, **kwargs)
    unbinned.declare_params(t2, {nm: (float(x[i]), np.nan, float(x[i]), 0)
                                 for i, nm in enumerate(names)
                                 if nm in t2.param_names})
    xx = np.array([x[names.index(nm)] for nm in t2.param_names])
    dens = t2.raw_density(tf.constant(xx, DT)).numpy().reshape(K, args.nbins)
    # the same truncation the likelihood applies
    z = np.sum(dens, axis=1) * dm
    dens = dens / np.maximum(z[:, None], 1e-300)
    ww = w[sub]
    model = (ww[:, None] * dens).sum(0) / ww.sum() * w.sum() * dm

    # ---- draw -------------------------------------------------------------
    fig = plt.figure(figsize=(8, 7))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.06)
    ax = fig.add_subplot(gs[0])
    rax = fig.add_subplot(gs[1], sharex=ax)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.errorbar(c, obs, yerr=obs_e, fmt="o", ms=2.5, color="k",
                label=f"CVH refit, {n} candidates")
    ax.plot(c, model, color="tab:red", lw=1.6,
            label=f"unbinned model ({K} candidates x {args.nbins} points)")
    ax.set_yscale("log")
    ax.set_ylabel(f"weighted candidates / {dm:.3g} GeV")
    ax.legend(fontsize=12)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = obs / model
        re = obs_e / model
    rax.errorbar(c, r, yerr=re, fmt="o", ms=2.5, color="k")
    rax.axhline(1.0, color="tab:red", lw=1.2)
    rax.set_ylim(0.8, 1.2)
    rax.set_xlabel(r"$m_{\mu\mu}$ [GeV]")
    rax.set_ylabel("data / model")
    plot_tools.add_decor(ax, args.title, args.subtitle, data=False, lumi=None,
                         loc=args.titlePos)
    name = "_".join(filter(None, [args.name, args.postfix]))
    plot_tools.save_pdf_and_png(outdir, name)
    output_tools.write_logfile(outdir, name, args=args,
                               wd=os.path.dirname(os.path.abspath(__file__)))
    ok = np.isfinite(r) & (obs > 0)
    chi2 = float(np.sum(((obs[ok] - model[ok]) / obs_e[ok]) ** 2))
    logger.info(f"chi2/ndof of the ratio = {chi2:.1f}/{int(ok.sum())} "
                f"= {chi2/max(int(ok.sum()),1):.2f}")
    logger.info(f"-> {outdir}{name}")


if __name__ == "__main__":
    logging.setup_logger(__file__, 3, False)
    main()
