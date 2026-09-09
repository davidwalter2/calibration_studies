#!/usr/bin/env python3
"""Post-fit mass spectra for the WG deck of 2026-09-09: `z_postfit.png`,
`z_postfit_mform.png`, `jpsi_postfit.png`.

Same arithmetic as `fullscale/plot_postfit.py` -- the model curve is the
ensemble average of the per-candidate densities, evaluated by REBUILDING a
`MassCFTerm` on the outer product of a random subsample of candidates with the
plotting grid, so the term's own code (and no re-derivation) produces
`p_i(m_j)`.  Three things are added here, and each is needed for one of the
three figures:

1. **term selection by name.**  `plot_postfit.py` takes `terms[0]`; the J/psi
   spectrum lives in the second unbinned term of a JOINT card.

2. **the v formulation.**  Under `--vpow p` the card stores `sigma` and `mobs`
   in `v = int dm/m^p`, and `m_ref` is the `alpha` LEVER `m_ref^(1-p)`, not a
   mass.  `plot_postfit.py`'s `m = mobs + m_ref` is therefore not a mass for a
   v card, and its rebuilt term is an m-form term.  Here the physical mass
   comes off `corr_mass` (which the v card must carry), the plotting grid is
   mapped m -> v with the same `u(m) = m^(1-p)/(1-p) - v(m_ref) + m_ref^(1-p)`
   that `make_card.py` used, and the density is multiplied by the Jacobian
   `du/dm = m^(-p)` so that what is drawn is `dP/dm` against `m` in GeV.
   The per-candidate fluctuation constants are still taken off the PARENT
   term, which is what makes the v-form constants (`a - p sigma/m` and the
   substitution's own curvature) the ones actually used.

3. **the WG style**: 13 x 8 in at 160 dpi, constrained layout, SHORT axis
   labels on both panels (the defect of `260907_fullscale/12_postfit_*`, where
   the long main y-label ran into the ratio panel's), MIT palette.

    ssh submit50 'cd .../fullscale && RABBIT=.../rabbit-vmass THREADS=16 \
      ./run_tf.sh python3 figs_wg260909/postfit_wg.py --card ... --name ...'
"""
import argparse
import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wgstyle as W  # noqa: E402

from wums import logging, output_tools, plot_tools  # noqa: E402

logger = logging.child_logger(__name__)
DT = tf.float64

OUTDIR = os.path.expanduser(
    "~/public_html/slides/260909_unbinned_likelihood_walter/assets")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--card", required=True)
    p.add_argument("--term", default=None,
                   help="unbinned term name; default = the first one")
    p.add_argument("--fit", default=None,
                   help="a `fit.py` or `native_dump.py` json (both carry "
                        "`params` and `fitted`); default = the card's start")
    p.add_argument("--outpath", default=OUTDIR)
    p.add_argument("--name", required=True, help="output basename")
    p.add_argument("--nsub", type=int, default=6000)
    p.add_argument("--nbins", type=int, default=240)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--xlim", type=float, nargs=2, default=None,
                   help="displayed mass range; the MODEL is always normalised "
                        "over the term's own truncation window")
    p.add_argument("--rlim", type=float, nargs=2, default=[0.8, 1.2])
    p.add_argument("--rlog", action="store_true",
                   help="log-scale the ratio panel (the J/psi tail "
                        "leaves any linear range)")
    p.add_argument("--xlabel", default=r"$m_{\mu\mu}$ [GeV]")
    p.add_argument("--ylim", type=float, nargs=2, default=None)
    p.add_argument("--linear", action="store_true", help="linear y axis")
    p.add_argument("--legend-loc", default="upper right")
    p.add_argument("--selfcheck", action="store_true",
                   help="rebuild the term on the candidates' OWN masses and "
                        "compare with the parent's density -- the gate that "
                        "says the grid rebuild is faithful")
    return p.parse_args()


def variable_map(cfg):
    """(m -> u, u -> m, du/dm, the physical window) for this term.

    `u` is the variable the term's `mobs + m_ref` lives in: the mass itself for
    an m-form card, and `v = int dm/m^p` for a v-form one.
    """
    p = cfg.get("vpow")
    mref_term = float(cfg["m_ref"])
    lo_u, hi_u = cfg["norm_window"]
    if p is None:
        return (lambda m: m), (lambda u: u), (lambda m: np.ones_like(m)), \
            (float(lo_u), float(hi_u)), mref_term
    p = float(p)
    q = 1.0 - p
    v_ref = mref_term / q                       # = v(m_ref)
    def u_of_m(m):
        return np.power(m, q) / q - v_ref + mref_term
    def m_of_u(u):
        return np.power((np.asarray(u) - mref_term + v_ref) * q, 1.0 / q)
    def dudm(m):
        return np.power(m, -p)
    win = (float(m_of_u(lo_u)), float(m_of_u(hi_u)))
    if win[0] > win[1]:                          # q < 0 flips the order
        win = (win[1], win[0])
    return u_of_m, m_of_u, dudm, win, mref_term


def main():
    import h5py
    from rabbit import unbinned

    args = parse_args()
    os.makedirs(args.outpath, exist_ok=True)

    with h5py.File(args.card, "r") as f:
        terms = unbinned.read_unbinned_terms_from_h5(f["unbinned_terms"])
    names = [t.name for t in terms]
    logger.info(f"card carries unbinned terms {names}")
    term = terms[0] if args.term is None else terms[names.index(args.term)]
    logger.info(f"plotting term '{term.name}'")

    # the family exponents come off the TERM, not off the HDF5: the datasets
    # are stored flat with an `original_shape` attribute
    keys = {}
    for f_ in term.families:
        for comp in ("re", "im"):
            if comp in f_:
                keys[f"S_{comp}_{f_['name']}"] = np.asarray(f_[comp])
    cfg = term.config()
    u_of_m, _, dudm, (lo, hi), mref = variable_map(cfg)
    logger.info(f"vpow = {cfg.get('vpow')}, m_ref (term) = {mref:.6g}, "
                f"physical window = [{lo:.4f}, {hi:.4f}] GeV")

    x = np.array(term.param_defaults, dtype=np.float64, copy=True)
    pnames = list(term.param_names)
    if args.fit:
        with open(args.fit) as fh:
            r = json.load(fh)
        nset = 0
        for nm, v in zip(r["params"], r.get("fitted", [])):
            if nm in pnames:
                x[pnames.index(nm)] = float(v)
                nset += 1
        logger.info(f"{nset} of {len(pnames)} parameters from {args.fit}")
    logger.info("parameter point: " + ", ".join(
        f"{nm}={x[i]:.6g}" for i, nm in enumerate(pnames)
        if not nm.startswith(("bfield_mode", "material_"))))

    n = term.n
    sigma = term.sigma.numpy()
    mobs = term.mobs.numpy()
    vgf = term.vgf.numpy() if term.vgf is not None else None
    w = term.weights.numpy() if term.weights is not None else np.ones(n)
    # the PHYSICAL mass: `corr_mass` where the card stores it (v form and
    # residual mode both need it), else the m-form identity
    if getattr(term, "corr_mass", None) is not None:
        m = np.asarray(term.corr_mass.numpy() if hasattr(term.corr_mass,
                                                         "numpy")
                       else term.corr_mass, dtype=np.float64)
    else:
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
    grid_u = np.tile(u_of_m(c), K)
    fams = []
    for f_ in term.families:
        e = {"name": f_["name"], "param": f_["param"], "kind": f_["kind"]}
        for comp in ("re", "im"):
            k = f"S_{comp}_{f_['name']}"
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
    if getattr(term, "corr_form", "residual") != "residual":
        kwargs["corr_form"] = term.corr_form
    t2 = unbinned.MassCFTerm(
        "grid", sigma=sigma[rep], mobs=grid_u - mref, tgrid=term.tgrid_stored,
        families=fams, vgf=None if vgf is None else vgf[rep],
        kernel=term.kernel, background=term.background, m_ref=mref,
        scale_param=cfg["scale_param"], bkg_frac=cfg["bkg_frac"],
        upsample=cfg["upsample"], chunk=131072, dtype=DT, **kwargs)
    # THE PER-CANDIDATE CONSTANTS COME FROM THE CANDIDATE, NOT FROM THE GRID.
    # `t2`'s rows are (candidate, mass) pairs, so `_build_fluct` would evaluate
    # each candidate's a_i, c_i, d_i at the PLOTTING mass.  Take them off the
    # parent -- which is also what makes the v-form constants (a - p sigma/m,
    # and the substitution's own curvature) the ones that are used, since `t2`
    # itself is built without `vpow`.
    if getattr(t2, "_fluct_active", False):
        t2._fl_a = tf.constant(term._fl_a.numpy()[rep], DT)
        t2._fl_g = tf.constant(term._fl_g.numpy()[rep], DT)
        t2._fl_d = (None if term._fl_d is None
                    else tf.constant(term._fl_d.numpy()[rep], DT))
    unbinned.declare_params(t2, {nm: (float(x[i]), np.nan, float(x[i]), 0)
                                 for i, nm in enumerate(pnames)
                                 if nm in t2.param_names})
    xx = np.array([x[pnames.index(nm)] for nm in t2.param_names])
    dens = t2.raw_density(tf.constant(xx, DT)).numpy().reshape(K, args.nbins)

    if args.selfcheck:
        # the same rebuild, but on the candidates' OWN `mobs`.  If this
        # reproduces the parent term's density row for row, then the grid
        # evaluation above differs from the likelihood only in WHERE it is
        # evaluated -- which is the whole claim of the figure.
        fams1 = []
        for f_ in term.families:
            e = {"name": f_["name"], "param": f_["param"], "kind": f_["kind"]}
            for comp in ("re", "im"):
                k = f"S_{comp}_{f_['name']}"
                if k in keys:
                    e[comp] = keys[k][sub]
            fams1.append(e)
        kw1 = dict(kwargs)
        for k in ("a_res", "jensen_s2"):
            if k in kw1:
                kw1[k] = getattr(term, k).numpy()[sub]
        t3 = unbinned.MassCFTerm(
            "self", sigma=sigma[sub], mobs=mobs[sub], tgrid=term.tgrid_stored,
            families=fams1, vgf=None if vgf is None else vgf[sub],
            kernel=term.kernel, background=term.background, m_ref=mref,
            scale_param=cfg["scale_param"], bkg_frac=cfg["bkg_frac"],
            upsample=cfg["upsample"], chunk=131072, dtype=DT, **kw1)
        if getattr(t3, "_fluct_active", False):
            t3._fl_a = tf.constant(term._fl_a.numpy()[sub], DT)
            t3._fl_g = tf.constant(term._fl_g.numpy()[sub], DT)
            t3._fl_d = (None if term._fl_d is None
                        else tf.constant(term._fl_d.numpy()[sub], DT))
        unbinned.declare_params(t3, {nm: (float(x[i]), np.nan, float(x[i]), 0)
                                     for i, nm in enumerate(pnames)
                                     if nm in t3.param_names})
        a = t3.raw_density(tf.constant(
            np.array([x[pnames.index(nm)] for nm in t3.param_names]),
            DT)).numpy()
        b = term.raw_density(tf.constant(x, DT)).numpy()[sub]
        rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-300)
        logger.info(f"SELFCHECK rebuilt vs parent density: max rel dev "
                    f"{rel.max():.3e}, median {np.median(rel):.3e}")
    # dP/du -> dP/dm, so the curve is a density in GeV whatever the card's
    # internal variable is
    dens = dens * dudm(c)[None, :]
    # the same truncation the likelihood applies
    z = np.sum(dens, axis=1) * dm
    dens = dens / np.maximum(z[:, None], 1e-300)
    ww = w[sub]
    model = (ww[:, None] * dens).sum(0) / ww.sum() * w.sum() * dm

    # ---- draw -------------------------------------------------------------
    W.use_style()
    fig = plt.figure(figsize=W.FIGSIZE, dpi=W.DPI, constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.045)
    ax = fig.add_subplot(gs[0])
    rax = fig.add_subplot(gs[1], sharex=ax)
    plt.setp(ax.get_xticklabels(), visible=False)

    ax.errorbar(c, obs, yerr=obs_e, fmt="o", ms=3.4, color="k", lw=1.2,
                label=f"MC candidates ({n:,})".replace(",", " "))
    ax.plot(c, model, color=W.MIT_RED, lw=2.2,
            label=f"unbinned model ({K} x {args.nbins} points)")
    if not args.linear:
        ax.set_yscale("log")
    if args.ylim:
        ax.set_ylim(*args.ylim)
    ax.set_ylabel(f"candidates / {dm:.3g} GeV")
    ax.legend(fontsize=17, loc=args.legend_loc, frameon=False)

    with np.errstate(invalid="ignore", divide="ignore"):
        r = obs / model
        re = obs_e / model
    rax.errorbar(c, r, yerr=re, fmt="o", ms=3.4, color="k", lw=1.2)
    rax.axhline(1.0, color=W.MIT_RED, lw=1.8)
    if args.rlog:
        rax.set_yscale("log")
        rax.yaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        rax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    rax.set_ylim(*args.rlim)
    rax.set_xlabel(args.xlabel)
    rax.set_ylabel("data / model")
    rax.grid(axis="y", color=W.MIT_GRAY, alpha=0.35, lw=0.8)
    if args.xlim:
        ax.set_xlim(*args.xlim)
        rax.set_xlim(*args.xlim)
    # `data=False` already prepends "Simulation", so the label is the rest of
    # it; `loc=0` puts the whole thing above the frame, clear of the legend
    plot_tools.add_decor(ax, "CMS", "work in progress", data=False,
                         lumi=None, loc=0)

    plot_tools.save_pdf_and_png(args.outpath, args.name, fig=fig)
    output_tools.write_logfile(args.outpath, args.name, args=args,
                               wd=os.path.dirname(os.path.abspath(__file__)))

    def chi2(mask):
        ok = mask & np.isfinite(r) & (obs > 0)
        v = float(np.sum(((obs[ok] - model[ok]) / obs_e[ok]) ** 2))
        nd = int(ok.sum())
        return v, nd
    full = chi2(np.ones_like(c, dtype=bool))
    logger.info(f"chi2 of the ratio, full window = {full[0]:.1f}/{full[1]} "
                f"= {full[0]/max(full[1],1):.2f}")
    if args.xlim:
        shown = chi2((c >= args.xlim[0]) & (c <= args.xlim[1]))
        logger.info(f"chi2 of the ratio, shown range = "
                    f"{shown[0]:.1f}/{shown[1]} "
                    f"= {shown[0]/max(shown[1],1):.2f}")
    # where the residual structure is: the mean ratio and the chi2 in blocks
    for blk in np.array_split(np.arange(args.nbins), 8):
        msk = np.zeros_like(c, dtype=bool)
        msk[blk] = True
        v, nd = chi2(msk)
        with np.errstate(invalid="ignore", divide="ignore"):
            mr = float(np.nanmean(r[msk & (obs > 0)]))
        logger.info(f"  block [{c[blk[0]]:.3f}, {c[blk[-1]]:.3f}] GeV: "
                    f"<data/model> = {mr:.4f}, chi2/n = {v:.1f}/{nd}")
    logger.info(f"-> {args.outpath}/{args.name}.png")
    plt.close(fig)


if __name__ == "__main__":
    logging.setup_logger(__file__, 3, False)
    main()
