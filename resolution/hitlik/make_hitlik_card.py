#!/usr/bin/env python3
"""Build a rabbit datacard for the residual-vector CF likelihood.

The card carries, over ONE set of parameters (42 material groups + 18 hit
classes, plus the 50 field modes when the quadratic term is present):

* the **residual-vector term**: an unbinned per-(track, component) likelihood
  of the whitened track-parameter residual ``z``, whose density is the inverse
  Fourier transform of the block CF product (arm ``cf``), or of the same
  blocks' variance-matched Gaussians (``gauss`` / ``gaussq``);
* optionally the **quadratic hit-chi2 term** from ``globalfit/extract.py`` over
  the SAME production, written exactly as ``globalfit/make_global_term.py``
  writes it.

Neither the field modes nor the alignment enter the residual term: on a
truth-referenced residual they move the MEAN, and the mean of ``z`` carries no
material or hit-resolution information -- so this term is a pure WIDTH/SHAPE
term, and field/alignment stay in the quadratic one.

Usage
-----
    python make_hitlik_card.py --npz runs/hitlik/mugun20k.npz \
      --quad-npz runs/hitlik/mugun_quad.npz --arm cf --comps 0123 \
      --whiten -o runs/hitlik/cards/cf.hdf5
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.dirname(_HERE)
_GF = os.path.join(_RES, "globalfit")
_MAT = os.path.join(_RES, "matres")
for _p in (_HERE, _RES, _GF, _MAT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hitlik_term as HT  # noqa: E402
import make_global_term as MGT  # noqa: E402


def log(m):
    print(m, flush=True)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", required=True)
    p.add_argument("--quad-npz", default=None)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--arm", choices=list(HT.ARMS), default="cf")
    p.add_argument("--comps", default="0123",
                   help="which whitened components to use, as digits")
    p.add_argument("--max-tracks", type=int, default=0)
    p.add_argument("--max-chi2-ndof", type=float, default=0.0)
    p.add_argument("--max-inflat", type=float, default=1e4,
                   help="drop tracks whose Cholesky variance inflation "
                        "V_kk/d_k exceeds this in any used component -- a "
                        "guard on the FIT COVARIANCE, not on the residual")
    p.add_argument("--groups", default=None,
                   help="materialGroups tier file (default: the one recorded "
                        "in the extraction)")
    p.add_argument("--coeffs", default=None)
    p.add_argument("--parmtypes", type=int, nargs="*", default=[14, 15])
    p.add_argument("--whiten", action="store_true")
    p.add_argument("--no-quadratic", action="store_true")
    p.add_argument("--no-residual", action="store_true")
    p.add_argument("--no-hits", action="store_true")
    p.add_argument("--prune-frac", type=float, default=0.001)
    p.add_argument("--amount-mode", choices=["exp", "linear"], default="exp")
    p.add_argument("--hit-mode", choices=["exp", "linear"], default="linear")
    p.add_argument("--chunk", type=int, default=8192)
    p.add_argument("--floor", choices=["softplus", "clip", "none"],
                   default="softplus")
    p.add_argument("--field-prior", type=float, default=0.0)
    p.add_argument("--material-prior-scale", type=float, default=1.0)
    p.add_argument("--hit-prior", type=float, default=0.0)
    p.add_argument("--poi", default="material")
    p.add_argument("--inject", action="append", default=[], metavar="NAME:VALUE",
                   help="a TRUE correction dtheta applied to BOTH terms: the "
                        "quadratic G -> G - K dtheta, and the residual term's "
                        "exponents of that material group scaled by A(dtheta) "
                        "(hit classes: the class variance scaled by H(dtheta))")
    p.add_argument("--inject-quad-only", action="store_true")
    p.add_argument("--inject-res-only", action="store_true")
    p.add_argument("--dense-max", type=int, default=4000)
    return p.parse_args()


def _poi_set(spec, names):
    if spec == "bfield":
        return {n for n in names if n.startswith("bfield_mode")}
    if spec == "material":
        return {n for n in names if n.startswith("material_")}
    if spec == "hitres":
        return {n for n in names if n.startswith("hitres_")}
    if spec == "all":
        return set(names)
    if spec == "none":
        return set()
    return {s for s in spec.split(",") if s}


def main():
    args = parse_args()
    comps = [int(c) for c in args.comps]
    sel = HT.load(args.npz, max_tracks=args.max_tracks, comps=comps,
                  max_chi2_ndof=args.max_chi2_ndof, max_inflat=args.max_inflat)
    groups_file = args.groups or sel["groups_file"]
    gnames_all = sel["group_names"]
    cnames_all = sel["hit_classes"]
    ngroups = len(gnames_all)
    log(f"{sel['ntrk']} tracks x {len(comps)} components = {len(sel['z'])} "
        f"rows; arm '{args.arm}'")

    # ---- the CARD UNIT of a material parameter -----------------------------
    # Defined by the groups file alone, so a residual-only card and a joint one
    # use the SAME units.  (Deriving it from the quadratic catalog, as an
    # earlier version did, silently left --no-quadratic cards unwhitened and
    # turned a 5 % injection into a 0.24 % one.)
    import groups as G
    gparams_grp, gpriors_grp = G.group_param_names(ngroups, groups_file)
    gscale = gpriors_grp.copy() if args.whiten else np.ones(ngroups)
    group_units = 1.0 / np.maximum(gscale, 1e-300)   # k_phys = value * units
    gprior_card = gpriors_grp * gscale               # 1 prior sigma, card units

    # ---- the global parameter catalog, from the quadratic extraction -------
    qd = None
    if args.quad_npz and not args.no_quadratic:
        qd = np.load(args.quad_npz, allow_pickle=False)
        parmtype, subidx = qd["parmtype"], qd["subidx"]
        isel = np.where(np.isin(parmtype, args.parmtypes))[0]
        names, prior_sigmas = MGT.name_params(
            parmtype[isel], subidx[isel], groups_file, args.field_prior,
            args.material_prior_scale)
        nfit = len(names)
        matcol = {int(si): j for j, (pt, si)
                  in enumerate(zip(parmtype[isel], subidx[isel]))
                  if int(pt) == 15}
        pscale = np.ones(nfit)
        if args.whiten:
            pscale, _ = MGT.param_scales(parmtype[isel], subidx[isel],
                                         args.coeffs, groups_file)
            prior_sigmas = prior_sigmas * pscale
        # GATE: the quadratic catalog's material scale must be the groups
        # file's, or the two terms are floating different variables.
        for g, j in matcol.items():
            if g < ngroups and abs(pscale[j] - gscale[g]) > 1e-12 * max(
                    gscale[g], 1e-30):
                sys.exit(f"material scale mismatch for group {g}: quadratic "
                         f"catalog {pscale[j]:g} vs groups file {gscale[g]:g}")
        log(f"{nfit} global parameters from the quadratic catalog; whiten="
            f"{bool(args.whiten)}")
    else:
        names, prior_sigmas = [], np.zeros(0)
        nfit, isel, matcol, pscale = 0, None, {}, None
        parmtype = subidx = None

    # ---- injection ---------------------------------------------------------
    hparams = [f"hitres_{c}" for c in cnames_all]
    allnames = list(names) if nfit else list(gparams_grp) + hparams
    nameidx = {nm: i for i, nm in enumerate(allnames)}
    for nm in hparams:
        nameidx.setdefault(nm, len(nameidx))

    inject_card = {}
    for spec in args.inject:
        nm, val = spec.rsplit(":", 1)
        if nm not in nameidx and nm not in hparams:
            sys.exit(f"--inject for unknown parameter '{nm}'")
        inject_card[nm] = float(val)
    inj_groups, inj_hits = {}, {}
    for g in range(ngroups):
        nm = gparams_grp[g]
        if nm in inject_card:
            inj_groups[g] = inject_card[nm] * group_units[g]
    for c, nm in enumerate(hparams):
        if nm in inject_card:
            inj_hits[c] = inject_card[nm]
    if inject_card:
        log("injecting " + ", ".join(f"{k}={v:+.6g}"
                                     for k, v in inject_card.items())
            + f"  -> physical k: " + ", ".join(
                f"{gparams_grp[g]}={v:+.6g} ({100*(np.exp(v)-1):+.3f} % material)"
                for g, v in inj_groups.items()))

    from rabbit import tensorwriter
    writer = tensorwriter.TensorWriter()
    writer.add_dummy_channel(name="calib_dummy")
    declared = []

    if not args.no_residual:
        term, data, meta = HT.build(
            sel, arm=args.arm, prune_frac=args.prune_frac,
            groups_file=groups_file, ngroups=ngroups,
            inject=(inj_groups if not args.inject_quad_only else None),
            hit_inject=(inj_hits if not args.inject_quad_only else None),
            amount_mode=args.amount_mode, hit_mode=args.hit_mode,
            chunk=args.chunk, no_hits=args.no_hits, floor=args.floor)
        # the group_units of the term must be the CARD units
        data["group_units"] = group_units
        term.group_units = group_units
        import tensorflow as tf
        term._gunits = tf.constant(group_units, term.dtype)
        log(f"residual term: {meta['n']} rows, {meta['nrows']} group rows "
            f"({meta['ndrop']} pruned), {len(term.param_names)} parameters")

        _x0 = tf.Variable(np.zeros(len(term.param_names)), dtype=tf.float64)
        with tf.GradientTape() as _tp:
            _v = term.nll(_x0)
        _g = np.asarray(_tp.gradient(_v, _x0).numpy())
        log(f"NLL(0) = {float(_v.numpy()):.6f}, max|grad| = "
            f"{np.abs(_g).max():.4g}")
        if not (np.isfinite(float(_v.numpy())) and np.isfinite(_g).all()):
            sys.exit("the residual term is not finite at theta = 0")

        poi_set = _poi_set(args.poi, term.param_names)
        prior_by_name = dict(zip(gparams_grp, gprior_card))
        if nfit:
            prior_by_name.update(dict(zip(names, prior_sigmas)))
        defaults, sig, means, ispoi = [], [], [], []
        for p in term.param_names:
            defaults.append(0.0)
            if p.startswith("hitres_"):
                sig.append(args.hit_prior if args.hit_prior > 0 else np.nan)
            else:
                sig.append(prior_by_name.get(p, np.nan))
            ispoi.append(1 if p in poi_set else 0)
            means.append(0.0)
        writer.add_unbinned_term(
            "hitres", term.config(), term.param_names, data,
            param_defaults=defaults, param_prior_sigmas=sig,
            param_prior_means=means, param_is_poi=ispoi)
        declared = list(term.param_names)

    # ---- the quadratic term -------------------------------------------------
    if qd is not None:
        grad_chi2 = qd["grad"][isel].astype(np.float64)
        hess_chi2 = qd["hess"][np.ix_(isel, isel)].astype(np.float64)
        if args.whiten:
            inv = 1.0 / np.maximum(pscale, 1e-300)
            grad_chi2 = grad_chi2 * inv
            hess_chi2 = hess_chi2 * np.outer(inv, inv)
        if inject_card and not args.inject_res_only:
            dth = np.zeros(nfit)
            for nm, v in inject_card.items():
                if nm in nameidx and nameidx[nm] < nfit:
                    dth[nameidx[nm]] = v
            grad_chi2 = grad_chi2 - hess_chi2 @ dth
            log("  quadratic term: G -> G - K dtheta")
        sparse = nfit > args.dense_max
        hg, hh = MGT.to_hists(names, 0.5 * grad_chi2, 0.5 * hess_chi2, sparse)
        writer.add_external_likelihood_term(grad=hg, hess=hh, name="hitchi2")
        log(f"external term 'hitchi2': {nfit} parameters")

        poi_set = _poi_set(args.poi, names)
        undeclared = [nm for nm in names if nm not in declared]
        if undeclared:
            ki = [i for i, nm in enumerate(names) if nm in set(undeclared)]
            writer.add_auxiliary("global_params", {
                "params": [names[i] for i in ki],
                "defaults": np.zeros(len(ki)),
                "prior_sigmas": prior_sigmas[ki],
                "prior_means": np.zeros(len(ki)),
                "is_poi": np.array([1 if names[i] in poi_set else 0
                                    for i in ki], np.int64)})
        writer.add_auxiliary("global_index_map", {
            "params": names,
            "parmtype": np.asarray(parmtype[isel], np.int64),
            "subidx": np.asarray(subidx[isel], np.int64),
            "prior_sigmas": prior_sigmas,
            "scale": pscale,
            "group_params": gparams_grp,
            "group_units": group_units,
            "hit_params": hparams,
            "provenance": [json.dumps(dict(argv=sys.argv, arm=args.arm,
                                           comps=args.comps,
                                           inject=inject_card))]})

    out = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    writer.write(os.path.dirname(out), os.path.basename(out))
    log(f"wrote {out} ({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
