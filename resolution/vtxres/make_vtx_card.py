#!/usr/bin/env python3
"""Build a rabbit datacard for the VERTEX-CONSTRAINT RESIDUAL term, the MASS
term, or BOTH over one parameter set.

The vertex term is the mass term with a DELTA KERNEL AT ZERO:

    mobs  = r_v   (the fitted track-track PCA distance, cm)
    sigma = sigma_v
    m_ref = 0,  no kernel, no theory, no PDG input

so it is a PURE RESOLUTION term.  Everything else -- the per-material-group
exponent `S_f(tau) = S_f^fix + sum_g exp(k_g) S_{f,g}` and the per-hit-class
Gaussian share `v = v_other + sum_c (1 + eps_c) v_c` -- is the SAME
`MaterialCFTerm` the mass term uses, on the SAME parameters, which is what
lets a joint fit float one set.

Arms (`--arm`)
    cf      the exported log-CF exponents
    gauss   each (row, family) replaced by -1/2 kappa2 tau^2 with kappa2 read
            off the SAME arrays (tau^4 eliminated); imaginary parts dropped
    gaussq  the variance the FIT used: `Gvqms` / `Gvqio` per group, nothing
            radiative or delta -- i.e. the Gaussian chi2 the CF is measured
            against.  `sum_g (vqms+vqio) + vgf == 1` exactly.

usage:
  python3 make_vtx_card.py --vtx-npz vtx.npz --mass-npz mass.npz \
      --groups .../materialGroups50.txt --whiten --arm cf -o cards/joint.hdf5
"""
import argparse, json, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.dirname(_HERE)
_MAT = os.path.join(_RES, "matres")
_GF = os.path.join(_RES, "globalfit")
for _p in (_HERE, _RES, _MAT, _GF):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import groups as G          # noqa: E402
import make_global_term as MGT  # noqa: E402

MJPSI = 3.0969
FAMS = ("ms", "io_re", "io_im", "rad_re", "rad_im")


def log(*a):
    print(*a, flush=True)


def kappa2_from_grid(S, tgrid):
    """-S''(0) from the tabulated exponent, tau^4 term eliminated:
    S(tau) = -kappa2 tau^2/2 + c tau^4 + O(tau^6), so with t1 and 2 t1
    (grid points 1 and 2 of a uniform grid) 16 S(t1) - S(2 t1) = -6 kappa2 t1^2."""
    t1 = float(tgrid[1])
    return -(16.0 * S[:, 1] - S[:, 2]) / (6.0 * t1 * t1)


def build_term(name, npz, arm, args, group_units, gparams, hparams, ngroups,
               inj_groups, inj_hits, idx=None):
    """One MaterialCFTerm from an `extract_vtx.py` npz."""
    idx_in = idx
    from rabbit import unbinned
    d = np.load(npz, allow_pickle=False)
    keys = set(d.files)
    n_all = len(d["sigma"])
    keep = np.ones(n_all, bool)
    if args.max_chi2_ndof > 0:
        keep &= d["chi2ndof"] < args.max_chi2_ndof
    if idx is None:
        idx = np.where(keep)[0]
        if args.maxn and args.maxn < len(idx):
            idx = idx[: args.maxn]
    # THE ONE CUT ON THE RESIDUAL, and why it exists.  A Gaussian density at
    # |z| = 40 is exp(-800) = 1e-348, BELOW the smallest float64 (2.2e-308):
    # the `gauss`/`gaussq` arms' log-density underflows to -inf and their NLL
    # is +inf, so no comparison is possible at all.  The cut is applied
    # IDENTICALLY to every arm and every channel so the comparison stays
    # like-for-like, and it removes ~2 candidates in 16 000 (0.012 %) whose
    # pull is beyond 40 sigma.  The UNCUT tails are reported separately
    # (`plot_vtx.py --densities`), so nothing about the tail is hidden by it.
    if args.max_abs_z > 0 and idx_in is None:
        mref0 = 0.0 if name == "vtx" else MJPSI
        zz = np.abs(d["m0"] - mref0) / np.maximum(d["sigma"], 1e-300)
        bad = zz[idx] > args.max_abs_z
        if bad.any():
            log(f"  {name}: dropping {int(bad.sum())} of {len(idx)} candidates "
                f"with |z| > {args.max_abs_z:g} (the Gaussian arms' float64 "
                f"underflow); max |z| kept {zz[idx][~bad].max():.1f}")
            idx = idx[~bad]
    n = len(idx)
    tg = np.asarray(d["tgrid"], np.float64)
    nt = len(tg)

    ptr = d["grp_ptr"].astype(np.int64)
    gid_all = d["grp_id"].astype(np.int64)
    cnt = np.diff(ptr)[idx]
    rows = (np.concatenate([np.arange(ptr[i], ptr[i + 1]) for i in idx])
            if n else np.zeros(0, np.int64))
    gid = gid_all[rows]
    seg = np.repeat(np.arange(n), cnt)

    # ---- the ARM ----------------------------------------------------------
    arrs = {}
    if arm == "cf":
        for f in FAMS:
            arrs[f] = d["S" + f][rows].astype(np.float64)
    elif arm == "gauss":
        t2 = -0.5 * tg ** 2
        for f in FAMS:
            if f.endswith("_im"):
                arrs[f] = np.zeros((len(rows), nt))
                continue
            arrs[f] = np.outer(kappa2_from_grid(d["S" + f][rows].astype(np.float64), tg), t2)
    elif arm == "gaussq":
        if "Gvqms" not in keys:
            sys.exit("the extraction carries no Gvqms/Gvqio -- rerun the "
                     "production with the per-group fit-Q export")
        t2 = -0.5 * tg ** 2
        for f in FAMS:
            if f == "ms":
                arrs[f] = np.outer(d["Gvqms"][rows].astype(np.float64), t2)
            elif f == "io_re":
                arrs[f] = np.outer(d["Gvqio"][rows].astype(np.float64), t2)
            else:
                arrs[f] = np.zeros((len(rows), nt))
    else:
        raise ValueError(arm)

    # ---- prune, and freeze the groups no candidate scales -----------------
    amp = np.zeros(len(rows))
    for f in FAMS:
        amp = np.maximum(amp, np.abs(arrs[f]).max(axis=1))
    top = np.zeros(n)
    if len(rows):
        np.maximum.at(top, seg, amp)
    drop = (amp < args.prune_frac * top[seg]) if args.prune_frac > 0 else \
        np.zeros(len(rows), bool)
    live = np.zeros(ngroups, bool)
    if len(rows):
        live[np.unique(gid[~drop])] = True
    drop = drop | ~live[gid]

    # ---- the injection, on the DATA side ----------------------------------
    w = np.ones(ngroups)
    for g, k in (inj_groups or {}).items():
        w[g] = np.exp(k) if args.amount_mode == "exp" else 1.0 + k
    fam_out = {}
    for f in FAMS:
        nm = f[:-3] if f.endswith(("_re", "_im")) else f
        comp = "im" if f.endswith("_im") else "re"
        a_ = arrs[f] * w[gid][:, None]
        fx = np.zeros((n, nt))
        if drop.any():
            np.add.at(fx, seg[drop], a_[drop])
        e = fam_out.setdefault(nm, {"name": nm})
        e[comp] = a_[~drop].astype(np.float32)
        if np.any(fx):
            e["fix_" + comp] = fx.astype(np.float32)
    gfam = [fam_out[k] for k in sorted(fam_out)]
    kcnt = np.zeros(n, np.int64)
    if len(rows):
        np.add.at(kcnt, seg[~drop], 1)
    nptr = np.concatenate([[0], np.cumsum(kcnt)]).astype(np.int64)
    ngid = gid[~drop]

    # ---- hit classes ------------------------------------------------------
    sigma = d["sigma"][idx].astype(np.float64)
    vgf = d["vgf"][idx].astype(np.float64)
    hptr_all = d["hit_ptr"].astype(np.int64)
    hcnt = np.diff(hptr_all)[idx]
    hrows = (np.concatenate([np.arange(hptr_all[i], hptr_all[i + 1]) for i in idx])
             if n else np.zeros(0, np.int64))
    hcls = d["hit_cls"][hrows].astype(np.int64)
    hv = d["hit_v"][hrows].astype(np.float64)
    nhptr = np.concatenate([[0], np.cumsum(hcnt)]).astype(np.int64)
    hseg = np.repeat(np.arange(n), hcnt) if len(hrows) else np.zeros(0, np.int64)
    vsum0 = np.zeros(n)
    if len(hrows):
        np.add.at(vsum0, hseg, hv)
    # `v_other` -- the part of the Gaussian share no CLASS scales -- is fixed
    # by the UNINJECTED shares; an injected class changes `hv` and, with it,
    # the TOTAL Gaussian share `vgf`.
    vother = vgf - vsum0
    for c, e in (inj_hits or {}).items():
        m = hcls == c
        hv = np.where(m, hv * ((1.0 + e) if args.hit_mode == "linear" else np.exp(e)), hv)
    if len(hrows):
        vsum1 = np.zeros(n)
        np.add.at(vsum1, hseg, hv)
        vgf = vother + vsum1
    if args.no_hits:
        share = (np.zeros(n + 1, np.int64), np.zeros(0, np.int64), np.zeros(0), vgf)
        hit_params = []
    else:
        share = (nhptr, hcls.astype(np.int64), hv, vother)
        hit_params = list(hparams)

    isvtx = name == "vtx"
    mref = 0.0 if isvtx else MJPSI
    mobs = d["m0"][idx].astype(np.float64) - mref
    data = {"sigma": sigma, "mobs": mobs, "tgrid": tg, "grp_ptr": nptr,
            "grp_id": ngid, "group_units": group_units,
            "hit_ptr": share[0], "hit_cls": share[1], "hit_v": share[2],
            "vg_other": share[3], "vgf": vgf,
            "hit_units": np.ones(len(hit_params))}
    for m in gfam:
        for c in ("re", "im"):
            if c in m:
                data[f"Sg_{c}_{m['name']}"] = m[c]
            if "fix_" + c in m:
                data[f"Sgfix_{c}_{m['name']}"] = m["fix_" + c]

    kw = {}
    if isvtx:
        # NO KERNEL and NO self-consistent-sigma correction.  The mean of the
        # vertex residual is zero BY CONSTRUCTION (both muons come from one
        # gen point on ideal geometry), and sigma_v is NOT proportional to
        # r_v the way sigma_m is to m -- the measured corr(sigma_v, z_v) is
        # consistent with zero, which is the justification, stated in
        # STATE.md with its number.
        wdw = float(args.vtx_window) if args.vtx_window > 0 else \
            float(8.0 * np.median(sigma))
        kw.update(background=unbinned.UniformBackground((-wdw, wdw)),
                  self_consistent_sigma=False, a_res=None)
    else:
        dm = d["mgen"][idx] - MJPSI
        tabs, phik = MGT.build_phik_table(dm, tg[-1] / max(sigma.min(), 1e-12), 4096)
        data["phik_t"] = tabs
        data["phik_re"] = phik.real.copy()
        data["phik_im"] = phik.imag.copy()
        kw.update(background=unbinned.UniformBackground((MJPSI - 0.5, MJPSI + 0.5)),
                  phik=(tabs, phik.real.copy(), phik.imag.copy()))

    term = unbinned.MaterialCFTerm(
        name, sigma=sigma, mobs=mobs, tgrid=tg, families=[], vgf=vgf,
        group_params=list(gparams),
        group_families=[{k: v for k, v in m.items()
                         if k in ("name", "re", "im", "fix_re", "fix_im")}
                        for m in gfam],
        grp_ptr=nptr, grp_id=ngid, group_units=group_units,
        hit_params=hit_params, hit_share=share,
        amount_mode=args.amount_mode, hit_mode=args.hit_mode,
        m_ref=mref, bkg_frac=0.0, floor=args.floor, chunk=args.chunk,
        channel="jpsi", **kw)
    log(f"{name} term: {n} candidates, {len(ngid)} group rows "
        f"({int(drop.sum())} pruned), {len(hcls)} hit rows, "
        f"{len(term.param_names)} parameters, arm '{arm}'")
    return term, data, idx


def common_index(npz_list, args):
    """The candidate set BOTH functionals keep, in order.

    A joint card pairs the two npz row by row, so the selection has to be the
    INTERSECTION -- and the |z| < max_abs_z guard has to be applied to both
    pulls before either term is built, not inside each one.
    """
    keep = None
    for name, npz in npz_list:
        d = np.load(npz, allow_pickle=False)
        k = np.ones(len(d["sigma"]), bool)
        if args.max_chi2_ndof > 0:
            k &= d["chi2ndof"] < args.max_chi2_ndof
        if args.max_abs_z > 0:
            mref0 = 0.0 if name == "vtx" else MJPSI
            k &= np.abs(d["m0"] - mref0) / np.maximum(d["sigma"], 1e-300) <= args.max_abs_z
        keep = k if keep is None else (keep & k)
    idx = np.flatnonzero(keep)
    if args.maxn and args.maxn < len(idx):
        idx = idx[: args.maxn]
    return idx


def _poi_set(spec, names):
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
    p = argparse.ArgumentParser()
    p.add_argument("--vtx-npz", default=None)
    p.add_argument("--mass-npz", default=None)
    p.add_argument("--groups", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--arm", choices=["cf", "gauss", "gaussq"], default="cf")
    p.add_argument("--mass-arm", default=None,
                   help="arm of the MASS term (default: same as --arm)")
    p.add_argument("--whiten", action="store_true")
    p.add_argument("--amount-mode", choices=["exp", "linear"], default="exp")
    p.add_argument("--hit-mode", choices=["exp", "linear"], default="linear")
    p.add_argument("--no-hits", action="store_true")
    p.add_argument("--max-chi2-ndof", type=float, default=3.0)
    p.add_argument("--maxn", type=int, default=0)
    p.add_argument("--prune-frac", type=float, default=1e-3)
    p.add_argument("--hit-prior", type=float, default=1.0)
    p.add_argument("--vtx-window", type=float, default=0.0)
    p.add_argument("--max-abs-z", type=float, default=40.0,
                   help="drop candidates beyond this pull: the Gaussian arms' "
                        "density underflows float64 there, so no arm could be "
                        "compared with them in. Applied to every arm alike.")
    p.add_argument("--floor", default="softplus")
    p.add_argument("--chunk", type=int, default=32768)
    p.add_argument("--poi", default="all")
    p.add_argument("--inject", nargs="*", default=[])
    p.add_argument("--same-candidates", action="store_true",
                   help="require the two npz to be the SAME candidates, and "
                        "use one index set for both (the JOINT card)")
    args = p.parse_args()
    if not (args.vtx_npz or args.mass_npz):
        sys.exit("nothing to build")
    if args.mass_arm is None:
        args.mass_arm = args.arm

    gmap, _ = G.read_groups(args.groups)
    ngroups = (max(gmap) + 1) if gmap else 0
    gparams, gpriors = G.group_param_names(ngroups, args.groups)
    gscale = gpriors.copy() if args.whiten else np.ones(ngroups)
    group_units = G.card_group_units(ngroups, args.groups, whiten=args.whiten)
    gprior_card = gpriors * gscale               # 1 prior sigma, card units
    import hitres_classes
    cnames = list(hitres_classes.CLASSES)
    hparams = [f"hitres_{c}" for c in cnames]

    inject_card = {}
    for spec in args.inject:
        nm, val = spec.rsplit(":", 1)
        inject_card[nm] = float(val)
    inj_groups = {g: inject_card[nm] * group_units[g]
                  for g, nm in enumerate(gparams) if nm in inject_card}
    inj_hits = {c: inject_card[nm] for c, nm in enumerate(hparams)
                if nm in inject_card}
    if inject_card:
        log("injecting " + ", ".join(f"{k}={v:+.6g}" for k, v in inject_card.items())
            + ("  -> physical k: " + ", ".join(
                f"{gparams[g]}={v:+.6g} ({100*(np.exp(v)-1):+.3f} % material)"
                for g, v in inj_groups.items()) if inj_groups else ""))

    from rabbit import tensorwriter
    writer = tensorwriter.TensorWriter()
    writer.add_dummy_channel(name="calib_dummy")

    import tensorflow as tf
    declared, idx = [], None
    _KEY = [None]
    if args.same_candidates:
        idx = common_index([(nm, npz) for nm, npz in
                            (("vtx", args.vtx_npz), ("mass", args.mass_npz))
                            if npz], args)
        log(f"--same-candidates: {len(idx)} candidates kept by BOTH functionals")
    for nm, npz, arm in (("vtx", args.vtx_npz, args.arm),
                         ("mass", args.mass_npz, args.mass_arm)):
        if not npz:
            continue
        term, data, idx_used = build_term(
            nm, npz, arm, args, group_units, gparams, hparams, ngroups,
            inj_groups, inj_hits, idx=(idx if args.same_candidates else None))
        if args.same_candidates and idx is None:
            idx = idx_used
        # GATE: a joint card pairs the two npz ROW BY ROW, so the candidate
        # keys must match exactly.
        if args.same_candidates:
            dd = np.load(npz, allow_pickle=False)
            if "event" in dd.files:
                k = np.stack([dd[q][idx] for q in ("run", "lumi", "event")], 1)
                if _KEY[0] is None:
                    _KEY[0] = k
                elif not np.array_equal(_KEY[0], k):
                    sys.exit("--same-candidates: the two npz do not carry the "
                             "same (run, lumi, event)")
        x0 = tf.Variable(np.zeros(len(term.param_names)), dtype=tf.float64)
        with tf.GradientTape() as tp:
            v = term.nll(x0)
        g = np.asarray(tp.gradient(v, x0).numpy())
        log(f"  {nm} NLL(0) = {float(v.numpy()):.6f}, max|grad| = {np.abs(g).max():.4g}")
        if not (np.isfinite(float(v.numpy())) and np.isfinite(g).all()):
            sys.exit(f"the {nm} term is not finite at theta = 0")
        poi_set = _poi_set(args.poi, term.param_names)
        prior_by_name = dict(zip(gparams, gprior_card))
        sig = [args.hit_prior if q.startswith("hitres_")
               else prior_by_name.get(q, np.nan) for q in term.param_names]
        writer.add_unbinned_term(
            nm, term.config(), term.param_names, data,
            param_defaults=[0.0] * len(term.param_names),
            param_prior_sigmas=sig,
            param_prior_means=[0.0] * len(term.param_names),
            param_is_poi=[1 if q in poi_set else 0 for q in term.param_names])
        declared = sorted(set(declared) | set(term.param_names))

    allp = list(gparams) + hparams
    units = np.concatenate([group_units, np.ones(len(hparams))])
    inj = np.zeros(len(allp))
    for i, nm in enumerate(allp):
        if nm in inject_card:
            inj[i] = inject_card[nm]
    writer.add_auxiliary("global_index_map", {
        "params": allp,
        "prior_sigmas": np.concatenate([gprior_card, np.full(len(hparams), args.hit_prior)]),
        "scale": np.concatenate([gscale, np.ones(len(hparams))]),
        "units": units, "injected": inj,
        "group_params": list(gparams), "group_units": group_units,
        "hit_params": hparams,
        "provenance": [json.dumps(dict(argv=sys.argv, arm=args.arm,
                                       mass_arm=args.mass_arm,
                                       inject=inject_card))]})
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out = os.path.abspath(args.output)
    writer.write(os.path.dirname(out), os.path.basename(out))
    log(f"-> {out} ({os.path.getsize(out)/1e6:.1f} MB, {len(declared)} parameters)")


if __name__ == "__main__":
    main()
