#!/usr/bin/env python3
"""Build a rabbit datacard whose mass term is parameterised by the CVH fit's
PHYSICAL parameters instead of the four ad-hoc per-family resolution knobs.

The card carries, over ONE set of parameters:

* the **quadratic hit-chi2 term** over the parmtype-14 field modes and the
  parmtype-15 material groups (from ``globalfit/extract.py``: ``G``, ``K``),
  written exactly as ``globalfit/make_global_term.py`` writes it -- same
  chi2 -> NLL factor of 2, same whitening, same trimming, same injection
  convention ``G -> G - K dtheta``;
* a **``MaterialCFTerm``** whose resolution exponent is
  ``S_f(tau) = S_f^fix(tau) + sum_g exp(k_g) S_{f,g}(tau)`` over the SAME
  ``material_<group>`` parameters, plus a per-hit-class Gaussian share
  ``v_i = v_other,i + sum_c (1 + eps_c) v_{c,i}`` over new ``hitres_<class>``
  parameters;
* the sparse ``D`` rows ``dm_i/dtheta`` on the field AND material parameters,
  so field and alignment enter the mass term only through the MEAN.

The material parameters therefore appear in three places at once -- the
quadratic term's curvature, the mass term's width, and the mass term's mean --
which is the whole point: one fit, one set of parameters, three constraints.

Inputs
------
``--groups-npz``  the ``matres/extract_groups.py`` output (per-group exponents,
                  per-class hit variances, sigma / m0 / mgen / chi2ndof, D rows)
``--quad-npz``    a ``globalfit/extract.py`` output for the quadratic term
                  (``--no-mass`` is enough and takes ~140 s / 48 files).  The
                  two need not have the same candidates: the quadratic term is
                  a sum over ITS candidates and the mass term a product over
                  ITS own.  When they ARE the same production the same
                  trimming must be applied to both, which is what
                  ``--max-chi2-ndof`` does on each side.

Usage
-----
    python make_material_card.py \\
      --groups-npz runs/matres/gun_groups.npz \\
      --quad-npz   runs/matres/gun_quad.npz \\
      --groups /work/.../materialGroups50.txt \\
      --whiten --max-chi2-ndof 3 -o runs/matres/cards/joint.hdf5
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.dirname(_HERE)
_GF = os.path.join(_RES, "globalfit")
for _p in (_HERE, _RES, _GF):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import groups as G  # noqa: E402
import make_global_term as MGT  # noqa: E402

MJPSI = 3.0969


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--groups-npz", required=True)
    p.add_argument("--quad-npz", default=None)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--groups", default=None, help="materialGroups tier file")
    p.add_argument("--coeffs", default=None, help="mode dump for --whiten")
    p.add_argument("--parmtypes", type=int, nargs="*", default=[14, 15])
    p.add_argument("--whiten", action="store_true")
    p.add_argument("--no-quadratic", action="store_true")
    p.add_argument("--no-mass", action="store_true")
    p.add_argument("--no-jac", action="store_true")
    p.add_argument("--no-hits", action="store_true",
                   help="drop the per-class hit parameters (the two-track "
                        "productions export no per-hit blocks, so this is the "
                        "default there)")
    p.add_argument("--amount-mode", choices=["exp", "linear"], default="exp")
    p.add_argument("--hit-mode", choices=["exp", "linear"], default="linear")
    p.add_argument("--legacy-families", action="store_true",
                   help="ALSO add the old per-family k knobs (k_ms, k_ioni, "
                        "...) on top of the physical parameterisation, for the "
                        "comparison fits only")
    p.add_argument("--max-chi2-ndof", type=float, default=0.0)
    p.add_argument("--maxn", type=int, default=0, help="use only N candidates")
    p.add_argument("--prune-frac", type=float, default=0.0,
                   help="fold groups contributing less than this fraction of "
                        "the candidate's max |S| into the fixed baseline")
    p.add_argument("--freeze-zero-info", action="store_true",
                   help="drop groups with no exponent anywhere in the sample")
    p.add_argument("--kernel-cache", default=None)
    p.add_argument("--phik-points", type=int, default=8192)
    p.add_argument("--phik-cache", default=None)
    p.add_argument("--with-alpha", action="store_true")
    p.add_argument("--mref", type=float, default=MJPSI)
    p.add_argument("--window", type=float, default=1.0)
    p.add_argument("--chunk", type=int, default=16384)
    p.add_argument("--floor", choices=["softplus", "clip", "none"],
                   default="softplus")
    p.add_argument("--fbkg", type=float, default=0.0)
    p.add_argument("--field-prior", type=float, default=0.0)
    p.add_argument("--material-prior-scale", type=float, default=1.0)
    p.add_argument("--hit-prior", type=float, default=0.0,
                   help="Gaussian prior sigma on every hitres_<class>")
    p.add_argument("--poi", default="bfield",
                   help="'bfield' | 'material' | 'all' | 'none' | comma list")
    p.add_argument("--inject", action="append", default=[],
                   metavar="NAME:VALUE",
                   help="a TRUE correction dtheta, applied consistently to "
                        "every term: quadratic G -> G - K dtheta, mass mean "
                        "mobs += D_card dtheta, and -- for a material group -- "
                        "the mass-term EXPONENTS of that group scaled by "
                        "A(dtheta) (the data would have had that much more "
                        "material)")
    p.add_argument("--inject-quad-only", action="store_true")
    p.add_argument("--inject-mass-only", action="store_true")
    p.add_argument("--inject-mean-only", action="store_true",
                   help="inject into the mean but NOT into the exponents "
                        "(isolates the D-row channel)")
    p.add_argument("--dense-max", type=int, default=4000)
    return p.parse_args()


def log(msg):
    print(msg, flush=True)


def main():
    args = parse_args()
    d = np.load(args.groups_npz, allow_pickle=False)
    keys = set(d.files)
    gnames_all = [str(x) for x in d["group_names"]]
    cnames_all = [str(x) for x in d["hit_classes"]]
    fams = [str(x) for x in d["families"]]
    ngroups = len(gnames_all)

    # ---- the catalog of D / quadratic columns -----------------------------
    if "fit_parmtype" not in keys and not args.no_jac:
        sys.exit("the groups npz has no D-row catalog; re-extract without "
                 "--no-jac, or pass --no-jac here")
    qd = None
    if args.quad_npz and not args.no_quadratic:
        qd = np.load(args.quad_npz, allow_pickle=False)
        parmtype, subidx = qd["parmtype"], qd["subidx"]
    else:
        parmtype = d.get("fit_parmtype")
        subidx = d.get("fit_subidx")
        if parmtype is None:
            sys.exit("no quadratic npz and no D catalog: nothing to name")
    sel = np.isin(parmtype, args.parmtypes)
    isel = np.where(sel)[0]
    names, prior_sigmas = MGT.name_params(
        parmtype[isel], subidx[isel], args.groups, args.field_prior,
        args.material_prior_scale)
    nfit = len(names)
    log(f"{nfit} global parameters: " + ", ".join(
        f"parmtype{pt} x{int((parmtype[isel]==pt).sum())}"
        for pt in sorted(set(parmtype[isel].tolist()))))

    # the material parameters, in GROUP-INDEX order, with the names the
    # quadratic term uses -- this is the join between the two terms
    matcol = {int(si): j for j, (pt, si)
              in enumerate(zip(parmtype[isel], subidx[isel])) if int(pt) == 15}
    if len(matcol) != ngroups:
        log(f"WARNING: {len(matcol)} parmtype-15 columns for {ngroups} groups")
    group_params = [names[matcol[g]] if g in matcol else f"material_group{g}"
                    for g in range(ngroups)]
    hit_params = [f"hitres_{c}" for c in cnames_all]

    # ---- whitening ---------------------------------------------------------
    pscale = np.ones(nfit)
    if args.whiten:
        pscale, _ = MGT.param_scales(parmtype[isel], subidx[isel],
                                     args.coeffs, args.groups)
        prior_sigmas = prior_sigmas * pscale
        log("whitened: 1 card unit = 1 T of RMS |dB| (parmtype 14) / 1 group "
            f"prior sigma (parmtype 15); scale {pscale.min():.3e} .. "
            f"{pscale.max():.3e}")
    # k_phys = value / s  (theta_card = theta_raw * s)
    group_units = np.array([1.0 / pscale[matcol[g]] if g in matcol else 1.0
                            for g in range(ngroups)])

    # ---- injection vector --------------------------------------------------
    inject = np.zeros(nfit)
    nameidx = {nm: i for i, nm in enumerate(names)}
    for spec in args.inject:
        nm, val = spec.rsplit(":", 1)
        if nm not in nameidx:
            sys.exit(f"--inject for unknown parameter '{nm}'")
        inject[nameidx[nm]] = float(val)
    if args.inject:
        log("injecting " + ", ".join(
            f"{nm}={inject[nameidx[nm]]:+.5g}" for nm in names
            if inject[nameidx[nm]]))

    # ---- quadratic term ----------------------------------------------------
    grad_chi2 = hess_chi2 = None
    if qd is not None:
        grad_chi2 = qd["grad"][isel].astype(np.float64)
        hess_chi2 = qd["hess"][np.ix_(isel, isel)].astype(np.float64)
        if args.whiten:
            inv = 1.0 / np.maximum(pscale, 1e-300)
            grad_chi2 = grad_chi2 * inv
            hess_chi2 = hess_chi2 * np.outer(inv, inv)
        if args.inject and not args.inject_mass_only:
            grad_chi2 = grad_chi2 - hess_chi2 @ inject
            log("  quadratic term: G -> G - K dtheta")

    # ---- mass term ---------------------------------------------------------
    from rabbit import tensorwriter, unbinned

    writer = tensorwriter.TensorWriter()
    writer.add_dummy_channel(name="calib_dummy")
    declared = []
    term = None

    if not args.no_mass:
        n_all = len(d["sigma"])
        keep = np.ones(n_all, bool)
        if args.max_chi2_ndof > 0.0:
            keep = d["chi2ndof"] < args.max_chi2_ndof
            log(f"chi2/ndof < {args.max_chi2_ndof:g}: {int(keep.sum())}/{n_all} "
                f"mass candidates ({100.*keep.mean():.2f} %)")
        idx = np.where(keep)[0]
        if args.maxn and args.maxn < len(idx):
            idx = idx[: args.maxn]
        n = len(idx)

        ptr = d["grp_ptr"].astype(np.int64)
        gid = d["grp_id"].astype(np.int64)
        # restrict the CSR store to the kept candidates
        cnt = np.diff(ptr)[idx]
        rows = np.concatenate([np.arange(ptr[i], ptr[i + 1]) for i in idx]) \
            if n else np.zeros(0, np.int64)
        nptr = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
        ngid = gid[rows]

        # per-group amplitude, for pruning and for the leverage report
        gid_full = ngid.copy()
        amp = np.zeros(len(rows))
        arrs = {}
        for f in fams:
            a = d["S" + f][rows]
            arrs[f] = a
            amp = np.maximum(amp, np.abs(a).max(axis=1))
        seg = np.repeat(np.arange(n), cnt)
        top = np.zeros(n)
        np.maximum.at(top, seg, amp)
        drop = np.zeros(len(rows), bool)
        if args.prune_frac > 0.0:
            drop = amp < args.prune_frac * top[seg]
            log(f"pruning: {int(drop.sum())}/{len(rows)} rows "
                f"({100.*drop.mean():.2f} %) folded into the fixed baseline")
        if args.freeze_zero_info:
            live = np.zeros(ngroups, bool)
            live[np.unique(ngid[~drop])] = True
            dead = [g for g in range(ngroups) if not live[g]]
            if dead:
                log("groups with NO exponent in this sample (folded): "
                    + ", ".join(gnames_all[g] for g in dead))
                drop = drop | np.isin(ngid, dead)

        nt = len(d["tgrid"])
        gfam, fixed = [], {}
        for f in fams:
            e = {"name": f}
            a = arrs[f]
            fx = np.zeros((n, nt), np.float64)
            if drop.any():
                np.add.at(fx, seg[drop], a[drop].astype(np.float64))
            if "Sfix" + f in keys:
                fx += d["Sfix" + f][idx].astype(np.float64)
            comp = "im" if f.endswith("_im") else "re"
            e["nm"] = f[:-3] if f.endswith(("_re", "_im")) else f
            e["comp"] = comp
            e["arr"] = a[~drop]
            e["fix"] = fx
            gfam.append(e)
        # merge the re/im halves of a family under one name
        merged = {}
        for e in gfam:
            m = merged.setdefault(e["nm"], {"name": e["nm"]})
            m[e["comp"]] = e["arr"].astype(np.float32)
            m["fix_" + e["comp"]] = e["fix"].astype(np.float32)
        group_families = [merged[k] for k in sorted(merged)]

        # renumber the CSR after pruning
        kcnt = np.zeros(n, np.int64)
        np.add.at(kcnt, seg[~drop], 1)
        nptr = np.concatenate([[0], np.cumsum(kcnt)]).astype(np.int64)
        ngid = ngid[~drop]

        data = {
            "sigma": d["sigma"][idx].astype(np.float64),
            "mobs": d["m0"][idx].astype(np.float64) - args.mref,
            "tgrid": np.asarray(d["tgrid"], np.float64),
            "grp_ptr": nptr,
            "grp_id": ngid,
            "group_units": group_units,
        }
        for m in group_families:
            for c in ("re", "im"):
                if c in m:
                    data[f"Sg_{c}_{m['name']}"] = m[c]
                if "fix_" + c in m and np.any(m["fix_" + c]):
                    data[f"Sgfix_{c}_{m['name']}"] = m["fix_" + c]
                elif "fix_" + c in m:
                    m.pop("fix_" + c)

        # ---- hit classes ---------------------------------------------------
        use_hits = (not args.no_hits) and "hit_ptr" in keys and \
            int(d["hit_ptr"][-1]) > 0
        vgf = d["vgf"][idx].astype(np.float64)
        if use_hits:
            hptr = d["hit_ptr"].astype(np.int64)
            hrows = np.concatenate(
                [np.arange(hptr[i], hptr[i + 1]) for i in idx]) if n else \
                np.zeros(0, np.int64)
            hcnt = np.diff(hptr)[idx]
            data["hit_ptr"] = np.concatenate([[0], np.cumsum(hcnt)]).astype(np.int64)
            data["hit_cls"] = d["hit_cls"][hrows].astype(np.int64)
            data["hit_v"] = d["hit_v"][hrows].astype(np.float64)
            data["vg_other"] = d["vg_other"][idx].astype(np.float64)
            data["hit_units"] = np.ones(len(hit_params))
            share = (data["hit_ptr"], data["hit_cls"], data["hit_v"],
                     data["vg_other"])
            log(f"per-class hit share: {len(hit_params)} classes, "
                f"{len(data['hit_cls'])/max(n,1):.2f} rows/candidate")
        else:
            # No floating class, but the Gaussian remainder must still be in
            # the model: in the flat MassCFTerm it rides as the `gauss` family
            # and it is the DOMINANT part of the mass CF.  It goes in as
            # `vg_other` with an empty class list, i.e. a fixed contribution.
            hit_params = []
            data["hit_ptr"] = np.zeros(n + 1, np.int64)
            data["hit_cls"] = np.zeros(0, np.int64)
            data["hit_v"] = np.zeros(0, np.float64)
            data["vg_other"] = vgf
            share = (data["hit_ptr"], data["hit_cls"], data["hit_v"], vgf)
            log("NO per-class hit share (the two-track maker exports no "
                "parmtype-8/9 resolution blocks); the Gaussian remainder "
                "enters as a FIXED vg_other")

        # ---- injection into the mass term ----------------------------------
        legacy = []
        if args.legacy_families:
            legacy = [{"name": "hit", "param": "k_hit", "kind": "gauss"}]
        if args.inject and not args.inject_quad_only:
            # (a) the MEAN, through the D rows
            if not args.no_jac and "D" in keys:
                Dphys = d["D"][idx][:, isel].astype(np.float64)
                if args.whiten:
                    Dphys = Dphys / np.maximum(pscale, 1e-300)[None, :]
                shift = (-Dphys) @ inject
                data["mobs"] = data["mobs"] + shift
                log(f"  mass mean: m_i^0 += D_card.dtheta, mean "
                    f"{shift.mean()*1e3:+.4f} MeV, rms {shift.std()*1e3:.4f} MeV")
            # (b) the WIDTH, through the exponents of the injected groups
            if not args.inject_mean_only:
                w = np.ones(ngroups)
                for g in range(ngroups):
                    if g in matcol and inject[matcol[g]]:
                        k = inject[matcol[g]] * group_units[g]
                        w[g] = np.exp(k) if args.amount_mode == "exp" else 1.0 + k
                if np.any(w != 1.0):
                    for m in group_families:
                        for c in ("re", "im"):
                            if c in m:
                                m[c] = (m[c] * w[ngid][:, None]).astype(np.float32)
                                data[f"Sg_{c}_{m['name']}"] = m[c]
                    log("  mass width: group exponents scaled by " + ", ".join(
                        f"{gnames_all[g]} x{w[g]:.5f}"
                        for g in range(ngroups) if w[g] != 1.0))

        # ---- D rows ---------------------------------------------------------
        jac, jac_params = None, []
        if not args.no_jac and "D" in keys:
            D = d["D"][idx][:, isel].astype(np.float64)
            if args.whiten:
                D = D / np.maximum(pscale, 1e-300)[None, :]
            D = -D  # MassCFTerm sign: its rows are d(predicted mass)/dtheta
            r_, c_ = np.nonzero(D)
            ji = np.stack([r_, c_], axis=1).astype(np.int64)
            jv = D[r_, c_]
            data["jac_indices"] = ji
            data["jac_values"] = jv
            data["jac_shape"] = np.array([n, nfit], np.int64)
            jac, jac_params = (ji, jv, (n, nfit)), names
            log(f"sparse D: {len(jv)} entries "
                f"({100.*len(jv)/max(n*nfit,1):.1f} % fill)")

        # ---- FSR kernel ------------------------------------------------------
        dm = None
        if args.kernel_cache:
            dm = np.load(args.kernel_cache)["dm"]
        elif "mgen" in keys:
            dm = d["mgen"][idx] - args.mref
            log("FSR kernel from the extraction's own gen masses")
        if dm is not None:
            tmax = data["tgrid"][-1] / data["sigma"].min()
            tabs, phik = MGT.build_phik_table(
                dm, tmax, args.phik_points,
                read=(args.phik_cache,) if args.phik_cache else (),
                write=args.phik_cache)
            data["phik_t"] = tabs
            data["phik_re"] = phik.real.copy()
            data["phik_im"] = phik.imag.copy()

        window = (args.mref - 0.5 * args.window, args.mref + 0.5 * args.window)
        term = unbinned.MaterialCFTerm(
            "mass",
            sigma=data["sigma"], mobs=data["mobs"], tgrid=data["tgrid"],
            families=legacy, vgf=vgf,
            group_params=group_params,
            group_families=[{k: v for k, v in m.items()
                             if k in ("name", "re", "im", "fix_re", "fix_im")}
                            for m in group_families],
            grp_ptr=data["grp_ptr"], grp_id=data["grp_id"],
            group_units=group_units,
            hit_params=hit_params, hit_share=share,
            amount_mode=args.amount_mode, hit_mode=args.hit_mode,
            background=unbinned.UniformBackground(window),
            m_ref=args.mref,
            scale_param="alpha" if args.with_alpha else None,
            bkg_frac=args.fbkg, floor=args.floor, chunk=args.chunk,
            channel="jpsi", jac=jac, jac_params=jac_params,
            phik=(data["phik_t"], data["phik_re"], data["phik_im"])
            if "phik_t" in data else None,
        )
        # `vg_other` STAYS in the written datasets whether or not any class
        # floats: it is what carries the Gaussian remainder into the term.
        data["vgf"] = vgf

        # SANITY GATE.  A card whose NLL or gradient is not finite AT ITS OWN
        # STARTING POINT cannot be fitted -- rabbit's Cholesky of the Hessian
        # fails and the covariance is lost, with a message that says nothing
        # about the cause.  Catch it here, where the cause is still visible.
        import tensorflow as tf

        _x0 = tf.Variable(np.zeros(len(term.param_names)), dtype=tf.float64)
        with tf.GradientTape() as _tp:
            _v = term.nll(_x0)
        _g = np.asarray(_tp.gradient(_v, _x0).numpy())
        _li = term.raw_density(tf.constant(np.zeros(len(term.param_names)))).numpy()
        log(f"NLL(0) = {float(_v.numpy()):.6f}, max|grad| = "
            f"{np.abs(_g).max():.4g}, raw density min {_li.min():.4g} "
            f"({int((_li <= 0).sum())} candidates <= 0)")
        if not (np.isfinite(float(_v.numpy())) and np.isfinite(_g).all()):
            sys.exit("the mass term is not finite at k = 0 -- refusing to write "
                     "the card.  The usual cause is a missing Gaussian share "
                     "(vg_other / a gauss family): without it the model is far "
                     "too narrow and the density underflows.")

        poi_set = _poi_set(args.poi, names)
        prior_by_name = dict(zip(names, prior_sigmas))
        defaults, sig, means, ispoi = [], [], [], []
        for p in term.param_names:
            if p == "alpha":
                defaults.append(0.0)
                sig.append(np.nan)
                ispoi.append(1)
            elif p.startswith("hitres_"):
                defaults.append(0.0)
                sig.append(args.hit_prior if args.hit_prior > 0 else np.nan)
                ispoi.append(0)
            elif p in prior_by_name:
                defaults.append(0.0)
                sig.append(prior_by_name[p])
                ispoi.append(1 if p in poi_set else 0)
            else:  # a legacy family knob
                defaults.append(1.0)
                sig.append(np.nan)
                ispoi.append(0)
            means.append(0.0 if np.isfinite(sig[-1]) else defaults[-1])
        writer.add_unbinned_term(
            "mass", term.config(), term.param_names, data,
            param_defaults=defaults, param_prior_sigmas=sig,
            param_prior_means=means, param_is_poi=ispoi)
        declared = list(term.param_names)
        log(f"unbinned term: {n} candidates, {len(term.param_names)} parameters "
            f"({len(group_params)} groups, {len(hit_params)} hit classes, "
            f"{len(jac_params)} D columns)")
        _report_leverage(gnames_all, gid_full, amp, drop, n)

    # ---- write --------------------------------------------------------------
    if grad_chi2 is not None:
        sparse = nfit > args.dense_max
        H = 0.5 * hess_chi2
        g = 0.5 * grad_chi2
        hg, hh = MGT.to_hists(names, g, H, sparse)
        writer.add_external_likelihood_term(grad=hg, hess=hh, name="hitchi2")
        log(f"external term 'hitchi2': {nfit} parameters")

    poi_set = _poi_set(args.poi, names)
    undeclared = [nm for nm in names if nm not in declared]
    if undeclared:
        keepi = [i for i, nm in enumerate(names) if nm in set(undeclared)]
        writer.add_auxiliary("global_params", {
            "params": [names[i] for i in keepi],
            "defaults": np.zeros(len(keepi)),
            "prior_sigmas": prior_sigmas[keepi],
            "prior_means": np.zeros(len(keepi)),
            "is_poi": np.array([1 if names[i] in poi_set else 0
                                for i in keepi], np.int64)})
    writer.add_auxiliary("global_index_map", {
        "params": names,
        "parmtype": np.asarray(parmtype[isel], np.int64),
        "subidx": np.asarray(subidx[isel], np.int64),
        "prior_sigmas": prior_sigmas,
        "scale": pscale,
        "injected": inject,
        "group_params": group_params,
        "group_units": group_units,
        "hit_params": hit_params if not args.no_hits else [],
        "provenance": [json.dumps(dict(argv=sys.argv,
                                       amount_mode=args.amount_mode,
                                       hit_mode=args.hit_mode,
                                       whiten=bool(args.whiten)))]})
    # TensorWriter.write takes (folder, filename), not a path
    out = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    writer.write(os.path.dirname(out), os.path.basename(out))
    log(f"wrote {out} ({os.path.getsize(out)/1e6:.1f} MB)")


def _poi_set(spec, names):
    if spec == "bfield":
        return {n for n in names if n.startswith("bfield_mode")}
    if spec == "material":
        return {n for n in names if n.startswith("material_")}
    if spec == "all":
        return set(names)
    if spec == "none":
        return set()
    return {s for s in spec.split(",") if s}


def _report_leverage(gnames, gid, amp, drop, n):
    """Per-group leverage: the share of the sample's total max_tau |S| each
    group carries, and the fraction of candidates it appears in."""
    lev = np.zeros(len(gnames))
    np.add.at(lev, gid, amp)
    occ = np.bincount(gid, minlength=len(gnames))
    kept = np.zeros(len(gnames), np.int64)
    np.add.at(kept, gid[~drop], 1)
    tot = lev.sum() or 1.0
    log("  per-group leverage (share of sum_i max_tau |S|, candidates touched,"
        " rows kept after pruning):")
    for g in np.argsort(-lev):
        if occ[g] == 0:
            continue
        log(f"    {gnames[g]:<26} {100.*lev[g]/tot:6.2f} %  "
            f"{100.*occ[g]/max(n,1):6.1f} % of cand   "
            f"{100.*kept[g]/max(occ[g],1):5.1f} % rows kept")


if __name__ == "__main__":
    main()
