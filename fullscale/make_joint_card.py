#!/usr/bin/env python3
"""PHASE 2 -- the joint J/psi + Z card over ONE set of calibration parameters.

The Z channel cannot measure `m_Z`: `m_Z` and a momentum scale are exactly
degenerate in a single-resonance fit. The J/psi fixes the scale, and in this
card it does so THROUGH THE PHYSICS rather than through a free `alpha`: the
J/psi term carries a delta kernel at the PDG mass and no scale parameter at
all, so the only way its candidates can move is through the field modes and the
material amounts -- exactly the parameters the Z term also depends on. That is
what makes the transfer a calibration rather than a fitted offset.

Three terms over one parameter vector:

* the **quadratic hit-chi2 term** on parmtypes 14 (50 field modes) and 15 (42
  material groups), summed over the J/psi and the DY productions
  (`globalfit/extract.py --no-mass`), written the way
  `globalfit/make_global_term.py` writes it -- same `chi2 -> NLL` factor of
  1/2, same whitening, same `D_card = -dm/dtheta` sign convention;
* the **J/psi mass term**: delta kernel at `MJPSI`, `scale_param=None`, both
  corrections, and the sparse `D` on the 92 parameters;
* the **Z mass term** of `make_card.py`: the `ZGammaLineshape` provider with
  the banded FSR fold, the acceptance, a floated 5-term `K(m)`, `m_Z` and
  `Gamma_Z` as POIs, both corrections, and the same sparse `D`.

The resolution scales `k_*` stay FIXED at the MC truth in phase 2; phase 3
replaces them with the parmtype-15 amounts themselves (`MaterialCFTerm`). The
two mass terms SHARE those scales by name (`k_hit`, `k_ms`, ...): they are
multiplicative factors on the per-candidate material exponents, so one number
per family is the physical statement, and the two terms' declarations of them
are checked to agree rather than silently taking the first.

The `f_ang` caveat: `jpsimc_20M_260905` predates `Jpsi_covrefmom`, so the
J/psi leg's Jensen `s^2` cannot be made truth-free from the candidate itself.
`--jpsi-fang` supplies the MC-measured value (gun 0.086 / data 0.106); the DY
leg uses its own per-candidate `Jpsi_fang`.

WHO DECLARES THE 92 (`--declare`)
---------------------------------
A parameter must appear in the fit vector exactly once. `MassCFTerm` always
appends its `jac_params` to its own `param_names` -- there is no way to give a
term a sparse `D` without the term listing those names -- and `UnbinnedParams`
declares the union of every term's `param_names`. So with two mass terms both
carrying all 92 columns:

`--declare bundle` (default)
    ALL parameters -- the 92 calibration parameters and the terms' own POIs and
    resolution knobs -- are declared once, by the `global_params` auxiliary
    bundle, and the card is run with `--paramModel ExternalParams
    bundle:global_params` alone. The 92 are then declared by the bundle and NOT
    by a param model reading the terms.
`--declare unbinned`
    `make_global_term.py`'s split: the unbinned terms declare everything they
    list (via `UnbinnedParams`) and the bundle carries only what no term lists
    -- which, with a full `D`, is nothing, so the bundle is not written at all.

Both give the same likelihood; they differ only in which param model puts the
names into the fit vector. `bundle` is the default because it stays correct as
terms are added, and because it is the only one in which the 92 are declared by
the bundle. There is NO configuration of the current rabbit in which
`UnbinnedParams` and `ExternalParams bundle:global_params` can be combined on a
card whose mass terms carry all 92 columns: the two would declare the same
names and `build_tf_unbinned_terms` refuses the duplicate.

A CAVEAT ABOUT `--chunk`
------------------------
`chunkfit.ChunkedObjective(chunk=)` -- and therefore `fit.py --chunk` and
`run_phase1.sh`'s `CHUNK=262144` -- rebuilds a term's `_chunks` list but leaves
its `_jac_chunks` (the per-chunk sparse `D`) at the size the term was BUILT
with. On a card with a sparse `D` that is an immediate
`Incompatible shapes: [c] vs [chunk]` in `_chunk_residual`. Until `chunkfit`
re-slices the Jacobian, the chunk size a joint card is fitted at must be the
one it was WRITTEN at, i.e. this file's `--chunk`.

usage::

    # the two-term card (no J/psi cache yet)
    python make_joint_card.py --z-pairs runs/zpairs_dyv2_jac.npz \\
        --quad runs/quad_dyv2.npz --z-maxn 100000 --whiten \\
        --fsr ../zchannel/data/kern_loose_band3.3e-4.npz \\
        --acc ../zchannel/data/acc_loose_d8.json \\
        -o cards/joint_z.hdf5

    rabbit_fit.py cards/joint_z.hdf5 -o out/ -t 0 --unblind \\
        --paramModel ExternalParams bundle:global_params
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.join(os.path.dirname(HERE), "resolution")
_GF = os.path.join(_RES, "globalfit")
for _p in (HERE, _RES, _GF):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MJPSI = 3.0969
MZ_REF = 91.1876


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jpsi-pairs", default=None,
                   help="cf_inmaker.py pairs cache built with "
                        "--jac-parmtypes 14 15. OPTIONAL: without it the card "
                        "is the quadratic term + the Z term only, which is the "
                        "configuration that can be built today (there is no "
                        "J/psi pairs cache until jpsimc_20M_260906_v2 lands).")
    p.add_argument("--z-pairs", required=True)
    p.add_argument("--quad", nargs="+", required=True,
                   help="one or more globalfit/extract.py --no-mass outputs; "
                        "they are SUMMED (the parameter map is bit-identical "
                        "across the two productions, verified)")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--groups", default=None, help="materialGroups tier file")
    p.add_argument("--coeffs", default=None, help="mode dump for --whiten")
    p.add_argument("--whiten", action="store_true", default=True)
    p.add_argument("--no-whiten", dest="whiten", action="store_false")
    p.add_argument("--fsr", default=None)
    p.add_argument("--acc", default=None)
    p.add_argument("--shape", type=int, default=5)
    p.add_argument("--jpsi-maxn", type=int, default=0)
    p.add_argument("--z-maxn", type=int, default=0)
    p.add_argument("--jpsi-window", type=float, default=0.35,
                   help="|m_obs - MJPSI| halfwidth [GeV]")
    p.add_argument("--jpsi-fang", type=float, default=0.096,
                   help="the angular share of the mass variance on the J/psi "
                        "leg, which v1 cannot supply per candidate. The MC "
                        "measurements are 0.086 (gun) and 0.106 (data); the "
                        "midpoint is the default and the spread is a "
                        "systematic to scan.")
    p.add_argument("--field-prior", type=float, default=0.0)
    p.add_argument("--material-prior-scale", type=float, default=1.0)
    p.add_argument("--max-chi2-ndof", type=float, default=3.0)
    p.add_argument("--max-sigma-rel", type=float, default=0.10)
    p.add_argument("--chunk", type=int, default=32768,
                   help="candidates per accumulation chunk. CHOOSE IT HERE, "
                        "not at fit time: `chunkfit.ChunkedObjective(chunk=)` "
                        "rebuilds a term's `_chunks` but NOT its `_jac_chunks`, "
                        "so a card with a sparse D crashes ('Incompatible "
                        "shapes: [c] vs [chunk]') if the fit driver re-chunks "
                        "it. `fit.py --chunk` and `run_phase1.sh`'s "
                        "CHUNK=262144 are therefore unusable on a joint card "
                        "until chunkfit re-slices the Jacobian too.")
    p.add_argument("--fit-upsample-z", type=int, default=4)
    p.add_argument("--fit-upsample-jpsi", type=int, default=1,
                   help="the J/psi window is +-0.35 GeV, i.e. a handful of "
                        "oscillation periods across the tau grid, so it does "
                        "not need the Z's 4x")
    p.add_argument("--seed", type=int, default=1234)
    # ---- the sparse D -----------------------------------------------------
    p.add_argument("--jac-prune", type=float, default=0.0,
                   help="drop |D_ik| below this fraction of the row's own "
                        "max |D_i.|. 0 keeps every non-zero entry. The block "
                        "is (n x 92) dense in the cache and sparse in the "
                        "card, so this is what sets the card size.")
    # ---- declaration / POIs ----------------------------------------------
    p.add_argument("--declare", choices=["bundle", "unbinned"], default="bundle",
                   help="see the module docstring: who puts the 92 into the "
                        "fit parameter vector")
    p.add_argument("--poi", default="bfield",
                   help="which CALIBRATION parameters are reported as POIs: "
                        "'bfield' (all parmtype-14 modes, the default), 'all', "
                        "'none', or a comma separated name list. The mass "
                        "terms' own POIs (m_Z, Gamma_Z) are unaffected.")
    # ---- injection tests ---------------------------------------------------
    p.add_argument("--inject", action="append", default=[], metavar="NAME:VALUE",
                   help="inject a shift, in CARD units: every mass term's "
                        "m_i^0 moves by (D_card dtheta)_i and the quadratic "
                        "term's g by -H dtheta, so the joint minimum moves by "
                        "exactly +dtheta. Repeatable.")
    p.add_argument("--inject-mass-only", action="store_true",
                   help="shift only the mass terms; the joint minimum then "
                        "moves by (A+B)^-1 A dtheta with A the mass-term and B "
                        "the quadratic information -- useful to measure the "
                        "information split, NOT a closure test")
    p.add_argument("--inject-quad-only", action="store_true")
    p.add_argument("--verify", action="store_true", default=True,
                   help="re-read the written card with "
                        "rabbit.unbinned.read_unbinned_terms_from_h5, which "
                        "asserts that each term's stored parameter list is the "
                        "one its configuration implies")
    p.add_argument("--no-verify", dest="verify", action="store_false")
    return p.parse_args(argv)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _quiet(*a, **k):
    pass


def sum_quadratic(paths, log=print):
    """G, K and the sandwich meat, summed over productions."""
    G = K = J = None
    cat = None
    tot = 0
    for f in paths:
        d = np.load(f)
        if cat is None:
            cat = {k: d[k] for k in ("fitidx", "parmtype", "subidx", "g2f",
                                     "nglobal")}
        else:
            for k in ("fitidx", "parmtype", "subidx"):
                if not np.array_equal(cat[k], d[k]):
                    raise SystemExit(
                        f"{f}: the parameter map differs from the first "
                        f"extraction on `{k}`; the two cannot be summed")
        G = d["grad"] if G is None else G + d["grad"]
        K = d["hess"] if K is None else K + d["hess"]
        if "jsand" in d.files:
            J = d["jsand"] if J is None else J + d["jsand"]
        tot += int(d["ncand_quadratic"])
        log(f"  {os.path.basename(f)}: {int(d['ncand_quadratic'])} candidates, "
            f"{int(d['ncand_chi2cut'])} cut")
    log(f"  quadratic term: {tot} candidates, {len(G)} parameters")
    return G, K, J, cat, tot


# ---------------------------------------------------------------------------
# the sparse D
# ---------------------------------------------------------------------------
def check_jac_map(d, cat, tag):
    """The pairs cache's own parameter map must equal the extraction's.

    `cf_inmaker.py --jac-parmtypes` stores `jac_globalidx` / `jac_parmtype` /
    `jac_subidx` (for parmtype 14/15 the runtree's `rawdetid` IS the mode /
    group index, which is exactly what `globalfit/extract.py` calls `subidx`).
    If the two disagree the columns of `D` mean something different from the
    columns of `K`, and every number downstream is silently wrong -- so this
    is an ABORT, not a warning.
    """
    have = set(d.files)
    need = ("jac_globalidx", "jac_parmtype", "jac_subidx")
    missing = [k for k in need if k not in have]
    if missing:
        raise SystemExit(
            f"{tag}: the pairs cache has no {missing}; it was not built with "
            "`cf_inmaker.py pairs --jac-parmtypes 14 15`")
    for ckey, jkey in (("fitidx", "jac_globalidx"),
                       ("parmtype", "jac_parmtype"),
                       ("subidx", "jac_subidx")):
        a = np.asarray(cat[ckey], dtype=np.int64)
        b = np.asarray(d[jkey], dtype=np.int64)
        if a.shape != b.shape:
            raise SystemExit(
                f"{tag}: the D block has {b.shape[0]} columns but the "
                f"quadratic extraction has {a.shape[0]} parameters "
                f"(disagreement on `{ckey}` / `{jkey}`)")
        if not np.array_equal(a, b):
            bad = int(np.argmax(a != b))
            raise SystemExit(
                f"{tag}: the pairs cache's parameter map disagrees with the "
                f"quadratic extraction on `{ckey}` vs `{jkey}`: first "
                f"difference at column {bad}, extraction {a[bad]} vs cache "
                f"{b[bad]}. The D columns and the K columns are not the same "
                "parameters; rebuild one of the two.")


def sparse_jac(D, prune, tag, log=print):
    """(indices, values, shape) of D_card, pruned, plus the density report.

    ``D`` is already ``D_card = -dm/dtheta`` and already whitened.
    """
    n, nfit = D.shape
    nz0 = int(np.count_nonzero(D))
    if prune > 0.0:
        rowmax = np.max(np.abs(D), axis=1)
        D = np.where(np.abs(D) >= prune * rowmax[:, None], D, 0.0)
    rows, cols = np.nonzero(D)
    vals = D[rows, cols]
    idx = np.stack([rows, cols], axis=1).astype(np.int64)
    dens0 = 100.0 * nz0 / max(n * nfit, 1)
    dens1 = 100.0 * len(vals) / max(n * nfit, 1)
    log(f"  {tag} sparse D: {nz0} non-zero of {n*nfit} ({dens0:.2f} % dense) "
        f"-> {len(vals)} after --jac-prune {prune:g} ({dens1:.2f} %), "
        f"{len(vals)*12/1e6:.1f} MB in the card")
    return (idx, vals, (n, nfit)), D, dict(nnz_raw=nz0, nnz=int(len(vals)),
                                           density_raw=dens0, density=dens1)


# ---------------------------------------------------------------------------
# the J/psi term (make_card.build's shape, with a delta kernel)
# ---------------------------------------------------------------------------
def norm_classes(sigma, vgf, arrays, nt, nclasses, log=print):
    """Resolution classes for the truncation normalisation.

    Transcribed from `make_card.build` (which cannot be imported piecewise:
    the block is inline there). Kept identical on purpose -- if the two ever
    disagree the Z and J/psi legs would be normalised differently.
    """
    n = len(sigma)
    K = n if nclasses <= 0 else min(nclasses, n)
    if K == n:
        cls = np.arange(n)
        sig_c, vgf_c = sigma.copy(), vgf.copy()
        fam_c = {nm: {c: a.astype(np.float64) for c, a in arr.items()}
                 for nm, arr in arrays.items()}
    else:
        edges = np.quantile(sigma, np.linspace(0, 1, K + 1))
        cls = np.clip(np.searchsorted(edges[1:-1], sigma, "right"), 0, K - 1)
        sig_c = np.empty(K)
        vgf_c = np.empty(K)
        fam_c = {nm: {c: np.empty((K, nt)) for c in arr}
                 for nm, arr in arrays.items()}
        for c in range(K):
            sel = cls == c
            if not sel.any():
                sel = np.ones(n, bool)
            sig_c[c] = np.median(sigma[sel])
            vgf_c[c] = np.mean(vgf[sel])
            for nm, arr in arrays.items():
                for comp, a in arr.items():
                    fam_c[nm][comp][c] = a[sel].mean(axis=0)
    norm = {"sigma": sig_c, "vgf": vgf_c, "class": cls,
            "families": [dict(name=nm, **fam_c[nm]) for nm in fam_c]}
    extra = {"norm_sigma": sig_c, "norm_vgf": vgf_c,
             "norm_class": cls.astype(np.int64)}
    for nm, arr in fam_c.items():
        for comp, a in arr.items():
            extra[f"S_{comp}_{nm}_norm"] = a.astype(np.float32)
    log(f"  truncation normalisation, {K} class(es)")
    return norm, extra


def build_jpsi(args, log=print):
    """The J/psi mass term: delta kernel at MJPSI, NO scale parameter.

    Returns (term, datasets, decl, idx, d) -- `idx` and the open cache so the
    caller can slice the D block on exactly the same candidates.
    """
    import make_card
    from rabbit import unbinned

    lo = MJPSI - args.jpsi_window
    hi = MJPSI + args.jpsi_window
    jargs = make_card.parse_args([
        "--pairs", args.jpsi_pairs,
        "--name", "jpsi", "--channel", "jpsi",
        "--mref", repr(MJPSI),
        "--window", repr(lo), repr(hi),
        "--maxn", str(args.jpsi_maxn),
        "--seed", str(args.seed),
        "--max-chi2-ndof", repr(args.max_chi2_ndof),
        "--max-sigma-rel", repr(args.max_sigma_rel),
        "--chunk", str(args.chunk),
        "--fit-upsample", str(args.fit_upsample_jpsi),
    ])

    d = np.load(args.jpsi_pairs, allow_pickle=True)
    tgrid = np.asarray(d["tgrid"], dtype=np.float64)
    nt = len(tgrid)
    idx, m_all, sig_all, w_all, _ = make_card.select(d, jargs, log)
    n = len(idx)
    if not n:
        raise SystemExit("the J/psi selection kept nothing")
    sigma = sig_all[idx]
    mreco = m_all[idx]
    vgf = d["vgf"].astype(np.float64)[idx]
    mobs = mreco - MJPSI
    weights, winfo = make_card.build_weights(w_all, idx, jargs, log)

    # the two corrections, exactly as on the Z leg
    a_res = (1.0 + vgf) * sigma / np.maximum(np.abs(mreco), 1e-9)
    nclip = int(np.sum(np.abs(a_res) > jargs.max_ares))
    a_res = np.clip(a_res, -jargs.max_ares, jargs.max_ares)
    log(f"  a_res: median {np.median(a_res):.5f}, "
        f"q99 {np.quantile(np.abs(a_res), 0.99):.5f}, {nclip} clipped")
    s2 = (sigma / np.maximum(np.abs(mreco), 1e-9)) ** 2
    if "fang" in d.files:
        fang = np.clip(np.asarray(d["fang"], dtype=np.float64)[idx], -0.5, 1.0)
        log(f"  jensen: PER-CANDIDATE f_ang median {np.median(fang):.3e}")
    else:
        fang = float(args.jpsi_fang)
        log(f"  jensen: the cache has no `fang`; using the scalar "
            f"--jpsi-fang {fang:g} (MC gun 0.086 / data 0.106) -- this is a "
            "systematic to scan, not a measurement")
    jensen_s2 = s2 * (1.5 - fang) / 1.5
    log(f"  jensen (exact): median 1.5 s^2 = {1.5*np.median(jensen_s2):.4e} "
        f"relative -> {1.5*np.median(jensen_s2)*MJPSI*1e3:.3f} MeV")

    fams = make_card.discover_families(set(d.files), False)
    families = [{"name": "hit", "param": "k_hit", "kind": "gauss"}]
    datasets = {"sigma": sigma, "mobs": mobs, "vgf": vgf, "tgrid": tgrid,
                "a_res": a_res, "jensen_s2": jensen_s2}
    if weights is not None:
        datasets["weights"] = weights
    arrays = {}
    for name, re_k, im_k in fams:
        families.append({"name": name, "param": f"k_{name}", "kind": "tab"})
        arrays[name] = {"re": np.asarray(d[re_k])[idx]}
        datasets[f"S_re_{name}"] = arrays[name]["re"]
        if im_k:
            arrays[name]["im"] = np.asarray(d[im_k])[idx]
            datasets[f"S_im_{name}"] = arrays[name]["im"]
    log(f"  {n} candidates, nt = {nt}, families "
        f"{[f['name'] for f in families]}, upsample {args.fit_upsample_jpsi}")

    norm, extra = norm_classes(sigma, vgf, arrays, nt, jargs.norm_classes, log)
    datasets.update(extra)

    term = unbinned.MassCFTerm(
        "jpsi", sigma=sigma, mobs=mobs, tgrid=tgrid,
        families=[dict(f, **arrays.get(f["name"], {})) for f in families],
        vgf=vgf, phik=None,
        kernel=unbinned.DeltaKernel(),
        background=None, m_ref=MJPSI,
        # NO alpha: the scale is carried by the field modes and the material
        # groups through D, which is the whole point of the joint card.
        scale_param=None,
        bkg_frac=0.0,
        weights=weights,
        a_res=a_res, self_consistent_sigma=True,
        jensen_s2=jensen_s2, jensen_mode="exact",
        norm_window=(lo, hi), norm_tpoints=jargs.norm_tpoints, norm=norm,
        upsample=args.fit_upsample_jpsi,
        chunk=args.chunk, channel="jpsi")
    decl = unbinned.declare_params(
        term, {f["param"]: (1.0, np.nan, 1.0, 0) for f in families})
    log(f"  parameters {list(term.param_names)}")
    info = {"n": n, "n_cache": int(len(d["z"])), "window": [lo, hi],
            "weights_info": winfo}
    return term, datasets, decl, idx, d, info


# ---------------------------------------------------------------------------
# card assembly
# ---------------------------------------------------------------------------
def attach_jac(cfg, datasets, decl, params, jac, names, prior_sigmas, poi_set):
    """Give an already-built MassCFTerm a sparse D, through its card image.

    `MassCFTerm.__init__` appends `jac_params` to `param_names` LAST (after the
    scale, the families, the kernel and the background), and
    `read_unbinned_terms_from_h5` rebuilds the term from `config` + `datasets`
    and then ASSERTS that the stored parameter list is the one the rebuilt term
    implies. So extending the card image the same way the constructor would is
    equivalent to having passed `jac=` in the first place -- and is checked on
    read-back rather than assumed. Doing it this way is what lets the Z term
    come straight out of `make_card.build` instead of being re-derived here.
    """
    cfg = dict(cfg)
    cfg["jac_params"] = list(names)
    idx, vals, shape = jac
    datasets = dict(datasets)
    datasets["jac_indices"] = idx
    datasets["jac_values"] = vals
    datasets["jac_shape"] = np.asarray(shape, dtype=np.int64)
    new = [nm for nm in names if nm not in params]
    if len(new) != len(names):
        raise SystemExit(
            "a calibration parameter name collides with one of the mass "
            f"term's own parameters: {sorted(set(names) - set(new))}")
    params = list(params) + new
    decl = {
        "param_defaults": np.concatenate(
            [decl["param_defaults"], np.zeros(len(new))]),
        "param_prior_sigmas": np.concatenate(
            [decl["param_prior_sigmas"], np.asarray(prior_sigmas, np.float64)]),
        "param_prior_means": np.concatenate(
            [decl["param_prior_means"], np.zeros(len(new))]),
        "param_is_poi": np.concatenate(
            [np.asarray(decl["param_is_poi"], np.int8),
             np.array([1 if nm in poi_set else 0 for nm in new], np.int8)]),
    }
    return cfg, datasets, decl, params


def merge_declarations(entries):
    """One (default, sigma, mean, is_poi) per name over all terms; disagreement
    is an error, not a silent first-wins."""
    out, order = {}, []
    for name, params, decl in entries:
        for i, nm in enumerate(params):
            row = (float(decl["param_defaults"][i]),
                   float(decl["param_prior_sigmas"][i]),
                   float(decl["param_prior_means"][i]),
                   int(decl["param_is_poi"][i]))
            if nm in out:
                a, b = out[nm][1], row
                same = all(
                    (np.isnan(x) and np.isnan(y)) or x == y
                    for x, y in zip(a, b))
                if not same:
                    raise SystemExit(
                        f"parameter '{nm}' is declared differently by term "
                        f"'{out[nm][0]}' {a} and term '{name}' {b}; rabbit "
                        "would refuse the card")
            else:
                out[nm] = (name, row)
                order.append(nm)
    return order, {nm: out[nm][1] for nm in order}


def main():
    args = parse_args()
    import make_card
    import make_global_term as mgt
    from rabbit import tensorwriter

    t_start = time.time()
    log(f"[make_joint_card] -> {args.output}")

    # -- 1. the quadratic term ---------------------------------------------
    G, K, J, cat, nquad = sum_quadratic(args.quad, log)
    parmtype, subidx = cat["parmtype"], cat["subidx"]
    nfit = len(G)
    names, prior_sigmas = mgt.name_params(
        parmtype, subidx, args.groups, args.field_prior,
        args.material_prior_scale)
    counts = {int(pt): int((parmtype == pt).sum())
              for pt in sorted(set(np.asarray(parmtype).tolist()))}
    log("  parameter types: " + ", ".join(
        f"{mgt.TYPENAMES.get(k, f'type{k}')}({k}) x{v}" for k, v in counts.items()))

    pscale = np.ones(nfit)
    if args.whiten:
        pscale, _ = mgt.param_scales(parmtype, subidx, args.coeffs, args.groups)
        inv = 1.0 / np.maximum(pscale, 1e-300)
        # theta_card = theta_raw * s  =>  G -> G/s, K -> K/(s s^T), D -> D/s
        G = G * inv
        K = K * np.outer(inv, inv)
        if J is not None:
            J = J * np.outer(inv, inv)
        prior_sigmas = prior_sigmas * pscale
        log("  whitened: 1 card unit = 1 T of RMS |dB| (parmtype 14) / 1 prior "
            f"sigma (parmtype 15); scale range {pscale.min():.3e} .. "
            f"{pscale.max():.3e}")
    else:
        inv = np.ones(nfit)
    Ksym = 0.5 * (K + K.T)
    ev = np.linalg.eigvalsh(Ksym)
    cond = float(ev[-1] / ev[0]) if ev[0] > 0 else float("inf")
    nnull = int((ev < 1e-12 * ev[-1]).sum())
    dead = [names[i] for i in range(nfit)
            if Ksym[i, i] < 1e-12 * np.max(np.diag(Ksym))]
    cond_eff = float(ev[-1] / ev[nnull]) if nnull < nfit else float("inf")
    log(f"  quadratic Hessian ({'whitened' if args.whiten else 'raw'}): "
        f"cond {cond:.4g}, eigenvalues [{ev[0]:.4g}, {ev[-1]:.4g}]; "
        f"{nnull} direction(s) below 1e-12 x max, cond over the rest "
        f"{cond_eff:.4g}")
    if dead:
        log(f"  the hit-chi2 term carries NO information on {len(dead)} "
            f"parameter(s): {dead}. They are constrained only by the mass "
            "term(s) (or not at all -- freeze them or give them a prior).")

    if args.poi == "bfield":
        poi_set = {nm for nm in names if nm.startswith("bfield_mode")}
    elif args.poi == "all":
        poi_set = set(names)
    elif args.poi == "none":
        poi_set = set()
    else:
        poi_set = {s for s in args.poi.split(",") if s}
        unknown = poi_set - set(names)
        if unknown:
            raise SystemExit(f"--poi names not among the 92: {sorted(unknown)}")

    # -- 2. the injection vector -------------------------------------------
    inject = np.zeros(nfit)
    if args.inject:
        byname = {nm: i for i, nm in enumerate(names)}
        for spec in args.inject:
            nm, val = spec.rsplit(":", 1)
            if nm not in byname:
                raise SystemExit(f"--inject for unknown parameter '{nm}'")
            inject[byname[nm]] = float(val)
        log("  injecting " + ", ".join(
            f"{nm}={inject[byname[nm]]:+.6g}" for nm in names if inject[byname[nm]]))

    # -- 3. the mass terms --------------------------------------------------
    # the parameter maps FIRST: an hour of term building is wasted if the D
    # columns turn out to mean something other than the K columns
    for tag, path in (("jpsi", args.jpsi_pairs), ("z", args.z_pairs)):
        if path:
            check_jac_map(np.load(path, allow_pickle=True), cat, tag)
    log(f"  parameter map: the D block of every pairs cache matches the "
        f"quadratic extraction on all {nfit} (fitidx, parmtype, subidx)")

    entries = []          # (name, cfg, params, datasets, decl)
    jacinfo = {}

    if args.jpsi_pairs:
        log("--- J/psi term ---")
        jterm, jdata, jdecl, jidx, jd, jinfo = build_jpsi(args, log)
        Dj = jd["D"][jidx].astype(np.float64)
        Dj *= -inv[None, :]
        del jd
        jac, Dj, jstat = sparse_jac(Dj, args.jac_prune, "jpsi", log)
        jacinfo["jpsi"] = jstat
        if args.inject and not args.inject_quad_only:
            shift = Dj @ inject
            jdata["mobs"] = jdata["mobs"] + shift
            log(f"  injected into the J/psi means: mean {shift.mean()*1e3:+.5f} "
                f"MeV, rms {shift.std()*1e3:.5f} MeV")
        cfg, jdata, jdecl, jparams = attach_jac(
            jterm.config(), jdata, jdecl, jterm.param_names, jac, names,
            prior_sigmas, poi_set)
        entries.append(("jpsi", cfg, jparams, jdata, jdecl))
        log(f"  J/psi term: {jinfo['n']} candidates, {len(jparams)} parameters")
    else:
        log("NO --jpsi-pairs: building the QUADRATIC + Z card only. Without "
            "the J/psi leg nothing pins the momentum scale independently of "
            "m_Z except the hit-chi2 term, so `m_Z` and the field modes are "
            "constrained only by K -- this card is the plumbing test, not the "
            "physics measurement.")

    log("--- Z term ---")
    zargv = ["--pairs", args.z_pairs,
             "--name", "zmass", "--channel", "z",
             "--mref", repr(MZ_REF),
             "--shape", str(args.shape),
             "--maxn", str(args.z_maxn),
             "--seed", str(args.seed),
             "--max-chi2-ndof", repr(args.max_chi2_ndof),
             "--max-sigma-rel", repr(args.max_sigma_rel),
             "--chunk", str(args.chunk),
             "--fit-upsample", str(args.fit_upsample_z)]
    if args.fsr:
        zargv += ["--fsr", args.fsr]
    if args.acc:
        zargv += ["--acc", args.acc]
    zargs = make_card.parse_args(zargv)
    zterm, zdata, zdecl, zinfo = make_card.build(zargs, log)

    zd = np.load(args.z_pairs, allow_pickle=True)
    zidx, _, _, _, _ = make_card.select(zd, zargs, _quiet)
    if len(zidx) != zinfo["n"]:
        raise SystemExit(
            f"the Z selection is not reproducible: make_card.build kept "
            f"{zinfo['n']} candidates, re-running make_card.select kept "
            f"{len(zidx)}")
    Dz = zd["D"][zidx].astype(np.float64)
    Dz *= -inv[None, :]
    del zd
    jac, Dz, zstat = sparse_jac(Dz, args.jac_prune, "z", log)
    jacinfo["z"] = zstat
    if args.inject and not args.inject_quad_only:
        shift = Dz @ inject
        zdata["mobs"] = zdata["mobs"] + shift
        log(f"  injected into the Z means: mean {shift.mean()*1e3:+.5f} MeV, "
            f"rms {shift.std()*1e3:.5f} MeV")
    cfg, zdata, zdecl, zparams = attach_jac(
        zterm.config(), zdata, zdecl, zterm.param_names, jac, names,
        prior_sigmas, poi_set)
    entries.append(("zmass", cfg, zparams, zdata, zdecl))
    log(f"  Z term: {zinfo['n']} candidates, {len(zparams)} parameters")

    # -- 4. the injection into the quadratic term ---------------------------
    if args.inject and not args.inject_mass_only:
        G = G - K @ inject
        log("  quadratic term: G -> G - K dtheta")

    # -- 5. write ------------------------------------------------------------
    writer = tensorwriter.TensorWriter()
    writer.add_dummy_channel(name="calib_dummy")
    for name, cfg, params, datasets, decl in entries:
        writer.add_unbinned_term(name, cfg, params, datasets, **decl)

    # chi2 -> NLL: the stored gradv/hesspackedv are in chi2 units (the factor 2
    # is inside the C++), rabbit's external term is L = g.theta + 0.5 theta H
    # theta in NLL units.  g = G/2, H = K/2.  Same minimum, same covariance.
    hg, hh = mgt.to_hists(names, 0.5 * G, 0.5 * np.asarray(K), False)
    writer.add_external_likelihood_term(grad=hg, hess=hh, name="hitchi2")
    log(f"  external term 'hitchi2': {nfit} parameters (dense Hessian), from "
        f"{nquad} candidates")

    # -- 6. who declares what -----------------------------------------------
    order, decl_all = merge_declarations(
        [(nm, params, decl) for nm, cfg, params, datasets, decl in entries])
    if args.declare == "bundle":
        bundle_names = order
        models = ["ExternalParams bundle:global_params"]
    else:
        bundle_names = [nm for nm in names if nm not in set(order)]
        models = ["UnbinnedParams"]
        if bundle_names:
            models.append("ExternalParams bundle:global_params")
    missing = [nm for nm in names if nm not in set(order) | set(bundle_names)]
    if missing:
        raise SystemExit(
            f"{len(missing)} calibration parameters would be used by the "
            f"quadratic term but declared by nobody: {missing[:5]}...")
    if bundle_names:
        writer.add_auxiliary("global_params", {
            "params": list(bundle_names),
            "defaults": np.array([decl_all[nm][0] if nm in decl_all else 0.0
                                  for nm in bundle_names]),
            "prior_sigmas": np.array(
                [decl_all[nm][1] if nm in decl_all
                 else prior_sigmas[names.index(nm)] for nm in bundle_names]),
            "prior_means": np.array([decl_all[nm][2] if nm in decl_all else 0.0
                                     for nm in bundle_names]),
            "is_poi": np.array(
                [decl_all[nm][3] if nm in decl_all else int(nm in poi_set)
                 for nm in bundle_names], dtype=np.int64),
        })
        log(f"  auxiliary bundle 'global_params': {len(bundle_names)} "
            f"parameters ({sum(1 for nm in bundle_names if nm in set(names))} "
            "of them the calibration parameters)")
    else:
        log("  auxiliary bundle 'global_params': NOT written (every parameter "
            "is declared by an unbinned term)")

    writer.add_auxiliary("global_index_map", {
        "params": names,
        "global_index": np.asarray(cat["fitidx"], dtype=np.int64),
        "parmtype": np.asarray(parmtype, dtype=np.int64),
        "subindex": np.asarray(subidx, dtype=np.int64),
        "prior_sigmas": np.asarray(prior_sigmas, dtype=np.float64),
        "scale": np.asarray(pscale, dtype=np.float64),
        "injected": inject,
        # the quadratic term's sandwich meat, in the card's own (whitened)
        # units: rabbit's external term cannot use it, but a robust covariance
        # downstream needs it and it would otherwise be lost here.
        "jsand": (np.zeros((nfit, nfit)) if J is None
                  else np.asarray(J, dtype=np.float64)),
        "provenance": [json.dumps({
            "quad": [os.path.abspath(f) for f in args.quad],
            "z_pairs": os.path.abspath(args.z_pairs),
            "jpsi_pairs": (os.path.abspath(args.jpsi_pairs)
                           if args.jpsi_pairs else None),
            "nglobal": int(cat["nglobal"]),
            "ncand_quadratic": int(nquad),
            "terms": [nm for nm, _, _, _, _ in entries],
            "jac_sign": "D_card = -dm/dtheta",
            "jac_prune": args.jac_prune,
            "jac_density": jacinfo,
            "chi2_to_nll": 0.5,
            "whiten": bool(args.whiten),
            "quad_cond": cond,
            "quad_cond_eff": cond_eff,
            "quad_null": nnull,
            "quad_dead_params": dead,
            "declare": args.declare,
            "field_prior": args.field_prior,
            "material_prior_scale": args.material_prior_scale,
            "jpsi_fang": args.jpsi_fang,
            "paramModels": models,
        })],
    })

    folder = os.path.dirname(os.path.abspath(args.output)) or "."
    base = os.path.basename(args.output)
    if base.endswith(".hdf5"):
        base = base[: -len(".hdf5")]
    os.makedirs(folder, exist_ok=True)
    t0 = time.time()
    writer.write(outfolder=folder, outfilename=base)
    path = os.path.join(folder, base) + ".hdf5"
    log(f"  wrote {path} ({os.path.getsize(path)/1e9:.3f} GB) in "
        f"{time.time()-t0:.1f} s")

    if args.verify:
        import h5py
        from rabbit import unbinned
        t0 = time.time()
        with h5py.File(path, "r") as f:
            terms = unbinned.read_unbinned_terms_from_h5(f["unbinned_terms"])
        for t in terms:
            miss = [nm for nm in names if nm not in set(t.param_names)]
            if miss:
                raise SystemExit(
                    f"term '{t.name}' does not carry {len(miss)} of the "
                    f"calibration parameters, e.g. {miss[:3]}")
        log(f"  verified: {len(terms)} term(s) re-read in {time.time()-t0:.1f} "
            "s, each parameter list is the one its configuration implies")

    log("  parameters in the fit vector:")
    for nm, cfg, params, datasets, decl in entries:
        own = [p for p in params if p not in set(names)]
        log(f"    {nm}: {own} + the {len(names)} calibration parameters")
    log(f"    calibration: {names[0]} .. {names[nfit-1]} "
        f"({len(poi_set)} flagged POI)")
    log("  run it with:  rabbit_fit.py %s -o out/ -t 0 --unblind %s"
        % (path, " ".join(f"--paramModel {m}" for m in models)))
    log(f"  total {time.time()-t_start:.1f} s")


if __name__ == "__main__":
    main()
