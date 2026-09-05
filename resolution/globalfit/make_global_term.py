#!/usr/bin/env python3
"""Turn an ``extract.py`` npz into a rabbit datacard for the unified CVH
calibration objective.

The objective has two pieces sharing the *same* global calibration parameters
``theta`` (scalar-potential B-field modes = parmtype 14, global material groups
= parmtype 15, optionally alignment = 0-5):

1. the **quadratic hit-chi2 term**, the per-candidate gradients and Hessians
   the residual makers store with ``fillGrads`` / ``fillGradsFactored``,
   already marginalized over the per-track parameters, summed over the
   production::

       chi2(theta) = chi2_0 + G^T theta + 0.5 theta^T K theta

   which ``global_corrections/fit_global_grads.py`` solves offline as
   ``theta = -K^-1 G``, ``cov = 2 K^-1``. **The stored ``gradv`` /
   ``hesspackedv`` / ``B^T B`` are in chi2 units** (the factor 2 is inside the
   C++, ``grad = 2 J^T R r``, ``hess = 2 J^T R J``). rabbit's external term is
   in **NLL** units, ``L_ext = g^T theta + 0.5 theta^T H theta``, so the card
   stores

       g = G / 2        H = K / 2

   giving the same minimum (``-H^-1 g = -K^-1 G``) and the same covariance
   (``H^-1 = 2 K^-1``). *This factor of two is the single most important
   convention in this file.*

2. the **unbinned mass term** (:class:`rabbit.unbinned.MassCFTerm`) over the
   same candidates, whose mass depends on ``theta`` through the per-candidate
   Jacobian rows ``D_i = dm_i/dtheta`` (the two-track maker's
   ``Jpsi_jacMass``).

   **Sign convention.** ``MassCFTerm`` evaluates the resolution density at
   ``delta_i = mobs_i - shift - (D_card theta)_i``, i.e. it treats ``D_card``
   as ``d(predicted mass)/dtheta``. Here it is the *reconstructed* mass that
   moves with theta, ``m_i(theta) = m_i^0 + D_i theta``, so the rows written
   into the card are ``D_card = -D``. Get this backwards and the two terms
   pull against each other instead of agreeing; ``--check-sign`` prints the
   diagnostic that catches it.

The two terms are joined by *name*: field modes ``bfield_mode<k>`` (``k`` =
the mode index, the same naming ``fit_global_grads.py`` prints), material
groups ``material_<group>``, anything else ``glob_t<parmtype>_<k>``.

Units of the shared parameters
------------------------------
* parmtype 14: a coefficient of the magnetic **scalar potential** in the
  ``ScalarPot3DEval`` basis, units **T cm**. Mode 0 is (l=1, m=0), a uniform
  ``Bz``: one unit = ``dBz = 1/r_scale = 1/319.9556 = 3.125e-3 T``, i.e.
  ``dB/B = 8.2e-4`` at 3.81 T. So ``dB/B = 1e-4`` is ``bfield_mode0 = 0.122``.
* parmtype 15: a dimensionless log energy-loss scale, ``dE = exp(k_g) dE_0``.

**There is no ``alpha``.** In the standalone mass fit the momentum scale is a
free parameter ``alpha`` multiplying ``m_ref``; here that degree of freedom is
carried by ``bfield_mode0`` (whose ``D`` column is nearly proportional to
``m_i``) together with the material groups. ``--with-alpha`` puts it back for
the standalone comparison against the step-1/step-2 mass fits; with the
Jacobian rows present it is nearly degenerate with ``bfield_mode0`` in the
mass term and is resolved only by the hit-chi2 term.

Priors follow ``fit_global_grads.py``: none on the field modes unless
``--field-prior`` is given, and the per-group sigmas from column 10 of the
materialGroups tier file (scaled by ``--material-prior-scale``) for the
material groups. They are declared through rabbit's ParamModel prior mechanism
(``0.5 ((p - mu)/sigma)^2``), identical to adding ``2/sigma^2`` to the chi2
Hessian as the offline fit does. The resolution family scales ``k_*`` stay
free nuisances.

Examples
--------
::

    # quadratic term only -- must reproduce fit_global_grads.py
    python make_global_term.py -i runs/globalfit_btojpsix.npz \\
        --parmtypes 14 --no-mass -o cards/quad.hdf5
    rabbit_fit.py cards/quad.hdf5 -o out/ -t 0 --unblind \\
        --paramModel ExternalParams bundle:global_params

    # joint card
    python make_global_term.py -i runs/globalfit_btojpsix.npz \\
        --parmtypes 14 15 --groups .../materialGroups50.txt -o cards/joint.hdf5
    rabbit_fit.py cards/joint.hdf5 -o out/ -t 0 --unblind \\
        --paramModel UnbinnedParams
"""

import argparse
import json
import os
import sys
import time

import numpy as np

MJPSI = 3.0969
MWIN = 0.7
FBKG = 0.005
GAUSS_FAMILY = "hit"
FAMILY_ORDER = ["ms", "ioni", "rad"]
FAMILY_ALIAS = {"io": "ioni"}
# 1 unit of a parmtype-14 coefficient = 1/r_scale Tesla for the (l=1, m=0)
# mode; r_scale from the coefficient dump used by the production.
R_SCALE_CM = 319.9555929341
B_NOMINAL = 3.8114
TYPENAMES = {
    0: "align_u",
    1: "align_v",
    2: "align_w",
    3: "align_a",
    4: "align_b",
    5: "align_g",
    7: "material_module",
    8: "res_hit_x",
    9: "res_hit_y",
    10: "res_ms",
    11: "res_ioni",
    14: "bfield",
    15: "material",
}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-i", "--input", required=True, help="extract.py output npz")
    p.add_argument("-o", "--output", required=True, help="output .hdf5 datacard")
    p.add_argument(
        "--parmtypes",
        type=int,
        nargs="+",
        default=None,
        help="restrict to these global parameter types (default: all in the "
        "extraction)",
    )
    p.add_argument(
        "--groups",
        default=None,
        help="materialGroups tier file (group names + prior sigmas, type 15)",
    )
    p.add_argument("--no-mass", action="store_true", help="quadratic term only")
    p.add_argument("--no-quadratic", action="store_true", help="mass term only")
    p.add_argument("--no-jac", action="store_true", help="drop the D rows")
    p.add_argument(
        "--dense-max",
        type=int,
        default=4000,
        help="store the external Hessian densely up to this many parameters; "
        "above it, store it as a sparse SparseHist",
    )
    p.add_argument(
        "--hess-drop",
        type=float,
        default=0.0,
        help="when storing sparsely, drop |H_ij| below this fraction of "
        "sqrt(H_ii H_jj)",
    )

    m = p.add_argument_group("mass term")
    m.add_argument("--kernel-cache", default=None, help="FSR kernel npz with 'dm'")
    m.add_argument(
        "--maxn", type=int, default=0, help="use only the first N candidates"
    )
    m.add_argument(
        "--max-chi2-ndof",
        type=float,
        default=0.0,
        help="drop mass-term candidates with chisqval/ndof above this (needs "
        "the 'chi2ndof' array in the extraction). The median is ~0.95 but the "
        "tail reaches 5e8 and a handful of runaway fits otherwise dominate "
        "everything they enter; see STATE.md.",
    )
    m.add_argument("--maxk", type=int, default=0, help="use only N kernel samples")
    m.add_argument(
        "--model",
        choices=["r", "families"],
        default="families",
        help="'r': one resolution scale for all families; 'families': one each",
    )
    m.add_argument("--with-alpha", action="store_true")
    m.add_argument("--alpha0", type=float, default=0.0)
    m.add_argument("--k0", type=float, default=1.0)
    m.add_argument("--fbkg", type=float, default=FBKG)
    m.add_argument("--float-bkg", action="store_true")
    m.add_argument("--mref", type=float, default=MJPSI)
    m.add_argument("--window", type=float, default=MWIN)
    m.add_argument("--chunk", type=int, default=32768)
    m.add_argument("--floor", choices=["softplus", "clip", "none"], default="softplus")
    m.add_argument("--phik-points", type=int, default=8192)
    m.add_argument("--phik-cache", default=None)

    pr = p.add_argument_group("priors")
    pr.add_argument("--field-prior", type=float, default=0.0)
    pr.add_argument("--material-prior-scale", type=float, default=1.0)
    pr.add_argument("--res-prior", type=float, default=0.0)
    pr.add_argument(
        "--poi",
        default="bfield",
        help="which parameters are reported as POIs: 'bfield' (all parmtype-14 "
        "modes, the default), 'all', 'none', or a comma separated name list",
    )

    inj = p.add_argument_group("injection tests")
    inj.add_argument(
        "--inject",
        action="append",
        default=[],
        metavar="NAME:VALUE",
        help="inject a shift: the mass term's m_i^0 moves by D_i.dtheta and "
        "the quadratic term's g by -K dtheta, so the joint minimum moves by "
        "exactly +dtheta. Repeatable.",
    )
    inj.add_argument("--inject-mass-only", action="store_true")
    inj.add_argument("--inject-quad-only", action="store_true")

    p.add_argument(
        "--whiten",
        action="store_true",
        help="write the card in the PHYSICAL basis: each parameter is scaled "
        "so its fitted value is Tesla of RMS |dB| in the tracker (parmtype 14) "
        "or its own prior sigma (parmtype 15). Strongly recommended -- the raw "
        "coefficient basis is ~350x worse conditioned and the raw values are "
        "not comparable between modes. The scale vector is stored in the "
        "global_index_map bundle so results convert back (theta_raw = "
        "theta_card / scale).",
    )
    p.add_argument("--coeffs", default=None, help="mode dump for --whiten")
    p.add_argument("--check-sign", action="store_true")
    return p.parse_args()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
def read_groups(fname):
    names, priors = {0: "other"}, {0: 0.2}
    if not fname:
        return names, priors
    for line in open(fname):
        if line.startswith("RULE"):
            f = line.split("\t")
            names[int(f[1])] = f[2]
            priors[int(f[1])] = float(f[10])
    return names, priors


def name_params(parmtype, subidx, groups_file, field_prior, mat_scale):
    """rabbit parameter names + prior sigmas for the selected global params."""
    gnames, gpriors = read_groups(groups_file)
    names = [None] * len(parmtype)
    sigmas = np.full(len(parmtype), np.nan)
    for i, (pt, si) in enumerate(zip(parmtype, subidx)):
        pt, si = int(pt), int(si)
        if pt == 14:
            names[i] = f"bfield_mode{si}"
            if field_prior > 0.0:
                sigmas[i] = field_prior
        elif pt == 15:
            names[i] = f"material_{gnames.get(si, f'group{si}')}"
            if mat_scale > 0.0:
                sigmas[i] = gpriors.get(si, 0.2) * mat_scale
        else:
            names[i] = f"glob_t{pt}_{si}"
    # the materialGroups tier file may reuse a group *name* for two group ids
    # (e.g. bpix_support appears twice in materialGroups50.txt); rabbit needs
    # unique parameter names, so disambiguate with the group index.
    seen = {}
    for i, nm in enumerate(names):
        seen.setdefault(nm, []).append(i)
    for nm, idxs in seen.items():
        if len(idxs) > 1:
            for i in idxs:
                names[i] = f"{nm}{int(subidx[i])}"
    if len(set(names)) != len(names):
        sys.exit("duplicate parameter names -- the (parmtype, subidx) map is not unique")
    return names, sigmas


# ---------------------------------------------------------------------------
# physical scales (used by --whiten and by diagnose_quadratic.py)
# ---------------------------------------------------------------------------
MFS = "/work/submit/david_w/ZMass/mfs"
DEFAULT_COEFFS = os.path.join(
    MFS, "data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt"
)


def read_modes(fname):
    """[(l, m, 'c'|'s')] in mode-index order, plus (r_scale, cmssw_norm)."""
    modes = []
    r_scale, cmssw_norm = R_SCALE_CM, False
    for line in open(fname):
        if line.startswith("#"):
            f = line.split()
            if len(f) >= 3 and f[1] == "r_scale":
                r_scale = float(f[2])
            if len(f) >= 3 and f[1] == "cmssw_norm":
                cmssw_norm = bool(int(f[2]))
            continue
        f = line.split()
        if len(f) < 5:
            continue
        modes.append((int(f[1]), int(f[2]), f[3]))
    return modes, r_scale, cmssw_norm


def field_scales(modes, r_scale, cmssw_norm, rmax=110.0, zmax=280.0,
                 nr=24, nz=41, nphi=16):
    """RMS over the tracker volume of |dB| per unit coefficient [T], per mode."""
    import sys as _sys

    if MFS not in _sys.path:
        _sys.path.insert(0, MFS)
    from harmonic_basis import bphi_basis, br_basis, bz_basis

    r = np.linspace(2.0, rmax, nr)
    z = np.linspace(-zmax, zmax, nz)
    phi = np.linspace(0.0, 2 * np.pi, nphi, endpoint=False)
    R, Z, P = np.meshgrid(r, z, phi, indexing="ij")
    R, Z, P = R.ravel(), Z.ravel(), P.ravel()
    w = R / R.sum()
    out = np.empty(len(modes))
    for j, (l, m, cs) in enumerate(modes):
        bz = bz_basis(l, m, cs, R, P, Z, r_scale=r_scale, cmssw_norm=cmssw_norm)
        br = br_basis(l, m, cs, R, P, Z, r_scale=r_scale, cmssw_norm=cmssw_norm)
        bp = bphi_basis(l, m, cs, R, P, Z, r_scale=r_scale, cmssw_norm=cmssw_norm)
        out[j] = np.sqrt(np.sum(w * (bz**2 + br**2 + bp**2)))
    return out


def param_scales(parmtype, subidx, coeffs_file=None, groups_file=None,
                 rmax=110.0, zmax=280.0):
    """Physical unit of every parameter.

    ``s_j`` such that ``theta_j * s_j`` is physical: Tesla of RMS |dB| in the
    tracker for a parmtype-14 coefficient, the group's own prior sigma (one
    "expected" unit of d ln dE/dx) for parmtype 15, and 1 otherwise. Raw
    parmtype-14 coefficients are NOT comparable to each other -- the basis is
    ``(R/r_scale)^(l-1)/r_scale``, so a high-l mode needs a huge coefficient to
    move the field at all -- and the raw field block is ~350x worse conditioned
    than the whitened one (6.6e9 vs 1.9e7 on the 260904f btojpsix sample).
    """
    modes, r_scale, cmssw_norm = read_modes(coeffs_file or DEFAULT_COEFFS)
    fs = field_scales(modes, r_scale, cmssw_norm, rmax, zmax)
    _, gprior = read_groups(groups_file)
    s = np.ones(len(parmtype))
    for i, (pt, si) in enumerate(zip(parmtype, subidx)):
        pt, si = int(pt), int(si)
        if pt == 14:
            s[i] = fs[si] if si < len(fs) else 1.0
        elif pt == 15:
            s[i] = gprior.get(si, 0.02)
    return s, fs


def discover_families(keys):
    fams = {}
    for k in keys:
        if not k.startswith("S") or not k[1:]:
            continue
        body = k[1:]
        if body.endswith("_re") or body.endswith("_im"):
            fam, comp = body[:-3], body[-2:]
        else:
            fam, comp = body, "re"
        fams.setdefault(FAMILY_ALIAS.get(fam, fam), {})[comp] = k
    return fams


def family_sort_key(name):
    if name in FAMILY_ORDER:
        return (0, FAMILY_ORDER.index(name), name)
    return (1, 0, name)


def build_phik_table(dm, tmax, npoints, read=(), write=None):
    tabs = np.linspace(0.0, tmax, npoints)
    for cache in read:
        if cache and os.path.exists(cache):
            z = np.load(cache)
            if len(z["tabs"]) == npoints and np.allclose(z["tabs"], tabs):
                log(f"kernel CF tabulation from cache {cache}")
                return tabs, z["phiK_tab"]
    t0 = time.time()
    blk = 1024
    tab = np.concatenate(
        [
            np.mean(np.exp(1j * np.outer(tabs[i : i + blk], dm)), axis=1)
            for i in range(0, npoints, blk)
        ]
    )
    log(f"kernel CF tabulation ({len(dm)} samples) built in {time.time()-t0:.1f} s")
    if write:
        try:
            np.savez(write, tabs=tabs, phiK_tab=tab)
        except OSError as e:
            log(f"kernel CF cache write failed: {e}")
    return tabs, tab


def to_hists(names, grad, hess, sparse):
    import hist

    hg = hist.Hist(hist.axis.StrCategory(names, name="params"))
    hg.values()[...] = grad
    ax0 = hist.axis.StrCategory(names, name="params0")
    ax1 = hist.axis.StrCategory(names, name="params1")
    if not sparse:
        hh = hist.Hist(ax0, ax1)
        hh.values()[...] = np.asarray(hess)
        return hg, hh
    import scipy.sparse
    from wums.sparse_hist import SparseHist

    coo = scipy.sparse.coo_array(hess)
    full = scipy.sparse.coo_array(
        (coo.data, (coo.coords[0], coo.coords[1])),
        shape=(ax0.extent, ax1.extent),
    ).tocsr()
    return hg, SparseHist(full, [ax0, ax1])


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    d = np.load(args.input, allow_pickle=False)
    keys = set(d.files)
    parmtype = d["parmtype"]
    subidx = d["subidx"]
    nall = len(parmtype)

    sel = (
        np.isin(parmtype, args.parmtypes)
        if args.parmtypes
        else np.ones(nall, dtype=bool)
    )
    if not sel.any():
        sys.exit(f"no extracted parameters of types {args.parmtypes}")
    isel = np.where(sel)[0]
    names, prior_sigmas = name_params(
        parmtype[isel],
        subidx[isel],
        args.groups,
        args.field_prior,
        args.material_prior_scale,
    )
    nfit = len(names)
    counts = {
        int(pt): int((parmtype[isel] == pt).sum())
        for pt in sorted(set(parmtype[isel].tolist()))
    }
    log(
        f"{nall} extracted parameters, using {nfit}: "
        + ", ".join(f"{TYPENAMES.get(k, f'type{k}')}({k}) x{v}" for k, v in counts.items())
    )

    grad_chi2 = d["grad"][isel] if "grad" in keys else None
    hess_chi2 = d["hess"][np.ix_(isel, isel)] if "hess" in keys else None
    if args.no_quadratic:
        grad_chi2 = hess_chi2 = None

    # -- mass inputs ---------------------------------------------------------
    data = families = D = None
    n = 0
    if not args.no_mass and "m0" in keys:
        sigma = d["sigma"].astype(np.float64)
        n = len(sigma)
        keep = np.ones(n, dtype=bool)
        if args.max_chi2_ndof > 0.0:
            if "chi2ndof" not in keys:
                sys.exit(
                    "--max-chi2-ndof needs the 'chi2ndof' array; re-run "
                    "extract.py (it stores it since 2026-09-04 evening)"
                )
            keep = d["chi2ndof"] < args.max_chi2_ndof
            log(
                f"chi2/ndof < {args.max_chi2_ndof:g}: keeping "
                f"{int(keep.sum())}/{n} mass-term candidates "
                f"({100.0*keep.mean():.2f}%)"
            )
        idx = np.where(keep)[0]
        if args.maxn and args.maxn < len(idx):
            idx = idx[: args.maxn]
        n = len(idx)
        sigma = sigma[idx]
        m0 = d["m0"][idx].astype(np.float64)
        data = {
            "sigma": sigma,
            "mobs": m0 - args.mref,
            "vgf": d["vgf"][idx].astype(np.float64),
            "tgrid": np.asarray(d["tgrid"], dtype=np.float64),
        }
        found = discover_families([k for k in d.files if k.startswith("S")])
        fam_names = sorted(found, key=family_sort_key)
        families = [{"name": GAUSS_FAMILY, "param": None, "kind": "gauss"}]
        for fam in fam_names:
            families.append({"name": fam, "param": None, "kind": "tab"})
            for comp, key in sorted(found[fam].items()):
                data[f"S_{comp}_{fam}"] = d[key][idx]
        log(f"{n} candidates, families {[f['name'] for f in families]}")
        if not args.no_jac and "D" in keys:
            D = d["D"][idx][:, isel].astype(np.float64)
            # SIGN, once and for all: the extraction stores the physical
            # response D = dm_i/dtheta. MassCFTerm evaluates the density at
            # delta_i = mobs_i - shift - (D_card theta)_i, i.e. it treats its
            # rows as d(predicted mass)/dtheta, whereas here it is the
            # *reconstructed* mass that moves with theta. So the card carries
            # D_card = -D, and everything downstream (the card rows AND the
            # injection) uses D_card.
            D = -D
        elif not args.no_jac:
            log("WARNING: the extraction carries no D rows; the mass term will "
                "not depend on theta")

    # -- whitening -----------------------------------------------------------
    pscale = np.ones(nfit)
    if args.whiten:
        pscale, _ = param_scales(
            parmtype[isel], subidx[isel], args.coeffs, args.groups
        )
        inv = 1.0 / np.maximum(pscale, 1e-300)
        # theta_card = theta_raw * s  =>  G -> G/s, K -> K/(s s^T), D -> D/s
        if grad_chi2 is not None:
            grad_chi2 = grad_chi2 * inv
            hess_chi2 = hess_chi2 * np.outer(inv, inv)
        if D is not None:
            D = D * inv[None, :]
        prior_sigmas = prior_sigmas * pscale
        log(
            "whitened: 1 card unit = 1 T of RMS |dB| (parmtype 14) / "
            "1 prior sigma (parmtype 15); scale range "
            f"{pscale.min():.3e} .. {pscale.max():.3e}"
        )

    # -- injection -----------------------------------------------------------
    inject = np.zeros(nfit)
    if args.inject:
        idx = {nm: i for i, nm in enumerate(names)}
        for spec in args.inject:
            nm, val = spec.rsplit(":", 1)
            if nm not in idx:
                sys.exit(f"--inject for unknown parameter '{nm}'")
            inject[idx[nm]] = float(val)
        log(
            "injecting "
            + ", ".join(f"{nm}={inject[idx[nm]]:+.5g}" for nm in names if inject[idx[nm]])
        )
        if D is not None and not args.inject_quad_only:
            # "the true correction is dtheta" means the uncorrected masses are
            # off by -(dm/dtheta) dtheta, i.e. mobs += D_card dtheta with the
            # card's (already negated) rows. Same convention as the quadratic
            # term's G -> G - K dtheta: both then prefer theta_base + dtheta.
            shift = D @ inject
            data["mobs"] = data["mobs"] + shift
            log(
                f"  mass term: m_i^0 += D_card.dtheta, mean "
                f"{shift.mean()*1e3:+.4f} MeV, rms {shift.std()*1e3:.4f} MeV"
            )
        if grad_chi2 is not None and not args.inject_mass_only:
            grad_chi2 = grad_chi2 - hess_chi2 @ inject
            log("  quadratic term: G -> G - K dtheta")

    # -- sign / units diagnostics -------------------------------------------
    if args.check_sign and D is not None:
        m = data["mobs"] + args.mref
        Dphys = -D  # report the physical response dm/dtheta, not the card rows
        for j in range(min(nfit, 8)):
            col = Dphys[:, j]
            if not np.any(col):
                continue
            rel = col / m
            log(
                f"  D[:,{j}] {names[j]:24s} mean {col.mean():+.4e} GeV  "
                f"rms {col.std():.3e}  dlnm/dtheta {rel.mean():+.4e} "
                f"(spread {rel.std()/max(abs(rel.mean()),1e-30):.3f})"
            )
        if "bfield_mode0" in names:
            j0 = names.index("bfield_mode0")
            rel0 = float((Dphys[:, j0] / m).mean())
            dbb = 1.0 / (R_SCALE_CM * B_NOMINAL)
            log(
                f"  mode0: dlnm/dtheta = {rel0:+.4e} per unit (unit = dB/B "
                f"{dbb:.3e}); dlnm/dlnB = {rel0/dbb:+.3f}"
            )
            if rel0 != 0.0:
                log(
                    f"  mapping: a momentum scale alpha of 1e-3 is equivalent to "
                    f"bfield_mode0 = {1e-3/rel0:+.5g} "
                    "(m(theta) = m0 (1 + rel0 theta))"
                )

    # -- card ----------------------------------------------------------------
    from rabbit import tensorwriter, unbinned

    writer = tensorwriter.TensorWriter()
    writer.add_dummy_channel(name="calib_dummy")
    declared = []

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
            sys.exit(f"--poi names not among the parameters: {sorted(unknown)}")

    if data is not None:
        for f in families:
            f["param"] = "r" if args.model == "r" else f"k_{f['name']}"

        dm = None
        if args.kernel_cache:
            dm = np.load(args.kernel_cache)["dm"]
        elif "dm" in keys:
            dm = d["dm"]
            log("FSR kernel taken from the extraction's own gen masses")
        if dm is not None:
            if args.maxk and args.maxk < len(dm):
                dm = dm[: args.maxk]
            tmax = data["tgrid"][-1] / data["sigma"].min()
            tabs, phik = build_phik_table(
                dm,
                tmax,
                args.phik_points,
                read=(args.phik_cache,),
                write=args.phik_cache,
            )
            data["phik_t"] = tabs
            data["phik_re"] = phik.real.copy()
            data["phik_im"] = phik.imag.copy()
        else:
            log("no FSR kernel: the resonance is a pure delta")

        jac = None
        jac_params = []
        if D is not None:
            # D already carries the MassCFTerm sign (D_card = -dm/dtheta)
            rows, cols = np.nonzero(D)
            idx = np.stack([rows, cols], axis=1).astype(np.int64)
            vals = D[rows, cols]
            data["jac_indices"] = idx
            data["jac_values"] = vals
            data["jac_shape"] = np.array([n, nfit], dtype=np.int64)
            jac = (idx, vals, (n, nfit))
            jac_params = names
            log(
                f"sparse D: {len(vals)} entries "
                f"({100.0*len(vals)/max(n*nfit,1):.1f}% fill), in the "
                "MassCFTerm sign convention (D_card = -dm/dtheta)"
            )

        window = (args.mref - 0.5 * args.window, args.mref + 0.5 * args.window)
        term = unbinned.MassCFTerm(
            "mass",
            sigma=data["sigma"],
            mobs=data["mobs"],
            tgrid=data["tgrid"],
            families=[
                dict(
                    f,
                    **{
                        c: data[f"S_{c}_{f['name']}"]
                        for c in ("re", "im")
                        if f"S_{c}_{f['name']}" in data
                    },
                )
                for f in families
            ],
            vgf=data["vgf"],
            background=unbinned.UniformBackground(window),
            m_ref=args.mref,
            scale_param="alpha" if args.with_alpha else None,
            bkg_frac_param="f_bkg" if args.float_bkg else None,
            bkg_frac=args.fbkg,
            floor=args.floor,
            chunk=args.chunk,
            channel="jpsi",
            jac=jac,
            jac_params=jac_params,
        )
        prior_by_name = dict(zip(names, prior_sigmas))
        defaults, sigmas, means, is_poi = [], [], [], []
        for p in term.param_names:
            if p == "alpha":
                defaults.append(args.alpha0)
                sigmas.append(np.nan)
                is_poi.append(1)
            elif p == "f_bkg":
                defaults.append(args.fbkg / unbinned.FBKG_UNIT)
                sigmas.append(np.nan)
                is_poi.append(0)
            elif p in prior_by_name:
                defaults.append(0.0)
                sigmas.append(prior_by_name[p])
                is_poi.append(1 if p in poi_set else 0)
            else:  # resolution family scale
                defaults.append(args.k0)
                sigmas.append(args.res_prior if args.res_prior > 0 else np.nan)
                is_poi.append(0)
            means.append(0.0 if np.isfinite(sigmas[-1]) else defaults[-1])
        writer.add_unbinned_term(
            "mass",
            term.config(),
            term.param_names,
            data,
            param_defaults=defaults,
            param_prior_sigmas=sigmas,
            param_prior_means=means,
            param_is_poi=is_poi,
        )
        declared = list(term.param_names)
        log(f"unbinned term parameters: {term.param_names}")

    if grad_chi2 is not None:
        sparse = nfit > args.dense_max
        H = 0.5 * np.asarray(hess_chi2)
        g = 0.5 * np.asarray(grad_chi2)
        if sparse:
            import scipy.sparse

            if args.hess_drop > 0.0:
                dsq = np.sqrt(np.abs(np.diag(H)))
                scale = np.outer(dsq, dsq)
                H = np.where(np.abs(H) > args.hess_drop * scale, H, 0.0)
            H = scipy.sparse.csr_array(H)
            log(f"sparse external Hessian: {H.nnz} nnz of {nfit*nfit}")
        hg, hh = to_hists(names, g, H, sparse)
        writer.add_external_likelihood_term(grad=hg, hess=hh, name="hitchi2")
        log(
            f"external term 'hitchi2': {nfit} parameters "
            f"({'sparse' if sparse else 'dense'} Hessian), from "
            f"{int(d['ncand_quadratic']) if 'ncand_quadratic' in keys else 0} "
            "candidates"
        )

    undeclared = [nm for nm in names if nm not in declared]
    if undeclared:
        keep = [i for i, nm in enumerate(names) if nm in set(undeclared)]
        writer.add_auxiliary(
            "global_params",
            {
                "params": [names[i] for i in keep],
                "defaults": np.zeros(len(keep)),
                "prior_sigmas": prior_sigmas[keep],
                "prior_means": np.zeros(len(keep)),
                "is_poi": np.array(
                    [1 if names[i] in poi_set else 0 for i in keep], dtype=np.int64
                ),
            },
        )

    writer.add_auxiliary(
        "global_index_map",
        {
            "params": names,
            "global_index": np.asarray(d["fitidx"][isel], dtype=np.int64),
            "parmtype": np.asarray(parmtype[isel], dtype=np.int64),
            "subindex": np.asarray(subidx[isel], dtype=np.int64),
            "prior_sigmas": np.asarray(prior_sigmas, dtype=np.float64),
            "scale": np.asarray(pscale, dtype=np.float64),
            "injected": inject,
            "provenance": [
                json.dumps(
                    {
                        "extract": os.path.abspath(args.input),
                        "nglobal": int(d["nglobal"]),
                        "ncand_quadratic": (
                            int(d["ncand_quadratic"])
                            if "ncand_quadratic" in keys
                            else 0
                        ),
                        "ncand_mass": int(n),
                        "hess_format": str(d["hess_format"]) if "hess_format" in keys else "",
                        "rad_model": int(d["rad_model"]) if "rad_model" in keys else 0,
                        "with_alpha": bool(args.with_alpha),
                        "jac_sign": "D_card = -dm/dtheta",
                        "chi2_to_nll": 0.5,
                        "whiten": bool(args.whiten),
                        "field_prior": args.field_prior,
                        "material_prior_scale": args.material_prior_scale,
                    }
                )
            ],
        },
    )

    models = []
    if declared:
        models.append("UnbinnedParams")
    if undeclared:
        models.append("ExternalParams bundle:global_params")
    folder = os.path.dirname(os.path.abspath(args.output)) or "."
    base = os.path.basename(args.output)
    if base.endswith(".hdf5"):
        base = base[: -len(".hdf5")]
    os.makedirs(folder, exist_ok=True)
    t0 = time.time()
    writer.write(outfolder=folder, outfilename=base)
    log(f"wrote {os.path.join(folder, base)}.hdf5 in {time.time()-t0:.1f} s")
    log(
        "run it with:  rabbit_fit.py %s.hdf5 -o out/ -t 0 --unblind %s"
        % (
            os.path.join(folder, base),
            " ".join(f"--paramModel {m}" for m in models),
        )
    )


if __name__ == "__main__":
    main()
