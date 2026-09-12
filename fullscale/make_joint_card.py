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

The `f_ang` caveat: the `jpsimc_20M_260905` production carries no
`Jpsi_covrefmom` branch, so the J/psi leg's Jensen `s^2` cannot be made
truth-free from the candidate itself. `--jpsi-fang` supplies the MC-measured
value (gun 0.086 / data 0.106); the DY leg uses its own per-candidate
`Jpsi_fang`.

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

WHAT `--inject` DOES AND DOES NOT CLOSE
---------------------------------------
`--inject NAME:VALUE` is `make_global_term.py`'s convention: every mass term's
`m_i^0` moves by `(D_card dtheta)_i` and the quadratic term's `G` by
`-K dtheta`, so a translation-invariant mass likelihood puts the joint minimum
at `theta*_base + dtheta` exactly.

In the FLUCTUATION form the closure is exact to numerical noise. On a
60 k + 60 k joint smoke card (`cards/joint_smoke.hdf5` vs
`cards/joint_smoke_inj.hdf5`, `fit_joint.py --free bfield_mode0 bfield_mode1
bfield_mode2`, `dtheta = (+0.010, -0.002, +0.003)` card units), as the
difference between the injected and the un-injected minimum:

    mode1   -0.00200000 vs -0.002    +1.5e-11  =  +0.0000 % of the injection
    mode2   +0.00300000 vs +0.003    -6.7e-11  =  -0.0000 % of the injection
    mode0   +0.01000023 vs +0.010    +2.3e-07  =  +0.0023 % of the injection

The fluctuation form's residual is linear in theta: the Jensen map contributes
a per-candidate CONSTANT `d_i` and a CF factor with no `delta` in it, so the
only thing that does not translate is the second-order dependence of `a_i`,
`c_i`, `d_i` themselves on the shifted masses -- the 2.3e-7 above, which is
1.6e-3 of `bfield_mode0`'s own statistical error and not a minimiser artefact
(the Newton step still implied by the residual gradient is 1.8e-12).

With `--corr-form residual` the same test closes to -3e-4 on modes 1 and 2 but
leaves +12 % on mode 0, and that defect is entirely `jensen_mode="exact"`:
`MassCFTerm._jensen_m` is `mobs + m_ref`, the OBSERVED mass, and it is the
denominator of `r = delta/m` (and of the discriminant floor) in the exact
second-order map, so the map does not commute with a shift of the observed
masses. Switching that one correction off and re-measuring the translation
defect in the gradient isolates it:

    jensen exact   defect  -8494  ->  +1.214e-3 = +12.14 % of the injection
    jensen off     defect   -235  ->  +3.35e-5  =  +0.34 % of the injection
    a_res off      defect  -8244  ->  +1.178e-3 = +11.78 %  (a_res is NOT it)

The sparse-`D` wiring is therefore right -- it closes on the two modes whose
`D` column is not proportional to `m` -- and it shows up ONLY on mode 0
because the map is invariant under a COMMON rescaling of `(m, sigma, delta)`,
which is exactly what a momentum-scale error is, while `--inject` moves the
mean and leaves `sigma`, `a_res` and `jensen_s2` at the values computed from
the unshifted masses. A faithful scale injection would have to rescale those
too, so in the residual form a `--inject` closure on `bfield_mode0` (or on any
mode whose `D` column is proportional to `m`) is a 12 %-level statement, not a
1e-2 one.

PHASE 3: `--material`
---------------------
`--material` replaces the four ad-hoc resolution knobs (`k_hit`, `k_ms`,
`k_ioni`, `k_rad`) by the PHYSICAL parameterisation of `MaterialCFTerm`, on
BOTH legs at once:

    S_f(tau; k) = S_f^fix(tau) + sum_g A(k_g) S_{f,g}(tau) ,   A(k) = exp(k)
    v_i(eps)    = v_other,i    + sum_c H(eps_c) v_{c,i} ,      H(eps) = 1 + eps

over the SAME `material_<group>` names the quadratic hit-chi2 term and the
sparse `D` already use. Each of the 42 parmtype-15 parameters therefore
appears in THREE places in one likelihood -- the quadratic curvature, the
mass terms' width, and the mass terms' mean -- which is the whole point of
phase 3. The 18 `hitres_<class>` parameters are new and come from the mass
terms alone.

The per-group exponents come from a `cf_inmaker.py pairs --groups` cache
(`Sgrp_*` in the CSR layout `MaterialCFTerm` consumes); the FLAT exponents
`Sms`/`Sio_re`/... in the same file are the k = 0 value of the same sum, to
7.3e-8 relative (`matres/validate_inmaker_groups.py`, the float32 storage
floor; the maker's own float64 closure is 1.4e-14).

WHAT IS EXACT AT k = 0, AND WHAT IS NOT
    * the per-candidate exponent: `sum_g A(0) S_{f,g} == S_f^flat` to the
      7.3e-8 above (float32 storage), NOT to float64;
    * the Gaussian share: `vg_other + sum_c H(0) v_c == vgf` EXACTLY -- the
      maker defines `vg_other` as `vgf - sum_c v_c`, so it is the algebraic
      remainder and not "noise". It is negative for 40.2 % of J/psi
      candidates (median |vg_other|/vgf = 3.2e-7, max 2.3e-4).  It is
      written through unclipped: clipping at 0 would break the identity for
      those 40 %;
    * the TRUNCATION NORMALISATION `Z_c`: `_norm_z` sums over resolution
      CLASSES, which have no per-group decomposition. In `--material` the
      class exponents are written as rabbit's FIXED norm families (coefficient
      1, `norm_fixed` in the term config), i.e. `Z_c` is evaluated at the
      production's own resolution -- which is EXACTLY the constant phase 2
      evaluates it at, since phase 2 runs with `--fix k_hit k_ms k_ioni k_rad`.
      Nothing is lost relative to phase 2; what is not modelled is the
      (second-order) response of `Z_c` to the material parameters themselves.

WHAT NOTHING CONSTRAINS (the census `--material` prints)
    `material_pp1_cables` is touched by NO candidate on either leg, and
    `material_thermal_screen` / `material_support_tube` by under 0.1 %; the
    hit-chi2 Hessian is blind to those three AND to `material_beampipe`. The
    card prints both censuses and a ready-to-paste `--freezeParameters` list;
    it does NOT freeze anything itself, because a frozen parameter is a
    physics decision and belongs in the fit command.

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
    # ---- phase 3: the physical resolution parameterisation ---------------
    p.add_argument("--material", action="store_true",
                   help="make BOTH mass terms MaterialCFTerms over the 42 "
                        "parmtype-15 `material_<group>` parameters the "
                        "quadratic term and the sparse D already carry, plus "
                        "18 new `hitres_<class>`. Both pairs caches must have "
                        "been built with `cf_inmaker.py pairs --groups`.")
    p.add_argument("--amount-mode", choices=["exp", "linear"], default="exp",
                   help="A(k) in S_f = S^fix + sum_g A(k_g) S_{f,g}. `exp` is "
                        "the C++ `matStepFact` convention.")
    p.add_argument("--hit-mode", choices=["exp", "linear"], default="linear")
    p.add_argument("--no-hits", action="store_true",
                   help="drop the 18 hitres_<class> parameters; the whole "
                        "Gaussian share then rides as a FIXED vg_other = vgf")
    p.add_argument("--legacy-families", action="store_true",
                   help="ALSO keep the ad-hoc k_hit knob on top of the "
                        "physical parameterisation, as "
                        "matres/make_material_card.py does. It DOUBLE COUNTS "
                        "the Gaussian share by construction and exists only "
                        "for comparison fits.")
    p.add_argument("--group-prune", type=float, default=0.0,
                   help="fold group rows contributing less than this fraction "
                        "of the candidate's max_tau |S| into the FIXED "
                        "baseline. 0 keeps every row; it is what sets the "
                        "card size (~30 kB/candidate at 0).")
    p.add_argument("--hit-prior", type=float, default=0.0,
                   help="Gaussian prior sigma on every hitres_<class>")
    p.add_argument("--material-maxrows", type=int, default=0,
                   help="abort if the group block would exceed this many CSR "
                        "rows (a guard against building a 100 GB card by "
                        "accident). 0 = no limit.")
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
    p.add_argument("--corr-form", choices=["fluctuation", "residual"],
                   default="fluctuation",
                   help="WHERE the two corrections act, on BOTH legs. "
                        "`fluctuation` applies them as one deterministic map "
                        "of the resolution fluctuation inside the convolution; "
                        "it is the only form defined at the Z, and it is also "
                        "the form in which the residual is LINEAR in theta, "
                        "which is what the --inject translation closure needs "
                        "(see the module docstring).")
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
    p.add_argument("--z-selection-aux", default=None,
                   help="aux npz carrying the standard selection's columns "
                        "for the Z leg (see make_card.py --selection-aux)")
    p.add_argument("--jpsi-selection-aux", default=None,
                   help="the same for the J/psi leg")
    p.add_argument("--no-standard-selection", action="store_true",
                   help="passed through to BOTH legs' make_card.select: do "
                        "not apply the standard two-track selection "
                        "(resolution/selection.py). For studies of the tail "
                        "itself only")
    return p.parse_args(argv)


def _selargv(args, which=None):
    """The standard-selection flags, forwarded to a `make_card` argv."""
    out = ["--no-standard-selection"] if args.no_standard_selection else []
    aux = getattr(args, f"{which}_selection_aux", None) if which else None
    if aux:
        out += ["--selection-aux", aux]
    return out


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
# phase 3: the parmtype-15 group block
# ---------------------------------------------------------------------------
def zip_member_array(path, key):
    """A read-only memmap of an UNCOMPRESSED ``.npz`` member, or ``None``.

    `np.load(npz)[key]` materialises the WHOLE member; the per-group exponent
    blocks are 3.8 GB each and there are five of them per leg, so a smoke card
    that keeps 20 000 candidates would still read 19 GB and hold it. Both group
    caches are written by `np.savez` (stored, not deflated), so the member's
    payload is a contiguous `.npy` inside the zip and can be memory-mapped in
    place; fancy-indexing the kept CSR rows then touches only those pages.

    Returns ``None`` for a deflated member, and the caller falls back to
    `np.load`, so this is an optimisation and never a correctness dependency.
    """
    import struct
    import zipfile

    z = zipfile.ZipFile(path)
    try:
        zi = z.getinfo(key + ".npy")
    except KeyError:
        return None
    if zi.compress_type != zipfile.ZIP_STORED:
        return None
    with open(path, "rb") as f:
        f.seek(zi.header_offset)
        hdr = struct.unpack("<IHHHHHIIIHH", f.read(30))
        if hdr[0] != 0x04034B50:
            return None
        f.seek(zi.header_offset + 30 + hdr[9] + hdr[10])
        ver = np.lib.format.read_magic(f)
        rd = (np.lib.format.read_array_header_1_0 if ver == (1, 0)
              else np.lib.format.read_array_header_2_0)
        shape, fortran, dt = rd(f)
        off = f.tell()
    return np.memmap(path, dtype=dt, mode="r", offset=off, shape=shape,
                     order="F" if fortran else "C")


def csr_rows(ptr, idx):
    """The flat CSR row indices of candidates ``idx``, and the new row pointer.

    Vectorised on purpose: the obvious
    ``np.concatenate([np.arange(ptr[i], ptr[i+1]) for i in idx])`` is 650 000
    python-level `arange`s on the full J/psi leg.
    """
    cnt = (ptr[idx + 1] - ptr[idx]).astype(np.int64)
    nptr = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
    if not len(idx):
        return np.zeros(0, np.int64), nptr, cnt
    rows = np.repeat(ptr[idx] - nptr[:-1], cnt) + np.arange(nptr[-1],
                                                            dtype=np.int64)
    return rows, nptr, cnt


class MaterialContext:
    """Everything `--material` needs that is common to the two legs.

    It owns the join between the three places a parmtype-15 parameter appears:
    the quadratic term's `names` (which is where the NAME comes from), the
    whitening scale (which is where `group_units` comes from), and the cache's
    own `group_names` (which is what the exponents are indexed by). The three
    are cross-checked rather than assumed -- a silent group-index shift would
    otherwise mis-associate every material amount with the wrong detector.
    """

    def __init__(self, args, names, parmtype, subidx, pscale, prior_sigmas,
                 poi_set):
        self.args = args
        self.names = list(names)
        self.prior_sigmas = np.asarray(prior_sigmas, np.float64)
        self.poi_set = set(poi_set)
        self.matcol = {int(si): j for j, (pt, si)
                       in enumerate(zip(parmtype, subidx)) if int(pt) == 15}
        self.pscale = np.asarray(pscale, np.float64)
        self.census = {}

    # -- the join ----------------------------------------------------------
    def group_params(self, gnames, tag, log):
        ng = len(gnames)
        if len(self.matcol) != ng:
            raise SystemExit(
                f"{tag}: the cache has {ng} material groups but the quadratic "
                f"extraction has {len(self.matcol)} parmtype-15 columns; the "
                "two were not produced from the same materialGroups file")
        gp = [self.names[self.matcol[g]] if g in self.matcol
              else f"material_group{g}" for g in range(ng)]
        bad = [(g, gnames[g], gp[g]) for g in range(ng) if gnames[g] != gp[g]]
        if bad:
            raise SystemExit(
                f"{tag}: the cache's own group name and the quadratic "
                f"extraction's parameter name disagree for {len(bad)} "
                f"group(s), e.g. {bad[:3]}. Pass the --groups tier file the "
                "production used; without it name_params() invents "
                "`material_group<i>` and the join is by index alone.")
        return gp

    def declarations(self, group_params, hit_params):
        """The (default, sigma, mean, is_poi) row of every parameter this
        block adds. The material rows are the CALIBRATION rows -- the same
        ones `attach_jac` writes -- so the two legs and the bundle agree."""
        out = {}
        for g, nm in enumerate(group_params):
            j = self.matcol.get(g)
            sig = float(self.prior_sigmas[j]) if j is not None else np.nan
            out[nm] = (0.0, sig, 0.0, 1 if nm in self.poi_set else 0)
        hp = self.args.hit_prior if self.args.hit_prior > 0 else np.nan
        for nm in hit_params:
            out[nm] = (0.0, hp, 0.0, 0)
        return out

    # -- the truncation normalisation --------------------------------------
    @staticmethod
    def fixed_norm(norm):
        """Turn the PARAMETERISED norm classes into FIXED ones.

        `_norm_z` sums over resolution classes with a coefficient
        `values[param]` per family. A `MaterialCFTerm` has no `k_*`, and the
        classes have no per-group decomposition, so the classes are handed to
        rabbit as `fixed` families (coefficient 1). Phase 2 fixes the same
        knobs at 1, so this reproduces phase 2's `Z_c` exactly rather than
        approximating it.
        """
        fixed = [{"name": "hit", "kind": "gauss"}]
        for f in norm.get("families", []):
            e = {"name": f["name"], "kind": "tab"}
            for c in ("re", "im"):
                if c in f:
                    e[c] = f[c]
            fixed.append(e)
        return {"sigma": norm["sigma"], "vgf": norm["vgf"],
                "class": norm["class"], "families": [], "fixed": fixed}

    # -- the block ---------------------------------------------------------
    def block(self, path, idx, tag, log):
        """``(constructor kwargs, datasets, census)`` for one leg."""
        args = self.args
        d = np.load(path, allow_pickle=False)
        keys = set(d.files)
        need = ("grp_ptr", "grp_id", "hit_ptr", "hit_cls", "hit_v",
                "vg_other", "group_names", "hit_classes", "families")
        miss = [k for k in need if k not in keys]
        if miss:
            raise SystemExit(
                f"{tag}: {os.path.basename(path)} has no {miss}; --material "
                "needs a cache built with `cf_inmaker.py pairs --groups`")
        gnames = [str(x) for x in d["group_names"]]
        cnames = [str(x) for x in d["hit_classes"]]
        fams = [str(x) for x in d["families"]]
        ngroups, nt = len(gnames), len(d["tgrid"])
        n = len(idx)
        gp = self.group_params(gnames, tag, log)
        # k_phys = theta_card * group_units  (theta_card = theta_raw * pscale)
        units = np.array([1.0 / max(self.pscale[self.matcol[g]], 1e-300)
                          if g in self.matcol else 1.0
                          for g in range(ngroups)])

        ptr = np.asarray(d["grp_ptr"], np.int64)
        rows, nptr, cnt = csr_rows(ptr, idx)
        nnz = int(nptr[-1])
        if args.material_maxrows and nnz > args.material_maxrows:
            raise SystemExit(
                f"{tag}: {nnz} CSR rows exceeds --material-maxrows "
                f"{args.material_maxrows} ({nnz*nt*4*len(fams)/1e9:.1f} GB of "
                "exponents); prune, cut the statistics, or raise the limit")
        _mm = zip_member_array(path, "grp_id")
        gid = np.asarray((_mm if _mm is not None else d["grp_id"])[rows],
                         np.int64)
        log(f"  {tag} group block: {n} candidates, {nnz} CSR rows "
            f"({nnz/max(n,1):.2f} groups/candidate), {len(fams)} families, "
            f"{nnz*nt*4*len(fams)/1e9:.2f} GB of exponents")

        arrs = {}
        amp = np.zeros(nnz)
        for f in fams:
            mm = zip_member_array(path, "S" + f)
            a = np.asarray(mm[rows] if mm is not None else d["S" + f][rows])
            arrs[f] = a
            np.maximum(amp, np.abs(a).max(axis=1), out=amp)

        seg = np.repeat(np.arange(n), cnt)
        drop = np.zeros(nnz, bool)
        if args.group_prune > 0.0:
            top = np.zeros(n)
            np.maximum.at(top, seg, amp)
            drop = amp < args.group_prune * top[seg]
            log(f"    --group-prune {args.group_prune:g}: {int(drop.sum())} of "
                f"{nnz} rows ({100.*drop.mean():.2f} %) folded into the fixed "
                "baseline")

        merged = {}
        for f in fams:
            nm = f[:-3] if f.endswith(("_re", "_im")) else f
            comp = "im" if f.endswith("_im") else "re"
            e = merged.setdefault(nm, {"name": nm})
            e[comp] = arrs[f][~drop].astype(np.float32)
            if drop.any():
                fx = np.zeros((n, nt), np.float32)
                np.add.at(fx, seg[drop], arrs[f][drop])
                e["fix_" + comp] = fx
            arrs[f] = None
        group_families = [merged[k] for k in sorted(merged)]

        kcnt = np.zeros(n, np.int64)
        np.add.at(kcnt, seg[~drop], 1)
        nptr = np.concatenate([[0], np.cumsum(kcnt)]).astype(np.int64)
        gid_kept = gid[~drop]

        datasets = {"grp_ptr": nptr, "grp_id": gid_kept.astype(np.int32),
                    "group_units": units}
        for m in group_families:
            for c in ("re", "im"):
                if c in m:
                    datasets[f"Sg_{c}_{m['name']}"] = m[c]
                if "fix_" + c in m:
                    datasets[f"Sgfix_{c}_{m['name']}"] = m["fix_" + c]

        # ---- the Gaussian hit share -------------------------------------
        hptr = np.asarray(d["hit_ptr"], np.int64)
        hrows, hnptr, _ = csr_rows(hptr, idx)
        vgf = np.asarray(d["vgf"], np.float64)[idx]
        vgo = np.asarray(d["vg_other"], np.float64)[idx]
        nneg = int((vgo < 0).sum())
        if args.no_hits:
            hit_params = []
            share = (np.zeros(n + 1, np.int64), np.zeros(0, np.int64),
                     np.zeros(0, np.float64), vgf)
            log("    --no-hits: the whole Gaussian share is a FIXED vg_other")
        else:
            hit_params = [f"hitres_{c}" for c in cnames]
            hv = np.asarray(d["hit_v"], np.float64)[hrows]
            hc = np.asarray(d["hit_cls"], np.int64)[hrows]
            share = (hnptr, hc, hv, vgo)
            tot = np.zeros(n)
            np.add.at(tot, np.repeat(np.arange(n), np.diff(hnptr)), hv)
            resid = np.abs(vgo + tot - vgf) / np.maximum(vgf, 1e-300)
            log(f"    hit share: {len(cnames)} classes, "
                f"{len(hc)/max(n,1):.2f} rows/candidate; "
                f"|vg_other + sum_c v_c - vgf|/vgf max {resid.max():.3e}; "
                f"vg_other < 0 for {nneg} "
                f"candidates ({100.*nneg/max(n,1):.2f} %)")
        datasets["hit_ptr"] = share[0]
        datasets["hit_cls"] = np.asarray(share[1], np.int32)
        datasets["hit_v"] = np.asarray(share[2], np.float64)
        datasets["vg_other"] = np.asarray(share[3], np.float64)
        datasets["hit_units"] = np.ones(len(hit_params))

        kw = dict(group_params=gp, group_families=group_families,
                  grp_ptr=nptr, grp_id=gid_kept, group_units=units,
                  hit_params=hit_params, hit_share=share,
                  amount_mode=args.amount_mode, hit_mode=args.hit_mode)

        occ = np.bincount(gid_kept, minlength=ngroups) / max(n, 1)
        lev = np.zeros(ngroups)
        np.add.at(lev, gid_kept, amp[~drop])
        self.census[tag] = {"names": gnames, "occ": occ, "lev": lev, "n": n,
                            "nnz": int(nptr[-1]), "nneg_vgother": nneg}
        return kw, datasets, self.census[tag]


def strip_flat_families(datasets, log=print):
    """Drop the per-candidate FLAT exponents from a material term's datasets.

    They are the k = 0 value of the group sum, so keeping them would add ~30 %
    to the card for arrays no `MaterialCFTerm` reads. The `_norm` copies
    (`S_*_norm`, one row per resolution CLASS) STAY: they are what the fixed
    truncation normalisation is built from.
    """
    gone = [k for k in list(datasets)
            if (k.startswith("S_re_") or k.startswith("S_im_"))
            and not k.endswith("_norm")]
    nb = sum(datasets[k].nbytes for k in gone)
    for k in gone:
        datasets.pop(k)
    if gone:
        log(f"    dropped the flat per-candidate exponents {gone} "
            f"({nb/1e9:.2f} GB); they are the k = 0 group sum")
    return datasets


def norm_from_datasets(datasets):
    """Rebuild the ``norm`` dict `make_card.build` handed to the term."""
    if "norm_sigma" not in datasets:
        return None
    fams = {}
    for k in datasets:
        if k.startswith("S_") and k.endswith("_norm"):
            body = k[len("S_"):-len("_norm")]
            comp, nm = body.split("_", 1)
            fams.setdefault(nm, {"name": nm})[comp] = datasets[k]
    return {"sigma": datasets["norm_sigma"], "vgf": datasets["norm_vgf"],
            "class": datasets["norm_class"],
            "families": [fams[k] for k in sorted(fams)]}


def report_census(matctx, names, dead, poi_set, log=print):
    """WHAT NOTHING CONSTRAINS -- printed so the fit can freeze it explicitly.

    Three separate blindnesses, and the union is what has to be named on the
    command line: the quadratic term's null space (`dead`), the groups no
    candidate of a leg touches, and the groups so few candidates touch that the
    mass terms cannot move them either.
    """
    if not matctx.census:
        return []
    tags = list(matctx.census)
    gnames = matctx.census[tags[0]]["names"]
    ng = len(gnames)
    log("")
    log("  === what constrains the 42 material amounts ===")
    log(f"    {'group':<28s}" + "".join(f"{t + ' occ':>14s}" for t in tags)
        + f"{'hit-chi2':>10s}")
    dead_set = set(dead)
    untouched, thin = [], []
    for g in range(ng):
        occ = [matctx.census[t]["occ"][g] for t in tags]
        nm = gnames[g]
        blind = "BLIND" if nm in dead_set else "ok"
        if max(occ) == 0.0:
            untouched.append(nm)
        elif max(occ) < 0.01:
            thin.append(nm)
        if max(occ) < 0.01 or nm in dead_set:
            log(f"    {nm:<28s}" + "".join(f"{o:14.6f}" for o in occ)
                + f"{blind:>10s}")
    log(f"    ({ng} groups; only those under 1 % occupancy or blind to the "
        "hit-chi2 term are listed)")
    # a group is unconstrainable when NO term can move it: no candidate
    # touches it, or so few do that only the quadratic term could -- and the
    # quadratic term is blind to it too.
    frozen = sorted(set(untouched) | (set(thin) & dead_set))
    log(f"    touched by NO candidate on any leg: {untouched or 'none'}")
    log(f"    touched by < 1 % of candidates:     {thin or 'none'}")
    log(f"    the hit-chi2 Hessian is blind to:   "
        f"{[nm for nm in dead] or 'none'}")
    if frozen:
        log("    NOTHING in this card constrains: " + " ".join(frozen))
        log("    -> add to the fit:  --freezeParameters " + " ".join(frozen))
    else:
        log("    every material amount is constrained by at least one term")
    # leverage, which is what says whether a CONSTRAINED group matters
    for t in tags:
        c = matctx.census[t]
        tot = c["lev"].sum() or 1.0
        order = np.argsort(-c["lev"])[:6]
        log(f"    {t}: top groups by share of sum_i max_tau |S| -- " + ", ".join(
            f"{gnames[g]} {100.*c['lev'][g]/tot:.1f} %" for g in order))
    return frozen


def rebuild_material(term, datasets, decl, matctx, path, idx, tag, log):
    """Re-instantiate a `make_card.build` MassCFTerm as a MaterialCFTerm.

    Everything the Z leg needs -- the `ZGammaLineshape` provider with its
    banded FSR fold and the acceptance, the floated 5-term `K(m)`, the
    truncation window and its classes, both corrections, the weights -- is
    expensive to build and easy to get subtly wrong a second time, so none of
    it is rebuilt. `MassCFTerm.config()` IS the constructor's argument list,
    and the kernel and background OBJECTS are carried over by reference; only
    the resolution parameterisation changes. That is what makes the k = 0
    comparison against the phase-2 term a gate on the resolution model alone.

    Returns the new term, the datasets it was built from (the flat exponents
    dropped, the group block added) and its declaration arrays.
    """
    from rabbit import unbinned

    args = matctx.args
    gkw, gdata, _ = matctx.block(path, idx, tag, log)
    datasets = dict(datasets)
    datasets.update(gdata)
    strip_flat_families(datasets, log)
    norm = norm_from_datasets(datasets)
    cfg = dict(term.config())
    for k in ("kind", "families", "norm_fixed", "kernel", "background",
              "jac_params"):
        cfg.pop(k, None)
    legacy = ([{"name": "hit", "param": "k_hit", "kind": "gauss"}]
              if args.legacy_families else [])
    old = {nm: (float(decl["param_defaults"][i]),
                float(decl["param_prior_sigmas"][i]),
                float(decl["param_prior_means"][i]),
                int(decl["param_is_poi"][i]))
           for i, nm in enumerate(term.param_names)}
    new = unbinned.MaterialCFTerm(
        term.name,
        sigma=datasets["sigma"], mobs=datasets["mobs"],
        tgrid=datasets["tgrid"], families=legacy,
        vgf=datasets.get("vgf"),
        weights=datasets.get("weights"), a_res=datasets.get("a_res"),
        jensen_s2=datasets.get("jensen_s2"),
        corr_mass=datasets.get("corr_mass"),
        phik=None,
        norm=(None if norm is None else MaterialContext.fixed_norm(norm)),
        kernel=term.kernel, background=term.background,
        **cfg, **gkw)
    keep = set(new.param_names)
    dcl = {nm: row for nm, row in old.items() if nm in keep}
    dcl.update(matctx.declarations(gkw["group_params"], gkw["hit_params"]))
    for f in legacy:
        dcl[f["param"]] = (1.0, np.nan, 1.0, 0)
    gone = [nm for nm in term.param_names if nm not in keep]
    log(f"    {tag}: MassCFTerm -> MaterialCFTerm; the resolution knobs {gone} "
        f"are REPLACED by {len(gkw['group_params'])} material amounts + "
        f"{len(gkw['hit_params'])} hit classes")
    return new, datasets, unbinned.declare_params(new, dcl)


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


def build_jpsi(args, log=print, matctx=None):
    """The J/psi mass term: delta kernel at MJPSI, NO scale parameter.

    Returns (term, datasets, decl, idx, d) -- `idx` and the open cache so the
    caller can slice the D block on exactly the same candidates.

    With ``matctx`` the term is a :class:`MaterialCFTerm` over the parmtype-15
    amounts and the hit classes instead of the four ``k_*`` knobs; everything
    else -- the selection, the weights, the two corrections, the truncation
    window -- is bit-identical, which is what makes the k = 0 comparison a
    gate rather than a coincidence.
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
        "--corr-form", args.corr_form,
        "--corr-clip", "0",
    ] + _selargv(args, "jpsi"))

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

    common = dict(
        sigma=sigma, mobs=mobs, tgrid=tgrid,
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
        corr_form=args.corr_form,
        norm_window=(lo, hi), norm_tpoints=jargs.norm_tpoints,
        upsample=args.fit_upsample_jpsi,
        # The positivity floor must be passed here too, not left at
        # MassCFTerm's own 1e-9: two of 3 000 000 J/psi candidates have a
        # Fourier-reconstructed density that undershoots to ~-6e-4, and 1e-9
        # underflows that to exactly zero, taking the joint NLL to inf at the
        # start point and making the gradient non-finite in most components.
        floor_scale=jargs.floor_scale,
        chunk=args.chunk, channel="jpsi")

    if matctx is None:
        term = unbinned.MassCFTerm(
            "jpsi",
            families=[dict(f, **arrays.get(f["name"], {})) for f in families],
            norm=norm, **common)
        decl = unbinned.declare_params(
            term, {f["param"]: (1.0, np.nan, 1.0, 0) for f in families})
    else:
        gkw, gdata, _ = matctx.block(args.jpsi_pairs, idx, "jpsi", log)
        datasets.update(gdata)
        strip_flat_families(datasets, log)
        legacy = ([{"name": "hit", "param": "k_hit", "kind": "gauss"}]
                  if args.legacy_families else [])
        term = unbinned.MaterialCFTerm(
            "jpsi", families=legacy,
            norm=MaterialContext.fixed_norm(norm), **common, **gkw)
        dcl = matctx.declarations(gkw["group_params"], gkw["hit_params"])
        for f in legacy:
            dcl[f["param"]] = (1.0, np.nan, 1.0, 0)
        decl = unbinned.declare_params(term, dcl)
    log(f"  parameters {list(term.param_names)}")
    info = {"n": n, "n_cache": int(len(d["z"])), "window": [lo, hi],
            "weights_info": winfo}
    return term, datasets, decl, idx, d, info


# ---------------------------------------------------------------------------
# card assembly
# ---------------------------------------------------------------------------
def attach_jac(cfg, datasets, decl, params, jac, names, prior_sigmas, poi_set,
               allow_shared=False):
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
    shared = [nm for nm in names if nm in params]
    if shared and not allow_shared:
        raise SystemExit(
            "a calibration parameter name collides with one of the mass "
            f"term's own parameters: {sorted(shared)}")
    prior = {nm: prior_sigmas[i] for i, nm in enumerate(names)}
    params = list(params) + new
    decl = {
        "param_defaults": np.concatenate(
            [decl["param_defaults"], np.zeros(len(new))]),
        "param_prior_sigmas": np.concatenate(
            [decl["param_prior_sigmas"],
             np.array([prior[nm] for nm in new], np.float64)]),
        "param_prior_means": np.concatenate(
            [decl["param_prior_means"], np.zeros(len(new))]),
        "param_is_poi": np.concatenate(
            [np.asarray(decl["param_is_poi"], np.int8),
             np.array([1 if nm in poi_set else 0 for nm in new], np.int8)]),
    }
    # A parameter the TERM already declares (the 42 material amounts, which a
    # MaterialCFTerm lists as its own) must still carry the CALIBRATION
    # declaration -- same prior, same POI flag as the bundle and as the other
    # leg -- or `merge_declarations` refuses the card. Overwrite rather than
    # trust the term's default row, and say how many were touched.
    if shared:
        pos = {nm: i for i, nm in enumerate(params)}
        for nm in shared:
            i = pos[nm]
            decl["param_defaults"][i] = 0.0
            decl["param_prior_sigmas"][i] = prior[nm]
            decl["param_prior_means"][i] = 0.0
            decl["param_is_poi"][i] = 1 if nm in poi_set else 0
    return cfg, datasets, decl, params, shared


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

    matctx = None
    if args.material:
        if not args.jpsi_pairs:
            raise SystemExit(
                "--material with no --jpsi-pairs: the material amounts would "
                "then be constrained by the Z leg and the quadratic term "
                "alone, which is not the phase-3 configuration. Give both "
                "group caches.")
        matctx = MaterialContext(args, names, parmtype, subidx, pscale,
                                 prior_sigmas, poi_set)
        log(f"  --material: {len(matctx.matcol)} parmtype-15 amounts are "
            "SHARED between the quadratic curvature, the mass-term widths and "
            f"the mass-term means; A(k) = {args.amount_mode}, "
            f"H(eps) = {args.hit_mode}")

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
        if args.material:
            bad = [nm for nm in names
                   if inject[byname[nm]] and nm.startswith("material_")]
            if bad:
                raise SystemExit(
                    "--material with --inject on a material amount "
                    f"({bad}): a TRUE extra amount of material moves the mass "
                    "mean (which this file injects through D) AND the "
                    "resolution EXPONENTS of that group by A(dtheta) (which "
                    "it does not). Injecting only the mean would look like a "
                    "closure failure of the width. Port "
                    "`matres/make_material_card.py`'s exponent scaling first, "
                    "or inject a field mode, which enters the mass terms "
                    "through the mean alone and is unaffected.")

    # -- 3. the mass terms --------------------------------------------------
    # the parameter maps FIRST: an hour of term building is wasted if the D
    # columns turn out to mean something other than the K columns
    for tag, path in (("jpsi", args.jpsi_pairs), ("z", args.z_pairs)):
        if path:
            check_jac_map(np.load(path, allow_pickle=True), cat, tag)
    log(f"  parameter map: the D block of every pairs cache matches the "
        f"quadratic extraction on all {nfit} (fitidx, parmtype, subidx)")

    # `f_ang`: the Jensen s^2 must be TRUTH-FREE on both legs, i.e. read per
    # candidate from `Jpsi_covrefmom`. A cache built from a production without
    # that branch falls back to the scalar --jpsi-fang; say explicitly which
    # source each leg of THIS card uses rather than leaving it in the middle of
    # make_card's selection log.
    fangsrc = {}
    for tag, path in (("jpsi", args.jpsi_pairs), ("z", args.z_pairs)):
        if not path:
            continue
        has = "fang" in set(np.load(path, allow_pickle=False).files)
        fangsrc[tag] = "per-candidate Jpsi_fang" if has else (
            f"the SCALAR --jpsi-fang {args.jpsi_fang:g}" if tag == "jpsi"
            else "none (s^2 unscaled)")
        log(f"  jensen f_ang, {tag} leg: {fangsrc[tag]}"
            + ("" if has else "  <-- NOT truth-free from the candidate"))

    entries = []          # (name, cfg, params, datasets, decl)
    jacinfo = {}

    if args.jpsi_pairs:
        log("--- J/psi term ---")
        jterm, jdata, jdecl, jidx, jd, jinfo = build_jpsi(args, log, matctx)
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
        cfg, jdata, jdecl, jparams, jshared = attach_jac(
            jterm.config(), jdata, jdecl, jterm.param_names, jac, names,
            prior_sigmas, poi_set, allow_shared=args.material)
        entries.append(("jpsi", cfg, jparams, jdata, jdecl))
        log(f"  J/psi term: {jinfo['n']} candidates, {len(jparams)} parameters"
            + (f" ({len(jshared)} of them SHARED with the term's own material "
               "amounts)" if jshared else ""))
        del jterm
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
             "--corr-form", args.corr_form,
             "--corr-clip", "0",
             "--fit-upsample", str(args.fit_upsample_z)]
    zargv += _selargv(args, "z")
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

    if matctx is not None:
        zterm, zdata, zdecl = rebuild_material(
            zterm, zdata, zdecl, matctx, args.z_pairs, zidx, "z", log)
    jac, Dz, zstat = sparse_jac(Dz, args.jac_prune, "z", log)
    jacinfo["z"] = zstat
    if args.inject and not args.inject_quad_only:
        shift = Dz @ inject
        zdata["mobs"] = zdata["mobs"] + shift
        log(f"  injected into the Z means: mean {shift.mean()*1e3:+.5f} MeV, "
            f"rms {shift.std()*1e3:.5f} MeV")
    cfg, zdata, zdecl, zparams, zshared = attach_jac(
        zterm.config(), zdata, zdecl, zterm.param_names, jac, names,
        prior_sigmas, poi_set, allow_shared=args.material)
    entries.append(("zmass", cfg, zparams, zdata, zdecl))
    log(f"  Z term: {zinfo['n']} candidates, {len(zparams)} parameters"
        + (f" ({len(zshared)} of them SHARED with the term's own material "
           "amounts)" if zshared else ""))
    del zterm

    # -- 4. the injection into the quadratic term ---------------------------
    if args.inject and not args.inject_mass_only:
        G = G - K @ inject
        log("  quadratic term: G -> G - K dtheta")

    frozen = report_census(matctx, names, dead, poi_set, log) if matctx else []
    if frozen:
        log("  the card does NOT freeze them: which parameters a fit floats is "
            "a physics decision and belongs in the fit command.")

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
            "fang_source": fangsrc,
            "material": bool(args.material),
            "amount_mode": args.amount_mode,
            "hit_mode": args.hit_mode,
            "group_prune": args.group_prune,
            "hit_prior": args.hit_prior,
            "legacy_families": bool(args.legacy_families),
            "unconstrained": frozen if args.material else [],
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
