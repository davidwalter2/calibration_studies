#!/usr/bin/env python3
"""One streaming pass over a two-track CVH production producing everything the
unified calibration datacard needs, from the SAME candidates:

* the **quadratic hit-chi2 term**: ``G = sum_i gradv_i`` and
  ``K = sum_i H_i`` over a chosen global-parameter subset (parmtype 14 = the
  50 scalar-potential field modes, 15 = the material groups, 0-5 = alignment),
  reading either ``hesspackedv`` (MC productions) or the factored
  ``hessfactorv`` / ``nRank`` (data productions). Same conventions and same
  numbers as ``global_corrections/fit_global_grads.py``: **chi2** units, so
  ``theta = -K^-1 G`` and ``cov = 2 K^-1``.

* the **per-candidate mass-likelihood inputs**: ``m0`` (``Jpsi_mass``),
  ``mgen`` (``Jpsigen_mass``), ``sigma`` (``Jpsi_sigmamass``), the Gaussian
  variance fraction ``vgf`` and the tabulated resolution-CF exponents of the
  MS / ionization / radiative families -- the identical construction of
  ``cf_mass_likelihood.build_pairs_tt`` (same sign conventions ``IONI_SGN`` /
  ``RAD_SGN``, same ``ioni_sq2`` normalization, same ``TG`` grid).

* the **per-candidate Jacobian rows** ``D_i = dm_i/dtheta``, taken straight
  from the two-track maker's ``Jpsi_jacMass`` branch (1 x nParms, contracted
  on the same ``globalidxv`` columns) rather than reassembled from
  ``jacrefv`` and a numerical mass Jacobian as ``cf_global_masslik.py`` does
  for single-track trees. ``jacrefv`` does not exist in two-track output.

* the **FSR kernel** samples ``dm = Jpsigen_mass - m_ref`` of the selected
  candidates, so the datacard's kernel matches its own sample.

Output: a single ``.npz`` consumed by ``make_global_term.py``.

Caveats
-------
* The exported Jacobians are built with the *expected* (Fisher) curvature; the
  per-candidate value scatters by ~6 % about an exact refit
  (``Documents/Resolution/NOTES_EXPORTS.md`` sec. 8 item 4). That is zero-mean
  and averages away for aggregate quantities, but the ``D`` rows here are used
  *per candidate*, so it is a genuine per-candidate error on the mass response.
* For parmtype 15 (material groups) and 7 (per-module ``dxi``) the same note
  reports a *coherent* +35..53 % Jacobian bias, i.e. not just scatter. Treat
  the material columns of ``D`` as indicative until an exact export exists.
* ``gradchisqv`` is empty in every two-track file, so the censored-likelihood
  trimming of ``fit_global_grads.py --censor-cut`` is not available here.

Usage::

    python extract.py \\
      --files '/ceph/.../resolution_trackres_btojpsix_v3_260904f_m0/task_*/globalcor_0.root' \\
      --parmtypes 14 15 -j 24 -o runs/globalfit_btojpsix_260904f.npz
"""

import argparse
import glob
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import uproot

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(_HERE))

# The CF primitives live in the parent directory and are expensive to import
# (cf_ms_exact builds its electron tables at import time: ~6 min), so they are
# pulled in lazily -- a --no-mass run that only accumulates gradv/hesspackedv
# must not pay for them. Pool workers are forked after load_cf_primitives()
# has run in the parent, so the cost is paid once.
TG = None
ioni_sq2 = ioni_step_exponent = ms_step_exponent = None
cf_brems_exact = None


def load_cf_primitives():
    """Import the resolution-CF primitives into module globals."""
    global TG, ioni_sq2, ioni_step_exponent, ms_step_exponent, cf_brems_exact
    if TG is not None:
        return
    t0 = time.time()
    import cf_brems_exact as _brems
    from cf_track_resolution import TG as _TG
    from cf_track_resolution import ioni_sq2 as _isq
    from cf_track_resolution import ioni_step_exponent as _ise
    from cf_track_resolution import ms_step_exponent as _mse

    TG, ioni_sq2, ioni_step_exponent, ms_step_exponent = _TG, _isq, _ise, _mse
    cf_brems_exact = _brems
    print(f"CF primitives imported in {time.time()-t0:.0f} s", flush=True)

# Physics conventions, identical to cf_mass_likelihood (imported by value so
# that the 24 worker processes do not have to pull in matplotlib):
#   IONI_SGN = RAD_SGN = -1  -- an energy loss on EITHER muon can only lower
#   the pair mass, so the mass functional of both blocks is negative for both
#   charges (see cf_mass_likelihood.py lines ~79-104 for the derivation).
MJPSI = 3.0969
IONI_SGN = -1.0
RAD_SGN = IONI_SGN
MASS_WINDOW = 0.35  # |m_gen - m_Jpsi| cut, as in build_pairs_tt


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--files", required=True, help="glob of globalcor_*.root files")
    p.add_argument("--ntasks", type=int, default=0, help="use only the first N files")
    p.add_argument(
        "--parmtypes",
        type=int,
        nargs="+",
        default=[14, 15],
        help="global parameter types to keep (14 = field modes, 15 = material)",
    )
    p.add_argument("-j", "--jobs", type=int, default=16, help="worker processes")
    p.add_argument(
        "--no-mass",
        action="store_true",
        help="accumulate the quadratic term only (skip the expensive CF build)",
    )
    p.add_argument(
        "--no-grads",
        action="store_true",
        help="build the mass inputs only (skip gradv / Hessian)",
    )
    p.add_argument(
        "--max-chi2-ndof",
        type=float,
        default=0.0,
        help="drop candidates with chisqval/ndof above this (0 = no cut) from "
        "BOTH the quadratic accumulation and the mass term. This is not "
        "cosmetic: on the 260904f productions the median chi2/ndof is 0.95 but "
        "the tail reaches 5e8, and ~0.02%% of candidates carry ~99.997%% of the "
        "summed chi2 -- so without a cut the global gradient and Hessian are "
        "the gradient and Hessian of a handful of runaway fits. Same role as "
        "fit_global_grads.py --max-chi2-per-hit / --censor-cut.",
    )
    p.add_argument(
        "--max-grad",
        type=float,
        default=0.0,
        help="drop candidates whose gradmax exceeds this (0 = no cut). "
        "Complements --max-chi2-ndof: a runaway fit can have an acceptable "
        "chi2 but a pathological Jacobian, and one such candidate is enough to "
        "leave a spurious 1e20 eigenvalue and a rank deficit in the summed "
        "Hessian (seen on the gun sample after the chi2 cut).",
    )
    p.add_argument(
        "--max-hess",
        type=float,
        default=0.0,
        help="drop candidates whose hessmax exceeds this (0 = no cut)",
    )
    p.add_argument(
        "--maxcand",
        type=int,
        default=0,
        help="stop after roughly this many selected candidates",
    )
    p.add_argument("-o", "--output", required=True)
    return p.parse_args()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
def build_catalog(fname, parmtypes):
    rt = uproot.open(fname)["runtree"]
    parmtype = rt["parmtype"].array(library="np")
    rawdetid = rt["rawdetid"].array(library="np")
    full_parmtype = parmtype
    nglobal = len(parmtype)
    fitidx = np.where(np.isin(parmtype, parmtypes))[0]
    g2f = -np.ones(nglobal, dtype=np.int64)
    g2f[fitidx] = np.arange(len(fitidx))
    return {
        "nglobal": nglobal,
        "fitidx": fitidx,
        "parmtype": parmtype[fitidx],
        # for parmtype 14/15 rawdetid IS the mode / group index
        "subidx": rawdetid[fitidx].astype(np.int64),
        "g2f": g2f,
        "full_parmtype": full_parmtype,
    }


_TRIU = {}


def triu_index(n):
    if n not in _TRIU:
        _TRIU[n] = np.triu_indices(n)
    return _TRIU[n]


# ---------------------------------------------------------------------------
_G2F = None
_ARGS = None
_PARMTYPE = None


def _init(g2f, parmtype, args):
    global _G2F, _ARGS, _PARMTYPE
    _G2F = g2f
    _PARMTYPE = parmtype
    _ARGS = args


def process_file(fname):
    """Per-file worker: returns (per-candidate arrays, grad, hess, counters)."""
    g2f = _G2F
    args = _ARGS
    nfit = int((g2f >= 0).sum())
    grad = np.zeros(nfit)
    hess = np.zeros((nfit, nfit))
    # chi2ndof is stored per selected candidate so a trimming can also be
    # applied later, at card-writing time, without re-extracting.
    out = {k: [] for k in ("m0", "mgen", "sigma", "vgf", "D", "chi2ndof")}
    for k in ("Sms", "Sio_re", "Sio_im", "Srad_re", "Srad_im"):
        out[k] = []
    nsel = ndrop = nquad = nchi2cut = 0
    try:
        f = uproot.open(fname)
        if "tree" not in f:
            return None
        t = f["tree"]
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: cannot open {fname} ({type(e).__name__}) -- skipping")
        return None
    if t.num_entries == 0:
        return None
    keys = set(t.keys())

    want = ["globalidxv", "chisqval", "ndof"]
    if args.max_grad > 0.0 and "gradmax" in keys:
        want.append("gradmax")
    if args.max_hess > 0.0 and "hessmax" in keys:
        want.append("hessmax")
    fmt = None
    if not args.no_grads:
        want.append("gradv")
        if "hessfactorv" in keys:
            fmt = "factored"
            want += ["hessfactorv", "nRank"]
        elif "hesspackedv" in keys:
            fmt = "packed"
            want.append("hesspackedv")
        else:
            sys.exit(f"{fname}: neither hesspackedv nor hessfactorv present")
    if "Jpsi_jacMass" not in keys:
        sys.exit(
            f"{fname}: no Jpsi_jacMass branch -- this is not a two-track "
            "production with the contracted mass Jacobian"
        )
    want.append("Jpsi_jacMass")
    want_rad = False
    if not args.no_mass:
        want += [
            "Jpsi_mass",
            "Jpsi_sigmamass",
            "Jpsigen_mass",
            "resinfv",
            "resinfvarv",
            "resinfcov",
            "reseigidx",
            "msmoliidx",
            "msmoliv",
            "ioniurbanidx",
            "ioniurbanv",
        ]
        want += [b for b in ("ioniqscaleidx", "ioniqscalev") if b in keys]
        rb = [
            b
            for b in ("radstepidx", "radstepv", "radstepspecv", "radvgrid")
            if b in keys
        ]
        if len(rb) not in (0, 4):
            raise ValueError(f"{fname}: partial radiative export {rb}")
        want_rad = len(rb) == 4
        want += rb
    a = t.arrays(want, library="np")
    pt_all = _PARMTYPE

    nent = len(a["globalidxv"])
    rchi2 = np.asarray(a["chisqval"], dtype=np.float64) / np.maximum(
        np.asarray(a["ndof"], dtype=np.float64), 1.0
    )
    chi2ok = rchi2 < args.max_chi2_ndof if args.max_chi2_ndof > 0.0 else None
    for br, cut in (("gradmax", args.max_grad), ("hessmax", args.max_hess)):
        if cut > 0.0 and br in a:
            ok = np.abs(np.asarray(a[br], dtype=np.float64)) < cut
            chi2ok = ok if chi2ok is None else (chi2ok & ok)
    for ic in range(nent):
        if chi2ok is not None and not chi2ok[ic]:
            nchi2cut += 1
            continue
        gi = np.asarray(a["globalidxv"][ic])
        fi = g2f[gi]
        keep = fi >= 0
        kk = np.where(keep)[0]

        # --- quadratic term: every candidate contributes ------------------
        if not args.no_grads and keep.any():
            nquad += 1
            g = np.asarray(a["gradv"][ic], dtype=np.float64)
            # globalidxv is deduped, so fi[keep] has no repeats and plain
            # fancy-index += is exact (and ~10x faster than np.add.at, which
            # is only needed when indices can repeat).
            grad[fi[keep]] += g[keep]
            n = len(gi)
            if fmt == "packed":
                h = np.asarray(a["hesspackedv"][ic], dtype=np.float64)
                H = np.zeros((n, n))
                iu = triu_index(n)
                H[iu] = h
                H = H + H.T - np.diag(np.diag(H))
                Hkk = H[np.ix_(kk, kk)]
            else:
                nrank = int(a["nRank"][ic])
                B = np.asarray(a["hessfactorv"][ic], dtype=np.float64).reshape(
                    nrank, n
                )
                Bk = B[:, kk]
                Hkk = Bk.T @ Bk
            fk = fi[kk]
            hess[np.ix_(fk, fk)] += Hkk

        if args.no_mass:
            continue

        # --- mass-likelihood selection (identical to build_pairs_tt) ------
        sig = float(a["Jpsi_sigmamass"][ic])
        mg = float(a["Jpsigen_mass"][ic])
        if not (np.isfinite(sig) and sig > 0.0) or abs(mg - MJPSI) > MASS_WINDOW:
            ndrop += 1
            continue
        rgi = np.asarray(a["reseigidx"][ic])
        if not len(rgi):
            ndrop += 1
            continue
        vb = np.asarray(a["resinfvarv"][ic], dtype=np.float64)
        uw = np.asarray(a["resinfv"][ic], dtype=np.float64).reshape(-1, 5)
        fam = pt_all[rgi]
        cov = float(a["resinfcov"][ic])
        vg = sig * sig - cov
        uvm = np.asarray(a["msmoliv"][ic], dtype=np.float64)
        uim = np.asarray(a["msmoliidx"][ic])
        uvm = (
            uvm.reshape(-1, len(uvm) // max(len(uim), 1))
            if len(uim)
            else uvm.reshape(0, 8)
        )
        uvi = np.asarray(a["ioniurbanv"][ic], dtype=np.float64)
        uii = np.asarray(a["ioniurbanidx"][ic])
        uvi = (
            uvi.reshape(-1, len(uvi) // max(len(uii), 1))
            if len(uii)
            else uvi.reshape(0, 11)
        )
        qsi = np.asarray(a["ioniqscaleidx"][ic]) if "ioniqscaleidx" in a else None
        qsv = (
            np.asarray(a["ioniqscalev"][ic], dtype=np.float64).reshape(-1, 2)
            if "ioniqscalev" in a
            else None
        )
        if want_rad:
            ridx = np.asarray(a["radstepidx"][ic])
            rrec = np.asarray(a["radstepv"][ic], dtype=np.float64).reshape(
                -1, cf_brems_exact.RADV_STRIDE
            )
            rspc = np.asarray(a["radstepspecv"][ic], dtype=np.float64).reshape(
                -1, 2 * cf_brems_exact.NRADV
            )
            rvg = np.asarray(a["radvgrid"][ic], dtype=np.float64)

        Sms = np.zeros(len(TG))
        Sio = np.zeros(len(TG), dtype=np.complex128)
        Srad = np.zeros(len(TG), dtype=np.complex128)
        ok = True
        for famcode, (uidx, uv) in ((10, (uim, uvm)), (11, (uii, uvi))):
            sel = fam == famcode
            for gg in np.unique(rgi[sel]):
                m = sel & (rgi == gg)
                vpool = vb[m].sum()
                if vpool <= 0.0:
                    continue
                steps = uv[uidx == gg]
                if not len(steps):
                    ok = False
                    break
                if famcode == 10:
                    sq2 = steps[:, 5].sum()
                else:
                    sq2 = ioni_sq2(steps, qsv[qsi == gg] if qsv is not None else None)
                if sq2 <= 0.0:
                    continue
                sgn = IONI_SGN if famcode == 11 else (np.sign(uw[m].sum()) or 1.0)
                if famcode == 10:
                    Sms += ms_step_exponent(steps, np.sqrt(vpool / sq2) / sig, TG)
                else:
                    wsc = np.sqrt(vpool / sq2) / sig
                    Sio += ioni_step_exponent(steps, sgn * wsc, TG)
                    if want_rad:
                        rm = ridx == gg
                        nrs = int(rm.sum())
                        if nrs:
                            Srad += cf_brems_exact.rad_exponent(
                                TG,
                                rrec[rm],
                                rspc[rm],
                                rvg,
                                weights=np.full(nrs, RAD_SGN * wsc),
                            )
            if not ok:
                break
        if not ok:
            ndrop += 1
            continue

        # --- D row: dm/dtheta on the fit block -----------------------------
        D = np.zeros(nfit)
        jm = np.asarray(a["Jpsi_jacMass"][ic], dtype=np.float64)
        if len(jm) != len(gi):
            raise ValueError(
                f"{fname}: Jpsi_jacMass has {len(jm)} entries but globalidxv "
                f"has {len(gi)}"
            )
        np.add.at(D, fi[keep], jm[keep])

        out["chi2ndof"].append(float(rchi2[ic]))
        out["m0"].append(float(a["Jpsi_mass"][ic]))
        out["mgen"].append(mg)
        out["sigma"].append(sig)
        out["vgf"].append(vg / (sig * sig))
        out["Sms"].append(Sms.astype(np.float32))
        out["Sio_re"].append(Sio.real.astype(np.float32))
        out["Sio_im"].append(Sio.imag.astype(np.float32))
        out["Srad_re"].append(Srad.real.astype(np.float32))
        out["Srad_im"].append(Srad.imag.astype(np.float32))
        out["D"].append(D.astype(np.float32))
        nsel += 1

    res = {k: (np.array(v) if len(v) else None) for k, v in out.items()}
    return (fname, res, grad, hess, nsel, ndrop, nquad, (fmt or "none"),
            int(want_rad), nchi2cut)


def main():
    args = parse_args()
    files = sorted(glob.glob(args.files))
    if not files:
        sys.exit(f"no files match {args.files}")
    if args.ntasks:
        files = files[: args.ntasks]
    log(f"{len(files)} files, {args.jobs} workers, parmtypes {args.parmtypes}")

    if not args.no_mass:
        load_cf_primitives()
    cat = build_catalog(files[0], args.parmtypes)
    nfit = len(cat["fitidx"])
    counts = {
        int(pt): int((cat["parmtype"] == pt).sum())
        for pt in sorted(set(cat["parmtype"].tolist()))
    }
    log(f"catalog: {cat['nglobal']} global params, floating {nfit} {counts}")

    grad = np.zeros(nfit)
    hess = np.zeros((nfit, nfit))
    chunks = {}
    nsel = ndrop = nquad = nchi2cut = 0
    fmts = set()
    rad = set()
    t0 = time.time()
    with Pool(
        args.jobs,
        initializer=_init,
        initargs=(cat["g2f"], cat["full_parmtype"], args),
    ) as pool:
        for i, r in enumerate(pool.imap(process_file, files)):
            if r is None:
                continue
            fname, res, g, h, ns, nd, nq, fmt, wr, ncut = r
            grad += g
            hess += h
            chunks[fname] = res
            nsel += ns
            ndrop += nd
            nquad += nq
            nchi2cut += ncut
            fmts.add(fmt)
            rad.add(wr)
            if (i + 1) % 5 == 0 or i + 1 == len(files):
                log(
                    f"  {i+1}/{len(files)} files, {nsel} candidates selected "
                    f"({ndrop} dropped), {nquad} in the quadratic term, "
                    f"{time.time()-t0:.0f} s"
                )
            if args.maxcand and nsel >= args.maxcand:
                log(f"reached --maxcand {args.maxcand}, stopping")
                pool.terminate()
                break
    if len(rad) > 1:
        sys.exit(
            "the radiative export is present in some input files and not in "
            "others; a cache mixing the two would carry the term for part of "
            "the sample only"
        )
    log(
        f"done in {time.time()-t0:.0f} s: {nsel} candidates, {nquad} in the "
        f"quadratic term, {nchi2cut} removed by the outlier cuts "
        f"(chi2/ndof < {args.max_chi2_ndof:g}, gradmax < {args.max_grad:g}, "
        f"hessmax < {args.max_hess:g}), Hessian format {sorted(fmts)}"
    )

    payload = {
        "grad": grad,
        "hess": hess,
        "fitidx": cat["fitidx"],
        "parmtype": cat["parmtype"],
        "subidx": cat["subidx"],
        "g2f": cat["g2f"],
        "nglobal": np.array(cat["nglobal"]),
        "ncand_quadratic": np.array(nquad),
        "ncand_chi2cut": np.array(nchi2cut),
        "max_chi2_ndof": np.array(args.max_chi2_ndof),
        "max_grad": np.array(args.max_grad),
        "max_hess": np.array(args.max_hess),
        "nfiles": np.array(len(chunks)),
        "tgrid": TG if TG is not None else np.zeros(0),
        "mref": np.array(MJPSI),
        "ioni_sign_fixed": np.array(1),
        "rad_model": np.array(int(rad.pop()) if rad else 0),
        "hess_format": np.array(sorted(fmts)[0] if fmts else "none"),
    }
    if not args.no_mass and nsel:
        keys = ("m0", "mgen", "sigma", "vgf", "chi2ndof", "Sms", "Sio_re",
                "Sio_im", "Srad_re", "Srad_im", "D")
        order = [f for f in files if f in chunks]
        for k in keys:
            parts = [chunks[f][k] for f in order if chunks[f][k] is not None]
            payload[k] = np.concatenate(parts) if parts else np.zeros(0)
        # the FSR kernel of this very sample
        payload["dm"] = (payload["mgen"] - MJPSI).astype(np.float64)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    np.savez(args.output, **payload)
    log(
        f"wrote {args.output} "
        f"({os.path.getsize(args.output)/1e9:.2f} GB, {nsel} candidates, "
        f"{nfit} parameters)"
    )


if __name__ == "__main__":
    main()
