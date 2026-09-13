#!/usr/bin/env python3
"""One streaming pass over a two-track CVH production producing everything the
unified calibration datacard needs, from the SAME candidates:

* the **quadratic hit-chi2 term**: ``G = sum_i gradv_i`` and
  ``K = sum_i H_i`` over a chosen global-parameter subset (parmtype 14 = the
  50 scalar-potential field modes, 15 = the material groups, 0-5 = alignment),
  reading either ``hesspackedv`` (MC productions) or the factored
  ``hessfactorv`` / ``nRank`` (data productions) -- plus, on a production run
  with ``exportVarianceGrads``, the separately shipped variance (log-det)
  block ``hessvaridxv`` / ``hessvarpackedv``, which ``hessfactorv`` does NOT
  contain and ``hesspackedv`` does. Same conventions and same
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
  per-candidate value scatters by ~6 % about an exact refit. That is zero-mean
  and averages away for aggregate quantities, but the ``D`` rows here are used
  *per candidate*, so it is a genuine per-candidate error on the mass response.
* For parmtype 15 (material groups) and 7 (per-module ``dxi``) the same note
  reports a *coherent* +35..53 % Jacobian bias, i.e. not just scatter. Treat
  the material columns of ``D`` as indicative until an exact export exists.
* ``gradchisqv`` is empty in every two-track file, so the censored-likelihood
  trimming of ``fit_global_grads.py --censor-cut`` is not available here.

Usage::

    python extract.py \\
      --files '/ceph/.../resolution_trackres_btojpsix_v3_260904f_m0/task_*/globalcor_*.root' \\
      --parmtypes 14 15 -j 24 -o runs/globalfit_btojpsix_260904f.npz
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import uproot

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(_HERE))

import prodfiles  # noqa: E402  (needs the parent directory on sys.path)
import selection  # noqa: E402  (the standard two-track selection)

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
#   charges.
MJPSI = 3.0969
IONI_SGN = -1.0
RAD_SGN = IONI_SGN
MASS_WINDOW = 0.35  # |m_gen - m_Jpsi| cut, as in build_pairs_tt


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--files",
        required=True,
        help="glob of globalcor_*.root files, a production directory, or "
        "@list.txt (see prodfiles.resolve). A glob naming stream 0 is widened "
        "to every stream of the task.",
    )
    p.add_argument(
        "--ntasks",
        type=int,
        default=0,
        help="use only the first N TASKS (0 = all). A task of a multi-stream "
        "production is globalcor_0..N-1.root and all of its streams are read; "
        "capping the FILE list instead would take 1/N of the intended tasks.",
    )
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
    # `--max-chi2-ndof` is the STANDARD SELECTION's (`resolution/selection.py`,
    # default 3.0).  It is not cosmetic here: on the B -> J/psi X productions
    # the median chi2/ndof is 0.95 but the tail reaches 5e8, and ~0.02 % of
    # candidates carry ~99.997 % of the summed chi2 -- so without it the global
    # gradient and Hessian are those of a handful of runaway fits.  Same role
    # as fit_global_grads.py --max-chi2-per-hit / --censor-cut.
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
        "--max-dEref-p",
        type=float,
        default=0.0,
        help="drop candidates whose WORSE leg has |dE_ref| / p above this "
        "(0 = no cut). The per-group material model is a MEAN-loss model and "
        "is biased above dE_ref/p ~ 0.03; 0.01 is the working point. "
        "`Mu{plus,minus}_dEref` is present in every two-track production "
        "(unlike `Mu*_maxfracloss`), so this cut is portable across "
        "production versions -- and it is the quantity itself, not a "
        "daughter-pT proxy for it. It costs ~0.1 %% of Z candidates and "
        "~22 %% of J/psi ones.",
    )
    p.add_argument(
        "--maxcand",
        type=int,
        default=0,
        help="stop after roughly this many selected candidates",
    )
    p.add_argument("-o", "--output", required=True)
    selection.add_args(p)
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
    # SANDWICH. `hess` is the model's own curvature K, and the covariance it
    # implies, 2 K^-1, is the truth only when the per-candidate model is
    # correct. The robust covariance is K^-1 J K^-1 with J = sum_i G_i G_i^T,
    # and J/(2K) is the factor by which a mis-specified model inflates the
    # error -- the one diagnostic that flags a parameter whose gradient is
    # carried by a handful of candidates. It is a 92x92 outer product per
    # candidate: free at this size, and impossible to recover afterwards.
    jout = np.zeros((nfit, nfit))
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
    want_deref = (args.max_dEref_p > 0.0
                  and all(f"Mu{q}_{b}" in keys
                          for q in ("plus", "minus")
                          for b in ("dEref", "pt", "eta")))
    if args.max_dEref_p > 0.0 and not want_deref:
        raise ValueError(
            f"{fname}: --max-dEref-p was asked for but the file has no "
            f"Mu*_dEref / Mu*_pt / Mu*_eta; a silently unapplied quality cut "
            f"is worse than no cut")
    if want_deref:
        want += [f"Mu{q}_{b}" for q in ("plus", "minus")
                 for b in ("dEref", "pt", "eta")]
    fmt = None
    if not args.no_grads:
        want.append("gradv")
        if "hessfactorv" in keys:
            fmt = "factored"
            want += ["hessfactorv", "nRank"]
            # THE VARIANCE (log-det) BLOCK.  With `exportVarianceGrads` the
            # two-track maker's Hessian is 2 J^T R J PLUS tr(dV_i R dV_j R)
            # over the variance parameters, and only the first is in
            # `hessfactorv` (the second is a Gram matrix whose rank IS the
            # number of variance parameters -- it does not compress, so it is
            # shipped as its own packed triangle).  Its presence is the flag:
            #     hess = B^T B + scatter(hessvarpackedv on hessvaridxv)
            if "hessvaridxv" in keys:
                want += ["hessvaridxv", "hessvarpackedv"]
        elif "hesspackedv" in keys:
            fmt = "packed"
            want.append("hesspackedv")
            # `hesspackedv` is COMPLETE -- the variance block is already in it
            # -- so it must NOT be added again.
        else:
            sys.exit(f"{fname}: neither hesspackedv nor hessfactorv present")
        # A file whose gradient carries the log-det term but whose Hessian
        # does not is an inconsistent (G, K) pair and would bias the fit; it
        # is much better to stop than to fit it.
        if (fmt == "factored" and "gradllv" in keys
                and "hessvaridxv" not in keys):
            gll = t["gradllv"].array(entry_stop=200, library="np")
            if any(len(x) for x in gll):
                sys.exit(
                    f"{fname}: gradllv is filled (the log-det term is in "
                    "gradv) but there is no hessvaridxv, so hessfactorv is "
                    "the MEAN Hessian only. Refusing to build an "
                    "inconsistent (G, K)."
                )
    # `Jpsi_jacMass` is the MASS term's input only.  A single-track production
    # has a perfectly good quadratic (hit-chi2) term -- gradv / hesspackedv --
    # and `--no-mass` must be able to accumulate it, so the guard belongs
    # inside the mass branch and not in front of it.
    if not args.no_mass and "Jpsi_jacMass" not in keys:
        sys.exit(
            f"{fname}: no Jpsi_jacMass branch -- this is not a two-track "
            "production with the contracted mass Jacobian"
        )
    if "Jpsi_jacMass" in keys:
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
    # the columns THE STANDARD TWO-TRACK SELECTION reads, where the
    # production has them (`resolution/selection.py`); each is optional.
    istwotrack = "Jpsi_jacMass" in keys
    if istwotrack:
        want += [b for b in sum(selection.ALIASES.values(), ()) if b in keys]
    want = list(dict.fromkeys(want))
    a = t.arrays(want, library="np")
    pt_all = _PARMTYPE

    nent = len(a["globalidxv"])
    chi2ok = None
    # the per-candidate reduced chi2, stored in the npz; the CUT on it is the
    # standard selection's (`resolution/selection.py`), which builds the same
    # quantity with the same convention
    rchi2 = selection.chi2ndof(a, set(a))
    for br, cut in (("gradmax", args.max_grad), ("hessmax", args.max_hess)):
        if cut > 0.0 and br in a:
            ok = np.abs(np.asarray(a[br], dtype=np.float64)) < cut
            chi2ok = ok if chi2ok is None else (chi2ok & ok)
    if want_deref:
        # dE_ref / p on the WORSE leg; p = pT cosh(eta) at the reference
        fr = None
        for q in ("plus", "minus"):
            pmag = (np.asarray(a[f"Mu{q}_pt"], dtype=np.float64)
                    * np.cosh(np.asarray(a[f"Mu{q}_eta"], dtype=np.float64)))
            r = np.abs(np.asarray(a[f"Mu{q}_dEref"], dtype=np.float64)) \
                / np.maximum(pmag, 1e-9)
            fr = r if fr is None else np.maximum(fr, r)
        ok = fr < args.max_dEref_p
        chi2ok = ok if chi2ok is None else (chi2ok & ok)
    # THE STANDARD SELECTION, which owns `--max-chi2-ndof` as well.  Its
    # two-track cuts have no column on a single-track production and are
    # reported absent there; the chi2 cut applies to both.
    m, stdsumm = selection.standard(a, args, n=nent)
    chi2ok = m if chi2ok is None else (chi2ok & m)
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
            gc = np.zeros(nfit)
            gc[fi[keep]] = g[keep]
            jout += np.outer(gc, gc)
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
                # ... plus the variance (log-det) block, which hessfactorv
                # does not contain. Scatter its packed triangle onto the
                # candidate's own column numbering first, then project.
                if "hessvaridxv" in a:
                    vi = np.asarray(a["hessvaridxv"][ic], dtype=np.int64)
                    if len(vi):
                        vp = np.asarray(
                            a["hessvarpackedv"][ic], dtype=np.float64
                        )
                        m = len(vi)
                        HV = np.zeros((m, m))
                        HV[triu_index(m)] = vp
                        HV = HV + HV.T - np.diag(np.diag(HV))
                        # rows of vi that survive the parmtype selection, and
                        # where they land in kk
                        pos = {int(c): j for j, c in enumerate(kk)}
                        sel = [(j, pos[int(c)]) for j, c in enumerate(vi)
                               if int(c) in pos]
                        if sel:
                            js = np.asarray([x[0] for x in sel])
                            ks = np.asarray([x[1] for x in sel])
                            Hkk[np.ix_(ks, ks)] += HV[np.ix_(js, js)]
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
            int(want_rad), nchi2cut, jout, stdsumm)


def main():
    args = parse_args()
    files = prodfiles.resolve(args.files, args.ntasks, logger=log)
    if not files:
        sys.exit(f"no files match {args.files}")
    st = prodfiles.last_stats()
    log(f"{st.get('ntasks_used', len(files))} tasks / {len(files)} files, "
        f"{args.jobs} workers, parmtypes {args.parmtypes}")

    if not args.no_mass:
        load_cf_primitives()
    # ONE runtree. Every stream file of every task carries a byte-identical
    # copy of the 13 MB parameter map, so the catalog is built from a single
    # file -- `files[0]` is the first existing stream of the first usable task.
    cat = build_catalog(files[0], args.parmtypes)
    nfit = len(cat["fitidx"])
    counts = {
        int(pt): int((cat["parmtype"] == pt).sum())
        for pt in sorted(set(cat["parmtype"].tolist()))
    }
    log(f"catalog: {cat['nglobal']} global params, floating {nfit} {counts}")

    grad = np.zeros(nfit)
    jsand = np.zeros((nfit, nfit))
    hess = np.zeros((nfit, nfit))
    chunks = {}
    nsel = ndrop = nquad = nchi2cut = 0
    fmts = set()
    rad = set()
    selsumms = []
    t0 = time.time()
    with Pool(
        args.jobs,
        initializer=_init,
        initargs=(cat["g2f"], cat["full_parmtype"], args),
    ) as pool:
        for i, r in enumerate(pool.imap(process_file, files)):
            if r is None:
                continue
            fname, res, g, h, ns, nd, nq, fmt, wr, ncut, jj, ss = r
            selsumms.append(ss)
            grad += g
            hess += h
            jsand += jj
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
    _sel = selection.merge(selsumms)
    if _sel is not None:
        _sel.log(log)
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
        "max_dEref_p": np.array(args.max_dEref_p),
        # J = sum_i G_i G_i^T; the sandwich covariance is K^-1 J K^-1, which
        # equals the model covariance 2 K^-1 exactly when J = 2 K.
        "jsand": jsand,
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
