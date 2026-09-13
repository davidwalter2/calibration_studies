#!/usr/bin/env python3
"""Per-MATERIAL-GROUP and per-HIT-CLASS resolution-CF inputs for the FIVE
whitened components of the track-parameter residual vector.

This is ``matres/extract_groups.py --functional qop`` generalized from ONE
functional (the fitted q/p) to the whole reference state.

The residual vector and its influence
-------------------------------------
``ResidualGlobalCorrectionMakerG4e`` exports, per resolution block ``b``,

    resinfbv[b] = B_b = W5[b]^T dV_b^{1/2}        (5 x 5, row major, padded)
    W5          = Vinv F C E5

i.e. the influence of the block's STANDARDIZED noise dofs on all five
reference-state parameters.  Hence, exactly,

    r  = refParms - genParms = sum_b B_b u_b ,     E[u u^T] = I
    V  = refCov              = sum_b B_b B_b^T                     (verified)

Whitening is by the LOWER Cholesky factor of ``V`` (not V^{-1/2}), so that

    z = L r,  L = inv(chol_lower(V))

has ``z_0 = r_0 / sqrt(V_00)`` = the existing q/p pull EXACTLY, and
``z_1..z_4`` are the successively orthogonalized lambda / phi / d0 / z0
residuals -- i.e. the components are NESTED and "what the vector buys over
q/p alone" is read off directly.  Each component is

    z_k = sum_b a_{b,k} . u_b ,    a_{b,k} = (L B_b)[k, :]

and its per-block standardized variance share is
``v_b^(k) = |a_{b,k}|^2``, with ``sum_b v_b^(k) = 1`` by construction.

The CF of component ``k`` is then the product over blocks of the SAME block
CFs the q/p model uses, at the block's own effective scalar weight

    wstd_k = sqrt( v_pool^(k) / sq2 )

(exact for the MS block under per-collision azimuthal isotropy -- any linear
functional of the two projected angles is again projected-Moliere with the
NORM of the weight -- and trivially exact for the 1-dof ionization/radiative
blocks).  This is ``matres``'s ``sqrt(v_pool/sq2)/sigma`` with the ``/sigma``
absorbed into the whitening.

Cost
----
The five components differ ONLY by the scalar ``wstd_k``, and every exponent
primitive (``ms_step_exponent``, ``ioni_step_exponent``,
``delta_step_exponent``, ``rad_exponent``) depends on ``(wstd, tau)`` through
the PRODUCT alone.  So all five are obtained from ONE call per (block, group)
on the concatenated grid ``tau_ext = concat_k(TG * wstd_k / wstd_ref)``, and
the extraction costs the per-call overhead of the single-functional one, not
five times it.

Output (npz).  The candidate axis is FLATTENED over components: row
``i = it * ncomp + k``, so a downstream ``MaterialCFTerm`` treats each
component as its own pseudo-candidate (the composite-likelihood
approximation; the components are uncorrelated by construction but not
independent -- see ``xcum`` below, which sizes it).

    tgrid       (nt,)            standardized tau grid
    z           (N,)             the whitened residual component
    sigma       (N,)             1.0 (the whitening already carries it)
    comp        (N,) int8        which component
    trk         (N,) int32       track index (0..n-1)
    grp_ptr     (N+1,) int64 / grp_id (nnz,) int16 / S<fam> (nnz,nt) f32
    vQms,vQio   (nnz,) f32       the FIT's Q-matrix variance of that
                                 (component, group) row -- the "Rossi"
                                 Gaussian convention, 14 % narrower than the
                                 model's own second moment for MS
    hit_ptr     (N+1,) int64 / hit_cls (nhz,) int16 / hit_v (nhz,) f32
    vgf,vg_other(N,)             total / unscaled Gaussian variance share
    eta,phi,charge,genPt,trackPt,chi2ndof,nvhit   (n,)  per TRACK
    covdev      (n,)             max |sum_b B B^T / refCov - 1| on the upper
                                 triangle: the closure of the decomposition
    xcum        (n, 6) f32       the four-cumulant cross terms
                                 kappa(z_j,z_j,z_k,z_k) of the MS blocks for
                                 the 6 pairs (j<k) of the first 4 components,
                                 in units of the diagonal kappa_4 -- the size
                                 of what the composite likelihood drops
    group_names, hit_classes

Usage
-----
    python extract_res5.py --files '/ceph/.../task_*/globalcor_*.root' \
      --groups .../materialGroups50.txt --ncomp 5 --decimate 4 --tmax 8 \
      --max-chi2-ndof 3 --max-hess 1e8 --max-grad 1e6 -j 24 -o out.npz
"""

import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import uproot

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.dirname(_HERE)
_MAT = os.path.join(_RES, "matres")
for _p in (_HERE, _RES, _MAT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import groups as G  # noqa: E402
import prodfiles  # noqa: E402
import selection  # noqa: E402  (the standard selection owns --max-chi2-ndof)

COVTOL = 5e-3
FAMS = ("ms", "del", "io_re", "io_im", "rad_re", "rad_im")

TG = None
ioni_sq2 = ioni_step_exponent = ms_step_exponent = None
cf_brems_exact = cf_delta_ray = hitres_classes = None
DELTA_TCUT = DELTA_TMAXCAP = None


def load_cf_primitives():
    global TG, ioni_sq2, ioni_step_exponent, ms_step_exponent
    global cf_brems_exact, cf_delta_ray, hitres_classes
    global DELTA_TCUT, DELTA_TMAXCAP
    if TG is not None:
        return
    import cf_brems_exact as _brems
    import cf_delta_ray as _delta
    import hitres_classes as _hc
    import cf_track_resolution as CTR
    cf_brems_exact, cf_delta_ray, hitres_classes = _brems, _delta, _hc
    TG = CTR.TG
    ioni_sq2 = CTR.ioni_sq2
    ioni_step_exponent = CTR.ioni_step_exponent
    ms_step_exponent = CTR.ms_step_exponent
    DELTA_TCUT = CTR.DELTA_TCUT
    DELTA_TMAXCAP = CTR.DELTA_TMAXCAP


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--files", required=True)
    p.add_argument("--ntasks", type=int, default=0)
    p.add_argument("--groups", default=None)
    p.add_argument("--ncomp", type=int, default=5)
    p.add_argument("--hitmode", choices=["subdet", "class18"], default="class18")
    p.add_argument("--ioni-norm", choices=["var", "raw"], default="var")
    p.add_argument("--no-rad", action="store_true")
    p.add_argument("--no-delta", action="store_true")
    p.add_argument("--max-hess", type=float, default=0.0)
    p.add_argument("--max-grad", type=float, default=0.0)
    p.add_argument("--max-cands", type=int, default=0,
                   help="stop after N selected TRACKS (per file: N/nfiles)")
    p.add_argument("--decimate", type=int, default=4)
    p.add_argument("--tmax", type=float, default=8.0)
    p.add_argument("--validate", action="store_true",
                   help="also compute component 0 the SINGLE-functional way "
                        "(wstd = sqrt(v/sq2)/sigma) and report the max "
                        "difference -- the gate that the 5-component route "
                        "reproduces the established q/p one")
    p.add_argument("-j", "--jobs", type=int, default=8)
    p.add_argument("-o", "--output", required=True)
    # THE STANDARD SELECTION owns `--max-chi2-ndof` (`resolution/selection.py`).
    # This is a SINGLE-track extraction, so its two-track cuts have no column
    # here and are reported absent; the chi2 cut is the one that applies.
    selection.add_args(p)
    return p.parse_args()


_ARGS = _PARMTYPE = _SUBDET = _IO2MS = _TSEL = None


def tau_subset(args):
    idx = np.arange(0, len(TG), max(1, int(args.decimate)))
    if args.tmax > 0.0:
        idx = idx[TG[idx] <= args.tmax + 1e-12]
    return idx


def _init(args, parmtype, subdet, io2ms):
    global _ARGS, _PARMTYPE, _SUBDET, _IO2MS, _TSEL
    _ARGS, _PARMTYPE, _SUBDET, _IO2MS = args, parmtype, subdet, io2ms
    load_cf_primitives()
    _TSEL = tau_subset(args)


def build_io2ms(fname):
    f = uproot.open(fname)
    pt = f["runtree"]["parmtype"].array(library="np")
    raw = f["runtree"]["rawdetid"].array(library="np")
    ms = {int(r): int(i) for i, r in zip(np.where(pt == 10)[0], raw[pt == 10])}
    out = np.full(len(pt), -1, np.int64)
    for i, r in zip(np.where(pt == 11)[0], raw[pt == 11]):
        out[i] = ms.get(int(r), -1)
    return out


def subdet_class_index(subdet, isy):
    sd = int(subdet)
    if sd not in G.SUBDET_NAMES:
        return None
    return (1 if isy else 0) * 6 + sd


def _sym(cov25):
    """refCov is stored UPPER-triangular in a float[25]; rebuild the full
    symmetric matrix.  Verified: the lower triangle is identically 0."""
    C = np.asarray(cov25, np.float64).reshape(5, 5)
    U = np.triu(C)
    return U + np.triu(U, 1).T


class Ext:
    """The concatenated-tau evaluator.  ``scales[k] = wstd_k / wstd_ref``;
    one primitive call on ``tau_ext`` returns all ncomp components."""

    def __init__(self, tg, ncomp):
        self.tg = tg
        self.nt = len(tg)
        self.ncomp = ncomp

    def grid(self, scales):
        return np.concatenate([self.tg * s for s in scales])

    def split(self, arr):
        return arr.reshape(self.ncomp, self.nt)


def process_file(fname):
    args, pt_all, subdet_all, io2ms = _ARGS, _PARMTYPE, _SUBDET, _IO2MS
    ncomp = args.ncomp
    fams = tuple(f for f in FAMS if not (f == "del" and args.no_delta))

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

    want = ["reseigidx", "resinfvarv", "resinfv", "resinfbv", "resinfcov",
            "msmoliidx", "msmoliv", "ioniurbanidx", "ioniurbanv",
            "refParms", "refCov", "genParms", "normalizedChi2", "nValidHits",
            "trackPt", "genPt", "chisqval", "ndof"]
    if "resinfbv" not in keys:
        sys.exit(f"{fname}: no `resinfbv` branch -- this production has no "
                 "generalized functional export (needs exportStepRecords)")
    for br, cut in (("gradmax", args.max_grad), ("hessmax", args.max_hess)):
        if cut > 0.0 and br in keys:
            want.append(br)
    want += [b for b in ("ioniqscaleidx", "ioniqscalev") if b in keys]
    _RB = ("radstepidx", "radstepv", "radstepspecv", "radvgrid")
    want_rad = (not args.no_rad) and all(b in keys for b in _RB)
    if want_rad:
        want += list(_RB)
    _CB = ("reshitidx", "hitDetId", "hitUProj", "clusterSizeX", "clusterChargeBin")
    if args.hitmode == "class18":
        if not all(b in keys for b in _CB):
            sys.exit(f"{fname}: --hitmode class18 needs {_CB}")
        want += list(_CB)
    elif "reshitidx" in keys:
        want.append("reshitidx")
    want = sorted(set(want))
    a = t.arrays(want, library="np")

    nent = len(a["reseigidx"])
    ok_cut, _stdsumm = selection.standard(a, args, n=nent)
    # the per-candidate reduced chi2, stored in the npz; the CUT on it is the
    # standard selection's (`resolution/selection.py`), which builds the same
    # quantity with the same convention
    rchi2 = selection.chi2ndof(a, set(a))
    for br, cut in (("gradmax", args.max_grad), ("hessmax", args.max_hess)):
        if cut > 0.0 and br in a:
            m = np.abs(np.asarray(a[br], np.float64)) < cut
            ok_cut = m if ok_cut is None else (ok_cut & m)

    E = Ext(TG[_TSEL], ncomp)
    store = G.GroupStore(fams, ncomp * len(_TSEL))   # nt slot = ncomp * nt
    out = {k: [] for k in ("eta", "phi", "charge", "chi2ndof", "nvhit",
                           "trackPt", "genPt", "covdev", "sigqop")}
    rres, infl_l = [], []
    zrow, vgfrow, vothrow = [], [], []
    hit_cls, hit_v, hit_ptr = [], [], [0]
    vQms_l, vQio_l = [], []
    xcum_l = []
    nsel = ndrop = ncut = 0
    valmax = 0.0
    grp_mult = []
    cap = 0 if not args.max_cands else max(1, args.max_cands)

    for ic in range(nent):
        if cap and nsel >= cap:
            break
        if ok_cut is not None and not ok_cut[ic]:
            ncut += 1
            continue
        qg = a["genParms"][ic][0]
        if qg == 0.0:
            ndrop += 1
            continue
        V = _sym(a["refCov"][ic])
        c00 = V[0, 0]
        if not (c00 > 0.0) or abs(float(a["resinfcov"][ic]) / c00 - 1.0) > COVTOL:
            ndrop += 1
            continue
        sig = np.sqrt(c00)
        chg = float(np.sign(qg))

        gi = np.asarray(a["reseigidx"][ic])
        if not len(gi):
            ndrop += 1
            continue
        vb = np.asarray(a["resinfvarv"][ic], np.float64)
        Bb = np.asarray(a["resinfbv"][ic], np.float64).reshape(-1, 5, 5)
        fam = pt_all[gi]

        # ---- whitening ----------------------------------------------------
        S5 = np.einsum("bpj,bqj->pq", Bb, Bb)
        iu = np.triu_indices(5)
        dev = np.abs(S5[iu] / np.where(V[iu] != 0.0, V[iu], np.inf) - 1.0)
        covdev = float(np.nanmax(dev))
        try:
            Lc = np.linalg.cholesky(V)
        except np.linalg.LinAlgError:
            ndrop += 1
            continue
        L = np.linalg.inv(Lc)                       # lower triangular
        A = np.einsum("kp,bpj->bkj", L, Bb)         # (nb, 5, 5) -> a_{b,k}
        vk = np.einsum("bkj,bkj->bk", A, A)         # (nb, ncomp<=5)
        vk = vk[:, :ncomp]
        tot = vk.sum(axis=0)
        if not np.all(np.isfinite(tot)) or np.any(np.abs(tot - 1.0) > 5e-3):
            ndrop += 1
            continue
        r = np.asarray(a["refParms"][ic], np.float64) - \
            np.asarray(a["genParms"][ic], np.float64)
        # PHI IS AN ANGLE.  `genParms[2] = g->phi()` is wrapped to (-pi, pi]
        # and the fitted `refParms[2]` is not, so a track near the branch cut
        # gets a residual of +-2 pi = 3.7e4 sigma(phi).  Measured before this
        # line existed: 8 tracks in 20 000 with |z_2| up to 24 011, which alone
        # made Var(z_2) = 2.9e4 (robust sigma 0.93) and, through the Cholesky
        # nesting, Var(z_3) = 9.7e4.
        r[2] = (r[2] + np.pi) % (2.0 * np.pi) - np.pi
        z = (L @ r)[:ncomp]
        # the whitening's conditioning: V_kk / d_k, d = diag(chol)^2, i.e. how
        # much smaller the CONDITIONAL variance of component k is than its
        # marginal.  A guard on this is a statement about the fit's covariance,
        # not about the residual being measured.
        infl = np.diag(V) / np.maximum(np.diag(Lc) ** 2, 1e-300)
        if not np.all(np.isfinite(z)):
            ndrop += 1
            continue

        # ---- step records --------------------------------------------------
        uim = np.asarray(a["msmoliidx"][ic])
        uvm = np.asarray(a["msmoliv"][ic], np.float64)
        uvm = (uvm.reshape(-1, len(uvm) // max(len(uim), 1)) if len(uim)
               else uvm.reshape(0, 10))
        if uvm.shape[1] < 10:
            sys.exit(f"{fname}: msmoliv stride {uvm.shape[1]} < 10")
        uii = np.asarray(a["ioniurbanidx"][ic])
        uvi = np.asarray(a["ioniurbanv"][ic], np.float64)
        uvi = (uvi.reshape(-1, len(uvi) // max(len(uii), 1)) if len(uii)
               else uvi.reshape(0, 11))
        qsi = np.asarray(a["ioniqscaleidx"][ic]) if "ioniqscaleidx" in a else None
        qsv = (np.asarray(a["ioniqscalev"][ic], np.float64).reshape(-1, 2)
               if "ioniqscalev" in a else None)
        if want_rad:
            ridx = np.asarray(a["radstepidx"][ic])
            rrec = np.asarray(a["radstepv"][ic], np.float64).reshape(
                -1, cf_brems_exact.RADV_STRIDE)
            rspc = np.asarray(a["radstepspecv"][ic], np.float64).reshape(
                -1, 2 * cf_brems_exact.NRADV)
            rvg = np.asarray(a["radvgrid"][ic], np.float64)

        per_group = {}
        vQms = {}
        vQio = {}
        val_ref = None
        val_new = None
        # the MS a-vectors kept for the cross-cumulant estimate
        ms_w = []

        def acc(g, f, arr):
            per_group.setdefault(g, {})
            per_group[g][f] = per_group[g].get(f, 0.0) + arr

        ok = True
        # ---- MS + delta (parmtype 10) ---------------------------------------
        sel10 = fam == 10
        for gg in np.unique(gi[sel10]):
            m = sel10 & (gi == gg)
            vpool = vk[m].sum(axis=0)                # (ncomp,)
            rows = np.where(uim == gg)[0]
            if not len(rows):
                ok = False
                break
            steps = uvm[rows]
            sq2 = steps[:, G.MS_THP2].sum()
            if sq2 <= 0.0 or vpool.max() <= 0.0:
                continue
            w = np.sqrt(np.maximum(vpool, 0.0) / sq2)   # (ncomp,)
            wref = float(w.max())
            if wref <= 0.0:
                continue
            scales = w / wref
            te = E.grid(scales)
            grp = steps[:, G.MS_GROUP].astype(np.int64)
            cf = (0.0 if ("del" not in fams) else
                  cf_delta_ray.carve_factor(steps, DELTA_TCUT, DELTA_TMAXCAP))
            for g in np.unique(grp):
                sg = steps[grp == g]
                sms = E.split(ms_step_exponent(sg, wref, te))
                acc(int(g), "ms", sms)
                vQms[int(g)] = vQms.get(int(g), 0.0) + \
                    (w ** 2) * float(sg[:, G.MS_THP2].sum())
                if "del" in fams:
                    sd = E.split(cf_delta_ray.delta_step_exponent(
                        sg, wref, te, DELTA_TCUT, tmax_cap=DELTA_TMAXCAP))
                    acc(int(g), "del", sd - cf * sms)
            # the block's a-vectors, for the cross-cumulant size
            ms_w.append(A[m][:, :ncomp, :].reshape(-1, ncomp, 5))
            if args.validate:
                s0 = ms_step_exponent(steps, float(w[0]), E.tg)
                s1 = ms_step_exponent(steps, np.sqrt(vb[m].sum() / sq2) / sig, E.tg)
                val_ref = s1 if val_ref is None else val_ref + s1
                val_new = s0 if val_new is None else val_new + s0
        if not ok:
            ndrop += 1
            continue

        # ---- ionization + radiative (parmtype 11) ---------------------------
        sel11 = fam == 11
        for gg in np.unique(gi[sel11]):
            m = sel11 & (gi == gg)
            vpool = vk[m].sum(axis=0)
            rows = np.where(uii == gg)[0]
            if not len(rows):
                ok = False
                break
            steps = uvi[rows]
            if args.ioni_norm == "raw":
                w0 = np.abs(np.asarray(a["resinfv"][ic], np.float64).reshape(
                    -1, 5)[m, 0])
                w0 = w0[w0 > 0.0]
                if not len(w0):
                    continue
                sq2 = None
                w = np.full(ncomp, np.sqrt(float(np.mean(w0 ** 2))) / sig)
            else:
                sq2 = ioni_sq2(steps, qsv[qsi == gg] if qsv is not None else None)
                if sq2 <= 0.0:
                    continue
                w = np.sqrt(np.maximum(vpool, 0.0) / sq2)
            wref = float(np.max(np.abs(w)))
            if wref <= 0.0:
                continue
            scales = w / wref
            te = E.grid(scales)
            g10k = io2ms[gg]
            if g10k < 0:
                ok = False
                break
            mrows = np.where(uim == g10k)[0]
            grp_ms = uvm[mrows, G.MS_GROUP].astype(np.int64)
            pair = G.pair_ioni_rows(uvm[mrows, G.MS_XG], len(rows))
            if pair is None:
                ok = False
                break
            grp_io = grp_ms[pair]
            for g in np.unique(grp_io):
                sub = steps[grp_io == g]
                Sc = E.split(ioni_step_exponent(sub, chg * wref, te))
                acc(int(g), "io_re", Sc.real)
                acc(int(g), "io_im", Sc.imag)
                if sq2 is not None:
                    q = ioni_sq2(sub, qsv[qsi == gg] if qsv is not None else None)
                    vQio[int(g)] = vQio.get(int(g), 0.0) + (w ** 2) * q
            if want_rad:
                rrows = np.where(ridx == gg)[0]
                if len(rrows):
                    if len(rrows) != len(mrows):
                        ok = False
                        break
                    for g in np.unique(grp_ms):
                        sub = grp_ms == g
                        ns = int(sub.sum())
                        Sc = E.split(cf_brems_exact.rad_exponent(
                            te, rrec[rrows[sub]], rspc[rrows[sub]], rvg,
                            weights=np.full(ns, chg * wref)))
                        acc(int(g), "rad_re", Sc.real)
                        acc(int(g), "rad_im", Sc.imag)
        if not ok:
            ndrop += 1
            continue

        # ---- hit classes, per component -------------------------------------
        selh = (fam == 8) | (fam == 9)
        cls_i = [dict() for _ in range(ncomp)]
        vhit = vk[selh].sum(axis=0) if selh.any() else np.zeros(ncomp)
        if "reshitidx" in a:
            hidx = np.asarray(a["reshitidx"][ic])
            if args.hitmode == "class18":
                hdet = np.asarray(a["hitDetId"][ic]).astype(np.uint32)
                hsd = (hdet >> 25) & 0x7
                hN = np.asarray(a["clusterSizeX"][ic])
                hU = np.asarray(a["hitUProj"][ic])
                hQ = np.asarray(a["clusterChargeBin"][ic])
            for j in np.where(selh)[0]:
                if args.hitmode == "class18":
                    hh = hidx[j]
                    if hh < 0 or hh >= len(hsd):
                        continue
                    cidx = hitres_classes.class_index(hitres_classes.class_of(
                        hsd[hh], hN[hh], hU[hh], hQ[hh], fam[j] == 9))
                else:
                    cidx = subdet_class_index(subdet_all[gi[j]], fam[j] == 9)
                    if cidx is None:
                        continue
                for k in range(ncomp):
                    if vk[j, k] > 0.0:
                        cls_i[k][cidx] = cls_i[k].get(cidx, 0.0) + vk[j, k]

        # ---- the composite-likelihood cross cumulants -------------------------
        # For an MS block the two projected angles are iso-Gaussian at leading
        # order plus a common non-Gaussian radial tail; the 4th cross cumulant
        # kappa(z_j,z_j,z_k,z_k) of one block is proportional to
        # (a_j.a_k)^2 + |a_j|^2|a_k|^2 / 2 against the diagonal
        # kappa(z_j,z_j,z_j,z_j) ~ 3/2 |a_j|^4 for the same block (both carry
        # the same radial 4th moment).  What is stored is the RATIO
        # sum_b [(a_j.a_k)^2 + |a_j|^2|a_k|^2/2] / sqrt(prod_j sum_b 3/2|a|^4),
        # i.e. the correlation the product-of-marginals likelihood drops.
        if ms_w:
            Aw = np.concatenate(ms_w, axis=0)           # (nb, ncomp, 5)
            gramm = np.einsum("bkj,blj->bkl", Aw, Aw)   # (nb, ncomp, ncomp)
            nrm = np.einsum("bkk->bk", gramm)
            num = (gramm ** 2 + 0.5 * nrm[:, :, None] * nrm[:, None, :]).sum(0)
            den = 1.5 * (nrm ** 2).sum(0)
            xc = num / np.sqrt(np.outer(den, den) + 1e-300)
            xcum_l.append(np.array([xc[j, k] for j in range(min(4, ncomp))
                                    for k in range(j + 1, min(4, ncomp))],
                                   np.float32))
        else:
            xcum_l.append(np.zeros(6, np.float32))

        if args.validate and val_ref is not None:
            valmax = max(valmax, float(np.max(np.abs(val_new - val_ref))))

        # ---- emit ncomp rows -------------------------------------------------
        grp_mult.append(len(per_group))
        store.add_candidate({g: {f: np.asarray(v).ravel()
                                 for f, v in rr.items()}
                             for g, rr in per_group.items()})
        gord = sorted(per_group)
        for g in gord:
            vQms_l.append(np.asarray(vQms.get(g, np.zeros(ncomp)), np.float64))
            vQio_l.append(np.asarray(vQio.get(g, np.zeros(ncomp)), np.float64))
        for k in range(ncomp):
            zrow.append(float(z[k]))
            vgfrow.append(float(vhit[k]))
            vothrow.append(float(vhit[k] - sum(cls_i[k].values())))
            for cidx in sorted(cls_i[k]):
                hit_cls.append(cidx)
                hit_v.append(cls_i[k][cidx])
            hit_ptr.append(len(hit_cls))
        rp = a["refParms"][ic]
        out["eta"].append(-np.log(np.tan((np.pi / 2.0 - rp[1]) / 2.0)))
        out["phi"].append(float(rp[2]))
        out["charge"].append(chg)
        out["chi2ndof"].append(float(rchi2[ic]))
        out["nvhit"].append(float(a["nValidHits"][ic]))
        out["trackPt"].append(float(a["trackPt"][ic]))
        out["genPt"].append(float(a["genPt"][ic]))
        out["covdev"].append(covdev)
        out["sigqop"].append(sig)
        rres.append(r)
        infl_l.append(infl)
        nsel += 1

    res = {k: (np.asarray(v) if len(v) else None) for k, v in out.items()}
    res.update(store.finish())
    res["z"] = np.asarray(zrow, np.float64)
    res["vgf"] = np.asarray(vgfrow, np.float64)
    res["vg_other"] = np.asarray(vothrow, np.float64)
    res["hit_ptr"] = np.asarray(hit_ptr, np.int64)
    res["hit_cls"] = np.asarray(hit_cls, np.int16)
    res["hit_v"] = np.asarray(hit_v, np.float32)
    res["vQms"] = (np.stack(vQms_l).astype(np.float32) if vQms_l
                   else np.zeros((0, ncomp), np.float32))
    res["vQio"] = (np.stack(vQio_l).astype(np.float32) if vQio_l
                   else np.zeros((0, ncomp), np.float32))
    res["xcum"] = (np.stack(xcum_l).astype(np.float32) if xcum_l
                   else np.zeros((0, 6), np.float32))
    res["rres"] = (np.stack(rres) if rres else np.zeros((0, 5)))
    res["inflat"] = (np.stack(infl_l).astype(np.float32) if infl_l
                     else np.zeros((0, 5), np.float32))
    stats = dict(nsel=nsel, ndrop=ndrop, ncut=ncut, sel=_stdsumm,
                 want_rad=int(want_rad),
                 grp_mult=grp_mult, valmax=valmax)
    return fname, res, stats


def main():
    args = parse_args()
    load_cf_primitives()
    fs = prodfiles.resolve(args.files, args.ntasks)
    if not fs:
        sys.exit(f"no files match {args.files}")
    print(f"{len(fs)} files", flush=True)
    if args.max_cands:
        args.max_cands = max(1, int(np.ceil(args.max_cands / len(fs))))
        print(f"cap {args.max_cands} tracks/file", flush=True)
    f0 = uproot.open(fs[0])
    parmtype = f0["runtree"]["parmtype"].array(library="np")
    subdet = f0["runtree"]["subdet"].array(library="np")
    io2ms = build_io2ms(fs[0])
    gnames = G.read_groups(args.groups)[0] if args.groups else None

    t0 = time.time()
    with Pool(args.jobs, initializer=_init,
              initargs=(args, parmtype, subdet, io2ms)) as p:
        parts = [r for r in p.map(process_file, fs) if r is not None]
    parts = [r for r in parts if r[2]["nsel"] > 0]
    print(f"extraction {time.time() - t0:.0f} s", flush=True)

    fams = tuple(f for f in FAMS if not (f == "del" and args.no_delta))
    stores = [r[1] for r in parts]
    ncomp = args.ncomp
    nt = len(tau_subset(args))

    merged = G.concat_group_stores(stores, fams)
    out = dict(merged)
    for k in ("eta", "phi", "charge", "chi2ndof", "nvhit", "trackPt",
              "genPt", "covdev", "sigqop"):
        out[k] = np.concatenate([r[1][k] for r in parts if r[1][k] is not None])
    n = len(out["eta"])
    for k in ("z", "vgf", "vg_other"):
        out[k] = np.concatenate([r[1][k] for r in parts])
    for k in ("rres", "inflat"):
        out[k] = np.concatenate([r[1][k] for r in parts])
    out["vQms"] = np.concatenate([r[1]["vQms"] for r in parts])
    out["vQio"] = np.concatenate([r[1]["vQio"] for r in parts])
    out["xcum"] = np.concatenate([r[1]["xcum"] for r in parts])
    # hit CSR: concatenate with a running offset
    hp = [np.array([0], np.int64)]
    hc, hv, off = [], [], 0
    for r in parts:
        d = r[1]
        hp.append(d["hit_ptr"][1:] + off)
        off += int(d["hit_ptr"][-1])
        hc.append(d["hit_cls"])
        hv.append(d["hit_v"])
    out["hit_ptr"] = np.concatenate(hp)
    out["hit_cls"] = np.concatenate(hc)
    out["hit_v"] = np.concatenate(hv)

    # `store` rows are per TRACK; the term wants per (track, component).
    # The stored S arrays are (nnz, ncomp*nt): reshape to (nnz, ncomp, nt) and
    # emit the per-component grp CSR by repeating each track's group block.
    gp = out.pop("grp_ptr")
    gid = out.pop("grp_id")
    cnt = np.diff(gp)
    assert len(cnt) == n, f"grp_ptr has {len(cnt)} tracks, expected {n}"
    newcnt = np.repeat(cnt, ncomp)
    out["grp_ptr"] = np.concatenate([[0], np.cumsum(newcnt)]).astype(np.int64)
    # row map: for track it and component k, the rows gp[it]:gp[it+1]
    rowmap = np.concatenate([np.arange(gp[it], gp[it + 1])
                             for it in range(n) for _ in range(ncomp)]) \
        if n else np.zeros(0, np.int64)
    compmap = np.concatenate([np.full(cnt[it], k, np.int64)
                              for it in range(n) for k in range(ncomp)]) \
        if n else np.zeros(0, np.int64)
    out["grp_id"] = gid[rowmap].astype(np.int16)
    for fam in fams:
        S = out.pop("S" + fam)
        S = S.reshape(-1, ncomp, nt)
        out["S" + fam] = S[rowmap, compmap, :].astype(np.float32)
    out["vQms"] = out["vQms"][rowmap, compmap].astype(np.float32)
    out["vQio"] = out["vQio"][rowmap, compmap].astype(np.float32)

    out["comp"] = np.tile(np.arange(ncomp, dtype=np.int8), n)
    out["trk"] = np.repeat(np.arange(n, dtype=np.int32), ncomp)
    out["sigma"] = np.ones(n * ncomp)
    out["tgrid"] = TG[tau_subset(args)]
    out["hit_classes"] = np.array(
        list(hitres_classes.CLASSES) if args.hitmode == "class18"
        else [f"{G.SUBDET_NAMES[sd]}_{pr}" for pr in ("x", "y")
              for sd in sorted(G.SUBDET_NAMES)], dtype=object)
    if gnames is not None:
        ng = int(out["grp_id"].max()) + 1 if len(out["grp_id"]) else 0
        out["group_names"] = np.array(
            [gnames.get(g, f"group{g}") for g in range(ng)], dtype=object)
    out["groups_file"] = np.array(args.groups or "", dtype=object)
    gm = np.concatenate([np.asarray(r[2]["grp_mult"]) for r in parts])
    prov = dict(files=len(fs), ntracks=int(n), ncomp=int(ncomp),
                nrows=int(n * ncomp),
                nsel=int(sum(r[2]["nsel"] for r in parts)),
                ndrop=int(sum(r[2]["ndrop"] for r in parts)),
                ncut=int(sum(r[2]["ncut"] for r in parts)),
                grp_mult_mean=float(gm.mean()) if len(gm) else 0.0,
                valmax=float(max(r[2]["valmax"] for r in parts)),
                argv=vars(args))
    # EVERY CALLER LOGS THE STANDARD SELECTION, cut by cut
    _sel = selection.merge([r[2].get("sel") for r in parts])
    if _sel is not None:
        _sel.log(lambda l: print(l, flush=True))
    out["provenance"] = json.dumps(prov)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".",
                exist_ok=True)
    np.savez_compressed(args.output, **out)
    print(json.dumps(prov, indent=1)[:2000], flush=True)
    print(f"wrote {args.output}: {n} tracks x {ncomp} components, "
          f"nnz {len(out['grp_id'])}", flush=True)
    print(f"covdev max {out['covdev'].max():.2e}  "
          f"median {np.median(out['covdev']):.2e}", flush=True)
    if args.validate:
        print(f"VALIDATE max|comp0(new) - qop(reference)| = {prov['valmax']:.3e}")


if __name__ == "__main__":
    main()
