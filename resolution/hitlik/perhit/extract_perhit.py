#!/usr/bin/env python3
"""Read the maker's per-hit (complement) residual export into an ``npz``.

This is the DATA-side twin of ``../extract_res5.py``, and it is deliberately
much smaller than it: the maker now evaluates the per-component CF exponents
itself (``phcf_*``), so nothing here rebuilds a physics model.  What is left is
a re-layout -- the tree's per-candidate jagged arrays into the flat, CSR
``npz`` that ``hitlik_term.load`` reads -- plus the two quantities that need
the per-block influence VECTORS and so cannot be formed from the term's own
inputs: the per-track cross cumulants and the algebraic ``F^T R = 0`` check.

WHAT IS DIFFERENT FROM THE PROTOTYPE'S npz
    * the number of components VARIES per track (``d = n_meas - 5`` per-hit
      ones, plus 5 truth-referenced ones on MC), so the rows carry an explicit
      ``row_ptr``/``trk``/``comp``/``ckind`` instead of a fixed ``ncomp``;
    * ``inflat`` is per ROW, not per track;
    * ``vQms``/``vQio`` come from the maker (``phcf_grp_vqms``/``vqio``), where
      the prototype rebuilt them from the raw step records.

OUTPUT (npz), N = sum over tracks of (d + nref), n = tracks

    tgrid       (nt,)             the maker's tau grid, from the runtree
    z           (N,)              the whitened residual component
    sigma       (N,)              1.0
    trk         (N,) int32        track index
    comp        (N,) int16        component index within the track
    ckind       (N,) int8         0 = per-hit innovation, 1 = truth-referenced
    hitidx      (N,) int16        which valid hit led the component (-1 = ref)
    dim         (N,) int8         local coordinate 0/1 of that hit (-1 = ref)
    cls         (N,) int16        hit-resolution class of that hit (-1 = ref)
    relpos      (N,)              hit position along the track, 0..1 (-1 = ref)
    inflat      (N,)              Gs_kk / pivot_k, the conditioning
    row_ptr     (n+1,) int64      rows of each track
    grp_ptr     (N+1,)/grp_id (nnz,)/S<fam> (nnz,nt)/vQms,vQio (nnz,)
    hit_ptr     (N+1,)/hit_cls (nhz,)/hit_v (nhz,)
    vgf,vg_other(N,)
    xc_ref0     (N,)              4th cross-cumulant correlation with the
                                  truth-referenced q/p component
    xc_prev     (N,)              ... with the previous per-hit component
    dot_ref0    (N,)              sum_b A_b[k] . A_b[ref0]: ZERO algebraically
                                  (F^T R = 0), so it is a per-track gate
    xc_refpairs (n,6)             the 6 pairs among the first 4 reference
                                  components -- directly comparable with the
                                  prototype's `xcum`
    eta,phi,charge,genPt,trackPt,chi2ndof,nvhit,nmeas,nhitcomp,nref  (n,)
    covdev      (n,)              `phres_vchk`, max_k |sum_b v - 1|
    rankgap     (n,)              lambda_d/lambda_(d+1) of Gs
    sigqop      (n,)              sqrt(refCov[0,0])
    rres        (n,5)             refParms - genParms
    cfms        (n,)              `phcf_msec`, the in-maker CF wall clock
    group_names, hit_classes, groups_file, provenance
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
_HL = os.path.dirname(_HERE)
_RES = os.path.dirname(_HL)
_MAT = os.path.join(_RES, "matres")
for _p in (_HERE, _HL, _RES, _MAT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hitres_classes  # noqa: E402

FAMS = ("ms", "del", "io_re", "io_im", "rad_re", "rad_im")
# the maker's branch suffix for each of them
BRFAM = {"ms": "ms", "del": "del", "io_re": "ioni_re", "io_im": "ioni_im",
         "rad_re": "rad_re", "rad_im": "rad_im"}

BR = ["phres_d", "phres_nref", "phres_nmeas", "phres_ok", "phres_vchk",
      "phres_rankgap", "phres_qrank", "phres_nfree",
      "phresz", "phresrow", "phreshit", "phresdim", "phrescls", "phresinflat",
      "phresvarv", "phresbv", "phcf_vgf", "phcf_nok", "phcf_msec",
      "phcf_grpcomp", "phcf_grp", "phcf_grp_vqms", "phcf_grp_vqio",
      "phcf_hitcomp", "phcf_hitcls", "phcf_hitv",
      "reseigidx", "reshitcls",
      "chisqval", "ndof", "nValidHits", "nValidPixelHits",
      "refParms", "genParms", "refCov", "trackPt", "genPt",
      "gradmax", "hessmax"] + ["phcf_grp_" + BRFAM[f] for f in FAMS]

NAN = float("nan")


def _cross_cum(A, sel):
    """Fourth cross-cumulant CORRELATION matrix from the influence vectors.

    ``A`` is ``(nblock, ncomp, 5)``; ``sel`` the blocks to use (the
    multiple-scattering ones -- they carry the non-Gaussianity).  For an MS
    block the two projected angles are iso-Gaussian at leading order plus a
    common radial tail, so
        kappa(z_j,z_j,z_k,z_k) ~ (A_j.A_k)^2 + |A_j|^2 |A_k|^2 / 2
    against a diagonal 3/2 |A_j|^4.  Returned is the ratio, i.e. exactly the
    correlation the product-of-marginals likelihood drops.  Identical formula
    to ``../extract_res5.py``.
    """
    Aw = A[sel]
    if Aw.shape[0] == 0:
        return None
    gram = np.einsum("bkj,blj->bkl", Aw, Aw)
    nrm = np.einsum("bkk->bk", gram)
    num = (gram ** 2 + 0.5 * nrm[:, :, None] * nrm[:, None, :]).sum(0)
    den = 1.5 * (nrm ** 2).sum(0)
    return num / np.sqrt(np.outer(den, den) + 1e-300)


def process(fname, args, ptype=None):
    t0 = time.time()
    if ptype is None:
        ptype = _PARMTYPE
    out = {k: [] for k in ("eta", "phi", "charge", "chi2ndof", "nvhit",
                           "nmeas", "nhitcomp", "nref", "trackPt", "genPt",
                           "covdev", "rankgap", "sigqop", "cfms")}
    rows = dict(z=[], comp=[], ckind=[], hitidx=[], dim=[], cls=[],
                relpos=[], inflat=[], vgf=[], vg_other=[],
                xc_ref0=[], xc_prev=[], dot_ref0=[])
    grp_id, grp_cnt, vQms, vQio = [], [], [], []
    Sacc = {f: [] for f in FAMS}
    hit_cls, hit_v, hit_cnt = [], [], []
    rres, xcref, row_cnt = [], [], []
    nsel = ndrop = 0

    with uproot.open(fname) as fh:
        t = fh["tree"]
        have = set(k.split(";")[0] for k in t.keys())
        want = [b for b in BR if b in have]
        if "phres_d" not in have:
            return fname, None, dict(nsel=0, ndrop=0, msg="no perhit branches")
        a = t.arrays(want, library="np")
        tg = None
        if "runtree" in fh:
            rt = fh["runtree"].arrays(["cftau"], library="np")
            if len(rt["cftau"]):
                tg = np.asarray(rt["cftau"][0], np.float64)
    if tg is None:
        raise RuntimeError(f"{fname}: no cftau in the runtree")
    nt = len(tg)
    nent = len(a["phres_d"])

    chi2ndof = (np.asarray(a["chisqval"], np.float64)
                / np.maximum(np.asarray(a["ndof"], np.float64), 1.0))
    keep = np.ones(nent, bool)
    if args.max_chi2_ndof > 0:
        keep &= chi2ndof < args.max_chi2_ndof
    for br, cut in (("gradmax", args.max_grad), ("hessmax", args.max_hess)):
        if cut > 0 and br in a:
            keep &= np.asarray(a[br], np.float64) < cut

    for ic in range(nent):
        if args.max_cands and nsel >= args.max_cands:
            break
        nd = int(a["phres_d"][ic])
        nref = int(a["phres_nref"][ic])
        ntot = nd + nref
        if ntot <= 0 or not keep[ic] or not bool(a["phres_ok"][ic]):
            ndrop += 1
            continue
        if int(a["phcf_nok"][ic]) != ntot:
            ndrop += 1
            continue
        nres = len(a["reseigidx"][ic])
        nmeas = int(a["phres_nmeas"][ic])

        z = np.asarray(a["phresz"][ic], np.float64)
        vgf = np.asarray(a["phcf_vgf"][ic], np.float64)
        if len(z) != ntot or len(vgf) != ntot:
            ndrop += 1
            continue

        # ---- per-(component, group) rows ---------------------------------
        gc = np.asarray(a["phcf_grpcomp"][ic], np.int64)
        gg = np.asarray(a["phcf_grp"][ic], np.int64)
        order = np.lexsort((gg, gc))          # component major, group minor
        gc, gg = gc[order], gg[order]
        cnt = np.bincount(gc, minlength=ntot)
        grp_id.append(gg)
        grp_cnt.append(cnt)
        vQms.append(np.asarray(a["phcf_grp_vqms"][ic], np.float64)[order])
        vQio.append(np.asarray(a["phcf_grp_vqio"][ic], np.float64)[order])
        for f in FAMS:
            S = np.asarray(a["phcf_grp_" + BRFAM[f]][ic], np.float64)
            Sacc[f].append(S.reshape(-1, nt)[order])

        # ---- per-(component, hit class) rows ------------------------------
        hc = np.asarray(a["phcf_hitcomp"][ic], np.int64)
        hk = np.asarray(a["phcf_hitcls"][ic], np.int64)
        hv = np.asarray(a["phcf_hitv"][ic], np.float64)
        ho = np.lexsort((hk, hc))
        hc, hk, hv = hc[ho], hk[ho], hv[ho]
        hcnt = np.bincount(hc, minlength=ntot)
        hit_cls.append(hk)
        hit_v.append(hv)
        hit_cnt.append(hcnt)
        vgo = vgf - np.bincount(hc, weights=hv, minlength=ntot)

        # ---- the cross cumulants and the F^T R = 0 check -------------------
        xc0 = np.full(ntot, NAN)
        xcp = np.full(ntot, NAN)
        dot0 = np.full(ntot, NAN)
        xcp4 = np.full(6, NAN)
        if "phresbv" in a and len(a["phresbv"][ic]) == nres * ntot * 5:
            A = np.asarray(a["phresbv"][ic], np.float64).reshape(nres, ntot, 5)
            # ONLY the parmtype-10 (multiple-scattering) blocks, exactly as
            # the prototype's `ms_w`: they carry the non-Gaussianity the
            # formula is derived for, and the ionization block of the same
            # rows would double-count them.  The family comes from the
            # runtree's `parmtype` at the block's global index.
            selb = ptype[np.asarray(a["reseigidx"][ic], np.int64)] == 10
            xc = _cross_cum(A, selb)
            if xc is not None:
                if nref >= 1:
                    xc0 = xc[:, nd]
                for k in range(1, nd):
                    xcp[k] = xc[k, k - 1]
                if nref >= 4:
                    xcp4 = np.array([xc[nd + j, nd + k] for j in range(4)
                                     for k in range(j + 1, 4)])
            if nref >= 1:
                dot0 = np.einsum("bkj,bj->k", A, A[:, nd, :])

        # ---- emit ---------------------------------------------------------
        rows["z"].append(z)
        rows["comp"].append(np.arange(ntot))
        rows["ckind"].append(np.concatenate([np.zeros(nd, np.int8),
                                             np.ones(nref, np.int8)]))
        rows["hitidx"].append(np.asarray(a["phreshit"][ic], np.int64))
        rows["dim"].append(np.asarray(a["phresdim"][ic], np.int64))
        rows["cls"].append(np.asarray(a["phrescls"][ic], np.int64))
        pr = np.asarray(a["phresrow"][ic], np.float64)
        rows["relpos"].append(np.where(pr >= 0, pr / max(nmeas - 1, 1), -1.0))
        rows["inflat"].append(np.asarray(a["phresinflat"][ic], np.float64))
        rows["vgf"].append(vgf)
        rows["vg_other"].append(vgo)
        rows["xc_ref0"].append(xc0)
        rows["xc_prev"].append(xcp)
        rows["dot_ref0"].append(dot0)
        row_cnt.append(ntot)
        xcref.append(xcp4)

        rp = np.asarray(a["refParms"][ic], np.float64)
        gp = np.asarray(a["genParms"][ic], np.float64)
        cv = np.asarray(a["refCov"][ic], np.float64)
        out["eta"].append(-np.log(np.tan((np.pi / 2.0 - rp[1]) / 2.0)))
        out["phi"].append(float(rp[2]))
        out["charge"].append(1.0 if rp[0] >= 0 else -1.0)
        out["chi2ndof"].append(float(chi2ndof[ic]))
        out["nvhit"].append(float(a["nValidHits"][ic]))
        out["nmeas"].append(float(nmeas))
        out["nhitcomp"].append(float(nd))
        out["nref"].append(float(nref))
        out["trackPt"].append(float(a["trackPt"][ic]))
        out["genPt"].append(float(a["genPt"][ic]) if "genPt" in a else NAN)
        out["covdev"].append(float(a["phres_vchk"][ic]))
        out["rankgap"].append(float(a["phres_rankgap"][ic]))
        out["sigqop"].append(float(np.sqrt(max(cv[0], 0.0))))
        out["cfms"].append(float(a["phcf_msec"][ic]))
        rres.append(rp - gp)
        nsel += 1

    res = {k: np.asarray(v, np.float64) for k, v in out.items()}
    for k, v in rows.items():
        res[k] = np.concatenate(v) if v else np.zeros(0)
    res["row_ptr"] = np.concatenate([[0], np.cumsum(row_cnt)]).astype(np.int64)
    res["trk"] = np.repeat(np.arange(len(row_cnt), dtype=np.int32),
                           np.asarray(row_cnt, np.int64))
    res["grp_cnt"] = np.concatenate(grp_cnt) if grp_cnt else np.zeros(0, np.int64)
    res["grp_id"] = np.concatenate(grp_id) if grp_id else np.zeros(0, np.int64)
    res["vQms"] = np.concatenate(vQms) if vQms else np.zeros(0)
    res["vQio"] = np.concatenate(vQio) if vQio else np.zeros(0)
    for f in FAMS:
        res["S" + f] = (np.concatenate(Sacc[f]) if Sacc[f]
                        else np.zeros((0, nt)))
    res["hit_cnt"] = np.concatenate(hit_cnt) if hit_cnt else np.zeros(0, np.int64)
    res["hit_cls"] = np.concatenate(hit_cls) if hit_cls else np.zeros(0, np.int64)
    res["hit_v"] = np.concatenate(hit_v) if hit_v else np.zeros(0)
    res["rres"] = np.stack(rres) if rres else np.zeros((0, 5))
    res["xc_refpairs"] = np.stack(xcref) if xcref else np.zeros((0, 6))
    res["tgrid"] = tg
    return fname, res, dict(nsel=nsel, ndrop=ndrop, dt=time.time() - t0)


_PARMTYPE = None


def _init(pt):
    global _PARMTYPE
    _PARMTYPE = pt


def _work(argt):
    f, args = argt
    try:
        return process(f, args)
    except Exception as e:                     # noqa: BLE001
        return f, None, dict(nsel=0, ndrop=0, msg=f"{type(e).__name__}: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--files", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("-j", "--jobs", type=int, default=24)
    p.add_argument("--max-chi2-ndof", type=float, default=3.0)
    p.add_argument("--max-hess", type=float, default=1e8)
    p.add_argument("--max-grad", type=float, default=1e6)
    p.add_argument("--max-cands", type=int, default=0)
    p.add_argument("--max-files", type=int, default=0)
    p.add_argument("--groups", default="")
    args = p.parse_args()

    files = sorted(glob.glob(args.files))
    if args.max_files:
        files = files[: args.max_files]
    print(f"# {len(files)} files, {args.jobs} workers", flush=True)
    t0 = time.time()
    parts = []
    with uproot.open(files[0]) as f0:
        ptype = f0["runtree"]["parmtype"].array(library="np")
    with Pool(args.jobs, initializer=_init, initargs=(ptype,)) as pool:
        for i, (f, r, st) in enumerate(
                pool.imap_unordered(_work, [(f, args) for f in files])):
            if r is None:
                print(f"  [skip] {os.path.basename(os.path.dirname(f))}: "
                      f"{st.get('msg', '')}", flush=True)
                continue
            parts.append(r)
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(files)}  {time.time()-t0:.0f} s", flush=True)
    if not parts:
        raise SystemExit("nothing extracted")

    out = {}
    ntrk = 0
    nrow = 0
    for k in ("eta", "phi", "charge", "chi2ndof", "nvhit", "nmeas",
              "nhitcomp", "nref", "trackPt", "genPt", "covdev", "rankgap",
              "sigqop", "cfms"):
        out[k] = np.concatenate([r[k] for r in parts])
    for k in ("z", "comp", "ckind", "hitidx", "dim", "cls", "relpos",
              "inflat", "vgf", "vg_other", "xc_ref0", "xc_prev", "dot_ref0"):
        out[k] = np.concatenate([r[k] for r in parts])
    out["rres"] = np.concatenate([r["rres"] for r in parts])
    out["xc_refpairs"] = np.concatenate([r["xc_refpairs"] for r in parts])
    trk = []
    rp = [0]
    for r in parts:
        trk.append(r["trk"] + ntrk)
        rp.append(rp[-1] + 0)
        ntrk += len(r["eta"])
    out["trk"] = np.concatenate(trk).astype(np.int32)
    cnts = np.concatenate([np.diff(r["row_ptr"]) for r in parts])
    out["row_ptr"] = np.concatenate([[0], np.cumsum(cnts)]).astype(np.int64)
    gcnt = np.concatenate([r["grp_cnt"] for r in parts])
    out["grp_ptr"] = np.concatenate([[0], np.cumsum(gcnt)]).astype(np.int64)
    out["grp_id"] = np.concatenate([r["grp_id"] for r in parts]).astype(np.int16)
    out["vQms"] = np.concatenate([r["vQms"] for r in parts]).astype(np.float32)
    out["vQio"] = np.concatenate([r["vQio"] for r in parts]).astype(np.float32)
    for f in FAMS:
        out["S" + f] = np.concatenate([r["S" + f] for r in parts]).astype(np.float32)
    hcnt = np.concatenate([r["hit_cnt"] for r in parts])
    out["hit_ptr"] = np.concatenate([[0], np.cumsum(hcnt)]).astype(np.int64)
    out["hit_cls"] = np.concatenate([r["hit_cls"] for r in parts]).astype(np.int16)
    out["hit_v"] = np.concatenate([r["hit_v"] for r in parts]).astype(np.float32)
    out["sigma"] = np.ones(len(out["z"]))
    out["tgrid"] = parts[0]["tgrid"]
    out["hit_classes"] = np.array(
        [str(c) for c in hitres_classes.CLASSES], dtype=object)
    ng = int(out["grp_id"].max()) + 1 if len(out["grp_id"]) else 0
    if args.groups:
        import groups as G
        names, _ = G.group_param_names(ng, args.groups)
        out["group_names"] = np.array([n.replace("material_", "")
                                       for n in names], dtype=object)
    else:
        out["group_names"] = np.array([f"g{i}" for i in range(ng)], dtype=object)
    out["groups_file"] = np.array(args.groups or "", dtype=object)
    out["provenance"] = json.dumps(dict(
        files=args.files, nfiles=len(files), cuts=dict(
            max_chi2_ndof=args.max_chi2_ndof, max_hess=args.max_hess,
            max_grad=args.max_grad), groups=args.groups,
        script=os.path.basename(__file__), when=time.strftime("%Y-%m-%d %H:%M")))
    nrow = len(out["z"])
    np.savez_compressed(args.output, **out)
    print(f"wrote {args.output}: {ntrk} tracks, {nrow} rows "
          f"({nrow/max(ntrk,1):.1f}/track), grp nnz {len(out['grp_id'])}, "
          f"hit nnz {len(out['hit_cls'])}, {time.time()-t0:.0f} s", flush=True)
    print(f"covdev median {np.median(out['covdev']):.2e} "
          f"max {out['covdev'].max():.2e}", flush=True)


if __name__ == "__main__":
    main()
