#!/usr/bin/env python3
"""Per-MATERIAL-GROUP and per-HIT-CLASS resolution-CF inputs from a CVH
production.

This is ``globalfit/extract.py`` / ``cf_mass_likelihood.build_pairs_tt`` /
``cf_track_resolution.extract`` with ONE change: the per-family log-CF
exponents are kept split by the material group of each Geant4 step
(``MoliereMsStep::stepGroup``, i.e. the parmtype-15 index) instead of summed
over the whole track, and the Gaussian hit share is kept split by hit class
instead of collapsed into the scalar ``vgf``.

See ``groups.py`` for the physics of the decomposition and for how the Urban
ionization rows are matched back to a group.  Summing the per-group arrays over
groups reproduces the flat cache bit-for-bit up to float32 rounding
(``--validate`` measures it).

Output (npz)
------------
common
    ``tgrid``           (nt,)          the standardized tau grid
    ``sigma``           (n,)           per-candidate resolution
    ``vgf``             (n,)           TOTAL Gaussian variance fraction (= the
                                       flat cache's ``vgf``; kept so a legacy
                                       card can be built from the same file)
    ``vg_other``        (n,)           the share of ``vgf`` no class scales
    ``hit_ptr``         (n+1,)  int64  CSR pointer into the hit-class axis
    ``hit_cls``         (nhz,)  int16  class index
    ``hit_v``           (nhz,) float32 v_c / sigma^2 of that class
    ``grp_ptr``         (n+1,)  int64  CSR pointer into the group axis
    ``grp_id``          (nnz,)  int16  material group index
    ``S<fam>``          (nnz,nt) f32   that group's contribution to family fam
    ``Sfix<fam>``       (n,nt)  f32    pruned groups, weight pinned to 1 (only
                                       written when ``--prune-frac`` > 0)
    ``chi2ndof``        (n,)
    ``hit_classes``     (ncls,) str    the class labels, in index order
    ``group_names``     (ngrp,) str    the material group names, in index order
mass functional (two-track trees)
    ``m0``, ``mgen``    (n,)           ``Jpsi_mass``, ``Jpsigen_mass``
    ``D``               (n,nfit) f32   ``dm_i/dtheta`` on the selected globals
    ``fit_parmtype``, ``fit_subidx``, ``fit_globalidx``  the D column catalog
qop functional (single-track trees)
    ``z``, ``eta``, ``phi``, ``charge``, ``chi2n``, ``nvhit``, ``trackPt``,
    ``genPt``

Usage
-----
    python extract_groups.py \\
      --files '/ceph/.../resolution_trackres_jpsigun_ul16_260905d_m0/task_*/globalcor_*.root' \\
      --functional mass --parmtypes 14 15 --groups .../materialGroups50.txt \\
      -j 16 --max-chi2-ndof 3 --max-hess 1e8 --max-grad 1e6 \\
      -o runs/matres/gun_groups.npz
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
_PARENT = os.path.dirname(_HERE)
for _p in (_HERE, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import groups as G  # noqa: E402
import prodfiles  # noqa: E402  (needs resolution/ on sys.path)
import selection  # noqa: E402  (the standard two-track selection)

# ---------------------------------------------------------------------------
# CF primitives -- imported lazily (cf_ms_exact builds its electron tables at
# import time), exactly as globalfit/extract.py does.
# ---------------------------------------------------------------------------
TG = None
ioni_sq2 = ioni_step_exponent = ms_step_exponent = None
cf_brems_exact = cf_delta_ray = None
hitres_classes = None


def load_cf_primitives():
    global TG, ioni_sq2, ioni_step_exponent, ms_step_exponent
    global cf_brems_exact, cf_delta_ray, hitres_classes
    if TG is not None:
        return
    t0 = time.time()
    import cf_brems_exact as _brems
    import cf_delta_ray as _delta
    import hitres_classes as _hc
    from cf_track_resolution import TG as _TG
    from cf_track_resolution import ioni_sq2 as _isq
    from cf_track_resolution import ioni_step_exponent as _ise
    from cf_track_resolution import ms_step_exponent as _mse

    TG, ioni_sq2, ioni_step_exponent, ms_step_exponent = _TG, _isq, _ise, _mse
    cf_brems_exact, cf_delta_ray, hitres_classes = _brems, _delta, _hc
    print(f"CF primitives imported in {time.time()-t0:.0f} s", flush=True)


# physics conventions, by value (see globalfit/extract.py)
MJPSI = 3.0969
IONI_SGN = -1.0
RAD_SGN = IONI_SGN
MASS_WINDOW = 0.35
COVTOL = 5e-3
DELTA_TCUT = 0.35e-3
DELTA_TMAXCAP = 0.05

FAMS_MASS = ("ms", "io_re", "io_im", "rad_re", "rad_im")
FAMS_QOP = ("ms", "del", "io_re", "io_im", "rad_re", "rad_im")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--files", required=True, help="glob of globalcor_*.root files")
    p.add_argument("--ntasks", type=int, default=0, help="use only the first N files")
    p.add_argument("--functional", choices=["mass", "qop"], default="mass")
    p.add_argument("--groups", default=None, help="materialGroups tier file")
    p.add_argument("--parmtypes", type=int, nargs="*", default=[14, 15],
                   help="global parmtypes to keep in the D rows")
    p.add_argument("--hitmode", choices=["subdet", "class18"], default="subdet",
                   help="'subdet' = 6 subdetectors x {local-x, local-y} (always "
                        "available); 'class18' = the hitres_classes bank "
                        "(needs hitDetId/hitUProj/clusterSizeX/clusterChargeBin, "
                        "present only in single-track productions)")
    p.add_argument("--ioni-norm", choices=["var", "raw"], default="var")
    p.add_argument("--no-rad", action="store_true")
    p.add_argument("--no-delta", action="store_true",
                   help="qop functional only: drop the delta-recoil family")
    p.add_argument("--no-jac", action="store_true", help="do not store the D rows")
    p.add_argument("--max-chi2-ndof", type=float, default=0.0)
    p.add_argument("--max-hess", type=float, default=0.0)
    p.add_argument("--max-grad", type=float, default=0.0)
    p.add_argument("--max-cands", type=int, default=0,
                   help="stop after N selected candidates (per file: N/nfiles)")
    p.add_argument("--prune-frac", type=float, default=0.0,
                   help="fold groups whose max|S| is below this fraction of the "
                        "candidate's max|S| into a fixed (weight-1) baseline")
    p.add_argument("--decimate", type=int, default=1,
                   help="keep every Nth tau point (4 + --tmax 8 reproduces the "
                        "in-maker 64-point grid TG[0:256:4])")
    p.add_argument("--tmax", type=float, default=0.0,
                   help="drop tau points above this value (0 = keep all)")
    p.add_argument("--validate", action="store_true",
                   help="also compute the FLAT exponents the reference builds and "
                        "report max|sum_g S_g - S_flat| per family")
    p.add_argument("-j", "--jobs", type=int, default=8)
    p.add_argument("-o", "--output", required=True)
    selection.add_args(p)
    return p.parse_args()


_ARGS = None
_PARMTYPE = None
_SUBDET = None
_G2F = None
_IO2MS = None


def tau_subset(args):
    """Indices of TG kept in the output (see --decimate / --tmax)."""
    idx = np.arange(0, len(TG), max(1, int(args.decimate)))
    if args.tmax > 0.0:
        idx = idx[TG[idx] <= args.tmax + 1e-12]
    return idx


def _init(args, parmtype, subdet, g2f, io2ms):
    global _ARGS, _PARMTYPE, _SUBDET, _G2F, _IO2MS, _TSEL
    _ARGS, _PARMTYPE, _SUBDET, _G2F, _IO2MS = args, parmtype, subdet, g2f, io2ms
    load_cf_primitives()
    _TSEL = tau_subset(args)


def build_io2ms(fname):
    """parmtype-11 global index -> the parmtype-10 global index of the SAME
    module.  Built from the runtree rather than assumed from the ordering, so a
    module that has one block and not the other cannot silently mis-pair the
    ionization records with the Moliere ones."""
    f = uproot.open(fname)
    pt = f["runtree"]["parmtype"].array(library="np")
    raw = f["runtree"]["rawdetid"].array(library="np")
    ms = {int(r): int(i) for i, r in zip(np.where(pt == 10)[0], raw[pt == 10])}
    out = np.full(len(pt), -1, np.int64)
    for i, r in zip(np.where(pt == 11)[0], raw[pt == 11]):
        j = ms.get(int(r), -1)
        out[i] = j
    return out


# ---------------------------------------------------------------------------
# hit classes
# ---------------------------------------------------------------------------
def subdet_classes():
    names = []
    for proj in ("x", "y"):
        for sd in sorted(G.SUBDET_NAMES):
            names.append(f"{G.SUBDET_NAMES[sd]}_{proj}")
    return names


def subdet_class_index(subdet, isy):
    sd = int(subdet)
    if sd not in G.SUBDET_NAMES:
        return None
    return (1 if isy else 0) * 6 + sd


# ---------------------------------------------------------------------------
# the worker
# ---------------------------------------------------------------------------
def process_file(fname):
    args, pt_all, subdet_all, g2f, io2ms = _ARGS, _PARMTYPE, _SUBDET, _G2F, _IO2MS
    ismass = args.functional == "mass"
    fams = FAMS_MASS if ismass else tuple(
        f for f in FAMS_QOP if not (f == "del" and args.no_delta))
    nfit = int((g2f >= 0).sum()) if g2f is not None else 0

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

    want = ["reseigidx", "resinfvarv", "resinfv",
            "msmoliidx", "msmoliv", "ioniurbanidx", "ioniurbanv"]
    if ismass:
        want += ["Jpsi_mass", "Jpsi_sigmamass", "Jpsigen_mass", "resinfcov",
                 "chisqval", "ndof"]
        if not args.no_jac:
            want += ["globalidxv", "Jpsi_jacMass"]
    else:
        want += ["refParms", "refCov", "genParms", "resinfcov",
                 "normalizedChi2", "nValidHits", "trackPt", "genPt",
                 "chisqval", "ndof"]
    for br, cut in (("gradmax", args.max_grad), ("hessmax", args.max_hess)):
        if cut > 0.0 and br in keys:
            want.append(br)
    want += [b for b in ("ioniqscaleidx", "ioniqscalev") if b in keys]
    _RB = ("radstepidx", "radstepv", "radstepspecv", "radvgrid")
    want_rad = (not args.no_rad) and all(b in keys for b in _RB)
    if want_rad:
        want += list(_RB)
    _CB = ("reshitidx", "hitDetId", "hitUProj", "clusterSizeX", "clusterChargeBin")
    have_cls18 = all(b in keys for b in _CB)
    if args.hitmode == "class18":
        if not have_cls18:
            sys.exit(f"{fname}: --hitmode class18 needs {_CB}")
        want += list(_CB)
    elif "reshitidx" in keys:
        want.append("reshitidx")
    # the columns THE STANDARD TWO-TRACK SELECTION reads, when the production
    # has them (`resolution/selection.py`).  Optional, one by one: the v2
    # productions predate `exportVtxResidual` and carry no `Jpsi_vtxz`.
    if ismass:
        want += [b for b in sum(selection.ALIASES.values(), ()) if b in keys]
    want = sorted(set(want))
    a = t.arrays(want, library="np")

    nent = len(a[want[0]])
    rchi2 = (np.asarray(a["chisqval"], np.float64)
             / np.maximum(np.asarray(a["ndof"], np.float64), 1.0))
    ok_cut = rchi2 < args.max_chi2_ndof if args.max_chi2_ndof > 0.0 else None
    for br, cut in (("gradmax", args.max_grad), ("hessmax", args.max_hess)):
        if cut > 0.0 and br in a:
            m = np.abs(np.asarray(a[br], np.float64)) < cut
            ok_cut = m if ok_cut is None else (ok_cut & m)
    # THE STANDARD TWO-TRACK SELECTION.  Only for the two-track (mass)
    # functional -- a single track has no vertex residual and no second leg.
    stdsumm = None
    if ismass:
        m, stdsumm = selection.standard(a, args, n=nent)
        ok_cut = m if ok_cut is None else (ok_cut & m)

    store = G.GroupStore(fams, len(_TSEL))
    out = {k: [] for k in ("sigma", "vgf", "vg_other", "chi2ndof", "fioni")}
    out.update({k: [] for k in (("m0", "mgen", "D") if ismass else
                                ("z", "eta", "phi", "charge", "chi2n",
                                 "nvhit", "trackPt", "genPt"))})
    hit_cls, hit_v, hit_ptr = [], [], [0]
    fixbuf = {f: [] for f in fams} if args.prune_frac > 0.0 else None
    flatmax = {f: 0.0 for f in fams} if args.validate else None
    nsel = ndrop = ncut = 0
    gap_min, lost_max, nblk_io = np.inf, 0.0, 0
    grp_mult = []
    cap = 0 if not args.max_cands else max(1, args.max_cands)

    for ic in range(nent):
        if cap and nsel >= cap:
            break
        if ok_cut is not None and not ok_cut[ic]:
            ncut += 1
            continue

        # ---- selection + standardization ---------------------------------
        if ismass:
            sig = float(a["Jpsi_sigmamass"][ic])
            mg = float(a["Jpsigen_mass"][ic])
            if not (np.isfinite(sig) and sig > 0.0) or abs(mg - MJPSI) > MASS_WINDOW:
                ndrop += 1
                continue
            chg = None
        else:
            qg = a["genParms"][ic][0]
            if qg == 0.0:
                ndrop += 1
                continue
            c00 = float(a["refCov"][ic][0])
            cov1 = float(a["resinfcov"][ic])
            if not (c00 > 0.0) or abs(cov1 / c00 - 1.0) > COVTOL:
                ndrop += 1
                continue
            sig = np.sqrt(c00)
            chg = float(np.sign(qg))

        gi = np.asarray(a["reseigidx"][ic])
        if not len(gi):
            ndrop += 1
            continue
        vb = np.asarray(a["resinfvarv"][ic], np.float64)
        uw = np.asarray(a["resinfv"][ic], np.float64).reshape(-1, 5)
        fam = pt_all[gi]

        # ---- step records --------------------------------------------------
        uim = np.asarray(a["msmoliidx"][ic])
        uvm = np.asarray(a["msmoliv"][ic], np.float64)
        uvm = (uvm.reshape(-1, len(uvm) // max(len(uim), 1)) if len(uim)
               else uvm.reshape(0, 10))
        if uvm.shape[1] < 10:
            sys.exit(f"{fname}: msmoliv stride {uvm.shape[1]} < 10 -- this "
                     "production has no per-step material group column")
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
        flat = {f: np.zeros(len(TG)) for f in fams} if args.validate else None

        def acc(g, f, arr):
            per_group.setdefault(g, {})
            per_group[g][f] = per_group[g].get(f, 0.0) + arr

        ok = True
        # ---- MS + delta blocks (parmtype 10) -------------------------------
        sel10 = fam == 10
        for gg in np.unique(gi[sel10]):
            m = sel10 & (gi == gg)
            vpool = vb[m].sum()
            if vpool <= 0.0:
                continue
            rows = np.where(uim == gg)[0]
            if not len(rows):
                ok = False
                break
            steps = uvm[rows]
            sq2 = steps[:, G.MS_THP2].sum()
            if sq2 <= 0.0:
                continue
            wstd = np.sqrt(vpool / sq2) / sig
            grp = steps[:, G.MS_GROUP].astype(np.int64)
            cf = (0.0 if ("del" not in fams) else
                  cf_delta_ray.carve_factor(steps, DELTA_TCUT, DELTA_TMAXCAP))
            for g in np.unique(grp):
                sg = steps[grp == g]
                sms = ms_step_exponent(sg, wstd, TG)
                acc(int(g), "ms", sms)
                if "del" in fams:
                    sd = cf_delta_ray.delta_step_exponent(
                        sg, wstd, TG, DELTA_TCUT, tmax_cap=DELTA_TMAXCAP)
                    acc(int(g), "del", sd - cf * sms)
            if args.validate:
                flat["ms"] += ms_step_exponent(steps, wstd, TG)
                if "del" in fams:
                    flat["del"] += (cf_delta_ray.delta_step_exponent(
                        steps, wstd, TG, DELTA_TCUT, tmax_cap=DELTA_TMAXCAP)
                        - cf * ms_step_exponent(steps, wstd, TG))
        if not ok:
            ndrop += 1
            continue

        # ---- ionization + radiative blocks (parmtype 11) -------------------
        # The parmtype-10 and parmtype-11 global indices of a module are
        # different, so the MS rows of the SAME module have to be found by the
        # module's parmtype-10 index -- `io2ms`, built from the runtree.
        sel11 = fam == 11
        g11 = np.unique(gi[sel11])
        for gg in g11:
            m = sel11 & (gi == gg)
            vpool = vb[m].sum()
            if vpool <= 0.0:
                continue
            rows = np.where(uii == gg)[0]
            if not len(rows):
                ok = False
                break
            steps = uvi[rows]
            if args.ioni_norm == "raw":
                w0 = np.abs(uw[m, 0])
                w0 = w0[w0 > 0.0]
                if not len(w0):
                    continue
                wsc = np.sqrt(float(np.mean(w0 ** 2))) / sig
            else:
                sq2 = ioni_sq2(steps, qsv[qsi == gg] if qsv is not None else None)
                if sq2 <= 0.0:
                    continue
                wsc = np.sqrt(vpool / sq2) / sig
            sgn = IONI_SGN if ismass else chg
            # the module's Moliere rows carry the group column
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
            gap, lost = G.ioni_pair_quality(uvm[mrows, G.MS_XG], len(rows))
            gap_min = min(gap_min, gap)
            lost_max = max(lost_max, lost)
            nblk_io += 1
            grp_io = grp_ms[pair]
            for g in np.unique(grp_io):
                sub = steps[grp_io == g]
                S = ioni_step_exponent(sub, sgn * wsc, TG)
                acc(int(g), "io_re", S.real)
                acc(int(g), "io_im", S.imag)
            if args.validate:
                S = ioni_step_exponent(steps, sgn * wsc, TG)
                flat["io_re"] += S.real
                flat["io_im"] += S.imag
            if want_rad:
                rrows = np.where(ridx == gg)[0]
                if len(rrows):
                    if len(rrows) != len(mrows):
                        # the radiative log is pushed under the same guard as
                        # the Moliere one, so this cannot happen; refuse rather
                        # than guess a correspondence
                        ok = False
                        break
                    rsgn = RAD_SGN if ismass else chg
                    for g in np.unique(grp_ms):
                        sub = grp_ms == g
                        ns = int(sub.sum())
                        S = cf_brems_exact.rad_exponent(
                            TG, rrec[rrows[sub]], rspc[rrows[sub]], rvg,
                            weights=np.full(ns, rsgn * wsc))
                        acc(int(g), "rad_re", S.real)
                        acc(int(g), "rad_im", S.imag)
                    if args.validate:
                        S = cf_brems_exact.rad_exponent(
                            TG, rrec[rrows], rspc[rrows], rvg,
                            weights=np.full(len(rrows), rsgn * wsc))
                        flat["rad_re"] += S.real
                        flat["rad_im"] += S.imag
        if not ok:
            ndrop += 1
            continue

        # ---- hit classes ----------------------------------------------------
        selh = (fam == 8) | (fam == 9)
        if ismass:
            vg = sig * sig - float(a["resinfcov"][ic])
        else:
            vg = vb[selh].sum()
        cls_i, v_i = {}, 0.0
        if "reshitidx" in a:
            hidx = np.asarray(a["reshitidx"][ic])
            if args.hitmode == "class18":
                hdet = np.asarray(a["hitDetId"][ic]).astype(np.uint32)
                hsd = (hdet >> 25) & 0x7
                hN = np.asarray(a["clusterSizeX"][ic])
                hU = np.asarray(a["hitUProj"][ic])
                hQ = np.asarray(a["clusterChargeBin"][ic])
            for j in np.where(selh)[0]:
                if vb[j] <= 0.0:
                    continue
                if args.hitmode == "class18":
                    hh = hidx[j]
                    if hh < 0 or hh >= len(hsd):
                        continue
                    ci = hitres_classes.class_index(hitres_classes.class_of(
                        hsd[hh], hN[hh], hU[hh], hQ[hh], fam[j] == 9))
                else:
                    ci = subdet_class_index(subdet_all[gi[j]], fam[j] == 9)
                    if ci is None:
                        continue
                cls_i[ci] = cls_i.get(ci, 0.0) + vb[j]
                v_i += vb[j]
        for ci in sorted(cls_i):
            hit_cls.append(ci)
            hit_v.append(cls_i[ci] / (sig * sig))
        hit_ptr.append(len(hit_cls))

        # ---- pruning ---------------------------------------------------------
        if args.prune_frac > 0.0 and per_group:
            amp = {g: max(float(np.max(np.abs(v))) for v in r.values())
                   for g, r in per_group.items()}
            top = max(amp.values()) if amp else 0.0
            drop = [g for g, v in amp.items() if v < args.prune_frac * top]
            fx = {f: np.zeros(len(TG)) for f in fams}
            for g in drop:
                for f, v in per_group.pop(g).items():
                    fx[f] += v
            for f in fams:
                fixbuf[f].append(fx[f][_TSEL].astype(np.float32))

        # ---- validation -------------------------------------------------------
        if args.validate:
            for f in fams:
                s = np.zeros(len(TG))
                for r in per_group.values():
                    if f in r:
                        s += r[f]
                if args.prune_frac > 0.0:
                    s = s[_TSEL] + fixbuf[f][-1]
                else:
                    s = s[_TSEL]
                flatmax[f] = max(flatmax[f],
                                 float(np.max(np.abs(s - flat[f][_TSEL]))))

        grp_mult.append(len(per_group))
        store.add_candidate({g: {f: v[_TSEL] for f, v in r.items()}
                             for g, r in per_group.items()})
        out["sigma"].append(sig)
        out["vgf"].append(vg / (sig * sig))
        out["vg_other"].append(vg / (sig * sig) - v_i / (sig * sig))
        # the parmtype-11 (ionization) influence share, the `f_ioni` of
        # MASSCFTERM_SPEC's a_i = (1 + f_hit - f_ioni) sigma / m_gen
        out["fioni"].append(float(vb[fam == 11].sum()) / (sig * sig))
        out["chi2ndof"].append(float(rchi2[ic]))
        if ismass:
            out["m0"].append(float(a["Jpsi_mass"][ic]))
            out["mgen"].append(mg)
            if not args.no_jac:
                D = np.zeros(nfit)
                gidx = np.asarray(a["globalidxv"][ic])
                fi = g2f[gidx]
                keep = fi >= 0
                jm = np.asarray(a["Jpsi_jacMass"][ic], np.float64)
                if len(jm) != len(gidx):
                    sys.exit(f"{fname}: Jpsi_jacMass/globalidxv length mismatch")
                np.add.at(D, fi[keep], jm[keep])
                out["D"].append(D.astype(np.float32))
        else:
            rp = a["refParms"][ic]
            out["z"].append((rp[0] - a["genParms"][ic][0]) / sig)
            out["eta"].append(-np.log(np.tan((np.pi / 2.0 - rp[1]) / 2.0)))
            out["phi"].append(float(rp[2]))
            out["charge"].append(chg)
            out["chi2n"].append(float(a["normalizedChi2"][ic]))
            out["nvhit"].append(float(a["nValidHits"][ic]))
            out["trackPt"].append(float(a["trackPt"][ic]))
            out["genPt"].append(float(a["genPt"][ic]))
        nsel += 1

    res = {k: (np.asarray(v) if len(v) else None) for k, v in out.items()}
    res.update(store.finish())
    res["hit_ptr"] = np.asarray(hit_ptr, np.int64)
    res["hit_cls"] = np.asarray(hit_cls, np.int16)
    res["hit_v"] = np.asarray(hit_v, np.float32)
    if fixbuf is not None:
        for f in fams:
            res["Sfix" + f] = (np.stack(fixbuf[f]) if fixbuf[f]
                               else np.zeros((0, len(_TSEL)), np.float32))
    stats = dict(nsel=nsel, ndrop=ndrop, ncut=ncut, sel=stdsumm,
                 want_rad=int(want_rad),
                 gap_min=float(gap_min), lost_max=float(lost_max),
                 nblk_io=nblk_io, grp_mult=grp_mult,
                 flatmax=(flatmax or {}))
    return fname, res, stats


def build_catalog(fname, parmtypes):
    """(g2f, parmtype, subidx, globalidx) for the selected global parmtypes."""
    f = uproot.open(fname)
    pt = f["runtree"]["parmtype"].array(library="np")
    raw = f["runtree"]["rawdetid"].array(library="np")
    sel = np.isin(pt, parmtypes)
    idx = np.where(sel)[0]
    g2f = np.full(len(pt), -1, np.int64)
    g2f[idx] = np.arange(len(idx))
    return g2f, pt[idx], raw[idx], idx


def main():
    args = parse_args()
    # prodfiles applies the .complete filter (and the empty/missing-stream
    # checks the bare sentinel test cannot make) per TASK, and --ntasks caps
    # TASKS: a task of a multi-stream production is globalcor_0..N-1.root and
    # all of its streams are read.
    files = prodfiles.resolve(args.files, args.ntasks, logger=lambda m: print(m, flush=True))
    if not files:
        sys.exit(f"no files match {args.files}")
    print(f"{len(files)} files", flush=True)

    f0 = uproot.open(files[0])
    parmtype = f0["runtree"]["parmtype"].array(library="np")
    subdet = f0["runtree"]["subdet"].array(library="np")
    io2ms = build_io2ms(files[0])
    g2f = cat_pt = cat_si = cat_gi = None
    if args.functional == "mass" and not args.no_jac:
        g2f, cat_pt, cat_si, cat_gi = build_catalog(files[0], args.parmtypes)
        print(f"D catalog: {len(cat_pt)} global parameters "
              f"{sorted(set(cat_pt.tolist()))}", flush=True)
    ngroups = int((parmtype == 15).sum())
    if ngroups == 0:
        sys.exit("this production has no parmtype-15 material groups")
    gnames, _ = G.group_param_names(ngroups, args.groups)
    if args.max_cands:
        args.max_cands = max(1, args.max_cands // len(files))

    load_cf_primitives()
    tsel = tau_subset(args)
    print(f"tau grid: {len(tsel)} of {len(TG)} points, max {TG[tsel][-1]:.4f}",
          flush=True)
    fams = FAMS_MASS if args.functional == "mass" else tuple(
        f for f in FAMS_QOP if not (f == "del" and args.no_delta))

    t0 = time.time()
    parts, stats = [], []
    with Pool(args.jobs, initializer=_init,
              initargs=(args, parmtype, subdet, g2f, io2ms)) as pool:
        for i, r in enumerate(pool.imap_unordered(process_file, files)):
            if r is None:
                continue
            fn, res, st = r
            parts.append(res)
            stats.append(st)
            done = sum(s["nsel"] for s in stats)
            print(f"[{i+1}/{len(files)}] {os.path.basename(os.path.dirname(fn))}: "
                  f"{st['nsel']} sel, {st['ndrop']} drop, {st['ncut']} cut "
                  f"-- cumulative {done} ({time.time()-t0:.0f} s)", flush=True)

    if not parts:
        sys.exit("nothing extracted")

    out = {"tgrid": TG[tsel],
           "hit_classes": np.array(
               subdet_classes() if args.hitmode == "subdet"
               else __import__("hitres_classes").CLASSES),
           "group_names": np.array(gnames),
           "families": np.array(fams),
           "amount_convention": np.array("exp(k_g) per group, weights frozen"),
           "hitmode": np.array(args.hitmode),
           "functional": np.array(args.functional)}
    scal = [k for k in parts[0] if parts[0][k] is not None
            and k not in ("grp_ptr", "grp_id", "hit_ptr", "hit_cls", "hit_v")
            and not k.startswith("S")]
    for k in scal:
        out[k] = np.concatenate([p[k] for p in parts if p.get(k) is not None])
    out.update(G.concat_group_stores(parts, fams))
    for pref in ("hit",):
        ptrs, off = [], 0
        for j, p in enumerate(parts):
            q = p[f"{pref}_ptr"]
            ptrs.append((q[1:] if j else q) + (off if j else 0))
            off += int(q[-1])
        out[f"{pref}_ptr"] = np.concatenate(ptrs)
        out[f"{pref}_cls"] = np.concatenate([p[f"{pref}_cls"] for p in parts])
        out[f"{pref}_v"] = np.concatenate([p[f"{pref}_v"] for p in parts])
    if args.prune_frac > 0.0:
        for f in fams:
            out["Sfix" + f] = np.concatenate([p["Sfix" + f] for p in parts])

    n = len(out["sigma"])
    mult = np.concatenate([np.asarray(s["grp_mult"]) for s in stats])
    nnz = out["grp_id"].size
    nt = len(tsel)
    raw_b = nnz * nt * 4 * len(fams) / max(n, 1)
    print("\n=== summary ===")
    _sel = selection.merge([s.get("sel") for s in stats])
    if _sel is not None:
        _sel.log(print)
    print(f"candidates          {n}")
    print(f"rad model           {stats[0]['want_rad']}")
    print(f"groups/candidate    mean {mult.mean():.2f}  median {np.median(mult):.0f}  "
          f"p1 {np.percentile(mult,1):.0f}  p99 {np.percentile(mult,99):.0f}  max {mult.max()}")
    print(f"hit rows/candidate  {out['hit_cls'].size/max(n,1):.2f}")
    print(f"exponent storage    {raw_b/1024:.2f} kB/candidate raw "
          f"({len(fams)} families x {nt} pts x float32)")
    print(f"flat equivalent     {len(fams)*nt*4/1024:.2f} kB/candidate")
    gmin = min(s["gap_min"] for s in stats)
    lmax = max(s["lost_max"] for s in stats)
    nio = sum(s["nblk_io"] for s in stats)
    print(f"ioni pairing        {nio} blocks, min xg gap {gmin:.3g}, "
          f"max dropped-xg fraction {lmax:.3g}")
    if args.validate:
        print("validation  max |sum_g S_g - S_flat| per family:")
        for f in fams:
            print(f"    {f:<8} {max(s['flatmax'].get(f,0.) for s in stats):.4e}")

    out["provenance"] = np.array(json.dumps(dict(
        files=len(files), argv=sys.argv, functional=args.functional,
        hitmode=args.hitmode, prune_frac=args.prune_frac,
        ioni_gap_min=float(gmin), ioni_lost_max=float(lmax))))
    if args.functional == "mass" and not args.no_jac:
        out["fit_parmtype"] = cat_pt
        out["fit_subidx"] = cat_si
        out["fit_globalidx"] = cat_gi

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(args.output, **out)
    print(f"\nwrote {args.output} "
          f"({os.path.getsize(args.output)/1e6:.1f} MB, {n} candidates)")


if __name__ == "__main__":
    main()
