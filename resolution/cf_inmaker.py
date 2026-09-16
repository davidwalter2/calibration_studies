#!/usr/bin/env python3
"""Read the IN-MAKER resolution-CF exponents into the existing cache format.

WHY IT EXISTS. `cf_track_resolution.extract` and
`cf_mass_likelihood.build_pairs_tt` build the per-candidate CF exponents
OFFLINE, from a raw export of every Geant4 step record -- 430 kB and 2.2 s per
candidate, i.e. 16 TB and 24k core-hours at the 40M candidates the full
calibration needs. The CVH makers compute the same exponents at fit time
(TrackPropagation/Geant4e/src/CvhCfExponents.cc, `cvhcf`) and write 6 x 64
floats per candidate. This module turns those branches into the SAME cache the
two extractors produce, so every consumer downstream -- `cf_skew_closure.py`,
`cf_track_resolution.py --closure`, `cf_masslik_fit.py` -- runs on it unchanged.

IT IS A READER, NOT A MODEL. Nothing here computes an exponent. The physics
lives in `cvhcf` (validated against the offline reference by
`cvhcf_validate.py`, worst |dS| ~ 1e-11 against a 1e-6 requirement) and the
offline modules remain the reference definition.

THE GRID. The maker exports 64 tau, the stride-4 subset of the offline
`linspace(0, 14, 448)` truncated at 8, and writes it into the runtree as
`cftau`. The cache's `tgrid` is that array. Two consumers need help with it:

  * `cf_track_resolution.py --closure` reads the MODULE GLOBAL `TG`, not
    `d["tgrid"]`, so it would broadcast a 448-point weight against a 64-point
    cache. `cf_inmaker.py closure ...` runs it with `TG` rebound to the
    cache's own grid -- the reference module is not edited.
  * `cf_masslik_fit.load_inputs` accepts any increasing grid starting at 0
    rather than only the 448-point one, so a 64-point cache loads and a
    448-point cache is unaffected.

usage:
  source /work/submit/david_w/ZMass/mfs/.venv/bin/activate

  # single-track (q/p functional) -> the cf_track_resolution --extract cache
  python3 cf_inmaker.py extract --files '<prod>/task_*/globalcor_resclosure_*.root' \
      --cache runs/cf_trackres_<tag>.npz

  # two-track (mass functional) -> the cf_mass_likelihood --pairs-tt cache
  python3 cf_inmaker.py pairs --files '<prod>/task_*/globalcor_*.root' \
      --cache runs/cf_masspairs_<tag>.npz

  # run the reference closure on a 64-point cache
  python3 cf_inmaker.py closure --cache runs/cf_trackres_<tag>.npz [--kms ...]

  # put a 448-point OFFLINE cache on the maker's 64-point grid, so that the
  # two can be compared without the grid change standing in between
  python3 cf_inmaker.py decimate --cache in448.npz --out in64.npz

  # max |dS| per family between two aligned caches
  python3 cf_inmaker.py compare --cache a.npz --other b.npz
"""
import argparse
import os
import sys

import numpy as np
import uproot
from wums import logging

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import prodfiles  # noqa: E402  (needs HERE on sys.path)

logger = logging.child_logger(__name__)

# The offline `extract` drop: the per-block variance shares must add up to the
# fit's own refCov(0,0), or the CF product is not describing the same track.
COVTOL = 5e-3


# --------------------------------------------------------------------------
# The maker's grid, exactly: the stride-4 subset of the offline
# `np.linspace(0, 14, 448)` truncated at 8. The file's own `cftau` is a
# vector<float>, so its values carry a 6e-8 relative rounding -- fine as a
# PROVENANCE record, not fine as the quadrature abscissae a decimated offline
# cache has to be compared against. The grid is a definition, so it is
# regenerated in double here and the file's copy is used to CHECK that this is
# the grid the maker actually used.
TAU_REF = np.linspace(0.0, 14.0, 448)[0:256:4]


def _runtree_grid(f):
    """(tgrid, model tag) from the runtree, or (None, '') on an old file."""
    if "runtree" not in f:
        return None, ""
    rt = f["runtree"]
    if "cftau" not in rt:
        return None, ""
    tg = np.asarray(rt["cftau"].array(library="np", entry_stop=1)[0], dtype=np.float64)
    if tg.shape == TAU_REF.shape and np.allclose(tg, TAU_REF, rtol=1e-6, atol=0.):
        tg = TAU_REF.copy()
    tag = ""
    if "cfmodel" in rt:
        tag = str(rt["cfmodel"].array(library="np", entry_stop=1)[0])
    return tg, tag


# --------------------------------------------------------------------------
# EXTRA PER-CANDIDATE COLUMNS OF THE MASS CACHE.
#
# The reference `pairs` cache holds only what the J/psi alpha fit needs:
# the standardized residual, sigma, the gen mass and `vgf`. Three other
# consumers need more, and none of it can be recovered after the pass:
#
#   * the Z channel is a WEIGHTED sample (MiNNLO `genweight`, 7.7 % negative);
#   * the Jensen second-order correction (MASSCFTERM_SPEC 4b) needs the two
#     legs' relative resolutions, their correlation and the angular share --
#     the maker reduces the exported `Jpsi_covrefmom` to
#     `Jpsi_sigmarelplus/minus`, `Jpsi_rhomom`, `Jpsi_fang`;
#   * a quality cut needs `chisqval`/`ndof` and the per-leg `maxfracloss`.
#
# Every one of these is OPTIONAL: a branch that is not in the file is not
# read and its column is simply absent from the cache, so a production
# without `Jpsi_covrefmom` or `maxfracloss` still produces the reference cache
# and every existing consumer is unaffected.
#
#            cache key : branch name
_MASS_AUX = {
    "w":            "genweight",
    "mpre":         "Jpsigenpre_mass",
    # the SELECTION variable. DiMuonTrackVertexCandidateProducer cuts on the
    # PRE-REFIT track-level dimuon mass, not on the refit `Jpsi_mass` the
    # likelihood models, so the truncation normalisation's assumption "the cut
    # is on the observable" is only approximately true. Keeping it makes the
    # size of that approximation measurable instead of assumed.
    "mtrk":         "Jpsitrk_mass",
    "sigrelp":      "Jpsi_sigmarelplus",
    "sigrelm":      "Jpsi_sigmarelminus",
    "rhomom":       "Jpsi_rhomom",
    "fang":         "Jpsi_fang",
    "chisqval":     "chisqval",
    "ndof":         "ndof",
    "maxfraclossp": "Muplus_maxfracloss",
    "maxfraclossm": "Muminus_maxfracloss",
    "pt":           "Jpsi_pt",
    "etapair":      "Jpsi_eta",
    "ptp":          "Muplus_pt",
    "ptm":          "Muminus_pt",
    "etap":         "Muplus_eta",
    "etam":         "Muminus_eta",
    # THE STANDARD TWO-TRACK SELECTION's columns (`resolution/selection.py`).
    # They are cached rather than cut on here, because this is the cache a
    # card builder selects FROM: a card that cuts must also normalise over
    # what it cut to, and that decision belongs to the card, not to the
    # cache. Absent in a production without `exportVtxResidual` -- every
    # entry of this map is optional.
    "vtxz":         "Jpsi_vtxz",
    "vtxsig":       "Jpsi_vtxsig",
    "sigmam":       "Jpsi_sigmamass",
}
# integer provenance columns, kept so a cache row can be joined against the
# quadratic term's own extraction of the same production
_MASS_AUX_INT = {"run": "run", "lumi": "lumi", "event": "event",
                 # the selection's per-leg hit counts and the vertex flag
                 "nvp": "Muplus_nvalid", "nvm": "Muminus_nvalid",
                 "vtxok": "Jpsi_vtxok"}


# --------------------------------------------------------------------------
# PER-MATERIAL-GROUP EXPONENTS (`--groups`).
#
# `exportCfGroupExponents=True` makes the maker write each family's log-CF
# exponent SPLIT by the parmtype-15 material group of the step that produced
# it, which is what turns the four ad-hoc `k_hit/k_ms/k_ioni/k_rad` knobs into
# the physical amounts:
#
#     S_f(tau; k) = S_f^fix(tau) + sum_g A(k_g) S_{f,g}(tau)
#
# (`Analysis/HitAnalyzer/doc/resolution-cf-export.md`, and the physics in
# `rabbit.unbinned.MaterialCFTerm`).  `matres/extract_groups.py` produces the
# same layout by rebuilding the exponents OFFLINE from the Geant4 step records;
# that path is unusable on a production with `exportStepRecords=False` (it dies
# on the missing `ioniurbanidx`), which is what the slim productions use.
# This reads the maker's own arrays instead.
#
# NAMING.  The per-group arrays cannot be called `Sms`/`Sio_re`/... in this
# cache: those keys are already the FLAT (n, nt) exponents every existing
# consumer reads.  They are `Sgrp_*` and the `families` array names them, which
# is what `matres/make_material_card.py` keys off (`d["S" + f]` for f in
# `d["families"]`), so ONE card builder serves both this cache and
# `extract_groups.py`'s.
_GRP_FAMS = ("grp_ms", "grp_io_re", "grp_io_im", "grp_rad_re", "grp_rad_im")
_GRP_BRANCH = ("ms", "ioni_re", "ioni_im", "rad_re", "rad_im")
# the flat branch suffix each per-group family must sum back to
_GRP_FLAT = ("ms", "ioni_re", "ioni_im", "rad_re", "rad_im")
_GRP_FLATKEY = ("Sms", "Sio_re", "Sio_im", "Srad_re", "Srad_im")


def _grp_branches(t, prefix):
    """The per-group branch names, or None if the file was not exported with
    `exportCfGroupExponents`."""
    names = [f"{prefix}_grp"] + [f"{prefix}_grp_{s}" for s in _GRP_BRANCH] \
        + [f"{prefix}_hitcls", f"{prefix}_hitv"]
    have = set(t.keys())
    if not all(n in have for n in names):
        return None
    if f"{prefix}_grp_closure" in have:
        names.append(f"{prefix}_grp_closure")
    return names


def _fam_arrays(t, prefix):
    """The six family branch names for `prefix`, or None if absent."""
    names = [f"{prefix}_{s}" for s in
             ("ms", "del", "ioni_re", "ioni_im", "rad_re", "rad_im")]
    return names if all(n in t.keys() for n in names) else None


def _jac_block(t, stop, idx, jaccat, fn):
    """Dense ``(len(idx), nfit)`` mass Jacobian, VECTORIZED.

    ``globalidxv`` and ``Jpsi_jacMass`` are jagged and a per-candidate python
    loop costs ~0.5 ms each -- five hours over a 16 M-candidate production.
    Read as awkward arrays instead and scatter once: flatten, map the global
    index through ``g2f``, and assign. ``globalidxv`` is deduped per candidate,
    so a plain fancy-index assignment is exact (no ``np.add.at`` needed).
    """
    import awkward as ak
    aj = t.arrays(["globalidxv", "Jpsi_jacMass"], library="ak", entry_stop=stop)
    gi, jm = aj["globalidxv"], aj["Jpsi_jacMass"]
    ng, nj = ak.to_numpy(ak.num(gi)), ak.to_numpy(ak.num(jm))
    if not np.array_equal(ng, nj):
        bad = int(np.argmax(ng != nj))
        raise SystemExit(
            f"{fn}: candidate {bad} has {nj[bad]} Jpsi_jacMass entries but "
            f"{ng[bad]} globalidxv entries")
    g2f, fidx = jaccat[0], jaccat[1]
    # candidate index -> output row, -1 for the ones the selection dropped
    row_of = np.full(len(ng), -1, dtype=np.int64)
    row_of[idx] = np.arange(len(idx))
    rows = np.repeat(row_of, ng)
    fi = g2f[ak.to_numpy(ak.flatten(gi))]
    keep = (rows >= 0) & (fi >= 0)
    D = np.zeros((len(idx), len(fidx)), dtype=np.float32)
    D[rows[keep], fi[keep]] = ak.to_numpy(ak.flatten(jm))[keep]
    return D


def _stack(a, key, n):
    """A jagged (n, 64) float branch -> a dense float32 array."""
    v = a[key]
    out = np.empty((len(v), n), dtype=np.float32)
    for i, r in enumerate(v):
        out[i] = r
    return out


# --------------------------------------------------------------------------
def read_files(args, mass):
    # --ntasks caps TASKS, not files: a task of a multi-stream production is
    # `globalcor_0..N-1.root`, and every stream of a usable task is read.
    files = prodfiles.resolve(args.files, args.ntasks, logger=logger.info)
    if not files:
        raise SystemExit(f"no files match {args.files}")
    logger.info(f"{len(files)} files ({'two-track/mass' if mass else 'single-track/qop'})")
    prefix = "cfmass" if mass else "cfqop"

    tgrid = None
    tag = ""
    cols = {}

    def push(k, v):
        cols.setdefault(k, []).append(v)

    nsel = ndrop = 0
    want_hitclass = None
    jacpt = getattr(args, "jac_parmtypes", None) or None
    jaccat = None
    grp = bool(getattr(args, "groups", False))
    if grp and not mass:
        raise SystemExit("--groups is implemented for the `pairs` (two-track "
                         "mass) path only; the single-track cache still goes "
                         "through matres/extract_groups.py")
    ngroups = None
    grpstats = {"clo": 0.0, "n": 0}
    for fn in files:
        try:
            f = uproot.open(fn)
            if "tree" not in f:
                continue
            t = f["tree"]
        except Exception as e:
            logger.warning(f"skipping {fn}: {type(e).__name__}")
            continue
        if jacpt is not None and jaccat is None:
            rt = f["runtree"]
            pt = rt["parmtype"].array(library="np")
            rid = rt["rawdetid"].array(library="np")
            fidx = np.where(np.isin(pt, jacpt))[0]
            g2f = -np.ones(len(pt), dtype=np.int64)
            g2f[fidx] = np.arange(len(fidx))
            jaccat = (g2f, fidx, pt[fidx], rid[fidx])
            logger.info(f"jacobian: {len(fidx)} global parameters of type "
                        f"{sorted(set(pt[fidx].tolist()))} out of {len(pt)}")
        tg, tg_tag = _runtree_grid(f)
        if tg is None:
            raise SystemExit(f"{fn} has no `cftau` in its runtree -- it was not "
                             f"produced with exportCfExponents")
        if tgrid is None:
            tgrid, tag = tg, tg_tag
            logger.info(f"tau grid {len(tgrid)} points [{tgrid[0]:.4f}, {tgrid[-1]:.4f}]")
            logger.info(f"model: {tag}")
        elif not np.array_equal(tgrid, tg) or tag != tg_tag:
            raise SystemExit(
                f"{fn} was exported on a different grid or model than the first "
                f"file; a cache mixing the two would carry two models")
        fams = _fam_arrays(t, prefix)
        if fams is None:
            raise SystemExit(f"{fn} has no {prefix}_* branches")
        nt = len(tgrid)
        gbr = None
        if grp:
            gbr = _grp_branches(t, prefix)
            if gbr is None:
                raise SystemExit(
                    f"{fn} has no {prefix}_grp* / {prefix}_hitcls branches -- "
                    f"it was not produced with exportCfGroupExponents=True")
            if ngroups is None:
                pt_all = f["runtree"]["parmtype"].array(library="np")
                ngroups = int((pt_all == 15).sum())
                if ngroups == 0:
                    raise SystemExit(
                        f"{fn}: the runtree declares no parmtype-15 material "
                        f"groups, so the per-group exponents index nothing")
                logger.info(f"per-group exponents: {ngroups} parmtype-15 "
                            f"material groups, families {list(_GRP_FAMS)}")

        need = list(fams) + [f"{prefix}_vgf", f"{prefix}_ok"]
        aux = auxi = {}
        if mass:
            need += ["Jpsi_mass", "Jpsi_sigmamass", "Jpsigen_mass"]
            have = set(t.keys())
            aux = {k: b for k, b in _MASS_AUX.items() if b in have}
            auxi = {k: b for k, b in _MASS_AUX_INT.items() if b in have}
            if want_hitclass is None:
                want_hitclass = False
                miss = sorted(set(_MASS_AUX) - set(aux))
                logger.info(f"aux columns: {len(aux) + len(auxi)} present"
                            + (f"; absent {miss}" if miss else ""))
            need += list(aux.values()) + list(auxi.values())
            if jacpt is not None:
                for b in ("Jpsi_jacMass", "globalidxv"):
                    if b not in have:
                        raise SystemExit(
                            f"{fn} has no `{b}`; --jac-parmtypes needs the "
                            f"two-track maker's mass Jacobian")
        else:
            need += ["refParms", "refCov", "genParms", "resinfcov",
                     "normalizedChi2", "nValidHits", "trackPt", "genPt",
                     "chisqval", "ndof"]
            # the per-hit class store, for `--hitmode class`; it survives the
            # slimming (reshitidx / resinfvarv / reseigidx are all kept)
            cls = ["reseigidx", "resinfvarv", "reshitidx", "hitDetId",
                   "hitUProj", "clusterSizeX", "clusterChargeBin"]
            hc = all(b in t.keys() for b in cls)
            if want_hitclass is None:
                want_hitclass = hc
                if not hc:
                    logger.warning("input has no per-hit class variables; only "
                                   "the Gaussian hit term will be available")
            if hc:
                need += cls
        stop = None
        if args.max_tracks:
            stop = min(t.num_entries, 3 * int(args.max_tracks) + 100)
        a = t.arrays(need, library="np", entry_stop=stop)

        S = {k: _stack(a, n, nt) for k, n in
             zip(("Sms", "Sdel", "Sio_re", "Sio_im", "Srad_re", "Srad_im"), fams)}
        ok = np.asarray(a[f"{prefix}_ok"]).astype(bool)
        vgf = np.asarray(a[f"{prefix}_vgf"], dtype=np.float64)

        if mass:
            sig = np.asarray(a["Jpsi_sigmamass"], dtype=np.float64)
            mg = np.asarray(a["Jpsigen_mass"], dtype=np.float64)
            mrec = np.asarray(a["Jpsi_mass"], dtype=np.float64)
            # Gen-mass acceptance. The branch names say Jpsi because the
            # two-track maker is resonance-agnostic and names its candidate
            # block after the channel it was written for; the WINDOW is not.
            # Default (3.0969, 0.35) = the J/psi; --mass-window is what lets
            # a Z (or Upsilon) production through. Unmatched candidates carry
            # -99 and are rejected by any window.
            mref, mhw = getattr(args, "mass_window", None) or (3.0969, 0.35)
            keep = ok & np.isfinite(sig) & (sig > 0.) & (np.abs(mg - mref) <= mhw)
            idx = np.where(keep)[0]
            ndrop += int((~keep).sum())
            # VECTORIZED: a per-candidate `push` loop costs ~0.1 ms/candidate,
            # i.e. hours over a 3.9M-candidate production, for arrays that are
            # already dense and contiguous. The columns hold the same values in
            # the same order a per-candidate loop would give.
            push("z", ((mrec - mg) / sig)[idx])
            push("sigma", sig[idx])
            push("eta", mg[idx])           # `eta` is the gen mass in this cache
            push("vgf", vgf[idx])
            for k in S:
                push(k, S[k][idx])
            for k, b in aux.items():
                push(k, np.asarray(a[b], dtype=np.float64)[idx])
            for k, b in auxi.items():
                push(k, np.asarray(a[b], dtype=np.int64)[idx])
            if jacpt is not None:
                push("D", _jac_block(t, stop, idx, jaccat, fn))
            if grp:
                _grp_block(t, stop, idx, prefix, nt, push, grpstats, fn)
            nsel += len(idx)
        else:
            pt = f["runtree"]["parmtype"].array(library="np")
            qg = np.array([r[0] for r in a["genParms"]], dtype=np.float64)
            c00 = np.array([r[0] for r in a["refCov"]], dtype=np.float64)
            cov = np.asarray(a["resinfcov"], dtype=np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                keep = (ok & (qg != 0.) & (c00 > 0.)
                        & (np.abs(cov / c00 - 1.) <= COVTOL))
            idx = np.where(keep)[0]
            ndrop += int((~keep).sum())
            for i in idx:
                sig = np.sqrt(c00[i])
                rp = a["refParms"][i]
                push("z", (rp[0] - qg[i]) / sig)
                push("sigma", sig)
                push("eta", -np.log(np.tan((np.pi / 2. - rp[1]) / 2.)))
                push("phi", rp[2])
                push("charge", np.sign(qg[i]))
                push("vgf", vgf[i])
                push("normchi2", float(a["normalizedChi2"][i]))
                push("nvalidhits", float(a["nValidHits"][i]))
                push("trackpt", float(a["trackPt"][i]))
                push("genpt", float(a["genPt"][i]))
                push("chisqval", float(a["chisqval"][i]))
                push("ndof", float(a["ndof"][i]))
                for k in S:
                    push(k, S[k][i])
                if want_hitclass:
                    _hitclass(a, i, pt, cols)
                else:
                    cols.setdefault("hitcnt", []).append(0)
            nsel += len(idx)
        logger.info(f"{'/'.join(fn.split('/')[-2:])}: cumulative {nsel} "
                    f"({'candidates' if mass else 'tracks'}), drop {ndrop}")
        if args.max_tracks and nsel >= args.max_tracks:
            break
    if not nsel:
        raise SystemExit(
            f"no entries selected out of {ndrop} read. If they were all "
            f"dropped on `{prefix}_ok`, the likeliest cause is a maker that "
            f"books the cf branches but does not fill them -- the THREE-TRACK "
            f"maker (ResidualGlobalCorrectionMakerNTrackG4e) has not been "
            f"given the export and writes them empty with ok = false.")
    if grp:
        logger.info(f"per-group closure (the maker's own "
                    f"max_j|sum_g S_g - S| / max_j|S|): worst "
                    f"{grpstats['clo']:.3e} over {grpstats['n']} candidates")
        cols["_ngroups"] = ngroups
    return tgrid, tag, cols, nsel, ndrop, bool(want_hitclass), jaccat


def _grp_block(t, stop, idx, prefix, nt, push, stats, fn):
    """The per-group and per-hit-class CSR rows of the SELECTED candidates.

    Vectorized with awkward for the same reason `_jac_block` is: a python loop
    over 26 groups x 64 tau per candidate is hours over a production.
    """
    import awkward as ak
    br = [f"{prefix}_grp"] + [f"{prefix}_grp_{s}" for s in _GRP_BRANCH] \
        + [f"{prefix}_hitcls", f"{prefix}_hitv"]
    have_clo = f"{prefix}_grp_closure" in t.keys()
    if have_clo:
        br.append(f"{prefix}_grp_closure")
    # Extra per-hit columns. `hit_v` is a VARIANCE contribution and is
    # unsigned, so it cannot say which way a hit's displacement pushes the
    # curvature -- a per-class LOCATION bias is a shift and a shift needs a
    # direction -- and a class label alone cannot say WHICH module a hit was
    # on.
    #
    # The branch names below are the ones a two-track v2 output actually
    # carries: the per-hit arrays are `reshitidx` and `reshitcls` (UNPREFIXED,
    # they belong to the res block) alongside the prefixed `{prefix}_hitcls` /
    # `{prefix}_hitv` already read above, plus `resinfcovhit`. **There is NO
    # detid branch**: `reshitidx` is the hit's INDEX and the module is reached
    # through the runtree, so `hit_idx` is what gets exported and the detid
    # mapping is a downstream join, not a column.
    #
    # NOTE, and it is a real gap: the v2 slim trees contain NO signed per-hit
    # weight and no detid. Both legs' trees (213 and 228 branches) carry
    # exactly `cfmass_hitcls`, `cfmass_hitv`, `reshitcls`, `reshitidx`,
    # `resinfcovhit` and nothing else per hit. `resinfv` / `resinfbv` are
    # booked inside `if (exportStepRecords_)` in
    # `ResidualGlobalCorrectionMakerBase.cc`, which is OFF for the
    # 81 kB/candidate v2 path, and `hitDetId` only in the
    # fitFromGenParms/validation block. So a signed mass-projected weight
    # needs either `exportStepRecords=True` (the 430 kB/candidate path, a
    # re-production) or a small maker change writing an int8 sign per block.
    _keys = set(t.keys())
    hit_extra = [(k, b) for k, b in
                 (("hit_idx", "reshitidx"),
                  ("hit_rescls", "reshitcls"),
                  ("hit_cov", "resinfcovhit"))
                 if b in _keys]
    br += [b for _, b in hit_extra]
    a = t.arrays(br, library="ak", entry_stop=stop)
    a = a[idx]
    ng = ak.to_numpy(ak.num(a[f"{prefix}_grp"])).astype(np.int64)
    push("grp_cnt", ng)
    push("grp_id", ak.to_numpy(ak.flatten(a[f"{prefix}_grp"])).astype(np.int16))
    tot = int(ng.sum())
    for key, s in zip(_GRP_FAMS, _GRP_BRANCH):
        v = ak.to_numpy(ak.flatten(a[f"{prefix}_grp_{s}"]))
        if v.size != tot * nt:
            raise SystemExit(
                f"{fn}: {prefix}_grp_{s} has {v.size} values but "
                f"{tot} groups x {nt} tau = {tot * nt} were expected; the "
                f"export is row-major (group, tau) and this file is not")
        push("S" + key, v.astype(np.float32).reshape(tot, nt))
    nh = ak.to_numpy(ak.num(a[f"{prefix}_hitcls"])).astype(np.int64)
    push("hit_cnt", nh)
    push("hit_cls", ak.to_numpy(ak.flatten(a[f"{prefix}_hitcls"])).astype(np.int16))
    push("hit_v", ak.to_numpy(ak.flatten(a[f"{prefix}_hitv"])).astype(np.float64))
    for key, brname in hit_extra:
        v = ak.to_numpy(ak.flatten(a[brname]))
        if v.size != int(nh.sum()):
            # the res-block per-hit arrays need not share the cfmass block's
            # jagged layout; say so rather than mis-align them silently
            raise SystemExit(
                f"{fn}: {brname} has {v.size} values against "
                f"{int(nh.sum())} hits from {prefix}_hitcls. They must share "
                "the same per-candidate jagged layout for hit_ptr to index "
                "both; if the res block is laid out per LEG instead, export it "
                "with its own pointer rather than forcing it onto this one")
        push(key, v.astype(np.uint32 if key == "hit_detid"
                           else np.int32 if key in ("hit_idx", "hit_rescls")
                           else np.float64))
    if have_clo:
        c = ak.to_numpy(a[f"{prefix}_grp_closure"]).astype(np.float64)
        push("grp_closure", c)
        if c.size:
            stats["clo"] = max(stats["clo"], float(np.nanmax(c)))
    stats["n"] += len(idx)


def _hitclass(a, ic, pt, cols):
    """Per-block hit class + amplitude, exactly as `extract` stores them."""
    import hitres_classes
    gi = np.asarray(a["reseigidx"][ic])
    vb = np.asarray(a["resinfvarv"][ic], dtype=np.float64)
    fam = pt[gi]
    hidx = np.asarray(a["reshitidx"][ic])
    hdet = np.asarray(a["hitDetId"][ic]).astype(np.uint32)
    hsd = (hdet >> 25) & 0x7
    hN = np.asarray(a["clusterSizeX"][ic])
    hU = np.asarray(a["hitUProj"][ic])
    hQ = np.asarray(a["clusterChargeBin"][ic])
    c00 = float(a["refCov"][ic][0])
    n = 0
    for j in np.where((fam == 8) | (fam == 9))[0]:
        hh = hidx[j]
        if hh < 0 or hh >= len(hsd) or vb[j] <= 0.:
            continue
        cl = hitres_classes.class_of(hsd[hh], hN[hh], hU[hh], hQ[hh], fam[j] == 9)
        cols.setdefault("hitcls", []).append(hitres_classes.class_index(cl))
        cols.setdefault("hitamp2", []).append(vb[j] / c00)
        n += 1
    cols.setdefault("hitcnt", []).append(n)


# --------------------------------------------------------------------------
def write_cache(path, tgrid, tag, cols, mass, nsel, ndrop, hitclass, keep_del,
                jaccat=None, groups_file=None, compress=True):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    ngroups = cols.pop("_ngroups", None)
    out = {}
    for k, v in cols.items():
        # The MASS path appends one array per input FILE (`read_files` is
        # vectorized there); the single-track path appends one row per
        # track. Only the first needs concatenating, and `mass` -- not a
        # shape guess -- says which it is: a single-track `Sms` row is also a
        # 1-D ndarray.
        if mass:
            v = np.concatenate(v, axis=0)
        if k in _MASS_AUX_INT:
            out[k] = np.asarray(v, dtype=np.int64)
        elif k in ("hitcls", "grp_id", "hit_cls", "hit_idx", "hit_rescls",
                   "hit_detid"):
            out[k] = np.asarray(v, dtype=np.int16)
        elif k in ("grp_cnt", "hit_cnt"):
            out[k] = np.asarray(v, dtype=np.int64)
        elif k in ("hit_v", "hit_cov"):
            out[k] = np.asarray(v, dtype=np.float32)
        elif k in ("hitamp2",):
            out[k] = np.asarray(v, dtype=np.float32)
        elif k in ("hitcnt",):
            out[k] = np.asarray(v, dtype=np.int32)
        elif k == "D":
            out[k] = np.concatenate(v, axis=0) if isinstance(v, list) else v
        elif k.startswith("S"):
            out[k] = np.asarray(v, dtype=np.float32)
        else:
            out[k] = np.asarray(v, dtype=np.float64)
    out["tgrid"] = np.asarray(tgrid, dtype=np.float64)
    if jaccat is not None:
        # the parameter map of the D block, so a card builder does not have to
        # re-open a runtree to know what its columns mean
        out["jac_globalidx"] = np.asarray(jaccat[1], dtype=np.int64)
        out["jac_parmtype"] = np.asarray(jaccat[2], dtype=np.int32)
        out["jac_subidx"] = np.asarray(jaccat[3], dtype=np.int64)
    # AFTER the jac block: the group aliases include `fit_parmtype` etc., which
    # `matres/make_material_card.py` needs and which are copies of the jac
    # catalog written just above
    if ngroups is not None:
        _finish_groups(out, ngroups, groups_file)
    # PROVENANCE. `cf_source` and `cf_model` are read by nothing downstream --
    # they are there so that a cache says which evaluator and which switch
    # configuration produced it.
    out["cf_source"] = np.array("cvhcf-inmaker")
    out["cf_model"] = np.array(tag)
    # `rad_model` IS read (cf_skew_closure, cf_masslik_fit): 1 = the radiative
    # family is in the model. `cvhcf` always builds it, so it is 1 whenever the
    # model tag says the term is present.
    out["rad_model"] = np.array(int("rad:" in tag))
    if mass:
        # The sign of an ionization block in the candidate mass CF is -1 for
        # both legs and both charges, hard-wired in the maker.
        out["ioni_sign_fixed"] = np.array(1)
        if not keep_del:
            # `build_pairs_tt` has no `Sdel`: the discrete delta-ray recoil is
            # not part of the published mass model. The maker computes it
            # anyway (it is free), and `--mass-del` keeps it so the two can be
            # compared -- but the default cache is the reference model.
            out.pop("Sdel", None)
    else:
        # The cached Sio_im already carries the charge of the ionization q/p
        # map. Never read as a value, only as a key.
        out["ioni_charge_signed"] = np.array(1)
        if hitclass:
            import hitres_classes
            out["hitclsnames"] = np.array(hitres_classes.CLASSES)
    (np.savez_compressed if compress else np.savez)(path, **out)
    logger.info(f"wrote {path} ({nsel} entries, {ndrop} dropped, "
                f"rad_model={int(out['rad_model'])}"
                + (", compressed" if compress else ", UNcompressed") + ")")


def _finish_groups(out, ngroups, groups_file):
    """CSR pointers, `vg_other`, the label arrays and the aliases
    `matres/make_material_card.py` reads.

    The per-file blocks were pushed as COUNTS (`grp_cnt`, `hit_cnt`), not
    pointers, precisely so that the generic concatenation above is enough and
    no pointer offset has to be fixed up per shard.
    """
    import sys as _sys
    _mat = os.path.join(HERE, "matres")
    if _mat not in _sys.path:
        _sys.path.insert(0, _mat)
    import groups as G

    for pref, cnt in (("grp", "grp_cnt"), ("hit", "hit_cnt")):
        c = np.asarray(out.pop(cnt), dtype=np.int64)
        out[f"{pref}_ptr"] = np.concatenate(
            [[0], np.cumsum(c)]).astype(np.int64)
    n = len(out["sigma"])
    if len(out["grp_ptr"]) != n + 1 or len(out["hit_ptr"]) != n + 1:
        raise SystemExit(
            f"group CSR pointer has {len(out['grp_ptr'])} entries and the hit "
            f"one {len(out['hit_ptr'])}, but there are {n} candidates")

    # v_other = vgf - sum_c v_c, EXACTLY the offline formula the export doc
    # prescribes.  The two-track `cfmass_vgf` is the TOTAL Gaussian share
    # (hits + beamspot + pointing) and the hit-class variances went into
    # `resinfcovhit`, NOT into `resinfcov`, so the remainder is a real physical
    # share and not a rounding residue -- see
    # Analysis/HitAnalyzer/doc/resolution-cf-export.md, "The hit-class blocks".
    hv = np.asarray(out["hit_v"], dtype=np.float64)
    hp = out["hit_ptr"]
    seg = np.zeros(n)
    if hv.size:
        np.add.at(seg, np.repeat(np.arange(n), np.diff(hp)), hv)
    out["vg_other"] = (np.asarray(out["vgf"], np.float64) - seg).astype(np.float64)

    gnames, _ = G.group_param_names(ngroups, groups_file)
    out["group_names"] = np.array(gnames)
    import hitres_classes
    out["hit_classes"] = np.array(hitres_classes.CLASSES)
    out["families"] = np.array(list(_GRP_FAMS))
    out["amount_convention"] = np.array("exp(k_g) per group, weights frozen")
    out["hitmode"] = np.array("class18")
    out["functional"] = np.array("mass")
    out["cf_groups"] = np.array("cvhcf-inmaker")

    # aliases so `matres/make_material_card.py` reads this cache unchanged
    if "m0" not in out:
        out["m0"] = np.asarray(out["z"]) * np.asarray(out["sigma"]) \
            + np.asarray(out["eta"])
    if "mgen" not in out:
        out["mgen"] = np.asarray(out["eta"], dtype=np.float64)
    if "chi2ndof" not in out and "chisqval" in out and "ndof" in out:
        out["chi2ndof"] = (np.asarray(out["chisqval"], np.float64)
                           / np.maximum(np.asarray(out["ndof"], np.float64), 1.0))
    for a, b in (("fit_parmtype", "jac_parmtype"),
                 ("fit_subidx", "jac_subidx"),
                 ("fit_globalidx", "jac_globalidx")):
        if b in out and a not in out:
            out[a] = out[b]


# --------------------------------------------------------------------------
def do_closure(argv):
    """`cf_track_resolution.closure` on a 64-point cache.

    The reference module reads its module-global `TG` rather than the cache's
    own `tgrid`, so it is rebound here for the duration of the call. That is a
    reader-side adaptation: `cf_track_resolution.py` itself is untouched, which
    is what keeps it the definition of the model.
    """
    import datetime
    import cf_track_resolution as ctr
    import pubhtml
    sys.argv = [os.path.join(HERE, "cf_track_resolution.py"), "--closure"] + argv
    args = ctr.parse_args()
    d = np.load(args.cache)
    tg = np.asarray(d["tgrid"], dtype=np.float64)
    if len(tg) != len(ctr.TG) or not np.allclose(tg, ctr.TG):
        logger.info(f"rebinding cf_track_resolution.TG to the cache grid "
                    f"({len(tg)} points, max {tg[-1]:.4f})")
        ctr.TG = tg
    outdir = args.outpath or os.path.expanduser(
        f"~/public_html/cvh/{datetime.date.today().strftime('%y%m%d')}_trackres/")
    os.makedirs(outdir, exist_ok=True)
    pubhtml.ensure_index(outdir, logger=logger)
    ctr.closure(args, outdir)


# --------------------------------------------------------------------------
def do_decimate(argv):
    """Restrict an offline 448-point cache to the maker's 64-point grid.

    The maker's grid IS a subset of the offline one (stride 4, truncated at 8),
    which is the whole reason for choosing it: no interpolation is involved, so
    an evaluator comparison on the decimated cache measures the EVALUATOR and
    not the grid. Every non-exponent column is copied through untouched.
    """
    p = argparse.ArgumentParser(prog="cf_inmaker.py decimate")
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ntau", type=int, default=64)
    p.add_argument("--stride", type=int, default=4)
    a = p.parse_args(argv)
    d = np.load(a.cache, allow_pickle=True)
    tg = np.asarray(d["tgrid"], dtype=np.float64)
    idx = np.arange(0, a.ntau * a.stride, a.stride)
    if idx[-1] >= len(tg):
        raise SystemExit(f"{a.cache} has only {len(tg)} tau points")
    out = {}
    for k in d.files:
        v = d[k]
        if k == "tgrid":
            out[k] = tg[idx]
        elif isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == len(tg):
            out[k] = np.ascontiguousarray(v[:, idx])
        else:
            out[k] = v
    np.savez_compressed(a.out, **out)
    logger.info(f"wrote {a.out} ({len(idx)} tau, max {out['tgrid'][-1]:.6f})")


def do_compare(argv):
    """max |difference| per exponent family between two aligned caches."""
    p = argparse.ArgumentParser(prog="cf_inmaker.py compare")
    p.add_argument("--cache", required=True)
    p.add_argument("--other", required=True)
    a = p.parse_args(argv)
    d1 = np.load(a.cache, allow_pickle=True)
    d2 = np.load(a.other, allow_pickle=True)
    n1, n2 = len(d1["z"]), len(d2["z"])
    print(f"A {a.cache}: {n1} entries, {len(d1['tgrid'])} tau")
    print(f"B {a.other}: {n2} entries, {len(d2['tgrid'])} tau")
    if not np.array_equal(np.asarray(d1["tgrid"]), np.asarray(d2["tgrid"])):
        print("  tau grids DIFFER -- run `decimate` first")
    n = min(n1, n2)
    # The two caches are only row-aligned if they were built from the same
    # files in the same order with the same drops; `z` is the cheapest proof.
    dz = np.max(np.abs(np.asarray(d1["z"])[:n] - np.asarray(d2["z"])[:n]))
    print(f"  rows aligned: max |dz| = {dz:.3e} over {n} entries"
          + ("" if dz < 1e-12 else "   <-- NOT ALIGNED, the rest is meaningless"))
    print(f"  {'key':<10} {'max |d|':>12} {'max |A|':>12}")
    for k in ("Sms", "Sdel", "Sio_re", "Sio_im", "Srad_re", "Srad_im",
              "sigma", "vgf", "z"):
        if k not in d1.files or k not in d2.files:
            continue
        A = np.asarray(d1[k], dtype=np.float64)[:n]
        B = np.asarray(d2[k], dtype=np.float64)[:n]
        print(f"  {k:<10} {np.max(np.abs(A - B)):>12.4e} {np.max(np.abs(A)):>12.4e}")


# --------------------------------------------------------------------------
def main():
    global logger
    logger = logging.setup_logger(__file__, 3, False)
    if len(sys.argv) > 1 and sys.argv[1] == "closure":
        return do_closure(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "decimate":
        return do_decimate(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        return do_compare(sys.argv[2:])
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["extract", "pairs"],
                   help="extract = single-track q/p cache; pairs = two-track mass cache")
    p.add_argument("--files", required=True,
                   help="a glob of stream files, a production directory, or "
                        "@list.txt (see prodfiles.resolve). A glob naming "
                        "stream 0 is widened to every stream of the task.")
    p.add_argument("--ntasks", type=int, default=1000,
                   help="cap on TASKS, not files; 0 = every usable task. The "
                        "default is unchanged, so a single-stream production "
                        "reads exactly what it read before.")
    p.add_argument("--max-tracks", type=int, default=0,
                   help="0 = no limit")
    p.add_argument("--cache", required=True)
    p.add_argument("--mass-window", nargs=2, type=float, default=None,
                   metavar=("M", "HALFWIDTH"),
                   help="gen-mass acceptance for `pairs`: |m_gen - M| <= "
                        "HALFWIDTH [GeV]. Default 3.0969 0.35 (the J/psi). "
                        "A Z production needs e.g. `--mass-window 91.1876 30`, "
                        "otherwise every candidate is dropped on a window it "
                        "was never in.")
    p.add_argument("--jac-parmtypes", type=int, nargs="*", default=None,
                   help="ALSO store the per-candidate mass Jacobian "
                        "`dm_i/dtheta_k` on these global parameter types, as a "
                        "dense (n, nfit) float32 block plus the parameter map. "
                        "`14 15` is the field modes + material groups. This is "
                        "what makes a mass term depend on a calibration "
                        "parameter through its MEAN, and it is the one thing a "
                        "joint fit needs that neither the CF exponents nor the "
                        "quadratic term carry. It is read from the same "
                        "`Jpsi_jacMass` branch `globalfit/extract.py` uses, in "
                        "the same pass, so the rows are aligned with the CF "
                        "rows by construction and no run/lumi/event join is "
                        "needed.")
    p.add_argument("--groups", action="store_true",
                   help="ALSO store the per-MATERIAL-GROUP log-CF exponents "
                        "and the per-HIT-CLASS Gaussian shares the maker "
                        "exports with `exportCfGroupExponents=True`, in the "
                        "CSR layout `rabbit.unbinned.MaterialCFTerm` and "
                        "`matres/make_material_card.py` consume. This is what "
                        "replaces the four ad-hoc k_* resolution knobs with "
                        "the physical parmtype-15 amounts. `pairs` only. "
                        "~27 kB/candidate raw against 1.4 kB flat.")
    p.add_argument("--groups-file",
                   default="/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2/"
                           "src/Analysis/HitAnalyzer/data/materialGroups50.txt",
                   help="materialGroups tier file, for the group NAMES only "
                        "(the ids come from the file). The names must be the "
                        "ones `make_global_term.name_params` builds or a joint "
                        "card will float two disjoint sets of parameters.")
    p.add_argument("--no-compress", action="store_true",
                   help="write with np.savez instead of np.savez_compressed. "
                        "A per-group cache is mostly float32 CF exponents, "
                        "which barely compress, and zipping 20 GB costs more "
                        "wall time than the disk it saves.")
    p.add_argument("--mass-del", action="store_true",
                   help="keep the delta-ray family in the MASS cache (the "
                        "reference `build_pairs_tt` model does not have it)")
    a = p.parse_args()
    mass = a.mode == "pairs"
    tgrid, tag, cols, nsel, ndrop, hitclass, jaccat = read_files(a, mass)
    write_cache(a.cache, tgrid, tag, cols, mass, nsel, ndrop, hitclass,
                a.mass_del, jaccat, groups_file=a.groups_file,
                compress=not a.no_compress)


if __name__ == "__main__":
    main()
