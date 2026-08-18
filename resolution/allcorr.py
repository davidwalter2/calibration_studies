#!/usr/bin/env python3
"""ALL SEVEN corrections ON, natively, and the radiation-harmonisation audit.

WHAT THIS IS
------------
NOTES_CLOSURE_FINAL measured the eight-species x radiation x {qop,locx} matrix
with the FOUR ionization corrections on.  NOTES_MOLIEREWRONG then added three
MS harmonisations and reported them as a per-cell SHIFT against that matrix's
base curve.  Nothing has ever been run with all seven set at once and reported
as an absolute closure with its own error bars.  That is what `closure` does.

THE SEVEN, and how each one comes on
------------------------------------
  IONIZATION (four; the C++/export half plus, for Kokoulin, an offline half)

    CVH_IONI_EXACTDELTA=1   exported into the model (`ctr.SWITCHES_ON`)
    CVH_IONI_KOKOULIN=1     exported AND `ctr.IONI_KOKOULIN`, driven from
                            `hp.kok_for` so the |PDG|==13 guard is honoured
    CVH_REF_CHARGEAWARE=1   exported into the model
    CVH_REF_SPECIESDEDX=1   exported into the model

  MULTIPLE SCATTERING (three, all pure-offline module globals in
  `cf_track_resolution`; none of them touches an exported record)

    MS_ELEC_TMAX=1, MS_ELEC_EDGE=1   the atomic-electron term gets its own HARD
                                     angular range at G4's `cosTetMaxElec`
                                     instead of the nucleus's
    MS_SNAP_YMAX=0                   each record's own form-factor ceiling is
                                     passed instead of being snapped to one of
                                     13 half-decade rows
    MS_WVI_SPLIT=1 (+ MS_FINE_G=1)   G4WentzelVIModel's internal
                                     variance-preserving Gaussian/single-
                                     scattering split.  NOTES_MOLIEREWRONG s6.1
                                     measured that the split alone is +0.00024
                                     and `gshape`'s own 21-node/decade
                                     quadrature is -0.00023, species-independent
                                     and opposite: the split MUST be applied
                                     together with the numerical fix that pays
                                     for it, so `MS_FINE_G` travels with it and
                                     is not a fourth MS item.

`MS_WVI_SPLIT` needs two MEASURED per-species inputs -- the delta-ray rate per
g/cm^2 (`MS_WVI_NPERX`, from the simulation's own census) and G4's transport log
per unit chi_c^2 (`MS_WVI_LG`, from `wvisplit_g4driver`).  At pT = 3 they are
`wvisplit.SP_INPUT`; at pT = 40 they are re-measured here (`sp40`), because both
are momentum-dependent.

IS IT NATIVE?
-------------
Yes, and the question is worth answering rather than assuming.  A "shift"
would mean adding a delta to a published curve.  What `moliere_probe gauge`
actually does is call `wvisplit._rows` TWICE -- once with the knobs at their
defaults and once with them set -- and print the difference; each call
re-evaluates the model CF, re-runs the Fisher inversion and re-scores against
the simulation.  So NOTES_MOLIEREWRONG s5's `left` column is already a
measurement.  What it is NOT is a closure reported against its own error: the
note printed only the rms of the resulting curve, took the errors from the BASE
cell, and never reported the outermost plane.  This module reports the absolute
curve, its per-probe error, its significance, and BOTH the ladder mean and the
outermost (fully accumulated) plane.

THE OUTERMOST PLANE
-------------------
Every closure row is cumulative over legs 0..k, so plane 13 is the whole-track
number -- the one the global fit integrates over.  `closure_rows` returns the
per-plane rows; the ladder statistic is their mean and is what every note in
this family has quoted.  Both are printed here.  The outermost plane's error is
the ordinary per-plane one (`errs[-1]`): the 2.8x plane-correlation inflation
of NOTES_GEOMCLOSURE s1 applies to the MEAN over planes and to nothing else.

SUBCOMMANDS
    live      every one of the seven switches, and the three radiation
              switches, measured before anything is interpreted
    control   reproduce a published row with the MS three OFF
    closure   TASK 1: 8 species x rad{ON,OFF} x {qop,locx}, all seven
    setup40   build the pT = 40 q=-1 and q=+1 toy areas
    sim40     the pT = 40 simulations
    export40  the pT = 40 models, all four ionization corrections ON
    sp40      the two measured MS_WVI inputs at pT = 40
    closure40 TASK 2: the muon at pT = 40, both charges, rad ON and OFF
    radharm   TASK 3: is the radiative channel harmonised?
"""

import argparse
import contextlib
import glob
import hashlib
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("RES_NO_PHI_CACHE", "1")

import cf_propagation_test as cpt                                # noqa: E402
import cf_ms_exact as mx
import cf_track_resolution as ctr                                # noqa: E402
import deltaspec as ds                                           # noqa: E402
import fisher_norm as fn                                         # noqa: E402
import geom_closure as gc                                        # noqa: E402
import hadron_probe as hp                                        # noqa: E402
import tail_probe as tp                                          # noqa: E402
# `radoff_species` pulls in speciesdedx -> barkas_probe (MODEL_SUFFIX shim on
# hp.model_path) and pion_probe (the `_s1*` SEED PIN on hp.sim_glob/census_glob).
import radoff_species as rs                                      # noqa: E402
import barkas_probe as bp                                        # noqa: E402
import speciesdedx as sd                                         # noqa: E402
import wvisplit as wv                                            # noqa: E402
import moliere_probe as mo                                       # noqa: E402  (registers MS_ELEC_EDGE / MS_FINE_G in wv._KNOBS)
import hbasis as hb                                              # noqa: E402

assert "_s1" in hp.sim_glob(13, "off"), "the seed pin is not installed"
assert "_s1" in hp.census_glob(13, "off"), "the census pin is not installed"

LOGD = os.path.join(hp.SCRATCH, "ac")
os.makedirs(LOGD, exist_ok=True)

UCURVE = fn.UCURVE
UROW = "".join(f"{u:>9g}" for u in UCURVE)
ORDER8 = bp.ORDER8            # [13, -13, -211, 211, -321, 321, -2212, 2212]

# ---------------------------------------------------------------- THE SEVEN
# The four ionization ones live in the EXPORT (ctr.SWITCHES_ON, greppable and
# printed in every run log); the three MS ones are offline module globals.
# MS_FINE_G is not an eighth item: it is the numerical companion rank 3 must
# not be applied without (NOTES_MOLIEREWRONG s6.1).
MS_SEVEN = {"MS_ELEC_TMAX": 1.0, "MS_ELEC_EDGE": 1.0,
            "MS_SNAP_YMAX": 0.0, "MS_WVI_SPLIT": 1.0, "MS_FINE_G": 1.0}
MS_OFF = {"MS_ELEC_TMAX": 0.0, "MS_ELEC_EDGE": 0.0,
          "MS_SNAP_YMAX": 1.0, "MS_WVI_SPLIT": 0.0, "MS_FINE_G": 0.0}

# The harmonisation knobs that live on cf_ms_exact rather than
# cf_track_resolution. Kept in the SAME place as MS_SEVEN deliberately: the
# switches are now spread over three modules (ctr, cf_ms_exact, and the C++
# CVH_* env flags), and a closure run that silently omits one compares a
# partly-harmonised model against the simulation and reports it as physics.
# Anything added to the harmonisation list must be added here too.
MX_HARM = {"MS_CHI0_G4": True, "MS_FF_G4": True}
MX_OFF = {"MS_CHI0_G4": False, "MS_FF_G4": False}


def switch_banner():
    return ("CVH_IONI_EXACTDELTA=1  CVH_IONI_KOKOULIN=1  CVH_REF_CHARGEAWARE=1  "
            "CVH_REF_SPECIESDEDX=1   (ctr.SWITCHES_ON, in the export)\n"
            "MS_ELEC_TMAX=1 MS_ELEC_EDGE=1   MS_SNAP_YMAX=0   "
            "MS_WVI_SPLIT=1 (+MS_FINE_G=1)   (cf_track_resolution globals)\n"
            "MS_CHI0_G4=1  MS_FF_G4=1   (cf_ms_exact globals; ranks 7-8)\n"
            "NOT enabled here and NOT default-on: CVH_REF_HADRAD (hadron "
            "radiative), MS_ELEC_EDGE=2 (beyond G4's transport XS)")


def _fmt(v, w=9, p=5):
    return "".join(f"{x:{w}.{p}f}" for x in v)


def _rms(v):
    return float(np.sqrt((np.asarray(v) ** 2).mean()))


# =========================================================================
# the closure cell, with all seven and with the per-plane rows kept
# =========================================================================

_SPIN = {}          # label -> (NPERX, LG); pT = 3 comes from wvisplit


def sp_input(label):
    return _SPIN.get(label, wv.SP_INPUT[label])


def cell(pdg, rad, func, ms=True, sim_arm=None, model_rad=None, useh=None):
    """One closure cell with the seven (or, with ms=False, the four).

    The three-cache discipline of `radoff_species._rows`, plus the per-species
    Kokoulin state and the two measured MS_WVI inputs, all set explicitly.
    Returns the ladder mean AND the per-plane rows, so the outermost plane is
    available rather than inferred.

    `useh` picks the a-vector BASIS (default: `hbasis.USE_H`, which is False,
    i.e. every published number is reproduced).  With it True the
    model's sigma is the width of the LOCAL component the sim residual reports
    rather than of its curvilinear partner -- see `hbasis`.  It is restored on
    the way out like every other switch here, and it is in both scale-cache
    keys, so the two bases cannot cross-contaminate."""
    assert os.environ.get("RES_NO_PHI_CACHE"), "phi cache is LIVE"
    lab = hp.SPECIES[pdg]["label"]
    old_h = hb.USE_H
    hb.USE_H = old_h if useh is None else bool(useh)
    if model_rad is None:
        model_rad = rad
    if sim_arm is None:
        sim_arm = rs.SIM_ARM[rad]
    old = {k: getattr(ctr, k) for k in MS_SEVEN}
    old["MS_WVI_NPERX"] = ctr.MS_WVI_NPERX
    old["MS_WVI_LG"] = ctr.MS_WVI_LG
    kok = hp.kok_for(pdg)
    cpt.RAD_CHANNEL = bool(model_rad)
    ctr.IONI_KOKOULIN = 1.0 if kok else 0.0
    ctr.IONI_KOKOULIN_TCUT = 0.0
    old_mx = {k: getattr(mx, k) for k in MX_HARM}
    for k, v in (MS_SEVEN if ms else MS_OFF).items():
        setattr(ctr, k, v)
    for k, v in (MX_HARM if ms else MX_OFF).items():
        setattr(mx, k, v)
    ctr.MS_WVI_NPERX, ctr.MS_WVI_LG = sp_input(lab)
    fn._SCALE_CACHE.clear()
    cpt._PHI_CACHE.clear()
    old_load, old_store = fn._scale_cache_load, fn._scale_cache_store
    fn._scale_cache_load = lambda *a, **k: None
    fn._scale_cache_store = lambda *a, **k: None
    try:
        path = rs.mp(pdg, model_rad)
        legs = cpt.load_model(path)
        # H needs dEdxlast/refglobz/zoff (not read by load_model) and the
        # SPECIES MASS, and it must be attached here, in the parent, before
        # any pool forks. Unconditional: binding is inert while USE_H is off.
        hb.bind(legs, path, pdg=pdg)
        sim = hp.sim_of(pdg, sim_arm)
        sc = rs._scale(legs, func, path, rs.chans(model_rad))
        rows, errs, ks, ns, err = gc.closure_rows(legs, sim, func, sc["sF"])
    finally:
        hb.USE_H = old_h
        fn._scale_cache_load, fn._scale_cache_store = old_load, old_store
        cpt.RAD_CHANNEL = True
        ctr.IONI_KOKOULIN = 0.0
        ctr.IONI_KOKOULIN_TCUT = 0.0
        for k, v in old.items():
            setattr(ctr, k, v)
        for k, v in old_mx.items():
            setattr(mx, k, v)
        fn._SCALE_CACHE.clear()
        cpt._PHI_CACHE.clear()
    return dict(m=rows.mean(axis=0), err=err, out=rows[-1], outerr=errs[-1],
                rows=rows, errs=errs, ks=ks, ns=ns,
                nev=int(sim["valid"].shape[0]), sF=float(np.mean(sc["sF"])),
                model=os.path.basename(path), sim_arm=sim_arm)


def _stats(m, err):
    s = m / np.where(err > 0, err, np.nan)
    i = int(np.nanargmax(np.abs(s)))
    return dict(rms=_rms(m), maxsig=float(np.abs(s)[i]), at=float(UCURVE[i]),
                chi2=float(np.nansum(s ** 2) / len(s)),
                n3=int(np.nansum(np.abs(s) > 3)),
                signs="".join("+" if x > 0 else ("-" if x < 0 else "0")
                              for x in m))


def print_cell(lab, tagl, r, prefix="  "):
    st = _stats(r["m"], r["err"])
    so = _stats(r["out"], r["outerr"])
    print(f"{prefix}{lab:<7}{tagl:<11}" + _fmt(r["m"])
          + f" {st['rms']:9.5f} {st['maxsig']:7.2f} {st['chi2']:8.2f}")
    print(f"{prefix}{'':<7}{'+- (ladder)':<11}" + _fmt(r["err"]))
    print(f"{prefix}{'':<7}{'OUTERMOST':<11}" + _fmt(r["out"])
          + f" {so['rms']:9.5f} {so['maxsig']:7.2f} {so['chi2']:8.2f}")
    print(f"{prefix}{'':<7}{'+- (outer)':<11}" + _fmt(r["outerr"]))
    return st, so


# =========================================================================
# 1.  LIVE -- every switch, measured
# =========================================================================

def cmd_live(args):
    print("=" * 140)
    print("EVERY SWITCH, MEASURED, BEFORE ANYTHING IS INTERPRETED")
    print("=" * 140)
    print(switch_banner())
    print()

    # ---- (A) the four ionization ones: are they IN the models being used?
    print("-" * 140)
    print("(A) THE FOUR IONIZATION CORRECTIONS -- are they in the exported "
          "model this study reads?")
    print("-" * 140)
    print("A1. THE TWO REFERENCE ONES ANNOUNCE THEMSELVES.  "
          "`G4EnergyLossForExtrapolatorForCVH` prints a line for each when the")
    print("    environment variable is set, so the model's OWN cmsRun log is "
          "the record.  Grepped, per model file.")
    print()
    MARK = {"CVH_REF_CHARGEAWARE": "CVH_REF_CHARGEAWARE set",
            "CVH_REF_SPECIESDEDX": "CVH_REF_SPECIESDEDX set"}
    print(f"  {'species':<8}{'rad':<5}{'model':<36}"
          + "".join(f"{k.replace('CVH_REF_', ''):>18}" for k in MARK))
    bad = 0
    for pdg in args.pdg:
        for rad in (True, False):
            p = rs.mp(pdg, rad)
            log = p[:-5] + ".log"
            txt = open(log, errors="ignore").read() if os.path.exists(log) else ""
            cells = []
            for k, mk in MARK.items():
                on = mk in txt
                cells.append(f"{'LIVE' if on else '*** NOT SET ***':>18}")
                bad += 0 if on else 1
            print(f"  {hp.SPECIES[pdg]['label']:<8}{'ON' if rad else 'OFF':<5}"
                  f"{os.path.basename(p):<36}" + "".join(cells))
    if bad:
        raise SystemExit(f"{bad} model/reference-switch combinations are not set")

    print()
    print("A2. THE TWO IONIZATION ONES DO NOT PRINT, so they are MEASURED, "
          "and each against the pairing that ISOLATES IT.")
    print("    CVH_IONI_EXACTDELTA: `model_*_pt3_off.root` vs "
          "`model_*_pt3_on.root` -- `hadron_probe.cmd_export` sets that one")
    print("    variable and nothing else between `corr=off` and `corr=on`, so "
          "the pair is a clean single-switch test.")
    print("    (NOT `_both` vs `_all4`: `_both` was itself exported with "
          "corr='on', i.e. it ALREADY carries EXACTDELTA, so that")
    print("    pairing isolates KOKOULIN and was mis-read as an exact-delta "
          "test on the first pass here.)")
    print()
    print(f"  {'species':<8}{'branches':>9}{'identical':>10}{'moved':>7}   "
          "moved branches")
    for pdg in args.pdg:
        a, b = bp._orig_model_path(pdg, "off"), bp._orig_model_path(pdg, "on")
        if not (os.path.exists(a) and os.path.exists(b)):
            print(f"  {hp.SPECIES[pdg]['label']:<8}(missing off/on model)")
            continue
        da, db = ds._branch_digests(a), ds._branch_digests(b)
        keys = sorted(set(da) | set(db))
        diff = [k for k in keys if da.get(k) != db.get(k)]
        print(f"  {hp.SPECIES[pdg]['label']:<8}{len(keys):>9}"
              f"{len(keys)-len(diff):>10}{len(diff):>7}   "
              + (", ".join(diff) if diff else
                 "*** BIT-IDENTICAL: CVH_IONI_EXACTDELTA is INERT ***"))
        if not diff:
            raise SystemExit("CVH_IONI_EXACTDELTA is inert in the export")

    print()
    print("    and the same pairing `_both` vs `_all4`, which differs ONLY in "
          "CVH_IONI_KOKOULIN (both carry corr='on'):")
    print(f"  {'species':<8}{'branches':>9}{'identical':>10}{'moved':>7}   "
          "moved branches")
    for pdg in args.pdg:
        a, b = sd._mp(pdg, "both"), rs.mp(pdg, True)
        if not os.path.exists(a):
            print(f"  {hp.SPECIES[pdg]['label']:<8}(no `both` model)")
            continue
        da, db = ds._branch_digests(a), ds._branch_digests(b)
        keys = sorted(set(da) | set(db))
        diff = [k for k in keys if da.get(k) != db.get(k)]
        print(f"  {hp.SPECIES[pdg]['label']:<8}{len(keys):>9}"
              f"{len(keys)-len(diff):>10}{len(diff):>7}   "
              + (", ".join(diff) if diff else
                 "BIT-IDENTICAL (the |PDG|==13 Kokoulin guard)"))

    print()
    print("A3. THE KOKOULIN |PDG|==13 GUARD, in both halves.  The C++ half: "
          "`_all4` against `_all4_nokok` must move the muon")
    print("    and be BIT-IDENTICAL for every hadron.  The offline half: "
          "`ctr.IONI_KOKOULIN` must move the muon CF only.")
    print()
    print(f"  {'species':<8}{'C++ moved':>11}{'verdict (C++)':<34}"
          f"{'offline max|dphi|':>19}  verdict (offline)")
    for pdg in args.pdg:
        nk = rs.mp(pdg, True)[:-5].replace("_all4", "_all4_nokok") + ".root"
        ismu = abs(pdg) == 13
        if os.path.exists(nk):
            da, db = ds._branch_digests(nk), ds._branch_digests(rs.mp(pdg, True))
            keys = sorted(set(da) | set(db))
            diff = [k for k in keys if da.get(k) != db.get(k)]
            ok = (len(diff) > 0) if ismu else (len(diff) == 0)
            vc = ("LIVE: " + ", ".join(diff) if ismu and ok else
                  "BIT-IDENTICAL (guard is live)" if ok else "*** UNEXPECTED ***")
            nd = len(diff)
        else:
            vc, nd, ok = "(no _all4_nokok model)", -1, True
        # the offline half, at fixed everything else
        legs = cpt.load_model(rs.mp(pdg, True))
        k = len(legs) - 1
        vals = {}
        for kk in (True, False):
            ctr.IONI_KOKOULIN = 1.0 if kk else 0.0
            ctr.IONI_KOKOULIN_TCUT = 0.0
            cpt._PHI_CACHE.clear()
            fn._SCALE_CACHE.clear()
            sc = rs._scale(legs, "qop", rs.mp(pdg, True), rs.chans(True))
            phi = cpt.model_phi(legs, k, cpt.FUNCTIONALS["qop"],
                                float(sc["sF"][k]), cpt.TAU)
            vals[kk] = np.array([cpt.weier_scalar(phi, u, cpt.TAU)
                                 for u in (0.01, 0.1, 1.0)])
        ctr.IONI_KOKOULIN = 0.0
        cpt._PHI_CACHE.clear()
        fn._SCALE_CACHE.clear()
        dof = float(np.abs(vals[True] - vals[False]).max())
        oko = (dof > 0) if ismu else (dof == 0.0)
        print(f"  {hp.SPECIES[pdg]['label']:<8}{nd:>11}{vc:<34}{dof:>19.3e}  "
              + ("LIVE (muon: required)" if ismu and oko else
                 "IDENTICAL to 0 ulp (guard is live)" if oko else
                 "*** UNEXPECTED ***"))
        if not (ok and oko):
            raise SystemExit(f"Kokoulin guard unexpected for "
                             f"{hp.SPECIES[pdg]['label']}")

    # ---- (B) the three MS ones: do they MOVE the closure?
    print()
    print("-" * 140)
    print("(B) THE THREE MS HARMONISATIONS -- each one alone, on the proton "
          "`locx` cell, at fixed simulation")
    print("-" * 140)
    print("A default-OFF knob that does nothing when set is the failure mode "
          "this whole study has been burned by.")
    print("Each row is the rms of the change it makes, and the max |sigma| of "
          "that change against the cell's own error.")
    print()
    pdg = args.probe
    base = cell(pdg, True, "locx", ms=False)
    print(f"  probe species = {hp.SPECIES[pdg]['label']}, locx, radiation ON, "
          f"the four ionization ON, the MS three OFF")
    print(f"  {'knob':<44}" + UROW + "      rms  max|sig|")
    print(f"  {'BASE':<44}" + _fmt(base["m"]) + f" {_rms(base['m']):9.5f}")
    knobs = [("MS_ELEC_TMAX=1,MS_ELEC_EDGE=1", dict(MS_ELEC_TMAX=1., MS_ELEC_EDGE=1.)),
             ("MS_SNAP_YMAX=0", dict(MS_SNAP_YMAX=0.)),
             ("MS_WVI_SPLIT=1", dict(MS_WVI_SPLIT=1.)),
             ("MS_FINE_G=1", dict(MS_FINE_G=1.)),
             ("all three (+MS_FINE_G) = THE SEVEN", dict(MS_SEVEN))]
    for name, spec in knobs:
        old = {k: getattr(ctr, k) for k in MS_SEVEN}
        oldn = (ctr.MS_WVI_NPERX, ctr.MS_WVI_LG)
        for k, v in MS_OFF.items():
            setattr(ctr, k, v)
        for k, v in spec.items():
            setattr(ctr, k, v)
        ctr.MS_WVI_NPERX, ctr.MS_WVI_LG = sp_input(hp.SPECIES[pdg]["label"])
        try:
            r = _cell_raw(pdg, True, "locx")
        finally:
            for k, v in old.items():
                setattr(ctr, k, v)
            ctr.MS_WVI_NPERX, ctr.MS_WVI_LG = oldn
        d = r["m"] - base["m"]
        s = np.nanmax(np.abs(d / np.where(base["err"] > 0, base["err"], np.nan)))
        verdict = "LIVE" if _rms(d) > 0 else "*** INERT ***"
        print(f"  {name:<44}" + _fmt(d) + f" {_rms(d):9.5f} {s:9.2f}  {verdict}")
        if _rms(d) == 0.0:
            raise SystemExit(f"{name} is INERT")

    # ---- (C) the three radiation switches
    print()
    print("-" * 140)
    print("(C) THE THREE RADIATION SWITCHES (R1 census / R2 CVH_IONONLY / "
          "R3 RAD_CHANNEL)")
    print("-" * 140)
    rs.cmd_live(argparse.Namespace(pdg=args.pdg, arms=["off", "norad"]))
    return True


def _cell_raw(pdg, rad, func):
    """`cell` without the MS overlay -- the globals are whatever the caller
    set.  Used only by `live`, which is measuring one knob at a time."""
    assert os.environ.get("RES_NO_PHI_CACHE"), "phi cache is LIVE"
    kok = hp.kok_for(pdg)
    cpt.RAD_CHANNEL = bool(rad)
    ctr.IONI_KOKOULIN = 1.0 if kok else 0.0
    ctr.IONI_KOKOULIN_TCUT = 0.0
    fn._SCALE_CACHE.clear()
    cpt._PHI_CACHE.clear()
    ol, os_ = fn._scale_cache_load, fn._scale_cache_store
    fn._scale_cache_load = lambda *a, **k: None
    fn._scale_cache_store = lambda *a, **k: None
    try:
        path = rs.mp(pdg, rad)
        legs = cpt.load_model(path)
        sim = hp.sim_of(pdg, rs.SIM_ARM[rad])
        sc = rs._scale(legs, func, path, rs.chans(rad))
        rows, errs, ks, ns, err = gc.closure_rows(legs, sim, func, sc["sF"])
    finally:
        fn._scale_cache_load, fn._scale_cache_store = ol, os_
        cpt.RAD_CHANNEL = True
        ctr.IONI_KOKOULIN = 0.0
        fn._SCALE_CACHE.clear()
        cpt._PHI_CACHE.clear()
    return dict(m=rows.mean(axis=0), err=err, out=rows[-1], outerr=errs[-1])


# =========================================================================
# 2.  CONTROL -- the published rows, with the MS three OFF
# =========================================================================

# NOTES_CLOSURE_FINAL s3.1 / s3.2, all four ionization corrections ON.
PUB = {
    ("qop", 13, True):   [0.00028, 0.00025, 0.00007, -0.00026, -0.00070, -0.00081, -0.00053, -0.00035, -0.00020],
    ("qop", 13, False):  [-0.00089, -0.00114, -0.00142, -0.00164, -0.00137, -0.00058, -0.00011, -0.00001, -0.00000],
    ("qop", -13, True):  [0.00005, 0.00011, 0.00014, 0.00018, 0.00052, 0.00073, 0.00050, 0.00028, 0.00014],
    ("qop", -13, False): [-0.00035, -0.00052, -0.00073, -0.00089, -0.00078, -0.00038, 0.00002, 0.00013, 0.00012],
    ("qop", -211, True): [-0.00028, -0.00042, -0.00065, -0.00080, -0.00068, -0.00061, -0.00056, -0.00035, -0.00023],
    ("qop", 211, True):  [-0.00054, -0.00070, -0.00075, -0.00062, -0.00019, 0.00007, 0.00014, 0.00018, 0.00012],
    ("qop", -321, True): [-0.00011, -0.00019, -0.00032, -0.00046, -0.00062, -0.00075, -0.00074, -0.00059, -0.00036],
    ("qop", 321, True):  [-0.00008, -0.00015, -0.00025, -0.00039, -0.00057, -0.00071, -0.00068, -0.00048, -0.00029],
    ("qop", -2212, True): [-0.00010, -0.00020, -0.00043, -0.00072, -0.00103, -0.00117, -0.00092, -0.00040, -0.00017],
    ("qop", 2212, True): [-0.00000, -0.00003, -0.00009, -0.00018, -0.00029, -0.00031, -0.00028, -0.00024, -0.00011],
    ("locx", 13, True):  [0.00007, 0.00017, 0.00039, 0.00078, 0.00142, 0.00184, 0.00153, 0.00087, 0.00044],
    ("locx", -13, True): [0.00006, 0.00014, 0.00033, 0.00064, 0.00108, 0.00125, 0.00078, 0.00004, -0.00041],
    ("locx", -211, True): [0.00004, 0.00009, 0.00018, 0.00031, 0.00053, 0.00070, 0.00066, 0.00056, 0.00051],
    ("locx", 211, True): [0.00005, 0.00012, 0.00028, 0.00056, 0.00097, 0.00115, 0.00063, 0.00004, -0.00020],
    ("locx", -321, True): [0.00009, 0.00022, 0.00055, 0.00113, 0.00217, 0.00305, 0.00269, 0.00145, 0.00056],
    ("locx", 321, True): [0.00009, 0.00021, 0.00051, 0.00102, 0.00187, 0.00251, 0.00230, 0.00165, 0.00100],
    ("locx", -2212, True): [0.00010, 0.00023, 0.00058, 0.00123, 0.00245, 0.00357, 0.00353, 0.00246, 0.00115],
    ("locx", 2212, True): [0.00012, 0.00029, 0.00073, 0.00153, 0.00298, 0.00426, 0.00423, 0.00313, 0.00213],
}


def cmd_control(args):
    """With the MS three at their defaults this must reproduce
    NOTES_CLOSURE_FINAL s3.1 / s3.2 digit for digit.  A control that does not
    reproduce is not a control, and the muon `off` glob is the one that has
    silently failed before (the stray seed-901 sample)."""
    print("=" * 152)
    print("CONTROL: the MS three at their DEFAULTS must reproduce "
          "NOTES_CLOSURE_FINAL s3.1 / s3.2")
    print("=" * 152)
    print(f"muon `off` sim glob : {hp.sim_glob(13, 'off')}")
    print(f"                      {len(glob.glob(hp.sim_glob(13, 'off')))} files "
          f"(must be 10; the stray `_s901` sample is what the pin excludes)")
    print(f"census glob         : {hp.census_glob(13, 'off')}  "
          f"({len(glob.glob(hp.census_glob(13, 'off')))} files)")
    for pdg in (13, -13):
        n = len(glob.glob(hp.sim_glob(pdg, "norad")))
        print(f"{hp.SPECIES[pdg]['label']} `norad` : {n} files")
    print()
    nbad = 0
    for func in args.funcs:
        print(f"  {func}")
        print(f"  {'species':<8}{'rad':<5}{'source':<12}" + UROW + "      rms")
        for pdg in args.pdg:
            for rad in args.rad:
                pub = PUB.get((func, pdg, bool(rad)))
                if pub is None:
                    continue
                r = cell(pdg, bool(rad), func, ms=False)
                d = r["m"] - np.asarray(pub)
                s = d / np.where(r["err"] > 0, r["err"], np.nan)
                lab = hp.SPECIES[pdg]["label"]
                print(f"  {lab:<8}{'ON' if rad else 'OFF':<5}{'PUBLISHED':<12}"
                      + _fmt(np.asarray(pub)) + f" {_rms(pub):9.5f}")
                print(f"  {'':<8}{'':<5}{'this run':<12}" + _fmt(r["m"])
                      + f" {_rms(r['m']):9.5f}")
                print(f"  {'':<8}{'':<5}{'sigma':<12}" + _fmt(s, 9, 2)
                      + f"   max {np.nanmax(np.abs(s)):.2f}")
                if np.nanmax(np.abs(s)) > 1.0:
                    nbad += 1
                    print("  *** DOES NOT REPRODUCE ***")
        print()
    print(f"  cells failing to reproduce: {nbad}")
    if nbad:
        raise SystemExit("the control does not reproduce")


# =========================================================================
# 3.  TASK 1 -- the closure, all seven
# =========================================================================

def cmd_closure(args):
    ms = not getattr(args, "msoff", False)
    print("=" * 152)
    print("TASK 1.  THE CLOSURE WITH ALL SEVEN CORRECTIONS ON, NATIVELY."
          if ms else
          "BASELINE.  THE SAME CLOSURE WITH THE FOUR IONIZATION CORRECTIONS "
          "ONLY (the MS three at their defaults).")
    print("=" * 152)
    print(switch_banner() if ms else
          "CVH_IONI_EXACTDELTA=1  CVH_IONI_KOKOULIN=1  CVH_REF_CHARGEAWARE=1  "
          "CVH_REF_SPECIESDEDX=1\nMS_ELEC_TMAX=0  MS_ELEC_EDGE=0  "
          "MS_SNAP_YMAX=1  MS_WVI_SPLIT=0  MS_FINE_G=0   (the defaults)")
    print(f"Layered toy, pT = {hp.PT} GeV, eta {hp.ETA}, phi {hp.PHI}, cut1e4 "
          f"(9.4862 keV), 14 planes, Fisher normalization, nothing fitted.")
    print("Errors: LADDER = the per-EVENT correlated estimator "
          "(err/sqrt(nplanes) is 2.8x too small); OUTERMOST = the ordinary")
    print("per-plane error, for which no correlation correction applies.")
    print("MS_WVI_NPERX / MS_WVI_LG per species: " +
          ", ".join(f"{k} {sp_input(k)[0]:.4f}/{sp_input(k)[1]:.4f}"
                    for k in ("mu-", "p")))
    print("=" * 152)
    res = {}
    for func in args.funcs:
        print()
        print("#" * 152)
        print(f"# FUNCTIONAL: {func}")
        print("#" * 152)
        print(f"  {'species':<7}{'cell':<11}" + UROW + "      rms  max|sig| chi2/ndf")
        for pdg in args.pdg:
            lab = hp.SPECIES[pdg]["label"]
            for rad in args.rad:
                rad = bool(rad)
                if not os.path.exists(rs.mp(pdg, rad)):
                    print(f"  {lab:<7}{'rad ' + ('ON' if rad else 'OFF'):<11}"
                          "(model missing)")
                    continue
                r = cell(pdg, rad, func, ms=ms)
                res[(func, pdg, rad)] = r
                print_cell(lab, "rad " + ("ON" if rad else "OFF"), r)
            if all((func, pdg, x) in res for x in (True, False)):
                a, b = res[(func, pdg, True)], res[(func, pdg, False)]
                for key, ek in (("m", "err"), ("out", "outerr")):
                    d = b[key] - a[key]
                    de = np.hypot(a[ek], b[ek])
                    sg = d / np.where(de > 0, de, np.nan)
                    nm = "ladder" if key == "m" else "outer"
                    print(f"  {'':<7}{'OFF-ON ' + nm:<11}" + _fmt(d)
                          + f" {_rms(d):9.5f} {np.nanmax(np.abs(sg)):7.2f}")
            print()

    _summary(res, args.funcs, args.pdg, args.rad)
    if args.npz:
        d = {}
        for (func, pdg, rad), r in res.items():
            k = f"{func}|{hp.SPECIES[pdg]['label']}|{'on' if rad else 'off'}"
            for f_ in ("m", "err", "out", "outerr"):
                d[k + "|" + f_] = r[f_]
        np.savez(args.npz, u=UCURVE, **d)
        print(f"\nwrote {args.npz}")
    return res


def _summary(res, funcs, pdgs, rads):
    print()
    print("=" * 152)
    print("SUMMARY -- the closure against its OWN statistical error, LADDER "
          "MEAN and OUTERMOST PLANE")
    print("=" * 152)
    for func in funcs:
        print(f"\n  {func}")
        print(f"  {'species':<8}{'rad':<5}"
              f"{'ladder rms':>12}{'max|sig|':>10}{'at u':>7}{'chi2/ndf':>10}"
              f"{'n>3s':>6}   "
              f"{'outer rms':>11}{'max|sig|':>10}{'at u':>7}{'chi2/ndf':>10}"
              f"{'n>3s':>6}{'out/lad':>9}   signs(ladder)")
        for pdg in pdgs:
            for rad in rads:
                r = res.get((func, pdg, bool(rad)))
                if r is None:
                    continue
                a = _stats(r["m"], r["err"])
                b = _stats(r["out"], r["outerr"])
                print(f"  {hp.SPECIES[pdg]['label']:<8}"
                      f"{'ON' if rad else 'OFF':<5}"
                      f"{a['rms']:12.5f}{a['maxsig']:10.2f}{a['at']:7g}"
                      f"{a['chi2']:10.2f}{a['n3']:6d}   "
                      f"{b['rms']:11.5f}{b['maxsig']:10.2f}{b['at']:7g}"
                      f"{b['chi2']:10.2f}{b['n3']:6d}"
                      f"{b['rms']/max(a['rms'],1e-12):9.2f}   {a['signs']}")


# =========================================================================
# 4.  the pT = 40 area
# =========================================================================

AC40_AREA = os.path.join(hp.CMSSW, "allcorr40")
AC40_OUT = os.path.join(hp.SCRATCH, "ac40")
PT40_REF_PLANES = os.path.join(hp.CMSSW, "tailhunt", "pt40", "Analysis",
                               "HitAnalyzer", "test", "toyPlanes_pt40.py")


def _pm40(g):
    return "toyPlanes_pt40" if g == "qm" else "toyPlanes_pt40_qp"


def _tag40(pdg, arm, pt=None):
    """`hadron_probe.tag_of` with the pT read at CALL time.

    THE LANDMINE, and it is exactly the class of defect this study keeps
    finding.  The original is

        def tag_of(pdg, arm, pt=PT):

    and a Python default argument is bound at DEF time, so `tag_of` returns
    `pt3` forever no matter what `hadron_probe.PT` is set to.  `_env` (which
    sets TOY_PT) and `_sim_one` (which passes `pt=`) both read the module
    global at call time, so the first pT = 40 smoke run produced files whose
    PHYSICS was pT = 40 (median first-plane |p| = 41.8133 GeV against 3.1358)
    and whose NAMES said `pt3`.  Nothing would have crashed; the pT = 40 rows
    would simply have been globbed under the pT = 3 tag.  Overridden here, and
    `sim40` asserts on the tag before it writes anything."""
    return (f"{hp.SPECIES[pdg]['label'].replace('-','m').replace('+','p')}"
            f"_pt{(hp.PT if pt is None else pt):g}_{arm}")


@contextlib.contextmanager
def pt40():
    """Everything that names pT = 3 in `hadron_probe`, swapped to 40.

    `sim_path`, `model_path`, `_env` and `planes_path` read the module globals
    at CALL time; `tag_of` does NOT (see `_tag40`) and is replaced.  The two
    file caches are keyed on (pdg, arm) only and are therefore cleared on both
    edges."""
    old = (hp.PT, hp.AREA, hp.OUT, hp.planes_mod, hp.tag_of, rs.OUT)
    hp.PT, hp.AREA, hp.OUT, hp.planes_mod = 40.0, AC40_AREA, AC40_OUT, _pm40
    hp.tag_of = _tag40
    rs.OUT = AC40_OUT
    hp._SIM.clear()
    hp._CEN.clear()
    fn._MODEL_CACHE.clear()
    fn._SIM_CACHE.clear()
    cpt._PHI_CACHE.clear()
    fn._SCALE_CACHE.clear()
    try:
        yield
    finally:
        (hp.PT, hp.AREA, hp.OUT, hp.planes_mod, hp.tag_of, rs.OUT) = old
        hp._SIM.clear()
        hp._CEN.clear()
        fn._MODEL_CACHE.clear()
        fn._SIM_CACHE.clear()
        cpt._PHI_CACHE.clear()
        fn._SCALE_CACHE.clear()


def cmd_setup40(args):
    """Build the two pT = 40 areas the same way `hadron_probe setup` builds the
    pT = 3 ones: the q = -1 planes are REGENERATED and byte-compared against the
    archived `tailhunt/pt40` file, and only then is the q = +1 mirror produced
    by the same generator."""
    ref_xml = os.path.join(hp.REFAREA, "Analysis/HitAnalyzer/data/tracker.xml")
    hx = hashlib.md5(open(ref_xml, "rb").read()).hexdigest()
    # the pT = 40 area's own tracker.xml must be the same file (concentric
    # shells; the geometry is pT- and charge-independent)
    x40 = os.path.join(hp.CMSSW, "tailhunt", "pt40",
                       "Analysis/HitAnalyzer/data/tracker.xml")
    if os.path.exists(x40):
        h40 = hashlib.md5(open(x40, "rb").read()).hexdigest()
        print(f"tailhunt/pt3 tracker.xml {hx}")
        print(f"tailhunt/pt40 tracker.xml {h40}   "
              + ("IDENTICAL" if h40 == hx else "*** DIFFERENT ***"))
        if h40 != hx:
            raise SystemExit("the two archived areas do not share tracker.xml")
    hp_ref = hashlib.md5(open(PT40_REF_PLANES, "rb").read()).hexdigest()
    print(f"archived toyPlanes_pt40.py {hp_ref}")

    with pt40():
        for g in ("qm", "qp"):
            td = hp.testdir(g)
            dd = os.path.join(hp.geomdir(g), "Analysis/HitAnalyzer/data")
            os.makedirs(td, exist_ok=True)
            os.makedirs(dd, exist_ok=True)
            shutil.copy(ref_xml, os.path.join(dd, "tracker.xml"))
            h = hashlib.md5(open(os.path.join(dd, "tracker.xml"), "rb").read()).hexdigest()
            assert h == hx
            print(f"[pt40/{g}] {h}  tracker.xml   IDENTICAL to the archived area")
        mine = hp._mk_planes(-1.0, os.path.join(hp.testdir("qm"),
                                                _pm40("qm") + ".py"))
        hm = hashlib.md5(mine.encode()).hexdigest()
        print(f"[pt40/qm] {hm}  {_pm40('qm')}.py   regenerated vs archived: "
              + ("IDENTICAL" if hm == hp_ref else "*** DIFFERENT ***"))
        if hm != hp_ref:
            raise SystemExit("the plane generator does not reproduce the "
                             "archived pT = 40 planes -- the q = +1 mirror "
                             "cannot be trusted")
        b = hp._mk_planes(+1.0, os.path.join(hp.testdir("qp"),
                                             _pm40("qp") + ".py"))
        print(f"[pt40/qp] {hashlib.md5(b.encode()).hexdigest()}  "
              f"{_pm40('qp')}.py   (q = +1 mirror, same generator)")
        # the two drivers, patched exactly as hadron_probe patches them
        for g in ("qm", "qp"):
            td = hp.testdir(g)
            s = open(f"{hp.SRCTEST}/runToyGeomCheck.py").read()
            s = hp._sub1(s, r"^import FWCore\.ParameterSet\.Config as cms$",
                         "import os\nimport FWCore.ParameterSet.Config as cms",
                         "sim: import os", flags=8)
            s = hp._sub1(s, r"^process\.g4SimHits\.Physics\.DefaultCutValue = "
                            r"cms\.double\(1\.0\)$",
                         'process.g4SimHits.Physics.DefaultCutValue = cms.double(\n'
                         '    float(os.environ.get("TOY_CUT", "1.0")))\n'
                         "print('[toy] DefaultCutValue = %g cm'\n"
                         "      % process.g4SimHits.Physics.DefaultCutValue.value())",
                         "sim: production cut", flags=8)
            s = hp._sub1(s, r"^    output=cms\.string\(opts\.output\),\n\)\)$",
                         hp._HAD_BLOCK.rstrip("\n"), "sim: watcher block", flags=8)
            open(f"{td}/runToyGeomCheck.py", "w").write(s)
            m = open(f"{hp.SRCTEST}/runToyModel.py").read()
            m = hp._sub1(m, r"^import toyPlanes_pt3 as planes$",
                         "import importlib, os as _os\nplanes = importlib."
                         'import_module(_os.environ["TOY_PLANES_MOD"])',
                         "model: planes import", flags=8)
            open(f"{td}/runToyModel.py", "w").write(m)
            print(f"[pt40/{g}] wrote runToyGeomCheck.py and runToyModel.py in {td}")
    os.makedirs(AC40_OUT, exist_ok=True)


def cmd_sim40(args):
    with pt40():
        print(f"pT = {hp.PT}  area = {hp.AREA}  out = {hp.OUT}")
        t = hp.tag_of(13, "off")
        print(f"tag check: {t}   glob: {hp.sim_glob(13, 'off')}")
        assert "_pt40_" in t, f"the pT is not in the tag ({t}) -- see _tag40"
        hp.cmd_sim(argparse.Namespace(
            pdg=args.pdg, arms=args.arms, events=args.events, jobs=args.jobs,
            seed0=args.seed0, workers=args.workers, force=args.force))


def cmd_export40(args):
    with pt40():
        print(f"pT = {hp.PT}  area = {hp.AREA}  out = {hp.OUT}")
        rs.cmd_export(argparse.Namespace(pdg=args.pdg, rad=args.rad,
                                         force=args.force))


SP40_JSON = os.path.join(LOGD, "sp40.json")


def cmd_sp40(args):
    """The two MEASURED per-species inputs to `MS_WVI_SPLIT`, at pT = 40.

    `wvisplit.SP_INPUT` is a pT = 3 measurement and BOTH entries move with
    momentum: `MS_WVI_NPERX` is the delta-ray rate per g/cm^2 (~1/beta^2, and
    the muon is already ultra-relativistic at pT = 3 so this barely moves) and
    `MS_WVI_LG` is G4's own transport log per unit chi_c^2, which contains
    ln(theta_max^2/chi_a^2) and therefore moves as ln p.  Carrying the pT = 3
    numbers over would be exactly the kind of silently-wrong constant this
    study keeps finding, so both are re-measured."""
    import json
    # `float(...)`, NOT `repr(np.float64(...))`.  numpy 2 reprs a scalar as
    # `np.float64(41813.5)`, the driver parses its arguments with `atof`, and
    # `atof("np.float64(...")` is 0.0 -- so the driver ran at p = 0 and returned
    # L_g = nan, silently, into a JSON the closure then reads.  Another member
    # of the s4.1 family: a wrong CONSTANT that does not raise.
    p_mev = float(1e3 * 40.0 * np.cosh(hp.ETA))
    log = os.path.join(LOGD, "wvi_pt40.log")
    if args.run or not os.path.exists(log):
        cmd = [wv.DRIVER, "--pathlen", str(float(wv.TOY_PATH_CM)),
               "--p", repr(p_mev),
               "--ndelta", "8.444", "--n", "2000", "--nfull", "2000",
               "--seed", "20260817"]
        print(" ".join(cmd))
        with open(log, "w") as fh:
            subprocess.run(cmd, stdout=fh, check=True)
    mat, ST, MOM, HIST = wv.parse_driver(log)
    print("=" * 132)
    print(f"THE TWO MS_WVI INPUTS AT pT = 40  (p = {p_mev:.1f} MeV/c, "
          f"eta = {hp.ETA})")
    print("=" * 132)
    print(f"  {'species':<8}{'L_g (pT=40)':>13}{'L_g (pT=3)':>13}"
          f"{'NPERX (pT=40)':>15}{'NPERX (pT=3)':>14}{'sum xg':>10}"
          f"{'ioni sec/ev':>13}{'nev':>9}")
    out = {}
    with pt40():
        for pdg in args.pdg:
            lab = hp.SPECIES[pdg]["label"]
            st = wv.state_of(ST, lab, mat)
            a, fb = st["screenZ"], st["screenZ"] * st["factB"]
            Lg = float(wv.g4_f(st["omcElec"] / a, fb)
                       + st["Z"] * wv.g4_f(st["omcNuc"] / a, fb)) / (st["Z"] + 1.0)
            c = hp.census_of(pdg, "off")
            P = tp.PROCS
            nsec = c[:, tp.I_NSECPROC:tp.I_NSECPROC + len(P)].mean(axis=0)
            nion = float(nsec[P.index("muIoni")] + nsec[P.index("otherIoni")])
            legs = cpt.load_model(rs.mp(pdg, True))
            xtot = float(sum(l["ms"][:, 2].sum() for l in legs))
            nperx = nion / xtot
            if not np.isfinite(Lg) or Lg <= 0.0:
                raise SystemExit(
                    f"L_g for {lab} is {Lg} -- the driver did not run at the "
                    f"right momentum (see {log}).  A nan here would reach the "
                    f"closure as MS_WVI_LG and silently poison every cell.")
            out[lab] = (nperx, Lg)
            print(f"  {lab:<8}{Lg:>13.4f}{wv.SP_INPUT[lab][1]:>13.4f}"
                  f"{nperx:>15.4f}{wv.SP_INPUT[lab][0]:>14.4f}{xtot:>10.4f}"
                  f"{nion:>13.2f}{len(c):>9}")
    json.dump(out, open(SP40_JSON, "w"), indent=1)
    print(f"\nwrote {SP40_JSON}")
    return out


def _load_sp40():
    import json
    if os.path.exists(SP40_JSON):
        return {k: tuple(v) for k, v in json.load(open(SP40_JSON)).items()}
    raise SystemExit(f"{SP40_JSON} missing -- run `allcorr.py sp40` first")


def cmd_closure40(args):
    """TASK 2: the muon at pT = 40, both charges, radiation ON and OFF."""
    global _SPIN
    _SPIN = _load_sp40()
    with pt40():
        print(f"pT = {hp.PT}   area = {hp.AREA}   out = {hp.OUT}")
        for pdg in args.pdg:
            for arm in ("off", "norad"):
                print(f"  {hp.SPECIES[pdg]['label']:<6} {arm:<6} "
                      f"{len(glob.glob(hp.sim_glob(pdg, arm))):>3} sim files   "
                      f"model {os.path.basename(rs.mp(pdg, arm == 'off'))}")
        print(f"  MS_WVI (NPERX, L_g) at pT = 40: "
              + ", ".join(f"{k} {v[0]:.4f}/{v[1]:.4f}" for k, v in _SPIN.items()))
        cmd_closure(args)
    _SPIN = {}


# =========================================================================
# 5.  TASK 3 -- is the radiative channel HARMONISED?
# =========================================================================

import cf_brems_exact as cbe                                     # noqa: E402

RHD = os.path.join(hp.SCRATCH, "rh")
os.makedirs(RHD, exist_ok=True)
DRIVER = os.path.join(hp.SCRATCH, "radharm_g4driver.sh")
GCUT = 0.00099          # MeV; the gamma production cut the toy's own
                        # G4ProductionCutsTable prints for ToyLayerMat (and it
                        # is G4's 990 eV floor, so it is the same everywhere)
XGMIN = 1e-9            # g/cm^2 below which a radv record is a zero-thickness
                        # filler and carries no material


def _bucket_key(rec, mass_gev):
    """(Z, A, rho, ekin[MeV], len[mm]) of a radv record, rounded so that the
    ~150 records collapse onto a few dozen distinct G4 driver calls.  The
    radiative cross sections vary by <0.1 % over a 1 % change in E, so ekin is
    binned at 1 %; everything else is exact for this toy."""
    Z, A = float(rec[cbe.R_EFFZ]), float(rec[cbe.R_EFFA])
    L = float(rec[cbe.R_STEPCM])
    xg = float(rec[cbe.R_XG])
    rho = xg / L if L > 0 else 0.0
    ekin = (float(rec[cbe.R_ETOT]) - mass_gev) * 1e3
    return (round(Z, 4), round(A, 4), float(f"{rho:.6g}"),
            float(f"{ekin:.4g}"), round(L * 10.0, 6))


def _run_driver(pdg, key, n, seed, tag):
    Z, A, rho, ekin, lmm = key
    out = os.path.join(RHD, tag)
    rec = out + ".rec"
    if not os.path.exists(rec):
        cmd = [DRIVER, "--pdg", str(pdg), "--Z", repr(Z), "--A", repr(A),
               "--rho", repr(rho), "--ekin", repr(ekin), "--len", repr(lmm),
               "--gcut", repr(GCUT), "--spectrum", "--emit",
               "--n", str(n), "--seed", str(seed), "--out", out]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode or not os.path.exists(rec):
            raise SystemExit(f"driver failed for {tag}: {r.stderr[-2000:]}")
    return _parse_rec(rec, out)


def _parse_rec(path, prefix):
    d = {"spec_v": [], "spec_b": [], "spec_p": []}
    for line in open(path):
        t = line.split()
        if not t or t[0].startswith("#"):
            continue
        if t[0] == "spec":
            d["spec_v"].append(float(t[2]))
            d["spec_b"].append(float(t[3]))
            d["spec_p"].append(float(t[4]))
        elif t[0] == "dedx":
            d["dedx_brem"], d["dedx_pair"] = float(t[2]), float(t[4])
        elif t[0] == "emitrates":
            d["lam_brem"], d["lam_pair"] = float(t[2]), float(t[4])
            d["sub_brem"], d["sub_pair"] = float(t[6]), float(t[8])
            d["len_mm"] = float(t[10])
        elif t[0] == "emit":
            w = t[1]
            d[f"n_{w}"] = int(t[3])
            d[f"mean_{w}"] = float(t[5])
            d[f"var_{w}"] = float(t[7])
    for w in ("brem", "pair"):
        f = prefix + f".{w}.bin"
        d[f"eps_{w}"] = (np.fromfile(f, dtype=np.float64)
                         if os.path.exists(f) else np.zeros(0))
    for k in ("spec_v", "spec_b", "spec_p"):
        d[k] = np.asarray(d[k])
    return d


def _g4_dndv(vg, etot_gev, lam, eps_mev, nbin=400, rng=None):
    """dN/dv on the model's OWN 48-point v grid, from G4's emission samples.

    A compound Poisson's intensity measure factorises into a RATE and a
    normalised emission density, and G4 gives both directly: `lam` is
    CrossSectionPerVolume * L and `eps_mev` are draws from the model's own
    SampleSecondaries.  With eps = v E,

        dN/dv = lam p(eps) de/dv = lam p(eps) E = lam (dP/dln eps) / v

    so the estimator is a histogram in ln eps -- which is also where the
    samples are uniform (dN/deps ~ 1/eps), i.e. where the statistics are best.
    `rng`, when given, resamples the emissions for the bootstrap."""
    if lam <= 0 or len(eps_mev) == 0:
        return np.zeros_like(vg)
    e = eps_mev if rng is None else eps_mev[rng.integers(0, len(eps_mev),
                                                         len(eps_mev))]
    x = np.log(e[e > 0])
    lo, hi = x.min(), x.max()
    if hi <= lo:
        return np.zeros_like(vg)
    h, edges = np.histogram(x, bins=nbin, range=(lo, hi))
    ctr = 0.5 * (edges[1:] + edges[:-1])
    dens = h / (len(x) * (edges[1] - edges[0]))          # dP/dln eps
    lnv = np.log(np.clip(vg, 1e-300, None) * etot_gev * 1e3)
    return lam * np.interp(lnv, ctr, dens, left=0., right=0.) / np.clip(vg, 1e-300, None)


def _g4_legs(legs, mass_gev, drv, rng=None, _cache=None):
    """A copy of `legs` whose radiative block is GEANT4's own sampler.

    Only `radspec` and the two dE/dx columns the offline normalization anchors
    on are replaced; every other record, and the whole ionization and MS
    machinery, is untouched.  `cf_brems_exact.step_spectrum` then renormalizes
    the substituted shape to the substituted dE/dx exactly as it does for the
    exported one, so the two arms differ by the INTENSITY MEASURE and by
    nothing else -- the 48-point grid, the trapezoid, the transport weights and
    the centring are common."""
    # The dN/dv of a record depends only on (bucket, process, E), and a track
    # has ~150 records in ~30 buckets, so histogramming per record repeats the
    # same 4e5-sample reduction five times over.  Cached, which is what makes
    # the bootstrap affordable (the cache is per CALL, so a resampled arm never
    # sees the nominal arm's table).
    _cache = {} if _cache is None else _cache
    out = []
    for leg in legs:
        leg = dict(leg)
        recs = np.array(leg["rad"], dtype=float)
        spec = np.array(leg["radspec"], dtype=float)
        vg = leg["radvgrid"]
        for i, rec in enumerate(recs):
            k = _bucket_key(rec, mass_gev)
            d = drv.get(k)
            if d is None or rec[cbe.R_XG] < XGMIN:
                spec[i, :] = 0.
                recs[i, cbe.R_DEDXBREM] = 0.
                recs[i, cbe.R_DEDXPAIR] = 0.
                recs[i, cbe.R_DEDXRAD] = 0.
                continue
            E = float(rec[cbe.R_ETOT])
            L = float(rec[cbe.R_STEPCM])
            nb = cbe.NRADV
            for j, w in enumerate(("brem", "pair")):
                lam, eps = d[f"lam_{w}"], d[f"eps_{w}"]
                ck = (k, w, round(E, 7))
                if ck not in _cache:
                    _cache[ck] = _g4_dndv(vg, E, lam, eps, rng=rng)
                spec[i, j * nb:(j + 1) * nb] = _cache[ck]
                # the ABSOLUTE G4 mean of this process over this step, put into
                # the column step_spectrum renormalizes to (GeV/cm)
                mean = d.get(f"mean_{w}", 0.) if lam > 0 else 0.
                col = cbe.R_DEDXBREM if j == 0 else cbe.R_DEDXPAIR
                recs[i, col] = (lam * mean * 1e-3 / L) if L > 0 else 0.
            recs[i, cbe.R_DEDXRAD] = recs[i, cbe.R_DEDXBREM] + recs[i, cbe.R_DEDXPAIR]
        leg["rad"], leg["radspec"] = recs, spec
        out.append(leg)
    return out


def _ref_shift(new, old, func):
    """The REFERENCE move that goes with changing a leg's radiative MEAN.

    THE POINT, and it is not a detail.  `cf_brems_exact.rad_exponent` is
    CENTRED -- it adds `e^{ix} - 1 - ix`, i.e. the fluctuation with the mean
    removed -- and that is right for the muon because the propagator's dE/dx
    table is built with ionOnly = false and has ALREADY subtracted the
    radiative mean.  For a HADRON it has not: `computeRadiativeDEDX` returns 0,
    so the exported reference subtracts NO radiative mean at all while the
    simulation loses one.  Inserting a centred CF and stopping there therefore
    supplies the fluctuation and silently leaves the mean wrong.

    The complete substitution moves the reference by the same per-step map the
    noise uses, applied to the DIFFERENCE of the two arms' radiative means:

        d(ref . a) = sum_steps [a . A_ioni]_qop  q  cs  (dE_new - dE_old)

    -- zero by construction when the arm's mean equals the exported one (the
    muon), and the whole G4 radiative mean when the exported one is zero (every
    hadron)."""
    avec = cpt.FUNCTIONALS[func]
    out = np.zeros(len(new))
    for k in range(len(new)):
        A_ms, A_ioni, A_ms0 = cpt.step_transports(new, k)
        tot = 0.0
        for j in range(k + 1):
            rn, ro = new[j].get("rad"), old[j].get("rad")
            if rn is None or not len(rn):
                continue
            q = np.sign(new[j]["refqop"]) or 1.0
            w = q * np.einsum("i,sij->sj", avec, A_ioni[j])[:, 0]
            wr = w if len(w) == len(rn) else np.full(len(rn), float(np.mean(w)))
            dE = (rn[:, cbe.R_DEDXRAD] * rn[:, cbe.R_STEPCM]
                  - ro[:, cbe.R_DEDXRAD] * ro[:, cbe.R_STEPCM])
            tot += float(np.sum(wr * rn[:, cbe.R_CS] * dE))
        out[k] = tot
    return out


def _shifted(legs, shift, func):
    br = cpt.REF_BRANCH[func]
    return [dict(l, **{br: float(l[br]) + float(s)}) for l, s in zip(legs, shift)]


def _closure_with(legs, pdg, func, sim_arm, rad=True, ms=True):
    """The closure curve for an explicitly supplied `legs` (so the radiative
    block can be substituted), with the seven applied exactly as in `cell`."""
    lab = hp.SPECIES[pdg]["label"]
    old = {k: getattr(ctr, k) for k in MS_SEVEN}
    old["MS_WVI_NPERX"] = ctr.MS_WVI_NPERX
    old["MS_WVI_LG"] = ctr.MS_WVI_LG
    cpt.RAD_CHANNEL = bool(rad)
    ctr.IONI_KOKOULIN = 1.0 if hp.kok_for(pdg) else 0.0
    ctr.IONI_KOKOULIN_TCUT = 0.0
    for k, v in (MS_SEVEN if ms else MS_OFF).items():
        setattr(ctr, k, v)
    ctr.MS_WVI_NPERX, ctr.MS_WVI_LG = sp_input(lab)
    fn._SCALE_CACHE.clear()
    cpt._PHI_CACHE.clear()
    ol, os_ = fn._scale_cache_load, fn._scale_cache_store
    fn._scale_cache_load = lambda *a, **k: None
    fn._scale_cache_store = lambda *a, **k: None
    try:
        sim = hp.sim_of(pdg, sim_arm)
        # the tag must move with the legs, or the scale cache keys collide
        sc = fn.plane_scales(legs, func, tag=f"rh|{lab}|{func}|{id(legs)}",
                             channels=rs.chans(rad))
        rows, errs, ks, ns, err = gc.closure_rows(legs, sim, func, sc["sF"])
    finally:
        fn._scale_cache_load, fn._scale_cache_store = ol, os_
        cpt.RAD_CHANNEL = True
        ctr.IONI_KOKOULIN = 0.0
        for k, v in old.items():
            setattr(ctr, k, v)
        fn._SCALE_CACHE.clear()
        cpt._PHI_CACHE.clear()
    return dict(m=rows.mean(axis=0), err=err, out=rows[-1], outerr=errs[-1])


def _kappa2_rad(legs, k, func, sigma):
    """The radiative channel's OWN second cumulant of the standardized
    residual, computed directly.

    Not via `model_phi` with the other channels stripped: emptying `leg["ioni"]`
    makes `A_ioni[j]` empty, and `model_phi`'s `wr` fallback then takes
    `np.mean` of an empty array and returns nan -- silently, because the nan
    only shows up as a RuntimeWarning.  A compound Poisson's second cumulant is
    `int dN (a eps)^2`, so it is one line:

        kappa2 = sum_records (w cs / sigma)^2 int dv (dN/dv) (v E)^2
    """
    avec = cpt.FUNCTIONALS[func]
    A_ms, A_ioni, A_ms0 = cpt.step_transports(legs, k)
    tot = 0.0
    for j in range(k + 1):
        leg = legs[j]
        r = leg.get("rad")
        if r is None or not len(r):
            continue
        q = np.sign(leg["refqop"]) or 1.0
        w = q * np.einsum("i,sij->sj", avec, A_ioni[j])[:, 0] / sigma
        wr = w if len(w) == len(r) else np.full(len(r), float(np.mean(w)))
        vg = leg["radvgrid"]
        for i, rec in enumerate(r):
            v, dNdv = cbe.step_spectrum(rec, leg["radspec"][i], vg)
            if not np.any(dNdv > 0):
                continue
            m2 = float(np.trapezoid(dNdv * (v * rec[cbe.R_ETOT]) ** 2, v))
            tot += (wr[i] * rec[cbe.R_CS]) ** 2 * m2
    return tot


def _kappa2(legs, k, func, sigma, tiny=1e-4):
    """The model's second cumulant of the standardized residual, from its own
    CF: ln phi(tau) -> -kappa2 tau^2/2 as tau -> 0."""
    tau = np.array([0.0, tiny])
    phi = cpt.model_phi(legs, k, cpt.FUNCTIONALS[func], sigma, tau)
    return float(-2.0 * np.log(np.abs(phi[1])) / tiny ** 2)


def cmd_radharm(args):
    print("=" * 152)
    print("TASK 3.  IS THE RADIATIVE CHANNEL HARMONISED BETWEEN THE "
          "SIMULATION AND THE MODEL?")
    print("=" * 152)
    print(switch_banner())
    print("Geant4 side: `radharm_g4driver`, which links G4MuBremsstrahlungModel "
          "/ G4MuPairProductionModel (muon) and")
    print("G4hBremsstrahlungModel / G4hPairProductionModel (hadrons) and drives "
          "their OWN SampleSecondaries and")
    print("CrossSectionPerVolume at the EXPORTED step conditions -- the "
          "urban_g4driver / msterms_g4driver pattern.")
    print("=" * 152)

    # ---------------------------------------------------------- 3.0 the model
    print()
    print("-" * 152)
    print("3.0  WHAT THE MODEL HAS, per species, read off the exported record")
    print("-" * 152)
    print("`Geant4ePropagator::fillRadiativeSpectrum` and "
          "`computeRadiativeDEDX` both open with `if (abs(pdg) != 13) return;`,")
    print("so for a hadron the radiative block is not merely small -- it is "
          "absent.  Measured on every exported column:")
    print()
    print(f"  {'species':<8}{'radv recs':>10}{'max dedxrad':>14}"
          f"{'max dedxbrem':>14}{'max dedxpair':>14}{'max |radspec|':>15}"
          f"{'  sum dE_rad [MeV]':>19}   verdict")
    MODEL_RAD = {}
    for pdg in args.pdg:
        legs, vg = cbe.load_radv(rs.mp(pdg, True))
        R = np.concatenate([r for r, _ in legs])
        SP = np.concatenate([s for _, s in legs])
        de = float((R[:, cbe.R_DEDXRAD] * R[:, cbe.R_STEPCM]).sum()) * 1e3
        MODEL_RAD[pdg] = de
        has = R[:, cbe.R_DEDXRAD].max() > 0
        print(f"  {hp.SPECIES[pdg]['label']:<8}{len(R):>10}"
              f"{R[:, cbe.R_DEDXRAD].max():>14.3e}"
              f"{R[:, cbe.R_DEDXBREM].max():>14.3e}"
              f"{R[:, cbe.R_DEDXPAIR].max():>14.3e}{np.abs(SP).max():>15.3e}"
              f"{de:>19.6f}   "
              + ("has a radiative block" if has else
                 "*** NO RADIATIVE CHANNEL AT ALL ***"))

    # ------------------------------------------- 3.1 the G4 side, per species
    print()
    print("-" * 152)
    print("3.1  WHAT GEANT4 HAS, at the same step conditions")
    print("-" * 152)
    print(f"gamma production cut = {GCUT} MeV (the toy's own "
          "G4ProductionCutsTable, i.e. G4's 990 eV floor).")
    print("`lambda` is CrossSectionPerVolume(ekin, gcut, ekin) * L, i.e. the "
          "number of DISCRETE emissions the simulation makes")
    print("in one step; `<eps>` is the mean TOTAL energy of one emission "
          "(pair production also costs 2 m_e of rest mass).")
    print()
    DRV = {}
    for pdg in args.pdg:
        legs, vg = cbe.load_radv(rs.mp(pdg, True))
        mgev = hp.SPECIES[pdg]["mass"] * 1e-3
        keys = {}
        for r, _ in legs:
            for rec in r:
                if float(rec[cbe.R_XG]) < XGMIN:
                    continue
                keys[_bucket_key(rec, mgev)] = keys.get(_bucket_key(rec, mgev), 0) + 1
        lab = hp.SPECIES[pdg]["label"].replace("-", "m").replace("+", "p")
        # the G4 driver calls are independent; run them in a pool, since a
        # species has ~30 buckets and each is a few 1e5 SampleSecondaries calls
        from concurrent.futures import ThreadPoolExecutor
        items = list(enumerate(sorted(keys.items())))
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            res = list(ex.map(
                lambda t: (t[1][0],
                           _run_driver(pdg, t[1][0], args.nemit,
                                       20260817 + 97 * t[0],
                                       f"{lab}_pt{hp.PT:g}_n{args.nemit}"
                                       f"_b{t[0]:02d}")),
                items))
        drv = dict(res)
        DRV[pdg] = drv
        # the totals over the track
        tot_l = tot_e = 0.
        for k, mult in keys.items():
            d = drv[k]
            tot_l += mult * (d["lam_brem"] + d["lam_pair"])
            tot_e += mult * (d["lam_brem"] * d.get("mean_brem", 0.)
                             + d["lam_pair"] * d.get("mean_pair", 0.))
        # the bucket carrying the most material: multiplicity x rho x length
        k0 = max(keys, key=lambda kk: keys[kk] * kk[2] * kk[4])
        d0 = drv[k0]
        print(f"  {hp.SPECIES[pdg]['label']:<6} buckets {len(keys):>3}   "
              f"thickest: Z {k0[0]:g} rho {k0[2]:g} ekin {k0[3]:g} MeV "
              f"L {k0[4]:.4f} mm  x{keys[k0]}")
        print(f"         lambda_brem {d0['lam_brem']:.4e}  <eps>_brem "
              f"{d0.get('mean_brem', 0.):9.4f} MeV   "
              f"lambda_pair {d0['lam_pair']:.4e}  <eps>_pair "
              f"{d0.get('mean_pair', 0.):9.4f} MeV")
        print(f"         TRACK: {tot_l:.5e} emissions, "
              f"sum dE_rad {tot_e:.6f} MeV     "
              f"MODEL: {MODEL_RAD[pdg]:.6f} MeV   "
              + (f"ratio {tot_e/MODEL_RAD[pdg]:.4f}" if MODEL_RAD[pdg] > 0
                 else "ratio -- (the model has nothing)"))

    # ------------------------------------- 3.2 the closure, three radiative arms
    print()
    print("-" * 152)
    print("3.2  THE THREE ARMS, IN THE CLOSURE'S OWN CURRENCY")
    print("-" * 152)
    print("  NONE   the radiative channel removed from the model CF "
          "(cf_propagation_test.RAD_CHANNEL = False)")
    print("  MODEL  the exported analytic block (the production configuration)")
    print("  G4f    the SAME machinery -- same 48-point v grid, same trapezoid, "
          "same transport weights, same centring --")
    print("         with the intensity measure replaced by "
          "lambda x SampleSecondaries.  FLUCTUATION ONLY.")
    print("  G4     G4f plus the REFERENCE move that goes with it "
          "(see `_ref_shift`): the centred CF assumes the mean was already")
    print("         subtracted, which is true for the muon and FALSE for every "
          "hadron.  This is the complete substitution.")
    print("  MODEL vs G4 therefore differ by the RATE, the SHAPE and the MEAN "
          "and by nothing else.  The simulation is held")
    print("  fixed at the radiation-ON arm throughout: this is a MODEL-side "
          "comparison.")
    print()
    for func in (() if args.k2only else args.funcs):
        print(f"  {func}")
        print(f"  {'species':<7}{'arm':<8}" + UROW + "      rms   d(rms) vs MODEL")
        for pdg in args.pdg:
            lab = hp.SPECIES[pdg]["label"]
            mgev = hp.SPECIES[pdg]["mass"] * 1e-3
            base = cpt.load_model(rs.mp(pdg, True))
            g4f = _g4_legs(base, mgev, DRV[pdg])
            sh = _ref_shift(g4f, base, func)
            g4 = _shifted(g4f, sh, func)
            arms = {}
            arms["NONE"] = _closure_with(base, pdg, func, "off", rad=False)
            arms["MODEL"] = _closure_with(base, pdg, func, "off", rad=True)
            arms["G4f"] = _closure_with(g4f, pdg, func, "off", rad=True)
            arms["G4"] = _closure_with(g4, pdg, func, "off", rad=True)
            for a in ("NONE", "MODEL", "G4f", "G4"):
                r = arms[a]
                d = _rms(r["m"] - arms["MODEL"]["m"])
                print(f"  {lab:<7}{a:<8}" + _fmt(r["m"])
                      + f" {_rms(r['m']):9.5f} {d:12.5f}")
            print(f"  {'':<7}{'+-':<8}" + _fmt(arms["MODEL"]["err"]))
            print(f"  {'':<7}{'ref shift (raw ' + func + ' units): plane 0 '}"
                  f"{sh[0]:.4e}   outermost {sh[-1]:.4e}")
            chan = _rms(arms["MODEL"]["m"] - arms["NONE"]["m"])
            harm = _rms(arms["MODEL"]["m"] - arms["G4"]["m"])
            # the bootstrap on the G4 arm
            boots = []
            for b in range(args.nboot):
                rng = np.random.default_rng(4242 + b)
                gb = _g4_legs(base, mgev, DRV[pdg], rng=rng)
                gb = _shifted(gb, _ref_shift(gb, base, func), func)
                boots.append(_closure_with(gb, pdg, func, "off", rad=True)["m"])
            bs = np.std(np.array(boots), axis=0, ddof=1) if boots else None
            mc = _rms(bs) if bs is not None else float("nan")
            print(f"  {'':<7}{'channel |MODEL-NONE|':<28}{chan:9.5f}"
                  f"   harmonisation error |MODEL-G4| {harm:9.5f}"
                  f"   MC {mc:9.5f}"
                  + (f"   cancel to {100*(1-harm/chan):6.2f} %"
                     if chan > 0 else "   (no model channel: the whole thing "
                                      "is missing)"))
            print()

    # --------------------------------- 3.3 the hadrons: kappa2 of the missing bit
    print()
    print("-" * 152)
    print("3.3  THE HADRONS: THE MISSING CHANNEL AS A FRACTION OF THE MODEL's "
          "OWN kappa2")
    print("-" * 152)
    print("kappa2 is read off the model's own CF as -2 ln|phi(tau)| / tau^2 at "
          "tau -> 0, at the OUTERMOST plane, in the")
    print("standardized variable the closure works in.  `rad` is the same "
          "quantity with ONLY the substituted G4 radiative")
    print("block present, so the ratio is the fraction of the total second "
          "cumulant the model does not represent.")
    print()
    print(f"  {'species':<8}{'func':<6}{'kappa2 model':>15}{'kappa2 rad(G4)':>17}"
          f"{'fraction':>12}{'  ':<2}{'sum dE_rad [MeV]':>18}")
    for pdg in args.pdg:
        mgev = hp.SPECIES[pdg]["mass"] * 1e-3
        base = cpt.load_model(rs.mp(pdg, True))
        g4 = _g4_legs(base, mgev, DRV[pdg])
        k = len(base) - 1
        for func in args.funcs:
            ctr.IONI_KOKOULIN = 1.0 if hp.kok_for(pdg) else 0.0
            for key, v in MS_SEVEN.items():
                setattr(ctr, key, v)
            ctr.MS_WVI_NPERX, ctr.MS_WVI_LG = sp_input(hp.SPECIES[pdg]["label"])
            fn._SCALE_CACHE.clear()
            cpt._PHI_CACHE.clear()
            ol, os_ = fn._scale_cache_load, fn._scale_cache_store
            fn._scale_cache_load = lambda *a, **kk: None
            fn._scale_cache_store = lambda *a, **kk: None
            try:
                cpt.RAD_CHANNEL = True
                sc = fn.plane_scales(base, func, tag=f"k2|{pdg}|{func}",
                                     channels=rs.chans(True))
                sig = float(sc["sF"][k])
                k2_tot = _kappa2(base, k, func, sig)
                k2_rad = _kappa2_rad(g4, k, func, sig)
            finally:
                fn._scale_cache_load, fn._scale_cache_store = ol, os_
                cpt.RAD_CHANNEL = True
                ctr.IONI_KOKOULIN = 0.0
                for key, v in MS_OFF.items():
                    setattr(ctr, key, v)
                fn._SCALE_CACHE.clear()
                cpt._PHI_CACHE.clear()
            print(f"  {hp.SPECIES[pdg]['label']:<8}{func:<6}{k2_tot:15.6e}"
                  f"{k2_rad:17.6e}{k2_rad/k2_tot:12.3e}"
                  f"{'  ':<2}{MODEL_RAD[pdg]:18.6f}")
    return True


def cmd_radscan(args):
    """The momentum at which the hadrons' missing radiative channel stops
    being negligible.

    Only the RATE and the SPECTRUM move with momentum; the geometry does not.
    So the scan is done on the SAME one-layer step (Z = 8, A = 16, rho = 9,
    L = 1.0625 mm, the toy's own ToyLayerMat crossing) with the kinetic energy
    swept, and the answer is expressed as the radiative second cumulant of the
    step's energy loss against the ionization one -- the ratio the closure's
    kappa2 fraction is built from, with the transport weights (common to both
    channels, since both are q/p-only) divided out."""
    print("=" * 152)
    print("TASK 3, THE MOMENTUM SCAN.  Where does the hadrons' missing "
          "radiative channel stop being negligible?")
    print("=" * 152)
    print("One ToyLayerMat crossing (Z = 8, A = 16, rho = 9 g/cm3, "
          "L = 1.0625 mm), Geant4's own hBrems + hPairProd.")
    print("kappa2_rad = lambda <eps^2> (a compound Poisson), from "
          "CrossSectionPerVolume and SampleSecondaries.")
    print("It is compared against the IONIZATION kappa2 of the same step, "
          "xi Tmax (the Landau second moment), so the")
    print("ratio is the fraction of the step's q/p variance the model does "
          "not have.  Both channels reach q/p through")
    print("the same cs = E/p^3, so the transport weight cancels exactly in "
          "the ratio.")
    print()
    ME = 0.51099895e-3          # GeV
    TWOPI_MC2_RCL2 = 0.1535e-3 * 2.0   # MeV cm2/g x 2 -> see below
    print(f"  {'species':<7}{'pT [GeV]':>9}{'p [GeV]':>9}{'ekin [MeV]':>12}"
          f"{'lam_brem':>11}{'lam_pair':>11}{'<eps> [MeV]':>13}"
          f"{'k2_rad [MeV^2]':>16}{'k2_ioni [MeV^2]':>17}{'k2_rad/k2_ioni':>16}")
    for pdg in args.pdg:
        m = hp.SPECIES[pdg]["mass"] * 1e-3          # GeV
        for pt in args.pt:
            p = pt * np.cosh(hp.ETA)
            etot = np.sqrt(p * p + m * m)
            ekin = (etot - m) * 1e3                 # MeV
            beta = p / etot
            gam = etot / m
            key = (8.0, 16.0, 9.0, float(f"{ekin:.4g}"), 1.0625)
            lab = hp.SPECIES[pdg]["label"].replace("-", "m").replace("+", "p")
            d = _run_driver(pdg, key, args.nemit, 909090,
                            f"scan_{lab}_pt{pt:g}")
            k2 = 0.
            emean = 0.
            for w in ("brem", "pair"):
                lam = d[f"lam_{w}"]
                if lam <= 0 or len(d[f"eps_{w}"]) == 0:
                    continue
                e = d[f"eps_{w}"]
                k2 += lam * float((e ** 2).mean())
                emean += lam * float(e.mean())
            # the ionization second moment of the same step: xi Tmax, with
            # xi = 0.1535 (Z/A) rho L / beta^2 MeV  (PDG 34.12)
            xg = 9.0 * 0.10625                       # g/cm2
            xi = 0.1535 * (8.0 / 16.0) * xg / (beta * beta)
            rat = ME / m
            tmax = 2. * ME * (beta * gam) ** 2 / (1. + 2. * gam * rat + rat * rat)
            k2i = xi * tmax * 1e3                    # MeV^2
            print(f"  {hp.SPECIES[pdg]['label']:<7}{pt:9g}{p:9.3f}{ekin:12.1f}"
                  f"{d['lam_brem']:11.3e}{d['lam_pair']:11.3e}"
                  f"{(emean/max(d['lam_brem']+d['lam_pair'],1e-300)):13.4f}"
                  f"{k2:16.4e}{k2i:17.4e}{k2/k2i:16.3e}")
        print()
    return True


# =========================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    s = p.add_subparsers(dest="cmd", required=True)

    q = s.add_parser("live")
    q.add_argument("--pdg", type=int, nargs="+", default=ORDER8)
    q.add_argument("--probe", type=int, default=2212)
    q.set_defaults(f=cmd_live)

    q = s.add_parser("control")
    q.add_argument("--pdg", type=int, nargs="+", default=ORDER8)
    q.add_argument("--rad", type=int, nargs="+", default=[1, 0])
    q.add_argument("--funcs", nargs="+", default=["qop", "locx"])
    q.set_defaults(f=cmd_control)

    q = s.add_parser("closure")
    q.add_argument("--pdg", type=int, nargs="+", default=ORDER8)
    q.add_argument("--rad", type=int, nargs="+", default=[1, 0])
    q.add_argument("--funcs", nargs="+", default=["qop", "locx"])
    q.add_argument("--npz", default=None)
    q.add_argument("--msoff", action="store_true",
                   help="the four ionization corrections only -- the baseline "
                        "the MS three are judged against")
    q.set_defaults(f=cmd_closure)

    q = s.add_parser("setup40")
    q.set_defaults(f=cmd_setup40)

    q = s.add_parser("sim40")
    q.add_argument("--pdg", type=int, nargs="+", default=[13, -13])
    q.add_argument("--arms", nargs="+", default=["off", "norad"])
    q.add_argument("--events", type=int, default=20000)
    q.add_argument("--jobs", type=int, default=10)
    q.add_argument("--seed0", type=int, default=101)
    q.add_argument("--workers", type=int, default=40)
    q.add_argument("--force", action="store_true")
    q.set_defaults(f=cmd_sim40)

    q = s.add_parser("export40")
    q.add_argument("--pdg", type=int, nargs="+", default=[13, -13])
    q.add_argument("--rad", type=int, nargs="+", default=[1, 0])
    q.add_argument("--force", action="store_true")
    q.set_defaults(f=cmd_export40)

    q = s.add_parser("sp40")
    q.add_argument("--pdg", type=int, nargs="+", default=[13, -13])
    q.add_argument("--run", action="store_true")
    q.set_defaults(f=cmd_sp40)

    q = s.add_parser("closure40")
    q.add_argument("--pdg", type=int, nargs="+", default=[13, -13])
    q.add_argument("--rad", type=int, nargs="+", default=[1, 0])
    q.add_argument("--funcs", nargs="+", default=["qop", "locx"])
    q.add_argument("--npz", default=None)
    q.add_argument("--msoff", action="store_true")
    q.set_defaults(f=cmd_closure40)

    q = s.add_parser("radharm")
    q.add_argument("--pdg", type=int, nargs="+", default=ORDER8)
    q.add_argument("--funcs", nargs="+", default=["qop", "locx"])
    q.add_argument("--nemit", type=int, default=400000)
    q.add_argument("--nboot", type=int, default=3)
    q.add_argument("--jobs", type=int, default=16)
    q.add_argument("--k2only", action="store_true",
                   help="skip s3.2 (the closure arms) and print s3.0/3.1/3.3 "
                        "only -- the driver outputs are cached")
    q.set_defaults(f=cmd_radharm)

    q = s.add_parser("radscan")
    q.add_argument("--pdg", type=int, nargs="+", default=[13, -211, -321, 2212])
    q.add_argument("--pt", type=float, nargs="+",
                   default=[3, 10, 40, 100, 300, 1000])
    q.add_argument("--nemit", type=int, default=400000)
    q.set_defaults(f=cmd_radscan)

    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
