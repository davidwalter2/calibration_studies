#!/usr/bin/env python3
"""Radiation OFF, with all four energy-loss corrections ON, for all eight species.

THE CELL THAT HAS NEVER BEEN MEASURED.  NOTES_RADOFF2 filled the 2x2
(radiation x exact-delta) and its addendum re-filled it with Kokoulin, but both
were run on ONE species (mu-) and with the two REFERENCE corrections OFF --
`CVH_REF_CHARGEAWARE` and `CVH_REF_SPECIESDEDX` did not exist when the first was
written and were not enabled in the second.  NOTES_HADRONS and
NOTES_SPECIESDEDX then measured all eight species, but only with radiation ON.
Nobody has run radiation OFF together with the reference corrections, and that
is the configuration the calibration would actually ship.

So the matrix here is

      8 species  x  {radiation ON, radiation OFF}  x  {qop, locx}

with ALL FOUR corrections set EXPLICITLY to 1 in every cell (they are
default-OFF again since the 2026-08-16 flip was reverted, so "explicitly" is
now the only way they come on), Fisher normalization, the full nine-probe u
curve, 200k events per cell, and nothing fitted.

--------------------------------------------------------------------------
THE THREE-SWITCH DISCIPLINE, and the one place it does not apply
--------------------------------------------------------------------------

NOTES_RADOFF s1 established at real cost that two of the three obvious ways to
switch radiation off in CMSSW are silent no-ops (`G4Commands` is inert; only
`ProcessActivationWatcher` compiled INTO the Simulation biglib works), and that
each half of the switch alone is enormous.  Radiation off therefore means three
switches, and each is shown live by measurement before anything is read:

  (R1) SIMULATION   arm `norad` -> `ProcessActivationWatcher` deactivates
                    muBrems + muPairProd (hBrems + hPairProd for hadrons).
                    Verified by the step census: a deactivated process must
                    define EXACTLY ZERO steps, i.e. must not appear at all.
  (R2) REFERENCE    `CVH_IONONLY=1` -> the extrapolator's dE/dx table drops the
                    radiative mean.  Verified by the shift in `refp`.
  (R3) MODEL CF     `cf_propagation_test.RAD_CHANNEL = False` -> the radiative
                    term leaves the model characteristic function, AND the
                    Fisher scale's channel set moves with it (1/I is the
                    inversion of the same CF the closure compares against).

**FOR A HADRON, (R2) AND (R3) ARE NO-OPS, AND THAT IS A RESULT, NOT AN
OVERSIGHT.**  `G4ePropagationExport`'s `radv` record carries the propagator's
own `dedxrad / dedxbrem / dedxpair`, and for pi, K, p and pbar all three are
EXACTLY ZERO (NOTES_HADRONS s9): the exported hadron model has no radiative
block at all.  So

  * the model CF is bit-identical with `RAD_CHANNEL` on and off,
  * `CVH_IONONLY` cannot remove a radiative mean that is not in the table,

and `radoff_species.py radid` MEASURES both rather than asserting them.  The
consequence is that for a hadron "radiation off" is a ONE-SIDED switch that
makes the SIMULATION consistent with the model, rather than making the model
consistent with the simulation -- which means the hadron rad-OFF cell is the
MORE consistent of the two, and the rad-ON cell (the one NOTES_HADRONS
published) is the one carrying an unmodelled channel.  For the muon it is the
other way round.  Both are reported; neither is silently called a null.

--------------------------------------------------------------------------
STATISTICAL ERRORS
--------------------------------------------------------------------------

Every closure is reported with the per-EVENT correlated error from
`geom_closure.closure_rows`.  `err_plane / sqrt(nplanes)` is WRONG here by a
factor 2.8 (NOTES_GEOMCLOSURE s1): every plane is evaluated on the SAME events,
so a ray that scattered early is displaced at every later plane and the
per-plane statistics are strongly positively correlated.  The estimator used is
the error of the per-EVENT plane average, validated against a 400-resample
bootstrap.  Significances are quoted per probe as well as on the rms, because
the rms hides sign changes and the u-shape has repeatedly been the
discriminator that excluded a candidate.

--------------------------------------------------------------------------
KOKOULIN, and why "all four ON" is species-correct by construction
--------------------------------------------------------------------------

Geant4 puts the Kokoulin factor in `G4MuBetheBlochModel` and NOWHERE else;
hadrons are ionized by `G4hIonisation` / `G4BetheBlochModel`, which has no such
factor (measured against Geant4's own `CrossSectionPerVolume` in
NOTES_HADRONS s3.1).  BOTH halves of the switch now carry a `|PDG| == 13`
guard -- the C++ in `G4UniversalFluctuationForExtrapolator.cc` and the offline
`cf_track_resolution._kokoulin_exponent` since NOTES_BARKAS s9.1 -- so setting
`CVH_IONI_KOKOULIN=1` for all eight species is species-correct rather than
wrong for six of them.  `radid` measures the C++ guard (a hadron export with
the switch on must be BIT-IDENTICAL) and `_rows` drives the offline half
EXPLICITLY from `hp.kok_for` rather than trusting either default.

Code: this file only.  Nothing in CMSSW is edited and no library is rebuilt.
"""

import argparse
import glob
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cf_propagation_test as cpt                                # noqa: E402
import cf_track_resolution as ctr                                # noqa: E402
import deltaspec as ds                                           # noqa: E402
import fisher_norm as fn                                         # noqa: E402
import geom_closure as gc                                        # noqa: E402
import hadron_probe as hp                                        # noqa: E402
# `speciesdedx` pulls in `barkas_probe` (which installs the MODEL_SUFFIX shim on
# hp.model_path) and `pion_probe` (which installs the `_s1*` SEED PIN on
# hp.sim_glob).  Both are needed here and neither is re-implemented.
import speciesdedx as sd                                         # noqa: E402
import barkas_probe as bp                                        # noqa: E402
from wums import logging as _wums_logging                        # noqa: E402

logger = _wums_logging.child_logger(__name__)

OUT = hp.OUT                       # shares NOTES_HADRONS' model/sim directory
LOGD = os.path.join(hp.SCRATCH, "rs")
os.makedirs(LOGD, exist_ok=True)

# THE SEED PIN, asserted rather than assumed.  NOTES_PION s1: the NOTES_BARKAS
# work added an eleventh mu- `off` seed (`mum_pt3_off_s901`), so the
# unrestricted glob returns 220 000 events for the muon against 200 000 for
# every hadron and the muon control comes back 0.00219 against the published
# 0.00217.  A control that does not reproduce is not a control.
assert "_s1" in hp.sim_glob(13, "off"), "the seed pin is not installed"

ORDER8 = bp.ORDER8                 # [13, -13, -211, 211, -321, 321, -2212, 2212]

# ------------------------------------------------------------------- the arms
#
# `rad` is the physical state; it selects BOTH the simulation arm and the model.
#
#   rad ON   sim arm `off`   (nuclear + Decay off, radiation ON)
#   rad OFF  sim arm `norad` (nuclear + Decay off, radiation OFF)
#
# The model side is all four corrections explicitly ON, plus CVH_IONONLY for
# the rad-OFF model.  `SWITCHES_ON` is `cf_track_resolution`'s own overlay, so
# the list of "the four" lives in exactly one place.
SIM_ARM = {True: "off", False: "norad"}
MODEL_ARM = {True: "all4", False: "all4_ionly"}
MODEL_ENV = {
    True:  dict(ctr.SWITCHES_ON),
    False: dict(ctr.SWITCHES_ON, CVH_IONONLY="1"),
}
# the control model: all four ON, radiation ON, but exported with CVH_IONONLY
# anyway -- used ONLY by `radid`, to show the reference switch is a no-op for a
# hadron.  (For a muon it must move, which is the same test's positive control.)
ARM_SUFFIX = {k: "_" + v for k, v in MODEL_ARM.items()}

UCURVE = fn.UCURVE
UROW = "".join(f"{u:>9g}" for u in UCURVE)


def mp(pdg, rad):
    """Model path for one species in one radiation state."""
    return bp._orig_model_path(pdg, "on")[:-5] + ARM_SUFFIX[rad] + ".root"


def _fmt(v, w=9, p=5):
    return "".join(f"{x:{w}.{p}f}" for x in v)


def _rms(v):
    return float(np.sqrt((np.asarray(v) ** 2).mean()))


# =========================================================================
# 1.  export -- 8 species x 2 radiation states, all four corrections ON
# =========================================================================

def cmd_export(args):
    """Export the 16 models.

    Everything goes through `hadron_probe.cmd_export`, whose `_run` funnel
    already pins the four corrections to their historical state as a BASE and
    lets the arm's overlay win (`deltaspec._clean_env`).  Passing all four as
    "1" therefore turns them on explicitly and visibly, in the one place a run
    log will show them."""
    for rad in args.rad:
        bp.MODEL_SUFFIX = ARM_SUFFIX[rad]
        sd.SD_ENV = dict(MODEL_ENV[rad])
        print(f"\n=== models, radiation {'ON' if rad else 'OFF'}   "
              f"suffix={bp.MODEL_SUFFIX!r}   env={sd.SD_ENV}")
        ns = argparse.Namespace(pdg=args.pdg, corr=["on"], kok=False,
                                force=args.force)
        hp.cmd_export(ns)
    bp.MODEL_SUFFIX, sd.SD_ENV = "", {}


def _export_one(pdg, suffix, env, force=False):
    """One extra model, for the `radid` controls."""
    bp.MODEL_SUFFIX = suffix
    sd.SD_ENV = dict(env)
    try:
        hp.cmd_export(argparse.Namespace(pdg=[pdg], corr=["on"], kok=False,
                                         force=force))
        return bp._orig_model_path(pdg, "on")[:-5] + suffix + ".root"
    finally:
        bp.MODEL_SUFFIX, sd.SD_ENV = "", {}


# =========================================================================
# 2.  sim -- the missing `norad` simulations
# =========================================================================

def cmd_sim(args):
    """The `norad` arm for whichever species do not have it.

    NOTES_HADRONS ran `norad` for its default `ORDER` only -- mu-, pi-, K-,
    pbar, p -- so the three POSITIVE partners (mu+, pi+, K+) have no rad-off
    sample.  They are what this produces.  `hadron_probe.cmd_sim` takes an
    exclusive flock (a `nohup` that outlived its shell once wrote the same
    paths twice) and counts a job as cached only if the census watcher printed
    the FULL record count, because cmsRun's TFileService creates the output at
    BeginJob and a job killed at event 1 leaves a 460-byte "valid" ROOT file."""
    have, want = [], []
    for pdg in args.pdg:
        n = len(glob.glob(hp.sim_glob(pdg, "norad")))
        (have if n >= args.jobs else want).append((pdg, n))
    for pdg, n in have:
        print(f"  {hp.SPECIES[pdg]['label']:<6} norad: {n} files, skipping")
    if not want:
        print("nothing to do")
        return
    print("producing: " + ", ".join(f"{hp.SPECIES[p]['label']} ({n} files)"
                                    for p, n in want))
    hp.cmd_sim(argparse.Namespace(
        pdg=[p for p, _ in want], arms=["norad"], events=args.events,
        jobs=args.jobs, seed0=args.seed0, workers=args.workers,
        force=args.force))


# =========================================================================
# 3.  live -- the three switches, measured
# =========================================================================

def cmd_live(args):
    """(R1) the step census, per species and arm.

    A deactivated process must define EXACTLY ZERO steps; absence from the
    census table IS zero, because the watcher prints only processes it saw.
    `hadron_probe.cmd_live` raises rather than prints if any requested-inactive
    process defined a step, so a silent no-op switch cannot get past here --
    which is the whole point, given that two of the three obvious ways to do
    this in CMSSW ARE silent no-ops."""
    hp.cmd_live(argparse.Namespace(pdg=args.pdg, arms=args.arms))

    print()
    print("=" * 108)
    print("(R1) SUMMARY: the radiative processes, rad ON vs rad OFF, summed "
          "over the ten seeds")
    print("=" * 108)
    print("The log set is derived from the PINNED sim glob, not from "
          "`sim_path(...).replace('_s0_','_s*_')`.  The latter is what")
    print("`hadron_probe.cmd_live` uses and it is NOT pinned: it picks up the "
          "stray `mum_pt3_off_s901` file (NOTES_PION s1),")
    print("which inflates the mu- `off` step counts by exactly 11/10 and would "
          "have shown as a spurious -8.7 % `muIoni` move")
    print("across the radiation switch, against +0.05 % for mu+.  That does "
          "NOT affect cmd_live's zero-check above -- zero is")
    print("zero however many logs are summed -- but it does affect any COUNT, "
          "and this table is counts.")
    print("=" * 108)
    print(f"  {'species':<8}{'arm':<7}{'logs':>6}" + "".join(f"{n:>14}" for n in
          ("brems", "pairprod", "ioni", "CoulombScat")))
    for pdg in args.pdg:
        for arm in args.arms:
            logs = sorted(f[:-5] + ".log"
                          for f in glob.glob(hp.sim_glob(pdg, arm)))
            if not logs:
                print(f"  {hp.SPECIES[pdg]['label']:<8}{arm:<7}"
                      f"{0:>6}  (no logs)")
                continue
            tot = {}
            for lg in logs:
                for k, v in hp.steps_from_log(lg).items():
                    tot[k] = tot.get(k, 0) + v[0]
            rb, pp_ = hp.RADPROC[pdg]
            ion = "muIoni" if abs(pdg) == 13 else "hIoni"
            cells = []
            for nm in (rb, pp_, ion, "CoulombScat"):
                v = tot.get(nm)
                cells.append("    (absent=0)" if v is None else f"{v:>14d}")
            print(f"  {hp.SPECIES[pdg]['label']:<8}{arm:<7}{len(logs):>6}"
                  + "".join(cells))


def cmd_radid(args):
    """(R2) and (R3): are the two MODEL-side switches live, per species?

    THIS IS THE CHECK THAT KEEPS THE HADRON ROWS HONEST.  For a hadron the
    exported model has `dedxrad = dedxbrem = dedxpair = 0` identically, so
    neither `CVH_IONONLY` (which removes a radiative MEAN from the reference
    table) nor `RAD_CHANNEL` (which removes a radiative TERM from the model CF)
    can do anything.  Both are measured:

      (R2)  export the same species twice, with and without CVH_IONONLY, and
            compare all 27 branches.  MUON: must move.  HADRON: must be
            BIT-IDENTICAL.
      (R3)  evaluate the model's own <e^{-u z^2}> with RAD_CHANNEL True and
            False.  MUON: must move.  HADRON: must be identical to 0 ulp.

    A hadron row that came back "no change with radiation off" would otherwise
    be indistinguishable from a switch that never fired."""
    print("=" * 118)
    print("(R2) THE REFERENCE SWITCH `CVH_IONONLY`, per species, at branch "
          "level")
    print("=" * 118)
    print("Radiation ON model vs the SAME model exported with CVH_IONONLY=1.  "
          "The muon must move (it has a")
    print("radiative mean in its table); every hadron must be BIT-IDENTICAL "
          "(there is nothing to remove).")
    print()
    print(f"  {'species':<8}{'branches':>10}{'identical':>11}{'moved':>7}"
          f"   {'d(refp) outermost [MeV]':>24}   verdict")
    r2 = {}
    for pdg in args.pdg:
        a, b = mp(pdg, True), mp(pdg, False)
        if not (os.path.exists(a) and os.path.exists(b)):
            print(f"  {hp.SPECIES[pdg]['label']:<8}(missing model)")
            continue
        da, db = ds._branch_digests(a), ds._branch_digests(b)
        keys = sorted(set(da) | set(db))
        diff = [k for k in keys if da.get(k) != db.get(k)]
        la, lb = cpt.load_model(a), cpt.load_model(b)
        # `refp` is in GeV.  Printed in MeV so it can be read straight against
        # NOTES_RADOFF2 s1 (R2), which quotes the pT = 3 radiative mean as
        # +0.02356 MeV -- and against the reference table's own radiative mean.
        drefp = 1e3 * float(lb[-1]["refp"] - la[-1]["refp"])
        rad = abs(pdg) == 13
        ok = (len(diff) > 0) if rad else (len(diff) == 0)
        r2[pdg] = (len(diff), drefp)
        print(f"  {hp.SPECIES[pdg]['label']:<8}{len(keys):>10}"
              f"{len(keys)-len(diff):>11}{len(diff):>7}   {drefp:>24.5f}   "
              + ("LIVE (muon: required)" if rad and ok else
                 "BIT-IDENTICAL (hadron: no radiative mean)" if ok else
                 "*** UNEXPECTED ***"))
        if not ok:
            raise SystemExit(f"(R2) unexpected for {hp.SPECIES[pdg]['label']}")

    print()
    print("=" * 118)
    print("(R3) THE MODEL-CF SWITCH `RAD_CHANNEL`, per species")
    print("=" * 118)
    print("The model's own <e^{-u z^2}> at the outermost plane, computed with "
          "the radiative term in and out.")
    print("The muon must move; every hadron must be identical to 0 ulp "
          "(dedxrad == 0, so the term is not there).")
    print()
    print(f"  {'species':<8}{'u=0.01':>12}{'u=0.1':>12}{'u=1':>12}"
          f"{'max|diff|':>12}   verdict")
    for pdg in args.pdg:
        a = mp(pdg, True)
        if not os.path.exists(a):
            continue
        legs = cpt.load_model(a)
        vals = {}
        k = len(legs) - 1
        for sw in (True, False):
            cpt.RAD_CHANNEL = sw
            cpt._PHI_CACHE.clear()
            fn._SCALE_CACHE.clear()
            try:
                # The scale is held FIXED at the radiation-ON one on purpose:
                # this row is asking whether the CHANNEL is in the CF, not
                # whether the Fisher inversion moved.  Letting s_F move too
                # would confound the two, and the muon would "pass" even if
                # the channel were absent.
                sc = _scale(legs, "qop", a, chans(True))
                phi = cpt.model_phi(legs, k, cpt.FUNCTIONALS["qop"],
                                    float(sc["sF"][k]), cpt.TAU)
                vals[sw] = np.array([cpt.weier_scalar(phi, u, cpt.TAU)
                                     for u in (0.01, 0.1, 1.0)])
            finally:
                cpt.RAD_CHANNEL = True
                cpt._PHI_CACHE.clear()
                fn._SCALE_CACHE.clear()
        d = np.abs(vals[True] - vals[False])
        rad = abs(pdg) == 13
        ok = (d.max() > 0) if rad else (d.max() == 0.0)
        print(f"  {hp.SPECIES[pdg]['label']:<8}"
              + "".join(f"{x:12.6f}" for x in vals[True])
              + f"{d.max():12.3e}   "
              + ("LIVE (muon: required)" if rad and ok else
                 "IDENTICAL to 0 ulp (hadron: dedxrad == 0)" if ok else
                 "*** UNEXPECTED ***"))
        if not ok:
            raise SystemExit(f"(R3) unexpected for {hp.SPECIES[pdg]['label']}")

    print()
    print("=" * 118)
    print("THE C++ KOKOULIN GUARD: CVH_IONI_KOKOULIN=1 must be INERT for every "
          "hadron")
    print("=" * 118)
    print("All four corrections are set to 1 in every cell of this note.  "
          "Geant4 puts the Kokoulin factor in")
    print("G4MuBetheBlochModel only, so for six of the eight species that "
          "setting must do nothing -- which is a")
    print("property of the C++ guard `|PDG| == 13`, measured here rather than "
          "read off the source.")
    print()
    print(f"  {'species':<8}{'branches':>10}{'identical':>11}{'moved':>7}   "
          "verdict")
    for pdg in args.pdg:
        base = _export_one(pdg, "_all4_nokok",
                           {k: v for k, v in MODEL_ENV[True].items()
                            if k != "CVH_IONI_KOKOULIN"} |
                           {"CVH_IONI_KOKOULIN": "0"}, force=args.force)
        da, db = ds._branch_digests(base), ds._branch_digests(mp(pdg, True))
        keys = sorted(set(da) | set(db))
        diff = [k for k in keys if da.get(k) != db.get(k)]
        rad = abs(pdg) == 13
        ok = (len(diff) > 0) if rad else (len(diff) == 0)
        print(f"  {hp.SPECIES[pdg]['label']:<8}{len(keys):>10}"
              f"{len(keys)-len(diff):>11}{len(diff):>7}   "
              + (f"LIVE, moved: {', '.join(diff)}" if rad and ok else
                 "BIT-IDENTICAL (guard is live)" if ok else
                 "*** UNEXPECTED ***"))
        if not ok:
            raise SystemExit(f"Kokoulin guard unexpected for "
                             f"{hp.SPECIES[pdg]['label']}")


# =========================================================================
# 4.  the closure
# =========================================================================

def chans(rad):
    """The Fisher scale's channel set.  It MUST track RAD_CHANNEL: 1/I is the
    inversion of the same CF the closure compares against, and
    `fisher_norm.scale_identity` keys the on-disk cache on the channel tuple,
    so a mismatched pair cannot even be served from cache."""
    return ("ioni", "ms", "rad") if rad else ("ioni", "ms")


def _scale(legs, func, tag, channels):
    return fn.plane_scales(legs, func, tag=tag, channels=channels)


def _rows(pdg, rad, func):
    """One closure cell.

    Kokoulin is driven EXPLICITLY from `hp.kok_for` (True for the muon, False
    for every hadron) rather than left to the environment or to a default,
    because the C++ and offline halves guard on |PDG| and the record carries no
    PDG code.  Three caches have to be busted and none of them keys on
    IONI_KOKOULIN or on the particle:

      * `fisher_norm._SCALE_CACHE`                (in-process s_F)
      * `fisher_norm._scale_cache_{load,store}`   (ON-DISK s_F)
      * `cf_propagation_test._PHI_CACHE`

    Without all three, a switched-ON cell silently reuses switched-OFF numbers.
    `s_F` is returned and printed so a switch that failed to reach the Fisher
    inversion is visible rather than silent."""
    assert os.environ.get("RES_NO_PHI_CACHE"), "phi cache is LIVE"
    kok = hp.kok_for(pdg)
    cpt.RAD_CHANNEL = bool(rad)
    ctr.IONI_KOKOULIN = 1.0 if kok else 0.0
    ctr.IONI_KOKOULIN_TCUT = 0.0
    fn._SCALE_CACHE.clear()
    cpt._PHI_CACHE.clear()
    old_load, old_store = fn._scale_cache_load, fn._scale_cache_store
    fn._scale_cache_load = lambda *a, **k: None
    fn._scale_cache_store = lambda *a, **k: None
    try:
        path = mp(pdg, rad)
        legs = cpt.load_model(path)
        sim = hp.sim_of(pdg, SIM_ARM[rad])
        sc = _scale(legs, func, path, chans(rad))
        rows, errs, ks, ns, err = gc.closure_rows(legs, sim, func, sc["sF"])
    finally:
        fn._scale_cache_load, fn._scale_cache_store = old_load, old_store
        cpt.RAD_CHANNEL = True
        ctr.IONI_KOKOULIN = 0.0
        ctr.IONI_KOKOULIN_TCUT = 0.0
        fn._SCALE_CACHE.clear()
        cpt._PHI_CACHE.clear()
    return dict(m=rows.mean(axis=0), err=err, ks=ks, ns=ns,
                nev=int(sim["valid"].shape[0]),
                sF=float(np.mean(sc["sF"])))


def cmd_closure(args):
    print("=" * 152)
    print(f"THE DEFINITIVE RADIATION CLOSURE: eight species, radiation ON and "
          f"OFF, ALL FOUR CORRECTIONS EXPLICITLY ON")
    print("=" * 152)
    print(f"Layered toy, pT = {hp.PT} GeV, eta {hp.ETA}, phi {hp.PHI}, "
          f"cut1e4 (9.4862 keV), 14 planes, plane mean, Fisher normalization, "
          f"nothing fitted.")
    print("CVH_IONI_EXACTDELTA=1  CVH_IONI_KOKOULIN=1  CVH_REF_CHARGEAWARE=1  "
          "CVH_REF_SPECIESDEDX=1  in every cell.")
    print("Kokoulin is species-correct by the |PDG|==13 guard in both halves "
          "(measured in `radid`), so it acts on the muon only.")
    print("Errors are the per-EVENT CORRELATED estimator; err/sqrt(nplanes) "
          "is 2.8x too small here (NOTES_GEOMCLOSURE s1).")
    print("=" * 152)
    res = {}
    for func in args.funcs:
        print()
        print("#" * 152)
        print(f"# FUNCTIONAL: {func}")
        print("#" * 152)
        print(f"  {'species':<8}{'rad':<6}{'sim arm':<8}" + UROW
              + "      rms       nev        s_F")
        for pdg in args.pdg:
            sp = hp.SPECIES[pdg]
            for rad in args.rad:
                if not os.path.exists(mp(pdg, rad)):
                    print(f"  {sp['label']:<8}{'ON' if rad else 'OFF':<6}"
                          f"{SIM_ARM[rad]:<8}(model missing)")
                    continue
                if not glob.glob(hp.sim_glob(pdg, SIM_ARM[rad])):
                    print(f"  {sp['label']:<8}{'ON' if rad else 'OFF':<6}"
                          f"{SIM_ARM[rad]:<8}(sim missing)")
                    continue
                r = _rows(pdg, rad, func)
                res[(func, pdg, rad)] = r
                print(f"  {sp['label']:<8}{'ON' if rad else 'OFF':<6}"
                      f"{SIM_ARM[rad]:<8}" + _fmt(r["m"])
                      + f" {_rms(r['m']):9.5f} {r['nev']:>9d}  {r['sF']:.8e}")
                print(f"  {'':<8}{'+-':<6}{'':<8}" + _fmt(r["err"]))
            # ---- the difference, per probe, with its significance
            if all((func, pdg, x) in res for x in (True, False)):
                a, b = res[(func, pdg, True)], res[(func, pdg, False)]
                d = b["m"] - a["m"]
                # The two cells are INDEPENDENT samples (different simulations,
                # different random streams), so the error on the difference is
                # the quadrature sum -- not the single-cell error.
                de = np.hypot(a["err"], b["err"])
                sig = d / np.where(de > 0, de, np.nan)
                print(f"  {'':<8}{'OFF-ON':<14}" + _fmt(d))
                print(f"  {'':<8}{'sigma':<14}" + _fmt(sig, 9, 2))
                print(f"  {'':<8}{'':<14}rms {_rms(a['m']):.5f} -> "
                      f"{_rms(b['m']):.5f}   "
                      f"({100*(_rms(a['m'])-_rms(b['m']))/max(_rms(a['m']),1e-12):+.1f} % "
                      f"of the rms removed)   max |sigma| "
                      f"{np.nanmax(np.abs(sig)):.2f}")
            print()

    # -------------------------------------------------------------- summary
    print()
    print("=" * 152)
    print("SUMMARY -- rms of the nine-probe curve, and the largest per-probe "
          "significance of the radiation-off shift")
    print("=" * 152)
    for func in args.funcs:
        print(f"\n  {func}")
        print(f"  {'species':<8}{'rad ON':>10}{'rad OFF':>10}{'diff':>10}"
              f"{'% removed':>11}{'max |sigma|':>13}{'n probes > 2s':>15}"
              f"{'events/cell':>13}")
        for pdg in args.pdg:
            if not all((func, pdg, x) in res for x in (True, False)):
                continue
            a, b = res[(func, pdg, True)], res[(func, pdg, False)]
            ra, rb = _rms(a["m"]), _rms(b["m"])
            d = b["m"] - a["m"]
            de = np.hypot(a["err"], b["err"])
            sig = np.abs(d / np.where(de > 0, de, np.nan))
            print(f"  {hp.SPECIES[pdg]['label']:<8}{ra:10.5f}{rb:10.5f}"
                  f"{rb-ra:10.5f}{100*(ra-rb)/max(ra,1e-12):10.1f} %"
                  f"{np.nanmax(sig):13.2f}{int(np.nansum(sig > 2)):15d}"
                  f"{a['nev']:13d}")

    # ------------------------------------- the closure against its OWN error
    print()
    print("=" * 152)
    print("IS THERE ANYTHING LEFT TO EXPLAIN?  Each cell's closure against its "
          "OWN statistical error.")
    print("=" * 152)
    print("`chi2/ndf` is over the nine probes against zero, using the "
          "per-probe correlated error.  The probes are NOT independent")
    print("(the same events at nine values of u), so it is a scale indicator "
          "and not a p-value; `max |sigma|` is the honest one.")
    print()
    for func in args.funcs:
        print(f"  {func}")
        print(f"  {'species':<8}{'rad':<6}{'rms':>10}{'max |sigma|':>13}"
              f"{'at u':>9}{'chi2/ndf':>11}{'n probes > 3s':>15}   sign pattern")
        for pdg in args.pdg:
            for rad in args.rad:
                r = res.get((func, pdg, rad))
                if r is None:
                    continue
                s = r["m"] / np.where(r["err"] > 0, r["err"], np.nan)
                i = int(np.nanargmax(np.abs(s)))
                signs = "".join("+" if x > 0 else ("-" if x < 0 else "0")
                                for x in r["m"])
                print(f"  {hp.SPECIES[pdg]['label']:<8}"
                      f"{'ON' if rad else 'OFF':<6}{_rms(r['m']):10.5f}"
                      f"{np.abs(s)[i]:13.2f}{UCURVE[i]:9g}"
                      f"{float(np.nansum(s ** 2) / len(s)):11.2f}"
                      f"{int(np.nansum(np.abs(s) > 3)):15d}   {signs}")
        print()

    if args.npz:
        d = {}
        for (func, pdg, rad), r in res.items():
            k = f"{func}|{hp.SPECIES[pdg]['label']}|{'on' if rad else 'off'}"
            d[k + "|m"] = r["m"]
            d[k + "|err"] = r["err"]
        np.savez(args.npz, u=UCURVE, **d)
        print(f"wrote {args.npz}")
    return res


# =========================================================================
# 5.  the control: reproduce the published rows before anything is read
# =========================================================================

def cmd_control(args):
    """Reproduce the published numbers on the archived models, then say what
    the new arm changes relative to them.

    NOTES_SPECIESDEDX s6's `both` column is the closest published configuration
    to this note's: all EIGHT species, both reference corrections on, radiation
    ON.  It differs from the arm here in that its models were exported WITHOUT
    `CVH_IONI_EXACTDELTA` / `CVH_IONI_KOKOULIN` in the C++ -- which, by the
    exact s_F invariance (NOTES_FISHERNORM s1), cannot move the closure at all,
    because the C++ half of Kokoulin changes only sigma and s_F = sigma
    sqrt(1/I) is invariant under a rescaling of sigma.  So the published `both`
    column IS the prediction for this note's rad-ON column, and reproducing it
    is a real test of the whole chain rather than a tautology."""
    print("=" * 140)
    print("CONTROL: the published NOTES_SPECIESDEDX s6 `both` column against "
          "this note's radiation-ON cells")
    print("=" * 140)
    print("The published models carry the two REFERENCE corrections only; this "
          "note's carry all four.  The two extra")
    print("switches move the exported VARIANCE (Q, gsig2) and not the "
          "reference, and s_F = sigma sqrt(1/I) is exactly")
    print("invariant under a rescaling of sigma (NOTES_FISHERNORM s1, verified "
          "to 1.5e-8), so the closure must NOT move.")
    print("=" * 140)
    print(f"  {'species':<8}{'source':<12}" + UROW + "      rms")
    nbad = 0
    for pdg in args.pdg:
        pub = PUBLISHED_BOTH.get(pdg)
        r = res_on = None
        if os.path.exists(mp(pdg, True)) and glob.glob(hp.sim_glob(pdg, "off")):
            res_on = _rows(pdg, True, "qop")
            r = res_on["m"]
        lab = hp.SPECIES[pdg]["label"]
        if pub is not None:
            print(f"  {lab:<8}{'PUBLISHED':<12}" + _fmt(np.asarray(pub))
                  + f" {_rms(pub):9.5f}")
        if r is not None:
            print(f"  {lab:<8}{'this note':<12}" + _fmt(r)
                  + f" {_rms(r):9.5f}")
        if pub is not None and r is not None:
            d = r - np.asarray(pub)
            s = d / np.where(res_on["err"] > 0, res_on["err"], np.nan)
            print(f"  {'':<8}{'difference':<12}" + _fmt(d))
            print(f"  {'':<8}{'sigma':<12}" + _fmt(s, 9, 2))
            if np.nanmax(np.abs(s)) > 1.0:
                nbad += 1
                print(f"  {'':<8}*** does not reproduce ***")
        print()
    print(f"  species failing to reproduce: {nbad}")


# NOTES_SPECIESDEDX s6, the `both` column (both reference corrections ON,
# radiation ON, exact-delta ON, Kokoulin species-correct, 200k events).
PUBLISHED_BOTH = {
    13:    [0.00028, 0.00025, 0.00007, -0.00026, -0.00070, -0.00081,
            -0.00053, -0.00035, -0.00020],
    -13:   [0.00005, 0.00011, 0.00014, 0.00018, 0.00052, 0.00073,
            0.00050, 0.00028, 0.00014],
    -211:  [-0.00028, -0.00042, -0.00065, -0.00080, -0.00068, -0.00061,
            -0.00056, -0.00035, -0.00023],
    211:   [-0.00054, -0.00070, -0.00075, -0.00062, -0.00019, 0.00007,
            0.00014, 0.00018, 0.00012],
    -321:  [-0.00011, -0.00019, -0.00032, -0.00046, -0.00062, -0.00075,
            -0.00074, -0.00059, -0.00036],
    321:   [-0.00008, -0.00015, -0.00025, -0.00039, -0.00057, -0.00071,
            -0.00068, -0.00048, -0.00029],
    -2212: [-0.00010, -0.00020, -0.00043, -0.00072, -0.00103, -0.00117,
            -0.00092, -0.00040, -0.00017],
    2212:  [-0.00000, -0.00003, -0.00009, -0.00018, -0.00029, -0.00031,
            -0.00028, -0.00024, -0.00011],
}


# =========================================================================

def main():
    os.environ.setdefault("RES_NO_PHI_CACHE", "1")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    s = p.add_subparsers(dest="cmd", required=True)

    q = s.add_parser("export", help="the 16 models, all four corrections ON")
    q.add_argument("--pdg", type=int, nargs="+", default=ORDER8)
    q.add_argument("--rad", type=int, nargs="+", default=[1, 0])
    q.add_argument("--force", action="store_true")
    q.set_defaults(f=cmd_export)

    q = s.add_parser("sim", help="the missing `norad` simulations")
    q.add_argument("--pdg", type=int, nargs="+", default=ORDER8)
    q.add_argument("--events", type=int, default=20000)
    q.add_argument("--jobs", type=int, default=10)
    q.add_argument("--seed0", type=int, default=101)
    q.add_argument("--workers", type=int, default=30)
    q.add_argument("--force", action="store_true")
    q.set_defaults(f=cmd_sim)

    q = s.add_parser("live", help="(R1) the step census")
    q.add_argument("--pdg", type=int, nargs="+", default=ORDER8)
    q.add_argument("--arms", nargs="+", default=["off", "norad"])
    q.set_defaults(f=cmd_live)

    q = s.add_parser("radid", help="(R2)+(R3) the model-side switches")
    q.add_argument("--pdg", type=int, nargs="+", default=ORDER8)
    q.add_argument("--force", action="store_true")
    q.set_defaults(f=cmd_radid)

    q = s.add_parser("closure", help="THE RESULT")
    q.add_argument("--pdg", type=int, nargs="+", default=ORDER8)
    q.add_argument("--rad", type=int, nargs="+", default=[1, 0])
    q.add_argument("--funcs", nargs="+", default=["qop", "locx"])
    q.add_argument("--npz", default=None)
    q.set_defaults(f=cmd_closure)

    q = s.add_parser("control", help="reproduce the published rows")
    q.add_argument("--pdg", type=int, nargs="+", default=ORDER8)
    q.set_defaults(f=cmd_control)

    a = p.parse_args()
    for k in ("rad",):
        if hasattr(a, k):
            setattr(a, k, [bool(x) for x in getattr(a, k)])
    a.f(a)


if __name__ == "__main__":
    main()
