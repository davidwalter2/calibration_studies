#!/usr/bin/env python3
"""RADIATION OFF, *WITH* THE EXACT-DELTA CORRECTION APPLIED -- the missing cell
of the 2x2.

WHY THIS EXISTS
---------------
NOTES_RADOFF concluded "radiation is not the cause": with brems + pair
production switched off consistently in the SIM, in the REFERENCE and in the
model CF, the pT = 40 `qop` non-closure was not removed anywhere and was
14-43 % LARGER at u >= 0.03.  That test predates NOTES_DELTASPEC's exact
delta-ray spectrum, which removes 74 % (pT = 3) / 79 % (pT = 40) of the same
non-closure.  Radiation was therefore being tested against a residual dominated
by a DIFFERENT, larger error -- and the fact that the residual GREW with
radiation off says the two were partially cancelling.  With 73-79 % of the
ionization error gone the balance is unknown, so the radiation-off test has to
be redone on the corrected model.

THE 2x2 (`qop` rms of the nine-probe Fisher-normalized closure curve)

                     correction OFF        correction ON
    radiation ON     published             published
    radiation OFF    published (grew)      *** this note ***

FOUR SWITCHES, AND ALL OF THEM MUST BE SHOWN LIVE
-------------------------------------------------
  RADIATION (three half-switches that must move TOGETHER; NOTES_RADOFF s1
  established at real cost that any one alone gives a large spurious answer --
  the CF half alone moved the pT = 40 closure by 68 % of the whole non-closure):
    (R1) SIM        `ProcessActivationWatcher` INSIDE the Simulation biglib.
                    `process.g4SimHits.G4Commands` is a silent no-op and a
                    watcher outside the biglib talks to a second, uninitialised
                    Geant4.  Evidence = the step census: muBrems and muPairProd
                    must define EXACTLY ZERO steps.
    (R2) REFERENCE  `CVH_IONONLY=1` -> the dE/dx table drops the radiative mean.
    (R3) MODEL CF   `cf_propagation_test.RAD_CHANNEL = False` drops the
                    `cf_brems_exact.rad_exponent` term, and the Fisher scale is
                    built from channels=("ioni","ms") to match.

  CORRECTION:
    (C)  EXPORT     `CVH_IONI_EXACTDELTA=1` -> regime-2 `ioniurbanv` record
                    (a3 becomes xi).  Offline `ioni_step_exponent` switches on
                    the regime, so there is no second offline flag to forget.

Nothing in this module is fitted and nothing is tuned.

SUBCOMMANDS
    export    the 2x2 of models (rad x correction), both momenta
    bitid     branch-level identity of the model matrix (the switch audit)
    sim       the NEW radiation-off exact-cut simulations
    live      switch-liveness evidence for all four switches
    pairs     model/sim pair validation
    closure   the 2x2 closure tables, full u curve, plane mean + per plane
"""

import argparse
import glob
import hashlib
import os
import re
import subprocess
import sys

import numpy as np
import uproot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "cleanprop"))

import cf_propagation_test as cpt                               # noqa: E402
import fisher_norm as fn                                        # noqa: E402
import geom_closure as gc                                       # noqa: E402
import deltaspec as ds                                          # noqa: E402
import tail_probe as tp                                         # noqa: E402

SCRATCH = fn.SCRATCH
OUT = os.path.join(SCRATCH, "ro2")
TH = os.path.join(SCRATCH, "th")          # NOTES_TAILHUNT's archived cut sims
TPD = os.path.join(SCRATCH, "tp")         # NOTES_TOY_PT40's archived models
RO = os.path.join(SCRATCH, "ro")          # NOTES_RADOFF's archived models
DSD = os.path.join(SCRATCH, "ds")         # NOTES_DELTASPEC's archived models

UCURVE = fn.UCURVE
UHEAD = fn.UHEAD
IHEAD = fn.IHEAD

RADPROCS = ["muBrems", "muPairProd"]

# ---------------------------------------------------------------- the matrix
# sim tag -> (area, DefaultCutValue [cm], radiation on?, where it lives)
#
# The rad-ON side is NOT re-run: it is NOTES_TAILHUNT's archived 200k sample,
# i.e. literally the simulation whose closure NOTES_DELTASPEC published.  The
# rad-OFF side is new and is produced here at the SAME cut, the same seeds and
# the same event count, through the same driver.
SIMS = {
    "pt3_cut1e4":     dict(area="pt3",  cut=1e-4, rad=True,  where=TH),
    "pt3_cut1e4_ro":  dict(area="pt3",  cut=1e-4, rad=False, where=OUT),
    "pt40_cut001":    dict(area="pt40", cut=0.01, rad=True,  where=TH),
    "pt40_cut001_ro": dict(area="pt40", cut=0.01, rad=False, where=OUT),
    "pt40_cut1e4":    dict(area="pt40", cut=1e-4, rad=True,  where=TH),
    "pt40_cut1e4_ro": dict(area="pt40", cut=1e-4, rad=False, where=OUT),
    # NOTES_RADOFF's own 400k DEFAULT-CUT pair, reused byte-identically.  It is
    # the configuration whose published numbers this note has to reproduce as a
    # control before its new numbers mean anything -- and, because the default
    # cut supplies a partial cancellation (NOTES_TAILHUNT s3), it is NOT the
    # configuration the 2x2 is measured in.
    "pt3_base":     dict(area="pt3",  cut=1.0, rad=True,  where=TPD,
                         path=f"{TPD}/pt3_K1_sim.root"),
    "pt3_base_ro":  dict(area="pt3",  cut=1.0, rad=False, where=RO,
                         path=f"{RO}/pt3_radoff_sim.root"),
    "pt40_base":    dict(area="pt40", cut=1.0, rad=True,  where=TPD,
                         path=f"{TPD}/pt40_K1_sim.root"),
    "pt40_base_ro": dict(area="pt40", cut=1.0, rad=False, where=RO,
                         path=f"{RO}/pt40_radoff_sim.root"),
}
AREA_PT = {"pt3": 3.0, "pt40": 40.0}

# (area, rad, corr) -> model label.  All four are exported HERE, from the same
# private area with the same build, so the four cells of the 2x2 are
# commensurate by construction rather than by argument.  `bitid` then checks
# each against the archive it should reproduce.
MODEL_LABEL = {(True, False): "radon_off", (True, True): "radon_on",
               (False, False): "radoff_off", (False, True): "radoff_on"}
MODEL_ENV = {"radon_off": {},
             "radon_on": {"CVH_IONI_EXACTDELTA": "1"},
             "radoff_off": {"CVH_IONONLY": "1"},
             "radoff_on": {"CVH_IONONLY": "1", "CVH_IONI_EXACTDELTA": "1"}}

# ---------------------------------------------------------------- KOKOULIN
# Geant4's radiative correction to the knock-on cross section, established by
# NOTES_SAMPLERGAP as a real parameter-free correction that removes a further
# 61 % (pT = 3) / 87 % (pT = 40) of the corrected residual.  It is a correction
# TO the exact-delta channel, so it exists only in regime 2/3: with the
# exact-delta correction OFF the record is regime 1 and neither the C++ nor the
# offline branch is entered at all.  That is asserted here, not assumed.
#
# TWO SITES, ONE SWITCH (`CVH_IONI_KOKOULIN`, NOTES_QVALID):
#   C++      `cvhcgf::ioniKokoulinEnabled()` adds the Kokoulin variance to the
#            exported `gsig2` and to the variance the fit consumes as Q(0,0).
#            Needs a RE-EXPORT.
#   offline  `cf_track_resolution.IONI_KOKOULIN` puts the term in the CF the
#            closure compares against.  Its DEFAULT is now derived from the same
#            environment variable, so a job cannot end up half-corrected.
# Both are exercised here; the offline-only arm is kept as the control that
# reproduces NOTES_SAMPLERGAP's published rows.
KOK_ENV = {"CVH_IONI_KOKOULIN": "1"}
# The e- production threshold of the SIMULATION each config is compared against.
# Geant4 applies the correction only to the EXPLICIT secondaries, i.e. only
# above the production cut, and only above its own 100 keV floor -- so the
# model's correction starts at max(tcut, 100 keV).  Both cut1e4 configs sit at
# 9.49 keV, i.e. BELOW the floor, so the floor is what acts and tcut = 0 is the
# correct entry (not a convenience).
MODEL_TCUT = {"pt3_cut1e4": 0.0, "pt3_cut1e4_ro": 0.0,
              "pt40_cut1e4": 0.0, "pt40_cut1e4_ro": 0.0,
              "pt40_cut001": 0.29638, "pt40_cut001_ro": 0.29638,
              "pt3_base": 17.8507, "pt3_base_ro": 17.8507,
              "pt40_base": 17.8507, "pt40_base_ro": 17.8507}
# what each fresh export must reproduce bit for bit (None = nothing archived)
MODEL_ARCHIVE = {
    ("pt3", "radon_off"):  f"{TPD}/pt3_K1_model.root",
    ("pt3", "radon_on"):   f"{DSD}/pt3_model_on.root",
    ("pt3", "radoff_off"): f"{RO}/pt3_radoff_model.root",
    ("pt3", "radoff_on"):  None,
    ("pt40", "radon_off"):  f"{TPD}/pt40_K1_model.root",
    ("pt40", "radon_on"):   f"{DSD}/pt40_model_on.root",
    ("pt40", "radoff_off"): f"{RO}/pt40_radoff_model.root",
    ("pt40", "radoff_on"):  None,
}


def model_path(area, rad, corr, kok=False):
    lab = MODEL_LABEL[(rad, corr)] + ("_kok" if kok else "")
    return os.path.join(OUT, f"{area}_model_{lab}.root")


def sim_glob(tag):
    p = SIMS[tag].get("path")
    return p if p else os.path.join(SIMS[tag]["where"], f"{tag}_s*_sim.root")


def sim_path(tag, seed):
    return os.path.join(SIMS[tag]["where"], f"{tag}_s{seed}_sim.root")


def planes_path(area):
    return os.path.join(tp.testdir(area),
                        f"toyPlanes_pt{int(AREA_PT[area])}.py")


# ==========================================================================
# model export
# ==========================================================================

def _libhash():
    p = ("/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/lib/"
         "el9_amd64_gcc12/libTrackPropagationGeant4e.so")
    return hashlib.md5(open(p, "rb").read()).hexdigest(), p


def cmd_export(args):
    os.makedirs(args.out or OUT, exist_ok=True)
    outdir = args.out or OUT
    h, p = _libhash()
    print(f"propagator library {h}  {p}")
    for area in args.areas:
        for rad in (True, False):
            for corr in (False, True):
                lab = MODEL_LABEL[(rad, corr)] + ("_kok" if args.kok else "")
                out = os.path.join(outdir, f"{area}_model_{lab}.root")
                log = out[:-5] + ".log"
                env = dict(MODEL_ENV[MODEL_LABEL[(rad, corr)]])
                if args.kok:
                    env.update(KOK_ENV)
                if os.path.exists(out) and not args.force:
                    print(f"[{area}/{lab}] exists, skipping")
                    continue
                rc = ds.run_model(area, out, log, env)
                txt = open(log, errors="ignore").read(400000)
                io = "CVH_IONONLY set" in txt
                print(f"[{area}/{lab}] rc={rc}  env={env}  ionOnlyInLog={io}")
                if rc:
                    raise SystemExit(f"export failed, see {log}")
                if io != (not rad):
                    raise SystemExit(
                        f"[{area}/{lab}] CVH_IONONLY marker {io}, expected "
                        f"{not rad} -- the reference switch did not take")
    h2, _ = _libhash()
    if h2 != h:
        raise SystemExit("the propagator library changed DURING the export "
                         f"({h} -> {h2}); the models are not commensurate")
    print(f"propagator library unchanged through the export: {h2}")


def cmd_bitid(args):
    """The switch audit, at branch level.

    NOTES_DELTASPEC s3.3 records the trap: a comparator that enumerates
    top-level keys finds ONE (the `propExport` TDirectory), compares ZERO
    branches and reports BIT-IDENTICAL for a file that moved.  `ds._branch_
    digests` enumerates TTrees explicitly; the number of branches compared is
    printed on every line and a zero is a hard failure.
    """
    print("=" * 100)
    print("MODEL MATRIX -- BRANCH-LEVEL IDENTITY")
    print("=" * 100)
    print("(a) each fresh export against the archived model it must reproduce")
    print(f"  {'pair':<44s} {'branches':>9} {'same':>6} {'diff':>6}  verdict")
    bad = 0
    for area in args.areas:
        for rad in (True, False):
            for corr in (False, True):
                lab = MODEL_LABEL[(rad, corr)]
                arch = MODEL_ARCHIVE[(area, lab)]
                mine = model_path(area, rad, corr)
                if arch is None:
                    print(f"  {area + '/' + lab + ' vs (no archive)':<44s} "
                          f"{'-':>9} {'-':>6} {'-':>6}  NEW")
                    continue
                a, b = ds._branch_digests(arch), ds._branch_digests(mine)
                keys = sorted(set(a) | set(b))
                same = [k for k in keys if a.get(k) == b.get(k)]
                diff = [k for k in keys if a.get(k) != b.get(k)]
                ok = len(keys) > 0 and not diff
                bad += 0 if ok else 1
                print(f"  {area + '/' + lab + ' vs archive':<44s} "
                      f"{len(keys):>9} {len(same):>6} {len(diff):>6}  "
                      f"{'BIT-IDENTICAL' if ok else 'DIFFER: ' + ','.join(diff[:6])}")
    print()
    print("(b) the switches, against the fresh radon_off corner: which branches "
          "each moves")
    print(f"  {'pair':<44s} {'branches':>9} {'same':>6} {'diff':>6}  moved")
    for area in args.areas:
        base = model_path(area, True, False)
        a = ds._branch_digests(base)
        for rad, corr in ((True, True), (False, False), (False, True)):
            mine = model_path(area, rad, corr)
            b = ds._branch_digests(mine)
            keys = sorted(set(a) | set(b))
            diff = [k for k in keys if a.get(k) != b.get(k)]
            lab = MODEL_LABEL[(rad, corr)]
            print(f"  {area + '/radon_off vs ' + lab:<44s} {len(keys):>9} "
                  f"{len(keys)-len(diff):>6} {len(diff):>6}  "
                  + ",".join(k.split('/')[-1] for k in diff[:8]))
    if bad:
        raise SystemExit(f"{bad} archive comparison(s) FAILED")
    print("\nall archive comparisons pass")


# ==========================================================================
# simulation
# ==========================================================================

def _run_one(arg):
    tag, seed, nev = arg
    cfg = SIMS[tag]
    out = sim_path(tag, seed)
    log = out[:-5] + ".log"
    ee = {"TOY_CUT": repr(cfg["cut"]),
          "TOY_INACT": "" if cfg["rad"] else ",".join(RADPROCS),
          "TOY_CENSUS": os.path.join(cfg["where"], f"{tag}_s{seed}_census.bin")}
    rc = tp._cmsrun(cfg["area"], "runToyGeomCheck.py",
                    f"events={nev} pt={AREA_PT[cfg['area']]} eta=0.30 "
                    f"output={out} seed={seed}", log, env_extra=ee)
    return tag, seed, rc, log


def cmd_sim(args):
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for tag in args.tags:
        if SIMS[tag]["where"] != OUT:
            raise SystemExit(f"{tag} is an ARCHIVED sim; it is not re-run")
        for i in range(args.jobs):
            jobs.append((tag, args.seed0 + i, args.events))
    print(f"launching {len(jobs)} jobs "
          f"({args.jobs} seeds x {args.events} events per tag)")
    nbad = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for tag, seed, rc, log in ex.map(_run_one, jobs):
            txt = open(log, errors="ignore").read(60000)
            st = ("TIGHT" if "[toy] TIGHT stepper" in txt else
                  ("LOOSE" if "LOOSE stepper" in txt else "UNKNOWN"))
            cut = re.search(r"\[toy\] DefaultCutValue = (\S+) cm", txt)
            ina = re.search(r"\[toy\] INACTIVATE requested: (\S+)", txt)
            cen = census_of_log(log)
            # A DEACTIVATED process does not appear in the census at all, which
            # is what "exactly zero steps" looks like.  An EMPTY census means
            # the log is truncated (the job died) and is a failure, not a pass.
            zero = bool(cen) and all(p not in cen or cen[p][0] == 0
                                     for p in RADPROCS)
            good = (rc == 0 and st == "TIGHT" and zero)
            nbad += 0 if good else 1
            print(f"  {tag} seed={seed} rc={rc} stepper={st} "
                  f"cut={cut.group(1) if cut else '?'} "
                  f"inact={ina.group(1) if ina else '-'} "
                  f"radsteps={ {p: cen[p][0] if p in cen else 'absent=0' for p in RADPROCS} } "
                  f"{'OK' if good else '*** FAILED ***'}")
    if nbad:
        raise SystemExit(f"{nbad} sim job(s) failed")


def census_of_log(path):
    """{process: (primary steps, all steps)} from ProcessActivationWatcher's
    EndOfRun census.  A process that is INACTIVE does not appear at all, which
    reads as zero -- that is the intended evidence and it is a statement about
    what Geant4 RAN, not about a flag."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, errors="ignore") as fh:
        for ln in fh:
            m = re.match(r"\[procact\]\s+(\S+)\s+primary (\d+)\s+all (\d+)", ln)
            if m:
                out[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    return out


def census_of_tag(tag):
    tot = {}
    for lg in sorted(glob.glob(sim_glob(tag)[:-5] + ".log")):
        for k, (a, b) in census_of_log(lg).items():
            p, q = tot.get(k, (0, 0))
            tot[k] = (p + a, q + b)
    return tot


# ==========================================================================
# loading
# ==========================================================================

_M, _S = {}, {}


def legs_of(area, rad, corr, kok=False):
    p = model_path(area, rad, corr, kok)
    if p not in _M:
        _M[p] = fn.load(p)          # fn.load registers provenance for the cache
    return _M[p]


def _planes_ns(area):
    ns = {}
    exec(open(planes_path(area)).read(), ns)
    return ns


def sim_of(tag):
    if tag not in _S:
        from toy_loader import load_toy_sim
        ns = _planes_ns(SIMS[tag]["area"])
        _S[tag] = load_toy_sim(sim_glob(tag), ns["origin"], ns["normal"],
                               ns["uaxis"])
    return _S[tag]


def legs_meta(area, rad, corr,
              fields=("refglobr", "refp", "refpt", "refqop", "detid")):
    f = uproot.open(model_path(area, rad, corr))
    tk = next(k for k in f.keys() if k.split(";")[0].endswith("/legs"))
    return {k: np.asarray(v) for k, v in
            f[tk].arrays(list(fields), library="np").items()}


def chans(rad):
    """The Fisher scale's channel set.  It MUST track RAD_CHANNEL: 1/I is built
    from the same CF the closure compares against."""
    return ("ioni", "ms", "rad") if rad else ("ioni", "ms")


def _rows(area, tag, rad, corr, func, kok=False, kokmodel=None):
    """One cell for one functional.

    `kok` puts Geant4's Kokoulin radiative correction into the model's own CF;
    `kokmodel` (default = `kok`) selects whether the MODEL was exported with the
    C++ half of the same switch, so the offline-only arm can be run as the
    control that reproduces NOTES_SAMPLERGAP's published rows.

    THREE CACHES HAVE TO BE BUSTED, and none of them keys on `IONI_KOKOULIN`:
      * `fisher_norm._SCALE_CACHE`  (in-process s_F)
      * `fisher_norm._scale_cache_{load,store}`  (ON-DISK s_F, keyed by the
        model file's content identity + a source fingerprint -- neither of
        which moves when a module global does)
      * `cf_propagation_test._PHI_CACHE` (`_phi_key` carries IONI_A3_SCALE,
        IONI_EXC_SCALE and IONI_TMAX_SCALE but NOT IONI_KOKOULIN)
    Without all three a switched-ON run silently reuses switched-OFF numbers.
    s_F is printed by the caller so a switch that failed to reach the scale
    computation is visible rather than silent.
    """
    import cf_track_resolution as ctr
    if kokmodel is None:
        kokmodel = kok
    cpt.RAD_CHANNEL = bool(rad)
    ctr.IONI_KOKOULIN = 1.0 if kok else 0.0
    ctr.IONI_KOKOULIN_TCUT = MODEL_TCUT.get(tag, 0.0) if kok else 0.0
    fn._SCALE_CACHE.clear()
    cpt._PHI_CACHE.clear()
    old_env = os.environ.get("RES_NO_PHI_CACHE")
    os.environ["RES_NO_PHI_CACHE"] = "1"
    old_load, old_store = fn._scale_cache_load, fn._scale_cache_store
    fn._scale_cache_load = lambda *a, **k: None
    fn._scale_cache_store = lambda *a, **k: None
    try:
        mp = model_path(area, rad, corr, kokmodel)
        legs, sim = legs_of(area, rad, corr, kokmodel), sim_of(tag)
        sc = fn.plane_scales(legs, func, tag=mp, channels=chans(rad))
        rows, errs, ks, ns, err = gc.closure_rows(legs, sim, func, sc["sF"])
    finally:
        fn._scale_cache_load, fn._scale_cache_store = old_load, old_store
        cpt.RAD_CHANNEL = True
        ctr.IONI_KOKOULIN = 0.0
        ctr.IONI_KOKOULIN_TCUT = 0.0
        fn._SCALE_CACHE.clear()
        cpt._PHI_CACHE.clear()
        if old_env is None:
            os.environ.pop("RES_NO_PHI_CACHE", None)
        else:
            os.environ["RES_NO_PHI_CACHE"] = old_env
    return dict(rows=rows, errs=errs, ks=ks, ns=ns, err=err, sc=sc)


# ==========================================================================
# subcommand: live
# ==========================================================================

def cmd_live(args):
    print("=" * 100)
    print("SWITCH LIVENESS.  Nothing below is interpreted until every switch "
          "is shown to DO SOMETHING.")
    print("=" * 100)

    # ---- (R1) the SIM switch: the step census
    print("\n### (R1) SIMULATION -- ProcessActivationWatcher step census")
    print("A deactivated process must define EXACTLY ZERO steps.  This is a "
          "statement about\nwhat Geant4 RAN, not about a flag: "
          "`process.g4SimHits.G4Commands` is applied in\nG4State_PreInit with "
          "the return code discarded and is a SILENT no-op "
          "(NOTES_RADOFF s1.1).")
    procs = ["muIoni", "muBrems", "muPairProd", "muonNuclear", "CoulombScat",
             "msc", "eIoni", "eBrem", "compt"]
    print(f"  {'sim tag':<18s}" + "".join(f"{p:>14s}" for p in procs))
    for tag in args.tags:
        c = census_of_tag(tag)
        print(f"  {tag:<18s}"
              + "".join(f"{c[p][0]:>14d}" if p in c else f"{'(absent=0)':>14s}"
                        for p in procs))
    for tag in args.tags:
        if not SIMS[tag]["rad"]:
            c = census_of_tag(tag)
            live = all(p not in c or c[p][0] == 0 for p in RADPROCS)
            verdict = "ZERO -- LIVE" if live else "*** NON-ZERO: INERT ***"
            print(f"  {tag}: radiative steps {verdict}")
            if not live:
                raise SystemExit(f"{tag}: the sim switch did not take")

    # ---- (R2) the REFERENCE switch
    print("\n### (R2) REFERENCE TRAJECTORY -- CVH_IONONLY=1")
    print(f"  {'area':>6} {'plane':>6} {'refp radON':>14} {'refp radOFF':>14} "
          f"{'d(refp) [MeV]':>15} {'d(refqop)/qop':>15}")
    for area in args.areas:
        a = legs_meta(area, True, False)
        b = legs_meta(area, False, False)
        for k in (0, len(a["refp"]) // 2, len(a["refp"]) - 1):
            print(f"  {area:>6} {k:>6} {a['refp'][k]:14.6f} "
                  f"{b['refp'][k]:14.6f} "
                  f"{1e3*(b['refp'][k]-a['refp'][k]):+15.5f} "
                  f"{(b['refqop'][k]-a['refqop'][k])/abs(a['refqop'][k]):+15.3e}")
        dEa = 1e3 * float(a["refp"][0] - a["refp"][-1])
        dEb = 1e3 * float(b["refp"][0] - b["refp"][-1])
        print(f"  {area:>6}  total reference dE: radON {dEa:.4f} MeV  "
              f"radOFF {dEb:.4f} MeV   radiative mean {dEa-dEb:.4f} MeV")
        if not dEa - dEb > 0:
            raise SystemExit(f"{area}: CVH_IONONLY did not move the reference")

    # ---- (C) the CORRECTION switch, in the record
    print("\n### (C) THE EXPORT -- CVH_IONI_EXACTDELTA=1 (the regime-2 record)")
    print("  regime 2 is a NEW value, so a consumer that cannot read it must "
          "refuse rather\n  than mis-read column 6 (a3 becomes xi, an energy, "
          "not a collision count).")
    print(f"  {'model':<20s} {'stride':>7} {'regime':>7} {'a1':>12} "
          f"{'a2':>10} {'a3 / xi':>13} {'e0':>10} {'tmax':>12} "
          f"{'gsig2':>13}")
    for area in args.areas:
        for rad in (True, False):
            for corr in (False, True):
                legs = legs_of(area, rad, corr)
                # the FIRST ToyLayerMat step of the first leg -- the same row
                # NOTES_DELTASPEC s3.2 tabulates
                m = np.asarray(legs[0]["ioni"], dtype=np.float64)
                r = m[_first_dense(m)]
                lab = MODEL_LABEL[(rad, corr)]
                print(f"  {area + '/' + lab:<20s} {m.shape[1]:>7d} "
                      f"{int(r[0]):>7d} {r[2]:>12.2f} {r[4]:>10.3f} "
                      f"{r[6]:>13.6g} {r[7]:>10.2g} {r[8]:>12.4g} "
                      f"{r[1]:>13.6g}")

    # ---- (R3) the MODEL CF switch
    print("\n### (R3) MODEL CF -- cf_propagation_test.RAD_CHANNEL")
    print("  the model's own <e^{-u z^2}> at the outermost plane, and the "
          "kappa2 shares.")
    import cgf_channels as cc
    from cf_propagation_test import FUNCTIONALS, model_variance
    print(f"  {'model':<18s} {'RAD_CHANNEL':<12s} {'ioni':>9} {'ms':>9} "
          f"{'rad':>9}  " + "".join(f"{'M(u=' + str(u) + ')':>13s}"
                                    for u in UHEAD))
    for area in args.areas:
        for rad in (True, False):
            for corr in (False, True):
                legs = legs_of(area, rad, corr)
                k = len(legs) - 1
                avec = FUNCTIONALS["qop"]
                sig = float(np.sqrt(model_variance(legs, k, avec)[0]))
                # kappa2 shares come from the SADDLEPOINT CGF, which refuses
                # regime-2 records by design (NOTES_DELTASPEC s5.3 item 5).
                # That refusal is itself part of the audit, so it is reported
                # rather than worked around.
                sh, tot = {}, None
                try:
                    for c in ("ioni", "ms", "rad"):
                        blk = cc.collect_block(legs, k, avec, sig,
                                               channels=(c,))
                        sh[c] = float(np.atleast_1d(
                            cc.block_cgf_derivs(blk, 0.0, order=2)[2])[0])
                    tot = sum(sh.values())
                except NotImplementedError:
                    sh, tot = {}, None
                for sw in (True, False):
                    cpt.RAD_CHANNEL = sw
                    tau = fn.closure_tau(float(np.max(UHEAD)))
                    phi = cpt.model_phi(legs, k, avec, sig, tau)
                    m = [cpt.weier_scalar(phi, u, tau) for u in UHEAD]
                    shtxt = ("".join(f"{sh[c]/tot:9.5f}"
                                     for c in ("ioni", "ms", "rad"))
                             if tot else f"{'(regime 2: saddlepoint refuses)':>27s}")
                    print(f"  {area + '/' + MODEL_LABEL[(rad, corr)]:<18s} "
                          f"{str(sw):<12s}" + shtxt
                          + "  " + "".join(f"{v:13.6f}" for v in m))
                cpt.RAD_CHANNEL = True
    print()


def _first_dense(m):
    """Index of the first record whose a1 is large, i.e. the first crossing of
    the dense layer material rather than of the beam pipe or of air.  Chosen by
    the record itself (max a1 over the first few steps), not by a step number,
    so it cannot silently point at a different material between models."""
    n = min(len(m), 40)
    return int(np.argmax(m[:n, 2]))


# ==========================================================================
# subcommand: pairs
# ==========================================================================

def cmd_pairs(args):
    print("=" * 100)
    print("MODEL / SIM PAIR VALIDATION")
    print("=" * 100)
    ok_all = True
    for tag in args.tags:
        cfg = SIMS[tag]
        area, rad = cfg["area"], cfg["rad"]
        legs = legs_of(area, rad, False)
        sim = sim_of(tag)
        ns = _planes_ns(area)
        npl = len(ns["radii"])
        a = legs_meta(area, rad, False)
        v = sim["valid"]
        nfull = int(v.all(axis=1).sum())
        rmed = np.array([np.nanmedian(sim["globr"][v[:, k], k])
                         for k in range(npl)])
        dE_m = 1e3 * float(a["refp"][0] - a["refp"][-1])
        p0 = np.asarray(sim["pabs"][v[:, 0], 0]).ravel()
        pl = np.asarray(sim["pabs"][v[:, npl - 1], npl - 1]).ravel()
        dE_s = 1e3 * float(np.median(p0) - np.median(pl))
        n_ok = len(a["detid"]) == npl == v.shape[1]
        seq_ok = n_ok and np.array_equal(a["detid"], np.arange(npl))
        dr = float(np.abs(a["refglobr"] - rmed).max())
        pt_ok = abs(float(a["refpt"][0]) - AREA_PT[area]) < 1e-3 * AREA_PT[area]
        logs = sorted(glob.glob(sim_glob(tag)[:-5] + ".log"))
        sts, cuts, inas = set(), set(), set()
        for lg in logs:
            txt = open(lg, errors="ignore").read(60000)
            sts.add("TIGHT" if "[toy] TIGHT stepper" in txt else
                    ("LOOSE" if "LOOSE stepper" in txt else "UNKNOWN"))
            m = re.search(r"\[toy\] DefaultCutValue = (\S+) cm", txt)
            cuts.add(m.group(1) if m else "?")
            m = re.search(r"\[toy\] (INACTIVATE requested: \S+|full physics)",
                          txt)
            inas.add(m.group(1) if m else "?")
        cen = census_of_tag(tag)
        radsteps = sum(cen.get(p, (0, 0))[0] for p in RADPROCS)
        # The archived NOTES_TOY_PT40 rad-ON sims predate the watcher, so they
        # carry NO census at all.  An empty census is the correct reading there
        # and only a rad-OFF sim with non-zero radiative steps is a failure.
        nocensus = not cen
        rad_ok = (radsteps > 0 or nocensus) if rad else (radsteps == 0
                                                        and not nocensus)
        print(f"--- {tag}   pT={AREA_PT[area]:g}  cut={cfg['cut']:g} cm  "
              f"rad={rad}   ({len(logs)} seeds)")
        print(f"    (1) legs {len(a['detid'])}  sim planes {v.shape[1]}  "
              f"planes {npl}   {'OK' if n_ok else 'MISMATCH'}")
        print(f"    (2) plane-index sequence identical: {seq_ok}")
        print(f"    (3) max |refglobr - sim median r| = {dr:.5f} cm")
        print(f"    (4) dE model(mean) {dE_m:8.3f} MeV   sim(median) "
              f"{dE_s:8.3f} MeV   gap {dE_m - dE_s:+7.3f} MeV")
        print(f"    (5) model refpt[0] = {float(a['refpt'][0]):.4f} GeV  "
              f"{'OK' if pt_ok else 'MISMATCH'}")
        print(f"        sim events {sim['ntot']}, complete-sequence {nfull} "
              f"({100.*nfull/sim['ntot']:.3f} %)")
        print(f"    (6) stepper {sorted(sts)}  cut {sorted(cuts)}  "
              f"{sorted(inas)}")
        print(f"    (7) radiative steps (primary, summed) "
              f"{'(no census in the archived log)' if nocensus else radsteps}"
              f"   {'OK' if rad_ok else '*** WRONG ***'}")
        good = (n_ok and seq_ok and dr < 1e-3 and pt_ok
                and sts <= {"TIGHT", "UNKNOWN"} and rad_ok and len(cuts) <= 2)
        ok_all &= good
        print(f"    VERDICT: {'PASS' if good else 'FAIL'}\n")
    print(f"ALL: {'PASS' if ok_all else 'SOME FAILED'}")
    if not ok_all:
        raise SystemExit("pair validation failed")


# ==========================================================================
# subcommand: closure -- THE 2x2
# ==========================================================================

def _rms(v):
    return float(np.sqrt(np.mean(np.asarray(v) ** 2)))


def cmd_closure(args):
    print("=" * 110)
    print("THE 2x2: RADIATION x EXACT-DELTA CORRECTION")
    print("=" * 110)
    print("Fisher normalization (s_F = sigma sqrt(1/I), 1/I by exact FFT "
          "inversion of the\nmodel's OWN CF with the SAME channel set the CF "
          "uses).  Mean over 14 planes.\nErrors are the correlated per-event "
          "estimator (NOTES_GEOMCLOSURE s1).  Nothing fitted.\n")

    res = {}
    for pair in args.pairs:
        ton, toff = pair.split(":")
        area = SIMS[ton]["area"]
        for func in args.funcs:
            for tag, rad in ((ton, True), (toff, False)):
                for corr in (False, True):
                    res[(tag, rad, corr, func)] = _rows(area, tag, rad, corr,
                                                        func)

    for func in args.funcs:
        for pair in args.pairs:
            ton, toff = pair.split(":")
            area = SIMS[ton]["area"]
            print(f"### {func}   pT = {AREA_PT[area]:g}   "
                  f"radON sim `{ton}`, radOFF sim `{toff}`   "
                  f"cut {SIMS[ton]['cut']:g} cm")
            print(f"  {'cell':<26s} " + "".join(f"{u:>10.3g}" for u in UCURVE)
                  + f"{'rms':>10}")
            cells = {}
            for rad in (True, False):
                tag = ton if rad else toff
                for corr in (False, True):
                    r = res[(tag, rad, corr, func)]
                    m = r["rows"].mean(axis=0)
                    cells[(rad, corr)] = (m, r["err"])
                    lab = (f"rad{'ON ' if rad else 'OFF'} corr"
                           f"{'ON ' if corr else 'OFF'}")
                    print(f"  {lab:<26s} "
                          + "".join(f"{v:+10.5f}" for v in m)
                          + f"{_rms(m):10.5f}"
                          + f"   [{len(r['ks'])} pl, "
                            f"{int(np.median(r['ns']))} ev]")
                    print(f"  {'   +-':<26s} "
                          + "".join(f"{v:10.5f}" for v in r["err"]))
                    print(f"  {'   s_F (mean)':<26s} "
                          f"{np.mean(r['sc']['sF']):.6e}   "
                          f"1/I (mean) {np.mean(r['sc']['invI']):.4f}")
            print()
            # the answer: radOFF - radON at fixed correction state
            for corr in (False, True):
                a, ea = cells[(True, corr)]
                b, eb = cells[(False, corr)]
                e = np.hypot(ea, eb)
                print(f"  --- radOFF - radON, correction "
                      f"{'ON' if corr else 'OFF'}")
                print(f"  {'diff':<26s} " + "".join(f"{v:+10.5f}"
                                                    for v in (b - a)))
                print(f"  {'sigma':<26s} " + "".join(f"{v:+10.2f}"
                                                     for v in (b - a) / e))
                with np.errstate(divide="ignore", invalid="ignore"):
                    frac = np.where(np.abs(a) > 1e-12, 1.0 - b / a, np.nan)
                print(f"  {'removed':<26s}"
                      + "".join(f"{100*v:9.1f}%" for v in frac))
                print(f"  {'rms':<26s} radON {_rms(a):.5f} -> radOFF "
                      f"{_rms(b):.5f}   "
                      f"({100*(1-_rms(b)/_rms(a)):+.1f} % of the rms removed)")
                print()
            # and the correction's effect at fixed radiation state, for the
            # 2x2 to be readable both ways
            for rad in (True, False):
                a, _ = cells[(rad, False)]
                b, _ = cells[(rad, True)]
                print(f"  --- corrON - corrOFF, radiation "
                      f"{'ON' if rad else 'OFF'}:  rms {_rms(a):.5f} -> "
                      f"{_rms(b):.5f}  "
                      f"({100*(1-_rms(b)/_rms(a)):+.1f} % removed)")
            print()

    if args.perplane:
        for func in args.funcs:
            for pair in args.pairs:
                ton, toff = pair.split(":")
                area = SIMS[ton]["area"]
                print(f"### {func} per plane (radius-ordered), "
                      f"pT = {AREA_PT[area]:g}")
                rg = legs_meta(area, True, False)["refglobr"]
                cols = [(True, False), (True, True), (False, False),
                        (False, True)]
                print(f"    {'k':>3} {'r[cm]':>8} " + "".join(
                    f"{('rad' + ('ON' if r else 'OFF') + '/c' + ('ON' if c else 'OFF')):>14s}"
                    for r, c in cols) + "   (u = 0.1)")
                got = {}
                for rad, corr in cols:
                    tag = ton if rad else toff
                    got[(rad, corr)] = res[(tag, rad, corr, func)]
                ks = got[(True, False)]["ks"]
                j = list(UCURVE).index(0.1)
                for i, k in enumerate(ks):
                    print(f"    {k:3d} {rg[k]:8.2f} " + "".join(
                        f"{got[(r, c)]['rows'][i, j]:+14.5f}" for r, c in cols))
                print()

    if args.dump:
        o = {}
        for (tag, rad, corr, func), r in res.items():
            key = f"{tag}|{int(rad)}|{int(corr)}|{func}"
            o[key + "|rows"] = r["rows"]
            o[key + "|err"] = r["err"]
            o[key + "|ks"] = r["ks"]
            o[key + "|sF"] = r["sc"]["sF"]
            o[key + "|invI"] = r["sc"]["invI"]
        np.savez(args.dump, u=UCURVE, **o)
        print(f"wrote {args.dump}")


# ==========================================================================
# subcommand: kok -- the 2x2 again, with the Kokoulin correction on
# ==========================================================================

def cmd_kok(args):
    print("=" * 118)
    print("THE 2x2 WITH GEANT4'S KOKOULIN RADIATIVE CORRECTION ON")
    print("=" * 118)
    print("Kokoulin is a correction TO the exact-delta channel and lives only "
          "in regime 2/3,\nso the correction-OFF column cannot move; that is "
          "checked, not assumed.  Both halves\nof the switch are exercised: "
          "`cf` = offline CF only (NOTES_SAMPLERGAP's configuration,\nthe "
          "control), `both` = C++ export + offline CF (the single structural "
          "switch).\n")

    arms = []
    for a in args.arms:
        if a == "off":
            arms.append(("kok OFF", False, False))
        elif a == "cf":
            arms.append(("kok CF only", True, False))
        elif a == "both":
            arms.append(("kok BOTH", True, True))
        else:
            raise SystemExit(f"unknown arm {a}")

    for pair in args.pairs:
        ton, toff = pair.split(":")
        area = SIMS[ton]["area"]
        for func in args.funcs:
            print(f"### {func}   pT = {AREA_PT[area]:g}   sims `{ton}` / "
                  f"`{toff}`   cut {SIMS[ton]['cut']:g} cm   "
                  f"Kokoulin from max(tcut, 100 keV), "
                  f"tcut = {MODEL_TCUT.get(ton, 0.0):g} MeV")
            print(f"  {'cell':<30s} " + "".join(f"{u:>10.3g}" for u in UCURVE)
                  + f"{'rms':>10}")
            cells = {}
            for corr in (False, True):
                for rad in (True, False):
                    tag = ton if rad else toff
                    for lab, kok, kmodel in arms:
                        # with the correction off the record is regime 1 and
                        # NEITHER Kokoulin branch is entered; only the OFF arm
                        # is run there, and the identity is asserted separately
                        if not corr and kok:
                            continue
                        r = _rows(area, tag, rad, corr, func, kok=kok,
                                  kokmodel=kmodel)
                        m = r["rows"].mean(axis=0)
                        cells[(corr, rad, lab)] = (m, r["err"], r["sc"])
                        name = (f"rad{'ON ' if rad else 'OFF'} "
                                f"corr{'ON ' if corr else 'OFF'} {lab}")
                        print(f"  {name:<30s} "
                              + "".join(f"{v:+10.5f}" for v in m)
                              + f"{_rms(m):10.5f}"
                              + f"   [{len(r['ks'])} pl]")
                        print(f"  {'   +-':<30s} "
                              + "".join(f"{v:10.5f}" for v in r["err"]))
                        print(f"  {'   s_F (mean)':<30s} "
                              f"{np.mean(r['sc']['sF']):.8e}   "
                              f"1/I (mean) {np.mean(r['sc']['invI']):.4f}")
            print()

            # THE ANSWER: radOFF - radON at correction ON, per Kokoulin arm
            for lab, _, _ in arms:
                if (True, True, lab) not in cells:
                    continue
                a, ea, _ = cells[(True, True, lab)]
                b, eb, _ = cells[(True, False, lab)]
                e = np.hypot(ea, eb)
                d = b - a
                print(f"  --- radOFF - radON, correction ON, {lab}")
                print(f"  {'diff':<30s} " + "".join(f"{v:+10.5f}" for v in d))
                print(f"  {'sigma':<30s} " + "".join(f"{v:+10.2f}"
                                                     for v in d / e))
                with np.errstate(divide="ignore", invalid="ignore"):
                    frac = np.where(np.abs(a) > 1e-12, 1.0 - b / a, np.nan)
                print(f"  {'removed':<30s}"
                      + "".join(f"{100*v:9.1f}%" for v in frac))
                print(f"  {'sign of the shift':<30s}"
                      + "".join(f"{('+' if v > 0 else '-'):>10s}" for v in d))
                print(f"  {'rms':<30s} radON {_rms(a):.5f} -> radOFF "
                      f"{_rms(b):.5f}   "
                      f"({100*(1-_rms(b)/_rms(a)):+.1f} % of the rms removed)")
                # the u >= 0.3 block, which is the feature the sign test is on
                j = [i for i, u in enumerate(UCURVE) if u >= 0.3]
                print(f"  {'u >= 0.3 only: radON':<30s}"
                      + "".join(f"{a[i]:+10.5f}" for i in j))
                print(f"  {'               radOFF':<30s}"
                      + "".join(f"{b[i]:+10.5f}" for i in j))
                print(f"  {'               change':<30s}"
                      + "".join(f"{100*(1-b[i]/a[i]):+9.1f}%" for i in j))
                print()

    if args.dump:
        print(f"(no dump implemented for `kok`; use the printed table)")


def cmd_kokid(args):
    """Two identities the Kokoulin arm rests on, both measured.

    (1) the NEW library reproduces the OLD models bit for bit with
        CVH_IONI_KOKOULIN unset -- i.e. this note's already-published 2x2 is
        unaffected by the rebuild that landed between the two runs;
    (2) with the exact-delta correction OFF the record is regime 1 and the
        Kokoulin switch cannot reach it, so `*_off_kok` must equal `*_off`.
    """
    print("=" * 100)
    print("KOKOULIN -- THE TWO IDENTITIES")
    print("=" * 100)
    h, p = _libhash()
    print(f"propagator library now {h}\n")
    print("(1) the rebuilt library against the models this note published "
          "(exported before it)")
    print(f"  {'pair':<48s} {'branches':>9} {'same':>6} {'diff':>6}  verdict")
    bad = 0
    for area in args.areas:
        for rad in (True, False):
            for corr in (False, True):
                lab = MODEL_LABEL[(rad, corr)]
                old = model_path(area, rad, corr)
                new = os.path.join(args.newdir, f"{area}_model_{lab}.root")
                if not os.path.exists(new):
                    print(f"  {area + '/' + lab:<48s} (missing {new})")
                    continue
                A, B = ds._branch_digests(old), ds._branch_digests(new)
                keys = sorted(set(A) | set(B))
                diff = [k for k in keys if A.get(k) != B.get(k)]
                ok = len(keys) > 0 and not diff
                bad += 0 if ok else 1
                print(f"  {area + '/' + lab + '  old lib vs new lib':<48s} "
                      f"{len(keys):>9} {len(keys)-len(diff):>6} {len(diff):>6}"
                      f"  {'BIT-IDENTICAL' if ok else 'DIFFER: ' + ','.join(k.split('/')[-1] for k in diff[:6])}")
    print()
    print("(2) with the exact-delta correction OFF the record is regime 1 and "
          "the Kokoulin\n    branch is not entered, in the C++ or offline")
    print(f"  {'pair':<48s} {'branches':>9} {'same':>6} {'diff':>6}  verdict")
    for area in args.areas:
        for rad in (True, False):
            for corr in (False, True):
                a = model_path(area, rad, corr, False)
                b = model_path(area, rad, corr, True)
                if not os.path.exists(b):
                    continue
                A, B = ds._branch_digests(a), ds._branch_digests(b)
                keys = sorted(set(A) | set(B))
                diff = [k for k in keys if A.get(k) != B.get(k)]
                lab = MODEL_LABEL[(rad, corr)]
                exp = "must be IDENTICAL" if not corr else "must MOVE gsig2/Q"
                verdict = ("BIT-IDENTICAL" if not diff
                           else "moved: " + ",".join(k.split('/')[-1]
                                                     for k in diff[:6]))
                good = (not diff) if not corr else bool(diff)
                bad += 0 if good else 1
                print(f"  {area + '/' + lab + ' vs _kok  (' + exp + ')':<48s} "
                      f"{len(keys):>9} {len(keys)-len(diff):>6} {len(diff):>6}"
                      f"  {verdict}")
    if bad:
        raise SystemExit(f"{bad} identity check(s) FAILED")
    print("\nboth identities hold")


# ==========================================================================
# subcommand: half -- how big is the radiative channel in the closure at all?
# ==========================================================================

def cmd_half(args):
    """The INCONSISTENT one-sided switch, quoted only for scale.

    Drop the radiative term from the model CF alone, keeping rad-ON data and a
    rad-ON reference.  This is not a physical configuration; it measures how
    much of the closure functional the radiative channel carries, i.e. the size
    of the error a WRONG radiative channel could produce.  NOTES_RADOFF s4
    measured it on the uncorrected model (-0.0626 at u = 0.03, pT = 40, 68 % of
    the whole non-closure).  Repeating it on the CORRECTED model bounds how
    much room is left for a radiative error now.

    The consistent switch is 4-20x smaller than this, which is the statement
    that the analytic brems CF and Geant4's sampled radiation agree far better
    than either agrees with zero.
    """
    print("=" * 110)
    print("ONE-SIDED (INCONSISTENT) CF HALF-SWITCH -- FOR SCALE ONLY")
    print("=" * 110)
    print("rad-ON sim, rad-ON reference, radiative term dropped from the model "
          "CF alone.\nNot a physical configuration.  It measures the SIZE of "
          "the radiative channel in the\nclosure functional, i.e. the room "
          "left for a radiative modelling error.\n")
    import cf_track_resolution as ctr
    for tag in args.tags:
        area = SIMS[tag]["area"]
        for corr, kok in args.states:
            sim = sim_of(tag)
            legs = legs_of(area, True, corr, kok)
            mp = model_path(area, True, corr, kok)
            out = {}
            for sw in (True, False):
                cpt.RAD_CHANNEL = sw
                ctr.IONI_KOKOULIN = 1.0 if kok else 0.0
                ctr.IONI_KOKOULIN_TCUT = MODEL_TCUT.get(tag, 0.0) if kok else 0.0
                fn._SCALE_CACHE.clear()
                cpt._PHI_CACHE.clear()
                os.environ["RES_NO_PHI_CACHE"] = "1"
                ld, st = fn._scale_cache_load, fn._scale_cache_store
                fn._scale_cache_load = lambda *a, **k: None
                fn._scale_cache_store = lambda *a, **k: None
                try:
                    sc = fn.plane_scales(legs, "qop", tag=mp,
                                         channels=chans(sw))
                    rows, _, ks, ns, err = gc.closure_rows(legs, sim, "qop",
                                                           sc["sF"])
                finally:
                    fn._scale_cache_load, fn._scale_cache_store = ld, st
                    cpt.RAD_CHANNEL = True
                    ctr.IONI_KOKOULIN = 0.0
                    ctr.IONI_KOKOULIN_TCUT = 0.0
                    fn._SCALE_CACHE.clear()
                    cpt._PHI_CACHE.clear()
                    os.environ.pop("RES_NO_PHI_CACHE", None)
                out[sw] = (rows.mean(axis=0), err)
            a, b = out[True][0], out[False][0]
            lab = (f"{tag} corr{'ON' if corr else 'OFF'}"
                   f" kok{'ON' if kok else 'OFF'}")
            print(f"### {lab}")
            print(f"  {'':<26s} " + "".join(f"{u:>10.3g}" for u in UCURVE)
                  + f"{'rms':>10}")
            print(f"  {'consistent (CF on)':<26s} "
                  + "".join(f"{v:+10.5f}" for v in a) + f"{_rms(a):10.5f}")
            print(f"  {'CF half-switch only':<26s} "
                  + "".join(f"{v:+10.5f}" for v in b) + f"{_rms(b):10.5f}")
            print(f"  {'the half-switch moved it':<26s} "
                  + "".join(f"{v:+10.5f}" for v in (b - a))
                  + f"{_rms(b - a):10.5f}")
            print()


# ==========================================================================

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)

    q = s.add_parser("export")
    q.add_argument("--areas", nargs="+", default=["pt3", "pt40"])
    q.add_argument("--force", action="store_true")
    q.add_argument("--kok", action="store_true",
                   help="export with CVH_IONI_KOKOULIN=1 (suffix _kok)")
    q.add_argument("--out", default="",
                   help="write elsewhere (used for the rebuilt-library "
                        "bit-identity control)")
    q.set_defaults(fn=cmd_export)

    q = s.add_parser("bitid")
    q.add_argument("--areas", nargs="+", default=["pt3", "pt40"])
    q.set_defaults(fn=cmd_bitid)

    q = s.add_parser("sim")
    q.add_argument("tags", nargs="+")
    q.add_argument("--events", type=int, default=20000)
    q.add_argument("--jobs", type=int, default=10)
    q.add_argument("--seed0", type=int, default=101)
    q.add_argument("--workers", type=int, default=20)
    q.set_defaults(fn=cmd_sim)

    q = s.add_parser("live")
    q.add_argument("--areas", nargs="+", default=["pt3", "pt40"])
    q.add_argument("--tags", nargs="+",
                   default=["pt3_cut1e4", "pt3_cut1e4_ro",
                            "pt40_cut001", "pt40_cut001_ro"])
    q.set_defaults(fn=cmd_live)

    q = s.add_parser("pairs")
    q.add_argument("--tags", nargs="+",
                   default=["pt3_cut1e4", "pt3_cut1e4_ro",
                            "pt40_cut001", "pt40_cut001_ro"])
    q.set_defaults(fn=cmd_pairs)

    q = s.add_parser("closure")
    q.add_argument("--pairs", nargs="+",
                   default=["pt3_cut1e4:pt3_cut1e4_ro",
                            "pt40_cut001:pt40_cut001_ro"])
    q.add_argument("--funcs", nargs="+", default=["qop"])
    q.add_argument("--perplane", action="store_true")
    q.add_argument("--dump", default="")
    q.set_defaults(fn=cmd_closure)

    q = s.add_parser("half")
    q.add_argument("--tags", nargs="+", default=["pt3_cut1e4", "pt40_cut001"])
    q.add_argument("--kok", action="store_true",
                   help="also measure on the Kokoulin-corrected model")
    q.set_defaults(fn=cmd_half)

    q = s.add_parser("kok")
    q.add_argument("--pairs", nargs="+",
                   default=["pt3_cut1e4:pt3_cut1e4_ro",
                            "pt40_cut1e4:pt40_cut1e4_ro"])
    q.add_argument("--funcs", nargs="+", default=["qop"])
    q.add_argument("--arms", nargs="+", default=["off", "cf", "both"])
    q.add_argument("--dump", default="")
    q.set_defaults(fn=cmd_kok)

    q = s.add_parser("kokid")
    q.add_argument("--areas", nargs="+", default=["pt3", "pt40"])
    q.add_argument("--newdir", default=os.path.join(SCRATCH, "ro2newlib"))
    q.set_defaults(fn=cmd_kokid)

    a = p.parse_args()
    if a.cmd == "half":
        a.states = ([(False, False), (True, False), (True, True)]
                    if a.kok else [(False, False), (True, False)])
    a.fn(a)


if __name__ == "__main__":
    main()
