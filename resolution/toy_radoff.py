#!/usr/bin/env python3
"""RADIATION-OFF closure test: is the momentum-dependent `qop` non-closure
caused by the radiative channel?

THE OBSERVATION BEING TESTED (NOTES_TOY_PT40)
---------------------------------------------
Going pT = 3 -> pT = 40 the layered toy's `qop` clean-propagation closure
degrades by 26-231 sigma at every probe (peak -0.0185 -> -0.0925) while `locx`
moves by at most 2.1 sigma.  Over the same step the radiative share of the
model's own `qop` kappa2 goes 2.5 % -> 18.6 %.  That is a CORRELATION.  This
module switches radiation off CONSISTENTLY ON BOTH SIDES and re-measures.

THREE SWITCHES, ALL OF WHICH MUST MOVE TOGETHER
-----------------------------------------------
1. SIMULATION.  `/process/inactivate muBrems` + `/process/inactivate
   muPairProd`.  Delta rays (muIoni) are untouched.
2. REFERENCE TRAJECTORY.  `CVH_IONONLY=1` makes the propagator's dE/dx table
   drop the radiative mean, so the reference is ionization-only -- which is the
   right reference for a sim that has no radiative loss.  Read in exactly one
   place (`cvhcgf::referenceIsIonOnly`), shared by the table and the block
   model.
3. MODEL CF.  `cf_propagation_test.RAD_CHANNEL = False` drops the
   `cf_brems_exact.rad_exponent` term from `model_phi`, and the Fisher scale is
   built from `channels=("ioni","ms")`.

Missing ANY one gives a plausible wrong answer: (1) alone leaves the model
subtracting a radiative mean that was never lost; (2) alone leaves the CF with
a fluctuation that is not in the data; (3) alone leaves the reference wrong.

WHY THE LIVENESS CHECKS ARE NOT OPTIONAL
----------------------------------------
`G4Commands` are applied by CMSSW in the G4 PreInit state and the return code
of `G4UImanager::ApplyCommand` is DISCARDED, so a refused command is completely
silent.  `live` therefore compares the two sims EVENT BY EVENT at a fixed seed:
if the switch were inert the two files would be bit-identical.  The same
applies to the reference (compare `refqop`) and to the CF (compare `model_phi`).

SUBCOMMANDS
-----------
    setup   <tag>   build the isolated area (geometry, planes, patched drivers)
    sim     <tag>   run the Geant4 sim
    model   <tag>   run the Geant4e model export
    live            switch-liveness evidence for all three switches
    pairs           model/sim pair validation
    closure         the closure tables, over the full u curve
"""

import argparse
import hashlib
import os
import re
import shutil
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
from cf_propagation_test import (FUNCTIONALS, load_model,       # noqa: E402
                                 model_variance)
import cgf_channels as cc                                       # noqa: E402

CMSSW = "/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev"
SRCTEST = f"{CMSSW}/src/Analysis/HitAnalyzer/test"
SCRATCH = fn.SCRATCH
AREAS = os.path.join(CMSSW, "toyradoff")     # must be under CMSSW_BASE
OUT = os.path.join(SCRATCH, "ro")
TP = os.path.join(SCRATCH, "tp")             # the published 400k rad-ON sample

UCURVE = fn.UCURVE
UHEAD = fn.UHEAD
IHEAD = fn.IHEAD

T_LAYER = 0.10

# The rad-ON side is NOT re-run: it is the published 400k NOTES_TOY_PT40 sample
# (same seed, same geometry, same stepper).  Re-running it could only introduce
# a difference that is not the one under test.
CONFIGS = {
    "pt3_radon":   dict(pt=3.0,  rad=True,
                        sim=f"{TP}/pt3_K1_sim.root",
                        model=f"{TP}/pt3_K1_model.root"),
    "pt3_radoff":  dict(pt=3.0,  rad=False),
    "pt40_radon":  dict(pt=40.0, rad=True,
                        sim=f"{TP}/pt40_K1_sim.root",
                        model=f"{TP}/pt40_K1_model.root"),
    "pt40_radoff": dict(pt=40.0, rad=False),
}
# one area per momentum -- geometry and planes do not depend on the rad switch
AREA_OF = {"pt3_radon": "pt3", "pt3_radoff": "pt3",
           "pt40_radon": "pt40", "pt40_radoff": "pt40"}
AREA_PT = {"pt3": 3.0, "pt40": 40.0}


def area(a):
    return os.path.join(AREAS, a)


def testdir(a):
    return os.path.join(area(a), "Analysis", "HitAnalyzer", "test")


def planes_name(pt):
    return f"toyPlanes_pt{int(pt) if float(pt).is_integer() else pt}.py"


def planes_path(tag):
    a = AREA_OF[tag]
    return os.path.join(testdir(a), planes_name(AREA_PT[a]))


def sim_path(tag, nev=None):
    if nev is not None:
        return os.path.join(OUT, f"{tag}_n{nev}_sim.root")
    return CONFIGS[tag].get("sim", os.path.join(OUT, f"{tag}_sim.root"))


def model_path(tag):
    return CONFIGS[tag].get("model", os.path.join(OUT, f"{tag}_model.root"))


# ==========================================================================
# area construction
# ==========================================================================

def _sub1(text, pat, repl, what, flags=0):
    new, n = re.subn(pat, repl, text, flags=flags)
    if n != 1:
        raise SystemExit(f"patch '{what}' matched {n} times, expected 1")
    return new


# The sim-side switch, injected mechanically into the private copy of
# runToyGeomCheck.py, immediately after the Watchers VPSet is defined.
#
# NOT `process.g4SimHits.G4Commands = ['/process/inactivate muBrems', ...]`.
# That route is SILENTLY INERT: CMSSW applies G4Commands while Geant4 is in
# G4State_PreInit, /process/inactivate is AvailableForStates(Idle, GeomClosed,
# EventProc), and the ApplyCommand return code is discarded.  Measured: two
# 2000-event pT = 40 runs at a fixed seed with those commands set came out
# BIT-IDENTICAL event for event (0 of 2000 differing).
#
# ProcessActivationWatcher instead calls G4ProcessTable::SetProcessActivation
# at BeginOfRun -- after the process managers exist -- and DUMPS the activation
# flag read back from each particle's G4ProcessManager, in BOTH configurations,
# so the log itself carries the before/after evidence.
RADOFF_BLOCK = '''    output=cms.string(opts.output),
))

# ---------------------------------------------------------------- RADIATION
_inact = []
if os.environ.get("TOY_RADOFF", "0") == "1":
    _inact = ['muBrems', 'muPairProd']
    print('[toy] RADOFF requested: muBrems + muPairProd inactivate')
else:
    print('[toy] RADON: full physics')
process.g4SimHits.Watchers.append(cms.PSet(
    type=cms.string('ProcessActivationWatcher'),
    inactivate=cms.untracked.vstring(*_inact),
    activate=cms.untracked.vstring(),
    particles=cms.untracked.vstring('mu-', 'mu+'),
))'''


def cmd_setup(args):
    a = args.area
    pt = AREA_PT[a]
    td = testdir(a)
    dd = os.path.join(area(a), "Analysis", "HitAnalyzer", "data")
    if os.path.exists(area(a)) and args.fresh:
        shutil.rmtree(area(a))
    os.makedirs(td, exist_ok=True)
    os.makedirs(dd, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)

    g = open(f"{SRCTEST}/gen_toy_config.py").read()
    g = _sub1(g,
              r"pt, eta, phi0, B, q = 3\.0, 0\.30, 0\.70, 3\.8, -1\.0",
              'pt, eta, phi0, B, q = float(os.environ["TOY_PT"]), '
              '0.30, 0.70, 3.8, -1.0',
              "gen: pT")
    g = _sub1(g,
              r'os\.path\.join\(HERE, "toyPlanes_pt3\.py"\)',
              'os.path.join(HERE, os.environ["TOY_PLANES"])',
              "gen: planes filename")
    open(f"{td}/gen_toy_config.py", "w").write(g)

    s = open(f"{SRCTEST}/runToyGeomCheck.py").read()
    s = _sub1(s, r"^import FWCore\.ParameterSet\.Config as cms$",
              "import os\nimport FWCore.ParameterSet.Config as cms",
              "sim: import os", flags=re.M)
    s = _sub1(s, r"^    output=cms\.string\(opts\.output\),\n\)\)$",
              RADOFF_BLOCK.rstrip("\n"), "sim: radiation switch", flags=re.M)
    open(f"{td}/runToyGeomCheck.py", "w").write(s)

    m = open(f"{SRCTEST}/runToyModel.py").read()
    m = _sub1(m, r"import toyPlanes_pt3 as planes",
              "import importlib, os as _os\n"
              'planes = importlib.import_module(_os.environ["TOY_PLANES_MOD"])',
              "model: planes import")
    open(f"{td}/runToyModel.py", "w").write(m)

    env = dict(os.environ, TOY_PT=repr(pt), TOY_PLANES=planes_name(pt))
    r = subprocess.run([sys.executable, "gen_toy_config.py",
                        f"{T_LAYER:.2f}", "1"],
                       cwd=td, env=env, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"gen failed:\n{r.stdout}\n{r.stderr}")
    print(f"[{a}] {r.stdout.strip()}")
    for f in (f"{dd}/tracker.xml",
              os.path.join(td, planes_name(pt)),
              f"{td}/runToyGeomCheck.py"):
        h = hashlib.md5(open(f, "rb").read()).hexdigest()
        print(f"[{a}]   {h}  {f}")


def _cmsrun(a, script, extra, log, env_extra=None):
    """cmsRun in the isolated area.  The environment is an ALLOWLIST, not
    `dict(os.environ)`: `calibration_studies/setup_env.sh` exports PYTHONPATH /
    PYTHONHOME / VIRTUAL_ENV / LD_LIBRARY_PATH for a different interpreter, and
    cmsRun's embedded Python then dies in a way that DEADLOCKS with zero output
    (see NOTES_TOY_PT40 section 0)."""
    td = testdir(a)
    keep = ("HOME", "USER", "LOGNAME", "SHELL", "TERM", "HOSTNAME", "TMPDIR",
            "X509_USER_PROXY", "KRB5CCNAME")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env["PATH"] = "/usr/local/bin:/usr/bin:/bin"
    # The four 2026-08-16 default-on corrections pinned to their
    # HISTORICAL state (all off); any explicit overlay below still
    # wins.  Same convention as deltaspec._clean_env -- an archived
    # model must stay comparable to a fresh export.
    import cf_track_resolution as _ctr
    env.update(_ctr.SWITCHES_OFF)
    env["TOY_PLANES_MOD"] = planes_name(AREA_PT[a])[:-3]
    env.update(env_extra or {})
    cmd = (f"source /cvmfs/cms.cern.ch/cmsset_default.sh >/dev/null 2>&1 && "
           f"cd {CMSSW}/src && eval $(scramv1 runtime -sh) && "
           f"export CMSSW_SEARCH_PATH={area(a)}:$CMSSW_SEARCH_PATH && "
           f"cd {td} && exec cmsRun {script} {extra}")
    os.makedirs(OUT, exist_ok=True)
    with open(log, "w") as fh:
        p = subprocess.run(["bash", "-c", cmd], stdout=fh,
                           stderr=subprocess.STDOUT, env=env)
    return p.returncode


def cmd_sim(args):
    tag = args.tag
    cfg = CONFIGS[tag]
    a = AREA_OF[tag]
    out = sim_path(tag, args.events if args.suffix else None)
    log = out[:-5] + ".log"
    rc = _cmsrun(a, "runToyGeomCheck.py",
                 f"events={args.events} pt={cfg['pt']} eta=0.30 output={out}",
                 log, env_extra={"TOY_RADOFF": "0" if cfg["rad"] else "1"})
    txt = open(log, errors="ignore").read(20000)
    st = ("TIGHT" if "[toy] TIGHT stepper" in txt else
          ("LOOSE" if "LOOSE stepper" in txt else "UNKNOWN"))
    rq = ("RADOFF" if "[toy] RADOFF requested" in txt else
          ("RADON" if "[toy] RADON" in txt else "UNKNOWN"))
    print(f"[{tag}] sim rc={rc} stepper={st} radswitch={rq} -> {out}")
    if rc or st != "TIGHT":
        raise SystemExit(f"[{tag}] SIM FAILED (rc={rc}, stepper={st}), see {log}")


def cmd_model(args):
    tag = args.tag
    cfg = CONFIGS[tag]
    a = AREA_OF[tag]
    out = model_path(tag)
    log = out[:-5] + ".log"
    ee = {} if cfg["rad"] else {"CVH_IONONLY": "1"}
    rc = _cmsrun(a, "runToyModel.py",
                 f"pt={cfg['pt']} eta=0.30 phi=0.70 output={out}", log,
                 env_extra=ee)
    txt = open(log, errors="ignore").read(200000)
    io = "CVH_IONONLY set" in txt
    print(f"[{tag}] model rc={rc} ionOnlyInLog={io} -> {out}")
    if rc or (io != (not cfg["rad"])):
        raise SystemExit(f"[{tag}] MODEL FAILED (rc={rc}, ionOnly={io}), "
                         f"see {log}")


# ==========================================================================
# loading
# ==========================================================================

_M, _S = {}, {}


def legs_of(tag):
    if tag not in _M:
        _M[tag] = load_model(model_path(tag))
    return _M[tag]


def _planes_ns(tag):
    ns = {}
    exec(open(planes_path(tag)).read(), ns)
    return ns


def sim_of(tag, path=None):
    key = (tag, path)
    if key not in _S:
        from toy_loader import load_toy_sim
        ns = _planes_ns(tag)
        _S[key] = load_toy_sim(path or sim_path(tag), ns["origin"],
                               ns["normal"], ns["uaxis"])
    return _S[key]


_R = {}


def legs_meta(tag, fields=("refglobr", "refp", "refpt", "refqop", "detid")):
    if tag not in _R:
        f = uproot.open(model_path(tag))
        tk = next(k for k in f.keys() if k.split(";")[0].endswith("/legs"))
        _R[tag] = {k: np.asarray(v) for k, v in
                   f[tk].arrays(list(fields), library="np").items()}
    return _R[tag]


def set_rad(on):
    """The model-CF switch.  `RAD_CHANNEL` is a module global of
    cf_propagation_test in exactly the same style as the pre-existing MS_NSUB /
    KMS_SCALE knobs, and defaults to True (production behaviour)."""
    cpt.RAD_CHANNEL = bool(on)


def chans(rad):
    return ("ioni", "ms", "rad") if rad else ("ioni", "ms")


# ==========================================================================
# subcommand: live -- switch liveness
# ==========================================================================

def _sim_arrays(path):
    t = uproot.open(path)["simstates"]
    return t.arrays(["detid", "globr", "pabs", "qop", "globx", "globy",
                     "globz"], library="np")


def cmd_live(args):
    print("=" * 78)
    print("SWITCH LIVENESS.  Each of the three switches must be shown to DO "
          "SOMETHING\nbefore any closure number is interpreted.")
    print("=" * 78)

    # ---------------- (1) the SIM switch
    print("\n### (1) SIMULATION: /process/inactivate muBrems + muPairProd")
    print("Fixed seed (initialSeed = 1) and identical geometry, so if the UI\n"
          "command were refused -- which CMSSW would not report, because\n"
          "G4UImanager::ApplyCommand's return code is discarded -- the two\n"
          "files would be BIT-IDENTICAL.\n")
    for pt in args.pts:
        nev = None if args.prod else args.events
        on = sim_path(f"pt{pt}_radon", nev)
        off = sim_path(f"pt{pt}_radoff", nev)
        if not (os.path.exists(on) and os.path.exists(off)):
            print(f"  pT={pt}: liveness sims missing ({on}) -- run `sim "
                  f"--suffix --events {args.events}` first")
            continue
        A, B = _sim_arrays(on), _sim_arrays(off)
        n = min(len(A["qop"]), len(B["qop"]))
        nd = sum(1 for i in range(n)
                 if not (len(A["qop"][i]) == len(B["qop"][i])
                         and np.array_equal(A["qop"][i], B["qop"][i])))
        # energy loss over the whole path, per event, first -> last crossing
        def dE(x):
            v = [1e3 * (x["pabs"][i][0] - x["pabs"][i][-1])
                 for i in range(len(x["pabs"])) if len(x["pabs"][i]) >= 2]
            return np.array(v)
        ea, eb = dE(A), dE(B)
        print(f"  pT={pt}  events {n}   differing events {nd} "
              f"({100.*nd/max(n,1):.2f} %)   "
              f"{'LIVE' if nd else '*** INERT -- SWITCH DID NOT TAKE ***'}")
        print(f"        dE [MeV]   radON  mean {ea.mean():8.4f} "
              f"median {np.median(ea):8.4f} p99 {np.percentile(ea,99):9.4f} "
              f"max {ea.max():10.3f}")
        print(f"                   radOFF mean {eb.mean():8.4f} "
              f"median {np.median(eb):8.4f} p99 {np.percentile(eb,99):9.4f} "
              f"max {eb.max():10.3f}")
        print(f"                   diff   mean {eb.mean()-ea.mean():+8.4f} "
              f"median {np.median(eb)-np.median(ea):+8.4f}")
        # ionization must be untouched: the LOW tail of the loss distribution
        for q in (1, 5, 16, 50, 84):
            print(f"          dE p{q:<2d}  radON {np.percentile(ea,q):9.4f}  "
                  f"radOFF {np.percentile(eb,q):9.4f}  "
                  f"diff {np.percentile(eb,q)-np.percentile(ea,q):+8.4f}")

    # ---------------- (2) the REFERENCE switch
    print("\n### (2) REFERENCE TRAJECTORY: CVH_IONONLY=1")
    print(f"  {'pT':>4} {'plane':>6} {'refp radON':>13} {'refp radOFF':>13} "
          f"{'d(refp) [MeV]':>14} {'d(refqop)/qop':>14}")
    for pt in args.pts:
        ton, toff = f"pt{pt}_radon", f"pt{pt}_radoff"
        if not os.path.exists(model_path(toff)):
            print(f"  pT={pt}: rad-off model missing")
            continue
        a, b = legs_meta(ton), legs_meta(toff)
        for k in (0, len(a["refp"]) // 2, len(a["refp"]) - 1):
            print(f"  {pt:>4} {k:>6} {a['refp'][k]:13.6f} "
                  f"{b['refp'][k]:13.6f} "
                  f"{1e3*(b['refp'][k]-a['refp'][k]):+14.5f} "
                  f"{(b['refqop'][k]-a['refqop'][k])/abs(a['refqop'][k]):+14.3e}")
        dEa = 1e3 * (a["refp"][0] - a["refp"][-1])
        dEb = 1e3 * (b["refp"][0] - b["refp"][-1])
        print(f"       total reference dE:  radON {dEa:.4f} MeV   "
              f"radOFF {dEb:.4f} MeV   radiative mean = {dEa-dEb:.4f} MeV")

    # ---------------- (3) the MODEL CF switch
    print("\n### (3) MODEL CF: cf_propagation_test.RAD_CHANNEL")
    print("  kappa2 share of the model's own CF (block_cgf_derivs at theta=0)\n"
          "  and the closure functional <e^{-u z^2}> of the model alone.")
    print(f"  {'config':<14s} {'func':<5s} {'RAD':<5s} {'ioni':>9} {'ms':>9} "
          f"{'rad':>9}  " + "".join(f"{'M(u=' + str(u) + ')':>13s}"
                                    for u in UHEAD))
    for pt in args.pts:
        for tag in (f"pt{pt}_radon", f"pt{pt}_radoff"):
            if not os.path.exists(model_path(tag)):
                continue
            legs = legs_of(tag)
            k = len(legs) - 1
            for func in ("qop",):
                avec = FUNCTIONALS[func]
                sig = float(np.sqrt(model_variance(legs, k, avec)[0]))
                sh = {}
                for c in ("ioni", "ms", "rad"):
                    blk = cc.collect_block(legs, k, avec, sig, channels=(c,))
                    sh[c] = float(np.atleast_1d(
                        cc.block_cgf_derivs(blk, 0.0, order=2)[2])[0])
                tot = sum(sh.values())
                for radsw in (True, False):
                    set_rad(radsw)
                    tau = fn.closure_tau(float(np.max(UHEAD)))
                    phi = cpt.model_phi(legs, k, avec, sig, tau)
                    m = [cpt.weier_scalar(phi, u, tau) for u in UHEAD]
                    print(f"  {tag:<14s} {func:<5s} {str(radsw):<5s} "
                          + "".join(f"{sh[c]/tot:9.5f}"
                                    for c in ("ioni", "ms", "rad"))
                          + "  " + "".join(f"{v:13.6f}" for v in m))
                set_rad(True)
    print()


# ==========================================================================
# subcommand: pairs
# ==========================================================================

def cmd_pairs(args):
    print("=" * 78)
    print("MODEL / SIM PAIR VALIDATION")
    print("=" * 78)
    ok_all = True
    for tag in args.configs:
        cfg = CONFIGS[tag]
        legs = legs_of(tag)
        sim = sim_of(tag)
        ns = _planes_ns(tag)
        npl = len(ns["radii"])
        a = legs_meta(tag)
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
        dr = np.abs(a["refglobr"] - rmed).max()
        set_rad(cfg["rad"])
        sc = fn.plane_scales(legs, "locx", tag=tag, channels=chans(cfg["rad"]))
        offx = np.array([np.nanmedian(sim["locx"][v[:, k], k])
                         for k in range(npl)])
        offx_sf = np.abs(offx / sc["sF"][:npl])
        pt_ok = abs(float(a["refpt"][0]) - cfg["pt"]) < 1e-3 * cfg["pt"]
        lg = os.path.splitext(sim_path(tag))[0] + ".log"
        st, rq = "N/A", "N/A"
        if os.path.exists(lg):
            txt = open(lg, errors="ignore").read(20000)
            st = ("TIGHT" if "[toy] TIGHT stepper" in txt else
                  ("LOOSE" if "LOOSE stepper" in txt else "UNKNOWN"))
            rq = ("RADOFF" if "[toy] RADOFF requested" in txt else
                  ("RADON" if "[toy] RADON" in txt else "UNKNOWN"))
        print(f"--- {tag}   pT={cfg['pt']:g}  rad={cfg['rad']}")
        print(f"    (1) legs {len(a['detid'])}  sim planes {v.shape[1]}  "
              f"planes {npl}   {'OK' if n_ok else 'MISMATCH'}")
        print(f"    (2) plane-index sequence identical: {seq_ok}")
        print(f"    (3) max |refglobr - sim median r| = {dr:.5f} cm")
        print(f"    (4) dE model(mean) {dE_m:8.3f} MeV   sim(median) "
              f"{dE_s:8.3f} MeV   gap {dE_m - dE_s:+7.3f} MeV")
        print(f"    (5) max |median locx| = {np.abs(offx).max():.5f} cm = "
              f"{offx_sf.max():.3f} s_F")
        print(f"    (6) model refpt[0] = {float(a['refpt'][0]):.4f} GeV  "
              f"{'OK' if pt_ok else 'MISMATCH'}")
        print(f"        sim events {sim['ntot']}, complete-sequence {nfull} "
              f"({100.*nfull/sim['ntot']:.3f} %)")
        print(f"    (7) stepper {st}   sim rad switch {rq}")
        # the rad-ON sims are the PUBLISHED NOTES_TOY_PT40 files, produced
        # before the switch existed, so their logs carry no marker: "UNKNOWN"
        # is the correct reading there and only "RADOFF" would be a failure.
        rq_ok = (rq in ("RADON", "UNKNOWN", "N/A") if cfg["rad"]
                 else rq == "RADOFF")
        good = (n_ok and seq_ok and dr < 1e-3 and offx_sf.max() < 5.0
                and pt_ok and st in ("TIGHT", "N/A") and rq_ok)
        ok_all &= good
        print(f"    VERDICT: {'PASS' if good else 'FAIL'}\n")
    print(f"ALL: {'PASS' if ok_all else 'SOME FAILED'}")
    set_rad(True)


# ==========================================================================
# subcommand: closure
# ==========================================================================

def _rows(tag, func):
    cfg = CONFIGS[tag]
    set_rad(cfg["rad"])
    legs, sim = legs_of(tag), sim_of(tag)
    sc = fn.plane_scales(legs, func, tag=tag, channels=chans(cfg["rad"]))
    rows, errs, ks, ns, err = gc.closure_rows(legs, sim, func, sc["sF"])
    set_rad(True)
    return dict(rows=rows, errs=errs, ks=ks, ns=ns, err=err, scales=sc,
                legs=legs, scale=sc["sF"])


def cmd_closure(args):
    print("=" * 78)
    print("RADIATION-OFF CLOSURE, FISHER NORMALIZATION")
    print("=" * 78)
    print("s_F = sigma sqrt(1/I); 1/I by exact FFT inversion of the model CF\n"
          "with the SAME channel set the CF uses (rad dropped when radiation\n"
          "is off).  The error is the CORRELATED per-event estimator of\n"
          "NOTES_GEOMCLOSURE section 1 -- err_plane/sqrt(nplanes) is wrong by\n"
          "~2.8x because every plane is evaluated on the same events.\n")
    res = {}
    for tag in args.configs:
        for func in args.funcs:
            res[(tag, func)] = _rows(tag, func)

    for func in args.funcs:
        print(f"### {func}   (mean over planes; u in units of s_F)")
        print(f"  {'config':<20s} " + "".join(f"{u:>10.3g}" for u in UCURVE))
        for tag in args.configs:
            r = res[(tag, func)]
            print(f"  {tag:<20s} "
                  + "".join(f"{v:+10.5f}" for v in r["rows"].mean(axis=0))
                  + f"   [{len(r['ks'])} planes, "
                    f"{int(np.median(r['ns']))} ev/plane]")
            print(f"  {'   +- (stat, corr)':<20s} "
                  + "".join(f"{v:10.5f}" for v in r["err"]))
            print(f"  {'   rms over planes':<20s} "
                  + "".join(f"{v:10.5f}" for v in r["rows"].std(axis=0)))
        print()

    # the difference that IS the answer
    print("### radOFF - radON, same momentum, errors added in quadrature "
          "(independent sims)")
    for func in args.funcs:
        for pt in args.pts:
            ton, toff = f"pt{pt}_radon", f"pt{pt}_radoff"
            if (ton, func) not in res or (toff, func) not in res:
                continue
            a = res[(ton, func)]["rows"].mean(axis=0)
            b = res[(toff, func)]["rows"].mean(axis=0)
            ea, eb = res[(ton, func)]["err"], res[(toff, func)]["err"]
            e = np.hypot(ea, eb)
            print(f"  {func:<5s} pT={pt}")
            print(f"    {'u':<10s}" + "".join(f"{u:>10.3g}" for u in UCURVE))
            print(f"    {'radON':<10s}" + "".join(f"{v:+10.5f}" for v in a))
            print(f"    {'radOFF':<10s}" + "".join(f"{v:+10.5f}" for v in b))
            print(f"    {'diff':<10s}" + "".join(f"{v:+10.5f}"
                                                 for v in (b - a)))
            print(f"    {'sigma':<10s}" + "".join(f"{v:+10.2f}"
                                                  for v in (b - a) / e))
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = np.where(np.abs(a) > 1e-12, 1.0 - b / a, np.nan)
            print(f"    {'removed':<10s}"
                  + "".join(f"{100*v:9.1f}%" for v in frac))
            print()

    if args.perplane:
        for func in args.funcs:
            print(f"### {func}: per-plane profile (radius-ordered)")
            for tag in args.configs:
                r = res[(tag, func)]
                rg = legs_meta(tag)["refglobr"]
                print(f"  {tag}")
                print(f"    {'k':>3} {'r[cm]':>8} {'1/I':>9} {'s_F':>12} "
                      + "".join(f"{'u=' + str(u):>19s}" for u in UHEAD))
                for i, k in enumerate(r["ks"]):
                    print(f"    {k:3d} {rg[k]:8.2f} "
                          f"{r['scales']['invI'][k]:9.4f} "
                          f"{r['scale'][k]:12.4e} "
                          + "".join(f"{v:+.5f}+-{e:.5f}".rjust(19)
                                    for v, e in zip(r["rows"][i, IHEAD],
                                                    r["errs"][i, IHEAD])))
                print(f"    {'mean':>3} {'':>8} {'':>9} {'':>12} "
                      + "".join(f"{v:+.5f}+-{e:.5f}".rjust(19)
                                for v, e in zip(r["rows"].mean(axis=0)[IHEAD],
                                                r["err"][IHEAD])))
            print()

    if args.dump:
        o = {}
        for (tag, func), r in res.items():
            o[f"{tag}|{func}|rows"] = r["rows"]
            o[f"{tag}|{func}|err"] = r["err"]
            o[f"{tag}|{func}|ks"] = r["ks"]
            o[f"{tag}|{func}|invI"] = r["scales"]["invI"]
            o[f"{tag}|{func}|sF"] = r["scales"]["sF"]
            o[f"{tag}|{func}|sigma"] = r["scales"]["sigma"]
        np.savez(args.dump, u=UCURVE, **o)
        print(f"wrote {args.dump}")


def cmd_cleanup(args):
    for f in (f"{CMSSW}/src/Analysis/HitAnalyzer/data/tracker.xml",
              f"{SRCTEST}/toyPlanes_pt3.py",
              f"{SRCTEST}/runToyGeomCheck.py",
              f"{SRCTEST}/gen_toy_config.py",
              f"{SRCTEST}/runToyModel.py"):
        h = hashlib.md5(open(f, "rb").read()).hexdigest()
        print(f"  {h}  {f}")
    for d in (AREAS, os.path.join(CMSSW, "toyscan_pt")):
        if os.path.isdir(d):
            shutil.rmtree(d)
            print(f"removed {d}")


# ==========================================================================

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)

    q = s.add_parser("setup")
    q.add_argument("area", choices=list(AREA_PT))
    q.add_argument("--fresh", action="store_true")
    q.set_defaults(fn=cmd_setup)

    q = s.add_parser("sim")
    q.add_argument("tag", choices=list(CONFIGS))
    q.add_argument("--events", type=int, default=400000)
    q.add_argument("--suffix", action="store_true",
                   help="name the output <tag>_n<events>_sim.root (liveness)")
    q.set_defaults(fn=cmd_sim)

    q = s.add_parser("model")
    q.add_argument("tag", choices=list(CONFIGS))
    q.set_defaults(fn=cmd_model)

    q = s.add_parser("live")
    q.add_argument("--pts", nargs="+", default=["3", "40"])
    q.add_argument("--events", type=int, default=2000)
    q.add_argument("--prod", action="store_true",
                   help="use the 400k production files instead of the small "
                        "fixed-seed liveness pair")
    q.set_defaults(fn=cmd_live)

    q = s.add_parser("pairs")
    q.add_argument("--configs", nargs="+", default=list(CONFIGS))
    q.set_defaults(fn=cmd_pairs)

    q = s.add_parser("closure")
    q.add_argument("--configs", nargs="+", default=list(CONFIGS))
    q.add_argument("--funcs", nargs="+", default=["qop", "locx"])
    q.add_argument("--pts", nargs="+", default=["3", "40"])
    q.add_argument("--perplane", action="store_true")
    q.add_argument("--dump", default="")
    q.set_defaults(fn=cmd_closure)

    q = s.add_parser("cleanup")
    q.set_defaults(fn=cmd_cleanup)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
