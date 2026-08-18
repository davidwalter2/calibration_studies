#!/usr/bin/env python3
"""The radiation-off closure test on the REAL CMS tracker geometry.

Companion to `toy_radoff.py`, which does the same thing on the layered toy.
The three switches are identical and are described there:

  1. SIM       -- muBrems + muPairProd deactivated by ProcessActivationWatcher
                  (SimG4Core/HelpfulWatchers).  NOT `G4Commands`, which is
                  silently inert, and NOT a watcher living outside the
                  Simulation biglib, whose Geant4 singletons are a DIFFERENT
                  instance from the simulation's statically-linked one.
  2. REFERENCE -- CVH_IONONLY=1 on the model job.
  3. MODEL CF  -- cf_propagation_test.RAD_CHANNEL = False, and the Fisher scale
                  built from channels ("ioni", "ms").

WHY BOTH SIDES ARE RE-RUN, INCLUDING RADIATION ON
-------------------------------------------------
The published real-geometry sample (sim_260808tight_pt3_eta0.30_phi0.70) was
produced with NO watcher at all.  Adding a watcher flips CMSSW's `hasWatchers`,
which changes whether the stepping action emits its signal.  That should not
change physics -- but "should not" is exactly what this study keeps having to
check, so the radiation-ON side is regenerated through the SAME driver with an
EMPTY inactivate list, and the published profile is used only as a target to
reproduce.

The real geometry is CHEAP here: 100 tasks x 2000 events is ~10 s per task.

SUBCOMMANDS
  setup            write the patched sim driver into an area under CMSSW_BASE
  sim   <on|off>   run the 100-task production
  model <on|off>   run the deterministic model export
  closure          per-plane closure profiles, radiation on vs off
"""

import argparse
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
from cf_propagation_test import FUNCTIONALS, load_model         # noqa: E402

CMSSW = "/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev"
SRCTEST = f"{CMSSW}/src/Analysis/HitAnalyzer/test"
AREA = os.path.join(CMSSW, "radoff_real")     # must be under CMSSW_BASE
CEPH = "/ceph/submit/data/user/d/david_w/ZMass/cvh/cleanprop"
OUT = os.path.join(fn.SCRATCH, "ro")
TARGETS = ("/work/submit/david_w/ZMass/calibration_studies/resolution/"
           "cleanprop/targets/targets_mu_pt3_eta0.30.txt")

PT, ETA, PHI = 3.0, 0.30, 0.70
UCURVE, UHEAD, IHEAD = fn.UCURVE, fn.UHEAD, fn.IHEAD

WATCHER_BLOCK = '''process.TFileService = cms.Service('TFileService', fileName=cms.string(opts.output))

# ---------------------------------------------------------------- RADIATION
# See real_radoff.py / NOTES_RADOFF.md.  The watcher is present in BOTH
# configurations so that `hasWatchers` -- which changes whether the stepping
# action emits its signal -- is the same on the two sides; only the
# `inactivate` list differs.  Its EndOfRun step census is the evidence: a
# deactivated process must show EXACTLY zero steps.
_inact = ['muBrems', 'muPairProd'] if os.environ.get("REAL_RADOFF", "0") == "1" else []
print('[cleanprop] RADOFF=%s inactivate=%s'
      % (os.environ.get("REAL_RADOFF", "0"), _inact))
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    type=cms.string('ProcessActivationWatcher'),
    inactivate=cms.untracked.vstring(*_inact),
    activate=cms.untracked.vstring(),
))'''


def simdir(rad):
    return os.path.join(OUT, f"real_pt3_rad{'on' if rad else 'off'}")


def modelpath(rad):
    return os.path.join(OUT, f"real_model_pt3_rad{'on' if rad else 'off'}.root")


def _sub1(text, pat, repl, what, flags=0):
    new, n = re.subn(pat, repl, text, flags=flags)
    if n != 1:
        raise SystemExit(f"patch '{what}' matched {n} times, expected 1")
    return new


def cmd_setup(args):
    if os.path.exists(AREA) and args.fresh:
        shutil.rmtree(AREA)
    os.makedirs(AREA, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    s = open(f"{SRCTEST}/runCleanPropSim.py").read()
    s = _sub1(s,
              r"^process\.TFileService = cms\.Service\('TFileService', "
              r"fileName=cms\.string\(opts\.output\)\)$",
              WATCHER_BLOCK, "sim: watcher", flags=re.M)
    open(f"{AREA}/runCleanPropSim.py", "w").write(s)
    shutil.copy(f"{SRCTEST}/runCleanPropModel.py", f"{AREA}/runCleanPropModel.py")
    print(f"wrote {AREA}/runCleanPropSim.py")
    print(subprocess.run(["diff", f"{SRCTEST}/runCleanPropSim.py",
                          f"{AREA}/runCleanPropSim.py"],
                         capture_output=True, text=True).stdout)


def _env(extra=None):
    keep = ("HOME", "USER", "LOGNAME", "SHELL", "TERM", "HOSTNAME", "TMPDIR",
            "X509_USER_PROXY", "KRB5CCNAME")
    e = {k: os.environ[k] for k in keep if k in os.environ}
    e["PATH"] = "/usr/local/bin:/usr/bin:/bin"
    # The four 2026-08-16 default-on corrections pinned to their
    # HISTORICAL state (all off); any explicit overlay below still
    # wins.  Same convention as deltaspec._clean_env -- an archived
    # model must stay comparable to a fresh export.
    import cf_track_resolution as _ctr
    e.update(_ctr.SWITCHES_OFF)
    e.update(extra or {})
    return e


def cmd_sim(args):
    rad = args.rad == "on"
    d = simdir(rad)
    os.makedirs(d, exist_ok=True)
    script = f"""
set -u
source /cvmfs/cms.cern.ch/cmsset_default.sh >/dev/null 2>&1
cd {CMSSW}/src && eval $(scramv1 runtime -sh)
cd {d}
run_task() {{
  i=$1
  f={d}/simstates_$(printf '%04d' $i).root
  t={d}/.tmp_$(printf '%04d' $i).root
  [ -s "$f" ] && return 0
  rm -f "$t"
  cmsRun {AREA}/runCleanPropSim.py nEvents={args.nev} pt={PT} eta={ETA} \
      phi={PHI} seed=$((i+1)) partId=13 output="$t" \
      > {d}/task_$(printf '%04d' $i).log 2>&1 && mv -f "$t" "$f" || rm -f "$t"
}}
export -f run_task
seq 0 $(({args.ntask}-1)) | xargs -P {args.npar} -I@ bash -c 'run_task @'
"""
    e = _env({"REAL_RADOFF": "0" if rad else "1"})
    p = subprocess.run(["bash", "-c", script], env=e)
    n = len([f for f in os.listdir(d) if f.startswith("simstates_")])
    print(f"[real rad={args.rad}] rc={p.returncode}  files {n}/{args.ntask} -> {d}")


def cmd_model(args):
    rad = args.rad == "on"
    out = modelpath(rad)
    log = out[:-5] + ".log"
    # ReferenceIonizationOnly is a cmsRun OPTION now, not an exported name
    import cf_track_resolution as _ctr
    _swopts, _env_rest = _ctr.split_switches({} if rad else {"CVH_IONONLY": "1"})
    cmd = (f"source /cvmfs/cms.cern.ch/cmsset_default.sh >/dev/null 2>&1 && "
           f"cd {CMSSW}/src && eval $(scramv1 runtime -sh) && "
           f"cd {AREA} && exec cmsRun runCleanPropModel.py pt={PT} eta={ETA} "
           f"phi={PHI} partId=13 targets={TARGETS} output={out} {_swopts}")
    e = _env(_env_rest)
    with open(log, "w") as fh:
        p = subprocess.run(["bash", "-c", cmd], stdout=fh,
                           stderr=subprocess.STDOUT, env=e)
    io = "ReferenceIonizationOnly set" in open(log, errors="ignore").read()
    print(f"[real model rad={args.rad}] rc={p.returncode} ionOnlyInLog={io} -> {out}")
    if p.returncode or (io == rad):
        raise SystemExit(f"model failed, see {log}")


# ==========================================================================

def cmd_census(args):
    """The step census the sim watcher wrote, summed over tasks."""
    import collections
    for rad in (True, False):
        d = simdir(rad)
        if not os.path.isdir(d):
            continue
        tot = collections.Counter()
        nlog = 0
        for f in sorted(os.listdir(d)):
            if not f.startswith("task_"):
                continue
            nlog += 1
            for ln in open(os.path.join(d, f), errors="ignore"):
                if ln.startswith("[procact]   ") and " primary " in ln:
                    p = ln.split()
                    tot[p[1]] += int(p[3])
        print(f"--- real geometry, radiation {'ON' if rad else 'OFF'} "
              f"({nlog} tasks): PRIMARY steps by defining process")
        for k in sorted(tot):
            print(f"    {k:<26s} {tot[k]:12d}")
        print()


def _rows(rad, func):
    cpt.RAD_CHANNEL = rad
    legs = load_model(modelpath(rad))
    sim = cpt.load_sim(os.path.join(simdir(rad), "simstates_*.root"),
                       acceptance="perplane")
    sc = fn.plane_scales(legs, func, tag=f"real_rad{'on' if rad else 'off'}",
                         channels=("ioni", "ms", "rad") if rad else ("ioni", "ms"))
    rows, errs, ks, ns, err = gc.closure_rows(legs, sim, func, sc["sF"])
    cpt.RAD_CHANNEL = True
    return dict(rows=rows, errs=errs, ks=ks, ns=ns, err=err, scales=sc,
                legs=legs, sim=sim)


def cmd_closure(args):
    print("=" * 78)
    print("REAL TRACKER, pT = 3, eta = 0.30, phi = 0.70 -- RADIATION ON vs OFF")
    print("=" * 78)
    res = {}
    for rad in (True, False):
        for func in args.funcs:
            res[(rad, func)] = _rows(rad, func)
    for func in args.funcs:
        print(f"### {func} (mean over planes; u in units of s_F)")
        print(f"  {'config':<12s} " + "".join(f"{u:>10.3g}" for u in UCURVE))
        for rad in (True, False):
            r = res[(rad, func)]
            print(f"  {'radON' if rad else 'radOFF':<12s} "
                  + "".join(f"{v:+10.5f}" for v in r["rows"].mean(axis=0))
                  + f"   [{len(r['ks'])} planes, "
                    f"{int(np.median(r['ns']))} ev/plane]")
            print(f"  {'   +-':<12s} "
                  + "".join(f"{v:10.5f}" for v in r["err"]))
        a = res[(True, func)]["rows"].mean(axis=0)
        b = res[(False, func)]["rows"].mean(axis=0)
        e = np.hypot(res[(True, func)]["err"], res[(False, func)]["err"])
        print(f"  {'diff':<12s} " + "".join(f"{v:+10.5f}" for v in (b - a)))
        print(f"  {'sigma':<12s} " + "".join(f"{v:+10.2f}" for v in (b - a) / e))
        print()

    for func in args.funcs:
        print(f"### {func}: per-plane profile at u = 1 "
              f"(the +0.0004 -> +0.0676 radial rise)")
        ra, rb = res[(True, func)], res[(False, func)]
        f = uproot.open(modelpath(True))
        tk = next(k for k in f.keys() if k.split(";")[0].endswith("/legs"))
        rg = np.asarray(f[tk].arrays(["refglobr"], library="np")["refglobr"])
        i1 = IHEAD[-1]
        print(f"    {'k':>3} {'r[cm]':>8} {'radON':>20} {'radOFF':>20} "
              f"{'diff':>12} {'sig':>7}")
        for i, k in enumerate(ra["ks"]):
            j = list(rb["ks"]).index(k) if k in list(rb["ks"]) else None
            if j is None:
                continue
            va, ea = ra["rows"][i, i1], ra["errs"][i, i1]
            vb, eb = rb["rows"][j, i1], rb["errs"][j, i1]
            d, ed = vb - va, np.hypot(ea, eb)
            print(f"    {k:3d} {rg[k]:8.2f} {va:+.5f}+-{ea:.5f}".ljust(45)
                  + f"{vb:+.5f}+-{eb:.5f}".rjust(20)
                  + f"{d:+12.5f} {d/ed:+7.1f}")
        print()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    q = s.add_parser("setup"); q.add_argument("--fresh", action="store_true")
    q.set_defaults(fn=cmd_setup)
    q = s.add_parser("sim"); q.add_argument("rad", choices=("on", "off"))
    q.add_argument("--ntask", type=int, default=100)
    q.add_argument("--nev", type=int, default=2000)
    q.add_argument("--npar", type=int, default=50)
    q.set_defaults(fn=cmd_sim)
    q = s.add_parser("model"); q.add_argument("rad", choices=("on", "off"))
    q.set_defaults(fn=cmd_model)
    q = s.add_parser("census"); q.set_defaults(fn=cmd_census)
    q = s.add_parser("closure")
    q.add_argument("--funcs", nargs="+", default=["qop", "locx"])
    q.set_defaults(fn=cmd_closure)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
