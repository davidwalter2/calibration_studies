#!/usr/bin/env python3
"""Which Geant4 process populates the tail of the layered toy's `qop`
clean-propagation residual, and does the DELTA-RAY PRODUCTION CUT matter?

THE TWO QUESTIONS
-----------------
1. PROCESS CENSUS.  `PrimaryLossCensusWatcher` (new, inside the Simulation
   biglib) writes one fixed-length float32 record per event giving the primary's
   energy loss split by the process that defined the step and by whether the
   energy went into an explicit SECONDARY or into the continuous straggling
   integral, plus the hardest single step and the hardest single secondary.
   Joined by entry index with `simstates`, that turns "the residual lives at
   |z| ~ 6 s_F" into a named process.

2. THE PRODUCTION CUT.  The offline model is built from the `ioniurbanv` record
   exported with tcut = Tmax, i.e. the FULL delta spectrum folded into the
   continuous straggling.  Geant4 instead applies a range production cut:
   below it the loss is continuous and Urban-sampled with tcut = the cut
   energy; above it a delta ray is a real secondary track and the primary loses
   that energy discretely.  The muon loses the energy either way, so the TOTAL
   should be cut-independent -- that is the design goal of the cut system.
   **Scanning the cut with the model held fixed is therefore a null test with
   teeth**: any movement of the closure is a cut artifact, and its size is the
   size of the production-cut mismatch.

   The model is NOT re-run for the scan.  It cannot be: the exported record has
   no notion of the sim's cut.  That is exactly what makes the sim-only scan the
   right test.

VERIFY BEFORE INTERPRETING
--------------------------
Three separate controls in this project were provably inert, so every switch
here is checked from the log and from the data:
  * `TOY_CUT`  -> the watcher DUMPS G4ProductionCutsTable, so the cut energy in
                  the toy material is a printed measurement, not an assumption;
                  and `census` shows the secondary counts moving with it.
  * `TOY_INACT`-> ProcessActivationWatcher's step census must show EXACTLY zero
                  steps for a deactivated process.
  * geometry   -> the planes file and tracker.xml are md5-compared against the
                  archived NOTES_TOY_PT40 / NOTES_RADOFF area before the
                  archived MODEL is paired with a new SIM.

SUBCOMMANDS
    setup <area>        build the isolated area (geometry, planes, patched driver)
    sim <tag>           run the sim, optionally split over seeds
    census <tag>        read the binary census, tail vs bulk
    closure             closure of each tag against the UNCHANGED archived model
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
from cf_propagation_test import FUNCTIONALS                     # noqa: E402

CMSSW = "/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev"
SRCTEST = f"{CMSSW}/src/Analysis/HitAnalyzer/test"
SCRATCH = fn.SCRATCH
AREAS = os.path.join(CMSSW, "tailhunt")      # must be under CMSSW_BASE
OUT = os.path.join(SCRATCH, "th")
TP = os.path.join(SCRATCH, "tp")             # archived NOTES_TOY_PT40 sample
RO = os.path.join(SCRATCH, "ro")             # archived NOTES_RADOFF area

UCURVE = fn.UCURVE
UHEAD = fn.UHEAD
IHEAD = fn.IHEAD
T_LAYER = 0.10

# ---- the record layout, mirroring PrimaryLossCensusWatcher.cc ----
PROCS = ["none", "muIoni", "muBrems", "muPairProd", "muonNuclear",
         "CoulombScat", "msc", "Transportation", "Decay", "otherIoni", "other"]
NPROC = len(PROCS)
I_EKIN0, I_NSTEP, I_DETOT, I_DEDEP, I_DESEC = 0, 1, 2, 3, 4
I_DEPROC = 5
I_ESECPROC = I_DEPROC + NPROC
I_NSECPROC = I_ESECPROC + NPROC
I_MAXDE = I_NSECPROC + NPROC
I_MAXDECODE = I_MAXDE + 1
I_MAXDER = I_MAXDE + 2
I_MAXSEC = I_MAXDE + 3
I_MAXSECCODE = I_MAXDE + 4
I_MAXSECR = I_MAXDE + 5
I_NHARD = I_MAXDE + 6
I_RLAST = I_MAXDE + 7
NREC = I_RLAST + 1

AREA_PT = {"pt3": 3.0, "pt40": 40.0}

# tag -> (area, DefaultCutValue [cm], processes to inactivate in the SIM)
CONFIGS = {
    # baseline: identical physics to the archived 400k sample, re-run through
    # the seeded-split path so the cut variations have a matched control
    "pt3_base":   dict(area="pt3",  cut=1.0,  inact=[]),
    "pt3_cut001": dict(area="pt3",  cut=0.01, inact=[]),
    "pt3_cut1e4": dict(area="pt3",  cut=1e-4, inact=[]),
    "pt3_nonuc":  dict(area="pt3",  cut=1.0,  inact=["muonNuclear"]),
    "pt40_base":   dict(area="pt40", cut=1.0,  inact=[]),
    "pt40_cut001": dict(area="pt40", cut=0.01, inact=[]),
    "pt40_cut1e4": dict(area="pt40", cut=1e-4, inact=[]),
    "pt40_nonuc":  dict(area="pt40", cut=1.0,  inact=["muonNuclear"]),
}
# the model is NEVER re-run: it is the archived NOTES_TOY_PT40 NSUB=1 model
MODEL_OF = {"pt3": f"{TP}/pt3_K1_model.root", "pt40": f"{TP}/pt40_K1_model.root"}
ARCHIVED_SIM = {"pt3": f"{TP}/pt3_K1_sim.root", "pt40": f"{TP}/pt40_K1_sim.root"}


def area(a):
    return os.path.join(AREAS, a)


def testdir(a):
    return os.path.join(area(a), "Analysis", "HitAnalyzer", "test")


def planes_name(pt):
    return f"toyPlanes_pt{int(pt) if float(pt).is_integer() else pt}.py"


def planes_path(tag):
    a = CONFIGS[tag]["area"]
    return os.path.join(testdir(a), planes_name(AREA_PT[a]))


def sim_glob(tag):
    return os.path.join(OUT, f"{tag}_s*_sim.root")


def sim_path(tag, seed):
    return os.path.join(OUT, f"{tag}_s{seed}_sim.root")


def census_glob(tag):
    return os.path.join(OUT, f"{tag}_s*_census.bin")


def model_path(tag):
    return MODEL_OF[CONFIGS[tag]["area"]]


def _sub1(text, pat, repl, what, flags=0):
    new, n = re.subn(pat, repl, text, flags=flags)
    if n != 1:
        raise SystemExit(f"patch '{what}' matched {n} times, expected 1")
    return new


# The whole sim-side switch set, injected right after the Watchers VPSet.
PROBE_BLOCK = '''    output=cms.string(opts.output),
))

# ------------------------------------------------------------ TAIL PROBE
# (1) process activation + step census -- the same plugin NOTES_RADOFF used
_inact = [s for s in os.environ.get("TOY_INACT", "").split(",") if s]
if _inact:
    print('[toy] INACTIVATE requested: %s' % ",".join(_inact))
else:
    print('[toy] full physics (nothing inactivated)')
process.g4SimHits.Watchers.append(cms.PSet(
    type=cms.string('ProcessActivationWatcher'),
    inactivate=cms.untracked.vstring(*_inact),
    activate=cms.untracked.vstring(),
    particles=cms.untracked.vstring('mu-', 'mu+'),
))
# (2) per-event, per-process energy-loss census for the PRIMARY
process.g4SimHits.Watchers.append(cms.PSet(
    type=cms.string('PrimaryLossCensusWatcher'),
    rmax=cms.untracked.double(float(os.environ.get("TOY_RMAX", "107.0"))),
    hard=cms.untracked.double(1.0),
    output=cms.untracked.string(os.environ.get("TOY_CENSUS",
                                               "primaryloss.bin")),
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
    g = _sub1(g, r"pt, eta, phi0, B, q = 3\.0, 0\.30, 0\.70, 3\.8, -1\.0",
              'pt, eta, phi0, B, q = float(os.environ["TOY_PT"]), '
              '0.30, 0.70, 3.8, -1.0', "gen: pT")
    g = _sub1(g, r'os\.path\.join\(HERE, "toyPlanes_pt3\.py"\)',
              'os.path.join(HERE, os.environ["TOY_PLANES"])',
              "gen: planes filename")
    open(f"{td}/gen_toy_config.py", "w").write(g)

    s = open(f"{SRCTEST}/runToyGeomCheck.py").read()
    s = _sub1(s, r"^import FWCore\.ParameterSet\.Config as cms$",
              "import os\nimport FWCore.ParameterSet.Config as cms",
              "sim: import os", flags=re.M)
    s = _sub1(s, r"^process\.g4SimHits\.Physics\.DefaultCutValue = "
                 r"cms\.double\(1\.0\)$",
              'process.g4SimHits.Physics.DefaultCutValue = cms.double(\n'
              '    float(os.environ.get("TOY_CUT", "1.0")))\n'
              "print('[toy] DefaultCutValue = %g cm'\n"
              "      % process.g4SimHits.Physics.DefaultCutValue.value())",
              "sim: production cut", flags=re.M)
    s = _sub1(s, r"^    output=cms\.string\(opts\.output\),\n\)\)$",
              PROBE_BLOCK.rstrip("\n"), "sim: watcher block", flags=re.M)
    open(f"{td}/runToyGeomCheck.py", "w").write(s)

    env = dict(os.environ, TOY_PT=repr(pt), TOY_PLANES=planes_name(pt))
    r = subprocess.run([sys.executable, "gen_toy_config.py",
                        f"{T_LAYER:.2f}", "1"],
                       cwd=td, env=env, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"gen failed:\n{r.stdout}\n{r.stderr}")
    print(f"[{a}] {r.stdout.strip()}")

    # GEOMETRY IDENTITY.  A new sim is going to be paired with the ARCHIVED
    # model, so the geometry must be provably the same one the model was
    # exported against.  toyradoff/<a> is the NOTES_RADOFF area, built from the
    # same generator at the same T and NSUB.
    ref = os.path.join(CMSSW, "toyradoff", a)
    for rel in (f"Analysis/HitAnalyzer/data/tracker.xml",
                f"Analysis/HitAnalyzer/test/{planes_name(pt)}"):
        mine = os.path.join(area(a), rel)
        theirs = os.path.join(ref, rel)
        hm = hashlib.md5(open(mine, "rb").read()).hexdigest()
        if os.path.exists(theirs):
            ht = hashlib.md5(open(theirs, "rb").read()).hexdigest()
            ok = "IDENTICAL" if hm == ht else "*** DIFFERENT ***"
            print(f"[{a}]   {hm}  {rel}   vs archived {ht}  {ok}")
            if hm != ht:
                raise SystemExit("geometry differs from the archived area; the "
                                 "archived model cannot be reused")
        else:
            print(f"[{a}]   {hm}  {rel}   (no archived copy to compare)")


def _cmsrun(a, script, extra, log, env_extra=None):
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
           f"cd {td} && exec {SRCTEST}/cmsswlock.sh run "
           f"cmsRun {script} {extra}")
    os.makedirs(OUT, exist_ok=True)
    with open(log, "w") as fh:
        p = subprocess.run(["bash", "-c", cmd], stdout=fh,
                           stderr=subprocess.STDOUT, env=env)
    return p.returncode


def _run_one(argstuple):
    tag, seed, nev = argstuple
    cfg = CONFIGS[tag]
    a = cfg["area"]
    out = sim_path(tag, seed)
    log = out[:-5] + ".log"
    ee = {"TOY_CUT": repr(cfg["cut"]),
          "TOY_INACT": ",".join(cfg["inact"]),
          "TOY_CENSUS": os.path.join(OUT, f"{tag}_s{seed}_census.bin")}
    rc = _cmsrun(a, "runToyGeomCheck.py",
                 f"events={nev} pt={AREA_PT[a]} eta=0.30 output={out} "
                 f"seed={seed}", log, env_extra=ee)
    return tag, seed, rc, log


def cmd_sim(args):
    from concurrent.futures import ThreadPoolExecutor
    jobs = []
    for tag in args.tags:
        for i in range(args.jobs):
            jobs.append((tag, args.seed0 + i, args.events))
    print(f"launching {len(jobs)} jobs "
          f"({args.jobs} seeds x {args.events} events per tag)")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for tag, seed, rc, log in ex.map(_run_one, jobs):
            txt = open(log, errors="ignore").read(40000)
            st = ("TIGHT" if "[toy] TIGHT stepper" in txt else
                  ("LOOSE" if "LOOSE stepper" in txt else "UNKNOWN"))
            cut = re.search(r"\[toy\] DefaultCutValue = (\S+) cm", txt)
            print(f"  {tag} seed={seed} rc={rc} stepper={st} "
                  f"cut={cut.group(1) if cut else '?'}")
            if rc or st != "TIGHT":
                print(f"    *** FAILED, see {log}")


# ==========================================================================
# reading
# ==========================================================================

_M, _S, _C = {}, {}, {}


def legs_of(tag):
    p = model_path(tag)
    if p not in _M:
        _M[p] = cpt.load_model(p)
    return _M[p]


def _planes_ns(tag):
    ns = {}
    exec(open(planes_path(tag)).read(), ns)
    return ns


def sim_of(tag, path=None):
    key = (tag, path)
    if key not in _S:
        from toy_loader import load_toy_sim
        ns = _planes_ns(tag)
        _S[key] = load_toy_sim(path or sim_glob(tag), ns["origin"],
                               ns["normal"], ns["uaxis"])
    return _S[key]


def census_of(tag):
    """(nev, NREC) float32, concatenated over seeds in the SAME sorted order
    `load_toy_sim` concatenates the ROOT files, so row i here is event i
    there."""
    import glob
    if tag in _C:
        return _C[tag]
    files = sorted(glob.glob(census_glob(tag)))
    if not files:
        raise FileNotFoundError(census_glob(tag))
    # the sim glob and the census glob differ only by the suffix, so sorting
    # both gives the same seed order -- assert it rather than trust it
    sims = sorted(glob.glob(sim_glob(tag)))
    ks = [os.path.basename(f).rsplit("_", 1)[0] for f in files]
    kss = [os.path.basename(f).rsplit("_", 1)[0] for f in sims]
    if ks != kss:
        raise SystemExit(f"census/sim file sets differ:\n {ks}\n {kss}")
    parts = [np.fromfile(f, dtype=np.float32).reshape(-1, NREC) for f in files]
    _C[tag] = np.concatenate(parts, axis=0)
    return _C[tag]


def scales_of(tag, func="qop"):
    legs = legs_of(tag)
    return fn.plane_scales(legs, func, tag=model_path(tag),
                           channels=("ioni", "ms", "rad"))


def zof(tag, func="qop", path=None):
    legs, sim = legs_of(tag), sim_of(tag, path)
    sc = scales_of(tag, func)
    nl = min(sim["valid"].shape[1], len(legs))
    z = np.full(sim["valid"].shape[:2], np.nan)
    for k in range(nl):
        good = sim["valid"][:, k] & np.isfinite(sim[cpt.SIM_BRANCH[func]][:, k])
        z[good, k] = ((sim[cpt.SIM_BRANCH[func]][good, k]
                       - legs[k][cpt.REF_BRANCH[func]]) / float(sc["sF"][k]))
    return z, sim, legs, sc, nl


# ==========================================================================
# subcommand: census
# ==========================================================================

ZBANDS = [(-1e9, -8), (-8, -4), (-4, 0), (0, 4), (4, 6), (6, 8), (8, 10),
          (10, 1e9)]


def cmd_census(args):
    print("=" * 110)
    print("PER-EVENT PROCESS CENSUS OF THE PRIMARY'S ENERGY LOSS, "
          "banded by the standardized qop residual z")
    print("=" * 110)
    for tag in args.tags:
        c = census_of(tag)
        z, sim, legs, sc, nl = zof(tag)
        k = nl - 1
        n = min(len(c), z.shape[0])
        c, zk = c[:n], z[:n, k]
        good = np.isfinite(zk)
        print(f"\n### {tag}   {n} events, plane {k}, s_F = {sc['sF'][k]:.4e}")
        print(f"    mean dE(total) = {c[:, I_DETOT].mean():.4f} MeV, "
              f"median {np.median(c[:, I_DETOT]):.4f}, "
              f"deposited {c[:, I_DEDEP].mean():.4f}, "
              f"to secondaries {c[:, I_DESEC].mean():.4f} "
              f"({100*c[:, I_DESEC].mean()/max(c[:, I_DETOT].mean(),1e-9):.2f} %)")
        print(f"    mean dE by defining process [MeV] and secondary energy "
              f"[MeV] / count, whole sample:")
        print(f"      {'process':<16}{'<dE_step>':>12}{'<E_sec>':>12}"
              f"{'<n_sec>':>10}{'frac dE':>10}")
        tot = c[:, I_DETOT].mean()
        for i, nm in enumerate(PROCS):
            d = c[:, I_DEPROC + i].mean()
            e = c[:, I_ESECPROC + i].mean()
            ns = c[:, I_NSECPROC + i].mean()
            if abs(d) < 1e-9 and abs(e) < 1e-9 and ns < 1e-9:
                continue
            print(f"      {nm:<16}{d:12.5f}{e:12.5f}{ns:10.4f}"
                  f"{d/tot:10.4f}")
        print(f"    hardest single step: mean {c[:, I_MAXDE].mean():.4f} MeV, "
              f"p99 {np.percentile(c[:, I_MAXDE], 99):.3f}, "
              f"max {c[:, I_MAXDE].max():.2f}")
        hb = np.bincount(c[:, I_MAXDECODE].astype(int), minlength=NPROC)
        print("      by process: " + "  ".join(
            f"{PROCS[i]} {100.*v/n:.2f}%" for i, v in enumerate(hb) if v))

        print(f"\n    BANDED BY z (plane {k}):")
        print(f"      {'z band':>12}{'n':>9}{'<dE>':>10}{'<dEsec>':>10}"
              f"{'<maxdE>':>10}{'<n_hard>':>10}  "
              + "".join(f"{'dE:'+p:>16}" for p in
                        ("muIoni", "muBrems", "muPairProd", "muonNuclear")))
        for lo, hi in ZBANDS:
            m = good & (zk >= lo) & (zk < hi)
            if m.sum() == 0:
                continue
            lab = f"{max(lo,-99):g}:{min(hi,99):g}"
            print(f"      {lab:>12}{m.sum():9d}{c[m, I_DETOT].mean():10.4f}"
                  f"{c[m, I_DESEC].mean():10.4f}{c[m, I_MAXDE].mean():10.4f}"
                  f"{c[m, I_NHARD].mean():10.4f}  "
                  + "".join(f"{c[m, I_DEPROC+PROCS.index(p)].mean():16.5f}"
                            for p in ("muIoni", "muBrems", "muPairProd",
                                      "muonNuclear")))
        print(f"\n    FRACTION OF EVENTS WITH A SECONDARY FROM <process>, "
              f"by z band")
        print(f"      {'z band':>12}{'n':>9}"
              + "".join(f"{p:>14}" for p in
                        ("muIoni", "muBrems", "muPairProd", "muonNuclear",
                         "Decay")))
        for lo, hi in ZBANDS:
            m = good & (zk >= lo) & (zk < hi)
            if m.sum() == 0:
                continue
            lab = f"{max(lo,-99):g}:{min(hi,99):g}"
            print(f"      {lab:>12}{m.sum():9d}"
                  + "".join(f"{100.*(c[m, I_NSECPROC+PROCS.index(p)]>0).mean():13.3f}%"
                            for p in ("muIoni", "muBrems", "muPairProd",
                                      "muonNuclear", "Decay")))


# ==========================================================================
# subcommand: hardrate -- the model's own hard-collision spectrum against what
# Geant4 actually produced
# ==========================================================================

def _model_hard(legs, k, T):
    """(N, E, Emean_total) the model's Urban a3 channel above T [MeV],
    summed over every step up to plane k.

    The exported record is the Urban parameter block with tcut = Tmax: a3
    collisions per step drawn from p(E) = C/E^2 on [e0, tmax], C = 1/(1/e0 -
    1/tmax).  So above T the model predicts

        N(>T) = a3 C (1/T - 1/tmax),      E(>T) = a3 C ln(tmax/T)

    and the block's whole ionization MEAN is a1 e1 + a2 e2 + a3 C ln(tmax/e0).
    Those three numbers are directly comparable to the simulation's delta-ray
    count, delta-ray energy and reference energy loss.
    """
    N = E = Emean = Eexc = 0.0
    nst = 0
    for j in range(k + 1):
        s = legs[j]["ioni"]
        if not len(s):
            continue
        reg = s[:, 0]
        gam = s[:, 9]
        a1, e1 = s[:, 2], s[:, 3] * gam
        a2, e2 = s[:, 4], s[:, 5] * gam
        a3, e0, tmx = s[:, 6], s[:, 7] * gam, s[:, 8] * gam
        m = (reg != 0) & (a3 > 0) & (tmx > e0) & (e0 > 0)
        C = np.zeros_like(a3)
        C[m] = 1.0 / (1.0 / e0[m] - 1.0 / tmx[m])
        hi = m & (tmx > T)
        N += float(np.sum(a3[hi] * C[hi] * (1.0 / T - 1.0 / tmx[hi])))
        E += float(np.sum(a3[hi] * C[hi] * np.log(tmx[hi] / T)))
        Emean += float(np.sum(a3[m] * C[m] * np.log(tmx[m] / e0[m])))
        Eexc += float(np.sum(np.where(reg != 0, a1 * e1 + a2 * e2, 0.0)))
        nst += len(s)
    return N, E, Emean, Eexc, nst


def cmd_hardrate(args):
    print("=" * 100)
    print("THE MODEL'S HARD-COLLISION SPECTRUM vs WHAT GEANT4 ACTUALLY MADE")
    print("=" * 100)
    print("The offline record is Urban with tcut = Tmax: the WHOLE delta "
          "spectrum is folded\ninto the continuous a3 channel.  Geant4 puts "
          "everything above the production cut on\nan explicit secondary "
          "track instead.  Both descriptions must give the same total,\nand "
          "the same rate and energy above any threshold T.\n")
    for tag in args.tags:
        legs = legs_of(tag)
        k = len(legs) - 1
        c = census_of(tag)
        n = len(c)
        print(f"### {tag}   ({n} events, {k+1} legs)")
        Tmax = float(np.max(legs[k]["ioni"][:, 8] * legs[k]["ioni"][:, 9]))
        print(f"    record Tmax on the last leg = {Tmax:.3f} MeV")
        _, _, Emean, Eexc, nst = _model_hard(legs, k, 1e9)
        print(f"    model ionization mean over {nst} steps: "
              f"excitation {Eexc:.4f} + delta-channel {Emean:.4f} = "
              f"{Eexc+Emean:.4f} MeV")
        print(f"    sim mean dE(primary, r<107) = {c[:, I_DETOT].mean():.4f} "
              f"MeV  [median {np.median(c[:, I_DETOT]):.4f}]")
        print()
        print(f"    {'T [MeV]':>10}{'N_model':>12}{'N_sim':>12}{'ratio':>9}"
              f"{'E_model':>12}{'E_sim':>12}{'ratio':>9}")
        for T in args.thresholds:
            Nm, Em, _, _, _ = _model_hard(legs, k, T)
            # the sim's explicit muIoni secondaries are exactly the deltas
            # above the production cut; for T above the cut this is a fair
            # comparison, for T below it the sim has no secondaries to count
            Ns = float(c[:, I_NSECPROC + PROCS.index("muIoni")].mean())
            Es = float(c[:, I_ESECPROC + PROCS.index("muIoni")].mean())
            print(f"    {T:>10.4g}{Nm:12.5f}{Ns:12.5f}"
                  f"{(Ns/Nm if Nm > 0 else np.nan):9.3f}"
                  f"{Em:12.5f}{Es:12.5f}"
                  f"{(Es/Em if Em > 0 else np.nan):9.3f}")
        print("    (N_sim / E_sim are the counts and energies of EXPLICIT "
              "muIoni secondaries,\n     i.e. everything above the production "
              "cut -- they do not vary with T.)")
        print()


# ==========================================================================
# subcommand: closure
# ==========================================================================

def _rows(tag, func, path=None):
    legs, sim = legs_of(tag), sim_of(tag, path)
    sc = scales_of(tag, func)
    return gc.closure_rows(legs, sim, func, sc["sF"])


def cmd_closure(args):
    print("=" * 100)
    print("CLOSURE AGAINST THE UNCHANGED ARCHIVED MODEL "
          "(Fisher normalization, mean over planes)")
    print("=" * 100)
    print("The model is the NOTES_TOY_PT40 NSUB=1 export.  Only the SIM "
          "changes between tags,\nso every difference below is a property of "
          "the simulation alone.\n")
    res = {}
    for tag in args.tags:
        for func in args.funcs:
            res[(tag, func)] = _rows(tag, func)
    if args.archived:
        for a in sorted({CONFIGS[t]["area"] for t in args.tags}):
            t0 = next(t for t in CONFIGS if CONFIGS[t]["area"] == a)
            for func in args.funcs:
                res[(f"{a}_ARCHIVED", func)] = _rows(t0, func,
                                                     path=ARCHIVED_SIM[a])
    order = list(args.tags) + [f"{a}_ARCHIVED" for a in
                               sorted({CONFIGS[t]["area"] for t in args.tags})
                               if args.archived]
    for func in args.funcs:
        print(f"### {func}")
        print(f"  {'config':<20s} " + "".join(f"{u:>10.3g}" for u in UCURVE))
        for tag in order:
            if (tag, func) not in res:
                continue
            rows, errs, ks, ns, err = res[(tag, func)]
            print(f"  {tag:<20s} "
                  + "".join(f"{v:+10.5f}" for v in rows.mean(axis=0))
                  + f"   [{len(ks)} pl, {int(np.median(ns))} ev]")
            print(f"  {'   +-':<20s} " + "".join(f"{v:10.5f}" for v in err))
        print()
    # differences against the per-area baseline
    print("### difference against the same-area baseline "
          "(errors in quadrature)")
    for func in args.funcs:
        for a in sorted({CONFIGS[t]["area"] for t in args.tags
                         if t in CONFIGS}):
            base = f"{a}_base"
            if (base, func) not in res:
                continue
            b, eb = res[(base, func)][0].mean(axis=0), res[(base, func)][4]
            for tag in order:
                if tag == base or (tag, func) not in res:
                    continue
                if tag in CONFIGS and CONFIGS[tag]["area"] != a:
                    continue
                if tag.endswith("_ARCHIVED") and not tag.startswith(a):
                    continue
                v, ev = res[(tag, func)][0].mean(axis=0), res[(tag, func)][4]
                d = v - b
                s = d / np.hypot(eb, ev)
                print(f"  {func:<5s} {tag} - {base}")
                print(f"    {'diff':<8s}" + "".join(f"{x:+10.5f}" for x in d))
                print(f"    {'sigma':<8s}" + "".join(f"{x:+10.2f}" for x in s))
    print()


# ==========================================================================
# subcommand: gauge -- move the MODEL's hard-delta channel and see whether an
# ionization hard-tail mismatch of the measured size has the observed shape
# ==========================================================================

def cmd_gauge(args):
    import cf_track_resolution as ctr
    print("=" * 100)
    print("MODEL-SIDE GAUGE: scale the Urban a3 (hard-delta) channel, holding "
          "the block's MEAN loss fixed")
    print("=" * 100)
    print("a3 is the model's ENTIRE representation of hard delta rays.  "
          "`hardrate` measures how\nfar it is from Geant4's own explicit delta "
          "production.  This propagates a change of\nthat size through the "
          "closure's own transform and weights.  The excitation channels are\n"
          "rescaled to keep the block's mean loss fixed, so this is a SHAPE "
          "change only.\n")
    for spec in args.targets:
        tag, _, simtag = spec.partition(":")
        a = CONFIGS[tag]["area"]
        path = sim_glob(simtag) if simtag else ARCHIVED_SIM[a]
        legs = legs_of(tag)
        base = None
        curves = {}
        print(f"### {tag}   sim = {os.path.basename(path)}")
        print(f"  {'f(a3)':>7} {'f(exc)':>8} " +
              "".join(f"{u:>10.3g}" for u in UCURVE) + f"{'rms':>10}")
        for f in args.scales:
            ctr.IONI_A3_SCALE = float(f)
            # hold the block's MEAN loss fixed, summed over every leg
            num = den = 0.0
            for j in range(len(legs)):
                s = legs[j]["ioni"]
                if not len(s):
                    continue
                reg, gam = s[:, 0], s[:, 9]
                a3, e0, tmx = s[:, 6], s[:, 7] * gam, s[:, 8] * gam
                m = (reg != 0) & (a3 > 0) & (tmx > e0) & (e0 > 0)
                C = np.zeros_like(a3)
                C[m] = 1.0 / (1.0 / e0[m] - 1.0 / tmx[m])
                num += float(np.sum(a3[m] * C[m] * np.log(tmx[m] / e0[m])))
                den += float(np.sum(np.where(
                    reg != 0, s[:, 2] * s[:, 3] * gam + s[:, 4] * s[:, 5] * gam,
                    0.0)))
            fe = max((den + num * (1.0 - float(f))) / den, 0.0) if den > 0 else 1.0
            ctr.IONI_EXC_SCALE = fe
            rows, errs, ks, ns, err = _rows(tag, args.func, path=path)
            m = rows.mean(axis=0)
            curves[float(f)] = m
            if base is None:
                base = m
            print(f"  {f:>7.3f} {fe:>8.4f} "
                  + "".join(f"{v:+10.5f}" for v in m)
                  + f"{np.sqrt((m ** 2).mean()):10.5f}")
        ctr.IONI_A3_SCALE = 1.0
        ctr.IONI_EXC_SCALE = 1.0
        # per-probe required f, by linear interpolation of the closure in f
        fs = np.array(sorted(curves))
        C = np.array([curves[f] for f in fs])          # (nf, nu)
        print(f"\n  f(a3) that would zero EACH probe separately "
              f"(linear interpolation in f):")
        req = []
        for i, u in enumerate(UCURVE):
            y = C[:, i]
            s = np.argsort(y)
            req.append(float(np.interp(0.0, y[s], fs[s])))
        print(f"    {'u':<10}" + "".join(f"{u:>10.3g}" for u in UCURVE))
        print(f"    {'f_req':<10}" + "".join(f"{v:10.4f}" for v in req))
        print(f"    spread over the nine probes: {min(req):.4f} - "
              f"{max(req):.4f}  (a mechanism with the WRONG SHAPE has an "
              f"f_req\n    that is inconsistent or changes sign; one with the "
              f"right shape has a single f)")
        # best single f by minimizing the rms over probes
        ff = np.linspace(fs[0], fs[-1], 401)
        rms = np.array([np.sqrt(np.mean(
            np.array([np.interp(x, fs, C[:, i]) for i in range(len(UCURVE))])
            ** 2)) for x in ff])
        ib = int(np.argmin(rms))
        r0 = np.sqrt(np.mean(curves[1.0] ** 2)) if 1.0 in curves else np.nan
        print(f"    best single f = {ff[ib]:.4f}: rms over probes "
              f"{rms[ib]:.5f} against {r0:.5f} at f = 1 "
              f"-> {100*(1-rms[ib]/r0):.1f} % of the rms removed")
        print()


def _exc_scale(legs, f):
    """The excitation scale that holds the WHOLE trajectory's mean ionization
    loss fixed when a3 is multiplied by f."""
    num = den = 0.0
    for j in range(len(legs)):
        s = legs[j]["ioni"]
        if not len(s):
            continue
        reg, gam = s[:, 0], s[:, 9]
        a3, e0, tmx = s[:, 6], s[:, 7] * gam, s[:, 8] * gam
        m = (reg != 0) & (a3 > 0) & (tmx > e0) & (e0 > 0)
        C = np.zeros_like(a3)
        C[m] = 1.0 / (1.0 / e0[m] - 1.0 / tmx[m])
        num += float(np.sum(a3[m] * C[m] * np.log(tmx[m] / e0[m])))
        den += float(np.sum(np.where(
            reg != 0, s[:, 2] * s[:, 3] * gam + s[:, 4] * s[:, 5] * gam, 0.0)))
    return max((den + num * (1.0 - float(f))) / den, 0.0) if den > 0 else 1.0


def cmd_realgauge(args):
    """The same gauge on the REAL tracker geometry, using the 400k pT = 3
    radiation-ON sample NOTES_RADOFF produced.  Offline only."""
    import cf_track_resolution as ctr
    import real_radoff as rr
    print("=" * 100)
    print("THE a3 GAUGE ON THE REAL TRACKER (pT = 3, eta = 0.30, phi = 0.70)")
    print("=" * 100)
    legs = cpt.load_model(rr.modelpath(True))
    sim = cpt.load_sim(os.path.join(rr.simdir(True), "simstates_*.root"),
                       acceptance="perplane")
    print(f"model {rr.modelpath(True)}   {len(legs)} legs")
    print(f"sim   {rr.simdir(True)}      {sim['valid'].shape[0]} events\n")
    curves = {}
    print(f"  {'f(a3)':>7} {'f(exc)':>8} " +
          "".join(f"{u:>10.3g}" for u in UCURVE) + f"{'rms':>10}")
    for f in args.scales:
        ctr.IONI_A3_SCALE = float(f)
        ctr.IONI_EXC_SCALE = _exc_scale(legs, f)
        sc = fn.plane_scales(legs, args.func, tag=rr.modelpath(True),
                             channels=("ioni", "ms", "rad"))
        rows, errs, ks, ns, err = gc.closure_rows(legs, sim, args.func,
                                                  sc["sF"])
        m = rows.mean(axis=0)
        curves[float(f)] = m
        print(f"  {f:>7.3f} {ctr.IONI_EXC_SCALE:>8.4f} "
              + "".join(f"{v:+10.5f}" for v in m)
              + f"{np.sqrt((m ** 2).mean()):10.5f}"
              + f"   [{len(ks)} planes]")
        if f == args.scales[0]:
            print(f"  {'   +-':<17s}" + "".join(f"{v:10.5f}" for v in err))
    ctr.IONI_A3_SCALE = 1.0
    ctr.IONI_EXC_SCALE = 1.0
    fs = np.array(sorted(curves))
    C = np.array([curves[f] for f in fs])
    ff = np.linspace(fs[0], fs[-1], 401)
    rms = np.array([np.sqrt(np.mean(np.array(
        [np.interp(x, fs, C[:, i]) for i in range(len(UCURVE))]) ** 2))
        for x in ff])
    ib = int(np.argmin(rms))
    r0 = np.sqrt(np.mean(curves[1.0] ** 2))
    print(f"\n  best single f = {ff[ib]:.4f}: rms {rms[ib]:.5f} against "
          f"{r0:.5f} at f = 1 -> {100*(1-rms[ib]/r0):.1f} % removed")


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
    q.add_argument("tags", nargs="+", choices=list(CONFIGS))
    q.add_argument("--events", type=int, default=20000)
    q.add_argument("--jobs", type=int, default=1)
    q.add_argument("--seed0", type=int, default=101)
    q.add_argument("--workers", type=int, default=20)
    q.set_defaults(fn=cmd_sim)

    q = s.add_parser("census")
    q.add_argument("tags", nargs="+")
    q.set_defaults(fn=cmd_census)

    q = s.add_parser("hardrate")
    q.add_argument("tags", nargs="+")
    q.add_argument("--thresholds", nargs="+", type=float,
                   default=[0.0095, 1.0, 17.8507, 50.0, 100.0])
    q.set_defaults(fn=cmd_hardrate)

    q = s.add_parser("gauge")
    q.add_argument("targets", nargs="+",
                   help="<modeltag>[:<simtag>]; without :simtag the ARCHIVED "
                        "400k NOTES_TOY_PT40 sim is used")
    q.add_argument("--scales", nargs="+", type=float,
                   default=[1.0, 1.05, 1.1, 1.15, 1.2, 1.3])
    q.add_argument("--func", default="qop")
    q.set_defaults(fn=cmd_gauge)

    q = s.add_parser("realgauge")
    q.add_argument("--scales", nargs="+", type=float,
                   default=[1.0, 1.1, 1.2, 1.3])
    q.add_argument("--func", default="qop")
    q.set_defaults(fn=cmd_realgauge)

    q = s.add_parser("closure")
    q.add_argument("tags", nargs="+")
    q.add_argument("--funcs", nargs="+", default=["qop"])
    q.add_argument("--archived", action="store_true")
    q.set_defaults(fn=cmd_closure)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
