#!/usr/bin/env python3
"""Does the hard-delta mechanism of NOTES_TAILHUNT reproduce on the REAL CMS
tracker, and how much of the real-geometry residual is production-cut artifact?

NOTES_TAILHUNT established, on the layered toy:
  * the `qop` tail is `muIoni` delta rays (73-88 % of tail events carry one);
  * Geant4's delta-ray rate is 1.40x the offline model's Urban `a3` prediction,
    at both momenta and over a factor 1900 in threshold;
  * a matching increase of `a3` removes 97 % of the pT = 40 non-closure;
  * the delta-ray PRODUCTION CUT, 17.85 MeV in the toy's 9 g/cm3 material at
    the CMSSW default `DefaultCutValue = 1.0` cm, changes the SIMULATION's own
    q/p distribution by 125 % of the pT = 3 residual, partially cancelling the
    model's error.

The real tracker is different in three ways that all matter here: many
materials, a much shorter step in each, and -- crucially -- its own production
cuts REGION. `Geometry/TrackerSimData/data/trackerProdCuts.xml` assigns
`//Tracker` to `TrackerDeadRegion` with `ProdCutsForElectrons = 1*mm`, ten
times tighter than the global default, so Geant4 samples much more of the hard
tail explicitly and the toy's cancellation should be smaller. This module
measures all of that rather than assuming it.

WHAT IT RUNS
  real_base    untouched physics -- must reproduce the archived NOTES_RADOFF
               radiation-ON sample
  real_flat1mm CutsPerRegion = False, DefaultCutValue = 0.1 cm.  The CONTROL
               that separates "regions switched off" from "cut lowered": inside
               the tracker this is the same 1 mm cut the region already
               applies, so it must agree with real_base.
  real_cut1e4  CutsPerRegion = False, DefaultCutValue = 1e-4 cm -- the e-
               threshold hits Geant4's 990 eV floor and essentially the whole
               delta spectrum is sampled explicitly from the exact cross
               section.  This is the physically faithful configuration.

The MODEL is never re-run: it is the archived `real_model_pt3_radon.root`,
which has no notion of the simulation's production cut.  Every difference
between tags is therefore a property of the simulation alone.

PAIRING IS CHECKED BY DetId, NOT ASSUMED.  `model_pt10_eta0.30_phi0.20.root`
famously shares not one module with its supposed sim, so `pairs` compares the
model's `detid` list against the sim's before any closure number is produced.
"""

import argparse
import glob
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
import cf_track_resolution as ctr                               # noqa: E402
import fisher_norm as fn                                        # noqa: E402
import geom_closure as gc                                       # noqa: E402
import real_radoff as rr                                        # noqa: E402
from cf_propagation_test import load_model                      # noqa: E402

CMSSW = rr.CMSSW
SRCTEST = rr.SRCTEST
AREA = os.path.join(CMSSW, "tailhunt_real")   # must be under CMSSW_BASE
OUT = os.path.join(fn.SCRATCH, "thr")
PT, ETA, PHI = rr.PT, rr.ETA, rr.PHI
UCURVE, UHEAD, IHEAD = fn.UCURVE, fn.UHEAD, fn.IHEAD

# the archived NOTES_RADOFF radiation-ON model and sim
MODEL = rr.modelpath(True)
ARCHIVED_SIM = os.path.join(rr.simdir(True), "simstates_*.root")

PROCS = ["none", "muIoni", "muBrems", "muPairProd", "muonNuclear",
         "CoulombScat", "msc", "Transportation", "Decay", "otherIoni", "other"]
NPROC = len(PROCS)
I_EKIN0, I_NSTEP, I_DETOT, I_DEDEP, I_DESEC = 0, 1, 2, 3, 4
I_DEPROC = 5
I_ESECPROC = I_DEPROC + NPROC
I_NSECPROC = I_ESECPROC + NPROC
I_MAXDE = I_NSECPROC + NPROC
I_MAXDECODE, I_MAXDER, I_MAXSEC, I_MAXSECCODE, I_MAXSECR, I_NHARD, I_RLAST = (
    I_MAXDE + 1, I_MAXDE + 2, I_MAXDE + 3, I_MAXDE + 4, I_MAXDE + 5,
    I_MAXDE + 6, I_MAXDE + 7)
NREC = I_RLAST + 1

CONFIGS = {
    "real_base":    dict(regions=True,  cut=None),
    "real_flat1mm": dict(regions=False, cut=0.1),
    "real_cut1e4":  dict(regions=False, cut=1e-4),
    # 0.01 cm -> 0.122 MeV in silicon.  The toy's cut scan SATURATED at
    # 0.296 MeV (cut001 and cut1e4 agreed to ~1 sigma at both momenta), so this
    # is already inside the plateau and is ~30x cheaper than 1e-4 cm, where
    # Geant4 tracks every sub-keV delta through the whole tracker (measured:
    # ~30 min per 2000 events per task against ~1 min at the default).
    "real_cut01mm": dict(regions=False, cut=0.01),
}

# The whole sim-side switch set, injected in place of the TFileService line of
# runCleanPropSim.py.  ProcessActivationWatcher is kept with an EMPTY
# inactivate list in every configuration so that `hasWatchers` -- which changes
# whether the stepping action emits its signal at all -- is identical across
# tags, exactly as NOTES_RADOFF does.
PROBE_BLOCK = '''process.TFileService = cms.Service('TFileService', fileName=cms.string(opts.output))

# ---------------------------------------------------------------- TAIL PROBE
# See real_probe.py / NOTES_TAILHUNT.md.
_cut = os.environ.get("REAL_CUT", "")
if _cut:
    process.g4SimHits.Physics.CutsPerRegion = cms.bool(False)
    process.g4SimHits.Physics.DefaultCutValue = cms.double(float(_cut))
    print('[cleanprop] CutsPerRegion=False DefaultCutValue=%s cm' % _cut)
else:
    print('[cleanprop] production cuts UNTOUCHED (CutsPerRegion=True, '
          'tracker region 1 mm)')
process.g4SimHits.Watchers = cms.VPSet(
    cms.PSet(
        type=cms.string('ProcessActivationWatcher'),
        inactivate=cms.untracked.vstring(),
        activate=cms.untracked.vstring(),
    ),
    cms.PSet(
        type=cms.string('PrimaryLossCensusWatcher'),
        rmax=cms.untracked.double(float(os.environ.get("REAL_RMAX", "120.0"))),
        hard=cms.untracked.double(1.0),
        output=cms.untracked.string(os.environ.get("REAL_CENSUS",
                                                   "primaryloss.bin")),
    ),
)'''


def simdir(tag):
    return os.path.join(OUT, tag)


def sim_glob(tag):
    return os.path.join(simdir(tag), "simstates_*.root")


def census_glob(tag):
    return os.path.join(simdir(tag), "census_*.bin")


def _sub1(text, pat, repl, what, flags=0):
    new, n = re.subn(pat, repl, text, flags=flags)
    if n != 1:
        raise SystemExit(f"patch '{what}' matched {n} times, expected 1")
    return new


def _env(extra):
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
    e.update(extra)
    return e


def cmd_setup(args):
    if os.path.exists(AREA) and args.fresh:
        shutil.rmtree(AREA)
    os.makedirs(AREA, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    s = open(f"{SRCTEST}/runCleanPropSim.py").read()
    if "\nimport os\n" not in s and not s.startswith("import os"):
        s = _sub1(s, r"^import FWCore\.ParameterSet\.Config as cms$",
                  "import os\nimport FWCore.ParameterSet.Config as cms",
                  "sim: import os", flags=re.M)
    s = _sub1(s,
              r"^process\.TFileService = cms\.Service\('TFileService', "
              r"fileName=cms\.string\(opts\.output\)\)$",
              PROBE_BLOCK, "sim: watcher block", flags=re.M)
    open(f"{AREA}/runCleanPropSim.py", "w").write(s)
    print(f"wrote {AREA}/runCleanPropSim.py")


def cmd_sim(args):
    for tag in args.tags:
        cfg = CONFIGS[tag]
        d = simdir(tag + getattr(args, "suffix", ""))
        os.makedirs(d, exist_ok=True)
        script = f"""
set -u
source /cvmfs/cms.cern.ch/cmsset_default.sh >/dev/null 2>&1
cd {CMSSW}/src && eval $(scramv1 runtime -sh)
cd {d}
run_task() {{
  i=$1
  n=$(printf '%04d' $i)
  f={d}/simstates_$n.root
  [ -s "$f" ] && return 0
  t={d}/.tmp_$n.root
  rm -f "$t"
  REAL_CENSUS={d}/.tmpcensus_$n.bin \
  {SRCTEST}/cmsswlock.sh run cmsRun {AREA}/runCleanPropSim.py \
      nEvents={args.nev} pt={PT} eta={ETA} phi={PHI} seed=$((i+1)) \
      partId=13 output="$t" > {d}/task_$n.log 2>&1 \
    && mv -f "$t" "$f" && mv -f {d}/.tmpcensus_$n.bin {d}/census_$n.bin \
    || {{ rm -f "$t" {d}/.tmpcensus_$n.bin; }}
}}
export -f run_task
seq 0 $(({args.ntask}-1)) | xargs -P {args.npar} -I@ bash -c 'run_task @'
"""
        e = _env({"REAL_CUT": "" if cfg["cut"] is None else repr(cfg["cut"]),
                  "REAL_RMAX": repr(args.rmax)})
        p = subprocess.run(["bash", "-c", script], env=e)
        n = len(glob.glob(sim_glob(tag)))
        nc = len(glob.glob(census_glob(tag)))
        print(f"[{tag}] rc={p.returncode}  sim {n}/{args.ntask}  "
              f"census {nc}/{args.ntask} -> {d}")


# ==========================================================================
# loading
# ==========================================================================

_M, _S, _C = {}, {}, {}


def legs_of():
    if MODEL not in _M:
        _M[MODEL] = load_model(MODEL)
    return _M[MODEL]


def sim_of(path):
    if path not in _S:
        _S[path] = cpt.load_sim(path, acceptance="perplane")
    return _S[path]


def census_of(tag):
    """(nev, NREC) float32 concatenated in the SAME sorted order `load_sim`
    concatenates the ROOT files, so row i here is event i there."""
    if tag in _C:
        return _C[tag]
    cf = sorted(glob.glob(census_glob(tag)))
    sf = sorted(glob.glob(sim_glob(tag)))
    if not cf:
        raise FileNotFoundError(census_glob(tag))
    ks = [os.path.basename(f).split("_")[-1].split(".")[0] for f in cf]
    kss = [os.path.basename(f).split("_")[-1].split(".")[0] for f in sf]
    if ks != kss:
        raise SystemExit(f"census/sim file sets differ for {tag}")
    _C[tag] = np.concatenate(
        [np.fromfile(f, dtype=np.float32).reshape(-1, NREC) for f in cf],
        axis=0)
    return _C[tag]


# ==========================================================================
# subcommand: pairs -- DetId-level validation before any closure number
# ==========================================================================

def cmd_pairs(args):
    print("=" * 100)
    print("MODEL / SIM PAIRING, BY DetId")
    print("=" * 100)
    legs = legs_of()
    f = uproot.open(MODEL)
    tk = next(k for k in f.keys() if k.split(";")[0].endswith("/legs"))
    a = f[tk].arrays(["detid", "refglobr", "refp", "refpt"], library="np")
    mdet = set(int(x) for x in np.asarray(a["detid"]))
    print(f"  model {MODEL}\n    {len(mdet)} modules, refpt[0] = "
          f"{float(a['refpt'][0]):.4f} GeV, r = {a['refglobr'][0]:.2f} .. "
          f"{a['refglobr'][-1]:.2f} cm")
    ok_all = True
    for tag in args.tags:
        path = ARCHIVED_SIM if tag == "ARCHIVED" else sim_glob(tag)
        files = sorted(glob.glob(path))
        if not files:
            print(f"  {tag}: NO FILES at {path}")
            ok_all = False
            continue
        det = set()
        for fn_ in files[:args.nfiles]:
            t = uproot.open(fn_)["simstates/simstates"]
            det |= set(int(x) for x in
                       np.concatenate(t.arrays(["detid"],
                                               library="np")["detid"]))
        inter = mdet & det
        print(f"  {tag}: {len(files)} files, {len(det)} distinct sim modules, "
              f"{len(inter)} shared with the model "
              f"({100.*len(inter)/max(len(mdet),1):.1f} % of the model's)")
        good = len(inter) >= 0.9 * len(mdet)
        ok_all &= good
        print(f"    {'PASS' if good else '*** FAIL -- NOT THE SAME GEOMETRY'}")
    print(f"\nALL: {'PASS' if ok_all else 'SOME FAILED'}")


# ==========================================================================
# subcommand: cuts -- what the production cut actually is, per material
# ==========================================================================

def cmd_cuts(args):
    print("=" * 100)
    print("THE e- PRODUCTION CUT IN THE REAL TRACKER, from "
          "G4ProductionCutsTable")
    print("=" * 100)
    for tag in args.tags:
        lg = sorted(glob.glob(os.path.join(simdir(tag), "task_*.log")))
        if not lg:
            print(f"  {tag}: no logs")
            continue
        txt = open(lg[0], errors="ignore").read()
        head = [l for l in txt.splitlines() if l.startswith("[cleanprop] ")]
        rows = [l for l in txt.splitlines() if l.startswith("[plcensus]   ")
                and "material" not in l]
        print(f"\n### {tag}   ({head[0] if head else '?'})")
        print(f"  {'material':<24}{'rho':>9}{'range[mm]':>11}"
              f"{'Ecut_e- [MeV]':>15}")
        seen = {}
        for l in rows:
            p = l.split()
            if len(p) < 7:
                continue
            try:
                idx, name, rho, rng, eg, ee = (p[1], " ".join(p[2:-4]), p[-4],
                                               p[-3], p[-2], p[-1])
                float(rho)
            except (ValueError, IndexError):
                continue
            if name in seen:
                continue
            seen[name] = (float(rho), float(rng), float(ee))
        for name in args.materials:
            for k, v in seen.items():
                if name.lower() in k.lower():
                    print(f"  {k:<24}{v[0]:9.3f}{v[1]:11.4g}{v[2]:15.5g}")
        print(f"  ({len(seen)} distinct materials in the couple table)")


# ==========================================================================
# subcommand: census
# ==========================================================================

ZBANDS = [(-1e9, -8), (-8, -4), (-4, 0), (0, 4), (4, 8), (8, 1e9)]


def _zof(path, func="qop"):
    legs, sim = legs_of(), sim_of(path)
    sc = fn.plane_scales(legs, func, tag=MODEL, channels=("ioni", "ms", "rad"))
    nl = min(sim["valid"].shape[1], len(legs))
    z = np.full(sim["valid"].shape[:2], np.nan)
    for k in range(nl):
        good = sim["valid"][:, k] & np.isfinite(sim[cpt.SIM_BRANCH[func]][:, k])
        z[good, k] = ((sim[cpt.SIM_BRANCH[func]][good, k]
                       - legs[k][cpt.REF_BRANCH[func]]) / float(sc["sF"][k]))
    return z, sim, legs, sc, nl


def cmd_census(args):
    print("=" * 110)
    print("REAL TRACKER: per-event process census of the primary's energy "
          "loss, banded by z")
    print("=" * 110)
    for tag in args.tags:
        c = census_of(tag)
        z, sim, legs, sc, nl = _zof(sim_glob(tag))
        k = nl - 1
        n = min(len(c), z.shape[0])
        c, zk = c[:n], z[:n, k]
        good = np.isfinite(zk)
        print(f"\n### {tag}   {n} events, outermost plane {k}, "
              f"s_F = {sc['sF'][k]:.4e}")
        print(f"    mean dE(total) = {c[:, I_DETOT].mean():.4f} MeV "
              f"[median {np.median(c[:, I_DETOT]):.4f}], deposited "
              f"{c[:, I_DEDEP].mean():.4f}, to secondaries "
              f"{c[:, I_DESEC].mean():.4f} "
              f"({100*c[:, I_DESEC].mean()/max(c[:, I_DETOT].mean(),1e-9):.2f} %)")
        print(f"      {'process':<16}{'<dE_step>':>12}{'<E_sec>':>12}"
              f"{'<n_sec>':>10}{'frac dE':>10}")
        tot = c[:, I_DETOT].mean()
        for i, nm in enumerate(PROCS):
            d, e, ns = (c[:, I_DEPROC + i].mean(), c[:, I_ESECPROC + i].mean(),
                        c[:, I_NSECPROC + i].mean())
            if abs(d) < 1e-9 and abs(e) < 1e-9 and ns < 1e-9:
                continue
            print(f"      {nm:<16}{d:12.5f}{e:12.5f}{ns:10.4f}{d/tot:10.4f}")
        print(f"\n    {'z band':>10}{'n':>9}{'<dE>':>10}{'<dEsec>':>10}"
              f"{'muIoni sec %':>14}{'muBrems %':>11}{'muPair %':>10}"
              f"{'muNucl %':>10}")
        for lo, hi in ZBANDS:
            m = good & (zk >= lo) & (zk < hi)
            if m.sum() == 0:
                continue
            lab = f"{max(lo,-99):g}:{min(hi,99):g}"
            print(f"    {lab:>10}{m.sum():9d}{c[m, I_DETOT].mean():10.4f}"
                  f"{c[m, I_DESEC].mean():10.4f}"
                  + "".join(f"{100.*(c[m, I_NSECPROC+PROCS.index(p)]>0).mean():>{w}.3f}"
                            for p, w in (("muIoni", 14), ("muBrems", 11),
                                         ("muPairProd", 10),
                                         ("muonNuclear", 10))))


# ==========================================================================
# subcommand: hardrate
# ==========================================================================

def _model_hard(legs, k, T):
    N = E = Emean = Eexc = 0.0
    nst = 0
    for j in range(k + 1):
        s = legs[j]["ioni"]
        if not len(s):
            continue
        reg, gam = s[:, 0], s[:, 9]
        a1, e1, a2, e2 = s[:, 2], s[:, 3] * gam, s[:, 4], s[:, 5] * gam
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
    print("REAL TRACKER: the model's Urban a3 channel against Geant4's own "
          "explicit deltas")
    print("=" * 100)
    legs = legs_of()
    k = len(legs) - 1
    _, _, Emean, Eexc, nst = _model_hard(legs, k, 1e12)
    Tmax = float(np.max(legs[k]["ioni"][:, 8] * legs[k]["ioni"][:, 9]))
    print(f"  model: {k+1} legs, {nst} ionization steps, "
          f"Tmax(last leg) = {Tmax:.3f} MeV")
    print(f"  model ionization mean: excitation {Eexc:.4f} + delta-channel "
          f"{Emean:.4f} = {Eexc+Emean:.4f} MeV")
    print()
    for tag in args.tags:
        c = census_of(tag)
        Ns = float(c[:, I_NSECPROC + PROCS.index("muIoni")].mean())
        Es = float(c[:, I_ESECPROC + PROCS.index("muIoni")].mean())
        print(f"### {tag}   {len(c)} events")
        print(f"    sim mean dE(primary, r < rmax) = {c[:, I_DETOT].mean():.4f}"
              f" MeV, explicit muIoni secondaries: n = {Ns:.5f}, "
              f"E = {Es:.5f} MeV")
        print(f"    {'T [MeV]':>10}{'N_model':>12}{'N_sim':>12}{'ratio':>9}"
              f"{'E_model':>12}{'E_sim':>12}{'ratio':>9}")
        for T in args.thresholds:
            Nm, Em, _, _, _ = _model_hard(legs, k, T)
            print(f"    {T:>10.4g}{Nm:12.5f}{Ns:12.5f}"
                  f"{(Ns/Nm if Nm > 0 else np.nan):9.3f}"
                  f"{Em:12.5f}{Es:12.5f}"
                  f"{(Es/Em if Em > 0 else np.nan):9.3f}")
        print()


# ==========================================================================
# subcommand: closure / gauge
# ==========================================================================

def _exc_scale(legs, f):
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


def _rows(path, func, f=1.0):
    legs = legs_of()
    ctr.IONI_A3_SCALE = float(f)
    ctr.IONI_EXC_SCALE = _exc_scale(legs, f)
    sc = fn.plane_scales(legs, func, tag=MODEL, channels=("ioni", "ms", "rad"))
    out = gc.closure_rows(legs, sim_of(path), func, sc["sF"])
    ctr.IONI_A3_SCALE = 1.0
    ctr.IONI_EXC_SCALE = 1.0
    return out + (sc,)


def cmd_closure(args):
    print("=" * 100)
    print("REAL TRACKER CLOSURE against the UNCHANGED archived model")
    print("=" * 100)
    res = {}
    paths = {t: (ARCHIVED_SIM if t == "ARCHIVED" else sim_glob(t))
             for t in args.tags}
    for func in args.funcs:
        print(f"### {func}")
        print(f"  {'config':<16s} " + "".join(f"{u:>10.3g}" for u in UCURVE))
        for t in args.tags:
            rows, errs, ks, ns, err, sc = _rows(paths[t], func)
            res[(t, func)] = (rows, err, ks)
            print(f"  {t:<16s} "
                  + "".join(f"{v:+10.5f}" for v in rows.mean(axis=0))
                  + f"   [{len(ks)} pl, {int(np.median(ns))} ev]")
            print(f"  {'   +-':<16s} " + "".join(f"{v:10.5f}" for v in err))
        print()
    base = args.tags[0]
    for func in args.funcs:
        b, eb, _ = res[(base, func)]
        for t in args.tags[1:]:
            v, ev, _ = res[(t, func)]
            d = v.mean(axis=0) - b.mean(axis=0)
            print(f"  {func:<5s} {t} - {base}")
            print(f"    {'diff':<8s}" + "".join(f"{x:+10.5f}" for x in d))
            print(f"    {'sigma':<8s}"
                  + "".join(f"{x:+10.2f}" for x in d / np.hypot(eb, ev)))
    print()


def cmd_gauge(args):
    print("=" * 100)
    print("REAL TRACKER: the a3 gauge, per configuration and PER PLANE")
    print("=" * 100)
    for t in args.tags:
        path = ARCHIVED_SIM if t == "ARCHIVED" else sim_glob(t)
        curves = {}
        print(f"\n### {t}" + (f"   planes k <= {args.kmax}"
                                if args.kmax < 10**6 else ""))
        print(f"  {'f(a3)':>7} {'f(exc)':>8} "
              + "".join(f"{u:>10.3g}" for u in UCURVE) + f"{'rms':>10}")
        rg = None
        for f in args.scales:
            rows, errs, ks, ns, err, sc = _rows(path, args.func, f)
            sel = np.array([k <= args.kmax for k in ks])
            m = rows[sel].mean(axis=0)
            curves[float(f)] = (m, rows, ks)
            print(f"  {f:>7.3f} {_exc_scale(legs_of(), f):>8.4f} "
                  + "".join(f"{v:+10.5f}" for v in m)
                  + f"{np.sqrt((m ** 2).mean()):10.5f}")
            if f == args.scales[0]:
                print(f"  {'   +-':<17s}" + "".join(f"{v:10.5f}" for v in err))
        fs = np.array(sorted(curves))
        C = np.array([curves[f][0] for f in fs])
        ff = np.linspace(fs[0], fs[-1], 401)
        rms = np.array([np.sqrt(np.mean(np.array(
            [np.interp(x, fs, C[:, i]) for i in range(len(UCURVE))]) ** 2))
            for x in ff])
        ib = int(np.argmin(rms))
        r0 = np.sqrt(np.mean(curves[1.0][0] ** 2))
        print(f"  best single f = {ff[ib]:.4f}: rms {rms[ib]:.5f} against "
              f"{r0:.5f} at f = 1 -> {100*(1-rms[ib]/r0):.1f} % removed")

        if args.perplane:
            fmeta = uproot.open(MODEL)
            tk = next(k for k in fmeta.keys()
                      if k.split(";")[0].endswith("/legs"))
            rgl = np.asarray(fmeta[tk].arrays(["refglobr"],
                                              library="np")["refglobr"])
            i1 = IHEAD[-1]
            print(f"\n  PER-PLANE at u = 1 (the +0.0004 -> +0.0676 radial "
                  f"growth of NOTES_GEOMCLOSURE)")
            hdr = "".join(f"{'f='+str(f):>12}" for f in args.scales)
            print(f"    {'k':>3} {'r[cm]':>8}{hdr}")
            ks0 = curves[args.scales[0]][2]
            for i, k in enumerate(ks0):
                cells = []
                for f in args.scales:
                    m_, rows_, ks_ = curves[f]
                    j = list(ks_).index(k) if k in list(ks_) else None
                    cells.append(f"{rows_[j, i1]:+12.5f}" if j is not None
                                 else f"{'--':>12}")
                print(f"    {k:3d} {rgl[k]:8.2f}" + "".join(cells))


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)

    q = s.add_parser("setup")
    q.add_argument("--fresh", action="store_true")
    q.set_defaults(fn=cmd_setup)

    q = s.add_parser("sim")
    q.add_argument("tags", nargs="+", choices=list(CONFIGS))
    q.add_argument("--nev", type=int, default=2000)
    q.add_argument("--ntask", type=int, default=100)
    q.add_argument("--npar", type=int, default=40)
    q.add_argument("--rmax", type=float, default=120.0,
                   help="census integration limit [cm].  The model's legs stop "
                        "at the outermost scored module (r = 106.76 cm), so a "
                        "census taken to a larger radius counts deltas the "
                        "model never had a chance to predict; `--suffix` lets "
                        "a matched rmax = 106.8 sample be produced to measure "
                        "that correction instead of estimating it.")
    q.add_argument("--suffix", default="",
                   help="append to the output directory name")
    q.set_defaults(fn=cmd_sim)

    q = s.add_parser("pairs")
    q.add_argument("tags", nargs="+")
    q.add_argument("--nfiles", type=int, default=3)
    q.set_defaults(fn=cmd_pairs)

    q = s.add_parser("cuts")
    q.add_argument("tags", nargs="+")
    q.add_argument("--materials", nargs="+",
                   default=["Silicon", "Air", "Carbon", "Alumin", "Copper",
                            "Epoxy", "Kapton", "Nomex", "Steel", "Beryl"])
    q.set_defaults(fn=cmd_cuts)

    q = s.add_parser("census")
    q.add_argument("tags", nargs="+")
    q.set_defaults(fn=cmd_census)

    q = s.add_parser("hardrate")
    q.add_argument("tags", nargs="+")
    q.add_argument("--thresholds", nargs="+", type=float,
                   default=[0.001, 0.01, 0.1, 0.6, 1.0, 10.0])
    q.set_defaults(fn=cmd_hardrate)

    q = s.add_parser("closure")
    q.add_argument("tags", nargs="+")
    q.add_argument("--funcs", nargs="+", default=["qop"])
    q.set_defaults(fn=cmd_closure)

    q = s.add_parser("gauge")
    q.add_argument("tags", nargs="+")
    q.add_argument("--scales", nargs="+", type=float,
                   default=[1.0, 1.1, 1.2])
    q.add_argument("--func", default="qop")
    q.add_argument("--perplane", action="store_true")
    q.add_argument("--kmax", type=int, default=10**6,
                   help="restrict the plane MEAN to k <= kmax.  The real "
                        "tracker's residual is the sum of the hard-delta term "
                        "and a separate radial growth that only appears "
                        "beyond r ~ 50 cm; restricting to the inner planes "
                        "isolates the first.")
    q.set_defaults(fn=cmd_gauge)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
