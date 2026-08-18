#!/usr/bin/env python3
"""Layered-toy clean-propagation closure at TWO momenta (pT = 3 and pT = 40),
NSUB = 1 / 4 / 16, at high statistics.

WHY
---
NOTES_GEOMCLOSURE settled the `locx` question (the +0.045 is a homogeneous-medium
artefact; the layered toy and the real tracker both sit at ~+0.001) but left
three things open, all of them statistics- or momentum-limited:

  * `qop` on the layered toy is +0.0008 to +0.0016 at u = 1 with a +-0.00034
    correlated error -- 2.4 to 4.6 sigma, i.e. suggestive and not settled;
  * the apparent NSUB trend in both channels is ~1.6 sigma (`qop`) and ~0.9
    sigma (`locx`), and an earlier NSUB "trend" at 2000 events turned out to be
    0.5-sigma noise that REVERSED at 100k;
  * everything was measured at pT = 3, where the radiative channel carries 2-3 %
    of the ionization kappa2.  At pT = 40 it carries a large fraction, so pT = 40
    tests a channel that is essentially absent at pT = 3 and dominant at Z
    momenta.

TWO THINGS THAT SILENTLY CORRUPT THIS TEST, AND HOW THEY ARE AVOIDED
--------------------------------------------------------------------
1.  `gen_toy_config.py` rewrites THREE shared files together -- the geometry
    (`Analysis/HitAnalyzer/data/tracker.xml`), the plane definitions
    (`test/toyPlanes_pt3.py`) and the watcher radii inside
    `test/runToyGeomCheck.py`.  Six geometry-dependent jobs cannot share one
    working area.

    This module therefore gives every configuration its OWN AREA:

        $CMSSW_BASE/toyscan_pt/<tag>/Analysis/HitAnalyzer/data/tracker.xml
        $CMSSW_BASE/toyscan_pt/<tag>/Analysis/HitAnalyzer/test/
              {gen_toy_config,runToyGeomCheck,runToyModel}.py
              toyPlanes_pt<PT>.py

    The area root is prepended to `CMSSW_SEARCH_PATH`, so `edm::FileInPath`
    resolves `Analysis/HitAnalyzer/data/tracker.xml` to the private copy and
    everything else falls through to the release.  `src/` is never written to
    -- not even transiently -- so the six generations run fully in parallel and
    the shared files stay byte-identical.  `toyscan_pt/` is a sibling of `src/`
    inside CMSSW_BASE and is removed by `cleanup`.

    WHY THE AREA MUST LIVE UNDER $CMSSW_BASE, AND NOT SOMEWHERE ARBITRARY.
    `edm::FileInPath::initialize_` (FWCore/Utilities/src/FileInPath.cc), having
    located a file under a search-path element, walks the element's parents
    looking for CMSSW_BASE / CMSSW_RELEASE_BASE / CMSSW_DATA_PATH:

        for (path br = pathPrefix.parent_path();
             !weakly_canonical(br).string().empty(); br = br.parent_path())

    `path("/").parent_path()` is `"/"` and `weakly_canonical("/")` is `"/"`,
    which is never empty -- so if NO ancestor matches, that loop never
    terminates.  An area under /tmp therefore hangs cmsRun in a tight
    `fstatat64` loop before the first line of output, and the eventual
    exception deadlocks in `pybind11::error_already_set::what()` on the GIL, so
    the failure is completely mute.  Diagnosed here the hard way; putting the
    area under CMSSW_BASE makes the walk terminate on its second step with
    `location_ = Local`.

    `gen_toy_config.py` locates its outputs as `dirname(__file__)` and
    `dirname(dirname(__file__))/data`, so a COPY placed in the private area
    writes to the private area with no path edits.

2.  pT is hard-coded in the plane generator (`pt, eta, phi0, B, q = 3.0, ...`).
    The scoring RADII are momentum-independent, but the helix INTERSECTION
    POINTS are not: at r = 106.8 cm the crossing azimuth differs by 0.189 rad
    between pT = 3 and pT = 40, i.e. ~20 cm of local x -- 130 s_F.  A pT = 40 sim
    scored against pT = 3 planes would not fail, it would just be wrong.

    Rather than editing the constant in place, the private copy of
    `gen_toy_config.py` is patched (mechanically, with an asserted replacement
    count) to read pT from `TOY_PT` and to name its output
    `toyPlanes_pt<PT>.py`; `runToyModel.py` is patched to import that module by
    name from `TOY_PLANES_MOD`.  The offline loader is handed the same file.
    `pairs` then checks the sim's own median crossing point against the planes
    the model used, which is the check that actually catches a momentum
    mismatch (a radius check does NOT -- the radii are identical by design).

SUBCOMMANDS
-----------
    setup     build one isolated area and generate its geometry+planes
    sim       run the Geant4 sim in an area
    model     run the Geant4e model export in an area
    pairs     model/sim correspondence, momentum matching, stepper provenance
    knobs     1/I convergence (grid/floor) -- the pre-check the Fisher scale needs
    closure   the closure table, over u, with per-plane profiles
    control   reproduce the published 100k pT=3 layered-toy numbers
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

import cf_ms_exact                                              # noqa: E402
import cgf_channels as cc                                       # noqa: E402
import fisher_norm as fn                                        # noqa: E402
import geom_closure as gc                                       # noqa: E402
from cf_propagation_test import (FUNCTIONALS, REF_BRANCH,       # noqa: E402
                                 SIM_BRANCH, load_model, model_variance)

CMSSW = "/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev"
SRCTEST = f"{CMSSW}/src/Analysis/HitAnalyzer/test"
SCRATCH = fn.SCRATCH
# must be under CMSSW_BASE -- see the FileInPath note in the module docstring
AREAS = os.path.join(CMSSW, "toyscan_pt")
OUT = os.path.join(SCRATCH, "tp")

UCURVE = fn.UCURVE
UHEAD = fn.UHEAD
IHEAD = fn.IHEAD

# the six configurations
T_LAYER = 0.10
CONFIGS = {}
for _pt in (3, 40):
    for _k in (1, 4, 16):
        CONFIGS[f"pt{_pt}_K{_k}"] = dict(pt=float(_pt), nsub=_k, t=T_LAYER)


def area(tag):
    return os.path.join(AREAS, tag)


def testdir(tag):
    return os.path.join(area(tag), "Analysis", "HitAnalyzer", "test")


def planes_name(pt):
    return f"toyPlanes_pt{int(pt) if float(pt).is_integer() else pt}.py"


def planes_path(tag):
    return os.path.join(testdir(tag), planes_name(CONFIGS[tag]["pt"]))


def sim_path(tag):
    return os.path.join(OUT, f"{tag}_sim.root")


def model_path(tag):
    return os.path.join(OUT, f"{tag}_model.root")


# ==========================================================================
# area construction
# ==========================================================================

def _sub1(text, pat, repl, what):
    """Exactly-one substitution, or die.  A silent no-op patch here would put a
    pT = 40 sim on pT = 3 planes, which is the failure mode this whole module
    exists to prevent."""
    new, n = re.subn(pat, repl, text)
    if n != 1:
        raise SystemExit(f"patch '{what}' matched {n} times, expected 1")
    return new


def cmd_setup(args):
    tag = args.tag
    cfg = CONFIGS[tag]
    a = area(tag)
    td = testdir(tag)
    dd = os.path.join(a, "Analysis", "HitAnalyzer", "data")
    if os.path.exists(a) and args.fresh:
        shutil.rmtree(a)
    os.makedirs(td, exist_ok=True)
    os.makedirs(dd, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)

    # --- gen_toy_config.py: pT from the environment, planes file named for it
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

    # --- runToyGeomCheck.py: verbatim.  gen_toy_config rewrites its watcher
    #     radii line in place, in the PRIVATE copy.
    shutil.copy(f"{SRCTEST}/runToyGeomCheck.py", f"{td}/runToyGeomCheck.py")

    # --- runToyModel.py: import the planes module named for this momentum
    m = open(f"{SRCTEST}/runToyModel.py").read()
    m = _sub1(m,
              r"import toyPlanes_pt3 as planes",
              "import importlib, os as _os\n"
              'planes = importlib.import_module(_os.environ["TOY_PLANES_MOD"])',
              "model: planes import")
    open(f"{td}/runToyModel.py", "w").write(m)

    # --- generate geometry + planes + watcher radii, TOGETHER, in the area
    env = dict(os.environ, TOY_PT=repr(cfg["pt"]),
               TOY_PLANES=planes_name(cfg["pt"]))
    r = subprocess.run([sys.executable, "gen_toy_config.py",
                        f"{cfg['t']:.2f}", str(cfg["nsub"])],
                       cwd=td, env=env, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"gen failed:\n{r.stdout}\n{r.stderr}")
    print(f"[{tag}] {r.stdout.strip()}")
    print(f"[{tag}] area {a}")
    for f in (f"{dd}/tracker.xml", planes_path(tag), f"{td}/runToyGeomCheck.py"):
        print(f"[{tag}]   {f}  {os.path.getsize(f)} B")


def _cmsrun(tag, script, extra, log):
    """cmsRun in the isolated area, with the area prepended to
    CMSSW_SEARCH_PATH so tracker.xml resolves to the private copy.

    THE ENVIRONMENT IS SANITIZED, and it has to be.  `calibration_studies/
    setup_env.sh` sources an LCG view and activates a venv, i.e. it exports
    PYTHONPATH / PYTHONHOME / VIRTUAL_ENV / LD_LIBRARY_PATH pointing at a
    DIFFERENT Python.  cmsRun embeds its own interpreter, picks those up, and
    dies -- and the failure is completely mute, because the resulting
    `pybind11::error_already_set` is converted to an EDM exception by
    `edm::convertException::stdToEDM`, whose `what()` blocks forever trying to
    re-acquire a GIL nobody holds.  Zero bytes of output, one thread, no CPU.
    Diagnosed here the hard way; hence the allowlist rather than
    `dict(os.environ)`.
    """
    td = testdir(tag)
    keep = ("HOME", "USER", "LOGNAME", "SHELL", "TERM", "HOSTNAME", "TMPDIR",
            "X509_USER_PROXY", "KRB5CCNAME")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env["PATH"] = "/usr/local/bin:/usr/bin:/bin"
    # The four 2026-08-16 default-on corrections pinned to their
    # HISTORICAL state (all off); any explicit overlay below still
    # wins.  Same convention as deltaspec._clean_env -- an archived
    # model must stay comparable to a fresh export.
    import cf_track_resolution as _ctr
    # The switches are ParameterSet parameters now (Geant4e b372e08), so the
    # historical pin has to travel as cmsRun OPTIONS; putting it in `env` would
    # export names nothing reads and silently give the archive the new defaults.
    _swopts, _ = _ctr.split_switches(dict(_ctr.SWITCHES_OFF))
    extra = f"{extra} {_swopts}"
    env["TOY_PLANES_MOD"] = planes_name(CONFIGS[tag]["pt"])[:-3]
    cmd = (f"source /cvmfs/cms.cern.ch/cmsset_default.sh >/dev/null 2>&1 && "
           f"cd {CMSSW}/src && eval $(scramv1 runtime -sh) && "
           f"export CMSSW_SEARCH_PATH={area(tag)}:$CMSSW_SEARCH_PATH && "
           f"cd {td} && exec cmsRun {script} {extra}")
    os.makedirs(OUT, exist_ok=True)
    with open(log, "w") as fh:
        p = subprocess.run(["bash", "-c", cmd], stdout=fh,
                           stderr=subprocess.STDOUT, env=env)
    return p.returncode


def cmd_sim(args):
    tag = args.tag
    cfg = CONFIGS[tag]
    log = os.path.join(OUT, f"{tag}_sim.log")
    rc = _cmsrun(tag, "runToyGeomCheck.py",
                 f"events={args.events} pt={cfg['pt']} eta=0.30 "
                 f"output={sim_path(tag)}", log)
    txt = open(log, errors="ignore").read(6000)
    st = ("TIGHT" if "[toy] TIGHT stepper" in txt else
          ("LOOSE" if "LOOSE stepper" in txt else "UNKNOWN"))
    print(f"[{tag}] sim rc={rc} stepper={st} -> {sim_path(tag)}")
    if rc or st != "TIGHT":
        raise SystemExit(f"[{tag}] SIM FAILED (rc={rc}, stepper={st}), see {log}")


def cmd_model(args):
    tag = args.tag
    cfg = CONFIGS[tag]
    log = os.path.join(OUT, f"{tag}_model.log")
    rc = _cmsrun(tag, "runToyModel.py",
                 f"pt={cfg['pt']} eta=0.30 phi=0.70 "
                 f"output={model_path(tag)}", log)
    print(f"[{tag}] model rc={rc} -> {model_path(tag)}")
    if rc:
        raise SystemExit(f"[{tag}] MODEL FAILED, see {log}")


# ==========================================================================
# loading
# ==========================================================================

_M, _S = {}, {}


def legs_of(tag):
    if tag not in _M:
        _M[tag] = load_model(model_path(tag))
    return _M[tag]


def sim_of(tag):
    """The sim, expressed in THIS configuration's own plane frames.

    `fn.load_sim` defaults to the shared pt=3 planes file AND caches on the sim
    path alone, so it is bypassed here: the planes are the thing that has to
    vary with momentum, and a cache that ignores them is exactly the silent
    failure this module is built to avoid.
    """
    if tag not in _S:
        from toy_loader import load_toy_sim
        ns = _planes_ns(tag)
        _S[tag] = load_toy_sim(sim_path(tag), ns["origin"], ns["normal"],
                               ns["uaxis"])
    return _S[tag]


def _planes_ns(tag):
    ns = {}
    exec(open(planes_path(tag)).read(), ns)
    return ns


_R = {}


def refglobr(tag):
    """`load_model` drops refglobr from the per-leg dict, so read it here."""
    if tag not in _R:
        f = uproot.open(model_path(tag))
        tk = next(k for k in f.keys() if k.split(";")[0].endswith("/legs"))
        _R[tag] = np.asarray(f[tk].arrays(["refglobr"],
                                          library="np")["refglobr"])
    return _R[tag]


# ==========================================================================
# subcommand: pairs
# ==========================================================================

def _median_locx_in(tag_sim, tag_planes):
    """max |median locx| / s_F of tag_sim's rays scored in tag_planes' frames."""
    from toy_loader import load_toy_sim
    ns = _planes_ns(tag_planes)
    s = load_toy_sim(sim_path(tag_sim), ns["origin"], ns["normal"],
                     ns["uaxis"])
    v = s["valid"]
    off = np.array([np.nanmedian(s["locx"][v[:, k], k])
                    for k in range(v.shape[1])])
    sc = fn.plane_scales(legs_of(tag_sim), "locx", tag=tag_sim)
    return float(np.abs(off).max()), float(np.abs(off / sc["sF"]).max())


def cmd_pairs(args):
    print("=" * 78)
    print("MODEL / SIM PAIR VALIDATION  (a mis-targeted pair is SILENT)")
    print("=" * 78)
    print("(1) same number of scoring surfaces\n"
          "(2) identical plane-index sequence\n"
          "(3) model reference sits at the sim's radius on every plane\n"
          "(4) reference energy loss compatible with the sim's (model quotes a\n"
          "    MEAN, sim a MEDIAN, so they differ by the Landau mean-mode gap)\n"
          "(5) MOMENTUM MATCHING -- the sim's median crossing POINT against the\n"
          "    planes the model was given.  Radii are momentum-independent by\n"
          "    construction, so (3) can NOT catch a pT mismatch; this can.\n"
          "    NOTE this is a TWO-HYPOTHESIS test, not an absolute bound: the\n"
          "    correctly paired configuration does NOT give zero.  The sim's\n"
          "    MEDIAN ray loses less energy than the model's MEAN reference (the\n"
          "    Landau mean-mode gap of test (4), 3.7 MeV out of 3.14 GeV), so it\n"
          "    is stiffer, bends less, and arrives at a slightly different\n"
          "    azimuth -- a real, physical, radius-growing offset of a few tenths\n"
          "    of s_F.  What separates a matched from a mismatched pair is the\n"
          "    RATIO to the same sim scored in the other momentum's planes.\n"
          "(6) the model's own reference pT\n")
    ok_all = True
    for tag in args.configs:
        cfg = CONFIGS[tag]
        legs = legs_of(tag)
        sim = sim_of(tag)
        ns = _planes_ns(tag)
        npl = len(ns["radii"])
        f = uproot.open(model_path(tag))
        tk = next(k for k in f.keys() if k.split(";")[0].endswith("/legs"))
        a = f[tk].arrays(["detid", "refglobr", "refp", "refpt", "reflocx"],
                         library="np")
        v = sim["valid"]
        nfull = int(v.all(axis=1).sum())
        rmed = np.array([np.nanmedian(sim["globr"][v[:, k], k])
                         for k in range(npl)])
        dE_m = 1e3 * float(a["refp"][0] - a["refp"][-1])
        p0 = np.array([sim["pabs"][v[:, 0], 0]]).ravel()
        pl = np.array([sim["pabs"][v[:, npl - 1], npl - 1]]).ravel()
        dE_s = 1e3 * float(np.median(p0) - np.median(pl))

        n_ok = len(a["detid"]) == npl == v.shape[1]
        seq_ok = n_ok and np.array_equal(np.asarray(a["detid"]),
                                         np.arange(npl))
        dr = np.abs(np.asarray(a["refglobr"]) - rmed).max()
        # (5): the sim's median local position in the model's plane frame.
        sc = fn.plane_scales(legs, "locx", tag=tag)
        offx = np.array([np.nanmedian(sim["locx"][v[:, k], k])
                         for k in range(npl)])
        offy = np.array([np.nanmedian(sim["locy"][v[:, k], k])
                         for k in range(npl)])
        offx_sf = np.abs(offx / sc["sF"][:npl])
        pt_ok = abs(float(a["refpt"][0]) - cfg["pt"]) < 1e-3 * cfg["pt"]
        # the same rays scored in the OTHER momentum's planes
        alt = next((t for t in CONFIGS
                    if CONFIGS[t]["pt"] != cfg["pt"]
                    and CONFIGS[t]["nsub"] == cfg["nsub"]
                    and os.path.exists(planes_path(t))), None)
        alt_cm, alt_sf = (_median_locx_in(tag, alt) if alt else (np.nan,
                                                                 np.nan))
        mom_ok = bool(offx_sf.max() < 5.0
                      and (np.isnan(alt_sf) or alt_sf > 20 * offx_sf.max()))

        print(f"--- {tag}   pT={cfg['pt']:g}  T={cfg['t']}  NSUB={cfg['nsub']}")
        print(f"    (1) legs {len(a['detid'])}  sim planes {v.shape[1]}  "
              f"planes file {npl}        {'OK' if n_ok else 'MISMATCH'}")
        print(f"    (2) plane-index sequence identical: {seq_ok}")
        print(f"    (3) max |refglobr - sim median r| = {dr:.5f} cm")
        print(f"    (4) dE model(mean) {dE_m:8.3f} MeV   "
              f"sim(median) {dE_s:8.3f} MeV   gap {dE_m - dE_s:+7.3f} MeV")
        print(f"    (5) OWN planes  max |median locx| = "
              f"{np.abs(offx).max():.5f} cm = {offx_sf.max():.3f} s_F ;  "
              f"max |median locy| = {np.abs(offy).max():.5f} cm")
        print(f"        {alt} planes  max |median locx| = {alt_cm:.5f} cm "
              f"= {alt_sf:.1f} s_F   -> ratio {alt_sf/max(offx_sf.max(),1e-12):.0f}x"
              f"    {'OK' if mom_ok else '<-- MOMENTUM MISMATCH'}")
        print(f"        per-plane median locx [s_F]: "
              + " ".join(f"{v:+.3f}" for v in (offx / sc['sF'][:npl])))
        print(f"    (6) model refpt[0] = {float(a['refpt'][0]):.4f} GeV, "
              f"requested {cfg['pt']:g}   {'OK' if pt_ok else 'MISMATCH'}")
        print(f"        sim events {sim['ntot']}, complete-sequence {nfull} "
              f"({100.*nfull/sim['ntot']:.3f} %)")
        # stepper
        lg = os.path.join(OUT, f"{tag}_sim.log")
        st = "MISSING"
        if os.path.exists(lg):
            txt = open(lg, errors="ignore").read(6000)
            st = ("TIGHT" if "[toy] TIGHT stepper" in txt else
                  ("LOOSE" if "LOOSE stepper" in txt else "UNKNOWN"))
        print(f"    (7) stepper from {os.path.basename(lg)}: {st}")
        # planes provenance
        hdr = open(planes_path(tag)).readline().strip()
        print(f"    (8) planes {os.path.basename(planes_path(tag))}  '{hdr}'")
        good = (n_ok and seq_ok and dr < 1e-3 and mom_ok and pt_ok
                and st == "TIGHT")
        ok_all &= good
        print(f"    VERDICT: {'PASS' if good else 'FAIL'}\n")
    print(f"ALL CONFIGURATIONS: {'PASS' if ok_all else 'SOME FAILED'}")

    # cross-momentum control: show what a mismatched pairing WOULD look like
    if args.crosscheck and len(args.configs) >= 2:
        print("\nCROSS-MOMENTUM CONTROL (deliberately wrong pairing, to show the\n"
              "check has teeth -- these numbers are NOT used for anything)")
        a3 = [t for t in args.configs if CONFIGS[t]["pt"] == 3.0]
        a4 = [t for t in args.configs if CONFIGS[t]["pt"] == 40.0]
        if a3 and a4:
            t3, t4 = a3[0], a4[0]
            ns3 = _planes_ns(t3)
            from toy_loader import load_toy_sim
            s = load_toy_sim(sim_path(t4), ns3["origin"], ns3["normal"],
                             ns3["uaxis"])
            v = s["valid"]
            off = np.array([np.nanmedian(s["locx"][v[:, k], k])
                            for k in range(v.shape[1])])
            sc = fn.plane_scales(legs_of(t4), "locx", tag=t4)
            print(f"  {t4} sim on {t3} planes: max |median locx| = "
                  f"{np.abs(off).max():.3f} cm = "
                  f"{np.abs(off / sc['sF']).max():.0f} s_F"
                  "   (correct pairing gives < 0.5 s_F)")


# ==========================================================================
# subcommand: knobs -- 1/I convergence
# ==========================================================================

def cmd_knobs(args):
    """1/I stability against the inversion grid.

    NOTES_FISHERNORM validated this at pT = 3 only.  At pT = 40 the radiative
    channel is a large share of the variance and its density is far more
    skewed, so the grid/floor convergence has to be re-established rather than
    inherited.  RELATIVE floor throughout (the absolute-floor trap of
    NOTES (XV)), and the grid is matched per block by `cgf_channels`.
    """
    print("=" * 78)
    print("1/I CONVERGENCE (relative floor, per-block matched grid)")
    print("=" * 78)
    for tag in args.configs:
        legs = legs_of(tag)
        print(f"--- {tag}")
        for func in args.funcs:
            avec = FUNCTIONALS[func]
            print(f"  {func}")
            print(f"    {'k':>3} {'floor1e-6':>11} {'1e-8':>11} {'1e-10':>11} "
                  f"{'nt x1/4':>11} {'npad x4':>11} {'lncut-120':>11} "
                  f"{'mass':>9}")
            for k in args.planes:
                if k >= len(legs):
                    continue
                sig = float(np.sqrt(model_variance(legs, k, avec)[0]))
                row, mass = [], np.nan
                for kw in (dict(floor=1e-6), dict(floor=1e-8),
                           dict(floor=1e-10), dict(nt=1 << 15),
                           dict(npad=128), dict(lncut=-120.0)):
                    fl = kw.pop("floor", 1e-8)
                    z, p, dp = cc.exact_density(
                        legs, k, avec, sig,
                        channels=("ioni", "ms", "rad"),
                        nt=kw.get("nt", 1 << 17), npad=kw.get("npad", 32),
                        lncut=kw.get("lncut", -60.0))
                    I, info = cc.fisher_exact(z, p, dp, floor=fl)
                    row.append(1.0 / I)
                    mass = info.get("mass_frac", np.nan)
                print(f"    {k:3d} " + "".join(f"{v:11.5f}" for v in row)
                      + f"{mass:9.5f}")
        print()


# ==========================================================================
# subcommand: closure
# ==========================================================================

def _rows(tag, func, norm="Fisher"):
    legs, sim = legs_of(tag), sim_of(tag)
    sc = fn.plane_scales(legs, func, tag=tag)
    scale = sc["sF"] if norm == "Fisher" else sc["sigma"]
    rows, errs, ks, ns, err = gc.closure_rows(legs, sim, func, scale)
    return dict(rows=rows, errs=errs, ks=ks, ns=ns, err=err, scales=sc,
                legs=legs, scale=scale)


def cmd_closure(args):
    print("=" * 78)
    print("LAYERED TOY, CLEAN-PROPAGATION CLOSURE, FISHER NORMALIZATION")
    print("=" * 78)
    print("s_F = sigma sqrt(1/I), 1/I by EXACT FFT inversion of the model CF\n"
          "(never the saddlepoint).  The error is the CORRELATED per-event\n"
          "estimator of NOTES_GEOMCLOSURE section 1 -- every plane is evaluated\n"
          "on the same events, so err_plane/sqrt(nplanes) is wrong by ~2.8x.\n")

    res = {}
    for tag in args.configs:
        for func in args.funcs:
            res[(tag, func)] = _rows(tag, func, norm=args.norm)

    # step structure, for the record
    print("### model step structure (per-step d/X0, msmoliv column 6)")
    print(f"  {'config':<12s} {'nms':>7} {'nioni':>7} {'ms/leg':>8} "
          f"{'ioni/leg':>9} {'sum d/X0':>10} {'<d/X0>_w':>11} {'dE[MeV]':>9}")
    for tag in args.configs:
        legs = legs_of(tag)
        f = uproot.open(model_path(tag))
        tk = next(k for k in f.keys() if k.split(";")[0].endswith("/legs"))
        a = f[tk].arrays(["msmoliv"], library="np")
        st = np.vstack([np.asarray(v, dtype=float).reshape(
            -1, 10 if np.asarray(v).size % 10 == 0 else 8)
            for v in a["msmoliv"]])
        d = st[:, 6]
        nms = sum(len(l["ms"]) for l in legs)
        nio = sum(len(l["ioni"]) for l in legs)
        dE = 1e3 * (1.0 / abs(legs[0]["refqop"]) - 1.0 / abs(legs[-1]["refqop"]))
        print(f"  {tag:<12s} {nms:7d} {nio:7d} {nms/len(legs):8.1f} "
              f"{nio/len(legs):9.1f} {d.sum():10.4f} "
              f"{(d**2).sum()/d.sum():11.3e} {dE:9.2f}")
    print()

    for func in args.funcs:
        print(f"### {func}   (mean over planes; u in units of s_F)")
        print(f"  {'config':<20s} " + "".join(f"{u:>10.3g}" for u in UCURVE))
        for tag in args.configs:
            r = res[(tag, func)]
            m = r["rows"].mean(axis=0)
            print(f"  {tag:<20s} " + "".join(f"{v:+10.5f}" for v in m)
                  + f"   [{len(r['ks'])} planes, "
                    f"{int(np.median(r['ns']))} ev/plane]")
            print(f"  {'   +- (stat, corr)':<20s} "
                  + "".join(f"{v:10.5f}" for v in r["err"]))
            print(f"  {'   rms over planes':<20s} "
                  + "".join(f"{v:10.5f}" for v in r["rows"].std(axis=0)))
        print()

    print("### headline probes")
    print(f"  {'config':<14s} " + "".join(f"{f + ' u=' + str(u):>20s}"
                                          for f in args.funcs for u in UHEAD))
    for tag in args.configs:
        cells = []
        for f in args.funcs:
            r = res[(tag, f)]
            m, e = r["rows"].mean(axis=0)[IHEAD], r["err"][IHEAD]
            cells += [f"{mi:+.5f}+-{ei:.5f}" for mi, ei in zip(m, e)]
        print(f"  {tag:<14s} " + "".join(f"{c:>20s}" for c in cells))
    print()

    # significance of the closure and of the NSUB trend
    print("### significance")
    for func in args.funcs:
        for tag in args.configs:
            r = res[(tag, func)]
            m, e = r["rows"].mean(axis=0)[IHEAD], r["err"][IHEAD]
            print(f"  {func:<5s} {tag:<14s} " +
                  "  ".join(f"u={u}: {mi/ei:+6.2f} sig"
                            for u, mi, ei in zip(UHEAD, m, e)))
    print()
    print("  NSUB trend (K16 - K1), same events per configuration but "
          "INDEPENDENT sims,\n  so the errors add in quadrature:")
    for func in args.funcs:
        for pt in sorted({CONFIGS[t]["pt"] for t in args.configs}):
            a1 = f"pt{int(pt)}_K1"
            a16 = f"pt{int(pt)}_K16"
            if a1 not in args.configs or a16 not in args.configs:
                continue
            r1, r16 = res[(a1, func)], res[(a16, func)]
            for i, u in zip(IHEAD, UHEAD):
                d = r16["rows"].mean(axis=0)[i] - r1["rows"].mean(axis=0)[i]
                e = np.hypot(r16["err"][i], r1["err"][i])
                print(f"    {func:<5s} pT={pt:g}  u={u:<5g} "
                      f"K16-K1 = {d:+.5f} +- {e:.5f}  ({d/e:+.2f} sigma)")
    print()

    if args.perplane:
        for func in args.funcs:
            print(f"### {func}: per-plane profile (radius-ordered)")
            for tag in args.configs:
                r = res[(tag, func)]
                rg = refglobr(tag)
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
                                                r["err"][IHEAD]))
                      + "   (correlated error on the plane mean)")
                print(f"    {'rms':>3} {'':>8} {'':>9} {'':>12} "
                      + "".join(f"{v:+19.5f}"
                                for v in r["rows"].std(axis=0)[IHEAD])
                      + "   (plane-to-plane rms)")
            print()

    if args.channels:
        # kappa2 of the model's OWN CF per channel (block_cgf_derivs at
        # theta = 0), not the Q-matrix variance: Q carries no radiative entry
        # at all, and the alpha-truncated ionization entry is a convention.
        # THIS is the number that says how much of the block the radiative
        # channel carries -- the pT = 40 question.
        print("### kappa2 share by channel, from the model CF "
              "(block_cgf_derivs at theta=0)")
        print(f"  {'config':<14s} {'func':<5s} {'ioni':>10} {'ms':>10} "
              f"{'rad':>10}   {'rad/ioni':>9}   (plane mean of the share)")
        for tag in args.configs:
            legs = legs_of(tag)
            for func in args.funcs:
                avec = FUNCTIONALS[func]
                sh = {c: [] for c in ("ioni", "ms", "rad")}
                for k in range(len(legs)):
                    sig = float(np.sqrt(model_variance(legs, k, avec)[0]))
                    for c in sh:
                        blk = cc.collect_block(legs, k, avec, sig,
                                               channels=(c,))
                        sh[c].append(float(np.atleast_1d(
                            cc.block_cgf_derivs(blk, 0.0, order=2)[2])[0]))
                v = {c: np.array(sh[c]) for c in sh}
                tot = v["ioni"] + v["ms"] + v["rad"]
                print(f"  {tag:<14s} {func:<5s} "
                      + "".join(f"{np.mean(v[c] / tot):10.5f}"
                                for c in ("ioni", "ms", "rad"))
                      + f"   {np.mean(v['rad'] / np.maximum(v['ioni'], 1e-300)):9.4f}")
        print()

    if args.dump:
        o = {}
        for (tag, func), r in res.items():
            o[f"{tag}|{func}|rows"] = r["rows"]
            o[f"{tag}|{func}|errs"] = r["errs"]
            o[f"{tag}|{func}|err"] = r["err"]
            o[f"{tag}|{func}|ks"] = r["ks"]
            o[f"{tag}|{func}|invI"] = r["scales"]["invI"]
            o[f"{tag}|{func}|sF"] = r["scales"]["sF"]
            o[f"{tag}|{func}|sigma"] = r["scales"]["sigma"]
        np.savez(args.dump, u=UCURVE, **o)
        print(f"wrote {args.dump}")


def cmd_diag(args):
    """The one toy-only approximation that could fake a momentum dependence.

    The watcher scores at the CYLINDER crossing while the model propagates to a
    TANGENT PLANE, so the sim state sits a normal offset `locz` off the plane
    and its `locx` is wrong by `locz * dx/dz`.  NOTES_GEOMCLOSURE section 6
    bounded this at 0.2 % of s_F -- AT pT = 3.  At pT = 40 the trajectory is
    straighter (smaller `locz`) but `s_F` is ~13x smaller too, so the ratio has
    to be re-measured rather than inherited: if it grew it would masquerade as
    a momentum-dependent `locx` residual.
    """
    print("=" * 78)
    print("TOY TANGENT-PLANE APPROXIMATION, in units of s_F")
    print("=" * 78)
    print("  QUOTED AS sigma68, NOT rms.  At 400k events the rms of `locz` is\n"
          "  dominated by a ~1e-5 tail of rays that curl and cross the outer\n"
          "  radius at a completely different azimuth (max |locz| = 43 cm on one\n"
          "  ray in pt3_K1) -- a real physical tail, not a defect, and one the\n"
          "  BOUNDED closure functional e^{-u z^2} cannot feel.  The rms is shown\n"
          "  alongside so the tail is visible rather than hidden.\n")
    print(f"  {'config':<12s} {'k':>3} {'s68 locz [um]':>14} "
          f"{'s68 locz dx/dz':>15} {'s_F [um]':>11} {'s68 ratio':>10} "
          f"{'rms ratio':>10} {'max|locz| [um]':>15}")

    def s68(x):
        return 0.5 * float(np.percentile(x, 84.135) - np.percentile(x, 15.865))

    for tag in args.configs:
        sim, legs = sim_of(tag), legs_of(tag)
        sc = fn.plane_scales(legs, "locx", tag=tag)
        for k in (0, 3, len(legs) - 1):
            m = sim["valid"][:, k] & np.isfinite(sim["dxdz"][:, k])
            lz = sim["locz"][m, k]
            c = lz * sim["dxdz"][m, k]
            print(f"  {tag:<12s} {k:3d} {1e4 * s68(lz):14.3f} "
                  f"{1e4 * s68(c):15.4f} {1e4 * sc['sF'][k]:11.2f} "
                  f"{s68(c) / sc['sF'][k]:10.6f} "
                  f"{np.std(c) / sc['sF'][k]:10.5f} "
                  f"{1e4 * np.abs(lz).max():15.1f}")
    print()


def cmd_guard(args):
    """Does the `cf_ms_exact` j0(x)-1 guard matter at pT = 40?

    NOTES_FISHERNORM section 6 measured the guard to be a no-op at pT = 3 (it
    changes the closure by < 1e-5) and then explicitly refused to generalize:
    a step lands in the corrupted small-argument branch when
    `sqrt(chi_a^2) * w * t < 1e-8`, and `chi_a` FALLS with momentum
    (~3e-6 at pT = 3, ~1e-7 quoted at pT = 100), so at higher pT a step with an
    ordinary transport weight can reach it.  That has to be measured here, not
    inherited.
    """
    from cf_ms_exact import moliere_params
    print("=" * 78)
    print("j0(x)-1 GUARD: is the small-argument MS branch reachable at this pT?")
    print("=" * 78)
    print(f"  {'config':<14s} {'min sqrt(chi_a^2)':>18} {'median':>12} "
          f"{'max':>12}   {'w needed for arg<1e-8 at t=1':>30}")
    for tag in args.configs:
        legs = legs_of(tag)
        ca = []
        for leg in legs:
            for s in leg["ms"]:
                p = (moliere_params(*s[:5], s[7], s[8]) if leg["ms"].shape[1] >= 10
                     else moliere_params(*s[:5]))
                ca.append(np.sqrt(p[1]))
        ca = np.array(ca)
        print(f"  {tag:<14s} {ca.min():18.3e} {np.median(ca):12.3e} "
              f"{ca.max():12.3e}   {1e-8 / np.median(ca):30.3e}")
    print()
    print("### closure with the guard ON (production default) and OFF")
    print(f"  {'config':<14s} {'func':<5s} {'guard':<6s} " +
          "".join(f"{'u=' + str(u):>12s}" for u in UHEAD))
    for tag in args.configs:
        for func in args.funcs:
            out = {}
            for g in (True, False):
                cf_ms_exact.set_j0_guard(g)
                fn._SCALE_CACHE.clear()
                r = _rows(tag, func)
                out[g] = r["rows"].mean(axis=0)[IHEAD]
            cf_ms_exact.set_j0_guard(True)
            fn._SCALE_CACHE.clear()
            for g in (True, False):
                print(f"  {tag:<14s} {func:<5s} {str(g):<6s} " +
                      "".join(f"{v:+12.5f}" for v in out[g]))
            print(f"  {'':<14s} {'':<5s} {'diff':<6s} " +
                  "".join(f"{a - b:+12.2e}" for a, b in zip(out[True],
                                                            out[False])))
    print()


def cmd_control(args):
    """Reproduce the published 100k pT=3 layered-toy numbers with THIS code
    path before any new number is believed."""
    print("=" * 78)
    print("CONTROL: the 100k pT=3 layered toy (NOTES_GEOMCLOSURE section 2/3)")
    print("=" * 78)
    print("targets, Fisher, u = 1, mean over planes:")
    print("   NSUB=1   locx +0.00002 +- 0.00083   qop +0.00081 +- 0.00034")
    print("   NSUB=4   locx +0.00055 +- 0.00083   qop +0.00131 +- 0.00034")
    print("   NSUB=16  locx +0.00111 +- 0.00083   qop +0.00157 +- 0.00034\n")
    print(f"  {'geom':<10s} {'func':<5s} " +
          "".join(f"{'u=' + str(u):>20s}" for u in UHEAD))
    for g in ("H_layK1", "H_layK4", "H_layK16"):
        for func in ("locx", "qop"):
            r = gc.geom_closure(g, func, norm="Fisher")
            m, e = r["rows"].mean(axis=0)[IHEAD], r["err"][IHEAD]
            print(f"  {g:<10s} {func:<5s} " +
                  "".join(f"{mi:+.5f}+-{ei:.5f}".rjust(20)
                          for mi, ei in zip(m, e)))


def cmd_cleanup(args):
    """Remove the private areas, leaving CMSSW as it was found.

    `src/` was never written to, so this is the ONLY thing to undo.  The md5s
    of the two shared files are printed so the claim is checkable rather than
    asserted.
    """
    import hashlib
    for f in (f"{CMSSW}/src/Analysis/HitAnalyzer/data/tracker.xml",
              f"{SRCTEST}/toyPlanes_pt3.py",
              f"{SRCTEST}/runToyGeomCheck.py",
              f"{SRCTEST}/gen_toy_config.py",
              f"{SRCTEST}/runToyModel.py"):
        h = hashlib.md5(open(f, "rb").read()).hexdigest()
        print(f"  {h}  {f}")
    if os.path.isdir(AREAS):
        shutil.rmtree(AREAS)
        print(f"removed {AREAS}")
    else:
        print(f"{AREAS} already absent")


# ==========================================================================

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)

    q = s.add_parser("setup")
    q.add_argument("tag", choices=list(CONFIGS))
    q.add_argument("--fresh", action="store_true")
    q.set_defaults(fn=cmd_setup)

    q = s.add_parser("sim")
    q.add_argument("tag", choices=list(CONFIGS))
    q.add_argument("--events", type=int, default=400000)
    q.set_defaults(fn=cmd_sim)

    q = s.add_parser("model")
    q.add_argument("tag", choices=list(CONFIGS))
    q.set_defaults(fn=cmd_model)

    q = s.add_parser("pairs")
    q.add_argument("--configs", nargs="+", default=list(CONFIGS))
    q.add_argument("--crosscheck", action="store_true")
    q.set_defaults(fn=cmd_pairs)

    q = s.add_parser("knobs")
    q.add_argument("--configs", nargs="+", default=["pt3_K1", "pt40_K1"])
    q.add_argument("--funcs", nargs="+", default=["qop", "locx"])
    q.add_argument("--planes", nargs="+", type=int, default=[0, 6, 13])
    q.set_defaults(fn=cmd_knobs)

    q = s.add_parser("closure")
    q.add_argument("--configs", nargs="+", default=list(CONFIGS))
    q.add_argument("--funcs", nargs="+", default=["qop", "locx"])
    q.add_argument("--norm", default="Fisher", choices=("Fisher", "sigma"))
    q.add_argument("--perplane", action="store_true")
    q.add_argument("--channels", action="store_true")
    q.add_argument("--dump", default="")
    q.set_defaults(fn=cmd_closure)

    q = s.add_parser("diag")
    q.add_argument("--configs", nargs="+",
                   default=["pt3_K1", "pt3_K16", "pt40_K1", "pt40_K16"])
    q.set_defaults(fn=cmd_diag)

    q = s.add_parser("guard")
    q.add_argument("--configs", nargs="+", default=["pt3_K1", "pt40_K1"])
    q.add_argument("--funcs", nargs="+", default=["qop", "locx"])
    q.set_defaults(fn=cmd_guard)

    q = s.add_parser("control")
    q.set_defaults(fn=cmd_control)

    q = s.add_parser("cleanup")
    q.set_defaults(fn=cmd_cleanup)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
