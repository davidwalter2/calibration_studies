#!/usr/bin/env python3
"""The clean-propagation closure on the REAL-MATERIAL toy, per species, in the
five local directions -- the realmat analogue of the layered-toy table in
~/public_html/slides/260819_cleanprop_walter.pdf (p. 17).

WHAT IS MEASURED
----------------
closure(u) = <e^{-u z^2}>_sim - <e^{-u z^2}>_model,  z = (sim - reference)/s_F,
s_F = sigma sqrt(1/I) (Fisher, exact FFT inversion), per plane, in the LOCAL
basis H (`hbasis`), for q/p, local x, local y, dx/dz, dy/dz.  The table quotes
u = 1 on the OUTERMOST plane (the whole-track number) with its per-plane
statistical error; the ladder mean and the full u curve are kept in the JSON.

THE GEOMETRY
------------
`realmat_full` (gen_toy_realmat.py --final-plane): the real radial material
sequence of the pT = 3, eta = 0.30, phi = 0.70 reference, as coaxial cylinders,
scored on the 18 sensor mid-planes plus the traversal's outer edge (19 planes).
Negative species use the planes of the q = -1 reference helix that the
geometry was built from; positive species cross the same cylinders at the
mirrored azimuth, so they get their own plane set from the SAME generator
(`gen_toy_realmat.helix_frames`, q = +1).  The q = -1 regeneration is checked
byte for byte against the published plane file before the mirror is trusted.
The cylinders are azimuthally symmetric, so the mirror sees identical material.

THE CONFIGURATION (matches the layered-toy table except where stated)
---------------------------------------------------------------------
  sim    tight stepper, DefaultCutValue = 1e-4 cm (`cut1e4`), clean arm:
         nuclear (inelastic + elastic) and Decay inactivated for the primary,
         verified by the step census (EXACTLY zero primary steps);
         10 seeds x 20 000 events
  model  all four ionisation corrections ON, passed explicitly
         (IoniExactDelta, IoniKokoulin, ReferenceChargeAware,
         ReferenceSpeciesDedx); every other switch at the cfi default
  offline radiation ON, Kokoulin per species (muon only), the six default-ON MS
         harmonisations (MS_ELEC_TMAX, MS_ELEC_EDGE, MS_SNAP_YMAX=0, MS_FINE_G,
         MS_CHI0_G4, MS_FF_G4).  MS_WVI_SPLIT stays OFF: its J0 series is
         valid only to q^2 U/4 = 60 and the real material sequence (long air
         gaps) sits orders of magnitude above that, where the guard fires
         (cf_track_resolution, "DEFAULT OFF").  On the layered toy it was ON;
         there it moved every MS direction by +0.00024 against MS_FINE_G's
         -0.00023, i.e. OFF here shifts the scattering directions by about
         -0.0002 relative to that table.

Everything lands on ceph (OUT), not in a session scratchpad: the previous
realmat samples lived in /tmp and are gone.

ARMS (`--arm`, all subcommands but setup)
    off      nuclear (inelastic + elastic) and Decay off for the primary
    elonly   hadElastic ON, inelastic and Decay off -- pair it with
             `closure --nucel`, which turns the nuclear-elastic CF channel
             (cf_nucel_exact, recoil included) on in the prediction, and with
             `--model-tag _mat`: models exported with the per-step material
             table, so the channel has per-ELEMENT targets (hydrogen in the
             composites); without it the channel rounds each step to one
             element and the far tails fail

SUBCOMMANDS
    setup    private areas (qm, qp): geometry, planes, patched drivers
    sim      seeded-split simulations
    live     switch liveness + acceptance + stepper/cut provenance
    export   the models
    pairs    model/sim pair validation (run BEFORE reading any closure)
    closure  the five-direction closure per species -> JSON
    table    the markdown table from the JSON
"""

import argparse
import glob
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "cleanprop"))

# the phi cache key omits IONI_KOKOULIN -- disabled for the whole process
os.environ["RES_NO_PHI_CACHE"] = "1"

import cf_ms_exact as mx                                         # noqa: E402
import cf_nucel_exact as cne                                     # noqa: E402
import cf_propagation_test as cpt                                # noqa: E402
import cf_track_resolution as ctr                                # noqa: E402
import deltaspec as ds                                           # noqa: E402
import fisher_norm as fn                                         # noqa: E402
import geom_closure as gc                                        # noqa: E402
import hadron_probe as hp                                        # noqa: E402
import hbasis as hb                                              # noqa: E402

CMSSW = "/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2"
SRC = f"{CMSSW}/src"
SRCTEST = f"{SRC}/Analysis/HitAnalyzer/test"
SRCDATA = f"{SRC}/Analysis/HitAnalyzer/data"
AREA = os.path.join(CMSSW, "rmfprobe")          # must live under CMSSW_BASE
OUT = "/ceph/submit/data/user/d/david_w/ZMass/cvh/cleanprop/realmat_full_260925"
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "runs", "realmat_closure_260925")

PT, ETA, PHI, BFIELD = 3.0, 0.30, 0.70, 3.8
CUT = 1e-4
SEEDS = list(range(101, 111))
NEV = 20000
ARM = "off"          # Geant4 process set of the sim; `--arm` sets it
MODEL_TAG = ""       # model-file suffix; `--model-tag` sets it (`_mat` = the
                     # export carrying the per-step G4 material table)
# The nuclear-elastic channel's Geant4 sampler (build_nucel.sh <dir>) and its
# kernel cache, both off the session scratchpad.
os.environ.setdefault("NUCEL_DRIVER", os.path.join(AREA, "bin", "nucel_g4driver.sh"))
os.environ.setdefault("NUCEL_CACHEDIR", os.path.join(OUT, "nucel"))

# geometry key -> (geometry directory, plane module, charge of the reference)
GEOM = {"qm": ("realmat_full", "toyPlanes_realmat_full_pt3", -1.0),
        "qp": ("realmat_full_qp", "toyPlanes_realmat_full_qp_pt3", +1.0)}

FUNCS = ("qop", "locx", "locy", "dxdz", "dydz")
ORDER8 = [13, -13, -211, 211, -321, 321, -2212, 2212]
FOUR_ON = dict(ctr.SWITCHES_ON)


def gkey(pdg):
    return "qm" if hp.SPECIES[pdg]["q"] < 0 else "qp"


def geomdir(g):
    return os.path.join(AREA, g)


def testdir(g):
    return os.path.join(geomdir(g), "Analysis", "HitAnalyzer", "test")


def toygeom(g):
    return f"Analysis/HitAnalyzer/data/{GEOM[g][0]}/tracker.xml"


def planes_path(g):
    return os.path.join(testdir(g), GEOM[g][1] + ".py")


def lab(pdg):
    return hp.SPECIES[pdg]["label"].replace("-", "m").replace("+", "p")


def sim_path(pdg, seed):
    return os.path.join(OUT, "sim", f"{lab(pdg)}_pt3_{ARM}_s{seed}_sim.root")


def sim_glob(pdg):
    return os.path.join(OUT, "sim", f"{lab(pdg)}_pt3_{ARM}_s1??_sim.root")


def model_path(pdg):
    return os.path.join(OUT, "model", f"model_{lab(pdg)}_pt3_all4{MODEL_TAG}.root")


def _md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


# =========================================================================
# setup
# =========================================================================

def _gen_module():
    spec = importlib.util.spec_from_file_location(
        "gen_toy_realmat", os.path.join(SRCTEST, "gen_toy_realmat.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _plane_body(radii, q):
    g = _gen_module()
    org, nrm, uu = g.helix_frames(radii, PT, ETA, PHI, BFIELD, q)
    f = lambda v: ", ".join("%.6f" % x for x in v)          # noqa: E731
    # byte for byte gen_toy_realmat.py's writer, minus its header line
    return (f"radii = [{f(radii)}]\norigin = [{f(org)}]\n"
            f"normal = [{f(nrm)}]\nuaxis  = [{f(uu)}]\n")


def cmd_setup(args):
    ref_xml = os.path.realpath(os.path.join(SRCDATA, "realmat_full", "tracker.xml"))
    ref_pl = os.path.join(SRCTEST, "toyPlanes_realmat_full_pt3.py")
    hx = _md5(ref_xml)
    print(f"reference geometry {ref_xml}\n  md5 {hx}")
    txt = open(ref_pl).read()
    head, body = txt.split("\n", 1)
    ns = {}
    exec(txt, ns)
    radii = list(ns["radii"])
    print(f"reference planes   {ref_pl}\n  {len(radii)} planes, "
          f"r = {radii[0]:.3f} ... {radii[-1]:.3f} cm")

    # the generator must reproduce the published q = -1 planes EXACTLY,
    # otherwise its q = +1 mirror cannot be trusted either
    mine = _plane_body(radii, -1.0)
    if mine != body:
        raise SystemExit("helix_frames(q=-1) does NOT reproduce the published "
                         "plane file -- the q=+1 mirror cannot be trusted")
    print("  helix_frames(q=-1) reproduces the published plane file: IDENTICAL")

    for g, (gdir, pmod, q) in GEOM.items():
        td = testdir(g)
        dd = os.path.join(geomdir(g), "Analysis", "HitAnalyzer", "data", gdir)
        os.makedirs(td, exist_ok=True)
        os.makedirs(dd, exist_ok=True)
        xo = os.path.join(dd, "tracker.xml")
        shutil.copyfile(ref_xml, xo)
        assert _md5(xo) == hx
        print(f"[{g}] {_md5(xo)}  {gdir}/tracker.xml  IDENTICAL to the reference")
        if q < 0:
            out = head + "\n" + body
        else:
            out = (f"# generated by realmat_closure.py setup: gen_toy_realmat."
                   f"helix_frames(q=+1) on the radii of "
                   f"toyPlanes_realmat_full_pt3.py; do not hand edit\n"
                   + _plane_body(radii, +1.0))
        open(planes_path(g), "w").write(out)
        print(f"[{g}] {_md5(planes_path(g))}  {pmod}.py  (q = {q:+.0f})")

        s = open(os.path.join(SRCTEST, "runToyGeomCheck.py")).read()
        s = hp._sub1(s, r"^import FWCore\.ParameterSet\.Config as cms$",
                     "import os\nimport FWCore.ParameterSet.Config as cms",
                     "sim: import os", flags=re.M)
        s = hp._sub1(s, r"^process\.g4SimHits\.Physics\.DefaultCutValue = "
                        r"cms\.double\(1\.0\)$",
                     'process.g4SimHits.Physics.DefaultCutValue = cms.double(\n'
                     '    float(os.environ.get("TOY_CUT", "1.0")))\n'
                     "print('[toy] DefaultCutValue = %g cm'\n"
                     "      % process.g4SimHits.Physics.DefaultCutValue.value())",
                     "sim: production cut", flags=re.M)
        s = hp._sub1(s, r"^    output=cms\.string\(opts\.output\),\n\)\)$",
                     hp._HAD_BLOCK.rstrip("\n"), "sim: watcher block", flags=re.M)
        open(os.path.join(td, "runToyGeomCheck.py"), "w").write(s)
        m = open(os.path.join(SRCTEST, "runToyModel.py")).read()
        m = hp._sub1(m, r"^import toyPlanes_pt3 as planes$",
                     "import importlib, os as _os\nplanes = importlib.import_module("
                     '_os.environ["TOY_PLANES_MOD"])', "model: planes import",
                     flags=re.M)
        open(os.path.join(td, "runToyModel.py"), "w").write(m)
        print(f"[{g}] wrote runToyGeomCheck.py and runToyModel.py in {td}")

    # the mirror, checked: same radii, same z, azimuth reflected about PHI
    a, b = {}, {}
    exec(open(planes_path("qm")).read(), a)
    exec(open(planes_path("qp")).read(), b)
    om = np.asarray(a["origin"]).reshape(-1, 3)
    op = np.asarray(b["origin"]).reshape(-1, 3)
    pm = np.arctan2(om[:, 1], om[:, 0])
    pp = np.arctan2(op[:, 1], op[:, 0])
    print(f"mirror check: max|r_qm - r_qp| = "
          f"{np.abs(np.hypot(*om[:, :2].T) - np.hypot(*op[:, :2].T)).max():.2e} cm, "
          f"max|z_qm - z_qp| = {np.abs(om[:, 2] - op[:, 2]).max():.2e} cm, "
          f"max|(phi_qm + phi_qp)/2 - PHI| = "
          f"{np.abs(0.5 * (pm + pp) - PHI).max():.2e} rad")


# =========================================================================
# running
# =========================================================================

def _env(pdg, extra=None):
    sp = hp.SPECIES[pdg]
    g = gkey(pdg)
    e = dict(TOY_PLANES_MOD=GEOM[g][1], TOY_PDG=str(pdg), TOY_PNAME=sp["g4"],
             TOY_RMAX="107.0", TOY_CUT=repr(CUT),
             TOY_INACT=",".join(hp.inact_of(pdg, ARM)))
    e.update(extra or {})
    return e


def _run(g, script, extra, log, env_extra):
    """hadron_probe._run with this module's area: CVH switches as options."""
    swopts, env_extra = ctr.split_switches(env_extra)
    if swopts:
        extra = f"{extra} {swopts}"
    cmd = ("source /cvmfs/cms.cern.ch/cmsset_default.sh >/dev/null 2>&1 && "
           f"cd {SRC} && eval $(scramv1 runtime -sh) && "
           f"export CMSSW_SEARCH_PATH={geomdir(g)}:$CMSSW_SEARCH_PATH && "
           f"cd {testdir(g)} && exec {SRCTEST}/cmsswlock.sh run "
           f"cmsRun {script} {extra}")
    with open(log, "w") as fh:
        fh.write(f"# {cmd}\n")
        fh.flush()
        return subprocess.run(["bash", "-c", cmd], stdout=fh,
                              stderr=subprocess.STDOUT,
                              env=ds._clean_env(env_extra)).returncode


def _complete(log, nev):
    if not os.path.exists(log):
        return False
    m = re.search(r"\[plcensus\] wrote (\d+) records",
                  open(log, errors="ignore").read())
    return bool(m) and int(m.group(1)) == nev


def _sim_one(a):
    pdg, seed, nev, force = a
    out = sim_path(pdg, seed)
    log = out[:-5] + ".log"
    if os.path.exists(out) and _complete(log, nev) and not force:
        return pdg, seed, 0, log, "cached"
    ee = _env(pdg, {"TOY_CENSUS": out[:-9] + "_census.bin"})
    rc = _run(gkey(pdg), "runToyGeomCheck.py",
              f"events={nev} pt={PT} eta={ETA} output={out} seed={seed} "
              f"toyGeom={toygeom(gkey(pdg))}", log, ee)
    if rc == 0 and not _complete(log, nev):
        rc = 99
    return pdg, seed, rc, log, "ran"


def cmd_sim(args):
    import fcntl
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(os.path.join(OUT, "sim"), exist_ok=True)
    lk = open(os.path.join(OUT, "sim", ".sim.lock"), "w")
    try:
        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another `sim` holds the lock")
    seeds = SEEDS[:args.jobs]
    jobs = [(pdg, s, args.events, args.force) for pdg in args.pdg for s in seeds]
    print(f"{len(jobs)} jobs: {len(args.pdg)} species x {len(seeds)} seeds x "
          f"{args.events} events, {args.workers} at once", flush=True)
    t0, bad = time.time(), 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for pdg, seed, rc, log, how in ex.map(_sim_one, jobs):
            print(f"  {lab(pdg):<5} seed={seed} rc={rc} {how:<6} "
                  f"{time.time() - t0:7.0f} s", flush=True)
            if rc:
                bad += 1
                print(f"    *** FAILED, see {log}", flush=True)
    if bad:
        raise SystemExit(f"{bad} jobs failed")


def cmd_export(args):
    os.makedirs(os.path.join(OUT, "model"), exist_ok=True)
    for pdg in args.pdg:
        out = model_path(pdg)
        log = out[:-5] + ".log"
        if os.path.exists(out) and not args.force:
            print(f"[{lab(pdg)}] exists -> {out}")
            continue
        g = gkey(pdg)
        rc = _run(g, "runToyModel.py",
                  f"pt={PT} eta={ETA} phi={PHI} partId={pdg} output={out} "
                  f"toyGeom={toygeom(g)}", log, _env(pdg, FOUR_ON))
        print(f"[{lab(pdg)}] rc={rc} -> {out}", flush=True)
        if rc:
            raise SystemExit(f"export failed, see {log}")
        txt = open(log, errors="ignore").read()
        eff = re.search(r"\[cvh\] effective: (.*)", txt)
        print(f"    {eff.group(0) if eff else '*** no [cvh] effective line ***'}")


# =========================================================================
# liveness and pairs
# =========================================================================

def cmd_live(args):
    for pdg in args.pdg:
        sp = hp.SPECIES[pdg]
        logs = sorted(glob.glob(sim_glob(pdg)[:-5] + ".log"))
        print(f"\n### {sp['label']}  PDG {pdg:+d}  geometry {gkey(pdg)}  "
              f"{len(logs)} logs")
        if not logs:
            continue
        tot, prov = {}, set()
        for lg in logs:
            t = open(lg, errors="ignore").read()
            for k, v in hp.steps_from_log(lg).items():
                a, b = tot.get(k, (0, 0))
                tot[k] = (a + v[0], b + v[1])
            st = "TIGHT" if "[toy] TIGHT stepper" in t else "NOT-TIGHT"
            cut = re.search(r"\[toy\] DefaultCutValue = (\S+) cm", t)
            pid = re.search(r"\[toy\] PartID = (-?\d+) \((\S+)\)", t)
            rad = re.search(r"\[toy\] watcher radii from (\S+) \((\d+)\)", t)
            prov.add((st, cut.group(1) if cut else "?",
                      pid.group(1) if pid else "?",
                      rad.group(1) + "/" + rad.group(2) if rad else "?"))
        for p in sorted(prov):
            print(f"    stepper={p[0]} cut={p[1]} cm PartID={p[2]} planes={p[3]}")
        if len(prov) != 1 or next(iter(prov))[0] != "TIGHT":
            raise SystemExit("inconsistent or loose provenance")
        if int(next(iter(prov))[2]) != pdg:
            raise SystemExit("the gun fired the wrong particle")
        # The criterion is the PRIMARY count.  ProcessActivationWatcher is
        # restricted to the primary's species, so a species-agnostic process
        # name (`Decay`, `hadElastic`) still fires on OTHER species'
        # secondaries -- on this geometry the photonuclear pions and the
        # neutrons -- and `all` counts those.  A
        # species-specific name (`muonNuclear`, `pi-Inelastic`) must be zero
        # in `all` as well, which is asserted for every other name.
        for nm in hp.inact_of(pdg, ARM):
            got = tot.get(nm, (0, 0))
            bad = got[0] != 0 or (nm not in ("Decay", "hadElastic")
                                  and got[1] != 0)
            print(f"    census {nm:<24} primary {got[0]:>9} all {got[1]:>9}  "
                  f"{'*** NONZERO ***' if bad else 'OK (primary zero)'}")
            if bad:
                raise SystemExit(f"{nm} is not switched off")
        # the OTHER direction: whatever this arm restores relative to `off`
        # must actually act on the primary, or the arm is inert and any closure
        # difference is a plumbing null rather than physics
        off = set(hp.inact_of(pdg, "off"))
        for nm in sorted(off - set(hp.inact_of(pdg, ARM))):
            got = tot.get(nm, (0, 0))
            print(f"    census {nm:<24} primary {got[0]:>9} all {got[1]:>9}  "
                  f"{'OK (restored, fires on the primary)' if got[0] > 0 else '*** INERT ***'}")
            if got[0] == 0:
                raise SystemExit(f"{nm} is restored by arm {ARM} but never fired")
        sim = _sim(pdg)
        v = sim["valid"]
        print(f"    events {v.shape[0]}, reached every plane "
              f"{100 * v.all(axis=1).mean():.3f} %, per plane "
              f"min {100 * v.mean(axis=0).min():.3f} % "
              f"(outermost {100 * v[:, -1].mean():.3f} %)")


_SIMC = {}


def _sim(pdg):
    if pdg not in _SIMC:
        from toy_loader import load_toy_sim
        ns = {}
        exec(open(planes_path(gkey(pdg))).read(), ns)
        _SIMC[pdg] = load_toy_sim(sim_glob(pdg), ns["origin"], ns["normal"],
                                  ns["uaxis"])
    return _SIMC[pdg]


def cmd_pairs(args):
    """The four pair tests of geom_closure.cmd_pairs, plus a frame test that
    catches a mis-mirrored plane set: on every plane the sim's median local x
    must sit on the model's reference (a plane at the wrong azimuth is tens of
    cm off, not microns)."""
    for pdg in args.pdg:
        mp = model_path(pdg)
        m = gc._model_meta(mp)
        files = sorted(glob.glob(sim_glob(pdg)))
        s = gc._toy_sim_meta(files[0])
        sim = _sim(pdg)
        dE_m = 1e3 * float(m["refp"][0] - m["refp"][-1])
        nok = len(m["detid"]) == s["npl"]
        seqok = nok and bool(np.array_equal(np.asarray(m["detid"]), s["seq"]))
        dr = np.abs(np.asarray(m["refglobr"]) - s["r"]).max() if nok else np.nan
        ns = {}
        exec(open(planes_path(gkey(pdg))).read(), ns)
        rpl = np.asarray(ns["radii"])
        dxs = np.array([np.nanmedian(sim["locx"][sim["valid"][:, k], k])
                        - m["reflocx"][k] for k in range(len(rpl))])
        ok = (nok and seqok and dr < 0.01 and np.abs(dxs).max() < 0.05
              and 0.0 < dE_m - s["dE"] < 6.0)
        print(f"--- {hp.SPECIES[pdg]['label']:<5} model {os.path.basename(mp)}  "
              f"sim {len(files)} files")
        print(f"    (1) legs {len(m['detid'])} vs sim planes {s['npl']} "
              f"{'OK' if nok else 'MISMATCH'}")
        print(f"    (2) plane sequence identical: {seqok}")
        print(f"    (3) max |refglobr - sim median r| = {dr:.5f} cm; "
              f"max |refglobr - plane radius| = "
              f"{np.abs(np.asarray(m['refglobr']) - rpl).max():.5f} cm")
        print(f"    (4) dE model(mean) {dE_m:.3f} MeV  sim(median, first file) "
              f"{s['dE']:.3f} MeV  gap {dE_m - s['dE']:+.3f} MeV")
        print(f"    (5) max |median(sim locx) - reflocx| = "
              f"{1e4 * np.abs(dxs).max():.1f} um  (outermost "
              f"{1e4 * dxs[-1]:+.1f} um)")
        print(f"    {'PASS' if ok else '*** FAIL ***'}")
        if not ok and not args.nofail:
            raise SystemExit("pair gate failed")


# =========================================================================
# the closure
# =========================================================================

def cell(pdg, func, useh=True, nucel=False, recoil=True):
    """One closure cell: the per-plane rows over the full u curve.

    Same cache discipline as allcorr.cell (in-process scale cache, on-disk
    scale cache and phi cache all bypassed), Kokoulin per species, radiation
    ON, the six default-ON MS harmonisations asserted rather than assumed."""
    assert os.environ.get("RES_NO_PHI_CACHE"), "phi cache is LIVE"
    assert (ctr.MS_ELEC_TMAX, ctr.MS_ELEC_EDGE, ctr.MS_SNAP_YMAX, ctr.MS_FINE_G,
            ctr.MS_WVI_SPLIT) == (1.0, 1.0, 0.0, 1.0, 0.0), "MS defaults moved"
    assert mx.MS_CHI0_G4 and mx.MS_FF_G4, "cf_ms_exact harmonisations off"
    was = bool(hb._canonical().USE_H)       # set_use_h returns the NEW value
    hb.set_use_h(useh)
    old_nucel, old_recoil = cne.NUCEL_CHANNEL, cne.NUCEL_RECOIL
    cne.NUCEL_CHANNEL = bool(nucel)
    cne.NUCEL_RECOIL = bool(recoil)
    cpt.RAD_CHANNEL = True
    ctr.IONI_KOKOULIN = 1.0 if hp.kok_for(pdg) else 0.0
    ctr.IONI_KOKOULIN_TCUT = 0.0
    fn._SCALE_CACHE.clear()
    cpt._PHI_CACHE.clear()
    old_load, old_store = fn._scale_cache_load, fn._scale_cache_store
    fn._scale_cache_load = lambda *a, **k: None
    fn._scale_cache_store = lambda *a, **k: None
    try:
        path = model_path(pdg)
        legs = cpt.load_model(path)
        hb.bind(legs, path, pdg=pdg)
        # build the elastic kernels in the PARENT, before either pool forks
        cne.warm(legs)
        sim = _sim(pdg)
        sc = fn.plane_scales(legs, func, tag=path,
                             channels=("ioni", "ms", "rad"))
        rows, errs, ks, ns, err = gc.closure_rows(legs, sim, func, sc["sF"])
    finally:
        hb.set_use_h(was)
        cne.NUCEL_CHANNEL, cne.NUCEL_RECOIL = old_nucel, old_recoil
        fn._scale_cache_load, fn._scale_cache_store = old_load, old_store
        cpt.RAD_CHANNEL = True
        ctr.IONI_KOKOULIN = 1.0
        fn._SCALE_CACHE.clear()
        cpt._PHI_CACHE.clear()
    return dict(rows=np.asarray(rows).tolist(), errs=np.asarray(errs).tolist(),
                ks=np.asarray(ks).tolist(), ns=np.asarray(ns).tolist(),
                err_ladder=np.asarray(err).tolist(),
                sF=np.asarray(sc["sF"]).tolist(),
                sigma=np.asarray(sc["sigma"]).tolist(),
                nev=int(sim["valid"].shape[0]), arm=ARM, nucel=bool(nucel),
                recoil=bool(recoil))


def _variant(nucel, recoil=True):
    """File-name tag. The recoil sub-channel is part of the elastic channel's
    default; only its OFF state (a diagnostic) is tagged."""
    if ARM == "off" and not nucel and not MODEL_TAG:
        return ""
    v = f"_{ARM}_nucel{int(nucel)}" + ("" if (recoil or not nucel) else "_norecoil")
    v += MODEL_TAG
    if nucel and MODEL_TAG and not cne.NUCEL_ELEMENTS:
        v += "_noelem"            # table present but per-element targets off
    return v


def _res_path(pdg, func=None, useh=True, nucel=False, recoil=True):
    """One JSON per (species, direction), written as soon as the cell is done,
    so a species can be split across processes and a crash loses one cell.
    The name carries the sim arm and the elastic-channel state."""
    tag = "" if func is None else f"_{func}"
    return os.path.join(RES, f"closure_{lab(pdg)}{tag}{_variant(nucel, recoil)}"
                             f"{'' if useh else '_legacy'}.json")


def _load_cells(pdg, nucel=False):
    """{func: cell} from the per-cell files, falling back to a per-species
    file holding several cells."""
    cells, meta = {}, None
    p = _res_path(pdg, nucel=nucel)
    if os.path.exists(p):
        d = json.load(open(p))
        cells.update(d["cells"])
        meta = d
    for func in FUNCS:
        q = _res_path(pdg, func, nucel=nucel)
        if os.path.exists(q):
            d = json.load(open(q))
            cells.update(d["cells"])
            meta = d
    return cells, meta


def cmd_closure(args):
    os.makedirs(RES, exist_ok=True)
    iu = list(fn.UCURVE).index(1.0)
    for pdg in args.pdg:
        for func in args.funcs:
            out = dict(pdg=pdg, label=hp.SPECIES[pdg]["label"],
                       ucurve=list(fn.UCURVE), model=model_path(pdg),
                       sim=sim_glob(pdg),
                       model_sha256=hashlib.sha256(open(model_path(pdg), "rb")
                                                   .read()).hexdigest(),
                       basis="H" if args.useh else "legacy", arm=ARM,
                       nucel=bool(args.nucel), recoil=bool(args.recoil),
                       nucel_elements=bool(cne.NUCEL_ELEMENTS),
                       model_tag=MODEL_TAG, cells={})
            if func == "locy" and not args.useh:
                continue
            t0 = time.time()
            r = cell(pdg, func, useh=args.useh, nucel=args.nucel,
                     recoil=args.recoil)
            out["cells"][func] = r
            row = np.asarray(r["rows"])[:, iu]
            err = np.asarray(r["errs"])[:, iu]
            print(f"{out['label']:<5} {func:<5} [{ARM} nucel={int(args.nucel)}"
                  f"{'' if args.recoil else ' norecoil'}] u=1  outer {row[-1]:+.4f} +- "
                  f"{err[-1]:.4f}   ladder {row.mean():+.4f} +- "
                  f"{r['err_ladder'][iu]:.4f}   ({time.time() - t0:.0f} s)",
                  flush=True)
            print("      per plane " + " ".join(f"{x:+.4f}" for x in row),
                  flush=True)
            p = _res_path(pdg, func, args.useh, args.nucel, args.recoil)
            json.dump(out, open(p, "w"), indent=1)
            print(f"-> {p}", flush=True)


def cmd_table(args):
    iu = None
    lines = ["| | $q/p$ | local $x$ | local $y$ | $dx/dz$ | $dy/dz$ |",
             "|---|--:|--:|--:|--:|--:|"]
    lad = list(lines)
    tex = {13: r"$\mu^-$", -13: r"$\mu^+$", -211: r"$\pi^-$", 211: r"$\pi^+$",
           -321: r"$K^-$", 321: r"$K^+$", -2212: r"$\bar p$", 2212: r"$p$"}
    for pdg in args.pdg:
        cells, d = _load_cells(pdg, args.nucel)
        if d is None:
            continue
        iu = d["ucurve"].index(1.0)
        c, cl = [], []
        for func in FUNCS:
            r = cells.get(func)
            if r is None:
                c.append("--")
                cl.append("--")
                continue
            rows, errs = np.asarray(r["rows"]), np.asarray(r["errs"])
            c.append(f"${rows[-1, iu]:+.4f}\\pm{errs[-1, iu]:.4f}$")
            cl.append(f"${rows[:, iu].mean():+.4f}\\pm{r['err_ladder'][iu]:.4f}$")
        lines.append(f"| {tex[pdg]} | " + " | ".join(c) + " |")
        lad.append(f"| {tex[pdg]} | " + " | ".join(cl) + " |")
    head = ("Clean-propagation closure on the real-material toy `realmat_full` "
            "(19 planes), pT = 3, eta = 0.30, phi = 0.70.\n"
            f"Sim: {NEV * len(SEEDS)} events per species, tight stepper, cut "
            f"{CUT:g} cm, arm `{ARM}`: {hp.ARMS[ARM]} (for the primary). Model: "
            "CMSSW dev2 cfi defaults with the four ionisation corrections passed "
            "explicitly (radiative reference and records ON for every species). "
            "Offline: radiation on, Kokoulin for muons, six default MS "
            "harmonisations, MS_WVI_SPLIT off, nuclear-elastic channel "
            f"{'ON (with recoil)' if args.nucel else 'OFF'}. Fisher normalisation, "
            "H basis. "
            "Produced by realmat_closure.py; per-cell JSON (full u curve, "
            "per-plane rows) next to this file.\n\n")
    txt = (head + "Outermost plane, u = 1, H basis:\n\n" + "\n".join(lines)
           + "\n\nLadder mean over planes, u = 1, H basis:\n\n" + "\n".join(lad)
           + "\n")
    print(txt)
    os.makedirs(RES, exist_ok=True)
    open(os.path.join(RES, f"table{_variant(args.nucel)}.md"), "w").write(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup").set_defaults(f=cmd_setup)
    for name, f in (("sim", cmd_sim), ("live", cmd_live), ("export", cmd_export),
                    ("pairs", cmd_pairs), ("closure", cmd_closure),
                    ("table", cmd_table)):
        q = sub.add_parser(name)
        q.add_argument("--pdg", type=int, nargs="+",
                       default=ORDER8 if name == "table" else [13, -13])
        q.set_defaults(f=f)
        q.add_argument("--model-tag", default="",
                       help="model-file suffix (`_mat`: export with the per-step "
                            "material table the per-element elastic channel needs)")
        q.add_argument("--arm", default="off", choices=sorted(hp.ARMS),
                       help="Geant4 process set of the sim (hadron_probe.ARMS); "
                            "`elonly` = hadElastic ON, inelastic and Decay OFF")
        if name in ("closure", "table"):
            q.add_argument("--nucel", action="store_true",
                           help="nuclear-elastic channel ON in the prediction")
        if name == "sim":
            q.add_argument("--jobs", type=int, default=len(SEEDS))
            q.add_argument("--events", type=int, default=NEV)
            q.add_argument("--workers", type=int, default=20)
            q.add_argument("--force", action="store_true")
        if name == "export":
            q.add_argument("--force", action="store_true")
        if name == "pairs":
            q.add_argument("--nofail", action="store_true")
        if name == "closure":
            q.add_argument("--funcs", nargs="+", default=list(FUNCS))
            q.add_argument("--legacy", dest="useh", action="store_false")
            q.add_argument("--no-recoil", dest="recoil", action="store_false",
                           help="diagnostic: the elastic channel without its "
                                "recoil (q/p) sub-channel")
    a = ap.parse_args()
    global ARM
    ARM = getattr(a, "arm", "off")
    global MODEL_TAG
    MODEL_TAG = getattr(a, "model_tag", "")
    a.f(a)


if __name__ == "__main__":
    main()
