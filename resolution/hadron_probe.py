#!/usr/bin/env python3
"""Does the clean-propagation ionization + MS + radiative model close on
HADRONS the way it does on the muon?

WHY THIS EXISTS
---------------
Two of the B-field-calibration channels this study exists to serve are hadronic:
`B+- -> J/psi K+-` fits the KAON track, and the V0 tests fit `KS -> pi pi` and
`Lambda -> p pi`.  Everything validated so far -- the exact delta-ray spectrum
(NOTES_DELTASPEC), the Kokoulin radiative correction (NOTES_SAMPLERGAP), the
radiation-off 2x2 (NOTES_RADOFF2) -- was measured on a MUON.  NOTES_DELTASPEC
section 9 took one step towards a hadron: it showed the parameter-free
delta-ray RATE prediction holds to 0.03 % for K- and pi-.  It explicitly did
NOT do the closure, and it did not touch the proton.

The closure is the harder test, and it is the one the calibration needs.

WHAT IS SWITCHED OFF, AND WHY IT IS OFF FIRST
---------------------------------------------
Three Geant4 channels the model does not represent AT ALL:

  * hadronic INELASTIC (`kaon-Inelastic`, `pi-Inelastic`, `protonInelastic`,
    `anti_protonInelastic`) -- the toy is 12.6 g/cm2 of Z = 8, i.e. ~15 % of a
    nuclear interaction length, so leaving it on truncates O(10 %) of tracks;
  * hadronic ELASTIC (`hadElastic`) -- a discrete large-angle kick with a
    nuclear recoil, not in the Moliere channel;
  * `Decay` -- K- has c*tau 3.7 m and pi- 7.8 m, so at a few GeV a percent-level
    fraction decays in flight.  A decay in flight is a KINK: a discrete change
    of the track state that the ionization + MS + radiative model does not
    model, landing exactly where the tail non-closure is measured.  It also
    TERMINATES the track, and acceptance loss has already been the single
    largest error in this study once (NOTES_STEPCORR: 5.6 % at pT = 3,
    tail-selective).

Measuring the model against a sample that contains them measures the union of
"the model is wrong" and "the model was never asked".  So they go off first,
the model is judged on what it claims to describe, and only THEN are they put
back to size the missing channel.  Both directions are reported.

EVERY SWITCH IS A MEASUREMENT
-----------------------------
`process.g4SimHits.G4Commands` is inert and a watcher outside the Simulation
biglib is inert (and segfaults) -- see ProcessActivationWatcher.cc.  The switch
is `ProcessActivationWatcher` INSIDE the biglib, and the evidence is its own
EndOfRun step census: a deactivated process must define EXACTLY ZERO steps.
NOTES_DELTASPEC records the trap this catches: a first pass fired a K+ while
deactivating `kaon--inelastic` for a K-, and the census reported
`kaon+Inelastic primary 20149`.  Here the process names are asserted against
the census's own output per species, and the ACCEPTANCE (fraction of primaries
reaching the outermost scoring surface) is reported alongside as the direct
observable.

THE KOKOULIN TRAP, WHICH IS SPECIES-DEPENDENT AND OFFLINE IS NOT GUARDED
-----------------------------------------------------------------------
Geant4 applies R. Kokoulin's radiative correction to the knock-on cross section
in `G4MuBetheBlochModel` ONLY.  Hadrons are ionized by `G4hIonisation`, whose
model above 2 MeV * m/m_p is `G4BetheBlochModel`, which has NO such factor.  So
the correct model configuration is species-dependent:

    muon    exact-delta ON  + Kokoulin ON
    pi/K/p  exact-delta ON  + Kokoulin OFF

The C++ half already knows this
(`G4UniversalFluctuationForExtrapolator.cc`: `kokoulinOn && ekin > kKokMuMin &&
std::abs(particle->GetPDGEncoding()) == 13`).  **The OFFLINE half did not**
(FIXED 2026-08-16, NOTES_BARKAS s9.1: `_kokoulin_exponent` now recovers the
mass from the record as `m = E sqrt(1-beta^2)` and guards on it; a hadron
record is bit-identical with the switch on).  The description below is the
state this note was measured in, and the driving-it-explicitly is kept
because it is still the right thing to do.
`cf_track_resolution._kokoulin_exponent` guards only on
`etot - m_mu > _KOK_MUMIN`, and the exported record carries no PDG code, so
`CVH_IONI_KOKOULIN=1` in a hadron job silently applies a muon-shaped Kokoulin
factor (a3 = ln(4E(E-T)/m_mu^2)) to a kaon.  This module therefore drives
`cf_track_resolution.IONI_KOKOULIN` explicitly per species rather than through
the environment default, and measures the size of the wrong configuration so
the trap is documented rather than merely avoided.

CACHES
------
Three, all of which have to be busted when the Kokoulin or exact-delta state
moves and NONE of which is keyed on the particle:
  * `fisher_norm._SCALE_CACHE`   (in process)
  * the ON-DISK scale cache      (keyed by model content + source fingerprint)
  * `cf_propagation_test._PHI_CACHE`, whose key carries IONI_A3_SCALE,
    IONI_EXC_SCALE and IONI_TMAX_SCALE but **not** IONI_KOKOULIN.
The last is the dangerous one: a cached phi survives the Kokoulin switch.  This
module sets `RES_NO_PHI_CACHE=1` for the whole process (belt) and clears the
dict between arms (braces), and asserts the cache is disabled before reading.

SUBCOMMANDS
    setup      private area(s): geometry, planes (q = -1 and the q = +1 mirror),
               patched drivers; md5 against the archived tailhunt area
    sim        seeded-split simulations, all species x arms
    live       switch liveness: the step census + acceptance, per species
    export     the models (species x exact-delta x Kokoulin)
    rate       the parameter-free delta-ray rate/energy check per species,
               with and without Kokoulin -- the empirical proof of which
               species Geant4 corrects
    closure    THE RESULT: the u curve per species against the muon control
    corr       per-species effect of exact-delta and of Kokoulin
    nuc        nuclear + decay ON vs OFF: the size of the unmodelled channel
"""

import argparse
import glob
import hashlib
import math
import os
import re
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "cleanprop"))

# the phi cache key omits IONI_KOKOULIN -- disable it before anything imports
# a legs list.  (Also cleared explicitly between arms in `_rows`.)
os.environ["RES_NO_PHI_CACHE"] = "1"

import cf_propagation_test as cpt                               # noqa: E402
import cf_track_resolution as ctr                               # noqa: E402
import fisher_norm as fn                                        # noqa: E402
import geom_closure as gc                                       # noqa: E402
import deltaspec as ds                                          # noqa: E402
import tail_probe as tp                                         # noqa: E402

CMSSW = "/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev"
SRCTEST = f"{CMSSW}/src/Analysis/HitAnalyzer/test"
SCRATCH = fn.SCRATCH
OUT = os.path.join(SCRATCH, "had")
AREA = os.path.join(CMSSW, "hadprobe")        # must live under CMSSW_BASE
TH = os.path.join(SCRATCH, "th")              # NOTES_TAILHUNT's archived sims
REFAREA = os.path.join(CMSSW, "tailhunt", "pt3")   # the md5 reference geometry

UCURVE = fn.UCURVE
UHEAD = fn.UHEAD

# --------------------------------------------------------------- the species
# SIGNED PDG codes everywhere.  The toy planes are the tangent planes of a
# q = -1 reference helix (gen_toy_config.py: `q = -1.0`), so a POSITIVE particle
# at the same pT/eta/phi bends the other way and crosses the cylinders ~43 cm
# away in azimuth from the plane it is supposed to hit.  The four primary
# species are therefore all NEGATIVE and share the archived, md5-verified
# geometry; the real (positive) proton gets its own mirrored plane set, built
# from the same generator with q = +1, and is used as a cross-check on the
# antiproton and for the nuclear-ON arm (where p and pbar differ enormously).
#
# `inel` is the name the census PRINTS for that species' hadronic inelastic
# process, and `nuc` is the full nuclear set that goes off in the clean arm.
SPECIES = {
    13: dict(name="mu-", g4="mu-", mass=105.6583745, q=-1, geom="qm",
             inel=None, nuc=["muonNuclear"], label="mu-"),
    -211: dict(name="pi-", g4="pi-", mass=139.57039, q=-1, geom="qm",
               inel="pi-Inelastic", nuc=["pi-Inelastic", "hadElastic"],
               label="pi-"),
    -321: dict(name="kaon-", g4="kaon-", mass=493.677, q=-1, geom="qm",
               inel="kaon-Inelastic", nuc=["kaon-Inelastic", "hadElastic"],
               label="K-"),
    -2212: dict(name="anti_proton", g4="anti_proton", mass=938.27209, q=-1,
                geom="qm", inel="anti_protonInelastic",
                nuc=["anti_protonInelastic", "hadElastic"], label="pbar"),
    2212: dict(name="proton", g4="proton", mass=938.27209, q=+1, geom="qp",
               inel="protonInelastic",
               nuc=["protonInelastic", "hadElastic"], label="p"),
    # The CHARGE-CONJUGATE set, on the mirrored geometry.  Added after the
    # first pass found the mean-loss bias to be charge-ODD (-0.042 to -0.065
    # MeV for every q = -1 species, -0.004 MeV for the proton); running the
    # positive partner of each species is the way to test that rather than
    # infer it.
    -13: dict(name="mu+", g4="mu+", mass=105.6583745, q=+1, geom="qp",
              inel=None, nuc=["muonNuclear"], label="mu+"),
    211: dict(name="pi+", g4="pi+", mass=139.57039, q=+1, geom="qp",
              inel="pi+Inelastic", nuc=["pi+Inelastic", "hadElastic"],
              label="pi+"),
    321: dict(name="kaon+", g4="kaon+", mass=493.677, q=+1, geom="qp",
              inel="kaon+Inelastic", nuc=["kaon+Inelastic", "hadElastic"],
              label="K+"),
}
ORDER = [13, -211, -321, -2212, 2212]
ORDER_PLUS = [-13, 211, 321]

# arm -> what is inactivated.  `off` is the clean arm the model is judged on;
# `dec` isolates decay; `on` is stock physics.
ARMS = {
    "off": "nuclear (inelastic + elastic) and Decay OFF",
    "dec": "nuclear OFF, Decay ON",
    "on":  "stock physics: nuclear and Decay ON",
    "norad": "nuclear, Decay AND bremsstrahlung + pair production OFF",
    # `elonly` and `inelonly` SPLIT the `off` arm's nuclear set, which until now
    # only ever went off as a unit.  The 0.00035-0.00141 rms figure quoted for
    # "nuclear" is elastic AND inelastic together; after the MS harmonisation
    # the locx residual is 0.8-2.9 sigma, so that combined figure is comparable
    # to what is left and the two halves have to be told apart before a nuclear
    # elastic channel is designed.  Each arm turns exactly ONE half back on
    # relative to `off`, so (elonly - off) is the elastic-alone effect and
    # (inelonly - off) the inelastic-alone one, in the same base-and-difference
    # form cmd_nuc already uses for (dec - off).
    "elonly": "inelastic and Decay OFF, hadElastic ON -- elastic alone",
    "inelonly": "hadElastic and Decay OFF, inelastic ON -- inelastic alone",
}
# the radiative process names, per species, as the census PRINTS them
RADPROC = {13: ["muBrems", "muPairProd"], -13: ["muBrems", "muPairProd"]}
for _p in (-211, -321, -2212, 2212, 211, 321):
    RADPROC[_p] = ["hBrems", "hPairProd"]


def inact_of(pdg, arm):
    sp = SPECIES[pdg]
    if arm == "off":
        return list(sp["nuc"]) + ["Decay"]
    if arm == "dec":
        return list(sp["nuc"])
    if arm == "on":
        return []
    if arm == "norad":
        # For a HADRON this is a ONE-SIDED switch that makes the simulation
        # consistent with the model rather than the other way round: the
        # exported model has dedxrad = 0 identically, so the sim's hBrems +
        # hPairProd have no counterpart at all.  For the MUON it is the
        # opposite -- the model DOES carry the channel -- so a muon `norad`
        # arm is deliberately inconsistent and is run only as a gauge of the
        # channel's size, never as a closure.
        return list(sp["nuc"]) + ["Decay"] + RADPROC[pdg]
    if arm in ("elonly", "inelonly"):
        # Built by REMOVING one member from the clean arm's nuclear set rather
        # than by listing what stays off, so a species that does not carry the
        # removed process degenerates to `off` by construction.  The muon has
        # no hadElastic and no inelastic entry, so BOTH muon arms come out
        # identical to `off` -- that is the null this measurement needs.  Note
        # it is a CONFIG-level identity (same inactivate list, same seeds, so
        # bit-identical output), which makes it a control on the plumbing, not
        # evidence that the elastic channel leaves muons alone.
        drop = "hadElastic" if arm == "elonly" else sp["inel"]
        return [p for p in sp["nuc"] if p != drop] + ["Decay"]
    raise KeyError(arm)


# the production cut: `cut1e4`, i.e. DefaultCutValue = 1e-4 cm -> 9.4862 keV
# e- threshold in ToyLayerMat.  This is the EXACT-CUT configuration; the default
# 1.0 cm cut supplies a partial cancellation (NOTES_TAILHUNT s3) and is not used.
CUT = 1e-4
PT = 3.0
ETA, PHI = 0.30, 0.70


def tag_of(pdg, arm, pt=PT):
    return f"{SPECIES[pdg]['label'].replace('-','m').replace('+','p')}" \
           f"_pt{pt:g}_{arm}"


# =========================================================================
# 0.  the private area
# =========================================================================

_HAD_BLOCK = '''    output=cms.string(opts.output),
))

# ------------------------------------------------------------ HADRON PROBE
_pdg = int(os.environ["TOY_PDG"])
_pname = os.environ["TOY_PNAME"]
process.generator.PGunParameters.PartID = cms.vint32(_pdg)
print('[toy] PartID = %d (%s)' % (_pdg, _pname))
_inact = [s for s in os.environ.get("TOY_INACT", "").split(",") if s]
print('[toy] INACTIVATE: %s' % (",".join(_inact) if _inact else "(nothing)"))
process.g4SimHits.Watchers.append(cms.PSet(
    type=cms.string('ProcessActivationWatcher'),
    inactivate=cms.untracked.vstring(*_inact),
    activate=cms.untracked.vstring(),
    particles=cms.untracked.vstring(_pname),
))
process.g4SimHits.Watchers.append(cms.PSet(
    type=cms.string('PrimaryLossCensusWatcher'),
    rmax=cms.untracked.double(float(os.environ.get("TOY_RMAX", "107.0"))),
    hard=cms.untracked.double(1.0),
    output=cms.untracked.string(os.environ.get("TOY_CENSUS",
                                               "primaryloss.bin")),
))'''


def geomdir(g):
    return os.path.join(AREA, g)


def testdir(g):
    return os.path.join(geomdir(g), "Analysis", "HitAnalyzer", "test")


def planes_mod(g):
    return "toyPlanes_pt3" if g == "qm" else "toyPlanes_pt3_qp"


def planes_path(g):
    return os.path.join(testdir(g), planes_mod(g) + ".py")


def _sub1(text, pat, repl, what, flags=0):
    new, n = re.subn(pat, repl, text, flags=flags)
    if n != 1:
        raise SystemExit(f"patch '{what}' matched {n} times, expected 1")
    return new


def _mk_planes(q, out):
    """The tangent planes of the reference helix, from gen_toy_config.py's own
    formula.  For q = -1 the result is byte-compared against the archived file
    rather than trusted, so the q = +1 mirror is produced by code that is
    PROVEN to reproduce the validated one."""
    src = open(f"{SRCTEST}/gen_toy_config.py").read()
    m = re.search(r"^RADII = (\[[^\]]*\])", src, re.M)
    radii = eval(m.group(1))
    R = PT / (0.3 * 3.8) * 100.0
    org, nrm, uu = [], [], []
    for r in radii:
        a = math.asin(min(r / (2 * R), 1.0))
        phi = PHI - q * a
        z = 2 * R * a * math.sinh(ETA)
        org += [r * math.cos(phi), r * math.sin(phi), z]
        nrm += [math.cos(phi), math.sin(phi), 0.0]
        uu += [-math.sin(phi), math.cos(phi), 0.0]
    f = lambda v: ", ".join("%.6f" % x for x in v)   # noqa: E731
    # byte for byte gen_toy_config.py's writer, T = 0.1 cm
    body = ("# generated by gen_toy_config.py for T=0.1 cm; do not hand edit\n"
            f"radii = [{f(radii)}]\norigin = [{f(org)}]\n"
            f"normal = [{f(nrm)}]\nuaxis  = [{f(uu)}]\n")
    open(out, "w").write(body)
    return body


def cmd_setup(args):
    """Build both geometry areas.  tracker.xml is charge- and pT-independent
    (concentric shells), so BOTH areas get the archived one, byte for byte; only
    the planes differ."""
    ref_xml = os.path.join(REFAREA, "Analysis/HitAnalyzer/data/tracker.xml")
    ref_pl = os.path.join(REFAREA, "Analysis/HitAnalyzer/test/toyPlanes_pt3.py")
    hx = hashlib.md5(open(ref_xml, "rb").read()).hexdigest()
    hp = hashlib.md5(open(ref_pl, "rb").read()).hexdigest()
    print(f"reference (tailhunt/pt3, the area every published toy number "
          f"was produced in):")
    print(f"  {hx}  tracker.xml")
    print(f"  {hp}  toyPlanes_pt3.py")

    for g in ("qm", "qp"):
        td, dd = testdir(g), os.path.join(geomdir(g), "Analysis/HitAnalyzer/data")
        os.makedirs(td, exist_ok=True)
        os.makedirs(dd, exist_ok=True)
        shutil.copy(ref_xml, os.path.join(dd, "tracker.xml"))
        h = hashlib.md5(open(os.path.join(dd, "tracker.xml"), "rb").read()).hexdigest()
        assert h == hx
        print(f"[{g}] {h}  tracker.xml   IDENTICAL to the archived area")

    # q = -1: the generator's own formula must reproduce the archived file
    # EXACTLY, otherwise the q = +1 mirror is not trustworthy either.
    mine = _mk_planes(-1.0, os.path.join(testdir("qm"), "toyPlanes_pt3.py"))
    theirs = open(ref_pl).read()
    hm = hashlib.md5(mine.encode()).hexdigest()
    ok = "IDENTICAL" if hm == hp else "*** DIFFERENT ***"
    print(f"[qm] {hm}  toyPlanes_pt3.py   regenerated vs archived: {ok}")
    if hm != hp:
        raise SystemExit("the plane generator does not reproduce the archived "
                         "planes -- the q=+1 mirror cannot be trusted")
    b = _mk_planes(+1.0, os.path.join(testdir("qp"), "toyPlanes_pt3_qp.py"))
    print(f"[qp] {hashlib.md5(b.encode()).hexdigest()}  toyPlanes_pt3_qp.py"
          f"   (q = +1 mirror, same generator, same RADII)")

    # the drivers
    for g in ("qm", "qp"):
        td = testdir(g)
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
                  _HAD_BLOCK.rstrip("\n"), "sim: watcher block", flags=re.M)
        open(f"{td}/runToyGeomCheck.py", "w").write(s)
        m = open(f"{SRCTEST}/runToyModel.py").read()
        m = _sub1(m, r"^import toyPlanes_pt3 as planes$",
                  "import importlib, os as _os\nplanes = importlib.import_module("
                  '_os.environ["TOY_PLANES_MOD"])', "model: planes import",
                  flags=re.M)
        open(f"{td}/runToyModel.py", "w").write(m)
        print(f"[{g}] wrote runToyGeomCheck.py and runToyModel.py in {td}")


# =========================================================================
# 1.  running
# =========================================================================

def _env(pdg, arm, extra=None):
    sp = SPECIES[pdg]
    e = dict(TOY_PT=repr(PT), TOY_PLANES=planes_mod(sp["geom"]) + ".py",
             TOY_PLANES_MOD=planes_mod(sp["geom"]),
             TOY_PDG=str(pdg), TOY_PNAME=sp["g4"], TOY_RMAX="107.0",
             TOY_CUT=repr(CUT), TOY_INACT=",".join(inact_of(pdg, arm)))
    e.update(extra or {})
    return e


def _run(g, script, extra, log, env_extra):
    td = geomdir(g)
    cmd = ("source /cvmfs/cms.cern.ch/cmsset_default.sh >/dev/null 2>&1 && "
           f"cd {CMSSW}/src && eval $(scramv1 runtime -sh) && "
           f"export CMSSW_SEARCH_PATH={td}:$CMSSW_SEARCH_PATH && "
           f"cd {testdir(g)} && exec {SRCTEST}/cmsswlock.sh run "
           f"cmsRun {script} {extra}")
    os.makedirs(OUT, exist_ok=True)
    with open(log, "w") as fh:
        return subprocess.run(["bash", "-c", cmd], stdout=fh,
                              stderr=subprocess.STDOUT,
                              env=ds._clean_env(env_extra)).returncode


def sim_path(pdg, arm, seed):
    return os.path.join(OUT, f"{tag_of(pdg, arm)}_s{seed}_sim.root")


# THE SEED PIN, INSTALLED AT THE DEFINITION.
#
# NOTES_BARKAS s3.4 ran a 20 000-event seed-901 mu- job on arm `off` to show
# the LD_PRELOAD shim is inert inside cmsRun; it wrote `mum_pt3_off_s901_*`
# into this same directory.  An unpinned `_s*` therefore returns ELEVEN files
# for the muon `off` arm -- 220 000 events against 200 000 for every other
# species -- and the muon control stops reproducing the published digits.
#
# The pin already existed, but only as an IMPORT SIDE-EFFECT in two downstream
# modules: chargeodd.py:154 (sim only, not census) and pion_probe.py:128-129
# (both).  allcorr.py:114-115 asserts it is installed and gets it because it
# imports pion_probe.  Bare `python hadron_probe.py ...` -- which is the form
# NOTES_HADRONS s8 documents -- imported neither, so the arm that the whole
# study takes its differences against was silently running on 11 files.  That
# is trap #9 (half-pinned globs) in its own home file.
#
# Pinning here makes the definition the source of truth.  Both downstream
# patches stay correct and become no-ops: pion_probe's `.replace("_s*","_s1*")`
# finds no `_s*` and returns this string unchanged, chargeodd's override
# produces the identical pattern, and allcorr's `"_s1" in ...` assert passes.
# `_s1??` (not `_s1*`) so it matches exactly the three-digit 1xx campaign
# seeds and cannot pick up a future `_s1_` or `_s1000_`.
def sim_glob(pdg, arm):
    return os.path.join(OUT, f"{tag_of(pdg, arm)}_s1??_sim.root")


def census_glob(pdg, arm):
    return os.path.join(OUT, f"{tag_of(pdg, arm)}_s1??_census.bin")


def log_glob(pdg, arm):
    """The job logs, pinned to the SAME seeds as sim_glob/census_glob.

    The step census that proves a switch is live is summed over these, so an
    unpinned log glob would attribute seed-901's steps to a 10-seed sample --
    the same half-pinning as the sim and census globs, one file further on.
    """
    return os.path.join(OUT, f"{tag_of(pdg, arm)}_s1??_sim.log")


def _complete(log, nev):
    """A job counts as done ONLY if PrimaryLossCensusWatcher printed that it
    wrote the full record count at EndOfRun.

    Not a nicety: a first attempt at this campaign left 460-byte ROOT files
    behind (cmsRun's TFileService creates the file at BeginJob, long before it
    holds any data), so a size-based cache check treats a job that was killed
    at event 1 as finished.  The census line is written after the last event
    and carries the count, so it cannot be faked by a truncated run."""
    if not os.path.exists(log):
        return False
    m = re.search(r"\[plcensus\] wrote (\d+) records", open(log, errors="ignore").read())
    return bool(m) and int(m.group(1)) == nev


def _sim_one(a):
    pdg, arm, seed, nev = a
    sp = SPECIES[pdg]
    out = sim_path(pdg, arm, seed)
    log = out[:-5] + ".log"
    if os.path.exists(out) and _complete(log, nev) and not _FORCE:
        return pdg, arm, seed, 0, log, "cached"
    ee = _env(pdg, arm, {"TOY_CENSUS": out[:-9] + "_census.bin"})
    rc = _run(sp["geom"], "runToyGeomCheck.py",
              f"events={nev} pt={PT} eta={ETA} output={out} seed={seed}",
              log, ee)
    if rc == 0 and not _complete(log, nev):
        rc = 99            # exited clean but did not finish the run
    return pdg, arm, seed, rc, log, "ran"


_FORCE = False


def cmd_sim(args):
    """EXCLUSIVE.  Two concurrent invocations write the SAME output paths --
    which happened once here (a `nohup ... &` that outlived the shell plus a
    second launch) and makes every file in the campaign untrustworthy.  A
    second `sim` therefore refuses to start rather than racing."""
    import fcntl
    global _FORCE
    _FORCE = args.force
    os.makedirs(OUT, exist_ok=True)
    lk = open(os.path.join(OUT, ".sim.lock"), "w")
    try:
        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another `hadron_probe.py sim` holds the lock -- "
                         "refusing to write the same files twice")
    lk.write(f"{os.getpid()}\n")
    lk.flush()
    from concurrent.futures import ThreadPoolExecutor
    jobs = []
    for pdg in args.pdg:
        for arm in args.arms:
            for i in range(args.jobs):
                jobs.append((pdg, arm, args.seed0 + i, args.events))
    print(f"{len(jobs)} jobs: {len(args.pdg)} species x {len(args.arms)} arms "
          f"x {args.jobs} seeds x {args.events} events")
    bad = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for pdg, arm, seed, rc, log, how in ex.map(_sim_one, jobs):
            txt = open(log, errors="ignore").read(60000) if os.path.exists(log) else ""
            st = ("TIGHT" if "[toy] TIGHT stepper" in txt else
                  ("LOOSE" if "LOOSE stepper" in txt else "?"))
            cut = re.search(r"\[toy\] DefaultCutValue = (\S+) cm", txt)
            pid = re.search(r"\[toy\] PartID = (\S+) \((\S+)\)", txt)
            print(f"  {tag_of(pdg, arm):<16} seed={seed} rc={rc} {how:<6} "
                  f"stepper={st} cut={cut.group(1) if cut else '?'} "
                  f"pid={pid.group(0)[13:] if pid else '?'}")
            if rc or (how == "ran" and st != "TIGHT"):
                bad += 1
                print(f"    *** FAILED, see {log}")
    if bad:
        raise SystemExit(f"{bad} jobs failed")


# =========================================================================
# 2.  switch liveness -- the step census and the acceptance
# =========================================================================

_CEN = {}


def census_of(pdg, arm):
    key = (pdg, arm)
    if key in _CEN:
        return _CEN[key]
    files = sorted(glob.glob(census_glob(pdg, arm)))
    sims = sorted(glob.glob(sim_glob(pdg, arm)))
    if not files:
        raise FileNotFoundError(census_glob(pdg, arm))
    ks = [os.path.basename(f).rsplit("_", 1)[0] for f in files]
    kss = [os.path.basename(f).rsplit("_", 1)[0] for f in sims]
    if ks != kss:
        raise SystemExit(f"census/sim file sets differ for {tag_of(pdg,arm)}")
    _CEN[key] = np.concatenate(
        [np.fromfile(f, dtype=np.float32).reshape(-1, tp.NREC) for f in files])
    return _CEN[key]


def steps_from_log(log):
    """The EndOfRun step census, as {process: (primary, all)}.  Absence from the
    table IS zero steps -- the watcher only prints processes it saw."""
    out = {}
    for line in open(log, errors="ignore"):
        m = re.match(r"\[procact\]   (\S+)  primary (\d+)  all (\d+)", line)
        if m:
            out[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    return out


def _acc(pdg, arm):
    c = census_of(pdg, arm)
    return c, float((c[:, tp.I_RLAST] > 106.85).mean())


def cmd_live(args):
    print("=" * 108)
    print("SWITCH LIVENESS.  A deactivated process must define EXACTLY ZERO "
          "steps; absence from the")
    print("census table IS zero.  Acceptance = primaries reaching the "
          "outermost scoring surface (r > 106.85 cm).")
    print("=" * 108)
    for pdg in args.pdg:
        sp = SPECIES[pdg]
        print(f"\n### {sp['label']}  (PDG {pdg:+d}, {sp['g4']}, "
              f"m = {sp['mass']} MeV, geometry {sp['geom']})")
        for arm in args.arms:
            logs = sorted(glob.glob(log_glob(pdg, arm)))
            if not logs:
                print(f"  [{arm}] no logs")
                continue
            tot = {}
            pid, req = None, None
            for lg in logs:
                for k, v in steps_from_log(lg).items():
                    a, b = tot.get(k, (0, 0))
                    tot[k] = (a + v[0], b + v[1])
                if pid is None:
                    t = open(lg, errors="ignore").read(60000)
                    m = re.search(r"\[toy\] PartID = (-?\d+) \((\S+)\)", t)
                    pid = m.groups() if m else None
                    req = re.findall(r"SetProcessActivation\((\S+), false\)", t)
            c, acc = _acc(pdg, arm)
            print(f"  [{arm:<3}] {ARMS[arm]}")
            print(f"        gun fired PartID = {pid[0]} ({pid[1]})   "
                  f"requested INACTIVE: {req if req else '(nothing)'}")
            if pid is not None and int(pid[0]) != pdg:
                raise SystemExit(f"*** the gun fired {pid[0]}, not {pdg}")
            want = inact_of(pdg, arm)
            for nm in want:
                got = tot.get(nm, (0, 0))
                flag = "OK zero" if got == (0, 0) else "*** NONZERO ***"
                print(f"        census  {nm:<24} primary {got[0]:>9}  "
                      f"all {got[1]:>9}   {flag}")
                if got != (0, 0):
                    raise SystemExit(f"{nm} defined {got} steps in "
                                     f"{tag_of(pdg,arm)} -- switch NOT live")
            # The OTHER direction, asserted rather than eyeballed.  Whatever
            # this arm RESTORES relative to `off` must actually have fired --
            # otherwise the arm is inert and its closure difference is a null
            # for a plumbing reason, not a physical one.  Four inert controls
            # have already been mistaken for nulls in this study, so the
            # positive direction gets a hard check too, not just a printout.
            # `restored` is empty for the muon `elonly`/`inelonly` arms (it has
            # neither hadElastic nor an inelastic process), which is the
            # degenerate-by-construction null and correctly asserts nothing.
            # The criterion is `all`, not `primary`.  A restored process that
            # fires on NOTHING (all == 0) was not actually reactivated -- that
            # is broken plumbing and a hard failure.  A process that fires on
            # secondaries but not on the primary (primary == 0, all > 0) is
            # reactivated and simply does not apply to this species: `Decay`
            # for the stable antiproton is exactly that, and NOTES_HADRONS s8
            # already reports pbar `dec` acceptance as 100.0000, identical to
            # `off`.  That is a real physical null, so it is reported loudly
            # and not failed -- failing it would have rejected a correct arm.
            restored = [p for p in inact_of(pdg, "off")
                        if p not in set(want)]
            for nm in restored:
                got = tot.get(nm, (0, 0))
                if got[1] == 0:
                    flag = "*** NOT REACTIVATED ***"
                elif got[0] == 0:
                    flag = "inert on PRIMARY (secondaries only)"
                else:
                    flag = "OK live"
                print(f"        restored {nm:<24} primary {got[0]:>9}  "
                      f"all {got[1]:>9}   {flag}")
                if got[1] == 0:
                    raise SystemExit(
                        f"{nm} was restored in {tag_of(pdg,arm)} but defined "
                        f"ZERO steps anywhere -- it was not reactivated, so "
                        f"any closure difference from this arm is not physics")
            # what is still on, for contrast
            live = [(k, v) for k, v in sorted(tot.items()) if v[0] > 0]
            print(f"        primary steps by defining process, still ON: "
                  + ", ".join(f"{k} {v[0]}" for k, v in live))
            print(f"        ACCEPTANCE {100*acc:8.4f} %   "
                  f"({len(c)} events, {int((c[:, tp.I_RLAST] > 106.85).sum())} "
                  f"reach r > 106.85 cm)")


# =========================================================================
# 3.  models
# =========================================================================

def model_path(pdg, corr, kok=False):
    """corr in {'off','on'}; `kok` only ever affects the EXPORTED VARIANCE
    (Q / gsig2), never the reference trajectory -- and for a hadron the C++
    guard makes it a no-op.  Kept as a separate file so that is measurable."""
    lab = corr + ("_kok" if kok else "")
    return os.path.join(OUT, f"model_{SPECIES[pdg]['label'].replace('-','m')}"
                             f"_pt{PT:g}_{lab}.root")


def cmd_export(args):
    for pdg in args.pdg:
        sp = SPECIES[pdg]
        for corr in args.corr:
            for kok in ([False, True] if args.kok else [False]):
                out = model_path(pdg, corr, kok)
                log = out[:-5] + ".log"
                if os.path.exists(out) and not args.force:
                    print(f"[{sp['label']}/{corr}{'+kok' if kok else ''}] "
                          f"exists -> {out}")
                    continue
                ev = {}
                if corr == "on":
                    ev["CVH_IONI_EXACTDELTA"] = "1"
                if kok:
                    ev["CVH_IONI_KOKOULIN"] = "1"
                rc = _run(sp["geom"], "runToyModel.py",
                          f"pt={PT} eta={ETA} phi={PHI} partId={pdg} "
                          f"output={out}", log, _env(pdg, "off", ev))
                print(f"[{sp['label']}/{corr}{'+kok' if kok else ''}] rc={rc} "
                      f"-> {out}")
                if rc:
                    raise SystemExit(f"export failed, see {log}")


def cmd_kokid(args):
    """The C++ half of the Kokoulin switch, per species.

    `G4UniversalFluctuationForExtrapolator.cc` guards it with
    `kokoulinOn && ekin > 1 GeV && std::abs(particle->GetPDGEncoding()) == 13`,
    so exporting a HADRON with `CVH_IONI_KOKOULIN=1` must be BIT-IDENTICAL to
    exporting it without, while the MUON must move exactly the variance
    branches.  That is the C++ guard shown live rather than read off the
    source."""
    print("=" * 100)
    print("C++ Kokoulin guard: CVH_IONI_KOKOULIN=1 must be INERT for every "
          "hadron and must move Q/gsig2 for the muon")
    print("=" * 100)
    for pdg in args.pdg:
        sp = SPECIES[pdg]
        a, b = model_path(pdg, "on", False), model_path(pdg, "on", True)
        if not os.path.exists(b):
            print(f"  {sp['label']}: {b} missing (export --kok)")
            continue
        da, db = ds._branch_digests(a), ds._branch_digests(b)
        keys = sorted(set(da) | set(db))
        same = [k for k in keys if da.get(k) == db.get(k)]
        diff = [k for k in keys if da.get(k) != db.get(k)]
        verdict = ("BIT-IDENTICAL (guard is live)" if not diff
                   else f"moved: {', '.join(diff)}")
        print(f"  {sp['label']:<6} {len(keys)} branches, {len(same)} same, "
              f"{len(diff)} diff   {verdict}")


def _record(pdg, corr="on"):
    """The regime-2/3 ionization record of the ToyLayerMat steps, plus the
    kinematics the exact-delta channel needs."""
    legs = cpt.load_model(model_path(pdg, corr))
    rows = []
    for j, leg in enumerate(legs):
        for r in leg["ioni"]:
            reg, gsig2, a1, e1, a2, e2, a3, e0, tmx, g = r[:10]
            rows.append(dict(leg=j, regime=int(reg), xi=a3 * g, e0=e0 * g,
                             tmax=tmx * g,
                             beta2=(r[11] if len(r) > 11 else np.nan),
                             Etot=(r[12] if len(r) > 12 else np.nan),
                             mat=ds.material_of(np.sqrt(e2 / 1e-5))[0]))
    return legs, rows


# =========================================================================
# 4.  the parameter-free rate check, per species, with/without Kokoulin
# =========================================================================

def _NE(xi, T, Tmax, beta2, Etot, spinhalf, kokoulin=False, mass=None,
        nquad=20000):
    """(N(>T), E(>T)) for Geant4's own knock-on spectrum

        dN/dT = xi/T^2 [1 - b2 T/Tmax (+ T^2/2E^2 if spin 1/2)] * f_K(T)

    `ds.exact_N_E_spin` takes a `kokoulin` argument and IGNORES it (the body
    never reads it), and `ds.exact_N_E` hard-codes the spin-1/2 term AND the
    MUON mass in the Kokoulin `a3 = ln(4E(E-T)/M^2)`.  Neither is usable for a
    hadron, so both branches are done here with the species mass.
    """
    T, Tmax = float(T), float(Tmax)
    if T >= Tmax:
        return 0.0, 0.0
    if not kokoulin:
        L = math.log(Tmax / T)
        N = xi * ((1.0 / T - 1.0 / Tmax) - beta2 * L / Tmax)
        E = xi * (L - beta2 * (Tmax - T) / Tmax)
        if spinhalf:
            N += xi * (Tmax - T) / (2.0 * Etot ** 2)
            E += xi * (Tmax ** 2 - T ** 2) / (4.0 * Etot ** 2)
        return float(N), float(E)
    M = float(mass)
    x = np.linspace(math.log(T), math.log(Tmax), nquad)
    t = np.exp(x)
    core = (1.0 / t ** 2) * (1.0 - beta2 * t / Tmax
                             + (t * t / (2.0 * Etot ** 2) if spinhalf else 0.0))
    f = np.ones_like(t)
    m = t > ds.KOKOULIN_TMIN
    a1 = np.log(1.0 + 2.0 * t[m] / ds.MEL)
    a3 = np.log(4.0 * Etot * (Etot - t[m]) / (M * M))
    f[m] = 1.0 + ds.ALPHA_PRIME * a1 * (a3 - a1)
    core = core * f
    return (float(xi * np.trapezoid(core * t, x)),
            float(xi * np.trapezoid(core * t * t, x)))


TWOPI_MC2_RCL2 = 2.549549439e-23     # MeV mm^2, CLHEP's twopi_mc2_rcl2
NEL_TOY = 9.0 * 6.02214076e23 * 8.0 / 16.0 * 1e-3      # electrons / mm^3


def cmd_g4xs(args):
    """DIRECTLY: does GEANT4's OWN ionization model for this species carry the
    Kokoulin radiative factor?

    `urban_g4driver --composite` prints `lambda = model->CrossSectionPerVolume(
    mat, part, ekin, tcut, tmax) * len`, i.e. the hard-delta RATE Geant4 itself
    will Poisson-sample, taken from `G4MuBetheBlochModel` for |pdg| = 13 and
    `G4BetheBlochModel` otherwise -- the same branch `G4hIonisation` /
    `G4MuIonisation` make.  Comparing it to the tree-level and to the
    Kokoulin-weighted integral of the same spectrum settles the question
    without a simulation and without a fit.  (The driver has no anti_proton, so
    the proton stands in: the Kokoulin factor is charge-even.)"""
    drv = os.path.join(SCRATCH, "urban_g4driver.sh")
    if not os.path.exists(drv):
        raise SystemExit(f"{drv} missing -- run build_urban_g4driver.sh")
    print("=" * 112)
    print("GEANT4's OWN hard-delta RATE (CrossSectionPerVolume x len) against "
          "the tree-level and Kokoulin integrals")
    print("=" * 112)
    for pdg in args.pdg:
        sp = SPECIES[pdg]
        legs, rows = _record(pdg, "on")
        r0 = [r for r in rows if r["mat"] == "ToyLayerMat"][0]
        beta2, etot, tmax, xi = r0["beta2"], r0["Etot"], r0["tmax"], r0["xi"]
        ekin = etot - sp["mass"]
        length = xi * beta2 / (TWOPI_MC2_RCL2 * NEL_TOY)     # mm
        tcut = args.tcut
        pd = 13 if abs(pdg) == 13 else abs(pdg)
        pre = os.path.join(OUT, f"g4xs_{sp['label'].replace('-','m')}")
        cmd = (f"{drv} --Z 8 --A 16 --rho 9.0 --ekin {ekin:.10g} "
               f"--len {length:.10g} --tmax {tmax:.10g} --tcut {tcut:.10g} "
               f"--n 1 --composite --pdg {pd} --out {pre}")
        rc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
        txt = open(pre + ".rec").read() if os.path.exists(pre + ".rec") else ""
        m = re.search(r"composite tcut \S+ dedxrestricted (\S+) "
                      r"dedxunrestricted (\S+) meanrestricted (\S+) "
                      r"lambda (\S+) len (\S+)", txt)
        mt = re.search(r"tmaxcmp .* tmaxMuBB (\S+) tmaxBB (\S+)", txt)
        if not m:
            print(f"  {sp['label']}: driver failed rc={rc.returncode}\n"
                  f"{rc.stdout[-2000:]}\n{rc.stderr[-2000:]}")
            continue
        lam = float(m.group(4))
        n0, _ = _NE(xi, tcut, tmax, beta2, etot, r0["regime"] == 2,
                    kokoulin=False)
        n1, _ = _NE(xi, tcut, tmax, beta2, etot, r0["regime"] == 2,
                    kokoulin=True, mass=sp["mass"])
        print(f"\n### {sp['label']}  (driver pdg {pd}, "
              f"{'G4MuBetheBlochModel' if pd == 13 else 'G4BetheBlochModel'})"
              f"  ekin {ekin:.1f} MeV  len {length:.5f} mm  tcut "
              f"{tcut*1e3:.4f} keV")
        print(f"    Tmax: record {tmax:.5g}   MuBB {float(mt.group(1)):.5g}   "
              f"BB {float(mt.group(2)):.5g}")
        print(f"    N(>tcut)  GEANT4 {lam:.6f}   tree {n0:.6f} "
              f"(G4/tree {lam/n0:.6f})   Kokoulin {n1:.6f} "
              f"(G4/kok {lam/n1:.6f})")
        print(f"    Kokoulin weight on the rate = {n1/n0-1:+.5%};  "
              f"G4 - tree = {lam/n0-1:+.5%}")


def cmd_rate(args):
    """NOTES_DELTASPEC s2's no-free-parameter test, extended to ask WHICH
    SPECIES Geant4 applies the Kokoulin radiative correction to.  The
    prediction is evaluated twice -- tree level and Kokoulin-weighted -- and
    compared to Geant4's own explicit secondaries.  If Geant4 corrects a
    species, the tree-level prediction is LOW by the Kokoulin weight; if it
    does not, the Kokoulin-weighted one is HIGH by it.  Nothing is fitted."""
    print("=" * 112)
    print("PARAMETER-FREE DELTA-RAY PREDICTION vs GEANT4's OWN EXPLICIT "
          "SECONDARIES, and the Kokoulin question")
    print("=" * 112)
    print("Geant4 puts the Kokoulin radiative factor 1 + (alpha/2pi) a1 (a3-a1) "
          "in G4MuBetheBlochModel ONLY.")
    print("Hadrons go through G4hIonisation -> G4BetheBlochModel, which has no "
          "such factor.  Measured below.")
    for pdg in args.pdg:
        sp = SPECIES[pdg]
        legs, rows = _record(pdg, "on")
        # EVERY material contributes, not just the toy layer: the beam pipe
        # (Beryllium) and the Air gaps carry their own xi and their own cut,
        # and leaving them out cost 18.6 % in the first version of this.  Plus
        # `--extra 1`, i.e. one synthetic copy of the LAST LEG's ToyLayerMat
        # crossing, because the census integrates through 14 layers while the
        # model's legs stop at plane 13 (ds.extra_crossings documents this).
        tl = [r for r in rows if r["mat"] == "ToyLayerMat"]
        if not tl:
            raise SystemExit(f"no ToyLayerMat steps for {sp['label']}")
        last = max(r["leg"] for r in tl)
        rr = rows + [dict(r) for r in tl if r["leg"] == last]
        regs = sorted({r["regime"] for r in tl})
        r0 = tl[0]
        # the cuts, from the watcher's own printed G4ProductionCutsTable
        log = sorted(glob.glob(log_glob(pdg, args.arm)))[0]
        cuts = ds.cuts_from_log(log)
        c = census_of(pdg, args.arm)
        # hIoni for hadrons, muIoni for the muon -- both are summed, so the
        # same code serves both and a mislabelled species shows up as a zero
        nsec = (c[:, tp.I_NSECPROC + tp.PROCS.index("muIoni")]
                + c[:, tp.I_NSECPROC + tp.PROCS.index("otherIoni")])
        esec = (c[:, tp.I_ESECPROC + tp.PROCS.index("muIoni")]
                + c[:, tp.I_ESECPROC + tp.PROCS.index("otherIoni")])
        full = c[:, tp.I_RLAST] > 106.85
        n = int(full.sum())
        print(f"\n### {sp['label']}  regime {regs} "
              f"({'2 = spin 1/2' if 2 in regs else '3 = spin 0'}),  "
              f"m = {sp['mass']} MeV")
        print(f"    ToyLayerMat step: xi {r0['xi']:.6f} MeV   "
              f"Tmax {r0['tmax']:.4f} MeV   beta^2 {r0['beta2']:.6f}   "
              f"E {r0['Etot']:.2f} MeV   Tmax/E {r0['tmax']/r0['Etot']:.5f}")
        print(f"    e- production cut in ToyLayerMat = "
              f"{cuts['ToyLayerMat']*1e3:.4f} keV;  Kokoulin floor 100 keV")
        for kok in (False, True):
            Nx = Ex = 0.0
            for r in rr:
                Tc = max(cuts[r["mat"]], r["e0"])
                if r["tmax"] <= Tc:
                    continue
                nn, ee = _NE(r["xi"], Tc, r["tmax"], r["beta2"], r["Etot"],
                             r["regime"] == 2, kokoulin=kok,
                             mass=sp["mass"])
                Nx += nn
                Ex += ee
            dN = nsec[full].std() / np.sqrt(n)
            dE = esec[full].std() / np.sqrt(n)
            lab = "Kokoulin ON " if kok else "tree level  "
            print(f"    [{lab}] N_pred {Nx:9.4f}  N_sim {nsec[full].mean():9.4f}"
                  f" +- {dN:6.4f}   ratio {nsec[full].mean()/Nx:8.5f}"
                  f"  ({(nsec[full].mean()/Nx-1)*100:+6.3f} %)")
            print(f"    {'':<14} E_pred {Ex:9.4f}  E_sim {esec[full].mean():9.4f}"
                  f" +- {dE:6.4f}   ratio {esec[full].mean()/Ex:8.5f}"
                  f"  ({(esec[full].mean()/Ex-1)*100:+6.3f} %)")


# =========================================================================
# 5.  the closure
# =========================================================================

_SIM = {}


def sim_of(pdg, arm):
    key = (pdg, arm)
    if key not in _SIM:
        from toy_loader import load_toy_sim
        ns = {}
        exec(open(planes_path(SPECIES[pdg]["geom"])).read(), ns)
        _SIM[key] = load_toy_sim(sim_glob(pdg, arm), ns["origin"], ns["normal"],
                                 ns["uaxis"])
    return _SIM[key]


def kok_for(pdg):
    """The SPECIES-CORRECT Kokoulin state.  Geant4 puts the factor in
    G4MuBetheBlochModel only, so it is on for the muon and off for every
    hadron.  Measured, not assumed -- see `rate`."""
    return abs(pdg) == 13


def _rows(pdg, arm, corr="on", kok=None, func="qop", tcut=0.0):
    """One closure curve.

    `kok` is set EXPLICITLY (None = the species-correct value), never left to
    the environment default.  (When this was written
    `cf_track_resolution._kokoulin_exponent` guarded only on
    `etot - m_mu > 1 GeV` and the record carries no PDG code; the species
    guard was added 2026-08-16, NOTES_BARKAS s9.1, and driving it explicitly
    is now belt-and-braces rather than the only defence.)

    THREE CACHES, none of which keys on IONI_KOKOULIN or on the particle:
      * `fisher_norm._SCALE_CACHE`            (in-process s_F)
      * `fisher_norm._scale_cache_{load,store}` (ON-DISK s_F, keyed by the
        model file's content identity + a source fingerprint)
      * `cf_propagation_test._PHI_CACHE`      (`_phi_key` carries A3/EXC/TMAX
        scales but NOT IONI_KOKOULIN)
    """
    if kok is None:
        kok = kok_for(pdg)
    assert os.environ.get("RES_NO_PHI_CACHE"), "phi cache is LIVE"
    ctr.IONI_KOKOULIN = 1.0 if kok else 0.0
    ctr.IONI_KOKOULIN_TCUT = float(tcut) if kok else 0.0
    fn._SCALE_CACHE.clear()
    cpt._PHI_CACHE.clear()
    old_load, old_store = fn._scale_cache_load, fn._scale_cache_store
    fn._scale_cache_load = lambda *a, **k: None
    fn._scale_cache_store = lambda *a, **k: None
    try:
        mp = model_path(pdg, corr)
        legs = cpt.load_model(mp)
        sc = fn.plane_scales(legs, func, tag=mp, channels=("ioni", "ms", "rad"))
        rows, errs, ks, ns, err = gc.closure_rows(legs, sim_of(pdg, arm), func,
                                                  sc["sF"])
    finally:
        fn._scale_cache_load, fn._scale_cache_store = old_load, old_store
        ctr.IONI_KOKOULIN = 0.0
        ctr.IONI_KOKOULIN_TCUT = 0.0
        fn._SCALE_CACHE.clear()
        cpt._PHI_CACHE.clear()
    return dict(rows=rows, errs=errs, ks=ks, ns=ns, err=err, sc=sc,
                m=rows.mean(axis=0),
                # The OUTERMOST plane, surfaced alongside the ladder mean.
                # Every row is cumulative over legs 0..k, so rows[-1] is the
                # whole-track number the global fit integrates over, and the
                # 2.8x plane-correlation inflation that `err` carries applies
                # to the MEAN over planes and to nothing else -- hence the
                # ordinary per-plane errs[-1] here.  Same convention as
                # allcorr.cell, so the two drivers can be compared directly.
                out=rows[-1], outerr=errs[-1])


def _fmt(v):
    return "".join(f"{x:9.5f}" for x in v)


def _rms(v):
    return float(np.sqrt((np.asarray(v) ** 2).mean()))


UROW = "".join(f"{u:>9g}" for u in UCURVE)


def cmd_closure(args):
    print("=" * 124)
    print(f"CLEAN-PROPAGATION CLOSURE, {args.func}, plane mean, Fisher "
          f"normalization, layered toy pT = {PT} GeV, eta {ETA}, cut1e4")
    print("exact-delta " + args.corr.upper()
          + ";  Kokoulin ON for the muon and OFF for hadrons "
            "(Geant4 corrects G4MuBetheBlochModel only).  Nothing fitted.")
    print("=" * 124)
    print(f"  {'species':<8}{'arm':<5}{'kok':<5}" + UROW + "      rms")
    ref = {}
    for pdg in args.pdg:
        sp = SPECIES[pdg]
        for arm in args.arms:
            r = _rows(pdg, arm, corr=args.corr, func=args.func)
            m, e = r["m"], r["err"]
            print(f"  {sp['label']:<8}{arm:<5}{int(kok_for(pdg)):<5}"
                  + _fmt(m) + f" {_rms(m):9.5f}   n={r['ns'][0]}"
                  f" planes={len(r['ks'])}")
            print(f"  {'':<8}{'+-':<5}{'':<5}" + _fmt(e))
            if pdg == 13 and arm == args.arms[0]:
                ref = dict(lab=sp["label"], m=m, e=e)
            elif args.dref and ref:
                d = m - ref["m"]
                s = d / np.sqrt(e ** 2 + ref["e"] ** 2)
                print(f"  {'':<8}{'d':<5}{'':<5}" + _fmt(d)
                      + f"        ({sp['label']} - {ref['lab']})")
                print(f"  {'':<8}{'sig':<5}{'':<5}"
                      + "".join(f"{x:9.2f}" for x in s))


# =========================================================================
# 6.  the controls -- reproduce the published muon numbers before anything
#     new is believed
# =========================================================================

PUBLISHED = {
    # NOTES_RADOFF2 A3 / NOTES_SAMPLERGAP 5: pt3_cut1e4, radiation ON,
    # exact-delta ON, Kokoulin ON, qop, plane mean, 200k, Fisher normalization
    "pt3_cut1e4_kok": ([-0.00062, -0.00097, -0.00173, -0.00305, -0.00407,
                        -0.00280, -0.00140, -0.00081, -0.00046], 0.00213),
    # the same with Kokoulin OFF (NOTES_DELTASPEC 4.1)
    "pt3_cut1e4_nokok": ([-0.00118, -0.00213, -0.00437, -0.00819, -0.01076,
                          -0.00663, -0.00284, -0.00153, -0.00083], 0.00540),
    # exact-delta OFF as well (NOTES_RADOFF2 s3)
    "pt3_cut1e4_off": (None, 0.02058),
}


def _archived_sim():
    """NOTES_TAILHUNT's archived 200k pt3_cut1e4 sample -- the simulation every
    published pT = 3 exact-cut number in this family was measured on."""
    from toy_loader import load_toy_sim
    ns = {}
    exec(open(planes_path("qm")).read(), ns)
    return load_toy_sim(os.path.join(TH, "pt3_cut1e4_s*_sim.root"),
                        ns["origin"], ns["normal"], ns["uaxis"])


def cmd_control(args):
    """Two controls, in this order:

    (1) THE MODEL.  This module's own muon model, exported from `hadprobe/qm`,
        run against the ARCHIVED 200k pt3_cut1e4 simulation.  Same events as
        the published table, so any difference is the model alone and there is
        no statistical excuse.  Must reproduce NOTES_RADOFF2's row digit for
        digit at Kokoulin ON and NOTES_DELTASPEC's at Kokoulin OFF.
    (2) THE SIMULATION.  This module's own muon `on` arm (nothing inactivated)
        against the same model -- a NEW 200k sample through a NEW private area,
        so it can only agree within statistics."""
    print("=" * 124)
    print("CONTROL 1: this module's muon model vs the ARCHIVED 200k "
          "pt3_cut1e4 simulation (same events as the published table)")
    print("=" * 124)
    sim = _archived_sim()
    print(f"  {'config':<22}" + UROW + "      rms")
    for corr, kok, ref in (("on", True, "pt3_cut1e4_kok"),
                           ("on", False, "pt3_cut1e4_nokok"),
                           ("off", False, "pt3_cut1e4_off")):
        assert os.environ.get("RES_NO_PHI_CACHE")
        ctr.IONI_KOKOULIN = 1.0 if kok else 0.0
        fn._SCALE_CACHE.clear()
        cpt._PHI_CACHE.clear()
        ol, os_ = fn._scale_cache_load, fn._scale_cache_store
        fn._scale_cache_load = lambda *a, **k: None
        fn._scale_cache_store = lambda *a, **k: None
        try:
            mp = model_path(13, corr)
            legs = cpt.load_model(mp)
            sc = fn.plane_scales(legs, "qop", tag=mp,
                                 channels=("ioni", "ms", "rad"))
            rows, errs, ks, ns, err = gc.closure_rows(legs, sim, "qop",
                                                      sc["sF"])
        finally:
            fn._scale_cache_load, fn._scale_cache_store = ol, os_
            ctr.IONI_KOKOULIN = 0.0
            fn._SCALE_CACHE.clear()
            cpt._PHI_CACHE.clear()
        m = rows.mean(axis=0)
        lab = f"corr {corr}, kok {int(kok)}"
        print(f"  {lab:<22}" + _fmt(m) + f" {_rms(m):9.5f}   "
              f"n={ns[0]} planes={len(ks)}  s_F[0]={sc['sF'][0]:.8e}")
        pub, prms = PUBLISHED[ref]
        if pub is not None:
            print(f"  {'  published':<22}" + _fmt(pub) + f" {prms:9.5f}")
            d = m - np.array(pub)
            print(f"  {'  difference':<22}" + _fmt(d))
        else:
            print(f"  {'  published rms':<22}" + " " * (9 * len(UCURVE))
                  + f" {prms:9.5f}")

    print()
    print("=" * 124)
    print("CONTROL 2: this module's own muon `on` arm (nothing inactivated) "
          "-- a NEW 200k sample, agrees only within statistics")
    print("=" * 124)
    for arm in ("on", "dec", "off"):
        try:
            r = _rows(13, arm, corr="on")
        except FileNotFoundError:
            print(f"  [{arm}] not produced")
            continue
        m, e = r["m"], r["err"]
        pub = np.array(PUBLISHED["pt3_cut1e4_kok"][0])
        print(f"  mu- {arm:<5}" + _fmt(m) + f" {_rms(m):9.5f}   n={r['ns'][0]}")
        print(f"  {'+-':<9}" + _fmt(e))
        print(f"  {'sigma':<9}" + "".join(f"{x:9.2f}" for x in (m - pub) / e)
              + "        (vs the published archived-sample row)")


def cmd_bias(args):
    """Split the non-closure into a MEAN-LOSS bias and a SHAPE mismatch.

    The model's reference trajectory subtracts a mean loss taken from the
    propagator's own dE/dx table.  If that table's mean is wrong by dE, every
    z is displaced by a constant and the closure moves without the FLUCTUATION
    model being wrong at all.  Re-running the closure against a reference
    shifted by the measured per-plane mean residual removes exactly that and
    nothing else, so the two contributions can be quoted separately.

    This is a diagnostic, not a correction: the shift is measured FROM the same
    data, so the shifted rms is a lower bound on what a correct mean would give
    and must never be quoted as a closure."""
    print("=" * 124)
    print("MEAN-LOSS BIAS vs SHAPE MISMATCH.  `shifted` re-centres the "
          "reference on the data's own per-plane mean --")
    print("a DIAGNOSTIC (one parameter per plane taken from the data), not a "
          "closure.")
    print("=" * 124)
    print(f"  {'species':<8}{'q':>3}{'<dE> [MeV]':>12}{'dE bias':>10}"
          f"{'std(z)':>10}{'skew':>8}   {'as exported':>12}{'re-centred':>12}")
    for pdg in args.pdg:
        sp = SPECIES[pdg]
        c = census_of(pdg, args.arm)
        sim = sim_of(pdg, args.arm)
        legs = cpt.load_model(model_path(pdg, "on"))
        k = min(sim["valid"].shape[1], len(legs)) - 1
        good = sim["valid"][:, k] & np.isfinite(sim["qop"][:, k])
        d = sim["qop"][good, k] - legs[k]["refqop"]
        q = float(np.sign(legs[k]["refqop"]))
        P = 1e3 / abs(legs[k]["refqop"])                     # MeV
        E = math.sqrt(P * P + sp["mass"] ** 2)
        dE = float(d.mean()) * P ** 3 / (q * E) / 1e3        # MeV
        base = _rows(pdg, args.arm, corr="on")
        # shifted: reference re-centred plane by plane
        legs2 = [dict(l) for l in cpt.load_model(model_path(pdg, "on"))]
        for j in range(len(legs2)):
            g = sim["valid"][:, j] & np.isfinite(sim["qop"][:, j])
            if g.sum() < 100:
                continue
            legs2[j]["refqop"] = float(sim["qop"][g, j].mean())
        assert os.environ.get("RES_NO_PHI_CACHE")
        ctr.IONI_KOKOULIN = 1.0 if kok_for(pdg) else 0.0
        fn._SCALE_CACHE.clear()
        cpt._PHI_CACHE.clear()
        try:
            r2 = gc.closure_rows(legs2, sim, "qop", base["sc"]["sF"])[0]
        finally:
            ctr.IONI_KOKOULIN = 0.0
            fn._SCALE_CACHE.clear()
            cpt._PHI_CACHE.clear()
        print(f"  {sp['label']:<8}{int(q):>3}{c[:, tp.I_DETOT].mean():12.4f}"
              f"{dE:10.4f}{d.std():10.3e}"
              f"{float(((d-d.mean())**3).mean()/d.std()**3):8.2f}   "
              f"{_rms(base['m']):12.5f}{_rms(r2.mean(axis=0)):12.5f}")


def cmd_radgap(args):
    """A SECOND unmodelled channel, found while setting this up and not
    anticipated by the brief: for a hadron the exported model has NO radiative
    block at all.

    `G4ePropagationExport`'s `radv` record carries the propagator's own
    `dedxrad / dedxbrem / dedxpair`, and for pi-, K-, p and pbar every one of
    them is EXACTLY ZERO -- so the model's radiative CF is a structural zero
    (verified: phi is bit-identical with `RAD_CHANNEL` on and off) and the
    reference trajectory subtracts no radiative mean.  The SIMULATION however
    runs `hBrems` and `hPairProd`, suppressed relative to the muon only by
    (m_mu/m)^2.  So for a hadron the radiative channel is missing from BOTH the
    mean and the fluctuation, and its size has to be quoted next to the nuclear
    one.  Measured here from the primary-loss census."""
    print("=" * 112)
    print("THE MISSING RADIATIVE CHANNEL.  Model: dedxrad = 0 for every "
          "hadron (structural zero).  Simulation: hBrems + hPairProd run.")
    print("=" * 112)
    IR = tp.I_DEPROC
    ib, ip, io = (tp.PROCS.index("muBrems"), tp.PROCS.index("muPairProd"),
                  tp.PROCS.index("other"))
    inm, ioi = tp.PROCS.index("muIoni"), tp.PROCS.index("otherIoni")
    print(f"  {'species':<8}{'(m_mu/m)^2':>12}{'<dE_rad> [MeV]':>16}"
          f"{'/ muon':>10}{'<dE_ioni>':>12}{'rad/ioni':>11}")
    mu0 = None
    for pdg in args.pdg:
        sp = SPECIES[pdg]
        c = census_of(pdg, args.arm)
        if abs(pdg) == 13:
            rad = c[:, IR + ib].mean() + c[:, IR + ip].mean()
        else:
            rad = c[:, IR + io].mean()
        ion = c[:, IR + inm].mean() + c[:, IR + ioi].mean()
        if mu0 is None:
            mu0 = rad
        print(f"  {sp['label']:<8}{(105.6583745/sp['mass'])**2:12.5f}"
              f"{rad:16.6f}{rad/mu0:10.4f}{ion:12.4f}{rad/ion:11.2e}")
    print("\n  `other` for a hadron in the `off` arm is hBrems + hPairProd + "
          "the at-rest capture, which\n  happens beyond the scoring volume "
          "(the census watcher is capped at r = 107 cm).")


def cmd_nuc(args):
    """Nuclear elastic + inelastic and Decay switched back ON: the size of the
    channel the model does not have.  Reported as the CHANGE in the closure and
    as the acceptance loss, because for a hadron the two are not separable --
    an inelastic interaction removes the track."""
    print("=" * 124)
    print("THE UNMODELLED CHANNEL: nuclear (elastic + inelastic) and Decay "
          "switched back ON")
    print("=" * 124)
    print(f"  {'species':<8}{'arm':<6}{'stat':<5}{'acc %':>9}" + UROW
          + "      rms")
    for pdg in args.pdg:
        sp = SPECIES[pdg]
        base = {}
        for arm in args.arms:
            try:
                r = _rows(pdg, arm, corr=args.corr, func=args.func)
                c, acc = _acc(pdg, arm)
            except FileNotFoundError:
                continue
            # BOTH statistics, every time.  The ladder alone is what this
            # driver used to print, and it is the one that improves while the
            # outermost plane degrades -- quoting it on its own is how the MS
            # commit came to mis-state its own effect (NOTES_CLOSURE_ALLCORR).
            for stat, (m, e) in (("lad", (r["m"], r["err"])),
                                 ("out", (r["out"], r["outerr"]))):
                head = f"  {sp['label']:<8}{arm:<6}" if stat == "lad" \
                    else f"  {'':<8}{'':<6}"
                accs = f"{100*acc:9.4f}" if stat == "lad" else f"{'':>9}"
                print(head + f"{stat:<5}" + accs + _fmt(m)
                      + f" {_rms(m):9.5f}")
                print(f"  {'':<8}{'':<6}{'+-':<5}{'':>9}" + _fmt(e))
                if arm == "off":
                    base[stat] = (m, e)
                elif stat in base:
                    d = m - base[stat][0]
                    s = d / np.sqrt(e ** 2 + base[stat][1] ** 2)
                    print(f"  {'':<8}{'':<6}{'d':<5}{'':>9}" + _fmt(d)
                          + f" {_rms(d):9.5f}   ({arm} - off, {stat})")
                    print(f"  {'':<8}{'':<6}{'sig':<5}{'':>9}"
                          + "".join(f"{x:9.2f}" for x in s))


def cmd_corr(args):
    """The per-species effect of each correction, separately.  Tmax is 61x
    smaller for a 3 GeV proton than for a 3 GeV muon, so the exact-delta
    correction -- which acts near Tmax -- has much less room to act; a small
    effect for the heavy species is a physically sensible result, not a
    failure."""
    print("=" * 124)
    print("PER-SPECIES EFFECT OF EACH CORRECTION, qop, plane mean, arm `off`")
    print("=" * 124)
    print(f"  {'species':<8}{'Tmax':>9}{'Tmax/E':>9}{'xi':>10}  "
          f"{'config':<22}" + "      rms   (rms removed)")
    for pdg in args.pdg:
        sp = SPECIES[pdg]
        legs, rows = _record(pdg, "on")
        r0 = [r for r in rows if r["mat"] == "ToyLayerMat"][0]
        got = {}
        for lab, corr, kok in (("exact-delta OFF", "off", False),
                               ("exact-delta ON, kok OFF", "on", False),
                               ("exact-delta ON, kok ON", "on", True)):
            try:
                r = _rows(pdg, args.arm, corr=corr, kok=kok)
            except FileNotFoundError:
                continue
            got[lab] = r
            head = (f"  {sp['label']:<8}{r0['tmax']:9.3f}"
                    f"{r0['tmax']/r0['Etot']:9.5f}{r0['xi']:10.6f}  "
                    if not got or len(got) == 1 else "  " + " " * 36)
            base = got.get("exact-delta OFF")
            rem = ("" if base is None or lab == "exact-delta OFF" else
                   f"   ({100*(1-_rms(r['m'])/_rms(base['m'])):+6.1f} % of "
                   f"the uncorrected rms)")
            print(head + f"{lab:<24}" + _fmt(r["m"])
                  + f" {_rms(r['m']):9.5f}{rem}")
        if "exact-delta ON, kok OFF" in got and "exact-delta ON, kok ON" in got:
            a = got["exact-delta ON, kok OFF"]["m"]
            b = got["exact-delta ON, kok ON"]["m"]
            e = got["exact-delta ON, kok OFF"]["err"]
            print("  " + " " * 36 + f"{'KOKOULIN moves it by':<24}"
                  + _fmt(b - a) + f" {_rms(b-a):9.5f}")
            print("  " + " " * 36 + f"{'  in sigma':<24}"
                  + "".join(f"{x:9.2f}" for x in (b - a) / e))


def main():
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)

    q = s.add_parser("setup")
    q.set_defaults(fn=cmd_setup)

    q = s.add_parser("sim")
    q.add_argument("--pdg", type=int, nargs="+", default=ORDER)
    q.add_argument("--arms", nargs="+", default=["off"])
    q.add_argument("--events", type=int, default=20000)
    q.add_argument("--jobs", type=int, default=10)
    q.add_argument("--seed0", type=int, default=101)
    q.add_argument("--workers", type=int, default=20)
    q.add_argument("--force", action="store_true")
    q.set_defaults(fn=cmd_sim)

    q = s.add_parser("live")
    q.add_argument("--pdg", type=int, nargs="+", default=ORDER)
    q.add_argument("--arms", nargs="+", default=["off"])
    q.set_defaults(fn=cmd_live)

    q = s.add_parser("export")
    q.add_argument("--pdg", type=int, nargs="+", default=ORDER)
    q.add_argument("--corr", nargs="+", default=["on", "off"])
    q.add_argument("--kok", action="store_true")
    q.add_argument("--force", action="store_true")
    q.set_defaults(fn=cmd_export)

    q = s.add_parser("rate")
    q.add_argument("--pdg", type=int, nargs="+", default=ORDER)
    q.add_argument("--arm", default="off")
    q.set_defaults(fn=cmd_rate)

    q = s.add_parser("closure")
    q.add_argument("--pdg", type=int, nargs="+", default=ORDER)
    q.add_argument("--arms", nargs="+", default=["off"])
    q.add_argument("--corr", default="on")
    q.add_argument("--func", default="qop")
    q.add_argument("--dref", action="store_true")
    q.set_defaults(fn=cmd_closure)

    q = s.add_parser("kokid")
    q.add_argument("--pdg", type=int, nargs="+", default=ORDER)
    q.set_defaults(fn=cmd_kokid)

    q = s.add_parser("g4xs")
    q.add_argument("--pdg", type=int, nargs="+",
                   default=[13, -211, -321, 2212])
    q.add_argument("--tcut", type=float, default=0.0094862)
    q.set_defaults(fn=cmd_g4xs)

    q = s.add_parser("control")
    q.set_defaults(fn=cmd_control)

    q = s.add_parser("bias")
    q.add_argument("--pdg", type=int, nargs="+", default=ORDER)
    q.add_argument("--arm", default="off")
    q.set_defaults(fn=cmd_bias)

    q = s.add_parser("radgap")
    q.add_argument("--pdg", type=int, nargs="+", default=ORDER)
    q.add_argument("--arm", default="off")
    q.set_defaults(fn=cmd_radgap)

    q = s.add_parser("nuc")
    q.add_argument("--pdg", type=int, nargs="+", default=ORDER)
    q.add_argument("--arms", nargs="+", default=["off", "dec", "on"])
    q.add_argument("--corr", default="on")
    q.add_argument("--func", default="qop")
    q.set_defaults(fn=cmd_nuc)

    q = s.add_parser("corr")
    q.add_argument("--pdg", type=int, nargs="+", default=ORDER)
    q.add_argument("--arm", default="off")
    q.set_defaults(fn=cmd_corr)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
