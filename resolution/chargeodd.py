#!/usr/bin/env python3
"""The charge-odd dE/dx diagnostic of NOTES_BARKAS, turned into a real switch.

WHAT IS BEING FIXED
-------------------
`G4TablesForExtrapolatorForCVH::Initialisation` builds its dE/dx tables with

    ComputeMuonDEDX(muonPlus,  dedxMuon);
    ComputeProtonDEDX(proton,  dedxProton);

`muonMinus` is a member of that class, is assigned in its constructor and is
never used; there is no antiproton at all.  `G4EnergyLossForExtrapolatorForCVH`
then hands the muon table to BOTH signs and scales the proton table by `q*q`,
which is even by construction.  So the CVH reference trajectory is not
charge-blind at the charge AVERAGE -- it is pinned to the POSITIVE particle,
and every negative track carries the whole of the error.

What it drops is the charge-odd half of Geant4's high-order stopping-power
block, `G4EmCorrections::HighOrderCorrections = 2*(Barkas + Bloch) + Mott`:
Bloch is a function of `q^2` and cannot contribute, Barkas (Ashley-Ritchie,
~1.29 z, FALLS with beta) supplies 0.65-0.75 % and Mott (Ahlen's z^3,
pi*alpha*beta*z, GROWS with beta) supplies 99.3 % at the toy's kinematics.
Size: 0.309-0.314 % of the restricted dE/dx, i.e. dp/p = 7.0-7.7e-6 one-sided
and 1.4-1.5e-5 between charge conjugates -- at the 1e-5 Z-mass target.

THE FIX, AND WHY IT IS THIS ONE
-------------------------------
`CVH_REF_CHARGEAWARE` (single C++ reader `cvhcgf::referenceIsChargeAware`)
builds the negative partners of the three muon and three hadron tables with the
SAME `ComputeMuonDEDX` / `ComputeProtonDEDX`, passing `G4MuonMinus` and
`G4AntiProton`; `ComputeDEDX`, `ComputeRange` and `ComputeEnergy` then choose
between them on the sign of the track's own `G4ParticleDefinition` charge.

Nothing is transcribed by hand.  The alternative -- keep one table and add
`pref*(2*Barkas + Mott)` with the sign of q -- was rejected: it hard-codes
which terms of Geant4's expansion are odd (so it silently goes wrong if Geant4
adds one), it has to be right at every beta by itself (Barkas and Mott exchange
dominance below beta*gamma ~ 1, i.e. exactly in the V0 regime), and it is more
code than calling the existing function with the other argument.

ALL THREE of ComputeDEDX / ComputeRange / ComputeEnergy switch together:
`EnergyAfterStep` picks between `step*ComputeDEDX` and `ComputeEnergy(range -
step)` on `linLossLimit`, so a partial fix makes the two branches disagree by
exactly this term.

WHY THIS IS A DIFFERENT ANIMAL FROM THE PREVIOUS TWO SWITCHES
-------------------------------------------------------------
`CVH_IONI_EXACTDELTA` and `CVH_IONI_KOKOULIN` change the FLUCTUATION and
preserve the mean by construction (NOTES_DELTASPEC 3.2 pins the block's mean to
the dE/dx table value).  This one changes the MEAN -- the reference trajectory
itself, i.e. the closure's origin and the linearization point of every
transport Jacobian.  NOTES_QVALID found that a *fluctuation* change already
moves the exported Jacobians coherently at 1e-5..5e-5 and invalidates existing
grads productions; a mean change is at least as consequential, so the same
measurement is repeated here against the same floor-and-determinism controls.

THE FREE REGRESSION TEST
------------------------
The correction is identically zero at q = +1.  So mu+, pi+, K+ and p must be
BIT-IDENTICAL with the switch on, at every level -- the exported model, the
closure, and the per-track CVH output.  Any movement of a positive is a bug,
not an improvement.

SUBCOMMANDS
    table     the extrapolator's dE/dx TABLE, C++ level, with and without the
              switch, against Geant4's own answer for each particle
    export    toy models: `caoff` (fresh, switch off) and `ca` (switch on)
    bitid     branch digests -- OFF bit-identity, positives unchanged,
              negatives LIVE
    refdiff   the C++ reference shift per species against the parameter-free
              offline prediction (Geant4's own Barkas + Mott at the exported
              step kinematics) -- the cross-check of NOTES_BARKAS s6.3
    refprof   the same shift plane by plane; it must ACCUMULATE
    closure   THE RESULT: per-species closure with the C++ charge-aware
              reference, positives shown unchanged
    registry  the offline cache registries must SEE this knob
    cvh       real-track CVH refit, with the floor and determinism controls
    cvhcmp    downstream of the reference, split by track charge
"""

import argparse
import hashlib
import math
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# hadron_probe sets RES_NO_PHI_CACHE=1 at import, before anything can populate
# the phi cache; barkas_probe brings the Geant4 driver used by `refdiff`.
import hadron_probe as hp                                        # noqa: E402
import barkas_probe as bp                                        # noqa: E402
import cf_propagation_test as cpt                                # noqa: E402
import deltaspec as ds                                           # noqa: E402
import qvalid as qv                                              # noqa: E402
import cf_track_resolution as ctr                                # noqa: E402

SCRATCH = hp.SCRATCH
OUT = hp.OUT                       # shares NOTES_HADRONS'/NOTES_BARKAS' dir
CO = os.path.join(SCRATCH, "co")   # the real-track CVH area for this note

ORDER8 = bp.ORDER8
PAIRS = bp.PAIRS
NEG = (13, -211, -321, -2212)
POS = (-13, 211, 321, 2212)

# The environment overlay for a toy MODEL export, per arm.  `caoff` is the
# control: a fresh export with the switch unset, which must reproduce the
# ARCHIVED model bit for bit.
ARM_ENV = {
    "caoff": {},
    "ca":    {"CVH_REF_CHARGEAWARE": "1"},
}
ARM_SUFFIX = {"caoff": "_caoff", "ca": "_ca"}

# chain OUR overlay on top of barkas_probe's (which is inert unless its shim
# arms are selected -- we never select them, so no LD_PRELOAD is ever set)
_bp_env = hp._env
CO_ENV = {}


def _env(pdg, arm, extra=None):
    e = _bp_env(pdg, arm, extra)
    e.update(CO_ENV)
    return e


hp._env = _env


# THE PUBLISHED `off` SAMPLE IS SEEDS 101-110, AND ONE SPECIES HAS AN ELEVENTH
# FILE IN THE SAME GLOB.
#
# NOTES_BARKAS s3.4 ran a 20 000-event, seed-901 mu- job four times to show the
# LD_PRELOAD shim is inert inside `cmsRun` (the md5-identical census files).
# Those jobs used arm `off` and therefore wrote
# `mum_pt3_off_s901_sim.root` into the same directory the closure globs, so
# `hadron_probe.sim_glob(13, "off")` now returns ELEVEN files -- 220 000 events
# for mu- against 200 000 for every other species.  It is the same physics and
# the same arm, so nothing is wrong with including it; but it is a DIFFERENT
# SAMPLE from the one NOTES_HADRONS s4 and NOTES_BARKAS s9.3 published, and a
# control that does not reproduce the published digits is not a control.
#
# Restricting the glob to the three-digit 1xx seeds restores the published
# sample exactly.  Verified below: with this in place every `nominal` row
# reproduces NOTES_BARKAS s9.3 digit for digit at all nine probes, including
# both s_F values.
_orig_sim_glob = hp.sim_glob


def sim_glob(pdg, arm):
    return os.path.join(OUT, f"{hp.tag_of(pdg, arm)}_s1??_sim.root")


hp.sim_glob = sim_glob


# =========================================================================
# 1.  export
# =========================================================================

def cmd_export(args):
    global CO_ENV
    for arm in args.arms:
        bp.MODEL_SUFFIX = ARM_SUFFIX[arm]
        CO_ENV = dict(ARM_ENV[arm])
        print(f"\n=== toy models, arm `{arm}`  suffix={bp.MODEL_SUFFIX!r}  "
              f"env={CO_ENV}")
        ns = argparse.Namespace(pdg=args.pdg, corr=["on"], kok=False,
                                force=args.force)
        hp.cmd_export(ns)
    bp.MODEL_SUFFIX, CO_ENV = "", {}


def _mp(pdg, arm=None):
    """model path for an arm ('' / None = the ARCHIVED nominal model)."""
    p = bp._orig_model_path(pdg, "on")
    return p if not arm else p[:-5] + ARM_SUFFIX[arm] + ".root"


# =========================================================================
# 2.  bit-identity, and liveness
# =========================================================================

def _cmp(a, b):
    """branch digests of two model files.  `deltaspec._branch_digests` is the
    comparator every switch in this study family has been proved with: sha256
    over every branch of every TTree, recursively, hard failure if no TTree is
    found.  Both guards are kept because both caught real bugs."""
    da, db = ds._branch_digests(a), ds._branch_digests(b)
    keys = sorted(set(da) | set(db))
    diff = [k for k in keys if da.get(k) != db.get(k)]
    return len(keys), diff


def cmd_bitid(args):
    print("=" * 118)
    print("BIT-IDENTITY OF THE OFF PATH, AND LIVENESS OF THE ON PATH")
    print("The branch COUNT is printed on every line and a file with no TTree "
          "is a hard failure (deltaspec._branch_digests).")
    print("=" * 118)

    print("\n--- control 1: ARCHIVED model vs a FRESH export with the switch "
          "OFF.")
    print("    The archived files predate this change (2026-08-15).  Anything "
          "but 0 different means the")
    print("    switch leaks when unset, or something else moved the model in "
          "the meantime.")
    print(f"  {'species':<8}{'branches':>10}{'identical':>11}"
          f"{'different':>11}   verdict")
    nbad = 0
    for pdg in args.pdg:
        a, b = _mp(pdg), _mp(pdg, "caoff")
        if not os.path.exists(b):
            print(f"  {hp.SPECIES[pdg]['label']:<8} (missing {b})")
            continue
        n, diff = _cmp(a, b)
        nbad += len(diff)
        print(f"  {hp.SPECIES[pdg]['label']:<8}{n:>10}{n - len(diff):>11}"
              f"{len(diff):>11}   "
              + ("BIT-IDENTICAL" if not diff else "DIFFERS: " + ",".join(diff)))
    print(f"    -> {'PASS' if nbad == 0 else 'FAIL'}: "
          f"{nbad} differing branches in total")

    print("\n--- control 2: the correction is IDENTICALLY ZERO at q = +1.")
    print("    (q - 1) = 0 for a positive particle, so a POSITIVE species must "
          "be bit-identical with the")
    print("    switch ON.  This is the cheapest regression test available and "
          "it is a test the fix can fail.")
    print("\n--- and the LIVE row: a NEGATIVE species must move.")
    print(f"  {'species':<8}{'q':>3}{'branches':>10}{'identical':>11}"
          f"{'different':>11}   verdict")
    bad = []
    for pdg in args.pdg:
        a, b = _mp(pdg), _mp(pdg, "ca")
        if not os.path.exists(b):
            print(f"  {hp.SPECIES[pdg]['label']:<8} (missing {b})")
            continue
        q = hp.SPECIES[pdg]["q"]
        n, diff = _cmp(a, b)
        if q > 0 and diff:
            ok, bad = "FAIL -- a positive moved: " + ",".join(diff[:6]), bad + [pdg]
        elif q > 0:
            ok = "BIT-IDENTICAL (as required)"
        elif not diff:
            ok, bad = "FAIL -- the switch is DEAD for a negative", bad + [pdg]
        else:
            ok = "LIVE: " + ",".join(sorted(diff))
        print(f"  {hp.SPECIES[pdg]['label']:<8}{q:>3}{n:>10}{n - len(diff):>11}"
              f"{len(diff):>11}   {ok}")
    print(f"    -> {'PASS' if not bad else 'FAIL'}")


# =========================================================================
# 3.  the reference shift, against the parameter-free offline prediction
# =========================================================================

def _refE(pdg, arm, k=None):
    """total energy at plane k (default: the outermost) of a model's reference
    trajectory, MeV."""
    legs = cpt.load_model(_mp(pdg, arm))
    k = len(legs) - 1 if k is None else k
    P = 1e3 / abs(legs[k]["refqop"])
    return math.sqrt(P * P + hp.SPECIES[pdg]["mass"] ** 2), legs


def cmd_refdiff(args):
    """Is the C++ shift the SAME number the offline diagnostic predicted?

    `NOTES_BARKAS s6` shifted the reference OFFLINE by
    `(q-1) * sum_s xi_s * (2*Barkas + Mott)_s`, every factor taken from
    Geant4's own G4EmCorrections at the exported step kinematics.  The C++
    switch instead rebuilds the whole table from `G4MuonMinus` /
    `G4AntiProton` and re-runs the entire Geant4e propagation -- table build,
    range table, inverse range, stepping -- none of which the offline formula
    models.  Agreement is therefore a real cross-check and not a tautology.

    It is also the SAME cross-check `barkas_probe refxcheck` made through the
    LD_PRELOAD `nohoc` export, from the other direction (that one REMOVED the
    block from the positive table; this one ADDS the odd part for the
    negative), so the two agree only if both are right."""
    print("=" * 124)
    print("THE C++ CHARGE-AWARE REFERENCE AGAINST THE PARAMETER-FREE OFFLINE "
          "PREDICTION")
    print("dE = E(charge-aware) - E(nominal) at the outermost plane.  "
          "`offline` = -(q-1)*sum_s xi_s*(2*Barkas+Mott)_s,")
    print("i.e. the ENERGY the charge-aware reference no longer subtracts, "
          "with Barkas and Mott from Geant4")
    print("at the SAME step kinematics.  Both MeV.  Nothing is fitted on "
          "either side.")
    print("=" * 124)
    print(f"  {'species':<8}{'q':>3}{'C++ (tables)':>16}{'offline':>14}"
          f"{'difference':>14}{'relative':>12}{'dp/p':>12}")
    for pdg in args.pdg:
        sp = hp.SPECIES[pdg]
        if not os.path.exists(_mp(pdg, "ca")):
            print(f"  {sp['label']:<8} (missing {_mp(pdg, 'ca')})")
            continue
        E0, legs0 = _refE(pdg, None)
        E1, _ = _refE(pdg, "ca")
        dE = E1 - E0
        bp.MODEL_SUFFIX = ""             # the offline prediction uses the
        _, cum = bp._hocodd_per_leg(pdg)  # NOMINAL model's own steps
        # `cum` is the change in the accumulated LOSS; the plane ENERGY moves
        # the other way.
        off = -cum[-1]
        P = 1e3 / abs(legs0[len(legs0) - 1]["refqop"])
        rel = (dE - off) / off if off != 0. else float("nan")
        dpp = abs(dE) * math.sqrt(P * P + sp["mass"] ** 2) / P / P
        print(f"  {sp['label']:<8}{sp['q']:>3}{dE:16.6f}{off:14.6f}"
              f"{dE - off:14.6f}{rel:12.2e}{dpp:12.3e}")


MEANLOSS_DRIVER = os.path.join(SCRATCH, "meanloss_g4driver.sh")


def _meanloss_run(env):
    """One meanloss_g4driver run -> {label: (dedxRef, dedxFluct)}, MeV/mm.

    Same env convention as everything else here: the four default-on
    corrections are pinned to their HISTORICAL state and `env` wins."""
    e = dict(os.environ)
    e.update(ctr.SWITCHES_OFF)
    e.update(env)
    r = subprocess.run([MEANLOSS_DRIVER], capture_output=True, text=True, env=e)
    if r.returncode:
        sys.stderr.write(r.stderr[-2000:])
        raise SystemExit(f"meanloss driver failed rc={r.returncode} "
                         f"(build it with ./build_meanloss.sh)")
    out = {}
    for line in r.stdout.splitlines():
        f = line.split()
        if f and f[0] == "MEANLOSS":
            out[f[2]] = (float(f[4]), float(f[5]))
    if len(out) != 8:
        raise SystemExit(f"meanloss driver produced {len(out)} species, want 8")
    return out


def cmd_meanloss(args):
    """DOES THE NOISE MODEL'S MEAN AGREE WITH THE REFERENCE'S MEAN?

    NOTES_SPECIESDEDX s2.1/s8 recorded, out of its own scope, that
    `CVH_REF_CHARGEAWARE` moved the REFERENCE trajectory's dE/dx and NOT
    `G4UniversalFluctuationForExtrapolator`'s `meanLoss`, because
    `SetParticleAndCharge` selected `fDedxMuon` / `fDedxProton` with no
    `isNegative` dispatch.  With the switch on, the two therefore disagreed by
    exactly the charge-odd part of Geant4's high-order block on every negative
    track.  Closed 2026-08-16 (NOTES_DEFAULTON s1); this is the measurement.

    WHAT IS AND IS NOT EXPECTED TO MATCH.  The two classes hold SEPARATE table
    objects on different grids -- the extrapolator's (nbins, 1 MeV, 100 TeV)
    carries the radiative mean, the fluctuation's (70, 1 MeV, 10 TeV) is always
    ionization-only -- so their ABSOLUTE values differ by ~1e-3 for a muon and
    always have.  That column is printed so it can be seen to be UNCHANGED by
    the fix.  What must match is the CHARGE-ODD PART, `dedx(+) - dedx(-)`,
    which is a property of the tables' physics and not of their grids.
    """
    print("=" * 122)
    print("THE REFERENCE'S MEAN vs THE NOISE MODEL'S `meanLoss`, C++ level, "
          "MeV/mm, toy material, p = 3136 MeV/c")
    print("`odd` is dedx(positive) - dedx(negative) in the same column.  It is "
          "0 with the switch off, and with it")
    print("on it must be the SAME in both columns -- a noise model centred on "
          "a different mean than the trajectory")
    print("it is the noise of is a new inconsistency, not a small one.  "
          "`ref-fluct` is the pre-existing grid /")
    print("radiative-mean offset between the two table sets and must not move.")
    print("=" * 122)
    arms = [("CA off", {"CVH_REF_CHARGEAWARE": "0"}),
            ("CA on ", {"CVH_REF_CHARGEAWARE": "1"})]
    res = {lab: _meanloss_run(ev) for lab, ev in arms}
    print(f"  {'arm':<8}{'pair':<10}{'odd(ref)':>22}{'odd(fluct)':>22}"
          f"{'odd ref-fluct':>16}{'ref-fluct(+)':>15}{'ref-fluct(-)':>15}")
    worst = 0.0
    for lab, _ in arms:
        r = res[lab]
        for m, p, name in PAIRS:
            pl, ml = hp.SPECIES[p]['label'], hp.SPECIES[m]['label']
            rp, fp = r[pl]
            rn, fn_ = r[ml]
            a, b = rp - rn, fp - fn_
            if lab.strip() == "CA on":
                worst = max(worst, abs(a - b))
            print(f"  {lab:<8}{pl + '/' + ml:<10}{a:22.16f}{b:22.16f}"
                  f"{a - b:16.2e}{rp - fp:15.2e}{rn - fn_:15.2e}")
        print()
    ok = worst < 1e-14
    print(f"  worst |odd(ref) - odd(fluct)| with the switch ON: {worst:.2e}  "
          f"-> {'PASS' if ok else '*** FAIL ***'}")
    print("  (the pre-fix value of that number was the whole charge-odd term, "
          "3.19e-3 to 3.33e-3)")
    if not ok:
        raise SystemExit("the reference and the fluctuation disagree on the "
                         "charge-odd mean")


def cmd_table(args):
    """The TABLE itself, at the C++ level, with and without the switch.

    `barkas_g4driver` links libTrackPropagationGeant4e and calls
    `G4EnergyLossForExtrapolatorForCVH::ComputeDEDX` for all eight particles in
    the toy material, alongside Geant4's OWN
    `G4(Mu)BetheBlochModel::ComputeDEDXPerVolume` for the same particle.  This
    is NOTES_BARKAS s2.3's table re-measured on the patched library: there the
    charge conjugates agreed to 0.00e+00 and the common value sat at the
    POSITIVE particle's, 2.2-2.4e-3 away from the negative one's.

    The driver must be REBUILT first (`barkas_probe.py build`): the class grew
    a member, so a binary compiled against the old header would allocate the
    old size.  Rebuilding changes its md5 against the NOTES_BARKAS fingerprint;
    the SOURCE is untouched, only the library it links has moved."""
    print("=" * 122)
    print("THE EXTRAPOLATOR'S dE/dx TABLE, C++ LEVEL, WITH AND WITHOUT THE "
          "SWITCH")
    print("`extrapolator` is G4EnergyLossForExtrapolatorForCVH::ComputeDEDX; "
          "`vs G4 (this charge)` is the relative")
    print("difference against Geant4's own unrestricted dE/dx FOR THAT "
          "PARTICLE.  MeV/mm, toy material, p = 3136 MeV/c.")
    print("(The muon column carries a constant ~+1e-3 offset because the muon "
          "table also includes the radiative")
    print("mean, which the comparison denominator does not -- NOTES_BARKAS "
          "s2.3.)")
    print("=" * 122)
    import re
    res = {}
    for arm, env in (("off", {}), ("on", {"CVH_REF_CHARGEAWARE": "1"})):
        txt, _ = bp._driver(env=env)
        # The tables are built LAZILY, inside the driver's first ComputeDEDX
        # call -- which happens between its `xs` printf and its `extrap`
        # printf.  The switch's G4cout banner therefore lands INSIDE a SPECIES
        # line (stdio and iostream do not share a buffer), splitting it in two.
        # Deleting the banner together with both of its newlines rejoins the
        # line exactly.  This is a property of the test driver's output, not of
        # the banner: in a cmsRun log the banner is on its own line and is the
        # liveness check used throughout this note.
        txt = re.sub(r"\n### G4EnergyLossForExtrapolatorForCVH:[^\n]*\n",
                     "", txt)
        res[arm], _ = bp._parse_species(txt)
    print(f"  {'species':<8}{'extrapolator OFF':>22}{'extrapolator ON':>22}"
          f"{'vs conj OFF':>14}{'vs conj ON':>13}"
          f"{'vs G4 OFF':>12}{'vs G4 ON':>12}")
    for m, p, lab in PAIRS:
        for pdg in (p, m):                     # positive first
            a, b = res["off"][pdg], res["on"][pdg]
            conj = p if pdg == m else m
            ca_, cb = res["off"][conj], res["on"][conj]
            print(f"  {a['lab']:<8}{a['extrap']:22.16f}{b['extrap']:22.16f}"
                  f"{a['extrap'] - ca_['extrap']:14.2e}"
                  f"{b['extrap'] - cb['extrap']:13.2e}"
                  f"{(a['extrap'] - a['dedxU']) / a['dedxU']:12.2e}"
                  f"{(b['extrap'] - b['dedxU']) / b['dedxU']:12.2e}")


def cmd_refprof(args):
    """The shift plane by plane -- it must ACCUMULATE monotonically along the
    trajectory rather than appear at one surface."""
    print("=" * 100)
    print("REFERENCE SHIFT PER PLANE, MeV (charge-aware minus nominal)")
    print("=" * 100)
    for pdg in args.pdg:
        sp = hp.SPECIES[pdg]
        if not os.path.exists(_mp(pdg, "ca")):
            continue
        l0 = cpt.load_model(_mp(pdg))
        l1 = cpt.load_model(_mp(pdg, "ca"))
        m = sp["mass"]
        d = []
        for a, b in zip(l0, l1):
            Pa, Pb = 1e3 / abs(a["refqop"]), 1e3 / abs(b["refqop"])
            d.append(math.sqrt(Pb * Pb + m * m) - math.sqrt(Pa * Pa + m * m))
        print(f"  {sp['label']:<6}" + "".join(f"{x:9.5f}" for x in d))


# =========================================================================
# 4.  the closure
# =========================================================================

def _rows_arm(pdg, arm, sim_arm, func="qop"):
    """One closure curve against a given MODEL arm.

    The Fisher normalization `s_F` is RECOMPUTED from the arm's own model (it
    is not inherited): the reference moved, so the linearization the scale is
    built on moved with it.  All three caches are bypassed exactly as
    `hadron_probe._rows` does."""
    import cf_track_resolution as ctr
    import fisher_norm as fn
    import geom_closure as gc
    assert os.environ.get("RES_NO_PHI_CACHE"), "phi cache is LIVE"
    ctr.IONI_KOKOULIN = 1.0 if hp.kok_for(pdg) else 0.0
    fn._SCALE_CACHE.clear()
    cpt._PHI_CACHE.clear()
    old_load, old_store = fn._scale_cache_load, fn._scale_cache_store
    fn._scale_cache_load = lambda *a, **k: None
    fn._scale_cache_store = lambda *a, **k: None
    try:
        mp = _mp(pdg, arm)
        legs = cpt.load_model(mp)
        sc = fn.plane_scales(legs, func, tag=mp, channels=("ioni", "ms", "rad"))
        rows, errs, ks, ns, err = gc.closure_rows(legs, hp.sim_of(pdg, sim_arm),
                                                  func, sc["sF"])
    finally:
        fn._scale_cache_load, fn._scale_cache_store = old_load, old_store
        ctr.IONI_KOKOULIN = 0.0
        fn._SCALE_CACHE.clear()
        cpt._PHI_CACHE.clear()
    return dict(m=rows.mean(axis=0), sc=sc, n=int(np.median(ns)))


def cmd_closure(args):
    print("=" * 132)
    print("PER-SPECIES CLOSURE WITH THE C++ CHARGE-AWARE REFERENCE, "
          f"{args.func}, plane mean, Fisher normalization,")
    print("layered toy pT = 3, eta 0.30, cut1e4, exact-delta ON, Kokoulin "
          "species-correct, nuclear + Decay OFF, 200k events.")
    print("The POSITIVES cannot move: the correction is identically zero at "
          "q = +1.  Nothing is fitted anywhere.")
    print("=" * 132)
    print(f"  {'species':<8}{'arm':<8}" + hp.UROW + "      rms")
    res = {}
    for pdg in args.pdg:
        for arm in args.arms:
            r = _rows_arm(pdg, None if arm == "nominal" else arm, args.simarm,
                          args.func)
            res[(pdg, arm)] = r
            print(f"  {hp.SPECIES[pdg]['label']:<8}{arm:<8}" + hp._fmt(r["m"])
                  + f" {hp._rms(r['m']):9.5f}   n={r['n']}  "
                  f"s_F {np.mean(r['sc']['sF']):.8e}")
    print()
    print("  rms, before and after")
    print(f"    {'species':<8}{'q':>3}" + "".join(f"{a:>12}" for a in args.arms)
          + f"{'change':>12}")
    for pdg in args.pdg:
        if (pdg, args.arms[0]) not in res:
            continue
        a0 = hp._rms(res[(pdg, args.arms[0])]["m"])
        row = f"    {hp.SPECIES[pdg]['label']:<8}{hp.SPECIES[pdg]['q']:>3}"
        for arm in args.arms:
            row += f"{hp._rms(res[(pdg, arm)]['m']):12.5f}"
        aN = hp._rms(res[(pdg, args.arms[-1])]["m"])
        row += ("       ZERO" if aN == a0 else f"{aN - a0:+12.5f}")
        print(row)
    print()
    print("  charge asymmetry of the closure rms, q = -1 / q = +1")
    print(f"    {'species':<8}" + "".join(f"{a:>14}" for a in args.arms))
    for m, p, lab in PAIRS:
        row = f"    {lab:<8}"
        for arm in args.arms:
            if (m, arm) in res and (p, arm) in res:
                a, b = hp._rms(res[(m, arm)]["m"]), hp._rms(res[(p, arm)]["m"])
                row += f"{a / b:14.2f}"
            else:
                row += f"{'':>14}"
        print(row)


# =========================================================================
# 5.  the offline cache registries must see the new knob
# =========================================================================

def cmd_registry(args):
    """NOTES_BARKAS s9.2 made `cf_propagation_test._phi_key` and
    `fisher_norm.scale_identity` registry-driven so that a NEW physics switch
    cannot be silently absent from a cache key, and `fisher_norm._MODEL_CACHE`
    sha256-keyed so that a model re-exported under the same name inside one
    session cannot serve the previous export's legs.

    Both claims are tested here against THIS knob, which is exactly the case
    they were built for: `chargeodd export` rewrites model files in place under
    `--force`, and the charge-aware reference is a new physics state."""
    import importlib
    import fisher_norm as fn

    print("=" * 108)
    print("DO THE CACHE REGISTRIES CATCH THIS KNOB?")
    print("=" * 108)

    print("\n1. the undeclared-knob audit still passes "
          "(every module-level numeric global is declared)")
    rows, bad = bp._knob_audit()
    print(f"   modules audited: {', '.join(bp._KNOB_MODULES)}   "
          f"globals: {len(rows)}")
    print(f"   undeclared knobs: {'NONE' if not bad else ', '.join(bad)}")

    print("\n2. the s_F CACHE KEY moves with this knob -- measured, not "
          "asserted, and it is CONSERVATIVE.")
    print("   `fisher_norm.scale_identity` hashes the MODEL FILE's sha256 "
          "together with every declared")
    print("   physics global, so a re-export can never serve a stale 1/I.  "
          "Note what the file hash is NOT:")
    print("   it moves for a POSITIVE species too, whose 27 branches are "
          "bit-identical -- ROOT embeds a UUID")
    print("   and per-write metadata, so two exports of identical DATA are "
          "different FILES (sizes differ by")
    print("   tens of bytes).  The key therefore over-invalidates, which is "
          "the safe direction, and it is not")
    print("   a content comparator: `bitid`'s branch digests are.  Reported "
          "here so the two are not confused.")
    print(f"   {'species':<7}{'q':>3}  {'nominal key':<18}"
          f"{'charge-aware key':<18}{'branches':>9}  verdict")
    bad = 0
    for pdg in args.pdg:
        a, b = _mp(pdg), _mp(pdg, "ca")
        if not os.path.exists(b):
            continue
        ka = []
        for p in (a, b):
            fn._MODEL_CACHE.clear()
            legs = fn.load(p)
            ka.append(fn.scale_identity(legs, "qop", 1e-8, 1 << 17, 32, -60.,
                                        ("ioni", "ms", "rad"), None)["_sha"])
        fn._MODEL_CACHE.clear()
        n, diff = _cmp(a, b)
        q = hp.SPECIES[pdg]["q"]
        moved = ka[0] != ka[1]
        bad += 0 if moved else 1
        ok = ("MOVED; content moved too (negative)" if (moved and diff) else
              "MOVED although the content did NOT (see above)" if moved else
              "*** STALE KEY -- FAIL ***")
        print(f"   {hp.SPECIES[pdg]['label']:<7}{q:>3}  {ka[0][:16]:<18}"
              f"{ka[1][:16]:<18}{n - len(diff):>4}/{n:<4}  {ok}")
    print(f"   -> {'PASS' if bad == 0 else 'FAIL'}: no cached 1/I can survive "
          f"this knob")

    print("\n3. a STALE model-cache hit is LOUD.  `fisher_norm.load` verifies "
          "a hit against the file's")
    print("   sha256 and RAISES on a mismatch.  Forced here by loading a "
          "model, then swapping the file")
    print("   underneath the cache entry.")
    src, alt = _mp(args.pdg[0]), _mp(args.pdg[0], "ca")
    if os.path.exists(alt) and hasattr(fn, "_MODEL_CACHE"):
        import shutil
        tmp = os.path.join(SCRATCH, "co_stale_probe.root")
        shutil.copyfile(src, tmp)
        try:
            fn.load(tmp)
            shutil.copyfile(alt, tmp)          # same path, different content
            try:
                fn.load(tmp)
                print("   -> FAIL: the stale hit was served silently")
            except Exception as e:              # noqa: BLE001
                print(f"   -> PASS: {type(e).__name__}: "
                      f"{str(e).splitlines()[0][:90]}")
        finally:
            fn._MODEL_CACHE.clear()
            if os.path.exists(tmp):
                os.remove(tmp)
    else:
        print("   (skipped: fisher_norm has no _MODEL_CACHE, or no `ca` model)")

    print("\n4. WHERE THIS KNOB DOES *NOT* BELONG, and why that is not an "
          "omission.")
    print("   `_phi_key` and `scale_identity` key the OFFLINE model's physics "
          "state.  CVH_REF_CHARGEAWARE")
    print("   changes the C++ REFERENCE and nothing in the offline model: "
          "there is no offline consumer to")
    print("   key, and the reference enters the offline chain only through the "
          "MODEL FILE, whose content")
    print("   identity is already in both keys (item 2/3).  Adding a python "
          "global that nothing reads")
    print("   would be a decoration.  The registry entry that matters is the "
          "file identity, and it is live.")


# =========================================================================
# 6.  downstream of the reference, on real tracks
# =========================================================================

# The arms.  `nominal2` and `dedxfloor` are the two controls WITHOUT WHICH
# NOTHING IN `cvhcmp` IS INTERPRETABLE.
CVHENV = {
    "nominal":   {},
    "ca":        {"CVH_REF_CHARGEAWARE": "1"},
    # THE CONTROL ON THE CONTROL: byte-identical configuration to `nominal`.
    # If these two differ, the "floor" is job nondeterminism and every
    # floor-relative statement collapses.
    "nominal2":  {},
    # THE FLOOR.  A physically null perturbation of the SAME rows this change
    # touches -- the mean-loss table itself.  CVH_DEDX_SCALE multiplies the
    # muon dE/dx table by a constant; 1 + 1e-9 is ~2e6 times smaller than the
    # charge-odd term it is the floor for (measured in `refdiff`), so anything
    # the fit does in response is the flat-direction reordering NOTES_CGFFIT
    # measured at gain ~1e12, not physics.
    #
    # It is deliberately a MEAN perturbation, not the `CVH_IONI_KOKOULIN_NBIN`
    # variance one NOTES_QVALID used: this change is a mean change and the
    # floor has to be measured on the same code path.
    "dedxfloor": {"CVH_DEDX_SCALE": "1.000000001"},
    # A second, LARGER null-in-kind perturbation.  1 + 1e-6 is still ~2000x
    # below the charge-odd term but far above the numerical floor, so the
    # response should be LINEAR in it if the fit is responding to physics and
    # not to reordering.  Used only as a sanity gauge.
    "dedxgauge": {"CVH_DEDX_SCALE": "1.000001"},
}


def cvh_out(tag, lab, script="runCvhSingleTrack.py"):
    return os.path.join(CO, f"cvh_{tag}_{lab}",
                        f"{qv.PREFIX.get(script, 'globalcor_single')}_0.root")


def cmd_cvh(args):
    os.makedirs(CO, exist_ok=True)
    for lab in args.labels:
        wd = os.path.join(CO, f"cvh_{args.tag}_{lab}")
        log = wd + ".log"
        if os.path.exists(cvh_out(args.tag, lab, args.script)) and not args.force:
            print(f"[{lab}] exists, skipping")
            continue
        rc = qv.run_cvh(args.script, args.input, wd, log,
                        dict(CVHENV[lab]), args.nev, args.extra)
        print(f"[{lab}] rc={rc} -> {cvh_out(args.tag, lab, args.script)}")
        if rc:
            raise SystemExit(f"cvh run failed, see {log}")


def _relnorm(x, y):
    n = np.sqrt(np.sum(x ** 2))
    return np.sqrt(np.sum((y - x) ** 2)) / n if n > 0 else 0.


def _stat(v, name, ind="    "):
    v = np.asarray(v, float)
    if not len(v):
        return f"{ind}{name:<20} (empty)"
    # BOTH statistics, always.  The per-track response is a MIXTURE -- a
    # coherent shift on every track plus a rare violent kick on the few percent
    # the fit's flat direction lets wander -- so the pooled/quadrature norm is a
    # TAIL statistic and understates signal-over-floor by up to 100x
    # (NOTES_QVALID 7.1).  The median is the one to read; the pooled column
    # says how big the chaotic tail is.
    return (f"{ind}{name:<20} median {np.median(v):.4e}   "
            f"trimmed95 {qv._trim(v, centred=False):.4e}   "
            f"pooled {np.sqrt(np.sum(v ** 2) / len(v)):.4e}   "
            f"exactly unchanged {np.count_nonzero(v == 0.)}/{len(v)}")


def _block(A, B, i, j, rtA, title, gen=False):
    """the whole downstream measurement on one subset of matched tracks."""
    print(f"  --- {title}   n = {len(i)}")
    if not len(i):
        return {}
    out = {}

    qa = np.array([p[0] for p in A["refParms"]])[i]
    qb = np.array([p[0] for p in B["refParms"]])[j]
    d = (qb - qa) / np.abs(qa)
    out["dqop_mean"] = d.mean()
    print(f"    fitted q/p  d(q/p)/|q/p|  mean {d.mean():+.4e} "
          f"+- {d.std() / np.sqrt(max(len(d), 1)):.2e}")
    print(f"      median|.| {np.median(np.abs(d)):.4e}   "
          f"trimmed95 {qv._trim(d):.4e}   full rms {d.std():.4e}   "
          f"max|.| {np.abs(d).max():.3e}   exactly unchanged "
          f"{np.count_nonzero(d == 0.)}/{len(d)}")
    pt = A["trackPt"][i]
    for lo, hi in ((0, 3), (3, 5), (5, 10), (10, 1e9)):
        m = (pt >= lo) & (pt < hi)
        if m.sum() > 5:
            print(f"      pT [{lo:g},{hi:g})  n {m.sum():5d}   "
                  f"mean {d[m].mean():+.4e}  rms {d[m].std():.4e}   "
                  f"x <pT> {d[m].mean() * np.mean(pt[m]):+.3e}")
    # The implied ENERGY shift.  This is the quantity that should be FLAT in pT
    # (a mean-loss change is an energy, not a scale) and it is the only way to
    # compare the real CMS tracker against the toy's 12.6 g/cm2 of Z = 8.
    #
    # The fit sets p_ref = p_measured_outside + L_model, so if the model's
    # accumulated loss L falls by `delta`, p_ref falls by `delta`.  Working in
    # MOMENTA rather than in the signed q/p keeps this charge-independent --
    # d(q/p)/|q/p| is +dp/p for a negative track and -dp/p for a positive one,
    # and a formula that gets that backwards would be a latent sign bug the
    # moment a positive track moved.
    _MMU = 105.6583745e-3
    Pa, Pb = 1.0 / np.abs(qa), 1.0 / np.abs(qb)
    dpp = (Pb - Pa) / Pa
    dEover = -dpp * Pa * Pa / np.sqrt(Pa * Pa + _MMU ** 2)
    print(f"    dp/p  mean {dpp.mean():+.4e}   |  over-subtraction removed, "
          f"dE = -dp/p * p^2/E:  mean {1e3 * dEover.mean():+.4f} MeV   "
          f"median {1e3 * np.median(dEover):+.4f}   rms {1e3 * dEover.std():.4f}")
    out["dE_MeV"] = 1e3 * float(dEover.mean())

    ca = np.array([c[0] for c in A["refCov"]])[i]
    cb = np.array([c[0] for c in B["refCov"]])[j]
    r = np.sqrt(cb / ca)
    print(f"    reported sigma(q/p) ratio  mean {r.mean():.7f}  "
          f"med {np.median(r):.7f}  min {r.min():.6f}  max {r.max():.6f}")

    for k, fmt in (("niter", "{:.4f}"), ("edmvalref", "{:.3e}"),
                   ("chisqval", "{:.5f}")):
        if k not in A:
            continue
        va, vb = np.asarray(A[k], float)[i], np.asarray(B[k], float)[j]
        print(f"    {k:<10} mean {fmt.format(va.mean())} -> "
              f"{fmt.format(vb.mean())}   changed on "
              f"{np.count_nonzero(va != vb)} / {len(i)}")
    if "ndof" in A:
        na = (np.asarray(A["chisqval"], float)[i]
              / np.maximum(np.asarray(A["ndof"], float)[i], 1))
        nb = (np.asarray(B["chisqval"], float)[j]
              / np.maximum(np.asarray(B["ndof"], float)[j], 1))
        print(f"    chi2/ndof  {na.mean():.5f} -> {nb.mean():.5f}")

    # IS IT EARNED?  On gen-truth MC this is the only question that matters
    # for a MEAN change: the reference moved, so the reconstructed q/p should
    # move TOWARD the generated one for a negative track and not at all for a
    # positive one.  A closure improvement in the toy is an argument; this is
    # the measurement.
    if gen and "genParms" in A:
        ga = np.array([p[0] for p in A["genParms"]])[i]
        gb = np.array([p[0] for p in B["genParms"]])[j]
        ok = (ga != 0.) & (gb != 0.) & (ga == gb)
        if ok.sum() > 20:
            for lab2, qq, cc in (("old", qa, ca), ("new", qb, cb)):
                # dp/p, NOT d(q/p)/|q/p|.  The latter changes SIGN with the
                # charge for one and the same physical momentum bias, so on a
                # charge-split table it manufactures an asymmetry that is not
                # there -- exactly the kind of statistic-vs-quantity mismatch
                # this study keeps walking into.  p = 1/|q/p|.
                res = (np.abs(ga[ok]) / np.abs(qq[ok])) - 1.0
                pl = (qq - ga)[ok] / np.sqrt(cc[ok])
                lo, hi = np.percentile(pl, [15.865, 84.135])
                lo2, hi2 = np.percentile(res, [15.865, 84.135])
                print(f"    [gen {lab2}]  (p_fit - p_gen)/p_gen: mean "
                      f"{res.mean():+.4e} +- "
                      f"{res.std() / np.sqrt(ok.sum()):.2e}   "
                      f"68% half-width {0.5 * (hi2 - lo2):.4e}   rms "
                      f"{res.std():.4e}")
                print(f"                pull rms {pl.std():.4f}   pull 68% "
                      f"half-width {0.5 * (hi - lo):.4f}   n {ok.sum()}")

    if "jacrefv" in A and "globalidxv" in A and rtA is not None:
        ptype = np.zeros(int(rtA["iidx"].max()) + 1, dtype=int)
        ptype[np.asarray(rtA["iidx"])] = np.asarray(rtA["parmtype"])
        acc = {}
        for ii, jj in zip(i, j):
            ga, gb = A["globalidxv"][ii], B["globalidxv"][jj]
            if len(ga) != len(gb) or np.any(ga != gb):
                continue
            n = len(ga)
            ja = np.asarray(A["jacrefv"][ii], float).reshape(5, n)
            jb = np.asarray(B["jacrefv"][jj], float).reshape(5, n)
            fam = np.array([qv._fam(ptype[g]) for g in ga])
            for f in np.unique(fam):
                m = fam == f
                acc.setdefault(f, []).append(_relnorm(ja[0, m], jb[0, m]))
        for f, v in sorted(acc.items()):
            print(_stat(v, f"jacref {f}"))
            out[f"jacref_{f}"] = float(np.median(v))

    for k in ("gradv", "hesspackedv"):
        if k not in A or k not in B:
            continue
        v = []
        for ii, jj in zip(i, j):
            x = np.asarray(A[k][ii], float)
            y = np.asarray(B[k][jj], float)
            if x.shape == y.shape:
                v.append(_relnorm(x, y))
        if v:
            print(_stat(v, k))
            out[k] = float(np.median(v))
    return out


def cmd_cvhcmp(args):
    print("=" * 108)
    print("DOWNSTREAM OF THE *REFERENCE*, ON REAL TRACKS")
    print("This changes the MEAN, i.e. the linearization point.  Split by "
          "TRACK CHARGE: the positives are")
    print("the free regression test (the correction is identically zero for "
          "them) and the negatives carry")
    print("the whole effect.  Read the MEDIAN; `pooled` is a tail statistic "
          "(NOTES_QVALID 7.1).")
    print("=" * 108)
    ref = args.labels[0]
    A, rtA, _ = qv._load_cvh(cvh_out(args.tag, ref, args.script))
    qA = np.asarray(A["trackCharge"])
    print(f"reference configuration: {ref}   {len(A['run'])} tracks   "
          f"({np.count_nonzero(qA < 0)} negative, "
          f"{np.count_nonzero(qA > 0)} positive)\n")
    for lab in args.labels[1:]:
        B, rtB, _ = qv._load_cvh(cvh_out(args.tag, lab, args.script))
        i, j = qv._match(A, B)
        q = np.asarray(A["trackCharge"])[i]
        print(f"### {lab}  vs  {ref}      matched {len(i)} / "
              f"{len(A['run'])} vs {len(B['run'])}")
        sel = {"ALL": np.ones(len(i), bool)}
        if args.split:
            sel["q = -1 (the correction applies)"] = q < 0
            sel["q = +1 (must be EXACTLY unchanged)"] = q > 0
        for name, m in sel.items():
            _block(A, B, i[m], j[m], rtA, name, gen=args.gen)
        print()


# =========================================================================

def main():
    ap = argparse.ArgumentParser()
    s = ap.add_subparsers(dest="cmd", required=True)

    q = s.add_parser("export")
    q.add_argument("--arms", nargs="+", default=["caoff", "ca"])
    q.add_argument("--pdg", nargs="+", type=int, default=ORDER8)
    q.add_argument("--force", action="store_true")

    q = s.add_parser("bitid")
    q.add_argument("--pdg", nargs="+", type=int, default=ORDER8)

    q = s.add_parser("refdiff")
    q.add_argument("--pdg", nargs="+", type=int, default=ORDER8)

    s.add_parser("table")
    s.add_parser("meanloss")

    q = s.add_parser("refprof")
    q.add_argument("--pdg", nargs="+", type=int, default=ORDER8)

    q = s.add_parser("closure")
    q.add_argument("--pdg", nargs="+", type=int, default=ORDER8)
    q.add_argument("--arms", nargs="+", default=["nominal", "caoff", "ca"])
    q.add_argument("--simarm", default="off")
    q.add_argument("--func", default="qop")

    q = s.add_parser("registry")
    q.add_argument("--pdg", nargs="+", type=int, default=[13, -13])

    q = s.add_parser("cvh")
    q.add_argument("--tag", default="jpsi")
    q.add_argument("--script", default="runCvhSingleTrack.py")
    q.add_argument("--input", required=True)
    q.add_argument("--nev", type=int, default=1200)
    q.add_argument("--extra", default="")
    q.add_argument("--labels", nargs="+", default=list(CVHENV))
    q.add_argument("--force", action="store_true")

    q = s.add_parser("cvhcmp")
    q.add_argument("--tag", default="jpsi")
    q.add_argument("--script", default="runCvhSingleTrack.py")
    q.add_argument("--labels", nargs="+", required=True)
    q.add_argument("--split", action="store_true", default=True)
    q.add_argument("--no-split", dest="split", action="store_false")
    q.add_argument("--gen", action="store_true")

    a = ap.parse_args()
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
