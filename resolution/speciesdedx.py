#!/usr/bin/env python3
"""The proton-table Tmax defect of NOTES_PION, turned into a real switch.

WHAT IS BEING FIXED
-------------------
`G4TablesForExtrapolatorForCVH` builds no hadron table except the proton's.
`G4EnergyLossForExtrapolatorForCVH::ComputeDEDX` serves every other hadron from
it at the SCALED kinetic energy `e = ekin * m_p / m`, and `ComputeRange` /
`ComputeEnergy` do the same with the matching `massratio` factors.

That scaling preserves `beta*gamma` exactly, hence beta, gamma, the mean
excitation energy I, the density-effect delta, the shell correction and the
whole Barkas/Bloch/Mott block -- but NOT `Tmax`, whose recoil denominator
carries the PROJECTILE mass.  Unrestricted Bethe-Bloch is
`dE = xi [ln(2 m_e bg^2 Tmax/I^2) - 2 beta^2 - delta]`, so
`d(dE)/d ln Tmax = xi` exactly and the whole error is

    d(dE/dx) = xi_perlength * ln( Tmax_species / Tmax_table ),
    xi_perlength = twopi_mc2_rcl2 * n_el * z^2 / beta^2

with nothing free.  Measured against Geant4's own unrestricted dE/dx for the
species: +5.24e-3 (pi), +2.86e-4 (K), and identically zero for the proton (it
IS the table) and for the muon (its own table).

THE FIX, AND WHY IT IS THIS ONE
-------------------------------
`CVH_REF_SPECIESDEDX` (single C++ reader `cvhcgf::referenceIsSpeciesDedx`)
adds that term at LOOKUP time in `ComputeDEDX`, carries it into `ComputeRange`
as the exact perturbation integral of the corrected stopping power, and makes
`ComputeEnergy` the numerical inverse of the corrected range.
`G4UniversalFluctuationForExtrapolator`'s own `meanLoss = length*dedx` reads
the same proton table through the SAME `cvhcgf` helper.

Why not a per-species TABLE, which is what `CVH_REF_CHARGEAWARE` did for the
charge:

  * the tables live on ONE energy grid, [1 MeV, 100 TeV], which is a grid in
    beta*gamma only AFTER the mass scaling.  A pion table on the same grid
    starts at bg = 0.12, below G4BetheBlochModel's validity, and the RANGE
    table is a cumulative integral from the bottom of the grid -- so the
    invalid region would corrupt the range at every energy.  The mass scaling
    is the mechanism that covers every hadron with one 81-bin grid; it is not
    an approximation to be removed.
  * what is transcribed is NOT a model.  `d(dE/dx)/d ln Tmax = xi` is the
    exact analytic derivative of the leading term and Tmax is exact two-body
    recoil kinematics; Geant4 cannot revise either without ceasing to be
    Bethe-Bloch.  That is precisely NOT true of the Barkas parameterization or
    the Mott series, which is why the charge-odd fix went the other way.
  * the species set is open (the propagator builds its particle from a
    configured name), so a per-species table needs either a PDG list -- which
    the charge-aware fix was right to avoid -- or a lazily mutated
    process-wide static.

THE FREE REGRESSION TEST
------------------------
The correction is identically zero for a MUON (own table, different branch)
and for a PROTON or ANTIPROTON (the table IS theirs, and the reference mass
used is the table particle's own PDG mass, so the log is bit-for-bit zero).
Those four species must be BIT-IDENTICAL with the switch on, at every level.
Any movement is a bug, not an improvement.

SUBCOMMANDS
    table     the extrapolator's dE/dx TABLE, C++ level, off / on / with
              charge-aware / both, against Geant4's own answer per particle
    spinterm  the residual after the fix, predicted and measured
    nbin      convergence of the range-defect quadrature
    export    toy models: `spdoff` (fresh, switch off), `spd`, `ca`, `both`
    bitid     branch digests -- OFF bit-identity, mu/p unchanged, pi/K LIVE
    refdiff   the C++ reference shift per species against NOTES_PION's
              parameter-free offline gauge
    refprof   the same shift plane by plane; it must ACCUMULATE
    closure   THE RESULT: per-species closure, mu/p shown unchanged
    both      all four arms, all eight species -- the composition
    registry  the offline cache registries must SEE this knob
    cvh       real-track CVH refit (muon null + V0 pion signal)
    cvhcmp    downstream of the reference
"""

import argparse
import hashlib
import math
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "cleanprop"))

os.environ["RES_NO_PHI_CACHE"] = "1"

import hadron_probe as hp                                        # noqa: E402
import barkas_probe as bp                                        # noqa: E402
import pion_probe as pp                                          # noqa: E402
import cf_propagation_test as cpt                                # noqa: E402
import cf_track_resolution as ctr                                # noqa: E402
import fisher_norm as fn                                         # noqa: E402
import geom_closure as gc                                        # noqa: E402
import deltaspec as ds                                           # noqa: E402
import qvalid as qv                                              # noqa: E402

SCRATCH = hp.SCRATCH
OUT = hp.OUT                       # shares NOTES_HADRONS'/NOTES_PION's dir
SD = os.path.join(SCRATCH, "sd")   # the real-track CVH area for this note
os.makedirs(SD, exist_ok=True)

ORDER8 = bp.ORDER8
PAIRS = bp.PAIRS

# THE SEED PIN.  pion_probe already replaces hadron_probe.sim_glob at import
# with the `_s1*` restriction (NOTES_PION s1: a stray seed-901 mu- `off` file
# makes the muon control 220k events against every hadron's 200k, and a
# control that does not reproduce the published digits is not a control).
# Importing pion_probe above is what installs it; assert rather than assume.
assert "_s1" in hp.sim_glob(13, "off"), "the seed pin is not installed"

MEL = pp.MEL
MPROT_CLHEP = pp.MPROT             # CLHEP::proton_mass_c2, used by the SCALING
MPROT_G4 = 938.272013              # G4Proton::GetPDGMass(), used by the FIX

ARM_ENV = {
    "spdoff": {},
    "spd":    {"CVH_REF_SPECIESDEDX": "1"},
    "ca":     {"CVH_REF_CHARGEAWARE": "1"},
    "both":   {"CVH_REF_CHARGEAWARE": "1", "CVH_REF_SPECIESDEDX": "1"},
    # THE LIBRARY-REBUILD CONTROL.  The `_ca` models were exported on
    # 2026-08-16 07:38 against the library as it stood BEFORE this change.
    # Both classes have since grown members, so `_carelib` is the same
    # configuration re-exported against the new library and must be
    # BIT-IDENTICAL to `_ca` -- without which no comparison that mixes the two
    # vintages means anything (the 2026-08-15 barkas driver died with a glibc
    # malloc assertion against the 2026-08-16 library for exactly this).
    "carelib": {"CVH_REF_CHARGEAWARE": "1"},
}
ARM_SUFFIX = {k: "_" + k for k in ARM_ENV}

_bp_env = hp._env
SD_ENV = {}


def _env(pdg, arm, extra=None):
    e = _bp_env(pdg, arm, extra)
    e.update(SD_ENV)
    return e


hp._env = _env


# =========================================================================
# 0.  the C++ table
# =========================================================================

def _table_arms(arms):
    res = {}
    for arm in arms:
        txt, _ = bp._driver(env=ARM_ENV[arm])
        # The tables are built LAZILY, inside the driver's first ComputeDEDX,
        # which happens between its `xs` printf and its `extrap` printf, so the
        # switch's G4cout banner lands INSIDE a SPECIES line (stdio and
        # iostream do not share a buffer).  Deleting it with both newlines
        # rejoins the line exactly.  Property of the driver's output, not of
        # the banner: in a cmsRun log it is on its own line and is the
        # liveness check.
        txt = re.sub(r"\n### G4EnergyLossForExtrapolatorForCVH:[^\n]*\n",
                     "", txt)
        res[arm], _ = bp._parse_species(txt)
    return res


def cmd_table(args):
    """The TABLE itself, C++ level, all four arms.

    `barkas_g4driver` links libTrackPropagationGeant4e and calls
    `G4EnergyLossForExtrapolatorForCVH::ComputeDEDX` for all eight particles in
    the toy material, alongside Geant4's OWN
    `G4(Mu)BetheBlochModel::ComputeDEDXPerVolume` for the same particle at the
    same energy.  `vs G4` is therefore the whole defect, not a model of it.

    REBUILD THE DRIVER FIRST (`barkas_probe.py build`): both classes grew
    members, so a binary compiled against the old headers allocates the old
    size.  The 2026-08-15 binary died with a glibc `malloc` assertion against
    the 2026-08-16 library for exactly this reason -- it presents as a physics
    crash."""
    res = _table_arms(("off" if False else "spdoff", "spd", "ca", "both"))
    print("=" * 128)
    print("THE EXTRAPOLATOR'S dE/dx TABLE, C++ LEVEL, MeV/mm, toy material, "
          "p = 3136 MeV/c")
    print("`vs G4` is the relative difference against Geant4's own "
          "UNRESTRICTED dE/dx FOR THAT PARTICLE.  The muon column carries a")
    print("constant ~+1e-3 because the muon table also holds the radiative "
          "mean, which the comparison denominator does not.")
    print("=" * 128)
    print(f"  {'species':<7}{'OFF':>21}{'SPECIESDEDX':>21}{'moved':>11}"
          f"{'vs G4 OFF':>12}{'vs G4 SPD':>12}{'vs G4 BOTH':>12}  bitwise-equal")
    nbad = []
    for m, p, lab in PAIRS:
        for pdg in (p, m):
            a, b, d = res["spdoff"][pdg], res["spd"][pdg], res["both"][pdg]
            eq = (a["extrap"] == b["extrap"])
            expect_null = abs(pdg) in (13, 2212)
            if eq != expect_null:
                nbad.append(a["lab"])
            print(f"  {a['lab']:<7}{a['extrap']:21.16f}{b['extrap']:21.16f}"
                  f"{(b['extrap']-a['extrap'])/a['extrap']:11.3e}"
                  f"{(a['extrap']-a['dedxU'])/a['dedxU']:12.3e}"
                  f"{(b['extrap']-b['dedxU'])/b['dedxU']:12.3e}"
                  f"{(d['extrap']-d['dedxU'])/d['dedxU']:12.3e}"
                  f"   {str(eq):<5} {'(required)' if expect_null else ''}")
    print(f"    -> {'PASS' if not nbad else 'FAIL ' + ','.join(nbad)}: "
          f"mu and p bit-identical, pi and K not")

    print()
    print("THE TWO SWITCHES ARE INDEPENDENT.  The Tmax term depends on "
          "(ekin, m, m_table, z^2, n_el) only, and")
    print("m(anti_proton) == m(proton), so the ABSOLUTE shift it adds must be "
          "the same on either table.")
    print(f"  {'species':<7}{'spd - spdoff':>18}{'both - ca':>18}"
          f"{'difference':>14}")
    for pdg in ORDER8:
        d1 = res["spd"][pdg]["extrap"] - res["spdoff"][pdg]["extrap"]
        d2 = res["both"][pdg]["extrap"] - res["ca"][pdg]["extrap"]
        print(f"  {res['spd'][pdg]['lab']:<7}{d1:18.15f}{d2:18.15f}"
              f"{d2 - d1:14.2e}")


def cmd_spinterm(args):
    """What is LEFT after the fix, predicted before it is looked at.

    At fixed beta*gamma the ONLY other mass-dependent term of
    G4BetheBlochModel's unrestricted bracket is the spin-1/2 recoil term
    `(0.5 Tmax/E_tot)^2`, present for the proton (spin 1/2) and absent for pi
    and K (spin 0).  Its relative size is that term divided by the bracket
    itself, which the driver prints as `Lrest` for the RESTRICTED loss; the
    unrestricted bracket is `dedxU/pref` less the high-order block.

    So the prediction is: after removing the Tmax term the extrapolator should
    still sit ABOVE Geant4's own value for pi and K by the proton's spin term,
    and the proton itself should not move at all."""
    # `both` for every species: at q = +1 the charge-aware term is
    # identically zero so `both` == `spd` there, and at q = -1 it is the only
    # way to see the Tmax residual without the (much larger) Barkas+Mott one
    # sitting on top of it.
    res = _table_arms(("spdoff", "both"))
    print("=" * 122)
    print("THE RESIDUAL AFTER THE FIX: the proton table's SPIN-1/2 RECOIL "
          "TERM, which a spin-0 hadron should not have.")
    print("Predicted = [s_species (0.5 Tmax_sp/E_sp)^2 - s_table "
          "(0.5 Tmax_p/E_p)^2] / bracket, at the SAME beta*gamma.")
    print("=" * 122)
    print(f"  {'species':<7}{'spin':>6}{'bracket':>10}{'spin term sp':>14}"
          f"{'spin term p':>13}{'predicted':>12}{'measured (both)':>16}{'ratio':>9}")
    for pdg in ORDER8:
        if abs(pdg) == 13:
            continue
        s = res["spdoff"][pdg]
        b = res["both"][pdg]
        bg = s["bg"]
        gam = math.sqrt(1.0 + bg * bg)
        m = s["mass"]
        tmax_sp = pp.tmax_of(bg, m)
        tmax_p = pp.tmax_of(bg, MPROT_G4)
        Esp = gam * m
        Ep = gam * MPROT_G4
        spin_sp = 0.0 if abs(pdg) in (211, 321) else (0.5 * tmax_sp / Esp) ** 2
        spin_p = (0.5 * tmax_p / Ep) ** 2
        # unrestricted bracket, high-order block removed (it is added AFTER
        # the multiplication in G4BetheBlochModel)
        bracket = (s["dedxU"] - s["hoc"]) / s["pref"]
        pred = (spin_sp - spin_p) / bracket
        meas = (b["extrap"] - b["dedxU"]) / b["dedxU"]
        print(f"  {s['lab']:<7}{0 if abs(pdg) in (211,321) else 0.5:>6}"
              f"{bracket:10.3f}{spin_sp:14.3e}{spin_p:13.3e}"
              f"{-pred:12.3e}{meas:16.3e}"
              f"{(meas / -pred) if pred else float('nan'):9.2f}")
    print()
    print("  The sign convention: the TABLE carries the proton's spin term, "
          "so the extrapolator is left HIGH by it.")
    print("  The proton's own row is the interpolation floor of the 81-bin "
          "log spline (-1.12e-5), which is charge- and")
    print("  species-even and is NOT touched by this fix.")


def cmd_nbin(args):
    """Convergence of the range-defect quadrature.

    `speciesRangeDefect` is a composite Simpson in `y = ln(1 + E'/m)` over the
    EXACT perturbation integral; the interval count is exposed
    (`CVH_REF_SPECIESDEDX_NBIN`, default 16) precisely so this is a
    measurement.  The observable is the exported reference itself -- if the
    quadrature mattered, the reference would move with the interval count."""
    import cf_propagation_test as _cpt
    print("=" * 108)
    print("RANGE-DEFECT QUADRATURE CONVERGENCE.  dE at the outermost plane of "
          "the exported reference, MeV, vs Simpson")
    print("intervals.  The dE/dx branch of EnergyAfterStep does not use the "
          "range at all except for the branch test,")
    print("so a flat column also says the range branch is not being taken in "
          "this geometry.")
    print("=" * 108)
    for pdg in args.pdg:
        sp = hp.SPECIES[pdg]
        vals = []
        for nb in args.nbin:
            path = _mp(pdg, "spd") + f".nbin{nb}"
            if not os.path.exists(path):
                vals.append(None)
                continue
            legs = _cpt.load_model(path)
            P = 1e3 / abs(legs[-1]["refqop"])
            vals.append(math.sqrt(P * P + sp["mass"] ** 2))
        base = next((v for v in vals if v is not None), None)
        print(f"  {sp['label']:<7}" + "".join(
            f"{('%14.9f' % v) if v is not None else '           n/a'}"
            for v in vals))
        if base:
            print(f"  {'':<7}" + "".join(
                f"{((v - base) if v is not None else float('nan')):14.2e}"
                for v in vals) + "   (vs the first)")


# =========================================================================
# 0b. the RANGE branch, which the toy geometry cannot exercise
# =========================================================================

BRANCH_DRIVER = os.path.join(SCRATCH, "speciesdedx_g4driver.sh")


def _branch_run(env):
    e = dict(os.environ)
    # historical switch state as the base, arm overlay wins -- see
    # deltaspec._clean_env
    e.update(ctr.SWITCHES_OFF)
    e.update(env)
    r = subprocess.run([BRANCH_DRIVER], capture_output=True, text=True, env=e)
    if r.returncode:
        sys.stderr.write(r.stderr[-2000:])
        raise SystemExit(f"driver failed rc={r.returncode}")
    out = {}
    for line in r.stdout.splitlines():
        f = line.split()
        if f and f[0] == "BRANCH":
            out[(int(f[1]), float(f[3]), float(f[6]))] = dict(
                lab=f[2], ekin=float(f[4]), R=float(f[5]), frac=float(f[7]),
                lossD=float(f[8]), lossR=float(f[9]), rt=float(f[10]))
    return out


def cmd_branch(args):
    """Do the two branches of EnergyAfterStep lose the SAME energy?

    `EnergyAfterStep` picks `step*ComputeDEDX` below `linLossLimit = 0.01` and
    `ComputeEnergy(range - step)` above it, so a ComputeDEDX-only fix would
    make them disagree by exactly the term it removed.  The toy closure cannot
    see this -- at pT = 3 the range is ~1e3 steps long, so the dE/dx branch is
    taken every time and `nbin` is flat.  This calls the three methods
    directly, with ABSOLUTE step lengths (a fraction of the range would
    compare two different steps, since the range itself moves).

    `PARTIAL` pairs the CORRECTED dE/dx branch with the UNCORRECTED range
    branch: that is what a partial fix would have left, and it must be
    `intrinsic + the defect`."""
    off, on = _branch_run({}), _branch_run({"CVH_REF_SPECIESDEDX": "1"})
    print("=" * 132)
    print("BRANCH CONSISTENCY of EnergyAfterStep.  D = step*ComputeDEDX(E); "
          "R = ekin - ComputeEnergy(range - step).")
    print("`rt` = ComputeEnergy(ComputeRange(E)) - E over E: is the corrected "
          "pair still a mutual inverse?")
    print("=" * 132)
    print(f"  {'species':<7}{'p':>7}{'step':>7}{'step/R':>9}{'dR/R':>12}"
          f"{'(R-D)/D OFF':>14}{'(R-D)/D ON':>13}{'PARTIAL':>12}"
          f"{'rt OFF':>11}{'rt ON':>11}")
    for k in sorted(off, key=lambda k: (abs(k[0]), -k[0], -k[1], k[2])):
        a, b = off[k], on[k]
        if b["frac"] < args.minfrac:
            continue
        print(f"  {a['lab']:<7}{k[1]:>7.0f}{k[2]:>7.0f}{b['frac']:>9.4f}"
              f"{(b['R'] - a['R']) / a['R']:>12.3e}"
              f"{(a['lossR'] - a['lossD']) / a['lossD']:>14.2e}"
              f"{(b['lossR'] - b['lossD']) / b['lossD']:>13.2e}"
              f"{(a['lossR'] - b['lossD']) / b['lossD']:>12.2e}"
              f"{a['rt'] / a['ekin']:>11.2e}{b['rt'] / b['ekin']:>11.2e}")


# =========================================================================
# 1.  export
# =========================================================================

def cmd_export(args):
    global SD_ENV
    for arm in args.arms:
        bp.MODEL_SUFFIX = ARM_SUFFIX[arm]
        SD_ENV = dict(ARM_ENV[arm])
        if args.nbin:
            SD_ENV["CVH_REF_SPECIESDEDX_NBIN"] = str(args.nbin)
            bp.MODEL_SUFFIX += ""      # written then renamed below
        print(f"\n=== toy models, arm `{arm}`  suffix={bp.MODEL_SUFFIX!r}  "
              f"env={SD_ENV}")
        ns = argparse.Namespace(pdg=args.pdg, corr=["on"], kok=False,
                                force=args.force)
        hp.cmd_export(ns)
        if args.nbin:
            for pdg in args.pdg:
                p = _mp(pdg, arm)
                os.replace(p, p + f".nbin{args.nbin}")
                print(f"  -> {p}.nbin{args.nbin}")
    bp.MODEL_SUFFIX, SD_ENV = "", {}


def _mp(pdg, arm=None):
    """model path for an arm ('' / None = the ARCHIVED nominal model)."""
    p = bp._orig_model_path(pdg, "on")
    return p if not arm else p[:-5] + ARM_SUFFIX[arm] + ".root"


# =========================================================================
# 2.  bit-identity, and liveness
# =========================================================================

def _cmp(a, b):
    da, db = ds._branch_digests(a), ds._branch_digests(b)
    keys = sorted(set(da) | set(db))
    diff = [k for k in keys if da.get(k) != db.get(k)]
    return len(keys), diff


def cmd_bitid(args):
    print("=" * 122)
    print("BIT-IDENTITY OF THE OFF PATH, AND LIVENESS OF THE ON PATH")
    print("The branch COUNT is printed on every line and a file with no TTree "
          "is a hard failure (deltaspec._branch_digests).")
    print("=" * 122)

    print("\n--- control 1: ARCHIVED model vs a FRESH export with the switch "
          "OFF.")
    print("    The archived files predate this change.  Anything but 0 "
          "different means the switch leaks when unset,")
    print("    or something else moved the model in the meantime.")
    print(f"  {'species':<8}{'branches':>10}{'identical':>11}"
          f"{'different':>11}   verdict")
    nbad = 0
    for pdg in args.pdg:
        a, b = _mp(pdg), _mp(pdg, "spdoff")
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

    print("\n--- control 2: the correction is IDENTICALLY ZERO for a MUON "
          "(own table) and for a PROTON/ANTIPROTON")
    print("    (the table IS theirs, and the reference mass is the table "
          "particle's own PDG mass).  Those four must")
    print("    be bit-identical with the switch ON.  --- and the LIVE row: "
          "pi and K must move.")
    print(f"  {'species':<8}{'branches':>10}{'identical':>11}"
          f"{'different':>11}   verdict")
    bad = []
    for pdg in args.pdg:
        a, b = _mp(pdg), _mp(pdg, "spd")
        if not os.path.exists(b):
            print(f"  {hp.SPECIES[pdg]['label']:<8} (missing {b})")
            continue
        n, diff = _cmp(a, b)
        null = abs(pdg) in (13, 2212)
        if null and diff:
            ok = "FAIL -- a null moved: " + ",".join(diff[:6])
            bad.append(pdg)
        elif null:
            ok = "BIT-IDENTICAL (as required)"
        elif not diff:
            ok = "FAIL -- the switch is DEAD for this species"
            bad.append(pdg)
        else:
            ok = "LIVE: " + ",".join(sorted(diff))
        print(f"  {hp.SPECIES[pdg]['label']:<8}{n:>10}{n - len(diff):>11}"
              f"{len(diff):>11}   {ok}")
    print(f"    -> {'PASS' if not bad else 'FAIL'}")


# =========================================================================
# 3.  the reference shift, against NOTES_PION's parameter-free offline gauge
# =========================================================================

def _refE(pdg, arm, k=None):
    legs = cpt.load_model(_mp(pdg, arm))
    k = len(legs) - 1 if k is None else k
    P = 1e3 / abs(legs[k]["refqop"])
    return math.sqrt(P * P + hp.SPECIES[pdg]["mass"] ** 2), legs


def cmd_refdiff(args):
    """Is the C++ shift the SAME number NOTES_PION's offline gauge predicted?

    That note shifted the exported reference OFFLINE by
    `sum_s xi_s ln(Tmax_p(bg)/Tmax_species(bg))`, every factor taken from the
    exported record itself.  The C++ switch instead corrects the lookup and
    re-runs the whole Geant4e propagation -- stepping, half-step rescaling,
    range table, inverse range -- none of which the offline formula models.
    Agreement is a cross-check, not a tautology."""
    print("=" * 124)
    print("THE C++ SPECIES-CORRECT REFERENCE AGAINST NOTES_PION's "
          "PARAMETER-FREE OFFLINE GAUGE")
    print("dE = E(species) - E(nominal) at the outermost plane; `offline` = "
          "sum_s xi_s ln(Tmax_p/Tmax_sp) from the record.")
    print("Both MeV.  Nothing is fitted on either side.")
    print("=" * 124)
    print(f"  {'species':<8}{'C++ (lookup)':>16}{'offline':>14}"
          f"{'difference':>14}{'relative':>12}{'dp/p':>12}")
    for pdg in args.pdg:
        sp = hp.SPECIES[pdg]
        if not os.path.exists(_mp(pdg, "spd")):
            print(f"  {sp['label']:<8} (missing {_mp(pdg, 'spd')})")
            continue
        E0, legs0 = _refE(pdg, "spdoff")
        E1, _ = _refE(pdg, "spd")
        dE = E1 - E0
        _, cum, _, _ = pp.tmax_defect_per_leg(pdg)
        off = cum[-1]
        P = 1e3 / abs(legs0[-1]["refqop"])
        rel = (dE - off) / off if off != 0. else float("nan")
        dpp = abs(dE) * math.sqrt(P * P + sp["mass"] ** 2) / P / P
        print(f"  {sp['label']:<8}{dE:16.6f}{off:14.6f}{dE - off:14.6f}"
              f"{rel:12.2e}{dpp:12.3e}")


def cmd_refprof(args):
    print("=" * 104)
    print("REFERENCE SHIFT PER PLANE, MeV (species-correct minus nominal).  "
          "It must ACCUMULATE along the trajectory.")
    print("=" * 104)
    for pdg in args.pdg:
        sp = hp.SPECIES[pdg]
        if not os.path.exists(_mp(pdg, "spd")):
            continue
        l0 = cpt.load_model(_mp(pdg, "spdoff"))
        l1 = cpt.load_model(_mp(pdg, "spd"))
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

    `s_F` is RECOMPUTED from the arm's own model -- the reference moved, so the
    linearization the scale is built on moved with it -- and printed, so a
    switch that failed to reach the Fisher inversion would be visible instead
    of silent.  All three caches bypassed exactly as `hadron_probe._rows`."""
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
        ctr.IONI_KOKOULIN = 0.0
        fn._SCALE_CACHE.clear()
        cpt._PHI_CACHE.clear()
        fn._scale_cache_load, fn._scale_cache_store = old_load, old_store
    return dict(m=rows.mean(axis=0), err=err, sF=float(np.mean(sc["sF"])))


def _closure(args, arms):
    print("=" * 140)
    print("PER-SPECIES CLOSURE.  qop, plane mean, Fisher normalization, arm "
          "`off`, exact-delta ON, Kokoulin species-correct,")
    print("200k events.  `nominal` MUST reproduce NOTES_HADRONS s4 / "
          "NOTES_PION s1 digit for digit before anything else is read.")
    print("=" * 140)
    print(f"  {'species':<8}{'arm':<8}" + hp.UROW + "      rms        s_F")
    out = {}
    for pdg in args.pdg:
        sp = hp.SPECIES[pdg]
        for arm in arms:
            if not os.path.exists(_mp(pdg, arm) if arm else _mp(pdg)):
                print(f"  {sp['label']:<8}{arm:<8} (missing)")
                continue
            r = _rows_arm(pdg, arm, "off")
            out[(pdg, arm)] = r
            lab = {"": "nominal", "spdoff": "spdoff", "spd": "spd",
                   "ca": "ca", "both": "both"}[arm or ""]
            print(f"  {sp['label']:<8}{lab:<8}" + hp._fmt(r["m"])
                  + f" {hp._rms(r['m']):9.5f}  {r['sF']:.8e}")
        # published control
        pub, prms = pp.PUBLISHED[pdg]
        print(f"  {'':<8}{'PUBLISHED':<8}" + hp._fmt(np.asarray(pub))
              + f" {prms:9.5f}")
        print()
    return out


def cmd_closure(args):
    out = _closure(args, [None, "spdoff", "spd"])
    print("  SUMMARY")
    print(f"  {'species':<8}{'nominal':>10}{'spdoff':>10}{'spd':>10}"
          f"{'change':>10}")
    for pdg in args.pdg:
        r = [out.get((pdg, a)) for a in (None, "spdoff", "spd")]
        v = [hp._rms(x["m"]) if x else float("nan") for x in r]
        print(f"  {hp.SPECIES[pdg]['label']:<8}{v[0]:10.5f}{v[1]:10.5f}"
              f"{v[2]:10.5f}{v[2]-v[1]:10.5f}")


def cmd_both(args):
    out = _closure(args, [None, "spd", "ca", "both"])
    print("  SUMMARY -- the composition")
    print(f"  {'species':<8}{'nominal':>10}{'spd':>10}{'ca':>10}{'both':>10}")
    for pdg in args.pdg:
        r = [out.get((pdg, a)) for a in (None, "spd", "ca", "both")]
        v = [hp._rms(x["m"]) if x else float("nan") for x in r]
        print(f"  {hp.SPECIES[pdg]['label']:<8}" + "".join(f"{x:10.5f}"
                                                           for x in v))


# =========================================================================
# 5.  the registries
# =========================================================================

def cmd_registry(args):
    """The offline cache keys must move when this knob does.

    NOTES_BARKAS s9.2 made `cf_propagation_test._phi_key` and
    `fisher_norm.scale_identity` registry-driven, and `fisher_norm._MODEL_CACHE`
    sha256-verified, so that a NEW physics switch could not be silently absent
    from a cache key.  NOTES_CHARGEODD s10 was the first test of that claim;
    this is the second, and it is not inherited."""
    import shutil

    print("=" * 112)
    print("1. UNDECLARED-KNOB AUDIT: every module-level numeric global must be "
          "in PHYSICS_GLOBALS or _NOT_PHYSICS.")
    print("=" * 112)
    rows, bad = bp._knob_audit()
    print(f"   modules audited: {', '.join(bp._KNOB_MODULES)}   "
          f"globals: {len(rows)}")
    print(f"   undeclared knobs: {'NONE' if not bad else ', '.join(bad)}")

    print()
    print("=" * 112)
    print("2. THE s_F CACHE KEY moves with this knob.  `fisher_norm."
          "scale_identity` hashes the MODEL FILE together")
    print("   with every declared physics global, so a re-export cannot serve "
          "a stale 1/I.  Note what the file hash")
    print("   is NOT: it moves for a MUON too, whose 27 branches are "
          "bit-identical, because ROOT embeds a UUID and")
    print("   per-write metadata.  The key OVER-invalidates, which is the safe "
          "direction, and it is not a content")
    print("   comparator -- `bitid`'s branch digests are.")
    print("=" * 112)
    print(f"   {'species':<7}  {'spdoff key':<18}{'species key':<18}"
          f"{'branches':>10}  verdict")
    nbad = 0
    for pdg in args.pdg:
        a_, b_ = _mp(pdg, "spdoff"), _mp(pdg, "spd")
        if not (os.path.exists(a_) and os.path.exists(b_)):
            continue
        ka = []
        for path in (a_, b_):
            fn._MODEL_CACHE.clear()
            legs = fn.load(path)
            ka.append(fn.scale_identity(legs, "qop", 1e-8, 1 << 17, 32, -60.,
                                        ("ioni", "ms", "rad"), None)["_sha"])
        fn._MODEL_CACHE.clear()
        n, diff = _cmp(a_, b_)
        moved = ka[0] != ka[1]
        nbad += 0 if moved else 1
        ok = ("MOVED; content moved too" if (moved and diff) else
              "MOVED although the content did NOT (see above)" if moved else
              "*** STALE KEY -- FAIL ***")
        print(f"   {hp.SPECIES[pdg]['label']:<7}  {ka[0][:16]:<18}"
              f"{ka[1][:16]:<18}{n - len(diff):>5}/{n:<4}  {ok}")
    print(f"   -> {'PASS' if nbad == 0 else 'FAIL'}: no cached 1/I can survive "
          f"this knob")

    print()
    print("3. a STALE model-cache hit RAISES.  Forced by loading a model, then "
          "swapping the file underneath it.")
    src, alt = _mp(args.pdg[0], "spdoff"), _mp(args.pdg[0], "spd")
    tmp = os.path.join(SCRATCH, "sd_stale_probe.root")
    shutil.copyfile(src, tmp)
    try:
        fn.load(tmp)
        shutil.copyfile(alt, tmp)
        try:
            fn.load(tmp)
            print("   -> FAIL: the stale hit was served silently")
        except Exception as e:                          # noqa: BLE001
            print(f"   -> PASS: {type(e).__name__}: "
                  f"{str(e).splitlines()[0][:90]}")
    finally:
        fn._MODEL_CACHE.clear()
        if os.path.exists(tmp):
            os.remove(tmp)

    print()
    print("4. WHERE THIS KNOB DOES *NOT* BELONG, and why that is not an "
          "omission.  `PHYSICS_GLOBALS` registers the")
    print("   OFFLINE model's physics state.  CVH_REF_SPECIESDEDX changes the "
          "C++ reference and nothing in the")
    print("   offline model: there is no offline consumer to key, and the "
          "reference reaches the offline chain only")
    print("   through the model FILE, whose content identity is already in "
          "both keys (items 2 and 3).  A python")
    print("   global that nothing reads would be a decoration.  Same "
          "conclusion as NOTES_CHARGEODD s10, re-tested.")


# =========================================================================
# 6.  real tracks
# =========================================================================

CVH_ENV = {
    # the identity control -- a second run of the SAME configuration
    "nominal":   {},
    "nominal2":  {},
    # THE FLOOR, on the same code path.  CVH_ELOSS_CYL_* adds `eps` to the
    # log-scale mean-loss offset of every step inside a cylinder; put the
    # cylinder outside the whole detector and every step's reference mean loss
    # is multiplied by exp(1e-9).  That is a null perturbation of the MEAN --
    # which is what this change is -- and unlike NOTES_CHARGEODD's
    # CVH_DEDX_SCALE it is species-independent: CVH_DEDX_SCALE lives inside
    # ComputeMuonDEDX and is not a floor for a pion at all.
    "meanfloor": {"CVH_ELOSS_CYL_R": "100000", "CVH_ELOSS_CYL_Z": "100000",
                  "CVH_ELOSS_CYL_EPS": "1e-9"},
    # and a GAUGE 1000x above the floor, so LINEARITY is measured rather than
    # assumed (NOTES_CHARGEODD s7.4 found jacref align and gradv still
    # floor-dominated at 1000x, and said so).
    "meangauge": {"CVH_ELOSS_CYL_R": "100000", "CVH_ELOSS_CYL_Z": "100000",
                  "CVH_ELOSS_CYL_EPS": "1e-6"},
    "spd":       {"CVH_REF_SPECIESDEDX": "1"},
    "ca":        {"CVH_REF_CHARGEAWARE": "1"},
    "both":      {"CVH_REF_CHARGEAWARE": "1", "CVH_REF_SPECIESDEDX": "1"},
}


def _co():
    """chargeodd's real-track machinery, pointed at this note's directory.

    Imported LAZILY and only here: `chargeodd` re-patches `hadron_probe._env`
    and `sim_glob` at import time, which would fight this module's own patches
    in the toy-model commands.  Nothing in `cvh` / `cvhcmp` touches those."""
    import chargeodd as co
    co.CO = SD
    co.CVHENV = CVH_ENV
    # the V0 single-track maker writes globalcor_v0single_<stream>.root
    qv.PREFIX.setdefault("runCvhSingleTrackV0.py", "globalcor_v0single")
    return co


def cmd_cvh(args):
    co = _co()
    ns = argparse.Namespace(tag=args.tag, script=args.script, input=args.input,
                            nev=args.nev, extra=args.extra,
                            labels=args.labels, force=args.force)
    co.cmd_cvh(ns)


def cmd_cvhcmp(args):
    co = _co()
    print("=" * 112)
    print("DOWNSTREAM OF THE *REFERENCE*, ON REAL TRACKS")
    print("This changes the MEAN, i.e. the linearization point of every "
          "transport Jacobian, and it is charge-EVEN,")
    print("so unlike the Barkas/Mott term it does not cancel between "
          "conjugates.  Read the MEDIAN; `pooled` is a")
    print("tail statistic that understates signal-over-floor by up to 100x on "
          "this class of response (NOTES_QVALID 7.1).")
    print("=" * 112)
    ns = argparse.Namespace(tag=args.tag, script=args.script,
                            labels=args.labels, split=args.split, gen=args.gen)
    co.cmd_cvhcmp(ns)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def _pdg(p, default=ORDER8):
        p.add_argument("--pdg", type=int, nargs="+", default=list(default))

    p = sub.add_parser("table"); p.set_defaults(f=cmd_table)
    p = sub.add_parser("spinterm"); p.set_defaults(f=cmd_spinterm)
    p = sub.add_parser("branch")
    p.add_argument("--minfrac", type=float, default=0.005)
    p.set_defaults(f=cmd_branch)
    p = sub.add_parser("nbin"); _pdg(p, (-211, 211, -321))
    p.add_argument("--nbin", type=int, nargs="+", default=[4, 16, 64])
    p.set_defaults(f=cmd_nbin)
    p = sub.add_parser("export"); _pdg(p)
    p.add_argument("--arms", nargs="+", default=["spdoff", "spd"])
    p.add_argument("--nbin", type=int, default=0)
    p.add_argument("--force", action="store_true"); p.set_defaults(f=cmd_export)
    p = sub.add_parser("bitid"); _pdg(p); p.set_defaults(f=cmd_bitid)
    p = sub.add_parser("refdiff"); _pdg(p); p.set_defaults(f=cmd_refdiff)
    p = sub.add_parser("refprof"); _pdg(p); p.set_defaults(f=cmd_refprof)
    p = sub.add_parser("closure"); _pdg(p); p.set_defaults(f=cmd_closure)
    p = sub.add_parser("both"); _pdg(p); p.set_defaults(f=cmd_both)
    p = sub.add_parser("registry"); _pdg(p); p.set_defaults(f=cmd_registry)
    p = sub.add_parser("cvh")
    p.add_argument("--tag", required=True)
    p.add_argument("--script", default="runCvhSingleTrack.py")
    p.add_argument("--input", required=True)
    p.add_argument("--nev", type=int, default=1200)
    p.add_argument("--extra", default="")
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--force", action="store_true")
    p.set_defaults(f=cmd_cvh)
    p = sub.add_parser("cvhcmp")
    p.add_argument("--tag", required=True)
    p.add_argument("--script", default="runCvhSingleTrack.py")
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--split", action="store_true", default=False)
    p.add_argument("--gen", action="store_true")
    p.set_defaults(f=cmd_cvhcmp)

    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
