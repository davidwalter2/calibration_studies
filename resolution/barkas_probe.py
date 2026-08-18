#!/usr/bin/env python3
"""Is the BARKAS-ANDERSEN term the source of the charge-odd energy-loss bias,
and is the Geant4e extrapolator's dE/dx table missing it?

WHY THIS EXISTS
---------------
NOTES_HADRONS s6 measured, in the layered toy at pT = 3, that

  * the SIMULATION's mean ionization loss is charge-ODD by 0.17-0.36 % (in the
    MEDIAN, so not a tail artifact), positive always losing more;
  * the CVH REFERENCE TRAJECTORY is charge-EVEN to 1.7e-9;
  * the charge-odd part of the reference's error equals the simulation's own
    charge-odd loss species by species, and removing it takes K- from 0.00311
    to 0.00036.

and attributed it to "Geant4's Barkas/Bloch z^3 term
(G4EmCorrections::HighOrderCorrections)".  That attribution was never
evaluated: no term of that expansion had been computed at the toy's
kinematics, and no switch had been thrown.  This module does both.

WHAT IS ACTUALLY IN THE EXPANSION
---------------------------------
G4EmCorrections::HighOrderCorrections is THREE terms, not one, and only two of
them are odd in the projectile charge:

    sum  = 2.0*(Barkas + Bloch) + Mott;
    sum *= material->GetElectronDensity()*q2*twopi_mc2_rcl2/beta2;

    Barkas  Ashley-Ritchie z^3 polarization term,  ~ 1.29 * charge * f(beta),
            ODD in z, FALLS with beta -- the Barkas-Andersen effect proper
    Bloch   -y2*term, y2 = q2/ba2                  EVEN in z (a function of
            q^2 only), so it cannot make a charge asymmetry at all
    Mott    pi * alpha * beta * charge             ODD in z, GROWS with beta
            -- the z^3 Mott term of Ahlen's expansion

The brief's hypothesis names the first.  The measurement in s3 below shows the
charge-odd effect is the THIRD, by a factor 150.

THE SWITCH
----------
Geant4 has no UI command, no G4EmParameters flag and no physics-list hook for
any of these (`SetUseMottCorrection` is G4GSMottCorrection, for e- multiple
scattering, not this).  The switch is made at the dynamic linker instead
(`barkas_shim.cc`): the calls from HighOrderCorrections into the three term
functions go through the PLT (verified with objdump -R -- R_X86_64_JUMP_SLOT),
so an LD_PRELOAD definition of the same mangled symbols is bound first.  The
shim always calls the original through RTLD_NEXT and prints a call census at
exit, so its own liveness is a number in the log.

Five arms, and two of them are the controls that make the other three mean
something:

    tee        shim loaded, nothing suppressed  -> must be BIT-IDENTICAL.
               Proves the shim is not itself the effect.
    nobarkas   Barkas -> 0                      -> the literal hypothesis
    nomott     Mott   -> 0                      -> the term the numbers point at
    nohoc      Barkas -> 0 AND Mott -> 0        -> the whole charge-odd part
    nobloch    Bloch  -> 0                      -> the EVEN control: must move
               the loss and NOT move the charge asymmetry

IT DOES NOT REACH THE CMSSW SIMULATION, and that is a fact about the build:
Geant4 is linked STATICALLY into biglib/pluginSimulation.so with LOCAL (`t`)
symbols and MottCorrection inlined away, so nothing can interpose it.  Shown
directly -- the shim loads into cmsRun, its census reads `Barkas n=0`, and the
census file is md5-identical across all four arms.  The simulation side is
therefore established by MEASUREMENT (`predict`, 1.0 M events per species) and
the fix is demonstrated on the REFERENCE, which is dynamically linked
(`export`, `refshift`, `refxcheck`).

CACHES
------
Inherited from hadron_probe, which is imported rather than copied:
RES_NO_PHI_CACHE is set at ITS import, the in-process and ON-DISK scale caches
are bypassed inside `_rows`, and `_PHI_CACHE` is cleared between arms.  As of
2026-08-16 that is a belt over a correct key rather than the only defence:
`_phi_key` and `fisher_norm.scale_identity` are now built from each module's
own PHYSICS_GLOBALS registry (so IONI_KOKOULIN, RAD_CHANNEL, G4_FF_SQUARED and
G4_SCREEN_F are in them), and `cf_track_resolution._kokoulin_exponent` has the
|PDG| == 13 guard the C++ always had.  `guards` re-tests both.

SUBCOMMANDS
    build     the driver + the shim
    dedx      Geant4's own dE/dx per species PER SIGN, the Barkas/Bloch/Mott
              decomposition, the knock-on cross section (charge-even), and the
              CVH extrapolator's table for the same eight particles
    scan      the beta*gamma dependence: which of the two odd terms is it
    shimlive  the shim, shown live and shown inert, in the driver
    sim       simulations (`more` = extra statistics on new seeds)
    census    the charge-odd loss, per species, per arm, with errors
    predict   prediction vs measurement, per species
    export    the models, with and without the shim
    refshift  THE RESULT: the reference made charge-aware, parameter-free
    refxcheck the offline shift against Geant4's own, via the shimmed export
    guards    the Kokoulin PDG guard and the cache keys, self-tested
    closure   the closure with the charge-odd term removed from BOTH sides
"""

import argparse
import glob
import hashlib
import math
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# hadron_probe sets RES_NO_PHI_CACHE=1 at import, before anything can populate
# the phi cache.  Importing it (rather than copying it) is what makes the
# controls in s2 of NOTES_HADRONS apply here unchanged.
import hadron_probe as hp                                       # noqa: E402
import cf_propagation_test as cpt                               # noqa: E402
import tail_probe as tp                                         # noqa: E402
import deltaspec as ds                                          # noqa: E402

SCRATCH = hp.SCRATCH
OUT = hp.OUT                       # shares NOTES_HADRONS' directory; the arm
                                   # names are new so no file is ever clobbered
DRIVER = os.path.join(SCRATCH, "barkas_g4driver.sh")
SHIM = os.path.join(SCRATCH, "libbarkasshim.so")
BUILD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_barkas.sh")

# the toy step, from NOTES_HADRONS s0.2 / the exported regime-2/3 record
TOY = dict(Z=8.0, A=16.0, rho=9.0, p=3136.0, tcut=0.0094862)

ORDER8 = [13, -13, -211, 211, -321, 321, -2212, 2212]
PAIRS = [(13, -13, "mu"), (-211, 211, "pi"), (-321, 321, "K"),
         (-2212, 2212, "p")]

# ------------------------------------------------------------------ the arms
# Every new arm inactivates exactly what `off` inactivates (nuclear + Decay),
# so it is like-for-like with the published clean arm; the only difference is
# the LD_PRELOAD overlay.
ARM_BASE = {"tee": "off", "nobarkas": "off", "nomott": "off", "nohoc": "off",
            "nobloch": "off",
            # `more` carries NO shim: it is simply the clean arm again on fresh
            # seeds, so the published 200k `off` sample is never touched and
            # the NOTES_HADRONS controls still reproduce, while the charge-odd
            # difference -- which is a 0.3 % effect on a Landau-tailed
            # distribution -- can be measured on 1.0 M events.
            "more": "off"}
ARM_SHIM = {
    "tee":      {},
    "nobarkas": {"CVH_SHIM_BARKAS_OFF": "1"},
    "nomott":   {"CVH_SHIM_MOTT_OFF": "1"},
    "nohoc":    {"CVH_SHIM_BARKAS_OFF": "1", "CVH_SHIM_MOTT_OFF": "1"},
    "nobloch":  {"CVH_SHIM_BLOCH_OFF": "1"},
}

_orig_inact = hp.inact_of
_orig_env = hp._env
_orig_model_path = hp.model_path

MODEL_SUFFIX = ""      # set by cmd_export / cmd_closure for the shimmed models
ENV_OVERLAY = {}       # set by cmd_export: the shim for a MODEL export


def inact_of(pdg, arm):
    return _orig_inact(pdg, ARM_BASE.get(arm, arm))


def _env(pdg, arm, extra=None):
    e = _orig_env(pdg, arm, extra)
    if arm in ARM_SHIM or ENV_OVERLAY:
        e["LD_PRELOAD"] = SHIM
    e.update(ARM_SHIM.get(arm, {}))
    e.update(ENV_OVERLAY)
    return e


def model_path(pdg, corr, kok=False):
    p = _orig_model_path(pdg, corr, kok)
    return (p[:-5] + MODEL_SUFFIX + ".root") if MODEL_SUFFIX else p


hp.inact_of = inact_of
hp._env = _env
hp.model_path = model_path
hp.ARMS = dict(hp.ARMS, **{k: f"clean arm + shim: {v or 'nothing suppressed'}"
                           for k, v in ARM_SHIM.items()})


# =========================================================================
# 0.  build
# =========================================================================

def cmd_build(args):
    rc = subprocess.run(["bash", BUILD], check=False).returncode
    if rc:
        raise SystemExit("build failed")
    for p in (DRIVER, SHIM):
        print(f"  {hashlib.md5(open(p,'rb').read()).hexdigest()}  {p}")


def _driver(extra=(), env=None):
    e = dict(os.environ)
    # The four default-on corrections are pinned to their HISTORICAL state
    # (all off) and the arm's own overlay wins -- the same convention, and for
    # the same reason, as deltaspec._clean_env.  Without this every `off` arm
    # here would silently become an `on` arm.
    import cf_track_resolution as _ctr
    e.update(_ctr.SWITCHES_OFF)
    e.update(env or {})
    cmd = [DRIVER, "--Z", repr(TOY["Z"]), "--A", repr(TOY["A"]),
           "--rho", repr(TOY["rho"]), "--p", repr(TOY["p"]),
           "--tcut", repr(TOY["tcut"])] + list(extra)
    r = subprocess.run(cmd, capture_output=True, text=True, env=e)
    if r.returncode:
        sys.stderr.write(r.stderr)
        raise SystemExit(f"driver failed rc={r.returncode}")
    return r.stdout, r.stderr


def _parse_species(txt):
    out = {}
    mat = {}
    for line in txt.splitlines():
        f = line.split()
        if not f:
            continue
        if f[0] == "MAT":
            # MAT name <n> Z <z> A <a> rho <r> ipot_eV <i> nel_percm3 <n>
            mat = dict(ipot_eV=float(f[10]), nel=float(f[12]))
        if f[0] != "SPECIES":
            continue
        out[int(f[1])] = dict(
            lab=f[2], mass=float(f[3]), ekin=float(f[4]), beta=float(f[5]),
            beta2=float(f[6]), bg=float(f[7]), tmax=float(f[8]), model=f[9],
            dedxR=float(f[10]), dedxU=float(f[11]), pref=float(f[12]),
            hoc=float(f[13]), barkas=float(f[14]), bloch=float(f[15]),
            mott=float(f[16]), Lrest=float(f[17]),
            # OFF-BY-ONE, FIXED 2026-08-16 (NOTES_CHARGEODD).  The driver
            # prints TWENTY fields -- it appends `xs`
            # (CrossSectionPerVolume, driver line 221) BEFORE `extrap`
            # (G4EnergyLossForExtrapolatorForCVH::ComputeDEDX, line 231) --
            # and this mapping read `extrap` from index 18, i.e. from `xs`.
            # `dedx`'s "extrapolator dE/dx" column therefore printed the
            # knock-on cross section, 7.2952331406579649 for the muon where
            # the extrapolator gives 1.7375069849772742, and its "vs G4 for
            # THIS charge" column read 3.2-4.1 (a 320 % difference, which is
            # the tell).  s2.3 of NOTES_BARKAS quotes the CORRECT extrapolator
            # values, so the note's numbers stand; it is the printer that
            # drifted.  The charge-conjugate column read 0.00e+00 either way,
            # because BOTH quantities are charge-even.
            xs=float(f[18]), extrap=float(f[19]))
    return out, mat


# =========================================================================
# 1.  what Geant4 and the extrapolator each do
# =========================================================================

def cmd_dedx(args):
    txt, _ = _driver()
    sp, mat = _parse_species(txt)
    print("=" * 126)
    print("GEANT4'S OWN MEAN STOPPING POWER, PER SIGN, at the toy step "
          f"(Z = {TOY['Z']:g}, A = {TOY['A']:g}, rho = {TOY['rho']:g} g/cm3, "
          f"I = {mat['ipot_eV']:.4g} eV,")
    print(f"p = {TOY['p']:g} MeV/c, tcut = {TOY['tcut']*1e3:.4f} keV).  "
          "dE/dx in MeV/mm.  Nothing here is fitted or simulated: these are "
          "the models' own values.")
    print("=" * 126)
    print(f"  {'species':<6}{'bg':>8}{'beta':>10}{'model':>22}"
          f"{'dE/dx|tcut':>13}{'dE/dx|unrestr':>15}{'HighOrderCorr':>15}"
          f"{'2*Barkas':>12}{'2*Bloch':>11}{'Mott':>11}")
    for pdg in ORDER8:
        s = sp[pdg]
        print(f"  {s['lab']:<6}{s['bg']:8.3f}{s['beta']:10.6f}"
              f"{s['model']:>22}{s['dedxR']:13.6f}{s['dedxU']:15.6f}"
              f"{s['hoc']:15.3e}{2*s['barkas']:12.3e}{2*s['bloch']:11.3e}"
              f"{s['mott']:11.3e}")

    print()
    print("THE CHARGE-ODD PART, term by term.  Barkas and Mott are odd, Bloch "
          "is even (it depends on q^2 only),")
    print("so the +/- difference of the restricted dE/dx is "
          "2*pref*(2*Barkas + Mott) exactly.")
    print(f"  {'species':<6}{'bg':>8}{'dE/dx(+)':>12}{'dE/dx(-)':>12}"
          f"{'difference':>13}{'fraction':>11}   {'from Barkas':>12}"
          f"{'from Mott':>11}")
    for m, p, lab in PAIRS:
        a, b = sp[p], sp[m]                       # a = positive, b = negative
        d = a["dedxR"] - b["dedxR"]
        avg = 0.5 * (a["dedxR"] + b["dedxR"])
        # decomposition of the difference into its two odd terms
        dB = 2 * a["pref"] * 2 * a["barkas"]
        dM = 2 * a["pref"] * a["mott"]
        print(f"  {lab:<6}{a['bg']:8.3f}{a['dedxR']:12.6f}{b['dedxR']:12.6f}"
              f"{d:13.3e}{100*d/avg:10.4f} %{100*dB/d:11.2f} %{100*dM/d:10.2f} %")

    print()
    print("THE EXTRAPOLATOR'S TABLE -- G4EnergyLossForExtrapolatorForCVH::"
          "ComputeDEDX, the dE/dx the CVH")
    print("reference trajectory actually integrates.  Same eight particles, "
          "same material, same energy.")
    print(f"  {'species':<6}{'extrapolator dE/dx':>24}   {'vs its charge conjugate':>26}"
          f"   {'vs G4 for THIS charge':>24}")
    for m, p, lab in PAIRS:
        a, b = sp[p], sp[m]
        rel = (a["extrap"] - b["extrap"]) / a["extrap"]
        print(f"  {a['lab']:<6}{a['extrap']:24.17g}   {rel:26.2e}"
              f"   {(a['extrap']-a['dedxU'])/a['dedxU']:24.3e}")
        print(f"  {b['lab']:<6}{b['extrap']:24.17g}   {'':>26}"
              f"   {(b['extrap']-b['dedxU'])/b['dedxU']:24.3e}")
    print()
    print("  (`vs G4 for THIS charge` compares the table against the "
          "UNRESTRICTED dE/dx of that particle.")
    print("   The muon table also carries the radiative mean -- brems + pair "
          "-- so its offset is not zero.)")


def cmd_scan(args):
    txt, _ = _driver(["--scan"])
    rows = {}
    for line in txt.splitlines():
        f = line.split()
        if f and f[0] == "SCAN":
            rows.setdefault(int(f[1]), []).append(
                [float(x) for x in f[2:]])
    print("=" * 108)
    print("beta*gamma DEPENDENCE OF THE TWO ODD TERMS.  The Barkas-Andersen "
          "term falls with beta; the Mott term")
    print("grows like beta.  `oddfrac` is the +/- fractional difference of the "
          "restricted dE/dx at that bg.")
    print("=" * 108)
    lab = {13: "mu-", -321: "K-", -2212: "pbar"}
    for pdg in (13, -321, -2212):
        if pdg not in rows:
            continue
        print(f"\n  {lab[pdg]}")
        print(f"    {'bg':>8}{'ekin':>11}{'2*Barkas':>13}{'Mott':>13}"
              f"{'Barkas/(odd)':>15}{'oddfrac':>12}")
        a = np.array(rows[pdg])
        for r in a[::4]:
            bg, ekin, barkas, bloch, mott, hoc, dedxR, oddfrac = r
            odd = 2 * barkas + mott
            print(f"    {bg:8.3f}{ekin:11.4g}{2*barkas:13.3e}{mott:13.3e}"
                  f"{100*2*barkas/odd:14.2f} %{100*oddfrac:11.4f} %")


# =========================================================================
# 2.  the shim, shown live
# =========================================================================

def cmd_shimlive(args):
    """The shim is a switch, so it is a measurement until proven otherwise.

    Three states in one process each, all through the DRIVER (deterministic,
    seconds, no simulation): loaded-and-inert must reproduce the unshimmed
    numbers bit for bit, and each suppression must move exactly the term it
    names and nothing else."""
    print("=" * 118)
    print("SHIM LIVENESS.  The call census is printed by the shim itself at "
          "exit; `passed` means it called the")
    print("original and returned it unchanged, `SUPPRESSED` means it returned "
          "0.  Bit-identity is over the full")
    print("17-digit dE/dx of all eight species.")
    print("=" * 118)
    base, _ = _driver()
    states = [("(no shim)", None, {}),
              ("tee", SHIM, {}),
              ("nobarkas", SHIM, ARM_SHIM["nobarkas"]),
              ("nomott", SHIM, ARM_SHIM["nomott"]),
              ("nohoc", SHIM, ARM_SHIM["nohoc"]),
              ("nobloch", SHIM, ARM_SHIM["nobloch"])]
    ref = None
    for lab, pre, ev in states:
        env = dict(ev)
        if pre:
            env["LD_PRELOAD"] = pre
        txt, err = _driver(env=env)
        sp, _ = _parse_species(txt)
        dig = hashlib.md5(
            ";".join(f"{sp[p]['dedxR']!r}" for p in ORDER8).encode()).hexdigest()
        if ref is None:
            ref = dig
        cen = [l for l in err.splitlines() if "census" in l]
        odd = {}
        for m, p, nm in PAIRS:
            d = sp[p]["dedxR"] - sp[m]["dedxR"]
            odd[nm] = 100 * d / (0.5 * (sp[p]["dedxR"] + sp[m]["dedxR"]))
        print(f"\n  {lab:<10} digest {dig[:16]} "
              f"{'IDENTICAL to no-shim' if dig == ref else 'MOVED'}")
        print(f"    charge-odd dE/dx fraction:  "
              + "  ".join(f"{k} {v:+.5f} %" for k, v in odd.items()))
        for c in cen:
            print(f"    {c.strip()}")


# =========================================================================
# 3.  simulations
# =========================================================================

def cmd_sim(args):
    hp.cmd_sim(args)


def cmd_live(args):
    hp.cmd_live(args)


# =========================================================================
# 4.  the charge-odd loss in the SIMULATION
# =========================================================================

def _loc(c, col=tp.I_DETOT):
    """Median and its bootstrap error.  The median is the right estimator
    here and that is not a matter of taste: the total loss is
    dE = X + Y with X the restricted continuous loss and Y the explicit
    delta rays above the cut.  Only X carries a charge-odd term, and it
    carries it as a pure TRANSLATION of X's distribution, so EVERY quantile of
    dE shifts by exactly the same amount -- while the mean is dominated by Y's
    Landau tail (the muon's sample mean moves by 3x between 20k and 200k
    events, NOTES_HADRONS s3.3)."""
    x = c[:, col]
    med = float(np.median(x))
    # Order-statistic error of the median: the 0.5 +- 0.5/sqrt(N) quantiles
    # bracket it at 1 sigma for any continuous distribution (the binomial
    # variance of the rank is N/4), so this is distribution-free and, unlike a
    # bootstrap, deterministic.
    n = len(x)
    lo, hi = np.quantile(x, [0.5 - 0.5 / math.sqrt(n), 0.5 + 0.5 / math.sqrt(n)])
    return med, float(hi - lo) / 2.0, float(x.mean()), float(x.std() / math.sqrt(n))


def _census(pdg, arms):
    """One census over several arms.  Only ever used to POOL arms that are the
    same physics on different seeds (`off` + `more`); the arms are otherwise
    kept apart."""
    return np.concatenate([hp.census_of(pdg, a) for a in arms])


def _restricted(c):
    """The part of the loss that carries the charge-odd term: total minus what
    left as explicit secondaries above the production cut."""
    return float(c[:, tp.I_DETOT].mean() - c[:, tp.I_DESEC].mean())


def cmd_census(args):
    txt, _ = _driver()
    sp, _ = _parse_species(txt)
    print("=" * 126)
    print("THE SIMULATION'S CHARGE-ODD MEAN LOSS, per arm.  dE = primary "
          "energy loss inside r < 107 cm, MeV.")
    print("The MEDIAN is the estimator (see `_loc`): the charge-odd term "
          "translates the restricted-loss")
    print("distribution, so every quantile shifts by exactly it, while the "
          "mean is Landau-tail limited.")
    print("=" * 126)
    for arm in args.arms:
        print(f"\n  arm `{arm}`")
        print(f"    {'species':<7}{'median(-)':>11}{'median(+)':>11}"
              f"{'difference':>13}{'+-':>9}{'restricted':>12}"
              f"{'odd fraction':>14}{'predicted':>12}")
        for m, p, lab in PAIRS:
            try:
                cm = _census(m, arm.split("+"))
                cp = _census(p, arm.split("+"))
            except FileNotFoundError as e:
                print(f"    {lab:<7} missing: {e}")
                continue
            mm, em, _, _ = _loc(cm)
            mp, ep, _, _ = _loc(cp)
            d = mp - mm
            e = math.hypot(em, ep)
            rest = 0.5 * (_restricted(cm) + _restricted(cp))
            pred = (sp[p]["dedxR"] - sp[m]["dedxR"]) / \
                   (0.5 * (sp[p]["dedxR"] + sp[m]["dedxR"]))
            print(f"    {lab:<7}{mm:11.4f}{mp:11.4f}{d:+13.4f}{e:9.4f}"
                  f"{rest:12.4f}{100*d/rest:13.4f} %{100*pred:11.4f} %")


def cmd_predict(args):
    """The prediction-versus-measurement table the brief asks for.

    Prediction: the fractional +/- difference of Geant4's own RESTRICTED
    stopping power at the toy step, from `dedx`.  It has no free parameter.
    Measurement: the shift of the simulated loss distribution, divided by the
    restricted loss the census itself reports.  Both dimensionless, so no path
    length enters anywhere."""
    txt, _ = _driver()
    sp, _ = _parse_species(txt)
    print("=" * 122)
    print("PREDICTION vs MEASUREMENT, per species.  Prediction = "
          "[dE/dx(+) - dE/dx(-)]/<dE/dx> from Geant4's model at")
    print("the toy step; measurement = [median dE(+) - median dE(-)] / "
          "<restricted loss>, from the census.  Nothing fitted.")
    print("=" * 122)
    print(f"  {'species':<7}{'bg':>8}{'beta':>9}{'2*Barkas':>12}{'Mott':>12}"
          f"{'Barkas share':>14}  {'predicted':>11}{'measured':>11}{'+-':>9}"
          f"{'ratio':>8}")
    for m, p, lab in PAIRS:
        a, b = sp[p], sp[m]
        pred = (a["dedxR"] - b["dedxR"]) / (0.5 * (a["dedxR"] + b["dedxR"]))
        odd = 2 * a["barkas"] + a["mott"]
        try:
            cm = _census(m, args.arm.split("+"))
            cp = _census(p, args.arm.split("+"))
        except FileNotFoundError:
            print(f"  {lab:<7} census missing for arm {args.arm}")
            continue
        mm, em, _, _ = _loc(cm)
        mp, ep, _, _ = _loc(cp)
        rest = 0.5 * (_restricted(cm) + _restricted(cp))
        meas = (mp - mm) / rest
        emeas = math.hypot(em, ep) / rest
        print(f"  {lab:<7}{a['bg']:8.3f}{a['beta']:9.5f}{2*a['barkas']:12.3e}"
              f"{a['mott']:12.3e}{100*2*a['barkas']/odd:13.2f} % "
              f"{100*pred:10.4f} %{100*meas:10.4f} %{100*emeas:8.4f}"
              f"{meas/pred:8.2f}")


# =========================================================================
# 5.  models and closure with the term removed from BOTH sides
# =========================================================================

def cmd_export(args):
    global MODEL_SUFFIX, ENV_OVERLAY
    for arm in args.arms:
        MODEL_SUFFIX = "" if arm == "off" else "_" + arm
        ENV_OVERLAY = dict(ARM_SHIM.get(arm, {}))
        if arm != "off":
            ENV_OVERLAY["LD_PRELOAD"] = SHIM
        print(f"\n=== models for arm `{arm}`  suffix={MODEL_SUFFIX!r} "
              f"shim={ {k: v for k, v in ENV_OVERLAY.items() if k != 'LD_PRELOAD'} }")
        ns = argparse.Namespace(pdg=args.pdg, corr=["on"], kok=False,
                                force=args.force)
        hp.cmd_export(ns)
    MODEL_SUFFIX, ENV_OVERLAY = "", {}


# =========================================================================
# 4b. THE TWO GUARDS -- the Kokoulin PDG guard and the cache keys
#
# Both were found by NOTES_HADRONS and worked around rather than fixed; both
# are fixed now and both are re-tested here rather than asserted.
# =========================================================================

_KNOB_MODULES = ("cf_propagation_test", "cf_track_resolution", "cf_ms_exact",
                 "cf_nucel_exact", "hbasis")


def _knob_audit():
    """Every module-level name that LOOKS like a knob must be declared, in
    exactly one of `PHYSICS_GLOBALS` (goes into every cache key) or
    `_NOT_PHYSICS` (explicitly does not).  A name in neither is the failure
    mode this whole exercise exists to prevent: a new physics switch that no
    cache key knows about."""
    import importlib
    bad = []
    rows = []
    for name in _KNOB_MODULES:
        mod = importlib.import_module(name)
        phys = set(getattr(mod, "PHYSICS_GLOBALS", ()))
        notp = set(getattr(mod, "_NOT_PHYSICS", ()))
        for n in sorted(vars(mod)):
            if n.startswith("_") or n.upper() != n:
                continue
            v = getattr(mod, n)
            if not isinstance(v, (int, float, bool)):
                continue
            where = ("PHYSICS" if n in phys else
                     "not-physics" if n in notp else "UNDECLARED")
            if where == "UNDECLARED":
                bad.append(f"{name}.{n}")
            rows.append((name, n, repr(v), where))
    return rows, bad


def cmd_guards(args):
    import importlib
    import cf_track_resolution as ctr
    import cf_ms_exact as ms
    import fisher_norm as fn

    print("=" * 116)
    print("GUARD 1: the Kokoulin correction is MUON-ONLY, and the offline half "
          "now enforces it the way the C++ does.")
    print("=" * 116)
    print("  mass recovered from the record as m = E*sqrt(1-beta^2) "
          "(an identity; both columns are in the stride-13 record):")
    print(f"    {'species':<8}{'PDG mass':>12}{'from record':>16}"
          f"{'relative':>12}{'guard says':>14}")
    ok = True
    for pdg in ORDER8:
        legs, rows = hp._record(pdg, "on")
        b2 = np.array([r["beta2"] for r in rows])
        et = np.array([r["Etot"] for r in rows])
        m = ctr._mass_from_record(b2, et)
        ref = hp.SPECIES[pdg]["mass"]
        rel = float(np.max(np.abs(m - ref)) / ref)
        ismu = bool(ctr._is_muon_record(b2, et).all())
        want = abs(pdg) == 13
        ok &= (ismu == want)
        print(f"    {hp.SPECIES[pdg]['label']:<8}{ref:12.4f}{m.mean():16.4f}"
              f"{rel:12.1e}{('MUON' if ismu else 'not a muon'):>14}"
              + ("" if ismu == want else "   *** WRONG ***"))
    print(f"  species identification: {'ALL CORRECT' if ok else 'FAILED'}")

    print()
    print("  the guard, exercised: IONI_KOKOULIN = 1 on every species, and the "
          "model CF compared against")
    print("  IONI_KOKOULIN = 0.  A hadron must be BIT-IDENTICAL (the correction "
          "cannot be picked up at all);")
    print("  the muon must MOVE.  This is the failure that already occurred "
          "(NOTES_HADRONS s3.4: it made K- look 4x better).")
    print(f"    {'species':<8}{'max |dphi|':>14}{'steps suppressed':>19}"
          f"   {'verdict':<28}")
    for pdg in ORDER8:
        legs = cpt.load_model(hp.model_path(pdg, "on"))
        k = len(legs) - 1
        avec = cpt.FUNCTIONALS["qop"]
        tau = np.linspace(0.0, 30.0, 4001)
        sig = float(np.sqrt(cpt.model_variance(legs, k, avec)[0]))
        ctr.IONI_KOKOULIN = 0.0
        cpt._PHI_CACHE.clear()
        p0 = cpt.model_phi(legs, k, avec, sig, tau)
        n0 = ctr._KOK_SUPPRESSED["n"]
        ctr.IONI_KOKOULIN = 1.0
        cpt._PHI_CACHE.clear()
        p1 = cpt.model_phi(legs, k, avec, sig, tau)
        nsup = ctr._KOK_SUPPRESSED["n"] - n0
        ctr.IONI_KOKOULIN = 0.0
        cpt._PHI_CACHE.clear()
        d = float(np.max(np.abs(p1 - p0)))
        want_move = abs(pdg) == 13
        good = (d > 0) == want_move
        print(f"    {hp.SPECIES[pdg]['label']:<8}{d:14.6e}{nsup:19d}   "
              + ("MOVES (muon: correct)" if d > 0 else
                 "BIT-IDENTICAL (guard is live)")
              + ("" if good else "   *** WRONG ***"))

    print()
    print("=" * 116)
    print("GUARD 2: the cache keys.  Every physics knob, flipped one at a "
          "time, must move BOTH the phi-cache key")
    print("and the scale-cache identity.  A knob that moves neither is a "
          "silent stale hit waiting to happen.")
    print("=" * 116)
    rows, bad = _knob_audit()
    print(f"  {'module':<24}{'global':<24}{'value':<12}{'declared as':<14}")
    for mod, n, v, where in rows:
        print(f"  {mod:<24}{n:<24}{v:<12}{where:<14}")
    print(f"  undeclared knobs: {bad if bad else 'NONE'}")
    if bad:
        raise SystemExit("a module has a knob in neither PHYSICS_GLOBALS nor "
                         "_NOT_PHYSICS -- declare it")

    legs = cpt.load_model(hp.model_path(13, "on"))
    avec = cpt.FUNCTIONALS["qop"]
    tau = np.linspace(0.0, 30.0, 257)
    ident0 = fn.scale_identity(legs, "qop", 1e-12, 1 << 16, 4, 40.0,
                               ("ioni", "ms", "rad"), None)
    key0 = cpt._phi_key(legs, 0, avec, 1.0, tau)
    print()
    print(f"  {'knob':<40}{'phi key':>12}{'scale key':>12}")
    nbadkey = 0
    for name in _KNOB_MODULES:
        mod = importlib.import_module(name)
        for n in getattr(mod, "PHYSICS_GLOBALS", ()):
            old = getattr(mod, n)
            new = (not old) if isinstance(old, bool) else (
                old + 1 if isinstance(old, int) else old * 1.5 + 0.25)
            setattr(mod, n, new)
            try:
                k1 = cpt._phi_key(legs, 0, avec, 1.0, tau)
                i1 = fn.scale_identity(legs, "qop", 1e-12, 1 << 16, 4, 40.0,
                                       ("ioni", "ms", "rad"), None)
            finally:
                setattr(mod, n, old)
            a = "MOVED" if k1 != key0 else "SAME"
            b = "MOVED" if i1["_sha"] != ident0["_sha"] else "SAME"
            nbadkey += (a == "SAME") + (b == "SAME")
            print(f"  {name + '.' + n:<40}{a:>12}{b:>12}"
                  + ("" if a == b == "MOVED" else "   *** NOT IN THE KEY ***"))
    if nbadkey:
        raise SystemExit(f"{nbadkey} knob/key pairs do not respond")
    print("  every knob moves both keys.")


_PLUS = {13: -13, -13: -13,          # PDG 13 is the mu-; the mu+ is -13
         -211: 211, 211: 211,
         -321: 321, 321: 321,
         -2212: 2212, 2212: 2212}


def _plus(pdg):
    """The POSITIVE representative of a species -- which is what the
    extrapolator's table is built from (G4MuonPlus, G4Proton), and therefore
    the sign at which its high-order block sits."""
    return _PLUS[pdg]


def _hocodd_per_leg(pdg, corr="on", force_q=None):
    """Accumulated charge-ODD reference error, MeV, one entry per leg.

    The reference integrates a dE/dx table whose high-order block is
    +(2*Barkas + Mott) -- the POSITIVE particle's, because
    G4TablesForExtrapolatorForCVH builds the muon table from G4MuonPlus and the
    hadron table from G4Proton.  A particle of charge q should carry
    q*(2*Barkas + Mott), so the reference's per-step energy error is

        dE_s = (q - 1) * xi_s * (2*Barkas + Mott)_s

    with xi_s the record's own Landau normalization
    (xi = twopi_mc2_rcl2 * n_e * len / beta^2, so xi * L-correction IS the
    energy: no step length, no electron density and no material constant
    enters from outside the record).

    Two facts make this exact rather than approximate:
      * HighOrderCorrections ignores its cutEnergy argument (the 4th parameter
        is unnamed in the signature), so restricted and unrestricted dE/dx
        carry the SAME high-order block;
      * Barkas and Mott depend on beta and z only, not on mass, so the
        proton-table scaling e -> e*m_p/m the extrapolator applies to hadrons
        reproduces the hadron's own high-order term exactly.
    """
    legs = cpt.load_model(hp.model_path(pdg, corr))
    mass = hp.SPECIES[pdg]["mass"]
    plus = _plus(pdg)
    # The record spans THREE materials -- ToyLayerMat (98.89 % of sum(xi)), the
    # Beryllium beam pipe (1.10 %) and Air (0.01 %) -- and Barkas, unlike
    # Mott, is material-dependent, so each step is evaluated in its own.
    rows = []
    for j, leg in enumerate(legs):
        for r in leg["ioni"]:
            reg, gsig2, a1, e1, a2, e2, a3, e0, tmx, g = r[:10]
            assert int(reg) in (2, 3), f"regime {int(reg)}: not an exact-delta record"
            name, Z, A, rho = ds.material_of(np.sqrt(e2 / 1e-5))
            rows.append(dict(leg=j, xi=a3 * g, ekin=r[12] - mass,
                             Z=Z, A=A, rho=rho, mat=name))
    lst = os.path.join(SCRATCH, f"barkas_ekin_{pdg}.txt")
    with open(lst, "w") as fh:
        for r in rows:
            # float(), not repr(): repr of a numpy scalar is "np.float64(...)",
            # which the driver's scanf reads as zero fields and silently
            # produces an EMPTY correction list.
            fh.write("%d %.17g %.17g %.17g %.17g\n"
                     % (plus, float(r["ekin"]), float(r["Z"]), float(r["A"]),
                        float(r["rho"])))
    txt, _ = _driver(["--ekinlist", lst])
    odd = []
    for line in txt.splitlines():
        f = line.split()
        if f and f[0] == "CORR":
            odd.append(2 * float(f[3]) + float(f[5]))
    assert len(odd) == len(rows), f"{len(odd)} corrections for {len(rows)} steps"
    q = force_q if force_q is not None else (
        -1.0 if pdg in (13, -211, -321, -2212) else +1.0)
    per = np.array([r["xi"] * o for r, o in zip(rows, odd)]) * (q - 1.0)
    cum = np.zeros(len(legs))
    for j in range(len(legs)):
        cum[j] = per[[i for i, r in enumerate(rows) if r["leg"] <= j]].sum()
    return legs, cum


def cmd_refxcheck(args):
    """Is the offline shift of `_hocodd_per_leg` the SAME number Geant4 removes?

    `export --arms nohoc` re-runs the whole Geant4e reference propagation with
    BarkasCorrection and MottCorrection interposed to zero, so the difference
    between that reference and the nominal one IS the high-order block's
    contribution to the reference, computed by Geant4 through its own table
    build, range table and stepping -- none of which the offline formula
    models.  If the two agree, the offline shift is not a parameterization of
    the effect, it is the effect."""
    print("=" * 116)
    print("CROSS-CHECK: the offline shift against Geant4's own, via the "
          "LD_PRELOAD reference export.")
    print("dE(nohoc - nominal) at each plane is what removing Barkas+Mott from "
          "the extrapolator's table does;")
    print("`offline` is sum_steps xi_s*(2*Barkas+Mott)_s at the SAME steps. "
          "Both in MeV, at the outermost plane.")
    print("=" * 116)
    print(f"  {'species':<8}{'Geant4 (C++)':>16}{'offline':>14}"
          f"{'difference':>14}{'relative':>12}")
    for pdg in args.pdg:
        sp = hp.SPECIES[pdg]
        a = _orig_model_path(pdg, "on")
        b = a[:-5] + "_nohoc.root"
        if not os.path.exists(b):
            print(f"  {sp['label']:<8} {b} missing (export --arms nohoc)")
            continue
        l0 = cpt.load_model(a)
        l1 = cpt.load_model(b)
        k = len(l0) - 1
        P0 = 1e3 / abs(l0[k]["refqop"])
        P1 = 1e3 / abs(l1[k]["refqop"])
        m = sp["mass"]
        dE = (math.sqrt(P1 * P1 + m * m) - math.sqrt(P0 * P0 + m * m))
        # nohoc loses LESS (or more) than nominal by exactly the block; the
        # reference's block sits at the POSITIVE particle's sign, so
        # dE(nohoc-nominal) = +S for a species whose table is the + one.
        _, cum = _hocodd_per_leg(pdg)
        # `cum` carries the (q-1) factor; the bare block is cum/(q-1) for a
        # negative species and has to be recomputed for a positive one.
        q = -1.0 if pdg in (13, -211, -321, -2212) else +1.0
        S = cum[-1] / (q - 1.0) if q < 0 else None
        if S is None:
            # positive species: (q-1) = 0, so redo without the factor
            S = _hocodd_bare(pdg)
        print(f"  {sp['label']:<8}{dE:16.6f}{S:14.6f}{dE - S:14.6f}"
              f"{(dE - S) / S:12.2e}")


def _hocodd_bare(pdg):
    """sum_steps xi_s * (2*Barkas + Mott)_s, MeV, WITHOUT the (q-1) factor."""
    _, cum = _hocodd_per_leg(pdg, force_q=-1.0)
    return cum[-1] / (-2.0)


def cmd_refshift(args):
    """THE PARAMETER-FREE CORRECTION.

    Geant4's simulation cannot be switched (s2: the EM code is statically
    linked into pluginSimulation.so with local symbols), so the switch is
    thrown on the side that CAN be: the reference.  Not by fitting anything to
    the data -- by adding to the reference the very term the extrapolator's
    table is missing, evaluated by Geant4 itself at the exported step
    kinematics.  There is no free parameter anywhere in `_hocodd_per_leg`.

    The prediction is sharp and one-sided: the POSITIVE species must not move
    (their reference is already right, s1), and the NEGATIVE species must
    improve to meet them."""
    print("=" * 128)
    print("REFERENCE MADE CHARGE-AWARE, PARAMETER-FREE.  The reference's "
          "per-plane energy is shifted by")
    print("(q-1) * sum_steps xi_s * (2*Barkas + Mott)_s, taken from Geant4's "
          "own G4EmCorrections at the")
    print("exported step kinematics.  Nothing is fitted.  `data re-centred` is "
          "NOTES_HADRONS s6.5's diagnostic")
    print("(one parameter per plane taken FROM the data) and is a lower bound, "
          "not a closure.")
    print("=" * 128)
    print(f"  {'species':<8}{'q':>3}{'dE shift @outer':>17}"
          f"{'as exported':>14}{'charge-aware ref':>18}{'data re-centred':>18}")
    for pdg in args.pdg:
        sp = hp.SPECIES[pdg]
        legs, cum = _hocodd_per_leg(pdg)
        base = hp._rows(pdg, args.arm, corr="on")
        legs2 = [dict(l) for l in cpt.load_model(hp.model_path(pdg, "on"))]
        for j, l in enumerate(legs2):
            P = 1e3 / abs(l["refqop"])                        # MeV
            E = math.sqrt(P * P + sp["mass"] ** 2) - cum[j]    # corrected
            P2 = math.sqrt(max(E * E - sp["mass"] ** 2, 1.0))
            l["refqop"] = math.copysign(1e3 / P2, l["refqop"])
        r2 = _closure_with(legs2, pdg, args.arm, base)
        # and the data-driven re-centring, for scale
        legs3 = [dict(l) for l in cpt.load_model(hp.model_path(pdg, "on"))]
        sim = hp.sim_of(pdg, args.arm)
        for j in range(len(legs3)):
            g = sim["valid"][:, j] & np.isfinite(sim["qop"][:, j])
            if g.sum() >= 100:
                legs3[j]["refqop"] = float(sim["qop"][g, j].mean())
        r3 = _closure_with(legs3, pdg, args.arm, base)
        print(f"  {sp['label']:<8}{int(np.sign(legs[0]['refqop'])):>3}"
              f"{cum[-1]:17.5f}{hp._rms(base['m']):14.5f}"
              f"{hp._rms(r2):18.5f}{hp._rms(r3):18.5f}")


def _closure_with(legs2, pdg, arm, base):
    """Run the closure against a modified reference, with the SAME Fisher
    normalization as the base (the noise model is untouched: only the
    reference's mean moves) and with every cache bypass hadron_probe's `_rows`
    takes."""
    import geom_closure as gc
    import fisher_norm as fn
    import cf_track_resolution as ctr
    assert os.environ.get("RES_NO_PHI_CACHE"), "phi cache is LIVE"
    ctr.IONI_KOKOULIN = 1.0 if hp.kok_for(pdg) else 0.0
    fn._SCALE_CACHE.clear()
    cpt._PHI_CACHE.clear()
    try:
        r = gc.closure_rows(legs2, hp.sim_of(pdg, arm), "qop",
                            base["sc"]["sF"])[0]
    finally:
        ctr.IONI_KOKOULIN = 0.0
        fn._SCALE_CACHE.clear()
        cpt._PHI_CACHE.clear()
    return r.mean(axis=0)


def cmd_closure(args):
    """The closure with the charge-odd term removed from BOTH sides.

    The reference's table is built from the POSITIVE representative of each
    species (G4MuonPlus, G4Proton -- see `dedx`), so it does not sit at the
    charge average: it sits at the positive value.  Removing the odd term from
    the model export AND from the simulation therefore has to collapse the
    q = -1 / q = +1 difference, not merely shrink it."""
    global MODEL_SUFFIX
    print("=" * 130)
    print("CLOSURE, qop, plane mean, Fisher normalization, layered toy pT = 3, "
          "eta 0.30, cut1e4, exact-delta ON,")
    print("Kokoulin species-correct, nuclear + Decay OFF, 200k events, "
          "nothing fitted.")
    print("=" * 130)
    print(f"  {'species':<8}{'arm':<10}" + hp.UROW + "      rms")
    res = {}
    for arm in args.arms:
        MODEL_SUFFIX = "" if arm == "off" else "_" + arm
        for pdg in args.pdg:
            r = hp._rows(pdg, arm, corr="on", func=args.func)
            m = r["m"]
            res[(pdg, arm)] = (m, r["err"])
            print(f"  {hp.SPECIES[pdg]['label']:<8}{arm:<10}" + hp._fmt(m)
                  + f" {hp._rms(m):9.5f}   n={r['ns'][0]}")
    MODEL_SUFFIX = ""
    print()
    print("  charge asymmetry of the closure rms")
    print(f"    {'species':<8}" + "".join(f"{a:>22}" for a in args.arms))
    for m, p, lab in PAIRS:
        row = f"    {lab:<8}"
        for arm in args.arms:
            if (m, arm) in res and (p, arm) in res:
                a, b = hp._rms(res[(m, arm)][0]), hp._rms(res[(p, arm)][0])
                row += f"{b:9.5f} /{a:8.5f}  "
            else:
                row += f"{'':>22}"
        print(row)


# =========================================================================

def main():
    ap = argparse.ArgumentParser()
    s = ap.add_subparsers(dest="cmd", required=True)

    s.add_parser("build")
    s.add_parser("dedx")
    s.add_parser("scan")
    s.add_parser("shimlive")

    q = s.add_parser("sim")
    q.add_argument("--pdg", type=int, nargs="+", default=ORDER8)
    q.add_argument("--arms", nargs="+", default=["nohoc"])
    q.add_argument("--events", type=int, default=20000)
    q.add_argument("--jobs", type=int, default=10)
    q.add_argument("--seed0", type=int, default=101)
    q.add_argument("--workers", type=int, default=45)
    q.add_argument("--force", action="store_true")

    q = s.add_parser("live")
    q.add_argument("--pdg", type=int, nargs="+", default=ORDER8)
    q.add_argument("--arms", nargs="+", default=["nohoc"])

    q = s.add_parser("census")
    q.add_argument("--arms", nargs="+", default=["off", "nohoc"])

    q = s.add_parser("predict")
    q.add_argument("--arm", default="off")

    q = s.add_parser("export")
    q.add_argument("--pdg", type=int, nargs="+", default=ORDER8)
    q.add_argument("--arms", nargs="+", default=["nohoc"])
    q.add_argument("--force", action="store_true")

    s.add_parser("guards")

    q = s.add_parser("refxcheck")
    q.add_argument("--pdg", type=int, nargs="+", default=ORDER8)

    q = s.add_parser("refshift")
    q.add_argument("--pdg", type=int, nargs="+", default=ORDER8)
    q.add_argument("--arm", default="off")

    q = s.add_parser("closure")
    q.add_argument("--pdg", type=int, nargs="+", default=ORDER8)
    q.add_argument("--arms", nargs="+", default=["off", "nohoc"])
    q.add_argument("--func", default="qop")

    a = ap.parse_args()
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
