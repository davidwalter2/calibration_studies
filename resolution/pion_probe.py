#!/usr/bin/env python3
"""Why does the PION fail the clean-propagation closure when mu, K and p pass?

NOTES_HADRONS s9 left the pion open: 3.5x (q=-1) and 15x (q=+1) the muon,
not radiative (`hBrems`/`hPairProd` off in the sim is a 0.00758 -> 0.00752
null), and not the mean as that note could measure it (re-centring on the
data's own per-plane mean removes only 41 %).  NOTES_BARKAS s6 then removed
the charge-odd term from the reference and the pion kept its excess
(0.00758 -> 0.00571 against pi+ 0.00540), which retires the charge asymmetry
as well.

The remaining charge-EVEN, species-dependent suspects, and what this module
does with each:

  1. the reference's dE/dx TABLE.  `G4EnergyLossForExtrapolatorForCVH`
     has no pion table: hadrons are served by the PROTON table, evaluated at
     `e = ekin * m_p / m` and multiplied by q^2.  That scaling preserves
     beta*gamma exactly, hence beta, gamma and the density-effect delta -- but
     NOT Tmax, whose recoil denominator carries the projectile mass.  The
     reference therefore integrates a stopping power with the PROTON's Tmax at
     the pion's beta*gamma, i.e. it is too large by

         dE_defect = xi * ln( Tmax_p(bg) / Tmax_species(bg) )

     (an identity: d(dE/dx)/d ln Tmax = xi for Bethe-Bloch, unrestricted).
     NOTES_BARKAS s2.4 measured the relative size -- +5.2e-3 for the pion,
     +2.9e-4 for the kaon, -1.1e-5 for the proton, and the muon does not go
     through this branch at all -- flagged it as "a lead, not a result", and
     did not test it.  `dedxdef` computes it from the exported record alone
     and `refshift` puts it into the reference and re-runs the closure.

  2. the SPIN branch.  pi and K are regime 3 (spin 0, no T^2/2E^2 term), mu
     and p regime 2.  The kaon closes on the same branch, but at Tmax/E =
     0.013 against the pion's 0.141, so the kaon cannot test any term that
     scales with Tmax/E.  NOTES_DELTASPEC s9.4 bounds the term at 2.1e-7 on
     the rate and 5.1e-4 on the delta energy for a pion and calls it
     unmeasurable; `spin` measures the closure directly rather than trusting
     the bound.

  3. MULTIPLE SCATTERING / transport.  NOTES_HADRONS s4 already answers this:
     the pion has the SMALLEST `locx` residual of all five species (0.00046
     against the proton's 0.00263 and the muon's 0.00052).  `locx` is
     reproduced here as a control, not re-derived.

EVERY NUMBER IS A MEASUREMENT.  `control` reproduces the published per-species
rows before anything new is believed; every gauge is applied through the same
channel, weights and transform as the effect it is testing, and is judged on
SHAPE as well as size.

CACHES.  `RES_NO_PHI_CACHE=1` is set at import, before anything can populate
`cf_propagation_test._PHI_CACHE`; `fisher_norm`'s two scale caches are cleared
and stubbed inside every closure call (hadron_probe._rows does this already
and is reused wherever possible).  Any new module global introduced here is
registered in the owning module's `PHYSICS_GLOBALS` so the sha256 cache keys
move with it.

VERDICT (NOTES_PION.md).  The cause is the proton-table Tmax defect of s1
below: 0.1223 MeV of reference over-subtraction for the pion, 0.0061 for the
kaon, exactly zero for the muon and the proton.  Removing it takes
pi+ 0.00540 -> 0.00045 and pi- 0.00758 -> 0.00237; with NOTES_BARKAS's
charge-odd correction as well every species closes at 0.00020-0.00068.

SUBCOMMANDS
    control    the published per-species rows (+ --locx), reproduced
    dedxdef    the proton-table Tmax defect, per species, from the record
    refshift   THE GAUGE: reference given the species' own Tmax; size + shape
    shape      the gauge's amplitude scanned -- is s = 1 the minimum?
    fit        least-squares amplitude of the gauge (--odd: charge-odd first)
    both       both parameter-free reference corrections, all eight species
    cylexport  re-export with GEANT4 applying the correction (CVH_ELOSS_CYL_EPS)
    cyl        the Geant4-side cross-check of the offline shift
    spin       the spin-0/spin-1/2 branch flipped as a gauge
    meanloss   the second-order leak into the fluctuation's excitation weights
    meancheck  is the sample-mean diagnostic usable?  (it is not, for pi/mu)
    trunc      the mean reference error vs how much tail enters the estimator
"""

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "cleanprop"))

# BEFORE any import that could populate the phi cache.
os.environ["RES_NO_PHI_CACHE"] = "1"

import hadron_probe as hp                                    # noqa: E402
import cf_propagation_test as cpt                            # noqa: E402
import cf_track_resolution as ctr                            # noqa: E402
import fisher_norm as fn                                     # noqa: E402
import geom_closure as gc                                    # noqa: E402

SCRATCH = fn.SCRATCH
OUT = os.path.join(SCRATCH, "pion")
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------ THE SEED PIN
# NOTES_HADRONS' tables are 200 000 events per cell, seeds 101-110.  The
# NOTES_BARKAS work running concurrently added an eleventh muon `off` seed
# (s901, 2026-08-15 23:40), so the unrestricted glob now returns 220 000 for
# the muon and only for the muon -- which silently makes the muon control
# irreproducible (rms 0.00219 against the published 0.00217, every low-u probe
# ~1 sigma out) while every hadron still reproduces exactly.  That is a
# statistics change, not a physics one, but a control that does not reproduce
# is not a control.  Pin the glob to the archived decade.
_SIM_GLOB = hp.sim_glob
hp.sim_glob = lambda pdg, arm: _SIM_GLOB(pdg, arm).replace("_s*", "_s1*")

MEL = 0.51099895                 # MeV, CLHEP electron_mass_c2
MPROT = 938.27208816             # MeV, CLHEP proton_mass_c2

# The published rows this module must reproduce before anything new is
# believed.  NOTES_HADRONS s4, arm `off`, exact-delta ON, Kokoulin
# species-correct, 200k events, qop, plane mean, Fisher normalization.
PUBLISHED = {
    13:    ([0.00020, -0.00004, -0.00090, -0.00262, -0.00437, -0.00337,
             -0.00172, -0.00099, -0.00054], 0.00217),
    -13:   ([0.00005, 0.00011, 0.00014, 0.00018, 0.00052, 0.00073,
             0.00050, 0.00028, 0.00014], 0.00037),
    -211:  ([-0.00056, -0.00139, -0.00399, -0.00924, -0.01502, -0.01180,
             -0.00592, -0.00323, -0.00177], 0.00758),
    211:   ([-0.00075, -0.00141, -0.00320, -0.00682, -0.01075, -0.00822,
             -0.00384, -0.00196, -0.00103], 0.00540),
    -321:  ([-0.00013, -0.00029, -0.00076, -0.00187, -0.00424, -0.00585,
             -0.00453, -0.00279, -0.00154], 0.00311),
    321:   ([-0.00008, -0.00016, -0.00031, -0.00056, -0.00100, -0.00132,
             -0.00114, -0.00075, -0.00044], 0.00076),
    -2212: ([-0.00011, -0.00024, -0.00060, -0.00138, -0.00322, -0.00529,
             -0.00519, -0.00327, -0.00181], 0.00301),
    2212:  ([-0.00000, -0.00003, -0.00009, -0.00018, -0.00029, -0.00031,
             -0.00028, -0.00024, -0.00011], 0.00020),
}
# NOTES_HADRONS s4, func = locx (the MS-dominated functional), arm `off`
PUBLISHED_LOCX_RMS = {13: 0.00103, -211: 0.00046, -321: 0.00168,
                      -2212: 0.00212, 2212: 0.00263}

ORDER = [13, -13, -211, 211, -321, 321, -2212, 2212]


# =========================================================================
# 0.  kinematics straight out of the exported record
# =========================================================================

def tmax_of(bg, mass):
    """Unrestricted maximum energy transfer to a free electron, MeV.

    Tmax = 2 m_e bg^2 / (1 + 2 gamma m_e/M + (m_e/M)^2)
    -- the same expression as G4BetheBlochModel::MaxSecondaryEnergy and as
    G4UniversalFluctuationForExtrapolator's `tmaxkine`."""
    bg = np.asarray(bg, float)
    gam = np.sqrt(1.0 + bg * bg)
    r = MEL / mass
    return 2.0 * MEL * bg * bg / (1.0 + 2.0 * gam * r + r * r)


def steps_of(pdg, corr="on", suffix=""):
    """Every ionization step of the exported model, as a dict of arrays, with
    the leg index.  Regime 2/3 only (exact-delta records); the record's own
    columns are used and nothing is assumed about the particle.

    columns: 0 regime, 1 gsig2, 2 a1, 3 e1, 4 a2, 5 e2, 6 a3/xi, 7 e0r,
             8 tmaxr, 9 scaling, 10 qop-per-MeV, 11 beta2, 12 etot
    """
    mp = hp.model_path(pdg, corr)
    if suffix:
        mp = mp[:-5] + suffix + ".root"
    legs = cpt.load_model(mp)
    out = dict(leg=[], reg=[], xi=[], e0=[], tmax=[], beta2=[], etot=[],
               a1=[], e1=[], a2=[], e2=[], gsig2=[], scal=[])
    for j, leg in enumerate(legs):
        for r in leg["ioni"]:
            g = r[9]
            out["leg"].append(j)
            out["reg"].append(int(r[0]))
            out["gsig2"].append(r[1])
            out["a1"].append(r[2])
            out["e1"].append(r[3] * g)
            out["a2"].append(r[4])
            out["e2"].append(r[5] * g)
            out["xi"].append(r[6] * g)
            out["e0"].append(r[7] * g)
            out["tmax"].append(r[8] * g)
            out["scal"].append(g)
            out["beta2"].append(r[11])
            out["etot"].append(r[12])
    d = {k: np.asarray(v) for k, v in out.items()}
    d["mass"] = d["etot"] * np.sqrt(np.clip(1.0 - d["beta2"], 0.0, None))
    d["bg"] = np.sqrt(d["beta2"] / np.clip(1.0 - d["beta2"], 1e-30, None))
    return legs, d


def tmax_defect_per_leg(pdg, corr="on"):
    """The reference's per-leg energy error from the proton-table Tmax
    mismatch, MeV, CUMULATIVE (one entry per leg), PARAMETER-FREE.

        dE_s = xi_s * ln( Tmax_proton(bg_s) / Tmax_species(bg_s) )

    Derivation.  Unrestricted Bethe-Bloch mean loss over a step is

        dE = xi [ ln(2 m_e bg^2 Tmax / I^2) - 2 beta^2 - delta ]

    so d(dE)/d ln Tmax = xi exactly.  `G4EnergyLossForExtrapolatorForCVH::
    ComputeDEDX` serves every hadron from the PROTON table at
    e = ekin*m_p/m, which preserves bg (hence beta, gamma, delta and I) and
    changes ONLY Tmax; the reference therefore carries Tmax_proton(bg) where
    it should carry Tmax_species(bg), and the error is the expression above.

    Everything comes from the record: xi is column 6 (times `scaling`),
    bg from the exported beta^2, and the mass from E*sqrt(1-beta^2).  The MUON
    is served by its own table (G4MuonPlus) and gets exactly zero.
    """
    legs, d = steps_of(pdg, corr)
    assert set(np.unique(d["reg"])) <= {2, 3}, \
        f"regime {np.unique(d['reg'])}: not an exact-delta record"
    if abs(pdg) == 13:
        return legs, np.zeros(len(legs)), d, np.zeros(len(d["xi"]))
    per = d["xi"] * np.log(tmax_of(d["bg"], MPROT) / tmax_of(d["bg"], d["mass"]))
    cum = np.array([per[d["leg"] <= j].sum() for j in range(len(legs))])
    return legs, cum, d, per


# =========================================================================
# 1.  controls
# =========================================================================

def cmd_control(args):
    print("=" * 132)
    print("CONTROL: the published NOTES_HADRONS s4 rows, reproduced.  arm "
          "`off`, exact-delta ON, Kokoulin species-correct, qop, plane mean.")
    print("=" * 132)
    print(f"  {'species':<9}" + hp.UROW + "      rms")
    bad = 0
    for pdg in args.pdg:
        r = hp._rows(pdg, "off", corr="on")
        m = r["m"]
        pub, prms = PUBLISHED[pdg]
        d = m - np.array(pub)
        print(f"  {hp.SPECIES[pdg]['label']:<9}" + hp._fmt(m)
              + f" {hp._rms(m):9.5f}   n={r['ns'][0]}")
        print(f"  {'  published':<9}" + hp._fmt(pub) + f" {prms:9.5f}")
        print(f"  {'  diff':<9}" + hp._fmt(d))
        if np.max(np.abs(d)) > 5e-6:
            bad += 1
            print("        *** DOES NOT REPRODUCE ***")
    if args.locx:
        print()
        print("  locx (MS-dominated functional) -- NOTES_HADRONS s4")
        for pdg in args.pdg:
            if pdg not in PUBLISHED_LOCX_RMS:
                continue
            r = hp._rows(pdg, "off", corr="on", func="locx")
            print(f"    {hp.SPECIES[pdg]['label']:<8}rms {hp._rms(r['m']):9.5f}"
                  f"   published {PUBLISHED_LOCX_RMS[pdg]:9.5f}")
    if bad:
        raise SystemExit(f"{bad} species do not reproduce")
    print("\n  every requested species reproduces the published row.")


# =========================================================================
# 2.  the proton-table Tmax defect, measured from the record
# =========================================================================

def cmd_dedxdef(args):
    print("=" * 132)
    print("THE PROTON-TABLE Tmax DEFECT.  `G4EnergyLossForExtrapolatorForCVH::"
          "ComputeDEDX` serves every hadron from the")
    print("PROTON table at e = ekin*m_p/m.  That preserves beta*gamma exactly "
          "and therefore beta, gamma, I and the")
    print("density-effect delta -- but NOT Tmax.  dE_defect = "
          "sum_steps xi_s ln(Tmax_p(bg)/Tmax_species(bg)), from the record.")
    print("=" * 132)
    print(f"  {'species':<8}{'regime':>7}{'bg':>9}{'m [MeV]':>10}"
          f"{'Tmax_sp':>10}{'Tmax_p':>10}{'ln ratio':>10}"
          f"{'sum xi':>9}{'dE defect':>11}{'/E_ioni':>10}{'dp/p':>10}")
    for pdg in args.pdg:
        sp = hp.SPECIES[pdg]
        legs, cum, d, per = tmax_defect_per_leg(pdg)
        # the reference's own total ionization loss, from the refqop chain
        P0 = 1e3 / abs(legs[0]["refqop"])
        # step 0 is at the innermost plane; take the record's own first-step
        # etot as the entry energy instead (it precedes any loss)
        E0 = d["etot"][0]
        Pk = 1e3 / abs(legs[-1]["refqop"])
        Ek = math.sqrt(Pk * Pk + float(d["mass"][0]) ** 2)
        Eioni = E0 - Ek
        i = 0
        lr = math.log(tmax_of(d["bg"][i], MPROT) / tmax_of(d["bg"][i],
                                                           d["mass"][i]))
        print(f"  {sp['label']:<8}{int(d['reg'][0]):>7}{d['bg'][i]:9.3f}"
              f"{d['mass'][i]:10.3f}{tmax_of(d['bg'][i], d['mass'][i]):10.2f}"
              f"{tmax_of(d['bg'][i], MPROT):10.2f}{lr:10.5f}"
              f"{d['xi'].sum():9.4f}{cum[-1]:11.5f}"
              f"{cum[-1] / Eioni:10.2e}{cum[-1] / P0:10.2e}")
    print()
    print("  `/E_ioni` is the relative dE/dx error, to be read against "
          "NOTES_BARKAS s2.3's directly measured")
    print("  extrapolator-vs-Geant4 numbers (pi+ +5.241e-03, K+ +2.859e-04, "
          "p -1.121e-05, all charge-EVEN).")
    print("  `dp/p` is what the reference's momentum at the outermost plane "
          "is wrong by, if nothing absorbs it.")


# =========================================================================
# 3.  THE GAUGE: hand the reference the species' own Tmax
# =========================================================================

def _closure_with(legs2, pdg, arm, base, func="qop"):
    """Closure against a modified reference, with the SAME Fisher scale as
    `base` -- only the reference's mean moves, the noise model is untouched --
    and with every cache bypass `hadron_probe._rows` takes."""
    assert os.environ.get("RES_NO_PHI_CACHE"), "phi cache is LIVE"
    ctr.IONI_KOKOULIN = 1.0 if hp.kok_for(pdg) else 0.0
    fn._SCALE_CACHE.clear()
    cpt._PHI_CACHE.clear()
    try:
        r, e = gc.closure_rows(legs2, hp.sim_of(pdg, arm), func,
                               base["sc"]["sF"])[:2]
    finally:
        ctr.IONI_KOKOULIN = 0.0
        fn._SCALE_CACHE.clear()
        cpt._PHI_CACHE.clear()
    return r.mean(axis=0)


def _shifted_legs(pdg, cum, corr="on"):
    """The exported legs with the reference's energy at plane j reduced by
    cum[j] MeV (the reference over-loses, so the correction ADDS momentum)."""
    legs = [dict(l) for l in cpt.load_model(hp.model_path(pdg, corr))]
    m = hp.SPECIES[pdg]["mass"]
    for j, l in enumerate(legs):
        P = 1e3 / abs(l["refqop"])
        E = math.sqrt(P * P + m * m) + cum[j]
        P2 = math.sqrt(max(E * E - m * m, 1.0))
        l["refqop"] = math.copysign(1e3 / P2, l["refqop"])
    return legs


def cmd_refshift(args):
    """THE GAUGE.  The reference is handed the species' own Tmax -- i.e. the
    proton-table defect is removed -- and the closure re-run.  Nothing is
    fitted: `cum` comes from the record via `tmax_defect_per_leg`.

    Prediction if this is the pion's cause: the pion must move by ~its whole
    excess AND the shape must match; the kaon must move by ~1/18 of it; the
    proton must not move at all (its table IS the proton's); the muon is not
    on this branch and is a null by construction."""
    print("=" * 140)
    print("REFERENCE GIVEN THE SPECIES' OWN Tmax (proton-table defect "
          "removed), PARAMETER-FREE.  arm `off`, qop, plane mean.")
    print("The muon does not use the proton table and must be an exact null; "
          "the proton IS the table and must be an exact null.")
    print("=" * 140)
    print(f"  {'species':<9}{'dE @outer':>11}  " + hp.UROW + "      rms")
    for pdg in args.pdg:
        sp = hp.SPECIES[pdg]
        legs, cum, d, per = tmax_defect_per_leg(pdg)
        base = hp._rows(pdg, args.arm, corr="on", func=args.func)
        m0 = base["m"]
        print(f"  {sp['label']:<9}{'as exported':>11}  " + hp._fmt(m0)
              + f" {hp._rms(m0):9.5f}")
        m1 = _closure_with(_shifted_legs(pdg, cum), pdg, args.arm, base,
                           args.func)
        print(f"  {'':<9}{cum[-1]:11.5f}  " + hp._fmt(m1)
              + f" {hp._rms(m1):9.5f}")
        print(f"  {'':<9}{'moved by':>11}  " + hp._fmt(m1 - m0))
        print(f"  {'':<9}{'per-probe +-':>11}  " + hp._fmt(base["err"]))
        print()


def cmd_shape(args):
    """SHAPE, not just size.  Inject a controlled reference shift of the SAME
    size as the Tmax defect but through the plain mean channel, and ask
    whether the pion's excess has that shape.

    `--scale` multiplies the parameter-free defect, so `--scale 0` is the
    null and larger values show how the u-curve responds to a pure mean
    displacement of the reference -- which is the only thing this defect can
    do."""
    pdg = args.pdg[0]
    sp = hp.SPECIES[pdg]
    legs, cum, d, per = tmax_defect_per_leg(pdg)
    base = hp._rows(pdg, args.arm, corr="on")
    m0 = base["m"]
    print("=" * 140)
    print(f"SHAPE OF A PURE REFERENCE-MEAN SHIFT, {sp['label']}, arm "
          f"{args.arm}.  The parameter-free Tmax defect is "
          f"{cum[-1]:.5f} MeV at the outermost plane.")
    print("=" * 140)
    print(f"  {'scale':<9}{'dE @outer':>11}  " + hp.UROW + "      rms")
    print(f"  {'0 (base)':<9}{0.0:11.5f}  " + hp._fmt(m0)
          + f" {hp._rms(m0):9.5f}")
    for s in args.scale:
        m1 = _closure_with(_shifted_legs(pdg, cum * s), pdg, args.arm, base)
        print(f"  {s:<9g}{cum[-1] * s:11.5f}  " + hp._fmt(m1)
              + f" {hp._rms(m1):9.5f}")
    print(f"  {'+-':<9}{'':>11}  " + hp._fmt(base["err"]))


# =========================================================================
# 4.  the spin branch, as a gauge
# =========================================================================

def cmd_spin(args):
    """Flip the delta channel's spin branch and measure the closure.

    NOTES_DELTASPEC s9.4 bounds the spin term at 2.1e-7 on the pion's delta
    RATE and 5.1e-4 on its delta ENERGY and calls it unmeasurable.  That is a
    bound on the spectrum, not on the closure, so it is measured here:
    regime 3 -> 2 for pi and K, regime 2 -> 3 for mu and p, by rewriting the
    record's regime column (the ONLY thing `ioni_step_exponent` dispatches
    on).  The size AND the shape are reported."""
    print("=" * 140)
    print("SPIN BRANCH FLIPPED AS A GAUGE.  regime 3 (spin 0: pi, K) <-> "
          "regime 2 (spin 1/2: mu, p), by rewriting the record's")
    print("regime column.  The exact-delta spectrum gains/loses the "
          "T^2/2E^2 term; nothing else moves.")
    print("=" * 140)
    print(f"  {'species':<9}{'regime':>8}  " + hp.UROW + "      rms")
    for pdg in args.pdg:
        sp = hp.SPECIES[pdg]
        base = hp._rows(pdg, args.arm, corr="on")
        m0 = base["m"]
        r0 = int(cpt.load_model(hp.model_path(pdg, "on"))[0]["ioni"][0][0])
        print(f"  {sp['label']:<9}{r0:>8}  " + hp._fmt(m0)
              + f" {hp._rms(m0):9.5f}")
        legs2 = []
        flip = {2: 3, 3: 2}
        for leg in cpt.load_model(hp.model_path(pdg, "on")):
            l = dict(leg)
            io = np.array(leg["ioni"], copy=True)
            for i in range(len(io)):
                io[i, 0] = flip.get(int(io[i, 0]), io[i, 0])
            l["ioni"] = io
            legs2.append(l)
        # the Fisher scale must be recomputed: the noise model itself moved
        assert os.environ.get("RES_NO_PHI_CACHE"), "phi cache is LIVE"
        ctr.IONI_KOKOULIN = 1.0 if hp.kok_for(pdg) else 0.0
        fn._SCALE_CACHE.clear()
        cpt._PHI_CACHE.clear()
        ol, os_ = fn._scale_cache_load, fn._scale_cache_store
        fn._scale_cache_load = lambda *a, **k: None
        fn._scale_cache_store = lambda *a, **k: None
        try:
            sc = fn.plane_scales(legs2, "qop", tag=None,
                                 channels=("ioni", "ms", "rad"))
            m1 = gc.closure_rows(legs2, hp.sim_of(pdg, args.arm), "qop",
                                 sc["sF"])[0].mean(axis=0)
        finally:
            fn._scale_cache_load, fn._scale_cache_store = ol, os_
            ctr.IONI_KOKOULIN = 0.0
            fn._SCALE_CACHE.clear()
            cpt._PHI_CACHE.clear()
        print(f"  {'':<9}{flip[r0]:>8}  " + hp._fmt(m1)
              + f" {hp._rms(m1):9.5f}   s_F[0] "
              f"{sc['sF'][0]:.8e} vs {base['sc']['sF'][0]:.8e}")
        print(f"  {'':<9}{'moved by':>8}  " + hp._fmt(m1 - m0))
        print(f"  {'':<9}{'+-':>8}  " + hp._fmt(base["err"]))
        print()


# =========================================================================
# 5.  BOTH parameter-free reference corrections at once
# =========================================================================

def cmd_both(args):
    """The two independent, parameter-free reference errors, together.

    (a) the charge-ODD one (NOTES_BARKAS): the extrapolator's tables are built
        from the POSITIVE representative (G4MuonPlus, G4Proton), so a negative
        particle is served the wrong sign of Geant4's high-order block
        2*Barkas + Mott.  `barkas_probe._hocodd_per_leg` evaluates it with
        Geant4's own G4EmCorrections at the exported step kinematics; it is
        identically zero for a positive species.

    (b) the charge-EVEN one (this note): the hadron branch of
        `G4EnergyLossForExtrapolatorForCVH::ComputeDEDX` is the PROTON table at
        e = ekin*m_p/m, which carries the proton's Tmax at the species'
        beta*gamma.  `tmax_defect_per_leg` evaluates it from the record; it is
        identically zero for the muon (own table) and for the proton (it IS
        the table).

    Nothing is fitted in either.  The prediction is that after both, every
    species closes at the muon's level in BOTH charges."""
    import barkas_probe as bp
    print("=" * 150)
    print("BOTH PARAMETER-FREE REFERENCE CORRECTIONS.  (a) charge-odd "
          "high-order block, from Geant4's G4EmCorrections;")
    print("(b) charge-even proton-table Tmax defect, from the record.  "
          "Nothing is fitted anywhere.  arm `off`, qop, plane mean.")
    print("=" * 150)
    print(f"  {'species':<8}{'q':>3}{'dE odd':>10}{'dE Tmax':>10}"
          f"{'as exported':>14}{'odd only':>11}{'Tmax only':>11}{'BOTH':>11}")
    keep = {}
    for pdg in args.pdg:
        sp = hp.SPECIES[pdg]
        _, cum_t, _, _ = tmax_defect_per_leg(pdg)
        _, cum_o = bp._hocodd_per_leg(pdg)
        base = hp._rows(pdg, args.arm, corr="on")
        m_t = _closure_with(_shifted_legs(pdg, cum_t), pdg, args.arm, base)
        # `_hocodd_per_leg` returns the reference's ERROR with the sign
        # convention of barkas_probe.cmd_refshift, which SUBTRACTS it.
        m_o = _closure_with(_shifted_legs(pdg, -cum_o), pdg, args.arm, base)
        m_b = _closure_with(_shifted_legs(pdg, cum_t - cum_o), pdg, args.arm,
                            base)
        keep[pdg] = (base["m"], m_o, m_t, m_b, base["err"])
        print(f"  {sp['label']:<8}{sp['q']:>3}{-cum_o[-1]:10.5f}"
              f"{cum_t[-1]:10.5f}{hp._rms(base['m']):14.5f}"
              f"{hp._rms(m_o):11.5f}{hp._rms(m_t):11.5f}"
              f"{hp._rms(m_b):11.5f}")
    print()
    print("  the u curves after BOTH")
    print(f"  {'species':<9}" + hp.UROW + "      rms")
    for pdg in args.pdg:
        m0, m_o, m_t, m_b, err = keep[pdg]
        print(f"  {hp.SPECIES[pdg]['label']:<9}" + hp._fmt(m_b)
              + f" {hp._rms(m_b):9.5f}")
        print(f"  {'  +-':<9}" + hp._fmt(err))


# =========================================================================
# 6.  SHAPE: is the parameter-free amplitude the one the data wants?
# =========================================================================

def cmd_fit(args):
    """One-parameter amplitude fit of the Tmax correction against the observed
    non-closure.  The correction's u-response is linear in its amplitude
    (verified by `shape`), so

        m(s) = m0 + s R,      R = m(1) - m(0)

    and the least-squares amplitude the DATA wants is
    s = sum(-m0 R/sig^2)/sum(R^2/sig^2).  The parameter-free prediction is
    s = 1.  This is the SHAPE test: a right-sized correction with the wrong
    shape leaves a large chi2 at its best amplitude.

    The nine probes are correlated (they are functionals of one CF), so the
    quoted error is a lower bound and the chi2 an upper bound; both are
    reported as such."""
    print("=" * 132)
    print("AMPLITUDE FIT.  The parameter-free proton-table Tmax correction is "
          "s = 1 by construction.  Probes are correlated,")
    print("so sigma(s) is a lower bound and chi2 an upper bound.")
    print("=" * 132)
    print(f"  {'species':<9}{'s_fit':>9}{'+-':>8}{'chi2/ndf before':>18}"
          f"{'chi2/ndf at s=1':>18}{'chi2/ndf at s_fit':>20}")
    base0 = np.zeros(1)
    for pdg in args.pdg:
        legs, cum, d, per = tmax_defect_per_leg(pdg)
        base = hp._rows(pdg, args.arm, corr="on")
        e = base["err"]
        # The charge-ODD reference error (NOTES_BARKAS) has a u-shape close
        # enough to this one that a fit on an uncorrected NEGATIVE species
        # simply absorbs it into this amplitude (pi- returns 1.47, K- 10.0).
        # It is a different, independently measured defect, so it is removed
        # FIRST and the amplitude fitted on what is left.  Positives are
        # unaffected (their odd shift is identically zero), so their fit is
        # unchanged and is the clean measurement.
        pre = np.zeros(len(legs))
        if args.odd:
            import barkas_probe as bp
            pre = -bp._hocodd_per_leg(pdg)[1]
        m0 = (_closure_with(_shifted_legs(pdg, pre), pdg, args.arm, base)
              if args.odd else base["m"])
        m1 = _closure_with(_shifted_legs(pdg, pre + cum), pdg, args.arm, base)
        R = m1 - m0
        w = 1.0 / e ** 2
        den = float(np.sum(R * R * w))
        if den <= 0:
            print(f"  {hp.SPECIES[pdg]['label']:<9}   (no response: "
                  f"dE defect is identically zero)")
            continue
        sfit = float(np.sum(-m0 * R * w)) / den
        sig = 1.0 / math.sqrt(den)
        n = len(m0)
        c0 = float(np.sum(m0 ** 2 * w)) / n
        c1 = float(np.sum((m0 + R) ** 2 * w)) / n
        cf = float(np.sum((m0 + sfit * R) ** 2 * w)) / (n - 1)
        print(f"  {hp.SPECIES[pdg]['label']:<9}{sfit:9.3f}{sig:8.3f}"
              f"{c0:18.1f}{c1:18.1f}{cf:20.1f}")


def cmd_meanloss(args):
    """How much of the defect leaks into the FLUCTUATION model, not just the
    reference mean.

    `G4UniversalFluctuationForExtrapolator` takes `meanLoss = length * dedx`
    from the SAME defective proton table, and the exact-delta block then
    rescales the excitation channels to hold that mean:

        fexc = (meanLoss - xi I1) / (a1 e1 + a2 e2)

    so a meanLoss that is too big by `f` inflates the excitation amplitudes by
    f * meanLoss / eExc.  xi itself is computed from the electron density and
    beta^2 and is NOT affected, and Tmax in the record comes from the
    particle's own mass (Geant4ePropagator's `Emax`), so the delta channel --
    which carries essentially all of the variance -- is untouched.  This
    command sizes what is left."""
    print("=" * 132)
    print("THE SECOND-ORDER LEAK: the defective dE/dx also sets `meanLoss` in "
          "the fluctuation exporter, hence the excitation weights.")
    print("=" * 132)
    print(f"  {'species':<8}{'meanLoss':>10}{'eDelta':>10}{'eExc':>10}"
          f"{'defect':>10}{'d(eExc)/eExc':>14}{'var_exc/var':>13}"
          f"{'d(var)/var':>12}")
    for pdg in args.pdg:
        legs, cum, d, per = tmax_defect_per_leg(pdg)
        eExc = d["a1"] * d["e1"] + d["a2"] * d["e2"]
        # the exact channel's own mean, from the record (regime 3 = no spin term)
        i1 = np.log(d["tmax"] / d["e0"]) - d["beta2"] * (d["tmax"] - d["e0"]) / d["tmax"]
        half = d["reg"] == 2
        i1 = i1 + np.where(half, (d["tmax"] ** 2 - d["e0"] ** 2)
                           / (4.0 * d["etot"] ** 2), 0.0)
        eDelta = d["xi"] * i1
        meanLoss = eDelta + eExc
        # excitation variance vs the (untruncated) delta variance
        var_exc = d["a1"] * d["e1"] ** 2 + d["a2"] * d["e2"] ** 2
        var_del = d["xi"] * (d["tmax"] - d["e0"])
        fr = per.sum() / meanLoss.sum()
        dex = per.sum() / eExc.sum()
        vx = var_exc.sum() / (var_exc.sum() + var_del.sum())
        print(f"  {hp.SPECIES[pdg]['label']:<8}{meanLoss.sum():10.4f}"
              f"{eDelta.sum():10.4f}{eExc.sum():10.4f}{per.sum():10.5f}"
              f"{dex:14.2e}{vx:13.2e}{dex * vx:12.2e}")
    print()
    print("  `d(var)/var` is the fractional change of the block's TOTAL "
          "ionization variance from the excitation")
    print("  channels being mis-weighted -- the only part of the defect the "
          "reference shift does not already carry.")


# =========================================================================
# 7.  why the MEAN diagnostic missed it
# =========================================================================

def cmd_meancheck(args):
    """Reconcile this note's answer with NOTES_HADRONS s6.4/s6.5.

    That note estimated the reference's mean-loss error from the SAMPLE MEAN
    of the qop residual, `dE = <dqop> p^3/(qE)`, and got -0.0654 MeV for the
    pion -- 2.7x smaller than the 0.175 MeV (0.122 even + 0.044 odd) that the
    u curve requires.  It then re-centred the reference on that sample mean,
    removed only 41 % of the pion's rms, and concluded "the pion's problem is
    not the mean".

    The sample mean of a residual whose skew is -12 and whose support runs to
    Tmax = 442 MeV is not a usable estimator at 200 000 events.  This command
    measures its actual spread rather than assuming a Gaussian standard error:
    <dqop> is computed on each 20 000-event seed file separately and the
    scatter across the ten is compared with the naive sqrt(N) error and with
    the MEDIAN, which is robust to the tail and shifts rigidly with a dE/dx
    error."""
    import glob
    from toy_loader import load_toy_sim
    print("=" * 140)
    print("IS THE SAMPLE MEAN A USABLE ESTIMATOR OF THE REFERENCE's MEAN-LOSS "
          "ERROR?  Outermost plane, per 20k seed file.")
    print("dE = <dqop> p^3/(qE) MeV, exactly NOTES_HADRONS s6.4's estimator.  "
          "`median` uses the median residual instead.")
    print("=" * 140)
    print(f"  {'species':<8}{'dE(mean)':>10}{'naive +-':>10}"
          f"{'seed scatter':>14}{'dE(median)':>12}{'med +-':>9}"
          f"{'skew':>8}{'  u-curve needs':>16}")
    for pdg in args.pdg:
        sp = hp.SPECIES[pdg]
        legs, cum, d, per = tmax_defect_per_leg(pdg)
        ns = {}
        exec(open(hp.planes_path(sp["geom"])).read(), ns)
        files = sorted(glob.glob(hp.sim_glob(pdg, args.arm)))
        k = len(legs) - 1
        P = 1e3 / abs(legs[k]["refqop"])                # MeV
        E = math.sqrt(P * P + sp["mass"] ** 2)
        # dqop is in 1/GeV in the loader; dE = dqop * p^3/(q E) with p in GeV
        pg = P * 1e-3
        conv = pg ** 3 / (sp["q"] * E * 1e-3) * 1e3     # -> MeV per (1/GeV)
        allres, permean, permed = [], [], []
        for f in files:
            s1 = load_toy_sim(f, ns["origin"], ns["normal"], ns["uaxis"])
            g = s1["valid"][:, k] & np.isfinite(s1["qop"][:, k])
            r = s1["qop"][g, k] - legs[k]["refqop"]
            allres.append(r)
            permean.append(r.mean() * conv)
            permed.append(np.median(r) * conv)
        r = np.concatenate(allres)
        dEm = r.mean() * conv
        naive = r.std(ddof=1) / math.sqrt(len(r)) * abs(conv)
        scat = np.std(permean, ddof=1) / math.sqrt(len(permean))
        dEmed = np.median(r) * conv
        # median error: 1.2533 sigma/sqrt(N) is Gaussian; use the seed scatter
        medscat = np.std(permed, ddof=1) / math.sqrt(len(permed))
        sk = float(((r - r.mean()) ** 3).mean() / r.std() ** 3)
        need = cum[-1]
        print(f"  {sp['label']:<8}{dEm:10.4f}{naive:10.4f}{scat:14.4f}"
              f"{dEmed:12.4f}{medscat:9.4f}{sk:8.1f}{need:16.4f}")
    print()
    print("  `u-curve needs` is this note's parameter-free charge-EVEN "
          "prediction (the charge-odd part is separate).")
    print("  A `seed scatter` far above `naive +-` means the sqrt(N) error is "
          "not the error and the mean is not a measurement.")


def cmd_trunc(args):
    """The reference's mean-loss error as a function of how much of the TAIL
    is allowed into the estimator.

    The closure's functional is <e^{-u z^2}>, which is bounded and therefore
    blind to the far tail: at u = 0.1 it has fallen to e^{-0.9} by |z| = 3 and
    to 1e-5 by |z| = 11.  The sample mean <z> is the opposite -- for a Landau
    whose support runs to Tmax = 442 MeV it is dominated by the events the
    closure cannot see.  So `dE(mean)` and `what the u curve needs` are
    estimators of DIFFERENT things whenever the model's far tail is imperfect,
    and NOTES_HADRONS s6.4/s6.5 used the tail-dominated one.

    This measures the crossover directly: the mean of the residual restricted
    to |z| < Z, as a function of Z, in the same MeV units."""
    import glob
    from toy_loader import load_toy_sim
    Zs = np.array(args.zcut, float)
    print("=" * 140)
    print("MEAN REFERENCE ERROR vs TAIL TRUNCATION.  dE from the residual "
          "restricted to |z| < Z, MeV; z is in Fisher units (s_F),")
    print("the same z the closure's e^{-u z^2} weights.  The last column is "
          "the untruncated sample mean = NOTES_HADRONS s6.4's estimator.")
    print("=" * 140)
    hdr = "".join(f"{f'Z<{z:g}':>10}" for z in Zs)
    print(f"  {'species':<8}" + hdr + f"{'no cut':>10}{'  even pred':>12}"
          f"{'odd pred':>10}{'kept @Zmin':>12}")
    import barkas_probe as bp
    for pdg in args.pdg:
        sp = hp.SPECIES[pdg]
        legs, cum, d, per = tmax_defect_per_leg(pdg)
        base = hp._rows(pdg, args.arm, corr="on")
        sF = base["sc"]["sF"]
        ns = {}
        exec(open(hp.planes_path(sp["geom"])).read(), ns)
        sim = hp.sim_of(pdg, args.arm)
        k = len(legs) - 1
        P = 1e3 / abs(legs[k]["refqop"])
        E = math.sqrt(P * P + sp["mass"] ** 2)
        conv = (P * 1e-3) ** 3 / (sp["q"] * E * 1e-3) * 1e3
        g = sim["valid"][:, k] & np.isfinite(sim["qop"][:, k])
        r = sim["qop"][g, k] - legs[k]["refqop"]
        z = r / float(sF[k])
        row = ""
        for Z in Zs:
            m = np.abs(z) < Z
            row += f"{r[m].mean() * conv:10.4f}"
        odd = -bp._hocodd_per_leg(pdg)[1][-1]
        kept = float((np.abs(z) < Zs.min()).mean())
        print(f"  {sp['label']:<8}" + row + f"{r.mean() * conv:10.4f}"
              f"{cum[-1]:12.4f}{odd:10.4f}{100 * kept:11.4f} %")
    print()
    print("  `even pred` is this note's proton-table Tmax defect and `odd "
          "pred` NOTES_BARKAS's high-order block;")
    print("  the reference's total predicted error is their SUM (both are "
          "over-subtractions, i.e. dE should be negative).")


# =========================================================================
# 8.  the C++ cross-check: let GEANT4 apply the correction, not me
# =========================================================================

CYLENV = dict(CVH_ELOSS_CYL_R="500.0", CVH_ELOSS_CYL_Z="900.0")


def cyl_model_path(pdg):
    return hp.model_path(pdg, "on")[:-5] + "_cyl.root"


def cmd_cylexport(args):
    """Re-export the model with GEANT4 ITSELF applying the correction.

    `G4ErrorEnergyLossForCVH::AlongStepDoIt` multiplies each step's reference
    mean loss by `exp(dxieff)`, and `CVH_ELOSS_CYL_EPS` adds a constant to
    `dxieff` for every step inside (r < CVH_ELOSS_CYL_R, |z| < CVH_ELOSS_CYL_Z).
    Set R and Z outside the whole toy and eps = -dE_defect/E_ioni and Geant4
    re-runs the entire reference propagation -- its own stepping, its own range
    tables, its own half-step rescaling -- with the defect removed.

    This is the analogue of NOTES_BARKAS's `refxcheck`: if the resulting
    closure matches the offline `refshift`, the offline shift is not a
    parameterization of the effect, it IS the effect.

    Two differences are deliberate and are reported rather than hidden:
      * the knob is proportional to each step's MEAN LOSS, the defect to each
        step's `xi`; they differ by the Bethe stopping number, which is
        material-dependent.  98.9 % of sum(xi) is one material, so the
        mis-distribution is sub-percent;
      * the knob does NOT touch `meanLoss` inside the fluctuation exporter, so
        the record's a1/a2 are unchanged -- which is the point: it isolates the
        reference mean, exactly what the offline shift models.
    """
    for pdg in args.pdg:
        sp = hp.SPECIES[pdg]
        legs, cum, d, per = tmax_defect_per_leg(pdg)
        P0, Pk = 1e3 / abs(legs[0]["refqop"]), 1e3 / abs(legs[-1]["refqop"])
        m = sp["mass"]
        Eioni = (math.sqrt(P0 * P0 + m * m) - math.sqrt(Pk * Pk + m * m))
        # eps is a LOG-scale offset on the mean loss; the correction is a
        # reduction, so eps < 0.  Use the whole-track ratio (the per-step
        # ratio varies by <1 % because one material carries 98.9 % of xi).
        eps = math.log(1.0 - cum[-1] / (Eioni + cum[-1]))
        out = cyl_model_path(pdg)
        log = out[:-5] + ".log"
        ev = dict(CYLENV)
        ev["CVH_ELOSS_CYL_EPS"] = repr(eps)
        ev["CVH_IONI_EXACTDELTA"] = "1"
        print(f"[{sp['label']}] E_ioni(leg0..outer) = {Eioni:.5f} MeV, "
              f"defect {cum[-1]:.5f} MeV -> eps = {eps:.9f}")
        if os.path.exists(out) and not args.force:
            print(f"  exists -> {out}")
            continue
        rc = hp._run(sp["geom"], "runToyModel.py",
                     f"pt={hp.PT} eta={hp.ETA} phi={hp.PHI} partId={pdg} "
                     f"output={out}", log, hp._env(pdg, "off", ev))
        print(f"  rc={rc} -> {out}")
        if rc:
            raise SystemExit(f"export failed, see {log}")


def cmd_cyl(args):
    """Compare GEANT4's own correction with the offline reference shift.

    LIVENESS FIRST: the re-exported reference must have lost exactly
    `dE_defect` less energy than the nominal one.  A knob that did not take
    shows up as a zero difference, not as a wrong closure."""
    print("=" * 140)
    print("GEANT4-SIDE CROSS-CHECK.  The reference re-propagated by Geant4 "
          "with the defect removed (CVH_ELOSS_CYL_EPS),")
    print("against the offline per-leg shift.  Both are parameter-free and "
          "neither is fitted to the other.")
    print("=" * 140)
    print(f"  {'species':<8}{'dE offline':>12}{'dE Geant4':>12}{'rel':>10}"
          f"   {'as exported':>12}{'offline shift':>15}{'Geant4 export':>15}")
    for pdg in args.pdg:
        sp = hp.SPECIES[pdg]
        legs, cum, d, per = tmax_defect_per_leg(pdg)
        cp = cyl_model_path(pdg)
        if not os.path.exists(cp):
            print(f"  {sp['label']}: {cp} missing (run `cylexport`)")
            continue
        l1 = cpt.load_model(cp)
        m = sp["mass"]
        P0 = 1e3 / abs(legs[-1]["refqop"])
        P1 = 1e3 / abs(l1[-1]["refqop"])
        dE = math.sqrt(P1 * P1 + m * m) - math.sqrt(P0 * P0 + m * m)
        base = hp._rows(pdg, args.arm, corr="on")
        m_off = _closure_with(_shifted_legs(pdg, cum), pdg, args.arm, base)
        # the Geant4-exported model, with its OWN Fisher scale (its record
        # moved too, if only in the last digits)
        assert os.environ.get("RES_NO_PHI_CACHE"), "phi cache is LIVE"
        ctr.IONI_KOKOULIN = 1.0 if hp.kok_for(pdg) else 0.0
        fn._SCALE_CACHE.clear()
        cpt._PHI_CACHE.clear()
        ol, os_ = fn._scale_cache_load, fn._scale_cache_store
        fn._scale_cache_load = lambda *a, **k: None
        fn._scale_cache_store = lambda *a, **k: None
        try:
            sc = fn.plane_scales(l1, "qop", tag=None,
                                 channels=("ioni", "ms", "rad"))
            m_g4 = gc.closure_rows(l1, hp.sim_of(pdg, args.arm), "qop",
                                   sc["sF"])[0].mean(axis=0)
        finally:
            fn._scale_cache_load, fn._scale_cache_store = ol, os_
            ctr.IONI_KOKOULIN = 0.0
            fn._SCALE_CACHE.clear()
            cpt._PHI_CACHE.clear()
        print(f"  {sp['label']:<8}{cum[-1]:12.5f}{dE:12.5f}"
              f"{(dE - cum[-1]) / cum[-1]:10.2e}   "
              f"{hp._rms(base['m']):12.5f}{hp._rms(m_off):15.5f}"
              f"{hp._rms(m_g4):15.5f}")
        print(f"  {'':<8}{'u curve G4':>12}  " + hp._fmt(m_g4))
        print(f"  {'':<8}{'u curve off':>12}  " + hp._fmt(m_off))
        print(f"  {'':<8}{'difference':>12}  " + hp._fmt(m_g4 - m_off))
        print(f"  {'':<8}{'+- (data)':>12}  " + hp._fmt(base["err"]))


def main():
    ap = argparse.ArgumentParser()
    s = ap.add_subparsers(dest="cmd", required=True)

    p = s.add_parser("control")
    p.add_argument("--pdg", type=int, nargs="+", default=ORDER)
    p.add_argument("--locx", action="store_true")
    p.set_defaults(f=cmd_control)

    p = s.add_parser("dedxdef")
    p.add_argument("--pdg", type=int, nargs="+", default=ORDER)
    p.set_defaults(f=cmd_dedxdef)

    p = s.add_parser("refshift")
    p.add_argument("--pdg", type=int, nargs="+",
                   default=[13, -211, 211, -321, 321, -2212, 2212])
    p.add_argument("--arm", default="off")
    p.add_argument("--func", default="qop")
    p.set_defaults(f=cmd_refshift)

    p = s.add_parser("shape")
    p.add_argument("--pdg", type=int, nargs="+", default=[-211])
    p.add_argument("--arm", default="off")
    p.add_argument("--scale", type=float, nargs="+",
                   default=[-2.0, -1.0, 1.0, 2.0])
    p.set_defaults(f=cmd_shape)

    p = s.add_parser("both")
    p.add_argument("--pdg", type=int, nargs="+", default=ORDER)
    p.add_argument("--arm", default="off")
    p.set_defaults(f=cmd_both)

    p = s.add_parser("fit")
    p.add_argument("--pdg", type=int, nargs="+", default=[-211, 211, -321, 321])
    p.add_argument("--arm", default="off")
    p.add_argument("--odd", action="store_true",
                   help="remove the charge-odd reference error first")
    p.set_defaults(f=cmd_fit)

    p = s.add_parser("meanloss")
    p.add_argument("--pdg", type=int, nargs="+", default=[-211, -321, -2212, 13])
    p.set_defaults(f=cmd_meanloss)

    p = s.add_parser("meancheck")
    p.add_argument("--pdg", type=int, nargs="+",
                   default=[13, -211, 211, -321, -2212, 2212])
    p.add_argument("--arm", default="off")
    p.set_defaults(f=cmd_meancheck)

    p = s.add_parser("trunc")
    p.add_argument("--pdg", type=int, nargs="+",
                   default=[13, -211, 211, -321, -2212, 2212])
    p.add_argument("--arm", default="off")
    p.add_argument("--zcut", type=float, nargs="+",
                   default=[3, 10, 30, 100, 300, 1000])
    p.set_defaults(f=cmd_trunc)

    p = s.add_parser("cylexport")
    p.add_argument("--pdg", type=int, nargs="+", default=[-211, 211, -321])
    p.add_argument("--force", action="store_true")
    p.set_defaults(f=cmd_cylexport)

    p = s.add_parser("cyl")
    p.add_argument("--pdg", type=int, nargs="+", default=[-211, 211, -321])
    p.add_argument("--arm", default="off")
    p.set_defaults(f=cmd_cyl)

    p = s.add_parser("spin")
    p.add_argument("--pdg", type=int, nargs="+", default=[-211, -321, 13])
    p.add_argument("--arm", default="off")
    p.set_defaults(f=cmd_spin)

    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
