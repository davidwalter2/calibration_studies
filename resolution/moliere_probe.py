#!/usr/bin/env python3
"""Where the Moliere / screened-Rutherford transform is PHYSICALLY WRONG.

CONTEXT
-------
NOTES_MSTERMS and NOTES_WVISPLIT enumerated the differences between the
offline transform and `G4WentzelVIModel` and retired all but one: the model's
`chi_c^2 ~ Z(Z+1)` gives the atomic electrons the NUCLEUS's angular range.
That candidate OVER-corrects (1.37-3.59x with the `ymax` snap), and its
implementation used a smooth dipole where the physics has a kinematic edge.
NOTES_MSTERMS s8.3 corrected for that analytically with a factor 1.27-1.68.

**That analytic correction is wrong**, and this module is what replaces it
with a measurement.  It compared the dipole against a SHARP CUT on the pure
Wentzel law.  The exact two-body kinematics of a heavy projectile on a free
electron give

    theta^2(T) = (2 m_e T / p^2) (1 - T/Tmax)

-- G4's `1 - cos = Tmax m_e/p^2` is the FIRST FACTOR ONLY -- so the deflection
turns over at T = Tmax/2 and is capped at theta_G4/2 EXACTLY, the density
carries the Jacobian of both branches (an integrable caustic at the edge), and

    L_e = ln(t_G4/chi_a^2) - 2 - beta^2/2

against G4's own transport-XS value of `- 1` and the dipole's `- 2.833`.  So
the dipole is 0.33 units of log too strong, not 1.83.

SUBCOMMANDS
    kin      per-species kinematics and every angular scale in the transform,
             read off the MODEL's own records (nothing assumed)
    assume   Moliere's assumptions, one by one, checked against our conditions
    kernels  the electron kernels against their closed-form transport logs,
             and the per-species removals in log units
    gauge    the closure, per species, for each kernel and combination
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("RES_NO_PHI_CACHE", "1")

import cf_ms_exact as cme                                        # noqa: E402
import cf_propagation_test as cpt                                # noqa: E402
import cf_track_resolution as ctr                                # noqa: E402
import fisher_norm as fn                                         # noqa: E402
import hadron_probe as hp                                        # noqa: E402
import radoff_species as rs                                      # noqa: E402
import wvisplit as wv                                            # noqa: E402

assert "_s1" in hp.sim_glob(13, "off"), "the seed pin is not installed"

LOGD = os.path.join(hp.SCRATCH, "moliere")
os.makedirs(LOGD, exist_ok=True)

UCURVE = fn.UCURVE
UROW = "".join(f"{u:>9g}" for u in UCURVE)
ORDER8 = [13, -13, -211, 211, -321, 321, -2212, 2212]

# the new knob, registered in wvisplit's dispatcher so cells can combine it
wv._KNOBS["MS_ELEC_EDGE"] = (ctr, float)
wv._KNOBS["MS_FINE_G"] = (ctr, float)

ALPHA = 1.0 / 137.035999084
ME = ctr._ME_GEV                     # GeV
HBARC = 0.19733                      # GeV fm


def _fmt(v, w=9, p=5):
    return "".join(f"{x:{w}.{p}f}" for x in v)


def _rms(v):
    return float(np.sqrt((np.asarray(v) ** 2).mean()))


# =========================================================================
# the per-species numbers, read off the model records
# =========================================================================

def _steps(pdg):
    """The pooled MS records of the OUTERMOST leg, with the derived Moliere
    parameters.  These are the model's own inputs, not a paraphrase."""
    legs = cpt.load_model(rs.mp(pdg, True))
    leg = legs[len(legs) - 1]
    st = leg["ms"]
    st = st[st[:, 5] > 0.]
    prm = np.array([cme.moliere_params(*s[:5], s[7], s[8]) if st.shape[1] >= 10
                    else cme.moliere_params(*s[:5]) for s in st])
    return leg, st, prm


def _species_scales(pdg):
    leg, st, prm = _steps(pdg)
    chic2, chia2, thff2 = prm[:, 0], prm[:, 1], prm[:, 2]
    w = chic2 / chic2.sum()                      # scattering-power weights
    pg = float(np.sum(w * st[:, 3]))
    bt = float(np.sum(w * st[:, 4]))
    gam = 1. / np.sqrt(1. - bt * bt)
    m = pg / (bt * gam)
    rat = ME / m
    tmax = 2. * ME * (bt * gam) ** 2 / (1. + 2. * gam * rat + rat * rat)
    tG4 = 2. * tmax * ME / (pg * pg)             # theta_G4^2 (G4's ceiling)
    d = dict(pdg=pdg, label=hp.SPECIES[pdg]["label"], p=pg, beta=bt, gamma=gam,
             mass=m, Tmax=tmax, nrec=len(st), xgtot=float(st[:, 2].sum()),
             chic2=float(np.sum(w * chic2)), chia2=float(np.sum(w * chia2)),
             thff2=float(np.sum(w * thff2)), tG4=tG4,
             te=0.25 * tG4,                      # the TRUE ceiling, theta_G4^2/4
             xg=float(np.mean(st[:, 2])), effZ=float(st[0, 0]),
             effA=float(st[0, 1]))
    d["ymFF"] = np.sqrt(d["thff2"] / d["chia2"])
    d["ymG4"] = np.sqrt(d["tG4"] / d["chia2"])
    d["ymTrue"] = np.sqrt(d["te"] / d["chia2"])
    d["nscat"] = d["chic2"] / d["chia2"]         # collisions per record
    d["sigp"] = np.sqrt(0.5 * d["chic2"] * (np.log(d["thff2"] / d["chia2"]) - 2.833))
    return d


def cmd_kin(args):
    print("=" * 140)
    print("EVERY ANGULAR SCALE IN THE TRANSFORM, per species, from the MODEL's "
          "own MS records (outermost leg).")
    print("=" * 140)
    print(f"  {'sp':<6}{'p [GeV]':>9}{'beta':>9}{'beta*gam':>10}{'m [MeV]':>10}"
          f"{'Tmax[MeV]':>11}{'chi_a[rad]':>12}{'th_FF':>10}{'th_G4':>10}"
          f"{'th_e,max':>10}{'sig_lay':>10}{'N_coll/lay':>11}")
    D = {}
    for pdg in ORDER8:
        d = _species_scales(pdg)
        D[pdg] = d
        print(f"  {d['label']:<6}{d['p']:9.4f}{d['beta']:9.6f}"
              f"{d['beta']*d['gamma']:10.3f}{d['mass']*1e3:10.3f}"
              f"{d['Tmax']*1e3:11.3f}{np.sqrt(d['chia2']):12.4e}"
              f"{np.sqrt(d['thff2']):10.5f}{np.sqrt(d['tG4']):10.2e}"
              f"{np.sqrt(d['te']):10.2e}{d['sigp']:10.2e}{d['nscat']:11.1f}")
    print()
    print("  the ceilings in units of the record's OWN rms projected deflection "
          "(this is what the closure sees):")
    print(f"  {'sp':<6}{'th_G4/sig':>12}{'th_e,max/sig':>14}{'th_FF/sig':>12}"
          f"{'ym_FF':>11}{'ym_G4':>11}{'ym_true':>11}{'records':>9}"
          f"{'xg tot':>9}")
    for pdg in ORDER8:
        d = D[pdg]
        print(f"  {d['label']:<6}{np.sqrt(d['tG4'])/d['sigp']:12.2f}"
              f"{np.sqrt(d['te'])/d['sigp']:14.2f}"
              f"{np.sqrt(d['thff2'])/d['sigp']:12.1f}"
              f"{d['ymFF']:11.4g}{d['ymG4']:11.4g}{d['ymTrue']:11.4g}"
              f"{d['nrec']:9d}{d['xgtot']:9.4f}")
    return D


# =========================================================================
# the assumption audit
# =========================================================================

def cmd_assume(args):
    D = {p: _species_scales(p) for p in ORDER8}
    d0 = D[13]
    Z, A = d0["effZ"], d0["effA"]
    print()
    print("=" * 140)
    print("MOLIERE'S ASSUMPTIONS, ONE BY ONE, AGAINST OUR CONDITIONS")
    print("=" * 140)

    # --- A1 screening -----------------------------------------------------
    print("\nA1.  THOMAS-FERMI SCREENING.  chi_a^2 = chi_0^2 (1.13 + 3.76 (alpha Z/beta)^2) "
          "* (1 + e^{-Z^2/1000}).")
    print("     The Born term 3.76 (alpha Z/beta)^2 is the ONLY beta-dependence, and at "
          "fixed p it is the")
    print("     only way the screening can be species-dependent at all.")
    az = ALPHA * Z
    print(f"     {'sp':<6}{'beta':>9}{'1.13+3.76(aZ/b)^2':>20}{'rel to mu':>12}"
          f"{'d ln chi_a^2':>14}{'as % of Lm=17':>15}")
    ref = None
    for pdg in ORDER8:
        b = D[pdg]["beta"]
        v = 1.13 + 3.76 * (az / b) ** 2
        if ref is None:
            ref = v
        print(f"     {D[pdg]['label']:<6}{b:9.6f}{v:20.6f}{v/ref:12.6f}"
              f"{np.log(v/ref):14.2e}{100*np.log(v/ref)/17.0:15.3f}")
    print("     -> the whole species spread of the SCREENING ANGLE is "
          f"{100*np.log((1.13+3.76*(az/D[2212]['beta'])**2)/ref)/17.0:.3f} % of the "
          "Moliere log.")
    print("     The (1+e^{-Z^2/1000}) factor IS the Hartree-Fock-vs-Thomas-Fermi "
          "correction the task asks")
    print("     about (Butkevich et al., NIM A488 (2002) 282, cited at "
          "G4WentzelOKandVIxSection.cc:135).")
    print(f"     It is already in the model (G4_SCREEN_F = {cme.G4_SCREEN_F}); "
          f"for Z = {Z:.0f} it is a factor {1+np.exp(-Z*Z*1e-3):.4f} on chi_a^2, "
          f"i.e. {np.log(1+np.exp(-Z*Z*1e-3)):.3f} units of log.")
    print("     It depends on Z and p ONLY.  At fixed pT every species has the SAME p, "
          "so it is EXACTLY")
    print("     species-independent.  VERDICT: real, already implemented, cannot "
          "produce a species ordering.")

    # --- A2 small angle ---------------------------------------------------
    print("\nA2.  THE SMALL-ANGLE APPROXIMATION.  The transform uses t = theta^2 with "
          "sin theta -> theta and")
    print("     1 - cos theta -> theta^2/2.  The relative error of the second is "
          "theta^2/12.")
    print(f"     {'sp':<6}{'sig_lay':>11}{'th at 30 sig':>14}{'err@30sig':>12}"
          f"{'th_FF':>10}{'err@th_FF':>12}")
    for pdg in ORDER8:
        d = D[pdg]
        s30 = 30 * d["sigp"]
        print(f"     {d['label']:<6}{d['sigp']:11.3e}{s30:14.3e}{s30**2/12:12.2e}"
              f"{np.sqrt(d['thff2']):10.5f}{d['thff2']/12:12.2e}")
    print("     -> at the largest angle any closure probe can reach the correction is "
          "~1e-6; at the")
    print("     form-factor angle, where the density is already suppressed by 1e-4, it "
          "is 5e-4.")
    print("     VERDICT: NULL.  Not species-dependent either (the angular scale spans "
          "4 % across species).")

    # --- A3 many collisions ----------------------------------------------
    print("\nA3.  THE MANY-COLLISIONS ASSUMPTION.  Moliere's B-expansion is asymptotic in "
          "Omega_0 = chi_c^2/(1.167 chi_a^2)")
    print("     and is quoted as needing Omega_0 >~ 20.  THE TRANSFORM DOES NOT MAKE "
          "THIS ASSUMPTION: it is the")
    print("     EXACT compound-Poisson CF exp[Int (J0(t theta) - 1) dn], not Moliere's "
          "series -- so the")
    print("     asymptotics cannot break it.  The census is printed anyway, because it "
          "is the number the")
    print("     question is about:")
    print(f"     {'sp':<6}{'coll/layer':>13}{'legs = layers':>15}"
          f"{'coll/layer':>12}{'coll/track':>12}{'Omega_0/layer':>15}")
    for pdg in ORDER8:
        d = D[pdg]
        rpl = d["nrec"] / 14.0
        cpl = d["nscat"] * rpl
        print(f"     {d['label']:<6}{d['nscat']:13.1f}{rpl:15.2f}{cpl:12.1f}"
              f"{cpl*14:12.0f}{cpl/1.167:15.1f}")
    print("     -> Omega_0 per LAYER is ~1e4 and per TRACK ~1e5, against the >~20 "
          "Moliere needs.  The")
    print("     assumption is satisfied by three to four orders of magnitude AND is not "
          "made.  VERDICT: NULL,")
    print("     twice over.")

    # --- A4 fixed beta ----------------------------------------------------
    print("\nA4.  FIXED beta THROUGH A STEP (no energy-loss / scattering correlation). "
          " chi_c^2 ~ 1/(p beta)^2,")
    print("     so the question is how much (p beta)^2 moves across one record.")
    print(f"     {'sp':<6}{'p first':>11}{'p last':>11}{'d(pbeta)^2 rel':>16}"
          f"{'per record':>13}")
    for pdg in ORDER8:
        leg, st, prm = _steps(pdg)
        pb = st[:, 3] * st[:, 4]
        rel = float(pb[0] ** 2 / pb[-1] ** 2 - 1.)
        print(f"     {hp.SPECIES[pdg]['label']:<6}{st[0,3]:11.5f}{st[-1,3]:11.5f}"
              f"{rel:16.3e}{rel/len(st):13.2e}")
    print("     -> the model carries the per-record p and beta explicitly, so the only "
          "error is WITHIN a")
    print("     record, and that is ~1e-6.  VERDICT: NULL.")

    # --- A5 the nuclear form factor / point nucleus ------------------------
    print("\nA5.  THE POINT NUCLEUS.  Moliere's single-scattering tail is 1/theta^4 to "
          "infinity; a real nucleus")
    print("     has a form factor.  The transform ALREADY carries G4's exponential FF "
          "SQUARED, (1+t/th_FF^2)^-4.")
    print(f"     th_FF^2 = 12 (hbar c/p R)^2 with R = 1.27 A^0.27 fm against G4's "
          f"2/formfactA: ratio 1.0048.")
    print("     VERDICT: already implemented and correct to 0.5 %; species-INDEPENDENT "
          "(depends on p and A only).")

    # --- A6 Z(Z+1): the range ---------------------------------------------
    print("\nA6.  Z(Z+1): ONE ANGULAR LAW AND ONE ANGULAR RANGE FOR THE NUCLEUS AND THE "
          "ATOMIC ELECTRONS.")
    print("     THIS IS THE DEFECT.  The electron term has a kinematic ceiling the "
          "nucleus does not, and it")
    print("     is the ONLY term in the enumeration that depends on the species at "
          "fixed momentum.")
    print(f"     {'sp':<6}{'L_e model':>11}{'L_e G4':>9}{'L_e exact':>11}"
          f"{'model-exact':>13}{'x 1/(Z+1)':>11}{'% of Lm':>9}{'th_e,max/sig':>14}")
    for pdg in ORDER8:
        d = D[pdg]
        Lmod = np.log(d["thff2"] / d["chia2"]) - 2.8333
        Lg4 = np.log(d["tG4"] / d["chia2"]) - 1.
        Lex = np.log(d["tG4"] / d["chia2"]) - 2. - 0.5 * d["beta"] ** 2
        dl = Lmod - Lex
        print(f"     {d['label']:<6}{Lmod:11.3f}{Lg4:9.3f}{Lex:11.3f}{dl:13.3f}"
              f"{dl/(d['effZ']+1):11.3f}{100*dl/(d['effZ']+1)/Lmod:9.2f}"
              f"{np.sqrt(d['te'])/d['sigp']:14.2f}")
    print("     -> the model over-states the TOTAL MS transport moment by 2.4-4.8 %, "
          "monotonically in 1/beta.")
    print("     But the log is NOT the currency: the ceiling sits at 1.4 sigma for the "
          "proton and 12 sigma")
    print("     for the muon, so the closure weights the same log very differently -- "
          "which is why `gauge`")
    print("     measures it instead of scaling it.")

    # --- A7 the two-branch kinematics -------------------------------------
    print("\nA7.  THE ELECTRON DEFLECTION IS NOT MONOTONIC IN THE ENERGY TRANSFER, and "
          "BOTH Moliere and G4")
    print("     assume it is.  theta^2(T) = (2 m_e T/p^2)(1 - T/Tmax) turns over at "
          "T = Tmax/2:")
    print(f"     {'sp':<6}{'th_G4 [rad]':>13}{'th_e,max':>11}{'ratio':>8}"
          f"{'T*(=Tmax/2) MeV':>17}{'L_e G4 - L_e exact':>20}")
    for pdg in ORDER8:
        d = D[pdg]
        print(f"     {d['label']:<6}{np.sqrt(d['tG4']):13.4e}{np.sqrt(d['te']):11.4e}"
              f"{np.sqrt(d['tG4']/d['te']):8.4f}{0.5*d['Tmax']*1e3:17.4f}"
              f"{1.+0.5*d['beta']**2:20.4f}")
    print("     -> theta_e,max = theta_G4/2 EXACTLY, for every species and every "
          "momentum, and G4's own")
    print("     transport cross section over-states the electron transport moment by "
          "1 + beta^2/2 = 1.5")
    print("     units of log.  Harmonising to G4 here would NOT be harmonising to the "
          "physics.")

    # --- A8 dsigma/dT -----------------------------------------------------
    print("\nA8.  THE ELECTRON CROSS SECTION IS PURE 1/T^2 IN BOTH.  The true spin-0 "
          "dsigma/dT carries")
    print("     (1 - beta^2 T/Tmax) -- the SAME factor whose omission was the first of "
          "the four ionization")
    print("     corrections.  It costs beta^2/2 of the electron log, i.e. "
          f"{0.5*0.999/(Z+1)/17.0*100:.2f} % of the total MS")
    print("     transport moment, and its species spread (beta^2 from 0.918 to 0.999) "
          "is 4 %.")
    print("     VERDICT: real, parameter-free, implemented here as MS_ELEC_EDGE = 2 vs 3.")

    # --- A9 spin / second Born --------------------------------------------
    print("\nA9.  NO SPIN AND NO SECOND-BORN (MOTT) TERM.  Measured in NOTES_MSTERMS "
          "s3.4: 1.6e-4 of the")
    print("     transport XS inside the WentzelVI range, zero for the spin-0 species, "
          "and ANTI-correlated")
    print("     with the observed ordering.  VERDICT: excluded, unchanged.")

    # --- A10 independence -------------------------------------------------
    print("\nA10. INDEPENDENT SUCCESSIVE COLLISIONS (the compound Poisson).  "
          "NOTES_STEPCORR measured the")
    print("     step-to-step correlation at +0.0037 and the closure needs -0.088 of the "
          "wrong sign.")
    print("     VERDICT: excluded on sign, unchanged.")
    return D


# =========================================================================
# the kernels
# =========================================================================

def cmd_kernels(args):
    print("=" * 132)
    print("V1.  THE ELECTRON KERNELS AGAINST THEIR CLOSED-FORM TRANSPORT LOGS.")
    print("     hard   : ln(1+Y) - Y/(1+Y)          Y = (theta_G4/chi_a)^2")
    print("     kine   : ln(4Y) - 2 - beta^2/2      Y = (theta_e,max/chi_a)^2 "
          "= (theta_G4/2chi_a)^2")
    print("     dipole : ln(Y) - 2.8333")
    print("=" * 132)
    print(f"  {'Y':>12} {'kind':>8} {'beta^2':>8} {'table':>12} {'closed form':>12} "
          f"{'rel':>10}")
    worst = 0.
    for Y in (7.3e4, 2.6e5, 1.0e6, 2.86e6, 4.51e6, 4.09e8):
        for kind, ana, b2 in (("hard", np.log(1 + Y) - Y / (1 + Y), 0.),
                              ("kine", np.log(4 * Y) - 2., 0.),
                              ("kine", np.log(4 * Y) - 2. - 0.4588, 0.9176),
                              ("dipole", np.log(Y) - 2.8333, 0.)):
            t = cme.elec_logrange(Y, kind, b2)
            worst = max(worst, abs(t / ana - 1))
            print(f"  {Y:12.4g} {kind:>8} {b2:8.4f} {t:12.6f} {ana:12.6f} "
                  f"{t/ana-1:10.2e}")
    print(f"\n  worst |rel| : {worst:.2e}")

    print()
    print("=" * 132)
    print("V2.  THE NUMERICS CONTROL.  `dipole` on the new 4.4x-finer quadrature "
          "against `gshape`'s own")
    print("     table at the SAME ymax.  This is the part of any MS_ELEC_EDGE move "
          "that is NOT physics.")
    print("=" * 132)
    print(f"  {'ymax':>12}{'gshape Lm':>13}{'new Lm':>13}{'rel':>11}")
    for ym in (271., 512., 1690., 2124., 2.026e4):
        t0 = cme._GTAU[0]
        g = cme.gshape(np.array([t0]), ymax=ym)[0]
        lm0 = -4. * g / t0 ** 2
        lm1 = cme.elec_logrange(ym ** 2, "dipole")
        print(f"  {ym:12.4g}{lm0:13.6f}{lm1:13.6f}{lm1/lm0-1:11.2e}")

    print()
    print("=" * 132)
    print("V3.  THE PER-SPECIES REMOVAL, in units of the model's own Moliere log, for "
          "each kernel.")
    print("     removal = L_e(model, dipole to theta_FF) - L_e(kernel), weighted by "
          "1/(Z+1).")
    print("=" * 132)
    print(f"  {'sp':<6}{'L_e model':>11}{'dipole@G4':>11}{'hard@G4':>10}"
          f"{'kine@true':>11}{'kine b2=0':>11}   "
          f"{'removal: dip':>13}{'hard':>8}{'kine':>8}{'kine0':>8}")
    for pdg in ORDER8:
        d = _species_scales(pdg)
        Zp1 = d["effZ"] + 1.
        Lmod = cme.elec_logrange(d["ymFF"] ** 2, "dipole")
        Ld = cme.elec_logrange(d["ymG4"] ** 2, "dipole")
        Lh = cme.elec_logrange(d["ymG4"] ** 2, "hard")
        Lk = cme.elec_logrange(d["ymTrue"] ** 2, "kine", d["beta"] ** 2)
        Lk0 = cme.elec_logrange(d["ymTrue"] ** 2, "kine", 0.)
        print(f"  {d['label']:<6}{Lmod:11.3f}{Ld:11.3f}{Lh:10.3f}{Lk:11.3f}"
              f"{Lk0:11.3f}   " + "".join(
                  f"{(Lmod-x)/Zp1:{w}.3f}" for x, w in
                  ((Ld, 13), (Lh, 8), (Lk, 8), (Lk0, 8))))
    print("\n  (the last four columns are the removal from the TOTAL Moliere log, "
          "which is 17.00 for this toy)")


# =========================================================================
# the closure
# =========================================================================

def cmd_gauge(args):
    print("=" * 160)
    print("THE CLOSURE.  Each row is a controlled change of the MODEL's MS channel at "
          "fixed simulation.")
    print(f"sim arm = {args.arm}   functional = {args.func}")
    print("=" * 160)
    print(f"  {'species':<7}{'cell':<40}" + UROW + "      rms   cos(resid)")
    res = {}
    for pdg in args.pdg:
        lab = hp.SPECIES[pdg]["label"]
        ctr.MS_WVI_NPERX, ctr.MS_WVI_LG = wv.SP_INPUT[lab]
        base = wv._rows(pdg, args.func, arm=args.arm)
        print(f"  {lab:<7}{'BASE (residual)':<40}" + _fmt(base["m"])
              + f" {_rms(base['m']):9.5f}      1.000")
        print(f"  {'':<7}{'+-':<40}" + _fmt(base["err"]))
        res[(lab, "BASE")] = base["m"]
        res[(lab, "ERR")] = base["err"]
        for cell in args.cells:
            spec = dict(kv.split("=", 1) for kv in cell.split(","))
            old = wv._set(spec)
            try:
                r = wv._rows(pdg, args.func, arm=args.arm)
            finally:
                wv._restore(old)
            d = r["m"] - base["m"]
            nb, nd = np.linalg.norm(base["m"]), np.linalg.norm(d)
            cos = float(d @ base["m"] / (nb * nd)) if nb * nd > 0 else np.nan
            res[(lab, cell)] = d
            print(f"  {'':<7}{cell:<40}" + _fmt(d)
                  + f" {_rms(d):9.5f} {cos:10.3f}")
        print()
    if args.npz:
        np.savez(args.npz, u=UCURVE,
                 **{f"{l}|{c}": v for (l, c), v in res.items()})
        print(f"wrote {args.npz}")
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    s = p.add_subparsers(dest="cmd", required=True)

    q = s.add_parser("kin")
    q.set_defaults(f=cmd_kin)

    q = s.add_parser("assume")
    q.set_defaults(f=cmd_assume)

    q = s.add_parser("kernels")
    q.set_defaults(f=cmd_kernels)

    q = s.add_parser("gauge")
    q.add_argument("--pdg", type=int, nargs="+", default=ORDER8)
    q.add_argument("--func", default="locx")
    q.add_argument("--arm", default="off")
    q.add_argument("--cells", nargs="+",
                   default=["MS_ELEC_TMAX=1", "MS_ELEC_TMAX=1,MS_ELEC_EDGE=0.5",
                            "MS_ELEC_TMAX=1,MS_ELEC_EDGE=1",
                            "MS_ELEC_TMAX=1,MS_ELEC_EDGE=2",
                            "MS_ELEC_TMAX=1,MS_ELEC_EDGE=3"])
    q.add_argument("--npz", default=None)
    q.set_defaults(f=cmd_gauge)

    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
