#!/usr/bin/env python3
"""The INTERNAL Gaussian/single-scattering split of `G4WentzelVIModel`,
modelled offline, validated against G4's own sampler, and propagated.

CONTEXT
-------
NOTES_MSTERMS s10 named this as "the strongest remaining" candidate for the
`locx` non-closure and did not test it.  The claim there was:

    G4WentzelVIModel replaces sub-threshold scattering with an EXACT Gaussian
    (`ComputeSecondTransportMoment` returns 0, so the Gamma(2,2) branch never
    fires) at a boundary sitting at ~1.1 sigma, while our analytic model
    carries a compound-Poisson tail there.  Sign and lineshape both match.

WHAT THE SPLIT ACTUALLY IS  (source read, not paraphrased)
----------------------------------------------------------
`G4WentzelVIModel::ComputeTrueStepLength` sets

    1 - cosThetaMin = ssFactor * t / lambda_transport          (ssFactor = 1.25)
    lambdaeff       = 1 / [ n * sigma_tr(cosThetaMin) ]        RESTRICTED
    xtsec           = n * [ nuclear(cosThetaMin, cosTetMaxNuc)
                          + electron(cosThetaMin, cosTetMaxElec) ]

and `SampleScattering` draws z ~ Exp(mean z0 = t/(2 lambdaeff)), cost = 1-2z,
plus explicit single scatters at rate xtsec.

THE STRUCTURAL POINT, and it decides the answer:

    <1 - cos>_Gaussian = 2 z0 = t / lambdaeff = t n sigma_tr(< theta_min)

i.e. the Gaussian is matched to the EXACT RESTRICTED FIRST TRANSPORT MOMENT of
the very law it replaces.  For a 2D isotropic deflection the first transport
moment IS the projected variance, and a compound Poisson's second cumulant IS
its transport moment.  **So the split preserves the variance exactly and
changes only the fourth and higher cumulants below theta_min.**  It is a
kurtosis transfer at fixed variance, not a magnitude change -- and the
residual it was proposed to explain is a magnitude effect (NOTES_MSTERMS s5).

SUBCOMMANDS
    steps     the sim's step structure inside the toy, MEASURED from the
              archived censuses (ionization secondaries per layer -> the
              Poisson rate that sets the step length distribution)
    g4        parse the driver, validate the offline re-implementation of
              G4's transport cross section against G4's own lambda, and
              compare the analytic CFs against what the sampler draws
    gauge     propagate through the closure, per species, alone and jointly
              with the electron-ceiling candidate
"""

import argparse
import glob
import os
import subprocess
import sys

import numpy as np
from scipy.special import j0 as _j0

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("RES_NO_PHI_CACHE", "1")

import cf_propagation_test as cpt                                # noqa: E402
import cf_track_resolution as ctr                                # noqa: E402
import cf_ms_exact as cme                                        # noqa: E402
import fisher_norm as fn                                         # noqa: E402
import geom_closure as gc                                        # noqa: E402
import hadron_probe as hp                                        # noqa: E402
import speciesdedx as sd                                         # noqa: E402
import radoff_species as rs                                      # noqa: E402
import tail_probe as tp                                          # noqa: E402

OUT = hp.OUT
LOGD = os.path.join(hp.SCRATCH, "wvi")
os.makedirs(LOGD, exist_ok=True)
DRIVER = os.path.join(hp.SCRATCH, "wvisplit_g4driver.sh")

UCURVE = fn.UCURVE
UROW = "".join(f"{u:>9g}" for u in UCURVE)

# The seed pin of NOTES_PION s1 must be installed (importing radoff_species is
# what installs it, via speciesdedx -> pion_probe).  Asserted, not assumed:
# NOTES_CLOSURE_FINAL s2.1 found this pin was HALF a pin and it moved a
# published number.
assert "_s1" in hp.sim_glob(13, "off"), "the seed pin is not installed"

ALPHA = 1.0 / 137.035999084
FACTB1 = 0.5 * np.pi * ALPHA
NUMLIMIT = 0.1          # G4WentzelOKandVIxSection.cc numlimit
SSFACTOR = 1.25         # G4WentzelVIModel::SetSingleScatteringFactor(1.25)

# Layer geometry of the toy, from Analysis/HitAnalyzer/data/tracker.xml in the
# private hadprobe area: 14 shells of 0.1 cm radial thickness, rho = 9 g/cm3,
# crossed at eta = 0.30 -> path 0.1/sin(theta) = 0.10450 cm.
TOY_PATH_CM = 0.1045
TOY_NLAYER = 14


def _fmt(v, w=9, p=5):
    return "".join(f"{x:{w}.{p}f}" for x in v)


def _rms(v):
    return float(np.sqrt((np.asarray(v) ** 2).mean()))


# =========================================================================
# 1.  G4's transport cross section, re-implemented offline
# =========================================================================

def g4_f(x, fb):
    """`G4WentzelOKandVIxSection::ComputeTransportCrossSectionPerAtom`'s inner
    function, both branches, quoted:

        x < numlimit : x2*((1 - 1.3333333 x + 3 x2) - fb*x*(0.6666667 - x)),
                       x2 = 0.5 x^2
        else         : ln(1+x) - x/(1+x) - fb*(x + x/(1+x) - 2 ln(1+x))
    """
    x = np.asarray(x, float)
    out = np.empty_like(x)
    lo = x < NUMLIMIT
    if np.any(lo):
        xl = x[lo]
        x2 = 0.5 * xl * xl
        out[lo] = x2 * ((1.0 - 1.3333333 * xl + 3 * x2) - fb * xl * (0.6666667 - xl))
    if np.any(~lo):
        xh = x[~lo]
        x1 = xh / (1.0 + xh)
        xlog = np.log1p(xh)
        out[~lo] = xlog - x1 - fb * (xh + x1 - 2 * xlog)
    return np.maximum(out, 0.0)


def sigma_tr(st, omc_max):
    """G4's transport cross section per atom at an upper limit `omc_max`
    (= 1 - cosTMax), with the electron term clipped at its own ceiling.
    Reproduces `ComputeTransportCrossSectionPerAtom` line for line."""
    a, fb = st["screenZ"], st["screenZ"] * st["factB"]
    omc_max = np.asarray(omc_max, float)
    # costm = max(cosTMax, cosTetMaxElec)  ->  omc_e = min(omc_max, omcElec)
    ome = np.minimum(omc_max, st["omcElec"])
    xs = g4_f(ome / a, fb) + st["Z"] * g4_f(omc_max / a, fb)
    return st["kinFactor"] * xs


def grej(st, z, electron=False):
    """The single-scattering rejection function, quoted from
    `SampleSingleScattering`:

        fm   = 1/(1 + formfactA z)^2                (fExponentialNF; formf = 0
                                                     for the electron branch)
        grej = (1 - z factB + factB1 Z sqrt(z factB)(2-z)) fm^2 / (1 + z factD)

    accepted with probability min(grej, 1) since the comparison is
    `fMottFactor*flat() <= grej` with fMottFactor = 1 (no Mott for mu/hadrons).
    """
    z = np.asarray(z, float)
    g = (1.0 - z * st["factB"]
         + FACTB1 * st["Z"] * np.sqrt(np.maximum(z * st["factB"], 0.0)) * (2.0 - z))
    g = g / (1.0 + z * st["factD"])
    if not electron:
        g = g / (1.0 + st["formfactA"] * z) ** 4
    return np.minimum(g, 1.0)


def _j0m1(x):
    """J0(x) - 1, guarded at small x.  `j0(x)-1` is -x^2/4 while both terms are
    ~1, so below |x| ~ 1e-4 it loses all precision -- the same cancellation
    NOTES_XXII s8.4 found corrupting `cf_ms_exact`'s G table."""
    x = np.asarray(x, float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-3
    if np.any(small):
        u = 0.25 * x[small] ** 2
        out[small] = -u * (1.0 - u / 4.0 + u * u / 36.0)
    if np.any(~small):
        out[~small] = _j0(x[~small]) - 1.0
    return out


def s_singles(st, s, z1, z2, C, electron=False, nz=40000):
    """log-CF of a compound Poisson of single scatters on [z1, z2]:

        S(s) = C int_{z1}^{z2} grej(z)/(z+a)^2 [ J0(s theta(z)) - 1 ] dz,
        theta(z) = arccos(1-z)                       (exact, not sqrt(2z))

    `C` = n L kinFactor Z (nucleus) or n L kinFactor (electrons): the
    normalization that makes int dn/dz reproduce G4's own
    Compute{Nuclear,Electron}CrossSection exactly (checked in `cmd_g4`)."""
    s = np.atleast_1d(np.asarray(s, float))
    if z2 <= z1:
        return np.zeros(len(s))
    a = st["screenZ"]
    lo = max(z1, a * 1e-8)
    zz = np.geomspace(lo, z2, nz)
    if z1 <= 0.0:
        # the [0, lo] head, where J0-1 = -s^2 theta^2/4 = -s^2 z/2 exactly to
        # 1e-16 and the integral is elementary
        head = -0.25 * s * s * C * (np.log1p(lo / a) - lo / (lo + a))
    else:
        head = np.zeros(len(s))
    th = np.arccos(np.clip(1.0 - zz, -1.0, 1.0))
    w = C * grej(st, zz, electron) / (zz + a) ** 2 * zz     # * z for dz -> dlnz
    ker = _j0m1(s[:, None] * th[None, :])
    return head + np.trapezoid(w[None, :] * ker, np.log(zz), axis=1)


def wvi_step(st, L):
    """The G4 quantities for one step of true length `L` (cm), exactly as
    `ComputeTrueStepLength` computes them."""
    invlam_full = st["natoms"] * sigma_tr(st, st["omcNuc"])
    zmin = SSFACTOR * L * invlam_full
    # restricted transport moment -> the Gaussian's projected variance
    sig2 = L * st["natoms"] * float(sigma_tr(st, min(zmin, st["omcNuc"])))
    return dict(zmin=float(zmin), sig2=float(sig2), invlam_full=float(invlam_full))


def s_split_step(st, s, L, nz=40000):
    """One step's log-CF as WentzelVI samples it: the variance-matched Gaussian
    below theta_min plus the explicit single scatters above it."""
    q = wvi_step(st, L)
    S = -0.5 * np.asarray(s, float) ** 2 * q["sig2"]
    if q["zmin"] < st["omcNuc"]:
        S = S + s_singles(st, s, q["zmin"], st["omcNuc"],
                          st["natoms"] * L * st["kinFactor"] * st["Z"], nz=nz)
    # the electron term contributes NOTHING above theta_min: omcElec = 4.93e-10
    # is three orders BELOW zmin ~ 7e-8, so ComputeElectronCrossSection returns
    # exactly 0 there.  Verified in the driver: elecRatio = 0 identically.
    return S


def s_full_path(st, s, D, nz=40000):
    """The pure compound Poisson over the whole path -- the structure the
    offline transform has."""
    return (s_singles(st, s, 0.0, st["omcNuc"], st["natoms"] * D * st["kinFactor"]
                      * st["Z"], nz=nz)
            + s_singles(st, s, 0.0, st["omcElec"], st["natoms"] * D * st["kinFactor"],
                        electron=True, nz=nz))


# =========================================================================
# 2.  the driver
# =========================================================================

def parse_driver(path):
    st, mom, hist = {}, {}, {}
    mat = {}
    for line in open(path):
        w = line.split()
        if not w:
            continue
        if w[0] == "MAT":
            mat = {w[i]: float(w[i + 1]) for i in range(1, len(w) - 1, 2)}
        elif w[0] == "STATE":
            d = {w[i]: float(w[i + 1]) for i in range(2, len(w) - 1, 2)}
            st[w[1]] = d
        elif w[0] == "MOM":
            mom[(w[1], w[2])] = {w[i]: float(w[i + 1]) for i in range(3, len(w) - 1, 2)}
        elif w[0] == "HIST":
            nb, lb0, dlb = int(w[3]), float(w[4]), float(w[5])
            h = np.zeros(nb + 2)
            for tok in w[6:]:
                i, v = tok.split(":")
                h[int(i)] = float(v)
            hist[(w[1], w[2])] = (nb, lb0, dlb, h)
    return mat, st, mom, hist


def ecf_from_hist(hb, s):
    """Empirical CF of the PROJECTED angle from the log histogram of the total
    polar angle: <J0(s theta)>, exact for an isotropic 2D deflection.  The
    error is the multinomial one, computed from the same histogram."""
    nb, lb0, dlb, h = hb
    N = h.sum()
    edges = np.exp(lb0 + dlb * np.arange(nb + 1))
    ctr_ = np.sqrt(edges[:-1] * edges[1:])
    th = np.concatenate(([0.0], ctr_, [np.pi]))
    w = h / N
    s = np.atleast_1d(np.asarray(s, float))
    v = _j0(s[:, None] * th[None, :])
    m = (w[None, :] * v).sum(axis=1)
    m2 = (w[None, :] * v * v).sum(axis=1)
    return m, np.sqrt(np.maximum(m2 - m * m, 0.0) / N)


def state_of(st, lab, mat):
    """Everything stays in GEANT4 INTERNAL UNITS (mm): `natomsG4` is 1/mm^3,
    `kinFactor` carries mm^2, `lambdaFull`/`tpathG4`/`lmeanG4` are mm.  Mixing
    in cm here is a silent factor 100 on every cross section."""
    d = dict(st[lab])
    d["Z"] = mat["Z"]
    d["natoms"] = d["natomsG4"]                # 1/mm^3
    return d


def cmd_steps(args):
    """The sim's step structure inside the toy, MEASURED.

    The number of G4 steps per layer is what the split's size scales with (the
    effect is a per-step fourth cumulant, so it goes as sum_s L_s^2 = D^2/N_eff),
    and NOTES_MSTERMS s9 inferred it from the FULL-TRACK muIoni census, which
    also counts the calorimeter -- the toy layers stop at r = 106.8 cm and the
    primary continues into the standard CMS geometry and stops there.  The
    census record has `iNstep` and `iNsecProc`, both gated at r < 107 cm, so
    the number can be measured instead."""
    print("=" * 118)
    print("STEP STRUCTURE INSIDE THE TOY, from the archived censuses "
          "(gated at r < 107 cm by the watcher itself).")
    print("=" * 118)
    print("Steps in a layer = 1 (the exit boundary) + Poisson(n) delta rays.  "
          "The delta count is the census's own")
    print("ionization-secondary count; msc defines 372 steps in 2e5 events and "
          "the range-based limits are metres, so")
    print("nothing else limits a step inside a 1.045 mm layer.")
    print()
    print(f"  {'species':<7}{'nev':>8}{'<Nstep> r<107':>15}{'ioni sec/ev':>13}"
          f"{'n/layer':>9}{'other/ev':>10}{'sum xg':>9}{'NPERX':>9}"
          f"{'sumL2/D2':>10}{'N_eff':>8}")
    out = {}
    for pdg in args.pdg:
        c = hp.census_of(pdg, "off")
        ns = float(c[:, tp.I_NSTEP].mean())
        P = tp.PROCS
        nsec = c[:, tp.I_NSECPROC:tp.I_NSECPROC + len(P)].mean(axis=0)
        nion = float(nsec[P.index("muIoni")] + nsec[P.index("otherIoni")])
        n = nion / TOY_NLAYER
        # the MODEL's own total material, so that NPERX is expressed in the
        # units the knob divides by (the record's `xg`) rather than in the
        # geometry's nominal thickness
        legs = cpt.load_model(rs.mp(pdg, True))
        xtot = float(sum(l["ms"][:, 2].sum() for l in legs))
        nperx = nion / xtot
        # E[sum L^2] for a Poisson process of n points on [0, D], the interval
        # ending at the boundary:  D^2 [ 2/n - 2(1-e^-n)/n^2 ]
        f = 2.0 / n - 2.0 * (1.0 - np.exp(-n)) / (n * n)
        out[hp.SPECIES[pdg]["label"]] = dict(n=n, f=f, nstep=ns, nperx=nperx)
        print(f"  {hp.SPECIES[pdg]['label']:<7}{len(c):>8}{ns:>15.2f}{nion:>13.2f}"
              f"{n:>9.3f}{ns - nion - 2 * TOY_NLAYER:>10.2f}{xtot:>9.4f}"
              f"{nperx:>9.4f}{f:>10.4f}{1.0 / f:>8.3f}")
    print()
    print("  `other/ev` = Nstep - deltas - 28 boundary crossings (14 layer exits +")
    print("  14 gap exits): 5 to 9 steps per event, i.e. the CoulombScat/msc/brems")
    print("  steps and a few field-stepper splits.  It is 3-6 % of the total and is")
    print("  NOT included in N_eff, which is therefore a conservative (LARGE) value.")
    return out


def cmd_g4(args):
    """Run the driver, validate the offline re-implementation against G4's own
    numbers, and compare the analytic CFs with what the sampler draws."""
    log = args.log or os.path.join(LOGD, "g4_layer.log")
    if args.run:
        cmd = [DRIVER, "--pathlen", str(TOY_PATH_CM), "--ndelta", str(args.ndelta),
               "--n", str(args.n), "--nfull", str(args.nfull), "--seed", "20260816"]
        print(" ".join(cmd))
        with open(log, "w") as fh:
            subprocess.run(cmd, stdout=fh, check=True)
    mat, ST, MOM, HIST = parse_driver(log)
    ndel = mat["ndelta"]
    # GEANT4 INTERNAL UNITS (mm) throughout -- see `state_of`.
    D = float(next(iter(ST.values()))["tpathG4"])
    lmean = float(next(iter(ST.values()))["lmeanG4"])

    print("=" * 128)
    print("V1.  THE OFFLINE TRANSPORT CROSS SECTION AGAINST G4's OWN lambda.")
    print("=" * 128)
    print("G4WentzelVIModel's transport mean free path is "
          "1/[n * ComputeTransportCrossSectionPerAtom(cosTetMaxNuc)].  The offline")
    print("re-implementation must reproduce the driver's `lambdaFull`, which the "
          "driver got from G4 itself.")
    print(f"  {'sp':<6}{'lambdaFull [mm]':>18}{'offline':>18}{'rel':>12}"
          f"{'Lg':>10}{'chia2=2screenZ':>18}")
    worst = 0.0
    LG = {}
    for lab in ST:
        st = state_of(ST, lab, mat)
        inv = st["natoms"] * float(sigma_tr(st, st["omcNuc"]))   # 1/mm
        lam_mm = 1.0 / inv
        rel = abs(lam_mm / st["lambdaFull"] - 1.0)
        worst = max(worst, rel)
        a, fb = st["screenZ"], st["screenZ"] * st["factB"]
        Lg = float(g4_f(st["omcElec"] / a, fb)
                   + st["Z"] * g4_f(st["omcNuc"] / a, fb)) / (st["Z"] + 1.0)
        LG[lab] = Lg
        print(f"  {lab:<6}{st['lambdaFull']:>18.10g}{lam_mm:>18.10g}{rel:>12.2e}"
              f"{Lg:>10.4f}{2 * st['screenZ']:>18.6e}")
    print(f"\n  worst |rel| over 8 species : {worst:.2e}")

    print()
    print("=" * 128)
    print("V2.  THE COUNTING NORMALIZATION AGAINST G4's OWN "
          "Compute{Nuclear,Electron}CrossSection.")
    print("=" * 128)
    print("dn/dz = C/(z+a)^2 with C = n L kinFactor Z (nucleus) / n L kinFactor "
          "(electrons).  Integrated over the")
    print("full range it must reproduce the driver's `nssFull` (= D * xtsec with "
          "cosThetaMin = 1) and `elecRatioFull`.")
    print(f"  {'sp':<6}{'nssFull(G4)':>16}{'offline':>16}{'rel':>11}"
          f"{'elecRatio(G4)':>15}{'offline':>12}{'rel':>11}")
    worst2 = 0.0
    for lab in ST:
        st = state_of(ST, lab, mat)
        a = st["screenZ"]
        Cn = st["natoms"] * D * st["kinFactor"] * st["Z"]
        Ce = st["natoms"] * D * st["kinFactor"]
        nn = Cn * (1.0 / a - 1.0 / (st["omcNuc"] + a))
        ne = Ce * (1.0 / a - 1.0 / (st["omcElec"] + a))
        r1 = abs((nn + ne) / st["nssFull"] - 1.0)
        er = ne / (nn + ne)
        r2 = abs(er / st["elecRatioFull"] - 1.0)
        worst2 = max(worst2, r1, r2)
        print(f"  {lab:<6}{st['nssFull']:>16.10g}{nn + ne:>16.10g}{r1:>11.2e}"
              f"{st['elecRatioFull']:>15.10g}{er:>12.8f}{r2:>11.2e}")
    print(f"\n  worst |rel| : {worst2:.2e}")

    print()
    print("=" * 128)
    print("V3.  VARIANCE PRESERVATION: the Gaussian against the law it replaces.")
    print("=" * 128)
    print("The exponential in z has <1-cos> = 2 z0 = t/lambdaeff, and lambdaeff is "
          "1/[n sigma_tr(cosThetaMin)] --")
    print("the EXACT restricted first transport moment of the very law being "
          "replaced.  So the Gaussian's projected")
    print("variance equals the sub-threshold compound Poisson's second cumulant "
          "IDENTICALLY.  Checked against the")
    print("offline quadrature of the same density (the only difference is the "
          "form factor / rejection, which the")
    print("transport cross section does not carry and which is 1 to 1e-5 below "
          "theta_min).")
    print(f"  {'sp':<6}{'theta_min [rad]':>17}{'theta_min/sig_step':>20}"
          f"{'sig2_gauss':>14}{'kappa2 quad':>14}{'rel':>11}")
    for lab in ST:
        st = state_of(ST, lab, mat)
        q = wvi_step(st, lmean)
        a = st["screenZ"]
        zz = np.geomspace(a * 1e-8, q["zmin"], 200000)
        Cn = st["natoms"] * lmean * st["kinFactor"] * st["Z"]
        Ce = st["natoms"] * lmean * st["kinFactor"]
        # kappa2 = (1/2) int theta^2 dn = int (1-cos) dn
        k2 = np.trapezoid(Cn * grej(st, zz) / (zz + a) ** 2 * zz * zz, np.log(zz))
        ze = np.geomspace(a * 1e-8, st["omcElec"], 200000)
        k2 += np.trapezoid(Ce * grej(st, ze, True) / (ze + a) ** 2 * ze * ze,
                           np.log(ze))
        thmin = np.sqrt(2 * q["zmin"])
        sigstep = np.sqrt(lmean * q["invlam_full"])   # projected sigma of the step
        print(f"  {lab:<6}{thmin:>17.6e}{thmin / sigstep:>20.4f}{q['sig2']:>14.6e}"
              f"{k2:>14.6e}{abs(k2 / q['sig2'] - 1):>11.2e}")

    print()
    print("=" * 128)
    print("V4.  THE ANALYTIC CFs AGAINST WHAT THE SAMPLER DRAWS, in sigma_MC.")
    print("=" * 128)
    print(f"One layer, path {D / 10:.5f} cm, {ndel} delta rays -> {ndel + 1:.2f} steps.  "
          f"`split` = WentzelVI's own split (n = {args.n:g});")
    print(f"`full` = the same physics with cosThetaMin = 1, i.e. the pure compound "
          f"Poisson the model has (n = {args.nfull:g}).")
    print()
    hdr = (f"  {'sp':<6}{'arm':<7}{'|ecf-CF_ana|':>14}{'sigma_MC':>10}"
           f"{'|ecf-CF_other|':>16}{'sigma_MC':>10}{'max dCF':>11}{'at s*sig':>10}")
    print(hdr)
    res = {}
    for lab in ST:
        st = state_of(ST, lab, mat)
        sig = np.sqrt(D * wvi_step(st, lmean)["invlam_full"])
        sgrid = np.linspace(0.05, 5.8, 24) / sig
        Sfull = s_full_path(st, sgrid, D)
        cf_full = np.exp(Sfull)
        # the split arm: mean-exponent over the layer's steps.  The step lengths
        # are random, so the true layer CF is E[exp(sum_k S(L_k))]; the
        # configuration average is done explicitly and both are reported.
        cf_split_mean, cf_split_cfg = split_layer_cf(st, sgrid, D, ndel,
                                                     nmc=args.nmc, seed=7)
        for arm, cfa, cfo in (("split", cf_split_cfg, cf_full),
                              ("full", cf_full, cf_split_cfg)):
            if (lab, arm) not in HIST:
                continue
            e, ee = ecf_from_hist(HIST[(lab, arm)], sgrid)
            d1 = np.abs(e - cfa) / np.where(ee > 0, ee, np.nan)
            d2 = np.abs(e - cfo) / np.where(ee > 0, ee, np.nan)
            k = int(np.nanargmax(np.abs(cf_split_cfg - cf_full)))
            print(f"  {lab:<6}{arm:<7}{np.nanmax(np.abs(e - cfa)):>14.3e}"
                  f"{np.nanmax(d1):>10.2f}{np.nanmax(np.abs(e - cfo)):>16.3e}"
                  f"{np.nanmax(d2):>10.2f}"
                  f"{(cf_split_cfg - cf_full)[k]:>11.3e}{sgrid[k] * sig:>10.2f}")
        res[lab] = dict(s=sgrid, sig=sig, cf_full=cf_full,
                        cf_split=cf_split_cfg, cf_split_mean=cf_split_mean,
                        Lg=LG[lab])
    print()
    print("  `CF_ana` is the arm's OWN analytic model; `CF_other` is the other "
          "arm's.  A row that agrees with its own")
    print("  model at O(1) sigma_MC and disagrees with the other at many sigma is "
          "the split, measured.")

    print()
    print("=" * 128)
    print("V5.  MEAN-EXPONENT vs CONFIGURATION-AVERAGED split CF.")
    print("=" * 128)
    print("The propagation knob implements exp(<sum_k S(L_k)>), i.e. it uses the "
          "MEAN step length structure; the true")
    print("layer CF is E[exp(sum_k S(L_k))] over step configurations.  The gap "
          "between the two bounds that choice.")
    print(f"  {'sp':<6}{'max |cfg - mean|':>18}{'as frac of |split-full|':>26}")
    for lab, r in res.items():
        dd = np.max(np.abs(r["cf_split"] - r["cf_split_mean"]))
        gap = np.max(np.abs(r["cf_split"] - r["cf_full"]))
        print(f"  {lab:<6}{dd:>18.3e}{dd / gap:>26.4f}")

    print()
    print("=" * 128)
    print("V6.  THE PROPAGATION KNOB against the validated offline model.")
    print("=" * 128)
    print("`cf_track_resolution.wvi_split_exponent` is the J0 SERIES form of the "
          "same object, written on the model's own")
    print("(chi_c^2, chi_a^2) and driven by L_g and the effective step thickness "
          "instead of by G4's kinFactor.  It must")
    print("reproduce [S_split - S_full] from the G4 quadrature above.")
    print(f"  {'sp':<6}{'max |dS| quad':>15}{'max |dS| series':>17}{'ratio':>9}"
          f"{'max abs diff':>15}{'as frac of dS':>15}")
    for lab, r in res.items():
        st = state_of(ST, lab, mat)
        s = r["s"]
        # the quadrature dS for the layer, mean-exponent (what the knob is)
        nsub = 1.0 / lmean_eff(ndel)
        dsq = nsub * s_split_step(st, s, D / nsub) - s_full_path(st, s, D)
        R = (st["natoms"] * D * st["kinFactor"] * (st["Z"] + 1.0)) / st["screenZ"]
        q = s * np.sqrt(2.0 * st["screenZ"])
        U = SSFACTOR * (R / nsub) * r["Lg"]
        fN = st["Z"] / (st["Z"] + 1.0)
        dss = R * fN * ctr.wvi_split_exponent(q, np.full_like(q, U))
        k = int(np.argmax(np.abs(dsq)))
        print(f"  {lab:<6}{dsq[k]:>15.5e}{dss[k]:>17.5e}{dss[k] / dsq[k]:>9.4f}"
              f"{np.max(np.abs(dss - dsq)):>15.3e}"
              f"{np.max(np.abs(dss - dsq)) / abs(dsq[k]):>15.4f}")
    np.savez(os.path.join(LOGD, "g4_cf.npz"),
             **{f"{lab}_{k}": v for lab, r in res.items() for k, v in r.items()})
    print(f"\n  Lg (G4's transport log per unit chi_c^2, = [f(x_e)+Z f(x_N)]/(Z+1)):")
    print("   " + "  ".join(f"{lab} {LG[lab]:.4f}" for lab in ST))
    return res


def lmean_eff(ndelta):
    """The step thickness that reproduces sum_s L_s^2 for a Poisson process of
    `ndelta` points on the layer: E[sum L^2]/D = D [2/n - 2(1-e^-n)/n^2].
    It is NOT the mean step -- the effect is quadratic in the step length, so
    the exponential spread matters (a factor 2 at large n)."""
    n = float(ndelta)
    return 2.0 / n - 2.0 * (1.0 - np.exp(-n)) / (n * n)


def split_layer_cf(st, s, D, ndelta, nmc=4000, seed=7, nz=20000):
    """The layer's split CF, two ways.

    mean : exp( sum_k S(L_k) ) with the step lengths at their MEAN structure --
           what a per-step model in `ms_step_exponent` can express
    cfg  : E over Poisson step configurations of exp( sum_k S(L_k) ) -- what the
           simulation actually realizes
    """
    s = np.asarray(s, float)
    rng = np.random.default_rng(seed)
    lmean = D / ndelta
    # a cache of S(L) on a log grid of L, so the MC does not re-quadrature.
    # S(L)/L is interpolated, not S(L): S is very nearly linear in L (the
    # single-scattering part exactly, the Gaussian part up to the log in
    # theta_min), so interpolating S itself in ln L carries a (dlnL)^2/8
    # curvature error -- 0.3 % on a 60-point grid, which is 5x the effect
    # being measured.
    Lg_ = np.geomspace(D * 1e-5, D, 400)
    Sg = np.array([s_split_step(st, s, L, nz=nz) / L for L in Lg_])   # (nL, ns)
    lnL = np.log(Lg_)

    def SofL(L):
        Lc = np.clip(L, Lg_[0], Lg_[-1])
        i = np.clip(np.searchsorted(lnL, np.log(Lc)) - 1, 0, len(lnL) - 2)
        f = (np.log(Lc) - lnL[i]) / (lnL[i + 1] - lnL[i])
        return (L[:, None] * ((1 - f)[:, None] * Sg[i] + f[:, None] * Sg[i + 1]))

    tot = np.zeros(len(s))
    acc = np.zeros(len(s))
    for _ in range(nmc):
        Ls, rem = [], D
        while rem > 0:
            g = rng.exponential(lmean)
            L = min(g, rem)
            Ls.append(L)
            rem -= L
        S = SofL(np.array(Ls)).sum(axis=0)
        acc += np.exp(S)
        tot += S
    return np.exp(tot / nmc), acc / nmc


# =========================================================================
# 3.  propagation
# =========================================================================

def _rows(pdg, func, arm="off", model_rad=True):
    """One closure cell, with the three-cache discipline of
    `radoff_species._rows` / `msterms._rows`."""
    assert os.environ.get("RES_NO_PHI_CACHE"), "phi cache is LIVE"
    kok = hp.kok_for(pdg)
    cpt.RAD_CHANNEL = bool(model_rad)
    ctr.IONI_KOKOULIN = 1.0 if kok else 0.0
    ctr.IONI_KOKOULIN_TCUT = 0.0
    fn._SCALE_CACHE.clear()
    cpt._PHI_CACHE.clear()
    old_load, old_store = fn._scale_cache_load, fn._scale_cache_store
    fn._scale_cache_load = lambda *a, **k: None
    fn._scale_cache_store = lambda *a, **k: None
    try:
        path = rs.mp(pdg, model_rad)
        legs = cpt.load_model(path)
        sim = hp.sim_of(pdg, arm)
        sc = rs._scale(legs, func, path, rs.chans(model_rad))
        rows, errs, ks, ns, err = gc.closure_rows(legs, sim, func, sc["sF"])
    finally:
        fn._scale_cache_load, fn._scale_cache_store = old_load, old_store
        cpt.RAD_CHANNEL = True
        ctr.IONI_KOKOULIN = 0.0
        ctr.IONI_KOKOULIN_TCUT = 0.0
        fn._SCALE_CACHE.clear()
        cpt._PHI_CACHE.clear()
    return dict(m=rows.mean(axis=0), err=err, nev=int(sim["valid"].shape[0]),
                sF=float(np.mean(sc["sF"])))


_KNOBS = {
    "MS_WVI_SPLIT": (ctr, float), "MS_WVI_NPERX": (ctr, float),
    "MS_WVI_LG": (ctr, float), "MS_ELEC_TMAX": (ctr, float),
    "MS_SNAP_YMAX": (ctr, float), "KMS_SCALE": (cpt, float),
    "MS_NSUB": (cpt, int), "_WVI_ARGMAX": (ctr, float),
}


def _set(spec):
    old = {}
    for k, v in spec.items():
        mod, cast = _KNOBS[k]
        old[k] = getattr(mod, k)
        setattr(mod, k, cast(v))
    return old


def _restore(old):
    for k, v in old.items():
        setattr(_KNOBS[k][0], k, v)


# MEASURED per-species inputs to the split knob.  NPERX from `wvisplit.py
# steps` (census ionization secondaries / the model's own total xg); L_g from
# `wvisplit.py g4` V1 (G4's own transport cross section).  Both are properties
# of the SIMULATION and of Geant4, not fitted.
SP_INPUT = {
    "mu-": (9.4914, 14.9988), "mu+": (9.4948, 14.9988),
    "pi-": (9.4965, 14.9990), "pi+": (9.4942, 14.9990),
    "K-": (9.6927, 14.9987), "K+": (9.6951, 14.9987),
    "pbar": (10.2697, 14.9979), "p": (10.2706, 14.9979),
}


def cmd_gauge(args):
    print("=" * 152)
    print("THE PROPAGATED EFFECT.  Each row is a controlled change of the MODEL's "
          "MS channel at fixed simulation.")
    print("=" * 152)
    print(f"  {'species':<8}{'knob':<30}" + UROW + "      rms   cos(resid)")
    res = {}
    for pdg in args.pdg:
        lab = hp.SPECIES[pdg]["label"]
        # the two MEASURED per-species inputs, set before anything is computed
        ctr.MS_WVI_NPERX, ctr.MS_WVI_LG = SP_INPUT[lab]
        base = _rows(pdg, args.func)
        print(f"  {lab:<8}{'NPERX=%.4f LG=%.4f' % SP_INPUT[lab]:<30}")
        print(f"  {lab:<8}{'BASE (residual)':<30}" + _fmt(base["m"])
              + f" {_rms(base['m']):9.5f}      1.000")
        print(f"  {'':<8}{'+-':<30}" + _fmt(base["err"]))
        res[(pdg, "BASE")] = base["m"]
        for cell in args.cells:
            spec = dict(kv.split("=", 1) for kv in cell.split(","))
            old = _set(spec)
            try:
                r = _rows(pdg, args.func)
            finally:
                _restore(old)
            d = r["m"] - base["m"]
            nb, nd = np.linalg.norm(base["m"]), np.linalg.norm(d)
            cos = float(d @ base["m"] / (nb * nd)) if nb * nd > 0 else np.nan
            res[(pdg, cell)] = d
            print(f"  {'':<8}{cell:<30}" + _fmt(d)
                  + f" {_rms(d):9.5f} {cos:10.3f}")
        print()
    if args.npz:
        np.savez(args.npz, u=UCURVE,
                 **{f"{hp.SPECIES[p]['label']}|{c}": v for (p, c), v in res.items()})
        print(f"wrote {args.npz}")
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    s = p.add_subparsers(dest="cmd", required=True)

    q = s.add_parser("steps")
    q.add_argument("--pdg", type=int, nargs="+",
                   default=[13, -13, -211, 211, -321, 321, -2212, 2212])
    q.set_defaults(f=cmd_steps)

    q = s.add_parser("g4")
    q.add_argument("--log", default=None)
    q.add_argument("--run", action="store_true")
    q.add_argument("--n", type=int, default=40000000)
    q.add_argument("--nfull", type=int, default=300000)
    q.add_argument("--ndelta", type=float, default=8.444)
    q.add_argument("--nmc", type=int, default=4000)
    q.set_defaults(f=cmd_g4)

    q = s.add_parser("gauge")
    q.add_argument("--pdg", type=int, nargs="+", default=[13, -211, -321, 2212])
    q.add_argument("--func", default="locx")
    q.add_argument("--cells", nargs="+", default=["MS_WVI_SPLIT=1"])
    q.add_argument("--npz", default=None)
    q.set_defaults(f=cmd_gauge)

    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
