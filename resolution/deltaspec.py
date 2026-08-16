#!/usr/bin/env python3
"""The EXACT delta-ray (knock-on electron) spectrum, against the Urban `a3`
channel's pure 1/T^2.

WHAT THIS IS FOR
----------------
NOTES_TAILHUNT located the `qop` clean-propagation non-closure in the hard end
of the ionization spectrum: the offline model's Urban `a3` channel -- its ENTIRE
representation of delta rays -- was measured to be short of Geant4's own
explicit delta production by ~1.4 in RATE and 1.19-1.31 in ENERGY, at two
momenta and on two materials.  A single `a3` scale removes 97 % of the pT = 40
non-closure but cannot fit both momenta (the required f runs 1.098 -> 1.261),
which is the signature of a SHAPE error, not a normalization error.

This module derives the correct spectrum, predicts the measured deficit from it
with NO free parameter, and provides the corrected characteristic function.

THE PHYSICS
-----------
For a spin-1/2 projectile of mass M, total energy E, velocity beta, scattering
off a free electron at rest, the differential cross section for a kinetic energy
transfer T is (PDG "Passage of particles through matter", eq. 34.7; identical to
G4MuBetheBlochModel::ComputeCrossSectionPerElectron / ::SampleSecondaries in the
Geant4 Physics Reference Manual, "Muon ionisation"):

    d sigma / dT = (2 pi r_e^2 m_e c^2 z^2 / beta^2) (1/T^2)
                   [ 1 - beta^2 T/Tmax + T^2/(2 E^2) ]

so that the number of collisions per unit path with transfer in [T, T+dT] is

    dN/dT = (xi / T^2) [ 1 - beta^2 T/Tmax + T^2/(2 E^2) ],
    xi    = 2 pi r_e^2 m_e c^2 n_e z^2 L / beta^2                      (MeV)

with n_e the electron density and L the step length.  `xi` is Geant4's
`twopi_mc2_rcl2 * electronDensity * length / beta2` -- the SAME quantity the
Urban model itself uses in its regime-0 variance -- and it is the whole
normalization of the spectrum.  There is nothing free in it.

Tmax for a muon on an ATOMIC ELECTRON is the two-body kinematic maximum

    Tmax = 2 m_e c^2 beta^2 gamma^2 / (1 + 2 gamma m_e/M + (m_e/M)^2)

(not 2 m_e c^2 beta^2 gamma^2, and not Tmax/2 -- that is the Moller case for
identical particles).  This is `G4MuBetheBlochModel::MaxSecondaryEnergy` and is
exactly what the extrapolator passes as `tmax` into the Urban block.

Above ~100 keV Geant4 additionally applies R. Kokoulin's radiative correction,
a multiplicative factor 1 + (alpha/2pi) a1 (a3 - a1) with
a1 = ln(1 + 2T/m_e), a3 = ln(4 E (E-T)/M^2), which reaches +6 % at the hard end.
It is implemented here (`kokoulin=True`) so the comparison to Geant4's OWN
secondaries is like for like, and is quantified separately from the tree-level
result.

WHAT THE URBAN a3 CHANNEL SAYS INSTEAD
--------------------------------------
    p(T) = C/T^2 on [e0, tmax],  C = 1/(1/e0 - 1/tmax),   a3 collisions
    a3 = rate * meanLoss (tmax - e0) / (e0 tmax ln(tmax/e0)),   rate = 0.56
=>  a3 C = rate * meanLoss / ln(tmax/e0)                                (MeV)

i.e. the a3 channel's normalization is `rate * meanLoss / ln(tmax/e0)`, an
Urban PARAMETERIZATION, while the physical normalization is `xi`.  Writing
meanLoss = xi * L_B with L_B the Bethe stopping number (the bracket of the
Bethe-Bloch formula, dimensionless), the ratio of the two is

    xi / (a3 C) = ln(tmax/e0) / (rate * L_B)                              (*)

-- a PARAMETER-FREE prediction of the rate deficit, and it is momentum
dependent through both ln(tmax/e0) and L_B.  The energy deficit is (*) times
the shape factor

    [ ln(Tmax/T) - beta^2 (1 - T/Tmax) + (Tmax^2-T^2)/(4E^2) ] / ln(Tmax/T)

which is < 1 -- the suppression near Tmax -- and that is why the measured
ENERGY deficit is smaller than the measured RATE deficit.

L_B is obtained from Geant4 itself (`urban_g4driver`, one call per material and
energy: meanLoss/xi at a known step length), NOT from a Bethe formula with a
hand-rolled density correction.  So the prediction uses the SAME dE/dx table
that produced the record.

SUBCOMMANDS
    xi       per-step xi vs a3 C, and the predicted rate / energy deficits
    census   the no-free-parameter test: predicted vs MEASURED explicit
             secondary rate and energy, at the simulation's own per-material
             production cuts, over the census's own path range
    shape    the shape test that does not need a threshold scan of the sim:
             the survival function of the HARDEST secondary in each event,
             P(max > T) = 1 - exp(-N(>T)), against the exact spectrum and
             against pure 1/T^2 at the same normalization
"""

import argparse
import glob
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cf_propagation_test as cpt                                    # noqa: E402
import cf_track_resolution as ctr                                    # noqa: E402
import fisher_norm as fn                                             # noqa: E402

SCRATCH = fn.SCRATCH
OUT = os.path.join(SCRATCH, "ds")
DRIVER = os.path.join(SCRATCH, "urban_g4driver.sh")

# CLHEP / PDG constants, in the units the record uses (MeV, cm)
MMU = 105.6583745
MEL = 0.51099895
ALPHA_PRIME = 1.0 / (2.0 * np.pi * 137.035999084)   # G4's alphaprime
KOKOULIN_TMIN = 0.1        # G4MuBetheBlochModel::limitKinEnergy = 100 keV
KOKOULIN_MUMIN = 1000.0    # G4MuBetheBlochModel::lowestKinEnergy = 1 GeV
URBAN_RATE = 0.56          # G4UniversalFluctuation*::rate


# ---------------------------------------------------------------- kinematics

def gamma_from_tmax(Tmax):
    """Invert Tmax = 2 me (g^2-1) / (1 + 2 g me/M + (me/M)^2) for gamma.

    The record carries tmax but not the step's energy, and the two are in
    one-to-one correspondence, so nothing has to be assumed about the muon."""
    r = MEL / MMU
    a = 2.0 * MEL
    b = -2.0 * np.asarray(Tmax) * r
    c = -(2.0 * MEL + np.asarray(Tmax) * (1.0 + r * r))
    return (-b + np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)


def tmax_of(gam):
    r = MEL / MMU
    return 2.0 * MEL * (gam * gam - 1.0) / (1.0 + 2.0 * gam * r + r * r)


# ------------------------------------------------------- the exact spectrum

def _kokoulin(T, Etot):
    """G4MuBetheBlochModel's radiative correction factor, applied by Geant4
    both in the cross section and in the sampling rejection."""
    T = np.asarray(T, dtype=np.float64)
    f = np.ones_like(T)
    m = T > KOKOULIN_TMIN
    if np.any(m):
        a1 = np.log(1.0 + 2.0 * T[m] / MEL)
        a3 = np.log(4.0 * Etot * (Etot - T[m]) / (MMU * MMU))
        f[m] = 1.0 + ALPHA_PRIME * a1 * (a3 - a1)
    return f


def exact_N_E(xi, T, Tmax, beta2, Etot, kokoulin=False, nquad=4000):
    """(N(>T), E(>T)) for dN/dT = xi/T^2 [1 - b2 T/Tmax + T^2/(2E^2)].

    Closed form for the tree-level spectrum; log-grid Gauss quadrature when the
    Kokoulin factor is included (it has no elementary primitive)."""
    T = float(T)
    if T >= Tmax:
        return 0.0, 0.0
    L = np.log(Tmax / T)
    if not kokoulin:
        N = xi * ((1.0 / T - 1.0 / Tmax) - beta2 * L / Tmax
                  + (Tmax - T) / (2.0 * Etot ** 2))
        E = xi * (L - beta2 * (Tmax - T) / Tmax
                  + (Tmax ** 2 - T ** 2) / (4.0 * Etot ** 2))
        return N, E
    x = np.linspace(np.log(T), np.log(Tmax), nquad)      # integrate in ln T
    t = np.exp(x)
    core = (1.0 / t ** 2) * (1.0 - beta2 * t / Tmax + t * t / (2.0 * Etot ** 2))
    if Etot - MMU > KOKOULIN_MUMIN:
        core = core * _kokoulin(t, Etot)
    N = xi * np.trapezoid(core * t, x)
    E = xi * np.trapezoid(core * t * t, x)
    return float(N), float(E)


def model_N_E(aC, T, tmax, e0):
    """(N(>T), E(>T)) for the Urban a3 channel: a3 collisions from C/E^2 on
    [e0, tmax], written through aC = a3*C."""
    if T >= tmax:
        return 0.0, 0.0
    T = max(T, e0)
    return aC * (1.0 / T - 1.0 / tmax), aC * np.log(tmax / T)


# ------------------------------------- L_B from Geant4's own dE/dx table

_LB_CACHE = {}


def bethe_stopping_number(Z, A, rho, ekin, tag="lb"):
    """L_B = meanLoss / xi for one material at one muon kinetic energy, taken
    from Geant4: `urban_g4driver` builds the material, calls the REAL
    G4UniversalFluctuationForExtrapolator::SampleFluctuations at a 1 mm step,
    and prints both the Urban record (from which meanLoss = scaling *
    (a1 e1 + a2 e2 + a3 C ln(tmax/e0)) is exact) and xi from the material's own
    electron density.  Dimensionless and step-length independent."""
    key = (round(Z, 6), round(A, 6), round(rho, 9), round(float(ekin), 3))
    if key in _LB_CACHE:
        return _LB_CACHE[key]
    os.makedirs(OUT, exist_ok=True)
    pre = os.path.join(OUT, f"lb_{tag}")
    subprocess.run([DRIVER, "--Z", f"{Z:.17g}", "--A", f"{A:.17g}",
                    "--rho", f"{rho:.17g}", "--ekin", f"{float(ekin):.17g}",
                    "--len", "1.0", "--n", "1", "--out", pre],
                   check=True, capture_output=True)
    d = {}
    for line in open(pre + ".rec"):
        p = line.split()
        if not p:
            continue
        if p[0] == "record":
            (d["a1"], d["e1"], d["a2"], d["e2"], d["a3"],
             d["e0"], d["tmax"], d["scal"]) = map(float, p[3:11])
        elif p[0] == "matxi":
            d["nel"], d["xi"], d["beta2"], d["gam"] = (
                float(p[2]), float(p[4]), float(p[6]), float(p[8]))
    C = 1.0 / (1.0 / d["e0"] - 1.0 / d["tmax"])
    mean = d["scal"] * (d["a1"] * d["e1"] + d["a2"] * d["e2"]
                        + d["a3"] * C * np.log(d["tmax"] / d["e0"]))
    res = dict(LB=mean / d["xi"], dedx=mean / 0.1, nel=d["nel"],
               beta2=d["beta2"], gam=d["gam"])
    _LB_CACHE[key] = res
    return res


# --------------------------------------------------- toy material catalogue
# The layered toy's path crosses exactly three materials; each is identified
# from the record ALONE by e2Fluct = 10 Zeff^2 eV, so nothing is assumed about
# which step is which.  (Zeff 7.374 is Geant4's Air; its share of the
# ionization mean is 1.1e-4, so its (A, rho) only has to be roughly right.)
TOY_MATS = {
    8.000: ("ToyLayerMat", 8.0, 16.0, 9.0),
    4.000: ("Beryllium", 4.0, 9.012182, 1.848),
    7.374: ("Air", 7.374, 14.79, 0.001214),
}


def material_of(Zeff):
    k = min(TOY_MATS, key=lambda z: abs(z - Zeff))
    assert abs(k - Zeff) < 0.05, f"unknown Zeff {Zeff}"
    return TOY_MATS[k]


def step_table(model_path, mats=None):
    """One row per exported ionization step, with everything the exact
    spectrum needs: xi, Tmax, beta^2, E, plus the model's own a3 C."""
    legs = cpt.load_model(model_path)
    rows = []
    for j, leg in enumerate(legs):
        for i, r in enumerate(leg["ioni"]):
            reg, gsig2, a1, e1, a2, e2, a3, e0, tmx, g, cs = r
            if reg == 0:
                rows.append(dict(leg=j, istep=i, regime=0, gsig2=gsig2))
                continue
            e1s, e2s, e0s, tms = e1 * g, e2 * g, e0 * g, tmx * g
            C = 1.0 / (1.0 / e0s - 1.0 / tms)
            ln = np.log(tms / e0s)
            mean = a1 * e1s + a2 * e2s + a3 * C * ln
            Zeff = np.sqrt(e2 / 1e-5)
            name, Z, A, rho = (mats or material_of)(Zeff) if callable(mats or material_of) \
                else material_of(Zeff)
            gam = gamma_from_tmax(tms)
            ek = (gam - 1.0) * MMU
            lb = bethe_stopping_number(Z, A, rho, ek, tag=name)
            rows.append(dict(leg=j, istep=i, regime=1, mat=name, Zeff=Zeff,
                             mean=mean, exc=a1 * e1s + a2 * e2s, aC=a3 * C,
                             a3=a3, e0=e0s, tmax=tms, gamma=gam, ekin=ek,
                             beta2=1.0 - 1.0 / gam ** 2, Etot=gam * MMU,
                             LB=lb["LB"], xi=mean / lb["LB"], cs=cs))
    return rows


def extra_crossings(rows, n):
    """The census integrates the primary's loss from the origin out to
    r < 107 cm, which crosses 14 toy layers; the model's legs stop at plane 13
    (r = 106.8 cm), i.e. 13 layers.  Rather than the 13/14 factor NOTES_TAILHUNT
    applied to the SIM side, add `n` synthetic copies of the LAST LEG's
    ToyLayerMat crossing to the MODEL side -- the layers are identical, and this
    keeps every material's own xi and cut attached to it.

    The last leg carries TWO ToyLayerMat steps at pT = 3 (the crossing, 1.843
    MeV, and a 1.4 keV remainder) and one at pT = 40, so the copy is the leg's
    SUM, not its last step: taking the last step instead loses a whole layer at
    pT = 3 and none at pT = 40, i.e. a 7 % error at one momentum only."""
    if n <= 0:
        return []
    tl = [r for r in rows if r.get("mat") == "ToyLayerMat"]
    assert tl, "no ToyLayerMat step to replicate"
    last = max(r["leg"] for r in tl)
    grp = [r for r in tl if r["leg"] == last]
    tot = dict(grp[-1])
    tot["mean"] = sum(r["mean"] for r in grp)
    tot["xi"] = sum(r["xi"] for r in grp)
    tot["aC"] = sum(r["aC"] for r in grp)
    tot["exc"] = sum(r["exc"] for r in grp)
    return [dict(tot) for _ in range(n)]


# ------------------------------------------------------------ the cuts table

def cuts_from_log(logpath):
    """The e- production cut per material, as PRINTED by
    PrimaryLossCensusWatcher from G4ProductionCutsTable -- a measurement of the
    simulation's own thresholds, not the XML's intent.

    A material appears once per production-cut REGION (133 couples in the toy).
    The muon's path is in the global region, i.e. the couple carrying
    DefaultCutValue, which is the FIRST occurrence -- so first-wins, and the
    per-config consistency of that choice is checked in `census` by the fact
    that the same rule reproduces three configurations spanning a factor 1900
    in threshold.
    """
    cuts = {}
    for line in open(logpath):
        if not line.startswith("[plcensus]"):
            continue
        p = line.split()
        if len(p) < 8 or p[1] == "idx":
            continue
        try:
            name = " ".join(p[2:-5])
            if name not in cuts:
                cuts[name] = float(p[-2])
        except ValueError:
            continue
    return cuts


# ================================================================ subcommands

def cmd_xi(args):
    print("=" * 100)
    print("THE EXACT SPECTRUM'S NORMALIZATION xi AGAINST THE URBAN a3 CHANNEL'S")
    print("=" * 100)
    print("xi = 2 pi re^2 me c^2 n_e L / beta^2 is the physical normalization; the "
          "Urban a3\nchannel instead carries a3 C = rate * meanLoss / ln(tmax/e0), "
          "rate = 0.56.\nTheir ratio is ln(tmax/e0) / (rate * L_B) with L_B = "
          "meanLoss/xi the Bethe stopping\nnumber, taken from Geant4's own dE/dx "
          "table.  NOTHING here is fitted.\n")
    for tag, mp in args.models:
        rows = [r for r in step_table(mp) if r["regime"] == 1]
        print(f"### {tag}   {mp}")
        print(f"  {'leg':>4}{'mat':>13}{'mean[MeV]':>11}{'L_B':>9}"
              f"{'xi[MeV]':>11}{'a3C[MeV]':>11}{'xi/a3C':>9}"
              f"{'ln(tmx/e0)':>12}{'Tmax[MeV]':>12}")
        for r in rows:
            print(f"  {r['leg']:>4}{r['mat']:>13}{r['mean']:11.5f}{r['LB']:9.3f}"
                  f"{r['xi']:11.6f}{r['aC']:11.6f}{r['xi']/r['aC']:9.4f}"
                  f"{np.log(r['tmax']/r['e0']):12.4f}{r['tmax']:12.3f}")
        sx = sum(r["xi"] for r in rows)
        sa = sum(r["aC"] for r in rows)
        sm = sum(r["mean"] for r in rows)
        print(f"  TOTAL  mean {sm:.5f} MeV   xi {sx:.6f}   a3C {sa:.6f}   "
              f"xi/a3C = {sx/sa:.4f}")
        # deficits at a common threshold, over the model's own path
        print(f"\n  {'T [MeV]':>10}{'N_model':>12}{'N_exact':>12}{'rate def':>10}"
              f"{'E_model':>12}{'E_exact':>12}{'ener def':>10}{'shape':>9}")
        for T in args.thresholds:
            Nm = Em = Nx = Ex = 0.0
            for r in rows:
                n, e = model_N_E(r["aC"], T, r["tmax"], r["e0"])
                Nm += n
                Em += e
                n, e = exact_N_E(r["xi"], max(T, r["e0"]), r["tmax"],
                                 r["beta2"], r["Etot"], kokoulin=args.kokoulin)
                Nx += n
                Ex += e
            print(f"  {T:>10.5g}{Nm:12.5f}{Nx:12.5f}{Nx/Nm:10.4f}"
                  f"{Em:12.5f}{Ex:12.5f}{Ex/Em:10.4f}{(Ex/Em)/(Nx/Nm):9.4f}")
        print()


def _census(tag):
    import tail_probe as tp
    fs = sorted(glob.glob(tp.census_glob(tag)))
    assert fs, f"no census for {tag}"
    return np.concatenate([np.fromfile(f, dtype=np.float32).reshape(-1, tp.NREC)
                           for f in fs]), fs


def cmd_census(args):
    import tail_probe as tp
    print("=" * 100)
    print("THE NO-FREE-PARAMETER TEST: predicted vs MEASURED explicit delta rays")
    print("=" * 100)
    print("The census counts every explicit muIoni secondary the primary made "
          "below r = rmax.\nGeant4 makes one whenever the transfer exceeds THAT "
          "MATERIAL's e- production cut, so\nthe prediction is a sum over steps "
          "of the exact spectrum above each step's own cut --\nnot a single "
          "threshold.  (The layered toy crosses three materials whose cuts "
          "differ\nby up to a factor 9.6 even at one DefaultCutValue; ignoring "
          "that is what turns the\ntrue rate deficit into an apparent one.)\n")
    for tag, mp, logp, nlayer_extra in args.configs:
        rows = [r for r in step_table(mp) if r["regime"] == 1]
        cuts = cuts_from_log(logp)
        c, fs = _census(tag)
        n = len(c)
        Ns = c[:, tp.I_NSECPROC + tp.PROCS.index("muIoni")]
        Es = c[:, tp.I_ESECPROC + tp.PROCS.index("muIoni")]
        # the census path is longer than the model's by `nlayer_extra` copies
        # of the last ToyLayerMat step -- stated, not hidden in a 13/14 factor
        allrows = rows + extra_crossings(rows, nlayer_extra)
        print(f"### {tag}   {n} events, {len(fs)} files")
        print(f"    cuts used:  " + "  ".join(
            f"{m}={cuts.get(m, float('nan')):.5g}"
            for m in sorted({r['mat'] for r in allrows})))
        for kok in ((False, True) if args.kokoulin else (False,)):
            Nx = Ex = 0.0
            for r in allrows:
                Tc = cuts[r["mat"]]
                nn, ee = exact_N_E(r["xi"], max(Tc, r["e0"]), r["tmax"],
                                   r["beta2"], r["Etot"], kokoulin=kok)
                Nx += nn
                Ex += ee
            lab = "tree+Kokoulin" if kok else "tree level   "
            dN = Ns.std() / np.sqrt(n)
            dE = Es.std() / np.sqrt(n)
            print(f"    {lab}   N_pred {Nx:10.4f}   N_sim {Ns.mean():10.4f}"
                  f" +- {dN:7.4f}   ratio {Ns.mean()/Nx:7.4f}")
            print(f"    {' '*len(lab)}   E_pred {Ex:10.4f}   E_sim {Es.mean():10.4f}"
                  f" +- {dE:7.4f}   ratio {Es.mean()/Ex:7.4f}")
        # what the model would have said at those same cuts
        Nm = Em = 0.0
        for r in allrows:
            nn, ee = model_N_E(r["aC"], max(cuts[r["mat"]], r["e0"]),
                               r["tmax"], r["e0"])
            Nm += nn
            Em += ee
        print(f"    Urban a3 model at the same cuts: N {Nm:10.4f}   E {Em:10.4f}"
              f"   ->  measured/model  N {Ns.mean()/Nm:6.3f}  E {Es.mean()/Em:6.3f}")
        print()


def cmd_shape(args):
    """The shape test.  The census stores the HARDEST single secondary of each
    event, so the survival function P(max > T) = 1 - exp(-N(>T)) measures
    N(>T) at ANY T above every material's cut -- with the existing samples and
    no threshold scan.  Only steps whose cut is below T contribute, so for T
    well above all cuts the material bookkeeping drops out entirely and what
    is left is the spectral SHAPE."""
    import tail_probe as tp
    print("=" * 100)
    print("SHAPE TEST: N(>T) from the hardest secondary per event")
    print("=" * 100)
    print("P(max secondary > T) = 1 - exp(-N(>T)) for independent collisions, so "
          "-ln(1-P) is\na direct measurement of N(>T) at every T above all the "
          "production cuts.  The two\nspectra are compared AT THE SAME xi (the "
          "exact one), so this isolates the shape:\npure 1/T^2 has no Tmax "
          "suppression and no spin term.\n")
    for tag, mp, logp, nlayer_extra in args.configs:
        rows = [r for r in step_table(mp) if r["regime"] == 1]
        allrows = rows + extra_crossings(rows, nlayer_extra)
        cuts = cuts_from_log(logp)
        cutmax = max(cuts[r["mat"]] for r in allrows)
        c, _ = _census(tag)
        n = len(c)
        mx = c[:, tp.I_MAXSEC].astype(np.float64)
        # The census's "hardest secondary" is the hardest of ANY process, and
        # muBrems / muPairProd put secondaries well above the ionization Tmax
        # (17 events above 664 MeV at pT = 3, where the delta spectrum predicts
        # 0.9).  iMaxSecCode carries the process that made it, so the delta rate
        # is recovered as the total rate times the muIoni share of the hardest
        # secondary -- exact to O(N^2) for independent Poisson processes, and N
        # is <= 0.1 over the whole range used here.
        code = c[:, tp.I_MAXSECCODE].astype(np.int32)
        isio = code == tp.PROCS.index("muIoni")
        Tmax = max(r["tmax"] for r in allrows)
        lo = max(cutmax * 1.5, args.tmin)
        Ts = np.exp(np.linspace(np.log(lo), np.log(0.95 * Tmax), args.npts))
        print(f"### {tag}   {n} events   (all cuts <= {cutmax:.4g} MeV, "
              f"Tmax = {Tmax:.4g} MeV)")
        print(f"  {'T [MeV]':>11}{'n(max>T)':>10}{'muIoni':>9}{'N_meas':>11}{'+-':>10}"
              f"{'N_exact':>11}{'meas/exact':>11}{'N_pure':>11}{'meas/pure':>11}")
        for T in Ts:
            hi = mx > T
            k = int(hi.sum())
            ki = int((hi & isio).sum())
            if ki < 8:
                continue
            p = k / n
            Nmeas = -np.log(1.0 - p) * (ki / k)
            dN = np.sqrt(ki) / n / (1.0 - p)
            Nx = Np = 0.0
            for r in allrows:
                if cuts[r["mat"]] > T or r["tmax"] <= T:
                    continue
                nn, _ = exact_N_E(r["xi"], T, r["tmax"], r["beta2"], r["Etot"],
                                  kokoulin=args.kokoulin)
                Nx += nn
                Np += r["xi"] * (1.0 / T - 1.0 / r["tmax"])
            print(f"  {T:>11.5g}{k:>10d}{ki:>9d}{Nmeas:11.6f}{dN:10.6f}{Nx:11.6f}"
                  f"{Nmeas/Nx:11.4f}{Np:11.6f}{Nmeas/Np:11.4f}")
        print()


# ==================================================== model export + bit-id

CMSSW = "/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev"
SRCTEST = f"{CMSSW}/src/Analysis/HitAnalyzer/test"
TOYAREA = os.path.join(CMSSW, "tailhunt")        # tail_probe.py's private areas
REALAREA = os.path.join(CMSSW, "tailhunt_real")


def _clean_env(extra):
    """cmsRun cannot inherit the calibration_studies venv / LCG view: it embeds
    its own interpreter, picks up PYTHONPATH / PYTHONHOME / LD_LIBRARY_PATH and
    dies MUTELY (toy_pt_scan._cmsrun documents the diagnosis).  Allowlist.

    EVERY ARM IS PINNED, IN BOTH DIRECTIONS.  The four corrections
    `CVH_IONI_EXACTDELTA`, `CVH_IONI_KOKOULIN`, `CVH_REF_CHARGEAWARE` and
    `CVH_REF_SPECIESDEDX` were DEFAULT-ON for one week (2026-08-16,
    NOTES_DEFAULTON) and are DEFAULT-OFF again (NOTES_CLOSURE_FINAL s1).
    Every published control arm in this directory -- `off`, `nominal`,
    `caoff`, `spdoff` -- is an EMPTY overlay, which under the default-ON state
    silently meant "all four on", i.e. the control became a second copy of the
    signal arm and every bit-identity table in NOTES_DELTASPEC / NOTES_QVALID /
    NOTES_CHARGEODD / NOTES_SPECIESDEDX would have read PASS for the wrong
    reason.

    The historical state is therefore applied as the BASE, and the arm's own
    overlay still wins.  THE PIN IS KEPT NOW THAT THE DEFAULTS ARE OFF AGAIN
    and it is deliberately not a no-op-by-luck: it makes a control arm
    independent of the ambient shell (and of any future re-flip), which is
    what a reproducibility harness owes its published numbers.  A closure
    study turns the corrections on through `ctr.SWITCHES_ON`, which is
    greppable and appears in the run log where a default does not.

    Two consequences worth being explicit about:

      * every existing arm dict keeps EXACTLY the meaning it had when its
        numbers were published, with no per-arm edit and therefore no arm
        that can be missed;
      * these drivers no longer track the C++ DEFAULT.  That is deliberate --
        a reproducibility harness should pin, not inherit -- and it means the
        default has to be demonstrated somewhere else, on an unpinned job.
        `python -c` on `ctr.env_flag`, and the cmsRun banner, are that
        somewhere else -- reachable from here through the `None` escape below.

    A value of `None` in `extra` means LEAVE THE VARIABLE UNSET, i.e. let the
    C++ default decide.  That is the one arm a pinning harness cannot express
    otherwise, and it is exactly the arm needed to demonstrate that a default
    is what it is claimed to be: `{"CVH_IONI_EXACTDELTA": None, ...}` runs an
    UNPINNED job.  It is per-variable, so "three pinned and one at its default"
    is expressible too.
    """
    keep = ("HOME", "USER", "LOGNAME", "SHELL", "TERM", "HOSTNAME", "TMPDIR",
            "X509_USER_PROXY", "KRB5CCNAME")
    e = {k: os.environ[k] for k in keep if k in os.environ}
    e["PATH"] = "/usr/local/bin:/usr/bin:/bin"
    e.update(ctr.SWITCHES_OFF)
    e.update(extra)
    return {k: v for k, v in e.items() if v is not None}


def run_model(which, out, log, env_extra=None):
    """Re-export the MODEL (the deterministic reference propagation and its
    per-step physics records).  `which` is 'pt3' / 'pt40' (the layered toy, in
    tail_probe.py's md5-verified private area) or 'real' (the CMS tracker).

    Every cmsRun goes through cmsswlock.sh run, per NOTES_PROFILE."""
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if which == "real":
        # Exactly real_radoff.cmd_model, whose output is the archived
        # `real_model_pt3_radon.root` this study's real-tracker closure uses.
        import real_radoff as rr
        td, script = rr.AREA, "runCleanPropModel.py"
        extra = (f"pt={rr.PT} eta={rr.ETA} phi={rr.PHI} partId=13 "
                 f"targets={rr.TARGETS} output={out}")
        search = ""
        env = {}
    else:
        area = os.path.join(TOYAREA, which)
        td = os.path.join(area, "Analysis", "HitAnalyzer", "test")
        pt = {"pt3": 3.0, "pt40": 40.0}[which]
        # runToyModel.py HARD-IMPORTS toyPlanes_pt3.  Run from the pt40 area it
        # still resolves -- from the release test/ directory, which cmsRun puts
        # on sys.path -- so it runs to rc = 0 with the WRONG planes and the
        # whole reference trajectory silently changes.  (Caught by the
        # branch-level bit-identity check: 22 of 27 branches moved with the
        # switch OFF.)  Patch it in the area, exactly as toy_pt_scan.cmd_setup
        # does.
        script = os.path.join(td, "runToyModel.py")
        src = open(f"{SRCTEST}/runToyModel.py").read()
        pat = "import toyPlanes_pt3 as planes"
        assert src.count(pat) == 1, "runToyModel.py planes import changed"
        src = src.replace(pat, "import importlib, os as _os\n"
                               'planes = importlib.import_module('
                               '_os.environ["TOY_PLANES_MOD"])')
        with open(script, "w") as fh:
            fh.write(src)
        extra = f"pt={pt} eta=0.30 phi=0.70 output={out}"
        search = f"export CMSSW_SEARCH_PATH={area}:$CMSSW_SEARCH_PATH && "
        env = {"TOY_PLANES_MOD": f"toyPlanes_pt{int(pt)}"}
    env.update(env_extra or {})
    cmd = ("source /cvmfs/cms.cern.ch/cmsset_default.sh >/dev/null 2>&1 && "
           f"cd {CMSSW}/src && eval $(scramv1 runtime -sh) && {search}"
           f"cd {td} && exec {SRCTEST}/cmsswlock.sh run cmsRun {script} {extra}")
    with open(log, "w") as fh:
        p = subprocess.run(["bash", "-c", cmd], stdout=fh,
                           stderr=subprocess.STDOUT, env=_clean_env(env))
    return p.returncode


def cmd_export(args):
    for which in args.which:
        combos = [("off", {}), ("on", {"CVH_IONI_EXACTDELTA": "1"})]
        if args.u2021:
            combos = [("u2021", {"CVH_IONI_URBAN2021": "1"}),
                      ("on_u2021", {"CVH_IONI_EXACTDELTA": "1",
                                    "CVH_IONI_URBAN2021": "1"})]
        for lab, ev in combos:
            if args.only and not lab.startswith(args.only):
                continue
            if args.t0 and lab == "on":
                ev = dict(ev, CVH_IONI_EXACTDELTA_T0=repr(args.t0))
                lab = f"on_t0{args.t0:g}"
            out = os.path.join(OUT, f"{which}_model_{lab}.root")
            log = os.path.join(OUT, f"{which}_model_{lab}.log")
            if os.path.exists(out) and not args.force:
                print(f"[{which}/{lab}] exists, skipping ({out})")
                continue
            rc = run_model(which, out, log, ev)
            print(f"[{which}/{lab}] rc={rc} -> {out}")
            if rc:
                raise SystemExit(f"export failed, see {log}")


def _branch_digests(path):
    """sha256 of every branch's raw basket bytes, over the WHOLE file.

    THE TRAP: the export writes into a `propExport/` TDirectory, so a
    comparison that enumerates only the top-level keys of the two files finds
    ONE key (the directory), hashes it, and reports a pass without ever having
    compared a branch.  `uproot`'s recursive iteration is used instead and the
    NUMBER of branches compared is printed, so a silent one-branch pass is
    visible."""
    import hashlib
    import uproot
    f = uproot.open(path)
    trees = [k for k, c in f.classnames(recursive=True).items() if c == "TTree"]
    if not trees:
        raise SystemExit(f"{path}: no TTree found (keys {list(f.keys())}) -- "
                         "a comparison that finds nothing is not a pass")
    out = {}
    for tn in trees:
        t = f[tn]
        for bn in t.keys():
            arr = t[bn].array(library="np")
            h = hashlib.sha256()
            for v in arr:
                a = np.ascontiguousarray(np.asarray(v))
                h.update(str(a.dtype).encode())
                h.update(a.tobytes())
            out[f"{tn}/{bn}"] = h.hexdigest()
    return out


def cmd_bitid(args):
    print("=" * 100)
    print("BIT-IDENTITY OF THE EXPORT WITH THE SWITCH OFF")
    print("=" * 100)
    print("Every branch of both files is hashed (sha256 over the raw values), "
          "recursively, so\nthe `propExport/` TDirectory cannot make a "
          "one-key comparison pass by accident.\n")
    for ref, new, lab in args.pairs:
        A, B = _branch_digests(ref), _branch_digests(new)
        ka, kb = set(A), set(B)
        same = sum(1 for k in ka & kb if A[k] == B[k])
        diff = sorted(k for k in ka & kb if A[k] != B[k])
        print(f"### {lab}")
        print(f"    {ref}\n    {new}")
        print(f"    branches: {len(ka)} vs {len(kb)}, common {len(ka & kb)}, "
              f"IDENTICAL {same}, different {len(diff)}")
        if ka ^ kb:
            print(f"    only on one side: {sorted(ka ^ kb)}")
        for k in diff[:40]:
            print(f"      DIFFERS  {k}")
        print("    ==> " + ("BIT-IDENTICAL" if not diff and not (ka ^ kb)
                            else "*** NOT IDENTICAL ***"))
        print()


# ================================ does the extrapolator sample stock Geant4?

def _drv(env, **kw):
    """One urban_g4driver run. Returns the parsed .rec dict and the sample
    arrays that were written."""
    pre = kw.pop("out")
    cmd = [DRIVER]
    for k, v in kw.items():
        cmd += ["--" + k, (f"{v:.17g}" if isinstance(v, float) else str(v))]
    cmd += ["--out", pre]
    if kw.get("stock_flag"):
        cmd.append("--stock")
    # historical switch state as the base, arm overlay wins -- see _clean_env
    subprocess.run(cmd, check=True, capture_output=True,
                   env=dict(os.environ, **ctr.SWITCHES_OFF, **env))
    d = {}
    for line in open(pre + ".rec"):
        p = line.split()
        if not p:
            continue
        if p[0] == "record":
            d["rec"] = [float(x) for x in p[1:11]]
        elif p[0] == "matxi":
            d["xi"] = float(p[4])
        elif p[0] == "sample2":
            d["mean2"], d["var2"] = float(p[2]), float(p[4])
        elif p[0] == "stock":
            d["stockmean"], d["stockvar"] = float(p[4]), float(p[6])
    return d


def _meanloss(rec):
    reg, gs2, a1, e1, a2, e2, a3, e0, tmx, sc = rec
    C = 1.0 / (1.0 / e0 - 1.0 / tmx)
    return sc * (a1 * e1 + a2 * e2 + a3 * C * np.log(tmx / e0))


def cmd_stockcmp(args):
    """THE test of the harmonization: at one fully specified step, does the
    extrapolator's own sampler draw the SAME distribution as stock Geant4
    11.2.2's G4UniversalFluctuation?  Empirical characteristic functions of N
    real C++ samples from each, compared in units of the MC error.  Nothing is
    fitted and nothing is inferred from a closure."""
    print("=" * 100)
    print("DOES THE EXTRAPOLATOR SAMPLE STOCK GEANT4 11.2.2?")
    print("=" * 100)
    print("`.bin` = G4UniversalFluctuationForExtrapolator::SampleFluctuations2, "
          "`.stock.bin` =\nstock G4UniversalFluctuation, both at the SAME "
          "material / particle / step / mean loss\nand tcut = tmax.  The CF is "
          "probed on the scale of the step's own CORE (the 16-84\npercentile "
          "half-width), not sqrt(kappa2), which the 1/E^2 tail makes useless.\n")
    os.makedirs(OUT, exist_ok=True)
    for ekin, length in args.steps:
        base = dict(Z=8.0, A=16.0, rho=9.0, ekin=float(ekin), len=float(length),
                    n=args.n, seed=args.seed)
        res = {}
        for lab, env in (("pre-2021 (as forked)", {}),
                         ("2021 (harmonized)", {"CVH_IONI_URBAN2021": "1"})):
            pre = os.path.join(OUT, "sc_" + lab.split()[0].replace("-", ""))
            d0 = _drv(env, out=pre, **base)
            ml = _meanloss(d0["rec"])
            d = _drv(env, out=pre, stock_flag=True, meanloss=float(ml), **base)
            x = np.fromfile(pre + ".bin", dtype=np.float64)
            y = np.fromfile(pre + ".stock.bin", dtype=np.float64)
            res[lab] = (d, ml, x, y)
        # common probe grid from the pre-2021 samples
        x0 = res["pre-2021 (as forked)"][2]
        core = 0.5 * (np.percentile(x0, 84) - np.percentile(x0, 16))
        t = np.linspace(0.0, 6.0 / core, 240)[1:]
        print(f"--- ekin {ekin} MeV, step {length} mm, core {core:.6g} MeV, "
              f"N = {args.n} each")
        for lab, (d, ml, x, y) in res.items():
            r = d["rec"]
            ecx = np.array([np.mean(np.exp(1j * tt * (x - x.mean()))) for tt in t])
            ecy = np.array([np.mean(np.exp(1j * tt * (y - y.mean()))) for tt in t])
            mc = np.sqrt(2.0 / len(x))     # |ecf| MC error, both samples
            dev = np.max(np.abs(ecx - ecy))
            print(f"    {lab:<22} a1 {r[2]:10.4g} e1 {r[3]:.5e} a2 {r[4]:9.4g} "
                  f"a3 {r[6]:10.5g}   meanLoss {ml:.6f} MeV")
            print(f"    {'':<22} mean  extrap {x.mean():.6f}  stock {y.mean():.6f}"
                  f"   var  extrap {x.var():.6e}  stock {y.var():.6e}")
            print(f"    {'':<22} max |ecf(extrap) - ecf(stock)| = {dev:.3e}"
                  f"  = {dev/mc:6.2f} sigma_MC")
        print()


# ======================================================= the hadron test
# The muon validation cannot exercise the parts of the construction that carry
# the particle: the mass in Tmax's recoil denominator, beta^2 (which multiplies
# the whole suppression term and is 0.9999 for a muon but 0.976 for a 3 GeV
# kaon), and the PDG-spin branch.  A hadron does.  Same layered toy, same
# single material, so the same parameter-free prediction applies.
#
# Hadronic inelastic scattering and decay are INACTIVATED, and verified off
# from ProcessActivationWatcher's own dump: at 9 g/cm3 over 14 layers the toy
# is 13.2 g/cm2, i.e. ~15 % of a nuclear interaction length, so leaving them on
# would truncate 15 % of the tracks and there would be no 0.1 %-level test to
# make.  Everything else is untouched.

HADAREA = os.path.join(CMSSW, "deltaspec_had")
# SIGNED PDG codes, so the gun and the model reference trajectory are the same
# particle -- a first pass used +321 for the gun (K+) and -321 for the model
# (K-) and, worse, told ProcessActivationWatcher to deactivate `kaon--inelastic`
# for a K+ primary. The watcher's own step census then showed
# `kaon+Inelastic primary 20149`, i.e. nothing had been switched off. The
# process names below are the ones the census PRINTS, not guesses.
HAD_PARTS = {-211: ("pi-", 139.57039), -321: ("kaon-", 493.677),
             2212: ("proton", 938.27209), 13: ("mu-", 105.6583745)}
HAD_INACT = {-211: ["pi-Inelastic", "hadElastic", "Decay", "CoulombScat"],
             -321: ["kaon-Inelastic", "hadElastic", "Decay", "CoulombScat"],
             2212: ["protonInelastic", "hadElastic", "CoulombScat"],
             13: []}

_HAD_BLOCK = '''    output=cms.string(opts.output),
))

# ---------------------------------------------------------- DELTASPEC HADRON
_pdg = int(os.environ.get("TOY_PDG", "13"))
_pname = os.environ.get("TOY_PNAME", "mu-")
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


def had_setup():
    """Private area: the pT = 3 toy geometry (md5-compared against the one the
    tail_probe campaign used) plus a runToyGeomCheck.py that takes the particle
    from the environment.  Nothing under tailhunt/ is written to."""
    import hashlib
    import re
    import shutil
    src = os.path.join(TOYAREA, "pt3")
    td = os.path.join(HADAREA, "Analysis", "HitAnalyzer", "test")
    dd = os.path.join(HADAREA, "Analysis", "HitAnalyzer", "data")
    os.makedirs(td, exist_ok=True)
    os.makedirs(dd, exist_ok=True)
    for rel in ("Analysis/HitAnalyzer/data/tracker.xml",
                "Analysis/HitAnalyzer/test/toyPlanes_pt3.py"):
        a, b = os.path.join(src, rel), os.path.join(HADAREA, rel)
        shutil.copy(a, b)
        h = hashlib.md5(open(b, "rb").read()).hexdigest()
        print(f"  {h}  {rel}   (copied from tailhunt/pt3)")
    s = open(f"{SRCTEST}/runToyGeomCheck.py").read()
    n = s.count("import FWCore.ParameterSet.Config as cms")
    assert n >= 1
    s = s.replace("import FWCore.ParameterSet.Config as cms",
                  "import os\nimport FWCore.ParameterSet.Config as cms", 1)
    pat = ("process.g4SimHits.Physics.DefaultCutValue = cms.double(1.0)")
    assert s.count(pat) == 1, "toy cut line changed"
    s = s.replace(pat,
                  'process.g4SimHits.Physics.DefaultCutValue = cms.double(\n'
                  '    float(os.environ.get("TOY_CUT", "1.0")))')
    pat2 = "    output=cms.string(opts.output),\n))"
    assert s.count(pat2) == 1, "toy TFileService block changed"
    s = s.replace(pat2, _HAD_BLOCK)
    open(f"{td}/runToyGeomCheck.py", "w").write(s)
    m = open(f"{SRCTEST}/runToyModel.py").read()
    m = m.replace("import toyPlanes_pt3 as planes",
                  "import importlib, os as _os\nplanes = importlib.import_module("
                  '_os.environ["TOY_PLANES_MOD"])')
    open(f"{td}/runToyModel.py", "w").write(m)
    print(f"  wrote {td}/runToyGeomCheck.py and runToyModel.py")
    return td


def _had_run(td, script, extra, log, env_extra):
    cmd = ("source /cvmfs/cms.cern.ch/cmsset_default.sh >/dev/null 2>&1 && "
           f"cd {CMSSW}/src && eval $(scramv1 runtime -sh) && "
           f"export CMSSW_SEARCH_PATH={HADAREA}:$CMSSW_SEARCH_PATH && "
           f"cd {td} && exec {SRCTEST}/cmsswlock.sh run cmsRun {script} {extra}")
    with open(log, "w") as fh:
        return subprocess.run(["bash", "-c", cmd], stdout=fh,
                              stderr=subprocess.STDOUT,
                              env=_clean_env(env_extra)).returncode


def cmd_hadron(args):
    import tail_probe as tp
    td = had_setup()
    pname, mass = HAD_PARTS[args.pdg]
    tag = f"had{args.pdg:+d}_pt{args.pt:g}"
    sim = os.path.join(OUT, f"{tag}_sim.root")
    cen = os.path.join(OUT, f"{tag}_census.bin")
    mod = os.path.join(OUT, f"{tag}_model_on.root")
    base = dict(TOY_PT=repr(args.pt), TOY_PLANES="toyPlanes_pt3.py",
                TOY_PLANES_MOD="toyPlanes_pt3", TOY_PDG=str(args.pdg),
                TOY_PNAME=pname, TOY_CENSUS=cen, TOY_RMAX="107.0",
                TOY_CUT=repr(args.cut),
                TOY_INACT=",".join(HAD_INACT[args.pdg]))
    if not os.path.exists(sim) or args.force:
        rc = _had_run(td, "runToyGeomCheck.py",
                      f"events={args.events} pt={args.pt} eta=0.30 output={sim}",
                      sim[:-5] + ".log", base)
        print(f"[{tag}] sim rc={rc} -> {sim}")
        if rc:
            raise SystemExit(f"sim failed, see {sim[:-5]}.log")
    if not os.path.exists(mod) or args.force:
        rc = _had_run(td, "runToyModel.py",
                      f"pt={args.pt} eta=0.30 phi=0.70 partId={args.pdg} "
                      f"output={mod}", mod[:-5] + ".log",
                      dict(base, CVH_IONI_EXACTDELTA="1"))
        print(f"[{tag}] model rc={rc} -> {mod}")
        if rc:
            raise SystemExit(f"model failed, see {mod[:-5]}.log")

    # --- the switches, verified from the logs rather than assumed
    slog = open(sim[:-5] + ".log", errors="ignore").read()
    print(f"\n### {tag}: switches, read back from the simulation log")
    for line in slog.splitlines():
        if line.startswith("[toy] PartID") or line.startswith("[toy] INACTIVATE") \
                or line.startswith("[procact]   SetProcessActivation") \
                or ("[procact]" in line and "steps" in line.lower()):
            print("   ", line.strip())
    cuts = cuts_from_log(sim[:-5] + ".log")
    print(f"    cuts: " + "  ".join(f"{m}={cuts[m]:.5g}"
                                    for m in ("ToyLayerMat", "Beryllium", "Air")
                                    if m in cuts))

    # --- xi and the spectrum straight from the regime-2/3 record
    legs = cpt.load_model(mod)
    rows = []
    for j, leg in enumerate(legs):
        for r in leg["ioni"]:
            reg, gsig2, a1, e1, a2, e2, a3, e0, tmx, g = r[:10]
            if reg not in (2., 3.):
                continue
            rows.append(dict(leg=j, regime=int(reg), xi=a3 * g, e0=e0 * g,
                             tmax=tmx * g, beta2=r[11], Etot=r[12],
                             mat=material_of(np.sqrt(e2 / 1e-5))[0]))
    assert rows, "model has no regime-2/3 steps -- was CVH_IONI_EXACTDELTA set?"
    r0 = [r for r in rows if r["mat"] == "ToyLayerMat"][0]
    lastleg = max(r["leg"] for r in rows if r["mat"] == "ToyLayerMat")
    extra = [dict(r) for r in rows
             if r["mat"] == "ToyLayerMat" and r["leg"] == lastleg]
    allr = rows + extra * args.extra
    print(f"\n    record: regime {sorted({r['regime'] for r in rows})} "
          f"(2 = spin 1/2, 3 = spin 0),  {pname} mass {mass} MeV")
    print(f"    ToyLayerMat step: xi {r0['xi']:.6f} MeV  Tmax {r0['tmax']:.4f} MeV"
          f"  beta^2 {r0['beta2']:.6f}  E {r0['Etot']:.2f} MeV"
          f"  Tmax/E {r0['tmax']/r0['Etot']:.4f}")
    gmu = float(gamma_from_tmax_mu(r0["tmax"]))
    print(f"    (assuming a MUON instead would give beta^2 "
          f"{1 - 1/gmu**2:.6f} and E {gmu*105.6583745:.2f} MeV -- which is why "
          f"beta2/etot are exported)")

    # --- the parameter-free prediction against the census
    c = np.fromfile(cen, dtype=np.float32).reshape(-1, tp.NREC)
    ns = c[:, tp.I_NSECPROC + tp.PROCS.index("muIoni")]
    es = c[:, tp.I_ESECPROC + tp.PROCS.index("muIoni")]
    oth = c[:, tp.I_NSECPROC + tp.PROCS.index("otherIoni")]
    nsec = ns + oth      # hIoni for hadrons, muIoni for muons
    esec = es + c[:, tp.I_ESECPROC + tp.PROCS.index("otherIoni")]
    rl = c[:, tp.I_RLAST]
    full = rl > 106.85
    print(f"\n    {len(c)} events, {100*full.mean():.2f} % reach the outermost "
          f"layer (r_last > 106.85 cm)")
    for sel, lab in ((np.ones(len(c), bool), "all events"),
                     (full, "reached r > 106.85")):
        n = int(sel.sum())
        for spin, half in ((True, "as exported"),
                           (False, "spin term FORCED off")):
            Nx = Ex = 0.0
            for r in allr:
                Tc = max(cuts[r["mat"]], r["e0"])
                if r["tmax"] <= Tc:
                    continue
                hh = (r["regime"] == 2) if spin else False
                nn, ee = exact_N_E_spin(r["xi"], Tc, r["tmax"], r["beta2"],
                                        r["Etot"], hh, kokoulin=False)
                Nx += nn
                Ex += ee
            if not spin and all(r["regime"] == 3 for r in rows):
                continue          # already spin-0; the forced-off row is a copy
            dN = nsec[sel].std() / np.sqrt(n)
            dE = esec[sel].std() / np.sqrt(n)
            print(f"    [{lab:<20}] {half:<20} N_pred {Nx:9.4f}  N_sim "
                  f"{nsec[sel].mean():9.4f} +- {dN:6.4f}  ratio "
                  f"{nsec[sel].mean()/Nx:7.4f}")
            print(f"    {'':<22} {'':<20} E_pred {Ex:9.4f}  E_sim "
                  f"{esec[sel].mean():9.4f} +- {dE:6.4f}  ratio "
                  f"{esec[sel].mean()/Ex:7.4f}")
    print()


def exact_N_E_spin(xi, T, Tmax, beta2, Etot, spinhalf, kokoulin=False):
    """(N(>T), E(>T)) with the spin term present only for spin 1/2, mirroring
    G4BetheBlochModel's `0.5 == spin` branch."""
    if T >= Tmax:
        return 0.0, 0.0
    L = np.log(Tmax / T)
    N = xi * ((1.0 / T - 1.0 / Tmax) - beta2 * L / Tmax)
    E = xi * (L - beta2 * (Tmax - T) / Tmax)
    if spinhalf:
        N += xi * (Tmax - T) / (2.0 * Etot ** 2)
        E += xi * (Tmax ** 2 - T ** 2) / (4.0 * Etot ** 2)
    return float(N), float(E)


def gamma_from_tmax_mu(tmax):
    """gamma from Tmax ASSUMING A MUON -- used only to show how wrong that
    assumption is for a hadron."""
    return gamma_from_tmax(tmax)


# ============================================================= the closure

def _closure_rows(modelpath, sim, func):
    """(rows, err, ks, n, err_mean) of the clean-propagation closure, Fisher
    normalization, for one model against one sim.  Same call as
    tail_probe._rows / real_probe._rows, only with the model given explicitly
    so the corrected and uncorrected models can be run against the SAME sim."""
    import geom_closure as gc
    legs = cpt.load_model(modelpath)
    sc = fn.plane_scales(legs, func, tag=modelpath,
                         channels=("ioni", "ms", "rad"))
    return gc.closure_rows(legs, sim, func, sc["sF"]), sc


def cmd_closure(args):
    import tail_probe as tp
    print("=" * 100)
    print("CLOSURE WITH THE CORRECTED SPECTRUM -- NOTHING FITTED")
    print("=" * 100)
    print("Only the MODEL changes between the OFF and ON rows; the simulation, "
          "the geometry\nand the reference trajectory are the same objects "
          "(bit-identity of the other 24\nbranches, `deltaspec bitid`).  "
          "Fisher normalization, mean over planes.\n")
    import real_probe as rp
    for tag in args.tags:
        if tag in tp.CONFIGS:
            sim, which = tp.sim_of(tag), tp.CONFIGS[tag]["area"]
        else:
            path = rp.ARCHIVED_SIM if tag == "ARCHIVED" else rp.sim_glob(tag)
            sim, which = rp.sim_of(path), "real"
        for func in args.funcs:
            print(f"### {tag}   {func}")
            print(f"  {'model':<22s} " + "".join(f"{u:>10.3g}" for u in fn.UCURVE)
                  + f"{'rms':>10}")
            base = None
            for lab in args.models:
                mp = os.path.join(OUT, f"{which}_model_{lab}.root")
                if not os.path.exists(mp):
                    print(f"  {lab:<22s} (missing {mp})")
                    continue
                (rows, errs, ks, ns, err), sc = _closure_rows(mp, sim, func)
                if args.kmax is not None:
                    sel = [i for i, k in enumerate(ks) if k <= args.kmax]
                    rows, err = rows[sel], err
                    ks = [ks[i] for i in sel]
                m = rows.mean(axis=0)
                if base is None:
                    base = m
                print(f"  {lab:<22s} " + "".join(f"{v:+10.5f}" for v in m)
                      + f"{np.sqrt(np.mean(m ** 2)):10.5f}"
                      + f"   [{len(ks)} pl, {int(np.median(ns))} ev]")
                print(f"  {'   +-':<22s} " + "".join(f"{v:10.5f}" for v in err))
                print(f"  {'   s_F (mean)':<22s} {np.mean(sc['sF']):.6e}"
                      f"   1/I (mean) {np.mean(sc['invI']):.4f}")
            print()


def cmd_perplane(args):
    """The per-plane profile at one probe -- the radial-growth null."""
    import tail_probe as tp
    import real_probe as rp
    print("=" * 100)
    print("PER-PLANE PROFILE (the radial-growth null)")
    print("=" * 100)
    for tag in args.tags:
        if tag in tp.CONFIGS:
            sim, which = tp.sim_of(tag), tp.CONFIGS[tag]["area"]
        else:
            sim, which = rp.sim_of(rp.sim_glob(tag)), "real"
        for func in args.funcs:
            print(f"### {tag}   {func}   probes {args.probes}")
            got = {}
            for lab in args.models:
                mp = os.path.join(OUT, f"{which}_model_{lab}.root")
                if not os.path.exists(mp):
                    continue
                (rows, errs, ks, ns, err), _ = _closure_rows(mp, sim, func)
                got[lab] = (rows, ks)
            if not got:
                continue
            ks = got[args.models[0]][1]
            legs = cpt.load_model(os.path.join(OUT, f"{which}_model_off.root"))
            iu = [list(fn.UCURVE).index(u) for u in args.probes]
            hdr = "".join(f"{lab[:8]:>11s}@u={u:g}" for u in args.probes
                          for lab in args.models)
            print(f"  {'k':>3}{'r [cm]':>9}" + hdr)
            for i, k in enumerate(ks):
                r = legs[k].get("refglobr", float("nan"))
                cells = "".join(f"{got[lab][0][i, j]:+18.5f}"
                                for j in iu for lab in args.models)
                print(f"  {k:>3}{r:9.2f}" + cells)
            print("  " + "-" * 60)
            for j, u in zip(iu, args.probes):
                for lab in args.models:
                    v = got[lab][0][:, j]
                    print(f"  span(last-first) u={u:g} {lab:<8s} "
                          f"{v[0]:+.5f} -> {v[-1]:+.5f}   span {v[-1]-v[0]:+.5f}")
            print()


# ==================================================================== driver

TP = os.path.join(SCRATCH, "tp")
TH = os.path.join(SCRATCH, "th")

MODELS = {"pt3": f"{TP}/pt3_K1_model.root", "pt40": f"{TP}/pt40_K1_model.root"}
# tag -> (model, a sim log carrying the cuts table, extra ToyLayerMat crossings
#         between the model's path (13 layers) and the census's (14, r < 107))
CFG = {
    "pt3_base":    ("pt3", f"{TH}/pt3_base_s101_sim.log", 1),
    "pt3_cut001":  ("pt3", f"{TH}/pt3_cut001_s101_sim.log", 1),
    "pt3_cut1e4":  ("pt3", f"{TH}/pt3_cut1e4_s101_sim.log", 1),
    "pt40_base":   ("pt40", f"{TH}/pt40_base_s101_sim.log", 1),
    "pt40_cut001": ("pt40", f"{TH}/pt40_cut001_s101_sim.log", 1),
    "pt40_cut1e4": ("pt40", f"{TH}/pt40_cut1e4_s101_sim.log", 1),
}


def _p(x):
    return x if os.path.sep in x else os.path.join(OUT, x)


def _resolve(tags):
    out = []
    for t in tags:
        m, lg, ex = CFG[t]
        out.append((t, MODELS[m], lg, ex))
    return out


def main():
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)

    q = s.add_parser("xi")
    q.add_argument("--models", nargs="+", default=["pt3", "pt40"])
    q.add_argument("--thresholds", type=float, nargs="+",
                   default=[0.00099, 0.0094862, 0.296, 1.0, 17.8507, 100.0])
    q.add_argument("--kokoulin", action="store_true")
    q.set_defaults(fn=cmd_xi)

    for name, f in (("census", cmd_census), ("shape", cmd_shape)):
        q = s.add_parser(name)
        q.add_argument("tags", nargs="+")
        q.add_argument("--kokoulin", action="store_true")
        if name == "shape":
            q.add_argument("--tmin", type=float, default=1.0)
            q.add_argument("--npts", type=int, default=14)
        q.set_defaults(fn=f)

    for name, f in (("closure", cmd_closure), ("perplane", cmd_perplane)):
        q = s.add_parser(name)
        q.add_argument("tags", nargs="+")
        q.add_argument("--models", nargs="+", default=["off", "on"])
        q.add_argument("--funcs", nargs="+", default=["qop"])
        q.add_argument("--kmax", type=int, default=None,
                       help="restrict the plane mean to k <= kmax")
        if name == "perplane":
            q.add_argument("--probes", type=float, nargs="+", default=[1.0])
        q.set_defaults(fn=f)

    q = s.add_parser("stockcmp")
    q.add_argument("--n", type=int, default=4000000)
    q.add_argument("--seed", type=int, default=12345)
    q.set_defaults(fn=cmd_stockcmp,
                   steps=[(3032.2, 1.0451), (3032.2, 0.01)])

    q = s.add_parser("hadron")
    q.add_argument("--pdg", type=int, default=-321,
                   choices=[-211, -321, 2212, 13])
    q.add_argument("--pt", type=float, default=3.0)
    q.add_argument("--events", type=int, default=20000)
    q.add_argument("--cut", type=float, default=1e-4)
    q.add_argument("--extra", type=int, default=1)
    q.add_argument("--force", action="store_true")
    q.set_defaults(fn=cmd_hadron)

    q = s.add_parser("export")
    q.add_argument("which", nargs="+", choices=["pt3", "pt40", "real"])
    q.add_argument("--only", choices=["off", "on", "u2021", "on_u2021"],
                   default=None)
    q.add_argument("--u2021", action="store_true")
    q.add_argument("--t0", type=float, default=0.0)
    q.add_argument("--force", action="store_true")
    q.set_defaults(fn=cmd_export)

    q = s.add_parser("bitid")
    q.add_argument("pairs", nargs="+",
                   help="ref.root:new.root[:label]")
    q.set_defaults(fn=cmd_bitid)

    a = p.parse_args()
    if a.cmd == "bitid":
        pp = []
        for s_ in a.pairs:
            t = s_.split(":")
            pp.append((_p(t[0]), _p(t[1]), t[2] if len(t) > 2 else "pair"))
        a.pairs = pp
        a.fn(a)
        return
    if a.cmd == "xi":
        a.models = [(t, MODELS[t]) for t in a.models]
    elif a.cmd in ("export", "closure", "perplane", "hadron", "stockcmp"):
        pass
    else:
        a.configs = _resolve(a.tags)
    a.fn(a)


if __name__ == "__main__":
    main()
