"""Radiative (bremsstrahlung + pair production) CF term for the muon transport model.

WHY THIS EXISTS
---------------
The mean and the fluctuation are inconsistent. The propagator's
mean-loss table is built with ionOnly=false, so the reference trajectory DOES
subtract the radiative mean; the fluctuation tables are built with
ionOnly=true, so the noise is ionization-only. The typical muon radiates
nothing but has the mean radiative loss removed anyway -- a mean-vs-mode bias
that grows with momentum (measured: 8.2e-6 at pT=10, 1.11e-5 at 40, 1.37e-5 at
100, after the T=0.413 fit transmission). At the 1e-5 Z-mass target that is not
negligible, and being pT-dependent it does not cancel in a scale calibration.

Excluding radiative loss from the VARIANCE was correct and must stay: for
dsigma/dnu ~ 1/nu the second moment is dominated by nu -> 1, so a radiative
variance describes the rare catastrophic radiator rather than the 99.9% that
radiate nothing. The fix is a CF term, not a variance -- which is what this is.

CONSTRUCTION
------------
Radiative loss is a compound Poisson like ionization, so the exponent adds:

    S_rad(t) = sum_steps INT deps (dN/deps)_step (e^{i a eps} - 1 - i a eps)

with a = cs * 1e-3 the step's dE[MeV] -> d(q/p) map (same convention as
UrbanIoniStep). The "-1 - i a eps" is the CENTRING: it removes the mean, which
is exactly right because the propagator already subtracted it. So "consistent
mean+fluctuation" and "fluctuation with the mean subtracted" are the same
object here -- centring forces S'(0) = 0 by construction.

SPECTRUM
--------
Nothing is reimplemented here. The propagator tabulates dN/dv per step for
bremsstrahlung and pair production SEPARATELY, from Geant4's own
G4MuBremsstrahlungModel / G4MuPairProductionModel differential cross sections
-- the same model objects that build the dE/dx table and that run in the full
simulation. Offline, each shape is renormalized to its own
ComputeDEDXPerVolume (see step_spectrum), which
  - absorbs the tabulation's quadrature and kinematic endpoint (measured:
    the exported shapes integrate to 1.02 of the brems and 0.993-0.998 of the
    pair dE/dx, constant to 0.4 %);
  - keeps the total mean exactly equal to what the propagator subtracted, so
    the centring leaves no residual bias;
  - gets the brems/pair MIXTURE right (pair is 58% of the radiative mean at
    100 GeV but is softer in v). A hand-built brems-only shape, normalized to
    the combined mean, under-predicts the simulated radiative rate by ~2.5x at
    5-15 GeV -- which is why the shapes come from G4.

usage:
    python cf_brems_exact.py --validate --model <model.root>
    python cf_brems_exact.py --validate-shape --model <model.root> --sim '<sim>/*.root'
"""

import argparse
import os

import numpy as np

import cf_knockon

# Leading columns of a radiative step record (Geant4ePropagator::RadiativeStep).
# The order must match G4ePropagationExport.cc's push_back sequence exactly.
(R_EFFZ, R_EFFA, R_XG, R_ETOT, R_P, R_DX0, R_STEPCM,
 R_DEDXRAD, R_DEDXBREM, R_DEDXPAIR, R_CS) = range(11)

# These eleven are all any reader here indexes -- but they are NOT the record
# width. `G4ePropagationExport`'s `radv` is exactly 11 wide; the maker's
# `radstepv` appends stepGroup and is 12. Both share columns 0..10, so the
# indices above serve both, and the stride must come from the file
# (`radstepstride`, or the record count) via `prodfiles.record_stride`.
# A module-level RADV_STRIDE used to stand here and was applied to BOTH
# branches: on `radstepv` that raises unless the step count happens to be a
# multiple of 11, and silently decodes garbage when it is. Do not restore it.
RADV_NCOLS = 11

NRADV = 48   # Geant4ePropagator::kNRadV; `radstepnv` in maker files

# The exported spectrum (48 points) is refined RAD_NSUB-fold before its
# quadrature (`refine_spectra`): log-log between points, the last interval
# continued to the step's kinematic endpoint.  Numerics, identical for every
# species and every caller; 4 is converged to 1e-4 in the CF.
RAD_NSUB = 4
PHYSICS_GLOBALS = ("RAD_NSUB",)


def physics_state():
    g = globals()
    return tuple((n, repr(g[n])) for n in PHYSICS_GLOBALS)

M_MU = 0.1056583745      # GeV
M_E = 0.510998946e-3     # GeV
SQRT_E = np.sqrt(np.e)


def refine_spectra(recs, spec, vg, nsub):
    """The exported spectra on a grid `nsub` times finer, for the quadrature.

    The export tabulates dN/dv at 48 log-spaced points on [1e-6, 1] and holds
    0 at points beyond the step's kinematic endpoint v_max = (E - M)/E, where
    the models are not evaluated.  Under rad_exponent's trapezoid that ramps
    the density to zero across the whole last interval -- [0.75, 1] on the
    export grid, i.e. the hardest emissions -- and a 1.34 point spacing
    miscounts a 1/v spectrum by 1.5 %.  Both are invisible when the
    radiative channel is a 1e-3 tail (muons) and several percent of the CF
    when it is the energy loss (e+-).

    Per step, and identically for every species:
      - log-log interpolation between exported points;
      - above the last in-range point, the last exported interval's power
        law continued to v_max;
      - the fine grid point that straddles v_max takes the value that makes
        the trapezoid's area over its interval equal the power law's over
        [v_i, v_max]; zero beyond.
    Returns (vf, spec_f) with vf shared by all steps."""
    vg = np.asarray(vg, dtype=np.float64)
    nb = len(vg)
    lv = np.log(vg)
    vf = np.exp(np.interp(np.linspace(0.0, nb - 1.0, (nb - 1) * nsub + 1),
                          np.arange(nb), lv))
    # the exported nodes exactly: exp(log(v)) is not always v, and the
    # spectrum's support below is decided by comparing against them
    vf[::nsub] = vg
    lvf = np.log(vf)
    out = np.zeros((len(recs), 2 * len(vf)))
    for s, (rec, sp) in enumerate(zip(recs, spec)):
        E, p = rec[R_ETOT], rec[R_P]
        vmax = (E - species_mass(E, p)) / E if E > 0.0 else 0.0
        for h in range(2):
            y = np.asarray(sp[h * nb:(h + 1) * nb], dtype=np.float64)
            pos = np.flatnonzero(y > 0.0)
            if len(pos) < 2:
                continue
            i0, iL = pos[0], pos[-1]
            f = np.zeros(len(vf))
            inner = (vf >= vg[i0]) & (vf <= vg[iL])
            f[inner] = np.exp(np.interp(lvf[inner], lv[pos], np.log(y[pos])))
            if vmax > vg[iL]:
                sl = (np.log(y[iL] / y[pos[-2]]) / (lv[iL] - lv[pos[-2]]))
                ext = (vf > vg[iL]) & (vf <= vmax)
                f[ext] = y[iL] * (vf[ext] / vg[iL]) ** sl
                last = np.flatnonzero(vf <= vmax)[-1]
                if last + 1 < len(vf):
                    a_, b_ = vf[last], vf[last + 1]
                    fa = f[last]
                    # INT_a^vmax fa (v/a)^sl dv
                    if abs(sl + 1.0) > 1e-9:
                        A = fa * a_ / (sl + 1.0) * ((vmax / a_) ** (sl + 1.0) - 1.0)
                    else:
                        A = fa * a_ * np.log(vmax / a_)
                    f[last + 1] = max(2.0 * A / (b_ - a_) - fa, 0.0)
            out[s, h * len(vf):(h + 1) * len(vf)] = f
    return vf, out


def step_spectrum(rec, spec, vg):
    """(v grid, dN/dv) for one step, from Geant4's own tabulated shapes.

    Each process is normalized SEPARATELY to its own ComputeDEDXPerVolume:

      dN/dv = sum_proc  shape_proc(v) * dE_proc / INT v E shape_proc dv

    Why per process and not on the sum: pair production is 58% of the
    radiative mean here but is SOFTER in v than brems, so normalizing a single
    combined shape gets the mixture wrong -- that is precisely what a
    hand-built brems-only shape gets wrong by ~2.5x at 5-15 GeV.

    Why normalize at all, given the shapes come from Geant4: the tabulated
    shapes integrate to 1.02 (brems) and 0.993-0.998 (pair) of each model's
    own ComputeDEDXPerVolume -- the 189-point grid's quadrature and the
    kinematic endpoint.  Renormalizing absorbs that, so the mean is right by
    construction and stays centred on what the propagator subtracted.  (The
    pair shape is only right if the pair model's per-element screening is set
    before each evaluation -- Geant4ePropagator's PairProbe; left unset it is
    the unscreened cross section, 700-1800x too large and several times too
    soft, which this renormalization cannot repair.)
    """
    v, brem, pair = step_spectrum_parts(rec, spec, vg)
    return v, brem + pair


def step_spectrum_parts(rec, spec, vg):
    """(v grid, dN/dv of bremsstrahlung, dN/dv of pair production), each
    normalized to its own process mean as step_spectrum describes."""
    E = rec[R_ETOT]
    L = rec[R_STEPCM]
    out = [np.zeros_like(vg), np.zeros_like(vg)]
    # the row holds the two shapes back to back on the SHARED grid, so the
    # split point is the grid the file itself shipped -- not a literal
    nb = len(vg)
    for i, (shape, dedx) in enumerate(((spec[:nb], rec[R_DEDXBREM]),
                                       (spec[nb:], rec[R_DEDXPAIR]))):
        dE = dedx * L
        if dE <= 0.:
            continue
        norm = np.trapezoid(shape * vg * E, vg)
        if norm > 0.:
            out[i] = shape * (dE / norm)
    return vg, out[0], out[1]


# The photon's polar angle, as the simulation's angular generators draw it:
#   e+-  G4ModifiedTsai (G4SeltzerBergerModel, G4eBremsstrahlungRelModel):
#        theta = u M/E, u = -ln(r1 r2) a, a = 1.6 w.p. 0.25 else 1.6/3,
#        i.e. u ~ Gamma(2, a); E[J0(c u)] = (1 + a^2 c^2)^(-3/2)
#   mu   G4ModifiedMephi (G4MuBremsstrahlungModel): gamma theta = w with
#        density 2w/(1+w^2)^2; E[J0(c w)] = c K1(c).  Its truncation at
#        w_max = gamma (pi/2) min(1, E/k - 1) matters only above v = 1/2
#        and is not carried.
# Measured for e- at 3 GeV on silicon with Geant4's own SampleSecondaries
# (ebrem_g4driver --sample): u mean 1.60 at every v, and the primary's recoil
# theta_e = k sin(theta_gamma)/p' to 1e-5 (momentum conservation).  Pair
# production leaves the muon's direction as it is and carries no recoil.
_ME_GEV = 0.51099895e-3
_MMU_GEV = 0.1056583745


def rad_angle_cfm1(c, mass):
    """E[J0(c w)] - 1 for the photon-angle law of the species of mass
    `mass` [GeV], c = (projected weight) x theta / w: G4ModifiedTsai for e+-,
    G4ModifiedMephi for every heavier species (muBrems and hBrems share it)."""
    c = np.asarray(c, dtype=np.float64)
    if is_epm(mass):
        a1, a2 = 1.6, 1.6 / 3.0
        return (0.25 * np.expm1(-1.5 * np.log1p((a1 * c) ** 2))
                + 0.75 * np.expm1(-1.5 * np.log1p((a2 * c) ** 2)))
    from scipy.special import k1
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(c > 0.0, c * k1(np.maximum(c, 1e-300)), 1.0) - 1.0
    return g


def is_epm(mass):
    """e+- or not, from a mass rebuilt from the records' float32 E and p:
    sqrt(E^2 - p^2) is only good to a few MeV there (to ~1 MeV for an e+- at
    3 GeV, ~9 MeV for a muon above 11 GeV), and the lightest heavy species is
    at 106 MeV."""
    return mass < 0.01


# the charged species a track can be: e, mu, pi, K, p [GeV]
_SPECIES_MASSES = np.array([_ME_GEV, _MMU_GEV, 0.13957039, 0.493677, 0.93827208816])


def species_mass(E, p):
    """The projectile's mass [GeV], snapped to the nearest charged species:
    the rebuilt sqrt(E^2 - p^2) is good to a few MeV (see is_epm) and the
    closest pair, mu and pi, is 34 MeV apart."""
    M = float(np.sqrt(max(E * E - p * p, 0.0)))
    return float(_SPECIES_MASSES[np.argmin(np.abs(_SPECIES_MASSES - M))])


def rad_exponent(tau, recs, spec, vg, weights=None, exact_qop=False,
                 bweights=None):
    """S_rad(t) = sum_steps INT dv (dN/dv) (e^{i a v E} - 1 - i a v E).

    tau      : (nt,) real CF argument grid
    recs     : (nstep, 11) radv records
    spec     : (nstep, 2*NRADV) tabulated shapes
    vg       : (NRADV,) shared v grid
    weights  : (nstep,) transport weight per step (default 1)
    exact_qop: map the loss into q/p EXACTLY, e^{i a T_eff} with
               T_eff = p^2 T (2E - T)/(E p' (p + p')), p' = sqrt((E-T)^2 - M^2),
               keeping the linear centring -i a vE (cf_knockon, QOP_EXACT: the
               reference subtracts the mean ENERGY loss, so the channel's mean
               becomes the Jensen excess).  False is the linear map, bit for bit.
    bweights : (nstep,) angular transport weight per radian (the scattering
               channel's), or None.  With it every bremsstrahlung emission is
               the JOINT event of its q/p change and the primary's recoil
               theta = k sin(theta_gamma)/p' against the photon (theta_gamma
               from the species' angular generator, rad_angle_cfm1), an
               isotropic 2D kick: the exponent gains
               sum_v dN_brem e^{i a X} (E[J0(b theta)] - 1).
    Returns the complex exponent; the CF factor is exp(S_rad).
    """
    tau = np.asarray(tau, dtype=float)
    S = np.zeros(len(tau), dtype=np.complex128)
    if weights is None:
        weights = np.ones(len(recs))
    # Trapezoid weights of the SHARED v grid, formed once. Folding them (and
    # dNdv) into a single vector turns the per-step
    # `np.trapezoid(term*dNdv, v, axis=1)` into one matrix-vector product,
    # which is the same sum of the same products -- but it stops numpy
    # materializing an (nt, nv) complex temporary per step: 6.0 ms -> 0.03 ms
    # for nt = 8001, nv = 48.
    wtrap = np.zeros(len(vg))
    dv = np.diff(np.asarray(vg, dtype=float))
    wtrap[:-1] += 0.5 * dv
    wtrap[1:] += 0.5 * dv
    for s_, (rec, sp, w) in enumerate(zip(recs, spec, weights)):
        v, dNb, dNp = step_spectrum_parts(rec, sp, vg)
        dNdv = dNb + dNp
        if not np.any(dNdv > 0.):
            continue
        E = rec[R_ETOT]
        bw = 0.0 if bweights is None else float(bweights[s_])
        if bw != 0.0 and np.any(dNb > 0.):
            # the recoil: theta = (k/p') theta_gamma, theta_gamma = w M/E
            p = rec[R_P]
            M = species_mass(E, p)
            T = v * E
            pp = np.sqrt(np.maximum((E - T) ** 2 - M * M, (1e-3 * p) ** 2))
            gm1 = rad_angle_cfm1(np.outer(tau * bw, T / pp * M / E), M)
            a_ = tau * rec[R_CS] * w
            ph = np.outer(a_, cf_knockon.qop_map(T, E, p, pp) if exact_qop else T)
            qb = wtrap * dNb
            S += ((np.cos(ph) * gm1) @ qb) + 1j * ((np.sin(ph) * gm1) @ qb)
        # a: dE -> the standardized variable, via the step's cs and the
        # caller's transport weight. cs = E/p^3 [GeV^-2] and v*E below is in
        # GeV, so d(qop) = cs * dE[GeV] directly -- NO MeV conversion. (The
        # 1e-3 in the UrbanIoniStep convention exists only because those
        # records store energies in MeV; putting a 1e3 here instead would make
        # the exponent 1e6 too large and collapse the CF to zero.)
        a = tau * rec[R_CS] * w
        x = np.outer(a, v * E)                       # (nt, nv)
        if exact_qop:
            T = v * E
            p = rec[R_P]
            # p' floored at 1e-3 p: a loss that leaves the primary below that
            # never reaches a plane, and the floor keeps T_eff finite
            M = species_mass(E, p)
            pp = np.sqrt(np.maximum((E - T) ** 2 - M * M, (1e-3 * p) ** 2))
            xe = np.outer(a, cf_knockon.qop_map(T, E, p, pp))
            sm = np.abs(xe) < 1e-4
            re = np.empty_like(xe)
            im = np.empty_like(xe)
            re[sm] = -0.5 * xe[sm] ** 2
            im[sm] = (xe[sm] - x[sm]) - xe[sm] ** 3 / 6.
            bg = ~sm
            sh = np.sin(0.5 * xe[bg])
            re[bg] = -2.0 * sh * sh
            im[bg] = np.sin(xe[bg]) - x[bg]
            q = wtrap * dNdv
            S += (re @ q) + 1j * (im @ q)
            continue
        # e^{ix} - 1 - ix, evaluated stably. The series is the accurate branch
        # only when |x| is small ACROSS THE WHOLE SUPPORT -- as in the
        # delta-ray term, where guarding on |a| instead of |a|*eps_max is
        # wrong by four orders of magnitude. Here x is formed explicitly, so
        # the guard is on x itself and cannot be misapplied.
        #
        # x is REAL, so e^{ix} - 1 = (cos x - 1) + i sin x needs no complex
        # transcendental: two real sines replace one complex expm1 (2.6x, and
        # the cos x - 1 branch is written as -2 sin^2(x/2), which is the same
        # cancellation-free form numpy's complex expm1 uses internally -- the
        # two agree to 4e-16 relative on real leg data, i.e. to rounding).
        small = np.abs(x) < 1e-4
        big = ~small
        re = np.empty_like(x)
        im = np.empty_like(x)
        xb = x[big]
        sh = np.sin(0.5 * xb)
        re[big] = -2.0 * sh * sh                     # cos(xb) - 1, stable
        im[big] = np.sin(xb) - xb
        xs = x[small]
        re[small] = -0.5 * xs ** 2
        im[small] = xs ** 3 / 6.
        q = wtrap * dNdv
        S += (re @ q) + 1j * (im @ q)
    return S


def load_radv(path):
    """(legs, vgrid): per-leg (recs, spec) from a model file."""
    import uproot
    t = uproot.open(path)
    key = next(k for k in t.keys() if k.split(";")[0].endswith("/legs"))
    import prodfiles
    a = t[key].arrays(["radv", "radspecv", "radvgrid", "stepnms"], library="np")
    vg = np.asarray(a["radvgrid"][0], dtype=float)
    legs = []
    # `stepnms` is CUMULATIVE and counts this log too: the propagator pushes
    # radStepLog_ in lockstep with msStepLog_.
    for r, sp, nms in zip(a["radv"], a["radspecv"], a["stepnms"]):
        n = prodfiles.cumulative_total(nms)
        recs = prodfiles.reshape_records(r, nrec=n, branch="radv",
                                         min_cols=RADV_NCOLS)
        spec = prodfiles.reshape_records(sp, nrec=n, branch="radspecv")
        assert len(recs) == len(spec)
        legs.append((recs, spec))
    return legs, vg


def validate_mean(legs, vg):
    """Gate 1: the modelled spectrum must reproduce the exported dE_rad."""
    print(f"{'leg':>4} {'steps':>6} {'dE_rad exported':>17} {'from spectrum':>15} {'ratio':>8}")
    tot_e = tot_m = 0.
    for i, (recs, spec) in enumerate(legs):
        de_exp = float((recs[:, R_DEDXRAD] * recs[:, R_STEPCM]).sum()) * 1e3
        de_mod = 0.
        for rec, sp in zip(recs, spec):
            v, dNdv = step_spectrum(rec, sp, vg)
            de_mod += np.trapezoid(dNdv * v * rec[R_ETOT], v) * 1e3
        tot_e += de_exp
        tot_m += de_mod
        if i < 6 or i == len(legs) - 1:
            r = de_mod / de_exp if de_exp else np.nan
            print(f"{i:4d} {len(recs):6d} {de_exp:17.5f} {de_mod:15.5f} {r:8.5f}")
    print(f"\nTOTAL exported {tot_e:.4f} MeV   from spectrum {tot_m:.4f} MeV   "
          f"ratio {tot_m/tot_e:.6f}")
    print("(ratio must be 1 by construction -- this checks the normalization "
          "integral, the grid and the kinematic limit, not the physics)")


def validate_shape(legs, vg, simglob):
    """Gate 2: the modelled spectrum vs the SIMULATED loss spectrum.

    The normalization is anchored to Geant4's dE/dx, so what is tested here is
    the SHAPE -- specifically the 1/eps plateau above ~1 GeV where the
    ionization delta-ray tail has died away.
    """
    import glob
    import uproot
    loss = []
    for f in sorted(glob.glob(simglob)):
        a = uproot.open(f)["simstates/simstates"].arrays(["pabs"], library="np")["pabs"]
        for v in a:
            v = np.asarray(v, dtype=float)
            if len(v) > 1:
                loss.append((v[0] - v[-1]) * 1e3)
    loss = np.array(loss)
    nev = len(loss)

    # Model: expected number of radiative transfers per track in each eps bin.
    # The per-step grid is only NRADV points over 6 decades, so a bin can
    # contain 0 or 1 of them -- sampling it directly gives spurious zeros.
    # Interpolate each step onto a common fine eps grid in log-log and
    # integrate there.
    fine = np.geomspace(100., 40000., 2000)          # MeV
    dNde = np.zeros_like(fine)
    for recs, spec in legs:
        for rec, sp in zip(recs, spec):
            v, dNdv = step_spectrum(rec, sp, vg)
            if not np.any(dNdv > 0.):
                continue
            E = rec[R_ETOT] * 1e3                    # MeV
            eps = v * E
            y = dNdv / E                             # dN/deps
            ok = y > 0.
            if ok.sum() < 2:
                continue
            dNde += np.exp(np.interp(np.log(fine), np.log(eps[ok]), np.log(y[ok]),
                                     left=-np.inf, right=-np.inf))

    # The simulated loss is TOTAL, so the radiative model alone cannot match
    # it: at p = 104 GeV the delta-ray kinematic limit is ~90 GeV, so the
    # ionization tail reaches across this whole range. Add it here, as the
    # standard Rutherford tail dN/deps = xi/eps^2 per step with
    # xi = 0.1535 (Z/A) rho d / beta^2 MeV -- the same xi computeErrorIoni
    # builds. Without this term the comparison shows a flat ~3.5x deficit that
    # looks like a normalization error in the radiative model and is not.
    dNde_delta = np.zeros_like(fine)
    for recs, spec in legs:
        for rec in recs:
            beta2 = (rec[R_P] / rec[R_ETOT]) ** 2
            xi = 0.1535 * (rec[R_EFFZ] / rec[R_EFFA]) * rec[R_XG] / max(beta2, 1e-9)
            tmax = 1e3 * 2 * M_E * (rec[R_P] / M_MU) ** 2 / (
                1. + 2. * (rec[R_ETOT] / M_MU) * M_E / M_MU + (M_E / M_MU) ** 2)
            dNde_delta += np.where(fine < tmax, xi / fine ** 2, 0.)

    eps_ed = np.geomspace(300., 30000., 12)          # MeV
    print(f"\nsim: {nev} events")
    print(f"{'eps bin [MeV]':>22} {'N_sim':>8} {'N_rad':>10} {'N_delta':>10} "
          f"{'N_total':>10} {'sim/tot':>8}")
    for j in range(len(eps_ed) - 1):
        ns = int(np.sum((loss >= eps_ed[j]) & (loss < eps_ed[j + 1])))
        m = (fine >= eps_ed[j]) & (fine < eps_ed[j + 1])
        nr = np.trapezoid(dNde[m], fine[m]) * nev if m.sum() > 1 else 0.
        nd = np.trapezoid(dNde_delta[m], fine[m]) * nev if m.sum() > 1 else 0.
        tot = nr + nd
        r = ns / tot if tot > 0 else np.nan
        print(f"{eps_ed[j]:10.0f} - {eps_ed[j+1]:8.0f} {ns:8d} {nr:10.1f} {nd:10.1f} "
              f"{tot:10.1f} {r:8.2f}")
    print("sim/tot ~ 1 across the range is the pass condition; the delta-ray "
          "term dominates below ~10 GeV even at this momentum")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="runCleanPropModel.py output (needs radv)")
    p.add_argument("--sim", default="", help="sim glob, for --validate-shape")
    p.add_argument("--validate", action="store_true")
    p.add_argument("--validate-shape", action="store_true")
    args = p.parse_args()

    legs, vg = load_radv(args.model)
    nst = sum(len(r) for r, _ in legs)
    print(f"{os.path.basename(args.model)}: {len(legs)} legs, {nst} radiative steps\n")
    if args.validate:
        validate_mean(legs, vg)
    if args.validate_shape:
        if not args.sim:
            raise SystemExit("--validate-shape needs --sim")
        validate_shape(legs, vg, args.sim)


if __name__ == "__main__":
    main()
