"""Radiative (bremsstrahlung + pair production) CF term for the muon transport model.

WHY THIS EXISTS
---------------
The mean and the fluctuation are currently inconsistent. The propagator's
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
  - absorbs the differing absolute-normalization convention of
    ComputeDMicroscopicCrossSection (measured factor 1.051 brems, 1.63e-3
    pair, both constant to ~1%, i.e. a convention offset not a shape error);
  - keeps the total mean exactly equal to what the propagator subtracted, so
    the centring leaves no residual bias;
  - gets the brems/pair MIXTURE right (pair is 58% of the radiative mean at
    100 GeV but is softer in v). An earlier hand-built brems-only shape,
    normalized to the combined mean, under-predicted the simulated radiative
    rate by ~2.5x at 5-15 GeV -- which is why the shapes now come from G4.

usage:
    python cf_brems_exact.py --validate --model <model.root>
    python cf_brems_exact.py --validate-shape --model <model.root> --sim '<sim>/*.root'
"""

import argparse
import os

import numpy as np

# radv layout, 11 doubles/step (Geant4ePropagator::RadiativeStep)
RADV_STRIDE = 11
# order must match G4ePropagationExport.cc's push_back sequence exactly
(R_EFFZ, R_EFFA, R_XG, R_ETOT, R_P, R_DX0, R_STEPCM,
 R_DEDXRAD, R_DEDXBREM, R_DEDXPAIR, R_CS) = range(11)
NRADV = 48   # Geant4ePropagator::kNRadV

M_MU = 0.1056583745      # GeV
M_E = 0.510998946e-3     # GeV
SQRT_E = np.sqrt(np.e)


def step_spectrum(rec, spec, vg):
    """(v grid, dN/dv) for one step, from Geant4's own tabulated shapes.

    Each process is normalized SEPARATELY to its own ComputeDEDXPerVolume:

      dN/dv = sum_proc  shape_proc(v) * dE_proc / INT v E shape_proc dv

    Why per process and not on the sum: pair production is 58% of the
    radiative mean here but is SOFTER in v than brems, so normalizing a single
    combined shape gets the mixture wrong -- that is precisely what the
    hand-built brems-only shape got wrong by ~2.5x at 5-15 GeV.

    Why normalize at all, given the shapes come from Geant4: the absolute
    normalization of ComputeDMicroscopicCrossSection does not match a naive
    dsigma/deps reading. Measured against each model's own ComputeDEDXPerVolume
    the required factor is 1.051 for brems and 1.63e-3 for pair, both CONSTANT
    (rel-rms 0.2% / 1.1%, <=4% residual Z-dependence) -- i.e. a convention
    offset, not a shape difference. Renormalizing absorbs it exactly, and also
    absorbs the v-grid cutoff, so the mean is right by construction and stays
    centred on what the propagator subtracted.
    """
    E = rec[R_ETOT]
    L = rec[R_STEPCM]
    out = np.zeros_like(vg)
    for shape, dedx in ((spec[:NRADV], rec[R_DEDXBREM]),
                        (spec[NRADV:], rec[R_DEDXPAIR])):
        dE = dedx * L
        if dE <= 0.:
            continue
        norm = np.trapezoid(shape * vg * E, vg)
        if norm > 0.:
            out = out + shape * (dE / norm)
    return vg, out


def rad_exponent(tau, recs, spec, vg, weights=None):
    """S_rad(t) = sum_steps INT dv (dN/dv) (e^{i a v E} - 1 - i a v E).

    tau      : (nt,) real CF argument grid
    recs     : (nstep, 11) radv records
    spec     : (nstep, 2*NRADV) tabulated shapes
    vg       : (NRADV,) shared v grid
    weights  : (nstep,) transport weight per step (default 1)
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
    # materializing an (nt, nv) complex temporary per step. Measured 2026-08-15
    # at 6.0 ms -> 0.03 ms for nt = 8001, nv = 48. (See the note below on why
    # the summation order change is harmless here.)
    wtrap = np.zeros(len(vg))
    dv = np.diff(np.asarray(vg, dtype=float))
    wtrap[:-1] += 0.5 * dv
    wtrap[1:] += 0.5 * dv
    for rec, sp, w in zip(recs, spec, weights):
        v, dNdv = step_spectrum(rec, sp, vg)
        if not np.any(dNdv > 0.):
            continue
        E = rec[R_ETOT]
        # a: dE -> the standardized variable, via the step's cs and the
        # caller's transport weight. cs = E/p^3 [GeV^-2] and v*E below is in
        # GeV, so d(qop) = cs * dE[GeV] directly -- NO MeV conversion. (The
        # 1e-3 in the UrbanIoniStep convention exists only because those
        # records store energies in MeV; putting a 1e3 here instead made the
        # exponent 1e6 too large and collapsed the CF to zero.)
        a = tau * rec[R_CS] * w
        x = np.outer(a, v * E)                       # (nt, nv)
        # e^{ix} - 1 - ix, evaluated stably. The series is the accurate branch
        # only when |x| is small ACROSS THE WHOLE SUPPORT -- the lesson from
        # the delta-ray term, where guarding on |a| instead of |a|*eps_max was
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
    a = t[key].arrays(["radv", "radspecv", "radvgrid"], library="np")
    vg = np.asarray(a["radvgrid"][0], dtype=float)
    legs = []
    for r, sp in zip(a["radv"], a["radspecv"]):
        recs = np.asarray(r, dtype=float).reshape(-1, RADV_STRIDE)
        spec = np.asarray(sp, dtype=float).reshape(-1, 2 * NRADV)
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
    # contain 0 or 1 of them -- sampling it directly gives spurious zeros (an
    # earlier version of this function did exactly that). Interpolate each
    # step onto a common fine eps grid in log-log and integrate there.
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
