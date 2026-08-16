#!/usr/bin/env python3
"""The composed gap: the analytic CF built from the exported record against
what STOCK Geant4 actually samples for the same step.

WHY THIS ONE AND NOT ANOTHER
----------------------------
The clean-propagation closure compares a SIMULATION drawn by stock Geant4
11.2.2 against an offline analytic model built from the per-step Urban record
`ioniurbanv`.  Only the two HALVES of that composition have ever been measured,
and each against something that is not the other:

  * NOTES_URBANSAMPLING: the analytic compound Poisson against
    `G4UniversalFluctuationForExtrapolator::SampleFluctuations2`.  532 sigma at
    the CF on a thin step; propagated, 0.008 - 3 % of the closure.  But that
    sampler is NOT in our chain: the record is, and its draws are thrown away.
  * NOTES_DELTASPEC section 10.3: that same extrapolator sampler against stock.
    17.6 sigma_MC (thick) / 101 (thin) even after the Urban-version
    harmonization -- i.e. a residual, unexplained sampler difference.

Neither is the closure's actual gap, which is
    analytic CF (from the record)   vs   what stock Geant4 draws.

WHAT "WHAT STOCK GEANT4 DRAWS" MEANS, AND WHY THE OBVIOUS TEST IS WRONG
----------------------------------------------------------------------
`G4VEnergyLossProcess` does NOT hand a step's whole energy loss to the
fluctuation model.  It splits it at the e- production threshold `tcut`:

    total loss  =  G4UniversalFluctuation(tcut, meanLoss = dE/dx(tcut) * L)
                +  sum of explicit G4Electron secondaries above tcut,
                   sampled from G4MuBetheBlochModel::SampleSecondaries

The offline model's record, by contrast, spans `e0` = 10 eV to the FULL Tmax:
the extrapolator passes no production cut.  So the correct comparison is the
COMPOSITE against the record's CF -- and driving stock at `tcut = tmax` (which
is what `urban_sampling.py stock` does) tests a configuration the simulation
never runs.  With `CVH_IONI_EXACTDELTA` on, that mistake would be large and
entirely artificial: it would "measure" the very correction the exporter
deliberately makes to the hard tail.

The composite has a structural consequence that is worth stating before any
number: because the analytic exponent is an integral over T and is therefore
additive in the integration range,

    S_composite - S_analytic
      = S_stockGlandz(tcut)  -  [ S_excitation(record) + S_exact(e0 -> tcut) ]

exactly -- the hard part cancels (up to the radiative correction, which is
measured here, not assumed).  THE COMPOSED GAP LIVES ENTIRELY BELOW THE
PRODUCTION CUT.

SUBCOMMANDS
    steps     the per-step parameter space: cuts, the sub-cut split, cumulants
    g4        drive the REAL C++ composite and validate the analytic composite
              CF against it (including which `scaling` convention stock uses --
              measured, not assumed)
    closure   propagate d(closure) over the u curve, both momenta, toy + real,
              with the same gauge that sized and then excluded three candidates
    shape     is the propagated shift the SHAPE of the observed residual?

Nothing in cf_propagation_test.py / cf_track_resolution.py / cgf_*.py /
deltaspec.py / urban_sampling.py is modified; they are imported.
"""

import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cf_propagation_test as cp                                   # noqa: E402
import cf_track_resolution as ctr                                  # noqa: E402
import fisher_norm as fn                                           # noqa: E402
import urban_sampling as us                                        # noqa: E402
import deltaspec as ds                                             # noqa: E402

SCRATCH = fn.SCRATCH
OUT = os.path.join(SCRATCH, "sg")
DSOUT = os.path.join(SCRATCH, "ds")
TP = os.path.join(SCRATCH, "tp")
DRIVER = os.path.join(SCRATCH, "urban_g4driver.sh")

R_REG, R_GSIG2, R_A1, R_E1, R_A2, R_E2, R_A3, R_E0, R_TMAX, R_SCAL, R_CS = range(11)
R_BETA2, R_ETOT = 11, 12

RATE = us.RATE      # 0.56
FW = us.FW          # 4
A0 = us.A0          # 42
KEV = 1e-3          # MeV


# =========================================================================
# the 2021 Glandz block, restricted to a production cut
# =========================================================================

def stock_glandz_block(meanloss, ipot, e0, tcut, cs=1e3, scaling=1.0):
    """The parameter block stock G4UniversalFluctuation builds for ONE step,
    in the same 11-column record layout the offline CF machinery consumes.

    Transcribed from the 2021 `SampleGlandz` -- which is present VERBATIM (and
    dead) inside G4UniversalFluctuationForExtrapolator, so this is a copy of a
    copy of the shipped code rather than a reading of the shipped code.  It is
    validated against real stock samples in `g4`.

    `meanloss` is the RESTRICTED mean loss (dE/dx(tcut) * L), already divided
    by `scaling` if the caller's convention applies one -- see `g4`, which
    MEASURES which convention stock uses instead of assuming it.  The returned
    block carries `scaling` in column 9, so every downstream consumer
    (sampler_exponent, ioni_step_exponent, step_cumulants) multiplies the
    energies by it exactly as it does for the extrapolator's record.
    """
    e1 = ipot
    a1 = 0.0
    if tcut > e1:
        a1 = meanloss * (1.0 - RATE) / e1
        if a1 < A0:
            fwnow = 0.1 + (FW - 0.1) * np.sqrt(a1 / A0)
            a1 /= fwnow
            e1 *= fwnow
        else:
            a1 /= FW
            e1 *= FW
    w1 = tcut / e0
    a3 = RATE * meanloss * (tcut - e0) / (e0 * tcut * np.log(w1))
    if a1 <= 0.0:
        a3 /= RATE
    return np.array([1.0, 0.0, a1, e1, 0.0, 0.0, a3, e0, tcut, scaling, cs])


def stock_scaling(tcut, tmax, mode):
    """The `scaling` width correction stock applies.  `mode` selects the
    hypothesis; `g4` measures which one is right rather than assuming."""
    if mode == "none":
        return 1.0
    if mode == "tcut":
        return min(1.0 + 0.5 * KEV / tcut, 1.50)
    if mode == "tmax":
        return min(1.0 + 0.5 * KEV / tmax, 1.50)
    raise ValueError(mode)


# =========================================================================
# the record's own decomposition at a production cut
# =========================================================================

def _I1(e0, tmax, beta2, etot, spinhalf=True):
    """int_{e0}^{tmax} T dN/dT dT  /  xi -- the exact spectrum's mean, in
    units of xi.  Same integrand as `exact_delta_exponent`."""
    v = np.log(tmax / e0) - beta2 * (tmax - e0) / tmax
    if spinhalf:
        v = v + (tmax ** 2 - e0 ** 2) / (4.0 * etot ** 2)
    return v


def _N(e0, tmax, beta2, etot, spinhalf=True):
    """int_{e0}^{tmax} dN/dT dT / xi -- the exact spectrum's RATE / xi."""
    v = (1.0 / e0 - 1.0 / tmax) - beta2 * np.log(tmax / e0) / tmax
    if spinhalf:
        v = v + (tmax - e0) / (2.0 * etot ** 2)
    return v


def record_split(rec, tcut):
    """(mu_tot, mu_hard, mu_soft, xi, e0, tmax, beta2, etot) for one regime-2
    record at a production cut `tcut` [MeV].

    mu_hard is what the SIMULATION puts into explicit secondaries; mu_soft is
    what is therefore left for the restricted straggling.  The record's OWN
    total mean is used, not Geant4's dE/dx table, so that the composite and
    the analytic model have the SAME mean and the CF comparison is a pure
    shape comparison.  (The two dE/dx tables differ by 0.4 %; that is a
    separate, already-known issue and it is reported by `g4`, not folded in
    here where it would contaminate the shape.)
    """
    gam = rec[R_SCAL]
    e0, tmax = rec[R_E0] * gam, rec[R_TMAX] * gam
    xi = rec[R_A3] * gam
    b2, et = rec[R_BETA2], rec[R_ETOT]
    half = rec[R_REG] == 2
    mu_exc = rec[R_A1] * rec[R_E1] * gam + rec[R_A2] * rec[R_E2] * gam
    mu_tot = mu_exc + xi * _I1(e0, tmax, b2, et, half)
    mu_hard = xi * _I1(tcut, tmax, b2, et, half) if tcut < tmax else 0.0
    return mu_tot, mu_hard, mu_tot - mu_hard, xi, e0, tmax, b2, et


def block_exponents(steps, tcuts, ipots, wstd, tau, scalmode="tcut",
                    kokoulin=True, nbin=96):
    """Every exponent this note needs, for one pooled ionization block, in
    standardized-z units.  Conventions identical to
    `cf_track_resolution.ioni_step_exponent`, so they can be differenced.

    Returns a dict:
      `ana`   the analytic model -- ioni_step_exponent VERBATIM
      `soft`  stock's restricted straggling, G4UniversalFluctuation at tcut
      `hard`  the explicit secondaries above tcut (tree-level exact spectrum)
      `kok`   the Kokoulin radiative correction to `hard` (0 if kokoulin=False)
      `comp`  soft + hard + kok = what the SIMULATION draws
      `asub`  the analytic model's OWN sub-cut part: excitation + exact
              spectrum on [e0, tcut]
      `gsub`  soft - asub          -- the straggling-law gap, BELOW the cut
      `gap`   comp - ana = gsub + kok   (checked, not assumed)

    The identity gap = gsub + kok holds because the analytic exponent is an
    integral over T and is therefore additive in the integration range:
    ana = asub + hard, so comp - ana = (soft - asub) + kok.
    """
    tau = np.asarray(tau, dtype=np.float64)
    nt = len(tau)
    gs = wstd * steps[:, R_CS].astype(np.float64) * 1e-3
    Z = lambda: np.zeros(nt, dtype=np.complex128)
    out = {k: Z() for k in ("soft", "hard", "kok", "asub")}

    blocks = []
    hv = {k: [] for k in ("xi", "t0", "tm", "b2", "et", "a", "half")}
    sv = {k: [] for k in ("xi", "e0", "t0", "tm", "b2", "et", "a", "half")}
    exc = []
    for i, r in enumerate(steps):
        if r[R_REG] == 0:
            out["soft"] += -0.5 * tau ** 2 * float(r[R_GSIG2] * gs[i] ** 2)
            out["asub"] += -0.5 * tau ** 2 * float(r[R_GSIG2] * gs[i] ** 2)
            continue
        tc = min(tcuts[i], r[R_TMAX] * r[R_SCAL])
        mu_tot, mu_hard, mu_soft, xi, e0, tmax, b2, et = record_split(r, tc)
        s = stock_scaling(tc, tmax, scalmode)
        # stock builds the block from meanLoss/scaling and multiplies the drawn
        # loss by scaling at the end; e0 and tcut go in UNSCALED.
        blocks.append(stock_glandz_block(mu_soft / s, ipots[i], r[R_E0], tc,
                                         cs=steps[i, R_CS], scaling=s))
        exc.append(i)
        half = r[R_REG] == 2
        if tc < tmax:
            for k, v in (("xi", xi), ("t0", tc), ("tm", tmax), ("b2", b2),
                         ("et", et), ("a", gs[i]), ("half", half)):
                hv[k].append(v)
        if e0 < tc:
            for k, v in (("xi", xi), ("e0", e0), ("t0", tc), ("tm", tmax),
                         ("b2", b2), ("et", et), ("a", gs[i]), ("half", half)):
                sv[k].append(v)
    if blocks:
        out["soft"] += us.sampler_exponent(np.array(blocks), wstd, tau)
    # the analytic model's excitation channels (its sub-cut part, part 1)
    if exc:
        st = steps[exc]
        gsu = gs[exc]
        gam = st[:, R_SCAL]
        for ja, je in ((R_A1, R_E1), (R_A2, R_E2)):
            aj, ej = st[:, ja], st[:, je] * gam
            act = (aj > 0.0) & (ej > 0.0)
            if act.any():
                th = (gsu[act] * ej[act])[:, None] * tau[None, :]
                out["asub"] += np.sum(
                    aj[act][:, None] * (np.exp(1j * th) - 1.0 - 1j * th), axis=0)
    for tag, v, dest in (("hard", hv, "hard"), ("sub", sv, "asub")):
        if not v["xi"]:
            continue
        arr = {k: np.array(v[k]) for k in v}
        aa = arr["a"][:, None] * tau[None, :]
        lo = arr["t0"] if tag == "hard" else arr["e0"]
        hi = arr["tm"] if tag == "hard" else arr["t0"]
        for spin in (True, False):
            m = arr["half"] == spin
            if not m.any():
                continue
            if tag == "hard":
                out["hard"] += ctr.exact_delta_exponent(
                    arr["xi"][m], lo[m], hi[m], arr["b2"][m], arr["et"][m],
                    aa[m], spinhalf=spin)
                if kokoulin:
                    out["kok"] += kokoulin_exponent(
                        arr["xi"][m], lo[m], hi[m], arr["b2"][m], arr["et"][m],
                        aa[m], nbin=nbin, spinhalf=spin)
            else:
                out["asub"] += subrange_delta_exponent(
                    arr["xi"][m], lo[m], hi[m], arr["tm"][m], arr["b2"][m],
                    arr["et"][m], aa[m], spinhalf=spin)
    out["ana"] = ctr.ioni_step_exponent(steps, wstd, tau)
    out["comp"] = out["soft"] + out["hard"] + out["kok"]
    out["gsub"] = out["soft"] - out["asub"]
    out["gap"] = out["comp"] - out["ana"]
    return out


def composite_exponent(steps, tcuts, ipots, wstd, tau, scalmode="tcut",
                       hard=True, kokoulin=True, nbin=96):
    e = block_exponents(steps, tcuts, ipots, wstd, tau, scalmode, kokoulin, nbin)
    return e["comp"] if hard else e["soft"]


# =========================================================================
# the Kokoulin radiative correction, in closed form per log bin
# =========================================================================
#
# G4MuBetheBlochModel multiplies the knock-on spectrum by Kokoulin's radiative
# correction above T = 100 keV (for muons above 1 GeV).  It is DELIBERATELY not
# in the offline model (NOTES_DELTASPEC section 1.5: no closed-form CF, a few
# percent of a channel that is itself a 28 % correction) and its size is quoted
# there as a known residual.  It has never been PROPAGATED, and it is part of
# "what stock Geant4 actually samples", so it is carried here.
#
# The correction factor kappa(T) = f_K(T) - 1 is smooth and slowly varying in
# ln T; the oscillatory part e^{i a T} is not, and at a T ~ 2e4 no practical
# quadrature in T resolves it.  So kappa is made piecewise constant on a log
# grid and the OSCILLATORY integral is done in closed form on each bin, with
# the SAME `_delta_terms_exact` the tree-level channel uses:
#
#     int_lo^hi (e^{iaT} - 1 - iaT) [1/T^2 - b2/(T Tmax) + 1/(2E^2)] dT
#
# is `exact_delta_exponent(xi, lo, hi, b2 * hi / Tmax, E, a)` -- the beta^2
# denominator is Tmax, the integration limit is hi, and the substitution
# b2 -> b2 hi/Tmax makes the existing routine compute exactly that.  Checked
# by summing contiguous bins against the full-range closed form.

ALPHA_PRIME = ds.ALPHA_PRIME
KOK_TMIN = ds.KOKOULIN_TMIN
KOK_MUMIN = ds.KOKOULIN_MUMIN
MMU = ds.MMU


def kokoulin_factor(T, etot):
    """f_K(T), vectorized over T AND over the per-step total energy."""
    T = np.asarray(T, dtype=np.float64)
    E = np.broadcast_to(np.asarray(etot, dtype=np.float64), T.shape)
    f = np.ones_like(T)
    m = (T > KOK_TMIN) & (T < E - MMU)
    if np.any(m):
        a1 = np.log(1.0 + 2.0 * T[m] / ds.MEL)
        a3 = np.log(4.0 * E[m] * (E[m] - T[m]) / (MMU * MMU))
        f[m] = 1.0 + ALPHA_PRIME * a1 * (a3 - a1)
    return f


def subrange_delta_exponent(xi, lo, hi, tmax, beta2, etot, a, spinhalf=True):
    """int_lo^hi (e^{iaT}-1-iaT) dN/dT dT with dN/dT the exact spectrum whose
    beta^2 term carries the KINEMATIC Tmax.  Vectorized over steps."""
    return ctr.exact_delta_exponent(
        np.asarray(xi, float), np.asarray(lo, float), np.asarray(hi, float),
        np.asarray(beta2, float) * np.asarray(hi, float) / np.asarray(tmax, float),
        np.asarray(etot, float), a, spinhalf=spinhalf)


def kokoulin_exponent(xi, t0, tmax, beta2, etot, a, nbin=48, spinhalf=True):
    """The Kokoulin CORRECTION's contribution to the delta channel's exponent,
    over [t0, tmax].  Zero when the whole range is below KOK_TMIN."""
    xi = np.atleast_1d(np.asarray(xi, float))
    t0 = np.atleast_1d(np.asarray(t0, float))
    tmax = np.atleast_1d(np.asarray(tmax, float))
    beta2 = np.atleast_1d(np.asarray(beta2, float))
    etot = np.atleast_1d(np.asarray(etot, float))
    a = np.atleast_2d(a)
    S = np.zeros(a.shape[1], dtype=np.complex128)
    lo0 = np.maximum(t0, KOK_TMIN)
    act = (lo0 < tmax) & (etot - MMU > KOK_MUMIN)
    if not act.any():
        return S
    xi, lo0, tmx, b2, et, aa = (xi[act], lo0[act], tmax[act], beta2[act],
                                etot[act], a[act])
    # log grid per step
    fr = np.linspace(0.0, 1.0, nbin + 1)
    edges = lo0[:, None] * (tmx / lo0)[:, None] ** fr[None, :]
    for m in range(nbin):
        lo, hi = edges[:, m], edges[:, m + 1]
        # energy-weighted centroid of the bin under the 1/T spectrum
        Tc = np.sqrt(lo * hi)
        kap = kokoulin_factor(Tc, et) - 1.0
        good = kap != 0.0
        if not good.any():
            continue
        S += subrange_delta_exponent(xi[good] * kap[good], lo[good], hi[good],
                                     tmx[good], b2[good], et[good], aa[good],
                                     spinhalf=spinhalf)
    return S


# =========================================================================
# material identification and the cut map
# =========================================================================

def ipot_map(off_model):
    """Zeff -> mean excitation energy [MeV], recovered from the OFF (regime 1,
    pre-2021) records, where `material_from_record`'s fwnow inversion applies
    unmodified.  Exact by construction: G4IonisParamMat DEFINES
    e1Fluct = (I / e2Fluct^f2)^(1/f1), so I = e1F^f1 e2F^f2.

    The ON records cannot be used blindly: CVH_IONI_EXACTDELTA rescales a1,
    which selects the fwnow branch for thin steps.
    """
    legs = cp.load_model(off_model)
    st = np.concatenate([l["ioni"] for l in legs if len(l["ioni"])])
    out = {}
    for r in st:
        if r[R_REG] == 0 or r[R_E2] <= 0:
            continue
        z = round(float(np.sqrt(r[R_E2] / 1e-5)), 4)
        if z not in out:
            out[z] = float(us.material_from_record(r)[0])
    return out


def zeff_of(steps):
    return np.round(np.sqrt(steps[:, R_E2] / 1e-5), 4)


TOY_CUTNAME = {8.0: "ToyLayerMat", 4.0: "Beryllium", 7.374: "Air"}


def toy_cuts(tag):
    """Zeff -> e- production cut [MeV] for the layered toy, from the
    simulation's own G4ProductionCutsTable dump."""
    _, logpath, _ = ds.CFG[tag]
    tbl = ds.cuts_from_log(logpath)
    return {z: tbl[n] for z, n in TOY_CUTNAME.items()}


def step_cuts(steps, cutmap, default=None):
    z = zeff_of(steps)
    out = np.empty(len(steps))
    for i, zz in enumerate(z):
        if cutmap is None:
            out[i] = default
            continue
        k = min(cutmap, key=lambda c: abs(c - zz))
        if abs(k - zz) > 0.05:
            if default is None:
                raise KeyError(f"no cut for Zeff {zz}")
            out[i] = default
        else:
            out[i] = cutmap[k]
    return out


def step_ipots(steps, imap):
    z = zeff_of(steps)
    out = np.empty(len(steps))
    for i, zz in enumerate(z):
        k = min(imap, key=lambda c: abs(c - zz))
        assert abs(k - zz) < 1e-3, f"no ipot for Zeff {zz}"
        out[i] = imap[k]
    return out


# =========================================================================
# subcommand: steps
# =========================================================================

CONFIGS = {
    # tag -> (on-model, off-model, cut source)
    "pt3_cut1e4": (f"{DSOUT}/pt3_model_on.root", f"{DSOUT}/pt3_model_off.root",
                   "pt3_cut1e4"),
    "pt3_base": (f"{DSOUT}/pt3_model_on.root", f"{DSOUT}/pt3_model_off.root",
                 "pt3_base"),
    "pt40_cut001": (f"{DSOUT}/pt40_model_on.root", f"{DSOUT}/pt40_model_off.root",
                    "pt40_cut001"),
    "pt40_base": (f"{DSOUT}/pt40_model_on.root", f"{DSOUT}/pt40_model_off.root",
                  "pt40_base"),
    "real": (f"{DSOUT}/real_model_on.root", f"{DSOUT}/real_model_off.root", None),
}


def load_cfg(tag, uniform_cut=None):
    on, off, cs = CONFIGS[tag]
    legs = cp.load_model(on)
    imap = ipot_map(off)
    cmap = toy_cuts(cs) if cs else None
    return legs, imap, cmap, uniform_cut


def cmd_steps(args):
    print("=" * 100)
    print("THE COMPOSED GAP, PER STEP: what the record says vs how the "
          "SIMULATION splits the same step")
    print("=" * 100)
    print("mu_soft is the mean the SIM leaves to G4UniversalFluctuation "
          "(everything below the e-\nproduction cut); mu_hard goes into "
          "explicit G4Electron secondaries.  `exc frac` is the\nfraction of "
          "mu_soft the two models put in the EXCITATION channel: the analytic "
          "side\nfrom the record, stock from its hard-wired rate = 0.56.  "
          "That split is the whole gap.\n")
    for tag in args.configs:
        legs, imap, cmap, ucut = load_cfg(tag, args.cut)
        st = np.concatenate([l["ioni"] for l in legs if len(l["ioni"])])
        cuts = step_cuts(st, cmap, ucut)
        ips = step_ipots(st, imap)
        print(f"--- {tag}   {len(st)} steps, {len(legs)} legs")
        print(f"    Zeff -> cut [MeV]: " + ", ".join(
            f"{z}:{c:.5g}" for z, c in sorted(
                {float(z): float(c) for z, c in zip(zeff_of(st), cuts)}.items())))
        print(f"    {'#':>4}{'Zeff':>7}{'tcut[keV]':>11}{'mu_tot':>10}"
              f"{'mu_soft':>10}{'mu_hard':>10}{'excA':>7}{'excS':>7}"
              f"{'a1S':>9}{'a3S':>10}{'scal':>7}"
              f"{'k2soft_A':>11}{'k2soft_S':>11}{'d/k2core':>10}"
              f"{'w share':>9}")
        # variance weight of each step in the block, z^2 units at the last plane
        avec = cp.FUNCTIONALS["qop"]
        klast = len(legs) - 1
        sig = float(np.sqrt(cp.model_variance(legs, klast, avec)[0]))
        wst = us.weighted_steps(legs, klast, avec, sig)
        gs = wst[:, R_CS] * 1e-3
        kA = np.array([us.step_cumulants(r, "analytic") for r in st])
        vz = kA[:, 1] * gs ** 2
        rows = []
        for i, r in enumerate(st):
            mu_tot, mu_hard, mu_soft, xi, e0, tmax, b2, et = record_split(r, cuts[i])
            s = stock_scaling(cuts[i], tmax, args.scalmode)
            blk = stock_glandz_block(mu_soft / s, ips[i], r[R_E0], cuts[i],
                                     cs=r[R_CS], scaling=s)
            mu_exc_A = r[R_A1] * r[R_E1] * r[R_SCAL] + r[R_A2] * r[R_E2] * r[R_SCAL]
            # sub-cut variances
            half = r[R_REG] == 2
            k2A = (r[R_A1] * (r[R_E1] * r[R_SCAL]) ** 2
                   + r[R_A2] * (r[R_E2] * r[R_SCAL]) ** 2
                   + xi * _exact_k2(e0, min(cuts[i], tmax), tmax, b2, et, half))
            k2S = us.step_cumulants(blk, "analytic")[1]
            core = core_scale(r) ** 2
            rows.append((i, float(zeff_of(st)[i]), cuts[i], mu_tot, mu_soft,
                         mu_hard, mu_exc_A / mu_soft, RATE and (1 - RATE),
                         blk[R_A1], blk[R_A3], s, k2A, k2S,
                         (k2S - k2A) / core, vz[i] / vz.sum()))
        order = np.argsort([-r[-1] for r in rows])
        show = order if args.all else order[:args.nshow]
        for j in show:
            (i, z, tc, mt, msf, mh, eA, eS, a1S, a3S, s, k2A, k2S, dk, w) = rows[j]
            print(f"    {i:4d}{z:7.3f}{1e3 * tc:11.4g}{mt:10.4g}{msf:10.4g}"
                  f"{mh:10.4g}{eA:7.3f}{eS:7.3f}{a1S:9.4g}{a3S:10.4g}{s:7.4f}"
                  f"{k2A:11.4e}{k2S:11.4e}{dk:+10.3e}{w:9.5f}")
        tot = sum(r[-1] * r[-2] for r in rows)
        print(f"    variance-weighted mean of d k2_soft / k2_core = {tot:+.4e}")
        print()


def _exact_k2(lo, hi, tmax, beta2, etot, spinhalf=True):
    """int_lo^hi T^2 dN/dT dT / xi.

    NOTE the beta^2 suppression's denominator is Tmax -- the KINEMATIC limit --
    not the upper integration limit.  Writing `hi` there is wrong by a factor
    Tmax/hi = 7e4 on the suppression term when the range is cut at a 9.5 keV
    production threshold, which is exactly how this was caught: it turned a
    4.7e-3 integrand into a 9.5e-3 one.
    """
    v = (hi - lo) - beta2 * (hi ** 2 - lo ** 2) / (2.0 * tmax)
    if spinhalf:
        v = v + (hi ** 3 - lo ** 3) / (6.0 * etot ** 2)
    return v


def core_scale(rec):
    """A width for the CORE of one step's loss: sqrt of the variance carried by
    collisions whose expected multiplicity exceeds one.

    `urban_sampling._core_scale` assumes regime 1, where column 6 is a
    COLLISION COUNT; in regime 2 it is the energy `xi`, so that function
    silently returns the excitation variance alone (5.4e-3 -> 1.7e-4 here, a
    factor 31) and every "% of core" would be wrong by that factor.  The
    delta-ray count above T is N(>T) = xi (1/T - 1/Tmax - ...) so N = 1 at
    T ~ xi, and the variance below that is the relevant one.
    """
    gam = rec[R_SCAL]
    e0, tmax = rec[R_E0] * gam, rec[R_TMAX] * gam
    v = rec[R_A1] * (rec[R_E1] * gam) ** 2 + rec[R_A2] * (rec[R_E2] * gam) ** 2
    if rec[R_REG] in (2, 3):
        xi = rec[R_A3] * gam
        b2, et = rec[R_BETA2], rec[R_ETOT]
        half = rec[R_REG] == 2
        from scipy.optimize import brentq
        f = lambda T: xi * _N(T, tmax, b2, et, half) - 1.0
        if f(e0) <= 0:
            return np.sqrt(v)
        E1 = brentq(f, e0, tmax, xtol=1e-14, rtol=1e-14)
        v += xi * _exact_k2(e0, E1, tmax, b2, et, half)
        return np.sqrt(v)
    return us._core_scale(rec)


# =========================================================================
# subcommand: g4 -- drive the real C++ and validate the analytic composite
# =========================================================================

def run_driver(Z, A, rho, ekin, length, tmax, tcut, n, seed, out,
               meanloss=None, pdg=13):
    cmd = [DRIVER, "--Z", f"{Z:.17g}", "--A", f"{A:.17g}", "--rho", f"{rho:.17g}",
           "--ekin", f"{float(ekin):.17g}", "--len", f"{float(length):.17g}",
           "--tmax", f"{float(tmax):.17g}", "--tcut", f"{float(tcut):.17g}",
           "--n", str(int(n)), "--seed", str(int(seed)), "--pdg", str(int(pdg)),
           "--composite", "--out", out]
    if meanloss is not None:
        cmd += ["--compmeanloss", f"{float(meanloss):.17g}"]
    subprocess.run(cmd, check=True, capture_output=True)
    d = {}
    for line in open(out + ".rec"):
        p = line.split()
        if not p:
            continue
        if p[0] == "record":
            d["rec"] = np.array([float(x) for x in p[1:11]] + [1e3])
        elif p[0] == "matxi":
            d["xi"], d["beta2"], d["gamma"] = float(p[4]), float(p[6]), float(p[8])
        elif p[0] == "mationi":
            d["ipot"] = float(p[2])
        elif p[0] == "composite":
            d.update(tcut=float(p[2]), dedxR=float(p[4]), dedxU=float(p[6]),
                     mlR=float(p[8]), lam=float(p[10]))
        elif p[0] == "compsample":
            d.update(cmean=float(p[2]), cvar=float(p[4]), nd=float(p[6]),
                     eh=float(p[8]), tseen=float(p[10]))
        elif p[0] == "sample2":
            d["m2"], d["v2"] = float(p[2]), float(p[4])
    d["comp"] = np.fromfile(out + ".comp.bin", dtype=np.float64)
    d["soft"] = np.fromfile(out + ".soft.bin", dtype=np.float64)
    return d


def _ecf(x, t):
    xc = x - x.mean()
    return np.array([np.mean(np.exp(1j * tt * xc)) for tt in t])


def _rec13(rec10, xi, beta2, etot):
    """An 11-column driver record turned into the regime-2, stride-13 record
    the offline model would have exported for the same step."""
    r = np.zeros(13)
    r[:11] = rec10
    r[R_REG] = 2.0
    gam = r[R_SCAL]
    e0, tmax = r[R_E0] * gam, r[R_TMAX] * gam
    # a1, a2 rescaled by a COMMON factor so the block mean is held at the
    # dE/dx value -- exactly what CVH_IONI_EXACTDELTA does in the exporter.
    C = 1.0 / (1.0 / e0 - 1.0 / tmax)
    mu = r[R_A1] * r[R_E1] * gam + r[R_A2] * r[R_E2] * gam \
        + r[R_A3] * C * np.log(tmax / e0)
    mu_exc = mu - xi * _I1(e0, tmax, beta2, etot)
    f = mu_exc / (r[R_A1] * r[R_E1] * gam + r[R_A2] * r[R_E2] * gam)
    r[R_A1] *= f
    r[R_A2] *= f
    r[R_A3] = xi / gam
    r[R_BETA2], r[R_ETOT] = beta2, etot
    return r


def cmd_g4(args):
    print("=" * 100)
    print("THE CF-LEVEL GAP, AGAINST REAL STOCK GEANT4")
    print("=" * 100)
    print("Every `stock` number below is N real C++ samples of\n"
          "  G4UniversalFluctuation::SampleFluctuations(couple, dp, tcut, "
          "tmax, L, dE/dx(tcut) L)\n"
          "plus explicit G4MuBetheBlochModel secondaries above tcut, i.e. the "
          "decomposition\nG4VEnergyLossProcess uses.  The CF is probed on the "
          "scale of the step's own CORE.\n")
    os.makedirs(OUT, exist_ok=True)
    for name, (Z, A, rho, ekin, length, tcut) in args.steps.items():
        pre = os.path.join(OUT, f"g4_{name}")
        # first pass: the record and Geant4's own dE/dx, no mean override
        d = run_driver(Z, A, rho, ekin, length, -1.0, tcut, args.n, args.seed, pre)
        rec = d["rec"]
        tmax = rec[R_TMAX] * rec[R_SCAL]
        etot = d["gamma"] * 105.6583715
        r13 = _rec13(rec, d["xi"], d["beta2"], etot)
        mu_tot, mu_hard, mu_soft, xi, e0, tmx, b2, et = record_split(r13, tcut)
        print(f"### {name}:  Z={Z:g} rho={rho:g} L={length:g} mm  ekin={ekin:g} MeV"
              f"  tcut={1e3 * tcut:.4g} keV  Tmax={tmax:.4g} MeV   N={args.n}")
        print(f"    record   a1 {rec[R_A1]:10.5g} e1 {rec[R_E1]:.5e} "
              f"a2 {rec[R_A2]:9.4g} a3 {rec[R_A3]:10.5g}  scaling {rec[R_SCAL]:.7f}")
        print(f"    xi {xi:.6g} MeV   ipot {d['ipot']:.6g} MeV   "
              f"beta2 {b2:.6f}  E {et:.6g} MeV")
        print(f"    MEAN LOSS   record(extrapolator dE/dx) {mu_tot:.6f}   "
              f"G4MuBetheBloch unrestricted {d['dedxU'] * length:.6f}   "
              f"rel {(d['dedxU'] * length / mu_tot - 1):+.3e}")
        print(f"    RESTRICTED  record split at tcut {mu_soft:.6f}   "
              f"G4 dE/dx(tcut) L {d['dedxR'] * length:.6f}   "
              f"rel {(d['dedxR'] * length / mu_soft - 1):+.3e}")
        print(f"    HARD DELTAS rate  exact xi N {xi * _N(tcut, tmx, b2, et) :.5f}"
              f"   G4 sigma L {d['lam']:.5f}   ratio {d['lam'] / (xi * _N(tcut, tmx, b2, et)):.5f}")
        print(f"                energy exact {mu_hard:.6f}   G4 sampled "
              f"{d['eh']:.6f}   ratio {d['eh'] / mu_hard:.5f}"
              f"   <- the Kokoulin radiative correction")

        # second pass: drive stock at the record's OWN restricted mean, so the
        # comparison is pure shape
        d2 = run_driver(Z, A, rho, ekin, length, -1.0, tcut, args.n,
                        args.seed + 1, pre + "b", meanloss=mu_soft)
        x, xs = d2["comp"], d2["soft"]
        core = 0.5 * (np.percentile(x, 84) - np.percentile(x, 16))
        t = np.linspace(0.0, args.tmaxcf / core, args.nt)[1:]
        nrm = 1.0 / np.sqrt(len(x))
        print(f"    core (16-84 half width) {core:.6g} MeV;  probe grid "
              f"0 .. {args.tmaxcf}/core, {args.nt} nodes")

        # --- which `scaling` convention does stock use?  MEASURED.
        st1 = np.array([r13])
        ips = np.array([d["ipot"]])
        tc = np.array([tcut])
        print(f"\n    {'':38s}{'max |ecf - CF|':>16}{'sigma_MC':>11}")
        ecs = _ecf(xs, t)
        best, bestv = None, np.inf
        for mode in ("none", "tcut", "tmax"):
            S = composite_exponent(st1, tc, ips, 1.0, t, scalmode=mode, hard=False)
            dev = float(np.max(np.abs(np.exp(S) - ecs)))
            s = stock_scaling(tcut, tmax, mode)
            print(f"    soft only vs CF_stockGlandz(scaling={mode:<5s} "
                  f"= {s:.5f}){dev:16.3e}{dev / nrm:11.2f}")
            if dev < bestv:
                best, bestv = mode, dev
        print(f"    ==> stock's width-correction convention is `{best}` "
              f"({bestv / nrm:.2f} sigma_MC)")

        # --- the HARD part on its own.  `comp` and `soft` are written from the
        #     same iteration, so their difference is exactly the explicit
        #     secondaries of that event -- the one term whose law is asserted
        #     (an exact Poisson process on the exact spectrum) rather than
        #     transcribed.  If the composite fails, this says whether it is
        #     here or in the straggling.
        xh = x - xs
        ex = block_exponents(st1, tc, ips, 1.0, t, scalmode=best)
        e0k = block_exponents(st1, tc, ips, 1.0, t, scalmode=best, kokoulin=False)
        dh0 = float(np.max(np.abs(np.exp(ex["hard"]) - _ecf(xh, t))))
        dh1 = float(np.max(np.abs(np.exp(ex["hard"] + ex["kok"]) - _ecf(xh, t))))
        print(f"    {'explicit deltas vs CF_exact (tree)':<38}{dh0:16.3e}"
              f"{dh0 / nrm:11.2f}")
        print(f"    {'explicit deltas vs CF_exact x Kokoulin':<38}{dh1:16.3e}"
              f"{dh1 / nrm:11.2f}   <- Geant4's radiative correction, "
              f"absent from the model")
        print(f"    hard: <n> {d2['nd']:.5f} (tree {xi * _N(tcut, tmx, b2, et):.5f})"
              f"   <E> {xh.mean():.6f} (tree {mu_hard:.6f})"
              f"   var {xh.var():.6g}   max {xh.max():.4g} (Tmax {tmx:.4g})")

        # --- the composed gap, and its exact decomposition
        ec = _ecf(x, t)
        dc = float(np.max(np.abs(np.exp(ex["comp"]) - ec)))
        da = float(np.max(np.abs(np.exp(ex["ana"]) - ec)))
        dp_ = float(np.max(np.abs(np.exp(ex["comp"]) - np.exp(ex["ana"]))))
        dsub = float(np.max(np.abs(np.exp(ex["soft"]) - np.exp(ex["asub"]))))
        dkok = float(np.max(np.abs(np.exp(ex["hard"] + ex["kok"])
                                   - np.exp(ex["hard"]))))
        print(f"\n    {'|ecf(stock composite) - CF_composite|':<38}{dc:16.3e}"
              f"{dc / nrm:11.2f}   <- validates the analytic composite")
        print(f"    {'|ecf(stock composite) - CF_analytic|':<38}{da:16.3e}"
              f"{da / nrm:11.2f}   <- THE COMPOSED GAP, measured")
        print(f"    {'|CF_composite - CF_analytic|':<38}{dp_:16.3e}"
              f"{dp_ / nrm:11.2f}   <- the same gap in closed form")
        print(f"    {'  of which  sub-cut straggling law':<38}{dsub:16.3e}"
              f"{dsub / nrm:11.2f}")
        print(f"    {'  of which  Kokoulin above the cut':<38}{dkok:16.3e}"
              f"{dkok / nrm:11.2f}")
        print(f"    exponents: max|gsub| {np.max(np.abs(ex['gsub'])):.5f}   "
              f"max|kok| {np.max(np.abs(ex['kok'])):.5f}   "
              f"max|gap| {np.max(np.abs(ex['gap'])):.5f}   "
              f"(identity gap = gsub + kok holds to "
              f"{np.max(np.abs(ex['gap'] - ex['gsub'] - ex['kok'])):.1e})")
        # cumulants -- regime-2 aware (us.step_cumulants reads column 6 as a
        # collision COUNT, which it is not in regime 2)
        k1A = mu_tot
        k2A = (r13[R_A1] * (r13[R_E1] * r13[R_SCAL]) ** 2
               + r13[R_A2] * (r13[R_E2] * r13[R_SCAL]) ** 2
               + xi * _exact_k2(e0, tmx, tmx, b2, et))
        print(f"\n    kappa2:  analytic {k2A:.6e}   sampled composite "
              f"{x.var():.6e}   core(mine)^2 {core_scale(r13) ** 2:.4e}")
        print(f"    mean:    analytic {k1A:.6f}   sampled composite "
              f"{x.mean():.6f}   rel {(x.mean() / k1A - 1):+.3e}")
        print()


# =========================================================================
# subcommand: closure
# =========================================================================

def _phis(legs, k, avec, sigma, tau, cuts_fn, imap, scalmode, nbin=96):
    """phi_analytic and the three modified phis, at plane k.

    phi_analytic is `cf_propagation_test.model_phi` VERBATIM -- the published
    closure's own function -- and each modified phi multiplies it by exp(dS)
    with both exponents built from the SAME weighted steps, so every other
    channel (MS, radiative) and every transport weight cancels identically.
    Exactly the technique NOTES_URBANSAMPLING used to size, and then exclude,
    three candidates.
    """
    phiA = cp.model_phi(legs, k, avec, sigma, tau)
    st = us.weighted_steps(legs, k, avec, sigma)
    if not len(st):
        return phiA, {"gap": phiA, "gsub": phiA, "kok": phiA}
    e = block_exponents(st, cuts_fn(st), step_ipots(st, imap), 1.0, tau,
                        scalmode, nbin=nbin)
    return phiA, {n: phiA * np.exp(e[n]) for n in ("gap", "gsub", "kok")}


OBSERVED = {
    # NOTES_DELTASPEC section 10.4, the `on` rows: the residual that remains
    # AFTER the exact-delta correction.  UCURVE = 1e-3 .. 10, 9 probes.
    "pt3_cut1e4": np.array([-0.00118, -0.00213, -0.00437, -0.00819, -0.01076,
                            -0.00663, -0.00284, -0.00153, -0.00083]),
    "pt40_cut001": np.array([-0.00306, -0.00716, -0.01671, -0.02342, -0.01138,
                             -0.00362, -0.00159, -0.00089, -0.00051]),
    "real": np.array([-0.00135, -0.00268, -0.00567, -0.00931, -0.00543,
                      +0.00520, +0.00564, +0.00357, +0.00206]),
    "pt3_base": np.array([+0.00059, +0.00212, +0.00601, +0.01183, +0.01265,
                          +0.00156, -0.00258, -0.00191, -0.00115]),
    "pt40_base": np.array([-0.00054, -0.00155, -0.00517, -0.01173, -0.01379,
                           -0.00812, -0.00425, -0.00243, -0.00134]),
}
OBSERR = {
    "pt3_cut1e4": np.array([0.00020, 0.00026, 0.00031, 0.00036, 0.00044,
                            0.00042, 0.00029, 0.00019, 0.00011]),
}


def cmd_closure(args):
    print("=" * 100)
    print("PROPAGATION: what the composed gap does to the clean-propagation "
          "closure")
    print("=" * 100)
    print("closure(u) = <e^{-u z^2}>_data - <e^{-u z^2}>_model.  Only the "
          "MODEL side can move, so\n"
          "  d closure(u) = <e^{-u z^2}>_model,analytic - "
          "<e^{-u z^2}>_model,composite\n"
          "is what the published closure would become if the model described "
          "what stock Geant4\ndraws.  s_F held at its published (analytic) "
          "value: the data z do not move.\n")
    probes = fn.UCURVE
    for tag in args.configs:
        legs, imap, cmap, ucut = load_cfg(tag, args.cut)
        for func in args.funcs:
            avec = cp.FUNCTIONALS[func]
            sc = fn.plane_scales(legs, func, tag=CONFIGS[tag][0],
                                 channels=("ioni", "ms", "rad"))
            print(f"### {tag}  {func}   "
                  f"{'uniform cut %.4g MeV' % ucut if cmap is None else 'per-material cuts'}")

            def cuts_fn(st):
                return step_cuts(st, cmap, ucut)

            rows = {n: [] for n in ("gap", "gsub", "kok")}
            for k in range(len(legs)):
                s = float(sc["sF"][k])
                tau = fn.closure_tau(float(probes.max()), n=args.taun)
                pA, pm = _phis(legs, k, avec, s, tau, cuts_fn, imap,
                               args.scalmode, nbin=args.nbin)
                base = np.array([fn.weier_scalar(pA, u, tau) for u in probes])
                for n in rows:
                    rows[n].append(base - np.array(
                        [fn.weier_scalar(pm[n], u, tau) for u in probes]))
            rows = {n: np.array(v) for n, v in rows.items()}
            m = rows["gap"].mean(axis=0)
            if args.perplane:
                print(f"  {'plane':>5} " + "".join(f"{u:>10.3g}" for u in probes))
                for k in range(len(legs)):
                    print(f"  {k:5d} " + "".join(
                        f"{v:+10.2e}" for v in rows["gap"][k]))
            for n, lab in (("gsub", "sub-cut law"), ("kok", "Kokoulin"),
                           ("gap", "TOTAL")):
                v = rows[n].mean(axis=0)
                print(f"  {lab:>11} " + "".join(f"{x:+10.2e}" for x in v)
                      + f"   rms {np.sqrt(np.mean(v ** 2)):.5f}")
            obs = OBSERVED.get(tag)
            if obs is not None:
                print(f"  {'observed':>11} " + "".join(f"{v:+10.2e}" for v in obs)
                      + f"   rms {np.sqrt(np.mean(obs ** 2)):.5f}")
                print(f"  {'ratio':>11} " + "".join(
                    f"{v / o:+10.2e}" for v, o in zip(m, obs)))
                # closure = data - model, so replacing the model by the
                # composite gives closure_new = closure_pub + d.  (Verified
                # from geom_closure.closure_rows: `rows = e.mean(axis=1) -
                # models[k]`, i.e. data MINUS model, and d = M_ana - M_comp.)
                r = obs + m
                print(f"  {'obs+TOTAL':>11} " + "".join(f"{v:+10.2e}" for v in r)
                      + f"   rms {np.sqrt(np.mean(r ** 2)):.5f}"
                      + f"   ({100 * (1 - np.sqrt(np.mean(r ** 2)) / np.sqrt(np.mean(obs ** 2))):+.1f} %)")
                # THE SHAPE TEST.  lambda is the multiple of the mechanism that
                # best cancels the residual: min_lambda ||obs + lambda d||.
                # lambda = 1 is the mechanism AS COMPUTED, so lambda near 1 with
                # a large rms reduction is a match in BOTH size and shape;
                # |cos| near 1 with lambda far from 1 is a shape match only.
                for n in ("gsub", "kok", "gap"):
                    v = rows[n].mean(axis=0)
                    d = float(np.dot(v, v))
                    if d <= 0:
                        continue
                    lam = -float(np.dot(v, obs) / d)
                    rr = obs + lam * v
                    cosa = -float(np.dot(v, obs) / np.sqrt(d * np.dot(obs, obs)))
                    print(f"  shape[{n:>4}] best lambda {lam:+9.3f} -> rms "
                          f"{np.sqrt(np.mean(rr ** 2)):.5f} "
                          f"({100 * (1 - np.sqrt(np.mean(rr ** 2)) / np.sqrt(np.mean(obs ** 2))):+6.1f} %)"
                          f"   cos(-d, obs) {cosa:+.4f}")
            print()

            if args.gauge:
                print("  GAUGE -- the SAME transform, channel and weights, for "
                      "a KNOWN relative\n  change eps of the block's ionization "
                      "variance.  Establishes that the\n  pipeline responds and "
                      "that the numbers above are measurements.")
                print(f"  {'eps':>9} " + "".join(f"{u:>10.3g}" for u in probes))
                for eps in args.gauge:
                    gr = []
                    for k in range(len(legs)):
                        s_ = float(sc["sF"][k])
                        tau = fn.closure_tau(float(probes.max()), n=args.taun)
                        phiA = cp.model_phi(legs, k, avec, s_, tau)
                        phiG = us._gauge_phi(legs, k, avec, s_, tau, eps)
                        gr.append([fn.weier_scalar(phiA, u, tau)
                                   - fn.weier_scalar(phiG, u, tau) for u in probes])
                    print(f"  {eps:9.1e} " + "".join(
                        f"{v:+10.2e}" for v in np.array(gr).mean(axis=0)))
                print()


# =========================================================================
# subcommand: real -- the ACTUAL closure, not the linearized shift
# =========================================================================
#
# The propagation above is a linearization: it multiplies the published model
# CF by exp(dS) and holds s_F fixed.  That technique has only ever been used
# in this study to produce small numbers that were then dismissed; producing a
# LARGE number with it demands that it be checked end to end.  `real` does the
# check by running `deltaspec._closure_rows` -- the same function that produced
# every published closure table -- against the SAME simulation files, with the
# model's own CF carrying the Kokoulin term.  s_F is recomputed, so the data z
# move too, and nothing is linearized.
#
# TWO TRAPS, both real and both checked rather than avoided:
#  1. `fisher_norm.plane_scales` caches s_F BOTH in-process and ON DISK, keyed
#     by the model's CONTENT identity -- which does not change when a module
#     global changes.  Without busting both caches the switched-ON run silently
#     reuses the switched-OFF s_F.  Both are bypassed here and the resulting
#     s_F is PRINTED, so a switch that failed to reach the scale computation is
#     visible instead of silent.
#  2. `fisher_norm.pmap` forks, so a module global set in the parent does
#     propagate -- but only because it forks.  Verified by the s_F change.

MODEL_TCUT = {"pt3_cut1e4": 0.0, "pt40_cut001": 0.29638, "pt3_base": 17.8507,
              "pt40_base": 17.8507, "real": 0.0}
# which SIMULATION each config's closure is run against (the real tracker's
# published rows are the ARCHIVED 400k pT = 3 sample)
SIMTAG = {"real": "ARCHIVED"}


def _run_real_closure(modelpath, tag, func, kok, tcut):
    import importlib
    ctr.IONI_KOKOULIN = kok
    ctr.IONI_KOKOULIN_TCUT = tcut
    # bust both layers of the s_F cache -- see the trap note above
    fn._SCALE_CACHE.clear()
    old_load, old_store = fn._scale_cache_load, fn._scale_cache_store
    fn._scale_cache_load = lambda *a, **k: None
    fn._scale_cache_store = lambda *a, **k: None
    try:
        import tail_probe as tp
        import real_probe as rp
        stag = SIMTAG.get(tag, tag)
        if stag in tp.CONFIGS:
            sim = tp.sim_of(stag)
        else:
            sim = rp.sim_of(rp.ARCHIVED_SIM if stag == "ARCHIVED"
                            else rp.sim_glob(stag))
        (rows, errs, ks, ns, err), sc = ds._closure_rows(modelpath, sim, func)
    finally:
        fn._scale_cache_load, fn._scale_cache_store = old_load, old_store
        ctr.IONI_KOKOULIN = 0.0
        ctr.IONI_KOKOULIN_TCUT = 0.0
        fn._SCALE_CACHE.clear()
    return rows, err, ks, ns, sc


def cmd_real(args):
    print("=" * 100)
    print("THE ACTUAL CLOSURE, WITH GEANT4'S RADIATIVE CORRECTION IN THE MODEL")
    print("=" * 100)
    print("Same simulation files, same reference trajectory, same "
          "`_closure_rows` that produced\nevery published table.  ONLY the "
          "model's ionization CF changes, and s_F is recomputed\nfrom it -- "
          "nothing is linearized and nothing is fitted.  `kok 0.0` must "
          "reproduce the\npublished row digit for digit; that is the "
          "switch-is-inert control.\n")
    for tag in args.configs:
        mp_ = CONFIGS[tag][0]
        tcut = args.tcut if args.tcut is not None else MODEL_TCUT.get(tag, 0.0)
        for func in args.funcs:
            print(f"### {tag}  {func}   model {os.path.basename(mp_)}   "
                  f"Kokoulin from max(tcut, 100 keV), tcut = {tcut:g} MeV")
            print(f"  {'kok':<8} " + "".join(f"{u:>10.3g}" for u in fn.UCURVE)
                  + f"{'rms':>10}")
            base = None
            for kok in args.kok:
                rows, err, ks, ns, sc = _run_real_closure(mp_, tag, func,
                                                          kok, tcut)
                if args.kmax is not None:
                    sel = [i for i, k in enumerate(ks) if k <= args.kmax]
                    rows, ks = rows[sel], [ks[i] for i in sel]
                m = rows.mean(axis=0)
                print(f"  {kok:<8.3g} " + "".join(f"{v:+10.5f}" for v in m)
                      + f"{np.sqrt(np.mean(m ** 2)):10.5f}"
                      + f"   [{len(ks)} pl, {int(np.median(ns))} ev]")
                if base is None:
                    base = m
                    print(f"  {'   +-':<8} " + "".join(f"{v:10.5f}" for v in err))
                else:
                    d = m - base
                    print(f"  {'   d':<8} " + "".join(f"{v:+10.5f}" for v in d)
                          + "     <- the ACTUAL move, against the linearized "
                            "prediction below")
                print(f"  {'   s_F':<8} {np.mean(sc['sF']):.8e}   "
                      f"1/I {np.mean(sc['invI']):.6f}"
                      + ("   <- must MOVE when the switch is on"
                         if kok else ""))
            print()


# =========================================================================
# subcommand: cutscan
# =========================================================================

def cmd_cutscan(args):
    print("=" * 100)
    print("CUT SCAN: the composed gap as a function of the e- production "
          "threshold")
    print("=" * 100)
    print("The gap lives entirely BELOW the cut, so this is the single "
          "variable it depends on.\nFor the real tracker the per-material cut "
          "map is not recoverable from the export\n(the record carries Zeff "
          "but not rho), so the scan BOUNDS it.\n")
    probes = fn.UCURVE
    for tag in args.configs:
        legs, imap, _, _ = load_cfg(tag)
        avec = cp.FUNCTIONALS[args.func]
        sc = fn.plane_scales(legs, args.func, tag=CONFIGS[tag][0],
                             channels=("ioni", "ms", "rad"))
        print(f"### {tag}  {args.func}")
        print(f"  {'tcut[MeV]':>10} " + "".join(f"{u:>10.3g}" for u in probes)
              + f"{'rms':>10}")
        obs = OBSERVED.get(tag)
        for tc in args.cuts:
            def cuts_fn(st, tc=tc):
                return np.full(len(st), tc)
            rows = {n: [] for n in ("gap", "gsub", "kok")}
            for k in range(len(legs)):
                s = float(sc["sF"][k])
                tau = fn.closure_tau(float(probes.max()), n=args.taun)
                pA, pm = _phis(legs, k, avec, s, tau, cuts_fn, imap,
                               args.scalmode)
                base = np.array([fn.weier_scalar(pA, u, tau) for u in probes])
                for n in rows:
                    rows[n].append(base - np.array(
                        [fn.weier_scalar(pm[n], u, tau) for u in probes]))
            for n in ("gsub", "kok", "gap") if args.parts else ("gap",):
                m = np.array(rows[n]).mean(axis=0)
                extra = ""
                if obs is not None and np.dot(m, m) > 0:
                    lam = -float(np.dot(m, obs) / np.dot(m, m))
                    cosa = -float(np.dot(m, obs)
                                 / np.sqrt(np.dot(m, m) * np.dot(obs, obs)))
                    extra = f"  lam {lam:+8.3f} cos {cosa:+.3f}"
                lab = f"{tc:10.4g}" if n == "gap" else f"{'  ' + n:>10}"
                print(f"  {lab} " + "".join(f"{v:+10.2e}" for v in m)
                      + f"{np.sqrt(np.mean(m ** 2)):10.5f}" + extra)
        print()


# =========================================================================

def _kv(s):
    """`name:Z,A,rho,ekin,len,tcut`"""
    n, v = s.split(":")
    p = [float(x) for x in v.split(",")]
    return n, tuple(p)


DEFAULT_STEPS = {
    # the layered toy's own steps, at the pt3_cut1e4 production cuts.
    # thick: a full ToyLayerMat plane crossing.  thin: the 0.01 mm remainder.
    "thick_pt3": (8.0, 16.0, 9.0, 3032.2, 1.045, 0.00948618),
    "thin_pt3": (8.0, 16.0, 9.0, 3032.2, 0.01, 0.00948618),
    "thick_pt40": (8.0, 16.0, 9.0, 40000.0, 1.045, 0.29638),
}


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("steps")
    q.add_argument("--configs", nargs="+", default=["pt3_cut1e4"])
    q.add_argument("--cut", type=float, default=None)
    q.add_argument("--scalmode", default="tcut")
    q.add_argument("--nshow", type=int, default=8)
    q.add_argument("--all", action="store_true")
    q.set_defaults(func=cmd_steps)

    q = sub.add_parser("g4")
    q.add_argument("--steps", type=lambda s: dict([_kv(s)]), default=None)
    q.add_argument("--only", nargs="*", default=None)
    q.add_argument("-n", type=int, default=4000000)
    q.add_argument("--seed", type=int, default=987)
    q.add_argument("--nt", type=int, default=400)
    q.add_argument("--tmaxcf", type=float, default=8.0)
    q.set_defaults(func=cmd_g4)

    q = sub.add_parser("closure")
    q.add_argument("--configs", nargs="+", default=["pt3_cut1e4", "pt40_cut001"])
    q.add_argument("--funcs", nargs="+", default=["qop"])
    q.add_argument("--cut", type=float, default=None)
    q.add_argument("--scalmode", default="tcut")
    q.add_argument("--taun", type=int, default=3000)
    q.add_argument("--gauge", nargs="*", type=float, default=[])
    q.add_argument("--perplane", action="store_true")
    q.add_argument("--nbin", type=int, default=96)
    q.set_defaults(func=cmd_closure)

    q = sub.add_parser("real")
    q.add_argument("--configs", nargs="+", default=["pt3_cut1e4", "pt40_cut001"])
    q.add_argument("--funcs", nargs="+", default=["qop"])
    q.add_argument("--kok", nargs="+", type=float, default=[0.0, 1.0])
    q.add_argument("--tcut", type=float, default=None)
    q.add_argument("--kmax", type=int, default=None)
    q.set_defaults(func=cmd_real)

    q = sub.add_parser("cutscan")
    q.add_argument("--configs", nargs="+", default=["real"])
    q.add_argument("--func", default="qop")
    q.add_argument("--cuts", nargs="+", type=float,
                   default=[0.00099, 0.01, 0.05, 0.1, 0.3, 1.0, 3.0])
    q.add_argument("--scalmode", default="tcut")
    q.add_argument("--taun", type=int, default=3000)
    q.add_argument("--parts", action="store_true")
    q.set_defaults(func=cmd_cutscan)

    a = p.parse_args()
    if a.cmd == "g4":
        st = dict(DEFAULT_STEPS) if a.steps is None else a.steps
        if a.only:
            st = {k: v for k, v in st.items() if k in a.only}
        a.steps = st
    a.func(a)


if __name__ == "__main__":
    main()
