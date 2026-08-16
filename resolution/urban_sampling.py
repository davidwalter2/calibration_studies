#!/usr/bin/env python3
"""Does Geant4's Urban sampler draw from the analytic compound-Poisson model?

WHY
---
Every ionization number in this study -- the block CGF, the saddlepoint mode,
the Fisher scale s_F, the clean-propagation closure -- is built from the
per-step Urban record

    ioniurbanv = (regime, gsig2, a1, e1, a2, e2, a3, e0, tmax, scaling, cs)

exported by `G4UniversalFluctuationForExtrapolator::SampleFluctuations`, and
from ONE assumption about it: that the step's energy loss is the exact compound
Poisson

    Poisson(a1) jumps of size e1  +  Poisson(a2) jumps of size e2
    +  Poisson(a3) jumps drawn from 1/E^2 on [e0, tmax],   all x scaling.

That CF has been validated NUMERICALLY to 1e-14 against mpmath (NOTES 2026-08-13
XIII-XXI).  That tests the mathematics of the model.  It does not test whether
the model is what Geant4 draws from -- and Geant4's sampler
(`SampleFluctuations2`, the same class, the same parameter block) is NOT that
compound Poisson.  Reading the C++ it differs in four places:

  1. an excitation channel with a <= nmaxCont = 8 is sampled as
     `((p+1) - 2*U) * e` with p ~ Poisson(a), U ~ Uniform(0,1), i.e. the
     Poisson is smeared by a uniform of full width 2e -- EXTRA variance
     e^2 (1 - e^{-a}) / 3;
  2. an excitation channel with a > 8 is replaced by a Gaussian of the same
     mean and variance, truncated by rejection to [0, 2*mean] (both channels
     pooled into ONE Gaussian);
  3. the delta-ray channel with a3 > 8 is SPLIT at w2 = alfa*e0: the collisions
     below w2 (mean number `namean`) are replaced by a Gaussian whose variance
     is `e0^2 namean (alfa - alfa1^2)` = lambda*Var(E), NOT lambda*E[E^2] --
     i.e. Geant4 drops the Poisson number fluctuation of the low-energy delta
     rays.  MISSING variance e0^2 namean alfa1^2.  Only the collisions above
     w2 -- exactly p3 = 8 a3/(8 + a3) of them -- keep their Poisson/1E^2 law;
  4. in regime 0 the sampled loss is a Gaussian truncated to [0, 2*meanLoss]
     (thick target) or a Gamma (sn < 2), while the record carries only the
     variance, so the analytic side cannot even see the truncation point.

None of these changes the MEAN (each replacement is mean-preserving, and the
Gaussians are truncated symmetrically), so they are invisible in every
mean/scale diagnostic and start at the second cumulant.

WHAT THIS MODULE DOES
---------------------
It compares the two distributions directly, in isolation from transport,
geometry, acceptance and the propagator, through a three-level chain in which
every level is checked against the one below it:

    real Geant4 C++  <->  Python transcription  <->  exact sampler CF
      (urban_g4driver)      (`emulate_step`)        (`sampler_exponent`)

and then propagates the per-step CF ratio through the SAME transport weights
the closure uses, so the question "can this account for the layered-toy q/p
non-closure, magnitude AND sign change" gets a number rather than an argument.

The exact sampler CF is possible because every random draw in the sampler is
either an exact Poisson (`G4Poisson` is exact for mean <= 16, and BOTH of its
call sites here have mean <= 8: the excitation branch by its `a <= nmaxCont`
guard, the delta branch because p3 = 8 a3/(8 + a3) < 8 identically), an exact
1/E^2 inverse-CDF draw, a uniform, or a symmetrically truncated Gaussian.

SUBCOMMANDS
    records   the per-step parameter space of the layered toy, and its weights
    g4check   the validation chain against the real C++ sampler
    perstep   analytic vs sampler cumulants and CF, per real step
    closure   propagate to the layered-toy closure: d<e^{-u z^2}> per probe
    scan      where in (a1, a3, regime) parameter space the two diverge

Nothing in cf_propagation_test.py / cf_track_resolution.py / cgf_*.py is
modified; they are imported.  No file in the CMSSW source area is written to
(the C++ driver links against the already-built library).
"""

import argparse
import os
import subprocess
import sys

import numpy as np
from scipy.special import wofz, erf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cf_propagation_test as cp                                 # noqa: E402
import fisher_norm as fn                                         # noqa: E402
from cf_track_resolution import _delta_term_2d, ioni_step_exponent  # noqa: E402

SCRATCH = fn.SCRATCH
TP = os.path.join(SCRATCH, "tp")
DRIVER = os.path.join(SCRATCH, "urban_g4driver.sh")

# Urban model constants, read off G4UniversalFluctuationForExtrapolator
NMAXCONT = 8.0
A0 = 42.0
FW = 4.0
RATE = 0.56

# record column names, for readability
R_REG, R_GSIG2, R_A1, R_E1, R_A2, R_E2, R_A3, R_E0, R_TMAX, R_SCAL, R_CS = range(11)

# |mu/sigma| above which a symmetric [mu-c sig, mu+c sig] truncation is dropped.
# The truncation moves at most the tail mass, 2(1-Phi(c)) = 1.2e-15 at c = 8,
# so beyond this the truncated and untruncated CFs agree to double precision.
CTRUNC_MAX = 8.0


# ==========================================================================
# the Urban delta-ray split, exactly as the C++ does it
# ==========================================================================

def urban_split(a3, e0, tmax):
    """(alfa, alfa1, namean, p3, w2, emean, sig2e) of the a3 > nmaxCont split.

    Transcribed from SampleFluctuations2.  For a3 <= nmaxCont the C++ leaves
    alfa = 1 and p3 = a3, which this returns too (emean = sig2e = 0), so the
    caller needs no branch.

    IDENTITY (verified in `scan`): p3 = a3 - namean = nmaxCont a3/(nmaxCont +
    a3) exactly, so the Poisson part of the delta channel never has mean above
    nmaxCont = 8 and `G4Poisson` is therefore always in its exact branch.
    """
    a3 = np.asarray(a3, dtype=np.float64)
    e0 = np.asarray(e0, dtype=np.float64)
    tmax = np.asarray(tmax, dtype=np.float64)
    w1 = tmax / e0
    alfa = np.ones_like(a3)
    alfa1 = np.zeros_like(a3)
    namean = np.zeros_like(a3)
    p3 = a3.copy()
    big = a3 > NMAXCONT
    if np.any(big):
        al = w1[big] * (NMAXCONT + a3[big]) / (w1[big] * NMAXCONT + a3[big])
        al1 = al * np.log(al) / (al - 1.0)
        nm = a3[big] * w1[big] * (al - 1.0) / ((w1[big] - 1.0) * al)
        alfa[big], alfa1[big], namean[big] = al, al1, nm
        p3[big] = a3[big] - nm
    w2 = alfa * e0
    emean = namean * e0 * alfa1
    sig2e = e0 * e0 * namean * (alfa - alfa1 * alfa1)
    return alfa, alfa1, namean, p3, w2, emean, sig2e


# ==========================================================================
# level 2: a Python transcription of SampleFluctuations2's Glandz body
# ==========================================================================

def emulate_step(rec, n, rng):
    """`n` samples of the Glandz-regime loss [MeV] for one record.

    Line-for-line transcription of G4UniversalFluctuationForExtrapolator::
    SampleFluctuations2 from the point where a1, a2, a3 are known -- which is
    exactly the information the record carries -- including AddExcitation2's
    uniform smear, the pooling of the two excitation Gaussians into a single
    SampleGauss2 call, the [0, 2 mean] rejection, and the final x scaling.

    Validated against the real C++ in `g4check`.
    """
    a1, e1 = rec[R_A1], rec[R_E1]
    a2, e2 = rec[R_A2], rec[R_E2]
    a3, e0, tmax, gam = rec[R_A3], rec[R_E0], rec[R_TMAX], rec[R_SCAL]
    loss = np.zeros(n)

    # --- excitations (AddExcitation2), both channels, ONE pooled Gaussian
    sig2e = 0.0
    eav = 0.0
    for a, e in ((a1, e1), (a2, e2)):
        if a <= 0.0:
            continue
        if a > NMAXCONT:
            eav += a * e
            sig2e += a * e * e
        else:
            p = rng.poisson(a, n)
            m = p > 0
            loss[m] += ((p[m] + 1) - 2.0 * rng.random(m.sum())) * e
    if sig2e > 0.0:
        loss += _sample_gauss2(eav, sig2e, n, rng)

    # --- delta rays
    if a3 > 0.0:
        alfa, alfa1, namean, p3, w2, gmean, gsig2 = [
            np.asarray(v).item() for v in urban_split(
                np.array([a3]), np.array([e0]), np.array([tmax]))]
        if tmax > w2:
            w = (tmax - w2) / tmax
            nnb = rng.poisson(p3, n)
            tot = int(nnb.sum())
            if tot:
                u = rng.random(tot)
                e = w2 / (1.0 - w * u)
                idx = np.repeat(np.arange(n), nnb)
                loss += np.bincount(idx, weights=e, minlength=n)
        if gsig2 > 0.0:
            loss += _sample_gauss2(gmean, gsig2, n, rng)

    return loss * gam


def _sample_gauss2(eav, esig2, n, rng):
    """SampleGauss2: uniform on [0, 2 eav] if eav < 0.25 sigma, else a Gaussian
    rejected outside [0, 2 eav]."""
    sig = np.sqrt(esig2)
    if eav < 0.25 * sig:
        return eav * 2.0 * rng.random(n)
    out = rng.normal(eav, sig, n)
    bad = (out < 0.0) | (out > 2.0 * eav)
    while bad.any():
        out[bad] = rng.normal(eav, sig, int(bad.sum()))
        bad = (out < 0.0) | (out > 2.0 * eav)
    return out


# ==========================================================================
# level 3: the EXACT characteristic function of what the sampler draws
# ==========================================================================

def _tn_logcf(mu, sig, t):
    """log CF of N(mu, sig^2) truncated to [0, 2 mu], CENTERED on mu.

    The truncation is symmetric about the mean, so the centered law is
    symmetric and its CF is REAL:

        phi(t) = [e^{-sig^2 t^2/2} - Re(e^{-c^2/2 + i c sig t}
                                        w((sig t + i c)/sqrt 2))] / erf(c/sqrt2)

    with c = mu/sig and w = Faddeeva.  Written this way nothing overflows: the
    Faddeeva argument stays in the upper half plane where w is bounded, and the
    e^{sig^2 t^2/2} of erfc's exponential prefactor has been cancelled
    analytically against the Gaussian factor.  Above c = CTRUNC_MAX the
    truncation is below double precision and the plain Gaussian is returned.
    """
    t = np.asarray(t, dtype=np.float64)
    if sig <= 0.0:
        return np.zeros_like(t, dtype=np.complex128)
    c = mu / sig
    base = -0.5 * sig ** 2 * t ** 2
    if c >= CTRUNC_MAX:
        return base.astype(np.complex128)
    st = sig * t
    z = (st + 1j * c) / np.sqrt(2.0)
    corr = np.real(np.exp(-0.5 * c ** 2 + 1j * c * st) * wofz(z))
    num = np.exp(base) - corr
    den = erf(c / np.sqrt(2.0))
    return np.log(np.asarray(num / den, dtype=np.complex128))


def _unif_logcf(mu, t):
    """log CF of Uniform(0, 2 mu) centered on mu:  sinc(mu t)."""
    x = mu * np.asarray(t, dtype=np.float64)
    return np.log(np.asarray(np.sinc(x / np.pi), dtype=np.complex128))


def _exc_discrete_logcf(a, e, t):
    """log CF, centered, of AddExcitation2's a <= nmaxCont branch:
    ((p+1) - 2U) e with p ~ Poisson(a), and nothing added when p = 0.

        phi(th) = e^{-a} + sinc(th) (e^{a(e^{i th} - 1)} - e^{-a}),  th = e t

    (the U integral gives e^{-i th} sin(th)/th, which cancels the +1 in p+1).
    Centering subtracts the mean a e, which is the SAME mean the analytic
    compound Poisson has -- the uniform smear is mean preserving.
    """
    th = e * np.asarray(t, dtype=np.float64)
    ea = np.exp(-a)
    phi = ea + np.sinc(th / np.pi) * (np.exp(a * (np.exp(1j * th) - 1.0)) - ea)
    return np.log(phi) - 1j * a * th


def sampler_exponent(steps, wstd, tau):
    """Centered log-CF exponent of what Geant4 SAMPLES, in standardized-z
    units.  Signature and conventions identical to
    `cf_track_resolution.ioni_step_exponent`, so the two can be differenced
    step by step.

    Regime 0 is NOT representable from the record alone (see module docstring
    point 4): the record carries the variance but not the mean loss, hence not
    the [0, 2 meanLoss] truncation point nor the Gamma shape.  Regime-0 steps
    are therefore returned with the analytic Gaussian, and `records` reports
    how many there are (none, in the layered toy).
    """
    tau = np.asarray(tau, dtype=np.float64)
    reg = steps[:, R_REG]
    g = steps[:, R_CS].astype(np.float64) * 1e-3
    gs = wstd * g
    S = np.zeros(len(tau), dtype=np.complex128)

    mg = reg == 0
    if mg.any():
        S += -0.5 * tau ** 2 * float(np.sum(steps[mg, R_GSIG2] * gs[mg] ** 2))

    mu = ~mg
    if not mu.any():
        return S
    st = steps[mu]
    gsu = gs[mu]
    gam = st[:, R_SCAL]

    for i in range(len(st)):
        t = gsu[i] * tau                      # conjugate variable per MeV
        gm = gam[i]
        # --- excitations: a <= 8 discrete-with-smear, a > 8 pooled Gaussian
        eav, esig2 = 0.0, 0.0
        for ja, je in ((R_A1, R_E1), (R_A2, R_E2)):
            a, e = float(st[i, ja]), float(st[i, je]) * gm
            if a <= 0.0 or e <= 0.0:
                continue
            if a > NMAXCONT:
                eav += a * e
                esig2 += a * e * e
            else:
                S += _exc_discrete_logcf(a, e, t)
        if esig2 > 0.0:
            S += (_unif_logcf(eav, t) if eav < 0.25 * np.sqrt(esig2)
                  else _tn_logcf(eav, np.sqrt(esig2), t))
        # --- delta rays
        a3 = float(st[i, R_A3])
        e0 = float(st[i, R_E0]) * gm
        tmax = float(st[i, R_TMAX]) * gm
        if a3 <= 0.0 or e0 <= 0.0 or tmax <= e0:
            continue
        alfa, alfa1, namean, p3, w2, gmean, gsig2 = [
            np.asarray(v).item() for v in urban_split(
                np.array([a3]), np.array([e0]), np.array([tmax]))]
        if tmax > w2:
            S += p3 * _delta_term_2d((w2 * gsu[i] * tau)[None, :],
                                     np.array([tmax / w2]))[0]
        if gsig2 > 0.0:
            S += (_unif_logcf(gmean, t) if gmean < 0.25 * np.sqrt(gsig2)
                  else _tn_logcf(gmean, np.sqrt(gsig2), t))
    return S


# ==========================================================================
# analytic and sampled cumulants (closed form, for the moment comparison)
# ==========================================================================

def step_cumulants(rec, which):
    """(kappa1, kappa2, kappa3, kappa4) of ONE step's loss [MeV^k], for
    which = 'analytic' (the compound Poisson) or 'sampler' (what Geant4 draws).

    Exact in both cases:
      compound Poisson         kappa_m = sum_j a_j x_j^m  (+ the 1/E^2 moment)
      truncated Gaussian       central moments of a symmetric truncation
      Poisson + uniform smear  from the mgf of ((p+1)-2U)e 1{p>0}
    """
    a1, e1 = rec[R_A1], rec[R_E1] * rec[R_SCAL]
    a2, e2 = rec[R_A2], rec[R_E2] * rec[R_SCAL]
    a3 = rec[R_A3]
    e0, tmax = rec[R_E0] * rec[R_SCAL], rec[R_TMAX] * rec[R_SCAL]
    k = np.zeros(4)

    def delta_cum(lam, lo, hi):
        """cumulants of a compound Poisson with rate lam on the 1/E^2 spectrum
        over [lo, hi] normalized to unit total probability."""
        A = 1.0 / (1.0 / lo - 1.0 / hi)
        return np.array([lam * A * np.log(hi / lo),
                         lam * A * (hi - lo),
                         lam * A * (hi ** 2 - lo ** 2) / 2.0,
                         lam * A * (hi ** 3 - lo ** 3) / 3.0])

    if which == "analytic":
        for a, e in ((a1, e1), (a2, e2)):
            if a > 0 and e > 0:
                k += a * np.array([e, e ** 2, e ** 3, e ** 4])
        if a3 > 0 and tmax > e0:
            k += delta_cum(a3, e0, tmax)
        return k

    # --- sampler
    eav, esig2 = 0.0, 0.0
    for a, e in ((a1, e1), (a2, e2)):
        if a <= 0 or e <= 0:
            continue
        if a > NMAXCONT:
            eav += a * e
            esig2 += a * e * e
        else:
            # ((p+1)-2U)e 1{p>0}: mean a e, var e^2 (a + (1-e^{-a})/3)
            k += _smear_cumulants(a, e)
    if esig2 > 0:
        k += _trunc_gauss_cumulants(eav, np.sqrt(esig2))
    if a3 > 0 and tmax > e0:
        _, _, _, p3, w2, gmean, gsig2 = [
            np.asarray(v).item() for v in urban_split(
                np.array([a3]), np.array([e0]), np.array([tmax]))]
        if tmax > w2:
            k += delta_cum(p3, w2, tmax)
        if gsig2 > 0:
            k += _trunc_gauss_cumulants(gmean, np.sqrt(gsig2))
    return k


def _smear_cumulants(a, e, nmax=200):
    """First four cumulants of X = ((p+1) - 2U) e 1{p>0}, p ~ Poisson(a)."""
    p = np.arange(nmax + 1)
    from scipy.stats import poisson
    w = poisson.pmf(p, a)
    # raw moments of (A - 2U) with A = p+1: E[(A-2U)^m] = (A^{m+1}-(A-2)^{m+1})/(2(m+1))
    m = np.zeros(5)
    m[0] = 1.0
    for j in range(1, 5):
        val = (p + 1.0) ** (j + 1) - (p - 1.0) ** (j + 1)
        val = val / (2.0 * (j + 1))
        val[0] = 0.0                       # p = 0 contributes nothing at all
        m[j] = float(np.sum(w * val)) * e ** j
    k1 = m[1]
    k2 = m[2] - m[1] ** 2
    k3 = m[3] - 3 * m[2] * m[1] + 2 * m[1] ** 3
    k4 = m[4] - 4 * m[3] * m[1] - 3 * m[2] ** 2 + 12 * m[2] * m[1] ** 2 - 6 * m[1] ** 4
    return np.array([k1, k2, k3, k4])


def _trunc_gauss_cumulants(mu, sig):
    """Cumulants of N(mu, sig^2) truncated to [mu - c sig, mu + c sig]."""
    c = mu / sig
    if c >= CTRUNC_MAX:
        return np.array([mu, sig ** 2, 0.0, 0.0])
    phi = np.exp(-0.5 * c ** 2) / np.sqrt(2 * np.pi)
    Z = erf(c / np.sqrt(2.0))
    # central moments of the symmetric truncation (odd ones vanish)
    m2 = sig ** 2 * (1.0 - 2.0 * c * phi / Z)
    m4 = sig ** 4 * (3.0 - 2.0 * (c ** 3 + 3.0 * c) * phi / Z)
    return np.array([mu, m2, 0.0, m4 - 3.0 * m2 ** 2])


# ==========================================================================
# transport weights, exactly as model_phi folds them
# ==========================================================================

def weighted_steps(legs, k, avec, sigma):
    """The plane-k ionization block: every step of legs 0..k with its transport
    weight folded into column 10, i.e. what model_phi feeds the exponent.

    Identical to cgf_channels.collect_ioni_steps; repeated here so this module
    does not depend on that function's default arguments.
    """
    _, A_ioni, _ = cp.step_transports(legs, k)
    out = []
    for j in range(k + 1):
        leg = legs[j]
        if not len(leg["ioni"]):
            continue
        q = np.sign(leg["refqop"]) or 1.0
        w = q * np.einsum("i,sij->sj", avec, A_ioni[j])[:, 0] / sigma
        st = leg["ioni"].copy().astype(np.float64)
        st[:, R_CS] *= w
        out.append(st)
    return np.concatenate(out) if out else np.zeros((0, 11))


# ==========================================================================
# the OTHER Urban model: stock Geant4 11.2.2, i.e. what the SIM samples
# ==========================================================================
#
# G4UniversalFluctuationForExtrapolator implements the PRE-2021 Urban model:
# two excitation channels at e1Fluct ~ ipot-derived and e2Fluct = 10 Zeff^2 eV
# with weights f1 = 1 - 2/Zeff, f2 = 2/Zeff.  Stock Geant4 11.2.2's
# G4UniversalFluctuation is the 2021 Urban model: ONE excitation channel at
# ipotFluct, and a different fwnow floor (0.1 instead of 0.5).  The header of
# the stock class has no f1Fluct/f2Fluct/e1Fluct/e2Fluct members at all, which
# is how the two are told apart without the .cc.
#
# The full CMSSW SIM -- the "data" side of the clean-propagation closure -- runs
# the stock class.  So the analytic model is compared with the extrapolator's
# sampler in `g4check` (the tasked question) and with the SIM's sampler here.

def material_from_record(rec):
    """(ipot, e1F, e2F, f1, f2, Zeff) [MeV] recovered from one record.

    e2 is never modified by the sampler, so e2F = e2 and Zeff = sqrt(e2/10 eV).
    e1 is modified: `a1 < a0 ? (fwnow = 0.5 + (fw-0.5) sqrt(a1/a0); a1 /= fwnow;
    e1 *= fwnow) : (a1 /= fw; e1 *= fw)`, with the branch taken on the
    PRE-division a1.  The two branches meet at a1_recorded = a0/fw = 10.5, and
    in the lower branch fwnow solves fwnow = 0.5 + (fw-0.5) sqrt(a1_rec fwnow/a0),
    a quadratic in sqrt(fwnow).  Validated against the driver's own
    material printout in `stock`.
    """
    e2F = float(rec[R_E2])
    Zeff = np.sqrt(e2F / 1e-5)                    # e2F = 10 Zeff^2 eV
    f2 = 2.0 / Zeff if Zeff > 2.0 else 0.0
    f1 = 1.0 - f2
    a1r, e1r = float(rec[R_A1]), float(rec[R_E1])
    if a1r >= A0 / FW:
        fwnow = FW
    else:
        # (fwnow - 0.5)^2 = (fw-0.5)^2 a1r fwnow / a0  ->  quadratic in fwnow
        b = (FW - 0.5) ** 2 * a1r / A0
        fwnow = 0.5 * (2.0 + b + np.sqrt(b * (b + 4.0)))   # root > 0.5
    e1F = e1r / fwnow
    ipot = np.exp(f1 * np.log(e1F) + f2 * np.log(e2F))
    return ipot, e1F, e2F, f1, f2, Zeff


def stock_record(meanloss, ipot, e0, tcut, cs=1e3):
    """The stock (Urban 2021) SampleGlandz parameter block, in the same
    11-column record layout, so the same CF machinery applies to it.

    a1 = meanLoss (1 - rate)/ipot with the 2021 fwnow floor 0.1, one channel
    only (a2 = e2 = 0); a3 identical in form to the pre-2021 model.  Validated
    against real stock samples in `stock`.
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
    return np.array([1.0, 0.0, a1, e1, 0.0, 0.0, a3, e0, tcut, 1.0, cs])


# ==========================================================================
# subcommand: records
# ==========================================================================

def _load(tag):
    """A toy tag (`pt3_K1`) or a bare model path/filename in the scratchpad."""
    if "/" in tag or tag.endswith(".root"):
        return cp.load_model(tag if "/" in tag else os.path.join(SCRATCH, tag))
    return cp.load_model(os.path.join(TP, f"{tag}_model.root"))


def cmd_records(args):
    print("=" * 78)
    print("URBAN PER-STEP PARAMETER SPACE OF THE LAYERED TOY")
    print("=" * 78)
    for tag in args.configs:
        legs = _load(tag)
        st = np.concatenate([l["ioni"] for l in legs if len(l["ioni"])])
        avec = cp.FUNCTIONALS["qop"]
        klast = len(legs) - 1
        sig = float(np.sqrt(cp.model_variance(legs, klast, avec)[0]))
        wst = weighted_steps(legs, klast, avec, sig)
        gs = wst[:, R_CS] * 1e-3
        # each step's own contribution to the block variance, in z^2 units
        kA = np.array([step_cumulants(r, "analytic") for r in st])
        kS = np.array([step_cumulants(r, "sampler") for r in st])
        vz = kA[:, 1] * gs ** 2
        print(f"\n--- {tag}:  {len(legs)} legs, {len(st)} ionization steps, "
              f"sigma(plane {klast}) = {sig:.4e}")
        print(f"    regimes: " + ", ".join(
            f"{int(r)}:{int(n)}" for r, n in zip(*np.unique(st[:, R_REG],
                                                            return_counts=True))))
        print(f"    a1 > 8: {int(np.sum(st[:, R_A1] > 8))}/{len(st)}   "
              f"a2 > 8: {int(np.sum(st[:, R_A2] > 8))}/{len(st)}   "
              f"a3 > 8: {int(np.sum(st[:, R_A3] > 8))}/{len(st)}")
        _, _, namean, p3, w2, gmean, gsg = urban_split(
            st[:, R_A3], st[:, R_E0], st[:, R_TMAX])
        big = st[:, R_A3] > NMAXCONT
        idn = np.max(np.abs(p3[big] - NMAXCONT * st[big, R_A3]
                            / (NMAXCONT + st[big, R_A3]))) if big.any() else 0.0
        print(f"    p3 = 8 a3/(8+a3) for a3 > 8 (max |dev| = {idn:.3e}); "
              f"max p3 over ALL steps = {p3.max():.4f} <= nmaxCont = 8, so\n"
              f"    every G4Poisson call in the sampler is in its EXACT branch "
              f"(mean <= 16)")
        print()
        print(f"    {'#':>3} {'a1':>10} {'a2':>10} {'a3':>10} {'p3':>7} "
              f"{'w2[keV]':>9} {'k2_ana':>11} {'k2_smp':>11} {'dk2/k2':>10} "
              f"{'k3_ana':>11} {'dk3/k3':>10} {'var share':>10}")
        order = np.argsort(-vz)
        for i in order:
            print(f"    {i:3d} {st[i, R_A1]:10.4g} {st[i, R_A2]:10.4g} "
                  f"{st[i, R_A3]:10.4g} {p3[i]:7.4f} {1e3 * w2[i]:9.4f} "
                  f"{kA[i, 1]:11.4e} {kS[i, 1]:11.4e} "
                  f"{(kS[i, 1] - kA[i, 1]) / kA[i, 1]:+10.3e} "
                  f"{kA[i, 2]:11.4e} "
                  f"{(kS[i, 2] - kA[i, 2]) / kA[i, 2]:+10.3e} "
                  f"{vz[i] / vz.sum():10.6f}")
        print(f"\n    block (plane {klast}) totals, z^2 units:")
        for m, nm in ((1, "kappa2"), (2, "kappa3"), (3, "kappa4")):
            A = float(np.sum(kA[:, m] * gs ** (m + 1)))
            S = float(np.sum(kS[:, m] * gs ** (m + 1)))
            print(f"      {nm}: analytic {A:+.6e}  sampler {S:+.6e}  "
                  f"rel diff {(S - A) / abs(A):+.3e}")


# ==========================================================================
# subcommand: g4check -- the validation chain
# ==========================================================================

def run_driver(Z, A, rho, ekin, length, tmax, n, seed, out, pdg=13,
               stock=False, meanloss=None, tcut=None):
    """Run the C++ driver; return (record dict, samples [MeV], stock samples)."""
    def f(x):
        # "%.17g", never repr(): numpy 2 renders a float64 as
        # "np.float64(1.045)", which atof() silently reads as 0.0 -- that
        # turned a step length into zero and a matched a1 into infinity.
        return "%.17g" % float(x)

    cmd = [DRIVER, "--Z", f(Z), "--A", f(A), "--rho", f(rho),
           "--ekin", f(ekin), "--len", f(length), "--tmax", f(tmax),
           "--n", str(int(n)), "--seed", str(int(seed)), "--out", out,
           "--pdg", str(int(pdg))]
    if stock:
        cmd += ["--stock", "--meanloss", f(meanloss)]
        if tcut is not None:
            cmd += ["--tcut", f(tcut)]
    env = dict(os.environ)
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"driver failed: {r.stdout}\n{r.stderr}")
    rec = {}
    for line in open(out + ".rec"):
        p = line.split()
        if not p or p[0].startswith("#"):
            continue
        if p[0] == "record":
            # column 10 is `cs`, the qop-per-MeV map, in units of 1e-3.
            # Setting it to 1e3 makes gs = cs*1e-3 = 1, i.e. the exponent
            # functions work directly in MeV^-1 conjugate units -- which is
            # what a single-step comparison wants. (Leaving it at 0, as an
            # earlier version did, silently returned CF == 1 everywhere.)
            rec["rec"] = np.array([float(x) for x in p[1:]] + [1e3])
        elif p[0] == "input":
            rec["input"] = [float(x) for x in p[1:9]]
        elif p[0] == "sample2":
            rec["mean"], rec["var"] = float(p[2]), float(p[4])
        elif p[0] == "stock":
            rec["stock_meanloss"] = float(p[2])
            rec["stock_mean"], rec["stock_var"] = float(p[4]), float(p[6])
        elif p[0] == "mationi":
            rec["ipot"] = float(p[2])
    s = np.fromfile(out + ".bin", dtype=np.float64)
    ss = (np.fromfile(out + ".stock.bin", dtype=np.float64)
          if stock and os.path.exists(out + ".stock.bin") else None)
    return rec, s, ss


def _ecf(x, t):
    """Empirical CF of a 1-D sample on a t grid (chunked to bound memory)."""
    out = np.zeros(len(t), dtype=np.complex128)
    for i in range(0, len(x), 200000):
        c = x[i:i + 200000]
        out += np.exp(1j * np.outer(t, c)).sum(axis=1)
    return out / len(x)


def _match_len(Z, A, rho, ekin, tmax, target_a1, seed=1, tol=1e-10):
    """Step length [mm] whose record reproduces `target_a1` (a1 is linear in
    the step's mean loss, so one secant iteration from a unit step suffices;
    iterated to `tol` relative anyway)."""
    L = 1.0
    for _ in range(30):
        rec, _, _ = run_driver(Z, A, rho, ekin, L, tmax, 8, seed,
                               os.path.join(SCRATCH, "_ulen"))
        a1 = float(rec["rec"][R_A1])
        if a1 <= 0.0:
            raise RuntimeError(f"driver returned a1 = 0 at len = {L}")
        if abs(a1 / target_a1 - 1.0) < tol:
            return L, rec["rec"]          # L is the one that MADE this record
        L = L * target_a1 / a1
    raise RuntimeError("step-length match did not converge")


def cmd_g4check(args):
    print("=" * 78)
    print("VALIDATION CHAIN: real Geant4 C++  <->  python emulator  <->  CF")
    print("=" * 78)
    print("Every point below is a SINGLE STEP. No transport, no geometry, no\n"
          "acceptance, no propagator: the driver calls\n"
          "  G4UniversalFluctuationForExtrapolator::SampleFluctuations  (record)\n"
          "  ...::SampleFluctuations2  N times                          (samples)\n"
          "at the SAME (material, particle, step), so the a1/a2/a3/e1/e2/e0/tmax\n"
          "recomputed inside the sampler are bit-identical to the recorded ones.\n")

    legs = _load(args.config)
    st = np.concatenate([l["ioni"] for l in legs if len(l["ioni"])])
    # the steps to reproduce: the largest-weight one, plus the extremes
    idx = args.steps if args.steps else [1, 0, 24]
    rng = np.random.default_rng(args.seed)

    for si in idx:
        tgt = st[si]
        ekin = _ekin_of_tmax(tgt[R_TMAX])
        L, got = _match_len(8.0, 16.0, 9.0, ekin, tgt[R_TMAX], tgt[R_A1])
        print(f"\n--- toy step {si}:  target a1 = {tgt[R_A1]:.6g}  "
              f"tmax = {tgt[R_TMAX]:.6g} MeV  ->  ekin = {ekin:.4f} MeV, "
              f"len = {L:.6f} mm")
        print("    record field   toy model        driver           rel")
        for nm, j in (("a1", R_A1), ("e1", R_E1), ("a2", R_A2), ("e2", R_E2),
                      ("a3", R_A3), ("e0", R_E0), ("tmax", R_TMAX),
                      ("scaling", R_SCAL), ("gsig2", R_GSIG2)):
            d = (got[j] - tgt[j]) / tgt[j] if tgt[j] else got[j] - tgt[j]
            print(f"    {nm:>12s}  {tgt[j]:14.8g}  {got[j]:14.8g}  {d:+10.2e}")

        out = os.path.join(SCRATCH, f"_u{si}")
        if args.jobs > 1:
            rec, x, _ = run_driver(8.0, 16.0, 9.0, ekin, L, tgt[R_TMAX],
                                   1000, args.seed, out)
        else:
            rec, x, _ = run_driver(8.0, 16.0, 9.0, ekin, L, tgt[R_TMAX],
                                   args.n, args.seed, out)
        r = rec["rec"]
        # the emulator is a cross-check on the C++, not the statistics driver:
        # cap it so a 1e8-sample C++ run does not also allocate 1e8 doubles here
        nemu = min(args.n, args.nemu)
        y = emulate_step(r, nemu, rng)

        kA = step_cumulants(r, "analytic")
        kS = step_cumulants(r, "sampler")
        print(f"\n    N = {len(x)} samples")
        print(f"    {'':<22s} {'mean':>14s} {'variance':>14s}")
        print(f"    {'G4 C++ sampler':<22s} {x.mean():14.8g} {x.var():14.8g}")
        print(f"    {'python emulator':<22s} {y.mean():14.8g} {y.var():14.8g}")
        print(f"    {'exact sampler cum.':<22s} {kS[0]:14.8g} {kS[1]:14.8g}")
        print(f"    {'exact analytic cum.':<22s} {kA[0]:14.8g} {kA[1]:14.8g}")
        print("    (the mean and variance of a 1/E^2 tail converge like 1/sqrt(N)\n"
              "     and 1/N^{1/4}: the CF below is the discriminating comparison)")

        # --- the CF comparison, on the scale of the step's own core.
        # NOT sqrt(kappa2): the 1/E^2 variance is tail dominated (39 MeV^2 for
        # a step whose distribution is 0.06 MeV wide), so standardizing by it
        # would put every probe in the far tail and hide the comparison. The
        # rule of this study is to match the grid to the thing being compared.
        core = _core_scale(r)
        sig = core
        t = np.linspace(0.0, args.tmaxcf / core, args.nt)
        cA = np.exp(ioni_step_exponent(r[None, :], 1.0, t) + 1j * t * kA[0])
        cS = np.exp(sampler_exponent(r[None, :], 1.0, t) + 1j * t * kS[0])
        if args.jobs > 1:
            eX, nx = _ecf_parallel(8.0, 16.0, 9.0, ekin, L, tgt[R_TMAX],
                                   args.n, args.seed, args.jobs, t, si)
            x = np.array([kA[0]])          # samples not kept at high N
        else:
            eX, nx = _ecf(x, t), len(x)
        eY = _ecf(y, t)
        nrm = 1.0 / np.sqrt(nx)          # MC error of the C++ sample
        nre = 1.0 / np.sqrt(len(y))      # MC error of the emulator sample
        print(f"\n    CF on t in [0, {t[-1]:.4g}] MeV^-1  "
              f"(core scale {core:.4g} MeV, MC error ~{nrm:.2e})")
        print(f"    N(G4) = {nx}")
        print(f"    {'|ecf(G4) - CF_sampler|':<30s} max {np.max(np.abs(eX - cS)):.3e}"
              f"   ({np.max(np.abs(eX - cS)) / nrm:.2f} sigma_MC)")
        print(f"    {'|ecf(emul) - CF_sampler|':<30s} max {np.max(np.abs(eY - cS)):.3e}"
              f"   ({np.max(np.abs(eY - cS)) / nre:.2f} sigma_MC[emul, N={len(y)}])")
        print(f"    {'|ecf(G4) - ecf(emul)|':<30s} max {np.max(np.abs(eX - eY)):.3e}"
              f"   ({np.max(np.abs(eX - eY)) / np.sqrt(nrm ** 2 + nre ** 2):.2f} sigma_MC)")
        print(f"    {'|CF_sampler - CF_analytic|':<30s} max "
              f"{np.max(np.abs(cS - cA)):.3e}   <-- THE QUESTION")
        print(f"    {'|ecf(G4) - CF_analytic|':<30s} max {np.max(np.abs(eX - cA)):.3e}"
              f"   ({np.max(np.abs(eX - cA)) / nrm:.2f} sigma_MC)")

        # --- the closure functional itself, on the single step
        print(f"\n    <e^{{-u z^2}}> on this step alone, z = (dE - <dE>)/{sig:.4g} MeV:")
        print(f"    {'u':>8} {'G4 sample':>12} {'emulator':>12} {'CF sampler':>12} "
              f"{'CF analytic':>12} {'samp - ana':>12}")
        for u in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
            tau = fn.closure_tau(u)
            pA = np.exp(ioni_step_exponent(r[None, :], 1.0 / sig, tau))
            pS = np.exp(sampler_exponent(r[None, :], 1.0 / sig, tau))
            zy = (y - kA[0]) / sig
            wA = fn.weier_scalar(pA, u, tau)
            wS = fn.weier_scalar(pS, u, tau)
            if args.jobs > 1:
                gx = "         n/a"     # samples not kept in the high-N mode
            else:
                gx = f"{np.mean(np.exp(-u * ((x - kA[0]) / sig) ** 2)):12.6f}"
            print(f"    {u:8.3g} {gx} "
                  f"{np.mean(np.exp(-u * zy ** 2)):12.6f} {wS:12.6f} "
                  f"{wA:12.6f} {wS - wA:+12.3e}")


def _ecf_parallel(Z, A, rho, ekin, length, tmax, ntot, seed, jobs, t, tag):
    """Empirical CF of `ntot` REAL-C++ samples, produced by `jobs` concurrent
    driver runs with distinct seeds and accumulated one file at a time.

    Needed because the quantity under test (|CF_sampler - CF_analytic| ~ 1e-3
    on a single step) is exactly the size of the MC error at a few million
    samples: at 4M the comparison cannot tell the two CFs apart, so it would be
    reported as "agreement" whichever model were right.
    """
    from concurrent.futures import ThreadPoolExecutor
    per = int(np.ceil(ntot / jobs))
    outs = [os.path.join(SCRATCH, f"_up{tag}_{j}") for j in range(jobs)]

    def one(j):
        # sample AND transform in the worker: only a (nt,) complex vector and
        # a count ever leave it, so nothing large is held or moved, and the
        # transform (which is the expensive half at 1e8 samples) is parallel
        # too. numpy releases the GIL inside the ufuncs.
        run_driver(Z, A, rho, ekin, length, tmax, per, seed + 1000 * (j + 1),
                   outs[j])
        v = np.fromfile(outs[j] + ".bin", dtype=np.float64)
        e = _ecf(v, t) * len(v)
        m = len(v)
        del v
        os.remove(outs[j] + ".bin")
        return e, m

    acc = np.zeros(len(t), dtype=np.complex128)
    n = 0
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for e, m in ex.map(one, range(jobs)):
            acc += e
            n += m
    return acc / n, n


def _ekin_of_tmax(tmax, mass=105.6583745, me=0.51099891):
    """Invert Geant4ePropagator's Emax for the kinetic energy."""
    from scipy.optimize import brentq

    def f(ek):
        E = ek + mass
        g = E / mass
        bg2 = g * g - 1.0
        mr = me / mass
        return 2.0 * me * bg2 / (1.0 + 2.0 * mr * g + mr * mr) - tmax

    return brentq(f, 1.0, 1e7, xtol=1e-12, rtol=1e-15)


def _core_scale(rec):
    """A width for the CORE of one step's loss: the sqrt of the variance
    accumulated by collisions whose expected multiplicity exceeds one.

    The full variance of a 1/E^2 channel, a3 A (tmax - e0), is tail dominated
    and says nothing about the scale on which the two CFs could differ; the
    number of delta rays above E is a3 A (1/E - 1/tmax), which reaches one at
    E1 = 1/(1/(a3 A) + 1/tmax), and the variance below E1 is the relevant one.
    """
    a3 = rec[R_A3]
    e0, tmax = rec[R_E0] * rec[R_SCAL], rec[R_TMAX] * rec[R_SCAL]
    v = rec[R_A1] * (rec[R_E1] * rec[R_SCAL]) ** 2 + \
        rec[R_A2] * (rec[R_E2] * rec[R_SCAL]) ** 2
    if a3 > 0 and tmax > e0:
        A = 1.0 / (1.0 / e0 - 1.0 / tmax)
        E1 = 1.0 / (1.0 / (a3 * A) + 1.0 / tmax)
        v += a3 * A * (E1 - e0)
    return np.sqrt(v)


# ==========================================================================
# subcommand: stock -- the model the SIM actually samples
# ==========================================================================

def cmd_stock(args):
    print("=" * 78)
    print("STOCK GEANT4 11.2.2 (what the full CMSSW SIM samples) vs the")
    print("EXTRAPOLATOR model (what the offline CF is built from)")
    print("=" * 78)
    print("The record comes from G4UniversalFluctuationForExtrapolator, which\n"
          "implements the PRE-2021 Urban model (two excitation channels).  The\n"
          "SIM runs stock G4UniversalFluctuation, the 2021 model (ONE excitation\n"
          "channel at the mean excitation energy).  Both are driven here at the\n"
          "SAME mean loss and the SAME tcut = tmax, so nothing but the model\n"
          "differs.  CAVEAT, stated up front: in the real SIM tcut is the e-\n"
          "production cut and the delta rays above it are explicit secondaries,\n"
          "which this single-call comparison does not reproduce; what is\n"
          "isolated here is the MODEL difference at fixed unrestricted\n"
          "straggling.\n")

    legs = _load(args.config)
    st = np.concatenate([l["ioni"] for l in legs if len(l["ioni"])])
    rng = np.random.default_rng(args.seed)
    for si in (args.steps or [2]):
        tgt = st[si]
        ekin = _ekin_of_tmax(tgt[R_TMAX])
        L, got = _match_len(8.0, 16.0, 9.0, ekin, tgt[R_TMAX], tgt[R_A1])
        kA = step_cumulants(got, "analytic")
        meanloss = kA[0]
        out = os.path.join(SCRATCH, f"_us{si}")
        rec, x, xs = run_driver(8.0, 16.0, 9.0, ekin, L, tgt[R_TMAX],
                                args.n, args.seed, out, stock=True,
                                meanloss=meanloss, tcut=tgt[R_TMAX])
        r = rec["rec"]
        ipot, e1F, e2F, f1, f2, Zeff = material_from_record(r)
        print(f"--- toy step {si}: len {L:.6f} mm, meanLoss {meanloss:.6f} MeV, "
              f"tcut = tmax = {tgt[R_TMAX]:.4f} MeV")
        print(f"    material recovered from the record vs the driver's own "
              f"G4IonisParamMat:")
        print(f"      ipot  {ipot:.8e}   driver {rec['ipot']:.8e}   "
              f"rel {(ipot / rec['ipot'] - 1):+.2e}")
        print(f"      Zeff  {Zeff:.6f}   f1 {f1:.4f}  f2 {f2:.4f}  "
              f"e1F {e1F:.6e}")
        sr = stock_record(meanloss, ipot, r[R_E0], tgt[R_TMAX])
        print(f"    stock block: a1 {sr[R_A1]:.6g}  e1 {sr[R_E1]:.6g}  "
              f"a3 {sr[R_A3]:.6g}      (extrapolator: a1 {r[R_A1]:.6g} "
              f"e1 {r[R_E1]:.6g}  a2 {r[R_A2]:.6g} e2 {r[R_E2]:.6g}  "
              f"a3 {r[R_A3]:.6g})")

        kS_stock = step_cumulants(sr, "sampler")
        kA_stock = step_cumulants(sr, "analytic")
        print(f"\n    excitation kappa2 [MeV^2]:  extrapolator "
              f"{r[R_A1] * (r[R_E1] * r[R_SCAL]) ** 2 + r[R_A2] * (r[R_E2] * r[R_SCAL]) ** 2:.6e}"
              f"   stock {sr[R_A1] * sr[R_E1] ** 2:.6e}")
        print(f"    total kappa2 [MeV^2]:       extrapolator {kA[1]:.6e}   "
              f"stock {kA_stock[1]:.6e}   rel {(kA_stock[1] / kA[1] - 1):+.3e}")
        core = _core_scale(r)
        print(f"    d kappa2 / kappa2_core:     "
              f"{(kA_stock[1] - kA[1]) / core ** 2:+.3e}   "
              f"(core scale {core:.4g} MeV)")

        if xs is None:
            print("    (no stock samples)")
            continue
        t = np.linspace(0.0, args.tmaxcf / core, args.nt)
        cE = np.exp(ioni_step_exponent(r[None, :], 1.0, t) + 1j * t * kA[0])
        cSt = np.exp(sampler_exponent(sr[None, :], 1.0, t) + 1j * t * kS_stock[0])
        cStA = np.exp(ioni_step_exponent(sr[None, :], 1.0, t) + 1j * t * kA_stock[0])
        e = _ecf(xs, t)
        nrm = 1.0 / np.sqrt(len(xs))
        print(f"\n    N = {len(xs)} stock samples, mean {xs.mean():.6f} "
              f"(driver reports {rec['stock_mean']:.6f})")
        print(f"    {'|ecf(stock) - CF_stockmodel|':<34s} max "
              f"{np.max(np.abs(e - cSt)):.3e}  ({np.max(np.abs(e - cSt)) / nrm:.2f} "
              f"sigma_MC)   <- validates the reconstruction")
        print(f"    {'|ecf(stock) - CF_extrapolator|':<34s} max "
              f"{np.max(np.abs(e - cE)):.3e}  ({np.max(np.abs(e - cE)) / nrm:.2f} "
              f"sigma_MC)   <- the SIM/model gap")
        print(f"    {'|CF_stockmodel - CF_extrap|':<34s} max "
              f"{np.max(np.abs(cSt - cE)):.3e}")
        print(f"    {'|CF_stock_ana - CF_extrap|':<34s} max "
              f"{np.max(np.abs(cStA - cE)):.3e}")


# ==========================================================================
# subcommand: perstep
# ==========================================================================

def cmd_perstep(args):
    print("=" * 78)
    print("PER-STEP: analytic compound Poisson vs the sampler's exact law")
    print("=" * 78)
    for tag in args.configs:
        legs = _load(tag)
        st = np.concatenate([l["ioni"] for l in legs if len(l["ioni"])])
        print(f"\n--- {tag}: {len(st)} steps")
        print(f"    {'#':>3} {'a3':>10} {'k2_ana':>12} {'dk2':>12} {'dk2/k2':>10} "
              f"{'dk2/k2_core':>12} {'dk4/k4':>11}")
        for i, r in enumerate(st):
            kA = step_cumulants(r, "analytic")
            kS = step_cumulants(r, "sampler")
            core = _core_scale(r) ** 2
            print(f"    {i:3d} {r[R_A3]:10.4g} {kA[1]:12.5e} "
                  f"{kS[1] - kA[1]:+12.4e} {(kS[1] - kA[1]) / kA[1]:+10.3e} "
                  f"{(kS[1] - kA[1]) / core:+12.3e} "
                  f"{(kS[3] - kA[3]) / abs(kA[3]):+11.3e}")


# ==========================================================================
# subcommand: closure -- propagate the per-step difference
# ==========================================================================

def stock_steps(steps):
    """The stock-Urban (2021) counterpart of a weighted step block.

    Each step is replaced by the stock model at the SAME mean loss, the same
    e0/tcut and the same weight, so differencing the two exponents isolates
    the model change and nothing else.  Returns (stock_steps, nskipped).
    """
    out, nskip = [], 0
    for r in steps:
        if r[R_REG] != 1 or r[R_E2] <= 0.0 or r[R_A1] <= 0.0:
            nskip += 1
            continue
        ipot, _, _, _, _, _ = material_from_record(r)
        ml = step_cumulants(r, "analytic")[0]
        out.append(stock_record(ml, ipot, r[R_E0] * r[R_SCAL],
                                r[R_TMAX] * r[R_SCAL], cs=r[R_CS]))
    return (np.array(out) if out else np.zeros((0, 11))), nskip


def _model_phi_pair(legs, k, avec, sigma, tau):
    """(phi_analytic, phi_sampler) of the standardized residual at plane k.

    phi_analytic is `cf_propagation_test.model_phi` verbatim -- the same
    function the published closure used, so the analytic side is not
    re-derived here.  phi_sampler multiplies it by exp(S_samp - S_ana) with
    both exponents built from the SAME weighted steps, so every non-ionization
    channel (MS, radiative) and every weight cancels identically.
    """
    phiA = cp.model_phi(legs, k, avec, sigma, tau)
    st = weighted_steps(legs, k, avec, sigma)
    if not len(st):
        return phiA, phiA
    dS = sampler_exponent(st, 1.0, tau) - ioni_step_exponent(st, 1.0, tau)
    return phiA, phiA * np.exp(dS)


def _model_phi_stock(legs, k, avec, sigma, tau):
    """phi with the ionization channel replaced by the STOCK Urban 2021 model
    (the one the SIM samples), same mean loss and same weights."""
    phiA = cp.model_phi(legs, k, avec, sigma, tau)
    st = weighted_steps(legs, k, avec, sigma)
    if not len(st):
        return phiA, 0
    sst, nskip = stock_steps(st)
    keep = np.array([r[R_REG] == 1 and r[R_E2] > 0 and r[R_A1] > 0 for r in st])
    dS = (ioni_step_exponent(sst, 1.0, tau)
          - ioni_step_exponent(st[keep], 1.0, tau))
    return phiA * np.exp(dS), nskip


def _gauge_phi(legs, k, avec, sigma, tau, eps):
    """phi with the ionization channel's variance changed by a RELATIVE eps,
    implemented as the extra Gaussian factor exp(-tau^2 eps V_ioni / 2).

    THE GAUGE. The propagated sampler/model difference is small; before that is
    read as "the mechanism is absent" the machinery has to be shown to respond
    to a difference of KNOWN size placed in the SAME channel, through the SAME
    weights and the SAME transform. V_ioni is the propagator's own ionization
    noise (model_variance's dQI term) in z^2 units, so eps = 1e-3 is literally
    "the block's ionization variance is 0.1 % larger".
    """
    phi = cp.model_phi(legs, k, avec, sigma, tau)
    _, _, vio = cp.model_variance(legs, k, avec)
    V = vio / sigma ** 2
    return phi * np.exp(-0.5 * tau ** 2 * eps * V)


def cmd_closure(args):
    print("=" * 78)
    print("PROPAGATION: what the sampler/model difference does to the closure")
    print("=" * 78)
    print("closure(u) = <e^{-u z^2}>_data - <e^{-u z^2}>_model, z = (sim-ref)/s_F.\n"
          "Only the MODEL side can move here, so the reported shift is\n"
          "    d closure(u) = <e^{-u z^2}>_model,analytic - <e^{-u z^2}>_model,sampler\n"
          "i.e. what the published closure would become if the model used the\n"
          "sampler's law.  s_F is held at its published (analytic) value: the\n"
          "data z are unchanged, and the s_F shift is second order (quoted).\n")

    probes = fn.UCURVE
    for tag in args.configs:
        legs = _load(tag)
        for func in args.funcs:
            avec = cp.FUNCTIONALS[func]
            sc = fn.plane_scales(legs, func, tag=tag)
            print(f"\n### {tag}  {func}")
            print(f"  {'plane':>5} " + "".join(f"{u:>10.3g}" for u in probes))
            rows = []
            for k in range(len(legs)):
                s = float(sc["sF"][k])
                tau = fn.closure_tau(float(probes.max()))
                phiA, phiS = _model_phi_pair(legs, k, avec, s, tau)
                row = np.array([fn.weier_scalar(phiA, u, tau)
                                - fn.weier_scalar(phiS, u, tau) for u in probes])
                rows.append(row)
                print(f"  {k:5d} " + "".join(f"{v:+10.2e}" for v in row))
            rows = np.array(rows)
            print(f"  {'MEAN':>5} " + "".join(f"{v:+10.2e}" for v in rows.mean(axis=0)))

            # --- the gauge: the same transform, the same channel, a KNOWN
            #     relative variance change. Establishes the sensitivity and
            #     shows the small answer above is not a dead pipeline.
            print("\n  GAUGE -- d closure for a KNOWN relative change eps of the\n"
                  "  block's ionization variance (same channel, same weights,\n"
                  "  same transform).  Mean over planes.")
            print(f"  {'eps':>9} " + "".join(f"{u:>10.3g}" for u in probes))
            for eps in args.gauge:
                gr = []
                for k in range(len(legs)):
                    s_ = float(sc["sF"][k])
                    tau = fn.closure_tau(float(probes.max()))
                    phiA = cp.model_phi(legs, k, avec, s_, tau)
                    phiG = _gauge_phi(legs, k, avec, s_, tau, eps)
                    gr.append([fn.weier_scalar(phiA, u, tau)
                               - fn.weier_scalar(phiG, u, tau) for u in probes])
                gm = np.array(gr).mean(axis=0)
                print(f"  {eps:9.1e} " + "".join(f"{v:+10.2e}" for v in gm))
            # the sampler's OWN relative variance change, for the comparison
            dv = []
            for k in range(len(legs)):
                st = weighted_steps(legs, k, avec, float(sc["sF"][k]))
                gs = st[:, R_CS] * 1e-3
                kA = np.array([step_cumulants(r, "analytic") for r in st])
                kS = np.array([step_cumulants(r, "sampler") for r in st])
                dv.append(float(np.sum((kS[:, 1] - kA[:, 1]) * gs ** 2)
                                / np.sum(kA[:, 1] * gs ** 2)))
            print(f"  the SAMPLER's own relative ionization-kappa2 change is "
                  f"eps = {np.mean(dv):+.3e}\n  (mean over planes; the row to "
                  f"interpolate the gauge at)")

            if args.stock:
                print("\n  STOCK URBAN 2021 (the model the SIM samples) in place of the\n"
                      "  extrapolator's pre-2021 model, same mean loss, same weights:")
                srows, nsk = [], 0
                for k in range(len(legs)):
                    s_ = float(sc["sF"][k])
                    tau = fn.closure_tau(float(probes.max()))
                    phiA = cp.model_phi(legs, k, avec, s_, tau)
                    phiSt, nsk = _model_phi_stock(legs, k, avec, s_, tau)
                    srows.append([fn.weier_scalar(phiA, u, tau)
                                  - fn.weier_scalar(phiSt, u, tau) for u in probes])
                sm = np.array(srows).mean(axis=0)
                print(f"  {'MEAN':>9} " + "".join(f"{v:+10.2e}" for v in sm))
                print(f"  ({nsk} step(s) per block skipped: not regime 1 / no "
                      f"excitation block)")

            if args.taun:
                print("\n  GRID CONVERGENCE of the sampler shift (tau nodes; the\n"
                      "  published default is 3000):")
                for nt in args.taun:
                    rr = []
                    for k in range(len(legs)):
                        s_ = float(sc["sF"][k])
                        tau = fn.closure_tau(float(probes.max()), n=int(nt))
                        phiA, phiS = _model_phi_pair(legs, k, avec, s_, tau)
                        rr.append([fn.weier_scalar(phiA, u, tau)
                                   - fn.weier_scalar(phiS, u, tau) for u in probes])
                    print(f"  {int(nt):9d} " + "".join(
                        f"{v:+10.2e}" for v in np.array(rr).mean(axis=0)))
            print(f"\n  observed layered-toy closure for comparison "
                  f"(NOTES_TOY_PT40, {tag}):")
            obs = OBSERVED.get((tag, func))
            if obs is not None:
                print(f"  {'obs':>5} " + "".join(f"{v:+10.2e}" for v in obs))
                print(f"  {'ratio':>5} " + "".join(
                    f"{(m / o if o else np.nan):+10.2e}"
                    for m, o in zip(rows.mean(axis=0), obs)))


# published layered-toy closure, NOTES_TOY_PT40 section 1, K1, mean over planes,
# on fn.UCURVE = [1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
OBSERVED = {
    ("pt3_K1", "qop"): np.array([-0.00432, -0.00756, -0.01326, -0.01854,
                                 -0.01167, -0.00042, +0.00142, np.nan, np.nan]),
    ("pt40_K1", "qop"): np.array([-0.01856, -0.04181, -0.08684, -0.09252,
                                  -0.02888, -0.00958, -0.00446, np.nan, np.nan]),
}


# ==========================================================================
# subcommand: scan
# ==========================================================================

def cmd_scan(args):
    print("=" * 78)
    print("PARAMETER-SPACE SCAN: where the sampler and the model diverge")
    print("=" * 78)
    print("One step of the toy material (Z=8, A=16, rho=9 g/cm3) at pT = 3,\n"
          "step length scanned over 5 decades so that a1 and a3 sweep through\n"
          "the nmaxCont = 8 thresholds and, at the long end, into regime 0.\n")
    ekin = _ekin_of_tmax(698.568)
    tmax = 698.568
    print(f"  {'len[mm]':>10} {'reg':>4} {'a1':>10} {'a2':>10} {'a3':>10} "
          f"{'p3':>7} {'k2_ana':>11} {'dk2/k2':>10} {'dk2/k2core':>11} "
          f"{'dk4/k4':>10}")
    for L in np.geomspace(args.lmin, args.lmax, args.npts):
        rec, _, _ = run_driver(8.0, 16.0, 9.0, ekin, float(L), tmax, 8, 1,
                               os.path.join(SCRATCH, "_uscan"))
        r = rec["rec"]
        if r[R_REG] == 0:
            print(f"  {L:10.4g} {int(r[R_REG]):4d} {'':>10} {'':>10} {'':>10} "
                  f"{'':>7} {r[R_GSIG2]:11.4e}   regime 0: the record carries "
                  f"only the variance, so the analytic side cannot see the\n"
                  f"{'':>12}[0, 2 meanLoss] truncation the sampler applies")
            continue
        kA = step_cumulants(r, "analytic")
        kS = step_cumulants(r, "sampler")
        _, _, _, p3, _, _, _ = urban_split(np.array([r[R_A3]]),
                                           np.array([r[R_E0]]),
                                           np.array([r[R_TMAX]]))
        core = _core_scale(r) ** 2
        print(f"  {L:10.4g} {int(r[R_REG]):4d} {r[R_A1]:10.4g} {r[R_A2]:10.4g} "
              f"{r[R_A3]:10.4g} {float(p3):7.4f} {kA[1]:11.4e} "
              f"{(kS[1] - kA[1]) / kA[1]:+10.3e} {(kS[1] - kA[1]) / core:+11.3e} "
              f"{(kS[3] - kA[3]) / abs(kA[3]):+10.3e}")


# ==========================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("records")
    q.add_argument("--configs", nargs="+", default=["pt3_K1", "pt40_K1"])
    q.set_defaults(func=cmd_records)

    q = sub.add_parser("g4check")
    q.add_argument("--config", default="pt3_K1")
    q.add_argument("--steps", nargs="*", type=int, default=None)
    q.add_argument("-n", type=int, default=4000000)
    q.add_argument("--seed", type=int, default=987)
    q.add_argument("--nt", type=int, default=400)
    q.add_argument("--tmaxcf", type=float, default=8.0,
                   help="CF grid upper limit in units of 1/core scale")
    q.add_argument("--nemu", type=int, default=4000000,
                   help="cap on the python-emulator sample count")
    q.add_argument("--jobs", type=int, default=1,
                   help="split -n over this many parallel driver runs and "
                        "accumulate only the empirical CF (for the high-N "
                        "test that can RESOLVE the model/sampler difference)")
    q.set_defaults(func=cmd_g4check)

    q = sub.add_parser("stock")
    q.add_argument("--config", default="pt3_K1")
    q.add_argument("--steps", nargs="*", type=int, default=None)
    q.add_argument("-n", type=int, default=4000000)
    q.add_argument("--seed", type=int, default=987)
    q.add_argument("--nt", type=int, default=400)
    q.add_argument("--tmaxcf", type=float, default=8.0)
    q.set_defaults(func=cmd_stock)

    q = sub.add_parser("perstep")
    q.add_argument("--configs", nargs="+", default=["pt3_K1", "pt40_K1"])
    q.set_defaults(func=cmd_perstep)

    q = sub.add_parser("closure")
    q.add_argument("--configs", nargs="+", default=["pt3_K1", "pt40_K1"])
    q.add_argument("--funcs", nargs="+", default=["qop"])
    q.add_argument("--gauge", nargs="*", type=float,
                   default=[1e-4, 1e-3, 1e-2])
    q.add_argument("--stock", action="store_true")
    q.add_argument("--taun", nargs="*", type=int, default=[])
    q.set_defaults(func=cmd_closure)

    q = sub.add_parser("scan")
    q.add_argument("--lmin", type=float, default=1e-3)
    q.add_argument("--lmax", type=float, default=1e3)
    q.add_argument("--npts", type=int, default=25)
    q.set_defaults(func=cmd_scan)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
