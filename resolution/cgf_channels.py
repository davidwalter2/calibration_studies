#!/usr/bin/env python3
"""Real-argument CGFs of the MULTIPLE-SCATTERING and RADIATIVE channels, and
the combined ionization + MS + radiative block.

WHY. `cgf_saddlepoint.ioni_cgf_derivs` gives K, K', K'', K''', K'''' of the
centred ionization CGF of a pooled block, from which the score
psi = theta + K'''/(2 K''^2), its derivative, the Fisher information
I = E[psi^2] and the Gaussian surrogate all follow analytically. That covered
ONE of the three noise channels. A physical block is

    ionization  +  multiple scattering  +  bremsstrahlung/pair production

and cumulant generating functions ADD over independent channels, so the block
CGF is the sum of the three. This module supplies the missing two, with the
same conventions as `ioni_cgf_derivs` (centred, K'(0) = 0; per-step transport
weight folded in by the caller; `order=` up to 4), plus the combined block.

CONVENTIONS, taken from `cf_propagation_test.model_phi` rather than assumed
------------------------------------------------------------------------
model_phi is the trusted imaginary-argument (characteristic function) code.
It weights the three channels DIFFERENTLY and the difference is not cosmetic:

  ionization  the noise sits in the q/p component ONLY, so the weight is the
              scalar  w = q * (a^T A_ioni)_qop / sigma  and it is folded into
              the record's qop-per-MeV column (col 10).
  radiative   same compound-Poisson structure, same qop-only scalar weight.
              The radv records are aligned with the MOLIERE log, the ioni
              weights with the IONIZATION log, so when the two counts differ
              model_phi falls back to the leg MEAN weight -- mirrored here.
  MS          an isotropic 2D kick in (lambda, phi). With the azimuthal angle
              measured as phi*cos(lambda) the two projected angles are iid, so
              by isotropy only the QUADRATURE SUM of the two transport weights
              enters:  w = sqrt(w_lambda^2 + (w_phi/cos lambda)^2) / sigma.
              Each step is further split into MS_NSUB sub-kicks with the
              material (xg, hence chi_c^2) divided by MS_NSUB and the weight
              interpolated linearly from the step start to the step end
              (`cf_propagation_test.MS_NSUB`, the sub-step quadrature of
              2026-08-08). `collect_ms_steps` reproduces that expansion, so
              the CGF and the CF see literally the same list of kicks.

MS: THE CGF IS *NOT* ENTIRE UNLESS THE ANGLE IS CUT OFF
-------------------------------------------------------
The modelled single-scattering density in t = theta_s^2 is

    dn/dt = chi_c^2 / (t + chi_a^2)^2 / (1 + t/theta_FF^2)^p ,   p = 4

(screened Rutherford times the G4 dipole nuclear form factor SQUARED). Its
tail is dn/dt ~ t^{-6}, so the moments mu_n = int t^n dn/dt dt DIVERGE for
n >= 5. The CGF is the moment series

    K(theta) = sum_{n>=1} (w theta)^{2n} mu_n / (4^n (n!)^2)

(from I0(u) = sum (u/2)^{2n}/(n!)^2), so with the spectrum extended to
arbitrarily large angle K(theta) = +infinity for EVERY theta != 0. The
characteristic function does not care -- |J0| <= 1 makes the same integral
converge -- which is why this never showed up in the CF code.

The fix is physical, not numerical: a single Coulomb scatter cannot deflect by
more than pi. With a cutoff theta_s <= theta_cut every jump is bounded, every
moment is finite, and the CGF IS entire. `theta_cut` is therefore an explicit
argument (default pi) and `ms_cut_scan` measures how much the answer moves
when it is varied -- it must be, and is, negligible in the theta range the
saddlepoint actually visits.

MS: HOW IT IS EVALUATED
-----------------------
The moment series above is exact, has ALL-POSITIVE terms (no cancellation
anywhere, unlike the alternating series the CF would need), and gives every
derivative for free:

    K^(m)(theta) = sum_n T_n ff(2n, m) theta^{2n-m},   ff = falling factorial
    T_n = sum_steps chi_c^2_s w_s^{2n} mu~_n^s / (4^n (n!)^2)

with mu~_n^s = int_0^{theta_cut^2} t^n dt/(t+chi_a^2)^2/(1+t/theta_FF^2)^p.
The whole pooled block collapses to ONE power series in theta^2, so evaluating
it on a 10^4-point theta grid costs nothing. Sums are done in log space
(logsumexp), so the dynamic range of w^{2n} mu~_n (which spans hundreds of
decades) never overflows or underflows spuriously, and the result becomes
+inf only when the true value leaves the double range.

RADIATIVE
---------
Bounded jumps by construction (the photon takes v*E with v <= 1 on the
exported grid), so the CGF is entire -- no caveat. It is evaluated directly on
the same 48-point v grid and with the same trapezoid rule as
`cf_brems_exact.rad_exponent`, so the two are exact analytic continuations of
each other:

    K^(0) = sum_s int dv (dN/dv) (e^x - 1 - x),     x = theta * cs * w * v E
    K^(1) = sum_s (cs w)   int dv (dN/dv) (vE)   (e^x - 1)
    K^(m) = sum_s (cs w)^m int dv (dN/dv) (vE)^m e^x         (m >= 2)

with expm1 and a small-|x| series so the centring subtraction does not cancel.

usage:
  python cgf_channels.py --model M.root [--planes 0,9,18] [--validate]
"""
import argparse
import os
import sys

import numpy as np
from scipy.special import gammaln, logsumexp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cf_brems_exact
import cf_propagation_test as cpt
from cf_ms_exact import G4_FF_SQUARED, moliere_params
from cf_propagation_test import FUNCTIONALS, model_variance, step_transports
from cgf_saddlepoint import _LOG_MAX, ioni_cgf_derivs

# ymax bucketing of cf_track_resolution.ms_step_exponent. That function SNAPS
# each step's theta_FF/chi_a to one of these 13 values (it calls gshape with
# the row's ymr, not with the step's own ymax), so mirroring the snap is what
# makes the CGF the exact continuation of the CF the code actually evaluates.
_YMAX_LO, _YMAX_HI, _YMAX_NROW = 1e1, 1e7, 12
# default maximum single-scatter deflection. pi is the kinematic bound.
THETA_CUT = np.pi
# |x| below which (e^x - 1 - x) is taken from its series (mirrors the 1e-4 of
# cf_brems_exact.rad_exponent).
_RAD_SER = 1e-4

# t grid for the REFERENCE inversions. Same shape as cf_propagation_test.TAU
# (logarithmic: the tail sets the decay near t ~ 1e-2 and the core near
# t ~ 10, three decades apart) but denser, because everything downstream
# interpolates it onto a uniform grid and that interpolation is the one
# remaining approximation on the exact side.
TAU_REF = np.concatenate([[0.0], np.geomspace(1e-4, 40.0, 6000)])


# ==================================================================== MS =====
def _snap_ymax(ymax):
    """The ymax value ms_step_exponent actually uses for a step."""
    lg = np.log(np.clip(ymax, _YMAX_LO, _YMAX_HI))
    step = np.log(_YMAX_HI / _YMAX_LO) / _YMAX_NROW
    r = np.round((lg - np.log(_YMAX_LO)) / step)
    return np.exp(np.log(_YMAX_LO) + r * step)


def ms_step_params(steps, snap_ymax=True):
    """(chic2, chia2, thff2) per msmoliv record, exactly as ms_step_exponent
    reads them (per-element sums from cols 7,8 when the stride is >= 10).

    Rows with thp2 <= 0, chic2 <= 0 or chia2 <= 0 are dropped, as there.
    """
    steps = np.atleast_2d(np.asarray(steps, dtype=np.float64))
    if steps.size == 0:
        return (np.zeros(0),) * 3 + (np.zeros(0, dtype=bool),)
    keep = steps[:, 5] > 0.0
    prm = np.zeros((len(steps), 3))
    for i in np.where(keep)[0]:
        s = steps[i]
        prm[i] = (moliere_params(*s[:5], s[7], s[8]) if steps.shape[1] >= 10
                  else moliere_params(*s[:5]))
    keep &= (prm[:, 0] > 0.0) & (prm[:, 1] > 0.0)
    chic2, chia2, thff2 = prm[:, 0], prm[:, 1], prm[:, 2]
    if snap_ymax:
        with np.errstate(invalid="ignore", divide="ignore"):
            ym = _snap_ymax(np.sqrt(np.where(chia2 > 0, thff2 / chia2, 1.0)))
        thff2 = np.where(chia2 > 0, ym ** 2 * chia2, thff2)
    return chic2, chia2, thff2, keep


def _gl_lognodes(t_lo, t_hi, dlog=0.1, npt=20):
    """Composite Gauss-Legendre nodes/weights for int_{t_lo}^{t_hi} f(t) dt,
    laid out uniformly in ln t (so t^n stays a slowly varying exponential on
    each panel and the rule is spectrally accurate up to large n)."""
    L = np.log(t_hi / t_lo)
    npan = max(int(np.ceil(L / dlog)), 1)
    edges = np.linspace(np.log(t_lo), np.log(t_hi), npan + 1)
    x, wq = np.polynomial.legendre.leggauss(npt)
    h = 0.5 * (edges[1] - edges[0])
    c = 0.5 * (edges[:-1] + edges[1:])
    u = (c[:, None] + h * x[None, :]).ravel()        # ln t nodes
    t = np.exp(u)
    w = (h * np.tile(wq, npan)) * t                  # dt = t d(ln t)
    return t, w


def ms_log_moments(chia2, thff2, nmax, theta_cut=THETA_CUT, ff_pow=None,
                   dlog=None, npt=20, chunk=4_000_000):
    """log of the truncated angular moments, (nstep, nmax):

        mu~_n = int_0^{theta_cut^2} t^n dt / (t + chia2)^2 / (1 + t/thff2)^p

    i.e. the moments of the single-scattering density WITHOUT its chi_c^2
    prefactor. Positive integrand, composite Gauss-Legendre in ln t with the
    panel width tied to nmax so that t^n = e^{n ln t} stays resolvable.

    THESE ARE THE OBJECT THAT DIVERGES. With the G4 form factor squared the
    integrand falls as t^{n-6}, so without the theta_cut every mu~_n with
    n >= 5 is infinite and the MS CGF is +inf for all theta != 0.
    """
    if ff_pow is None:
        ff_pow = 4 if G4_FF_SQUARED else 2
    chia2 = np.atleast_1d(np.asarray(chia2, float))
    thff2 = np.atleast_1d(np.asarray(thff2, float))
    Tc = float(theta_cut) ** 2
    t_lo = float(np.min(chia2)) * 1e-8
    if dlog is None:
        # n * dlog <~ 4 keeps the Gauss-Legendre truncation far below double
        # precision on every panel.
        dlog = min(0.1, 4.0 / max(nmax, 1))
    t, wq = _gl_lognodes(t_lo, Tc, dlog=dlog, npt=npt)
    n = np.arange(1, nmax + 1)
    lr = np.log(t / Tc)                       # <= 0, so (t/Tc)^n never grows
    pw = np.exp(np.outer(lr, n))
    per = max(int(chunk // max(len(t), 1)), 1)
    acc = []
    for a in range(0, len(chia2), per):
        b = min(a + per, len(chia2))
        f = (wq[None, :] / (t[None, :] + chia2[a:b, None]) ** 2
             / (1.0 + t[None, :] / thff2[a:b, None]) ** ff_pow)
        with np.errstate(divide="ignore"):
            acc.append(np.log(f @ pw))
    return np.concatenate(acc, axis=0) + n * np.log(Tc)


def ms_series_coeffs(chic2, chia2, thff2, weights, nmax,
                     theta_cut=THETA_CUT, ff_pow=None, dlog=None, npt=20,
                     chunk=4_000_000):
    """log T_n, n = 1..nmax, for the pooled MS block.

        T_n = sum_s chic2_s w_s^{2n} mu~_n^s / (4^n (n!)^2)

    Every term is positive, so the accumulation is a logsumexp and there is no
    cancellation anywhere. Returned as logs because w^{2n} mu~_n spans several
    hundred decades across the block.
    """
    chic2 = np.asarray(chic2, float)
    chia2 = np.asarray(chia2, float)
    thff2 = np.asarray(thff2, float)
    w = np.asarray(weights, float)
    ok = (chic2 > 0) & (chia2 > 0) & (w > 0)
    if not ok.any() or nmax < 1:
        return np.full(max(nmax, 0), -np.inf)
    chic2, chia2, thff2, w = chic2[ok], chia2[ok], thff2[ok], w[ok]
    lmu = ms_log_moments(chia2, thff2, nmax, theta_cut=theta_cut,
                         ff_pow=ff_pow, dlog=dlog, npt=npt, chunk=chunk)
    n = np.arange(1, nmax + 1)
    lg = (lmu + np.log(chic2)[:, None] + 2.0 * np.outer(np.log(w), n))
    return (logsumexp(lg, axis=0) - n * np.log(4.0)
            - 2.0 * gammaln(n + 1.0))


def _series_eval(logT, theta, order, ltol=32.0):
    """K ... K^(order) of K(theta) = sum_{n>=1} T_n theta^{2n} from log T_n.

    All-positive terms: for even m the sum is positive, for odd m it carries
    sign(theta). Accumulated with the largest log factored out, so the result
    is +-inf only when the true value leaves the double range.

    SELF-DIAGNOSING TRUNCATION. The series needs n ~ w theta_cut |theta| terms
    before the terms turn over, so a fixed nmax is safe only up to some
    |theta|. Rather than silently returning a truncated (i.e. far too small)
    value, any theta whose LAST retained term is still within e^{-ltol} of the
    partial sum is returned as +inf -- the same "the value is not
    representable here" signal an overflow gives, and the saddlepoint code
    masks it the same way.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=np.float64))
    nmax = len(logT)
    n = np.arange(1, nmax + 1)
    out = []
    at = np.abs(theta)
    nz = at > 0.0
    lt = np.full(at.shape, -np.inf)
    lt[nz] = np.log(at[nz])
    bad = None
    for m in range(order + 1):
        # falling factorial ff(2n, m) = prod_{i<m} (2n - i); zero when 2n < m
        ff = np.ones(nmax)
        for i in range(m):
            ff = ff * (2.0 * n - i)
        val = np.zeros_like(theta)
        good = ff > 0
        if good.any():
            lc = logT[good] + np.log(ff[good])
            p = (2.0 * n[good] - m)
            with np.errstate(invalid="ignore"):
                lg = lc[None, :] + p[None, :] * lt[:, None]
            lg = np.where(np.isnan(lg), -np.inf, lg)
            lsum = logsumexp(lg, axis=1)
            with np.errstate(over="ignore"):
                val = np.exp(lsum)
            if bad is None:
                bad = nz & (lg[:, -1] > lsum - ltol)
            # theta = 0: only the p = 0 term survives
            if (~nz).any():
                z = np.zeros(len(lc))
                z[p == 0] = np.exp(lc[p == 0])
                val[~nz] = z.sum()
        if m % 2 == 1:
            val = val * np.sign(theta)
        if bad is not None and bad.any():
            val = np.where(bad, np.inf, val)
        out.append(val)
    return tuple(out)


def ms_cgf_derivs(steps, weights, theta, order=2, theta_cut=THETA_CUT,
                  nmax=None, snap_ymax=True, logT=None, ff_pow=None):
    """K and its first `order` derivatives (order <= 4) for the CENTRED
    multiple-scattering CGF of a pooled block.

    steps   : (n, 8) or (n, 10) msmoliv records, already sub-step expanded by
              `collect_ms_steps` (i.e. xg already divided by MS_NSUB).
    weights : (n,) per-record effective scalar weight
              w = sqrt(w_lambda^2 + (w_phi/cos lambda)^2)/sigma, the quadrature
              sum model_phi uses -- NOT the qop weight of the ionization
              channel.
    theta   : scalar or array, EITHER SIGN.
    order   : defaults to 2, matching ioni_cgf_derivs.

    Centred by construction: the projected deflection is symmetric, so K is
    EVEN in theta and K'(0) = 0 exactly (no subtraction is needed, and none is
    done -- the "-1" under the integral removes only the n = 0 term).

    theta_cut is the maximum single-scatter deflection. It is NOT optional:
    without it the modelled t^{-6} tail makes every moment above the 8th
    divergent and K(theta) = +inf for all theta != 0. See the module docstring.

    logT lets a caller reuse precomputed series coefficients.
    """
    if not 0 <= order <= 4:
        raise ValueError("order must be 0..4")
    theta = np.atleast_1d(np.asarray(theta, dtype=np.float64))
    if logT is None:
        if nmax is None:
            nmax = ms_series_nmax(steps, weights, theta, theta_cut=theta_cut,
                                  snap_ymax=snap_ymax)
        chic2, chia2, thff2, keep = ms_step_params(steps, snap_ymax=snap_ymax)
        wv = np.asarray(weights, float)
        if len(wv) != len(keep):
            raise ValueError("weights must be aligned with steps")
        logT = ms_series_coeffs(chic2[keep], chia2[keep], thff2[keep],
                                wv[keep], nmax, theta_cut=theta_cut,
                                ff_pow=ff_pow)
    return _series_eval(logT, theta, order)


MS_NMAX_CAP = 200


def ms_series_nmax(steps, weights, theta, theta_cut=THETA_CUT, snap_ymax=True,
                   pad=40, cap=MS_NMAX_CAP):
    """Number of moment terms needed: the series behaves like
    (w theta_cut theta / 2)^{2n}/(n!)^2, so n must reach ~ w theta_cut |theta|
    before the terms start falling.

    The cap is not a silent truncation: `_series_eval` returns +inf wherever
    `cap` terms are not enough, and the saddlepoint masks those theta the same
    way it masks a genuine overflow. `ms_nmax_scan` checks that the masked
    region carries no probability mass.
    """
    w = np.asarray(weights, float)
    _, _, _, keep = ms_step_params(steps, snap_ymax=snap_ymax)
    if not np.any(keep) or not len(w):
        return 1
    amax = float(np.max(np.abs(w[keep]))) * float(theta_cut)
    tmax = float(np.max(np.abs(np.atleast_1d(theta))))
    return int(min(max(int(np.ceil(amax * tmax)) + pad, pad), cap))


def ms_cf_exponent_quad(steps, weights, tau, theta_cut=THETA_CUT,
                        snap_ymax=True, ny=6000, ff_pow=None):
    """The MS CF exponent by DIRECT high-accuracy quadrature of the defining
    integral,

        S(t) = sum_s chic2_s int_0^{Tc} (J0(w_s t sqrt(t_ang)) - 1) g_s dt_ang

    on a fine log grid. Slow, and only used as a reference: it is the same
    object cf_ms_exact.gshape returns from its 360-node y^2 grid and 1600-node
    tau table with linear interpolation, so comparing the two measures the
    accuracy of the TABLE rather than of anything here.
    """
    from scipy.special import j0
    if ff_pow is None:
        ff_pow = 4 if G4_FF_SQUARED else 2
    c2, a2, f2, keep = ms_step_params(steps, snap_ymax=snap_ymax)
    w = np.asarray(weights, float)
    c2, a2, f2, w = c2[keep], a2[keep], f2[keep], w[keep]
    tau = np.atleast_1d(np.asarray(tau, float))
    Tc = float(theta_cut) ** 2
    out = np.zeros(len(tau))
    for i in range(len(c2)):
        t = np.geomspace(a2[i] * 1e-8, Tc, ny)
        g = 1.0 / (t + a2[i]) ** 2 / (1.0 + t / f2[i]) ** ff_pow
        arg = np.outer(tau * w[i], np.sqrt(t))
        # J0(x) - 1 CANCELS CATASTROPHICALLY at small x: it is -x^2/4 while
        # both terms are ~1, so float64 gives at best 1e-16/x^2 relative. Over
        # most of this grid x is 1e-10 or smaller (chi_a ~ 1e-7 rad at
        # p = 100 GeV), and taking the difference directly made this reference
        # -- MY reference, not the code under test -- wrong by a factor 3 at
        # pT = 100. Use the series below |x| = 1e-2, where it is exact to
        # x^6/2304 < 4e-13 relative.
        x2 = arg ** 2
        km1 = np.where(np.abs(arg) < 1e-2,
                       -0.25 * x2 * (1.0 - x2 / 16.0 * (1.0 - x2 / 36.0)),
                       j0(arg) - 1.0)
        out += c2[i] * np.trapezoid(km1 * g[None, :], t, axis=1)
    return out


def ms_cf_exponent_series(steps, weights, tau, theta_cut=THETA_CUT,
                          nmax=None, snap_ymax=True, logT=None):
    """The IMAGINARY-argument exponent of the same series,
    S(t) = sum_n (-1)^n T_n t^{2n} -- i.e. K(i t). Only for the
    continuation cross-check against cf_track_resolution.ms_step_exponent;
    it is alternating, so it is not the way to evaluate the CF in production.
    """
    tau = np.atleast_1d(np.asarray(tau, dtype=np.float64))
    if logT is None:
        if nmax is None:
            nmax = ms_series_nmax(steps, weights, tau, theta_cut=theta_cut,
                                  snap_ymax=snap_ymax)
        chic2, chia2, thff2, keep = ms_step_params(steps, snap_ymax=snap_ymax)
        w = np.asarray(weights, float)
        logT = ms_series_coeffs(chic2[keep], chia2[keep], thff2[keep],
                                w[keep], nmax, theta_cut=theta_cut)
    n = np.arange(1, len(logT) + 1)
    sgn = (-1.0) ** n
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        terms = sgn[None, :] * np.exp(logT[None, :]
                                      + 2.0 * np.outer(np.log(np.abs(tau)), n))
    terms = np.where(np.isfinite(terms), terms, 0.0)
    out = terms.sum(axis=1)
    out[np.abs(tau) == 0.0] = 0.0
    return out


# ============================================================== RADIATIVE =====
def rad_step_spectra(recs, spec, vgrid):
    """(v, dN/dv per step, jump size vE per step) -- the same
    cf_brems_exact.step_spectrum used by rad_exponent, evaluated once."""
    recs = np.atleast_2d(np.asarray(recs, dtype=np.float64))
    spec = np.atleast_2d(np.asarray(spec, dtype=np.float64))
    v = np.asarray(vgrid, dtype=np.float64)
    dn = np.zeros((len(recs), len(v)))
    eps = np.zeros((len(recs), len(v)))
    for i, (rec, sp) in enumerate(zip(recs, spec)):
        _, y = cf_brems_exact.step_spectrum(rec, sp, v)
        dn[i] = y
        eps[i] = v * rec[cf_brems_exact.R_ETOT]
    return v, dn, eps


def rad_cgf_derivs(recs, spec, vgrid, weights, theta, order=2, pre=None):
    """K and its first `order` derivatives (order <= 4) for the CENTRED
    radiative (brems + pair) CGF of a pooled block.

    recs/spec/vgrid : radv records, tabulated shapes and the shared v grid,
                      exactly as cf_brems_exact.rad_exponent takes them.
    weights         : (nstep,) transport weight, the SAME qop-only scalar the
                      ionization channel uses (see model_phi).
    theta           : scalar or array, either sign.

    Centred: the "-1 - x" removes the mean, which is right because the
    propagator's reference already subtracted the radiative mean loss.
    Jumps are bounded by v <= 1, so the CGF is entire; it overflows only at
    |theta| > _LOG_MAX / max|cs w vE|, which `rad_theta_overflow_limit`
    returns.
    """
    if not 0 <= order <= 4:
        raise ValueError("order must be 0..4")
    theta = np.atleast_1d(np.asarray(theta, dtype=np.float64))
    K = [np.zeros_like(theta) for _ in range(5)]
    if pre is None:
        if recs is None or len(recs) == 0:
            return tuple(K[:order + 1])
        v, dn, eps = rad_step_spectra(recs, spec, vgrid)
        cs = np.asarray(recs, dtype=np.float64)[:, cf_brems_exact.R_CS]
    else:
        v, dn, eps, cs = pre
    w = np.asarray(weights, float)
    if len(w) != len(dn):
        raise ValueError("weights must be aligned with the radv records")
    a = cs * w                                    # d(qop)/dE for this step
    for i in range(len(dn)):
        if not np.any(dn[i] > 0.0) or a[i] == 0.0:
            continue
        x = np.outer(theta * a[i], eps[i])        # (nt, nv)
        # overflow/invalid are expected far out in theta: the TRUE value
        # leaves the double range there, and the saddlepoint masks it. What
        # must not happen is a silently clipped finite value, and nothing
        # here clips.
        with np.errstate(over="ignore", invalid="ignore"):
            ex = np.exp(x)
            em = np.expm1(x)
            small = np.abs(x) < _RAD_SER
            if order >= 0:
                t0 = np.where(small,
                              0.5 * x ** 2 * (1.0 + x / 3.0 * (1.0 + 0.25 * x)),
                              em - x)
                K[0] += np.trapezoid(t0 * dn[i][None, :], v, axis=1)
            if order >= 1:
                K[1] += a[i] * np.trapezoid(
                    em * eps[i][None, :] * dn[i][None, :], v, axis=1)
            for m in range(2, order + 1):
                K[m] += a[i] ** m * np.trapezoid(
                    ex * eps[i][None, :] ** m * dn[i][None, :], v, axis=1)
    return tuple(K[:order + 1])


def rad_theta_overflow_limits(recs, spec, vgrid, weights, margin=0.98,
                              pre=None):
    """(bound on -theta, bound on +theta) from e^{theta cs w vE}.

    SIGNED, and that matters. The radiative jump cs*w*vE has ONE sign (energy
    loss, times a fixed transport weight), so the exponential grows on only
    ONE side of theta -- the same side the ionization delta-ray term grows on.
    A symmetric bound would cap the safe side too, which is exactly the side
    the mode sits on, and would truncate the saddlepoint curve before it
    reaches the mode (measured: the mode came out 0.73 instead of 1.37).

    A FLOATING-POINT bound, not a convergence bound: the radiative CGF is
    entire (v <= 1). It is nevertheless the binding one for a physical block
    on its growing side -- a catastrophic photon moves q/p by O(10^3-10^4)
    sigma, so this limit is ~4x tighter than the ionization one.
    """
    if recs is None or len(recs) == 0:
        return np.inf, np.inf
    if pre is None:
        v, dn, eps = rad_step_spectra(recs, spec, vgrid)
        cs = np.asarray(recs, dtype=np.float64)[:, cf_brems_exact.R_CS]
    else:
        v, dn, eps, cs = pre
    a = cs * np.asarray(weights, float)
    live = np.any(dn > 0.0, axis=1)
    out = []
    for sgn in (-1.0, +1.0):
        m = 0.0
        sel = live & (np.sign(a) == sgn)          # theta*a > 0 for theta*sgn>0
        if sel.any():
            m = float(np.max(np.abs(a[sel, None]) * eps[sel]))
        out.append(np.inf if m <= 0 else margin * _LOG_MAX / m)
    return tuple(out)


def rad_theta_overflow_limit(recs, spec, vgrid, weights, margin=0.98, pre=None):
    """min of the two signed bounds (kept for reporting)."""
    return float(min(rad_theta_overflow_limits(recs, spec, vgrid, weights,
                                               margin, pre)))


# =================================================== collectors / block =====
def collect_ms_steps(legs, k, avec, sigma):
    """(steps, weights) for every MS sub-kick feeding plane k.

    A literal transcription of the MS branch of model_phi, INCLUDING the
    MS_NSUB sub-step quadrature: xg is divided by MS_NSUB and the weight is
    interpolated linearly between the step-start and step-end transports.
    """
    A_ms, _, A_ms_start = step_transports(legs, k)
    rows, wts = [], []
    for j in range(k + 1):
        leg = legs[j]
        if not len(leg["ms"]):
            continue
        wv = np.einsum("i,sij->sj", avec, A_ms[j])
        wv0 = np.einsum("i,sij->sj", avec, A_ms_start[j])
        coslam = leg["refpt"] / leg["refp"] if leg["refp"] > 0 else 1.0

        def _weff(v):
            return np.sqrt(v[:, 1] ** 2
                           + (v[:, 2] / max(coslam, 1e-3)) ** 2) / sigma
        weff, weff0 = _weff(wv), _weff(wv0)
        for s in range(len(leg["ms"])):
            if weff[s] <= 0.0 and weff0[s] <= 0.0:
                continue
            rec = leg["ms"][s].astype(np.float64).copy()
            rec[2] /= cpt.MS_NSUB
            for i in range(cpt.MS_NSUB):
                f = (i + 0.5) / cpt.MS_NSUB
                w = weff0[s] + f * (weff[s] - weff0[s])
                if w > 0.0:
                    rows.append(rec)
                    wts.append(cpt.KMS_SCALE * w)
    if not rows:
        return np.zeros((0, 10)), np.zeros(0)
    return np.array(rows), np.array(wts)


def collect_rad_steps(legs, k, avec, sigma):
    """(recs, spec, vgrid, weights) for every radiative step feeding plane k,
    with model_phi's leg-mean fallback when the radv and ioni logs differ in
    length."""
    _, A_ioni, _ = step_transports(legs, k)
    R, S, W = [], [], []
    vg = None
    for j in range(k + 1):
        leg = legs[j]
        if leg.get("rad") is None or not len(leg["rad"]):
            continue
        vg = leg["radvgrid"]
        q = np.sign(leg["refqop"]) or 1.0
        w = q * np.einsum("i,sij->sj", avec, A_ioni[j])[:, 0] / sigma
        wr = w if len(w) == len(leg["rad"]) else np.full(len(leg["rad"]),
                                                         np.mean(w))
        R.append(leg["rad"])
        S.append(leg["radspec"])
        W.append(wr)
    if not R:
        return None, None, None, np.zeros(0)
    return np.concatenate(R), np.concatenate(S), vg, np.concatenate(W)


def collect_ioni_steps(legs, k, avec, sigma):
    """Ionization steps with the transport weight folded into column 10.

    Same as cgf_phase0_validate.collect_ioni_steps; repeated here so this
    module does not import a script.
    """
    _, A_ioni, _ = step_transports(legs, k)
    out = []
    for j in range(k + 1):
        leg = legs[j]
        if not len(leg["ioni"]):
            continue
        q = np.sign(leg["refqop"]) or 1.0
        w = q * np.einsum("i,sij->sj", avec, A_ioni[j])[:, 0] / sigma
        st = leg["ioni"].copy().astype(np.float64)
        st[:, 10] *= w
        out.append(st)
    return np.concatenate(out) if out else np.zeros((0, 11))


def collect_block(legs, k, avec, sigma, theta_cut=THETA_CUT, nmax=None,
                  theta_max=None, snap_ymax=True, channels=("ioni", "ms", "rad")):
    """Everything plane k's block needs, with the MS moment series and the
    radiative spectra precomputed once.

    theta_max sizes the MS series (see ms_series_nmax); it defaults to a few
    times the block's own overflow limit, which is more than the saddlepoint
    ever visits.
    """
    blk = dict(k=k, sigma=float(sigma), channels=tuple(channels),
               theta_cut=float(theta_cut))
    blk["ioni"] = (collect_ioni_steps(legs, k, avec, sigma)
                   if "ioni" in channels else np.zeros((0, 11)))
    if "ms" in channels:
        st, wt = collect_ms_steps(legs, k, avec, sigma)
    else:
        st, wt = np.zeros((0, 10)), np.zeros(0)
    blk["ms_steps"], blk["ms_weights"] = st, wt
    if "rad" in channels:
        R, S, VG, W = collect_rad_steps(legs, k, avec, sigma)
    else:
        R, S, VG, W = None, None, None, np.zeros(0)
    blk["rad"] = (R, S, VG, W)
    blk["rad_pre"] = None
    if R is not None and len(R):
        v, dn, eps = rad_step_spectra(R, S, VG)
        blk["rad_pre"] = (v, dn, eps,
                          np.asarray(R, dtype=np.float64)[:, cf_brems_exact.R_CS])
    # the MS series must exist before the theta limits, because its truncation
    # length is one of the two MS bounds (see block_theta_limits)
    if len(st):
        nm = nmax if nmax else (
            ms_series_nmax(st, wt, theta_max, theta_cut=theta_cut,
                           snap_ymax=snap_ymax) if theta_max is not None
            else MS_NMAX_CAP)
        c2, a2, f2, keep = ms_step_params(st, snap_ymax=snap_ymax)
        blk["ms_logT"] = ms_series_coeffs(c2[keep], a2[keep], f2[keep],
                                          wt[keep], nm, theta_cut=theta_cut)
    else:
        blk["ms_logT"] = np.zeros(0)
    blk["theta_limits"] = block_theta_limits(blk)
    blk["theta_overflow"] = float(min(blk["theta_limits"]))
    blk["theta_max"] = theta_max if theta_max is not None else max(
        blk["theta_limits"])
    return blk


def block_theta_limits(blk, margin=0.98):
    """(bound on -theta, bound on +theta): the tightest double-precision bound
    over the three channels, computed SEPARATELY for the two signs.

    Ionization and radiative each grow on one side only (a delta ray and a
    photon both move q/p the same way); MS is even, so it grows on both. Only
    the growing side may be capped -- see rad_theta_overflow_limits.
    """
    from cgf_saddlepoint import theta_overflow_limit, heavy_tail_sign
    lim = [np.inf, np.inf]                        # [-theta, +theta]
    if len(blk["ioni"]):
        t = theta_overflow_limit(blk["ioni"], margin=margin)
        # heavy_tail_sign is the side on which e^{bw} DECAYS -> the other side
        # is the one that overflows
        grow = -heavy_tail_sign(blk["ioni"])
        lim[1 if grow > 0 else 0] = min(lim[1 if grow > 0 else 0], t)
    R, S, VG, W = blk["rad"]
    if R is not None and len(R):
        lo, hi = rad_theta_overflow_limits(R, S, VG, W, margin=margin,
                                           pre=blk.get("rad_pre"))
        lim[0] = min(lim[0], lo)
        lim[1] = min(lim[1], hi)
    st, wt = blk["ms_steps"], blk["ms_weights"]
    if len(st):
        _, _, _, keep = ms_step_params(st)
        if keep.any():
            amax = float(np.max(np.abs(wt[keep]))) * blk["theta_cut"]
            if amax > 0:
                # two bounds, and the SERIES one usually binds first:
                #   overflow: e^{a |theta|} leaves the double range
                #   series:   nmax terms no longer reach past the turnover at
                #             n ~ a |theta|, so _series_eval returns +inf
                t = margin * _LOG_MAX / amax
                nm = len(blk.get("ms_logT", ()))
                if nm:
                    t = min(t, 0.7 * nm / amax)
                lim[0] = min(lim[0], t)
                lim[1] = min(lim[1], t)
    return float(lim[0]), float(lim[1])


def block_theta_overflow_limit(blk, margin=0.98):
    """min over channels and signs (reporting only)."""
    return float(min(block_theta_limits(blk, margin)))


def block_cgf_derivs(blk, theta, order=2):
    """K ... K^(order) of the FULL block: ionization + MS + radiative.

    CGFs add over independent channels, so this is literally the sum of the
    three -- which is the whole point: no new approximation is introduced by
    combining them.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=np.float64))
    n = order + 1
    tot = [np.zeros_like(theta) for _ in range(n)]
    if len(blk["ioni"]):
        for i, v in enumerate(ioni_cgf_derivs(blk["ioni"], theta, order=order)):
            tot[i] = tot[i] + v
    if len(blk["ms_logT"]):
        for i, v in enumerate(_series_eval(blk["ms_logT"], theta, order)):
            tot[i] = tot[i] + v
    R, S, VG, W = blk["rad"]
    if R is not None and len(R):
        for i, v in enumerate(rad_cgf_derivs(R, S, VG, W, theta, order=order,
                                             pre=blk["rad_pre"])):
            tot[i] = tot[i] + v
    return tuple(tot)


# ============================================== saddlepoint on the block =====
def block_theta_grid(blk, n=2000, lo=1e-8, hi=1e10):
    """Log-spaced theta on BOTH signs, always containing 0, each side capped
    by its OWN overflow bound (block_theta_limits).

    Same construction as cgf_saddlepoint.theta_grid, generalized to three
    channels: the scale is 1/sqrt(K''(0)) of the SUM, and the two sides are
    capped independently because the two one-sided channels (ionization,
    radiative) grow on the same single side.
    """
    _, _, k2 = block_cgf_derivs(blk, 0.0, order=2)
    s = 1.0 / np.sqrt(float(np.atleast_1d(k2)[0]))
    tneg, tpos = blk.get("theta_limits") or block_theta_limits(blk)
    gp = np.geomspace(lo * s, min(hi * s, tpos), n)
    gn = np.geomspace(lo * s, min(hi * s, tneg), n)
    return np.sort(np.concatenate([-gn, [0.0], gp]))


def block_saddlepoint_curve(blk, thetas, order=4):
    """(r, log p, psi) along the saddlepoint curve, indexed by theta."""
    K, K1, K2, K3, K4 = block_cgf_derivs(blk, thetas, order=4)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ok = (np.isfinite(K) & np.isfinite(K1) & np.isfinite(K2)
              & np.isfinite(K3) & (K2 > 0))
        lp = np.where(ok, K - thetas * K1 - 0.5 * np.log(2 * np.pi * K2),
                      -np.inf)
        psi = np.where(ok, thetas + 0.5 * K3 / K2 ** 2, 0.0)
    return np.where(ok, K1, np.nan), lp, psi, K2, ok


def block_mode(blk, n=4001):
    """Mode of the full block's loss-deviation density, in z units."""
    th = block_theta_grid(blk, n=(n - 1) // 2)
    r, lp, _, _, _ = block_saddlepoint_curve(blk, th)
    if not np.any(np.isfinite(lp)):
        return np.nan
    i = int(np.nanargmax(np.where(np.isfinite(lp), lp, -np.inf)))
    if 0 < i < len(r) - 1 and np.all(np.isfinite(lp[i - 1:i + 2])):
        y0, y1, y2 = lp[i - 1], lp[i], lp[i + 1]
        den = y0 - 2 * y1 + y2
        if den != 0 and np.isfinite(r[i - 1]) and np.isfinite(r[i + 1]):
            h = 0.5 * (r[i + 1] - r[i - 1])
            return float(r[i] - 0.5 * h * (y2 - y0) / den)
    return float(r[i])


def block_fisher_spa(blk, n=4000, lo=1e-8):
    """(I, diagnostics) from the saddlepoint density along the theta curve,
    the exact analogue of cgf_fisher.fisher_spa for the full block.

    DO NOT TRUST THIS FOR A MULTI-CHANNEL BLOCK (NOTES 2026-08-13 XXII).
    I of these blocks is EDGE dominated -- 86 % of it comes from the last
    ~20 % of the mass, at the delta-ray hard edge -- and the saddlepoint
    reaches that edge only as theta -> infinity, where the MS term takes over
    the asymptotics of K and fabricates support out to r ~ 200. Measured at
    plane 18: adding MS, worth 4.7e-5 of the variance, moves the SPA 1/I from
    1.70 to 1.14 while the EXACT I moves by 0.01 %; and the answer depends on
    the MS series length (nmax = 60 -> 3.68, nmax >= 200 -> 1.14), a knob that
    must be irrelevant. A relative density floor -- the fix for the 57x error
    of NOTES XV -- does NOT repair it (floors 1e-6 and 1e-8 return the
    unfloored value).

    Use `fisher_exact` on `invert_cf(..., deriv=True)` instead. This function
    is kept because it is the like-for-like continuation of cgf_fisher's route
    A and is what demonstrates the failure.
    """
    th = block_theta_grid(blk, n=n, lo=lo)
    r, lp, psi, K2, ok = block_saddlepoint_curve(blk, th)
    with np.errstate(over="ignore", invalid="ignore"):
        meas = np.where(ok, np.exp(lp) * K2, 0.0)
    meas = np.where(np.isfinite(meas), meas, 0.0)
    sel = meas > 0.0
    integ = np.zeros_like(meas)
    with np.errstate(over="ignore", invalid="ignore"):
        integ[sel] = psi[sel] ** 2 * meas[sel]
    nbad = int(np.sum(~np.isfinite(integ[sel])))
    if nbad:
        raise FloatingPointError(f"{nbad} non-finite integrand values")
    Z = float(np.trapezoid(meas, th))
    I = float(np.trapezoid(integ, th))
    return (I / Z if Z > 0 else np.nan,
            dict(Z=Z, rmin=float(np.nanmin(r[sel])) if sel.any() else np.nan,
                 rmax=float(np.nanmax(r[sel])) if sel.any() else np.nan,
                 thmin=float(th[sel].min()) if sel.any() else np.nan,
                 thmax=float(th[sel].max()) if sel.any() else np.nan))


# ==================================================== exact CF reference =====
def block_cf_exponent(legs, k, avec, sigma, tau, channels=("ioni", "ms", "rad")):
    """The trusted imaginary-argument exponent of model_phi, per channel
    switchable. Verbatim from cf_propagation_test.model_phi."""
    from cf_track_resolution import ioni_step_exponent, ms_step_exponent
    A_ms, A_ioni, A_ms_start = step_transports(legs, k)
    S = np.zeros(len(tau), dtype=np.complex128)
    for j in range(k + 1):
        leg = legs[j]
        if "ioni" in channels and len(leg["ioni"]):
            q = np.sign(leg["refqop"]) or 1.0
            w = q * np.einsum("i,sij->sj", avec, A_ioni[j])[:, 0] / sigma
            st = leg["ioni"].copy()
            st[:, 10] *= w
            S += ioni_step_exponent(st, 1.0, tau)
        if ("rad" in channels and leg.get("rad") is not None
                and len(leg["rad"])):
            q = np.sign(leg["refqop"]) or 1.0
            w = q * np.einsum("i,sij->sj", avec, A_ioni[j])[:, 0] / sigma
            wr = (w if len(w) == len(leg["rad"])
                  else np.full(len(leg["rad"]), np.mean(w)))
            S += cf_brems_exact.rad_exponent(tau, leg["rad"], leg["radspec"],
                                             leg["radvgrid"], weights=wr)
        if "ms" in channels and len(leg["ms"]):
            wv = np.einsum("i,sij->sj", avec, A_ms[j])
            wv0 = np.einsum("i,sij->sj", avec, A_ms_start[j])
            coslam = leg["refpt"] / leg["refp"] if leg["refp"] > 0 else 1.0

            def _weff(v):
                return np.sqrt(v[:, 1] ** 2
                               + (v[:, 2] / max(coslam, 1e-3)) ** 2) / sigma
            weff, weff0 = _weff(wv), _weff(wv0)
            for s in range(len(leg["ms"])):
                if weff[s] <= 0.0 and weff0[s] <= 0.0:
                    continue
                rec = leg["ms"][s:s + 1].copy()
                rec[:, 2] /= cpt.MS_NSUB
                for i in range(cpt.MS_NSUB):
                    f = (i + 0.5) / cpt.MS_NSUB
                    w = weff0[s] + f * (weff[s] - weff0[s])
                    if w > 0.0:
                        S += cpt.KMS_SCALE * ms_step_exponent(rec, w, tau)
    return S


_REACH_CACHE = {}


def tau_reach(legs, k, avec, sigma, channels=("ioni", "ms", "rad"),
              lncut=-60.0, lo=1e-3, hi=1e6, n=400):
    """t at which Re S(t) = lncut, i.e. |phi| = e^{lncut}.

    Bisected on a coarse log scan. The contribution of t > t_cut to p(z) is
    bounded by (1/pi) int |phi| dt, so lncut = -60 (|phi| = 9e-27) is far below
    any relative floor used downstream.

    THIS MUST BE DONE PER MOMENTUM. The pT = 3 grid (TAU_REF, t <= 40) reaches
    only e^{-13} at plane 0 and is not adequate at pT = 40 or 100, where the
    radiative channel is 20-80 % of the ionization one and the CF decays
    differently.
    """
    key = (k, tuple(channels), float(lncut), float(sigma))
    if key in _REACH_CACHE:
        return _REACH_CACHE[key]
    t = np.geomspace(lo, hi, n)
    S = block_cf_exponent(legs, k, avec, sigma, t, channels=channels).real
    below = np.where(S < lncut)[0]
    if not len(below):
        out = float(hi)
    elif int(below[0]) == 0:
        out = float(lo)
    else:
        i = int(below[0])
        a, b = t[i - 1], t[i]
        # the coarse grid already brackets to one log step of 400 over 9
        # decades (factor 1.053); four refinements are ample, since the caller
        # multiplies by 1.3 anyway
        for _ in range(4):
            m = np.sqrt(a * b)
            sv = float(block_cf_exponent(legs, k, avec, sigma, np.array([m]),
                                         channels=channels).real[0])
            if sv < lncut:
                b = m
            else:
                a = m
        out = float(b)
    _REACH_CACHE[key] = out
    return out


def auto_tau(legs, k, avec, sigma, channels=("ioni", "ms", "rad"),
             lncut=-60.0, n=8000, factor=1.3, tlo=1e-4):
    """Log-spaced t grid matched to THIS block: dense enough for the three
    decades of structure, long enough that |phi| has underflowed at the end."""
    top = factor * tau_reach(legs, k, avec, sigma, channels=channels,
                             lncut=lncut)
    return np.concatenate([[0.0], np.geomspace(tlo, top, n)])


def exact_density(legs, k, avec, sigma, channels=("ioni", "ms", "rad"),
                  tau=None, nt=1 << 17, npad=32, deriv=True, lncut=-60.0):
    """(z, p, dp) of the block by FFT inversion, with the t grid, the uniform
    resolution and the zero padding all matched to this block.

    nt fixes z_max = pi nt / tau_max (aliasing), npad fixes
    dz = 2 pi / (npad tau_max) (peak resolution). Both are scanned in the
    report rather than assumed.
    """
    if tau is None:
        tau = auto_tau(legs, k, avec, sigma, channels=channels, lncut=lncut)
    S = block_cf_exponent(legs, k, avec, sigma, tau, channels=channels)
    return invert_cf(S, tau, npad=npad, nt=nt, deriv=deriv)


def invert_cf(S, tau, npad=64, nt=1 << 16, deriv=False):
    """(z, p[, dp/dz]) by FFT inversion of exp(S) given on the log grid `tau`.

    The uniform t grid is matched to `tau` (dt = tau[-1]/nt) and then ZERO
    PADDED by npad, which refines dz = 2 pi / (npad tau[-1]) without inventing
    any phi beyond where it was computed. Matching the grid to the thing being
    inverted, and only then refining, is the rule this study keeps relearning.

    dp/dz comes from the same transform with phi -> -i t phi, so no numerical
    differencing of a density (or of its log) ever enters.
    """
    tu = np.linspace(0.0, tau[-1], nt)
    Su = np.interp(tu, tau, S.real) + 1j * np.interp(tu, tau, S.imag)
    with np.errstate(over="ignore"):
        phi = np.exp(Su)
    dt = tu[1] - tu[0]
    wgt = np.full(nt, dt)
    wgt[0] = 0.5 * dt
    c = np.zeros(nt * npad, dtype=np.complex128)
    c[:nt] = phi * wgt
    p = np.fft.fft(c).real / np.pi
    z = 2 * np.pi * np.fft.fftfreq(nt * npad, d=dt)
    # ascending-z order. `np.argsort(z)` is what this used to be, and on the
    # 4.2M-point padded grid it cost more than the FFT itself (5.8 s of the
    # 496 s `geom_closure closure` run). fftfreq's output is
    # [0..n/2-1, -n/2..-1], whose ascending permutation is exactly the roll
    # that `fftshift` applies -- same indices, no comparison sort. Checked
    # element-by-element against argsort on the grids this code uses.
    if not deriv:
        return np.fft.fftshift(z), np.fft.fftshift(p)
    c2 = np.zeros_like(c)
    c2[:nt] = phi * wgt * (-1j * tu)
    dp = np.fft.fft(c2).real / np.pi
    return np.fft.fftshift(z), np.fft.fftshift(p), np.fft.fftshift(dp)


def mode_from_density(z, p):
    i = int(np.argmax(p))
    if 0 < i < len(z) - 1:
        y0, y1, y2 = p[i - 1], p[i], p[i + 1]
        den = y0 - 2 * y1 + y2
        if den != 0:
            return float(z[i] - 0.5 * (z[1] - z[0]) * (y2 - y0) / den)
    return float(z[i])


def _support(p, floor):
    """Largest CONTIGUOUS run with p > floor*max(p), containing the mode.
    Same as cgf_fisher._support: a non-contiguous mask makes np.trapezoid
    stitch across gaps that are not in the support.

    The two element-at-a-time walks this replaces are the same slice: the run
    ends at the nearest index on each side that FAILS `p > thr`, so those
    indices come straight out of `flatnonzero(~mask)`. NaNs fail the
    comparison and so still terminate the run, exactly as `p[i] > thr` did.
    Vectorized because the walk was 4.6 s of the 496 s `geom_closure closure`
    run -- the padded grid is 4.2M points and the support covers most of it."""
    i0 = int(np.nanargmax(p))
    thr = floor * np.nanmax(p)
    with np.errstate(invalid="ignore"):
        bad = np.flatnonzero(~(p > thr))
    j = np.searchsorted(bad, i0)
    lo = int(bad[j - 1]) + 1 if j > 0 else 0
    hi = int(bad[j]) - 1 if j < len(bad) else len(p) - 1
    return slice(lo, hi + 1)


def fisher_exact(z, p, dp, floor=1e-8):
    """I = int (p')^2/p dz on the contiguous support, normalized by the mass
    there. RELATIVE floor: an absolute one was a 57x error in this study."""
    s = _support(p, floor)
    if s.stop - s.start < 10:
        return np.nan, {}
    I = float(np.trapezoid(dp[s] ** 2 / p[s], z[s]))
    mass = float(np.trapezoid(p[s], z[s]))
    tot = float(np.trapezoid(np.maximum(p, 0.0), z))
    return I / mass, dict(mass=mass, mass_frac=mass / tot,
                          zlo=float(z[s][0]), zhi=float(z[s][-1]))


def _spa_mass(blk, nth=20000):
    """(measure, theta) whose trapezoid is Z = int p_SPA dr -- the mass the
    saddlepoint density carries, which is NOT 1."""
    th = block_theta_grid(blk, n=nth // 2)
    r, lp, _, K2, ok = block_saddlepoint_curve(blk, th)
    with np.errstate(over="ignore", invalid="ignore"):
        meas = np.where(ok, np.exp(lp) * K2, 0.0)
    return np.where(np.isfinite(meas), meas, 0.0), th


def spa_density(blk, z, nth=20000):
    """SPA density on a given z grid, by interpolating the parametric
    saddlepoint curve (r(theta), log p(theta)) -- r is monotone in theta
    because K'' > 0, so this is a legitimate 1D interpolation (unlike psi,
    which is NOT monotone and must never be inverted this way)."""
    th = block_theta_grid(blk, n=nth // 2)
    r, lp, _, _, ok = block_saddlepoint_curve(blk, th)
    m = ok & np.isfinite(r) & np.isfinite(lp)
    r, lp = r[m], lp[m]
    o = np.argsort(r)
    r, lp = r[o], lp[o]
    out = np.full(len(z), -np.inf)
    inside = (z >= r[0]) & (z <= r[-1])
    out[inside] = np.interp(z[inside], r, lp)
    return np.exp(out), (float(r[0]), float(r[-1]))


# ===================================================================== tests ==
try:
    import mpmath as mp
    HAVE_MP = True
except ImportError:                                  # pragma: no cover
    HAVE_MP = False


def _relerr(a, b):
    if b is None or a is None:
        return np.nan
    if not np.isfinite(a) or not np.isfinite(b):
        return 0.0 if (np.isfinite(a) == np.isfinite(b)) else np.inf
    return abs(a) if b == 0 else abs(a - b) / abs(b)


def _i0_deriv_mp(u, m):
    """d^m I0(u)/du^m = 2^{-m} sum_j C(m,j) I_{|m-2j|}(u), in mpmath."""
    if m == 0:
        return mp.besseli(0, u)
    s = mp.mpf(0)
    for j in range(m + 1):
        s += mp.binomial(m, j) * mp.besseli(abs(m - 2 * j), u)
    return s / mp.mpf(2) ** m


def ms_cgf_mp(chic2, chia2, thff2, w, theta, order=0, theta_cut=THETA_CUT,
              ff_pow=None, dps=50):
    """mpmath reference for the MS CGF and its derivatives, from the DEFINING
    integral (no series, no quadrature grid of ours):

        K(theta)   = sum_s chic2_s int_0^{Tc} (I0(w_s theta sqrt t) - 1) g_s dt
        K^(m)      = sum_s chic2_s w_s^m int_0^{Tc} t^{m/2}
                     I0^(m)(w_s theta sqrt t) g_s dt
        g_s(t)     = 1/(t + chia2_s)^2 / (1 + t/thff2_s)^p
    """
    if ff_pow is None:
        ff_pow = 4 if G4_FF_SQUARED else 2
    with mp.workdps(dps):
        Tc = mp.mpf(theta_cut) ** 2
        tot = mp.mpf(0)
        for c2, a2, f2, ws in zip(np.atleast_1d(chic2), np.atleast_1d(chia2),
                                  np.atleast_1d(thff2), np.atleast_1d(w)):
            c2, a2, f2, ws = (mp.mpf(float(c2)), mp.mpf(float(a2)),
                              mp.mpf(float(f2)), mp.mpf(float(ws)))
            th = mp.mpf(float(theta))

            def f(t, c2=c2, a2=a2, f2=f2, ws=ws, th=th):
                g = 1 / (t + a2) ** 2 / (1 + t / f2) ** ff_pow
                u = ws * th * mp.sqrt(t)
                if order == 0:
                    return c2 * (mp.besseli(0, u) - 1) * g
                return (c2 * ws ** order * t ** (mp.mpf(order) / 2)
                        * _i0_deriv_mp(u, order) * g)
            # split at the two scales of g (chi_a^2 and theta_FF^2) so the
            # quadrature never has to resolve them on one interval
            pts = sorted({mp.mpf(0), a2, f2, Tc})
            tot += mp.quad(f, pts)
        return float(tot)


def rad_cgf_mp(v, dn, eps, a, theta, order=0, dps=50):
    """mpmath reference for the radiative CGF, evaluated with the SAME
    trapezoid rule on the SAME 48-point v grid as rad_cgf_derivs and
    cf_brems_exact.rad_exponent.

    That is deliberate. The v spectrum is a TABLE, so there is no continuous
    'defining integral' to integrate independently; what can fail in float64
    is the cancellation in (e^x - 1 - x) and the accumulation over 400 steps,
    and this reference tests exactly that at 50 digits. The quadrature itself
    is tested separately against an analytic 1/v spectrum (rad_selftest_mp).
    """
    with mp.workdps(dps):
        th = mp.mpf(float(theta))
        tot = mp.mpf(0)
        for i in range(len(dn)):
            if not np.any(dn[i] > 0):
                continue
            ai = mp.mpf(float(a[i]))
            vals = []
            for j in range(len(v)):
                x = th * ai * mp.mpf(float(eps[i][j]))
                e = mp.mpf(float(eps[i][j]))
                if order == 0:
                    y = mp.e ** x - 1 - x
                elif order == 1:
                    y = ai * e * (mp.e ** x - 1)
                else:
                    y = ai ** order * e ** order * mp.e ** x
                vals.append(y * mp.mpf(float(dn[i][j])))
            s = mp.mpf(0)
            for j in range(len(v) - 1):
                s += (vals[j] + vals[j + 1]) / 2 * (mp.mpf(float(v[j + 1]))
                                                    - mp.mpf(float(v[j])))
            tot += s
        return float(tot)


def rad_selftest_mp(theta, order=0, A=1.0, v0=1e-6, E=3.0, cs=0.1, w=1.0,
                    nv=20001, dps=50):
    """(code, mpmath) for an ANALYTIC radiative spectrum dN/dv = A/v on
    [v0, 1] -- the physical brems shape, and a case whose integral mpmath can
    do continuously. Tests the quadrature and the whole weight chain, not just
    the per-node arithmetic."""
    v = np.geomspace(v0, 1.0, nv)
    dn = A / v
    eps = v * E
    pre = (v, dn[None, :], eps[None, :], np.array([cs]))
    got = rad_cgf_derivs(None, None, None, np.array([w]), theta, order=order,
                         pre=pre)[order]
    with mp.workdps(dps):
        b = mp.mpf(float(theta)) * mp.mpf(cs) * mp.mpf(w)
        Em = mp.mpf(E)

        def f(vv):
            x = b * vv * Em
            if order == 0:
                y = mp.e ** x - 1 - x
            elif order == 1:
                y = (mp.mpf(cs) * w) * vv * Em * (mp.e ** x - 1)
            else:
                y = (mp.mpf(cs) * w) ** order * (vv * Em) ** order * mp.e ** x
            return y * mp.mpf(A) / vv
        ref = float(mp.quad(f, [mp.mpf(v0), mp.mpf(1)]))
    return float(np.atleast_1d(got)[0]), ref


def fd_cumulants_from_cf(cf_fun, h, mmax=4):
    """kappa_2..kappa_mmax of a distribution from central differences of its
    log-CF at t = 0. cf_fun(t_array) -> complex exponent S(t)."""
    o = np.arange(-4, 5) * h
    S = cf_fun(o)
    S = np.asarray(S, dtype=np.complex128)
    # central-difference stencils for the 2nd..4th derivative, 4th order
    d2 = (-S[0] / 560 + 8 * S[1] / 315 - S[2] / 5 + 8 * S[3] / 5
          - 205 * S[4] / 72 + 8 * S[5] / 5 - S[6] / 5 + 8 * S[7] / 315
          - S[8] / 560) / h ** 2
    d3 = (-7 * S[0] / 240 + 3 * S[1] / 10 - 169 * S[2] / 120 + 61 * S[3] / 30
          - 61 * S[5] / 30 + 169 * S[6] / 120 - 3 * S[7] / 10
          + 7 * S[8] / 240) / h ** 3
    d4 = (7 * S[0] / 240 - 2 * S[1] / 5 + 169 * S[2] / 60 - 122 * S[3] / 15
          + 91 * S[4] / 8 - 122 * S[5] / 15 + 169 * S[6] / 60 - 2 * S[7] / 5
          + 7 * S[8] / 240) / h ** 4
    return {2: (d2 / 1j ** 2), 3: (d3 / 1j ** 3), 4: (d4 / 1j ** 4)}


def fd_deriv(f, x, h, n=1):
    """8th/6th-order central difference of a scalar function, for checking
    K''' and K'''' against numerical derivatives of K''."""
    o = np.arange(-4, 5) * h
    y = np.asarray([f(x + d) for d in o], dtype=np.float64)
    if n == 1:
        return (y[0] / 280 - 4 * y[1] / 105 + y[2] / 5 - 4 * y[3] / 5
                + 4 * y[5] / 5 - y[6] / 5 + 4 * y[7] / 105 - y[8] / 280) / h
    if n == 2:
        return (-y[0] / 560 + 8 * y[1] / 315 - y[2] / 5 + 8 * y[3] / 5
                - 205 * y[4] / 72 + 8 * y[5] / 5 - y[6] / 5 + 8 * y[7] / 315
                - y[8] / 560) / h ** 2
    raise ValueError(n)


# =============================================================== validation ==
def validate(model, planes=(0, 9, 18), func="qop", theta_cut=THETA_CUT):
    """The acceptance test for both new channels and for the combined block."""
    legs = cpt.load_model(model)
    avec = FUNCTIONALS[func]
    print(f"model {os.path.basename(model)}, func {func}, "
          f"theta_cut = {theta_cut:.4f} rad, mpmath {'yes' if HAVE_MP else 'NO'}")

    # ---------------------------------------------------------------- MS: mp
    print("\n" + "=" * 96)
    print("V1. MS: truncated angular moments mu~_n against mpmath (50 dps) on "
          "the defining integral")
    print("=" * 96)
    chia2, thff2 = 2.4e-11, 5.2e-3
    lmu = ms_log_moments([chia2], [thff2], 40)[0]
    ff = 4 if G4_FF_SQUARED else 2
    print(f"  chi_a^2 = {chia2:.3g}, theta_FF^2 = {thff2:.3g}, FF power {ff}")
    print(f"  {'n':>4} {'ln mu~ code':>18} {'ln mu~ mpmath':>18} {'rel err':>10}")
    worst = 0.0
    if HAVE_MP:
        for n in (1, 2, 3, 4, 5, 8, 12, 20, 40):
            with mp.workdps(50):
                Tc = mp.mpf(theta_cut) ** 2
                f = lambda t: (t ** n / (t + mp.mpf(chia2)) ** 2
                               / (1 + t / mp.mpf(thff2)) ** ff)
                lref = float(mp.log(mp.quad(f, [0, mp.mpf(chia2),
                                                mp.mpf(thff2), Tc])))
            e = abs(lmu[n - 1] - lref)
            worst = max(worst, e)
            print(f"  {n:4d} {lmu[n-1]:18.10f} {lref:18.10f} {e:10.2e}")
        print(f"  worst |delta ln mu~| = {worst:.2e}  (= relative error on mu~)")

    print("\n" + "=" * 96)
    print("V2. MS: K ... K'''' against mpmath on the defining integral, BOTH "
          "signs of theta")
    print("=" * 96)
    c2 = np.array([3.0e-9, 1.2e-8, 5.0e-9])
    a2 = np.array([2.4e-11, 1.1e-11, 4.0e-11])
    f2 = np.array([5.2e-3, 9.0e-3, 3.0e-3])
    ws = np.array([2.4, 0.9, 4.7])
    logT = ms_series_coeffs(c2, a2, f2, ws, 200, theta_cut=theta_cut)
    thl = (-15.0, -3.0, -0.3, -0.01, 0.01, 0.3, 3.0, 15.0)
    print(f"  {'theta':>8} " + " ".join(f"{'rel K'+chr(39)*m:>10}"
                                        for m in range(5)))
    wmax = np.zeros(5)
    if HAVE_MP:
        for th in thl:
            v = ms_cgf_derivs(None, None, th, order=4, logT=logT)
            row = []
            for m in range(5):
                r = _relerr(float(np.atleast_1d(v[m])[0]),
                            ms_cgf_mp(c2, a2, f2, ws, th, order=m,
                                      theta_cut=theta_cut, dps=40))
                row.append(r)
                wmax[m] = max(wmax[m], r)
            print(f"  {th:8.3g} " + " ".join(f"{r:10.2e}" for r in row))
        print("  worst per order: " + " ".join(f"K{chr(39)*m}={wmax[m]:.1e}"
                                               for m in range(5)))
    print("  (K is EVEN in theta and K' ODD -- the projected MS kick is "
          "symmetric, so\n   K'(0) = 0 holds by construction, not by "
          "subtraction.)")

    # ------------------------------------------------------------- MS: entire
    print("\n" + "=" * 96)
    print("V3. MS: is the CGF entire?  moment growth with the angular cutoff")
    print("=" * 96)
    print("  mu~_n for the SAME step at four cutoffs. The FF-squared tail is")
    print("  dn/dt ~ t^-6, so mu~_n must diverge as theta_cut -> inf for "
          "n >= 5 and\n  converge for n <= 4. That is the whole question: "
          "with no cutoff the moment\n  series -- hence K(theta) -- is +inf "
          "for every theta != 0.")
    cuts = (0.3, 1.0, np.pi, 10.0, 100.0)
    print(f"  {'n':>3} " + " ".join(f"{'cut='+f'{c:g}':>13}" for c in cuts)
          + f" {'ratio 100/pi':>13}")
    for n in (1, 2, 3, 4, 5, 6, 8):
        row = [float(ms_log_moments([chia2], [thff2], n, theta_cut=c)[0][-1])
               for c in cuts]
        print(f"  {n:3d} " + " ".join(f"{np.exp(v):13.4e}" for v in row)
              + f" {np.exp(row[-1] - row[2]):13.4e}")
    print("  -> n <= 4 flat (converged), n >= 5 grows like theta_cut^(2n-10):")
    print("     the CGF is entire ONLY with a finite maximum deflection.")

    # ---------------------------------------------------------------- RAD: mp
    print("\n" + "=" * 96)
    print("V4. RADIATIVE: analytic dN/dv = A/v spectrum, continuous mpmath "
          "reference")
    print("=" * 96)
    print("  Tests the quadrature AND the whole weight chain on a case whose "
          "integral\n  mpmath can do in closed form. x_max = theta cs w E is "
          "the exponent argument.")
    print(f"  {'x_max':>10} " + " ".join(f"{'rel K'+chr(39)*m:>10}"
                                         for m in range(5)))
    if HAVE_MP:
        for th in (-2300.0, -100.0, -1.0, -1e-3, -1e-6, 1e-6, 1e-3, 1.0,
                   100.0, 2300.0):
            row = [_relerr(*rad_selftest_mp(th, order=m, nv=200001))
                   for m in range(5)]
            print(f"  {th*0.3:10.3g} " + " ".join(f"{r:10.2e}" for r in row))
        print("  The floor at large |x| is the TRAPEZOID's own error on the "
              "geomspace v grid\n  (verified: it falls exactly as 1/nv^2), "
              "not the CGF evaluation -- see V5.")

    print("\n" + "=" * 96)
    print("V5. RADIATIVE: the REAL block, same 48-point v grid, against "
          "mpmath at 50 dps")
    print("=" * 96)
    print("  The v spectrum is a TABLE, so this is the only like-for-like "
          "reference: same\n  grid, same trapezoid, 50 digits. It isolates "
          "the float64 arithmetic -- the\n  (e^x - 1 - x) cancellation and "
          "the accumulation over hundreds of steps.")
    for k in planes:
        var, _, _ = model_variance(legs, k, avec)
        sig = float(np.sqrt(var))
        R, S, VG, W = collect_rad_steps(legs, k, avec, sig)
        v, dn, eps = rad_step_spectra(R, S, VG)
        cs = np.asarray(R, dtype=np.float64)[:, cf_brems_exact.R_CS]
        pre = (v, dn, eps, cs)
        tn, tp = rad_theta_overflow_limits(R, S, VG, W, pre=pre)
        # one of the two is +inf when every step's jump has the same sign
        # (it does): the exponential decays on that whole side, so probe it at
        # the mirror of the finite bound instead of at infinity.
        fin = min(t for t in (tn, tp) if np.isfinite(t))
        tn = fin if not np.isfinite(tn) else tn
        tp = fin if not np.isfinite(tp) else tp
        row = []
        ths = [-0.5 * tn, -1e-3 * tn, 1e-3 * tp, 0.5 * tp]
        if HAVE_MP:
            for th in ths:
                got = rad_cgf_derivs(None, None, None, W, th, order=4,
                                     pre=pre)
                rr = [_relerr(float(np.atleast_1d(got[m])[0]),
                              rad_cgf_mp(v, dn, eps, cs * W, th, order=m,
                                         dps=50)) for m in range(5)]
                row.append(max(rr))
            print(f"  plane {k:2d}: theta in "
                  f"[{-0.5*tn:+.3g}, {0.5*tp:+.3g}]  worst rel err over "
                  f"K..K'''' and 4 theta = {max(row):.2e}")

    # ------------------------------------------- continuation from the CF
    print("\n" + "=" * 96)
    print("V6. Both channels ARE the analytic continuation of the CF the code "
          "already uses")
    print("=" * 96)
    print("  kappa_m = K^(m)(0) from the new REAL-argument code, against "
          "kappa_m read off\n  the TRUSTED IMAGINARY-argument exponent by "
          "central differences at t = 0\n  (S(t) = sum kappa_m (it)^m/m!). "
          "Independent code paths: gshape's tabulated\n  Moliere quadrature "
          "and cf_brems_exact.rad_exponent on one side, the moment\n  series "
          "and the direct v quadrature on the other.")
    print("  The FD step is NOT free. These CFs are dominated by their high "
          "cumulants --\n  the radiative jump reaches 3e4 z units, so "
          "kappa4/kappa2 ~ 1e12 and any h that\n  looks reasonable for a "
          "kappa2 of 19 is contaminated by a factor 1e7. h is\n  therefore "
          "set from kappa4/kappa2 so that the kappa4 h^2/12 leakage is 1e-4.")
    print(f"  {'plane':>5} {'chan':>5} {'h':>10} {'kappa2':>13} "
          f"{'rel k2':>9} {'rel k3':>9} {'rel k4':>9}")
    for k in planes:
        var, _, _ = model_variance(legs, k, avec)
        sig = float(np.sqrt(var))
        for name in ("ms", "rad"):
            blk = collect_block(legs, k, avec, sig, theta_cut=theta_cut,
                                channels=(name,))
            g = block_cgf_derivs(blk, 0.0, order=4)
            K2 = float(np.atleast_1d(g[2])[0])
            K4 = float(np.atleast_1d(g[4])[0])
            h = np.sqrt(12e-4 * K2 / K4) if K4 > 0 else 1e-3
            cf = lambda t: block_cf_exponent(legs, k, avec, sig,
                                             np.asarray(t, float),
                                             channels=(name,))
            ref = fd_cumulants_from_cf(cf, h)
            rr = [_relerr(float(np.atleast_1d(g[m])[0]),
                          float(np.real(ref[m]))) for m in (2, 3, 4)]
            print(f"  {k:5d} {name:>5} {h:10.3e} {K2:13.5e} "
                  + " ".join(f"{r:9.2e}" for r in rr))
    print("  MS floors at ~2e-3: that is gshape's TABLE (see V7), not the "
          "series.\n  Radiative kappa4 is limited by rad_exponent's small-|x| "
          "branch, which is\n  truncated at x^3 and so has no x^4 term to "
          "give.")

    print("\n" + "=" * 96)
    print("V7. MS at IMAGINARY argument: moment series vs gshape table vs a "
          "fine quadrature")
    print("=" * 96)
    print("  S(t) = sum_n (-1)^n T_n t^{2n} (mine, alternating -- only usable "
          "while\n  w theta_cut t <~ 20, beyond which it cancels), "
          "cf_track_resolution.\n  ms_step_exponent (the production CF), and a "
          "6000-node direct J0 quadrature\n  of the same defining integral. "
          "The third column says which of the first two\n  is right.")
    for k in planes:
        var, _, _ = model_variance(legs, k, avec)
        sig = float(np.sqrt(var))
        st, wt = collect_ms_steps(legs, k, avec, sig)
        blk = collect_block(legs, k, avec, sig, theta_cut=theta_cut,
                            channels=("ms",))
        _, _, _, keep = ms_step_params(st)
        amax = float(np.max(np.abs(wt[keep]))) * theta_cut
        tg = np.geomspace(0.05, min(10.0, 15.0 / amax), 5)
        mine = ms_cf_exponent_series(None, None, tg, logT=blk["ms_logT"])
        prod = block_cf_exponent(legs, k, avec, sig, tg,
                                 channels=("ms",)).real
        quad = ms_cf_exponent_quad(st, wt, tg, theta_cut=theta_cut)
        print(f"  plane {k:2d} (series valid to t = {15.0/amax:.2f}):")
        print(f"    {'t':>8} {'series':>14} {'gshape':>14} {'quad':>14} "
              f"{'ser/quad-1':>11} {'gsh/quad-1':>11}")
        for t, a, b, c in zip(tg, mine, prod, quad):
            print(f"    {t:8.3f} {a:14.6e} {b:14.6e} {c:14.6e} "
                  f"{a/c-1:11.2e} {b/c-1:11.2e}")

    # ------------------------------------------------- K''' / K'''' by FD
    print("\n" + "=" * 96)
    print("V8. K''' and K'''' against central differences of K''  (all three "
          "channels + block)")
    print("=" * 96)
    for k in planes:
        var, _, _ = model_variance(legs, k, avec)
        sig = float(np.sqrt(var))
        for chans in (("ioni",), ("ms",), ("rad",), ("ioni", "ms", "rad")):
            blk = collect_block(legs, k, avec, sig, theta_cut=theta_cut,
                                channels=chans)
            tn, tp = blk["theta_limits"]
            th = 0.1 * min(tn, tp)
            f2 = lambda t: float(np.atleast_1d(block_cgf_derivs(blk, t,
                                                                order=2)[2])[0])
            h = 1e-3 * abs(th)
            _, _, _, k3, k4 = block_cgf_derivs(blk, th, order=4)
            r3 = _relerr(float(np.atleast_1d(k3)[0]), fd_deriv(f2, th, h, 1))
            r4 = _relerr(float(np.atleast_1d(k4)[0]), fd_deriv(f2, th, h, 2))
            print(f"  plane {k:2d} {'+'.join(chans):16s} theta={th:+.4g}  "
                  f"rel K'''={r3:.2e}  rel K''''={r4:.2e}")


# ===================================================================== main ===
_tau_cache = {}


def _inv(legs, k, avec, sigma, channels, args):
    """Cached exact inversion on a grid matched to THIS (plane, channels)."""
    key = (k, tuple(channels))
    if key not in _tau_cache:
        tau = auto_tau(legs, k, avec, sigma, channels=channels,
                       lncut=args.lncut)
        S = block_cf_exponent(legs, k, avec, sigma, tau, channels=channels)
        _tau_cache[key] = invert_cf(S, tau, npad=args.npad, nt=args.nt,
                                    deriv=True)
    return _tau_cache[key]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--planes", default="0,9,18")
    ap.add_argument("--func", default="qop")
    ap.add_argument("--nth", type=int, default=4000)
    ap.add_argument("--theta-cut", type=float, default=THETA_CUT)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--nt", type=int, default=1 << 17,
                    help="uniform t samples for the FFT (sets z_max)")
    ap.add_argument("--npad", type=int, default=32,
                    help="zero padding factor (sets dz)")
    ap.add_argument("--lncut", type=float, default=-60.0,
                    help="Re S at which the reference t grid is cut")
    args = ap.parse_args()

    planes0 = [int(x) for x in args.planes.split(",")]
    if args.validate:
        validate(args.model, planes=planes0, func=args.func,
                 theta_cut=args.theta_cut)
        return

    legs = cpt.load_model(args.model)
    avec = FUNCTIONALS[args.func]
    planes = [int(x) for x in args.planes.split(",")]

    print(f"model {os.path.basename(args.model)}, func {args.func}, "
          f"{len(legs)} planes, theta_cut = {args.theta_cut:.4f} rad\n")

    print("=" * 100)
    print("1. CHANNEL CGFs AT theta = 0 (kappa2 per channel, z units) and the "
          "overflow bounds")
    print("=" * 100)
    print(f"{'plane':>5} {'k2 ioni':>12} {'k2 ms':>12} {'k2 rad':>12} "
          f"{'k2 total':>12} | {'th_of ioni':>10} {'th_of ms':>10} "
          f"{'th_of rad':>10} | {'block -th':>10} {'block +th':>10}")
    blocks = {}
    from cgf_saddlepoint import theta_overflow_limit
    for k in planes:
        var, _, _ = model_variance(legs, k, avec)
        sig = float(np.sqrt(var))
        blk = collect_block(legs, k, avec, sig, theta_cut=args.theta_cut)
        blocks[k] = blk
        k2i = float(ioni_cgf_derivs(blk["ioni"], 0.0)[2][0])
        k2m = float(_series_eval(blk["ms_logT"], 0.0, 2)[2][0]) if len(blk["ms_logT"]) else 0.0
        R, S, VG, W = blk["rad"]
        k2r = float(rad_cgf_derivs(R, S, VG, W, 0.0, order=2,
                                   pre=blk["rad_pre"])[2][0]) if R is not None else 0.0
        st, wt = blk["ms_steps"], blk["ms_weights"]
        _, _, _, keep = ms_step_params(st)
        msof = (0.98 * _LOG_MAX / (float(np.max(np.abs(wt[keep]))) * args.theta_cut)
                if keep.any() else np.inf)
        tn, tp = blk["theta_limits"]
        print(f"{k:5d} {k2i:12.5g} {k2m:12.5g} {k2r:12.5g} "
              f"{k2i + k2m + k2r:12.5g} | "
              f"{theta_overflow_limit(blk['ioni']):10.4g} {msof:10.4g} "
              f"{rad_theta_overflow_limit(R, S, VG, W):10.4g} | "
              f"{tn:10.4g} {tp:10.4g}")

    print()
    print("=" * 100)
    print("1a. INVERSION GRID, matched per plane (NOT reused across momenta)")
    print("=" * 100)
    print("  t_reach = where Re S(t) = -60, i.e. |phi| = 9e-27. The reference "
          "grid runs to\n  1.3 t_reach; z_max = pi nt / t_max controls "
          "aliasing and dz = 2 pi/(npad t_max)\n  the peak resolution. All "
          "three are scanned below, per momentum.")
    print(f"{'plane':>5} {'t_reach':>9} {'t_max':>9} {'nt':>9} {'z_max':>10} "
          f"{'dz':>10} {'noise/peak':>11} {'mode':>10}")
    for k in planes:
        blk = blocks[k]
        tr = tau_reach(legs, k, avec, blk["sigma"], lncut=args.lncut)
        tau = auto_tau(legs, k, avec, blk["sigma"], lncut=args.lncut)
        z, p, dp = _inv(legs, k, avec, blk["sigma"], ("ioni", "ms", "rad"),
                        args)
        dt = tau[-1] / args.nt
        print(f"{k:5d} {tr:9.2f} {tau[-1]:9.2f} {args.nt:9d} "
              f"{np.pi/dt:10.1f} {z[1]-z[0]:10.3e} "
              f"{abs(float(np.min(p)))/float(np.max(p)):11.2e} "
              f"{mode_from_density(z, p):10.5f}")
    print("  convergence of the full-block mode under each grid knob "
          "separately:")
    print(f"{'plane':>5} | " + " ".join(f"{c:>11}" for c in
          ("lncut -30", "lncut -60", "lncut -120", "nt/4", "nt*2",
           "npad/4", "npad*4")))
    for k in planes:
        blk = blocks[k]
        row = []
        for lnc in (-30.0, -60.0, -120.0):
            tau = auto_tau(legs, k, avec, blk["sigma"], lncut=lnc)
            z, p = invert_cf(block_cf_exponent(legs, k, avec, blk["sigma"],
                                               tau), tau, npad=args.npad,
                             nt=args.nt)
            row.append(mode_from_density(z, p))
        tau = auto_tau(legs, k, avec, blk["sigma"])
        Sfix = block_cf_exponent(legs, k, avec, blk["sigma"], tau)
        for nt, npd in ((args.nt // 4, args.npad), (args.nt * 2, args.npad),
                        (args.nt, max(args.npad // 4, 1)),
                        (args.nt, args.npad * 4)):
            z, p = invert_cf(Sfix, tau, npad=npd, nt=nt)
            row.append(mode_from_density(z, p))
        print(f"{k:5d} | " + " ".join(f"{v:11.5f}" for v in row))

    print()
    print("=" * 100)
    print("1b. IS EACH CHANNEL's CGF ENTIRE? K, K', K'' through theta = 0 on "
          "BOTH signs")
    print("=" * 100)
    print("  Finite, smooth and K'' > 0 on both sides is the operational test "
          "(NOTES XVIII\n  did exactly this for ionization). theta is quoted "
          "as a fraction of that\n  channel's own overflow bound, so the "
          "table probes the whole usable range.")
    for k in planes:
        blk = blocks[k]
        for name in ("ioni", "ms", "rad"):
            b = collect_block(legs, k, avec, blk["sigma"],
                              theta_cut=args.theta_cut, channels=(name,))
            tn, tp = b["theta_limits"]
            fin = min(t for t in (tn, tp) if np.isfinite(t))
            tn = fin if not np.isfinite(tn) else tn
            tp = fin if not np.isfinite(tp) else tp
            ths = np.array([-0.5 * tn, -1e-2 * tn, 0.0, 1e-2 * tp, 0.5 * tp])
            K, K1, K2 = block_cgf_derivs(b, ths, order=2)
            print(f"  plane {k:2d} {name:4s}  " + "  ".join(
                f"th={t:+.2e}:K={a:+.3e},K'={c:+.3e},K''={d:.3e}"
                for t, a, c, d in zip(ths, K, K1, K2)))
            assert np.all(np.isfinite(K)) and np.all(K2 > 0), \
                f"{name} plane {k}: CGF not finite / K'' <= 0 on the grid"
    print("  All finite with K'' > 0 on both signs -> entire in the usable "
          "range.\n  For MS that statement is CONDITIONAL on theta_cut (V3); "
          "for ionization and\n  radiative it is unconditional (bounded "
          "jumps).")

    print()
    print("=" * 100)
    print("2. MODE: saddlepoint of the summed CGF vs FFT inversion of the "
          "full model_phi")
    print("=" * 100)
    print(f"{'plane':>5} | {'SPA all':>9} {'exact all':>9} | {'SPA ioni':>9} "
          f"{'exact ioni':>10} | {'d(ms+rad) SPA':>13} {'d exact':>9}")
    for k in planes:
        blk = blocks[k]
        var, _, _ = model_variance(legs, k, avec)
        sig = float(np.sqrt(var))
        blk_i = collect_block(legs, k, avec, sig, theta_cut=args.theta_cut,
                              channels=("ioni",))
        ma, mi = block_mode(blk), block_mode(blk_i)
        z, p, _ = _inv(legs, k, avec, sig, ("ioni", "ms", "rad"), args)
        ea = mode_from_density(z, p)
        z, p, _ = _inv(legs, k, avec, sig, ("ioni",), args)
        ei = mode_from_density(z, p)
        print(f"{k:5d} | {ma:9.4f} {ea:9.4f} | {mi:9.4f} {ei:10.4f} | "
              f"{ma - mi:+13.4f} {ea - ei:+9.4f}")
    print("  and the exact-inversion mode split by channel:")
    print(f"{'plane':>5} {'ioni':>11} {'ioni+ms':>11} {'ioni+rad':>11} "
          f"{'all':>11} | {'d(ms)':>10} {'d(rad)':>10} {'d(ms+rad)':>10}")
    for k in planes:
        sig = blocks[k]["sigma"]
        m = {}
        for ch in (("ioni",), ("ioni", "ms"), ("ioni", "rad"),
                   ("ioni", "ms", "rad")):
            z, p, _ = _inv(legs, k, avec, sig, ch, args)
            m[ch] = mode_from_density(z, p)
        a = m[("ioni",)]
        print(f"{k:5d} {a:11.6f} {m[('ioni','ms')]:11.6f} "
              f"{m[('ioni','rad')]:11.6f} {m[('ioni','ms','rad')]:11.6f} | "
              f"{m[('ioni','ms')]-a:+10.2e} {m[('ioni','rad')]-a:+10.2e} "
              f"{m[('ioni','ms','rad')]-a:+10.2e}")

    print()
    print("=" * 100)
    print("3. DENSITY: SPA of the summed CGF vs FFT inversion of the full "
          "model_phi")
    print("=" * 100)
    print("  p_SPA/p_exact at quantiles of the EXACT density, so the sample "
          "points follow the\n  distribution rather than a grid. '/Z' divides "
          "the SPA by its own mass Z, the\n  only constant the saddlepoint "
          "leaves undetermined.  Ionization-only is shown\n  alongside so the "
          "two new channels can be blamed or acquitted.")
    qs = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
    for k in planes:
        blk = blocks[k]
        blk_i = collect_block(legs, k, avec, blk["sigma"],
                              theta_cut=args.theta_cut, channels=("ioni",))
        z, pe, _ = _inv(legs, k, avec, blk["sigma"], ("ioni", "ms", "rad"),
                        args)
        _, pei, _ = _inv(legs, k, avec, blk["sigma"], ("ioni",), args)
        cdf = np.cumsum(np.maximum(pe, 0.0)) * (z[1] - z[0])
        cdf /= cdf[-1]
        zq = np.interp(qs, cdf, z)
        zq = np.append(zq, mode_from_density(z, pe))
        Za = float(np.trapezoid(*_spa_mass(blk)))
        Zi = float(np.trapezoid(*_spa_mass(blk_i)))
        pa, _ = spa_density(blk, zq)
        pi, _ = spa_density(blk_i, zq)
        ea = np.interp(zq, z, pe)
        ei = np.interp(zq, z, pei)
        print(f"  plane {k:2d}   Z(all) = {Za:.4f}, Z(ioni) = {Zi:.4f}, "
              f"inversion noise floor = "
              f"{abs(float(np.min(pe)))/float(np.max(pe)):.1e} of the peak")
        print(f"    {'quantile':>9} {'z':>9} {'all raw':>9} {'all /Z':>9} "
              f"{'ioni raw':>9} {'ioni /Z':>9}")
        for q, zz, a, b, c, d in zip(list(qs) + ["mode"], zq, pa, ea, pi, ei):
            qq = f"{q:9.2f}" if not isinstance(q, str) else f"{q:>9}"
            print(f"    {qq} {zz:9.3f} {a/b:9.4f} {a/Za/b:9.4f} "
                  f"{c/d:9.4f} {c/Zi/d:9.4f}")

    print()
    print("=" * 100)
    print("4. FISHER INFORMATION, by channel subset, saddlepoint vs exact "
          "inversion")
    print("=" * 100)
    print(f"{'plane':>5} {'channels':>14} | {'1/I SPA':>9} {'1/I exact':>10} "
          f"{'SPA/exact':>10} | {'Z':>7}")
    subs = (("ioni",), ("ioni", "ms"), ("ioni", "rad"), ("ioni", "ms", "rad"))
    for k in planes:
        for ch in subs:
            b = collect_block(legs, k, avec, blocks[k]["sigma"],
                              theta_cut=args.theta_cut, channels=ch)
            I, d = block_fisher_spa(b, n=args.nth)
            z, p, dp = _inv(legs, k, avec, blocks[k]["sigma"], ch, args)
            Ie, _ = fisher_exact(z, p, dp)
            print(f"{k:5d} {'+'.join(ch):>14} | {1/I:9.4f} {1/Ie:10.4f} "
                  f"{Ie/I:10.3f} | {d['Z']:7.4f}")
    print("  floor scan of the exact 1/I (8 decades) and the SPA theta-"
          "quadrature scan, full block:")
    print(f"{'plane':>5} | " + " ".join(f"{f'floor 1e-{e}':>11}"
                                        for e in (4, 6, 8, 10, 12))
          + f" | {'n=1000':>9} {'n=4000':>9} {'n=16000':>9}")
    for k in planes:
        blk = blocks[k]
        z, p, dp = _inv(legs, k, avec, blk["sigma"], ("ioni", "ms", "rad"),
                        args)
        row = [1.0 / fisher_exact(z, p, dp, floor=10.0 ** (-e))[0]
               for e in (4, 6, 8, 10, 12)]
        row2 = [1.0 / block_fisher_spa(blk, n=n)[0] for n in (1000, 4000, 16000)]
        print(f"{k:5d} | " + " ".join(f"{v:11.4f}" for v in row) + " | "
              + " ".join(f"{v:9.4f}" for v in row2))

    print()
    print("=" * 100)
    print("5. ROBUSTNESS of the two MS knobs: the angular cutoff and the "
          "series length")
    print("=" * 100)
    print("  FIRST, the knobs must actually MOVE something -- two 'controls' "
          "in this study\n  turned out to be inert. K_ms at the largest theta "
          "the MS series can reach:")
    print(f"{'plane':>5} {'theta':>10} | " + " ".join(f"{f'cut={c:g}':>13}"
                                                      for c in (1.0, 2.0,
                                                                np.pi, 6.0,
                                                                12.0)))
    for k in planes:
        row, th = [], None
        for c in (1.0, 2.0, np.pi, 6.0, 12.0):
            b = collect_block(legs, k, avec, blocks[k]["sigma"], theta_cut=c,
                              channels=("ms",))
            if th is None:
                th = 0.3 * min(b["theta_limits"])
            row.append(float(np.atleast_1d(
                block_cgf_derivs(b, th, order=0)[0])[0]))
        print(f"{k:5d} {th:10.3g} | " + " ".join(f"{v:13.5e}" for v in row))
    print("  -> the cutoff changes K_ms by orders of magnitude out there, so "
          "it is live.\n  It nonetheless leaves the MODE untouched, because "
          "that region carries no mass:")
    print(f"{'plane':>5} | " + " ".join(f"{f'cut={c:g}':>12}" for c in
                                        (1.0, 2.0, np.pi, 6.0, 12.0))
          + " (mode of the full block)")
    for k in planes:
        row = []
        for c in (1.0, 2.0, np.pi, 6.0, 12.0):
            b = collect_block(legs, k, avec, blocks[k]["sigma"], theta_cut=c)
            row.append(block_mode(b))
        print(f"{k:5d} | " + " ".join(f"{v:12.6f}" for v in row))
    print(f"{'plane':>5} | " + " ".join(f"{f'nmax={n}':>12}" for n in
                                        (60, 100, 200, 400))
          + " (mode; +inf-masked theta must carry no mass)")
    for k in planes:
        row = []
        for nm in (60, 100, 200, 400):
            b = collect_block(legs, k, avec, blocks[k]["sigma"],
                              theta_cut=args.theta_cut, nmax=nm)
            row.append(block_mode(b))
        print(f"{k:5d} | " + " ".join(f"{v:12.6f}" for v in row))


if __name__ == "__main__":
    main()
