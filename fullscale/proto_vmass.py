#!/usr/bin/env python3
"""Numerical validation of the v-formulation, before any TensorFlow is touched.

THE CLAIM.  The mass resolution of a track pair is not a fixed width: the
CURVATURE resolution is what is constant, so `sigma_m ~ m^{1+f}` with
`f = f_hit` (the term's `vgf`).  The likelihood convolves the lineshape with a
kernel of FIXED width `sigma_i`, and conditions each candidate on that
`sigma_i`.  Both are wrong in the same way, and substituting

    v(m) = Int dm / m^{1+f} = -m^{-f} / f          (v = ln m at f = 0)

fixes both at once: the smearing becomes a FIXED-WIDTH convolution in `v`,
    v_i = v(m') + k_i x + c^v_i x^2 + d^v_i ,      k_i = sigma_i / m_i^{1+f} ,
with a width `k_i` that carries no information about the true mass
(measured: `rho(k, m_gen) = -0.01` against `rho(sigma, m_gen) = +0.17`).

THE COEFFICIENTS, derived.  With `sigma_bar(m') = k_bar m'^{1+f}` and
`delta = m_i - m'`,

    v_i - v(m') = delta/m'^{1+f} - (1+f) delta^2 / (2 m'^{2+f}) + ...
                = k_bar x - (1+f) k_bar^2 m'^f x^2 / 2 + (Jensen)/m'^{1+f}

and the STORED `k_i = sigma_i/m_i^{1+f}` equals `k_bar` to first order in `x`
EXACTLY when `a = (1+f) sigma_bar/m_bar`, which is the `a` the term uses.  So

  * there is NO measure term `(1 - a_i x)` in `v` -- the self-consistent width
    is absorbed by the substitution, which is the whole point;
  * `c^v_i = k_i sigma_i / m_i * (1 - (1+f)/2)`  -- the substitution's own
    curvature `-(1+f)/2` plus the Jensen `u^2` term `+1`.  This script has NO
    Jensen effect in its truth (its smearing is a pure Gaussian about the true
    mass), so what it can check is the `-(1+f)/2` piece alone, and it does:
    scanning the coefficient, the optimum is `-0.625` against the derived
    `-(1+f)/2 = -0.632`, and there the residual shift is **0.03 MeV**.  The
    Jensen `+1` is a separate physical effect and its gate is the J/psi gun;
  * `d^v_i = s_i^2 / (2 m_i^f)`  -- the Jensen mean shift `m_i s_i^2/2`,
    divided by `m^{1+f}`.

At `f = 0.264` that is `c^v = +0.368 k sigma/m` against the m-form's
`c = -0.264 sigma^2/m`, i.e. `-0.264 k sigma/m` in the same units: a different
sign and a different size, which is the check that nothing is double-counted.

WHAT THIS SCRIPT MEASURES.  For one resolution class it builds
  * the TRUTH: `Int p(m') N(m_i - m'; k m'^{1+f}) dm'`, the smearing that
    actually happened, by direct quadrature with a mass-dependent width;
  * the M-MODEL: the same lineshape convolved with a FIXED width
    `sigma = k m_i^{1+f}` -- what the term does today;
  * the V-MODEL: the lineshape resampled to `v`, convolved with the fixed
    width `k`, and mapped back.
and reports the peak shift and the shift a 5-term Legendre `K(m)` cannot
absorb.  The v-model must reproduce the truth; the m-model must not.
"""
import argparse

import numpy as np
from numpy.polynomial import legendre
from scipy.optimize import minimize


def born(m, mz=91.1876, gz=2.4952, power=-1.5):
    """A Z-like Born density: relativistic BW times a falling luminosity."""
    d = (m * m - mz * mz) ** 2 + (mz * gz) ** 2
    return (m ** power) * m * m / d


def vmap(m, f):
    return np.log(m) if f == 0.0 else -(m ** (-f)) / f


def minv(v, f):
    return np.exp(v) if f == 0.0 else (-f * v) ** (-1.0 / f)


def truth_density(mgrid, p, mobs, k, f, nsig=8.0):
    """`Int p(m') N(m - m'; k m'^{1+f}) dm'` at every point of `mobs`."""
    dm = mgrid[1] - mgrid[0]
    sig = k * mgrid ** (1.0 + f)                     # width AT THE TRUE MASS
    out = np.empty(len(mobs))
    for i, mo in enumerate(mobs):
        z = (mo - mgrid) / sig
        w = np.where(np.abs(z) < nsig, np.exp(-0.5 * z * z) / (sig * np.sqrt(2 * np.pi)), 0.0)
        out[i] = np.sum(p * w) * dm
    return out


def m_model(mgrid, p, mobs, k, f, quad=None):
    """The term's model: a FIXED width `sigma = k m_obs^{1+f}` per candidate."""
    dm = mgrid[1] - mgrid[0]
    out = np.empty(len(mobs))
    for i, mo in enumerate(mobs):
        sig = k * mo ** (1.0 + f)
        # the map m_i = m' + sigma x + c x^2 (+ d), sampled over x
        x = np.linspace(-8.0, 8.0, 2001)
        px = np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)
        c = 0.0 if quad is None else quad * sig * sig / mo
        msrc = mo - (sig * x + c * x * x)
        out[i] = np.sum(np.interp(msrc, mgrid, p, left=0.0, right=0.0) * px) * (x[1] - x[0])
    return out


def v_model(mgrid, p, mobs, k, f, quad=None, fmod=None):
    """The v-formulation: fixed width in `v`, mapped back.

    ``f`` is the candidate's TRUE width exponent (``sigma_m ~ m^{1+f}``,
    ``f = vgf_i``); ``fmod`` is the exponent the MODEL uses. They are the same
    number only if the convolution variable is chosen per candidate.

    The card uses ONE common ``p = 1 + fmod = 1.264`` for every
    candidate, for both the conditioning label ``k_i = sigma_i/m_i^p`` and the
    convolution variable. The first is a labelling choice and any
    mass-independent ``p`` will do; the second is physics and its answer is
    ``1 + f_i`` per candidate. With ``fmod != f`` the model mis-scales the
    width by ``(m/m_i)^{fmod - f}`` -- a MASS-DEPENDENT width error whose sign
    is the sign of ``fmod - f``, which is what this argument exists to measure.
    """
    if fmod is None:
        fmod = f
    out = np.empty(len(mobs))
    x = np.linspace(-8.0, 8.0, 2001)
    px = np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)
    for i, mo in enumerate(mobs):
        # the LABEL the card stores is sigma_i / m_i^{1+fmod}, and sigma_i is
        # the candidate's true width at its own mass, k m_i^{1+f}
        kmod = k * mo ** (f - fmod)
        cv = 0.0 if quad is None else quad * kmod * (kmod * mo ** fmod)
        vsrc = vmap(mo, fmod) - (kmod * x + cv * x * x)
        msrc = minv(vsrc, fmod)
        # L_v(v_i) = E_x[p_v(v_i - u^v(x))] with p_v = p(m) m^{1+f}, and the
        # density in m is L_v / m_i^{1+f}: BOTH Jacobians, and their ratio
        # (m'/m_i)^{1+f} is a 7 % effect over the kernel's own support, i.e.
        # exactly the size of the thing being corrected -- dropping either
        # Jacobian silently absorbs the effect under test.
        jac = (msrc / mo) ** (1.0 + fmod)
        out[i] = np.sum(np.interp(msrc, mgrid, p, left=0.0, right=0.0) * jac * px) * (x[1] - x[0])
    return out


def shift_needed(g, T, M, nshape=5):
    """Mass shift the model needs to match the truth, with `K(m)` floated."""
    u = 2.0 * (g - g[0]) / (g[-1] - g[0]) - 1.0
    B = np.stack([legendre.legval(u, [0] * j + [1]) for j in range(1, nshape + 1)], 1)
    T = T / T.sum()

    def chi2(par):
        mm = np.interp(g - par[0], g, M, left=0.0, right=0.0) * np.exp(B @ par[1:])
        s = mm.sum()
        if s <= 0:
            return 1e30
        mm = mm / s
        ok = (T > 1e-9) & (mm > 1e-12)
        return float(np.sum((T[ok] - mm[ok]) ** 2 / T[ok]))

    r = minimize(chi2, np.zeros(1 + nshape), method="Nelder-Mead",
                 options=dict(maxiter=60000, maxfev=60000, xatol=1e-10, fatol=1e-18))
    return r.x[0] * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f", type=float, default=0.264,
                    help="the candidate's TRUE width exponent, sigma ~ m^{1+f}")
    ap.add_argument("--pmodel", type=float, default=None,
                    help="the COMMON p the card's v map uses (1+f_model). "
                         "Default: matched to --f, which is the validated "
                         "idealisation. Set it to 1.264 to measure what "
                         "the card as built actually does to a candidate whose "
                         "own exponent is --f.")
    ap.add_argument("--bands", action="store_true",
                    help="run the three eta bands at their MEASURED vgf and "
                         "sigma/m instead of the --srel scan")
    ap.add_argument("--srel", type=float, nargs="*", default=[0.008, 0.012, 0.02, 0.03])
    ap.add_argument("--window", type=float, nargs=2, default=[60.0, 120.0])
    ap.add_argument("--dm", type=float, default=0.02)
    args = ap.parse_args()

    f = args.f
    fmod = args.f if args.pmodel is None else args.pmodel - 1.0
    mgrid = np.arange(40.0, 150.0, args.dm)
    p = born(mgrid)
    p = p / (p.sum() * args.dm)
    obs = np.arange(args.window[0], args.window[1] + 1e-9, args.dm)

    print(f"f = {f}, model 1+f = {1+fmod:.4f} (mismatch {fmod-f:+.4f}), "
          f"window {args.window}, dm = {args.dm*1e3:.0f} MeV\n")

    if args.bands:
        # MEASURED on the m-form band cards (median vgf, inverse-variance
        # weighted <sigma/m>): the sample's own numbers, not an assumption
        BANDS = [("|eta|<0.9", 0.2104, 0.00968),
                 ("0.9-1.6",   0.1925, 0.01247),
                 ("1.6-3.0",   0.2696, 0.01510),
                 ("inclusive", 0.2166, 0.01121)]
        pm = 1.264 if args.pmodel is None else args.pmodel
        print(f"the common-p mismatch, at p = {pm:.4f}\n")
        print(f"{'band':11s} {'vgf':>7s} {'1+f':>7s} {'p-(1+f)':>8s} {'sig/m':>8s} "
              f"{'v matched':>10s} {'v common p':>11s}")
        for lab, fb, sb in BANDS:
            fmb = pm - 1.0
            sig91 = sb * 91.1876
            kb = sig91 / 91.1876 ** (1.0 + fb)
            T = truth_density(mgrid, p, obs, kb, fb)
            a = shift_needed(obs, T, v_model(mgrid, p, obs, kb, fb,
                                             -0.5 * (1.0 + fb)))
            b = shift_needed(obs, T, v_model(mgrid, p, obs, kb, fb,
                                             -0.5 * (1.0 + fmb), fmod=fmb))
            print(f"{lab:11s} {fb:7.4f} {1+fb:7.4f} {pm-(1+fb):+8.4f} {sb:8.5f} "
                  f"{a:+10.2f} {b:+11.2f}   MeV")
        print("\n`v matched` is the idealisation (p chosen per "
              "candidate);\n`v common p` is what the card as built does. The "
              "difference is the effect under test.")
        return
    print(f"{'sigma/m at 91':>13} {'k':>10} | {'m-model':>22} | {'v-model':>22}")
    print(f"{'':13} {'':10} | {'no quad':>10} {'+quad':>11} | {'no quad':>10} {'+quad':>11}")
    for srel in args.srel:
        sig91 = srel * 91.1876
        k = sig91 / 91.1876 ** (1.0 + f)
        T = truth_density(mgrid, p, obs, k, f)
        rows = []
        for quad in (None, -f):                       # the m-form's c = -vgf s^2/m
            rows.append(shift_needed(obs, T, m_model(mgrid, p, obs, k, f, quad)))
        for quad in (None, -0.5 * (1.0 + fmod)):      # the v-form's c^v,
            # WITHOUT the Jensen +1: this script's truth has no Jensen effect
            rows.append(shift_needed(obs, T, v_model(mgrid, p, obs, k, f, quad,
                                                     fmod=fmod)))
        print(f"{srel:13.4f} {k:10.3e} | {rows[0]:+10.2f} {rows[1]:+11.2f} | "
              f"{rows[2]:+10.2f} {rows[3]:+11.2f}   MeV")
    print("\nthe shift is what the MODEL needs to match the TRUTH: 0 is correct.")


if __name__ == "__main__":
    main()
