#!/usr/bin/env python3
"""Effective Gaussian blocks (mu_eff, sigma2_eff) for a CGF-based track fit.

WHAT THIS IS FOR. The CVH fit solves a Gaussian least-squares problem: every
block must look like (r - mu)^2 / 2 sigma^2. The true per-block term is
-ln p(r), with p the compound-Poisson ionization density, which is nothing like
a Gaussian. This module builds, at each iteration, the quadratic that matches
-ln p in gradient and curvature at the current residual. Those matched values
are mu_eff and sigma2_eff. They are NOT the mean and variance of the physical
distribution -- they are the parameters of a local surrogate.

WHY THIS REMOVES THE alpha TRUNCATION. A Gaussian block needs kappa2 = K''(0),
which for ionization is delta-ray dominated (366 in z units against a core of
~1). That divergence is the ONLY reason the fit truncates the spectrum at
alpha = 0.999, and the truncation is applied per step, so it is not additive:
measured 2026-08-13, it is the entire step-length dependence AND the dominant
qop closure error. This scheme never evaluates K''(0). It evaluates K'' at a
NONZERO tilt theta_hat, where the exponential tilt suppresses the 1/E^2 tail.
Finite variance, no cut, and exactly additive because CGFs add under
convolution.

THE IDENTITIES (derived, not fitted). From the saddlepoint density
    p(r) ~ exp[K(th) - th r] / sqrt(2 pi K''(th)),   K'(th) = r
differentiating with dth/dr = 1/K''(th) makes the K-terms cancel:
    dln p/dr    = -th - 0.5 K'''/(K'')^2
    -d2ln p/dr2 = 1/K''(th)
so, to leading order,
    sigma2_eff = K''(theta_hat)
    mu_eff     = r - theta_hat * K''(theta_hat)
The score IS the saddlepoint. Gaussian check: K = mu th + s2 th^2/2 gives
th = (r-mu)/s2, K'' = s2, hence mu_eff = mu and sigma2_eff = s2 identically --
ordinary least squares is the exact special case.

KNOWN LIMITATIONS, all real:
  * (RETRACTED 2026-08-13 XIX) "The CGF is ONE-SIDED" was wrong. All jumps are
    bounded (delta rays on [e0, tmax], excitations at fixed e1, e2), so
    E[e^{th x}] is finite for every real theta and the CGF is ENTIRE. The
    apparent divergence was e^{bw} hitting a clip in _delta_derivs, which
    returned a wrong finite number instead of signalling. theta_hat and psi
    exist on BOTH sides; what remains is a floating-point bound on |theta|
    (cgf_saddlepoint.theta_overflow_limit), which in z units already sits
    hundreds of sigma into the power-law tail.
  * The +0.5 K'''/(K'')^2 term in the score is a genuine skewness correction --
    at small r it DOMINATES theta_hat rather than correcting it. It is now
    analytic (ioni_cgf_derivs(..., order=4)), not finite-differenced.
  * Real CVH blocks are 5x5, so theta_hat is a 5-vector and sigma2_eff a 5x5
    Hessian. This module is the scalar (q/p) case -- the one that carries the
    ionization -- and is meant to establish the identities before the
    multivariate version.
  * IRLS/EM converges monotonically for log-concave p. This p is NOT log-concave
    in the tail, so convergence must be tested, not assumed.

usage: python cgf_irls.py --model model_toy.root [--plane K]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cgf_saddlepoint import (ioni_cgf_derivs, heavy_tail_sign, theta_grid,
                             theta_overflow_limit, solve_saddlepoint,
                             log_density)


def effective_gaussian(steps, r, sign=None):
    """(mu_eff, sigma2_eff, theta_hat) for residual(s) r.

    r is in the same units as the centered CGF variable, i.e. deviation from
    the mean loss -- what the fit's reference already subtracts.

    Returns NaN only where the Newton solve fails or K'' is not positive; the
    `sign` argument is accepted for backward compatibility and ignored, since
    the CGF has no divergent half-line (see module docstring).
    """
    r = np.atleast_1d(np.asarray(r, dtype=np.float64))
    mu = np.full_like(r, np.nan)
    s2 = np.full_like(r, np.nan)
    th = np.full_like(r, np.nan)
    for i, ri in enumerate(r):
        t = solve_saddlepoint(steps, ri)
        if not np.isfinite(t):
            continue
        _, _, K2 = ioni_cgf_derivs(steps, t)
        K2 = float(np.atleast_1d(K2)[0])
        if not np.isfinite(K2) or K2 <= 0:
            continue
        th[i] = t
        s2[i] = K2
        mu[i] = ri - t * K2
    return mu, s2, th


def surrogate_curve(steps, thetas):
    """Analytic (r, psi, dpsi/dr, sigma2_eff, mu_eff) along the theta curve.

    No root finding and no finite differences: r = K'(theta) sweeps the whole
    line as theta does, and with dtheta/dr = 1/K''

        psi     = theta + K'''/(2 K''^2)                 (= -dln p/dr)
        dpsi/dr = 1/K'' + K''''/(2 K''^3) - K'''^2/K''^4

    Newton/observed curvature is dpsi/dr; it goes NEGATIVE in the power-law
    region, which is why the Fisher information (cgf_fisher) is used for the
    block weight instead. Both are returned so the two can be compared.
    """
    K, K1, K2, K3, K4 = ioni_cgf_derivs(steps, thetas, order=4)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        psi = thetas + 0.5 * K3 / K2 ** 2
        dpsi = 1.0 / K2 + 0.5 * K4 / K2 ** 3 - K3 ** 2 / K2 ** 4
        s2 = np.where(dpsi > 0, 1.0 / dpsi, np.nan)
        mu = K1 - s2 * psi
    return K1, psi, dpsi, s2, mu


# ---------------------------------------------------------------------------
# Safeguarded Fisher scoring (2026-08-13 XXI)
#
# The block objective is f(r) = -ln p_SPA(r). Two things about it decide the
# design, and BOTH were checked before the code was written rather than after:
#
#  1. THE SADDLEPOINT SOLVE IS THE FRAGILE PART, not the scoring. The damped
#     Newton in solve_saddlepoint fails badly on the heavy-tail side: at
#     plane 18 it returns theta = -1.008e-2 for r = -50 (K'(theta) = -52.5, so
#     it simply stopped) and theta = -0.397 for r = -300 and for r = -1e4 alike
#     (K'(theta) = -1.7e96). Feeding that into psi is what makes an iteration
#     "wander to 1e4". K' is MONOTONE in theta, so a bracketed solve cannot
#     fail; solve_theta below is a safeguarded Newton (rtsafe) in
#     u = asinh(theta/scale), which compresses the 1e-10..1e9 range of theta
#     into u in [-24, 24].
#
#  2. THE SPA DENSITY REALLY DOES TURN UP AT THE HARD EDGE. p_SPA =
#     e^{K-theta r}/sqrt(2 pi K''), and K'' -> 0 at the edge, so the prefactor
#     -0.5 ln(2 pi K'') -> +infinity and eventually beats e^{K - theta r}.
#     Measured at plane 18: ln p_SPA bottoms out at -142418.79 at r = 24.48176
#     and rises again. So psi genuinely has a SECOND zero there -- it is a
#     property of the saddlepoint density, not a numerical artefact (though it
#     is numerically filthy too: lambda3 = K'''/K''^{3/2} passes 1 just before
#     it). It is NOT reachable by a descent method: getting there means
#     crossing a barrier of 1.4e5 in -ln p, and backtracking accepts only
#     decreases. The domain is capped at the turn-up anyway, so the region is
#     simply not in the model.
# ---------------------------------------------------------------------------

def make_block(steps, nfisher=3000, verbose=False):
    """Precompute everything a block needs for repeated evaluation.

    Returns a dict with the Fisher information I (two-sided, cgf_fisher), the
    trusted theta domain and the r range it maps to. The domain is capped on
    the bounded side at the turn-up of ln p_SPA (see the note above) and on the
    heavy-tail side at theta_overflow_limit.
    """
    from cgf_fisher import fisher_spa
    _, _, k2 = ioni_cgf_derivs(steps, 0.0)
    k20 = float(np.atleast_1d(k2)[0])
    scale = 1.0 / np.sqrt(k20)
    sgn = heavy_tail_sign(steps)
    I, _ = fisher_spa(steps, n=nfisher, two_sided=True)

    def lnp_of(th):
        K, K1, K2 = ioni_cgf_derivs(steps, th)
        with np.errstate(divide="ignore", invalid="ignore"):
            return K - th * K1 - 0.5 * np.log(2.0 * np.pi * K2), K1, K2

    # bounded side: walk out until ln p stops falling, or K'' dies
    g = sgn * scale * np.geomspace(1e-3, 1e12, 400)
    lp, r_, k2_ = lnp_of(g)
    good = np.isfinite(lp) & np.isfinite(r_) & (k2_ > 0)
    idx = np.where(good)[0]
    if len(idx):
        j = idx[int(np.argmin(lp[idx]))]     # the turn-up
        th_bounded = float(g[j])
    else:
        th_bounded = sgn * scale
    # heavy-tail side: the double-precision cap
    cap = theta_overflow_limit(steps)
    h = -sgn * np.geomspace(1e-3 * scale, 0.999 * cap, 400) if np.isfinite(cap) \
        else -sgn * scale * np.geomspace(1e-3, 1e12, 400)
    lp2, r2, k22 = lnp_of(h)
    good2 = np.isfinite(lp2) & np.isfinite(r2) & (k22 > 0)
    th_tail = float(h[np.where(good2)[0][-1]]) if good2.any() else -sgn * scale

    th_lo, th_hi = (th_tail, th_bounded) if sgn > 0 else (th_bounded, th_tail)
    _, r_lo, _ = lnp_of(np.array([th_lo]))
    _, r_hi, _ = lnp_of(np.array([th_hi]))
    blk = dict(steps=steps, I=float(I), k20=k20, scale=scale, sgn=sgn,
               th_lo=float(th_lo), th_hi=float(th_hi),
               r_lo=float(r_lo[0]), r_hi=float(r_hi[0]), nfev=0)
    if verbose:
        print(f"    block: {len(steps):4d} steps  kappa2 {k20:9.2f}  "
              f"1/I {1/blk['I']:8.4f}  theta in [{blk['th_lo']:.3e}, "
              f"{blk['th_hi']:.3e}]  r in [{blk['r_lo']:.4g}, {blk['r_hi']:.4g}]")
    return blk


def solve_theta(blk, r, hint=None, ftol=1e-11, itmax=200):
    """K'(theta) = r by safeguarded Newton (rtsafe) with BOTH axes in asinh.

    u = asinh(theta/scale) compresses theta; the residual is matched as
    asinh(K'/rscale) - asinh(r/rscale) rather than K' - r. The second transform
    is not cosmetic: K'(theta) is EXPONENTIAL in theta (measured on a 10-step
    block, K' runs from -2 to -8e296 as theta goes from -0.012 to -1.2), so
    Newton on K' - r converges only LINEARLY -- one factor of e per step. It
    needed ~230 steps to come down from K' = -4.6e99 to K' = -5, blew through
    itmax = 200, and returned the bracket end: block_eval then reported the
    same f = 1.39e12 for every r from -5 to -1e5. In asinh-asinh both axes are
    O(1), the map is nearly linear, and it converges in a handful of steps.

    The bracket is maintained throughout and a bisection is forced every 8th
    iteration, so worst case is plain bisection and failure is impossible.
    Returns NaN if r is outside the block's trusted r range.
    """
    s = blk["scale"]
    if not np.isfinite(r) or r <= blk["r_lo"] or r >= blk["r_hi"]:
        return np.nan
    rs = np.sqrt(blk["k20"])
    tgt = np.arcsinh(r / rs)
    a = np.arcsinh(blk["th_lo"] / s)
    b = np.arcsinh(blk["th_hi"] / s)
    u = np.arcsinh(hint / s) if (hint is not None and np.isfinite(hint)) else 0.0
    u = min(max(u, a), b)
    for it in range(itmax):
        th = s * np.sinh(u)
        _, K1, K2 = ioni_cgf_derivs(blk["steps"], th)
        blk["nfev"] += 1
        k1 = float(K1[0])
        if abs(k1 - r) <= ftol * max(1.0, abs(r)):
            return float(th)
        g = np.arcsinh(k1 / rs) - tgt
        if g < 0.0:
            a = u
        else:
            b = u
        # d/du asinh(K'/rs) = K'' * s cosh(u) / sqrt(K'^2 + rs^2)
        with np.errstate(over="ignore", invalid="ignore"):
            d = float(K2[0]) * s * np.cosh(u) / np.hypot(k1, rs)
        un = u - g / d if (np.isfinite(d) and d > 0) else 0.5 * (a + b)
        if not (a < un < b) or not np.isfinite(un) or (it % 8 == 7):
            un = 0.5 * (a + b)
        if abs(un - u) <= 1e-16 * max(1.0, abs(u)):
            return float(s * np.sinh(un))
        u = un
    return float(s * np.sinh(u))


def block_eval(blk, r, hint=None):
    """(f, psi, dpsi, theta) at residual r, with f = -ln p_SPA(r).

    Outside the trusted domain f = +inf (and psi = NaN), which is what makes
    the backtracking line search reject those steps rather than act on garbage.
    """
    th = solve_theta(blk, r, hint=hint)
    if not np.isfinite(th):
        return np.inf, np.nan, np.nan, np.nan
    K, K1, K2, K3, K4 = ioni_cgf_derivs(blk["steps"], th, order=4)
    blk["nfev"] += 1
    K, K1, K2, K3, K4 = (np.float64(v[0]) for v in (K, K1, K2, K3, K4))
    if not np.isfinite(K2) or K2 <= 0.0:
        return np.inf, np.nan, np.nan, np.nan
    lnp = K - th * K1 - 0.5 * np.log(2.0 * np.pi * K2)
    if not np.isfinite(lnp):
        return np.inf, np.nan, np.nan, np.nan
    # Ratios taken ONE K2 at a time. K2 reaches ~1e300 far out in the tail
    # (r_lo ~ -5e297), so K2**3 and K2**4 overflow -- as Python floats that
    # raises OverflowError, as numpy it silently gives inf and then inf/inf =
    # nan, which would poison psi exactly where the iteration needs it.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        a = K3 / K2
        b = K4 / K2
        psi = th + 0.5 * (a / K2)
        dpsi = 1.0 / K2 + 0.5 * (b / K2) / K2 - (a / K2) ** 2
    return -lnp, float(psi), float(dpsi), th


def objective(blocks, zs, c, hints=None):
    """F(c) = sum_k -ln p_k(z_k - c) and its derivatives w.r.t. c.

    r_k = z_k - c so dr/dc = -1:  dF/dc = -sum psi_k,  d2F/dc2 = sum dpsi_k.
    The Fisher curvature is sum I_k, positive by construction.
    """
    F = 0.0
    G = 0.0
    Hn = 0.0
    Hf = 0.0
    th = []
    for i, (blk, z) in enumerate(zip(blocks, zs)):
        hi = hints[i] if hints is not None else None
        f, psi, dpsi, t = block_eval(blk, z - c, hint=hi)
        th.append(t)
        if not np.isfinite(f):
            return np.inf, np.nan, np.nan, np.nan, th
        F += f
        G += -psi
        Hn += dpsi
        Hf += blk["I"]
    return F, G, Hn, Hf, th


def fisher_scoring(blocks, zs, c0, mode="fisher", itmax=500, gtol=1e-8,
                   maxstep=8.0, nback=60, expand=True, history=False):
    """Safeguarded scoring iteration for the shared parameter c.

    mode: 'fisher' (curvature = sum I_k, constant and positive),
          'newton' (curvature = sum dpsi_k, may be negative -> falls back),
          'hybrid' (Newton where the summed curvature is positive, else Fisher).

    SAFEGUARDS, in order of application:
      * direction: if the chosen curvature is not positive, use Fisher's.
      * step limiting: |dc| <= maxstep * sqrt(1/I_tot), so a single step can
        never jump across the distribution.
      * backtracking: halve until F strictly decreases (up to nback times).
        This is what makes the second stationary point at the hard edge
        unreachable -- it sits behind a barrier of ~1e5 in F -- and what
        rejects steps that leave the trusted domain (F = +inf there).
      * optional expansion: while doubling keeps improving F, keep doubling.
        Without it the deep tail is unusable: psi -> 0 like 1/|r| there, so the
        Fisher step from r = -300 is ~0.006 and 5e4 iterations would be needed.
    """
    c = float(c0)
    F, G, Hn, Hf, th = objective(blocks, zs, c)
    if not np.isfinite(F):
        return dict(c=np.nan, iters=0, converged=False, reason="start invalid",
                    F=np.inf, nfev=sum(b["nfev"] for b in blocks), hist=[])
    sc = 1.0 / np.sqrt(Hf)
    hist = [(c, F, G)]
    nb_tot = 0
    for it in range(1, itmax + 1):
        if abs(G) <= gtol * np.sqrt(Hf):
            return dict(c=c, iters=it - 1, converged=True, reason="gradient",
                        F=F, G=G, nfev=sum(b["nfev"] for b in blocks),
                        nback=nb_tot, hist=hist if history else [])
        H = {"fisher": Hf, "newton": Hn,
             "hybrid": (Hn if (np.isfinite(Hn) and Hn > 0) else Hf)}[mode]
        if not (np.isfinite(H) and H > 0):
            H = Hf
        step = -G / H
        cap = maxstep * sc
        if abs(step) > cap:
            step = np.sign(step) * cap
        # forward tracking: take the largest doubling that still improves
        if expand:
            best = (step, None)
            s2 = step
            for _ in range(40):
                F2, *_ = objective(blocks, zs, c + s2, hints=th)
                if not np.isfinite(F2) or F2 >= F:
                    break
                best = (s2, F2)
                s2 = 2.0 * s2
            if best[1] is not None:
                step = best[0]
        acc = False
        for _ in range(nback):
            Fn, Gn, Hnn, Hfn, thn = objective(blocks, zs, c + step, hints=th)
            if np.isfinite(Fn) and Fn < F:
                c, F, G, Hn, Hf, th = c + step, Fn, Gn, Hnn, Hfn, thn
                acc = True
                break
            step *= 0.5
            nb_tot += 1
        hist.append((c, F, G))
        if not acc:
            return dict(c=c, iters=it, converged=abs(G) <= 1e-6 * np.sqrt(Hf),
                        reason="no decrease", F=F, G=G,
                        nfev=sum(b["nfev"] for b in blocks), nback=nb_tot,
                        hist=hist if history else [])
    return dict(c=c, iters=itmax, converged=False, reason="itmax", F=F, G=G,
                nfev=sum(b["nfev"] for b in blocks), nback=nb_tot,
                hist=hist if history else [])


def c_domain(blocks, zs):
    """(c_min, c_max) on which F(c) is finite: every block needs
    r_lo < z_k - c < r_hi, i.e. c in (z_k - r_hi, z_k - r_lo).

    The blocks have BOUNDED support on the loss side (a block cannot gain
    energy), so this is a hard constraint, not a numerical nicety, and the
    binding block is the one with the SMALLEST hard edge. An iteration started
    outside it has F = +inf and no gradient at all -- a real failure mode for a
    shared parameter, and the reason a fit must project its start inside.
    """
    lo = max(z - b["r_hi"] for b, z in zip(blocks, zs))
    hi = min(z - b["r_lo"] for b, z in zip(blocks, zs))
    return lo, hi


def block_mode(blk, n=6000):
    """argmax of ln p_SPA over the block's trusted domain, by scanning the
    theta curve (no root finding)."""
    r, lp = block_curve(blk, n=n)
    m = np.isfinite(lp)
    r, lp = r[m], lp[m]
    i = int(np.argmax(lp))
    if 0 < i < len(r) - 1:
        y0, y1, y2 = lp[i - 1], lp[i], lp[i + 1]
        den = y0 - 2 * y1 + y2
        if den != 0:
            h = 0.5 * (r[i + 1] - r[i - 1])
            return float(r[i] - 0.5 * h * (y2 - y0) / den), i == len(r) - 1
    return float(r[i]), i == len(r) - 1


def block_curve(blk, n=6000):
    """(r, ln p_SPA) along the theta curve, for INTERPOLATION-based ground
    truth. Uses no root finding at all, so a scan built on it is independent of
    solve_theta and of the iteration -- which is the point.
    """
    th = np.sort(np.concatenate([
        np.geomspace(blk["scale"] * 1e-8, abs(blk["th_hi"]), n) * blk["sgn"],
        [0.0],
        -np.geomspace(blk["scale"] * 1e-8, abs(blk["th_lo"]), n) * blk["sgn"]]))
    th = th[(th >= blk["th_lo"]) & (th <= blk["th_hi"])]
    K, K1, K2 = ioni_cgf_derivs(blk["steps"], th)
    with np.errstate(divide="ignore", invalid="ignore"):
        lnp = K - th * K1 - 0.5 * np.log(2.0 * np.pi * K2)
    ok = np.isfinite(lnp) & np.isfinite(K1) & (K2 > 0)
    r, lp = K1[ok], lnp[ok]
    o = np.argsort(r)
    return r[o], lp[o]


def score_fd(steps, r, h=None):
    """-dln p/dr by finite difference, to test the score == theta_hat identity."""
    r = np.atleast_1d(np.asarray(r, dtype=np.float64))
    out = np.full_like(r, np.nan)
    for i, ri in enumerate(r):
        hh = h if h is not None else max(1e-4 * max(abs(ri), 1.0), 1e-6)
        lp = log_density(steps, np.array([ri - hh, ri + hh]))
        if np.all(np.isfinite(lp)):
            out[i] = -(lp[1] - lp[0]) / (2 * hh)
    return out


def curvature_fd(steps, r, h=None):
    """-d2ln p/dr2 by finite difference, to test 1/K''(theta_hat)."""
    r = np.atleast_1d(np.asarray(r, dtype=np.float64))
    out = np.full_like(r, np.nan)
    for i, ri in enumerate(r):
        hh = h if h is not None else max(1e-3 * max(abs(ri), 1.0), 1e-5)
        lp = log_density(steps, np.array([ri - hh, ri, ri + hh]))
        if np.all(np.isfinite(lp)):
            out[i] = -(lp[2] - 2 * lp[1] + lp[0]) / hh ** 2
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--plane", type=int, default=None,
                   help="plane index; default = outermost")
    p.add_argument("--func", default="qop")
    args = p.parse_args()

    from cf_propagation_test import load_model, model_variance, FUNCTIONALS
    from cgf_phase0_validate import collect_ioni_steps

    legs = load_model(args.model)
    k = args.plane if args.plane is not None else len(legs) - 1
    avec = FUNCTIONALS[args.func]
    var, _, _ = model_variance(legs, k, avec)
    sigma = float(np.sqrt(var))
    steps = collect_ioni_steps(legs, k, avec, sigma)
    sgn = heavy_tail_sign(steps)

    _, _, K2_0 = ioni_cgf_derivs(steps, 0.0)
    kap2 = float(np.atleast_1d(K2_0)[0])
    print(f"plane {k}, {len(steps)} ionization steps, bounded-edge side "
          f"{sgn:+.0f}")
    print(f"kappa2 = K''(0) = {kap2:.3f}   (in z units; the tail-dominated "
          f"number the fit truncates to avoid)\n")

    # probe the convergent side, in z units
    rs = sgn * np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0])
    mu, s2, th = effective_gaussian(steps, rs, sign=sgn)
    psi = score_fd(steps, rs)
    cur = curvature_fd(steps, rs)

    # the same score and curvature from the ANALYTIC K''' / K'''', evaluated at
    # the theta_hat solved above: this is the test of the new derivatives
    _, psi_an, dpsi_an, _, _ = surrogate_curve(steps, th)

    print(f"{'r':>7} {'theta_hat':>11} {'K2(th)':>10} {'psi_an':>10} "
          f"{'score_FD':>10} {'an/FD':>8} {'dpsi_an':>10} {'curv_FD':>10} "
          f"{'an/FD':>8} {'sig2/kap2':>10}")
    print("-" * 104)
    for i in range(len(rs)):
        r1 = psi_an[i] / psi[i] if np.isfinite(psi[i]) and psi[i] != 0 else np.nan
        r2 = dpsi_an[i] / cur[i] if np.isfinite(cur[i]) and cur[i] != 0 else np.nan
        print(f"{rs[i]:7.2f} {th[i]:11.5f} {s2[i]:10.4f} {psi_an[i]:10.5f} "
              f"{psi[i]:10.5f} {r1:8.4f} {dpsi_an[i]:10.5f} {cur[i]:10.5f} "
              f"{r2:8.4f} {s2[i]/kap2:10.2e}")

    print("\nCHECKS")
    print("  psi_an  = theta_hat + K'''/(2 K''^2), all analytic. an/FD -> 1")
    print("            validates it against -dln p/dr taken by finite difference")
    print("            of the saddlepoint log-density. Note psi_an is NOT")
    print("            theta_hat: at r = 0.25 theta_hat is 6e-4 and psi is -0.37,")
    print("            so the prefactor term DOMINATES (NOTES 2026-08-13 XIII).")
    print("  dpsi_an = 1/K'' + K''''/(2 K''^3) - K'''^2/K''^4 against -d2ln p/dr2.")
    print("            It goes NEGATIVE below r ~ 2: no positive-variance")
    print("            Gaussian surrogate exists there, which is what the Fisher")
    print("            information in cgf_fisher.py is for.")
    print("  sig2/kap2 << 1 is the POINT: the tilted variance is far below the")
    print("              tail-dominated kappa2, which is why no truncation is needed.")

    # the OTHER side, which used to return NaN because _delta_derivs clipped
    other = -sgn * np.array([1.0, 3.0, 10.0])
    mub, s2b, thb = effective_gaussian(steps, other)
    print(f"\n  other side r = {other}")
    print(f"    theta_hat = {thb}")
    print(f"    sigma2    = {s2b}")
    print("  (finite, not NaN: the CGF is entire and the saddlepoint exists "
          "here too.\n   Before 2026-08-13 XIX this returned NaN, which was a "
          "clipped exponential,\n   not a divergence.)")

    print("\n  ANALYTIC surrogate along theta (no root finding, no FD):")
    th_probe = sgn * np.array([1e-4, 1e-3, 1e-2, 1e-1])
    th_probe = np.sort(np.concatenate([-th_probe[::-1], th_probe]))
    rr, ps, dps, s2c, muc = surrogate_curve(steps, th_probe)
    print(f"{'theta':>11} {'r=K1':>12} {'psi':>12} {'dpsi/dr':>12} "
          f"{'sig2_eff':>12} {'mu_eff':>12}")
    for i in range(len(th_probe)):
        print(f"{th_probe[i]:11.1e} {rr[i]:12.4g} {ps[i]:12.4g} "
              f"{dps[i]:12.4g} {s2c[i]:12.4g} {muc[i]:12.4g}")
    print("  sigma2_eff = NaN where dpsi/dr <= 0: the density is not "
          "log-concave there, which\n  is why the block weight has to come "
          "from the Fisher information (cgf_fisher.py).")


if __name__ == "__main__":
    main()
