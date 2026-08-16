#!/usr/bin/env python3
"""Phase 0: the cumulant generating function of the ionization noise, and its
saddlepoint density / mode.

WHY. The track fit subtracts the MEAN energy loss in its reference and gives
the per-block deviation a Gaussian prior centred on zero. The typical track
loses the MODE, which is less. That mismatch is the leading candidate for the
~1 MeV J/psi bias and the +0.2e-3 common offset (NOTES 2026-08-08/10).

To fix it in the fit we need the mode of the exact per-block loss distribution.
This module gets it WITHOUT any Fourier inversion, by working with the
cumulant generating function

    K(theta) = ln E[e^{theta q}]  =  ln phi(-i theta)

i.e. the same object as the characteristic function used everywhere else in
this package, evaluated at a real rather than imaginary argument. The exponents
still add over steps, so a pooled block's K is the sum of its steps'.

The density then comes from the saddlepoint approximation

    p(q) ~ exp[K(th) - th q] / sqrt(2 pi K''(th)),    K'(th) = q

whose log is what the fit would use as a penalty. Note the sqrt term: it is
what displaces the mode from the mean, so it must NOT be dropped.

DO NOT try to shortcut the mode with a cumulant (Edgeworth) correction. For the
1/E^2 delta-ray spectrum kappa2 ~ a3 e0 tmax and kappa3 ~ a3 e0 tmax^2 / 2, so
-kappa3/2kappa2 ~ -tmax/4, which is divergently large: the cumulants are
tail-dominated, exactly the pathology that breaks the Gaussian fit in the first
place. The saddlepoint is safe because it never expands in cumulants.

Record layout (ioniurbanv, mirrors cf_track_resolution.ioni_step_exponent):
    0 regime (0 = already Gaussian, 1 = Urban/Glandz)
    1 gsig2      2 a1   3 e1   4 a2   5 e2
    6 a3         7 e0   8 tmax    9 scaling    10 qop-per-MeV (x1e-3)
"""
import numpy as np
from scipy.special import expi

_EULER = 0.5772156649015328606
# ln(largest finite float64). Beyond this the TRUE value of the delta-ray
# integral is genuinely larger than a double can hold, so +inf is returned.
_LOG_MAX = 709.782712893384
# |b w| below which the Taylor series is used. The series converges for every
# argument; this bound keeps the alternating cancellation (b < 0) below one
# digit while removing the e^{bw} - e^b cancellation of the closed forms.
_SER_X = 2.0
# x above which e^{-x} Ei(x) is taken from its asymptotic series rather than
# from expi (which overflows at x > 709 and loses digits well before that).
_EI_ASYM = 50.0
_NSER = 60


def _g_scaled(x):
    """G(x) = e^{-x} Ei(x) for x > 0.

    Direct below _EI_ASYM; asymptotic Ei(x) = e^x/x sum_k k!/x^k above, which
    is what lets the b > 0 branch keep working past the point where Ei itself
    overflows. At x = 50 the asymptotic truncation error is ~1e-16 relative,
    so the two branches agree to double precision.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    lo = x < _EI_ASYM
    if lo.any():
        out[lo] = np.exp(-x[lo]) * expi(x[lo])
    hi = ~lo
    if hi.any():
        xx = x[hi]
        term = np.ones_like(xx)
        acc = np.ones_like(xx)
        for k in range(1, 26):
            term = term * k / xx
            acc = acc + term
        out[hi] = acc / xx
    return out


def _s_scaled(x):
    """S(x) = e^{-x} Ei(x) - 1/x for x > 0, without the 1/x cancellation.

    Needed because b*G(bw) cancels the -1/w term of D exactly; keeping the
    difference analytic is what makes D accurate at large positive b w.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    lo = x < _EI_ASYM
    if lo.any():
        xx = x[lo]
        out[lo] = np.exp(-xx) * expi(xx) - 1.0 / xx
    hi = ~lo
    if hi.any():
        xx = x[hi]
        term = np.ones_like(xx)
        acc = np.zeros_like(xx)
        for k in range(1, 26):
            term = term * k / xx
            acc = acc + term
        out[hi] = acc / xx
    return out


def _ei_gamma(s):
    """Ei(s) - ln|s| = gamma + sum_{k>=1} s^k/(k k!). Finite as s -> 0.

    The ln|s| is split off because D' = [Ei(bw) - Ei(b) - ln w]/N is exactly
    [ei_gamma(bw) - ei_gamma(b)]/N (since |bw|/|b| = w), which removes the
    ln w cancellation for free.
    """
    s = np.asarray(s, dtype=np.float64)
    out = np.empty_like(s)
    sm = np.abs(s) <= 1.0
    if sm.any():
        ss = s[sm]
        term = np.ones_like(ss)
        acc = np.zeros_like(ss)
        for k in range(1, 40):
            term = term * ss / k
            acc = acc + term / k
        out[sm] = _EULER + acc
    bg = ~sm
    if bg.any():
        out[bg] = expi(s[bg]) - np.log(np.abs(s[bg]))
    return out


def _delta_series(b, w):
    """Taylor series of the five derivatives, valid for any b, used for
    |b w| <= _SER_X.

    With x = b w, y = b, and S_m = int_1^w E^m e^{bE} dE = sum_k (w^{m+1} x^k
    - y^k) / (k! (k+m+1)), the five quantities are S_{-2}, S_{-1}, S_0, S_1,
    S_2 with the k = 0, 1 terms of S_{-2} and the k = 0 term of S_{-1}
    cancelled analytically against the -1 - bE subtraction. Written in terms of
    x and y rather than b^k w^k so that w^k never overflows.
    """
    x = b * w
    y = b
    N = 1.0 - 1.0 / w
    w2, w3 = w * w, w * w * w
    D = np.zeros_like(b)
    D1 = np.zeros_like(b)
    D2 = np.zeros_like(b)
    D3 = np.zeros_like(b)
    D4 = np.zeros_like(b)
    tx = np.ones_like(b)          # x^k / k!
    ty = np.ones_like(b)          # y^k / k!
    for k in range(0, _NSER):
        if k > 0:
            tx = tx * x / k
            ty = ty * y / k
        if k >= 2:
            D += (tx / w - ty) / (k - 1.0)
        if k >= 1:
            D1 += (tx - ty) / k
        D2 += (w * tx - ty) / (k + 1.0)
        D3 += (w2 * tx - ty) / (k + 2.0)
        D4 += (w3 * tx - ty) / (k + 3.0)
    return D / N, D1 / N, D2 / N, D3 / N, D4 / N


def _delta_derivs(b, w):
    """D, D', D'', D''', D'''' of <(e^{bE} - 1 - bE)/E^2> for the 1/E^2
    spectrum on [1, w], with N = 1 - 1/w.

        D    = [(-e^{bw}/w + e^b) + b(Ei(bw) - Ei(b)) - (1 - 1/w) - b ln w]/N
        D'   = [Ei(bw) - Ei(b) - ln w] / N
        D''  = [(e^{bw} - e^b) / b] / N
        D''' = [(w e^{bw} - e^b)/b - (e^{bw} - e^b)/b^2] / N
        D''''= [(w^2 e^{bw} - e^b)/b - 2(w e^{bw} - e^b)/b^2
                + 2(e^{bw} - e^b)/b^3] / N

    Real-argument continuation of cf_track_resolution._delta_term_2d: with
    ia -> b the b(E1(-b) - E1(-bw)) term becomes b(Ei(bw) - Ei(b)), the two
    -i*pi branch offsets cancelling.

    BOTH SIGNS OF b ARE SUPPORTED. The CGF of this block is entire (the jumps
    are bounded: delta rays live on [e0, tmax]), so there is no divergent
    half-line -- the old code's _EXP_MAX clip was a floating-point guard that
    silently returned a WRONG finite value, and that is what made theta look
    one-sided (NOTES 2026-08-13 XVII/XVIII). Three regimes:

      |b w| <= _SER_X   Taylor series (exact for b = 0, no cancellation)
      b < 0             direct closed form; e^{bw}, e^b <= 1 so nothing can
                        overflow, and Ei of a negative argument is bounded
      b > 0             everything is factored by e^{bw}: the reduced brackets
                        are O(1) and the single exponential is reinstated in
                        log space, so the result overflows to +inf ONLY when
                        the true value exceeds the double range

    Where the true value is not representable the return is +inf (the value is
    genuinely too large), and where an input is invalid (w <= 1) it is NaN.
    Nothing is clipped.
    """
    b = np.asarray(b, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    b, w = np.broadcast_arrays(b, w)
    b = np.array(b, dtype=np.float64, copy=True)
    w = np.array(w, dtype=np.float64, copy=True)
    N = 1.0 - 1.0 / w
    x = b * w                      # b w
    out = [np.full(b.shape, np.nan) for _ in range(5)]

    bad = ~(w > 1.0) | ~np.isfinite(b)
    ser = (np.abs(x) <= _SER_X) & ~bad
    neg = (x < -_SER_X) & ~bad
    pos = (x > _SER_X) & ~bad

    if ser.any():
        vals = _delta_series(b[ser], w[ser])
        for o, v in zip(out, vals):
            o[ser] = v

    if neg.any():
        bb, ww, NN, xx = b[neg], w[neg], N[neg], x[neg]
        u = np.exp(xx)             # e^{bw} <= 1
        v = np.exp(bb)             # e^{b}  <= 1
        t = bb * (ww - 1.0)
        # (u - v)/b: exact as a difference unless w ~ 1, where expm1 is used
        du = np.where(np.abs(t) < 0.5, v * np.expm1(t), u - v)
        D1 = (_ei_gamma(xx) - _ei_gamma(bb)) / NN
        D = (np.expm1(bb) - np.expm1(xx) / ww) / NN + bb * D1
        D2 = du / bb / NN
        D3 = (u * (xx - 1.0) + v * (1.0 - bb)) / bb ** 2 / NN
        q = lambda s: s * s - 2.0 * s + 2.0
        D4 = (u * q(xx) - v * q(bb)) / bb ** 3 / NN
        for o, val in zip(out, (D, D1, D2, D3, D4)):
            o[neg] = val

    if pos.any():
        bb, ww, NN, xx = b[pos], w[pos], N[pos], x[pos]
        em = np.exp(-xx)                       # e^{-bw}, underflows to 0 safely
        et = np.exp(bb - xx)                   # e^{-b(w-1)} <= 1
        Sx = _s_scaled(xx)                     # e^{-x}Ei(x) - 1/x
        Gy = _g_scaled(bb)                     # e^{-b}Ei(b)
        lw = np.log(ww)
        # reduced (i.e. e^{-bw} times) values; every bracket is O(1) and > 0
        rD1 = Sx + 1.0 / xx - em * lw - et * Gy
        # b*(1/x) cancels the -1/w of D exactly; done analytically here
        rD0 = em * np.expm1(bb) + em / ww + bb * Sx - bb * em * lw - bb * et * Gy
        rD2 = (1.0 - et) / bb
        rD3 = ((xx - 1.0) + et * (1.0 - bb)) / bb ** 2
        q = lambda s: s * s - 2.0 * s + 2.0
        rD4 = (q(xx) - et * q(bb)) / bb ** 3
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for o, r in zip(out, (rD0, rD1, rD2, rD3, rD4)):
                # reinstate e^{bw} in log space: +inf iff the true value really
                # does exceed the double range, 0 iff it really underflows
                lg = xx + np.log(np.abs(r)) - np.log(NN)
                o[pos] = np.sign(r) * np.exp(lg)
    return tuple(out)


def ioni_cgf_derivs(steps, theta, order=2):
    """K and its first `order` derivatives (order <= 4) for the CENTERED
    ionization CGF of a pooled block.

    steps: (n, 11) ioniurbanv records whose column 10 already carries the
    per-step transport weight (same convention as model_phi, which folds the
    weight into the qop-per-MeV column and then calls the exponent with unit
    weight). theta: scalar or array.

    Centered means K'(0) = 0 -- the variable is the deviation from the mean
    loss, which is what the fit's reference already subtracts.

    order defaults to 2 so that existing callers keep getting (K, K1, K2).
    order=4 adds the analytic K''' and K'''', which the score
    psi = theta + K'''/(2 K''^2) and its derivative need; they used to be taken
    by finite difference of K''.
    """
    if not 0 <= order <= 4:
        raise ValueError("order must be 0..4")
    theta = np.atleast_1d(np.asarray(theta, dtype=np.float64))
    n = order + 1
    K = [np.zeros_like(theta) for _ in range(5)]
    if len(steps) == 0:
        return tuple(K[:n])

    reg = steps[:, 0]
    # regime 2 (CVH_IONI_EXACTDELTA) repurposes column 6 as the exact delta
    # spectrum's normalization xi, not a collision count; reading it as one
    # would silently give a wrong CGF. Refuse rather than mis-read.
    if np.any(reg == 2):
        raise NotImplementedError(
            "ioni_cgf_derivs: regime-2 (exact delta spectrum) records are not "
            "supported by the saddlepoint CGF; use the exact-inversion route "
            "(cf_track_resolution.ioni_step_exponent).")
    gs = steps[:, 10].astype(np.float64) * 1e-3     # weighted qop per MeV

    # --- steps already in the Gaussian regime: exact quadratic
    mg = reg == 0
    if mg.any():
        V = float(np.sum(steps[mg, 1].astype(np.float64) * gs[mg] ** 2))
        K[0] += 0.5 * V * theta ** 2
        K[1] += V * theta
        K[2] += V

    mu = ~mg
    if not mu.any():
        return tuple(K[:n])
    gam = steps[mu, 9]
    gsu = gs[mu]

    # --- the two discrete excitation channels: compound Poisson at fixed loss
    #     K = sum a (e^{theta x} - 1 - theta x),  x = gs * e_j
    for ja, je in ((2, 3), (4, 5)):
        aj = steps[mu, ja]
        ej = steps[mu, je] * gam
        act = (aj > 0.0) & (ej > 0.0)
        if not act.any():
            continue
        x = (gsu[act] * ej[act])[:, None]           # (ns, 1)
        a = aj[act][:, None]
        xt = x * theta[None, :]
        # no clip: if this genuinely overflows the caller must see inf
        with np.errstate(over="ignore"):
            ex = np.exp(xt)
        K[0] += np.sum(a * (ex - 1.0 - xt), axis=0)
        K[1] += np.sum(a * x * (ex - 1.0), axis=0)
        for m in (2, 3, 4):
            if m < n:
                K[m] += np.sum(a * x ** m * ex, axis=0)

    # --- the delta-ray channel: 1/E^2 spectrum on [e0, tmax]
    a3 = steps[mu, 6]
    e0 = steps[mu, 7] * gam
    tmax = steps[mu, 8] * gam
    act = (a3 > 0.0) & (tmax > e0) & (e0 > 0.0)
    if act.any():
        c = (gsu[act] * e0[act])[:, None]           # (ns, 1) qop per e0
        w = (tmax[act] / e0[act])[:, None]
        a = a3[act][:, None]
        b = c * theta[None, :]
        D = _delta_derivs(b, np.broadcast_to(w, b.shape))
        for m in range(n):
            K[m] += np.sum(a * c ** m * D[m], axis=0)
    return tuple(K[:n])


def heavy_tail_sign(steps):
    """Sign of theta on which the exponentials of the delta-ray term GROW.

    A delta ray shifts z by gs*E with E > 0, so the heavy (power-law) tail of
    the block lies in the direction sign(gs) and the bounded hard edge in the
    direction -sign(gs). Returns the sign of theta on which e^{b w} decays,
    i.e. the numerically easy side.

    IT IS NOT A CONVERGENCE BOUNDARY. The jumps are bounded (delta rays on
    [e0, tmax], excitations at fixed e1, e2), so E[e^{theta X}] is finite for
    every real theta and the CGF is entire (NOTES 2026-08-13 XVIII). K is
    defined, smooth and evaluable on BOTH sides.
    """
    if len(steps) == 0:
        return 1.0
    gs = steps[:, 10].astype(np.float64)
    tot = float(np.sum(gs))
    return -1.0 if tot > 0 else 1.0


def convergent_sign(steps):
    """DEPRECATED alias of heavy_tail_sign, kept so existing callers are not
    silently changed.

    It was named for the belief that K(theta) diverges on the other half-line.
    That belief was wrong: the CGF is entire and the "divergence" was e^{bw}
    overflowing against a clip in _delta_derivs (NOTES 2026-08-13 XVII/XVIII).
    New code should use heavy_tail_sign (for the direction of the tail) or
    theta_overflow_limit (for the actual numerical bound), and should sample
    theta on BOTH sides.
    """
    return heavy_tail_sign(steps)


def theta_overflow_limit(steps, margin=0.98):
    """|theta| beyond which the delta-ray term of K exceeds the double range.

    On the side where e^{bw} grows, b w = theta * gs * tmax, so the bound is
    theta_max = _LOG_MAX / max|gs * tmax|. This is a FLOATING-POINT limit, not
    a property of the distribution: K itself is finite for all theta.
    """
    if len(steps) == 0:
        return np.inf
    reg = steps[:, 0]
    mu = reg != 0
    if not mu.any():
        return np.inf
    gam = steps[mu, 9]
    gs = steps[mu, 10].astype(np.float64) * 1e-3
    tmax = steps[mu, 8] * gam
    m = np.max(np.abs(gs * tmax))
    return np.inf if m <= 0 else margin * _LOG_MAX / m


def theta_grid(steps, n=1500, lo=1e-8, two_sided=True):
    """Log-spaced theta grid, both signs, always containing 0.

    The scale is 1/sqrt(K''(0)). The two sides are NOT capped the same way:
      * on the side where e^{bw} decays (heavy_tail_sign) nothing can overflow,
        so theta runs far out -- K' saturates at the block's hard edge there;
      * on the other side theta is capped at theta_overflow_limit, which is a
        double-precision bound, not a convergence bound. In z units that cap
        already corresponds to r ~ -1e2 and beyond, i.e. hundreds of sigma into
        the power-law tail, so nothing of the distribution is lost.

    two_sided=False reproduces the old one-sided sampling, for A/B tests only.
    """
    _, _, k2 = ioni_cgf_derivs(steps, 0.0)
    s = 1.0 / np.sqrt(float(np.atleast_1d(k2)[0]))
    top = theta_overflow_limit(steps)
    sgn = heavy_tail_sign(steps)
    hi_safe = 1e10 * s
    hi_over = min(1e10 * s, top)
    g_safe = sgn * np.geomspace(lo * s, hi_safe, n)
    if not two_sided:
        return np.sort(np.concatenate([[0.0], g_safe]))
    g_over = -sgn * np.geomspace(lo * s, hi_over, n)
    return np.sort(np.concatenate([g_over, [0.0], g_safe]))


def saddlepoint_curve(steps, thetas):
    """Parametric (q, log p) along the saddlepoint, indexed by theta.

    Far more robust than solving K'(theta) = q for a given q: K' is monotone,
    so scanning theta sweeps q monotonically and no root-find is needed. This
    matters here because the variance is tail-dominated (kappa2 can be ~1e2 x
    the core width), so a q-grid built from sqrt(K'') is wildly mis-scaled.
    """
    K, K1, K2 = ioni_cgf_derivs(steps, thetas)
    ok = np.isfinite(K) & np.isfinite(K1) & np.isfinite(K2) & (K2 > 0)
    q = np.where(ok, K1, np.nan)
    lp = np.where(ok, K - thetas * K1 - 0.5 * np.log(2.0 * np.pi * np.abs(K2)),
                  -np.inf)
    return q, lp


def solve_saddlepoint(steps, q, th0=0.0, tol=1e-12, itmax=60):
    """Solve K'(theta) = q. K'' > 0 so K' is monotone: Newton with bisection
    fallback converges in a few steps."""
    th = float(th0)
    for _ in range(itmax):
        _, k1, k2 = ioni_cgf_derivs(steps, th)
        f = float(k1[0]) - q
        if abs(f) < tol * max(1.0, abs(q)):
            break
        d = float(k2[0])
        if not np.isfinite(d) or d <= 0.0:
            return np.nan
        step = f / d
        # damp: the delta-ray CGF is very stiff on the positive-loss side
        step = np.clip(step, -0.5 * max(abs(th), 1.0), 0.5 * max(abs(th), 1.0))
        th -= step
        if not np.isfinite(th):
            return np.nan
    return th


def log_density(steps, q):
    """log p(q) by saddlepoint, including the -1/2 log K'' term that carries
    the mean-vs-mode displacement."""
    q = np.atleast_1d(np.asarray(q, dtype=np.float64))
    out = np.full(q.shape, -np.inf)
    th = 0.0
    for i, qi in enumerate(q):
        th = solve_saddlepoint(steps, qi, th0=th)
        if not np.isfinite(th):
            th = 0.0
            continue
        k, _, k2 = ioni_cgf_derivs(steps, th)
        if k2[0] <= 0.0:
            continue
        out[i] = float(k[0]) - th * qi - 0.5 * np.log(2.0 * np.pi * float(k2[0]))
    return out


def mode(steps, thmax=1.0e3, n=4001, two_sided=True):
    """Mode of the block's loss-deviation density, in the units of the
    weighted records (z units when the weights carry 1/sigma).

    Scans theta rather than q. Do NOT build a q-grid from sqrt(K''(0)): the
    variance is tail-dominated (measured kappa2 ~ 366 in z units at pT=3
    against a core width ~ 1), so such a grid is off by two orders of magnitude
    and lands entirely in the tail.

    theta is now scanned on BOTH sides (the CGF is entire; see heavy_tail_sign).
    The mode sits on the bounded side, so this is expected to leave the answer
    unchanged -- two_sided=False is kept so that can be checked rather than
    assumed.

    Returns (q_mode, kappa2). kappa2 is returned for diagnostics only -- it is
    NOT a width.
    """
    _, _, k2 = ioni_cgf_derivs(steps, 0.0)
    kap2 = float(k2[0])
    if not np.isfinite(kap2) or kap2 <= 0.0:
        return np.nan, np.nan
    s = heavy_tail_sign(steps)
    g = np.logspace(-8, np.log10(thmax), n - 1)
    if two_sided:
        top = theta_overflow_limit(steps)
        th = np.sort(np.concatenate([-s * g[g < top], [0.0], s * g]))
    else:
        th = s * np.concatenate([[0.0], g])
    q, lp = saddlepoint_curve(steps, th)
    if not np.any(np.isfinite(lp)):
        return np.nan, kap2
    i = int(np.nanargmax(np.where(np.isfinite(lp), lp, -np.inf)))
    if 0 < i < len(q) - 1 and np.all(np.isfinite(lp[i - 1:i + 2])):
        y0, y1, y2 = lp[i - 1], lp[i], lp[i + 1]
        den = y0 - 2 * y1 + y2
        # parabolic refine in q (grid is non-uniform, so use the local spacing)
        if den != 0 and np.isfinite(q[i - 1]) and np.isfinite(q[i + 1]):
            h = 0.5 * (q[i + 1] - q[i - 1])
            return float(q[i] - 0.5 * h * (y2 - y0) / den), kap2
    return float(q[i]), kap2
