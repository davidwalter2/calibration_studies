#!/usr/bin/env python3
"""Hard knock-on collisions exactly: the joint law of energy loss and
deflection, and the exact energy -> q/p map.

THE TWO PIECES
--------------
The ionisation channel (cf_track_resolution.ioni_step_exponent, regime 2/3)
carries the knock-on spectrum

    dN/dT = (xi/T^2) [1 - b2 T/tmax (+ T^2/(2E^2), spin 1/2)] f_K(T)   on [e0, tmax]

and for e+- projectiles (regime 4/5) G4MollerBhabhaModel's Moller / Bhabha
law, which ends at T0/2 for e- (law_rate, law_top).  Every piece below takes
the law and its range from the record's regime, so one code path serves all
species; tmax is the kinematic ceiling in all regimes.

as a centred compound Poisson in the ENERGY, mapped LINEARLY into q/p
(d qop = q cs dT, cs = E/p^3).  Geant4 delivers every collision above the e-
production threshold (KNOCKON_TCUT) as an explicit delta ray: one event with
both

    d qop = q (1/p' - 1/p) = q cs T_eff(T),
            T_eff = p^2 (p - p')/(E p') = p^2 T (2E - T)/(E p' (p + p')),
            p' = sqrt((E - T)^2 - M^2)
    theta = sqrt(2 m_e T (1 - T/tmax)) / p'
            (the electron's transverse momentum, exact for a free electron,
            over the primary's momentum after the collision)

with the kick's azimuth uniform; its msc carries the electrons only below the
threshold.  The model mirrors that split: the scattering channel's electron
term stops at KNOCKON_TCUT (cf_track_resolution.ms_step_exponent) and this
channel turns each collision above it from the ionisation channel's

    [e^{i a T} - 1 - i a T]      into      [e^{i a T_eff} J0(b theta) - 1 - i a T].

The centring stays linear: the reference subtracts the mean ENERGY loss, so
the channel's mean becomes the Jensen excess of q/p over q cs E[dE].  The
difference, added to the exponent, is

    dS = INT dN(T) [ e^{i a X} J - e^{i a T} ] dT ,

X = T_eff under QOP_EXACT (else T), J = J0(b theta) under KNOCKON_JOINT
(else 1; the scattering channel's electron term then runs to the kinematic
ceiling instead, the independent-channel model).  Both DEFAULT ON; with both
off dS = 0, nothing is evaluated and every number is bit-identical to the
linear model.

VALIDATION (realmat_full, 200 k per sample; CLOSURE_STATE.md)
  J/psi one-plane vertex mass (realmat_ditrack.py), 19 planes x 9 probes:
    even > 3 sigma 98 -> 3 of 171, odd 14 -> 1, mean 3.3 -> 0.9 sigma;
    each switch alone leaves one failure (joint: shape and core; exact: mean).
  single track, 8 species x 5 directions: > 3 sigma 471 -> 349 of 6840, all of
    it in the bending plane of muons and pions; kaons and protons (Tmax 40,
    10 MeV) move by <= 2.2e-4.

    KNOCKON_JOINT  e^{iaX} (J0 - 1): the collision's deflection with exact
                   two-body kinematics, jointly with its loss -- the angular
                   marginal of every delta ray above the threshold (where the
                   scattering channel's electron term stops) and the shape of
                   any direction that mixes q/p with the angles.
    QOP_EXACT      e^{i a T_eff} - e^{i a T}: the Jensen term.  Only the hard
                   collisions are stretched (T_eff/T - 1 is 29 % at Tmax for a
                   3.1 GeV muon, 1e-3 at 1 MeV), so the tail and the mean move
                   and the core does not.  The radiative channel gets the same
                   substitution (cf_brems_exact.rad_exponent, exact_qop).

WEIGHTS
-------
per unit T:   a = t q (a_k . A_s)[qop] cs / sigma          (the ionisation weight)
per radian:   b = t |((a_k . A_s)[lambda], (a_k . A_s)[phi]/cos lambda)| / sigma
                                                            (the scattering weight)
with the collision point uniform along the step: KNOCKON_NSUB sub-steps at the
centres of NSUB equal parts, the transport interpolated linearly between the
step's start and end, each with 1/NSUB of the rate (the scattering channel's
sub-step rule).  NSUB = 1 is the step midpoint; NSUB = 2 changes dS by
< 2e-7 (mu-, planes 0 and 9).

QUADRATURE
----------
a tmax reaches 1e3-1e4 where the CF still matters, so no rule in T resolves
e^{iaT}.  Both oscillatory pieces are done by FILON-SIMPSON on the node set
(amplitude quadratic per panel, phase exact): INT dN e^{iaT} dT in the variable T and
INT dN J e^{iaX} dT in the variable x = X(T) (dT/dx folded into the
amplitude), so the phase is linear in the integration variable in both.  The
non-oscillatory INT dN (J - 1) dT is Simpson on the same panels.  Nodes:
KNOCKON_NPERDEC per decade from max(e0, KNOCKON_TCUT) to 0.9 tmax, then 30
log-spaced in (tmax - T) down to 1e-7 tmax, where theta -> 0; steps whose xi is
below KNOCKON_THIN of the leg's largest (the air gaps: 142 of 247 steps, 1 % of
the material on realmat_full) take KNOCKON_NPERDEC_THIN.  The lower limit is
PHYSICS, not numerics: at large t the joint term's integrand does not vanish
at small T (b theta(T) is not small there), and below the production threshold
the simulation's channels are independent.  A piecewise
LINEAR amplitude is not enough: the two oscillatory pieces are large and
nearly cancel, each carrying the (dx/x)^2 error of interpolating 1/T^2, and
the Jensen piece came out 12 % wrong at 40 nodes per decade; the quadratic
panels take it to 2e-3 at the same cost.
"""

import os

import numpy as np
from scipy.special import j0

import cf_track_resolution as ctr

ME_MEV = 0.51099895


def _flag(name, dflt):
    v = os.environ.get(name)
    return dflt if v is None else v not in ("0", "", "false", "False")


# DEFAULT ON, like the MS harmonisations: the model models the simulation.
# CF_KNOCKON_JOINT=0 / CF_QOP_EXACT=0 (or the module globals) give the linear,
# independent-channel model bit for bit.
KNOCKON_JOINT = 1.0 if _flag("CF_KNOCKON_JOINT", True) else 0.0
QOP_EXACT = 1.0 if _flag("CF_QOP_EXACT", True) else 0.0
# The q/p VARIABLE of the exact map (with QOP_EXACT).  0: q/p itself, one
# collision's change q (1/p' - 1/p).  1: q/p on a LOGARITHMIC scale,
# |q/p|_ref ln(|q/p| / |q/p|_ref) -- q ln(p/p')/p per collision, which ADDS
# over successive collisions exactly (p_final = p0 prod(1 - v_i)), so
# repeated radiation compounds without error; both reduce to q cs T as
# T -> 0.  The closure's statistic follows the switch (`qop_dev`).
QOP_LOG = 1.0 if _flag("CF_QOP_LOG", False) else 0.0
KNOCKON_NSUB = 1
KNOCKON_NPERDEC = 40
# The joint law starts at the SIMULATION's e- production threshold: below it
# Geant4 has no knock-on electron -- the loss is continuous (restricted dE/dx
# plus Urban straggling) and the deflection comes from msc, sampled
# independently -- so the channels ARE independent there.  0.99 keV is
# Geant4's floor and the e- cut of air, silicon, carbon fibre and most of the
# realmat_full path at DefaultCutValue = 1e-4 cm (a few keV in the dense
# composites).  [MeV]
KNOCKON_TCUT = 0.99e-3
KNOCKON_NPERDEC_THIN = 10   # node density for thin steps ...
KNOCKON_THIN = 1e-2         # ... whose xi is below this fraction of the leg's largest
# A collision that leaves the primary with less than PMIN_FRAC of its momentum
# never reaches a plane; the exact map's p' -> 0 endpoint (an e+ giving the
# electron all its kinetic energy) is cut there, as cf_brems_exact's radiative
# map floors p'.  It binds for no heavy species (tmax << T0).
PMIN_FRAC = 1e-3
# ioniurbanv regimes carrying an exact knock-on law: 2/3 Bethe-Bloch spin 1/2
# and 0, 4/5 Moller (e-) and Bhabha (e+)
EXACT_REGIMES = (2, 3, 4, 5)
PHYSICS_GLOBALS = ("KNOCKON_JOINT", "QOP_EXACT", "QOP_LOG", "KNOCKON_NSUB",
                   "KNOCKON_NPERDEC", "KNOCKON_TCUT", "KNOCKON_NPERDEC_THIN",
                   "KNOCKON_THIN", "PMIN_FRAC")
_NOT_PHYSICS = ("ME_MEV", "_TCHUNK", "_NSER", "_ZSER", "EXACT_REGIMES")
_TCHUNK = 2048          # t points per block (memory only)


def physics_state():
    g = globals()
    return tuple((n, repr(g[n])) for n in PHYSICS_GLOBALS)


def active():
    return bool(KNOCKON_JOINT) or bool(QOP_EXACT)


# ------------------------------------------------------------- kinematics
def qop_map(T, E, p, pp):
    """The q/p-variable change of an energy loss T at (E, p) with outgoing
    momentum pp, in units of the linear one (-> T as T -> 0): p^2 T (2E - T) /
    (E pp (p + pp)) for q/p itself, (p^2/E) ln(p/pp) under QOP_LOG.
    Cancellation-free: ln(p/pp) = -log1p(r)/2 with r = T (T - 2E)/p^2 unless
    pp is floored (pp^2 above the unfloored value), then r = pp^2/p^2 - 1."""
    if QOP_LOG:
        r = T * (T - 2.0 * E) / (p * p)
        fl = pp * pp > (p * p) * (1.0 + r) * (1.0 + 1e-9) + 1e-300
        r = np.where(fl, pp * pp / (p * p) - 1.0, r)
        return (p * p / E) * (-0.5 * np.log1p(r))
    return p * p * T * (2.0 * E - T) / (E * pp * (p + pp))


def dqop_map(T, E, p, pp):
    """d qop_map / dT (pp unfloored)."""
    if QOP_LOG:
        return (p * p / E) * (E - T) / (pp * pp)
    return (p ** 3 / E) * (E - T) / pp ** 3


def qop_dev(qop, refqop):
    """The q/p deviation the closure statistics carry, in the variable the
    map uses: qop - refqop, or |refqop| ln(qop/refqop) with refqop's sign
    under QOP_LOG (qop and refqop share the charge's sign)."""
    qop = np.asarray(qop, dtype=np.float64)
    if QOP_LOG and QOP_EXACT:
        return refqop * np.log(qop / refqop)
    return qop - refqop


def t_eff(T, E, p):
    """The q/p change of an energy loss T in units of the linear one (so
    T_eff -> T as T -> 0), in the map's variable (`qop_map`)."""
    M2 = E * E - p * p
    pp = np.sqrt(np.maximum((E - T) ** 2 - M2, 1e-300))
    return qop_map(T, E, p, pp)


def dteff_dt(T, E, p):
    M2 = E * E - p * p
    pp = np.sqrt(np.maximum((E - T) ** 2 - M2, 1e-300))
    return dqop_map(T, E, p, pp)


def theta_kick(T, E, p, tmax):
    M2 = E * E - p * p
    pp = np.sqrt(np.maximum((E - T) ** 2 - M2, 1e-300))
    return np.sqrt(np.maximum(2.0 * ME_MEV * T * (1.0 - T / tmax), 0.0)) / pp


def nodes(e0, tmax, nper=None, tlo=None):
    """Nodes on [max(e0, tlo), tmax]; tlo in MeV, default KNOCKON_TCUT."""
    nper = KNOCKON_NPERDEC if nper is None else nper
    tlo = KNOCKON_TCUT if tlo is None else tlo
    lo = max(e0, tlo)
    top = 0.9 * tmax
    n1 = max(int(np.ceil(np.log10(top / lo) * nper)), 2)
    a = np.geomspace(lo, top, n1 + 1)
    d = tmax * np.geomspace(0.1, 1e-7, 31)[1:]
    x = np.concatenate([a, tmax - d, [tmax]])
    if len(x) % 2 == 0:                 # Filon-Simpson panels need an odd count
        x = np.insert(x, 1, np.sqrt(x[0] * x[1]))
    return x


def rate(T, xi, tmax, b2, E, spinhalf, kok):
    """dN/dT of the channel, exactly as ioni_step_exponent builds it."""
    r = (xi / T ** 2) * (1.0 - b2 * T / tmax)
    if spinhalf:
        r = r + xi / (2.0 * E * E)
    if kok:
        f = ctr.kokoulin_factor(T, E)
        f = np.where(T > ctr.IONI_KOKOULIN_TCUT, f, 1.0)
        r = r * (1.0 + ctr.IONI_KOKOULIN * (f - 1.0))
    return r


def law_rate(T, xi, tmax, b2, E, reg, kok):
    """dN/dT of the record's knock-on law.  Regimes 2/3 are the Bethe-Bloch
    forms (`rate`); 4/5 are G4MollerBhabhaModel's (Geant4ePropagator's
    UrbanFluctRecord), obtained by differentiating
    ComputeCrossSectionPerElectron in xmax, with x = T/tmax and tmax = T0 the
    projectile's kinetic energy:
      4 Moller (e-): (xi/T^2) [1 + (1-gg) x^2 + x^2/(1-x)^2 - gg x/(1-x)],
                     gg = (2 gamma - 1)/gamma^2, zero above x = 1/2
      5 Bhabha (e+): (xi/T^2) [1 + b2 (-c1 x + c2 x^2 - c3 x^3 + c4 x^4)],
                     y = 1/(1+gamma), c1 = 2 - y^2, c2 = (1-2y)(3+y^2),
                     c4 = (1-2y)^3, c3 = c4 + (1-2y)^2"""
    if reg == 2 or reg == 3:
        return rate(T, xi, tmax, b2, E, reg == 2, kok)
    x = T / tmax
    gam = E / ME_MEV
    if reg == 4:
        gg = (2.0 * gam - 1.0) / (gam * gam)
        with np.errstate(divide="ignore", invalid="ignore"):
            y = x / (1.0 - x)
            r = 1.0 + x * x * (1.0 - gg) + y * y - gg * y
        return np.where(x <= 0.5, (xi / T ** 2) * r, 0.0)
    if reg == 5:
        yy = 1.0 / (1.0 + gam)
        y12 = 1.0 - 2.0 * yy
        c1 = 2.0 - yy * yy
        c2 = y12 * (3.0 + yy * yy)
        c4 = y12 ** 3
        c3 = c4 + y12 * y12
        r = 1.0 + b2 * (-c1 * x + c2 * x * x - c3 * x ** 3 + c4 * x ** 4)
        return np.where(x <= 1.0, (xi / T ** 2) * r, 0.0)
    raise ValueError(f"knock-on law: no regime {reg}")


def law_base_exponent(xi, e0, tmax, b2, E, reg, a):
    """The ionisation channel's knock-on part for one regime-4/5 step, linear
    map:  INT_{e0}^{top} dN (e^{i a T} - 1 - i a T) dT,  a (nt,) per MeV,
    top = the law's own ceiling (T0/2 Moller, T0 Bhabha).  The 1/T^2 part in
    closed form (cf_track_resolution.exact_delta_exponent with beta^2 = 0, no
    spin term), the law's O(T/T0) remainder by Filon-Simpson on `nodes` --
    whose quadratic panels make the constant and linear terms of the
    remainder's e^{iaT} cancel against the Simpson terms panel by panel."""
    top = 0.5 * tmax if reg == 4 else tmax
    a = np.asarray(a, dtype=np.float64)
    S = ctr.exact_delta_exponent(np.array([xi]), np.array([e0]), np.array([top]),
                                 np.zeros(1), np.array([E]), a[None, :],
                                 spinhalf=False)
    T = nodes(e0, top, KNOCKON_NPERDEC, tlo=e0)
    h = law_rate(T, xi, tmax, b2, E, reg, False) - xi / T ** 2
    S = S + filon(T, h, a)
    S = S - simpson(T, h[None, :])[0]
    S = S - 1j * a * simpson(T, (h * T)[None, :])[0]
    return S


def law_top(reg, tmax, E, p):
    """Where the law's channel ends: the record's tmax, T0/2 for Moller, and
    never past the PMIN_FRAC floor on the primary's outgoing momentum."""
    top = 0.5 * tmax if reg == 4 else tmax
    tcap = E - np.sqrt(E * E - p * p + (PMIN_FRAC * p) ** 2)
    return tcap if tcap < top else top


# ------------------------------------------------------------- quadrature
_NSER = 14
_ZSER = 0.2             # series below |z| = 0.2 (last term 0.2^14/14! ~ 2e-21);
                        # the closed forms lose < 1e-13 relative above it


def _fmom(z):
    """f_n(z) = INT_0^1 s^n e^{zs} ds for n = 0, 1, 2, elementwise.
    Series sum_k z^k/(k! (n+k+1)) for |z| < _ZSER (no cancellation), closed
    forms beyond."""
    f = [np.empty_like(z) for _ in range(3)]
    sm = np.abs(z) < _ZSER
    if sm.any():
        w = z[sm]
        term = np.ones_like(w)
        acc = [np.zeros_like(w) for _ in range(3)]
        for k in range(_NSER):
            if k:
                term = term * w / k
            for n in range(3):
                acc[n] += term / (n + k + 1)
        for n in range(3):
            f[n][sm] = acc[n]
    b = ~sm
    if b.any():
        w = z[b]
        ez = np.exp(w)
        f[0][b] = np.expm1(w) / w
        f[1][b] = (ez * (w - 1.0) + 1.0) / (w * w)
        f[2][b] = (ez * (w * w - 2.0 * w + 2.0) - 2.0) / (w * w * w)
    return f


def filon(x, h, a):
    """INT e^{i a x} h(x) dx by Filon-Simpson: h quadratic on each panel
    [x_{2i}, x_{2i+2}] (Newton form through the three nodes, nonuniform), the
    phase exact.  x: (n,) with n odd, h: (n,) or (nt, n), a: (nt,).  Error
    O((dx/x)^4) for the 1/T^2-like amplitudes here.  Returns (nt,) complex."""
    if len(x) % 2 == 0:
        raise ValueError("filon needs an odd node count")
    h = np.broadcast_to(h, (len(a), len(x)))
    x0, x1, x2 = x[0:-2:2], x[1:-1:2], x[2::2]
    h0, h1, h2 = h[:, 0:-2:2], h[:, 1:-1:2], h[:, 2::2]
    d01 = (h1 - h0) / (x1 - x0)
    d12 = (h2 - h1) / (x2 - x1)
    c2 = (d12 - d01) / (x2 - x0)
    c1 = d01 - c2 * (x1 - x0)
    D = x2 - x0
    z = 1j * a[:, None] * D[None, :]
    f0, f1, f2 = _fmom(z)
    e0 = np.exp(1j * a[:, None] * x0[None, :])
    return np.sum(D * e0 * (h0 * f0 + c1 * D * f1 + c2 * D * D * f2), axis=1)


def _fmom_centred(z):
    """(f_n, g_n, q_n)(z), n = 0, 1, 2: f_n = INT_0^1 s^n e^{zs} ds (`_fmom`),
    g_n = INT_0^1 s^n (e^{zs} - 1) ds and q_n = INT_0^1 s^n (e^{zs} - 1 - zs) ds.
    Series sum_{k >= 0, 1, 2} z^k/(k! (n+k+1)) for |z| < _ZSER (no
    cancellation: g_n, q_n are O(z), O(z^2) there), closed forms beyond."""
    f = [np.empty_like(z) for _ in range(3)]
    g = [np.empty_like(z) for _ in range(3)]
    q = [np.empty_like(z) for _ in range(3)]
    sm = np.abs(z) < _ZSER
    if sm.any():
        w = z[sm]
        term = np.ones_like(w)
        af = [np.zeros_like(w) for _ in range(3)]
        ag = [np.zeros_like(w) for _ in range(3)]
        aq = [np.zeros_like(w) for _ in range(3)]
        for k in range(_NSER):
            if k:
                term = term * w / k
            for n in range(3):
                t = term / (n + k + 1)
                af[n] += t
                if k >= 1:
                    ag[n] += t
                if k >= 2:
                    aq[n] += t
        for n in range(3):
            f[n][sm], g[n][sm], q[n][sm] = af[n], ag[n], aq[n]
    b = ~sm
    if b.any():
        w = z[b]
        ez = np.exp(w)
        fb = (np.expm1(w) / w, (ez * (w - 1.0) + 1.0) / (w * w),
              (ez * (w * w - 2.0 * w + 2.0) - 2.0) / (w * w * w))
        for n in range(3):
            f[n][b] = fb[n]
            g[n][b] = fb[n] - 1.0 / (n + 1)
            q[n][b] = g[n][b] - w / (n + 2)
    return f, g, q


_SINM_C = (-1.0 / 6.0, 1.0 / 120.0, -1.0 / 5040.0, 1.0 / 362880.0, -1.0 / 39916800.0,
           1.0 / 6227020800.0, -1.0 / 1307674368000.0, 1.0 / 355687428096000.0)


def _sinm(y):
    """sin(y) - y without the cancellation at small y: the Taylor series (Horner
    in y^2, complete to double precision) below |y| = 0.5, the direct form
    above."""
    y = np.asarray(y, dtype=np.float64)
    y2 = y * y
    p = np.full(y.shape, _SINM_C[-1])
    for c in _SINM_C[-2::-1]:
        p = p * y2 + c
    return np.where(np.abs(y) < 0.5, p * y2 * y, np.sin(y) - y)


def filon_centred(x, h, a):
    """INT h(x) (e^{i a x} - 1 - i a x) dx on the Filon-Simpson panels of
    `filon` (h quadratic per panel, the phase exact), the phase's constant and
    linear terms subtracted INSIDE each panel: with x = x0 + s D, y = a x0 and
    z = i a D,
        e^{iax} - 1 - iax = E0 e^{zs} + (e^{zs} - 1 - zs) + i y (e^{zs} - 1),
        E0 = e^{iy} - 1 - iy = -2 sin^2(y/2) + i (sin y - y),
    every piece O(a^2) or O(a) x O(a) where a is small, so the result keeps
    its relative precision for any a -- where `filon` minus the Simpson terms
    would be the O(a^2) difference of O(1) sums.  x: (n,) odd, h: (n,), a:
    (nt,).  Returns (nt,) complex."""
    if len(x) % 2 == 0:
        raise ValueError("filon needs an odd node count")
    x0, x1, x2 = x[0:-2:2], x[1:-1:2], x[2::2]
    h0, h1, h2 = h[0:-2:2], h[1:-1:2], h[2::2]
    d01 = (h1 - h0) / (x1 - x0)
    d12 = (h2 - h1) / (x2 - x1)
    c2 = (d12 - d01) / (x2 - x0)
    c1 = d01 - c2 * (x1 - x0)
    D = x2 - x0
    y = a[:, None] * x0[None, :]
    z = 1j * a[:, None] * D[None, :]
    f, g, q = _fmom_centred(z)
    hy = np.sin(0.5 * y)
    E0 = (-2.0 * hy * hy) + 1j * _sinm(y)
    c1D, c2D2 = c1 * D, c2 * D * D
    tot = (E0 * (h0 * f[0] + c1D * f[1] + c2D2 * f[2])
           + (h0 * q[0] + c1D * q[1] + c2D2 * q[2])
           + (1j * y) * (h0 * g[0] + c1D * g[1] + c2D2 * g[2]))
    return np.sum(D * tot, axis=1)


def simpson(x, h):
    """filon at a = 0 (the same panels, real amplitude): (nt,) from (nt, n)."""
    x0, x1, x2 = x[0:-2:2], x[1:-1:2], x[2::2]
    h0, h1, h2 = h[:, 0:-2:2], h[:, 1:-1:2], h[:, 2::2]
    d01 = (h1 - h0) / (x1 - x0)
    d12 = (h2 - h1) / (x2 - x1)
    c2 = (d12 - d01) / (x2 - x0)
    c1 = d01 - c2 * (x1 - x0)
    D = x2 - x0
    return np.sum(D * (h0 + c1 * D / 2.0 + c2 * D * D / 3.0), axis=1)


def step_correction(tau_a, tau_b, T, dN, X, dXdT, th, joint, exact, part="all"):
    """dS on the t points of one step: tau_a = t*a (nt,), tau_b = t*b (nt,).

    part: "all"   -- INT dN [e^{iaX} J - e^{iaT}];
          "map"   -- its J = 1 piece, INT dN [e^{iaX} - e^{iaT}];
          "joint" -- the rest, INT dN e^{iaX} (J - 1): the collision's
                     deflection, jointly with its loss."""
    if part == "map":
        return step_correction(tau_a, tau_b, T, dN, X, dXdT, None, False, exact)
    if part == "joint":
        if not joint:
            return np.zeros(len(tau_a), dtype=np.complex128)
        J = j0(tau_b[:, None] * th[None, :])
        amp = dN[None, :] * (J - 1.0)
        return filon(X, amp / dXdT[None, :], tau_a) if exact else filon(T, amp, tau_a)
    B = filon(T, dN, tau_a)
    J = j0(tau_b[:, None] * th[None, :]) if joint else 1.0
    if exact:
        A = filon(X, dN[None, :] * J / dXdT[None, :], tau_a)
    else:
        A = filon(T, dN[None, :] * J, tau_a)
    return A - B


# ------------------------------------------------------------- the channel
def knockon_rows(tau, st, rid, wq, wb, frac, part="all"):
    """dS of a set of ionisation ROWS (`cf_rows`): per entry (record rid,
    q/p weight wq [z per unit q/p, charge included], angular weight wb [z per
    radian], share frac of the record's rate), summed.  The record's law and
    nodes are evaluated once however many entries share it.  `part` selects
    the exact-map or the joint piece (`step_correction`)."""
    tau = np.asarray(tau, dtype=np.float64)
    S = np.zeros(len(tau), dtype=np.complex128)
    if not active() or not len(st) or not len(rid):
        return S
    st = np.asarray(st)
    reg = st[:, 0]
    if not np.isin(reg, EXACT_REGIMES).any():
        # regime-1 records carry no exact knock-on spectrum (and no beta^2, E)
        return S
    if st.shape[1] < 13:
        raise ValueError("knock-on channel needs the stride-13 exact-delta record")
    gam = st[:, 9]
    xi = st[:, 6] * gam * ctr.IONI_A3_SCALE
    e0 = st[:, 7] * gam
    tmax = st[:, 8] * gam * ctr.IONI_TMAX_SCALE
    b2, E = st[:, 11], st[:, 12]
    g = st[:, 10] * 1e-3
    p = E * np.sqrt(b2)
    kok_on = ctr.IONI_KOKOULIN != 0.0
    ismu = ctr._is_muon_record(b2, E, reg) & (E - ctr._MMU > ctr._KOK_MUMIN)
    act = np.isin(reg, EXACT_REGIMES) & (xi > 0) & (tmax > e0) & (e0 > 0)
    joint, exact = bool(KNOCKON_JOINT), bool(QOP_EXACT)
    xmax = float(xi[act].max()) if act.any() else 0.0
    rid = np.asarray(rid)
    order = np.argsort(rid, kind="stable")
    bounds = np.searchsorted(rid[order], np.arange(len(st) + 1))
    for s in np.flatnonzero(act):
        ents = order[bounds[s]:bounds[s + 1]]
        if not len(ents):
            continue
        thin = xi[s] < KNOCKON_THIN * xmax
        T = nodes(e0[s], law_top(reg[s], tmax[s], E[s], p[s]),
                  KNOCKON_NPERDEC_THIN if thin else KNOCKON_NPERDEC)
        dN = law_rate(T, xi[s], tmax[s], b2[s], E[s], int(reg[s]),
                      kok_on and bool(ismu[s]))
        X = t_eff(T, E[s], p[s]) if exact else T
        dXdT = dteff_dt(T, E[s], p[s]) if exact else np.ones_like(T)
        th = theta_kick(T, E[s], p[s], tmax[s]) if joint else None
        for e in ents:
            al = wq[e] * g[s]
            be = wb[e]
            if al == 0.0 and (be == 0.0 or not joint):
                continue
            for lo in range(0, len(tau), _TCHUNK):
                t = tau[lo:lo + _TCHUNK]
                S[lo:lo + _TCHUNK] += step_correction(
                    t * al, t * be, T, dN, X, dXdT, th, joint, exact, part) * frac[e]
    return S


def knockon_exponent(leg, A_end, A_start, avec, sigma, tau):
    """dS of one clean-propagation leg at plane k: its ionisation rows at the
    exact transport weights, KNOCKON_NSUB sub-steps (`cf_rows`)."""
    if not active() or not len(leg["ioni"]):
        return np.zeros(len(tau), dtype=np.complex128)
    import cf_rows
    rid, wq, wb, frac = cf_rows.transport_entries(leg, A_end, A_start, avec,
                                                  sigma, KNOCKON_NSUB, "vector")
    return cf_rows.knockon_rows(tau, leg["ioni"], rid, wq, wb, frac)


def mean_shift(leg, A_end, A_start, avec, sigma):
    """d/d(it) of dS at t = 0: the channel's mean in z units, analytic
    (INT dN (T_eff - T) a, sub-step averaged).  Zero unless QOP_EXACT."""
    if not QOP_EXACT or not len(leg["ioni"]):
        return 0.0
    st = leg["ioni"]
    reg, gam = st[:, 0], st[:, 9]
    if not np.isin(reg, EXACT_REGIMES).any() or st.shape[1] < 13:
        return 0.0
    xi = st[:, 6] * gam * ctr.IONI_A3_SCALE
    e0, tmax = st[:, 7] * gam, st[:, 8] * gam * ctr.IONI_TMAX_SCALE
    b2, E, g = st[:, 11], st[:, 12], st[:, 10] * 1e-3
    p = E * np.sqrt(b2)
    q = np.sign(leg["refqop"]) or 1.0
    v1 = np.einsum("i,sij->sj", avec, A_end)
    v0 = np.einsum("i,sij->sj", avec, A_start)
    kok_on = ctr.IONI_KOKOULIN != 0.0
    ismu = ctr._is_muon_record(b2, E, reg) & (E - ctr._MMU > ctr._KOK_MUMIN)
    act = np.isin(reg, EXACT_REGIMES) & (xi > 0) & (tmax > e0) & (e0 > 0)
    out = 0.0
    for s in np.flatnonzero(act):
        T = nodes(e0[s], law_top(reg[s], tmax[s], E[s], p[s]))
        dN = law_rate(T, xi[s], tmax[s], b2[s], E[s], int(reg[s]),
                      kok_on and bool(ismu[s]))
        m = np.trapezoid(dN * (t_eff(T, E[s], p[s]) - T), T)
        vbar = 0.5 * (v0[s] + v1[s])
        out += q * vbar[0] * g[s] / sigma * m
    return float(out)


# ================================================== THE FIT-LEVEL FAMILIES
# The same two pieces in the resolution CF of the CVH track fit
# (cf_track_resolution.extract / model_phi, cf_mass_likelihood.build_pairs /
# build_pairs_tt, and the in-maker `cvhcf` that is validated against them).
# There the model has no per-step transports: every step row carries the
# scalar weight of its block, fixed by the fit's influence coefficients, and
# the pieces ride on the rows that already carry their marginals.
#
#   MAP   dS_x = INT dN [e^{i a T_eff} - e^{i a T}]   on the IONISATION rows
#         (`ioniurbanv`, regime 2/3), over the channel's own spectrum and
#         range [max(e0, FIT_MAP_TLO tmax), tmax], a = w_io,b * cs * 1e-3 per
#         MeV (map_block), the part below the lower limit in closed form
#         (_map_below); and on the RADIATIVE rows with the radiative
#         block's weight (map_rad).
#   JOINT dS_j = INT dN (e^{i a X} - 1)(J0(b theta) - 1)   on the `msmoliv`
#         rows, over exactly what the discrete delta-ray family S_del
#         (cf_delta_ray.delta_step_exponent) carries: rate xi/T^2 times its
#         first-order spin factor, theta = sqrt(2 m_e T)/p, T in
#         [DELTA_TCUT, min(Tmax, DELTA_TMAXCAP)] (joint_rows).  X = T_eff when
#         the map is on, T otherwise.  b is the row's MS block weight, a the
#         ionisation weight of the same leg at the same step (pair_weights).
#
# The identity  e^{iaX}J - e^{iaT} - J + 1 = (e^{iaX} - e^{iaT})
# + (e^{iaX} - 1)(J - 1)  makes the two the split of step_correction's single
# integrand; the joint piece is evaluated as
#     INT dN (J - 1) e^{iaX} dT  -  INT dN (J - 1) dT
# (Filon-Simpson in x = X(T), Simpson), whose oscillatory amplitude dN (J - 1)
# vanishes at small T, so it carries none of the map piece's cancellation.
#
# Every row is its own quadrature (the per-row reference form); nothing is
# pooled here.
FIT_NPERDEC = 40
FIT_MAP_TLO = 1e-4       # map lower limit as a fraction of tmax (above e0)
PHYSICS_GLOBALS = PHYSICS_GLOBALS + ("FIT_NPERDEC", "FIT_MAP_TLO")


def _odd(x):
    if len(x) % 2 == 0:
        x = np.insert(x, 1, np.sqrt(x[0] * x[1]))
    return x


def _joint_nodes(tlo, thi, tkin, nper):
    """[tlo, thi] at nper per decade; when the range ends at the kinematic
    limit (thi == tkin, a hadron below the cap) the 30 nodes in (tmax - T)
    of `nodes` are added, where theta -> 0 under the exact factor."""
    if thi >= tkin:
        return nodes(tlo, tkin, nper, tlo=tlo)
    n1 = max(int(np.ceil(np.log10(thi / tlo) * nper)), 2)
    return _odd(np.geomspace(tlo, thi, n1 + 1))


def _refuse_epm(rows):
    """The fit-level families are the Bethe-Bloch (regime 2/3) forms; an e+-
    record (regime 4/5) is refused rather than silently skipped."""
    if len(rows) and np.isin(rows[:, 0], (4, 5)).any():
        raise ValueError("fit-level knock-on families: regime 4/5 (e+- "
                         "Moller/Bhabha) records are not implemented here")


def map_rows(rows, alpha, tau):
    """dS_x summed over ionisation rows (stride >= 13 `ioniurbanv`, energies
    in MeV) with per-row signed weights alpha [z per MeV].  Regime-1/0 rows
    carry no exact spectrum and contribute nothing (as knockon_exponent)."""
    _refuse_epm(rows)
    tau = np.asarray(tau, dtype=np.float64)
    S = np.zeros(len(tau), dtype=np.complex128)
    if not len(rows):
        return S
    if rows.shape[1] < 13:
        if ((rows[:, 0] == 2) | (rows[:, 0] == 3)).any():
            raise ValueError("knock-on map needs the stride-13 regime-2/3 record")
        return S
    reg, gam = rows[:, 0], rows[:, 9]
    xi = rows[:, 6] * gam * ctr.IONI_A3_SCALE
    e0 = rows[:, 7] * gam
    tmax = rows[:, 8] * gam * ctr.IONI_TMAX_SCALE
    b2, E = rows[:, 11], rows[:, 12]
    p = E * np.sqrt(b2)
    kok_on = ctr.IONI_KOKOULIN != 0.0
    ismu = ctr._is_muon_record(b2, E, reg) & (E - ctr._MMU > ctr._KOK_MUMIN)
    alpha = np.broadcast_to(np.asarray(alpha, dtype=np.float64), (len(rows),))
    act = (((reg == 2) | (reg == 3)) & (xi > 0) & (tmax > e0) & (e0 > 0)
           & (alpha != 0.0))
    for s in np.flatnonzero(act):
        T = nodes(e0[s], tmax[s], FIT_NPERDEC,
                  tlo=max(e0[s], FIT_MAP_TLO * tmax[s]))
        dN = rate(T, xi[s], tmax[s], b2[s], E[s], reg[s] == 2,
                  kok_on and bool(ismu[s]))
        X = t_eff(T, E[s], p[s])
        amp = dN / dteff_dt(T, E[s], p[s])
        c = _teff_c2(E[s], p[s])
        for lo in range(0, len(tau), _TCHUNK):
            ta = tau[lo:lo + _TCHUNK] * alpha[s]
            S[lo:lo + _TCHUNK] += (filon(X, amp, ta) - filon(T, dN, ta)
                                   + _map_below(e0[s], T[0], xi[s], c, ta))
    return S


def _teff_c2(E, p):
    """c in T_eff = T + c T^2 + O(T^3):  c = 3E/(2p^2) - 1/(2E)."""
    return 1.5 * E / (p * p) - 0.5 / E


def _map_below(e0, lo, xi, c, a):
    """The map piece on [e0, lo] in closed form.  There dN = xi/T^2 (the
    spin and beta^2 terms are O(T/tmax)) and a (T_eff - T) = a c T^2 << 1, so
        INT xi/T^2 (e^{i a T_eff} - e^{i a T}) dT = i a xi c INT e^{i a T} dT
                                                 = xi c (e^{i a lo} - e^{i a e0})
    up to O(a c lo^2, lo/E).  It is NOT negligible: its amplitude xi c does
    not fall with t (it is the boundary term of a constant amplitude times
    e^{i a T}), and truncating at 1e-4 tmax instead of e0 moves dS by 6 % of
    its largest value on Z-momentum muons."""
    if not lo > e0:
        return 0.0
    return xi * c * 2j * np.sin(0.5 * a * (lo - e0)) * np.exp(0.5j * a * (lo + e0))


def map_block(rows, wstd, tau):
    """dS_x of one pooled ionisation block: rows with the block's signed
    standardized weight wstd (the weight ioni_step_exponent takes);
    a = wstd * cs * 1e-3 per row."""
    if not len(rows):
        return np.zeros(len(tau), dtype=np.complex128)
    return map_rows(rows, wstd * rows[:, 10].astype(np.float64) * 1e-3, tau)


def map_mean(rows, wstd):
    """d dS_x / d(i t) at t = 0 of map_block, analytic:
    sum_rows a INT dN (T_eff - T) dT on the same range (fine Simpson)."""
    _refuse_epm(rows)
    if not len(rows) or rows.shape[1] < 13:
        return 0.0
    reg, gam = rows[:, 0], rows[:, 9]
    xi = rows[:, 6] * gam * ctr.IONI_A3_SCALE
    e0 = rows[:, 7] * gam
    tmax = rows[:, 8] * gam * ctr.IONI_TMAX_SCALE
    b2, E = rows[:, 11], rows[:, 12]
    p = E * np.sqrt(b2)
    al = wstd * rows[:, 10].astype(np.float64) * 1e-3
    kok_on = ctr.IONI_KOKOULIN != 0.0
    ismu = ctr._is_muon_record(b2, E, reg) & (E - ctr._MMU > ctr._KOK_MUMIN)
    act = ((reg == 2) | (reg == 3)) & (xi > 0) & (tmax > e0) & (e0 > 0)
    out = 0.0
    for s in np.flatnonzero(act):
        T = nodes(e0[s], tmax[s], 400, tlo=max(e0[s], FIT_MAP_TLO * tmax[s]))
        dN = rate(T, xi[s], tmax[s], b2[s], E[s], reg[s] == 2,
                  kok_on and bool(ismu[s]))
        h = (dN * (t_eff(T, E[s], p[s]) - T))[None, :]
        below = xi[s] * _teff_c2(E[s], p[s]) * max(T[0] - e0[s], 0.0)
        out += al[s] * (float(simpson(T, h)[0]) + below)
    return out


def map_rad(tau, recs, spec, vg, weights):
    """dS_x of radiative rows: rad_exponent(exact_qop=True) minus the linear
    one, on the same trapezoid rule, formed cancellation-free as
    e^{i xe} - e^{i x} = 2i sin((xe - x)/2) e^{i (xe + x)/2}."""
    import cf_brems_exact as cbe
    tau = np.asarray(tau, dtype=float)
    S = np.zeros(len(tau), dtype=np.complex128)
    wtrap = np.zeros(len(vg))
    dv = np.diff(np.asarray(vg, dtype=float))
    wtrap[:-1] += 0.5 * dv
    wtrap[1:] += 0.5 * dv
    for rec, sp, w in zip(recs, spec, weights):
        if w == 0.0:
            continue
        v, dNdv = cbe.step_spectrum(rec, sp, vg)
        if not np.any(dNdv > 0.):
            continue
        E, p = rec[cbe.R_ETOT], rec[cbe.R_P]
        T = v * E
        pp = np.sqrt(np.maximum((E - T) ** 2 - (E * E - p * p),
                                (1e-3 * p) ** 2))
        a = tau * rec[cbe.R_CS] * w
        x = np.outer(a, T)
        xe = np.outer(a, qop_map(T, E, p, pp))
        h = np.sin(0.5 * (xe - x))
        c = 0.5 * (xe + x)
        q = wtrap * dNdv
        S += (-2.0 * h * np.sin(c)) @ q + 1j * ((2.0 * h * np.cos(c)) @ q)
    return S


def joint_rows(ms_rows, alpha, beta, tau, tcut_gev, tmaxcap_gev, exact,
               groups=None):
    """dS_j summed over `msmoliv` rows [effZ, effA, xg, pGeV, beta, thp2,
    dOverX0, ...] with per-row signed energy weights alpha [z per MeV] and
    angular weights beta [z per radian] (pair_weights).  S_del's own range,
    rate, spin factor and theta: T in [tcut, min(Tmax, cap)], dN =
    f_spin xi/T^2, theta = sqrt(2 m_e T)/p.  With `groups` (the rows'
    material group) also returns {group: dS_j}."""
    import cf_delta_ray as cdr
    tau = np.asarray(tau, dtype=np.float64)
    nt = len(tau)
    S = np.zeros(nt, dtype=np.complex128)
    grp = {}
    if len(ms_rows):
        Z, A, xg, pg = (ms_rows[:, 0], ms_rows[:, 1], ms_rows[:, 2],
                        ms_rows[:, 3])
        bt = np.clip(ms_rows[:, 4], 1e-9, 1. - 1e-15)
        tkin = cdr._tmx(ms_rows, tcut_gev, 0) * 1e3          # MeV, uncapped
        thi = np.minimum(tkin, tmaxcap_gev * 1e3) if tmaxcap_gev else tkin
        tlo = tcut_gev * 1e3
        alpha = np.broadcast_to(np.asarray(alpha, np.float64), (len(ms_rows),))
        beta = np.abs(np.broadcast_to(np.asarray(beta, np.float64),
                                      (len(ms_rows),)))
        good = ((xg > 0.) & (thi > tlo) & (pg > 0.) & (alpha != 0.)
                & (beta != 0.))
        for s in np.flatnonzero(good):
            xi = cdr.xi_mev(Z[s], A[s], xg[s], bt[s])
            fsp = 1. - 0.5 * bt[s] ** 2 / np.log(max(thi[s] / tlo, 1.0001))
            p = pg[s] * 1e3
            E = p / bt[s]
            T = _joint_nodes(tlo, thi[s], tkin[s], FIT_NPERDEC)
            dN = fsp * xi / T ** 2
            th = np.sqrt(2.0 * ME_MEV * T) / p
            if exact:
                X = t_eff(T, E, p)
                dXdT = dteff_dt(T, E, p)
            else:
                X, dXdT = T, np.ones_like(T)
            val = np.empty(nt, dtype=np.complex128)
            for lo in range(0, nt, _TCHUNK):
                t = tau[lo:lo + _TCHUNK]
                h = dN[None, :] * (j0((t * beta[s])[:, None] * th[None, :]) - 1.0)
                val[lo:lo + _TCHUNK] = (filon(X, h / dXdT[None, :], t * alpha[s])
                                        - simpson(T, h))
            S += val
            if groups is not None:
                g = int(groups[s])
                grp[g] = grp.get(g, 0.0) + val
    return (S, grp) if groups is not None else S


def pair_weights(midx, ridx, ms_rows, rad_rows, wms, wio):
    """Per `msmoliv` row: (beta, alpha) = (w_ms of its MS block,
    w_io of the ionisation block of the same leg at the same step * cs * 1e-3).
    The pairing of ks_nucel_cf.maker_step_weights and cvhcf's elastic family:
    the radiative rows are one per Geant4 step, parallel to the MS rows, and
    carry the leg's ionisation block index; wms / wio are {global index:
    weight} of the blocks the caller formed (wio signed)."""
    if len(rad_rows) != len(ms_rows) or (
            len(ms_rows) and not np.array_equal(rad_rows[:, 4], ms_rows[:, 3])):
        raise ValueError("radiative rows are not parallel to the MS rows")
    beta = np.array([wms.get(int(g), 0.0) for g in midx], dtype=np.float64)
    alpha = (np.array([wio.get(int(g), 0.0) for g in ridx], dtype=np.float64)
             * rad_rows[:, 10].astype(np.float64) * 1e-3)
    return beta, alpha


# ---------------------------------------------------------------- pooling
# One quadrature per pool instead of per row: the pieces are linear in each
# row's xi at fixed kinematics and weights, so a pool of rows with nearly
# equal (a, b, p, tmax) is replaced by one row with the summed xi and the
# xi-weighted means of the rest.  The per-row functions above are the
# reference; the pooled forms are accepted only where they agree with it at
# the float32 floor (knockon_fit/gates.py pool).
def _wmean(x, w):
    return float(np.sum(x * w) / np.sum(w))


def map_block_pooled(rows, wstd, tau, groups=None):
    """map_block with the block's rows pooled per (regime, material group);
    `groups` (the rows' group, the last `ioniurbanv` column) None pools per
    regime only.  With groups also returns {group: dS_x}, whose sum is the
    flat result exactly (the same pools)."""
    _refuse_epm(rows)
    tau = np.asarray(tau, dtype=np.float64)
    S = np.zeros(len(tau), dtype=np.complex128)
    grp = {}
    if not len(rows) or rows.shape[1] < 13:
        S = map_block(rows, wstd, tau)
        return (S, grp) if groups is not None else S
    reg, gam = rows[:, 0], rows[:, 9]
    xi = rows[:, 6] * gam * ctr.IONI_A3_SCALE
    e0 = rows[:, 7] * gam
    tmax = rows[:, 8] * gam * ctr.IONI_TMAX_SCALE
    b2, E = rows[:, 11], rows[:, 12]
    cs = rows[:, 10].astype(np.float64)
    act = ((reg == 2) | (reg == 3)) & (xi > 0) & (tmax > e0) & (e0 > 0)
    gcol = np.zeros(len(rows), np.int64) if groups is None else np.asarray(groups).astype(np.int64)
    for rg, gg in sorted({(int(a), int(b)) for a, b in zip(reg[act], gcol[act])}):
        m = act & (reg == rg) & (gcol == gg)
        w = xi[m]
        pool = np.zeros((1, rows.shape[1]))
        pool[0, 0] = rg
        pool[0, 6] = w.sum() / ctr.IONI_A3_SCALE
        pool[0, 7] = _wmean(e0[m], w)
        pool[0, 8] = _wmean(tmax[m], w) / ctr.IONI_TMAX_SCALE
        pool[0, 9] = 1.0
        pool[0, 10] = _wmean(cs[m], w)
        pool[0, 11] = _wmean(b2[m], w)
        pool[0, 12] = _wmean(E[m], w)
        v = map_rows(pool, wstd * pool[0, 10] * 1e-3, tau)
        S += v
        if groups is not None:
            grp[gg] = grp.get(gg, 0.0) + v
    return (S, grp) if groups is not None else S


def joint_rows_pooled(ms_rows, alpha, beta, keys, tau, tcut_gev, tmaxcap_gev,
                      exact, groups=None):
    """joint_rows with the rows of equal `keys` (the (MS block, ionisation
    block, material group) of each row) pooled into one.  With `groups`
    (the rows' group) also returns {group: dS_j}."""
    import cf_delta_ray as cdr
    tau = np.asarray(tau, dtype=np.float64)
    S = np.zeros(len(tau), dtype=np.complex128)
    if not len(ms_rows):
        return S
    alpha = np.broadcast_to(np.asarray(alpha, np.float64), (len(ms_rows),))
    beta = np.abs(np.broadcast_to(np.asarray(beta, np.float64), (len(ms_rows),)))
    Z, A, xg, pg = ms_rows[:, 0], ms_rows[:, 1], ms_rows[:, 2], ms_rows[:, 3]
    bt = np.clip(ms_rows[:, 4], 1e-9, 1. - 1e-15)
    tkin = cdr._tmx(ms_rows, tcut_gev, 0) * 1e3
    thi = np.minimum(tkin, tmaxcap_gev * 1e3) if tmaxcap_gev else tkin
    tlo = tcut_gev * 1e3
    good = (xg > 0.) & (thi > tlo) & (pg > 0.) & (alpha != 0.) & (beta != 0.)
    if not good.any():
        return S
    xi = np.where(good, cdr.xi_mev(Z, A, xg, bt), 0.0)
    fsp = 1. - 0.5 * bt ** 2 / np.log(np.maximum(thi / tlo, 1.0001))
    w = xi * fsp
    grp = {}
    keys = np.asarray(keys).reshape(len(ms_rows), -1)
    for k in np.unique(keys[good], axis=0):
        m = good & np.all(keys == np.reshape(k, (1, -1)), axis=1)
        ww = w[m]
        p = _wmean(pg[m], ww) * 1e3
        b = _wmean(bt[m], ww)
        tk = _wmean(tkin[m], ww)
        th_ = min(tk, tmaxcap_gev * 1e3) if tmaxcap_gev else tk
        al, be = _wmean(alpha[m], ww), _wmean(beta[m], ww)
        E = p / b
        T = _joint_nodes(tlo, th_, tk, FIT_NPERDEC)
        dN = ww.sum() / T ** 2
        th = np.sqrt(2.0 * ME_MEV * T) / p
        if exact:
            X, dXdT = t_eff(T, E, p), dteff_dt(T, E, p)
        else:
            X, dXdT = T, np.ones_like(T)
        for lo in range(0, len(tau), _TCHUNK):
            t = tau[lo:lo + _TCHUNK]
            h = dN[None, :] * (j0((t * be)[:, None] * th[None, :]) - 1.0)
            v = filon(X, h / dXdT[None, :], t * al) - simpson(T, h)
            S[lo:lo + _TCHUNK] += v
            if groups is not None:
                g = int(np.asarray(groups)[m][0])
                if g not in grp:
                    grp[g] = np.zeros(len(tau), dtype=np.complex128)
                grp[g][lo:lo + _TCHUNK] += v
    return (S, grp) if groups is not None else S


# The consumers' entry points.  FIT_POOL selects the pooled forms (default:
# they agree with the per-row reference to <= 3e-7 on S -- below the float32
# floor of the caches -- on low-pT and Z-momentum muons and on kaons, at 1/40
# of the cost); 0 is the per-row reference.
FIT_POOL = 1
PHYSICS_GLOBALS = PHYSICS_GLOBALS + ("FIT_POOL",)


# The pools are (block, material group): the material group is the last
# column of `ioniurbanv` (stride 14) and column 9 of `msmoliv` (stride 10),
# as the in-maker split reads them, so a flat family is exactly the sum of its
# per-group split.  Rows without the column pool per block.
IONI_GROUP_STRIDE = 14
MS_GROUP_COL = 9


def ioni_groups(rows):
    return rows[:, -1] if rows.shape[1] >= IONI_GROUP_STRIDE else None


def ms_groups(rows):
    return rows[:, MS_GROUP_COL] if rows.shape[1] > MS_GROUP_COL else None


def fit_map(rows, wstd, tau):
    """dS_x of one ionisation block (map_block / map_block_pooled)."""
    if not FIT_POOL:
        return map_block(rows, wstd, tau)
    g = ioni_groups(rows) if len(rows) else None
    return map_block_pooled(rows, wstd, tau, g)[0] if g is not None \
        else map_block_pooled(rows, wstd, tau)


def fit_joint(ms_rows, alpha, beta, midx, ridx, tau, tcut_gev, tmaxcap_gev,
              exact):
    """dS_j of MS rows, pooled by (MS block, ionisation block, group)."""
    if not FIT_POOL:
        return joint_rows(ms_rows, alpha, beta, tau, tcut_gev, tmaxcap_gev,
                          exact)
    cols = [np.asarray(midx), np.asarray(ridx)]
    g = ms_groups(ms_rows) if len(ms_rows) else None
    if g is not None:
        cols.append(g)
    keys = np.stack(cols, axis=1)
    return joint_rows_pooled(ms_rows, alpha, beta, keys, tau, tcut_gev,
                             tmaxcap_gev, exact)
