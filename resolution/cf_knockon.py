#!/usr/bin/env python3
"""Hard knock-on collisions exactly: the joint law of energy loss and
deflection, and the exact energy -> q/p map.

THE TWO PIECES
--------------
The ionisation channel (cf_track_resolution.ioni_step_exponent, regime 2/3)
carries the knock-on spectrum

    dN/dT = (xi/T^2) [1 - b2 T/tmax (+ T^2/(2E^2), spin 1/2)] f_K(T)   on [e0, tmax]

as a centred compound Poisson in the ENERGY, mapped LINEARLY into q/p
(d qop = q cs dT, cs = E/p^3).  The scattering channel carries the same
collisions' angular kicks -- its electron term runs to the kinematic ceiling
theta(Tmax) -- as an INDEPENDENT compound Poisson.  Per collision the truth is
one event with both:

    d qop = q (1/p' - 1/p) = q cs T_eff(T),
            T_eff = p^2 (p - p')/(E p') = p^2 T (2E - T)/(E p' (p + p')),
            p' = sqrt((E - T)^2 - M^2)
    theta = sqrt(2 m_e T (1 - T/tmax)) / p'
            (the electron's transverse momentum, exact for a free electron,
            over the primary's momentum after the collision)

with the kick's azimuth uniform.  So per collision the model's

    [e^{i a T} - 1 - i a T]  +  [J0(b theta) - 1]

becomes the exact  [e^{i a T_eff} J0(b theta) - 1 - i a T].  The centring
stays linear: the reference subtracts the mean ENERGY loss, so the channel's
mean becomes the Jensen excess of q/p over q cs E[dE].  The difference, added
to the exponent, is

    dS = INT dN(T) [ e^{i a X} J - e^{i a T} - J + 1 ] dT ,

X = T_eff under QOP_EXACT (else T), J = J0(b theta) under KNOCKON_JOINT
(else 1).  Both DEFAULT ON; with both off dS = 0, nothing is evaluated and
every number is bit-identical to the linear model.

VALIDATION (realmat_full, 200 k per sample; CLOSURE_STATE.md)
  J/psi one-plane vertex mass (realmat_ditrack.py), 19 planes x 9 probes:
    even > 3 sigma 98 -> 3 of 171, odd 14 -> 1, mean 3.3 -> 0.9 sigma;
    each switch alone leaves one failure (joint: shape and core; exact: mean).
  single track, 8 species x 5 directions: > 3 sigma 471 -> 349 of 6840, all of
    it in the bending plane of muons and pions; kaons and protons (Tmax 40,
    10 MeV) move by <= 2.2e-4.

    KNOCKON_JOINT  (e^{iaT} - 1)(J0 - 1): the shape of any direction that mixes
                   q/p with the angles.  Mean and variance unchanged (the cross
                   moment vanishes by azimuthal symmetry).
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
PHYSICS_GLOBALS = ("KNOCKON_JOINT", "QOP_EXACT", "KNOCKON_NSUB",
                   "KNOCKON_NPERDEC", "KNOCKON_TCUT", "KNOCKON_NPERDEC_THIN",
                   "KNOCKON_THIN")
_NOT_PHYSICS = ("ME_MEV", "_TCHUNK", "_NSER", "_ZSER")
_TCHUNK = 2048          # t points per block (memory only)


def physics_state():
    g = globals()
    return tuple((n, repr(g[n])) for n in PHYSICS_GLOBALS)


def active():
    return bool(KNOCKON_JOINT) or bool(QOP_EXACT)


# ------------------------------------------------------------- kinematics
def t_eff(T, E, p):
    """The q/p change of an energy loss T in units of the linear one (so
    T_eff -> T as T -> 0).  Cancellation-free form."""
    M2 = E * E - p * p
    pp = np.sqrt(np.maximum((E - T) ** 2 - M2, 1e-300))
    return p * p * T * (2.0 * E - T) / (E * pp * (p + pp))


def dteff_dt(T, E, p):
    M2 = E * E - p * p
    pp = np.sqrt(np.maximum((E - T) ** 2 - M2, 1e-300))
    return (p ** 3 / E) * (E - T) / pp ** 3


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


def step_correction(tau_a, tau_b, T, dN, X, dXdT, th, joint, exact):
    """dS on the t points of one step: tau_a = t*a (nt,), tau_b = t*b (nt,)."""
    B = filon(T, dN, tau_a)
    if joint:
        J = j0(tau_b[:, None] * th[None, :])
        C = simpson(T, dN[None, :] * (J - 1.0))
    else:
        J = 1.0
        C = 0.0
    if exact:
        A = filon(X, dN[None, :] * J / dXdT[None, :], tau_a)
    else:
        A = filon(T, dN[None, :] * J, tau_a)
    return A - B - C


# ------------------------------------------------------------- the channel
def knockon_exponent(leg, A_end, A_start, avec, sigma, tau):
    """dS of one leg at plane k: summed over its ionisation records."""
    S = np.zeros(len(tau), dtype=np.complex128)
    if not active() or not len(leg["ioni"]):
        return S
    st = leg["ioni"]
    reg = st[:, 0]
    if not ((reg == 2) | (reg == 3)).any():
        # regime-1 records carry no exact knock-on spectrum (and no beta^2, E)
        return S
    if st.shape[1] < 13:
        raise ValueError("knock-on channel needs the stride-13 regime-2/3 record")
    gam = st[:, 9]
    xi = st[:, 6] * gam * ctr.IONI_A3_SCALE
    e0 = st[:, 7] * gam
    tmax = st[:, 8] * gam * ctr.IONI_TMAX_SCALE
    b2, E = st[:, 11], st[:, 12]
    g = st[:, 10] * 1e-3
    p = E * np.sqrt(b2)
    q = np.sign(leg["refqop"]) or 1.0
    coslam = max(leg["refpt"] / leg["refp"] if leg["refp"] > 0 else 1.0, 1e-3)
    v1 = np.einsum("i,sij->sj", avec, A_end)
    v0 = np.einsum("i,sij->sj", avec, A_start)
    kok_on = ctr.IONI_KOKOULIN != 0.0
    ismu = ctr._is_muon_record(b2, E) & (E - ctr._MMU > ctr._KOK_MUMIN)
    act = ((reg == 2) | (reg == 3)) & (xi > 0) & (tmax > e0) & (e0 > 0)
    joint, exact = bool(KNOCKON_JOINT), bool(QOP_EXACT)
    nsub = max(int(KNOCKON_NSUB), 1)
    xmax = float(xi[act].max()) if act.any() else 0.0
    for s in np.flatnonzero(act):
        thin = xi[s] < KNOCKON_THIN * xmax
        T = nodes(e0[s], tmax[s],
                  KNOCKON_NPERDEC_THIN if thin else KNOCKON_NPERDEC)
        dN = rate(T, xi[s], tmax[s], b2[s], E[s], reg[s] == 2,
                  kok_on and bool(ismu[s]))
        X = t_eff(T, E[s], p[s]) if exact else T
        dXdT = dteff_dt(T, E[s], p[s]) if exact else np.ones_like(T)
        th = theta_kick(T, E[s], p[s], tmax[s]) if joint else None
        for i in range(nsub):
            f = (i + 0.5) / nsub
            v = v0[s] + f * (v1[s] - v0[s])
            al = q * v[0] * g[s] / sigma
            be = np.hypot(v[1], v[2] / coslam) / sigma
            if al == 0.0 and (be == 0.0 or not joint):
                continue
            for lo in range(0, len(tau), _TCHUNK):
                t = tau[lo:lo + _TCHUNK]
                S[lo:lo + _TCHUNK] += step_correction(
                    t * al, t * be, T, dN, X, dXdT, th, joint, exact) / nsub
    return S


def mean_shift(leg, A_end, A_start, avec, sigma):
    """d/d(it) of dS at t = 0: the channel's mean in z units, analytic
    (INT dN (T_eff - T) a, sub-step averaged).  Zero unless QOP_EXACT."""
    if not QOP_EXACT or not len(leg["ioni"]):
        return 0.0
    st = leg["ioni"]
    reg, gam = st[:, 0], st[:, 9]
    if not ((reg == 2) | (reg == 3)).any() or st.shape[1] < 13:
        return 0.0
    xi = st[:, 6] * gam * ctr.IONI_A3_SCALE
    e0, tmax = st[:, 7] * gam, st[:, 8] * gam * ctr.IONI_TMAX_SCALE
    b2, E, g = st[:, 11], st[:, 12], st[:, 10] * 1e-3
    p = E * np.sqrt(b2)
    q = np.sign(leg["refqop"]) or 1.0
    v1 = np.einsum("i,sij->sj", avec, A_end)
    v0 = np.einsum("i,sij->sj", avec, A_start)
    kok_on = ctr.IONI_KOKOULIN != 0.0
    ismu = ctr._is_muon_record(b2, E) & (E - ctr._MMU > ctr._KOK_MUMIN)
    act = ((reg == 2) | (reg == 3)) & (xi > 0) & (tmax > e0) & (e0 > 0)
    out = 0.0
    for s in np.flatnonzero(act):
        T = nodes(e0[s], tmax[s])
        dN = rate(T, xi[s], tmax[s], b2[s], E[s], reg[s] == 2,
                  kok_on and bool(ismu[s]))
        m = np.trapezoid(dN * (t_eff(T, E[s], p[s]) - T), T)
        vbar = 0.5 * (v0[s] + v1[s])
        out += q * vbar[0] * g[s] / sigma * m
    return float(out)
