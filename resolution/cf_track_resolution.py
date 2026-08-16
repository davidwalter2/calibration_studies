"""Per-track resolution prediction from the CF product ("Level 2").

The linearized CVH fit maps the noise vector n to the momentum error via
delta(q/p) = w^T n with w = (C e_qop)^T F^T V^{-1}. The refit exports, per
resolution entry b (aligned with reseigidx), the variance contribution
v_b = w_b^T V_b w_b (resinfvarv) and their total resinfcov, which equals
refCov(0,0) EXACTLY (validated: ratio 1.0000 on every smoke track) -- every
noise dof feeding q/p carries an entry.

Independence of the blocks then gives the full non-Gaussian CF of the
standardized momentum error z = (qop_reco - qop_gen)/sigma, sigma^2 =
refCov(0,0):

  log phi_z(t) = -(Vg/sigma^2) t^2/2                        [hits, Gaussian]
    + sum_MS  e^{k_ms}  sum_steps (chic2/chia2) G(t w_std sqrt(chia2); FF)
    + sum_ion e^{k_ion} sum_steps S_urban(t w_std g_s ...)  [centered]

where per pooled block w_std = sqrt(v_pool / sigma_Q^2) / sigma is the
effective scalar weight (exact under per-collision azimuthal isotropy:
any linear functional of the two projected angles is again projected-
Moliere with |weight|), and sigma_Q^2 is the fit-assumed block variance
(sum thp2 for MS, sum gsig2 g^2 for ionization). Blocks are pooled by
global parameter index (steps are matched the same way as in cf_ms_exact /
cf_ioni_exact); pooling merges same-family crossings with a shared w --
exact for the common one-crossing case.

Closure ("--closure"): on gen-matched MC with the NOMINAL fit
(fitFromGenParms=False), compare per track
  data  e^{-u z_obs^2}          vs   model  E[e^{-u z^2}] (Weierstrass)
averaged in bins of predicted sigma and probe u, plus the averaged
predicted lineshape p(z) against the pull histogram.

usage:
  python cf_track_resolution.py --extract [--files GLOB] [--ntasks N]
  python cf_track_resolution.py --closure [--probes ...]
"""

import argparse
import datetime
import glob
import os
import sys

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot

from wums import logging, output_tools, plot_tools
from cf_ms_exact import moliere_params, gshape
import cf_ioni_exact

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

TG = np.linspace(0.0, 14.0, 448)  # t conjugate to standardized z
COVTOL = 5e-3                     # |resinfcov/refCov00 - 1| guard


# ------------------------------------------------------------------ SWITCHES
# The tri-state environment convention, IDENTICAL to `cvhcgf::envFlag` in
# TrackPropagation/Geant4e/src/CGFQoPBlock.cc:
#
#     unset                               -> `dflt`
#     "0"/"false"/"off"/"no"/"" (any case) -> False
#     anything else                        -> True
#
# It exists because on 2026-08-16 four corrections went DEFAULT-ON, and a
# presence-only reader (`os.environ.get(name)`) cannot express "off" once the
# default is on.  Attribution needs both directions: the whole point of the
# gated global-fit exercise is to run all four off against all four on.
#
# This lives in the DEEPEST module of the offline stack (nothing local is
# imported here), so every driver can reach it without a cycle.
def env_flag(name, dflt):
    v = os.environ.get(name)
    if v is None:
        return bool(dflt)
    return v.strip().lower() not in ("", "0", "false", "off", "no")


# The four corrections that went default-ON on 2026-08-16.  See
# Documents/Resolution/NOTES_DEFAULTON.md.
CVH_DEFAULT_ON = ("CVH_IONI_EXACTDELTA", "CVH_IONI_KOKOULIN",
                  "CVH_REF_CHARGEAWARE", "CVH_REF_SPECIESDEDX")

# The environment overlay that restores the PRE-2026-08-16 state, i.e. the
# state every published `off`/`nominal`/`caoff` control arm in this directory
# was measured in.  Those arms used to be `{}`; an empty dict now means ALL
# FOUR ON, so every one of them has to carry this explicitly or the control
# silently becomes a second copy of the signal arm.
SWITCHES_OFF = {n: "0" for n in CVH_DEFAULT_ON}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--files",
                   default="/ceph/submit/data/user/d/david_w/ZMass/cvh/"
                           "resolution_trackres_260802/task_*/globalcor_resclosure_0.root")
    p.add_argument("--ntasks", type=int, default=100)
    p.add_argument("--max-tracks", type=int, default=100000)
    p.add_argument("--cache", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "runs/cf_trackres_cache.npz"))
    p.add_argument("--extract", action="store_true")
    p.add_argument("--closure", action="store_true")
    p.add_argument("--probes", type=float, nargs="+",
                   default=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
    p.add_argument("--khit", type=float, default=0.0,
                   help="log-scale applied to the hit (Gaussian) variance share")
    p.add_argument("--kms", type=float, default=0.0,
                   help="log-scale applied to the MS exponent (collision counts)")
    p.add_argument("--kioni", type=float, default=0.0,
                   help="log-scale applied to the ionization exponent")
    p.add_argument("--nsigma-bins", type=int, default=4)
    p.add_argument("--outpath", default=None)
    p.add_argument("--postfix", default="")
    return p.parse_args()


_EULER = 0.5772156649015328606
# |a w| below which the Taylor series is used. The series converges for every
# argument; this bound keeps the alternating cancellation below one digit.
_DT_SER = 2.0
_DT_NSER = 80


def _ein_neg(s, out=None):
    """Ein(-s) = -sum_{k>=1} s^k/(k k!), for complex s. Entire, and finite at
    s = 0 (unlike E1, which is what makes it the right object here).

    Series for |s| <= _DT_SER; Ein(-s) = gamma + Log(-s) + E1(-s) beyond, where
    E1 comes from scipy (accurate to 1e-16 on the imaginary axis -- measured,
    2026-08-13 XX, so it is NOT the weak link).
    """
    s = np.asarray(s, dtype=np.complex128)
    res = np.empty(s.shape, dtype=np.complex128) if out is None else out
    sm = np.abs(s) <= _DT_SER
    if sm.any():
        ss = s[sm]
        term = np.ones_like(ss)
        acc = np.zeros_like(ss)
        for k in range(1, _DT_NSER):
            term = term * ss / k
            acc = acc - term / k
            if k > 3 and np.max(np.abs(term)) < 1e-19 * max(
                    float(np.max(np.abs(acc))), 1e-300):
                break
        res[sm] = acc
    bg = ~sm
    if bg.any():
        from scipy.special import exp1
        ss = s[bg]
        res[bg] = _EULER + np.log(-ss) + exp1(-ss)
    return res


def _delta_term_2d(a, w):
    """Vectorized <(e^{iaE} - 1 - iaE)/E^2> for the 1/E^2 spectrum on
    [1, w] (units of e0): a (ns, nt) real, w (ns,) scalar per step.
    Same math as cf_ioni_exact.delta_term, and the imaginary-argument twin of
    cgf_saddlepoint._delta_derivs (b -> ia).

    With x = i a w and y = i a, and using E1(-y) - E1(-x) - ln w
    = Ein(-y) - Ein(-x) exactly (the ln w and the two branch offsets cancel
    analytically, since x/y = w > 0):

        <...> * N = expm1(y) - expm1(x)/w + y (Ein(-y) - Ein(-x)),   N = 1 - 1/w
                  = sum_{k>=2} (x^k/w - y^k) / (k! (k-1))            [series]

    THE EXPANSION PARAMETER IS a*w, NOT a. The series of <e^{iaE}-1-iaE>
    requires a*E << 1 over the WHOLE support, and the support reaches E = w.
    For muons w = tmax/e0 ~ 1e8-1e10 (keV-scale e0 against a multi-GeV
    kinematic tmax), so a guard on |a| alone selects the series in a regime
    where it is wrong by orders of magnitude: at w = 1e9, a = 1e-6 the old
    |a|-guard gave Im = -8.3e-2 against the true -6.5e-6, and that spurious
    phase, stacked over ~350 steps, turned the model CF into a pure
    oscillation at small t. That guard is KEPT -- the switch below is on
    |a| * w.

    WHAT CHANGED 2026-08-13 (XX). Both branches were rewritten; neither was
    giving wrong VALUES (the argument is imaginary, |e^{iaw}| = 1, so nothing
    ever overflowed) but both lost precision, worst AT the old crossover:
      * the series kept only k = 2, 3, so it was truncated at (a w)^2/36
        -- 2.8e-6 at a w = 1e-2, 2.5e-5 at 3e-2. It now runs to convergence.
      * the closed form grouped e^{ia} - (1 - 1/w). fl(1 - 1/w) carries a
        rounding error of eps/2 ~ 5.5e-17 while the result is only O(a^2 w)
        = O(x^2/w), so the RELATIVE error grew linearly in w: 3e-13 at
        w = 20, 2.4e-10 at w = 1e4, 7.0e-5 at w = 1e9, 7.3e-4 at w = 1e10.
        expm1(y) - expm1(x)/w cancels the 1/w analytically instead.
    Measured worst error against mpmath at 60 dps is now 6e-15 over
    w = 20..1e10 and a w = 1e-6..1e2.
    """
    a = np.asarray(a, dtype=np.complex128)
    wb = np.broadcast_to(np.asarray(w, dtype=np.float64)[:, None], a.shape)
    y = 1j * a
    x = y * wb
    N = 1.0 - 1.0 / wb
    out = np.empty(a.shape, dtype=np.complex128)

    ser = np.abs(x) <= _DT_SER
    if ser.any():
        xs, ys, ws = x[ser], y[ser], wb[ser]
        tx = np.ones_like(xs)
        ty = np.ones_like(ys)
        acc = np.zeros_like(xs)
        for k in range(1, _DT_NSER):
            tx = tx * xs / k
            ty = ty * ys / k
            if k >= 2:
                acc = acc + (tx / ws - ty) / (k - 1.0)
                if np.max(np.abs(tx)) < 1e-19 * max(
                        float(np.max(np.abs(acc))), 1e-300):
                    break
        out[ser] = acc

    bg = ~ser
    if bg.any():
        xb, yb, ww = x[bg], y[bg], wb[bg]
        with np.errstate(all="ignore"):
            out[bg] = (np.expm1(yb) - np.expm1(xb) / ww
                       + yb * (_ein_neg(yb) - _ein_neg(xb)))
    return out / N


# ------------------------------------------------------- EXACT DELTA SPECTRUM
# The three integrals the exact spin-1/2 knock-on spectrum needs, in units of
# the channel's lower limit e0 (u = T/e0, alpha = a e0, W = tmax/e0):
#
#   J0 = int_1^W (e^{i alpha u} - 1 - i alpha u) / u^2 du     <- the 1/T^2 part
#   J1 = int_1^W (e^{i alpha u} - 1 - i alpha u) / u   du     <- the -b2 T/Tmax
#   J2 = int_1^W (e^{i alpha u} - 1 - i alpha u)       du     <- the spin term
#
# so that, for a step whose delta channel is
#     dN/dT = (xi/T^2) [1 - b2 T/tmax + T^2/(2 E^2)]  on [e0, tmax],
# the centred compound-Poisson log-CF at conjugate variable a is
#     S = (xi/e0) [ J0 - (b2 e0/tmax) J1 + (e0^2/(2 E^2)) J2 ].
#
# J0 is exactly the (unnormalized) integrand of `_delta_term_2d`, so the pure
# 1/E^2 branch is recovered by dropping the other two -- which is the check the
# regime-1 / regime-2 comparison uses.
#
# BRANCHING is on |alpha W| = |a * tmax|, NOT |alpha|, for the reason
# `_delta_term_2d` documents at length: the support reaches T = tmax and a
# guard on |a| alone selects the series in a regime where it is wrong by orders
# of magnitude.  Series:
#   J0 = sum_{k>=2} (x^k/W - y^k) / (k! (k-1))
#   J1 = sum_{k>=2} (x^k    - y^k) / (k!  k)
#   J2 = sum_{k>=2} (x^k W  - y^k) / (k! (k+1)),     x = i alpha W, y = i alpha
# Closed forms (E1 via the entire Ein, so nothing is evaluated near a branch
# cut and the ln W cancels analytically):
#   J0 = expm1(y) - expm1(x)/W + y (Ein(-y) - Ein(-x))
#   J1 = (Ein(-y) - Ein(-x)) - y (W - 1)
#   J2 = (e^x - e^y)/y - (W - 1) - y (W^2 - 1)/2

def _delta_terms_exact(a, w):
    """(J0, J1, J2) as above.  a: (ns, nt) real; w: (ns,) = tmax/e0."""
    a = np.asarray(a, dtype=np.complex128)
    wb = np.broadcast_to(np.asarray(w, dtype=np.float64)[:, None], a.shape)
    y = 1j * a
    x = y * wb
    J0 = np.empty(a.shape, dtype=np.complex128)
    J1 = np.empty(a.shape, dtype=np.complex128)
    J2 = np.empty(a.shape, dtype=np.complex128)

    ser = np.abs(x) <= _DT_SER
    if ser.any():
        xs, ys, ws = x[ser], y[ser], wb[ser]
        tx = np.ones_like(xs)
        ty = np.ones_like(ys)
        a0 = np.zeros_like(xs)
        a1_ = np.zeros_like(xs)
        a2_ = np.zeros_like(xs)
        for k in range(1, _DT_NSER):
            tx = tx * xs / k
            ty = ty * ys / k
            if k >= 2:
                a0 = a0 + (tx / ws - ty) / (k - 1.0)
                a1_ = a1_ + (tx - ty) / k
                a2_ = a2_ + (tx * ws - ty) / (k + 1.0)
                if np.max(np.abs(tx)) * float(np.max(ws)) < 1e-19 * max(
                        float(np.max(np.abs(a2_))), 1e-300):
                    break
        J0[ser], J1[ser], J2[ser] = a0, a1_, a2_

    bg = ~ser
    if bg.any():
        xb, yb, ww = x[bg], y[bg], wb[bg]
        with np.errstate(all="ignore"):
            ein = _ein_neg(yb) - _ein_neg(xb)
            J0[bg] = np.expm1(yb) - np.expm1(xb) / ww + yb * ein
            J1[bg] = ein - yb * (ww - 1.0)
            J2[bg] = (np.exp(xb) - np.exp(yb)) / yb - (ww - 1.0) \
                - yb * (ww * ww - 1.0) / 2.0
    return J0, J1, J2


def exact_delta_exponent(xi, e0, tmax, beta2, etot, a, spinhalf=True):
    """Centred log-CF of the exact delta channel, summed over steps.

    xi, e0, tmax, etot: (ns,) in MeV; beta2: (ns,); a: (ns, nt) conjugate
    variable in 1/MeV.  Returns (nt,) complex.

    NOTE `_delta_terms_exact` works in units of e0 -- its first argument is
    alpha = a * e0, exactly as `_delta_term_2d`'s is -- so the conversion is
    done HERE and not by the caller."""
    w = tmax / e0
    J0, J1, J2 = _delta_terms_exact(a * e0[:, None], w)
    pref = (xi / e0)[:, None]
    c1 = (beta2 * e0 / tmax)[:, None]
    S = pref * (J0 - c1 * J1)
    if spinhalf:
        # the spin term is present only for spin 1/2 (mu, p), exactly as in
        # G4BetheBlochModel's `0.5 == spin` branch -- pi and K do not have it
        S = S + pref * (e0 ** 2 / (2.0 * etot ** 2))[:, None] * J2
    return np.sum(S, axis=0)


# `gamma` from `tmax`, ASSUMING A MUON. Kept only as a cross-check of the
# exported beta2/etot columns for muon models -- it is NOT used to build the
# CF, because Tmax fixes gamma only once the particle mass is known, and
# inverting a 3 GeV KAON's Tmax = 41.1 MeV with the muon mass returns E = 698
# MeV instead of 3175 MeV and beta^2 = 0.9989 instead of 0.9761. beta^2
# multiplies the entire suppression term, so that is not a small error.
_MMU = 105.6583745
_MEL = 0.51099895


def gamma_from_tmax(tmax):
    r = _MEL / _MMU
    b = -2.0 * np.asarray(tmax, dtype=np.float64) * r
    c = -(2.0 * _MEL + np.asarray(tmax, dtype=np.float64) * (1.0 + r * r))
    return (-b + np.sqrt(b * b - 8.0 * _MEL * c)) / (4.0 * _MEL)


# ---------------------------------------------------------------- A3 GAUGE
# DIAGNOSTIC KNOB, default-inert (1.0), in exactly the style of the pre-existing
# KMS_SCALE / MS_NSUB module globals.
#
# The exported Urban record describes the block's ionization as two Poisson
# excitation channels (a1,e1),(a2,e2) plus `a3` collisions drawn from 1/E^2 on
# [e0, tmax = Tmax].  That third channel is the model's ENTIRE representation of
# hard delta rays.  Geant4's own explicit delta production can be counted
# directly in the simulation (tail_probe.py hardrate), and it is measurably not
# the same number.  `IONI_A3_SCALE` multiplies a3 so the size of that
# disagreement can be propagated through the closure's own transform and
# weights, and `IONI_EXC_SCALE` rescales the excitation channels so the block's
# MEAN loss can be held fixed while its hard-tail weight moves -- otherwise the
# gauge is a mean change as well as a shape change and the two cannot be told
# apart.  (The CF is centred, so the mean does not enter z directly; keeping it
# fixed is what makes the modified block a physically admissible alternative
# rather than a different amount of material.)
IONI_A3_SCALE = 1.0
IONI_EXC_SCALE = 1.0

# SYSTEMATIC KNOB, default-inert. Multiplies the regime-2/3 delta channel's
# Tmax -- the spectrum's upper limit AND the beta^2 T/Tmax suppression -- while
# holding beta^2 and E at their exported values. It answers exactly one
# question: how much would the closure move if the record's Tmax disagreed with
# the one Geant4 sampled from? (Measured disagreement: +2.3e-7 relative, from
# the propagator's hard-coded electron mass. See NOTES_DELTASPEC section 8.)
# It is NOT a tune and nothing is fitted to it.
IONI_TMAX_SCALE = 1.0


def a3_gauge_exc_scale(steps, f_a3):
    """The excitation scale that holds a pooled block's mean loss fixed when a3
    is multiplied by `f_a3`.  Returns 1.0 if there is no excitation to give."""
    reg, gam = steps[:, 0], steps[:, 9]
    a1, e1 = steps[:, 2], steps[:, 3] * gam
    a2, e2 = steps[:, 4], steps[:, 5] * gam
    a3, e0, tmx = steps[:, 6], steps[:, 7] * gam, steps[:, 8] * gam
    m = (reg != 0) & (a3 > 0) & (tmx > e0) & (e0 > 0)
    m1 = m & (reg != 2) & (reg != 3)
    m2 = m & ((reg == 2) | (reg == 3))
    C = np.zeros_like(a3)
    C[m1] = 1.0 / (1.0 / e0[m1] - 1.0 / tmx[m1])
    Ed = float(np.sum(a3[m1] * C[m1] * np.log(tmx[m1] / e0[m1])))
    if m2.any():
        # regime 2/3: column 6 is xi (already gam-scaled above); the channel's
        # mean is xi [ln(tmax/e0) - b2 (1-e0/tmax) (+ (tmax^2-e0^2)/(4E^2))]
        b2, et = steps[:, 11], steps[:, 12]
        t = (np.log(tmx[m2] / e0[m2]) - b2[m2] * (tmx[m2] - e0[m2]) / tmx[m2])
        half = reg[m2] == 2
        t = t + np.where(half, (tmx[m2] ** 2 - e0[m2] ** 2)
                         / (4.0 * np.maximum(et[m2], 1e-30) ** 2), 0.0)
        Ed += float(np.sum(a3[m2] * t))
    Ee = float(np.sum(np.where(reg != 0, a1 * e1 + a2 * e2, 0.0)))
    if Ee <= 0.0:
        return 1.0
    return max((Ee + Ed * (1.0 - f_a3)) / Ee, 0.0)


def ioni_step_exponent(steps, wstd, tau):
    """Centered Urban log-CF exponent of one pooled ionization block,
    in standardized-z units: per step the raw qop-noise CF evaluated at
    wstd*tau (wstd = w/sigma includes the fit-unit conversion). steps:
    (n, 11) ioniurbanv records. Vectorized over steps. Returns complex
    array on tau."""
    if IONI_A3_SCALE != 1.0 or IONI_EXC_SCALE != 1.0:
        steps = steps.copy()
        steps[:, 6] *= IONI_A3_SCALE
        steps[:, 2] *= IONI_EXC_SCALE
        steps[:, 4] *= IONI_EXC_SCALE
    reg = steps[:, 0]
    gsig2 = steps[:, 1].astype(np.float64)
    g = steps[:, 10].astype(np.float64) * 1e-3  # qop per MeV
    gs = wstd * g                               # (ns,)
    S = np.zeros(len(tau), dtype=np.complex128)

    mg = reg == 0
    if mg.any():
        S += -0.5 * tau ** 2 * float(np.sum(gsig2[mg] * gs[mg] ** 2))

    mu = ~mg
    if mu.any():
        gam = steps[mu, 9]
        gsu = gs[mu]
        for ja, je in ((2, 3), (4, 5)):
            aj = steps[mu, ja]
            ej = steps[mu, je] * gam
            act = (aj > 0.) & (ej > 0.)
            if act.any():
                th = (gsu[act] * ej[act])[:, None] * tau[None, :]
                S += np.sum(aj[act][:, None] * (np.exp(1j * th) - 1. - 1j * th), axis=0)
        a3 = steps[mu, 6]
        e0 = steps[mu, 7] * gam
        tmax = steps[mu, 8] * gam
        # regime 2 (CVH_IONI_EXACTDELTA): the delta channel is the exact
        # spin-1/2 knock-on spectrum and column 6 carries its normalization xi
        # (an energy, so it takes the `scaling` factor) rather than a count.
        # Split on the regime rather than on a heuristic, so a regime-2 record
        # can never be read as a regime-1 one.
        regmu = reg[mu]
        ex = (regmu == 2) | (regmu == 3)
        act = (a3 > 0.) & (tmax > e0) & (e0 > 0.)
        if (act & ~ex).any():
            m = act & ~ex
            aarg = (gsu[m] * e0[m])[:, None] * tau[None, :]
            S += np.sum(a3[m][:, None] * _delta_term_2d(aarg, tmax[m] / e0[m]),
                        axis=0)
        if (act & ex).any():
            if steps.shape[1] < 13:
                raise ValueError(
                    "regime 2/3 record with stride 11: beta^2 and E are not in "
                    "the file. Re-export with CVH_IONI_EXACTDELTA (which writes "
                    "stride 13); they cannot be recovered from tmax without the "
                    "particle mass.")
            b2 = steps[mu, 11]
            et = steps[mu, 12]
            if IONI_TMAX_SCALE != 1.0:
                tmax = tmax * IONI_TMAX_SCALE
            for spin, half in ((2, True), (3, False)):
                m = act & (regmu == spin)
                if not m.any():
                    continue
                S += exact_delta_exponent(a3[m] * gam[m], e0[m], tmax[m],
                                          b2[m], et[m],
                                          gsu[m][:, None] * tau[None, :],
                                          spinhalf=half)
                if IONI_KOKOULIN != 0.0:
                    S += IONI_KOKOULIN * _kokoulin_exponent(
                        a3[m] * gam[m],
                        np.maximum(e0[m], max(IONI_KOKOULIN_TCUT, _KOK_TMIN)),
                        tmax[m], b2[m], et[m],
                        gsu[m][:, None] * tau[None, :],
                        nbin=IONI_KOKOULIN_NBIN, spinhalf=half)
    return S


# ---------------------------------------------------- KOKOULIN, default-inert
# Geant4's G4MuBetheBlochModel multiplies the knock-on cross section by
# R. Kokoulin's radiative correction  1 + (alpha/2pi) a1 (a3 - a1),
# a1 = ln(1 + 2T/m_e), a3 = ln(4 E (E-T)/M^2), above T = 100 keV for muons
# above 1 GeV.  It reaches +6 % at the hard end.  It is NOT part of the
# tree-level spectrum this module implements, and NOTES_DELTASPEC section 1.5
# recorded that omission as a known residual (it has no closed-form CF).
#
# `IONI_KOKOULIN` is a default-inert gauge in the style of IONI_A3_SCALE /
# IONI_TMAX_SCALE / MS_NSUB: 0.0 reproduces the published model bit-for-bit
# (the branch is not entered at all), 1.0 puts Geant4's own correction into
# the model.  `IONI_KOKOULIN_TCUT` is the e- production threshold of the
# simulation being compared against: the SIMULATION applies the correction
# only to the EXPLICIT secondaries, i.e. only above its production cut -- the
# Urban straggling below the cut has no radiative correction -- so the
# correction must start at max(tcut, 100 keV) and not at e0.
#
# THE SWITCH IS SHARED WITH THE C++ SIDE (2026-08-15, NOTES_QVALID).
# `CVH_IONI_KOKOULIN` is read by `cvhcgf::ioniKokoulinEnabled()` in
# TrackPropagation/Geant4e, which is the SINGLE C++ reader and is what puts the
# correction into the variance the track fit consumes (Q(0,0), and `gsig2` in
# the exported record).  Deriving this module's default from the SAME variable
# is the offline half of that coupling, in the pattern of
# `cvhcgf::referenceIsIonOnly()`: a job that exports with the correction on and
# then analyses in the same environment cannot end up half-corrected because
# somebody forgot a convention.
#
# Setting the module global directly still works and still wins (that is how
# `samplergap.py real --kok 0 1` runs both arms in one process); the
# environment only supplies the DEFAULT.
#
# DEFAULT ON since 2026-08-16, mirroring `cvhcgf::ioniKokoulinEnabled()`
# (Documents/Resolution/NOTES_DEFAULTON.md).  `env_flag` is the SAME tri-state
# convention the C++ reader uses -- unset means the default, `=0` means off --
# so the two halves of the shared switch cannot disagree in either direction.
IONI_KOKOULIN = 1.0 if env_flag("CVH_IONI_KOKOULIN", True) else 0.0
IONI_KOKOULIN_TCUT = 0.0
IONI_KOKOULIN_NBIN = 96

# ------------------------------------------------------- the knob REGISTRY
# Every module-level global that changes what this module COMPUTES, in one
# place, so that a cache key can be built from the registry instead of from
# somebody's memory of the list.  That is not bookkeeping: `_PHI_CACHE`'s key
# carried IONI_A3_SCALE, IONI_EXC_SCALE and IONI_TMAX_SCALE but not
# IONI_KOKOULIN, which is why NOTES_RADOFF2 and NOTES_HADRONS both had to
# disable the cache by hand and why a switched-ON cell could silently reuse
# switched-OFF numbers.
#
# `_NOT_PHYSICS` is the explicit counterpart: names that look like knobs and
# are NOT, so the completeness audit (`barkas_probe.py guards`) can tell the
# two apart and fail on anything in neither list.
PHYSICS_GLOBALS = ("IONI_A3_SCALE", "IONI_EXC_SCALE", "IONI_TMAX_SCALE",
                   "IONI_KOKOULIN", "IONI_KOKOULIN_TCUT", "IONI_KOKOULIN_NBIN")
_NOT_PHYSICS = ("COVTOL",)          # a plotting/validation tolerance only


def physics_state():
    """The module's physics knobs, as a hashable, printable tuple."""
    g = globals()
    return tuple((n, repr(g[n])) for n in PHYSICS_GLOBALS)

_ALPHA_PRIME = 1.0 / (2.0 * np.pi * 137.035999084)   # G4's alphaprime
_KOK_TMIN = 0.1          # G4MuBetheBlochModel::limitKinEnergy = 100 keV
_KOK_MUMIN = 1000.0      # G4MuBetheBlochModel::lowestKinEnergy = 1 GeV

# ---------------------------------------------------- THE MUON-ONLY GUARD
# Geant4 puts the Kokoulin factor in `G4MuBetheBlochModel` and NOWHERE else.
# Hadrons are ionized by `G4hIonisation`, whose high-energy model is
# `G4BetheBlochModel`, which has no such factor -- measured against Geant4's
# own `CrossSectionPerVolume` in NOTES_HADRONS s3.1 (the muon agrees with the
# Kokoulin-weighted integral to 3.8e-5 and every hadron with TREE LEVEL to
# 4e-5).  The C++ export has always known this:
#
#   G4UniversalFluctuationForExtrapolator.cc
#     kokoulinOn && ekin > kKokMuMin && std::abs(particle->GetPDGEncoding()) == 13
#
# This module did not, and the record carries no PDG column, so
# `CVH_IONI_KOKOULIN=1` in a hadron job silently applied a MUON-shaped factor
# (a3 = ln(4E(E-T)/m_mu^2)) to a kaon.  NOTES_HADRONS s3.4 measured the damage:
# it makes K- look 4x BETTER (0.00311 -> 0.00071) and pbar 3x better, because
# its u-shape mimics the charge-odd mean-loss bias -- i.e. the mistake is in
# the direction that looks like success.
#
# The mass is RECOVERABLE from the record, exactly, and does not need a new
# column: the regime-2/3 record carries `beta^2` and `E` (added by
# NOTES_DELTASPEC s9.2 precisely so that a kaon's Tmax is not inverted as a
# muon's), and
#
#       E * sqrt(1 - beta^2) = (gamma m) * (1/gamma) = m
#
# is an identity, not an approximation.  Measured on the exported records it
# returns the PDG masses to 1e-9 relative (`_mass_from_record` self-test in
# barkas_probe.py `guards`).  So the guard is applied HERE, on every existing
# file, rather than waiting for a record-layout change and a re-export of
# every model in the study.
#
# The tolerance is loose (1 MeV) on purpose: it has to separate the muon from
# the pion, its nearest neighbour, and 105.66 vs 139.57 MeV is a 34 MeV gap.
_KOK_MASS_TOL = 1.0      # MeV
_KOK_SUPPRESSED = {"n": 0, "masses": set()}


def _mass_from_record(beta2, etot):
    """The projectile mass implied by the regime-2/3 record, in MeV.

    m = E sqrt(1 - beta^2).  Exact for any particle; the record has both
    columns, and a record that does not (stride 11) never reaches this code --
    `ioni_step_exponent` raises on it."""
    return np.asarray(etot, float) * np.sqrt(
        np.clip(1.0 - np.asarray(beta2, float), 0.0, None))


def _is_muon_record(beta2, etot):
    """Elementwise: is this step a MUON's?  The offline half of the C++
    `std::abs(GetPDGEncoding()) == 13`."""
    return np.abs(_mass_from_record(beta2, etot) - _MMU) < _KOK_MASS_TOL


def kokoulin_factor(T, etot):
    """f_K(T), vectorized over T and over the per-step total energy."""
    T = np.asarray(T, dtype=np.float64)
    E = np.broadcast_to(np.asarray(etot, dtype=np.float64), T.shape)
    f = np.ones_like(T)
    m = (T > _KOK_TMIN) & (T < E - _MMU)
    if np.any(m):
        a1 = np.log(1.0 + 2.0 * T[m] / _MEL)
        a3 = np.log(4.0 * E[m] * (E[m] - T[m]) / (_MMU * _MMU))
        f[m] = 1.0 + _ALPHA_PRIME * a1 * (a3 - a1)
    return f


def _kokoulin_exponent(xi, t0, tmax, beta2, etot, a, nbin=96, spinhalf=True):
    """The Kokoulin CORRECTION's contribution to the delta channel's exponent.

    f_K - 1 is smooth and slowly varying in ln T; e^{i a T} is not, and at
    a T ~ 2e4 no practical quadrature in T resolves it.  So f_K - 1 is made
    piecewise constant on a log grid and the OSCILLATORY integral is done in
    CLOSED FORM on each bin with the same `exact_delta_exponent`.  The
    substitution beta^2 -> beta^2 hi/Tmax is what turns that routine's
    (upper limit == beta^2 denominator) convention into a genuine sub-range
    integral whose beta^2 term still carries the KINEMATIC Tmax; summing
    contiguous bins reproduces the full-range closed form to 7e-15.
    """
    xi = np.atleast_1d(np.asarray(xi, float))
    t0 = np.atleast_1d(np.broadcast_to(np.asarray(t0, float), xi.shape))
    tmax = np.atleast_1d(np.asarray(tmax, float))
    beta2 = np.atleast_1d(np.asarray(beta2, float))
    etot = np.atleast_1d(np.asarray(etot, float))
    a = np.atleast_2d(a)
    S = np.zeros(a.shape[1], dtype=np.complex128)
    lo0 = np.maximum(t0, _KOK_TMIN)
    ismu = _is_muon_record(beta2, etot)
    # `etot - _MMU > _KOK_MUMIN` is the C++ `ekin > kKokMuMin` and is only the
    # KINEMATIC half of the guard; `ismu` is the species half, which the C++
    # has had all along and this module did not.
    act = (lo0 < tmax) & (etot - _MMU > _KOK_MUMIN) & ismu
    nsup = int((~ismu).sum())
    if nsup:
        # Not an error -- Geant4 does exactly this -- but a job that ASKED for
        # the correction on a hadron has a configuration problem, so it is
        # said once per distinct mass rather than never.
        m0 = float(np.round(_mass_from_record(beta2[~ismu], etot[~ismu])[0], 3))
        if m0 not in _KOK_SUPPRESSED["masses"]:
            _KOK_SUPPRESSED["masses"].add(m0)
            msg = (f"IONI_KOKOULIN is on but {nsup} step(s) carry m = "
                   f"{m0:.3f} MeV, not the muon's {_MMU:.3f}: Geant4 applies "
                   f"the Kokoulin factor in G4MuBetheBlochModel ONLY, so the "
                   f"correction is SUPPRESSED for those steps (NOTES_HADRONS "
                   f"s3).")
            logger.warning(msg)
            # ...and to stderr as well, because a caller that has not set up
            # the wums root logger would otherwise see NOTHING, and silence
            # read as success is the exact failure this guard exists to stop.
            print("[cf_track_resolution] WARNING: " + msg, file=sys.stderr)
    _KOK_SUPPRESSED["n"] += nsup
    if not act.any():
        return S
    xi, lo0, tmx, b2, et, aa = (xi[act], lo0[act], tmax[act], beta2[act],
                                etot[act], a[act])
    fr = np.linspace(0.0, 1.0, nbin + 1)
    edges = lo0[:, None] * (tmx / lo0)[:, None] ** fr[None, :]
    for m in range(nbin):
        lo, hi = edges[:, m], edges[:, m + 1]
        kap = kokoulin_factor(np.sqrt(lo * hi), et) - 1.0
        g = kap != 0.0
        if not g.any():
            continue
        S += exact_delta_exponent(xi[g] * kap[g], lo[g], hi[g],
                                  b2[g] * hi[g] / tmx[g], et[g], aa[g],
                                  spinhalf=spinhalf)
    return S


def ms_step_exponent(steps, wstd, tau):
    """Moliere log-CF exponent of one pooled MS block in standardized-z
    units, FF-cut per step, vectorized over steps with the ymax bucketing
    of cf_ms_moliere._slot_S. steps: (n, 8) msmoliv records [effZ, effA,
    xg, pGeV, beta, thp2, dOverX0, stepGroup]."""
    ok = steps[:, 5] > 0.
    if not ok.any():
        return np.zeros(len(tau))
    # stride 10 (2026-08-08) carries the per-element sums in cols 8,9;
    # stride 8 files fall back to the effZ approximation inside
    # moliere_params.
    _st = steps[ok]
    if _st.shape[1] >= 10:
        prm = np.array([moliere_params(*s[:5], s[7], s[8]) for s in _st])
    else:
        prm = np.array([moliere_params(*s[:5]) for s in _st])  # (ns, 3)
    chic2, chia2, thff2 = prm[:, 0], prm[:, 1], prm[:, 2]
    act = (chic2 > 0.) & (chia2 > 0.)
    if not act.any():
        return np.zeros(len(tau))
    chic2, chia2, thff2 = chic2[act], chia2[act], thff2[act]
    args = np.sqrt(chia2)[:, None] * (wstd * tau)[None, :]
    ym = np.sqrt(thff2 / chia2)
    lg = np.log(np.clip(ym, 1e1, 1e7))
    rows = np.round((lg - np.log(1e1)) / (np.log(1e7 / 1e1) / 12)).astype(int)
    S = np.zeros(len(tau))
    for r in np.unique(rows):
        m = rows == r
        ymr = float(np.exp(np.log(1e1) + r * (np.log(1e7 / 1e1) / 12)))
        gsh = gshape(args[m].ravel(), ymax=ymr).reshape(int(m.sum()), len(tau))
        S += np.sum((chic2[m] / chia2[m])[:, None] * gsh, axis=0)
    return S


def extract(args):
    """Per selected track: z_obs, sigma, eta, and the three exponent
    components on TG (Gaussian variance share Vg_frac; MS real exponent;
    ionization complex exponent), stored separately so per-family k
    scalings can be applied at closure time."""
    files = sorted(glob.glob(args.files))[:args.ntasks]
    logger.info(f"{len(files)} files")
    zs, sigs, etas, phis, chgs, vgf = [], [], [], [], [], []
    # Track-quality and kinematics, stored so the closure can be scanned
    # AGAINST THE SELECTION rather than reported at one arbitrary cut. The
    # historical trimming-dependence (MS +0.007 at chi2/hit < 10 against
    # -0.073 at < 3) was never separable from model error without these
    # (2026-08-08).
    chi2n, nvhit, ptrk, ptgen, chisq, ndofs = [], [], [], [], [], []
    Sms_l, Sio_re_l, Sio_im_l = [], [], []
    nsel = ndropcov = ndropgen = 0
    pt = None
    for fn in files:
        try:
            f = uproot.open(fn)
            if "tree" not in f:
                continue
            if pt is None:
                pt = f["runtree"]["parmtype"].array(library="np")
            t = f["tree"]
        except Exception as e:
            logger.warning(f"skipping {fn}: {type(e).__name__}")
            continue
        a = t.arrays(["refParms", "refCov", "genParms", "resinfcov",
                      "resinfvarv", "reseigidx", "msmoliidx", "msmoliv",
                      "ioniurbanidx", "ioniurbanv",
                      "normalizedChi2", "nValidHits", "trackPt", "genPt",
                      "chisqval", "ndof"],
                     library="np")
        for ic in range(len(a["resinfcov"])):
            qg = a["genParms"][ic][0]
            if qg == 0.:
                ndropgen += 1
                continue
            c00 = float(a["refCov"][ic][0])
            cov = float(a["resinfcov"][ic])
            if not (c00 > 0.) or abs(cov / c00 - 1.) > COVTOL:
                ndropcov += 1
                continue
            sig = np.sqrt(c00)
            gi = np.asarray(a["reseigidx"][ic])
            vb = np.asarray(a["resinfvarv"][ic], dtype=np.float64)
            fam = pt[gi]
            vgauss = vb[(fam == 8) | (fam == 9)].sum()

            uvm = np.asarray(a["msmoliv"][ic], dtype=np.float64)
            uim = np.asarray(a["msmoliidx"][ic])
            uvm = uvm.reshape(-1, len(uvm) // max(len(uim), 1)) if len(uim) else uvm.reshape(0, 8)
            uvi = np.asarray(a["ioniurbanv"][ic], dtype=np.float64)
            uii = np.asarray(a["ioniurbanidx"][ic])
            uvi = uvi.reshape(-1, len(uvi) // max(len(uii), 1)) if len(uii) else uvi.reshape(0, 11)

            Sms = np.zeros(len(TG))
            Sio = np.zeros(len(TG), dtype=np.complex128)
            ok = True
            for famcode, (uidx, uv) in ((10, (uim, uvm)), (11, (uii, uvi))):
                sel = fam == famcode
                for g in np.unique(gi[sel]):
                    vpool = vb[sel & (gi == g)].sum()
                    if vpool <= 0.:
                        continue
                    steps = uv[uidx == g]
                    if not len(steps):
                        ok = False
                        break
                    if famcode == 10:
                        sq2 = steps[:, 5].sum()
                        if sq2 <= 0.:
                            continue
                        wstd = np.sqrt(vpool / sq2) / sig
                        Sms += ms_step_exponent(steps, wstd, TG)
                    else:
                        gq = steps[:, 10] * 1e-3
                        sq2 = float(np.sum(steps[:, 1] * gq * gq))
                        if sq2 <= 0.:
                            continue
                        wstd = np.sqrt(vpool / sq2) / sig
                        Sio += ioni_step_exponent(steps, wstd, TG)
                if not ok:
                    break
            if not ok:
                continue

            zs.append((a["refParms"][ic][0] - qg) / sig)
            sigs.append(sig)
            etas.append(-np.log(np.tan((np.pi / 2. - a["refParms"][ic][1]) / 2.)))
            # phi is stored so a SIM-vs-refit field mismatch can be TESTED
            # rather than assumed: the OAE tracker parametrization is
            # phi-symmetric, so any residual phi structure in <z> is the
            # sharpest handle on a field-model difference (2026-08-07).
            phis.append(a["refParms"][ic][2])
            # charge from sign(gen q/p): the discriminator between a
            # curvature-like (charge-ODD) and a material/eloss-like
            # (charge-EVEN) bias. Useless on the mu- only gun sample,
            # essential on the both-charge one (2026-08-07).
            chgs.append(np.sign(qg))
            vgf.append(vgauss / c00)
            chi2n.append(float(a["normalizedChi2"][ic]))
            nvhit.append(float(a["nValidHits"][ic]))
            ptrk.append(float(a["trackPt"][ic]))
            ptgen.append(float(a["genPt"][ic]))
            # chisqval/nValidHits is the HISTORICAL trim variable (the
            # --max-chi2-per-hit of fit_global_grads). Stored raw so the exact
            # historical cut can be reproduced rather than approximated by
            # normalizedChi2 = chisqval/ndof.
            chisq.append(float(a["chisqval"][ic]))
            ndofs.append(float(a["ndof"][ic]))
            Sms_l.append(Sms.astype(np.float32))
            Sio_re_l.append(Sio.real.astype(np.float32))
            Sio_im_l.append(Sio.imag.astype(np.float32))
            nsel += 1
        logger.info(f"{fn.split('/')[-2]}: cumulative {nsel} tracks "
                    f"(drop gen {ndropgen}, cov {ndropcov})")
        if nsel >= args.max_tracks:
            break
    os.makedirs(os.path.dirname(args.cache), exist_ok=True)
    np.savez_compressed(args.cache, z=np.array(zs), sigma=np.array(sigs),
                        eta=np.array(etas), phi=np.array(phis),
                        charge=np.array(chgs), vgf=np.array(vgf),
                        normchi2=np.array(chi2n), nvalidhits=np.array(nvhit),
                        trackpt=np.array(ptrk), genpt=np.array(ptgen),
                        chisqval=np.array(chisq), ndof=np.array(ndofs),
                        Sms=np.array(Sms_l), Sio_re=np.array(Sio_re_l),
                        Sio_im=np.array(Sio_im_l), tgrid=TG)
    logger.info(f"wrote {args.cache} ({nsel} tracks)")


def model_phi(d, args):
    """Complex phi_z(t) per track on TG with the per-family k applied.
    Total variance is renormalized so z stays standardized to the
    *scaled* model: sigma_model^2/sigma^2 = e^khit*vgf + scaled tails --
    handled by evaluating phi of the scaled physics and letting closure
    compare against z_obs built with the unscaled sigma."""
    vg = np.exp(args.khit) * d["vgf"][:, None]
    S = (-0.5 * vg * TG[None, :] ** 2
         + np.exp(args.kms) * d["Sms"]
         + np.exp(args.kioni) * (d["Sio_re"] + 1j * d["Sio_im"]))
    return np.exp(S)


def weier(phi, u):
    """E[e^{-u z^2}] per track from phi on TG (even weight, Re phi)."""
    w = np.exp(-TG ** 2 / (4. * u)) / np.sqrt(np.pi * u)
    return np.clip(np.trapezoid(phi.real * w[None, :], TG, axis=1), 0., 1.)


def closure(args, outdir):
    d = np.load(args.cache)
    n = len(d["z"])
    logger.info(f"{n} tracks from {args.cache}")
    phi = model_phi(d, args)
    z = d["z"]

    # per-probe closure, inclusive and in predicted-sigma quartiles
    qs = np.quantile(d["sigma"], np.linspace(0., 1., args.nsigma_bins + 1))
    lines = []
    for u in args.probes:
        Em = weier(phi, u)
        Ed = np.exp(-u * z ** 2)
        dm, sm = Ed.mean() - Em.mean(), Ed.std() / np.sqrt(n)
        lines.append(f"u={u:5.2f}  <data>-<model> = {dm:+.5f} +- {sm:.5f}  "
                     f"(<model> = {Em.mean():.4f})")
    logger.info("inclusive closure:\n" + "\n".join("  " + s for s in lines))

    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    # left: pull histogram vs averaged predicted lineshape
    ax = axs[0]
    zg = np.linspace(-8., 8., 161)
    # p(z) = 1/pi Int_0^inf Re[phi(t) e^{-itz}] dt, averaged over tracks
    ph = phi.mean(axis=0)
    pz = np.array([np.trapezoid((ph * np.exp(-1j * TG * zz)).real, TG) / np.pi
                   for zz in zg])
    ax.hist(np.clip(z, zg[0], zg[-1]), bins=zg, density=True,
            histtype="step", color="black", label="pulls (gen-matched MC)")
    ax.plot(zg, pz, color="crimson", label="CF-product prediction")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 2.)
    ax.set_xlabel(r"$z = \Delta(q/p)/\sigma_{\mathrm{pred}}$")
    ax.set_ylabel("density")
    ax.legend()
    # right: data-model difference of the bounded statistic vs u, by sigma bin
    ax = axs[1]
    uu = np.geomspace(0.02, 4., 25)
    for ib in range(args.nsigma_bins):
        m = (d["sigma"] >= qs[ib]) & (d["sigma"] <= qs[ib + 1])
        dd = np.array([np.exp(-u * z[m] ** 2).mean() - weier(phi[m], u).mean()
                       for u in uu])
        ss = np.array([np.exp(-u * z[m] ** 2).std() / np.sqrt(m.sum()) for u in uu])
        ax.errorbar(uu, dd, ss, marker="o", markersize=4, linestyle="-",
                    label=rf"$\sigma \in [{1e2*qs[ib]:.2f}, {1e2*qs[ib+1]:.2f}]\times 10^{{-2}}$")
    ax.axhline(0., color="gray", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"probe $u$")
    ax.set_ylabel(r"$\langle e^{-uz^2}\rangle_{\mathrm{data}} - \langle e^{-uz^2}\rangle_{\mathrm{model}}$")
    ax.legend(fontsize="x-small")
    name = f"trackres_closure{args.postfix}"
    plot_tools.save_pdf_and_png(outdir, name, fig)
    output_tools.write_logfile(outdir, name, args=args,
                               wd=os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(outdir, f"{name}.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    logger.info(f"wrote {outdir}/{name}")


def main():
    global logger
    logger = logging.setup_logger(__file__, 3, False)
    args = parse_args()
    outdir = args.outpath or os.path.expanduser(
        f"~/public_html/cvh/{datetime.date.today().strftime('%y%m%d')}_trackres/")
    os.makedirs(outdir, exist_ok=True)
    if args.extract:
        extract(args)
    if args.closure:
        closure(args, outdir)


if __name__ == "__main__":
    main()
