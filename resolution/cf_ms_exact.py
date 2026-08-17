"""Phase B: exact screened-Rutherford compound-Poisson CF fit of the
multiple-scattering scale, using the raw per-step material export
(msmoliidx / msmoliv branches) and gradllv.

Model per MS block b (parmtype 10), mirroring cf_ioni_exact.py:

  z_b = sqrt(h) * z_true + sqrt(1-h) * s_est * z_gauss,  h = nu (hat value)

  z_true: weighted sum of per-step projected Moliere deflections,
  standardized by the model (sum_s thp2_s = the Rossi variances actually in
  Q). Per step, the single-scattering density in theta^2 is
      n(t2) = chi_c^2 / (t2 + chi_a^2)^2      (N_scat = chi_c^2/chi_a^2)
  with (PDG/Bethe conventions, p in GeV, x in g/cm^2):
      chi_c^2 = 0.157e-6 * Z(Z+1)/A * x / (p^2 beta^2)         [rad^2]
      chi_a^2 = chi_0^2 * (1.13 + 3.76 (alpha Z / beta)^2)
      chi_0   = 2.007e-5 * Z^(1/3) * (1 + 3.34 (alpha Z / beta)^2)^0 / p
                -- classic Moliere screening angle m_e alpha Z^(1/3)/(0.885 p)
  The projected-angle CF of one step's compound is
      S_step(t) = Int_0^inf (J0(t*theta) - 1) n(theta^2) dtheta^2
  (J0 from the azimuthal average; even -> exactly centered). The material
  scale k multiplies chi_c^2 (collision rate) only, so the block exponent
  is again linear in e^k: phi_true = exp(e^k * S_b(tau)).

  Per-step weights into the block scalar: w_s^2 = thp2_s / sum thp2 (the
  R-projection lever arms are not exported; variance-matched uniform
  weighting is the v1 approximation, same spirit as the ioni fit).

Fit: joint (k, s_est) over the probe statistics E[e^{-u z^2}] via the
Weierstrass transform. With h(ms) ~ 0.3 the signal is O(30%) of the block
residual (vs 1e-6 for ioni), so the joint fit is expected well-conditioned.
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
from scipy.optimize import minimize
from scipy.special import j0

from wums import logging, output_tools, plot_tools

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)

NPARS = 7
TGRID = np.linspace(0.0, 12.0, 512)
ALPHA_EM = 1.0 / 137.036


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input",
                   default="/ceph/submit/data/user/d/david_w/ZMass/cvh/"
                           "resolution_closure_260724_moli_alpha999_fb01e7b7741/"
                           "task_*/globalcor_resclosure_0.root")
    p.add_argument("--ntasks", type=int, default=10)
    p.add_argument("--max-blocks", type=int, default=40000)
    p.add_argument("--cache", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "runs/cf_ms_cache.npz"))
    p.add_argument("--extract", action="store_true")
    p.add_argument("--fit", action="store_true")
    p.add_argument("--probes", type=float, nargs="+",
                   default=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
    p.add_argument("--outpath", default=None)
    p.add_argument("--postfix", default="")
    return p.parse_args()


# Switchable so the two G4-matching corrections can be A/B tested against
# the clean-propagation ground truth (see Documents/Resolution/NOTES.md).
# Coefficient f in chi_a^2 *= (1 + f*exp(-Z^2/1000)).
#   f = 0 -> the original (pre-2026-08-07) screening
#   f = 1 -> Geant4's G4WentzelOKandVIxSection.cc:154 exactly
# Scannable so the clean-propagation ground truth can say whether the
# residual is a MAGNITUDE problem (some f closes all probes) or a
# SHAPE problem (no single f does).
G4_SCREEN_F = 1.0   # G4's value. NOT tuned -- see NOTES: the 0.7 the data wants is UNEXPLAINED.
# G4 carries the nuclear form factor SQUARED in the cross section:
#   G4WentzelOKandVIxSection.cc:356-357  fm = 1/(1+formf*z1)^2   <- this is F(q^2)
#   :373                                 grej = (...)*fm*fm      <- |F|^2
# so the weight is (1+q^2 R^2/12)^-4 and ours was (1+..)^-2, i.e. |F|.
# The SCALE is right: thff2 = 2/formfactA holds to 1.0049 for every
# material and momentum (C/Al/Si/Cu at 3 and 40 GeV), the 0.5% being
# R = 1.27 A^0.27 fm vs G4's constn = 6.937e-6. An earlier note here
# claimed a scale error to justify leaving this off -- that was WRONG.
# Enabled because it is the correct equation; it moves the closure the
# right way on its own, and that it adds to the screening overshoot is
# a statement about the UNEXPLAINED residual, not about this term.
G4_FF_SQUARED = True

# ---------------------------------------------------------------------------
# Two constants where the model and Geant4 disagree. Both are HARMONISATIONS --
# G4's value is the target, ours is simply a different number for the same
# quantity -- and both are ~1e-5/3e-5 in the closure, i.e. 10-100x BELOW the
# residual. They are ranks 7 and 8 of the harmonisation list in
# NOTES_MOLIEREWRONG s8. Expect no measurable closure change; the argument is
# correctness, exactly as for the Urban excitation-version harmonisation.
#
# Kept as SEPARATE switches so each is attributable. Note the MS channel
# currently OVER-corrects on the outermost-plane statistic (144/144 probes
# negative, NOTES_CLOSURE_ALLCORR), so a term that happens to push the same way
# makes the accumulated number slightly worse; the sign was not recorded when
# these were ranked, which is why they are measured rather than assumed benign.

# Rank 7. chi_0 = alpha m_e / 0.88534 (Thomas-Fermi screening momentum).
#   ours     4.214e-6   GeV
#   Geant4   4.211714e-6 GeV = ALPHA_EM * m_e / 0.88534
# +0.054 % on chi_0, hence +0.109 % on chi_a^2, which enters the Moliere log.
MS_CHI0_G4 = False
_CHI0_OURS = 4.214e-6
_CHI0_G4 = ALPHA_EM * 0.51099895e-3 / 0.88534

# Rank 8. The nuclear form-factor angle. Ours is built from R = 1.27 A^0.27 fm;
# G4 uses formfactA with constn = 6.937e-6 MeV^-2 and FormFactor ~ A^0.54, and
# thff2 = 2/formfactA. The two agree to 1.0045-1.0049 across C/Al/Si/Cu at 3
# and 40 GeV -- ours is HIGH -- the difference being the nuclear-radius
# constant, not the form.
MS_FF_G4 = False
_FF_CONSTN_MEV2 = 6.937e-6

def moliere_params(effZ, effA, xg, pGeV, beta, zzp1OverA=None, lnScreenW=None):
    """chi_c^2, chi_a^2 and the nuclear form-factor cutoff theta_FF^2
    [rad^2] for one step (WentzelVI-consistent single-scattering inputs).

    chi_0 = m_e alpha / 0.885 * Z^(1/3) / p  (Thomas-Fermi screening;
    4.214e-6 GeV -- NOTE: an earlier version used 2.007e-5, conflating the
    Lynch-Dahl chi_a^2 constant with a linear chi_0 formula: chi_a^2 was
    ~23x too large, N_scat ~23x too small. Fixed 2026-07-24.)
    theta_FF = hbar c / (p R_N), R_N = 1.27 A^0.27 fm: dipole/nuclear
    form-factor scale that terminates the single-Rutherford tail (G4
    WentzelVI's FF); implemented as a hard cutoff of the y^2 integral.
    Mott (McKinley-Feshbach) factor not yet included (few-% tail shape).
    """
    # PER-ELEMENT sums when the exporter provides them (msmoliv stride 10,
    # 2026-08-08). Both Moliere parameters are non-linear in Z and Geant4
    # evaluates them per element, so using the mass-averaged effZ/effA is
    # wrong for compounds -- it was what forced the empirical G4_SCREEN_F=0.7.
    #   zzp1OverA = sum_i massfrac_i Z_i(Z_i+1)/A_i
    #   lnScreenW = scattering-power-weighted mean of
    #               ln[Z^(2/3)(1.13+3.76(alpha Z/beta)^2)(1+exp(-Z^2/1000))]
    # The geometric mean is the right one for chi_a: the exponent depends on
    # it only through ln(chi_a).
    if zzp1OverA is not None and zzp1OverA > 0.:
        chic2 = 0.157e-6 * zzp1OverA * xg / (pGeV ** 2 * beta ** 2)
    else:
        chic2 = 0.157e-6 * effZ * (effZ + 1.) / effA * xg / (pGeV ** 2 * beta ** 2)
    az = ALPHA_EM * effZ / beta
    _chi0c = _CHI0_G4 if MS_CHI0_G4 else _CHI0_OURS
    chi0 = _chi0c * effZ ** (1. / 3.) / pGeV
    # G4 SCREENING FACTOR (2026-08-07). G4WentzelOKandVIxSection.cc:154
    #   ScreenRSquare[j]     = afact*(1 + G4Exp(-j*j*0.001))*Z^(2/3)   <- NUCLEUS
    #   ScreenRSquareElec[j] = afact*Z^(2/3)                           <- electrons
    # and for muons (:217) screenZ = (1.13 + 3.76 Z^2 alpha^2/beta^2)
    # * ScreenRSquare[Z]/p^2.  G4's Wentzel form is 1/(1-cos+screenZ)^2 in
    # d(1-cos), which maps to chi_a^2 = 2*screenZ; and 2*afact = 1.776e-11
    # = chi0^2 to 0.1%.  So G4's screening angle is OURS times
    # (1 + exp(-Z^2/1000)) -- 1.822 for Si, 1.965 for C, 1.431 for Cu.
    # Without it chi_a^2 is ~1.8x too SMALL for silicon, which inflates the
    # Moliere log and over-predicts the scattering.
    if lnScreenW is not None and np.isfinite(lnScreenW) and lnScreenW != 0.:
        # exact: chi_a^2 = (4.214e-6/p)^2 * exp(<ln screening>_weighted)
        chia2 = (_chi0c / pGeV) ** 2 * np.exp(lnScreenW)
    else:
        # legacy fallback for stride-8 files: apply the G4 factor at effZ with
        # the empirically calibrated coefficient (mix-dependent, see NOTES).
        g4screen = 1. + G4_SCREEN_F * np.exp(-effZ * effZ * 1.0e-3)
        chia2 = chi0 ** 2 * (1.13 + 3.76 * az ** 2) * g4screen
    rn_fm = 1.27 * max(effA, 1.) ** 0.27
    # dipole form-factor characteristic angle^2 (G4WentzelOKandVIxSection
    # convention: FF = 1/(1 + q^2 R^2/12)^2, i.e. theta_c^2 = 12 (hbarc/pR)^2
    # -- the earlier hard cutoff at (hbarc/pR)^2 was 12x too tight and the
    # wrong shape)
    if MS_FF_G4:
        # G4's own: formfactA = constn * A^0.54 * p^2 (p in MeV), thff2 = 2/formfactA
        _pmev = pGeV * 1.0e3
        thff2 = 2. / (_FF_CONSTN_MEV2 * max(effA, 1.) ** 0.54 * _pmev ** 2)
    else:
        thff2 = 12. * (0.19733 / (pGeV * rn_fm)) ** 2
    return chic2, chia2, thff2


# Universal Moliere shape: with theta = sqrt(chia2)*y,
#   S_step(t) = (chic2/chia2) * G(t*sqrt(chia2)),
#   G(tau) = Int_0^inf (J0(tau*y) - 1) / (1+y^2)^2 dy^2
# Precomputed once on a log-tau grid; every step is then an interpolation.
# G(0) = 0, G(inf) -> -1 (one unit per scatter). The y^2 grid upper end is
# the kinematic-scale cutoff of the single-scattering spectrum; the CF
# converges (unlike the variance) so the exact value is uncritical.
_Y2 = np.logspace(-4, 13, 360)
_DY2 = np.gradient(_Y2)
_GTAU = np.logspace(-8, 4, 1600)
# 2D shape table: G(tau; ymax) with the y^2 integral cut at ymax^2
# (nuclear form-factor cutoff, ymax = theta_FF/chi_a per step).
_YMAXG = np.logspace(1, 7, 13)

# --------------------------------------------------------------------------
# j0(x) - 1 CANCELLATION GUARD  (bug found in NOTES_XXII 8.4, fixed 2026-08-14)
#
# The table was built as `j0(outer(_GTAU, sqrt(_Y2))) - 1`, unguarded. Both
# terms are ~1 and the difference is -x^2/4, so below |x| ~ 1e-8 every digit is
# lost and the entry collapses to 0. `_GTAU` starts at 1e-8 and `_Y2` at 1e-4,
# so the SMALLEST-TAU ROWS -- exactly the ones `gshape` uses directly and
# through its row[0]*(tau/_GTAU[0])^2 extrapolation -- were corrupted, measured
# 1.3-1.7 % LOW.  Negligible for q/p (MS is <= 5e-5 of that variance);
# ~0.7 % on the predicted width wherever MS dominates (position, angle).
#
# `set_j0_guard(False)` restores the exact legacy table so any number recorded
# before this date can be reproduced bit-for-bit.
# --------------------------------------------------------------------------
J0M1_GUARD = True

# The knob registry (see cf_track_resolution.PHYSICS_GLOBALS).  All three of
# these change the Moliere kernel, and only J0M1_GUARD was in any cache key
# before 2026-08-16.  `_NOT_PHYSICS` are fixed constants / array shapes.
PHYSICS_GLOBALS = ("J0M1_GUARD", "G4_FF_SQUARED", "G4_SCREEN_F", "MS_CHI0_G4", "MS_FF_G4")
_NOT_PHYSICS = ("NPARS", "ALPHA_EM")


def physics_state():
    g = globals()
    return tuple((n, repr(g[n])) for n in PHYSICS_GLOBALS)


def _j0m1(x):
    """J0(x) - 1, evaluated stably at small argument.

    J0(x) - 1 = -x^2/4 (1 - x^2/16 + x^4/576 - ...), truncated at 1e-16
    relative for |x| < 1e-2, where the direct difference has already lost
    ~4 significant digits.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty(x.shape, dtype=np.float64)
    small = np.abs(x) < 1e-2
    x2 = x[small] ** 2
    out[small] = -0.25 * x2 * (1. - x2 / 16. + x2 * x2 / 576.)
    out[~small] = j0(x[~small]) - 1.
    return out


def _build_tables(guard=True):
    """(Re)build the universal shape table.  Sets the module globals used by
    `gshape`, so callers that did `from cf_ms_exact import gshape` still see
    the change."""
    global _G2D, _G, J0M1_GUARD
    J0M1_GUARD = bool(guard)
    k0 = (_j0m1(np.outer(_GTAU, np.sqrt(_Y2))) if guard
          else j0(np.outer(_GTAU, np.sqrt(_Y2))) - 1.)
    g2d = np.empty((len(_YMAXG), len(_GTAU)))
    for iy, ym in enumerate(_YMAXG):
        # smooth dipole form factor 1/(1+y^2/ymax^2)^2 (G4 convention), not a
        # hard cutoff
        # G4 applies the exponential nuclear form factor SQUARED:
        # G4WentzelOKandVIxSection.cc:355 fm = 1/(1+formfactA*z1)^2 and the
        # cross section carries fm*fm (:373), i.e. |F|^2 = (1+q^2R^2/12)^-4.
        # Ours had the square root of that.
        ffpow = 4 if G4_FF_SQUARED else 2
        w = _DY2 / (1. + _Y2) ** 2 / (1. + _Y2 / ym ** 2) ** ffpow
        g2d[iy] = k0 @ w
    _G2D = g2d
    _G = _G2D[-1]  # backwards-compatible: effectively uncut


def set_j0_guard(flag):
    """Switch the j0(x)-1 guard on/off and rebuild the table (for A/B tests)."""
    _build_tables(bool(flag))


_build_tables(J0M1_GUARD)


def gshape(tau, ymax=None):
    """Universal exponent shape, optionally with the FF cutoff ymax;
    linear interpolation in log(ymax) between precomputed rows."""
    tau = np.abs(np.asarray(tau, dtype=np.float64))
    if ymax is None:
        row = _G
    else:
        ly = np.clip(np.log(ymax), np.log(_YMAXG[0]), np.log(_YMAXG[-1]))
        fi = (ly - np.log(_YMAXG[0])) / (np.log(_YMAXG[1]) - np.log(_YMAXG[0]))
        i0 = int(np.clip(np.floor(fi), 0, len(_YMAXG) - 2))
        f = fi - i0
        row = (1. - f) * _G2D[i0] + f * _G2D[i0 + 1]
    out = np.interp(tau, _GTAU, row, left=np.nan, right=row[-1])
    tiny = tau < _GTAU[0]
    if np.any(tiny):
        out[tiny] = row[0] * (tau[tiny] / _GTAU[0]) ** 2
    return out


# =========================================================================
# THE ATOMIC-ELECTRON KERNELS  (2026-08-17, NOTES_MOLIEREWRONG)
#
# Moliere's `chi_c^2 ~ Z(Z+1)` gives the atomic electrons the NUCLEUS's angular
# law and the NUCLEUS's angular range.  Both are wrong, and the second one is
# the only species-dependent term in the whole MS enumeration.  The electron
# term's single-scattering density is
#
#     dn/dt = chi_c,e^2 / (t + chi_a^2)^2 * R_beta(t/t_e) * Theta(t < t_e)
#
# with t = theta^2 and, from the EXACT two-body kinematics of a heavy
# projectile on a free electron at rest,
#
#     t(T) = t_G4 * tau (1 - tau),   tau = T/Tmax,   t_G4 = 2 m_e Tmax / p^2
#
# -- i.e. G4's own `1 - cos = Tmax m_e/p^2` is the LEADING TERM of a relation
# that turns over at tau = 1/2.  The projectile's deflection is therefore
# capped at
#
#     t_e = t_G4/4          (theta_e,max = theta_G4/2, EXACTLY)
#
# and, since t(T) is two-valued, the density carries the Jacobian of both
# branches.  With sigma = sqrt(1 - t/t_e):
#
#     R_0(x)     = (2 - x) / (2 sqrt(1-x))            [pure 1/T^2]
#     R_beta(x)  = [ (1 + sigma^2) - beta^2 (1 - sigma^2)/2 ] / (2 sigma)
#
# the second form carrying the (1 - beta^2 T/Tmax) of the true spin-0
# `dsigma/dT` -- the same factor whose omission was the FIRST of the four
# ionization corrections.  R has an integrable inverse-square-root caustic at
# the edge and is exactly ZERO above it.
#
# THE CHECK THAT DECIDES THE DECOMPOSITION, and it is analytic:
#
#     Int_0^{t_e} t dn/dt dt / chi_c,e^2  =  ln(t_G4/chi_a^2) - 1
#
# because Int_0^1 [R_0(x) - 1] dx/x = 2 ln 2 - 1 exactly, and ln(t_e) + 2ln2
# = ln(t_G4).  That is IDENTICALLY `G4WentzelOKandVIxSection`'s own electron
# transport log `f(x_e) = ln(1+x) - x/(1+x)`.  **G4's ceiling is the right
# TRANSPORT MOMENT with the wrong SHAPE**: the caustic pile-up at t_G4/4
# exactly pays for the missing range between t_G4/4 and t_G4.  So `hard` and
# `kine` below differ only in shape, at identical variance, and `dipole`
# differs from both by 1.833 units of log (the `(1+t/Y)^-4` starts biting a
# decade below Y where a kinematic edge does not).
#
# Everything here is DEFAULT-INERT: the tables are built at import, nothing
# reads them unless `cf_track_resolution.MS_ELEC_EDGE` is set.
# =========================================================================

# 141 rows, 0.1 decade, in Y = ymax^2 (the y^2 scale, NOT the y scale
# `gshape` takes) -- ten times finer than `_YMAXG`, because the electron
# ceiling is a per-species number and must not be snapped the way the nuclear
# form-factor ceiling is.
_ELEC_Y = np.logspace(2., 16., 141)
# quadrature in v = -ln(y^2/Y) (below the ceiling) and w = +ln(y^2/Y) (above,
# for the dipole only).  h = 0.0247 in ln(y^2) against `_Y2`'s 0.1088: 4.4x
# finer, so the `dipole` row is the NUMERICS CONTROL for `gshape` itself.
_ELEC_V = np.linspace(0., 50.6, 2048)
_ELEC_W = np.linspace(0., 30.0, 512)
_GE = {}


def _build_elec_tables():
    """G_e(tau; Y) for the three electron kernels, on `_GTAU` x `_ELEC_Y`.

    `kine` is stored as TWO tables, the beta^2-independent piece and the
    coefficient of -beta^2, so one build covers every species exactly:
        G_kine(tau; Y, beta^2) = GE['kine0'] - beta^2 * GE['kine1'].
    """
    global _GE
    out = {k: np.empty((len(_ELEC_Y), len(_GTAU))) for k in
           ("dipole", "hard", "kine0", "kine1")}
    ev, ew = np.exp(-_ELEC_V), np.exp(_ELEC_W)
    sig = 1. - ev                      # sigma = 1 - e^-v  in [0, 1)
    for iy, Y in enumerate(_ELEC_Y):
        # --- below the ceiling: y^2 = Y e^-v, dy^2 = -Y e^-v dv
        y2 = Y * ev
        k0 = _j0m1(np.outer(_GTAU, np.sqrt(y2)))
        base = Y * ev / (1. + y2) ** 2
        out["hard"][iy] = np.trapezoid(k0 * base[None, :], _ELEC_V, axis=1)
        # --- the exact kinematic kernel: y^2 = Y (1 - sigma^2), the caustic
        # removed analytically by the sigma substitution (R dx = -[...] dsigma)
        y2k = Y * (1. - sig ** 2)
        kk = _j0m1(np.outer(_GTAU, np.sqrt(y2k)))
        den = ev / (1. + y2k) ** 2      # the e^-v is dsigma/dv
        out["kine0"][iy] = Y * np.trapezoid(kk * ((1. + sig ** 2) * den)[None, :],
                                            _ELEC_V, axis=1)
        out["kine1"][iy] = Y * np.trapezoid(kk * (0.5 * (1. - sig ** 2) * den)[None, :],
                                            _ELEC_V, axis=1)
        # --- the dipole, on the SAME quadrature: below plus above the ceiling
        ffb = (1. + ev) ** -4
        lo = np.trapezoid(k0 * (base * ffb)[None, :], _ELEC_V, axis=1)
        y2a = Y * ew
        ka = _j0m1(np.outer(_GTAU, np.sqrt(y2a)))
        hi = np.trapezoid(ka * (Y * ew / (1. + y2a) ** 2 / (1. + ew) ** 4)[None, :],
                          _ELEC_W, axis=1)
        out["dipole"][iy] = lo + hi
    _GE = out


_build_elec_tables()


def gshape_elec(tau, y2max, kind="hard", beta2=0.0):
    """Electron-term exponent shape with its own ceiling, `y2max = (theta_e,max
    /chi_a)^2` -- note this is the SQUARE of what `gshape` takes -- by linear
    interpolation in log(y2max) between the 141 rows of `_ELEC_Y`."""
    tau = np.abs(np.asarray(tau, dtype=np.float64))
    ly = np.clip(np.log(y2max), np.log(_ELEC_Y[0]), np.log(_ELEC_Y[-1]))
    fi = (ly - np.log(_ELEC_Y[0])) / (np.log(_ELEC_Y[1]) - np.log(_ELEC_Y[0]))
    i0 = int(np.clip(np.floor(fi), 0, len(_ELEC_Y) - 2))
    f = fi - i0

    def _row(name):
        return (1. - f) * _GE[name][i0] + f * _GE[name][i0 + 1]

    if kind == "kine":
        row = _row("kine0") - float(beta2) * _row("kine1")
    else:
        row = _row(kind)
    out = np.interp(tau, _GTAU, row, left=np.nan, right=row[-1])
    tiny = tau < _GTAU[0]
    if np.any(tiny):
        out[tiny] = row[0] * (tau[tiny] / _GTAU[0]) ** 2
    return out


def elec_logrange(Y, kind, beta2=0.0):
    """The kernel's own transport log, Int t dn/dt dt / chi_c,e^2, read off the
    table at small tau (G -> -(tau^2/4) Lm).  Used to validate the tables
    against the analytic values:
        hard   : ln(1+Y) - Y/(1+Y)          -> ln Y - 1
        kine   : ln(4Y) - 1 - beta^2/2      (Y is the TRUE ceiling)
        dipole : ln Y - 2.833...
    """
    t0 = _GTAU[0]
    g = gshape_elec(np.array([t0]), Y, kind=kind, beta2=beta2)[0]
    return -4. * g / t0 ** 2


def block_exponent(steps, tau):
    """S_b(tau) = sum_s (chic2/chia2) G(tau/sigma_ref * sqrt(chia2)),
    absolute-chic2 normalization: sigma_ref from the exported thp2 (the Q
    matrix), chic2/chia2 from first principles. k=0 closure then tests
    Geant4's realized scattering against Moliere theory directly.
    steps: (n, NPARS) = [effZ, effA, xg, pGeV, beta, thp2, dOverX0]."""
    thp2 = steps[:, 5].astype(np.float64)
    tot = thp2.sum()
    if tot <= 0.:
        return None
    sig = np.sqrt(tot)
    S = np.zeros(len(tau))
    for s in range(len(steps)):
        if thp2[s] <= 0.:
            continue
        chic2, chia2, thff2 = moliere_params(*steps[s, :5])
        if chic2 <= 0. or chia2 <= 0.:
            continue
        S += (chic2 / chia2) * gshape(tau / sig * np.sqrt(chia2),
                                      ymax=np.sqrt(thff2 / chia2))
    return S


def extract(files, args):
    z2s, hs, Ss, reffs = [], [], [], []
    pt = None
    nblocks = 0
    for fn in files[:args.ntasks]:
        f = uproot.open(fn)
        if "tree" not in f:
            continue
        if pt is None:
            pt = f["runtree"]["parmtype"].array(library="np")
        t = f["tree"]
        a = t.arrays(["globalidxv", "gradchisqv", "gradllv", "hesspackedv",
                      "msmoliidx", "msmoliv"], library="np")
        for ic in range(len(a["globalidxv"])):
            gi = np.asarray(a["globalidxv"][ic])
            tp = pt[gi]
            m10 = np.where(tp == 10)[0]
            if not len(m10):
                continue
            nu = np.asarray(a["gradllv"][ic], dtype=np.float64)
            q = -np.asarray(a["gradchisqv"][ic], dtype=np.float64)
            hp = np.asarray(a["hesspackedv"][ic], dtype=np.float64)
            n = len(gi)
            rr = np.arange(n)
            hdiag = hp[(rr * n - rr * (rr - 1) // 2).astype(np.int64)]
            uidx = np.asarray(a["msmoliidx"][ic])
            uv = np.asarray(a["msmoliv"][ic], dtype=np.float64)
            # stride detection: 7 floats/step (260724_moli) or 8 (+stepGroup)
            stride = len(uv) // max(len(uidx), 1)
            uv = uv.reshape(-1, stride)
            for j in m10:
                h = nu[j]
                if not (0. < h < 1.) or q[j] < 0.:
                    continue
                steps = uv[uidx == gi[j]]
                if not len(steps):
                    continue
                # effective rank of the block quadratic form from the
                # Fisher diagonal (c = 1 exactly, validated in cf0)
                reff = max(h * h / hdiag[j], 1.0) if hdiag[j] > 0. else 1.0
                # per-dof variance is 1/r (E[z^2]=1 pooled over r dofs), so
                # every per-dof CF argument carries 1/sqrt(r)
                S = block_exponent(steps, np.sqrt(h / reff) * TGRID)
                if S is None:
                    continue
                z2s.append(q[j] / h)
                hs.append(h)
                Ss.append(S)
                reffs.append(reff)
                nblocks += 1
            if nblocks >= args.max_blocks:
                break
        logger.info(f"{fn.split('/')[-2]}: cumulative {nblocks} blocks")
        if nblocks >= args.max_blocks:
            break
    np.savez_compressed(args.cache, z2=np.array(z2s), h=np.array(hs),
                        S=np.array(Ss), reff=np.array(reffs), tgrid=TGRID)
    logger.info(f"wrote {args.cache} ({nblocks} blocks)")


def model_stat(u, k, sest, S, h, tgrid, reff=None):
    """Dimension-r radial Weierstrass transform per entry:
      E[e^{-u z^2}]_b = Int_0^inf w_r(t) phi_b(t) dt,
      w_r(t) = 2 t^{r-1} e^{-t^2/4u} / ((4u)^{r/2} Gamma(r/2)),
    exact for any r in the Gaussian limit (gives (1+2u)^{-r/2}); the block's
    isotropic per-dof exponent S is shared across its r_eff dofs (J0 is the
    2D deflection CF, so r=2 is the exact two-projection case and non-integer
    r is the two-moment interpolation of the eigenvalue spectrum)."""
    from scipy.special import gammaln
    ek = np.exp(k)
    if reff is None:
        r = np.ones(len(h))
    else:
        r = reff
    # per-dof Gaussian estimation part also carries variance 1/r
    phi = np.exp(ek * S) * np.exp(-0.5 * sest * ((1. - h) / r)[:, None] * tgrid[None, :] ** 2)
    # avoid t=0 singularity for r<1 is not needed (r >= 1); t^{r-1} at t=0
    # is 1 for r=1 and 0 for r>1 -- handle t=0 node explicitly
    t = tgrid.copy()
    t[0] = 1e-12
    logw = (np.log(2.) + (r[:, None] - 1.) * np.log(t[None, :])
            - t[None, :] ** 2 / (4. * u)
            - 0.5 * r[:, None] * np.log(4. * u)
            - gammaln(0.5 * r)[:, None])
    vals = np.trapz(np.exp(logw) * phi, tgrid, axis=1)
    return np.mean(np.clip(vals, 0., 1.))


def fit(args, outdir):
    d = np.load(args.cache)
    z2, h, S, tgrid = d["z2"], d["h"], d["S"], d["tgrid"]
    reff = d["reff"] if "reff" in d.files else None
    logger.info(f"{len(z2)} blocks; h median {np.median(h):.3g} "
                f"[{np.percentile(h,16):.3g}, {np.percentile(h,84):.3g}]")
    Fd = np.array([np.mean(np.exp(-u * z2)) for u in args.probes])
    sF = np.array([np.std(np.exp(-u * z2)) / np.sqrt(len(z2)) for u in args.probes])

    print(f"{'u':>6s} {'F_data':>10s} {'F_model(0,1)':>12s}")
    for u, fd in zip(args.probes, Fd):
        print(f"{u:6.2f} {fd:10.6f} {model_stat(u, 0., 1., S, h, tgrid, reff):12.6f}")

    def chi2(p):
        return np.sum(((Fd - np.array([model_stat(u, p[0], p[1], S, h, tgrid, reff)
                                       for u in args.probes])) / sF) ** 2)

    res = minimize(chi2, [0., 0.95], method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-3})
    k, sest = res.x
    eps = np.array([1e-2, 2e-3])
    H = np.zeros((2, 2))
    for i in range(2):
        for j in range(i, 2):
            di = np.zeros(2); di[i] = eps[i]
            dj = np.zeros(2); dj[j] = eps[j]
            H[i, j] = H[j, i] = (chi2(res.x + di + dj) - chi2(res.x + di - dj)
                                 - chi2(res.x - di + dj) + chi2(res.x - di - dj)) \
                                / (4 * eps[i] * eps[j])
    cov = 2. * np.linalg.inv(H)
    errs = np.sqrt(np.diag(cov))
    rho = cov[0, 1] / (errs[0] * errs[1])
    ndf = len(args.probes) - 2
    print(f"\njoint fit: k = {k:.4f} +- {errs[0]:.4f}   "
          f"s_est = {sest:.4f} +- {errs[1]:.4f}   rho = {rho:.2f}   "
          f"chi2/ndf = {res.fun:.1f}/{ndf}")
    for u, fd in zip(args.probes, Fd):
        print(f"   u={u:5.2f}  F_data={fd:.6f}  "
              f"F_joint={model_stat(u, k, sest, S, h, tgrid, reff):.6f}")

    fig, ax = plt.subplots(figsize=(9, 7))
    kk = np.linspace(k - 0.3, k + 0.3, 25)
    prof = []
    for kv in kk:
        r1 = minimize(lambda s: chi2([kv, s[0]]), [sest], method="Nelder-Mead",
                      options={"xatol": 1e-4})
        prof.append(r1.fun)
    ax.plot(kk, np.array(prof) - res.fun, color="#2ca02c", lw=2)
    ax.axhline(1., color="k", ls="--", lw=1)
    ax.set_xlabel(r"MS material scale $k$")
    ax.set_ylabel(r"$\Delta\chi^2$ (s$_{est}$ profiled)")
    plot_tools.add_decor(ax, "CMS", "Work in progress (B→J/ψ+X MC gen closure)",
                         data=False, lumi=None, loc=2)
    name = "_".join(filter(None, ["cf_ms_profile", args.postfix]))
    plot_tools.save_pdf_and_png(outdir, name)
    output_tools.write_logfile(outdir, name, args=args,
                               wd=os.path.dirname(os.path.abspath(__file__)))
    np.savez(os.path.join(outdir, "cf_ms_results.npz"),
             k=k, sest=sest, errs=errs, rho=rho, chi2=res.fun,
             probes=np.array(args.probes), Fd=Fd)


def main():
    args = parse_args()
    if args.extract:
        files = sorted(glob.glob(args.input))
        if not files:
            sys.exit(f"no files match {args.input}")
        extract(files, args)
    if args.fit:
        today = datetime.date.today().strftime("%y%m%d")
        outdir = output_tools.make_plot_dir(
            args.outpath or os.path.expanduser(
                f"~/public_html/calibration_studies/{today}_cf_ms_exact/"))
        fit(args, outdir)


if __name__ == "__main__":
    main()
