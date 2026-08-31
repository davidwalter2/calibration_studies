"""Discrete delta-ray (knock-on electron) transverse recoil as a CF block.

PHYSICS.  Above the e- production cut Geant4's muIoni creates a real knock-on
electron and conserves 4-momentum exactly, so the muon is DEFLECTED by the
electron's transverse momentum.  Below the cut the loss is continuous and
collinear (no recoil), and that sub-cut electron scattering is carried by the
MS model instead (the Z(Z+1) in Moliere/Highland).  The CF has an angular MS
block and a LONGITUDINAL ionization block; the transverse recoil sits in
neither.  This module supplies it.

The kinematics collapse exactly.  For a muon on a free electron, the electron
of kinetic energy T comes out at cos(th_e) = T(E+m_e)/(p p_e) with
p_e^2 = T^2 + 2 m_e T, so

    p_eT^2 = p_e^2 sin^2(th_e) = 2 m_e T          (EXACT, no approximation)

hence the muon's deflection is th = sqrt(2 m_e T)/p.  This is the same relation
the existing MS_ELEC_TMAX ceiling uses (theta^2 = 2 Tmax m_e / p^2), which is
what makes the split at Tcut consistent between the two blocks.

Emission is a Poisson process with the Rutherford spectrum
dN/dT = xi/T^2 per step, xi = 0.1535 (Z/A) xg / beta^2 [MeV], so the block is
compound Poisson like MS and ionization already are.  Azimuth is uniform, so
projecting onto the bending plane gives a Bessel J0:

    S(t) = xi * Int_{Tcut}^{Tmax} T^-2 [ J0(w t sqrt(2 m_e T)/p) - 1 ] dT

Substituting u = sqrt(T) and s = a u with a = w t sqrt(2 m_e)/p:

    S(t) = xi a^2 [ Phi(a sqrt(Tcut)) - Phi(a sqrt(Tmax)) ],
    Phi(x) = Int_x^inf (2/s^3) [J0(s) - 1] ds

so the whole T-integral is ONE universal 1-D table Phi(x) -- the p-dependence
scales out.  Cost is two interpolations per (step, t), the same cost class as
ms_step_exponent, and it reuses the EXISTING MS influence weights (the kick is
angular at the same planes), so no new Jacobian, no refit, no change to the
CVH/CGF output.
"""
import numpy as np
from scipy.special import j0

ME = 0.51099895e-3        # electron mass  [GeV]
MMU = 0.1056583755        # muon mass      [GeV]
HALF_K = 0.1535           # (1/2) K = (1/2) 0.307075  [MeV cm^2/g]

# ---------------------------------------------------------------- Phi table
_LX = np.linspace(-14., 9., 4001)          # ln x
def _build_phi():
    """Phi(x) = int_x^inf 2 s^-3 [J0(s)-1] ds, by outward cumulative quadrature
    from the largest node (where the remaining tail is analytic to O(s^-3))."""
    x = np.exp(_LX)
    # integrand in ln s:  2 s^-3 [J0-1] * s ds/dlns = 2 s^-2 [J0(s)-1]
    g = 2. * np.exp(-2. * _LX) * (j0(x) - 1.)
    # cumulative from the RIGHT (x -> inf); tail beyond xmax: J0~0 so
    # int_xmax^inf -2 s^-3 ds = -1/xmax^2
    tail = -1. / x[-1] ** 2
    c = np.concatenate(([0.], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(_LX))))
    return (c[-1] - c) + tail                # Phi at each node
_PHI = _build_phi()

def phi_tab(x):
    """Phi(x), vectorized. Small-x limit is analytic: J0-1 ~ -s^2/4 gives
    Phi -> -(1/2)ln(1/x) + const, so extrapolate in ln x with slope 1/2."""
    lx = np.log(np.clip(x, 1e-300, None))
    out = np.interp(lx, _LX, _PHI)
    lo = lx < _LX[0]
    if np.any(lo):
        out = np.where(lo, _PHI[0] + 0.5 * (lx - _LX[0]), out)
    hi = lx > _LX[-1]
    if np.any(hi):
        out = np.where(hi, -np.exp(-2. * lx), out)
    return out

# ---------------------------------------------------------------- kinematics
def tmax_gev(p, beta, m):
    g = 1. / np.sqrt(np.clip(1. - beta * beta, 1e-30, None))
    r = ME / np.clip(m, 1e-6, None)
    return 2. * ME * (beta * g) ** 2 / (1. + 2. * g * r + r * r)

def xi_mev(effZ, effA, xg, beta):
    # effA >= 1: a zero/absurd A from a malformed record would otherwise make
    # Z/A blow up and drive the carve straight into its 0.5 clip.
    return HALF_K * (effZ / np.clip(effA, 1., None)) * xg / np.clip(beta * beta, 1e-9, None)

# ---------------------------------------------------------------- the block
def _tmx(steps, tcut_gev, tmax_cap):
    bt = np.clip(steps[:, 4], 1e-9, 1. - 1e-12)
    g = 1. / np.sqrt(1. - bt * bt)
    m = np.clip(steps[:, 3] / (bt * g), 1e-6, None)
    t = tmax_gev(steps[:, 3], bt, m)
    return np.minimum(t, tmax_cap) if tmax_cap else t


def delta_step_exponent(steps, wstd, tau, tcut_gev=0.35e-3, spin=True,
                        tmax_cap=0.05):
    """log-CF exponent of the discrete delta-ray recoil for one pooled MS block,
    in standardized-z units. `steps` is the msmoliv record
    [effZ, effA, xg, pGeV, beta, thp2, dOverX0, stepGroup]; `wstd` the same
    per-block standardized weight ms_step_exponent uses."""
    Z, A, xg, p, bt = (steps[:, 0], steps[:, 1], steps[:, 2],
                       steps[:, 3], np.clip(steps[:, 4], 1e-9, 1. - 1e-15))
    tmx = _tmx(steps, tcut_gev, tmax_cap)
    good = (xg > 0.) & (tmx > tcut_gev) & (p > 0.)
    if not good.any():
        return np.zeros(len(tau))
    Z, A, xg, p, bt, tmx = Z[good], A[good], xg[good], p[good], bt[good], tmx[good]
    xi = xi_mev(Z, A, xg, bt) * 1e-3                   # -> GeV, so T in GeV throughout
    a = (wstd * np.sqrt(2. * ME) / p)[:, None] * tau[None, :]
    lo = a * np.sqrt(tcut_gev)
    hi = a * np.sqrt(tmx)[:, None]
    S = xi[:, None] * a ** 2 * (phi_tab(lo) - phi_tab(hi))
    if spin:
        # the (1 - beta^2 T/Tmax) spin-0 term of the PDG delta spectrum, applied
        # as a first-order variance correction (the 1/T^2 weight makes the
        # T~Tmax region where it matters a small share)
        S *= 1. - 0.5 * (bt * bt)[:, None] / np.log(np.clip(tmx / tcut_gev, 1.0001, None))[:, None]
    return S.sum(axis=0)

def delta_variance(steps, tcut_gev=0.35e-3, tmax_cap=0.05):
    """Var(theta_proj) of the discrete block, summed over steps -- the closed
    form  xi * m_e * ln(Tmax/Tcut) / p^2 , for validating the CF against."""
    Z, A, xg, p, bt = (steps[:, 0], steps[:, 1], steps[:, 2],
                       steps[:, 3], np.clip(steps[:, 4], 1e-9, 1. - 1e-15))
    tmx = _tmx(steps, tcut_gev, tmax_cap)
    good = (xg > 0.) & (tmx > tcut_gev) & (p > 0.)
    if not good.any():
        return 0.
    xi = xi_mev(Z[good], A[good], xg[good], bt[good]) * 1e-3
    return float(np.sum(xi * ME * np.log(tmx[good] / tcut_gev) / p[good] ** 2))


def carve_factor(steps, tcut_gev=0.35e-3, tmax_cap=0.05):
    """Fraction of the pooled MS variance that the discrete block REPLACES.

    Moliere/Highland already carries electron scattering continuously via the
    Z(Z+1) in X0, so the discrete block must be carved out, not added.  We carve
    by matching variance -- scale the MS exponent by (1 - carve_factor) -- which
    preserves the block's total variance EXACTLY and moves only the shape from
    a Gaussian core to a heavy discrete tail.  That is the conservative choice:
    it cannot change any variance-level result, only the tails."""
    vd = delta_variance(steps, tcut_gev, tmax_cap)
    vms = float(np.sum(steps[:, 5]))
    if vms <= 0.:
        return 0.
    return float(np.clip(vd / vms, 0., 0.5))
