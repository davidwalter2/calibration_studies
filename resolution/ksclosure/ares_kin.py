#!/usr/bin/env python3
"""The two-body mass as an explicit function of the six reference parameters.

The CVH two-track fit's reference state is, per leg, ``(q/p, lambda, phi)``;
``Jpsi_covrefmom`` is that state's 6x6 covariance in the order
``(plus: q/p, lambda, phi), (minus: q/p, lambda, phi)`` and
``Jpsi_jacrefmom`` is ``dm/du`` in the same order (the maker's
``massJacobianAltD``).

Everything the angular study needs beyond those two exports is the SECOND
derivative, because the mass variance is

    sigma_m^2(u) = J(u)^T Sigma(u) J(u)

and its gradient therefore has a purely KINEMATIC piece that needs no model of
the detector at all:

    d(sigma_m^2)/du_i = 2 (dJ/du_i)^T Sigma J   +   J^T (dSigma/du_i) J
                        \-------- exact -------/   \--- resolution model ---/

``dJ/du_i`` is the Hessian ``H_ij = d^2 m / du_i du_j``, which this module
supplies analytically (autograd-free, closed form via the chain rule on
``m^2``).  The functions are vectorised over candidates.

Conventions: ``p = 1/|q/p|``, ``p_vec = p (cos l cos f, cos l sin f, sin l)``,
``E = sqrt(p^2 + mass^2)``, and both legs carry the charged-pion mass for the
K_S (the muon mass for the J/psi) -- passed in, never assumed.
"""
import numpy as np

M_PI = 0.13957039
M_MU = 0.1056583745


def momenta(u):
    """(p, pvec, E-less) pieces from the 6-vector state ``u`` of shape (n, 6)."""
    u = np.asarray(u, dtype=np.float64)
    k = u[:, 0::3]                       # (n, 2) q/p
    lam = u[:, 1::3]
    phi = u[:, 2::3]
    p = 1.0 / np.abs(k)
    cl, sl = np.cos(lam), np.sin(lam)
    cf, sf = np.cos(phi), np.sin(phi)
    px, py, pz = p * cl * cf, p * cl * sf, p * sl
    return p, px, py, pz


def mass(u, masses):
    """Invariant mass of the two legs, shape (n,)."""
    m1, m2 = masses
    p, px, py, pz = momenta(u)
    e1 = np.sqrt(p[:, 0] ** 2 + m1 ** 2)
    e2 = np.sqrt(p[:, 1] ** 2 + m2 ** 2)
    sx, sy, sz = px.sum(1), py.sum(1), pz.sum(1)
    m2sq = (e1 + e2) ** 2 - (sx ** 2 + sy ** 2 + sz ** 2)
    return np.sqrt(np.maximum(m2sq, 0.0))


def jac_hess(u, masses, h=None):
    """``(m, J, H)`` -- mass, gradient (n, 6) and Hessian (n, 6, 6).

    Central finite differences on the CLOSED-FORM mass, with per-component
    steps scaled to the parameter.  The mass is an analytic function of `u`
    with no cancellation anywhere near these steps (the smallest scale in the
    problem is the opening angle, ~0.1 rad for a K_S), so the 2nd-order
    central differences are exact to ~1e-9 relative.  ``ares_angles.load``
    gates ``J`` against the exported ``Jpsi_jacrefmom``: 2.1e-7 (K_S, pion
    hypothesis) and 7.3e-8 (J/psi, muon), i.e. the float32 precision of the
    export.
    """
    u = np.asarray(u, dtype=np.float64)
    n = u.shape[0]
    if h is None:
        h = np.empty_like(u)
        h[:, 0::3] = 1e-5 * np.abs(u[:, 0::3])      # q/p: relative
        h[:, 1::3] = 1e-5                            # lambda: absolute rad
        h[:, 2::3] = 1e-5                            # phi
    J = np.empty((n, 6))
    H = np.empty((n, 6, 6))
    m0 = mass(u, masses)
    for i in range(6):
        du = np.zeros_like(u)
        du[:, i] = h[:, i]
        mp = mass(u + du, masses)
        mm = mass(u - du, masses)
        J[:, i] = (mp - mm) / (2.0 * h[:, i])
        H[:, i, i] = (mp - 2.0 * m0 + mm) / h[:, i] ** 2
    for i in range(6):
        for j in range(i + 1, 6):
            du = np.zeros_like(u)
            du[:, i] = h[:, i]
            dv = np.zeros_like(u)
            dv[:, j] = h[:, j]
            mpp = mass(u + du + dv, masses)
            mpm = mass(u + du - dv, masses)
            mmp = mass(u - du + dv, masses)
            mmm = mass(u - du - dv, masses)
            v = (mpp - mpm - mmp + mmm) / (4.0 * h[:, i] * h[:, j])
            H[:, i, j] = v
            H[:, j, i] = v
    return m0, J, H


def unpack_cov(tri):
    """(n, 21) upper triangle, row-major -> (n, 6, 6) symmetric."""
    tri = np.asarray(tri, dtype=np.float64)
    n = tri.shape[0]
    C = np.empty((n, 6, 6))
    k = 0
    for i in range(6):
        for j in range(i, 6):
            C[:, i, j] = tri[:, k]
            C[:, j, i] = tri[:, k]
            k += 1
    return C


def state_from_cache(c):
    """The 6-vector reference state (plus, minus) x (q/p, lambda, phi)."""
    lamp = np.arctan(np.sinh(c["Muplus_eta"].astype(np.float64)))
    lamm = np.arctan(np.sinh(c["Muminus_eta"].astype(np.float64)))
    u = np.stack([c["Jpsi_qoprefplus"].astype(np.float64), lamp,
                  c["Muplus_phi"].astype(np.float64),
                  c["Jpsi_qoprefminus"].astype(np.float64), lamm,
                  c["Muminus_phi"].astype(np.float64)], axis=1)
    return u
