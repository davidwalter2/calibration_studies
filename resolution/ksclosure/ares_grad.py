#!/usr/bin/env python3
"""The gradient of the mass resolution with respect to the fitted state.

The sigma-artefact coefficient of ``MASSCFTERM_SPEC`` is a regression slope,

    a = Cov(dm, dsigma) / Var(dm) = (J^T Sigma G) / (J^T Sigma J) ,
    J = grad_u m ,   G = grad_u sigma_m ,

over the six reference parameters ``u = (plus, minus) x (q/p, lambda, phi)``.
Both ``J`` (``Jpsi_jacrefmom``) and ``Sigma`` (``Jpsi_covrefmom``) are exported
per candidate; ``G`` is not, and this module builds it.

``sigma_m^2(u) = J(u)^T Sigma(u) J(u)`` gives it two pieces:

    d(sigma_m^2)/du_i = 2 (dJ/du_i)^T Sigma J  +  J^T (dSigma/du_i) J
                        \------ KINEMATIC -----/  \----- RESOLUTION -----/

The kinematic piece is exact -- ``dJ/du`` is the mass Hessian, ``ares_kin``.
It is NOT the physical answer on its own: holding ``Sigma`` fixed holds
``sigma_(q/p)`` fixed, which is ``sigma_p/p ~ p``, i.e. the pure HIT limit.
The resolution piece is what carries the track back to the actual mix of hit,
multiple scattering and ionisation, and it is where the ANGULAR dependence of
the covariance lives -- the path length in material goes as ``1/cos lambda``,
so an angular fluctuation moves ``Sigma`` as well as ``m``.

MODEL FOR ``dSigma/du``.  Write ``Sigma_ab = s_a s_b rho_ab`` with
``s_a = sqrt(Sigma_aa)``.  The correlations ``rho_ab`` are dimensionless shape
quantities; taking them locally constant leaves

    dSigma_ab/du_i = (g^i_a + g^i_b) Sigma_ab ,  g^i_a = d ln s_a / du_i

so that, with ``w_a = J_a (Sigma J)_a`` the component's share of the mass
variance (``sum_a w_a = sigma_m^2``),

    R_i = J^T (dSigma/du_i) J = 2 sum_a g^i_a w_a .

Only the leg that owns ``u_i`` responds (the cross-leg block is a second-order
effect and its correlation is 0.003 here), so ``g^i_a`` is non-zero for
``a`` in the same leg as ``i``.

The six log-derivatives per leg, ``d ln Sigma_aa / d ln p_l`` and
``d ln Sigma_aa / d lambda_l`` for ``a`` in ``(q/p, lambda, phi)``, come two
ways:

* ``population`` -- a LOCAL LINEAR regression of ``ln Sigma_aa`` on
  ``(ln p_l, lambda_l)`` inside cells of the same two variables, with the
  detector controls (valid hits, the decay vertex) partialled out linearly.
  This is a derivative of a SMOOTH function of the fitted state, so it is
  measured on the FITTED kinematics: it is not a residual, and the rule that
  a residual must be conditioned on truth does not apply to it.
* ``msmodel`` -- the first-principles path-length scaling: material
  ``~ 1/cos lambda`` makes every multiple-scattering and ionisation variance
  go as ``1/cos lambda``, hence ``d ln Sigma_aa/d lambda = tan lambda``, and
  ``sigma_p/p`` flat in ``p`` makes ``d ln Sigma_(q/p),(q/p)/d ln p = -2``,
  with the angular variances ``~1/p^2`` giving the same ``-2``.

The population form includes what the analytic one cannot -- the lever arm and
layer count changing with ``lambda``, and the incidence-angle dependence of the
hit errors.  It also includes the DISCRETE layer transitions, which a
1e-3 rad fluctuation does not actually cross; that is the known bias of the
population route and the reason both are quoted.

The ``lambda`` response of ``Sigma`` turns out to be worth 0.3 % of the slope
at the K_S and 0.07 % at the J/psi (``STATE.md`` section 9.5): the angular
sector matters, but through the mass Hessian, not through this term.
"""
import numpy as np


def leg_table(u, C, aux):
    """Flatten the candidates into 2n legs for the population regression."""
    n = u.shape[0]
    out = {}
    out["lnp"] = np.concatenate([np.log(1.0 / np.abs(u[:, 0])),
                                 np.log(1.0 / np.abs(u[:, 3]))])
    out["lam"] = np.concatenate([u[:, 1], u[:, 4]])
    out["dphi"] = np.concatenate([u[:, 2] - aux["phivtx"], u[:, 5] - aux["phivtx"]])
    out["dphi"] = (out["dphi"] + np.pi) % (2 * np.pi) - np.pi
    out["nv"] = np.concatenate([aux["nvp"], aux["nvm"]]).astype(np.float64)
    out["lnr"] = np.concatenate([np.log(aux["rvtx"] + 0.5)] * 2)
    out["zv"] = np.concatenate([aux["zvtx"]] * 2)
    for k, a in enumerate(("qop", "lam", "phi")):
        out["lnS_" + a] = np.concatenate([np.log(C[:, k, k]),
                                          np.log(C[:, 3 + k, 3 + k])])
    out["cand"] = np.concatenate([np.arange(n), np.arange(n)])
    out["leg"] = np.concatenate([np.zeros(n, int), np.ones(n, int)])
    return out


def local_linear(y, x1, x2, ctrl, nb1=8, nb2=8, minn=200, extra=None):
    """Per-row partial derivatives ``dy/dx1``, ``dy/dx2`` by local linear fit.

    Cells are quantile bins of ``(x1, x2)``; inside each cell the design is
    ``[1, x1, x2, *ctrl]`` (all centred), so the controls are partialled out
    linearly and the two coefficients of interest are local slopes.  Returns
    ``(d1, d2, info)`` with one value per row, taken from the row's own cell.
    """
    n = len(y)
    q1 = np.quantile(x1, np.linspace(0, 1, nb1 + 1))
    q2 = np.quantile(x2, np.linspace(0, 1, nb2 + 1))
    q1[0], q1[-1] = -np.inf, np.inf
    q2[0], q2[-1] = -np.inf, np.inf
    i1 = np.clip(np.searchsorted(q1, x1, "right") - 1, 0, nb1 - 1)
    i2 = np.clip(np.searchsorted(q2, x2, "right") - 1, 0, nb2 - 1)
    d1 = np.full(n, np.nan)
    d2 = np.full(n, np.nan)
    info = []
    for a in range(nb1):
        for b in range(nb2):
            sel = (i1 == a) & (i2 == b)
            ns = int(sel.sum())
            if ns < minn:
                continue
            cols = [np.ones(ns), x1[sel] - x1[sel].mean(), x2[sel] - x2[sel].mean()]
            for cvec in ctrl:
                cols.append(cvec[sel] - cvec[sel].mean())
            if extra is not None:
                for cvec in extra:
                    cols.append(cvec[sel] - cvec[sel].mean())
            X = np.stack(cols, axis=1)
            beta, *_ = np.linalg.lstsq(X, y[sel], rcond=None)
            d1[sel] = beta[1]
            d2[sel] = beta[2]
            info.append((a, b, ns, float(np.median(x1[sel])), float(np.median(x2[sel])),
                         float(beta[1]), float(beta[2]),
                         float(beta[3 + len(ctrl)]) if extra is not None else np.nan))
    return d1, d2, info


def grad_sigma(u, C, J, H, glnp, glam, sig):
    """``grad_u sigma_m`` from the exact kinematic term and the log-derivatives.

    ``glnp``/``glam`` are ``(n, 6)`` arrays of ``d ln Sigma_aa / d ln p_l`` and
    ``d ln Sigma_aa / d lambda_l`` with ``l`` the component's own leg.
    Returns ``(grad, Kvec, Rvec)`` -- the gradient of ``sigma_m`` and the two
    contributions to the gradient of ``sigma_m^2``.
    """
    SJ = np.einsum("nij,nj->ni", C, J)
    Kvec = 2.0 * np.einsum("nij,nj->ni", H, SJ)
    w = J * SJ                                     # (n, 6) variance shares
    Rvec = np.zeros_like(Kvec)
    # d/d(q/p)_l: the chain rule from ln p, p = 1/|q/p| -> d ln p/d(q/p) = -1/(q/p)
    for l in (0, 1):
        comp = slice(3 * l, 3 * l + 3)
        rl = (glnp[:, comp] * w[:, comp]).sum(1)   # sum_a (dlnS_aa/dlnp_l) w_a
        Rvec[:, 3 * l] += 2.0 * 0.5 * rl * (-1.0 / u[:, 3 * l])
        ra = (glam[:, comp] * w[:, comp]).sum(1)
        Rvec[:, 3 * l + 1] += 2.0 * 0.5 * ra
    grad = (Kvec + Rvec) / (2.0 * sig[:, None])
    return grad, Kvec, Rvec


def slope(J, C, grad, sig):
    """``a = J^T Sigma grad(sigma) / sigma^2``."""
    SJ = np.einsum("nij,nj->ni", C, J)
    return np.einsum("ni,ni->n", SJ, grad) / sig ** 2


def ms_model(u, fhit_qop=0.0):
    """The analytic path-length log-derivatives, ``(glnp, glam)`` of (n, 6).

    Pure multiple scattering/ionisation: every variance scales as the material
    ``1/cos lambda`` and as ``1/p^2``.  ``fhit_qop`` optionally gives the
    ``q/p`` variance a hit component, which is ``p``-flat in ``sigma_(q/p)``
    and carries ``-tan lambda`` from ``sigma_(q/p),hit = cos lambda
    sigma(1/p_T)``.
    """
    n = u.shape[0]
    glnp = np.full((n, 6), -2.0)
    glam = np.empty((n, 6))
    for l in (0, 1):
        t = np.tan(u[:, 3 * l + 1])
        glam[:, 3 * l:3 * l + 3] = t[:, None]
    if fhit_qop:
        h = np.asarray(fhit_qop, dtype=np.float64)
        if h.ndim == 0:
            h = np.full(n, float(h))
        for l in (0, 1):
            t = np.tan(u[:, 3 * l + 1])
            glnp[:, 3 * l] = -2.0 * (1.0 - h)
            glam[:, 3 * l] = h * (-2.0 * t) + (1.0 - h) * t
    return glnp, glam
