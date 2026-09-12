#!/usr/bin/env python3
"""Bases for compressing the per-candidate CF exponents.

Two competing representations, both of the form  S_f(t) ~ sum_k a_ik B_kf(t)
with a UNIVERSAL (candidate-independent) basis B and k scalars per candidate:

* :func:`pca_fit` -- the empirical optimum: SVD of the concatenated data
  matrix.  A JOINT fit over all families gives ONE coefficient vector per
  candidate that still reconstructs each family separately, so every k_f can
  still float independently in the likelihood.

* the PHYSICS dictionaries below -- fixed, model-derived, no data pass:

  ``levy_design``   S_io/S_rad are compound-Poisson exponents,
                    S(t) = sum_s a_s (e^{i v_s t} - 1 - i v_s t)
                    (`cf_ioni_exact.block_exponent`, `cf_brems_exact`), so a
                    quadrature of the Levy measure on fixed nodes v_j
                    reproduces BOTH the real and imaginary part from the SAME
                    k non-negative scalars.
  ``moliere_design``  S_ms(t) = sum_s (chic2/chia2)_s G(b_s t; ymax_s) with
                    G = `cf_ms_exact.gshape`, so the dictionary is dilations
                    of that one tabulated universal function.
  ``poly_design``   the cumulant/Taylor baseline, S ~ sum_p kappa_p (it)^p/p!.
"""
import os

import numpy as np

GTAB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "results", "gshape_tab.npz")
_G = None


# ---------------------------------------------------------------- PCA -----
def pca_fit(X, rmax):
    """Centred PCA via the (m x m) Gram matrix.  X: (n, m) float64."""
    mu = X.mean(axis=0)
    Xc = X - mu
    G = Xc.T @ Xc
    ev, V = np.linalg.eigh(G)
    o = np.argsort(ev)[::-1]
    return mu, V[:, o[:rmax]], np.sqrt(np.clip(ev[o], 0, None))


def pca_apply(X, mu, V, r):
    """Rank-r reconstruction of X in the (mu, V) basis."""
    Xc = X - mu
    return mu + (Xc @ V[:, :r]) @ V[:, :r].T


# ------------------------------------------------------- physics bases ----
def gshape(tau, ymax=None):
    """Copy of `cf_ms_exact.gshape` reading the dumped tables (importing the
    module costs ~60 s of table building)."""
    global _G
    if _G is None:
        _G = dict(np.load(GTAB))
    GTAU, G, G2D, YMAXG = _G["GTAU"], _G["G"], _G["G2D"], _G["YMAXG"]
    tau = np.abs(np.asarray(tau, dtype=np.float64))
    if ymax is None:
        row = G
    else:
        ly = np.clip(np.log(ymax), np.log(YMAXG[0]), np.log(YMAXG[-1]))
        fi = (ly - np.log(YMAXG[0])) / (np.log(YMAXG[1]) - np.log(YMAXG[0]))
        i0 = int(np.clip(np.floor(fi), 0, len(YMAXG) - 2))
        f = fi - i0
        row = (1. - f) * G2D[i0] + f * G2D[i0 + 1]
    out = np.interp(tau, GTAU, row, left=np.nan, right=row[-1])
    tiny = tau < GTAU[0]
    if np.any(tiny):
        out[tiny] = row[0] * (tau[tiny] / GTAU[0]) ** 2
    return out


def levy_design(TG, v):
    """[Re | Im] design of the compound-Poisson dictionary on nodes v (signed).

    Returns (2*nt, K): rows 0:nt are  cos(v t) - 1,  rows nt: are
    sin(v t) - v t   -- i.e. the real and imaginary parts of
    e^{i v t} - 1 - i v t, which is EXACTLY the per-step term of
    `cf_ioni_exact.block_exponent` and of the radiative channel.
    """
    v = np.atleast_1d(np.asarray(v, np.float64))
    th = np.outer(TG, v)                       # (nt, K)
    return np.concatenate([np.cos(th) - 1.0, np.sin(th) - th], axis=0)


def moliere_design(TG, b, ymax=1e4):
    """(nt, K) dictionary of dilated Moliere shapes G(b_j t; ymax)."""
    b = np.atleast_1d(np.asarray(b, np.float64))
    return gshape(np.outer(TG, b).ravel(), ymax=ymax).reshape(len(TG), len(b))


def poly_design(TG, kre, kim=0):
    """Cumulant/Taylor baseline.  kre even powers t^2, t^4, ... (real part),
    kim odd powers t^3, t^5, ... (imaginary part)."""
    cols = [TG ** (2 * (j + 1)) for j in range(kre)]
    if kim:
        cols += [TG ** (2 * j + 3) for j in range(kim)]
    return np.stack(cols, axis=1) if cols else np.zeros((len(TG), 0))


def ls_coeffs(Y, D, ridge=0.0):
    """Least squares Y ~ C D^T for every row of Y at once.  Y: (n, m),
    D: (m, K).  Returns C (n, K)."""
    G = D.T @ D
    if ridge:
        G = G + ridge * np.trace(G) / len(G) * np.eye(len(G))
    return np.linalg.solve(G, (Y @ D).T).T
