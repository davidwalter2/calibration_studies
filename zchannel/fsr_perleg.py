#!/usr/bin/env python3
"""Per-leg factorised QED FSR kernel and the selection-conditional mass shape.

The inclusive kernel of `fsr_analytic` is a density in the **pair** variable
``z = (m_post/m_pre)^2``.  A lepton ``p_T`` cut does not act on ``z``: it acts on
each muon separately, so the kernel has to be resolved leg by leg first.

**The collinear factorisation.**  In the quasi-collinear limit each muon keeps
its direction and its post-FSR four-momentum is ``p' = x p``.  In the massless
limit that is exactly Lorentz covariant, so ``x = p_T'/p_T = E'/E`` in *any*
frame and ``eta' = eta``, ``phi' = phi``.  The pair mass then factorises,

    m'^2 = (x_+ p_+ + x_- p_-)^2 = x_+ x_- m^2 ,   i.e.   z = x_+ x_- ,

so ``u = -ln(m'/m) = (u_+ + u_-)/2`` with ``u_q = -ln x_q``, and in Mellin space
the inclusive kernel is the **square** of the single-leg time-like QED structure
function ``D(x; L)``:  ``K~(n) = D~(n)^2``.

**The single-leg radiator.**  `LegRadiator` is `fsr_analytic.FSRKernel` with the
Mellin exponent halved.  With ``a = alpha/pi``, ``L = ln(m^2/m_mu^2)`` and the
inclusive exponent

    E(n) = a (L g0 + c1) + a^2 (L/2) g1 ,          K~ = exp E ,

the leg is ``D~ = exp(E/2)``, which the construction of `FSRKernel` realises by

  * ``beta_D = beta/2 = a (L - 1)``      (the single-leg soft exponent, the
    ``beta_e/2`` of Kuraev-Fadin), used both in ``C beta_D (1-x)^{beta_D-1}``
    and in the O(alpha^2) LL term ``(beta_D^2/8) [P (x) P]_reg`` ;
  * ``h_D(x) = -(beta_D/2)(1+x) + (a/2) (1+x^2) ln x/(1-x)`` -- exactly half the
    inclusive ``h``, as ``D_1 = K_1/2`` requires at O(alpha) ;
  * the O(alpha^2 L) remainder ``(a^2 L/4) [G_D(x) + g_delta,D delta(1-x)]`` with

        G_D      = P1T + A - kappa (1+x)  =  G + P1T/2
        g_delta,D = P1_delta + (3/2) kappa

    (``G``, ``A``, ``P1T``, ``kappa`` as in `fsr_analytic`).  The extra
    ``P1T/2`` relative to half the inclusive ``G`` is not a choice: the generic
    construction with coupling ``a_hat`` has exponent
    ``a_hat (L g0 + c1) + a_hat^2 (L/2) g1``, so ``a_hat = a/2`` undershoots the
    wanted ``(a^2 L/4) g1`` by ``(a^2 L/8) g1``, i.e. by ``(a^2 L/8) P1T``.
    ``int G_D dx + g_delta,D = 0`` exactly, as for the inclusive term, because
    ``int P1T = -P1_delta``, ``int A = 0`` and ``int kappa(1+x) = (3/2) kappa``:
    the NLL piece leaves ``int D dx = 1`` alone with no numerical rescale.

  * **pairs** are emitted from one leg, so per leg the pair factor is
    ``(1 - N/2) delta(1-x) + R_pair(x)/2`` with the *same* ``R_pair`` the
    inclusive kernel uses.  Its square is
    ``(1-N) delta + R_pair + O(alpha^4)``, the inclusive pair factor.

``D (x) D`` then reproduces ``K`` through O(alpha^2) by construction; what is
left is the O(alpha^3) difference between the two-leg soft resummations
``[beta_D (1-x)^{beta_D-1}]^{(x)2}`` and ``beta (1-z)^{beta-1}``, measured by
``fsr_perleg.py check``.

**The selection.**  With ``|eta| < eta_cut`` FSR-independent in the collinear
limit, the post-FSR ``p_T > p_T^cut`` condition on a pre-FSR configuration
``(p_T+, p_T-)`` is ``x_q > a_q = p_T^cut/p_T q``, i.e. ``u_q < b_q = -ln a_q``.
The selected density of the pair variable is therefore the convolution of the
two **truncated** leg radiators, and integrating over the pre-FSR kinematics at
fixed ``m``,

    K_sel(u | m) = int du_+ du_- D_u(u_+) D_u(u_-) G(u_+, u_- | m)
                                                  delta(u - (u_+ + u_-)/2)
    A(m)         = int du_+ du_- D_u(u_+) D_u(u_-) G(u_+, u_- | m)

with **G(u_+, u_- | m) = P(p_T+ > p_T^cut e^{u_+}, p_T- > p_T^cut e^{u_-},
|eta_q| < eta_cut | m)** the joint survival function of the *pre-FSR* muon
transverse momenta at exponentially scaled thresholds.  ``G`` is the whole
boson-kinematics input -- production (``p_T^Z``, ``y_Z``), the decay angles with
their angular coefficients, and the PDFs all sit inside it and nothing else
does.  It is the 2-D reverse cumulative of the table ``h(a_+, a_- | m)``, which
`htable` measures from a generator record and which SCETLib + DYTurbo can
supply instead without touching any of the QED above.

**The effective leg of a tabulated kernel.**  ``D`` is *defined* by
``D (x) D = K``, so a kernel that is a table rather than a closed form -- the
``mc`` configuration of `fsr_config`, i.e. standalone Photos -- has a per-leg
radiator too, the numerical convolution square root (`legsqrt`, `conv_sqrt`).
Feeding it to `kernel --mc-leg` gives a per-leg model whose QED is exactly the
MC's, which is what separates the collinear factorisation from the kernel
physics.  `condker` completes the separation by reading the MC's *own*
conditional kernel in the model's mass and selection variables.
"""

import argparse
import json
import math
import os
import time

import numpy as np

from fsr_analytic import (ALPHA, A_PI, KAPPA, M_MU, P1_DELTA, FSRKernel,
                          PairTerm, band_edges, beta_fsr, coll_log,
                          conv_p0_pln, p1_timelike, _merge_cells)

#: delta(1-x) coefficient of the leg's O(alpha^2 L) remainder,
#: g_delta,D = P1_delta + (3/2) kappa = -int_0^1 G_D dx
G_DELTA_LEG = P1_DELTA + 1.5 * KAPPA


def nll_reg_leg(x_, v=None):
    """``G_D(x) = P1T(x) + A(x) - kappa (1+x)``, the leg's O(alpha^2 L) kernel."""
    x_ = np.asarray(x_, float)
    v = (1.0 - x_) if v is None else np.asarray(v, float)
    return p1_timelike(x_, v) + conv_p0_pln(x_, v) - KAPPA * (1.0 + x_)


class _HalfPair:
    """``R_pair/2``: the pair radiator seen by **one** leg."""

    def __init__(self, pt):
        self._pt = pt
        self.B = None if pt.B is None else 0.5 * pt.B
        self._u = getattr(pt, "_u", None)
        self._lnu = getattr(pt, "_lnu", None)
        self.species = pt.species

    def __call__(self, z, x=None):
        return 0.5 * self._pt(z, x)


class LegRadiator(FSRKernel):
    """``D(x; m)``: the single-leg time-like QED structure function.

    Same construction as `FSRKernel` with the Mellin exponent halved; see the
    module docstring.  ``x`` is the muon's energy fraction, ``u = -ln x``.
    """

    def __init__(self, m, variant="exp1", pair=(), minus_one=True,
                 x_cut=1e-7, zmin=None, beta_at=None, pair_table=None):
        self.m = float(m)
        self.variant = variant
        if variant not in ("exp1", "exp2", "exp2nll", "oalpha", "born"):
            raise ValueError(f"unknown variant {variant}")
        mb = self.m if beta_at is None else float(beta_at)
        self.beta_at = beta_at
        self.L = float(coll_log(mb))
        #: the pair exponent, kept for reference
        self.beta_pair = float(beta_fsr(mb, minus_one))
        #: the single-leg soft exponent beta_D = beta/2 = a (L-1)
        self.beta = self.beta_gam = 0.5 * self.beta_pair
        self.pair = tuple(pair)
        self._pair = _HalfPair(PairTerm(mb, pair, path=pair_table))
        self.x_cut = float(x_cut)
        #: support floor ``x_min = 2 m_mu/m``: the muon cannot keep less than
        #: its own mass, and ``x_min^2 = 4 m_mu^2/m^2`` is exactly the pair's
        #: own threshold, so ``D (x) D`` has support ``z >= zmin`` with nothing
        #: to trim.
        self.zmin = (float(zmin) if zmin is not None
                     else 2.0 * M_MU / self.m)
        self.pair_rate = self.int_pair()
        self._c_nll = 0.25 * A_PI**2 * self.L if variant == "exp2nll" else 0.0
        self._C = (1.0 - self.int_h()
                   - (self.int_q2() if variant in ("exp2", "exp2nll") else 0.0)
                   - self.int_nll())

    # -- the O(alpha) pieces, exactly half the inclusive ones --------------
    def r1(self, z, x=None):
        z = np.asarray(z, float)
        x = (1.0 - z) if x is None else np.asarray(x, float)
        return 0.5 * A_PI * (1.0 + z * z) / x * (self.L - 1.0 + np.log1p(-x))

    def h(self, z, x=None):
        z = np.asarray(z, float)
        x = (1.0 - z) if x is None else np.asarray(x, float)
        return (-0.5 * self.beta * (1.0 + z)
                + 0.5 * A_PI * (1.0 + z * z) * np.log1p(-x) / x)

    def int_h(self):
        return -0.75 * self.beta + 0.5 * A_PI * (1.25 - math.pi**2 / 3.0)

    # -- the leg variable is u = -ln x, not the pair's -(1/2) ln z ---------
    # `FSRKernel` measures the mass loss of the *pair*, u = -(1/2) ln z; the leg
    # carries u = -ln x of its own energy fraction, and z = x_+ x_- makes the
    # two coincide only in the mean.  Everything that converts between the
    # quadrature variable t = (1-x)^{beta_D} and u is therefore overridden.
    def _t_of_u(self, u):
        return np.power(-np.expm1(-np.asarray(u, float)), self.beta)

    def _u_of_t(self, t):
        return -np.log1p(-np.power(np.asarray(t, float), 1.0 / self.beta))

    def _u_max(self):
        return -math.log(self.zmin)

    def _panel(self, t_lo, t_hi, ng, clip=True):
        u, x, w = super()._panel(t_lo, t_hi, ng, clip)
        return 2.0 * u, x, w

    def pdf_u(self, u):
        """``D_u(u) = x D(x)``, the density in ``u = -ln x``."""
        u = np.asarray(u, float)
        x = -np.expm1(-u)
        return (1.0 - x) * self.pdf_z(1.0 - x, x)

    def pair_cells(self, n=400, ng=8, u_max=None, with_x=False):
        """``(w, int u R/2, int u^2 R/2)`` in the **leg** variable ``u = -ln x``.

        The pair table is tabulated against the pair's ``-(1/2) ln z``; a pair
        radiated off one leg has ``z = x``, so the leg's ``u`` is twice it.
        """
        if self._pair.B is None:
            return np.zeros((4 if with_x else 3, 0))
        u_max = self._u_max() if u_max is None else float(u_max)
        e = np.geomspace(2.0 * self._pair._u[0],
                         min(2.0 * self._pair._u[-1], u_max), n + 1)
        g, wg = np.polynomial.legendre.leggauss(ng)
        a, b = np.log(e[:-1])[:, None], np.log(e[1:])[:, None]
        lu = 0.5 * (b - a) * (g[None, :] + 1.0) + a
        w = 0.5 * (b - a) * wg[None, :]
        u = np.exp(lu)
        x = -np.expm1(-u)
        z = 1.0 - x
        f = w * u * z * self._pair(z, x)                  # R(x) dx -> f dlnu
        out = [f.sum(1), (f * u).sum(1), (f * u * u).sum(1)]
        if with_x:
            out.append((f * x).sum(1))
        return np.stack(out)

    def int_pair(self, n=4000):
        """``N/2 = int_0^1 R_pair(x)/2 dx``, the leg's real-pair rate."""
        if self._pair.B is None:
            return 0.0
        u = np.geomspace(2.0 * self._pair._u[0],
                         min(2.0 * self._pair._u[-1], self._u_max()), n)
        x = -np.expm1(-u)
        z = 1.0 - x
        f = z * self._pair(z, x) * u
        lu = np.log(u)
        return float(np.trapezoid(f, lu) if hasattr(np, "trapezoid")
                     else np.trapz(f, lu))

    # -- the O(alpha^2 L) remainder ---------------------------------------
    def nll(self, z, x=None):
        if self._c_nll == 0.0:
            return np.zeros_like(np.asarray(z, float))
        return self._c_nll * nll_reg_leg(z, x)

    def int_nll(self):
        return -self._c_nll * G_DELTA_LEG


# --------------------------------------------------------------------------
# Mellin moments
# --------------------------------------------------------------------------
def mellin(k, n, n_fine=4000, ng=16, npair=400, ngpair=8):
    """``int_0^1 y^{n-1} K(y) dy`` of a `FSRKernel` or a `LegRadiator`.

    The pair factor multiplies: the kernel is the convolution
    ``K_phot (x) [(1 - N) delta(1-y) + R(y)]`` (halved per leg), so its Mellin
    transform is ``M[K_phot] (1 - N + M[R])``.
    """
    n = np.atleast_1d(np.asarray(n, float))
    tg = k._fine_grid(None, n_fine)
    _, x, w = k._panel(tg[:-1], tg[1:], ng)
    y = 1.0 - x
    out = np.array([float(np.sum(w * np.power(y, kk - 1.0))) for kk in n])
    if k._pair.B is not None:
        e = np.geomspace(k._pair._u[0], min(k._pair._u[-1], k._u_max()),
                         npair + 1)
        g, wg = np.polynomial.legendre.leggauss(ngpair)
        a, b = np.log(e[:-1])[:, None], np.log(e[1:])[:, None]
        lu = 0.5 * (b - a) * (g[None, :] + 1.0) + a
        wq = 0.5 * (b - a) * wg[None, :]
        uu = np.exp(lu)
        z = np.exp(-2.0 * uu)
        xx = -np.expm1(-2.0 * uu)
        f = wq * uu * 2.0 * z * k._pair(z, xx)
        mp = np.array([float(np.sum(f * np.power(z, kk - 1.0))) for kk in n])
        out = out * (1.0 - k.pair_rate + mp)
    return out


# --------------------------------------------------------------------------
# the leg on a ladder in u = -ln x
# --------------------------------------------------------------------------
def leg_u_edges(D, n=2000, u_min=1e-9, u_max=None):
    """Cell edges of the leg ladder: ``0`` then geometric to ``u_max``.

    Geometric in ``u`` because the radiator is scale free there
    (``D_u -> C beta_D u^{beta_D-1}``), so a uniform ladder in ``ln u`` carries
    nearly equal weight per cell over the ten decades between the infrared end
    and the kinematic limit ``x_min = 2 m_mu/m``.
    """
    u_max = D._u_max() if u_max is None else float(u_max)
    return np.concatenate([[0.0], np.geomspace(u_min, u_max, n)])


def leg_cells(D, u_edges, ng=16, n_phot_pair=600, n_pair=300):
    """``(w, int u D_u, int u^2 D_u)`` per cell of ``u_edges``, pairs folded in.

    The photonic radiator is integrated cell by cell with the ``t = (1-x)^beta_D``
    quadrature of `fsr_analytic`; the pair factor
    ``(1 - N/2) delta(1-x) + R_pair(x)/2`` is convoluted onto the same ladder by
    re-binning the outer product of a coarse photonic discretisation with the
    pair cells -- its error is weighted by ``N/2 ~ 1.8e-3``, so the coarse grid
    is enough.
    """
    tg = D._t_of_u(u_edges)
    w, m1, m2, _ = D._cells(tg, ng)
    N = float(D.pair_rate)
    if D._pair.B is None or N <= 0.0:
        return np.stack([w, m1, m2])
    uk, wk = _merge_cells(D.pair_cells(n=n_pair, ng=8), 1e-30)
    # coarse photonic atoms for the pair branch
    tc = D._t_of_u(leg_u_edges(D, n=n_phot_pair, u_min=u_edges[1],
                               u_max=u_edges[-1]))
    cw, cm1, cm2, _ = D._cells(tc, ng)
    cu = np.where(cw > 0, cm1 / np.maximum(cw, 1e-300), 0.0)
    uu = (cu[:, None] + uk[None, :]).ravel()
    ww = (cw[:, None] * wk[None, :]).ravel()
    idx = np.clip(np.searchsorted(u_edges, uu, "right") - 1, 0, len(w) - 1)
    n = len(w)
    return np.stack([(1.0 - N) * w + np.bincount(idx, ww, n),
                     (1.0 - N) * m1 + np.bincount(idx, ww * uu, n),
                     (1.0 - N) * m2 + np.bincount(idx, ww * uu * uu, n)])


def leg_atoms(cells, u_edges):
    """Cell weights and their exact conditional means ``(u_p, w_p)``."""
    w, m1, _ = cells
    ok = w > 0.0
    u = np.zeros_like(w)
    u[ok] = m1[ok] / w[ok]
    return u[ok], w[ok]


# --------------------------------------------------------------------------
# D (x) D: the inclusive kernel rebuilt from the leg
# --------------------------------------------------------------------------
def conv_atoms(u, w, u_out_edges):
    """Bin ``(u_+ + u_-)/2`` of the outer product of a leg measure with itself."""
    n = len(u_out_edges) - 1
    s0 = np.zeros(n)
    s1 = np.zeros(n)
    s2 = np.zeros(n)
    for a in range(0, len(u), 512):
        up = u[a:a + 512]
        wp = w[a:a + 512]
        U = 0.5 * (up[:, None] + u[None, :]).ravel()
        W = (wp[:, None] * w[None, :]).ravel()
        i = np.clip(np.searchsorted(u_out_edges, U, "right") - 1, 0, n - 1)
        s0 += np.bincount(i, W, n)
        s1 += np.bincount(i, W * U, n)
        s2 += np.bincount(i, W * U * U, n)
    return np.stack([s0, s1, s2])


# --------------------------------------------------------------------------
# the numerical convolution square root: `D` from a *tabulated* kernel
# --------------------------------------------------------------------------
#: grid spacing of the square root, in the pair variable ``u``.  It is the
#: standalone Photos histograms' own resolution (`photos_gen.cc`'s ``U_FINE``):
#: one node per filled histogram bin, so the kernel is a gap-free positive
#: vector.  On a finer grid the tabulated kernel is a *comb* -- isolated atoms
#: with empty nodes between them -- and a comb is not infinitely divisible, so
#: its square root rings at the lattice scale (see `legsqrt --check-band`).
KSQRT_DU = 2e-5
#: the grid's extent in ``u``; the standalone's support ends at ``u`` ~ 6.
KSQRT_UMAX = 8.0


def kernel_atoms(s0, s1, w_norad, tail0=None, tail1=None):
    """``(u, w)`` of a tabulated kernel: one atom per filled bin, exact mean.

    ``w_norad`` is the weight of the events the generator left untouched; they
    are a genuine ``delta(u)``, and the first bin's *continuum* is what is left
    of it, at its own conditional mean (the untouched events contribute zero to
    ``s1``, so no modelling is involved).
    """
    w = np.asarray(s0, float).copy()
    s1 = np.asarray(s1, float)
    w[0] -= float(w_norad)
    u = np.zeros_like(w)
    ok = w > 0.0
    u[ok] = s1[ok] / w[ok]
    uu, ww = [np.zeros(1)], [np.array([float(w_norad)])]
    uu.append(u[ok])
    ww.append(w[ok])
    if tail0 is not None:
        t0 = np.asarray(tail0, float)
        t1 = np.asarray(tail1, float)
        ok = t0 > 0.0
        uu.append(t1[ok] / t0[ok])
        ww.append(t0[ok])
    u = np.concatenate(uu)
    w = np.concatenate(ww)
    return u, w / w.sum()


def deposit(u, w, du, n):
    """Atoms onto a uniform grid, **mass and mean preserved exactly**.

    Each atom is split linearly between its two neighbouring nodes.  The sum of
    two nodes is again a node, so the convolution of two deposited measures is
    the exact convolution of the node measures -- no positioning error enters
    the square root.
    """
    p = np.asarray(u, float) / du
    i = np.floor(p).astype(np.int64)
    t = p - i
    if i.min() < 0 or i.max() + 1 >= n:
        raise ValueError(f"atoms outside the grid: u_max = {np.max(u):.4g}")
    w = np.asarray(w, float)
    return (np.bincount(i, w * (1.0 - t), n)
            + np.bincount(i + 1, w * t, n)[:n])


def _fft_len(n):
    m = 1
    while m < n:
        m <<= 1
    return m


def conv_sqrt(g, method="spectral", tol=1e-15, maxit=200):
    """``D`` with ``D (x) D = g``, on the grid ``g`` lives on.

    ``g`` is a kernel with a ``delta`` at the origin (``g[0] > 0``), so the
    square root exists and is unique as a *signed* measure: with
    ``g = g0 e0 + g_c`` and ``D = d0 e0 + D_c``, ``D_c`` is supported on nodes
    ``>= 1`` and ``D_c (x) D_c`` on nodes ``>= 2``, so ``d0 = sqrt(g0)`` and

        D_c = (g_c - D_c (x) D_c) / (2 sqrt(g0))

    is a contraction with factor ``||D_c||/sqrt(g0)``.  ``spectral`` solves the
    same equation pointwise in Fourier space, where it is the scalar quadratic
    ``d^2 + 2 sqrt(g0) d - g_c = 0`` and the contracting root is
    ``d = -sqrt(g0) + sqrt(g_hat)``: the two are the same object, and the
    branch of the square root is the one the iteration selects.
    """
    n = _fft_len(2 * len(g))
    G = np.zeros(n)
    G[:len(g)] = g
    if method == "spectral":
        Gh = np.fft.rfft(G)
        D = np.fft.irfft(np.sqrt(Gh), n)
        info = dict(method="spectral", k_abs_min=float(np.abs(Gh).min()),
                    k_arg_max=float(np.abs(np.angle(Gh)).max()))
    elif method == "fixedpoint":
        s = math.sqrt(G[0])
        Kc = G.copy()
        Kc[0] -= G[0]
        Dc = Kc / (2.0 * s)
        for it in range(1, maxit + 1):
            new = (Kc - np.fft.irfft(np.fft.rfft(Dc) ** 2, n)) / (2.0 * s)
            step = float(np.abs(new - Dc).max())
            Dc = new
            if step < tol:
                break
        info = dict(method="fixedpoint", iterations=it, last_step=step,
                    contraction=float(np.abs(Dc).sum() / s))
        D = Dc
        D[0] += s
    else:
        raise ValueError(method)
    info["roundtrip"] = float(
        np.abs(np.fft.irfft(np.fft.rfft(D) ** 2, n) - G).max())
    info["neg_mass"] = float(D[D < 0.0].sum())
    info["d0"] = float(D[0])
    return D, info


def merge_positive(cells):
    """Merge neighbouring cells forward until every weight is positive.

    The square root is a *signed* measure on the grid: wherever the tabulated
    kernel is a comb rather than a dense histogram -- which above ``u`` = 2 it
    is, the standalone's tail block being binned at 0.05 against a 2e-5 grid --
    it rings at the lattice scale.  Merging is exact in the mass and in the
    first two moments, so it changes no integral of the leg against a function
    that is smooth on the merged cell; it only says where the leg is resolved.
    """
    w, m1, m2 = np.asarray(cells, float)
    out = []
    a = np.zeros(3)
    for k in range(len(w)):
        a += (w[k], m1[k], m2[k])
        if a[0] > 0.0 and a[1] >= 0.0:
            out.append(a.copy())
            a[:] = 0.0
    if a.any() and out:
        out[-1] += a
    while len(out) > 1 and (out[-1][0] <= 0.0 or out[-1][1] < 0.0):
        out[-2] += out.pop()
    return np.stack(out, axis=1) if out else np.zeros((3, 0))


def leg_cells_from_grid(D, du, u_edges, positive=True):
    """``(w, int u, int u^2)`` per cell of ``u_edges`` in the **leg** variable.

    ``D`` is a measure in the leg's contribution to the pair variable,
    ``v = u_leg/2``, so the ladder in ``u_leg = 2 v`` is read at ``2 du``.
    """
    n = len(u_edges) - 1
    u = np.arange(len(D)) * (2.0 * du)
    i = np.clip(np.searchsorted(u_edges, u, "right") - 1, 0, n - 1)
    c = np.stack([np.bincount(i, D, n), np.bincount(i, D * u, n),
                  np.bincount(i, D * u * u, n)])
    return merge_positive(c) if positive else c


# --------------------------------------------------------------------------
# h(a_+, a_- | m): the boson-kinematics table
# --------------------------------------------------------------------------
#: default edges of the threshold-ratio grid, in ``b = -ln a = ln(pT/pT_cut)``:
#: 1e-3 up to 0.08, 5e-3 to 0.5, 2e-2 to 3.0.  ``b`` is the leg's own variable
#: (``u_q < b_q`` is the pass condition), so the grid has to resolve ``G`` where
#: the radiator has its weight, which is ``u -> 0``.
B_EDGES = np.concatenate([np.arange(0.0, 0.08, 1e-3),
                          np.arange(0.08, 0.5, 5e-3),
                          np.arange(0.5, 3.0 + 1e-9, 2e-2)])


def build_htable(m, pt1, eta1, pt2, eta2, w, bands, pt_cut=25.0,
                 eta_cut=2.4, b_edges=None):
    """``h(a_+, a_- | m)``: the 2-D threshold-ratio table, one per ``m`` band.

    ``a_q = pT_cut / pT_q`` on the **pre-FSR** muons; the table counts the
    weight of events inside each ``(a_+, a_-)`` cell that pass the pre-FSR
    ``|eta| < eta_cut``, normalised to the band's total weight, so the sum of a
    band is ``P(|eta| < eta_cut and both pT > pT_cut | m)`` and everything the
    selection can never accept (``a > 1``, i.e. ``pT < pT_cut``, or an ``eta``
    failure) is simply absent.

    The table is stored **symmetrised**, ``(h + h^T)/2``: the QED radiator is
    the same on both legs, so only the symmetric part of ``h`` can ever enter.
    Its 2-D reverse cumulative
    ``G(u_+, u_-) = sum_{b_+ > u_+, b_- > u_-} h`` -- exact on the grid edges,
    bilinear between them -- is what the kernel uses.
    """
    b_edges = B_EDGES if b_edges is None else np.asarray(b_edges, float)
    nb = len(b_edges) - 1                      # + one overflow bin
    m = np.asarray(m, float)
    w = np.asarray(w, float)
    ok = (np.abs(np.asarray(eta1, float)) < eta_cut)
    ok &= (np.abs(np.asarray(eta2, float)) < eta_cut)
    b1 = np.log(np.maximum(np.asarray(pt1, float), 1e-9) / pt_cut)
    b2 = np.log(np.maximum(np.asarray(pt2, float), 1e-9) / pt_cut)
    ok &= (b1 > 0.0) & (b2 > 0.0)
    i1 = np.clip(np.searchsorted(b_edges, b1, "right") - 1, 0, nb)
    i2 = np.clip(np.searchsorted(b_edges, b2, "right") - 1, 0, nb)

    H = np.zeros((len(bands), nb + 1, nb + 1))
    tot = np.zeros(len(bands))
    mbar = np.zeros(len(bands))
    nev = np.zeros(len(bands), np.int64)
    for k, (lo, hi) in enumerate(bands):
        s = (m >= lo) & (m < hi)
        if not s.any():
            continue
        tot[k] = w[s].sum()
        mbar[k] = float(np.sum(w[s] * m[s]) / tot[k]) if tot[k] else 0.0
        nev[k] = int(s.sum())
        t = s & ok
        if not t.any():
            continue
        f = np.bincount(i1[t] * (nb + 1) + i2[t], w[t], (nb + 1) ** 2)
        f = f.reshape(nb + 1, nb + 1)
        H[k] = 0.5 * (f + f.T) / tot[k]
    return dict(h=H, b_edges=b_edges, bands=np.asarray(bands, float),
                m_bar=mbar, sumw=tot, nev=nev,
                pt_cut=float(pt_cut), eta_cut=float(eta_cut))


def survival(h, b_edges):
    """``G`` on the grid **nodes**: the 2-D reverse cumulative of one band's h.

    Node ``(p, q)`` is ``P(b_+ > b_edges[p], b_- > b_edges[q])``, exact because
    an event in bin ``i`` has ``b >= b_edges[i]``.  The overflow row/column
    (``b`` beyond the grid) is folded into the last bin, so the last node is
    ``P(b_+ > b_edges[-1], ...)`` and everything above it is kept.
    """
    g = np.cumsum(np.cumsum(np.asarray(h, float)[::-1, ::-1], 0), 1)[::-1, ::-1]
    return g                                   # shape (nb+1, nb+1)


def eval_G(g, b_edges, u):
    """``G(u_+, u_-)`` on the outer grid of ``u`` by bilinear interpolation.

    ``u`` beyond the last node returns that node's value -- the table's last
    bin holds everything above it, and the leg weight out there is ~1e-5.
    """
    u = np.asarray(u, float)
    nb = len(b_edges) - 1
    j = np.clip(np.searchsorted(b_edges, u, "right") - 1, 0, nb - 1)
    t = np.clip((u - b_edges[j]) / (b_edges[j + 1] - b_edges[j]), 0.0, 1.0)
    # rows first, then columns
    a = g[j] * (1.0 - t)[:, None] + g[j + 1] * t[:, None]
    return a[:, j] * (1.0 - t)[None, :] + a[:, j + 1] * t[None, :]


# --------------------------------------------------------------------------
# the selection-conditional kernel
# --------------------------------------------------------------------------
def ksel_band(u_leg, w_leg, g, b_edges, u_out_edges, idx=None, chunk=400):
    """``(cells, A)`` for one band: the selected pair-``u`` measure and ``P(pass)``.

    ``K_sel(u) = sum_{p,q} w_p w_q G(u_p, u_q) delta(u - (u_p + u_q)/2)`` and
    ``A = sum_{p,q} w_p w_q G(u_p, u_q)``.  Nothing here is approximated beyond
    the leg ladder and the bilinear reading of ``G``: swapping the ``a``
    integral inside the ``u`` one turns the double integral over ``h`` into the
    survival function evaluated at the ladder points, so the ``(a_+, a_-)``
    binning is a discretisation of ``G`` only, never of the kernel.

    ``idx`` is the precomputed output bin of ``(u_p + u_q)/2``; it depends on the
    ladder alone, so one band's worth of `np.searchsorted` is reused by all.
    """
    n = len(u_out_edges) - 1
    s0 = np.zeros(n)
    s1 = np.zeros(n)
    s2 = np.zeros(n)
    A = 0.0
    R = _interp_rows(g, b_edges, u_leg)              # G(b_edges[p], u_q)
    for a in range(0, len(u_leg), chunk):
        sl = slice(a, a + chunk)
        W = (w_leg[sl][:, None] * w_leg[None, :]) * _interp_cols(
            R, b_edges, u_leg[sl])
        U = 0.5 * (u_leg[sl][:, None] + u_leg[None, :])
        i = (idx[sl] if idx is not None else np.clip(
            np.searchsorted(u_out_edges, U, "right") - 1, 0, n - 1)).ravel()
        Wf, Uf = W.ravel(), U.ravel()
        s0 += np.bincount(i, Wf, n)
        s1 += np.bincount(i, Wf * Uf, n)
        s2 += np.bincount(i, Wf * Uf * Uf, n)
        A += float(W.sum())
    return np.stack([s0, s1, s2]), A


def _interp_rows(g, b_edges, u):
    """``G(b_edges[p], u_q)`` -- interpolate the *column* argument only."""
    nb = len(b_edges) - 1
    j = np.clip(np.searchsorted(b_edges, u, "right") - 1, 0, nb - 1)
    t = np.clip((u - b_edges[j]) / (b_edges[j + 1] - b_edges[j]), 0.0, 1.0)
    return g[:, j] * (1.0 - t)[None, :] + g[:, j + 1] * t[None, :]


def _interp_cols(R, b_edges, u):
    """Finish `_interp_rows` by interpolating the row argument."""
    nb = len(b_edges) - 1
    j = np.clip(np.searchsorted(b_edges, u, "right") - 1, 0, nb - 1)
    t = np.clip((u - b_edges[j]) / (b_edges[j + 1] - b_edges[j]), 0.0, 1.0)
    return R[j] * (1.0 - t)[:, None] + R[j + 1] * t[:, None]


# --------------------------------------------------------------------------
# drivers
# --------------------------------------------------------------------------
def make_bands(lo, hi, width):
    e = band_edges(lo, hi, width)
    b = list(zip(e[:-1], e[1:]))
    return [(0.0, e[0])] + b + [(e[-1], np.inf)]


def empirical_leg(u1, u2, w, bands, m, u_edges):
    """Per-band leg cells measured from a generator record's own ``u = -ln x``.

    Both legs are pooled -- the radiator does not know the charge -- so the
    measured ``D`` is the average of the two marginals.
    """
    n = len(u_edges) - 1
    out = []
    for lo, hi in bands:
        s = (m >= lo) & (m < hi)
        c = np.zeros((3, n))
        for uu in (u1, u2):
            x = uu[s]
            ww = w[s]
            i = np.clip(np.searchsorted(u_edges, x, "right") - 1, 0, n - 1)
            c[0] += np.bincount(i, ww, n)
            c[1] += np.bincount(i, ww * x, n)
            c[2] += np.bincount(i, ww * x * x, n)
        tot = c[0].sum()
        out.append(c / tot if tot > 0 else c)
    return out


def build_selected_kernel(ht, variant="exp2nll", pair=(), minus_one=True,
                          pair_table=None, n_leg=3000, u_min=1e-9,
                          var_budget=6e-10, sigma_cap=None, n_out=40000,
                          ng=16, leg_cells_by_band=None, verbose=True):
    """Atoms + ``A(m)`` of the selection-conditional kernel, band by band."""
    bands = ht["bands"]
    b_edges = np.asarray(ht["b_edges"], float)
    H = np.asarray(ht["h"], float)
    R, W, LO, HI = [], [], [], []
    macc, aacc, info = [], [], []
    t0 = time.time()
    for k, (lo, hi) in enumerate(bands):
        mc = 0.5 * (lo + hi)
        if not np.isfinite(mc):
            mc = float(lo) * 1.05
        if k == 0:
            mc = float(bands[0][1]) * 0.95
        D = LegRadiator(mc, variant=variant, pair=pair, minus_one=minus_one,
                        pair_table=pair_table)
        ue = leg_u_edges(D, n=n_leg, u_min=u_min)
        if leg_cells_by_band is not None:
            cells = leg_cells_by_band[k]
        else:
            cells = leg_cells(D, ue, ng=ng)
        u_leg, w_leg = leg_atoms(cells, ue)
        w_leg = w_leg / w_leg.sum()
        u_out = np.concatenate([[0.0], np.geomspace(u_min, D._u_max(), n_out)])
        g = survival(H[k], b_edges)
        c, A = ksel_band(u_leg, w_leg, g, b_edges, u_out)
        if A <= 0.0:
            info.append(dict(m=mc, A=0.0, natoms=0, mean_u=0.0))
            continue
        uj, wj = _merge_cells(c / A, var_budget, sigma_cap)
        rj = np.exp(-uj)
        o = np.argsort(-rj)
        rj, wj = rj[o], wj[o]
        wj = wj / wj.sum()
        R.append(rj)
        W.append(wj)
        LO.append(np.full(len(rj), float(lo)))
        HI.append(np.full(len(rj), float(hi)))
        macc.append(float(ht["m_bar"][k]) if ht["m_bar"][k] > 0 else mc)
        aacc.append(float(A))
        info.append(dict(m=mc, m_bar=macc[-1], A=float(A), natoms=len(rj),
                         beta=D.beta, mean_u=-float(np.sum(wj * np.log(rj)))))
    # the outermost bands that actually carry atoms are opened, so every
    # pre-FSR mass the provider can reach finds a band
    LO[0] = np.zeros_like(LO[0])
    HI[-1] = np.full_like(HI[-1], np.inf)
    if verbose:
        print(f"[ksel] {len(R)} bands, {sum(len(r) for r in R)} atoms, "
              f"{time.time()-t0:.0f} s")
    return (dict(r=np.concatenate(R), w=np.concatenate(W),
                 m_lo=np.concatenate(LO), m_hi=np.concatenate(HI)),
            dict(kind="grid", m=macc, a=aacc), info)


def _run_bands(run):
    """``(lo, hi, m_bar, s0, s1, tail0, tail1, w_norad)`` per band of a run."""
    d = np.load(run, allow_pickle=True)
    key = "n_noemit" if "n_noemit" in d.files else "n_nophot"
    # the run file is compressed: pull each block out once, not once per band
    n = np.asarray(d["n"], float)
    lo, hi, mp = d["bands_lo"], d["bands_hi"], d["mpre_s1"]
    f0, f1 = d["fine_s0"], d["fine_s1"]
    t0, t1 = d["tail_s0"], d["tail_s1"]
    wn = d[key]
    out = []
    for k in range(len(lo)):
        nk = n[k] if n[k] > 0 else 1.0
        out.append((float(lo[k]), float(hi[k]), float(mp[k] / nk),
                    f0[k] / nk, f1[k] / nk, t0[k] / nk, t1[k] / nk,
                    float(wn[k] / nk)))
    return out


def leg_from_run(run, du=KSQRT_DU, u_max=KSQRT_UMAX, n_leg=3000,
                 method="spectral", verbose=True):
    """The numerical convolution square root of a tabulated kernel, per band.

    Returns ``(u_edges, cells, meta)`` with ``cells[k]`` the leg's
    ``(w, int u, int u^2)`` on ``u_edges`` in the leg variable ``u_leg``.
    """
    rows = _run_bands(run)
    n_grid = int(round(u_max / du))
    ue = np.concatenate([[0.0], np.geomspace(2.0 * du, 2.0 * u_max, n_leg)])
    cells = np.zeros((len(rows), 3, len(ue) - 1))
    info = []
    t0 = time.time()
    for k, (lo, hi, mb, s0, s1, t0_, t1_, wn) in enumerate(rows):
        u, w = kernel_atoms(s0, s1, wn, t0_, t1_)
        g = deposit(u, w, du, n_grid)
        D, dg = conv_sqrt(g, method=method)
        raw = leg_cells_from_grid(D[:2 * n_grid], du, ue, positive=False)
        c = merge_positive(raw)
        if (c[0] <= 0.0).any() or (c[1] < 0.0).any():
            raise RuntimeError(f"band {k}: leg cells are not a positive measure")
        cells[k, :, :c.shape[1]] = c
        info.append(dict(lo=lo, hi=hi, m_bar=mb, u_max=float(u.max()),
                         mean_u_K=float((u * w).sum()),
                         mean_u_leg=float(c[1].sum() / c[0].sum()),
                         cells=int(c.shape[1]),
                         neg_cells=int((raw[0] < 0.0).sum()),
                         neg_cell_mass=float(raw[0][raw[0] < 0.0].sum()), **dg))
        if verbose and (k % 8 == 0 or k == len(rows) - 1):
            b = info[-1]
            print(f"    band {k:3d} [{lo:6.1f}, {hi:7.1f})  d0 = {dg['d0']:.6f}"
                  f"  neg = {dg['neg_mass']:+.2e}  cells = {b['cells']}"
                  f"  merged = {b['neg_cells']}"
                  f"  <u>_leg/<u>_K - 1 = "
                  f"{b['mean_u_leg'] / b['mean_u_K'] - 1:+.2e}")
    if verbose:
        print(f"[legsqrt] {len(rows)} bands, {time.time() - t0:.0f} s")
    return ue, cells, info


def _legsqrt_cmd(args):
    if args.check_band is not None:
        return _legsqrt_check(args)
    ue, cells, info = leg_from_run(args.run, du=args.du, u_max=args.u_max,
                                   n_leg=args.n_leg, method=args.method)
    meta = dict(kind="legsqrt", run=os.path.abspath(args.run), du=args.du,
                u_max=args.u_max, n_leg=args.n_leg, method=args.method,
                bands=info)
    np.savez_compressed(args.output, u_edges=ue, cells=cells,
                        bands=np.array([[b["lo"], b["hi"]] for b in info]),
                        m_bar=np.array([b["m_bar"] for b in info]),
                        provenance=np.array([json.dumps(meta)]))
    print(f"[legsqrt] -> {args.output}")


def _surv(u, w, cuts):
    """``P(u > t)`` of a discrete measure."""
    o = np.argsort(u, kind="stable")
    u, w = np.asarray(u)[o], np.asarray(w)[o]
    c = np.concatenate([np.cumsum(w[::-1])[::-1], [0.0]])
    return np.array([c[np.searchsorted(u, t, "right")] for t in cuts])


def _w1(u1, w1, u2, w2):
    """The 1-D Wasserstein-1 distance ``int |F_1 - F_2| du`` of two measures.

    Cut-position free, and the natural size of the grid representation: it is
    the mean distance the deposition moves the kernel's mass in ``u``.
    """
    x = np.unique(np.concatenate([np.asarray(u1, float), np.asarray(u2, float)]))
    def cdf(u, w):
        o = np.argsort(u, kind="stable")
        return np.cumsum(np.asarray(w)[o])[
            np.clip(np.searchsorted(np.asarray(u)[o], x, "right") - 1, 0, None)
        ] * (np.searchsorted(np.sort(u), x, "right") > 0)
    return float(np.sum(np.abs(cdf(u1, w1) - cdf(u2, w2))[:-1] * np.diff(x)))


def _legsqrt_check(args):
    """``D (x) D`` against the tabulated kernel: grid scan and fixed point."""
    rows = _run_bands(args.run)
    k = args.check_band
    lo, hi, mb, s0, s1, t0_, t1_, wn = rows[k]
    u, w = kernel_atoms(s0, s1, wn, t0_, t1_)
    cuts = np.array([1.3e-4, 1.1e-3, 1.07e-2, 5.3e-2, 0.21, 0.53])
    pk, uk = _surv(u, w, cuts), float((u * w).sum())
    print(f"band {k} [{lo:.0f}, {hi:.0f}), <m_pre> = {mb:.3f} GeV, "
          f"P(no emission) = {wn:.6f}, support to u = {u.max():.3f}\n")
    print(f"K (atoms):  <u> = {uk:.8e}")
    print("     t     " + " ".join(f"{v:8.2e}" for v in cuts))
    print("  P(u>t)   " + " ".join(f"{v:8.6f}" for v in pk))
    print(f"\n{'du':>9s} {'|K^|min':>9s} {'max|arg|':>9s} {'D[0]':>10s}"
          f" {'neg mass':>10s} {'roundtrip':>10s} {'d<u>/<u>':>10s} {'W1':>9s}"
          f"   P(grid)/P(atoms) - 1")
    for du in args.du_scan:
        ng = int(round(args.u_max / du))
        g = deposit(u, w, du, ng)
        D, dg = conv_sqrt(g, method="spectral")
        ug = np.arange(len(g)) * du
        pg = _surv(ug, g, cuts)
        print(f"{du:9.1e} {dg['k_abs_min']:9.4f} {dg['k_arg_max']:9.3f}"
              f" {dg['d0']:10.6f} {dg['neg_mass']:+10.2e} {dg['roundtrip']:10.1e}"
              f" {(ug * g).sum() / uk - 1:+10.1e} {_w1(ug, g, u, w):9.2e}  "
              + " ".join(f"{a / b - 1:+8.1e}" for a, b in zip(pg, pk)))
        if abs(du - args.du) < 1e-12:
            e = np.concatenate([[0.0], np.geomspace(du, 2.0 * u.max(), 300)])
            i = np.clip(np.searchsorted(e, np.arange(len(D)) * du, "right") - 1,
                        0, len(e) - 2)
            b = np.bincount(i, D, len(e) - 1)
            print(f"{'':9s} log-binned D: {int((b < 0).sum())} negative cells "
                  f"of {int((b != 0).sum())}, most negative {b.min():+.2e}")
            Df, fi = conv_sqrt(g, method="fixedpoint")
            print(f"{'':9s} fixed point:  {fi['iterations']} iterations, "
                  f"contraction {fi['contraction']:.3f}, last step "
                  f"{fi['last_step']:.1e}, max|D_fp - D_sqrt| = "
                  f"{np.abs(Df - D).max():.1e}")


def _htable_cmd(args):
    d = np.load(args.gen)
    w = d["weight"].astype(np.float64)
    if args.wclip:
        aw = np.abs(w)
        w = np.clip(w, -args.wclip * np.median(aw), args.wclip * np.median(aw))
        w = w * (len(w) / w.sum())
    m = d["m_pre"].astype(np.float64)
    if "ptp_pre" in d:
        pt1, eta1 = d["ptp_pre"], d["etap_pre"]
        pt2, eta2 = d["ptm_pre"], d["etam_pre"]
    else:
        pt1, eta1 = d["pt1_pre"], d["eta1_pre"]
        pt2, eta2 = d["pt2_pre"], d["eta2_pre"]
    bands = make_bands(args.band_lo, args.band_hi, args.band_width)
    be = B_EDGES if args.b_thin <= 1 else np.unique(
        np.concatenate([B_EDGES[::args.b_thin], B_EDGES[-1:]]))
    ht = build_htable(m, pt1, eta1, pt2, eta2, w, bands, pt_cut=args.pt_cut,
                      eta_cut=args.eta_cut, b_edges=be)
    meta = dict(gen=os.path.abspath(args.gen), pt_cut=args.pt_cut,
                eta_cut=args.eta_cut, band_lo=args.band_lo,
                band_hi=args.band_hi, band_width=args.band_width,
                wclip=args.wclip, b_thin=args.b_thin,
                n_b=len(ht["b_edges"]))
    np.savez_compressed(args.output, h=ht["h"].astype(np.float32),
                        b_edges=ht["b_edges"], bands=ht["bands"],
                        m_bar=ht["m_bar"], sumw=ht["sumw"], nev=ht["nev"],
                        provenance=np.array([json.dumps(meta)]))
    print(f"[htable] {len(bands)} bands x {ht['h'].shape[1]}^2 cells, "
          f"pT > {args.pt_cut}, |eta| < {args.eta_cut} -> {args.output}")
    for k in range(0, len(bands), max(1, len(bands) // 8)):
        print(f"    m = {ht['m_bar'][k]:7.2f}  n = {ht['nev'][k]:8d}  "
              f"sum h = {ht['h'][k].sum():.5f}")


def _kernel_cmd(args):
    with np.load(args.htable, allow_pickle=False) as d:
        ht = {k: d[k] for k in ("h", "b_edges", "bands", "m_bar", "sumw", "nev")}
    prov = json.loads(str(np.load(args.htable,
                                  allow_pickle=False)["provenance"][0]))
    legs = None
    if args.empirical_leg:
        legs = _empirical_leg_cells(args.empirical_leg, ht, args)
    elif getattr(args, "mc_leg", None):
        legs = _mc_leg_cells(args.mc_leg, ht)
    ker, acc, info = build_selected_kernel(
        ht, variant=args.variant, pair=tuple(args.pair or ()),
        minus_one=not args.no_minus_one, pair_table=args.pair_table,
        n_leg=args.n_leg, var_budget=(None if args.sigma_cap
                                      else args.var_budget),
        sigma_cap=args.sigma_cap, n_out=args.n_out, leg_cells_by_band=legs)
    meta = dict(kind="perleg", variant=args.variant,
                pair=list(args.pair or ()), htable=os.path.abspath(args.htable),
                htable_prov=prov, empirical_leg=args.empirical_leg,
                n_leg=args.n_leg, var_budget=args.var_budget,
                mc_leg=getattr(args, "mc_leg", None),
                sigma_cap=args.sigma_cap, alpha=ALPHA, m_mu=M_MU, bands=info)
    np.savez(args.output, r=ker["r"], w=ker["w"], m_lo=ker["m_lo"],
             m_hi=ker["m_hi"], provenance=np.array([json.dumps(meta)]))
    if args.acceptance:
        with open(args.acceptance, "w") as fh:
            json.dump(dict(kind="grid", m=acc["m"], a=acc["a"],
                           _meta=dict(pt_cut=prov["pt_cut"],
                                      eta_cut=prov["eta_cut"],
                                      source=os.path.abspath(args.output))),
                      fh, indent=1)
        print(f"[kernel] acceptance -> {args.acceptance}")
    print(f"[kernel] {len(ker['r'])} atoms -> {args.output}")
    for b in info[:: max(1, len(info) // 10)]:
        if b["natoms"]:
            print(f"    m = {b['m_bar']:7.2f}  A = {b['A']:.5f}"
                  f"  atoms = {b['natoms']:4d}  <u> = {b['mean_u']*1e3:8.4f}e-3")


def _mc_leg_cells(path, ht):
    """Per-band leg cells from a `legsqrt` file, matched to the h table's bands.

    The square root is taken band by band on the run's own (2 GeV) bands; an
    `h` band is served by the run band its centre falls in.  The leg depends on
    ``m`` only through the radiator, so the mapping is a coarsening, not an
    interpolation.
    """
    d = np.load(path, allow_pickle=False)
    cells, bl, bh = d["cells"], d["bands"][:, 0], d["bands"][:, 1]
    out = []
    for lo, hi in ht["bands"]:
        mc = 0.5 * (lo + hi)
        if not np.isfinite(mc):
            mc = float(lo) * 1.05
        j = np.nonzero((bl <= mc) & (bh > mc))[0]
        j = int(j[0]) if len(j) else int(np.argmin(np.abs(0.5 * (bl + bh) - mc)))
        out.append(cells[j])
    return out


def _empirical_leg_cells(path, ht, args):
    """Per-band leg cells measured from a per-leg generator record."""
    d = np.load(path)
    w = d["weight"].astype(np.float64)
    aw = np.abs(w)
    w = np.clip(w, -100.0 * np.median(aw), 100.0 * np.median(aw))
    w = w * (len(w) / w.sum())
    m = d["m_pre"].astype(np.float64)
    u1 = -np.log(np.maximum(d["xp"], 1e-300))
    u2 = -np.log(np.maximum(d["xm"], 1e-300))
    wid = float(args.leg_band_width)
    out = []
    for lo, hi in ht["bands"]:
        mc = 0.5 * (lo + hi)
        if not np.isfinite(mc):
            mc = float(lo) * 1.05
        D = LegRadiator(mc, variant=args.variant)
        ue = leg_u_edges(D, n=args.n_leg, u_min=1e-9)
        # widen the window the leg is measured in, centred on the band
        w_lo, w_hi = (mc - 0.5 * wid, mc + 0.5 * wid) if wid > 0 else (lo, hi)
        c = empirical_leg(u1, u2, w, [(w_lo, w_hi)], m, ue)[0]
        if c[0].sum() <= 0:
            c = empirical_leg(u1, u2, w, [(0.0, np.inf)], m, ue)[0]
        out.append(c)
    return out


def _check_cmd(args):
    """``D (x) D`` against ``K``: Mellin moments, their order, and the density."""
    m = args.mass
    n = np.arange(1, 9)
    print(f"D~(n)^2 / K~(n) - 1 at m = {m} GeV\n")
    print(f"{'n':>4s}" + "".join(f"{v:>14s}" for v in ("exp1", "exp2",
                                                       "exp2nll", "data cfg")))
    cfgs = [("exp1", ()), ("exp2", ()), ("exp2nll", ()),
            ("exp2nll", ("e", "mu", "tau", "had"))]
    rows = []
    for var, pair in cfgs:
        K = FSRKernel(m, variant=var, pair=pair, pair_table=args.pair_table)
        D = LegRadiator(m, variant=var, pair=pair, pair_table=args.pair_table)
        rows.append(mellin(D, n) ** 2 / mellin(K, n) - 1.0)
    for i, nn in enumerate(n):
        print(f"{nn:4d}" + "".join(f"{r[i]:14.3e}" for r in rows))

    print("\nthe residual is O(alpha^3): res(n) / beta^k across masses")
    print(f"{'m':>8s} {'beta':>9s} {'exp1 n=8':>12s} {'/beta^2':>10s}"
          f" {'exp2nll n=8':>13s} {'/beta^3':>10s}")
    for mm in (15.0, 30.0, 91.1876, 300.0, 1000.0):
        b = beta_fsr(mm)
        r1 = (mellin(LegRadiator(mm, variant="exp1"), [8.0])[0] ** 2
              / mellin(FSRKernel(mm, variant="exp1"), [8.0])[0] - 1.0)
        r2 = (mellin(LegRadiator(mm, variant="exp2nll"), [8.0])[0] ** 2
              / mellin(FSRKernel(mm, variant="exp2nll"), [8.0])[0] - 1.0)
        print(f"{mm:8.2f} {b:9.5f} {r1:12.3e} {r1/b**2:10.4f}"
              f" {r2:13.3e} {r2/b**3:10.4f}")

    print("\ndensity: P(u > u0) and <u | u < u0> of D (x) D against K")
    ue = np.concatenate([[0.0], np.geomspace(1e-9, 7.0, 6000)])
    for var, pair in cfgs:
        K = FSRKernel(m, variant=var, pair=pair, pair_table=args.pair_table)
        D = LegRadiator(m, variant=var, pair=pair, pair_table=args.pair_table)
        e = leg_u_edges(D, n=args.n_leg)
        u, w = leg_atoms(leg_cells(D, e), e)
        s = conv_atoms(u, w / w.sum(), ue)
        um = np.where(s[0] > 0, s[1] / np.maximum(s[0], 1e-300), 0.0)
        # `leg_cells` is generic: run on the inclusive kernel it folds in the
        # full pair factor on the pair's own u ladder, so both sides of the
        # comparison carry the pairs
        ek = leg_u_edges(K, n=args.n_leg)
        kc = leg_cells(K, ek)
        ku = np.where(kc[0] > 0, kc[1] / np.maximum(kc[0], 1e-300), 0.0)
        tag = var + ("+pairs" if pair else "")
        print(f"\n  {tag}   int D dx = {w.sum():.9f}"
              f"   <u>_K = {K.moments()['u']:.6e}"
              f"   <u>_DD = {(s[1].sum()/s[0].sum()):.6e}"
              f"   ratio-1 = {(s[1].sum()/s[0].sum())/K.moments()['u']-1:+.2e}")
        print(f"  {'u0':>8s} {'P_K(u>u0)':>13s} {'P_DD':>13s} {'ratio-1':>10s}"
              f" {'<u|u<u0>_K':>13s} {'ratio-1':>10s}")
        for u0 in (1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5):
            pk = kc[0][ku > u0].sum() / kc[0].sum()
            pd = s[0][um > u0].sum() / s[0].sum()
            ak = kc[1][ku < u0].sum() / kc[0][ku < u0].sum()
            ad = s[1][um < u0].sum() / s[0][um < u0].sum()
            print(f"  {u0:8.3g} {pk:13.8f} {pd:13.8f} {pd/pk-1:+10.2e}"
                  f" {ak:13.8f} {ad/ak-1:+10.2e}")


# --------------------------------------------------------------------------
# the conditional kernel read in the model's own variables
# --------------------------------------------------------------------------
#: the two approximations of the collinear picture, as switches on the
#: generator record: which mass loss is histogrammed, and which decision the
#: selection takes.  ``true`` is the MC's own selected kernel; ``coll`` is what
#: the per-leg construction would give if the two legs kept their measured
#: correlation, so ``coll`` minus ``true`` is the cost of ``z = x_+ x_-`` and of
#: taking the ``p_T`` decision on ``x p_T^{pre}``, and the per-leg model minus
#: ``coll`` is the cost of treating the legs as independent.
COND_MODES = {
    "true": ("u", "post"),
    "collmass": ("coll", "post"),
    "collsel": ("u", "coll"),
    "coll": ("coll", "coll"),
}
#: the bands the selected kernel is measured in; the selection makes ``<u|m>``
#: run by a factor two across the window, so the kernel has to be banded, and
#: these are the bands the MC-conditional reference uses.
COND_BANDS = (70.0, 80.0, 85.0, 88.0, 91.0, 94.0, 98.0, 105.0, 115.0, 130.0)


def _condker_cmd(args):
    import fit_gen as FG

    d = np.load(args.gen)
    w = np.asarray(d["weight"], float)
    aw = np.abs(w)
    w = np.clip(w, -args.wclip * np.median(aw), args.wclip * np.median(aw))
    w = w * (len(w) / w.sum())
    m = np.asarray(d["m_pre"], float)
    xp = np.asarray(d["xp"], float)
    xm = np.asarray(d["xm"], float)
    mass = {"u": np.asarray(d["m_post"], float),
            "coll": m * np.sqrt(np.maximum(xp * xm, 1e-300))}
    sel = {
        "post": ((d["ptp"] > args.pt_cut) & (d["ptm"] > args.pt_cut)
                 & (np.abs(d["etap"]) < args.eta_cut)
                 & (np.abs(d["etam"]) < args.eta_cut)),
        "coll": ((xp * d["ptp_pre"] > args.pt_cut)
                 & (xm * d["ptm_pre"] > args.pt_cut)
                 & (np.abs(d["etap_pre"]) < args.eta_cut)
                 & (np.abs(d["etam_pre"]) < args.eta_cut)),
    }
    mkey, skey = COND_MODES[args.mode]
    s = sel[skey]
    bands = list(zip(COND_BANDS[:-1], COND_BANDS[1:]))
    bands = [(0.0, COND_BANDS[0])] + bands + [(COND_BANDS[-1], np.inf)]
    ker, info = FG.build_banded_kernel(m[s], mass[mkey][s], w[s], bands,
                                       sigma_cap=args.sigma_cap)
    meta = dict(kind="condker", mode=args.mode, mass=mkey, selection=skey,
                gen=os.path.abspath(args.gen), pt_cut=args.pt_cut,
                eta_cut=args.eta_cut, sigma_cap=args.sigma_cap,
                bands=list(COND_BANDS), n=int(s.sum()))
    np.savez(args.output, r=ker["r"], w=ker["w"], m_lo=ker["m_lo"],
             m_hi=ker["m_hi"], provenance=np.array([json.dumps(meta)]))
    print(f"[condker] {args.mode}: mass = {mkey}, selection = {skey}, "
          f"{int(s.sum())} events, {len(ker['r'])} atoms -> {args.output}")
    for lo, hi, n, na, pn, md in info:
        print(f"    [{lo:6.1f}, {hi:6.1f})  n = {n:9d}  atoms = {na:4d}"
              f"  P(no rad) = {pn:.4f}  <u> = {md * 1e3:+8.4f}e-3")
    if args.acceptance:
        e = np.arange(args.acc_lo, args.acc_hi + 1e-9, args.acc_width)
        i = np.clip(np.searchsorted(e, m, "right") - 1, 0, len(e) - 2)
        tot = np.bincount(i, w, len(e) - 1)
        npass = np.bincount(i, w * s, len(e) - 1)
        mb = np.bincount(i, w * m, len(e) - 1) / np.maximum(tot, 1e-30)
        ok = tot > 0
        with open(args.acceptance, "w") as fh:
            json.dump(dict(kind="grid", m=mb[ok].tolist(),
                           a=(npass[ok] / tot[ok]).tolist(),
                           _meta=dict(source=os.path.abspath(args.output),
                                      selection=skey, pt_cut=args.pt_cut,
                                      eta_cut=args.eta_cut)), fh, indent=1)
        print(f"[condker] acceptance ({skey}) -> {args.acceptance}")


def fit_bernstein(m, a, lo, hi, degree):
    """Least-squares Bernstein fit of a tabulated ``A(m)``, unweighted in m."""
    from math import comb
    m = np.asarray(m, float)
    a = np.asarray(a, float)
    s = (m >= lo) & (m <= hi)
    u = (m[s] - lo) / (hi - lo)
    B = np.stack([comb(degree, k) * u**k * (1.0 - u) ** (degree - k)
                  for k in range(degree + 1)], axis=1)
    c, *_ = np.linalg.lstsq(B, a[s], rcond=None)
    return c, float(np.max(np.abs(B @ c - a[s])))


def _accfit_cmd(args):
    with open(args.acceptance) as fh:
        d = json.load(fh)
    c, dev = fit_bernstein(d["m"], d["a"], args.lo, args.hi, args.degree)
    with open(args.output, "w") as fh:
        json.dump(dict(kind="bernstein", lo=args.lo, hi=args.hi,
                       coef=c.tolist(),
                       _meta=dict(degree=args.degree, maxdev=dev,
                                  source=os.path.abspath(args.acceptance))),
                  fh, indent=1)
    print(f"[accfit] degree {args.degree}, max |A_fit - A| = {dev:.2e} "
          f"-> {args.output}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="D (x) D against the inclusive K")
    c.add_argument("--mass", type=float, default=91.1876)
    c.add_argument("--n-leg", type=int, default=6000)
    c.add_argument("--pair-table", default=None)

    t = sub.add_parser("htable", help="h(a_+, a_- | m) from a generator record")
    t.add_argument("--gen", required=True)
    t.add_argument("-o", "--output", required=True)
    t.add_argument("--pt-cut", type=float, default=25.0)
    t.add_argument("--eta-cut", type=float, default=2.4)
    t.add_argument("--band-lo", type=float, default=50.0)
    t.add_argument("--band-hi", type=float, default=200.0)
    t.add_argument("--band-width", type=float, default=1.0)
    t.add_argument("--wclip", type=float, default=100.0)
    t.add_argument("--b-thin", type=int, default=1,
                   help="keep every n-th edge of the (a_+, a_-) grid")

    k = sub.add_parser("kernel", help="the selection-conditional atoms + A(m)")
    k.add_argument("--htable", required=True)
    k.add_argument("-o", "--output", required=True)
    k.add_argument("--acceptance", default=None, help="write A(m) as a json")
    k.add_argument("--variant", default="exp2nll",
                   choices=("exp1", "exp2", "exp2nll"))
    k.add_argument("--pair", nargs="*", default=None)
    k.add_argument("--no-minus-one", action="store_true")
    k.add_argument("--pair-table", default=None)
    k.add_argument("--n-leg", type=int, default=3000)
    k.add_argument("--n-out", type=int, default=40000)
    k.add_argument("--var-budget", type=float, default=6e-10)
    k.add_argument("--sigma-cap", type=float, default=None)
    k.add_argument("--empirical-leg", default=None,
                   help="per-leg gen npz: use its measured D instead")
    k.add_argument("--mc-leg", default=None,
                   help="`legsqrt` npz: use the numerical convolution square "
                        "root of a tabulated kernel as D")
    k.add_argument("--leg-band-width", type=float, default=10.0,
                   help="m_pre band width the empirical D is measured in; the "
                        "leg radiator depends on m only through beta(m), so it "
                        "is measured in wide bands while h keeps the narrow "
                        "ones (a 1 GeV empirical D carries a 7 %% band-to-band "
                        "statistical jitter that the smooth K(m) cannot absorb)")

    s = sub.add_parser("legsqrt",
                       help="D from a tabulated kernel by convolution sqrt")
    s.add_argument("--run", required=True,
                   help="a `photos_standalone/merge.py` run npz")
    s.add_argument("-o", "--output", default=None)
    s.add_argument("--du", type=float, default=KSQRT_DU)
    s.add_argument("--u-max", type=float, default=KSQRT_UMAX)
    s.add_argument("--n-leg", type=int, default=3000)
    s.add_argument("--method", default="spectral",
                   choices=("spectral", "fixedpoint"))
    s.add_argument("--check-band", type=int, default=None,
                   help="run the grid scan and the fixed-point cross-check on "
                        "this band instead of writing a file")
    s.add_argument("--du-scan", type=float, nargs="*",
                   default=(1e-4, 4e-5, 2e-5, 1e-5, 5e-6))

    d = sub.add_parser("condker",
                       help="the MC's own conditional kernel, in the model's "
                            "mass and selection variables")
    d.add_argument("--gen", required=True, help="a per-leg gen record")
    d.add_argument("-o", "--output", required=True)
    d.add_argument("--mode", required=True, choices=tuple(COND_MODES))
    d.add_argument("--acceptance", default=None)
    d.add_argument("--pt-cut", type=float, default=25.0)
    d.add_argument("--eta-cut", type=float, default=2.4)
    d.add_argument("--sigma-cap", type=float, default=3.3e-4)
    d.add_argument("--wclip", type=float, default=100.0)
    d.add_argument("--acc-lo", type=float, default=50.0)
    d.add_argument("--acc-hi", type=float, default=200.0)
    d.add_argument("--acc-width", type=float, default=1.0)

    f = sub.add_parser("accfit", help="Bernstein fit of a tabulated A(m)")
    f.add_argument("--acceptance", required=True)
    f.add_argument("-o", "--output", required=True)
    f.add_argument("--lo", type=float, default=50.0)
    f.add_argument("--hi", type=float, default=200.0)
    f.add_argument("--degree", type=int, default=8)

    args = ap.parse_args()
    if args.cmd == "accfit":
        _accfit_cmd(args)
    elif args.cmd == "check":
        _check_cmd(args)
    elif args.cmd == "htable":
        _htable_cmd(args)
    elif args.cmd == "kernel":
        _kernel_cmd(args)
    elif args.cmd == "legsqrt":
        _legsqrt_cmd(args)
    elif args.cmd == "condker":
        _condker_cmd(args)


if __name__ == "__main__":
    main()
