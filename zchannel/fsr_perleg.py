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

**The correlated two-leg density (`corr`).**  The product ``D(x_+) D(x_-)``
draws the two momentum fractions **independently**, and they are not: at
O(alpha) the exact three-body kinematics put them on the LINE
``x_+ + x_- = 1 + z`` with the sharing fixed by the photon angle, and the ~1/L
of the rate away from the collinear end points is a photon taking energy off
both muons at once.  `corr` replaces the product by the exact O(alpha)
correlated density -- ``z`` drawn from ``K`` (untouched, ``int df p_1 = 1`` at
every ``z``) and the sharing from the spin-summed matrix element -- so the
selected kernel is the inclusive kernel reweighted by

    Gbar(u | m) = int df p_1(f | z) G(u_+(z, f), u_-(z, f) | m) ,   z = e^{-2u}

with the same ``G`` and the same ``h`` table.  ``--mode matched`` is the
alternative composition of the resummation, ``K = D_< (x) D_< (x) H_>`` with the
exact hard emission above ``u_c`` and a collinear-independent soft remainder
below it; the difference between the two bounds the multi-emission recoil, and
the matching scale itself is worth 0.2 MeV.
"""

import argparse
import json
import math
import os
import time

import numpy as np

import fsr_analytic as FA
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
        tail = out.pop()
        out[-1] += tail
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


def build_htable(m, pt1, eta1, pt2, eta2, w, bands, pt_ref=25.0,
                 eta_cut=2.4, b_edges=None):
    """``h(b_+, b_- | m)``: the 2-D threshold-ratio table, one per ``m`` band.

    ``b_q = ln(pT_q / pT_ref)`` on the **pre-FSR** muons; the table counts the
    weight of events inside each ``(b_+, b_-)`` cell that pass the pre-FSR
    ``|eta| < eta_cut``, normalised to the band's total weight, so the sum of a
    band is ``P(|eta| < eta_cut and both pT > pT_ref e^{b_min} | m)`` and
    everything below the grid's own floor -- which no cut at or above
    ``pT_ref`` can accept, up to the resolution -- is simply absent.

    ``pT_ref`` is the reference of the logarithmic axis, NOT a cut: a table
    built at ``pT_ref = 10`` serves 25/10, 25/25 and 10/10 alike, and which
    cuts are applied is `PassRegion`'s business.  Its floor ``b_edges[0]``
    must sit below every threshold the table will be asked about, and below it
    by enough resolution that an up-fluctuation cannot reach the cut.

    The table is stored **symmetrised**, ``(h + h^T)/2``: the QED radiator is
    the same on both legs, so only the symmetric part of ``h`` can ever enter.
    Its 2-D reverse cumulative
    ``G(u_+, u_-) = sum_{b_+ > u_+, b_- > u_-} h`` -- exact on the grid edges,
    bilinear between them -- is what a symmetric step cut uses directly.
    """
    b_edges = B_EDGES if b_edges is None else np.asarray(b_edges, float)
    nb = len(b_edges) - 1                      # + one overflow bin
    m = np.asarray(m, float)
    w = np.asarray(w, float)
    ok = (np.abs(np.asarray(eta1, float)) < eta_cut)
    ok &= (np.abs(np.asarray(eta2, float)) < eta_cut)
    b1 = np.log(np.maximum(np.asarray(pt1, float), 1e-9) / pt_ref)
    b2 = np.log(np.maximum(np.asarray(pt2, float), 1e-9) / pt_ref)
    ok &= (b1 > b_edges[0]) & (b2 > b_edges[0])
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
                pt_ref=float(pt_ref), eta_cut=float(eta_cut))


def build_h4table(m, pt1, eta1, pt2, eta2, w, bands, pt_ref=10.0,
                  eta_cut=2.4, b_edges=None, eta_edges=None):
    """``h4(b_+, eta_+, b_-, eta_-|m)``: the same table with the ``eta`` axis.

    The resolution depends on ``eta``, so a pass PROBABILITY -- unlike a step
    in ``pT`` -- needs to know which ``eta`` each leg had.  This is the whole
    extension of the boson-kinematics interface that the detector level forces:
    one more axis per leg, on a coarse grid, and nothing else.

    Stored as ``(n_band, n_eta, n_b+1, n_eta, n_b+1)`` float32, symmetrised
    under the simultaneous swap of both axes of the two legs.  The ``b`` grid
    is deliberately coarser than the step table's: the step is smoothed by the
    resolution, so ``G`` is read through a kernel of width ``sigma`` and the
    cell midpoint rule converges second order (`smeared_G`).
    """
    b_edges = B_EDGES if b_edges is None else np.asarray(b_edges, float)
    eta_edges = (np.array([0.0, 0.9, 1.6, 2.1, 2.4]) if eta_edges is None
                 else np.asarray(eta_edges, float))
    nb = len(b_edges) - 1
    ne = len(eta_edges) - 1
    nc = ne * (nb + 1)
    m = np.asarray(m, float)
    w = np.asarray(w, float)
    a1 = np.abs(np.asarray(eta1, float))
    a2 = np.abs(np.asarray(eta2, float))
    ok = (a1 < eta_cut) & (a2 < eta_cut) & (a1 >= eta_edges[0]) \
        & (a2 >= eta_edges[0])
    b1 = np.log(np.maximum(np.asarray(pt1, float), 1e-9) / pt_ref)
    b2 = np.log(np.maximum(np.asarray(pt2, float), 1e-9) / pt_ref)
    ok &= (b1 > b_edges[0]) & (b2 > b_edges[0])
    i1 = np.clip(np.searchsorted(b_edges, b1, "right") - 1, 0, nb)
    i2 = np.clip(np.searchsorted(b_edges, b2, "right") - 1, 0, nb)
    e1 = np.clip(np.searchsorted(eta_edges, a1, "right") - 1, 0, ne - 1)
    e2 = np.clip(np.searchsorted(eta_edges, a2, "right") - 1, 0, ne - 1)
    c1 = e1 * (nb + 1) + i1
    c2 = e2 * (nb + 1) + i2

    H = np.zeros((len(bands), ne, nb + 1, ne, nb + 1), np.float32)
    # the cell's own mean b, per leg marginal.  The contraction reads the pass
    # probability at ONE b per cell; taking the cell's mean instead of its
    # midpoint removes the first-order discretisation error and is one extra
    # number per (eta, b) cell -- 0.2 % of the table.
    B = np.zeros((len(bands), ne, nb + 1))
    mid = np.concatenate([0.5 * (b_edges[:-1] + b_edges[1:]),
                          [b_edges[-1] + 0.5 * (b_edges[-1] - b_edges[-2])]])
    tot = np.zeros(len(bands))
    mbar = np.zeros(len(bands))
    nev = np.zeros(len(bands), np.int64)
    for k, (lo, hi) in enumerate(bands):
        s = (m >= lo) & (m < hi)
        B[k] = np.broadcast_to(mid, (ne, nb + 1))
        if not s.any():
            continue
        tot[k] = w[s].sum()
        mbar[k] = float(np.sum(w[s] * m[s]) / tot[k]) if tot[k] else 0.0
        nev[k] = int(s.sum())
        t = s & ok
        if not t.any():
            continue
        f = np.bincount(c1[t] * nc + c2[t], w[t], nc * nc).reshape(nc, nc)
        f = (0.5 * (f + f.T) / tot[k]).reshape(ne, nb + 1, ne, nb + 1)
        H[k] = f.astype(np.float32)
        cw = (np.bincount(c1[t], w[t], nc) + np.bincount(c2[t], w[t], nc))
        cb = (np.bincount(c1[t], w[t] * b1[t], nc)
              + np.bincount(c2[t], w[t] * b2[t], nc))
        g = cw > 0
        bm = B[k].ravel().copy()
        bm[g] = cb[g] / cw[g]
        B[k] = bm.reshape(ne, nb + 1)
    return dict(h4=H, b_mean=B, b_edges=b_edges, eta_edges=eta_edges,
                bands=np.asarray(bands, float), m_bar=mbar, sumw=tot, nev=nev,
                pt_ref=float(pt_ref), eta_cut=float(eta_cut))


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
# the pass region: which (u_+, u_-) survive the lepton pT cuts
# --------------------------------------------------------------------------
# The cuts are a LEADING and a TRAILING threshold, ``c_L >= c_T``, and which
# muon is which is decided AFTER the loss (and, at detector level, after the
# smearing).  With ``b_q = ln(pT_q^pre / pT_ref)`` and the loss ``u_q``, the
# post-FSR transverse momenta are ``pT_ref e^{b_q - u_q}``, so the pass
# condition is
#
#     max(pT'_+, pT'_-) > c_L   and   min(pT'_+, pT'_-) > c_T
#
# i.e. the UNION of the two rectangles
#
#     A = {b_+ - u_+ > s_L , b_- - u_- > s_T} ,  s_c = ln(c / pT_ref)
#     B = {b_- - u_- > s_L , b_+ - u_+ > s_T}
#
# in the ``(b_+, b_-)`` plane at fixed ``(u_+, u_-)``.  ``A and B`` is
# ``{both over c_L}`` because ``c_L >= c_T``, so
#
#     P(pass) = P_A + P_B - P_{A and B}
#             = P_+(c_L) P_-(c_T) + P_+(c_T) P_-(c_L) - P_+(c_L) P_-(c_L)
#
# with ``P_q(c)`` the probability that leg ``q`` alone clears ``c``.  Every term
# is a PRODUCT of two per-leg thresholds, which is what makes the whole
# construction work for a smooth pass probability as well as for a step: the
# two legs' resolutions are independent given their own ``(pT, eta)``, so the
# joint pass probability of a rectangle factorises inside the boson-kinematics
# average and only the ``(b, eta)`` pair correlation is left in the table.
#
# ``c_L = c_T`` collapses the three terms to ``P_+ P_-``, the symmetric case,
# and the code returns the plain survival function there, bit for bit.
def b_grid(anchors=(0.0,), b_min=0.0, b_max=3.0,
           fine=((0.08, 1e-3), (0.5, 5e-3)), coarse=2e-2):
    """Edges in ``b = ln(pT/pT_ref)``: the fine pattern replicated at each cut.

    ``G`` has to be resolved where the leg radiator has its weight, which is at
    ``u -> 0`` **above each threshold**.  With an asymmetric pair of cuts there
    are two thresholds, ``b = ln(c/pT_ref)``, so the fine part of the grid is
    replicated at each of them; between and beyond them the grid is coarse.

    The replication is exact: ``anchor + pattern`` is the same expression the
    shifted reading of the survival function evaluates, so a node of the grid
    shifted by ``s_L - s_T`` lands on another node and the reading stays exact
    there rather than interpolating.

    ``anchors = (0,)``, ``b_min = 0``, ``b_max = 3`` reproduces `B_EDGES`.
    """
    anchors = np.unique(np.asarray(anchors, float))
    parts = []
    if b_min < anchors[0] - 1e-12:
        e = np.arange(b_min, anchors[0], coarse)
        parts.append(e[e < anchors[0] - 1e-9])
    for i, a in enumerate(anchors):
        lo = 0.0
        for hi, step in fine:
            parts.append(a + np.arange(lo, hi, step))
            lo = hi
        nxt = anchors[i + 1] if i + 1 < len(anchors) else None
        if nxt is not None:
            e = np.arange(a + lo, nxt, coarse)
            parts.append(e[e < nxt - 1e-9])
    e = np.arange(anchors[-1] + fine[-1][0], b_max + 1e-9, coarse)
    parts.append(e)
    b = np.concatenate(parts)
    keep = np.concatenate([[True], np.diff(b) > 1e-9])
    return b[keep]


def cut_shifts(cuts, pt_ref):
    """``(s_L, s_T) = ln(c/pT_ref)`` of the leading and trailing thresholds."""
    c = np.atleast_1d(np.asarray(cuts, float))
    if c.size == 1:
        c = np.array([c[0], c[0]])
    c = np.sort(c)[::-1]
    return float(np.log(c[0] / pt_ref)), float(np.log(c[1] / pt_ref))


def rect_terms(s_lead, s_trail, tol=1e-12):
    """``[(shift_+, shift_-, sign)]`` of the union-of-rectangles pass region."""
    if s_lead - s_trail <= tol:
        return [(s_trail, s_trail, 1.0)]
    return [(s_lead, s_trail, 1.0), (s_trail, s_lead, 1.0),
            (s_lead, s_lead, -1.0)]


def step_G(h, b_edges, terms):
    """``G`` on the grid nodes for a step acceptance: shifted survivals.

    Each rectangle is the survival function read at ``(b_p + shift_+,
    b_q + shift_-)``; the sum is the union.  A symmetric cut returns
    `survival` itself, unread and unshifted.
    """
    g = survival(h, b_edges)
    if len(terms) == 1 and terms[0][0] == 0.0 and terms[0][1] == 0.0:
        return g
    out = np.zeros_like(g)
    for sp, sm, sgn in terms:
        R = _interp_rows(g, b_edges, b_edges + sm)
        out += sgn * _interp_cols(R, b_edges, b_edges + sp)
    return out


def pass_matrix(resol, cut, iband, b_src, b_out, s):
    """``Pi[i, p] = P(pT_reco > cut | b_i, u = b_out[p], eta band iband)``.

    ``v = ln(pT^post/cut) = b_i - u - ln(cut/pT_ref)`` is the distance to the
    threshold in the leg's own logarithmic variable, and the resolution is
    evaluated at the TRUE post-FSR ``pT = cut e^v``, which is what the law of
    ``r = pT^reco/pT^gen - 1`` is conditioned on.
    """
    v = np.asarray(b_src, float)[:, None] - np.asarray(b_out, float)[None, :] - s
    return resol.pass_prob(v, cut, iband)


def smeared_G(h4, b_src, eta_edges, b_out, terms, pt_ref, resol):
    """``G`` on the grid nodes with the resolution folded into the acceptance.

    ``h4[a, i, c, j]`` is the weight of the ``(eta, b)`` pair cell, so

        G(u_+, u_-) = sum_terms sign sum_{a,i,c,j} h4 Pi(b_i - u_+ - s_+, eta_a)
                                                      Pi(b_j - u_- - s_-, eta_c)

    and the inner double sum is the matrix triple product
    ``Pi_a^T h4[a, :, c, :] Pi_c``.  The ``b`` argument of each cell is its
    midpoint: the first-order error of that cancels against the symmetric part
    of the cell's own content, which is what makes a coarse ``b`` grid usable
    once the step is smoothed (checked against a direct event count by
    `fsr_perleg.py gcheck`).
    """
    ne = len(eta_edges) - 1
    shifts = sorted({s for t in terms for s in t[:2]})
    P = {s: [pass_matrix(resol, pt_ref * np.exp(s), a, b_src[a], b_out, s)
             for a in range(ne)] for s in shifts}
    n = len(b_out)
    G = np.zeros((n, n))
    for a in range(ne):
        for c in range(ne):
            M = np.asarray(h4[a, :, c, :], float)
            if not M.any():
                continue
            for sp, sm, sgn in terms:
                G += sgn * (P[sp][a].T @ (M @ P[sm][c]))
    return G


class PassRegion:
    """``G(u_+, u_-|m)`` for one selection, on the ``h`` table's own node grid.

    It is the only object that knows what the selection is; everything
    downstream -- `ksel_band`, `gbar`, `matched_gbar` -- takes the array it
    returns and never asks how it was built.  Three cases, in increasing
    generality:

    * symmetric step: the plain 2-D survival function of ``h``, unchanged;
    * asymmetric step: the union of two rectangles, three shifted readings;
    * with a resolution: the step is replaced by the per-leg pass probability
      of `ptres.Resolution`, which needs the ``eta`` of each leg and therefore
      the 4-D table ``h4(b_+, eta_+, b_-, eta_-|m)``.
    """

    def __init__(self, ht, cuts=None, h4=None, resol=None):
        self.b_edges = np.asarray(ht["b_edges"], float)
        self.pt_ref = float(ht["pt_ref"])
        if cuts is None:
            self.cuts = (self.pt_ref, self.pt_ref)
        else:
            c = np.atleast_1d(np.asarray(cuts, float))
            self.cuts = ((float(c[0]), float(c[0])) if c.size == 1
                         else (float(np.max(c)), float(np.min(c))))
        self.shifts = cut_shifts(self.cuts, self.pt_ref)
        self.terms = rect_terms(*self.shifts)
        self.H = np.asarray(ht["h"], float)
        self.resol = resol
        self.h4 = h4
        if resol is not None:
            if h4 is None:
                raise ValueError("a resolution needs the h4 table (--h4)")
            self.h4_b = np.asarray(h4["b_edges"], float)
            self.h4_eta = np.asarray(h4["eta_edges"], float)
            b = self.h4_b
            mid = np.concatenate([0.5 * (b[:-1] + b[1:]),
                                  [b[-1] + 0.5 * (b[-1] - b[-2])]])
            self.H4 = h4["h4"]
            self.h4_mean = h4.get("b_mean")
            if self.h4_mean is None:
                self.h4_mean = np.broadcast_to(
                    mid, (len(self.H4), len(self.h4_eta) - 1, len(mid)))
        if self.shifts[0] < self.b_edges[0] or self.shifts[1] < self.b_edges[0]:
            raise ValueError(f"cuts {self.cuts} below the table's pT floor "
                             f"{self.pt_ref * np.exp(self.b_edges[0]):.3f} GeV")

    def grid(self, k):
        """``G`` of band ``k`` on ``b_edges`` x ``b_edges``."""
        if self.resol is None:
            return step_G(self.H[k], self.b_edges, self.terms)
        return smeared_G(np.asarray(self.H4[k], float),
                         np.asarray(self.h4_mean[k], float),
                         self.h4_eta, self.b_edges, self.terms, self.pt_ref,
                         self.resol)

    def label(self):
        c = self.cuts
        s = f"pT > {c[0]:g}/{c[1]:g}"
        return s + (f", resolution ({self.resol.mode})" if self.resol else
                    ", step")


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
# the correlated two-leg density: the exact O(alpha) recoil sharing
# --------------------------------------------------------------------------
# ``D(x_+) D(x_-)`` is the collinear limit, in which a photon is emitted by one
# leg and the other leg does not know.  It is exact at O(alpha) *in that limit*
# -- the exact single-photon kinematics put the two energy fractions on the LINE
#
#     x_+ + x_- = 1 + z ,   z = (m_post/m_pre)^2                     (exact)
#
# in the pre-FSR rest frame, and the collinear end points of that line are
# ``(1, z)`` and ``(z, 1)``, all-on-one-leg.  What the product misses is the
# ~1/L of the rate away from the end points, where the photon takes energy from
# **both** muons.  Under a lepton ``p_T`` cut that matters, because the pass
# decision is taken on the two legs separately.
#
# The correlated model replaces the product by
#
#     z ~ K(z) ,   f ~ p_1(f | z) ,
#     x_+ = 1 - (1 - z) f ,   x_- = 1 - (1 - z)(1 - f) ,
#
# with ``p_1`` the exact spin-summed O(alpha) matrix element of
# `fsr_analytic.share_nodes`.  Two properties:
#
#   * the ``z`` marginal is **exactly** the kernel handed in -- ``int df p_1 = 1``
#     at every ``z`` -- so ``K(z)`` is untouched by construction and the inclusive
#     fit is unchanged.  Nothing is refitted, rescaled or matched;
#   * the mass is the exact one.  ``z = x_+ x_-`` is *not* imposed (it is the
#     collinear corollary of the product and is wrong off axis by
#     ``(1-z)^2/8`` at ``f = 1/2``); the model carries ``z`` itself.
#
# The selected kernel is then the inclusive kernel reweighted atom by atom,
#
#     K_sel(u | m) = K(u | m) Gbar(u | m) ,
#     Gbar(u | m)  = int df p_1(f | z) G(u_+(z, f), u_-(z, f) | m) ,
#     A(m)         = int du K(u|m) Gbar(u|m) ,          z = e^{-2u} ,
#
# with the **same** ``G`` -- the 2-D survival function of the pre-FSR muon
# transverse momenta -- and the same ``h`` table.  The boson-kinematics
# interface does not change: the pass decision is still
# ``x_q p_T,q^{pre} > p_T^cut``, whose error against the true post-FSR decision
# is +1.9e-4 on ``A`` and +0.13/+0.03 MeV at fit level (README, check (e)).
#
# ``G`` is symmetric because the ``h`` table is symmetrised, and ``p_1`` is
# exactly symmetric under ``f -> 1 - f``, so only the half branch is summed and
# the result doubled.
def share_x(z, f):
    """``(x_+, x_-)`` on the exact single-photon line ``x_+ + x_- = 1 + z``."""
    d = (1.0 - z)[:, None] if np.ndim(z) else (1.0 - z)
    return 1.0 - d * f, 1.0 - d * (1.0 - f)


def eval_G_at(g, b_edges, up, um):
    """``G(u_+, u_-)`` at paired points, bilinear on the table's own grid."""
    nb = len(b_edges) - 1
    i = np.clip(np.searchsorted(b_edges, up, "right") - 1, 0, nb - 1)
    s = np.clip((up - b_edges[i]) / (b_edges[i + 1] - b_edges[i]), 0.0, 1.0)
    j = np.clip(np.searchsorted(b_edges, um, "right") - 1, 0, nb - 1)
    t = np.clip((um - b_edges[j]) / (b_edges[j + 1] - b_edges[j]), 0.0, 1.0)
    return ((g[i, j] * (1.0 - t) + g[i, j + 1] * t) * (1.0 - s)
            + (g[i + 1, j] * (1.0 - t) + g[i + 1, j + 1] * t) * s)


def G_norad(g, b_edges):
    """``G(0, 0)``: the pass probability of an event that radiated nothing.

    ``u = 0`` is a node of every grid built by `b_grid`, but it is NOT node 0
    once the grid reaches below the threshold (which it must, for a resolution
    to be able to promote a muon over the cut), so it is read rather than
    indexed.
    """
    return float(eval_G_at(g, b_edges, np.zeros(1), np.zeros(1))[0])


def gbar(m, u_nodes, g, b_edges, npanel=64, ng=8, chunk=512):
    """``Gbar(u)``: the selection weight of the pair variable, sharing folded in.

    ``u = 0`` is ``G(0, 0)`` -- nothing is radiated, the sharing has no meaning
    and the matrix element is 0/0 there -- so it is taken from the table
    directly and the quadrature starts at the first finite node.
    """
    u_nodes = np.asarray(u_nodes, float)
    z = np.exp(-2.0 * u_nodes)
    out = np.empty(len(u_nodes))
    out[u_nodes <= 0.0] = G_norad(g, b_edges)
    for i0 in range(0, len(u_nodes), chunk):
        sl = slice(i0, min(i0 + chunk, len(u_nodes)))
        if u_nodes[sl][-1] <= 0.0:
            continue
        zs = np.maximum(z[sl], 0.0)
        pos = zs < 1.0
        f, w = FA.share_nodes(m, zs[pos], npanel=npanel, ng=ng)
        xp, xm = share_x(zs[pos], f)
        up = -np.log(np.maximum(xp, 1e-300))
        um = -np.log(np.maximum(xm, 1e-300))
        o = out[sl]
        o[pos] = 2.0 * np.sum(w * eval_G_at(g, b_edges, up, um), axis=1)
        out[sl] = o
    return out


def shift_matrix(b_edges, u_p, w_p):
    """``M_{ij}``: reading ``sum_p w_p f(b_i + u_p)`` off the grid values ``f_j``.

    The bilinear read is linear in the table, so convoluting a per-leg measure
    onto **both** legs of ``G`` is the matrix triple product ``M G M^T`` -- one
    dense 290x290 product per leg instead of a double sum over the ladder.
    Beyond the last node ``G`` is constant (the table's last bin holds
    everything above it), which the clipping reproduces.
    """
    nb = len(b_edges) - 1
    b = b_edges[:, None] + np.asarray(u_p, float)[None, :]
    i = np.clip(np.searchsorted(b_edges, b, "right") - 1, 0, nb - 1)
    t = np.clip((b - b_edges[i]) / (b_edges[i + 1] - b_edges[i]), 0.0, 1.0)
    M = np.zeros((nb + 1, nb + 1))
    wp = np.broadcast_to(np.asarray(w_p, float)[None, :], b.shape)
    for r in range(nb + 1):
        M[r] = (np.bincount(i[r], wp[r] * (1.0 - t[r]), nb + 1)
                + np.bincount(i[r] + 1, wp[r] * t[r], nb + 2)[:nb + 1])
    return M


def corr_cells(cells, u_nodes, gb, u_min):
    """Reweight kernel cells ``(w, int u, int u^2)`` by ``Gbar`` at their mean."""
    w, m1, m2 = np.asarray(cells, float)
    ok = w > 0.0
    u = np.zeros_like(w)
    u[ok] = m1[ok] / w[ok]
    lg = np.interp(np.log(np.maximum(u, u_min)), np.log(u_nodes[1:]), gb[1:],
                   left=gb[0], right=gb[-1])
    lg = np.where(u <= u_min, gb[0], lg)
    return np.stack([w * lg, m1 * lg, m2 * lg])


# --------------------------------------------------------------------------
# the fine cells of the two kernel configurations
# --------------------------------------------------------------------------
def analytic_cells(m, variant="exp2nll", pair=(), minus_one=True,
                   pair_table=None, var_budget=6e-10, n_fine=4000, ng=16):
    """``(cells, w_delta, u_max)`` of `fsr_analytic.FSRKernel` at mass ``m``."""
    k = FSRKernel(m, variant=variant, pair=tuple(pair), minus_one=minus_one,
                  pair_table=pair_table)
    return k.cells(var_budget=var_budget, n_fine=n_fine, ng=ng), 0.0, k._u_max()


def tabulated_cells(run, bands, u_tail0=2.0, u_tail_w=0.05):
    """``(cells, w_delta, u_max)`` per band of a standalone Photos run.

    One cell per filled histogram bin at the bin's exact first and second
    moment, plus the genuine ``delta(u)`` of the events the generator left
    untouched.  An ``h``-table band is served by the run band its centre falls
    in, exactly as `_mc_leg_cells` does for the leg.
    """
    d = np.load(run, allow_pickle=True)
    bl, bh = np.asarray(d["bands_lo"], float), np.asarray(d["bands_hi"], float)
    n = np.asarray(d["n"], float)
    key = "n_noemit" if "n_noemit" in d.files else "n_nophot"
    f0, f1, f2 = d["fine_s0"], d["fine_s1"], d["fine_s2"]
    t0, t1, t2 = d["tail_s0"], d["tail_s1"], d["tail_s2"]
    wn = np.asarray(d[key], float)
    out = []
    for lo, hi in bands:
        mc = 0.5 * (lo + hi)
        if not np.isfinite(mc):
            mc = float(lo) * 1.05
        j = np.nonzero((bl <= mc) & (bh > mc))[0]
        j = int(j[0]) if len(j) else int(np.argmin(np.abs(0.5 * (bl + bh) - mc)))
        nk = n[j] if n[j] > 0 else 1.0
        s0 = np.concatenate([f0[j], t0[j]]).astype(float) / nk
        s1 = np.concatenate([f1[j], t1[j]]).astype(float) / nk
        s2 = np.concatenate([f2[j], t2[j]]).astype(float) / nk
        wd = float(wn[j] / nk)
        s0[0] = max(s0[0] - wd, 0.0)               # the delta is split off
        keep = s0 > 0.0
        out.append((np.stack([s0[keep], s1[keep], s2[keep]]), wd,
                    u_tail0 + u_tail_w * len(t0[j])))
    return out


# --------------------------------------------------------------------------
# the matched construction: exact hard sharing + collinear soft remainder
# --------------------------------------------------------------------------
def hard_spectrum(m, u_c, u_max, n=600, ng=8, var_budget=1e-11):
    """``(u_h, w_h, N)`` of the exact O(alpha) emission above ``u_c``.

    ``N = int_{u > u_c} R_1 du`` is the hard rate, and the measure returned
    carries it; the matching delta carries ``1 - N``.
    """
    e = np.geomspace(u_c, u_max, n + 1)
    g, wq = np.polynomial.legendre.leggauss(ng)
    a, b = np.log(e[:-1])[:, None], np.log(e[1:])[:, None]
    lu = 0.5 * (b - a) * (g[None, :] + 1.0) + a
    wt = 0.5 * (b - a) * wq[None, :]
    u = np.exp(lu)
    z = np.exp(-2.0 * u)
    # R_1(z) dz as a density in u: K_u(u) = 2 z R_1(z), and u dln u = du
    r = FA.spec_gammastar(m, z.ravel(), np.zeros(z.size), ng=24).reshape(z.shape)
    f = wt * u * 2.0 * z * r
    uu, ww = _merge_cells(np.stack([f.sum(1), (f * u).sum(1),
                                    (f * u * u).sum(1)]), var_budget)
    return uu, ww, float(ww.sum())


#: grid of the matched construction.  Coarser than `KSQRT_DU` on purpose: the
#: soft remainder only has to smear ``G``, whose own resolution is 1e-3 in the
#: leg variable, and the *analytic* kernel is a comb on a 2e-5 grid above
#: ``u`` ~ 2e-3 (its cells are geometric), which a square root cannot divide.
MATCH_DU = 1e-4


def kernel_grid(k, du, n, pair_atoms=None):
    """A **dense** uniform-grid image of an analytic kernel.

    Each grid cell carries the kernel's own integral over that cell, not a
    deposited atom, so there is no comb at any ``u`` -- which is what the
    convolution square root of the matched construction needs.  The pair branch
    is convoluted on the grid rather than multiplied cell by cell.
    """
    ue = np.minimum(np.arange(n + 1) * du, k._u_max())
    w = k._cells(k._t_of_u(ue), 16)[0]
    if pair_atoms is not None and len(pair_atoms[0]):
        uk, wk = pair_atoms
        P = deposit(np.concatenate([[0.0], np.clip(uk, 0.0, (n - 2) * du)]),
                    np.concatenate([[1.0 - wk.sum()], wk]), du, n)
        L = _fft_len(2 * n)
        w = np.fft.irfft(np.fft.rfft(w, L) * np.fft.rfft(P, L), L)[:n]
    return w / w.sum()


def soft_leg(Kg, u_h, w_h, n_hard, du=MATCH_DU, n_leg=1200, u_min=1e-9):
    """``(u_p, w_p)`` of ``D_<`` defined by ``D_< (x) D_< (x) H_> = K``.

    ``H_>`` is the hard measure ``(1 - N) delta + (u_h, w_h)``.  The
    deconvolution and the square root are one spectral operation,
    ``D^_< = sqrt(K^ / H^_>)``, on the uniform grid `conv_sqrt` uses; the
    matching keeps ``|H^_>|`` bounded away from zero as long as ``N < 1/2``,
    which is what limits how low the matching scale can go.
    """
    n = len(Kg)
    H = deposit(np.concatenate([[0.0], np.clip(u_h, 0.0, (n - 2) * du)]),
                np.concatenate([[1.0 - n_hard], w_h]), du, n)
    L = _fft_len(2 * n)
    R = np.fft.rfft(Kg, L) / np.fft.rfft(H, L)
    D = np.fft.irfft(np.sqrt(R), L)[:n]
    ue = np.concatenate([[0.0], np.geomspace(u_min, 2.0 * n * du, n_leg)])
    c = leg_cells_from_grid(D, du, ue)
    u, w = leg_atoms(c, ue)
    return u, w / w.sum()


def ladder_groups(u, n_v):
    """Split an ordered leg ladder into ``n_v`` contiguous groups.

    The ladder is geometric in ``u``, so equal counts is geometric binning; the
    unradiated node ``u = 0`` is kept on its own because it carries most of the
    soft remainder's weight and must not be averaged with anything.
    """
    i0 = 1 if (len(u) and u[0] <= 0.0) else 0
    rest = np.arange(i0, len(u))
    out = ([np.arange(i0)] if i0 else []) + [
        s for s in np.array_split(rest, max(n_v - i0, 1)) if len(s)]
    return [s for s in out if len(s)]


def matched_gbar(m, u_bin_edges, g, b_edges, Kg, u_c, u_max,
                 n_v=16, npanel=16, ng=4, n_leg=1200, du=MATCH_DU):
    """``Gbar(u)`` of the matched construction, on ``u_bin_edges``.

    ``K = D_< (x) D_< (x) H_>`` with ``H_>`` the exact O(alpha) emission above
    ``u_c`` carrying its exact sharing and ``D_<`` the soft-collinear remainder,
    independent leg by leg.  The total is ``u = u_h + (u_p + u_q)/2`` and the
    two legs are ``u_q = u_q^h + u_q^<``, so

        Gbar(u) = < G(u_+, u_-) | u >

    is the ratio of two sums over the same configurations, and the
    discretisation of the soft ladder cancels between them.  The inner double
    sum over the soft ladder is done group by group with `shift_matrix`, so its
    cost is ``n_v^2`` dense products and not ``n_leg^2`` per hard configuration.
    """
    uh, wh, N = hard_spectrum(m, u_c, u_max)
    up_, wp_ = soft_leg(Kg, uh, wh, N, du=du, n_leg=n_leg)
    # hard configurations: the matching delta, then (hard atom) x (sharing node)
    f, wf = FA.share_nodes(m, np.exp(-2.0 * uh), npanel=npanel, ng=ng)
    xp, xm = share_x(np.exp(-2.0 * uh), f)
    A = np.concatenate([[0.0], (-np.log(np.maximum(xp, 1e-300))).ravel()])
    B = np.concatenate([[0.0], (-np.log(np.maximum(xm, 1e-300))).ravel()])
    WH = np.concatenate([[0.5 * (1.0 - N)], (wh[:, None] * wf).ravel()])
    UH = np.concatenate([[0.0], np.repeat(uh, f.shape[1])])
    grp = ladder_groups(up_, n_v)
    M = [shift_matrix(b_edges, up_[s], wp_[s]) for s in grp]
    ub = [float(np.sum(wp_[s] * up_[s]) / np.sum(wp_[s])) for s in grp]
    wb = [float(np.sum(wp_[s])) for s in grp]
    nb = len(u_bin_edges) - 1
    num = np.zeros(nb)
    den = np.zeros(nb)
    sum_u = np.zeros(nb)
    for j in range(len(grp)):
        Aj = M[j] @ g
        for k in range(len(grp)):
            P = Aj @ M[k].T
            v = UH + 0.5 * (ub[j] + ub[k])
            i = np.clip(np.searchsorted(u_bin_edges, v, "right") - 1, 0, nb - 1)
            # both orientations of the half sharing branch: G is symmetric and
            # the (j, k) sum is, so one read and a factor two
            num += np.bincount(i, 2.0 * WH * eval_G_at(P, b_edges, A, B), nb)
            w = 2.0 * WH * wb[j] * wb[k]
            den += np.bincount(i, w, nb)
            sum_u += np.bincount(i, w * v, nb)
    return num, den, sum_u


#: extent of the matched construction's uniform grid, in the pair variable
MATCH_GRID_MAX = 8.0


def rebin_cells(cells, w_delta, du, n):
    """A tabulated kernel's cells on a uniform grid, cell by cell (dense)."""
    w, m1, _ = np.asarray(cells, float)
    ok = w > 0.0
    u = np.zeros_like(w)
    u[ok] = m1[ok] / w[ok]
    g = deposit(np.concatenate([[0.0], np.clip(u[ok], 0.0, (n - 2) * du)]),
                np.concatenate([[w_delta], w[ok]]), du, n)
    return g / g.sum()


def _fill_ratio(num, den, sum_u, u_nodes, g00, u_min=1e-9):
    """``Gbar`` on ``u_nodes`` from the binned conditional expectation.

    The bins are the model's own configurations, so the numerator and the
    denominator carry the same discretisation and it cancels in the ratio; each
    filled bin is attached to its weighted mean ``u`` and the ladder is read off
    by interpolation in ``ln u``.
    """
    ok = den > 0.0
    if not ok.any():
        return np.full(len(u_nodes), g00)
    r = num[ok] / den[ok]
    ub = np.maximum(sum_u[ok] / den[ok], u_min)
    o = np.argsort(ub)
    out = np.interp(np.log(np.maximum(u_nodes, u_min)), np.log(ub[o]), r[o],
                    left=g00, right=r[o][-1])
    out[u_nodes <= 0.0] = g00
    return out


# --------------------------------------------------------------------------
# the multi-emission two-leg density: the exponentiated exact O(alpha) sharing
# --------------------------------------------------------------------------
# The single-photon model above shares the WHOLE loss as if it were one photon;
# the collinear product ``D (x) D`` shares it as if every photon were collinear.
# Both are limits of one object.  Photon energies ADD, so in the pre-FSR rest
# frame the two leg losses
#
#     eps_q = 1 - x_q ,    eps_+ + eps_- = eps = 2 sum_i E_i / m       (exact)
#
# are sums over photons, and the two-leg law is the 2-D compound Poisson whose
# Levy measure is the exact O(alpha) emission density carrying its exact
# sharing,
#
#     nu_2(eps_+, eps_-) = nu(delta) p_1(f | 1 - delta) ,
#                          eps_+ = delta f ,  eps_- = delta (1 - f) ,
#     log P^(s_+, s_-)   = nu_2^(s_+, s_-) - N ,   N = int nu_2 .
#
# Three properties, none of them fitted and none of them a matching:
#
#   * the MARGINAL IDENTITY is exact.  On the diagonal ``s_+ = s_- = s`` the
#     sharing integrates to one at every ``delta``, so
#     ``log P^(s,s) = nu^(s) - N = log K^(s)``: the total-loss law is the
#     kernel that was handed in, to machine precision, and the inclusive fit is
#     untouched;
#   * at O(alpha) the law IS ``nu_2``, the exact matrix element's angular
#     distribution -- no collinear approximation in the recoil;
#   * in the collinear limit ``p_1 -> [delta(f) + delta(f-1)]/2`` it collapses
#     EXACTLY to ``D (x) D``: ``log P^ = [nu^(s_+) - N]/2 + [nu^(s_-) - N]/2``
#     ``= log K^(s_+)^{1/2} + log K^(s_-)^{1/2}``, and ``K^{1/2}`` is the leg
#     radiator.  The construction interpolates between the two limits with no
#     matching scale and no subtraction, and it is a positive measure wherever
#     ``nu`` is.
#
# ``nu`` is read off the kernel by the same spectral operation the convolution
# square root is -- ``log`` in place of ``sqrt`` -- on a uniform grid in
# ``eps``, which is the additive variable.  ``K`` is infinitely divisible in it
# by construction (it is the exponential of a Mellin exponent), so ``nu >= 0``
# up to the grid's own representation error, which is measured.
#
# The convention for the multi-photon part is forced by the same additivity:
# the model's mass is ``z = 1 - eps`` with ``eps = eps_+ + eps_-``, which is
# EXACT for one photon and neglects ``(sum k)^2/m^2`` beyond it -- the same
# quantity the generator record measures at -8.0e-3 on the 1 % tail of the
# radiating events.  The collinear product's own convention, ``z = x_+ x_-``,
# differs from it by ``eps_+ eps_-`` and is exact for collinear photons and
# wrong at O(alpha) off axis by ``(1-z)^2/8`` at ``f = 1/2``.

#: uniform step of the ``eps`` grid.  ``eps = 1 - z`` runs over [0, 1], so the
#: node count is ``1/MULTI_H``; the step has to resolve ``G``, whose own
#: resolution is 1e-3 in the leg variable.
MULTI_H = 5e-4


def eps_of_u(u):
    return -np.expm1(-2.0 * np.asarray(u, float))


def u_of_eps(e):
    return -0.5 * np.log1p(-np.minimum(np.asarray(e, float), 1.0 - 1e-300))


def kernel_eps_grid(k, h, n, pair_atoms=None):
    """A **dense** uniform-``eps`` image of an analytic kernel.

    Each cell carries the kernel's own integral over that cell, so there is no
    comb at any ``eps`` -- which is what the logarithm of the transform needs.
    The pair branch is convoluted on the grid; ``eps`` is the additive variable
    for a lepton pair exactly as it is for a photon.
    """
    ee = np.minimum(np.arange(n + 1) * h, eps_of_u(k._u_max()))
    w = k._cells(k._t_of_u(u_of_eps(ee)), 16)[0]
    if pair_atoms is not None and len(pair_atoms[0]):
        uk, wk = pair_atoms
        P = deposit(np.concatenate([[0.0], np.clip(eps_of_u(uk), 0.0,
                                                   (n - 2) * h)]),
                    np.concatenate([[1.0 - wk.sum()], wk]), h, n)
        L = _fft_len(2 * n)
        w = np.fft.irfft(np.fft.rfft(w, L) * np.fft.rfft(P, L), L)[:n]
    return w / w.sum()


def eps_grid_cells(cells, w_delta, h, n):
    """A tabulated kernel's cells on the uniform ``eps`` grid, cell by cell.

    Atoms above the grid are **dropped**, not clipped, and the result is
    renormalised: the Levy measure obeys a causal recursion,
    ``k g_k = sum_j j nu_j g_{k-j}``, so ``nu`` below ``eps_max`` is the same
    for the kernel and for the kernel conditioned on ``eps < eps_max`` -- which
    is what lets a fine short grid resolve the sharing at small ``u``, and
    which piling the tail into the last cell would destroy.
    """
    w, m1, _ = np.asarray(cells, float)
    ok = w > 0.0
    u = np.zeros_like(w)
    u[ok] = m1[ok] / w[ok]
    e = eps_of_u(u[ok])
    keep = e < (n - 1) * h
    g = deposit(np.concatenate([[0.0], e[keep]]),
                np.concatenate([[w_delta], w[ok][keep]]), h, n)
    return g / g.sum()


def levy(g):
    """``(nu, N, info)`` with ``g = exp_*(nu - N e_0)`` on ``g``'s own grid.

    The same spectral operation as `conv_sqrt`, with ``log`` in place of
    ``sqrt``: ``log g^ = nu^ - N`` pointwise.  ``nu[0] = 0`` and
    ``N = -log g[0]`` come out of it rather than being imposed -- the
    ``n``-photon term of ``exp_*`` starts at node ``n``, so node 0 sees only
    ``e^{-N}``.
    """
    n = len(g)
    L = _fft_len(4 * n)
    G = np.zeros(L)
    G[:n] = g
    Gh = np.fft.rfft(G)
    lam = np.fft.irfft(np.log(Gh), L)
    N = -float(lam[0])
    nu = lam.copy()
    nu[0] = 0.0
    rt = np.fft.irfft(np.exp(np.fft.rfft(nu) - N), L)[:n]
    info = dict(N=N, k_abs_min=float(np.abs(Gh).min()),
                k_arg_max=float(np.abs(np.angle(Gh)).max()),
                neg=float(nu[nu < 0.0].sum()),
                wrap=float(np.abs(nu[n:]).sum()),
                roundtrip=float(np.abs(rt - g).max()))
    return nu[:n], N, info


def levy_recursion(g):
    """``(nu, N)`` by the causal moment recursion ``k g_k = sum_j j nu_j g_{k-j}``.

    ``nu[1..k]`` depends only on ``g[0..k]``, so a grid truncated at some
    ``eps_max`` carries the same Levy measure below it as the full one -- which
    is what lets a *fine* short grid be used for the sharing diagnostics.
    O(n^2), and the cross-check of the spectral branch.
    """
    g = np.asarray(g, float)
    n = len(g)
    nu = np.zeros(n)
    jn = np.zeros(n)
    for k in range(1, n):
        nu[k] = ((k * g[k] - float(np.dot(jn[1:k], g[k - 1:0:-1])))
                 / (k * g[0]))
        jn[k] = k * nu[k]
    return nu, -math.log(g[0])


def deposit_diag(kk, p, w, n):
    """Atoms onto the 2-D grid **along their own anti-diagonal**.

    An atom of total loss ``kk h`` at ``eps_+ = p h`` is split between the
    nodes ``(i, kk - i)`` and ``(i + 1, kk - i - 1)``, both of which have
    ``i + j = kk`` exactly.  The projection onto the total is therefore the
    1-D measure the atoms came from, to machine precision -- which is what
    makes the marginal identity exact -- and the first moment in ``eps_+`` is
    preserved as well.
    """
    kk = np.asarray(kk, np.int64)
    i = np.floor(p).astype(np.int64)
    t = np.asarray(p, float) - i
    i = np.clip(i, 0, kk)
    hi = np.minimum(i + 1, kk)
    j, jh = kk - i, kk - hi
    ok = (i < n) & (j < n)
    A = np.bincount((i[ok] * n + j[ok]), (w * (1.0 - t))[ok], n * n)
    ok = (hi < n) & (jh < n)
    A += np.bincount((hi[ok] * n + jh[ok]), (w * t)[ok], n * n)
    return A.reshape(n, n)


def share_measure(m, w, h, n, npanel=16, ng=4, mmu=M_MU):
    """``nu_2`` (or ``S_1``) from a 1-D measure ``w`` on the ``eps`` grid.

    Every node ``k >= 1`` carries a loss ``delta = k h``; the exact O(alpha)
    sharing of that loss between the two legs is `fsr_analytic.share_nodes` at
    ``z = 1 - delta``, tabulated on the half branch ``f <= 1/2`` and mirrored.
    Below the ``2 m_mu`` threshold the muons are at rest in their own frame and
    the sharing is ``f = 1/2``.
    """
    k = np.arange(1, n)
    d = k * h
    use = (w[1:n] != 0.0) & (d < 1.0)
    out = np.zeros((n, n))
    if not use.any():
        return out
    kk, dd, ww = k[use], d[use], w[1:n][use]
    z = 1.0 - dd
    live = z * m * m > 4.0 * mmu * mmu
    if live.any():
        f, wf = FA.share_nodes(m, z[live], npanel=npanel, ng=ng, mmu=mmu)
        K = np.repeat(kk[live], f.shape[1])
        W = (ww[live][:, None] * wf).ravel()
        P = (np.repeat(dd[live], f.shape[1]) * f.ravel()) / h
        out += deposit_diag(K, P, W, n)                       # f  <= 1/2
        out += deposit_diag(K, np.repeat(kk[live], f.shape[1]) - P, W, n)
    if (~live).any():
        K = kk[~live]
        out += deposit_diag(K, 0.5 * K, ww[~live], n)
    return out


def coll_measure(w, n):
    """The collinear limit of `share_measure`: half the loss on each end point.

    The 2-D exponential of this is ``D (x) D`` exactly, with ``D`` the
    convolution square root of the kernel in ``eps``.
    """
    out = np.zeros((n, n))
    k = np.arange(1, n)
    out[k, 0] += 0.5 * w[1:n]
    out[0, k] += 0.5 * w[1:n]
    return out


def multi_law(nu2, N, n, dtype=np.float64):
    """``exp_*(nu_2 - N e_0)`` on the 2-D grid, by FFT.

    The grid is padded to twice its extent: a configuration aliasing back into
    ``eps_+ < 1`` needs a total loss above 2, i.e. three photons each taking
    the whole mass, and carries no weight.  The wrap is measured by the
    marginal identity.
    """
    L = _fft_len(2 * n)
    A = np.zeros((L, L), dtype)
    A[:n, :n] = nu2
    F = np.fft.rfft2(A)
    del A
    F -= N
    np.exp(F, out=F)
    P = np.fft.irfft2(F, s=(L, L))
    del F
    return np.ascontiguousarray(P[:n, :n])


def conv2(A, B, n):
    """The 2-D convolution of two measures on the ``eps`` grid, truncated."""
    L = _fft_len(2 * n)
    X = np.zeros((L, L))
    X[:n, :n] = A
    F = np.fft.rfft2(X)
    X[:] = 0.0
    X[:n, :n] = B
    F *= np.fft.rfft2(X)
    del X
    return np.ascontiguousarray(np.fft.irfft2(F, s=(L, L))[:n, :n])


def G_eps_matrix(g, b_edges, h, n):
    """``G(u_+, u_-)`` on the ``eps`` grid, bilinear on the table's own grid."""
    u = np.minimum(u_of_eps(np.arange(n) * h), b_edges[-1])
    nb = len(b_edges) - 1
    i = np.clip(np.searchsorted(b_edges, u, "right") - 1, 0, nb - 1)
    s = np.clip((u - b_edges[i]) / (b_edges[i + 1] - b_edges[i]), 0.0, 1.0)
    r0 = g[i][:, i] * (1.0 - s) + g[i][:, i + 1] * s
    r1 = g[i + 1][:, i] * (1.0 - s) + g[i + 1][:, i + 1] * s
    return r0 * (1.0 - s)[:, None] + r1 * s[:, None]


def diag_reduce(P, Gm, n):
    """``(num, den)`` per anti-diagonal ``eps_+ + eps_- = k h``."""
    i = np.arange(n, dtype=np.int64)
    kk = (i[:, None] + i[None, :]).ravel()
    den = np.bincount(kk, P.ravel(), 2 * n)[:n]
    num = np.bincount(kk, (P * Gm).ravel(), 2 * n)[:n]
    return num, den


def multi_laws(m, Ke, h, n, modes, npanel=16, ng=4):
    """Yield ``(mode, P, info)``: the two-leg laws that share ``Ke``.

    One law is alive at a time.  ``modes`` picks the sharing:

      ``single``  the whole loss carried by one exact-angle photon (the `corr`
                  model), i.e. ``S_1`` without exponentiation;
      ``multi``   the exponentiated exact sharing (this section);
      ``coll``    the exponentiated collinear sharing, which is ``D (x) D``;
      ``lin``     the same correction at first order only, ``D (x) D`` plus one
                  wide-angle emission.
    """
    nu, N, info = levy(Ke)
    for md in modes:
        if md == "single":
            P = share_measure(m, Ke, h, n, npanel, ng)
            P[0, 0] += Ke[0]                # the unradiated atom, no sharing
        elif md == "multi":
            P = multi_law(share_measure(m, nu, h, n, npanel, ng), N, n)
        elif md == "coll":
            P = multi_law(coll_measure(nu, n), N, n)
        elif md == "lin":
            # the LINEARISED matching: D (x) D plus one wide-angle correction,
            # ``P = D (x) D + [D (x) D] (x) dnu_2`` with
            # ``dnu_2 = nu_2^exact - nu_2^collinear``.  It is the first term of
            # ``exp_*(dnu_2)`` acting on the collinear law, so it has the same
            # O(alpha) angle and the same exact z marginal (``dnu_2`` has zero
            # projection on every anti-diagonal), and differs from ``multi`` by
            # the multi-WIDE-angle configurations only -- which is the size of
            # the resummation of the sharing.  Unlike ``multi`` it is a signed
            # measure.
            dn = (share_measure(m, nu, h, n, npanel, ng)
                  - coll_measure(nu, n))
            P = multi_law(coll_measure(nu, n), N, n)
            P += conv2(P, dn, n)
        else:
            raise ValueError(md)
        yield md, P, info


def share_q(m, w, h, n, npanel=16, ng=4, mmu=M_MU):
    """``sum_i delta_i^2 f_i (1-f_i)`` carried by a 1-D measure ``w``, per node.

    The mass of an ``n``-photon configuration is
    ``z = 1 - eps + (sum k)^2/m^2``, and the light-cone decomposition of each
    photon along the two muon directions gives, averaged over the relative
    azimuth,

        (sum k)^2/m^2 = eps_+ eps_- - sum_i eps_+^i eps_-^i ,
        x_+ x_- - z   = sum_i eps_+^i eps_-^i = sum_i delta_i^2 f_i (1-f_i) .

    So ``R = <x_+ x_- - z> / <eps_+ eps_->`` is 1 when one photon carries the
    loss at any angle (``z = 1 - eps``) and 0 when every photon is collinear to
    one leg (``z = x_+ x_-``): it says which mass-leg relation a configuration
    obeys, and it is measurable on the generator record as well as on the model.
    """
    k = np.arange(1, n)
    d = k * h
    out = np.zeros(n)
    use = (w[1:n] != 0.0) & (d < 1.0) & ((1.0 - d) * m * m > 4.0 * mmu * mmu)
    if not use.any():
        return out
    f, wf = FA.share_nodes(m, 1.0 - d[use], npanel=npanel, ng=ng, mmu=mmu)
    out[1:n][use] = w[1:n][use] * d[use] ** 2 * (2.0 * (wf * f
                                                       * (1.0 - f)).sum(1))
    return out


def multi_R(m, Ke, nu, P, h, n, npanel=16, ng=4, exclusive=False):
    """``(R, num, den)``: the model's own mass-leg ratio, per anti-diagonal.

    For a compound Poisson the first moment of an additive functional at fixed
    total is ``E[sum_i g_i ; total = eps] = (nu_g (x) P_total)(eps)`` (Mecke),
    and ``P_total`` is the kernel itself -- so the numerator is one 1-D
    convolution.  The denominator ``<eps_+ eps_- ; eps>`` is read off the same
    2-D law the selection weight is.
    """
    nug = share_q(m, nu, h, n, npanel, ng)
    if exclusive:
        # the whole loss is ONE photon: no ladder to convolute with, and the
        # answer is 1 by construction
        num = nug
    else:
        L = _fft_len(2 * n)
        num = np.fft.irfft(np.fft.rfft(nug, L) * np.fft.rfft(Ke, L), L)[:n]
    e = np.arange(n) * h
    i = np.arange(n, dtype=np.int64)
    kk = (i[:, None] + i[None, :]).ravel()
    den = np.bincount(kk, (P * np.outer(e, e)).ravel(), 2 * n)[:n]
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den, num, den


def multi_gbar(m, Ke, g, b_edges, h=MULTI_H, npanel=16, ng=4, modes=("multi",),
               verbose=False):
    """``Gbar`` of the two-leg laws that share ``Ke`` on the ``eps`` grid.

    ``modes`` picks which laws to build on the SAME grid so that the
    discretisation cancels in their ratio:

      ``single``  the whole loss carried by one exact-angle photon (the `corr`
                  model), i.e. ``S_1`` without exponentiation;
      ``multi``   the exponentiated exact sharing (this section);
      ``coll``    the exponentiated collinear sharing, which is ``D (x) D``.

    Returns ``(u_nodes, {mode: Gbar}, info)``; ``u_nodes[k]`` is the exact
    ``u`` of the anti-diagonal, ``-ln(1 - k h)/2``.
    """
    n = len(Ke)
    Gm = G_eps_matrix(g, b_edges, h, n)
    out = {}
    marg = {}
    info = {}
    for md, P, info in multi_laws(m, Ke, h, n, modes, npanel, ng):
        num, den = diag_reduce(P, Gm, n)
        neg = float(P.min())
        negsum = float(P[P < 0.0].sum())
        del P
        ok = den > 0.0
        v = np.full(n, Gm[0, 0])
        v[ok] = num[ok] / den[ok]
        out[md] = v
        info["den"] = den
        marg[md] = (float(np.abs(den - Ke).max() / Ke.max()),
                    float(1.0 - den.sum()), float(min(neg, 0.0)), negsum)
        if verbose:
            print(f"    [{md}] marginal {marg[md][0]:.2e}  leak {marg[md][1]:.2e}"
                  f"  min P {marg[md][2]:.2e}  Gbar(1e-2) = "
                  f"{np.interp(1e-2, u_of_eps(np.arange(n) * h), v):.6f}",
                  flush=True)
    info["marginal"] = marg
    return u_of_eps(np.arange(n) * h), out, info


def u_leg_of_eps(e):
    """The leg's own variable, ``u_q = -ln x_q`` with ``x_q = 1 - eps_q``."""
    return -np.log1p(-np.minimum(np.asarray(e, float), 1.0 - 1e-300))


def leg_tails(P, h, t_grid):
    """``(t, P(u_leg>t), P(u_+>t, u_->t), joint/product)`` of a two-leg law."""
    n = P.shape[0]
    u = u_leg_of_eps(np.arange(n) * h)
    tot = float(P.sum())
    Pl = P.sum(1) / tot
    out = []
    for t in t_grid:
        i = int(np.searchsorted(u, t, "right"))
        mg = float(Pl[i:].sum())
        jt = float(P[i:, i:].sum()) / tot
        out.append((t, mg, jt, jt / max(mg * mg, 1e-300)))
    return out


def share_hist(P, h, u_lo, u_hi, f_edges):
    """``(hist, weight, P(0.01 < f < 0.99))`` of the sharing in a ``u`` slice.

    ``f = eps_+ / eps`` is read off the anti-diagonal the configuration sits
    on, so the slice is weighted by the kernel itself.  The resolution in ``f``
    is ``1/k`` at total node ``k``: the two collinear end bins are grid-limited
    and so is the MC's own measurement, which has its own infrared cutoff.
    """
    n = P.shape[0]
    k0 = max(int(np.ceil(eps_of_u(u_lo) / h)), 1)
    k1 = min(int(np.floor(eps_of_u(u_hi) / h)), n - 1)
    nb = len(f_edges) - 1
    hst = np.zeros(nb)
    tot = win = 0.0
    if k1 < k0:                                  # the slice is off the grid
        return np.full(nb, np.nan), 0.0, np.nan
    for k in range(k0, k1 + 1):
        i = np.arange(k + 1)
        w = P[i, k - i]
        f = i / float(k)
        j = np.clip(np.searchsorted(f_edges, f, "right") - 1, 0, nb - 1)
        hst += np.bincount(j, w, nb)
        tot += float(w.sum())
        win += float(w[(f > 0.01) & (f < 0.99)].sum())
    return hst, tot, win / max(tot, 1e-300)


#: the ``f`` binning of `cmp_perleg.fig_share`, symmetric under ``f -> 1-f``
def share_edges(n=25, f0=1e-6):
    e = np.geomspace(f0, 0.5, n)
    return np.unique(np.concatenate([e, 1.0 - e[::-1]]))


#: floor, relative to ``Gbar(0)``, below which the two-leg law's selection
#: weight is not resolved on the ``eps`` grid and the variant correction is
#: continued at 1.  The kernel weight there is below 1e-4 of the total and the
#: answer is flat in it over three decades (README).
SHARE_FLOOR = 1e-6


#: masses at which an alternative construction's correction is evaluated.  The
#: alternatives are systematic variants and their ratio to the single-photon
#: sharing is a smooth, slowly varying function of ``m`` -- the whole model's
#: ``m`` dependence is analytic -- so they are built on this grid and read off
#: band by band, which is what makes them affordable.
VARIANT_MASSES = (55.0, 62.0, 70.0, 78.0, 85.0, 91.0, 97.0, 105.0, 115.0,
                  130.0, 150.0, 180.0)


def variant_ratio(ht, mode, masses=VARIANT_MASSES, n_u=1500, u_ref_max=7.0,
                  u_min=1e-9, cells_by_band=None, variant="exp2nll", pair=(),
                  minus_one=True, pair_table=None, var_budget=6e-10,
                  u_c=0.01, n_v=12, multi_h=MULTI_H, npanel=16, ng=4,
                  share_floor=SHARE_FLOOR, pr=None, verbose=True):
    """``rho(m, u) = Gbar_variant / Gbar_single-photon`` on a coarse mass grid.

    ``mode`` is

      ``"matched"``  the exact hard emission above ``u_c`` with its sharing,
                     convoluted with a collinear-independent soft remainder;
      ``"multi"``    the exponentiated exact sharing, i.e. every photon of the
                     Levy measure carrying the exact O(alpha) angle;
      ``"coll"``     the exponentiated *collinear* sharing, which is
                     ``D (x) D`` in the ``eps`` convention -- the collinear
                     limit of ``"multi"``, kept as a closure test of the
                     construction against the per-leg model.

    All three have the same ``z`` marginal as the single-photon model, so
    ``rho`` is a pure statement about the two-leg law and is read off at the
    band's own ``Gbar``.
    """
    b_edges = np.asarray(ht["b_edges"], float)
    H = np.asarray(ht["h"], float)
    bands = ht["bands"]
    ctr = np.array([0.5 * (a + b) if np.isfinite(b) else a * 1.05
                    for a, b in bands])
    u_ref = np.concatenate([[0.0], np.geomspace(u_min, u_ref_max, n_u)])
    rho = np.ones((len(masses), len(u_ref)))
    t0 = time.time()
    for i, mm in enumerate(masses):
        k = int(np.argmin(np.abs(ctr - mm)))
        g = pr.grid(k) if pr is not None else survival(H[k], b_edges)
        if cells_by_band is None:
            cells, wd, u_max = analytic_cells(ctr[k], variant, pair, minus_one,
                                              pair_table, var_budget)
        else:
            cells, wd, u_max = cells_by_band[k][:3]
        un = np.minimum(u_ref, u_max)
        if mode in ("multi", "coll", "lin"):
            nn = int(round(1.0 / multi_h))
            if cells_by_band is None:
                kk = FSRKernel(ctr[k], variant=variant, pair=tuple(pair),
                               minus_one=minus_one, pair_table=pair_table)
                Ke = kernel_eps_grid(kk, multi_h, nn,
                                     kk.pair_atoms(var_budget=1e-9))
            else:
                Ke = eps_grid_cells(cells, wd, multi_h, nn)
            ue, gv, inf = multi_gbar(ctr[k], Ke, g, b_edges, h=multi_h,
                                     npanel=npanel, ng=ng,
                                     modes=("single", mode))
            # the ratio is formed only where the conditional expectation is
            # RESOLVED: the anti-diagonal has to carry kernel weight and the
            # single-photon model's own selection weight has to be above a
            # floor, or ``Gbar`` is 0/0 and the ratio is noise.  Above the last
            # resolved node the two constructions are both selecting nothing
            # and the ratio is continued at 1.  ``share_floor`` is scanned.
            gs, gm, den = gv["single"], gv[mode], inf["den"]
            ok = (den > 1e-8 * den.max()) & (gs > share_floor * max(gs[0],
                                                                    1e-300))
            ok[0] = False
            kmax = int(np.nonzero(ok)[0].max()) if ok.any() else 1
            v = np.ones(len(gs))
            v[ok] = gm[ok] / gs[ok]
            rho[i] = np.interp(np.log(np.maximum(u_ref, u_min)),
                               np.log(np.maximum(ue[1:kmax + 1], u_min)),
                               v[1:kmax + 1], left=v[1], right=1.0)
            rho[i][u_ref <= 0.0] = 1.0
            if verbose:
                mg, lk, mp = inf["marginal"][mode][:3]
                print(f"[rho:{mode}] m = {ctr[k]:6.2f}  rho(u=1e-2) = "
                      f"{np.interp(1e-2, u_ref, rho[i]):.5f}  "
                      f"rho(0.1) = {np.interp(0.1, u_ref, rho[i]):.5f}  "
                      f"rho(0.5) = {np.interp(0.5, u_ref, rho[i]):.5f}  "
                      f"u_res = {ue[kmax]:.2f}  "
                      f"[marg {mg:.1e} leak {lk:.1e} minP {mp:.1e}]  "
                      f"{time.time()-t0:.0f} s", flush=True)
            continue
        gb = gbar(ctr[k], un, g, b_edges, npanel=64, ng=8)
        if mode == "matched":
            n = int(round(MATCH_GRID_MAX / MATCH_DU))
            if cells_by_band is None:
                kk = FSRKernel(ctr[k], variant=variant, pair=tuple(pair),
                               minus_one=minus_one, pair_table=pair_table)
                Kg = kernel_grid(kk, MATCH_DU, n,
                                 kk.pair_atoms(var_budget=1e-9))
            else:
                Kg = rebin_cells(cells, wd, MATCH_DU, n)
            num, den, su = matched_gbar(ctr[k], un, g, b_edges, Kg, u_c, u_max,
                                        n_v=n_v)
        else:
            raise ValueError(mode)
        gv = _fill_ratio(num, den, su, un, G_norad(g, b_edges), u_min)
        rho[i] = np.where(gb > 0, gv / np.maximum(gb, 1e-300), 1.0)
        if verbose:
            print(f"[rho:{mode}] m = {ctr[k]:6.2f}  rho(u=1e-2) = "
                  f"{np.interp(1e-2, un, rho[i]):.5f}  "
                  f"rho(u=0.1) = {np.interp(0.1, un, rho[i]):.5f}  "
                  f"{time.time()-t0:.0f} s", flush=True)
    return np.asarray(masses, float), u_ref, rho, "ratio"


def macc_m(ht, k, fallback):
    """The band's weighted mean pre-FSR mass, or its centre if it is empty."""
    v = float(ht["m_bar"][k])
    return v if v > 0 else fallback


def apply_rho(m, un, gb, rho, u_min=1e-9):
    """Read a variant's correction off the coarse mass grid onto a band's ladder.

    ``rho`` is ``(masses, u, table, kind)``; ``kind`` is ``ratio`` for the
    matched construction and ``delta`` for the two-leg laws of the
    multi-emission section, whose correction is a difference of two selection
    weights and is therefore well defined where both of them vanish.
    """
    mg, ug, R = rho[:3]
    kind = rho[3] if len(rho) > 3 else "ratio"
    j = np.clip(np.searchsorted(mg, m) - 1, 0, len(mg) - 2)
    t = np.clip((m - mg[j]) / (mg[j + 1] - mg[j]), 0.0, 1.0)
    r = R[j] * (1.0 - t) + R[j + 1] * t
    out = np.interp(np.log(np.maximum(un, u_min)),
                    np.log(np.maximum(ug[1:], u_min)), r[1:],
                    left=r[1], right=r[-1])
    out[un <= 0.0] = r[0]
    return np.maximum(gb + out, 0.0) if kind == "delta" else gb * out


def build_corr_kernel(ht, cells_by_band=None, variant="exp2nll", pair=(),
                      minus_one=True, pair_table=None, var_budget=6e-10,
                      sigma_cap=None, n_gbar=2000, npanel=64, ng=8,
                      u_min=1e-9, n_fine=4000, rho=None, pr=None,
                      verbose=True):
    """Atoms + ``A(m)`` of the correlated selection-conditional kernel.

    ``cells_by_band`` is ``(cells, w_delta, u_max)`` per band of the *inclusive*
    kernel; ``None`` builds them from `fsr_analytic.FSRKernel` at the band
    centre.  The ``z`` marginal of the result divided by ``A(m)`` is the
    inclusive kernel of that band reweighted by ``Gbar``, and nothing else.
    """
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
        if cells_by_band is None:
            cells, wd, u_max = analytic_cells(mc, variant, pair, minus_one,
                                              pair_table, var_budget, n_fine)
        else:
            cells, wd, u_max = cells_by_band[k][:3]
        g = pr.grid(k) if pr is not None else survival(H[k], b_edges)
        un = np.concatenate([[0.0], np.geomspace(u_min, u_max, n_gbar)])
        gb = gbar(mc, un, g, b_edges, npanel=npanel, ng=ng)
        if rho is not None:
            gb = apply_rho(macc_m(ht, k, mc), un, gb, rho, u_min)
        c = corr_cells(cells, un, gb, u_min)
        tot = float(np.sum(cells[0]) + wd)
        A = float((c[0].sum() + wd * gb[0]) / tot)
        if A <= 0.0:
            info.append(dict(m=mc, A=0.0, natoms=0, mean_u=0.0))
            continue
        uj, wj = _merge_cells(c / (tot * A), var_budget, sigma_cap)
        rj, wj = np.exp(-uj), wj
        if wd > 0.0:
            rj = np.concatenate([[1.0], rj])
            wj = np.concatenate([[wd * gb[0] / (tot * A)], wj])
        o = np.argsort(-rj)
        rj, wj = rj[o], wj[o]
        R.append(rj)
        W.append(wj)
        LO.append(np.full(len(rj), float(lo)))
        HI.append(np.full(len(rj), float(hi)))
        macc.append(float(ht["m_bar"][k]) if ht["m_bar"][k] > 0 else mc)
        aacc.append(A)
        info.append(dict(m=mc, m_bar=macc[-1], A=A, natoms=len(rj),
                         gb0=float(gb[0]),
                         mean_u=-float(np.sum(wj * np.log(np.maximum(rj, 1e-300))))))
    LO[0] = np.zeros_like(LO[0])
    HI[-1] = np.full_like(HI[-1], np.inf)
    if verbose:
        print(f"[corr] {len(R)} bands, {sum(len(r) for r in R)} atoms, "
              f"{time.time()-t0:.0f} s")
    return (dict(r=np.concatenate(R), w=np.concatenate(W),
                 m_lo=np.concatenate(LO), m_hi=np.concatenate(HI)),
            dict(kind="grid", m=macc, a=aacc), info)


# --------------------------------------------------------------------------
# drivers
# --------------------------------------------------------------------------
def load_htable(path):
    """An ``h`` table plus the provenance fields the pass region needs."""
    with np.load(path, allow_pickle=False) as d:
        ht = {k: d[k] for k in ("h", "b_edges", "bands", "m_bar", "sumw", "nev")}
        prov = json.loads(str(d["provenance"][0]))
    # `pt_cut` is the pre-asymmetric name of the same number: the reference of
    # the logarithmic axis.  Tables written before the cuts were separated from
    # the axis carry it and are read unchanged.
    ht["pt_ref"] = float(prov.get("pt_ref", prov.get("pt_cut")))
    ht["eta_cut"] = float(prov["eta_cut"])
    ht["provenance"] = prov
    return ht


def load_h4table(path):
    """The ``(b, eta)`` per leg table of `build_h4table`."""
    with np.load(path, allow_pickle=False) as d:
        h4 = {k: d[k] for k in ("h4", "b_mean", "b_edges", "eta_edges",
                                "bands", "m_bar", "sumw", "nev")}
        prov = json.loads(str(d["provenance"][0]))
    h4["pt_ref"] = float(prov["pt_ref"])
    h4["provenance"] = prov
    return h4


def make_pass(ht, args):
    """The `PassRegion` an ``args`` namespace asks for (None-safe)."""
    resol = None
    h4 = None
    if getattr(args, "resol", None):
        import ptres
        resol = ptres.Resolution(args.resol, mode=args.resol_mode)
        h4 = load_h4table(args.h4)
        if abs(h4["pt_ref"] - ht["pt_ref"]) > 1e-9:
            raise ValueError("h and h4 tables have different pT references")
    cuts = getattr(args, "pt_cuts", None)
    if cuts is None and resol is None:
        return None
    return PassRegion(ht, cuts=cuts, h4=h4, resol=resol)


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
                          ng=16, leg_cells_by_band=None, pr=None,
                          verbose=True):
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
        g = pr.grid(k) if pr is not None else survival(H[k], b_edges)
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


def _sel_meta(pr, ht):
    """The selection, as it goes into a kernel's provenance."""
    if pr is None:
        return dict(pt_lead=ht["pt_ref"], pt_trail=ht["pt_ref"],
                    eta_cut=ht["eta_cut"], resolution=None)
    return dict(pt_lead=pr.cuts[0], pt_trail=pr.cuts[1],
                eta_cut=ht["eta_cut"],
                resolution=None if pr.resol is None else
                dict(file=pr.resol.path, mode=pr.resol.mode,
                     h4=pr.h4["provenance"]["gen"] if pr.h4 else None))


def _edges_from_args(args):
    """The ``b`` grid an ``args`` namespace asks for.

    No ``--anchor-cuts`` and the default range reproduces `B_EDGES` exactly, so
    every table and every number built before the cuts became asymmetric is
    reproduced bit for bit.
    """
    if not args.anchor_cuts and args.b_min == 0.0 and args.b_max == 3.0 \
            and args.b_thin <= 1 and args.fine_step is None:
        return B_EDGES
    anchors = [0.0] + [float(np.log(c / args.pt_ref))
                       for c in (args.anchor_cuts or ())]
    fine = (((0.08, 1e-3), (0.5, 5e-3)) if args.fine_step is None
            else ((args.fine_width, args.fine_step),))
    be = b_grid(anchors, b_min=args.b_min, b_max=args.b_max, fine=fine,
                coarse=args.coarse_step)
    if args.b_thin > 1:
        be = np.unique(np.concatenate([be[::args.b_thin], be[-1:]]))
    return be


def _gen_legs(d):
    """``(pt1, eta1, pt2, eta2)`` of the PRE-FSR muons of a gen record."""
    if "ptp_pre" in d:
        return d["ptp_pre"], d["etap_pre"], d["ptm_pre"], d["etam_pre"]
    return d["pt1_pre"], d["eta1_pre"], d["pt2_pre"], d["eta2_pre"]


def _h4table_cmd(args):
    d = np.load(args.gen)
    w = np.asarray(d["weight"], np.float64)
    if args.wclip:
        aw = np.abs(w)
        w = np.clip(w, -args.wclip * np.median(aw), args.wclip * np.median(aw))
        w = w * (len(w) / w.sum())
    m = np.asarray(d["m_pre"], np.float64)
    pt1, eta1, pt2, eta2 = _gen_legs(d)
    bands = make_bands(args.band_lo, args.band_hi, args.band_width)
    be = _edges_from_args(args)
    ee = np.asarray(args.eta_edges, float)
    ht = build_h4table(m, pt1, eta1, pt2, eta2, w, bands, pt_ref=args.pt_ref,
                       eta_cut=args.eta_cut, b_edges=be, eta_edges=ee)
    meta = dict(gen=os.path.abspath(args.gen), pt_ref=args.pt_ref,
                anchor_cuts=list(args.anchor_cuts or ()),
                b_min=args.b_min, b_max=args.b_max, eta_cut=args.eta_cut,
                eta_edges=ee.tolist(), band_lo=args.band_lo,
                band_hi=args.band_hi, band_width=args.band_width,
                wclip=args.wclip, b_thin=args.b_thin, n_b=len(ht["b_edges"]))
    np.savez_compressed(args.output, h4=ht["h4"], b_mean=ht["b_mean"],
                        b_edges=ht["b_edges"],
                        eta_edges=ht["eta_edges"], bands=ht["bands"],
                        m_bar=ht["m_bar"], sumw=ht["sumw"], nev=ht["nev"],
                        provenance=np.array([json.dumps(meta)]))
    print(f"[h4table] {len(bands)} bands x ({len(ee)-1} x "
          f"{ht['h4'].shape[2]})^2 cells = "
          f"{ht['h4'].nbytes/1e6:.0f} MB -> {args.output}")
    for k in range(0, len(bands), max(1, len(bands) // 8)):
        print(f"    m = {ht['m_bar'][k]:7.2f}  n = {ht['nev'][k]:8d}  "
              f"sum h4 = {ht['h4'][k].sum():.5f}")


def _gcheck_cmd(args):
    """``G`` off the table against a direct weighted event count.

    The only approximation between the two is the table: the ``(b_+, b_-)``
    binning for a step, and additionally the ``eta`` binning and the cell
    midpoint rule once the pass probability is smooth.  The resolution model
    itself is the SAME object on both sides, so what this measures is the
    discretisation and nothing else.
    """
    ht = load_htable(args.htable)
    pr = make_pass(ht, args)
    d = np.load(args.gen)
    w = np.asarray(d["weight"], np.float64)
    aw = np.abs(w)
    w = np.clip(w, -args.wclip * np.median(aw), args.wclip * np.median(aw))
    w = w * (len(w) / w.sum())
    m = np.asarray(d["m_pre"], np.float64)
    pt1, eta1, pt2, eta2 = (np.asarray(x, float) for x in _gen_legs(d))
    c0 = 0.5 * (args.band[0] + args.band[1])
    k = int(np.argmin([abs(0.5 * (a + b) - c0) if np.isfinite(b) else 1e9
                       for a, b in ht["bands"]]))
    # the direct count must use the SAME band the table's G belongs to: A(m)
    # moves by ~1e-3 per GeV, which would swamp the discretisation this checks
    lo, hi = ht["bands"][k]
    s = (m >= lo) & (m < hi)
    w, pt1, eta1, pt2, eta2 = w[s], pt1[s], eta1[s], pt2[s], eta2[s]
    tot = w.sum()
    ok = (np.abs(eta1) < ht["eta_cut"]) & (np.abs(eta2) < ht["eta_cut"])
    cL, cT = pr.cuts
    G = pr.grid(k)
    print(f"[gcheck] band {lo}-{hi} ({int(s.sum())} events), {pr.label()}, "
          f"table band {ht['bands'][k]}")
    print(f"{'u_+':>9s} {'u_-':>9s} {'G table':>12s} {'G direct':>12s} "
          f"{'diff':>11s}")
    for up in args.u:
        for um in args.u:
            a1 = pt1 * np.exp(-up)
            a2 = pt2 * np.exp(-um)
            b1 = pt1 * np.exp(-um)
            b2 = pt2 * np.exp(-up)
            if pr.resol is None:
                p1 = ((np.maximum(a1, a2) > cL) & (np.minimum(a1, a2) > cT))
                p2 = ((np.maximum(b1, b2) > cL) & (np.minimum(b1, b2) > cT))
                direct = float(np.sum(w * ok * 0.5
                                      * (p1.astype(float)
                                         + p2.astype(float))) / tot)
            else:
                acc = 0.0
                for x1, x2 in ((a1, a2), (b1, b2)):
                    L1 = pr.resol.pass_prob_ev(x1, cL, eta1)
                    T1 = pr.resol.pass_prob_ev(x1, cT, eta1)
                    L2 = pr.resol.pass_prob_ev(x2, cL, eta2)
                    T2 = pr.resol.pass_prob_ev(x2, cT, eta2)
                    acc = acc + 0.5 * (L1 * T2 + T1 * L2 - L1 * L2)
                direct = float(np.sum(w * ok * acc) / tot)
            gt = float(eval_G_at(G, ht["b_edges"], np.array([up]),
                                 np.array([um]))[0])
            print(f"{up:9.4f} {um:9.4f} {gt:12.6f} {direct:12.6f} "
                  f"{gt - direct:+11.2e}")


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
    be = _edges_from_args(args)
    ht = build_htable(m, pt1, eta1, pt2, eta2, w, bands, pt_ref=args.pt_ref,
                      eta_cut=args.eta_cut, b_edges=be)
    meta = dict(gen=os.path.abspath(args.gen), pt_ref=args.pt_ref,
                anchor_cuts=list(args.anchor_cuts or ()),
                b_min=args.b_min, b_max=args.b_max,
                eta_cut=args.eta_cut, band_lo=args.band_lo,
                band_hi=args.band_hi, band_width=args.band_width,
                wclip=args.wclip, b_thin=args.b_thin,
                n_b=len(ht["b_edges"]))
    np.savez_compressed(args.output, h=ht["h"].astype(np.float32),
                        b_edges=ht["b_edges"], bands=ht["bands"],
                        m_bar=ht["m_bar"], sumw=ht["sumw"], nev=ht["nev"],
                        provenance=np.array([json.dumps(meta)]))
    print(f"[htable] {len(bands)} bands x {ht['h'].shape[1]}^2 cells, "
          f"pT ref {args.pt_ref}, b in [{be[0]:.2f}, {be[-1]:.2f}], "
          f"|eta| < {args.eta_cut} -> {args.output}")
    for k in range(0, len(bands), max(1, len(bands) // 8)):
        print(f"    m = {ht['m_bar'][k]:7.2f}  n = {ht['nev'][k]:8d}  "
              f"sum h = {ht['h'][k].sum():.5f}")


def _kernel_cmd(args):
    ht = load_htable(args.htable)
    prov = ht["provenance"]
    pr = make_pass(ht, args)
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
        sigma_cap=args.sigma_cap, n_out=args.n_out, leg_cells_by_band=legs,
        pr=pr)
    meta = dict(kind="perleg", selection=_sel_meta(pr, ht), variant=args.variant,
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
                           _meta=dict(selection=_sel_meta(pr, ht),
                                      eta_cut=prov["eta_cut"],
                                      source=os.path.abspath(args.output))),
                      fh, indent=1)
        print(f"[kernel] acceptance -> {args.acceptance}")
    print(f"[kernel] {len(ker['r'])} atoms -> {args.output}")
    for b in info[:: max(1, len(info) // 10)]:
        if b["natoms"]:
            print(f"    m = {b['m_bar']:7.2f}  A = {b['A']:.5f}"
                  f"  atoms = {b['natoms']:4d}  <u> = {b['mean_u']*1e3:8.4f}e-3")


def _corr_cmd(args):
    ht = load_htable(args.htable)
    prov = ht["provenance"]
    pr = make_pass(ht, args)
    cells = None
    if args.run:
        cells = tabulated_cells(args.run, ht["bands"])
    pair = tuple(args.pair if args.pair is not None else ())
    rho = None
    if args.mode != "single":
        if args.rho and os.path.exists(args.rho):
            with np.load(args.rho) as z:
                rho = (z["m"], z["u"], z["rho"],
                       str(z["kind"]) if "kind" in z.files else "ratio")
            print(f"[corr] {rho[3]} table from {args.rho}")
        else:
            rho = variant_ratio(ht, args.mode, cells_by_band=cells,
                                variant=args.variant, pair=pair,
                                minus_one=not args.no_minus_one,
                                pair_table=args.pair_table, u_c=args.u_c,
                                n_v=args.n_v, pr=pr, multi_h=args.multi_h,
                                npanel=args.share_npanel, ng=args.share_ng,
                                share_floor=args.share_floor,
                                masses=(tuple(args.rho_masses)
                                        if args.rho_masses else VARIANT_MASSES))
            if args.rho:
                np.savez(args.rho, m=rho[0], u=rho[1], rho=rho[2],
                         kind=np.array(rho[3]))
    ker, acc, info = build_corr_kernel(
        ht, cells_by_band=cells, variant=args.variant, pair=pair,
        minus_one=not args.no_minus_one, pair_table=args.pair_table,
        var_budget=(None if args.sigma_cap else args.var_budget),
        sigma_cap=args.sigma_cap, n_gbar=args.n_gbar, npanel=args.npanel,
        ng=args.ng, rho=rho, pr=pr)
    meta = dict(kind="corr", selection=_sel_meta(pr, ht), variant=args.variant,
                run=args.run,
                mode=args.mode, u_c=args.u_c, n_v=args.n_v,
                multi_h=args.multi_h, share_npanel=args.share_npanel,
                share_ng=args.share_ng, share_floor=args.share_floor,
                pair=list(args.pair or ()), htable=os.path.abspath(args.htable),
                htable_prov=prov, n_gbar=args.n_gbar, npanel=args.npanel,
                ng=args.ng, var_budget=args.var_budget,
                sigma_cap=args.sigma_cap, alpha=ALPHA, m_mu=M_MU, bands=info)
    np.savez(args.output, r=ker["r"], w=ker["w"], m_lo=ker["m_lo"],
             m_hi=ker["m_hi"], provenance=np.array([json.dumps(meta)]))
    if args.acceptance:
        with open(args.acceptance, "w") as fh:
            json.dump(dict(kind="grid", m=acc["m"], a=acc["a"],
                           _meta=dict(selection=_sel_meta(pr, ht),
                                      eta_cut=prov["eta_cut"],
                                      source=os.path.abspath(args.output))),
                      fh, indent=1)
        print(f"[corr] acceptance -> {args.acceptance}")
    print(f"[corr] {len(ker['r'])} atoms -> {args.output}")
    for b in info[:: max(1, len(info) // 10)]:
        if b["natoms"]:
            print(f"    m = {b['m_bar']:7.2f}  A = {b['A']:.5f}"
                  f"  atoms = {b['natoms']:4d}  <u> = {b['mean_u']*1e3:8.4f}e-3")


#: the four ``u`` slices the generator record's own sharing table is quoted in
REPORT_SLICES = ((1e-3, 1e-2), (1e-2, 0.05), (0.05, 0.2), (0.2, 0.6))
#: the profile version, on `cmp_perleg.fig_share_u`'s own ``u`` binning
PROFILE_EDGES = np.geomspace(6e-4, 0.8, 15)
SHARE_SLICES = tuple(REPORT_SLICES) + tuple(zip(PROFILE_EDGES[:-1],
                                                PROFILE_EDGES[1:]))

#: thresholds of the leg law, matching the generator record's own table
LEG_T = (1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.2)


def _multi_cells(args, ht, k, mc):
    """``(cells, w_delta, u_max, m)`` of the band's inclusive kernel."""
    if args.run:
        c, wd, um = tabulated_cells(args.run, [ht["bands"][k]])[0]
        return c, wd, um
    kk = FSRKernel(mc, variant=args.variant,
                   pair=tuple(args.pair if args.pair is not None else ()),
                   minus_one=not args.no_minus_one,
                   pair_table=args.pair_table)
    return kk.cells(var_budget=args.var_budget), 0.0, kk._u_max()


def _multi_Ke(args, cells, wd, mc, h, n):
    """The band's inclusive kernel on the uniform ``eps`` grid.

    A tabulated (standalone Photos) kernel is deposited cell by cell; an
    analytic one is integrated cell by cell, which keeps it dense -- a comb is
    not infinitely divisible and its logarithm is not a measure.
    """
    if args.run:
        return eps_grid_cells(cells, wd, h, n)
    k = FSRKernel(mc, variant=args.variant,
                  pair=tuple(args.pair if args.pair is not None else ()),
                  minus_one=not args.no_minus_one, pair_table=args.pair_table)
    return kernel_eps_grid(k, h, n, k.pair_atoms(var_budget=1e-9))


def _multicheck_cmd(args):
    ht = load_htable(args.htable)
    pr = make_pass(ht, args)
    b_edges = np.asarray(ht["b_edges"], float)
    bands = ht["bands"]
    ctr = np.array([0.5 * (a + b) if np.isfinite(b) else a * 1.05
                    for a, b in bands])
    k = int(np.argmin(np.abs(ctr - args.mass)))
    mc = float(ctr[k])
    g = (pr.grid(k) if pr is not None
         else survival(np.asarray(ht["h"], float)[k], b_edges))
    cells, wd, u_max = _multi_cells(args, ht, k, mc)
    print(f"# band {k} = [{bands[k][0]:g}, {bands[k][1]:g}) GeV, m = {mc:.3f}, "
          f"selection {_sel_meta(pr, ht)}, kernel = "
          f"{'tabulated ' + os.path.basename(args.run) if args.run else args.variant}")
    u_probe = (1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0)
    keep = {}
    print("\n# the Levy measure of the kernel in eps = 1 - z, and the "
          "marginal identity of the two-leg law")
    for h in args.h_scan:
        n = int(round(args.eps_max / h))
        Ke = _multi_Ke(args, cells, wd, mc, h, n)
        nu, N, inf = levy(Ke)
        line = (f"h = {h:.1e}  n = {n:5d}  N = {N:.6f}  |K^|min = "
                f"{inf['k_abs_min']:.4f}  |arg K^|max = {inf['k_arg_max']:.4f}  "
                f"nu<0 = {float(nu[nu < 0].sum()):+.2e}  roundtrip = "
                f"{inf['roundtrip']:.1e}")
        if n <= args.recursion_max:
            nur, Nr = levy_recursion(Ke)
            line += (f"  |nu - nu_rec|max = "
                     f"{float(np.abs(nu - nur).max()):.1e}")
        print(line, flush=True)
        un, gb, inf2 = multi_gbar(mc, Ke, g, b_edges, h=h, npanel=args.npanel,
                                  ng=args.ng, modes=tuple(args.modes))
        for md in args.modes:
            mg, lk, mp, ms = inf2["marginal"][md]
            print(f"    {md:7s} marginal {mg:.2e}  leak {lk:+.2e}  "
                  f"min P {mp:+.2e}  sum P<0 {ms:+.2e}")
        keep[h] = (un, gb)
    print("\n# Gbar(u), and the two-leg law's correction to the single-photon "
          "model")
    hdr = "  u        " + "".join(f"{md:>12s}" for md in args.modes)
    for md in args.modes[1:]:
        hdr += f"{md + '/single':>14s}"
    print(hdr)
    for h in args.h_scan:
        un, gb = keep[h]
        print(f"  -- h = {h:.1e}")
        for u0 in u_probe:
            row = f"  {u0:<9.4g}"
            for md in args.modes:
                row += f"{np.interp(u0, un, gb[md]):12.6f}"
            for md in args.modes[1:]:
                row += (f"{np.interp(u0, un, gb[md]) / np.interp(u0, un, gb['single']):14.5f}")
            print(row)
    h = args.h_scan[0]
    n = int(round(args.eps_max / h))
    Ke = _multi_Ke(args, cells, wd, mc, h, n)
    fe = share_edges()
    Dref = conv_sqrt(Ke)[0][:n]
    out = dict(h=h, n=n, m=mc, band=np.asarray(bands[k], float),
               f_edges=fe, slices=np.asarray(SHARE_SLICES, float),
               t=np.asarray(LEG_T, float), Ke=Ke,
               u_nodes=u_of_eps(np.arange(n) * h),
               modes=np.asarray(list(args.modes)))
    for md in args.modes:
        out[f"gbar_{md}"] = keep[h][1][md]
    print("\n# the leg law: P(u_leg > t), the joint tail, and their ratio")
    print(f"  {'t':>8s}" + "".join(f"{md + ' marg':>14s}{md + ' j/p':>10s}"
                                   for md in args.modes))
    rows = {}
    shr = {}
    for md, P, _ in multi_laws(mc, Ke, h, n, tuple(args.modes),
                               args.npanel, args.ng):
        rows[md] = leg_tails(P, h, LEG_T)
        shr[md] = [share_hist(P, h, a, b, fe) for a, b in SHARE_SLICES]
        nu, N, _ = levy(Ke)
        Rm = multi_R(mc, Ke, Ke if md == "single" else
                     (np.zeros(n) if md == "coll" else nu), P, h, n,
                     args.npanel, args.ng, exclusive=(md == "single"))[0]
        out[f"R_{md}"] = Rm
        if md == "coll":
            pl = P.sum(1)
            out["coll_legmarg"] = pl
            print(f"  [coll] leg marginal against conv_sqrt(K_eps): "
                  f"max |diff| = {float(np.abs(pl - Dref).max()):.2e}, "
                  f"  <u_leg> {float(np.sum(pl * u_leg_of_eps(np.arange(n) * h))):.6e}"
                  f" vs {float(np.sum(Dref * u_leg_of_eps(np.arange(n) * h))):.6e}")
        out[f"legtail_{md}"] = np.asarray(rows[md], float)
        out[f"share_{md}"] = np.stack([x[0] for x in shr[md]])
        out[f"sharewin_{md}"] = np.asarray([x[2] for x in shr[md]], float)
        del P
    for i, t in enumerate(LEG_T):
        row = f"  {t:8.1e}"
        for md in args.modes:
            row += f"{rows[md][i][1]:14.5f}{rows[md][i][3]:10.3f}"
        print(row)
    print("\n# the model's own mass-leg ratio "
          "R = <x_+ x_- - z>/<eps_+ eps_->  (1 = z is 1 - eps, 0 = z is x_+x_-)")
    un = u_of_eps(np.arange(n) * h)
    print(f"  {'u':>10s}" + "".join(f"{md:>12s}" for md in args.modes))
    for u0 in (1e-3, 1e-2, 0.05, 0.1, 0.2, 0.35, 0.5):
        print(f"  {u0:10.4g}" + "".join(
            f"{np.interp(u0, un, out['R_' + md]):12.4f}" for md in args.modes))
    print("\n# the sharing: P(0.01 < f < 0.99) per u slice "
          f"(f resolution 1/k, k = eps/h, h = {h:.1e})")
    print(f"  {'u slice':>16s}" + "".join(f"{md:>12s}" for md in args.modes)
          + f"{'exact O(a)':>12s}")
    for i, (a, b) in enumerate(REPORT_SLICES):
        zb = float(np.exp(-2.0 * math.sqrt(a * b)))
        fm, wm = FA.share_nodes(mc, np.array([zb]), npanel=256, ng=8)
        ex = float(2.0 * wm[0][fm[0] > 0.01].sum())
        row = f"  [{a:g}, {b:g})".rjust(18)
        for md in args.modes:
            row += f"{shr[md][i][2]:12.4f}"
        print(row + f"{ex:12.4f}")
    if args.output:
        np.savez_compressed(args.output, **out)
        print(f"\n[multicheck] -> {args.output}")


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
    cuts = args.pt_cuts if args.pt_cuts else [args.pt_cut]
    seed = (FG.SMEAR_SEED if args.smear_seed is None else args.smear_seed)
    args.smear_seed = seed
    rsm = FG.load_resolution(args.smear, args.smear_mode)
    if "xp" in d.files:
        xp = np.asarray(d["xp"], float)
        xm = np.asarray(d["xm"], float)
        mass = {"u": np.asarray(d["m_post"], float),
                "coll": m * np.sqrt(np.maximum(xp * xm, 1e-300))}
        rec = {"post": dict(m_pre=m, pt1=d["ptp"], eta1=d["etap"],
                            pt2=d["ptm"], eta2=d["etam"]),
               "coll": dict(m_pre=m, pt1=xp * d["ptp_pre"],
                            eta1=d["etap_pre"], pt2=xm * d["ptm_pre"],
                            eta2=d["etam_pre"])}
        sel = {k: FG.fiducial(v, cuts, args.eta_cut, post=True, smear=rsm,
                              seed=args.smear_seed) for k, v in rec.items()}
    else:
        # the full gen record: no per-leg x, so only the MC's own kernel under
        # its own selection -- which is the reference row of the benchmark, and
        # is measured on the SAME events (and the same smearing draw) the fit
        # sees, so it carries no record floor of its own.
        mass = {"u": np.asarray(d["m_post"], float)}
        sel = {"post": FG.fiducial(
            {k: np.asarray(d[k], float) for k in
             ("m_pre", "pt1", "eta1", "pt2", "eta2")},
            cuts, args.eta_cut, post=True, smear=rsm, seed=args.smear_seed)}
    mkey, skey = COND_MODES[args.mode]
    if mkey not in mass or skey not in sel:
        raise ValueError(f"mode {args.mode} needs a per-leg gen record")
    s = sel[skey]
    bands = list(zip(COND_BANDS[:-1], COND_BANDS[1:]))
    bands = [(0.0, COND_BANDS[0])] + bands + [(COND_BANDS[-1], np.inf)]
    ker, info = FG.build_banded_kernel(m[s], mass[mkey][s], w[s], bands,
                                       sigma_cap=args.sigma_cap)
    meta = dict(kind="condker", mode=args.mode, mass=mkey, selection=skey,
                gen=os.path.abspath(args.gen), pt_cuts=list(cuts),
                smear=args.smear, smear_mode=args.smear_mode,
                smear_seed=args.smear_seed,
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
                                      selection=skey, pt_cuts=list(cuts),
                                      smear=args.smear,
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

    def _table_args(q, pt_ref):
        q.add_argument("--gen", required=True)
        q.add_argument("-o", "--output", required=True)
        q.add_argument("--pt-ref", type=float, default=pt_ref,
                       help="reference of the log axis b = ln(pT/pT_ref); NOT "
                            "a cut -- the table serves every cut above it")
        q.add_argument("--anchor-cuts", nargs="*", type=float, default=None,
                       help="extra thresholds [GeV] the fine part of the b "
                            "grid is replicated at (the leading cut of an "
                            "asymmetric pair)")
        q.add_argument("--b-min", type=float, default=0.0,
                       help="floor of the b grid; below 0 for a resolution, so "
                            "a muon under the cut can fluctuate over it")
        q.add_argument("--b-max", type=float, default=3.0)
        q.add_argument("--fine-step", type=float, default=None,
                       help="single-step fine pattern instead of the default "
                            "1e-3 / 5e-3 two-block one")
        q.add_argument("--fine-width", type=float, default=0.2)
        q.add_argument("--coarse-step", type=float, default=2e-2)
        q.add_argument("--eta-cut", type=float, default=2.4)
        q.add_argument("--band-lo", type=float, default=50.0)
        q.add_argument("--band-hi", type=float, default=200.0)
        q.add_argument("--band-width", type=float, default=1.0)
        q.add_argument("--wclip", type=float, default=100.0)
        q.add_argument("--b-thin", type=int, default=1,
                       help="keep every n-th edge of the b grid")

    t = sub.add_parser("htable", help="h(b_+, b_- | m) from a generator record")
    _table_args(t, 25.0)

    t4 = sub.add_parser("h4table",
                        help="h4(b_+, eta_+, b_-, eta_- | m): the same table "
                             "with the eta axis the resolution needs")
    _table_args(t4, 10.0)
    t4.add_argument("--eta-edges", nargs="*", type=float,
                    default=[0.0, 0.9, 1.6, 2.1, 2.4])

    k = sub.add_parser("kernel", help="the selection-conditional atoms + A(m)")
    k.add_argument("--htable", required=True)
    k.add_argument("-o", "--output", required=True)
    k.add_argument("--acceptance", default=None, help="write A(m) as a json")
    k.add_argument("--pt-cuts", nargs="*", type=float, default=None,
                   help="leading and trailing pT thresholds [GeV]; one value "
                        "or none is the symmetric cut at the table's pT ref")
    k.add_argument("--h4", default=None,
                   help="h4 table, required with --resol")
    k.add_argument("--resol", default=None,
                   help="a `ptres.py` resolution npz: the cuts then act on the "
                        "RECONSTRUCTED pT and the pass region is smooth")
    k.add_argument("--resol-mode", default="shape",
                   choices=("shape", "gauss"))
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

    r = sub.add_parser("corr", help="the CORRELATED selection-conditional "
                                    "kernel: exact O(alpha) recoil sharing")
    r.add_argument("--htable", required=True)
    r.add_argument("-o", "--output", required=True)
    r.add_argument("--acceptance", default=None, help="write A(m) as a json")
    r.add_argument("--pt-cuts", nargs="*", type=float, default=None,
                   help="leading and trailing pT thresholds [GeV]; one value "
                        "or none is the symmetric cut at the table's pT ref")
    r.add_argument("--h4", default=None,
                   help="h4 table, required with --resol")
    r.add_argument("--resol", default=None,
                   help="a `ptres.py` resolution npz: the cuts then act on the "
                        "RECONSTRUCTED pT and the pass region is smooth")
    r.add_argument("--resol-mode", default="shape",
                   choices=("shape", "gauss"))
    r.add_argument("--run", default=None,
                   help="a standalone Photos run npz: use its tabulated kernel "
                        "instead of the analytic one (the `mc` configuration)")
    r.add_argument("--variant", default="exp2nll",
                   choices=("exp1", "exp2", "exp2nll"))
    r.add_argument("--pair", nargs="*", default=None)
    r.add_argument("--no-minus-one", action="store_true")
    r.add_argument("--pair-table", default=None)
    r.add_argument("--var-budget", type=float, default=6e-10)
    r.add_argument("--sigma-cap", type=float, default=None)
    r.add_argument("--n-gbar", type=int, default=2000,
                   help="nodes of the Gbar(u) ladder")
    r.add_argument("--npanel", type=int, default=64,
                   help="Gauss panels of the sharing integral, half branch")
    r.add_argument("--ng", type=int, default=8)
    r.add_argument("--mode", default="single",
                   choices=("single", "matched", "multi", "coll", "lin"),
                   help="single: the exact O(alpha) sharing applied to the "
                        "whole loss.  matched: the exact hard emission above "
                        "--u-c with its sharing, convoluted with a "
                        "collinear-independent soft remainder below it.  "
                        "multi: the exponentiated exact sharing -- every "
                        "photon of the kernel's Levy measure carries the "
                        "exact O(alpha) angle.  coll: its collinear limit, "
                        "D (x) D")
    r.add_argument("--multi-h", type=float, default=MULTI_H,
                   help="uniform step of the eps grid of --mode multi/coll")
    r.add_argument("--share-npanel", type=int, default=16,
                   help="Gauss panels of the per-photon sharing, half branch")
    r.add_argument("--share-ng", type=int, default=4)
    r.add_argument("--share-floor", type=float, default=SHARE_FLOOR,
                   help="floor on Gbar/Gbar(0) below which the two-leg law is "
                        "not resolved on the eps grid and the correction is "
                        "continued at 1")
    r.add_argument("--u-c", type=float, default=0.01,
                   help="matching scale of --mode matched")
    r.add_argument("--n-v", type=int, default=12,
                   help="groups of the soft remainder's ladder")
    r.add_argument("--rho", default=None,
                   help="cache file of the variant/single ratio table")
    r.add_argument("--rho-masses", type=float, nargs="*", default=None,
                   help="masses the variant/single ratio is evaluated at")

    mc = sub.add_parser("multicheck",
                        help="the exponentiated-sharing two-leg law: the Levy "
                             "measure, the marginal identity, the leg law and "
                             "the sharing, against the collinear product")
    mc.add_argument("--htable", required=True)
    mc.add_argument("-o", "--output", default=None)
    mc.add_argument("--mass", type=float, default=91.0,
                    help="the h-table band whose centre is closest is used")
    mc.add_argument("--run", default=None,
                    help="a standalone Photos run npz: the `mc` kernel")
    mc.add_argument("--pt-cuts", nargs="*", type=float, default=None)
    mc.add_argument("--h4", default=None)
    mc.add_argument("--resol", default=None)
    mc.add_argument("--resol-mode", default="shape", choices=("shape", "gauss"))
    mc.add_argument("--variant", default="exp2nll",
                    choices=("exp1", "exp2", "exp2nll"))
    mc.add_argument("--pair", nargs="*", default=None)
    mc.add_argument("--no-minus-one", action="store_true")
    mc.add_argument("--pair-table", default=None)
    mc.add_argument("--var-budget", type=float, default=6e-10)
    mc.add_argument("--eps-max", type=float, default=1.0,
                    help="extent of the eps grid; below 1 it is a FINE short "
                         "grid for the small-u sharing, which the causal Levy "
                         "recursion makes exact there")
    mc.add_argument("--h-scan", nargs="*", type=float,
                    default=(5e-4, 2.5e-4, 1e-3))
    mc.add_argument("--modes", nargs="*", default=("single", "multi", "coll"))
    mc.add_argument("--npanel", type=int, default=16)
    mc.add_argument("--ng", type=int, default=4)
    mc.add_argument("--recursion-max", type=int, default=6000,
                    help="grid size up to which the causal recursion is run as "
                         "a cross-check of the spectral logarithm")

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
    d.add_argument("--pt-cuts", nargs="*", type=float, default=None,
                   help="leading and trailing pT thresholds [GeV]")
    d.add_argument("--smear", default=None,
                   help="a `ptres.py` resolution npz: select on a smeared pT")
    d.add_argument("--smear-mode", default="shape", choices=("shape", "gauss"))
    d.add_argument("--smear-seed", type=int, default=None)
    d.add_argument("--eta-cut", type=float, default=2.4)
    d.add_argument("--sigma-cap", type=float, default=3.3e-4)
    d.add_argument("--wclip", type=float, default=100.0)
    d.add_argument("--acc-lo", type=float, default=50.0)
    d.add_argument("--acc-hi", type=float, default=200.0)
    d.add_argument("--acc-width", type=float, default=1.0)

    gc = sub.add_parser("gcheck",
                        help="G off the table against a direct event count")
    gc.add_argument("--gen", required=True)
    gc.add_argument("--htable", required=True)
    gc.add_argument("--pt-cuts", nargs="*", type=float, default=None)
    gc.add_argument("--h4", default=None)
    gc.add_argument("--resol", default=None)
    gc.add_argument("--resol-mode", default="shape", choices=("shape", "gauss"))
    gc.add_argument("--band", nargs=2, type=float, default=(90.0, 92.0))
    gc.add_argument("--u", nargs="*", type=float,
                    default=(0.0, 1e-3, 1e-2, 0.05, 0.2, 0.5))
    gc.add_argument("--wclip", type=float, default=100.0)

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
    elif args.cmd == "h4table":
        _h4table_cmd(args)
    elif args.cmd == "gcheck":
        _gcheck_cmd(args)
    elif args.cmd == "kernel":
        _kernel_cmd(args)
    elif args.cmd == "corr":
        _corr_cmd(args)
    elif args.cmd == "multicheck":
        _multicheck_cmd(args)
    elif args.cmd == "legsqrt":
        _legsqrt_cmd(args)
    elif args.cmd == "condker":
        _condker_cmd(args)


if __name__ == "__main__":
    main()
