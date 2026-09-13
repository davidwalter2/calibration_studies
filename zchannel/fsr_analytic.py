#!/usr/bin/env python3
r"""Analytic QED final-state-radiation kernel for the Z channel.

Replaces the MC-derived (Photos++) FSR kernel of `fit_gen.py kernel` by a
closed-form QED radiator whose dependence on the pre-FSR mass is exact, so the
kernel is a *function of the fitted mass* rather than a table measured at one
generator setting.

Variables
---------
``m``   pre-FSR dimuon (Z virtuality) mass,      ``s  = m^2``
``m'``  post-FSR bare dimuon mass,               ``s' = m'^2``
``z  = s'/s = (m'/m)^2``,  ``x = 1 - z``,  ``u = -ln(m'/m) = -(1/2) ln z``.
For a single emission ``x = 2 E_gamma / m`` in the Z rest frame.

``L(m) = ln(m^2/m_mu^2)``,  ``beta(m) = (2 alpha / pi) (L - 1)``.

`alpha` is the **on-shell (Thomson) coupling** 1/137.035999: the radiated
photons are real and on shell, so no vacuum-polarisation running enters the
emission vertex.  (The running of alpha in the *propagator* is a property of
the Born lineshape, not of the radiator; the provider treats it separately.)

The O(alpha) spectrum
---------------------
For a vector *or* axial current, integrated over all angles at fixed ``z``,
the exact spin-summed 3-body result collapses in the limit
``m_mu^2/s -> 0`` to

    R1(z; m) = (alpha/pi) (1+z^2)/(1-z) [ ln(z s / m_mu^2) - 1 ]                (1)
             = (alpha/pi) (1+z^2)/(1-z) [ L - 1 + ln z ] ,

i.e. the Altarelli-Parisi splitting function times the collinear logarithm
**evaluated at the outgoing pair mass** ``s' = z s``, minus one.  The ``-1``
is the non-logarithmic part of the soft eikonal of a back-to-back pair, and
the ``ln z`` is the entire non-logarithmic hard remainder: the O(alpha)
radiator has *no* other non-log term.  Equation (1) is the standard
final-state radiator of Berends-Kleiss-Jadach (Nucl. Phys. B202 (1982) 63),
the ZFITTER final-state radiator (Bardin et al., Comput. Phys. Commun. 133
(2001) 229) and Bardin-Passarino, "The Standard Model in the Making".

`validate` checks (1) against a numerical evaluation of the exact
spin-summed |M|^2 for ``V* -> mu+ mu- gamma`` (Dirac traces, exact muon mass,
transverse projector ``-g + QQ/s``, 2-D quadrature over the Dalitz variable)
and finds

    |R_exact/R1 - 1|  <  6e-6    at m = 91.19 GeV  (all z; scales as m_mu^2/s)
    |R_exact/R1 - 1|  <  9e-3    at m = 3.097 GeV  (J/psi)
    |R_A - R_V| / R   <  1e-4    at m = 91.19 GeV,

so the muon mass terms and the vector/axial (i.e. gamma*/Z) decomposition are
irrelevant at the Z to far below the 3e-3 target, and only matter at the J/psi.

Soft limit:  ``R1 -> beta/(1-z)`` as ``z -> 1``, reproduced by the exact
numerics to 1e-6.

The kernels
-----------
Every variant is a probability density in ``z`` on ``(0, 1]`` normalised to
``int_0^1 K dz = 1``.  The overall O(alpha) *rate* correction ``3 alpha/4pi``
is z- and m-independent and therefore drops out of the normalised kernel.

``exp1`` - **exponentiated exact O(alpha)** (YFS / Kuraev-Fadin form):

    K(z) = C beta (1-z)^{beta-1} + h(z) ,
    h(z) = -(beta/2)(1+z) + (alpha/pi) (1+z^2) ln z / (1-z) ,                   (2)
    C    = 1 - int_0^1 h dz = 1 + 3 beta/4 + (alpha/pi)(pi^2/3 - 5/4) .

    `h` is (1) minus the soft term ``beta/(1-z)`` that the exponentiated
    factor already supplies, so the O(alpha) expansion of (2),
    ``beta(1-z)^{beta-1} = delta(1-z) + beta [1/(1-z)]_+ + O(beta^2)``,
    reproduces (1) exactly.  ``int_0^1 (1+z^2) ln z/(1-z) dz = 5/4 - pi^2/3``.

``exp2`` - ``exp1`` plus the **O(alpha^2) leading log**.  The LL radiator for
    the pair-mass fraction is, in Mellin space, ``R~(n) = exp[(beta/2) g(n)]``
    with ``g(n) = 3/2 - 2 S_{n-1} - 1/n - 1/(n+1)`` the QED non-singlet
    anomalous dimension (two radiating legs, each carrying ``beta/2``).  Its
    O(beta^2) term is ``(beta^2/8) [P (x) P](z)`` with
    ``P = [(1+z^2)/(1-z)]_+``; the explicit convolution is

    [P(x)P](z) = (9/4 - 2pi^2/3) delta(1-z) + 6 [1/(1-z)]_+
               + 8 [ln(1-z)/(1-z)]_+
               + (1+z)[3 ln z - 4 ln(1-z)] - 4 ln z/(1-z) - 5 - z .             (3)

    The two plus-distributions of (3) are *already* contained in (2): the
    ``beta^2 [ln(1-z)/(1-z)]_+`` term of ``beta(1-z)^{beta-1}`` matches
    ``(beta^2/8) . 8`` exactly, and ``(3 beta/4) . beta [1/(1-z)]_+`` from the
    constant ``C`` matches ``(beta^2/8) . 6`` exactly.  Only the regular part
    of (3) is new, and its delta coefficient is fixed by ``int K = 1``
    (``int [P(x)P] = 0``, as probability conservation requires).  This is the
    Kuraev-Fadin second-order structure function (Sov. J. Nucl. Phys. 41
    (1985) 466) written for the two-leg exponent ``beta``.

``oalpha`` - fixed-order O(alpha) with a soft cutoff ``x_cut``: a delta at
    ``z = 1`` carrying ``1 - P(x > x_cut)`` plus (1) above the cutoff.  Not a
    model - it exists to show the size of the exponentiation.

``pair`` suffixes add **real lepton/hadron pair emission** from the muon line
    (see `beta_pair`), which Photos does not generate: the soft part is
    exponentiated together with the photon (``beta -> beta + beta_pair``) and
    the hard remainder ``-(beta_pair/2)(1+z)`` is added to `h`.

Discretisation
--------------
The provider's fold ``p_post(m) = sum_j w_j p_born(m/r_j)/r_j`` is a midpoint
quadrature of ``int dr k(r) p_born(m/r)/r``.  Atoms are groups of ``u`` placed
at their *exact* conditional mean, so the residual of a group is
``w_j Var(u|group) m^2 |p''| / 2``; groups are grown until ``w_j Var_j``
exceeds ``--var-budget`` (or the group sd exceeds ``--sigma-cap``).  Both the
weights and the conditional moments are computed by Gauss-Legendre quadrature
in ``t = (1-z)^beta``, the substitution that removes the ``(1-z)^{beta-1}``
endpoint singularity exactly (``dt = beta (1-z)^{beta-1} dz``).

Banding in ``m_pre`` is free for an analytic kernel: one kernel per band with
its own ``beta(m)``, written as per-atom ``m_lo``/``m_hi`` for
`rabbit.lineshapes.zgamma`.
"""
import argparse
import json
import math

import numpy as np

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------
ALPHA = 1.0 / 137.035999            # on-shell (Thomson) fine-structure constant
M_MU = 0.1056583745                 # PDG muon mass [GeV]
M_E = 0.51099895e-3                 # PDG electron mass [GeV]
M_TAU = 1.77686                     # PDG tau mass [GeV]
A_PI = ALPHA / math.pi

#: int_0^1 (1+z^2) ln z / (1-z) dz = 5/4 - pi^2/3
INT_F = 1.25 - math.pi**2 / 3.0     # = -2.0398684...
#: int_0^1 of the regular part of [P (x) P](z), eq. (3)
INT_Q2 = 4.3298684                  # = 2 pi^2/3 - 9/4, computed below


def _int_q2():
    """int_0^1 { (1+z)(3 ln z - 4 ln(1-z)) - 4 ln z/(1-z) - 5 - z } dz."""
    return 3.0 * (-1.25) - 4.0 * (-1.75) - 4.0 * (-math.pi**2 / 6.0) - 5.5


INT_Q2 = _int_q2()


def coll_log(m):
    """L(m) = ln(m^2 / m_mu^2)."""
    return 2.0 * np.log(np.asarray(m, float) / M_MU)


def beta_fsr(m, minus_one=True):
    """beta(m) = (2 alpha/pi) (L - 1); ``minus_one=False`` drops the -1.

    The ``-1`` is the exact non-logarithmic part of the soft eikonal integral
    for a back-to-back pair; dropping it is the conventional NLL sensitivity
    bound on the radiator strength (it moves beta by 1/(L-1) = 8 %).
    """
    L = coll_log(m)
    return 2.0 * A_PI * (L - (1.0 if minus_one else 0.0))


# --------------------------------------------------------------------------
# pair emission
# --------------------------------------------------------------------------
def _rho_lepton(q2, ml):
    """Im Pi(q^2)/pi for a lepton pair: (alpha/3pi)(1 + 2 m^2/q^2) beta_l."""
    q2 = np.asarray(q2, float)
    b = np.sqrt(np.maximum(1.0 - 4.0 * ml * ml / q2, 0.0))
    return (ALPHA / (3.0 * math.pi)) * (1.0 + 2.0 * ml * ml / q2) * b


def _rho_had(q2, r_had=2.0, q2_min=1.0):
    """Im Pi(q^2)/pi for hadrons: (alpha/3pi) R(q^2), R ~ const above 1 GeV^2.

    A deliberately crude R: the hadronic pair term is 0.4 % of the photonic
    radiator, so a 20 % error on R is 1e-3 of it.
    """
    q2 = np.asarray(q2, float)
    return np.where(q2 >= q2_min, (ALPHA / (3.0 * math.pi)) * r_had, 0.0)


def beta_pair(m, species=("e",), r_had=2.0, ngauss=400):
    r"""Soft exponent of real pair emission off the final-state muon line.

    A virtual photon of mass^2 ``q^2`` radiated off the muon converts to a
    pair; the pair removes the same energy as a photon would, so at leading
    log the ``z`` dependence is the photon one and only the coefficient
    changes.  The collinear logarithm is cut off at ``q^2`` instead of
    ``m_mu^2``, so

        beta_pair = (2 alpha/pi) int_{q2_thr}^{s} (dq^2/q^2) rho(q^2)
                                 [ ln(s/q^2) - 1 ]                              (4)

    with ``rho = Im Pi/pi`` the vacuum-polarisation spectral density.  This is
    the leading-log (alpha^2 L^2) pair correction; the ``-1`` and the upper
    limit are NLL and carry a ~1/L = 8 % relative uncertainty on (4).

    Virtual pairs (the vacuum-polarisation insertion in the one-loop vertex)
    cancel the soft part of (4) and otherwise change only the overall rate, so
    they do not enter a normalised kernel.
    """
    s = float(m) ** 2
    out = 0.0
    for sp in species:
        if sp in ("e", "mu", "tau"):
            ml = {"e": M_E, "mu": M_MU, "tau": M_TAU}[sp]
            q2lo, rho = 4.0 * ml * ml, (lambda q2, ml=ml: _rho_lepton(q2, ml))
        elif sp == "had":
            q2lo, rho = 1.0, (lambda q2: _rho_had(q2, r_had))
        else:
            raise ValueError(f"unknown pair species {sp!r}")
        if q2lo >= s:
            continue
        # log-spaced Gauss-Legendre in y = ln(q^2)
        ylo, yhi = math.log(q2lo), math.log(s)
        x, wq = np.polynomial.legendre.leggauss(ngauss)
        y = 0.5 * (yhi - ylo) * (x + 1.0) + ylo
        w = 0.5 * (yhi - ylo) * wq
        q2 = np.exp(y)
        out += float(np.sum(w * rho(q2) * (np.log(s / q2) - 1.0)))
    return 2.0 * A_PI * out


# --------------------------------------------------------------------------
# the kernel
# --------------------------------------------------------------------------
VARIANTS = ("exp1", "exp2", "oalpha", "born")


class FSRKernel:
    """K(z; m): the normalised probability density of z = (m_post/m_pre)^2.

    Parameters
    ----------
    m : float
        pre-FSR mass at which beta and the collinear log are evaluated.
    variant : {"exp1", "exp2", "oalpha", "born"}
    pair : tuple of str
        pair-emission species to add, e.g. ``("e",)`` or ``("e","mu","had")``.
    minus_one : bool
        keep the ``-1`` in ``beta = (2 alpha/pi)(L-1)``.
    x_cut : float
        soft cutoff of the ``oalpha`` variant, ``x = 1-z``.
    zmin : float
        lower end of the support; the default is the exact 2-muon threshold
        ``4 m_mu^2/s``.
    """

    def __init__(self, m, variant="exp1", pair=(), minus_one=True,
                 x_cut=1e-7, zmin=None, beta_at=None):
        self.m = float(m)
        self.variant = variant
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        mb = self.m if beta_at is None else float(beta_at)
        self.beta_at = beta_at
        self.L = float(coll_log(mb))
        self.beta_gam = float(beta_fsr(mb, minus_one))
        self.beta_pair = float(beta_pair(mb, pair)) if pair else 0.0
        self.pair = tuple(pair)
        self.beta = self.beta_gam + self.beta_pair
        self.x_cut = float(x_cut)
        self.zmin = float(zmin) if zmin is not None else 4.0 * M_MU**2 / self.m**2
        self._C = 1.0 - self.int_h() - (self.int_q2() if variant == "exp2" else 0.0)

    # -- the pieces --------------------------------------------------------
    # Everything is written in terms of BOTH z and x = 1-z.  The quadrature
    # variable is t = x^beta, so x can be as small as 1e-170 while z rounds to
    # 1 exactly; passing x explicitly is what keeps (1-z)^{beta-1} and
    # ln z = log1p(-x) accurate there.
    def r1(self, z, x=None):
        """Exact O(alpha) real spectrum, eq. (1) (photon only)."""
        z = np.asarray(z, float)
        x = (1.0 - z) if x is None else np.asarray(x, float)
        return A_PI * (1.0 + z * z) / x * (self.L - 1.0 + np.log1p(-x))

    def h(self, z, x=None):
        """Hard remainder of eq. (2): O(alpha) minus the exponentiated soft."""
        z = np.asarray(z, float)
        x = (1.0 - z) if x is None else np.asarray(x, float)
        out = (-0.5 * self.beta * (1.0 + z)
               + A_PI * (1.0 + z * z) * np.log1p(-x) / x)
        return out

    def int_h(self):
        return -0.75 * self.beta + A_PI * INT_F

    def q2(self, z, x=None):
        """Regular part of the O(alpha^2) LL term, (beta^2/8) x eq. (3)."""
        z = np.asarray(z, float)
        x = (1.0 - z) if x is None else np.asarray(x, float)
        lz = np.log1p(-x)
        return (self.beta**2 / 8.0) * (
            (1.0 + z) * (3.0 * lz - 4.0 * np.log(x))
            - 4.0 * lz / x - 5.0 - z)

    def int_q2(self):
        return (self.beta**2 / 8.0) * INT_Q2

    # -- the density -------------------------------------------------------
    def pdf_z(self, z, x=None):
        """K(z) for z < 1.  ``x = 1-z`` may be supplied at full precision.

        The delta at z = 1 of the ``oalpha`` variant is not included here.
        """
        z = np.asarray(z, float)
        x = (1.0 - z) if x is None else np.asarray(x, float)
        if self.variant == "born":
            return np.zeros_like(z)
        if self.variant == "oalpha":
            out = np.where(x > self.x_cut, self.r1(z, x), 0.0)
            return np.where(z >= self.zmin, np.maximum(out, 0.0), 0.0)
        out = self.h(z, x) + self._C * self.beta * np.power(x, self.beta - 1.0)
        if self.variant == "exp2":
            out = out + self.q2(z, x)
        return np.where(z >= self.zmin, np.maximum(out, 0.0), 0.0)

    # -- quadrature in t = (1-z)^beta -------------------------------------
    # dt = beta x^{beta-1} dz with x = 1-z, so the endpoint singularity of the
    # exponentiated term is absorbed exactly:
    #     K dz  ->  [ C + (h + q2) x^{1-beta}/beta ] dt ,
    # a bounded integrand that plain Gauss-Legendre in t integrates to machine
    # precision.  The integrand is evaluated in this form rather than as
    # K(z) |dz/dt| because x reaches 1e-100 near t = 0, where x^{beta-1} and
    # x^{1-beta} individually overflow/underflow while their product does not.
    def _t_of_u(self, u):
        """t = (1-z)^beta from u = -ln(m'/m); x = 1-z = -expm1(-2u)."""
        return np.power(-np.expm1(-2.0 * np.asarray(u, float)), self.beta)

    def _u_of_t(self, t):
        return -0.5 * np.log1p(-np.power(np.asarray(t, float), 1.0 / self.beta))

    def _panel(self, t_lo, t_hi, ng):
        """GL nodes on [t_lo, t_hi]: returns (u, x, w) with sum(w) = int K dz.

        ``t_lo``/``t_hi`` may be arrays, one entry per panel; the returned
        arrays then carry a leading panel axis.
        """
        g, wg = np.polynomial.legendre.leggauss(ng)
        t_lo = np.asarray(t_lo, float)[..., None]
        t_hi = np.asarray(t_hi, float)[..., None]
        t = 0.5 * (t_hi - t_lo) * (g + 1.0) + t_lo
        w = 0.5 * (t_hi - t_lo) * wg
        x = np.power(t, 1.0 / self.beta)
        z = 1.0 - x
        u = -0.5 * np.log1p(-x)
        jac = np.power(x, 1.0 - self.beta) / self.beta          # = |dz/dt|
        if self.variant in ("exp1", "exp2"):
            f = self.h(z, x)
            if self.variant == "exp2":
                f = f + self.q2(z, x)
            integ = self._C + f * jac
            integ = np.where(z >= self.zmin, np.maximum(integ, 0.0), 0.0)
        else:
            integ = self.pdf_z(z, x) * jac
        return u, x, w * integ

    def _u_max(self):
        return -0.5 * math.log(self.zmin)

    def _fine_grid(self, u_max=None, n=4000):
        """Panel edges in ``t``: uniform in t, merged with a geometric u ladder.

        Uniform in ``t`` gives equal weight per panel for the soft term and
        resolves ``u -> 0``; the geometric ladder in ``u`` keeps the hard tail,
        where ``t`` saturates at 1, from being covered by a handful of panels.
        """
        u_max = self._u_max() if u_max is None else float(u_max)
        t_end = float(self._t_of_u(u_max))
        tg = np.concatenate([np.linspace(0.0, t_end, n + 1),
                             self._t_of_u(np.geomspace(1e-7, u_max, n // 2))])
        return np.unique(np.clip(tg, 0.0, t_end))

    def _cells(self, tg, ng=16):
        """Per-panel (w, int u K, int u^2 K, int (1-z) K)."""
        u, x, w = self._panel(tg[:-1], tg[1:], ng)
        return np.stack([w.sum(1), (w * u).sum(1), (w * u * u).sum(1),
                         (w * x).sum(1)])

    def moments(self, u_max=None, n=4000, ng=16):
        """(norm, <u>, <u^2>, <1-z>) of the normalised kernel."""
        c = self._cells(self._fine_grid(u_max, n), ng).sum(axis=1)
        n0 = max(c[0], 1.0) if self.variant == "oalpha" else c[0]
        return dict(norm=float(c[0]), u=float(c[1] / n0), u2=float(c[2] / n0),
                    x=float(c[3] / n0))

    def tail(self, u0, n=4000, ng=16):
        """P(u > u0) of the normalised kernel."""
        u0 = np.atleast_1d(np.asarray(u0, float))
        tg = self._fine_grid(None, n)
        c = self._cells(tg, ng)[0]
        n0 = max(c.sum(), 1.0) if self.variant == "oalpha" else c.sum()
        above = np.concatenate([np.cumsum(c[::-1])[::-1], [0.0]])   # weight above tg[i]
        ug = self._u_of_t(tg)
        out = np.zeros_like(u0)
        for i, uu in enumerate(u0):
            if uu >= ug[-1]:
                continue
            j = int(max(min(np.searchsorted(ug, uu) - 1, len(tg) - 2), 0))
            _, _, w = self._panel(self._t_of_u([uu]), tg[j + 1:j + 2], ng)
            out[i] = (above[j + 1] + w.sum()) / n0
        return out

    def moments_below(self, u_cut, n=4000, ng=16):
        """(P, <u>, <u^2>) of the kernel restricted to u < u_cut.

        The window-truncated mean is what a narrow-resonance fit actually
        sees: events radiating out of the selection window are lost, not
        shifted, so the bias on a fitted mass is ``-m <u | u < u_cut>``.
        """
        tg = self._fine_grid(None, n)
        tc = float(self._t_of_u(u_cut))
        tg = np.unique(np.clip(np.concatenate([tg, [tc]]), 0.0, tg[-1]))
        tg = tg[tg <= tc]
        c = self._cells(tg, ng).sum(axis=1)
        tot = self._cells(self._fine_grid(None, n), ng)[0].sum()
        return dict(p=float(c[0] / tot), u=float(c[1] / c[0]),
                    u2=float(c[2] / c[0]))

    def p_norad(self, x_cut=None):
        """P(1-z < x_cut): the 'unradiated' fraction at an IR cutoff."""
        x_cut = self.x_cut if x_cut is None else x_cut
        return 1.0 - float(self.tail(-0.5 * math.log1p(-x_cut))[0])

    def pdf_u(self, u):
        """Density in u = -ln(m'/m):  K_u(u) = 2 z K(z), z = e^{-2u}."""
        u = np.asarray(u, float)
        x = -np.expm1(-2.0 * u)
        return 2.0 * (1.0 - x) * self.pdf_z(1.0 - x, x)

    # -- atoms -------------------------------------------------------------
    def atoms(self, u_max=None, var_budget=6e-10, sigma_cap=None,
              n_fine=4000, ng=16):
        """Discretise into ``(r_j, w_j)`` with the first moment of u exact.

        Panels of the fine grid are merged while ``w Var(u|group)`` stays below
        ``var_budget`` (or the group sd below ``sigma_cap``); each atom sits at
        the exact conditional mean of ``u``, which makes the provider's
        midpoint fold first-order exact and leaves a residual
        ``w_j Var_j m^2 |p''| / 2`` per atom.
        """
        w, m1, m2, _ = self._cells(self._fine_grid(u_max, n_fine), ng)
        rj, wj = [], []
        a0 = a1 = a2 = 0.0
        for i in range(len(w)):
            b0, b1, b2 = a0 + w[i], a1 + m1[i], a2 + m2[i]
            if b0 <= 0.0:
                continue
            mu = b1 / b0
            var = max(b2 / b0 - mu * mu, 0.0)
            too_wide = (math.sqrt(var) > sigma_cap if sigma_cap
                        else b0 * var > var_budget)
            if too_wide and a0 > 0.0:
                rj.append(math.exp(-a1 / a0))
                wj.append(a0)
                a0, a1, a2 = w[i], m1[i], m2[i]
            else:
                a0, a1, a2 = b0, b1, b2
        if a0 > 0.0:
            rj.append(math.exp(-a1 / a0))
            wj.append(a0)
        rj = np.asarray(rj)
        wj = np.asarray(wj)
        if self.variant == "oalpha":
            rj = np.concatenate([[1.0], rj])
            wj = np.concatenate([[max(1.0 - wj.sum(), 0.0)], wj])
        o = np.argsort(-rj)
        rj, wj = rj[o], wj[o]
        tot = wj.sum()
        return rj, wj / tot, tot


# --------------------------------------------------------------------------
# banded kernels
# --------------------------------------------------------------------------
def band_edges(lo, hi, width):
    n = int(math.ceil((hi - lo) / width))
    return lo + width * np.arange(n + 1)


def build_banded(edges, variant="exp1", pair=(), minus_one=True,
                 beta_at=None, u_max=None, **kw):
    """One kernel per ``m_pre`` band -> flat ``(r, w, m_lo, m_hi)`` arrays.

    The outermost bands are opened to ``0`` and ``inf`` so every pre-FSR mass
    the provider can reach is covered; ``beta`` is evaluated at the band
    centre (at the open edges, at the finite edge).
    """
    edges = np.asarray(edges, float)
    R, W, LO, HI, info = [], [], [], [], []
    n = len(edges) - 1
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        mc = 0.5 * (lo + hi)
        k = FSRKernel(mc, variant=variant, pair=pair, minus_one=minus_one,
                      beta_at=beta_at)
        r, w, tot = k.atoms(u_max=u_max, **kw)
        blo = 0.0 if i == 0 else lo
        bhi = np.inf if i == n - 1 else hi
        R.append(r)
        W.append(w)
        LO.append(np.full(len(r), blo))
        HI.append(np.full(len(r), bhi))
        info.append(dict(m=mc, lo=blo, hi=float(bhi) if np.isfinite(bhi) else None,
                         beta=k.beta, beta_pair=k.beta_pair, natoms=len(r),
                         captured=tot, mean_u=-float(np.sum(w * np.log(r)))))
    return (dict(r=np.concatenate(R), w=np.concatenate(W),
                 m_lo=np.concatenate(LO), m_hi=np.concatenate(HI)), info)


# --------------------------------------------------------------------------
# exact O(alpha) V* -> mu+ mu- gamma, by numerical Dirac traces
# --------------------------------------------------------------------------
# Ground truth for eq. (1): the spin-summed squared matrix element with the
# exact muon mass, contracted with the transverse projector
# ``P_{mu mu'} = -g_{mu mu'} + Q_mu Q_{mu'}/s`` on the current indices and with
# ``-g_{nu nu'}`` on the photon index (legitimate because
# ``k_nu M^{mu nu} = 0`` for either current), integrated over the muon
# direction in the mu-mu rest frame.  The two propagators are exactly
# ``1 -+ beta_mu cos(theta*)``, so the collinear region is resolved by the
# substitution ``1 -+ beta_mu cos = (1 - beta_mu) e^t``.
_I2 = np.eye(2, dtype=complex)
_S = [np.array([[0, 1], [1, 0]], dtype=complex),
      np.array([[0, -1j], [1j, 0]], dtype=complex),
      np.array([[1, 0], [0, -1]], dtype=complex)]
_Z2 = np.zeros((2, 2), dtype=complex)
_GAMMA = np.zeros((4, 4, 4), dtype=complex)
_GAMMA[0] = np.block([[_I2, _Z2], [_Z2, -_I2]])
for _i, _s in enumerate(_S):
    _GAMMA[_i + 1] = np.block([[_Z2, _s], [-_s, _Z2]])
_G5 = np.block([[_Z2, _I2], [_I2, _Z2]])
_ETA = np.diag([1.0, -1.0, -1.0, -1.0])
_LOW = np.array([1.0, -1.0, -1.0, -1.0])


def _slash(p):
    return np.einsum('...m,mij->...ij', p * _LOW, _GAMMA)


def _bar(X):
    return np.einsum('ij,...kj,kl->...il', _GAMMA[0], X.conj(), _GAMMA[0])


def _T_rad(s, z, c, mmu, va):
    """(1/3) P (-g) sum_spins |M|^2 for V* -> mu mu gamma, without e^2."""
    v, a = va
    c = np.atleast_1d(np.asarray(c, float))
    sp = z * s
    E = 0.5 * math.sqrt(sp)
    bm = math.sqrt(max(1.0 - 4.0 * mmu * mmu / sp, 0.0))
    q = E * bm
    st = np.sqrt(np.maximum(1.0 - c * c, 0.0))
    zer = np.zeros_like(c)
    one = np.ones_like(c)
    pm = np.stack([E * one, q * st, zer, q * c], -1)
    pp = np.stack([E * one, -q * st, zer, -q * c], -1)
    w = (s - sp) / (2.0 * math.sqrt(sp))
    k = np.stack([w * one, zer, zer, w * one], -1)
    Q = pm + pp + k
    Gam = v * _GAMMA + a * np.einsum('mij,jk->mik', _GAMMA, _G5)
    num1 = _slash(pm) + _slash(k) + mmu * np.eye(4)
    num2 = mmu * np.eye(4) - _slash(pp) - _slash(k)
    dot = lambda x, y: (x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
                        - x[..., 2] * y[..., 2] - x[..., 3] * y[..., 3])
    d1 = (2.0 * dot(pm, k))[:, None, None, None, None]
    d2 = (2.0 * dot(pp, k))[:, None, None, None, None]
    A = np.einsum('nij,pjk->pnik', _GAMMA, num1)
    Xa = np.einsum('pnij,mjk->pnmik', A, Gam) / d1
    B = np.einsum('mij,pjk->pmik', Gam, num2)
    Xb = np.einsum('pmij,njk->pnmik', B, _GAMMA) / d2
    X = Xa + Xb
    L = _slash(pm) + mmu * np.eye(4)
    R = _slash(pp) - mmu * np.eye(4)
    LX = np.einsum('pij,pnmjk->pnmik', L, X)
    RXb = np.einsum('pij,pNMjk->pNMik', R, _bar(X))
    Tr = np.einsum('pnmij,pNMji->pnmNM', LX, RXb)
    Ql = Q * _LOW
    Pmm = -_ETA + np.einsum('pi,pj->pij', Ql, Ql) / s
    return np.einsum('nN,pmM,pnmNM->p', -_ETA, Pmm, Tr).real / 3.0


def _T_born(s, mmu, va):
    v, a = va
    E = 0.5 * math.sqrt(s)
    q = E * math.sqrt(1.0 - 4.0 * mmu * mmu / s)
    pm = np.array([[E, 0.0, 0.0, q]])
    pp = np.array([[E, 0.0, 0.0, -q]])
    Q = pm + pp
    Gam = v * _GAMMA + a * np.einsum('mij,jk->mik', _GAMMA, _G5)
    L = _slash(pm) + mmu * np.eye(4)
    R = _slash(pp) - mmu * np.eye(4)
    LG = np.einsum('pij,mjk->pmik', L, Gam)
    RG = np.einsum('pij,Mjk->pMik', R, _bar(Gam))
    Tr = np.einsum('pmij,pMji->pmM', LG, RG)
    Ql = Q * _LOW
    Pmm = -_ETA + np.einsum('pi,pj->pij', Ql, Ql) / s
    return float(np.einsum('pmM,pmM->', Pmm, Tr).real / 3.0)


def r1_exact(z, m, mmu=M_MU, va=(1.0, 0.0), ng=120):
    """(1/Gamma_0) dGamma/dz from the exact matrix element and muon mass.

    ``va = (v, a)`` selects the current: ``(1, 0)`` vector, ``(0, 1)`` axial.
    """
    s = m * m
    sp = z * s
    if sp <= 4.0 * mmu * mmu:
        return 0.0
    bm = math.sqrt(1.0 - 4.0 * mmu * mmu / sp)
    bm0 = math.sqrt(1.0 - 4.0 * mmu * mmu / s)
    tA = -math.log(1.0 - bm)
    g, wq = np.polynomial.legendre.leggauss(ng)
    t = 0.5 * tA * (g + 1.0)
    wt = 0.5 * tA * wq * ((1.0 - bm) / bm) * np.exp(t)
    cA = (1.0 - (1.0 - bm) * np.exp(t)) / bm          # cos theta* in [0, 1]
    cB = ((1.0 - bm) * np.exp(t) - 1.0) / bm          # cos theta* in [-1, 0]
    c = np.concatenate([cA, cB])
    w = np.concatenate([wt, wt])
    T = _T_rad(s, z, c, mmu, va)
    return (ALPHA * s * (1.0 - z) / (4.0 * math.pi) * (bm / bm0)
            * 0.5 * float(np.sum(w * T)) / _T_born(s, mmu, va))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _kernel_cmd(args):
    pair = tuple(args.pair or ())
    if args.band_width > 0:
        edges = band_edges(args.band_lo, args.band_hi, args.band_width)
    else:
        edges = np.array([args.band_lo, args.band_hi])
    ker, info = build_banded(
        edges, variant=args.variant, pair=pair, minus_one=not args.no_minus_one,
        beta_at=args.freeze_beta_at, u_max=args.u_max,
        var_budget=(None if args.sigma_cap else args.var_budget),
        sigma_cap=args.sigma_cap, n_fine=args.n_fine, ng=args.ng)
    meta = dict(kind="analytic", variant=args.variant, pair=list(pair),
                minus_one=not args.no_minus_one,
                freeze_beta_at=args.freeze_beta_at, alpha=ALPHA, m_mu=M_MU,
                band_lo=args.band_lo, band_hi=args.band_hi,
                band_width=args.band_width, u_max=args.u_max,
                var_budget=args.var_budget, sigma_cap=args.sigma_cap,
                n_fine=args.n_fine, ng=args.ng, bands=info)
    np.savez(args.output, r=ker["r"], w=ker["w"], m_lo=ker["m_lo"],
             m_hi=ker["m_hi"], provenance=np.array([json.dumps(meta)]))
    print(f"[kernel] {args.variant}{'+pairs' + '/'.join(pair) if pair else ''}: "
          f"{len(edges)-1} bands, {len(ker['r'])} atoms -> {args.output}")
    for b in info[:: max(1, len(info) // 8)]:
        print(f"    m = {b['m']:7.2f}  beta = {b['beta']:.6f}"
              f"  beta_pair = {b['beta_pair']:.3e}  atoms = {b['natoms']:4d}"
              f"  <u> = {b['mean_u']*1e3:8.4f}e-3")


def _validate_cmd(args):
    print("Exact O(alpha) matrix element vs the closed form")
    print("  R1(z) = (alpha/pi) (1+z^2)/(1-z) [ ln(z m^2/m_mu^2) - 1 ]\n")
    ms = [91.1876, 9.4603, 3.0969]
    print("relative deviation R_exact/R1 - 1, vector current:")
    print(f"{'z':>10s}" + "".join(f"{'m=%.4g' % m:>14s}" for m in ms)
          + f"{'m=mZ,mmu/30':>14s}")
    zs = [0.9999, 0.999, 0.99, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1, 0.05, 0.02]
    for z in zs:
        cells = []
        for m in ms:
            cells.append(r1_exact(z, m) / (A_PI * (1 + z * z) / (1 - z)
                                           * (math.log(z * m * m / M_MU**2) - 1)) - 1)
        mm = M_MU / 30
        cells.append(r1_exact(z, 91.1876, mmu=mm)
                     / (A_PI * (1 + z * z) / (1 - z)
                        * (math.log(z * 91.1876**2 / mm**2) - 1)) - 1)
        print(f"{z:10.4f}" + "".join(f"{c:14.3e}" for c in cells))
    print("\nsoft limit R_exact (1-z)/beta  (should -> 1):")
    for x in (1e-3, 1e-4, 1e-5, 1e-6):
        z = 1 - x
        print(f"   1-z = {x:.0e}: {r1_exact(z, 91.1876) * x / beta_fsr(91.1876):.6f}")
    print("\naxial vs vector, |R_A/R_V - 1|:")
    for m in ms:
        d = [abs(r1_exact(z, m, va=(0., 1.)) / r1_exact(z, m, va=(1., 0.)) - 1)
             for z in (0.99, 0.9, 0.5)]
        print(f"   m = {m:8.4f}: z=0.99 {d[0]:.2e}  z=0.9 {d[1]:.2e}  z=0.5 {d[2]:.2e}")


def _moments_cmd(args):
    for m in args.mass:
        if args.u_cut:
            k = FSRKernel(m, variant="exp1")
            k2 = FSRKernel(m, variant="exp2")
            for kk, tag in ((k, "exp1"), (k2, "exp2")):
                b = kk.moments_below(args.u_cut)
                print(f"  [{tag}] m = {m:8.4f}  beta = {kk.beta:.6f}"
                      f"  P(u < {args.u_cut:.5f}) = {b['p']:.6f}"
                      f"  <u | in window> = {b['u']*1e3:8.4f}e-3"
                      f"  -> dm = {-m*b['u']*1e3:8.3f} MeV")
        print(f"\nm_pre = {m} GeV   L = {float(coll_log(m)):.6f}")
        for v in ("exp1", "exp2", "oalpha"):
            for pair in ((), ("e",), ("e", "mu", "tau", "had")):
                if v != "exp1" and pair:
                    continue
                k = FSRKernel(m, variant=v, pair=pair)
                d = k.moments()
                tag = v + ("+pair:" + "/".join(pair) if pair else "")
                print(f"  {tag:<24s} beta = {k.beta:.6f}"
                      f"  (pair {k.beta_pair:.3e})"
                      f"  <u> = {d['u']*1e3:8.4f}e-3  <1-z> = {d['x']*1e3:8.4f}e-3"
                      f"  P(u>0.01) = {float(k.tail([0.01])[0]):.6f}"
                      f"  P0(1e-7) = {k.p_norad(1e-7):.6f}")
        k = FSRKernel(m, minus_one=False)
        print(f"  {'exp1, L instead of L-1':<24s} beta = {k.beta:.6f}"
              f"                   <u> = {k.moments()['u']*1e3:8.4f}e-3")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    k = sub.add_parser("kernel", help="write the atom npz for the provider")
    k.add_argument("-o", "--output", required=True)
    k.add_argument("--variant", default="exp1", choices=VARIANTS)
    k.add_argument("--pair", nargs="*", default=None,
                   choices=["e", "mu", "tau", "had"])
    k.add_argument("--no-minus-one", action="store_true",
                   help="use L instead of L-1 in beta (NLL sensitivity bound)")
    k.add_argument("--freeze-beta-at", type=float, default=None,
                   help="evaluate beta at this mass in every band")
    k.add_argument("--band-lo", type=float, default=50.0)
    k.add_argument("--band-hi", type=float, default=200.0)
    k.add_argument("--band-width", type=float, default=2.0,
                   help="m_pre band width in GeV; 0 = a single band")
    k.add_argument("--u-max", type=float, default=None,
                   help="default: the 2-muon threshold, u = -ln(2 m_mu/m)")
    k.add_argument("--var-budget", type=float, default=6e-10)
    k.add_argument("--sigma-cap", type=float, default=None)
    k.add_argument("--n-fine", type=int, default=4000)
    k.add_argument("--ng", type=int, default=16)
    k.set_defaults(func=_kernel_cmd)

    v = sub.add_parser("validate", help="exact matrix element vs the closed form")
    v.set_defaults(func=_validate_cmd)

    m = sub.add_parser("moments", help="analytic moments at a few masses")
    m.add_argument("--mass", nargs="*", type=float,
                   default=[91.1876, 9.4603, 3.0969])
    m.add_argument("--u-cut", type=float, default=None,
                   help="also report the window-truncated mean below this u")
    m.set_defaults(func=_moments_cmd)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
