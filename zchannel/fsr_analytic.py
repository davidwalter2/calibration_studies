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

``exp2nll`` - ``exp2`` plus the **O(alpha^2) next-to-leading log**, i.e. every
    term of order ``alpha^2 L`` (``alpha^2 L^2`` is already in ``exp2``).

    The pair-mass fraction factorises, ``z = z_+ z_-``, so in Mellin space the
    kernel is the *square* of the muon's time-like (fragmentation) QED
    structure function ``D(z; L)``, which evolves with ``a = alpha/2pi`` as

        dD/dL = a P0 (x) D + a^2 P1T (x) D ,  D(z; 0) = delta(1-z) + a C1 + ...

    Solving to O(a^2) and squaring,

        K~(n) = exp[ (alpha/pi)(L g0 + c1) + (alpha/pi)^2 (L/2) g1 ] + NNLL ,  (5)

    lower case = Mellin moment of the corresponding capital.  The O(alpha) term
    of (5) *is* the exact spectrum (1), which therefore **defines** the O(alpha)
    coefficient function: with ``p(z) = (1+z^2)/(1-z)`` and
    ``P0 = [p]_+ = 2[1/(1-z)]_+ - (1+z) + (3/2) delta(1-z)``,

        C1(z) = [ p(z) (ln z - 1) ]_+ = Chat(z) - P0(z) ,                      (6)
        Chat(z) = p(z) ln z + (pi^2/3 - 5/4) delta(1-z) ,

    and the O(alpha^2) term of (5) is

        (alpha/pi)^2 { (L^2/2) P0 (x) P0 + L [ (1/2) P1T + P0 (x) C1 ] } .     (7)

    ``exp2`` already carries the whole ``L^2`` coefficient and part of the ``L``
    one: its O(alpha^2) content is exactly
    ``(beta^2/8) P0 (x) P0 + (alpha/pi) beta (pi^2/3 - 5/4) [1/(1-z)]_+``, and
    because ``beta = (2 alpha/pi)(L-1)`` the cross term of ``(L-1)^2`` supplies
    ``-(alpha/pi)^2 L P0 (x) P0``, which is precisely the ``-P0 (x) P0`` hiding
    inside ``P0 (x) C1``.  The remainder - the whole of ``exp2nll`` - is

        Delta K(z) = (alpha/pi)^2 L [ G(z) + g_delta delta(1-z) ] ,            (8)
        G(z)  = 3 p(z) ln z ln(1-z) + [ 9/(2(1-z)) - 5 - 2z ] ln z
              + [ (11/4)(1+z) - 4/(1-z) ] ln^2 z
              - (5/2)(1-z) - (pi^2/3 - 5/4)(1+z) ,
        g_delta = P1_delta/2 + (3/2)(pi^2/3 - 5/4)
                = 3 zeta_3 + pi^2/4 - 27/16 = 4.386072 = - int_0^1 G dz ,

    so ``int Delta K dz = 0`` analytically and the normalisation is untouched -
    no numerical rescale anywhere.  ``G = (1/2) P1T + A - (pi^2/3 - 5/4)(1+z)``
    with the two ingredients:

    * ``P1T`` is the **two-loop time-like non-singlet splitting function in its
      abelian part** (C_F^2 -> 1, C_A = 0, n_f = 0), in the normalisation
      ``P = a P0 + a^2 P1``:

          P1T(z) = 2 p ln z ln(1-z) + [ 3/(1-z) - 7 - 5z ] ln z
                 + [ (5/2)(1+z) - 4/(1-z) ] ln^2 z - 9(1-z) - 2 p(-z) S_2(z)
                 + (3/8 - pi^2/2 + 6 zeta_3) delta(1-z) ,                      (9)
          p(-z) = 2/(1+z) - 1 + z ,
          S_2(z) = ln^2 z/2 - pi^2/6 - 2 Li_2(-z) - 2 ln z ln(1+z) ,  S_2(1) = 0

      the **valence (C-odd) combination** ``P_NS,- = P_qq,V - P_qqbar,V``: the Z
      couples to the C-odd vector current, what is counted on the ``mu-`` leg is
      ``mu - mubar``, and only this combination has ``int P dz = 0``, i.e. muon
      number conservation, which the photonic sector must obey exactly.  The
      ``S_2(z)`` term is the crossed-photon interference in ``mu -> mu gamma
      gamma`` (``T^a T^b T^a T^b = C_F(C_F - C_A/2) -> 1`` in QED), not a pair
      effect.  The space-like kernel is Curci-Furmanski-Petronzio (Nucl. Phys.
      B175 (1980) 27) and Floratos-Kounnas-Lacaze (Nucl. Phys. B192 (1981) 417),

          P1S(z) = -2 p ln z ln(1-z) - [ 3/(1-z) + 2 + 4z ] ln z
                 - (1+z) ln^2 z / 2 - 9(1-z) - 2 p(-z) S_2(z)
                 + (3/8 - pi^2/2 + 6 zeta_3) delta(1-z) ;                     (10)

      its QED form is de Florian-Sborlini-Rodrigo, JHEP 10 (2016) 056, eqs.
      (57), (58), (63), (64).  The time-like difference is the Drell-Levy-Yan /
      Gribov-Lipatov-violating term

          P1T - P1S = 2 [ ln z P0 ] (x) P0                                    (11)

      (Curci-Furmanski-Petronzio; Mitov-Moch-Vogt, Phys. Lett. B638 (2006) 61
      and the accompanying ``tlike-ns.h``, where ``diffP1ns`` is 4x (11) in the
      ``alpha_s/4pi`` normalisation).  `nll` reproduces ``diffP1ns`` from (11)
      to 1.4e-13.

    * ``A(z) = (P0 (x) Chat_reg)(z)``, the convolution needed for ``P0 (x) C1``,
      is the *same* object: ``Chat_reg = p ln z``, so
      ``A = ([p]_+ (x) [p ln]) = (P1T - P1S)/2``, closed form

          A(z) = 2 p ln z ln(1-z) + (3/2) p ln z - p ln^2 z
               + (z-1) ln z + (1+z) ln^2 z / 2 .                              (12)

    **Scheme.**  The abelian kernel is the complete *photonic* two-loop
    splitting function of QED: C_F C_A has no QED analogue, and the ``n_f T_F``
    terms need a real fermion pair.  Those belong to the pair sector, which this
    kernel carries *exactly* at O(alpha^2) through `pair_radiator`; nothing is
    double counted.

    **Where the fixed order stops working.**  ``G`` carries ``-(7/4) ln^2 z`` at
    small ``z`` and ``(alpha/pi) L ln z = -0.36`` at ``z = 1e-5``: NLL is not
    enough once the effective collinear log ``ln(z m^2/m_mu^2)`` closes, and the
    ``alpha^2 L^2`` term of ``exp2`` is no better there.  ``exp2nll`` turns
    negative for ``u > 4.85`` at the Z (``z < 6.2e-5``) and `pdf_z` clips it;
    that region carries ``P = 5.5e-7`` of the kernel and 1.9e-7 of the norm.

    `nll` checks all of this: the closed forms against numerical convolutions,
    the sum rules, the Mellin moments against the two-loop anomalous dimension
    of Moch-Vermaseren-Vogt (Nucl. Phys. B688 (2004) 101) eq. (3.6), and - the
    decisive one - the ``L^2`` and ``L`` coefficients of the O(alpha^2) term of
    the **full kernel** against ``P0 (x) P0 / 2`` and ``P1T/2 + P0 (x) C1``.
    All residuals are below 1.4e-13.

``oalpha`` - fixed-order O(alpha) with a soft cutoff ``x_cut``: a delta at
    ``z = 1`` carrying ``1 - P(x > x_cut)`` plus (1) above the cutoff.  Not a
    model - it exists to show the size of the exponentiation.

``pair`` selects the species of **real lepton/hadron pair emission** off the
    muon line.  The spectrum is the exact O(alpha^2) `pair_radiator`
    ``R_pair(z)`` and the kernel is the convolution

        K = K_photonic (x) [ (1 - N_pair) delta(1-z) + R_pair(z) ] ,

    done on the atoms (`atoms`), so the two mean mass losses add exactly and
    the photon-pair cross term - 17 % of the pair ``<u>`` at the Z - is kept.
    The pair term is *not* exponentiated: a real pair of mass ``q`` cannot be
    softer than ``1 - z = 2q/sqrt(s)``, and the massive-photon phase space
    closes at ``q^2 = s (1 - sqrt z)^2``, so the pair spectrum has a threshold
    at ``1 - sqrt z = 2 m_l/m`` and no soft singularity to resum.

Discretisation
--------------
The provider's fold ``p_post(m) = sum_j w_j p_born(m/r_j)/r_j`` is a midpoint
quadrature of ``int dr k(r) p_born(m/r)/r``.  Atoms are groups of ``u`` placed
at their *exact* conditional mean, so the residual of a group is
``w_j Var(u|group) m^2 |p''| / 2``; groups are grown until ``w_j Var_j``
exceeds ``--var-budget`` (or the group sd exceeds ``--sigma-cap``).  Both the
weights and the conditional moments are computed by Gauss-Legendre quadrature
in ``t = (1-z)^beta``, the substitution that removes the ``(1-z)^{beta-1}``
endpoint singularity exactly (``dt = beta (1-z)^{beta-1} dz``).  The pair
convolution keeps the full photonic cell grid under the ``delta(1-z)`` branch
and multiplies a coarse photonic discretisation - budget relaxed by
``1/N_pair``, which is where its error is weighted - by the pair cells; the
atom count grows by ~1 %.

Banding in ``m_pre`` is free for an analytic kernel: one kernel per band with
its own ``beta(m)``, written as per-atom ``m_lo``/``m_hi`` for
`rabbit.lineshapes.zgamma`.
"""
import argparse
import json
import math
import os

import numpy as np

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------
ALPHA = 1.0 / 137.035999            # on-shell (Thomson) fine-structure constant
M_MU = 0.1056583745                 # PDG muon mass [GeV]
M_E = 0.51099895e-3                 # PDG electron mass [GeV]
M_TAU = 1.77686                     # PDG tau mass [GeV]
M_PI_CH = 0.13957039                # PDG charged pion mass [GeV]
M_C, M_B = 1.27, 4.18               # PDG MS-bar quark masses [GeV]
A_PI = ALPHA / math.pi

#: int_0^1 (1+z^2) ln z / (1-z) dz = 5/4 - pi^2/3
INT_F = 1.25 - math.pi**2 / 3.0     # = -2.0398684...
#: int_0^1 of the regular part of [P (x) P](z), eq. (3)
INT_Q2 = 4.3298684                  # = 2 pi^2/3 - 9/4, computed below


def _int_q2():
    """int_0^1 { (1+z)(3 ln z - 4 ln(1-z)) - 4 ln z/(1-z) - 5 - z } dz."""
    return 3.0 * (-1.25) - 4.0 * (-1.75) - 4.0 * (-math.pi**2 / 6.0) - 5.5


INT_Q2 = _int_q2()

#: Riemann zeta(3)
ZETA3 = 1.2020569031595942854
#: kappa = pi^2/3 - 5/4: the delta(1-z) part of the O(alpha) coefficient
#: function Chat, eq. (6).  Equal to -INT_F because int Chat dz = 0.
KAPPA = -INT_F
#: delta(1-z) coefficient of the two-loop non-singlet splitting function in its
#: abelian part, eqs. (9)-(10)
P1_DELTA = 3.0 / 8.0 - math.pi**2 / 2.0 + 6.0 * ZETA3
#: delta(1-z) coefficient of the O(alpha^2 L) remainder, eq. (8).  It is
#: (1/2) P1_DELTA + (3/2) KAPPA and equals -int_0^1 G dz, which is what makes
#: the NLL term leave the normalisation exactly alone.
G_DELTA = 0.5 * P1_DELTA + 1.5 * KAPPA


def _lnz(z, x):
    """ln z, accurate at both ends: ``log1p(-x)`` near z = 1, ``log z`` near 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x < 0.5, np.log1p(-x), np.log(z))


#: terms of the Li_2 power series; ``0.5**K/K^2`` is 2e-20 at K = 55
_DILOG_K = 55


def dilog_neg(z):
    """``Li_2(-z)`` for ``z >= 0``, to 1e-16, with numpy only.

    Landen: ``Li_2(-z) = -Li_2(z/(1+z)) - ln^2(1+z)/2``, which maps
    ``z in [0, inf)`` onto ``y in [0, 1)`` and, for the ``z <= 1`` this module
    needs, onto ``y in [0, 1/2]`` where ``sum_k y^k/k^2`` converges geometrically.
    """
    z = np.asarray(z, float)
    l1p = np.log1p(z)
    y = z / (1.0 + z)
    s = np.zeros_like(y)
    for k in range(_DILOG_K, 0, -1):
        s = s * y + 1.0 / (k * k)
    return -y * s - 0.5 * l1p * l1p


def s2_cfp(z, x=None):
    """``S_2(z) = int_{z/(1+z)}^{1/(1+z)} (dw/w) ln((1-w)/w)``, the crossed-photon
    function of Curci-Furmanski-Petronzio:

        S_2 = ln^2 z / 2 - pi^2/6 - 2 Li_2(-z) - 2 ln z ln(1+z) ,   S_2(1) = 0.
    """
    z = np.asarray(z, float)
    x = (1.0 - z) if x is None else np.asarray(x, float)
    lz = _lnz(z, x)
    return (0.5 * lz * lz - math.pi**2 / 6.0 - 2.0 * dilog_neg(z)
            - 2.0 * lz * np.log1p(z))


def p_minus(z):
    """``p(-z) = 2/(1+z) - 1 + z``, the crossed-photon splitting function."""
    z = np.asarray(z, float)
    return 2.0 / (1.0 + z) - 1.0 + z


def p1_timelike(z, x=None):
    """Regular part of eq. (9): abelian two-loop TIME-LIKE non-singlet kernel.

    Normalisation ``P = (alpha/2pi) P0 + (alpha/2pi)^2 P1``; the ``delta(1-z)``
    coefficient is `P1_DELTA`.  Every term is integrable at z = 1 - the abelian
    two-loop cusp vanishes, so there is no ``[1/(1-z)]_+``.
    """
    z = np.asarray(z, float)
    x = (1.0 - z) if x is None else np.asarray(x, float)
    lz = _lnz(z, x)
    lzx = lz / x                                  # -> -1 as x -> 0
    return (2.0 * (1.0 + z * z) * lzx * np.log(x)
            + 3.0 * lzx - (7.0 + 5.0 * z) * lz
            + 2.5 * (1.0 + z) * lz * lz - 4.0 * lzx * lz
            - 9.0 * x - 2.0 * p_minus(z) * s2_cfp(z, x))


def p1_spacelike(z, x=None):
    """Regular part of eq. (10): the same kernel, SPACE-LIKE (for `validate`)."""
    z = np.asarray(z, float)
    x = (1.0 - z) if x is None else np.asarray(x, float)
    lz = _lnz(z, x)
    lzx = lz / x
    return (-2.0 * (1.0 + z * z) * lzx * np.log(x)
            - 3.0 * lzx - (2.0 + 4.0 * z) * lz
            - 0.5 * (1.0 + z) * lz * lz
            - 9.0 * x - 2.0 * p_minus(z) * s2_cfp(z, x))


def conv_p0_pln(z, x=None):
    """``A(z) = ([p]_+ (x) [p ln])(z)``, eq. (12); also ``(P1T - P1S)/2``."""
    z = np.asarray(z, float)
    x = (1.0 - z) if x is None else np.asarray(x, float)
    lz = _lnz(z, x)
    lzx = lz / x
    return ((1.0 + z * z) * (2.0 * lzx * np.log(x) + 1.5 * lzx - lzx * lz)
            - x * lz + 0.5 * (1.0 + z) * lz * lz)


def nll_reg(z, x=None):
    """``G(z)``, eq. (8): the regular part of the O(alpha^2 L) remainder.

    ``G = (1/2) P1T + A - kappa (1+z)``, written out so that every 1/(1-z) is
    paired with the ``ln z`` that cancels it.
    """
    z = np.asarray(z, float)
    x = (1.0 - z) if x is None else np.asarray(x, float)
    lz = _lnz(z, x)
    lzx = lz / x
    return (3.0 * (1.0 + z * z) * lzx * np.log(x)
            + 4.5 * lzx - (6.0 + 3.0 * z) * lz
            + 2.75 * (1.0 + z) * lz * lz - 4.0 * lzx * lz
            - 4.5 * x - KAPPA * (1.0 + z) - p_minus(z) * s2_cfp(z, x))


def coll_log(m):
    """L(m) = ln(m^2 / m_mu^2)."""
    return 2.0 * np.log(np.asarray(m, float) / M_MU)


def coll_log_exact(m, mmu=M_MU):
    """The collinear log with the muon mass kept exactly.

    The angular integral of the soft eikonal factor of a back-to-back massive
    pair is ``(1+b^2)/(2b) ln((1+b)/(1-b))`` with ``b = sqrt(1 - 4 m_mu^2/s)``
    the muon velocity.  It reduces to `coll_log` as ``m_mu^2/s -> 0`` and is
    what the exact matrix element's soft limit contains, so it is the ``L``
    that makes ``R_exact - beta/x`` regular.  At the Z it differs from
    `coll_log` by 6e-8 of ``L-1``; at the J/psi by -4.3e-4.
    """
    s = float(m) * float(m)
    b = math.sqrt(max(1.0 - 4.0 * mmu * mmu / s, 0.0))
    if b <= 0.0:
        return 0.0
    return (1.0 + b * b) / (2.0 * b) * math.log((1.0 + b) / (1.0 - b))


def beta_fsr(m, minus_one=True):
    """beta(m) = (2 alpha/pi) (L - 1); ``minus_one=False`` drops the -1.

    The ``-1`` is the exact non-logarithmic part of the soft eikonal integral
    for a back-to-back pair; dropping it is the conventional NLL sensitivity
    bound on the radiator strength (it moves beta by 1/(L-1) = 8 %).
    """
    L = coll_log(m)
    return 2.0 * A_PI * (L - (1.0 if minus_one else 0.0))


# --------------------------------------------------------------------------
# real pair emission: the spectral densities
# --------------------------------------------------------------------------
# A virtual photon of mass^2 q^2 radiated off the muon line converts to a
# fermion pair.  The photon propagator with one self-energy insertion, cut, is
# exactly a dispersive integral over the massive-photon emission cross section:
#
#     dSigma_pair = int (dq^2/q^2) rho(q^2) dSigma_{gamma*}(q^2) ,           (P1)
#     rho(q^2) = Im Pi(q^2)/pi ,
#
# with ``dSigma_{gamma*}`` the emission of a VECTOR OF MASS^2 q^2 with the same
# coupling e and the polarisation sum ``-g + q q/q^2`` (equal to ``-g`` here,
# because the muon emission current is conserved).  (P1) is exact for the
# non-singlet channel - both attachments and their interference are inside
# ``dSigma_{gamma*}`` - so the only O(alpha^2) real-pair terms it misses are the
# singlet ones, in which the observed muon pair is not the one the current
# produced; those are a background to the dimuon spectrum, not FSR, and their
# rate inside a 60-120 GeV window is below 1e-7 (`pair` prints it).
#
#     rho_lepton = (alpha/3pi) (1 + 2 m_l^2/q^2) beta_l ,
#     rho_had    = (alpha/3pi) R(q^2) .
#
# Kniehl, Krawczyk, Kuhn, Stuart, Phys. Lett. B209 (1988) 337.


def rho_lepton(q2, ml):
    """``Im Pi(q^2)/pi`` for a lepton pair: ``(alpha/3pi)(1+2m^2/q^2) beta_l``."""
    q2 = np.asarray(q2, float)
    b = np.sqrt(np.maximum(1.0 - 4.0 * ml * ml / q2, 0.0))
    return (ALPHA / (3.0 * math.pi)) * (1.0 + 2.0 * ml * ml / q2) * b


#: ``1 + alpha_s/pi`` in each open-flavour region, ``alpha_s`` at a
#: representative scale of the region (2, 7 and 30 GeV).  The ``alpha_s^2`` and
#: ``alpha_s^3`` terms of the PDG QCD review eq. (9.7) add 0.2 % to
#: ``int R dln q^2`` and are dropped.
R_QCD = (1.093, 1.062, 1.045)
#: open-flavour thresholds: the physical ones, ``2 m_D0`` and ``2 m_B``, not
#: ``2 m_q``.  Using the quark masses instead overshoots ``int R dln q^2`` by
#: 2.4 %.
Q2_CHARM, Q2_BOTTOM = 3.7300**2, 10.5590**2
#: narrow vector resonances: ``(name, M [GeV], Gamma_ee [keV])`` (PDG).  A
#: resonance contributes ``int R dln q^2 = (9 pi/alpha^2) Gamma_ee/M`` at
#: ``q^2 = M^2`` - derived from ``int sigma_had ds = 12 pi^2 Gamma_ee/M`` and
#: ``sigma_pt = 4 pi alpha^2/3s`` - and is carried as a discrete node of the
#: ``q^2`` quadrature, which is exact for ``Gamma << M``.  ``Gamma_ee`` is the
#: PDG (vacuum-polarisation-dressed) value; undressing it, as the ``Delta
#: alpha_had`` compilations do, would lower each by 3-5 %, and the narrow
#: formula is itself 5 % high for the ``rho``.
R_RESONANCES = (("rho", 0.77526, 7.04), ("omega", 0.78266, 0.63),
                ("phi", 1.019461, 1.27), ("Jpsi", 3.096900, 5.53),
                ("psi2S", 3.68610, 2.33), ("Y1S", 9.46040, 1.340),
                ("Y2S", 10.02326, 0.612), ("Y3S", 10.3552, 0.443),
                ("Y4S", 10.5794, 0.272))


def r_had_cont(q2):
    """Continuum ``R(q^2) = sigma(hadrons)/sigma(mu mu)``, resonances excluded.

    Zero below 1 GeV, where the cross section *is* the ``rho``/``omega``/``phi``
    of `R_RESONANCES`; a linear ramp in ``sqrt(q^2)`` from 1 to 1.5 GeV onto the
    non-resonant plateau; then the parton model ``3 sum Q_q^2`` with the QCD
    factor and the physical open-flavour thresholds, and a flat 3.40 across the
    open-charm region 3.73-5 GeV where the data sit above the parton model.
    PDG "Quantum Chromodynamics" review eqs. (9.7)-(9.9); the sub-2 GeV
    normalisation follows KNT19 (arXiv:1911.00367) and DHMZ19 (arXiv:1908.00921).
    """
    q2 = np.asarray(q2, float)
    e = np.sqrt(np.maximum(q2, 0.0))
    R = np.clip((e - 1.0) / 0.5, 0.0, 1.0) * 2.0 * R_QCD[0]
    R = np.where(q2 > Q2_CHARM, 3.40, R)
    R = np.where(q2 > 25.0, (10.0 / 3.0) * R_QCD[1], R)
    R = np.where(q2 > Q2_BOTTOM, (11.0 / 3.0) * R_QCD[2], R)
    return R


def r_resonance_weights():
    """``(M^2, int R dln q^2)`` of the narrow resonances of `R_RESONANCES`."""
    q2 = np.array([m * m for _, m, _ in R_RESONANCES])
    a = np.array([(9.0 * math.pi / ALPHA**2) * (g * 1e-6) / m
                  for _, m, g in R_RESONANCES])
    return q2, a


#: lowest ``q^2`` at which each species can be produced
PAIR_Q2LO = {"e": 4 * M_E**2, "mu": 4 * M_MU**2, "tau": 4 * M_TAU**2,
             "had": 4 * M_PI_CH**2}
PAIR_SPECIES = tuple(PAIR_Q2LO)


def beta_pair_ll(m, species=("e",), ngauss=400):
    r"""Dispersive LEADING-LOG soft exponent, kept as the reference of `pair`.

        beta_pair = (2 alpha/pi) int_{q2_thr}^{s} (dq^2/q^2) rho(q^2)
                                 [ ln(s/q^2) - 1 ]

    i.e. the photon radiator with its collinear log moved from ``m_mu^2`` to
    ``q^2``.  It overestimates the mass loss by 46 %: for ``q^2 < m_mu^2`` -
    most of the ``dq^2/q^2`` range of an ``e+e-`` pair - the collinear log is
    cut off by the MUON mass and saturates at ``L``, it has no ``ln z``, and it
    ignores the massive-photon phase space ``q^2 < s (1-sqrt z)^2``.  The
    kernel uses the exact `pair_radiator` instead.
    """
    s = float(m) ** 2
    out = 0.0
    x, wq = np.polynomial.legendre.leggauss(ngauss)
    for sp in species:
        q2lo = PAIR_Q2LO[sp]
        if q2lo >= s:
            continue
        ylo, yhi = math.log(q2lo), math.log(s)
        y = 0.5 * (yhi - ylo) * (x + 1.0) + ylo
        w = 0.5 * (yhi - ylo) * wq
        q2 = np.exp(y)
        if sp == "had":
            rho = (ALPHA / (3.0 * math.pi)) * np.where(q2 >= 1.0, 2.0, 0.0)
        else:
            rho = rho_lepton(q2, {"e": M_E, "mu": M_MU, "tau": M_TAU}[sp])
        out += float(np.sum(w * rho * (np.log(s / q2) - 1.0)))
    return 2.0 * A_PI * out


# --------------------------------------------------------------------------
# the kernel
# --------------------------------------------------------------------------
def _merge_cells(c, var_budget=None, sigma_cap=None):
    """Merge ordered cells ``(w, int u, int u^2)`` into point masses ``(u, w)``.

    Groups grow while ``w Var(u|group)`` stays below ``var_budget`` (or the
    group sd below ``sigma_cap``); each atom sits at the exact conditional
    mean of ``u``, which makes the provider's midpoint fold first-order exact.
    """
    w, m1, m2 = c
    uj, wj = [], []
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
            uj.append(a1 / a0); wj.append(a0)
            a0, a1, a2 = w[i], m1[i], m2[i]
        else:
            a0, a1, a2 = b0, b1, b2
    if a0 > 0.0:
        uj.append(a1 / a0); wj.append(a0)
    return np.asarray(uj), np.asarray(wj)


VARIANTS = ("exp1", "exp2", "exp2nll", "oalpha", "born")
#: variants whose soft end is exponentiated as C beta (1-z)^{beta-1}
EXP_VARIANTS = ("exp1", "exp2", "exp2nll")


class FSRKernel:
    """K(z; m): the normalised probability density of z = (m_post/m_pre)^2.

    Parameters
    ----------
    m : float
        pre-FSR mass at which beta and the collinear log are evaluated.
    variant : {"exp1", "exp2", "exp2nll", "oalpha", "born"}
    pair : tuple of str
        pair-emission species to add, e.g. ``("e",)`` or ``("e","mu","had")``.
    minus_one : bool
        keep the ``-1`` in ``beta = (2 alpha/pi)(L-1)``.
    x_cut : float
        soft cutoff of the ``oalpha`` variant, ``x = 1-z``.
    zmin : float
        lower end of the support; the default is the exact 2-muon threshold
        ``4 m_mu^2/s``.
    mass_exact : bool
        keep the muon mass in the O(alpha) spectrum: ``L`` becomes the exact
        massive eikonal `coll_log_exact` and the hard remainder gets
        ``R_exact - r1`` from the exact matrix element (`r1_fast`).  It moves
        ``<u>`` by -2.8e-3 at the J/psi, -2.7e-4 at the Upsilon and below 1e-5
        at the Z, so it is the option a NARROW resonance needs and the Z does
        not.  ``mass_exact_xmin`` (derived from the cancellation floor when
        left at ``None``) and ``mass_exact_n`` control the ``dh`` tabulation.
    """

    def __init__(self, m, variant="exp1", pair=(), minus_one=True,
                 x_cut=1e-7, zmin=None, beta_at=None, pair_table=None,
                 mass_exact=False, mass_exact_xmin=None, mass_exact_n=20000):
        self.m = float(m)
        self.variant = variant
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        mb = self.m if beta_at is None else float(beta_at)
        self.beta_at = beta_at
        self.mass_exact = bool(mass_exact)
        self.L = float(coll_log_exact(mb) if self.mass_exact else coll_log(mb))
        self.beta = self.beta_gam = float(
            2.0 * A_PI * (self.L - (1.0 if minus_one else 0.0)))
        self.pair = tuple(pair)
        #: R_pair(z), the exact O(alpha^2) real-pair spectrum, added to the
        #: kernel with its integral taken out of the delta(1-z)
        self._pair = PairTerm(mb, pair, path=pair_table)
        self.x_cut = float(x_cut)
        self.zmin = float(zmin) if zmin is not None else 4.0 * M_MU**2 / self.m**2
        #: the exact-mass remainder of the O(alpha) spectrum, `_build_dh`
        self._dh_lx = self._dh_v = None
        self.dh_xmin = None
        self._int_dh = 0.0
        if self.mass_exact:
            self._build_dh(mass_exact_xmin, mass_exact_n)
        self.pair_rate = self.int_pair()
        #: (alpha/pi)^2 L, the prefactor of the O(alpha^2) NLL remainder (8).
        #: Photonic only - the n_f sector is the exact `pair_radiator`.
        self._c_nll = A_PI**2 * self.L if variant == "exp2nll" else 0.0
        self._C = (1.0 - self.int_h()
                   - (self.int_q2() if variant in ("exp2", "exp2nll") else 0.0)
                   - self.int_nll())

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

    # -- the exact-mass remainder -----------------------------------------
    # `r1` is the MASSLESS closed form.  With `L` set to `coll_log_exact` its
    # soft limit is already the exact massive eikonal `beta/x`, so
    #
    #     dh(z) = R_exact(z) - r1(z)
    #
    # is REGULAR at x -> 0, where it tends to a CONSTANT (-7.32e-5 at the
    # J/psi, 2.7e-3 of beta).  It is nevertheless computed there as the
    # difference of two numbers of size beta/x, so below `x_min` -- 1e-6, where
    # the plateau is still clean to four digits and the cancellation has not
    # yet eaten it -- the constant is continued rather than re-evaluated.
    # Everything above is tabulated in ln x and interpolated, and `int_h` gets
    # the same table's integral, so `C` and the normalisation stay exact.
    #: relative accuracy of `r1_fast`; the cancellation floor of `dh` is
    #: `DH_RTOL * r1(x)`, which is what sets `x_min`
    DH_RTOL = 3e-11

    def _dh_xmin(self):
        """Smallest ``x`` at which ``R_exact - r1`` still has two clean digits.

        The cancellation error is ``DH_RTOL * r1 ~ DH_RTOL * beta/x`` while the
        plateau is ``dh(x -> 0)``, so the usable range ends where the two are
        within a factor 50 of each other.  Derived rather than fixed because
        the plateau is 7.3e-5 at the J/psi and 1.7e-7 at the Z -- one constant
        cannot serve both, and a too-small ``x_min`` continues NOISE over the
        60 % of the kernel's weight that sits below it.
        """
        x0 = 1e-3                                  # safely on the plateau
        p = abs(float(r1_fast(self.m, np.array([1.0 - x0]))[0]
                      - self.r1(np.array([1.0 - x0]), np.array([x0]))[0]))
        if p <= 0.0:
            return 1e-4
        # DH_RTOL * beta / x = p / 50
        return min(max(50.0 * self.DH_RTOL * self.beta / p, 1e-8), 1e-2)

    def _build_dh(self, x_min, n):
        # the grid stops at the 2-muon threshold: above it `pdf_z` is zero
        # anyway, and `r1` itself diverges at x = 1.  `n` has to resolve the
        # bottom decades: `<u>` is converged to 2e-6 of the correction at
        # n = 20000 (the default) and to 3 % of it at n = 5000.
        x_min = self._dh_xmin() if x_min is None else float(x_min)
        self.dh_xmin = x_min
        lx = np.linspace(math.log(x_min), math.log1p(-self.zmin), n)
        x = np.exp(lx)
        z = 1.0 - x
        v = r1_fast(self.m, z) - self.r1(z, x)
        self._dh_lx, self._dh_v = lx, v
        # int_0^1 dh dz = int_0^1 dh dx = int dh x dlnx  (z = 1 - x)
        self._int_dh = float(np.trapezoid(v * x, lx)
                             if hasattr(np, "trapezoid")
                             else np.trapz(v * x, lx))

    def dh(self, z, x=None):
        """``R_exact(z) - r1(z)``, continued as a constant below ``x_min``."""
        if self._dh_v is None:
            return 0.0
        x = (1.0 - np.asarray(z, float)) if x is None else np.asarray(x, float)
        lx = np.log(np.maximum(x, 1e-300))
        return np.interp(lx, self._dh_lx, self._dh_v)

    def h(self, z, x=None):
        """Hard remainder of eq. (2): O(alpha) minus the exponentiated soft."""
        z = np.asarray(z, float)
        x = (1.0 - z) if x is None else np.asarray(x, float)
        out = (-0.5 * self.beta * (1.0 + z)
               + A_PI * (1.0 + z * z) * np.log1p(-x) / x)
        if self.mass_exact:
            out = out + self.dh(z, x)
        return out

    def int_h(self):
        return -0.75 * self.beta + A_PI * INT_F + self._int_dh

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

    def nll(self, z, x=None):
        """The O(alpha^2 L) remainder (8) without its ``delta(1-z)``."""
        if self._c_nll == 0.0:
            return np.zeros_like(np.asarray(z, float))
        return self._c_nll * nll_reg(z, x)

    def int_nll(self):
        """``int_0^1 nll dz = -(alpha/pi)^2 L g_delta``, exactly (no quadrature).

        Its negative goes into ``C``, i.e. into the ``delta(1-z)`` of (8), so
        the NLL term is normalisation-preserving by construction.
        """
        return -self._c_nll * G_DELTA

    def int_pair(self, n=4000):
        """``N_pair = int_0^1 R_pair dz``, the real-pair rate.

        Integrated as ``int 2 z R_pair(z) du`` on a log-spaced ``u`` ladder,
        the variable the table is smooth in.
        """
        if self._pair.B is None:
            return 0.0
        u = np.geomspace(self._pair._u[0], min(self._pair._u[-1],
                                               self._u_max()), n)
        z = np.exp(-2.0 * u)
        f = 2.0 * z * self._pair(z, -np.expm1(-2.0 * u)) * u   # d u -> d ln u
        lu = np.log(u)
        return float(np.trapezoid(f, lu) if hasattr(np, "trapezoid")
                     else np.trapz(f, lu))

    # -- the density -------------------------------------------------------
    def pdf_z(self, z, x=None):
        """K(z) for z < 1.  ``x = 1-z`` may be supplied at full precision.

        The **photonic** kernel: the pair term is a separate factor, carried by
        `pair_cells` and folded in by `atoms` and `moments`.  The delta at
        z = 1 of the ``oalpha`` variant is not included here either.
        """
        z = np.asarray(z, float)
        x = (1.0 - z) if x is None else np.asarray(x, float)
        if self.variant == "born":
            return np.zeros_like(z)
        if self.variant == "oalpha":
            out = np.where(x > self.x_cut, self.r1(z, x), 0.0)
            return np.where(z >= self.zmin, np.maximum(out, 0.0), 0.0)
        out = self.h(z, x) + self._C * self.beta * np.power(x, self.beta - 1.0)
        if self.variant in ("exp2", "exp2nll"):
            out = out + self.q2(z, x)
        if self.variant == "exp2nll":
            out = out + self.nll(z, x)
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

    def _panel(self, t_lo, t_hi, ng, clip=True):
        """GL nodes on [t_lo, t_hi]: returns (u, x, w) with sum(w) = int K dz.

        ``t_lo``/``t_hi`` may be arrays, one entry per panel; the returned
        arrays then carry a leading panel axis.  ``clip=False`` keeps the
        (negative) far tail of the fixed-order O(alpha^2) terms, which is what
        the normalisation check integrates.
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
        if self.variant in EXP_VARIANTS:
            f = self.h(z, x)
            if self.variant in ("exp2", "exp2nll"):
                f = f + self.q2(z, x)
            if self.variant == "exp2nll":
                f = f + self.nll(z, x)
            integ = self._C + f * jac
            integ = np.where(z >= self.zmin,
                             np.maximum(integ, 0.0) if clip else integ, 0.0)
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

    def _cells(self, tg, ng=16, clip=True):
        """Per-panel (w, int u K, int u^2 K, int (1-z) K)."""
        u, x, w = self._panel(tg[:-1], tg[1:], ng, clip)
        return np.stack([w.sum(1), (w * u).sum(1), (w * u * u).sum(1),
                         (w * x).sum(1)])

    def moments(self, u_max=None, n=4000, ng=16):
        """(norm, <u>, <u^2>, <1-z>) of the normalised kernel.

        Includes the pair term through the convolution: ``u`` is additive, so
        the means add, and ``1 - z`` combines as ``x_p + x_q - x_p x_q``.
        """
        c = self._cells(self._fine_grid(u_max, n), ng).sum(axis=1)
        n0 = max(c[0], 1.0) if self.variant == "oalpha" else c[0]
        u, u2, x = float(c[1] / n0), float(c[2] / n0), float(c[3] / n0)
        if self._pair.B is not None:
            p = self.pair_cells(u_max=u_max, with_x=True).sum(axis=1)
            u2 = u2 + 2.0 * u * p[1] + p[2]
            x = x + p[3] - x * p[3]
            u = u + p[1]
        return dict(norm=float(c[0]), u=u, u2=u2, x=x)

    def tail(self, u0, n=4000, ng=16):
        """P(u > u0) of the normalised PHOTONIC kernel (see `pdf_z`)."""
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

    # -- the pair kernel ---------------------------------------------------
    def pair_cells(self, n=400, ng=8, u_max=None, with_x=False):
        """``(w, int u R, int u^2 R)`` of ``R_pair`` on a log-``u`` ladder.

        Excludes the ``delta(1-z)``; ``sum w = N_pair``.  ``with_x`` appends
        ``int (1-z) R``.
        """
        if self._pair.B is None:
            return np.zeros((4 if with_x else 3, 0))
        u_max = self._u_max() if u_max is None else float(u_max)
        e = np.geomspace(self._pair._u[0], min(self._pair._u[-1], u_max), n + 1)
        g, wg = np.polynomial.legendre.leggauss(ng)
        a, b = np.log(e[:-1])[:, None], np.log(e[1:])[:, None]
        lu = 0.5 * (b - a) * (g[None, :] + 1.0) + a
        w = 0.5 * (b - a) * wg[None, :]
        u = np.exp(lu)
        z = np.exp(-2.0 * u)
        x = -np.expm1(-2.0 * u)
        f = w * u * 2.0 * z * self._pair(z, x)                    # K_u(u) u dlnu
        out = [f.sum(1), (f * u).sum(1), (f * u * u).sum(1)]
        if with_x:
            out.append((f * x).sum(1))
        return np.stack(out)

    def pair_atoms(self, var_budget=6e-10, **kw):
        """Merge `pair_cells` into point masses ``(u_k, w_k)``, no ``delta``."""
        c = self.pair_cells(**kw)
        if c.shape[1] == 0:
            return np.zeros(0), np.zeros(0)
        return _merge_cells(c, var_budget)

    # -- atoms -------------------------------------------------------------
    def cells(self, u_max=None, var_budget=6e-10, sigma_cap=None,
              n_fine=4000, ng=16):
        """Fine cells ``(w, int u K, int u^2 K)`` of the full kernel, u-ordered.

        The photonic ladder with the pair branch folded in; `atoms` is
        `_merge_cells` of this, and anything that has to reweight the kernel
        *before* the merge -- the selection-conditional construction of
        `fsr_perleg` -- needs the cells rather than the atoms.
        """
        w, m1, m2, _ = self._cells(self._fine_grid(u_max, n_fine), ng)
        if self._pair.B is not None:
            # K = K_phot (x) [ (1 - N) delta(1-z) + R_pair ]: the delta branch
            # keeps the full photonic cell grid, the pair branch multiplies a
            # COARSE photonic discretisation (its error is weighted by N, so
            # the budget there is relaxed by 1/N) with the pair cells.
            N = float(self.pair_rate)
            uk, wk = self.pair_atoms(
                var_budget=(var_budget / max(N, 1e-12) if var_budget else 1e-6),
                u_max=u_max)
            bud = (var_budget / max(N, 1e-12)) if var_budget else None
            uc, wc = _merge_cells(np.stack([w, m1, m2]), bud,
                                  None if bud else (sigma_cap or 1e-2) * 30.0)
            uu = (uc[:, None] + uk[None, :]).ravel()
            ww = (wc[:, None] * wk[None, :]).ravel()
            w = np.concatenate([(1.0 - N) * w, ww])
            m1 = np.concatenate([(1.0 - N) * m1, ww * uu])
            m2 = np.concatenate([(1.0 - N) * m2, ww * uu * uu])
            o = np.argsort(np.where(w > 0, m1 / np.maximum(w, 1e-300), 0.0))
            w, m1, m2 = w[o], m1[o], m2[o]
        return np.stack([w, m1, m2])

    def atoms(self, u_max=None, var_budget=6e-10, sigma_cap=None,
              n_fine=4000, ng=16):
        """Discretise into ``(r_j, w_j)`` with the first moment of u exact.

        Panels of the fine grid are merged while ``w Var(u|group)`` stays below
        ``var_budget`` (or the group sd below ``sigma_cap``); each atom sits at
        the exact conditional mean of ``u``, which makes the provider's
        midpoint fold first-order exact and leaves a residual
        ``w_j Var_j m^2 |p''| / 2`` per atom.
        """
        uj, wj = _merge_cells(self.cells(u_max, var_budget, sigma_cap,
                                         n_fine, ng), var_budget, sigma_cap)
        rj = np.exp(-uj)
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
                 beta_at=None, u_max=None, pair_table=None, **kw):
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
                      beta_at=beta_at, pair_table=pair_table)
        r, w, tot = k.atoms(u_max=u_max, **kw)
        blo = 0.0 if i == 0 else lo
        bhi = np.inf if i == n - 1 else hi
        R.append(r)
        W.append(w)
        LO.append(np.full(len(r), blo))
        HI.append(np.full(len(r), bhi))
        info.append(dict(m=mc, lo=blo, hi=float(bhi) if np.isfinite(bhi) else None,
                         beta=k.beta, pair_rate=k.pair_rate, natoms=len(r),
                         captured=tot, mean_u=-float(np.sum(w * np.log(r)))))
    return (dict(r=np.concatenate(R), w=np.concatenate(W),
                 m_lo=np.concatenate(LO), m_hi=np.concatenate(HI)), info)


# --------------------------------------------------------------------------
# scalar QED: spin-0 -> two charged SCALARS, exact inner bremsstrahlung
# --------------------------------------------------------------------------
# K_S -> pi+ pi- (gamma) is the case this was written for.  NOTHING about it is
# soft or collinear: ln(M^2/m_pi^2) = 2.54 is O(1) and the pion velocity in the
# pair frame is beta_0 = 0.828, so the quasi-collinear structure-function forms
# of `FSRKernel` do not apply and the spectrum has to be the EXACT scalar-QED
# matrix element integrated over the three-body phase space at fixed z.
#
# THE MATRIX ELEMENT.  With a CONSTANT (s-wave, non-derivative) weak vertex
# ``M_0 = G`` the two bremsstrahlung diagrams give, in a gauge with
# ``eps.k = 0``, the Low amplitude
#
#     M = e G [ p_+.eps/(p_+.k) - p_-.eps/(p_-.k) ] ,                      (S1)
#
# which is already gauge invariant on its own (``eps -> k`` gives 1 - 1 = 0),
# so a neutral spin-0 parent needs no seagull at O(e): the scalar-QED
# ``e^2 A^2 phi* phi`` vertex first contributes with TWO photons.  Summing the
# photon polarisations with ``-g^{mu nu}``,
#
#     sum_pol |M|^2 = e^2 |G|^2 [ 2 p_+.p_-/((p_+.k)(p_-.k))
#                                 - m^2/(p_+.k)^2 - m^2/(p_-.k)^2 ] ,      (S2)
#
# the classic ``-e^2 |G|^2 J^2`` eikonal current squared -- which here is not an
# approximation but the complete O(alpha) real-emission matrix element, because
# Low's theorem is saturated by a pointlike constant vertex.  (Structure
# dependence is DIRECT EMISSION, which for K_S -> pi+ pi- gamma is bounded by
# measurement below 2.3 % of IB -- see `ks_fsr_kernel`.)
#
# THE ANGULAR INTEGRAL IN CLOSED FORM.  Work in the pi-pi rest frame, where the
# pions are back to back with energy ``E = sqrt(s')/2`` and momentum
# ``q = E beta``, ``beta = sqrt(1 - 4 m^2/s')``, and the photon has energy
# ``w = (M^2 - s')/(2 sqrt(s'))``.  Then ``p_-.k = w(E - q c)``,
# ``p_+.k = w(E + q c)``, ``p_+.p_- = E^2 + q^2`` with ``c = cos theta*``, and
# (S2) collapses to the dipole pattern
#
#     sum_pol |M|^2 = e^2 |G|^2 . 4 beta^2 sin^2 theta*
#                                 / ( w^2 (1 - beta^2 cos^2 theta*)^2 ) .  (S3)
#
# Its angular average is elementary,
#
#     int_-1^1 dc (1-c^2)/(1-beta^2 c^2)^2 = [ (1+beta^2) Lam/(2 beta) - 1 ]
#                                    / beta^2 ,  Lam = ln((1+beta)/(1-beta))
#
# and folding in the two-body x two-body phase space of
# ``dPhi_3 = (ds'/2pi) dPhi_2(M; s', 0) dPhi_2(s'; m, m)`` against
# ``Gamma_0 = |G|^2 beta_0/(16 pi M)`` leaves, with ``z = s'/M^2``,
#
#     R1(z) = (1/Gamma_0) dGamma/dz
#           = (alpha/pi) (beta/beta_0) Bcal(beta) z/(1-z) ,                (S4)
#     Bcal(beta) = (1+beta^2)/beta ln((1+beta)/(1-beta)) - 2 ,
#
# i.e. the YFS soft function of the pair evaluated at the OUTGOING pair
# velocity, times the scalar-QED splitting weight ``z/(1-z)`` and the
# phase-space ratio ``beta/beta_0``.  (S4) is exact in the pion mass.
#
# Two rewritings make it numerically clean.  With ``r = 4 m^2/M^2`` (= z_min)
# and ``beta(z) = sqrt(1 - r/z)``,
#
#     R1(z) = (alpha/pi) g(z)/(beta_0 (1-z)) ,
#     g(z)  = (2z - r) Lam(z) - 2 z beta(z) ,   dg/dz = 2 Lam(z) ,         (S5)
#
# the second identity exact and elementary.  The soft limit of (S5) is
# ``R1 -> b/(1-z)`` with
#
#     b = (alpha/pi) g(1)/beta_0 = (alpha/pi) Bcal(beta_0) ,               (S6)
#
# the exact YFS exponent of the pi+ pi- pair -- so the hard remainder
# ``h = R1 - b/(1-z)`` is regular at z = 1 BY CONSTRUCTION, with no massless
# limit taken anywhere and no tabulated cancellation of the kind
# `FSRKernel.dh` needs.  This is the scalar analogue of the `mass_exact` path:
# the eikonal is the exact massive one, evaluated at the Born configuration.
# `ScalarIBKernel.h` evaluates it through
#
#     g(z) - g(1) = (2z-r) . 2 artanh(y)
#                   + 2 (1-z) [ r/(beta+beta_0) + beta_0 - Lam_0 ] ,
#     y = (beta - beta_0)/(1 - beta beta_0) ,
#     (beta - beta_0)/(1-z) = -r/(z (beta + beta_0)) ,                     (S7)
#
# where every difference is written in a cancellation-free form, so ``h`` is
# accurate to machine precision down to x = 1e-300.
#
# EXPONENTIATION.  ``K(z) = C b (1-z)^{b-1} + h(z)`` on the physical support
# ``[r, 1]``, with ``C = (1 - int h)/(1-r)^b`` fixing ``int_r^1 K dz = 1``.  The
# soft factor resums the eikonal to all orders (YFS); expanding it to O(alpha)
# returns (S4) plus the delta(1-z) that unitarity fixes.  ``b = 6.526e-3`` at
# the K_S, so the resummation itself is a 0.65 % effect and the uncontrolled
# O(alpha^2) hard remainder is a further factor b down.

#: PDG K^0_S mass [GeV], the parent `ScalarIBKernel` was written for
M_KS = 0.497611


def soft_B_scalar(beta):
    """``Bcal(beta) = (1+beta^2)/beta ln((1+beta)/(1-beta)) - 2``.

    The YFS soft-photon exponent of a pair of opposite unit charges with
    velocity ``beta`` in their own rest frame: the number of photons radiated
    between ``w`` and ``w + dw`` is ``(alpha/pi) Bcal(beta) dw/w``.  Goes to
    ``(8/3) beta^2`` at threshold (dipole radiation switches off when the pair
    is at rest) and to ``2[ln(s'/m^2) - 1]`` in the ultrarelativistic limit.
    """
    beta = np.asarray(beta, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(beta > 0.0,
                       (1.0 + beta * beta) * 2.0 * np.arctanh(np.minimum(
                           beta, 1.0 - 1e-16)) / np.maximum(beta, 1e-300) - 2.0,
                       0.0)
    # the beta -> 0 limit, (8/3) beta^2 + O(beta^4), where the form above
    # cancels 2 against 2 and loses every digit
    small = beta < 1e-4
    return np.where(small, (8.0 / 3.0) * beta * beta, out)


class ScalarIBKernel(FSRKernel):
    """K(z): inner bremsstrahlung of spin-0 -> two charged scalars, exact.

    ``z = (m'/M)^2`` with ``m'`` the post-radiation invariant mass of the two
    charged tracks, so ``u = -ln(m'/M)`` and ``dm = M(e^{-u} - 1)`` exactly as
    in `FSRKernel`; the whole quadrature, cell and atom machinery of the parent
    is reused unchanged -- only the O(alpha) spectrum, the soft exponent and
    the support differ.

    Parameters
    ----------
    m : float
        parent mass ``M``.
    mch : float
        charged-daughter mass ``m`` (default the charged pion).
    variant : {"exp1", "oalpha", "born"}
        ``exp1`` exponentiates the exact eikonal (S6); ``oalpha`` is the
        fixed-order spectrum (S4) with an explicit soft cutoff ``x_cut``.
        There is no ``exp2``/``exp2nll`` here: the O(alpha^2) structure
        functions of `FSRKernel` are collinear objects and ``ln(M^2/m^2)``
        is 2.54 at the K_S, so they have nothing to resum.
    coulomb : bool
        multiply by the ratio of Sommerfeld factors ``S(z)/S(1)`` for the
        pi+ pi- final state and renormalise.  OFF by default: it is a pure
        normalisation at O(alpha) and its z dependence is 2e-4 across the
        whole radiating range (`ks_fsr_kernel validate` measures it).
    """

    def __init__(self, m, mch=M_PI_CH, variant="exp1", x_cut=1e-7,
                 coulomb=False):
        if variant not in ("exp1", "oalpha", "born"):
            raise ValueError("ScalarIBKernel variant must be exp1/oalpha/born")
        self.m = float(m)
        self.mch = float(mch)
        self.variant = variant
        self.coulomb = bool(coulomb)
        #: r = 4 m^2/M^2 is both the threshold z and 1 - beta_0^2
        self.r = 4.0 * self.mch**2 / self.m**2
        if not 0.0 < self.r < 1.0:
            raise ValueError(f"2 m_ch = {2*self.mch} is not below M = {self.m}")
        self.zmin = self.r
        self.beta0 = math.sqrt(1.0 - self.r)
        self.lam0 = 2.0 * math.atanh(self.beta0)
        #: g(1) = (1 + beta_0^2) Lam_0 - 2 beta_0
        self.g1 = (2.0 - self.r) * self.lam0 - 2.0 * self.beta0
        #: the soft exponent, (S6).  `FSRKernel` calls it `beta`, and every
        #: inherited method means THIS by it (the quadrature variable is
        #: t = x^beta), so the name is kept.
        self.beta = self.beta_gam = A_PI * self.g1 / self.beta0
        #: ln(M^2/m^2), quoted because it is what is NOT large here
        self.L = 2.0 * math.log(self.m / self.mch)
        self.x_cut = float(x_cut)
        self.mass_exact = True          # there is no massless limit anywhere
        self._pair = PairTerm(self.m, (), path=None)
        self.pair_rate = 0.0
        self._c_nll = 0.0
        self._coul0 = float(self._coulomb(np.array([1.0]))[0]) if coulomb else 1.0
        self._int_h = self.int_h()
        #: int_r^1 C b x^{b-1} dz = C (1-r)^b, NOT C: the support stops at the
        #: two-pion threshold and (1-r)^b is 0.9975, a 2.5e-3 effect
        self._C = (1.0 - self._int_h) / (1.0 - self.r) ** self.beta

    # -- kinematics --------------------------------------------------------
    def beta_of_z(self, z):
        """Daughter velocity in the pair rest frame at pair mass^2 = z M^2."""
        z = np.asarray(z, float)
        return np.sqrt(np.maximum(1.0 - self.r / np.maximum(z, 1e-300), 0.0))

    def egamma(self, z):
        """Photon energy in the PARENT rest frame, ``E*_gamma = M(1-z)/2``."""
        return 0.5 * self.m * (1.0 - np.asarray(z, float))

    # -- the O(alpha) spectrum --------------------------------------------
    def r1(self, z, x=None):
        """Exact O(alpha) IB spectrum (S4)/(S5): ``(1/Gamma_0) dGamma/dz``."""
        z = np.asarray(z, float)
        x = (1.0 - z) if x is None else np.asarray(x, float)
        b = self.beta_of_z(z)
        g = (2.0 * z - self.r) * 2.0 * np.arctanh(b) - 2.0 * z * b
        with np.errstate(divide="ignore", invalid="ignore"):
            out = A_PI * g / (self.beta0 * x)
        return np.where(z >= self.r, out, 0.0)

    def r1_egamma(self, eg):
        """``(1/Gamma_0) dGamma/dE*_gamma`` -- the form the K -> pi pi gamma
        literature is written in: ``(alpha/pi)(beta/beta_0) Bcal(beta) z/E*``.
        """
        eg = np.asarray(eg, float)
        z = 1.0 - 2.0 * eg / self.m
        return self.r1(z) * 2.0 / self.m

    def h(self, z, x=None):
        """``R1(z) - b/(1-z)``: the hard remainder, regular at z = 1.

        Evaluated through (S7), where every difference is cancellation-free,
        so the result is accurate down to ``x = 1e-300`` without the tabulated
        continuation `FSRKernel.dh` needs.
        """
        z = np.asarray(z, float)
        x = (1.0 - z) if x is None else np.asarray(x, float)
        b = self.beta_of_z(z)
        sb = b + self.beta0
        d_over_x = -self.r / (np.maximum(z, 1e-300) * sb)      # (beta-beta0)/x
        y = d_over_x * x / (1.0 - b * self.beta0)
        # artanh(y)/y, exactly 1 in floating point for |y| below ~1e-8
        ay = np.where(np.abs(y) > 1e-8,
                      np.arctanh(np.clip(y, -1.0 + 1e-16, 1.0 - 1e-16))
                      / np.where(np.abs(y) > 1e-8, y, 1.0),
                      1.0 + y * y / 3.0)
        yx = d_over_x / (1.0 - b * self.beta0)                 # y/x
        out = (A_PI / self.beta0) * (
            (2.0 * z - self.r) * 2.0 * ay * yx
            + 2.0 * (self.r / sb + self.beta0 - self.lam0))
        return np.where(z >= self.r, out, 0.0)

    # -- normalisation -----------------------------------------------------
    #: Gauss-Legendre panels in beta for `int_h` (beta removes the sqrt branch
    #: point of the spectrum at the two-daughter threshold exactly)
    INT_H_PANELS, INT_H_NG = 400, 16

    def int_h(self, npan=None, ng=None):
        """``int_r^1 h dz``, by quadrature in ``beta``.

        ``z = r/(1-beta^2)``, ``dz = 2 r beta/(1-beta^2)^2 dbeta``: ``h`` is a
        smooth function of ``beta`` on ``[0, beta_0]`` with no endpoint
        structure at all, so plain Gauss-Legendre converges geometrically.
        `int_h_parts` is the independent cross-check.
        """
        npan = self.INT_H_PANELS if npan is None else npan
        ng = self.INT_H_NG if ng is None else ng
        e = np.linspace(0.0, self.beta0, npan + 1)
        g, wg = np.polynomial.legendre.leggauss(ng)
        b = 0.5 * (e[1:] - e[:-1])[:, None] * (g[None, :] + 1.0) + e[:-1][:, None]
        w = 0.5 * (e[1:] - e[:-1])[:, None] * wg[None, :]
        z = self.r / (1.0 - b * b)
        jac = 2.0 * self.r * b / (1.0 - b * b) ** 2
        return float(np.sum(w * self.h(z, 1.0 - z) * jac))

    def int_h_parts(self, npan=None, ng=None, zsplit=0.5):
        """``int_r^1 h dz`` again, from ``dg/dz = 2 Lam`` by parts:

            int h dz = (alpha/pi beta_0) [ -g(1) ln(1-r)
                                           + 2 int_r^1 Lam(z) ln(1-z) dz ] .

        A different integrand, a different singularity structure and a
        different quadrature, so agreement with `int_h` tests the identity
        ``dg/dz = 2 Lam`` -- the one piece of algebra behind (S5) and (S7) --
        as well as both quadratures.  ``Lam ln(1-z)`` has a sqrt branch point
        at ``z = r`` and a log one at ``z = 1``, so the range is split: below
        ``zsplit`` in ``beta`` (which removes the first exactly), above it in
        ``v = -ln(1-z)`` (which removes the second exactly).
        """
        npan = self.INT_H_PANELS if npan is None else npan
        ng = self.INT_H_NG if ng is None else ng
        g, wg = np.polynomial.legendre.leggauss(ng)

        def _gl(lo, hi):
            e = np.linspace(lo, hi, npan + 1)
            n = 0.5 * (e[1:] - e[:-1])[:, None]
            return n * (g[None, :] + 1.0) + e[:-1][:, None], n * wg[None, :]

        b, w = _gl(0.0, math.sqrt(1.0 - self.r / zsplit))
        z = self.r / (1.0 - b * b)
        lo = float(np.sum(w * 2.0 * (2.0 * np.arctanh(b)) * np.log1p(-z)
                          * (2.0 * self.r * b / (1.0 - b * b) ** 2)))
        x_lo = 1e-14
        lx, w = _gl(math.log(x_lo), math.log1p(-zsplit))
        x = np.exp(lx)
        hi = float(np.sum(w * 2.0 * (2.0 * np.arctanh(self.beta_of_z(1.0 - x)))
                          * lx * x))                  # dx = x dlnx
        # the sliver below x_lo, where Lam is Lam_0 to 13 digits and
        # int_0^X ln x dx = X(ln X - 1) exactly
        hi += 2.0 * self.lam0 * x_lo * (math.log(x_lo) - 1.0)
        return (A_PI / self.beta0) * (-self.g1 * math.log1p(-self.r) + lo + hi)

    # -- the pieces the parent's quadrature expects to exist ---------------
    def q2(self, z, x=None):
        return np.zeros_like(np.asarray(z, float))

    def int_q2(self):
        return 0.0

    def nll(self, z, x=None):
        return np.zeros_like(np.asarray(z, float))

    def int_nll(self):
        return 0.0

    def dh(self, z, x=None):
        return 0.0

    def int_pair(self, n=4000):
        return 0.0

    # -- the final-state Coulomb (Sommerfeld) factor -----------------------
    def _coulomb(self, z):
        """``S(eta) = 2 pi eta/(e^{2 pi eta} - 1)``, ``eta = -alpha/v_rel``.

        The attractive pi+ pi- Coulomb enhancement at pair mass^2 = z M^2, with
        ``v_rel = 2 beta/(1+beta^2)`` the relative velocity.  Returned RAW;
        `pdf_z` divides by its value at z = 1, so only the z DEPENDENCE ever
        enters the normalised kernel.
        """
        b = self.beta_of_z(z)
        v = np.where(b > 0.0, 2.0 * b / (1.0 + b * b), 1e-12)
        a = -2.0 * math.pi * ALPHA / v
        return np.where(np.abs(a) > 1e-12, a / np.expm1(a), 1.0)

    def pdf_z(self, z, x=None):
        out = FSRKernel.pdf_z(self, z, x)
        if self.coulomb:
            out = out * self._coulomb(z) / self._coul0
        return out

    def _panel(self, t_lo, t_hi, ng, clip=True):
        # for `oalpha` the parent already went through `pdf_z`, which carries
        # the factor; only the exponentiated branch is built from `h` and `_C`
        # directly and needs it applied here
        u, x, w = FSRKernel._panel(self, t_lo, t_hi, ng, clip)
        if self.coulomb and self.variant in EXP_VARIANTS:
            w = w * self._coulomb(1.0 - x) / self._coul0
        return u, x, w

    # -- rates, for the PDG normalisation gate -----------------------------
    def rate_above(self, egamma_cut, npan=4000, ng=16):
        """``Gamma(parent -> 2 charged + gamma, E*_gamma > cut)/Gamma_0``.

        The fixed-order O(alpha) integral of (S4).  This is the number the PDG
        ratio ``Gamma(K_S -> pi+ pi- gamma)/Gamma(K_S -> pi+ pi-)`` measures.
        Quadrature in ``beta``, as `int_h`.
        """
        zc = 1.0 - 2.0 * float(egamma_cut) / self.m
        if zc <= self.r:
            return 0.0
        bmax = math.sqrt(1.0 - self.r / zc)
        e = np.linspace(0.0, bmax, npan + 1)
        g, wg = np.polynomial.legendre.leggauss(ng)
        b = 0.5 * (e[1:] - e[:-1])[:, None] * (g[None, :] + 1.0) + e[:-1][:, None]
        w = 0.5 * (e[1:] - e[:-1])[:, None] * wg[None, :]
        z = self.r / (1.0 - b * b)
        jac = 2.0 * self.r * b / (1.0 - b * b) ** 2
        return float(np.sum(w * self.r1(z, 1.0 - z) * jac))


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
# the exact O(alpha) energy sharing between the two legs
# --------------------------------------------------------------------------
# At O(alpha) with ONE photon, in the V*(m) rest frame,
#
#     m'^2 = (Q - k)^2 = m^2 (1 - 2 E_gamma/m)  =  z m^2            (exact)
#     E_+ + E_- + E_gamma = m   ->   x_+ + x_- = 1 + z              (exact)
#
# with x_q = 2 E'_q/m the muon energy fractions.  The two are therefore on a
# LINE, not on the collinear hyperbola x_+ x_- = z, and one number fixes where:
#
#     x_+ = 1 - (1 - z) f ,   x_- = 1 - (1 - z)(1 - f) ,   f = (1 - beta_mu c)/2
#
# with ``c = cos theta*`` the angle of the ``mu-`` to the photon in the mu-mu
# rest frame -- the variable `_T_rad`/`_T_pair` use.  ``f = 0`` is the photon
# collinear with ``mu-`` (that leg takes the whole loss), ``f = 1`` collinear
# with ``mu+``, and the ~1/L of the rate in between is the recoil sharing the
# collinear factorisation has no room for.
#
# The sharing density at fixed z is the exact spin-summed matrix element,
# normalised:  p(f|z) df = T(s, z, c) dc / int T dc, whose ``c`` integral is
# `r1_exact` by construction.  It is EXACTLY symmetric under f -> 1 - f (C
# invariance of a neutral current; checked at 1e-16), so only the half branch
# is tabulated.
#
# ``T (1 - beta_mu^2 c^2)^2`` is a quadratic in ``c^2`` -- the amplitude has the
# two propagators ``2 p_-.k = s(1-x_+)`` and ``2 p_+.k = s(1-x_-)``, both linear
# in ``c``, and the numerators are quadratic -- so three evaluations of
# `_T_pair` per ``z`` give the closed form exactly (verified against direct
# evaluation at the 1e-14 level from ``z`` = 1 - 1e-6 down to the ``2 m_mu``
# threshold).  In the massless limit it collapses to the textbook
# ``(x_+^2 + x_-^2)/((1-x_+)(1-x_-)) = 2 (1 + zeta c^2)/((1-c^2) (1-z)^2/(1+z)^2)``
# with ``zeta = ((1-z)/(1+z))^2``.
#: the three ``cos theta*`` at which ``_T_pair`` is sampled for `share_coeffs`
SHARE_C = (0.0, 0.6, 0.9)


def share_coeffs(m, z, mmu=M_MU, va=(1.0, 0.0)):
    """``(A, beta_mu)`` with ``T (1 - beta_mu^2 c^2)^2 = A0 + A1 c^2 + A2 c^4``."""
    s = m * m
    z = np.atleast_1d(np.asarray(z, float))
    bm = np.sqrt(np.maximum(1.0 - 4.0 * mmu * mmu / (z * s), 0.0))
    cc = np.asarray(SHARE_C, float)
    zf = np.repeat(z, len(cc))
    cf = np.tile(cc, len(z))
    T = _T_pair(s, zf, np.zeros_like(zf), cf, mmu, va).reshape(len(z), len(cc))
    F = T * (1.0 - np.outer(bm * bm, cc * cc)) ** 2
    return np.linalg.solve(np.vander(cc * cc, 3, increasing=True), F.T).T, bm


def r1_fast(m, z, npanel=64, ng=8, mmu=M_MU, va=(1.0, 0.0), chunk=4000):
    """`r1_exact` for an ARRAY of ``z``, through the closed form of `share_coeffs`.

    Three `_T_pair` evaluations per ``z`` instead of ``2 ng`` of them; the
    ``cos theta*`` integral is then arithmetic.  Agrees with `r1_exact` to
    1e-11.
    """
    z = np.atleast_1d(np.asarray(z, float))
    s = m * m
    out = np.zeros(len(z))
    ok = z * s > 4.0 * mmu * mmu
    if not ok.any():
        return out
    _, w = share_nodes(m, z[ok], npanel, ng, mmu, va, chunk, raw=True)
    bm0 = math.sqrt(1.0 - 4.0 * mmu * mmu / s)
    bm = np.sqrt(1.0 - 4.0 * mmu * mmu / (z[ok] * s))
    out[ok] = (ALPHA * s * (1.0 - z[ok]) / (4.0 * math.pi) * (bm / bm0)
               * 0.5 * 2.0 * w.sum(1) / _T_born(s, mmu, va))
    return out


def share_nodes(m, z, npanel=64, ng=8, mmu=M_MU, va=(1.0, 0.0), chunk=4000,
                raw=False):
    """``(f, w)`` of the exact O(alpha) sharing density, half branch ``f <= 1/2``.

    ``f`` has shape ``(len(z), npanel*ng)`` and ``w`` sums to **1/2** per row:
    the density is exactly symmetric under ``f -> 1 - f``, so the other half is
    the mirror image with the same weights.

    The quasi-collinear pole ``p ~ 1/f`` is resolved by the substitution
    ``1 - beta_mu c = (1 - beta_mu) e^t``, i.e. ``f = (1 - beta_mu) e^t/2``:
    the nodes are uniform in ``ln f`` over the ``L = ln(z m^2/m_mu^2)`` e-folds
    between the mass regulator ``f_min = (1 - beta_mu)/2`` and ``1/2``, which is
    where the rate is.  ``t`` is covered by ``npanel`` Gauss-Legendre panels of
    ``ng`` nodes.  ``raw`` returns the unnormalised ``T |dc/dt| dt`` instead,
    whose row sum doubled is the ``cos theta*`` integral `r1_fast` needs.
    """
    z = np.atleast_1d(np.asarray(z, float))
    g, wg = np.polynomial.legendre.leggauss(ng)
    e = np.linspace(0.0, 1.0, npanel + 1)
    tt = (0.5 * (e[1:] - e[:-1])[:, None] * (g[None, :] + 1.0)
          + e[:-1][:, None]).ravel()
    ww = (0.5 * (e[1:] - e[:-1])[:, None] * wg[None, :]).ravel()
    F = np.empty((len(z), len(tt)))
    W = np.empty_like(F)
    for i0 in range(0, len(z), chunk):
        sl = slice(i0, min(i0 + chunk, len(z)))
        A, bm = share_coeffs(m, z[sl], mmu, va)
        # at the 2 m_mu threshold the muons are at rest in the mu-mu frame and
        # the sharing collapses to f = 1/2; there is no t range to integrate
        dead = bm <= 0.0
        bs = np.where(dead, 1.0, bm)
        tA = np.where(dead, 0.0, -np.log1p(-bs))
        om = (1.0 - bs)[:, None] * np.exp(tA[:, None] * tt[None, :])
        y = ((1.0 - om) / bs[:, None]) ** 2
        T = ((A[:, 0:1] + A[:, 1:2] * y + A[:, 2:3] * y * y)
             / (1.0 - (bs * bs)[:, None] * y) ** 2)
        w = T * (tA[:, None] * ww[None, :]) * om / bs[:, None]
        f = 0.5 * om
        f[dead] = 0.5
        w[dead] = 1.0
        F[sl] = f
        W[sl] = w if raw else 0.5 * w / w.sum(1, keepdims=True)
    return F, W


# --------------------------------------------------------------------------
# exact O(alpha^2) real pair emission
# --------------------------------------------------------------------------
# The same Dirac-trace machinery with a MASSIVE emitted vector: ``k`` is given
# mass^2 = q2, the propagator denominators become ``2 p.k + q2``, and the
# photon index is still contracted with ``-g`` (``k_nu M^{mu nu} = 0`` makes
# the ``k k/q^2`` piece vanish - checked below at the 1e-15 level).  The
# massless limit reproduces `r1_exact` to 6e-12.
#
# ``eikonal=True`` replaces the matrix element by its soft limit
# ``T_born x (-J^2)`` with ``J = p_-/(p_-.k) - p_+/(p_+.k)``.  That is exactly
# what Photos++ 3.61 generates (`pairs.cxx`, ``YOT1``), and it reproduces the
# standalone Photos pair rate and mass loss to 0.2 %.


def _T_pair(s, z, q2, c, mmu=M_MU, va=(1.0, 0.0), eikonal=False):
    """Spin-summed ``|M|^2`` (without ``e^2``) for ``V* -> mu+ mu- gamma*(q2)``.

    ``z``, ``q2``, ``c`` are equal-length arrays; ``c = cos theta*`` is the
    muon direction in the ``mu mu`` rest frame, with the ``gamma*`` along +z.
    """
    v, a = va
    z = np.asarray(z, float); q2 = np.asarray(q2, float); c = np.asarray(c, float)
    sp = z * s
    E = 0.5 * np.sqrt(sp)
    bm = np.sqrt(np.maximum(1.0 - 4.0 * mmu * mmu / sp, 0.0))
    pq = E * bm
    w = (s - sp - q2) / (2.0 * np.sqrt(sp))
    kk = np.sqrt(np.maximum(w * w - q2, 0.0))
    if eikonal:
        pmk = E * w - pq * kk * c            # p_- . k
        ppk = E * w + pq * kk * c            # p_+ . k
        pmpp = E * E + pq * pq               # p_- . p_+
        return _T_born(s, mmu, va) * (2.0 * pmpp / (pmk * ppk)
                                      - mmu**2 / pmk**2 - mmu**2 / ppk**2)
    st = np.sqrt(np.maximum(1.0 - c * c, 0.0))
    zer = np.zeros_like(c)
    pm = np.stack([E, pq * st, zer, pq * c], -1)
    pp = np.stack([E, -pq * st, zer, -pq * c], -1)
    k = np.stack([w, zer, zer, kk], -1)
    Q = pm + pp + k
    I4 = np.eye(4)
    Gam = v * _GAMMA + a * np.einsum('mij,jk->mik', _GAMMA, _G5)
    num1 = _slash(pm) + _slash(k) + mmu * I4
    num2 = mmu * I4 - _slash(pp) - _slash(k)
    dot = lambda x, y: (x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
                        - x[..., 2] * y[..., 2] - x[..., 3] * y[..., 3])
    d1 = (2.0 * dot(pm, k) + q2)[:, None, None, None, None]
    d2 = (2.0 * dot(pp, k) + q2)[:, None, None, None, None]
    A = np.einsum('nij,pjk->pnik', _GAMMA, num1)
    X = np.einsum('pnij,mjk->pnmik', A, Gam) / d1
    B = np.einsum('mij,pjk->pmik', Gam, num2)
    X += np.einsum('pmij,njk->pnmik', B, _GAMMA) / d2
    L = _slash(pm) + mmu * I4
    R = _slash(pp) - mmu * I4
    # P_{m M} = -g + Qt Qt/s.  In this frame Q = (Q0, 0, 0, Q3) with Q^2 = s,
    # so P is the two transverse unit vectors (eigenvalue 1) plus ONE direction
    # in the (0, 3) plane with eigenvalue 1 + 2 Q3^2/s: rank 3, no eigenvalue
    # decomposition needed at run time.
    Q3, Q0 = Q[:, 3], Q[:, 0]
    aa = Q3 * Q3 / s
    bb = -Q0 * Q3 / s
    nrm = np.sqrt(aa * aa + bb * bb)
    small = nrm < 1e-300
    nrm = np.where(small, 1.0, nrm)
    vv = np.zeros_like(Q)
    vv[:, 0] = np.where(small, 1.0, aa / nrm)
    vv[:, 3] = np.where(small, 0.0, bb / nrm)
    lam3 = 1.0 + 2.0 * Q3 * Q3 / s
    Y = np.stack([X[:, :, 1], X[:, :, 2],
                  np.einsum('pm,pnmij->pnij', vv, X)], axis=2)
    wa = np.stack([np.ones_like(lam3), np.ones_like(lam3), lam3], axis=-1)
    LY = np.einsum('pij,pnajk->pnaik', L, Y)
    RY = np.einsum('pij,pnajk->pnaik', R, _bar(Y))
    tr = np.einsum('pnaij,pnaji->pna', LY, RY).real
    g = np.array([-1.0, 1.0, 1.0, 1.0])
    return np.einsum('n,pa,pna->p', g, wa, tr) / 3.0


def spec_gammastar(m, z, q2, mmu=M_MU, va=(1.0, 0.0), ng=24, chunk=20000,
                   eikonal=False):
    """``(1/Gamma_0) dGamma/dx`` for ``V*(m) -> mu+ mu- gamma*(q2)``.

    ``x = 2 E_gamma*/sqrt(s) = 1 - z + q2/s`` is the emitted energy fraction;
    the mass loss is ``1 - z = x - q2/s``.  The muon angular integral uses
    ``1 -+ beta_mu beta_k cos = (1 - beta_mu beta_k) e^t``, which resolves both
    quasi-collinear poles exactly.  ``ng = 24`` converges to 1e-13.
    """
    s = m * m
    z = np.asarray(z, float); q2 = np.asarray(q2, float)
    out = np.zeros(np.broadcast(z, q2).shape)
    z, q2 = np.broadcast_arrays(z, q2)
    z = np.ascontiguousarray(z); q2 = np.ascontiguousarray(q2)
    sp = z * s
    x = 1.0 - z + q2 / s
    lam2 = x * x - 4.0 * q2 / s
    ok = (sp > 4.0 * mmu * mmu) & (lam2 > 0.0)
    if not ok.any():
        return out
    idx = np.nonzero(ok.ravel())[0]
    zo = z.ravel()[idx]; qo = q2.ravel()[idx]
    spo = zo * s
    bm = np.sqrt(1.0 - 4.0 * mmu * mmu / spo)
    bm0 = math.sqrt(1.0 - 4.0 * mmu * mmu / s)
    w = 0.5 * (s - spo - qo) / np.sqrt(spo)
    bk = np.sqrt(np.maximum(w * w - qo, 0.0)) / w
    aa = np.clip(bm * bk, 1e-300, 1.0 - 1e-16)
    tA = -np.log1p(-aa)
    gq, wq = np.polynomial.legendre.leggauss(ng)
    t = 0.5 * tA[:, None] * (gq[None, :] + 1.0)
    wt = 0.5 * tA[:, None] * wq[None, :] * (1.0 - aa)[:, None] / aa[:, None] * np.exp(t)
    cA = (1.0 - (1.0 - aa)[:, None] * np.exp(t)) / aa[:, None]
    C = np.concatenate([cA, -cA], axis=1)
    W = np.concatenate([wt, wt], axis=1)
    n = C.shape[1]
    zf = np.repeat(zo, n); qf = np.repeat(qo, n); cf = C.ravel()
    T = np.empty(zf.size)
    for i0 in range(0, zf.size, chunk):
        sl = slice(i0, min(i0 + chunk, zf.size))
        T[sl] = _T_pair(s, zf[sl], qf[sl], cf[sl], mmu, va, eikonal)
    T = T.reshape(-1, n)
    o = out.ravel()
    o[idx] = (ALPHA * s * np.sqrt(lam2.ravel()[idx]) / (4.0 * math.pi)
              * (bm / bm0) * 0.5 * np.sum(W * T, axis=1) / _T_born(s, mmu, va))
    return out


def _pair_q2_nodes(m, z, species, nq=16):
    """``(q2, w)`` of the ``q^2`` quadrature at fixed ``z``, ``w`` = rho dln q^2.

    The range is ``4 m_thr^2 < q^2 < s (1 - sqrt z)^2`` - the massive-photon
    phase-space limit.  Log-spaced Gauss panels split at every species
    threshold, plus a top panel in ``v = sqrt(q2max - q2)`` that resolves the
    square-root edge, plus one discrete node per narrow vector resonance.
    """
    s = m * m
    q2hi = s * (1.0 - math.sqrt(z)) ** 2
    q2lo = min(PAIR_Q2LO[sp] for sp in species)
    if q2hi <= q2lo:
        return np.zeros(0), np.zeros(0)
    g, wg = np.polynomial.legendre.leggauss(nq)
    edges = [q2lo, 4 * M_MU**2, 4 * M_PI_CH**2, 1.0, 2.25, 3.24, Q2_CHARM,
             25.0, 4 * M_TAU**2, Q2_BOTTOM]
    edges = sorted(set([e for e in edges if q2lo <= e < 0.5 * q2hi] + [q2lo, 0.5 * q2hi]))
    Q, W = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        a, b = math.log(lo), math.log(hi)
        y = 0.5 * (b - a) * (g + 1.0) + a
        Q.append(np.exp(y)); W.append(0.5 * (b - a) * wg)
    vmax = math.sqrt(0.5 * q2hi)
    v = 0.5 * vmax * (g + 1.0)
    q2t = q2hi - v * v
    Q.append(q2t); W.append(0.5 * vmax * wg * 2.0 * v / q2t)
    q2 = np.concatenate(Q); dl = np.concatenate(W)
    rho = np.zeros_like(q2)
    for sp in species:
        if sp == "had":
            rho += (ALPHA / (3.0 * math.pi)) * r_had_cont(q2)
        else:
            rho += rho_lepton(q2, {"e": M_E, "mu": M_MU, "tau": M_TAU}[sp]) \
                * (q2 >= PAIR_Q2LO[sp])
    w = rho * dl
    if "had" in species:
        rq2, ra = r_resonance_weights()
        keep = rq2 < q2hi
        q2 = np.concatenate([q2, rq2[keep]])
        w = np.concatenate([w, (ALPHA / (3.0 * math.pi)) * ra[keep]])
    return q2, w


def pair_radiator(m, z, species=PAIR_SPECIES, nq=16, ng=24, eikonal=False):
    """``R_pair(z)``, the exact O(alpha^2) real-pair spectrum, eq. (P1).

    A density in ``z = (m'/m)^2`` per unit ``z``, to be ADDED to the photonic
    kernel with ``int R_pair dz`` removed from the ``delta(1-z)``: to O(alpha^2)
    that is the product of the photonic kernel with the pair kernel
    ``(1 - N_pair) delta(1-z) + R_pair(z)``.
    """
    z = np.atleast_1d(np.asarray(z, float))
    out = np.zeros_like(z)
    for i, zz in enumerate(z):
        if not (0.0 < zz < 1.0):
            continue
        q2, w = _pair_q2_nodes(m, float(zz), species, nq)
        if q2.size == 0:
            continue
        f = spec_gammastar(m, np.full_like(q2, zz), q2, ng=ng, eikonal=eikonal)
        out[i] = float(np.sum(w * f))
    return out


def pair_moments(m, species=PAIR_SPECIES, nq=12, nth=48, npan=3, ng=24,
                 eikonal=False, q2lo=None):
    """``(rate, <u> rate, <u^2> rate)`` of real pair emission.

    Integrated the other way round - over the emitted energy at fixed ``q^2``,
    with ``x = 2 sqrt(q^2/s) cosh(theta)`` removing the square-root edge - so
    it is an independent check of `pair_radiator`.  ``qcut`` restricts to pair
    masses ``q > qcut``.
    """
    s = m * m
    per = {}
    q2, wq, dl = _moment_q2_grid(m, species, nq, q2lo)
    N = np.zeros(len(q2)); U = np.zeros(len(q2)); U2 = np.zeros(len(q2))
    g, wg = np.polynomial.legendre.leggauss(nth)
    for i, qq in enumerate(q2):
        r = qq / s
        xmax = 1.0 + r - 4.0 * M_MU**2 / s
        x0 = 2.0 * math.sqrt(r)
        if x0 >= xmax:
            continue
        thmax = math.acosh(xmax / x0)
        e = np.linspace(0.0, thmax, npan + 1)
        th = np.concatenate([0.5 * (b - a) * (g + 1.0) + a
                             for a, b in zip(e[:-1], e[1:])])
        wth = np.concatenate([0.5 * (b - a) * wg for a, b in zip(e[:-1], e[1:])])
        x = x0 * np.cosh(th)
        wx = wth * x0 * np.sinh(th)
        z = 1.0 - x + r
        f = spec_gammastar(m, z, np.full_like(z, qq), ng=ng, eikonal=eikonal)
        u = -0.5 * np.log(np.maximum(z, 1e-300))
        N[i] = np.sum(wx * f); U[i] = np.sum(wx * f * u)
        U2[i] = np.sum(wx * f * u * u)
    for sp in species:
        w = wq[sp]
        per[sp] = (float(np.sum(w * N)), float(np.sum(w * U)),
                   float(np.sum(w * U2)))
    return per


def _moment_q2_grid(m, species, nq, q2lo=None):
    """Shared ``q^2`` nodes for `pair_moments` with per-species rho weights."""
    s = m * m
    g, wg = np.polynomial.legendre.leggauss(nq)
    lo = 4 * M_E**2 if q2lo is None else float(q2lo)
    edges = sorted(set([lo, 4 * M_E**2, 4 * M_MU**2, 4 * M_PI_CH**2, 1.0,
                        2.25, 3.24, Q2_CHARM, 25.0, 4 * M_TAU**2, Q2_BOTTOM,
                        0.25 * s, s]))
    edges = [e for e in edges if lo <= e <= s]
    Q, W = [], []
    for e0, e1 in zip(edges[:-1], edges[1:]):
        a, b = math.log(e0), math.log(e1)
        Q.append(np.exp(0.5 * (b - a) * (g + 1.0) + a))
        W.append(0.5 * (b - a) * wg)
    q2 = np.concatenate(Q); dl = np.concatenate(W)
    rq2, ra = r_resonance_weights()
    keep = rq2 >= lo
    rq2, ra = rq2[keep], ra[keep]
    q2 = np.concatenate([q2, rq2]); dl = np.concatenate([dl, np.zeros(len(rq2))])
    wq = {}
    for sp in species:
        if sp == "had":
            w = (ALPHA / (3.0 * math.pi)) * r_had_cont(q2) * dl
            if len(rq2):
                w[-len(rq2):] = (ALPHA / (3.0 * math.pi)) * ra
        else:
            w = rho_lepton(q2, {"e": M_E, "mu": M_MU, "tau": M_TAU}[sp]) \
                * (q2 >= PAIR_Q2LO[sp]) * dl
        wq[sp] = w
    return q2, wq, dl


# --------------------------------------------------------------------------
# the pair table
# --------------------------------------------------------------------------
# `pair_radiator` costs a few hundred ms per ``z``, so the kernel reads a
# precomputed table of
#
#     B(u; m) = R_pair(z) / [ (alpha/pi) (1+z^2)/(1-z) ] ,   u = -(1/2) ln z ,
#
# one column per species, log-spaced in ``u`` and linearly interpolated in
# ``ln m``.  ``B`` is O(0.1) and smooth in ``ln u``; dividing out the
# Altarelli-Parisi pole is what makes the interpolation accurate at both ends.
# Rebuild with ``fsr_analytic.py pairtable``.
PAIR_TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "data", "fsr", "pairkern.npz")
_PAIR_CACHE = {}


def _pair_table_cell(arg):
    mm, sp, z, nq, ng, eik = arg
    P = (1.0 + z * z) / (1.0 - z)
    return pair_radiator(mm, z, (sp,), nq=nq, ng=ng, eikonal=eik) / (A_PI * P)


def build_pair_table(masses=None, nu=200, u_lo=1e-6, u_hi=7.0,
                     species=PAIR_SPECIES, nq=16, ng=24, procs=1,
                     eikonal=False):
    """Tabulate ``B(u; m)`` for every species; returns a dict for ``np.savez``."""
    if masses is None:
        masses = np.geomspace(40.0, 260.0, 13)
    masses = np.asarray(masses, float)
    u = np.geomspace(u_lo, u_hi, nu)
    z = np.exp(-2.0 * u)
    jobs = [(float(mm), sp, z, nq, ng, eikonal)
            for sp in species for mm in masses]
    if procs > 1:
        import multiprocessing as mp
        with mp.Pool(procs) as pool:
            res = pool.map(_pair_table_cell, jobs, chunksize=1)
    else:
        res = [_pair_table_cell(j) for j in jobs]
    out = dict(u=u, m=masses, species=np.array(list(species)))
    for i, sp in enumerate(species):
        out["B_" + sp] = np.stack(res[i * len(masses):(i + 1) * len(masses)])
    return out


def pair_pdf_u(m, u, species=PAIR_SPECIES, path=None):
    """``K_u(u) = 2 z R_pair(z)``: the pair kernel's density in ``u``.

    The full pair kernel is ``(1 - N_pair) delta(u) + K_u(u)``; this is its
    continuous part, which is what a pair-only Photos run measures away from
    ``u = 0``.
    """
    u = np.asarray(u, float)
    z = np.exp(-2.0 * u)
    return 2.0 * z * PairTerm(m, species, path=path)(z, -np.expm1(-2.0 * u))


class PairTerm:
    """``R_pair(z)`` for a species set, from `PAIR_TABLE`, at one mass."""

    def __init__(self, m, species, path=None):
        self.species = tuple(species)
        self.m = float(m)
        if not self.species:
            self.B = None
            self.rate = self.mean_u = 0.0
            return
        path = PAIR_TABLE if path is None else path
        d = _PAIR_CACHE.get(path)
        if d is None:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"pair table {path} missing; build it with "
                    f"`python3 fsr_analytic.py pairtable -o {path}`")
            d = {k: v for k, v in np.load(path, allow_pickle=True).items()}
            _PAIR_CACHE[path] = d
        self._u = d["u"]
        mg = d["m"]
        lm = math.log(self.m)
        j = int(np.clip(np.searchsorted(np.log(mg), lm) - 1, 0, len(mg) - 2))
        f = (lm - math.log(mg[j])) / (math.log(mg[j + 1]) - math.log(mg[j]))
        f = min(max(f, 0.0), 1.0)
        B = np.zeros_like(self._u)
        for sp in self.species:
            Bs = d["B_" + sp]
            B = B + (1.0 - f) * Bs[j] + f * Bs[j + 1]
        self.B = B
        self._lnu = np.log(self._u)

    def __call__(self, z, x=None):
        """``R_pair(z)``; ``x = 1-z`` may be supplied at full precision."""
        z = np.asarray(z, float)
        x = (1.0 - z) if x is None else np.asarray(x, float)
        if self.B is None:
            return np.zeros_like(z)
        u = -0.5 * np.log1p(-x)
        B = np.interp(np.log(np.maximum(u, 1e-300)), self._lnu, self.B,
                      left=0.0, right=0.0)
        return A_PI * (1.0 + z * z) / x * B


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
        pair_table=args.pair_table,
        var_budget=(None if args.sigma_cap else args.var_budget),
        sigma_cap=args.sigma_cap, n_fine=args.n_fine, ng=args.ng)
    meta = dict(kind="analytic", variant=args.variant, pair=list(pair),
                minus_one=not args.no_minus_one,
                freeze_beta_at=args.freeze_beta_at, alpha=ALPHA, m_mu=M_MU,
                pair_table=args.pair_table,
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
              f"  pair = {b['pair_rate']:.3e}  atoms = {b['natoms']:4d}"
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


# --------------------------------------------------------------------------
# validation of the O(alpha^2) NLL term
# --------------------------------------------------------------------------
# Everything here is a *check*, not part of the kernel: closed forms against
# numerical convolutions, sum rules, the two-loop anomalous dimension in Mellin
# space, and the O(alpha^2) expansion of the kernel itself.
def _gl_panels(n=64, npan=1600, lo=1e-13):
    """Gauss-Legendre nodes/weights on (0,1), clustered at both endpoints.

    The panels are built once on ``t in (0, 1/2]`` and used twice, as ``z = t``
    below the midpoint and as ``x = t`` above it, so both ``z`` and ``x = 1-z``
    keep full relative precision at their own endpoint and no node ever lands
    on 0 or 1.
    """
    g, wg = np.polynomial.legendre.leggauss(n)
    e = np.unique(np.concatenate([[0.0, 0.5], np.geomspace(lo, 0.5, npan)]))
    half = 0.5 * (e[1:] - e[:-1])[:, None]
    t = half * (g + 1.0) + e[:-1][:, None]
    w = half * wg
    return (np.concatenate([t, 1.0 - t]), np.concatenate([1.0 - t, t]),
            np.concatenate([w, w]))


def _integ(f, **kw):
    z, x, w = _gl_panels(**kw)
    return float(np.sum(w * f(z, x)))


def _mellin(f, n, **kw):
    z, x, w = _gl_panels(**kw)
    return float(np.sum(w * np.power(z, n - 1.0) * f(z, x)))


def _dilog(z):
    """``Li_2(z)`` on [0, 1], from `dilog_neg` by Landen + reflection."""
    z = np.asarray(z, float)
    lo = np.minimum(z, 0.5)
    d_lo = -dilog_neg(lo / (1.0 - lo)) - 0.5 * np.log1p(-lo) ** 2
    # the z <= 1/2 branch is the one used there; clamp the other so that the
    # unselected half of the `where` does not raise on z -> 0
    hi = np.clip(1.0 - z, 1e-300, 0.5)
    d_hi_1mz = -dilog_neg(hi / (1.0 - hi)) - 0.5 * np.log1p(-hi) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        d_hi = (math.pi**2 / 6.0
                - np.log(np.maximum(z, 1e-300)) * np.log(np.maximum(1.0 - z, 1e-300))
                - d_hi_1mz)
    return np.where(z <= 0.5, d_lo, d_hi)


def diff_p1_hpl(z, x=None):
    """``diffP1ns`` of Mitov-Moch-Vogt `tlike-ns.h`, transcribed literally.

    ``H(R(0)) = ln z``, ``H(R(0,0)) = ln^2 z/2``,
    ``H(R(1,0)) = -ln z ln(1-z) - Li_2(z)``, ``H(R(2)) = Li_2(z)``; the colour
    factor is ``cf^2 -> 1`` and the normalisation ``(alpha_s/4pi)^2``.
    """
    z = np.asarray(z, float)
    x = (1.0 - z) if x is None else np.asarray(x, float)
    lz, l1, li2 = _lnz(z, x), np.log(x), _dilog(z)
    H0, H00 = lz, 0.5 * lz * lz
    H10, H2 = -lz * l1 - li2, li2
    return ((24.0 / x - 20.0 - 4.0 * z) * H0
            + (-32.0 / x + 24.0 + 24.0 * z) * H00
            + (-32.0 / x + 16.0 + 16.0 * z) * H10
            + (-32.0 / x + 16.0 + 16.0 * z) * H2)


def _conv_p0(z, f, n=48, npan=900, lo=1e-9):
    """``([p]_+ (x) f)(z)`` by quadrature, for one scalar ``z``.

    ``[p]_+ = 2[1/(1-y)]_+ - (1+y) + (3/2) delta(1-y)``; the plus-distribution is
    integrated as ``int dy [f(z/y)/y - f(z)]/(1-y) + f(z) ln(1-z)``.
    """
    g, wg = np.polynomial.legendre.leggauss(n)
    e = z + (1.0 - z) * np.unique(np.concatenate(
        [[0.0, 1.0], np.geomspace(lo, 0.5, npan // 2),
         1.0 - np.geomspace(lo, 0.5, npan // 2)]))
    y = 0.5 * (e[1:] - e[:-1])[:, None] * (g + 1.0) + e[:-1][:, None]
    w = 0.5 * (e[1:] - e[:-1])[:, None] * wg
    out = 2.0 * np.sum(w * ((f(z / y) / y - f(z)) / (1.0 - y)))
    out += 2.0 * f(z) * math.log1p(-z) - np.sum(w * (1.0 + y) * f(z / y) / y)
    return out + 1.5 * f(z)


def _harm(k, m):
    """``S_k(m)`` for integer or half-integer ``m >= 0``.

    Half-integer: ``psi``-continuation, written as finite odd-denominator sums,
    ``S_1(m+1/2) = -2 ln 2 + 2 sum 1/(2k+1)`` etc.
    """
    if abs(m - round(m)) < 1e-12:
        return float(sum(1.0 / j**k for j in range(1, int(round(m)) + 1)))
    mm = int(round(m - 0.5))
    t = sum(1.0 / (2 * j + 1) ** k for j in range(mm + 1))
    return {1: -2.0 * math.log(2.0) + 2.0 * t,
            2: -2.0 * math.pi**2 / 6.0 + 4.0 * t,
            3: -6.0 * ZETA3 + 8.0 * t}[k]


def gamma1_ns_minus_abelian(n):
    """``gamma^{(1)}_{ns,-}(n)``, C_F^2 part, in ``alpha_s/(4 pi)`` units.

    Moch-Vermaseren-Vogt (Nucl. Phys. B688 (2004) 101) eq. (3.6); transcribed
    from the harmonic-sum form in `eko` (``gqq1m_cfcf``).  ``g3n`` is the Mellin
    transform of ``Li_2(x)/(1+x)``, done here by quadrature.
    """
    z2, log2 = math.pi**2 / 6.0, math.log(2.0)
    S1, S2 = _harm(1, n), _harm(2, n)
    Sp1m, Sp2m, Sp3m = (_harm(k, (n - 1) / 2.0) for k in (1, 2, 3))
    g3n = _integ(lambda z, x: np.power(z, n - 1.0) * _dilog(z) / (1.0 + z))
    return (-32.0 * g3n
            + (24.0 - n * (-32.0 + 3.0 * n * (-8.0 + n * (3.0 + n)
                                              * (3.0 + n**2))))
            / (2.0 * n**3 * (1.0 + n) ** 3)
            + (12.0 - 8.0 / n + 8.0 / (1.0 + n)) * S2
            + S1 * (-24.0 / n**2 - 8.0 / (1.0 + n) ** 2 + 16.0 * S2
                    - 16.0 * Sp2m)
            + 8.0 * Sp2m / (n + n * n) - 4.0 * Sp3m - 20.0 * ZETA3
            + z2 * (-32.0 * S1 + 32.0 * Sp1m + 32.0 * (1.0 / n + log2)))


def _q2_reg(z, x=None):
    """Regular part of ``[P0 (x) P0]``, eq. (3), without the ``beta^2/8``."""
    z = np.asarray(z, float)
    x = (1.0 - z) if x is None else np.asarray(x, float)
    lz = _lnz(z, x)
    return ((1.0 + z) * (3.0 * lz - 4.0 * np.log(x)) - 4.0 * lz / x - 5.0 - z)


def _nll_cmd(args):
    ok = lambda v, t=1e-10: "ok " if abs(v) < t else "FAIL"   # nan -> FAIL
    P = lambda lab, v, t=1e-10: print(f"  {ok(v,t)}  {lab:<58s} {v:12.3e}")

    print("\n=== 1. the time-like difference, eq. (11) ===")
    print("  Mitov-Moch-Vogt tlike-ns.h `diffP1ns` (HPLs, with Li_2) against")
    print("  8 x ([p]_+ (x) [p ln]) with A(z) of eq. (12), and against a direct")
    print("  numerical convolution.")
    print(f"  {'z':>10s}{'diffP1ns':>16s}{'8 A(z)':>16s}{'8 x numeric':>16s}"
          f"{'A/num - 1':>13s}")
    fln = lambda w: (1.0 + w * w) * np.log(w) / (1.0 - w)
    worst = 0.0
    for z in (0.9, 0.7, 0.5, 0.3, 0.1, 0.03, 0.01, 1e-3, 1e-5):
        d = float(diff_p1_hpl(np.array([z]))[0])
        a = float(conv_p0_pln(np.array([z]))[0])
        c = _conv_p0(z, fln)
        worst = max(worst, abs(d - 8.0 * a), abs(a / c - 1.0))
        print(f"  {z:10.4g}{d:16.8f}{8*a:16.8f}{8*c:16.8f}{a/c-1:13.2e}")
    P("max |diffP1ns - 8 A| and |A/numeric - 1|", worst, 1e-9)

    print("\n=== 2. sum rules ===")
    iS, iT = _integ(p1_spacelike), _integ(p1_timelike)
    P("int_0^1 P1S dz + delta   (fermion number, NS_-)", iS + P1_DELTA)
    P("int_0^1 P1T dz + delta   (fermion number, NS_-)", iT + P1_DELTA)
    P("int_0^1 A dz            (= g0(1) x int p ln z = 0)", _integ(conv_p0_pln))
    P("int_0^1 G dz + g_delta  (normalisation of (8))",
      _integ(nll_reg) + G_DELTA)
    P("int_0^1 [P0 (x) P0]_reg dz - INT_Q2", _integ(_q2_reg) - INT_Q2)
    P("g_delta - (P1_DELTA/2 + 3 kappa/2)",
      G_DELTA - (0.5 * P1_DELTA + 1.5 * KAPPA))
    P("P1T - P1S - 2 A   (max over z)",
      max(abs(float(p1_timelike(np.array([z]))[0]
                    - p1_spacelike(np.array([z]))[0]
                    - 2.0 * conv_p0_pln(np.array([z]))[0]))
          for z in (0.9, 0.5, 0.1, 0.01, 1e-4)), 1e-12)
    P("G - (P1T/2 + A - kappa(1+z))   (max over z)",
      max(abs(float(nll_reg(np.array([z]))[0]
                    - 0.5 * p1_timelike(np.array([z]))[0]
                    - conv_p0_pln(np.array([z]))[0]
                    + KAPPA * (1.0 + z)))
          for z in (0.9, 0.5, 0.1, 0.01, 1e-4)), 1e-12)

    print("\n=== 3. two-loop anomalous dimension, Mellin moments ===")
    print("  x-space P1S (abelian NS_-) against gamma^(1)_{ns,-}|_{C_F^2} of")
    print("  Moch-Vermaseren-Vogt eq. (3.6) in harmonic sums; the normalisation")
    print("  is P1(n) = -gamma^(1)(n)/4.")
    print(f"  {'n':>3s}{'M[P1S](n)':>18s}{'-gamma/4':>18s}{'difference':>13s}")
    wn = 0.0
    for n in range(1, 9):
        m = _mellin(p1_spacelike, n) + P1_DELTA
        g = -0.25 * gamma1_ns_minus_abelian(float(n))
        wn = max(wn, abs(m - g))
        print(f"  {n:3d}{m:18.10f}{g:18.10f}{m-g:13.2e}")
    P("max |M[P1S](n) + gamma^(1)_{ns,-}(n)/4|, n = 1..8", wn, 1e-9)

    print("\n=== 4. soft limit: the abelian two-loop cusp vanishes ===")
    print("  no [1/(1-z)]_+ in P1T, so (1-z) P1T -> 0 and P1T itself is finite")
    print(f"  {'1-z':>10s}{'P1T(z)':>16s}{'(1-z) P1T':>16s}{'G(z)':>16s}")
    for x in (1e-2, 1e-4, 1e-6, 1e-8):
        zz, xx = np.array([1.0 - x]), np.array([x])
        print(f"  {x:10.0e}{float(p1_timelike(zz,xx)[0]):16.6f}"
              f"{float(x*p1_timelike(zz,xx)[0]):16.3e}"
              f"{float(nll_reg(zz,xx)[0]):16.6f}")
    xx = np.array([1e-8])
    P("(1-z) P1T at 1-z = 1e-8", float((xx * p1_timelike(1.0 - xx, xx))[0]),
      1e-6)
    print("  delta(1-z) coefficient of P1T:"
          f"  3/8 - pi^2/2 + 6 zeta_3 = {P1_DELTA:.12f}")

    print("\n=== 5. the O(alpha^2) expansion of the kernel, in Mellin space ===")
    print("  a^2 coefficient of int z^{n-1} K dz must be")
    print("      (1/2) L^2 g0^2 + L [ (1/2) g1 + g0 c1 ] + (NNLL, L^0)")
    print("  with g0, g1, c1 the moments of P0, P1T, C1 = Chat - P0.")
    print(f"  {'n':>3s}{'L^2: kernel':>15s}{'L^2: g0^2/2':>15s}{'diff':>11s}"
          f"{'L: kernel':>15s}{'L: NLL pred':>15s}{'diff':>11s}")
    jn = lambda n: _mellin(lambda z, x: (1.0 + z * z) / x * _lnz(z, x), n)
    w2 = w1 = 0.0
    for n in range(1, 9):
        S1, S2 = _harm(1, n) - 1.0 / n, _harm(2, n) - 1.0 / n**2   # S_{n-1}
        g0 = 1.5 - 2.0 * S1 - 1.0 / n - 1.0 / (n + 1.0)
        c1 = jn(n) + KAPPA - g0
        g1 = _mellin(p1_timelike, n) + P1_DELTA
        gq, gn = _mellin(_q2_reg, n), _mellin(nll_reg, n)
        def coef2(L):
            b1 = 1.5 * (L - 1.0) + KAPPA
            b2 = -0.5 * (L - 1.0) ** 2 * INT_Q2 + G_DELTA * L
            return (b2 - 2.0 * b1 * (L - 1.0) * S1
                    + 4.0 * (L - 1.0) ** 2 * (0.5 * S2 + 0.5 * S1 * S1)
                    + 0.5 * (L - 1.0) ** 2 * gq + L * gn)
        V = np.linalg.solve(np.array([[L * L, L, 1.0] for L in (1.0, 2.0, 3.0)]),
                            np.array([coef2(L) for L in (1.0, 2.0, 3.0)]))
        pred2, pred1 = 0.5 * g0 * g0, 0.5 * g1 + g0 * c1
        w2, w1 = max(w2, abs(V[0] - pred2)), max(w1, abs(V[1] - pred1))
        print(f"  {n:3d}{V[0]:15.8f}{pred2:15.8f}{V[0]-pred2:11.2e}"
              f"{V[1]:15.8f}{pred1:15.8f}{V[1]-pred1:11.2e}")
    P("max |L^2 coefficient - P0 (x) P0 / 2|, n = 1..8", w2, 1e-9)
    P("max |L coefficient - (P1T/2 + P0 (x) C1)|, n = 1..8", w1, 1e-9)

    print("\n=== 6. normalisation of the kernel itself ===")
    for m in args.mass:
        for v in ("exp1", "exp2", "exp2nll"):
            k = FSRKernel(m, variant=v)
            tg = k._fine_grid(None, 8000)
            raw = float(k._cells(tg, 32, clip=False)[0].sum())
            cut = float(k._cells(tg, 32, clip=True)[0].sum())
            print(f"  m = {m:7.3f}  {v:8s}  C = {k._C:.10f}"
                  f"   int K dz - 1 = {raw - 1.0:10.2e} (unclipped)"
                  f"  {cut - 1.0:10.2e} (clipped at K >= 0)")
        print("  The delta(1-z) of (8) is put into C analytically, so int K dz")
        print("  is 1 by construction; what is left is quadrature.  Clipping")
        print("  the negative far tail of the fixed-order O(alpha^2) terms")
        print("  (z < 7e-5 at the Z, see the NLL note in the README) is the")
        print("  difference between the two columns.")

    print("\n=== 7. discretisation ===")
    print("  <u> of the atoms against the panel count and the Gauss order")
    for m in args.mass[:1]:
        for v in ("exp2", "exp2nll"):
            base = None
            for nf, ng in ((4000, 16), (8000, 32), (16000, 32)):
                k = FSRKernel(m, variant=v)
                r, w, tot = k.atoms(n_fine=nf, ng=ng)
                mu = -float(np.sum(w * np.log(r)))
                base = mu if base is None else base
                print(f"  m = {m:7.3f}  {v:8s}  n_fine = {nf:6d}  ng = {ng:3d}"
                      f"  atoms = {len(r):5d}  <u> = {mu*1e3:.8f}e-3"
                      f"  d = {mu-base:9.2e}  captured = {tot:.10f}")

    print("\n=== 8. additive vs exponentiated NLL: the difference is O(alpha^3) ===")
    print("  (8) puts its delta(1-z) into C, i.e. under the exponentiated soft")
    print("  factor.  Adding it as an explicit atom at z = 1 instead changes")
    print("  the kernel by c_nll g_delta [beta (1-z)^{beta-1} - delta(1-z)]")
    print("  = c_nll g_delta beta [1/(1-z)]_+ + O(alpha^4), i.e. O(alpha^3);")
    print("  there is nothing else to exponentiate, because the abelian")
    print("  two-loop cusp is zero.")
    for m in args.mass:
        k = FSRKernel(m, variant="exp2nll")
        d = k._c_nll * G_DELTA
        u_soft = _integ(lambda z, x, k=k: (-0.5 * _lnz(z, x)) * k.beta
                        * np.power(x, k.beta - 1.0))
        du = -d * u_soft
        print(f"  m = {m:7.3f}  c_nll g_delta = {d:.6e}  <u>_soft = {u_soft:.6e}"
              f"  d<u> = {du:.3e}  ({du/FSRKernel(m,'exp2nll').moments()['u']:.2e}"
              f" relative, alpha^3/alpha = {ALPHA/math.pi*(coll_log(m)-1)*2:.3f})")


def _pairtable_cmd(args):
    t = build_pair_table(masses=args.masses, nu=args.nu, u_lo=args.u_lo,
                         u_hi=args.u_hi, nq=args.nq, ng=args.ng,
                         procs=args.procs, eikonal=args.eikonal)
    meta = dict(kind="pairtable", alpha=ALPHA, m_mu=M_MU, m_e=M_E, m_tau=M_TAU,
                nu=args.nu, u_lo=args.u_lo, u_hi=args.u_hi, nq=args.nq,
                ng=args.ng, masses=list(map(float, t["m"])),
                eikonal=bool(args.eikonal),
                r_qcd=R_QCD, resonances=[list(r) for r in R_RESONANCES])
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(args.output, provenance=np.array([json.dumps(meta)]), **t)
    print(f"[pairtable] {len(t['m'])} masses x {args.nu} u x "
          f"{len(PAIR_SPECIES)} species -> {args.output}")
    for sp in PAIR_SPECIES:
        i = int(np.argmin(np.abs(t["m"] - 91.1876)))
        print(f"    {sp:>4s}: max B = {t['B_' + sp][i].max():.5f} at m = "
              f"{t['m'][i]:.2f}")


def _pair_cmd(args):
    """Exact O(alpha^2) pair emission: closure, the LL error, and Photos."""
    m = args.mass
    print(f"Real pair emission off the muon line, m = {m} GeV\n")
    print("1. massive-photon matrix element")
    for z, q2 in ((0.9, 1e-12), (0.5, 1e-12), (0.1, 1e-12)):
        ex = r1_exact(z, m)
        mv = float(spec_gammastar(m, np.array([z]), np.array([q2]))[0])
        print(f"   q2 -> 0, z = {z:4.2f}: massive/massless - 1 = {mv / ex - 1:+.2e}")
    for z, q2 in ((0.9, 1e-4), (0.5, 1.0), (0.1, 100.0)):
        a = float(spec_gammastar(m, np.array([z]), np.array([q2]),
                                 va=(1.0, 0.0))[0])
        print(f"   current conservation at z = {z:4.2f}, q2 = {q2:8.1e}: "
              f"R = {a:.6e}")
    print("\n2. rate and mass loss, exact vs the dispersive leading log")
    print(f"{'species':>8s}{'rate':>13s}{'<u> N':>13s}{'<u2> N':>13s}"
          f"{'LL <u> N':>13s}{'exact/LL':>10s}")
    per = pair_moments(m)
    tot = [0.0, 0.0, 0.0]
    for sp in PAIR_SPECIES:
        v = per[sp]
        ll = beta_pair_ll(m, (sp,)) * 0.5099671
        print(f"{sp:>8s}{v[0]:13.5e}{v[1]:13.5e}{v[2]:13.5e}{ll:13.5e}"
              f"{v[1] / ll:10.4f}")
        for i in range(3):
            tot[i] += v[i]
    llall = beta_pair_ll(m, PAIR_SPECIES) * 0.5099671
    print(f"{'TOTAL':>8s}{tot[0]:13.5e}{tot[1]:13.5e}{tot[2]:13.5e}"
          f"{llall:13.5e}{tot[1] / llall:10.4f}")
    print("\n3. the eikonal limit = Photos++ 3.61 (`pairs.cxx` YOT1)")
    pe = pair_moments(m, species=("e", "mu"), eikonal=True)
    ph = {"e": (2.41855e-3, 1.95999e-4), "mu": (3.12600e-4, 5.68119e-5)}
    print(f"{'species':>8s}{'eik rate':>13s}{'Photos':>13s}{'ratio':>9s}"
          f"{'eik <u>N':>13s}{'Photos':>13s}{'ratio':>9s}")
    for sp in ("e", "mu"):
        v = pe[sp]
        print(f"{sp:>8s}{v[0]:13.5e}{ph[sp][0]:13.5e}{ph[sp][0] / v[0]:9.4f}"
              f"{v[1]:13.5e}{ph[sp][1]:13.5e}{ph[sp][1] / v[1]:9.4f}")
    print("   (Photos: 1e9 standalone events at this mass, photons off)")
    print("\n4. table closure: int R_pair dz and int u R_pair dz")
    k = FSRKernel(m, variant="exp2nll", pair=PAIR_SPECIES)
    k0 = FSRKernel(m, variant="exp2nll")
    (r, w, t1), (r0, w0, t0) = k.atoms(), k0.atoms()
    du = -float(np.sum(w * np.log(r))) + float(np.sum(w0 * np.log(r0)))
    print(f"   table rate = {k.pair_rate:.5e}   quadrature = {tot[0]:.5e}"
          f"   ratio = {k.pair_rate / tot[0]:.4f}")
    print(f"   atoms {len(r0)} -> {len(r)};  d<u> = {du:.6e}"
          f"   quadrature = {tot[1]:.6e}   ratio = {du / tot[1]:.5f}")
    print(f"   captured weight = {t1:.9f} (no pairs {t0:.9f})")
    print("\n5. singlet background: pairs with q inside a 60-120 GeV window")
    if m > 60.0:
        pc = pair_moments(m, species=("e", "mu"), q2lo=3600.0)
        print(f"   rate(q > 60 GeV) = {sum(v[0] for v in pc.values()):.3e} "
              f"per event")


def _moments_cmd(args):
    for m in args.mass:
        if args.u_cut:
            for tag in ("exp1", "exp2", "exp2nll"):
                kk = FSRKernel(m, variant=tag)
                b = kk.moments_below(args.u_cut)
                print(f"  [{tag}] m = {m:8.4f}  beta = {kk.beta:.6f}"
                      f"  P(u < {args.u_cut:.5f}) = {b['p']:.6f}"
                      f"  <u | in window> = {b['u']*1e3:8.4f}e-3"
                      f"  -> dm = {-m*b['u']*1e3:8.3f} MeV")
        print(f"\nm_pre = {m} GeV   L = {float(coll_log(m)):.6f}")
        for v in ("exp1", "exp2", "exp2nll", "oalpha"):
            for pair in ((), ("e",), ("e", "mu", "tau", "had")):
                if v != "exp1" and pair:
                    continue
                k = FSRKernel(m, variant=v, pair=pair)
                d = k.moments()
                tag = v + ("+pair:" + "/".join(pair) if pair else "")
                print(f"  {tag:<24s} beta = {k.beta:.6f}"
                      f"  (pair {k.pair_rate:.3e})"
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
    k.add_argument("--pair-table", default=None,
                   help="pair table npz; default data/fsr/pairkern.npz")
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

    nl = sub.add_parser("nll", help="validate the O(alpha^2) NLL term")
    nl.add_argument("--mass", nargs="*", type=float, default=[91.1876])
    nl.set_defaults(func=_nll_cmd)

    pt = sub.add_parser("pairtable", help="build the exact pair-emission table")
    pt.add_argument("-o", "--output", default=PAIR_TABLE)
    pt.add_argument("--nu", type=int, default=200)
    pt.add_argument("--u-lo", type=float, default=1e-6)
    pt.add_argument("--u-hi", type=float, default=7.0)
    pt.add_argument("--masses", nargs="*", type=float, default=None)
    pt.add_argument("--nq", type=int, default=16)
    pt.add_argument("--ng", type=int, default=24)
    pt.add_argument("--procs", type=int, default=1)
    pt.add_argument("--eikonal", action="store_true",
                    help="soft-limit matrix element = what Photos generates")
    pt.set_defaults(func=_pairtable_cmd)

    pr = sub.add_parser("pair", help="validate the exact pair term")
    pr.add_argument("--mass", type=float, default=91.1876)
    pr.set_defaults(func=_pair_cmd)

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
