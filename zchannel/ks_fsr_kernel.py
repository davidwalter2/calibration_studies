#!/usr/bin/env python3
r"""The K_S -> pi+ pi-(gamma) QED kernel for the unbinned CVH mass likelihood.

A K_S mass term carries a ``DeltaKernel`` at ``MKS = 0.497611``: the modelled
pre-radiation mass is the PDG value with no spread (the K_S total width is
7.35e-15 GeV, eleven orders below any resolution).  In CMS simulation the K_S
is decayed by Geant4 WITHOUT radiation, so for MC that delta is exact and the
kernel is 1.  In DATA it is not: the decay radiates, the two tracks carry the
post-radiation momenta, and the reconstructed mass is the bare pi+ pi- mass.
This module builds the object that repairs it -- the kernel characteristic
function ``phi_K(t)`` that ``rabbit.unbinned.MassCFTerm`` takes as
``phik = (phik_t, phik_re, phik_im)``.

For a **delta** lineshape the kernel CF is exact and additive, exactly as for
the J/psi (`jpsi_fsr_kernel`): with ``m_pre == M`` the post-radiation density
is ``p(m') = K(m'/M)/M`` and

    phi_K(t) = Int K(r) exp(i t M (r - 1)) dr = < exp(i t dm) > ,  dm = m' - M .

One function of ``t``, no ``m_pre`` dependence -- so the banded fold matrix of
`fsr_table` (which exists because the Z's ``m_pre`` varies and its kernel is
multiplicative) does NOT apply here, and the cells this module stores are the
same ones `cf_from_cells` integrates, not a fold table.

WHY THE FERMION KERNEL CANNOT BE REUSED
    Nothing about this decay is soft or collinear.  ``m_pi/M = 0.280``, the
    pion velocity in the pair frame is ``beta_0 = 0.828``, the collinear log
    ``ln(M^2/m_pi^2) = 2.542`` is O(1) and the hardest photon is 170.5 MeV.
    The Altarelli-Parisi/structure-function machinery of `fsr_analytic`
    (`FSRKernel`, its ``exp2``/``exp2nll`` variants, the ``mass_exact``
    remainder) is an expansion in ``1/L`` around a massless splitting, and at
    ``L = 2.54`` there is nothing to expand.  What replaces it is the EXACT
    scalar-QED matrix element integrated over the exact three-body phase space
    at fixed ``z = (m'/M)^2``, which for a pointlike constant weak vertex is
    available in closed form -- `fsr_analytic.ScalarIBKernel`, whose derivation
    and formula numbers (S1)-(S7) are in that module.  In one line,

        R1(z) = (alpha/pi) (beta(z)/beta_0) Bcal(beta(z)) z/(1-z) ,
        Bcal(beta) = (1+beta^2)/beta ln((1+beta)/(1-beta)) - 2 ,

    the YFS soft function of the pion pair at the OUTGOING pair velocity times
    the scalar splitting weight ``z/(1-z)``.  Its soft limit is the exact
    massive eikonal ``b/(1-z)``, ``b = (alpha/pi) Bcal(beta_0) = 6.526e-3``,
    which is what is exponentiated; the hard remainder is regular by
    construction, with no cancellation and no tabulation.

WHAT IS IN THE KERNEL AND WHAT IS BOUNDED OUT OF IT
    * **inner bremsstrahlung**, exactly, to O(alpha) with the eikonal
      exponentiated.  This is the whole kernel.
    * **direct emission** is bounded by MEASUREMENT, not by theory: TAUREG 76
      finds < 0.06e-3 of Gamma(pi+ pi-) at 90 % CL with the same 50 MeV cut,
      i.e. < 2.3 % of the IB rate there (BURGUN 73: 0.3 +- 0.6 in the same
      units).  (The CP argument runs the other way round from K_L, where IB is
      CP suppressed and the CP-allowed direct emission is M1: for K_S the IB is
      the CP-allowed amplitude and the E1 direct emission is CP allowed too --
      what suppresses it is the chiral counting, O(p^4) against the
      DeltaI = 1/2 enhanced O(p^2) of the IB, and there is no reason to lean on
      CP here.)  What keeps it OUT of the kernel's windows is its SPECTRUM:
      direct emission rises as ``E*_gamma^3``, so the fraction of it below a
      window edge ``E*_w`` is ``~(E*_w/E*_max)^4`` -- 1.8e-4 at the +-20 MeV
      edge -- and the resulting shift of the window-conditional mean is below
      1e-9 of the scale.  Even a deliberately wrong FLAT direct-emission
      spectrum leaves it below 1e-6.
    * **the e+ e- conversion channel** (K_S -> pi+ pi- e+ e-, PDG
      (4.79 +- 0.15)e-5, i.e. 6.92e-5 of pi+ pi-) is a real pair replacing a
      real photon.  In a window of halfwidth ``w`` its contribution to
      ``<dm>`` is bounded by ``P_pair . w`` -- 3.5e-4 MeV at ``w = 5 MeV``,
      7e-7 of the scale -- because a pair event that loses more than ``w``
      leaves the window instead of biasing it.
    * **virtual corrections** are proportional to ``delta(1-z)`` at O(alpha)
      and drop out of a normalised kernel exactly.
    * **the pi+ pi- Coulomb (Sommerfeld) factor** is a normalisation, not a
      shape: ``S(z)/S(1)`` moves by 2.0e-4 between ``z = 1`` and the 50 MeV
      point.  ``--coulomb`` switches it on and `validate` measures what it
      does (< 1e-8 of the scale in every window).

usage::

    python ks_fsr_kernel.py analytic -o data/ks_kern_data.npz --report
    python ks_fsr_kernel.py validate            # every check, with numbers
"""

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import fsr_analytic as fa                                        # noqa: E402
from jpsi_fsr_kernel import cf_from_cells                        # noqa: E402

#: PDG K^0_S mass [GeV]
MKS = fa.M_KS
#: PDG charged pion mass [GeV]
MPI = fa.M_PI_CH
#: PDG Gamma(K_S -> pi+ pi- gamma)/Gamma(K_S -> pi+ pi-) for p_gamma > 50 MeV/c
#: ("OUR AVERAGE" of RAMBERG 93 E731 and TAUREG 76; unchanged 2014 -> 2024)
PDG_IB_50 = (2.59e-3, 0.08e-3)
#: RAMBERG 93 E731 at p_gamma > 20 MeV/c (PDG lists it, does not average it)
PDG_IB_20 = (7.10e-3, 0.22e-3)
#: TAUREG 76 limit on the direct-emission contribution, same 50 MeV cut, 90 % CL
PDG_DE_LIMIT_50 = 0.06e-3
#: PDG B(K_S -> pi+ pi- e+ e-) / B(K_S -> pi+ pi-) = 4.79e-5/0.6920
PAIR_RATE = 4.79e-5 / 0.6920

#: the windows a K_S mass term would plausibly normalise over, in GeV.  The
#: K_S mass resolution is ~1 % (5 MeV), so these are 1-4 sigma.
WINDOWS = (0.005, 0.010, 0.020)
#: u = -ln(m'/M) ladder the tails are reported on
U_GRID = (1e-5, 1e-4, 1e-3, 0.0101, 0.0203, 0.0410, 0.1, 0.2, 0.4)


# --------------------------------------------------------------------------
# the kernel
# --------------------------------------------------------------------------
def build(mass=MKS, mch=MPI, variant="exp1", coulomb=False,
          n_fine=20000, ng=16, log=print):
    """``(kernel, w, ubar, var)``: fine cells of the normalised kernel in ``u``."""
    k = fa.ScalarIBKernel(mass, mch=mch, variant=variant, coulomb=coulomb)
    c = k.cells(n_fine=n_fine, ng=ng)
    w, m1, m2 = c[0], c[1], c[2]
    good = w > 0
    w, m1, m2 = w[good], m1[good], m2[good]
    ub = m1 / w
    var = np.maximum(m2 / w - ub * ub, 0.0)
    tot = w.sum()
    log(f"  ScalarIBKernel(M={mass:.6f}, m={mch:.8f}, variant={variant}, "
        f"coulomb={coulomb}):")
    log(f"    z_min = 4m^2/M^2 = {k.r:.9f}   beta_0 = {k.beta0:.9f}   "
        f"L = ln(M^2/m^2) = {k.L:.6f}")
    log(f"    soft exponent b = (alpha/pi) Bcal(beta_0) = {k.beta:.9e}   "
        f"C = {k._C:.9f}   int h dz = {k._int_h:.9e}")
    log(f"    E*_gamma,max = {k.egamma(k.r) * 1e3:.4f} MeV   "
        f"u_max = {k._u_max():.6f}   dm_min = "
        f"{mass * math.expm1(-k._u_max()) * 1e3:.3f} MeV")
    log(f"    {len(w)} cells, captured {tot:.12f}")
    return k, w / tot, ub, var


def report(k, w, ub, mass, log, windows=WINDOWS):
    """Tails, window means and the out-of-window fractions, from the cells."""
    out = {}
    dm = mass * np.expm1(-ub)
    mom = k.moments()
    log(f"\n  === K(z) for M = {mass:.6f} GeV ===")
    log(f"  <u> = {mom['u']:.9e}   <u^2> = {mom['u2']:.9e}   "
        f"<1-z> = {mom['x']:.9e}   norm - 1 = {mom['norm'] - 1:.2e}")
    log(f"  inclusive <dm> = {float(np.sum(w * dm)) * 1e3:+.6f} MeV "
        f"= {float(np.sum(w * dm)) / mass:+.6e} relative")
    out["mean_u"] = float(mom["u"])
    out["mean_u2"] = float(mom["u2"])
    out["mean_dm"] = float(np.sum(w * dm))
    log("  P(u > u0)                                   dm(u0) [MeV]")
    for u0 in U_GRID:
        p = float(np.sum(w[ub > u0]))
        log(f"    u0 = {u0:<8g}  {p:.6e}                  "
            f"{mass * math.expm1(-u0) * 1e3:9.3f}")
        out[f"tail_{u0:g}"] = p
    log("\n  window        P(in)        out-of-window   <dm|in> [MeV]   "
        "relative        E*_gamma(edge)")
    out["windows"] = {}
    for hw in windows:
        m = np.abs(dm) <= hw
        pw = float(np.sum(w[m]))
        mw = float(np.sum(w[m] * dm[m]) / pw)
        eg = 0.5 * mass * (1.0 - (1.0 - hw / mass) ** 2)
        log(f"  +-{hw * 1e3:5.1f} MeV   {pw:.6f}     {1.0 - pw:.6e}    "
            f"{mw * 1e3:+.6f}       {mw / mass:+.6e}    {eg * 1e3:6.3f} MeV")
        out["windows"][f"{hw:g}"] = dict(p_in=pw, p_out=1.0 - pw,
                                         mean_dm=mw, rel=mw / mass,
                                         egamma_edge=float(eg))
    return out


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
def v_soft(k, log):
    """(a) the soft limit of ``R1`` against the classic soft-photon formula."""
    log("\n(a) SOFT LIMIT.  R1(z)(1-z) -> b = (alpha/pi) Bcal(beta_0), the")
    log("    eikonal number spectrum dn = (alpha/pi) Bcal(beta_0) dw/w.")
    b_soft = float(fa.A_PI * fa.soft_B_scalar(np.array([k.beta0]))[0])
    log(f"    (alpha/pi) Bcal(beta_0)          = {b_soft:.15e}")
    log(f"    k.beta (from g(1)/beta_0)        = {k.beta:.15e}")
    log(f"    relative difference              = {abs(b_soft / k.beta - 1):.2e}")
    worst = 0.0
    for lx in (-2, -4, -6, -8, -12, -20):
        x = 10.0 ** lx
        v = float(k.r1(np.array([1.0 - x]), np.array([x]))[0]) * x / k.beta
        worst = max(worst, abs(v - 1.0))
        log(f"    x = 1e{lx:<4d}  R1 x / b = {v:.12f}")
    log(f"    -> R1 x/b - 1 = {worst:.2e} at x = 1e-2, and O(x) below it")
    # the threshold limit, Bcal -> (8/3) beta^2
    bt = 1e-3
    log(f"    threshold: Bcal({bt:g})/( (8/3) beta^2 ) - 1 = "
        f"{float(fa.soft_B_scalar(np.array([bt]))[0]) / (8.0 / 3.0 * bt * bt) - 1.0:+.2e}")
    return dict(b_soft=b_soft, b_kernel=float(k.beta), worst_soft=worst)


def v_norm(k, log, n_fine=20000, ng=16):
    """(b) ``int K dz = 1``, and ``int h dz`` by two independent quadratures."""
    log("\n(b) NORMALISATION.")
    n = float(k._cells(k._fine_grid(None, n_fine), ng)[0].sum())
    a, bq = k.int_h(), k.int_h_parts()
    log(f"    int K dz - 1                     = {n - 1.0:+.3e}   "
        f"(t = x^b Gauss-Legendre, {n_fine} panels x {ng})")
    log(f"    int h dz, beta quadrature        = {a:.15e}")
    log(f"    int h dz, by parts via dg/dz=2Lam= {bq:.15e}")
    log(f"    relative difference              = {abs(a / bq - 1):.2e}")
    return dict(norm_minus_one=n - 1.0, int_h=a, int_h_parts=bq)


def v_rate(k, log):
    """(c) the PDG normalisation gate on the IB rate above a photon-energy cut."""
    log("\n(c) IB RATE GATE.  The fixed-order O(alpha) integral of R1 above a")
    log("    photon-energy cut, against the measured radiative fraction.")
    out = {}
    for cut, (val, err), tag in ((0.050, PDG_IB_50, "PDG average"),
                                 (0.020, PDG_IB_20, "RAMBERG 93 E731")):
        th = k.rate_above(cut)
        log(f"    E*_gamma > {cut * 1e3:.0f} MeV: this work {th:.5e}   "
            f"measured ({val:.3e} +- {err:.2e})  [{tag}]")
        log(f"                      ratio {th / val:.4f}   "
            f"pull {(th - val) / err:+.2f} sigma")
        out[f"rate_{cut:g}"] = dict(theory=th, meas=val, err=err,
                                    ratio=th / val, pull=(th - val) / err)
    log(f"    direct-emission limit at the 50 MeV cut: "
        f"< {PDG_DE_LIMIT_50:.2e} (TAUREG 76, 90 % CL) = "
        f"{PDG_DE_LIMIT_50 / PDG_IB_50[0] * 100:.1f} % of IB; its spectrum")
    log(f"    rises as E*^3, and the 50 MeV point is dm = "
        f"{MKS * ((1 - 2 * 0.050 / MKS) ** 0.5 - 1) * 1e3:.1f} MeV.")
    return out


def v_expansion(k, log, lams=(1.0, 0.3, 0.1, 0.03, 0.01)):
    """(d) the O(alpha) expansion of the exponentiated kernel reproduces R1.

    ``<u>`` of the exponentiated kernel is, at O(alpha), exactly
    ``int u R1 dz`` -- the ``delta(1-z)`` carries no ``u``.  Scaling the
    coupling by ``lam`` and watching ``<u>(lam)/lam`` approach
    ``int u R1 dz`` linearly in ``lam`` is the statement, checked numerically
    rather than asserted: the residual must fall by 10 when ``lam`` does.
    """
    log("\n(d) O(alpha) EXPANSION of the exponentiated kernel.")
    log("    <u>_exp/lam -> int u R1 dz as the coupling lam -> 0, with an")
    log("    O(lam) residual: the exponentiation adds nothing at O(alpha).")
    # int u R1 dz, in beta (the substitution that removes the sqrt at z = r);
    # u R1 is integrable at z = 1 because u ~ (1-z)/2 kills the pole
    npan, ng = 4000, 24
    e = np.linspace(0.0, k.beta0, npan + 1)
    g, wg = np.polynomial.legendre.leggauss(ng)
    b = 0.5 * (e[1:] - e[:-1])[:, None] * (g[None, :] + 1.0) + e[:-1][:, None]
    w = 0.5 * (e[1:] - e[:-1])[:, None] * wg[None, :]
    z = k.r / (1.0 - b * b)
    x = 1.0 - z
    jac = 2.0 * k.r * b / (1.0 - b * b) ** 2
    u = -0.5 * np.log(z)
    i_uR1 = float(np.sum(w * u * k.r1(z, x) * jac))
    log(f"    int u R1 dz (O(alpha), exact)    = {i_uR1:.12e}")
    prev = None
    out = {"int_u_r1": i_uR1, "scan": []}
    for lam in lams:
        kk = fa.ScalarIBKernel(k.m, mch=k.mch, variant=k.variant)
        kk.beta = kk.beta_gam = k.beta * lam
        kk._scale = lam
        # h is linear in alpha, and so is C - 1
        kk.h = lambda z_, x_=None, _k=k, _l=lam: _l * _k.h(z_, x_)
        kk._int_h = lam * k._int_h
        kk._C = (1.0 - kk._int_h) / (1.0 - kk.r) ** kk.beta
        m = kk.moments(n=20000)["u"] / lam
        d = m - i_uR1
        rate = "" if prev is None else f"   ratio to previous {d / prev:.4f}"
        log(f"    lam = {lam:<6g} <u>/lam = {m:.12e}   residual "
            f"{d:+.3e}{rate}")
        out["scan"].append(dict(lam=lam, mean_u_over_lam=float(m),
                                residual=float(d)))
        prev = d
    log("    (the residual falls linearly in lam: it is the O(alpha^2) of the")
    log("     exponentiated form, not a defect of the O(alpha) content)")
    return out



def v_resum(k, w, ub, log, windows=WINDOWS, npan=8000, ng=24):
    """What the exponentiation is worth, and therefore the theory uncertainty.

    At FIXED ORDER the window-conditional mean is cutoff independent: the soft
    cutoff only moves weight at ``u ~ 0``, where ``dm ~ 0``, so both

        num = int_{|dm| < w} dm R1 dz      and      P_out = int_{|dm| > w} R1 dz

    are finite, and ``<dm|in> = num/(1 - P_out)`` with the virtual + soft
    ``delta(1-z)`` supplying the rest of the probability.  The difference
    against the exponentiated kernel is the WHOLE effect of the resummation.
    Since the eikonal exponentiates exactly (YFS), what is left uncontrolled is
    the non-eikonal O(alpha^2), smaller than this difference by a further
    factor ~b = 6.5e-3 -- so this line is the QED theory uncertainty of the
    kernel, not merely a variation of it.
    """
    log("\n(j) WHAT THE EXPONENTIATION IS WORTH (= the theory uncertainty).")
    g, wg = np.polynomial.legendre.leggauss(ng)

    def _gl(lo, hi):
        e = np.linspace(lo, hi, npan + 1)
        n = 0.5 * (e[1:] - e[:-1])[:, None]
        return n * (g[None, :] + 1.0) + e[:-1][:, None], n * wg[None, :]

    dm = k.m * np.expm1(-ub)
    out = {"windows": {}}
    log("      window    exponentiated      fixed order        difference/M"
        "      P_out exp / P_out FO")
    for hw in windows:
        zc = (1.0 - hw / k.m) ** 2
        lx, ww = _gl(math.log(1e-300), math.log1p(-zc))
        x = np.exp(lx)
        z = 1.0 - x
        # dz = x dlnx
        num = float(np.sum(ww * k.m * (np.sqrt(z) - 1.0) * k.r1(z, x) * x))
        b, ww = _gl(0.0, math.sqrt(1.0 - k.r / zc))
        z = k.r / (1.0 - b * b)
        p_out = float(np.sum(ww * k.r1(z, 1.0 - z)
                             * (2.0 * k.r * b / (1.0 - b * b) ** 2)))
        fo = num / (1.0 - p_out)
        s = np.abs(dm) <= hw
        ex = float(np.sum(w[s] * dm[s]) / np.sum(w[s]))
        log(f"      +-{hw * 1e3:5.1f} MeV  {ex * 1e3:+.6f} MeV   "
            f"{fo * 1e3:+.6f} MeV   {(ex - fo) / k.m:+.3e}    "
            f"{1.0 - float(np.sum(w[s])):.6e} / {p_out:.6e}")
        out["windows"][f"{hw:g}"] = dict(exp=ex, fixed_order=fo,
                                         rel=(ex - fo) / k.m, p_out_fo=p_out)
    log("    The resummation itself is a few 1e-6 of the scale; the non-eikonal")
    log("    O(alpha^2) it does not carry is another factor b = 6.5e-3 below")
    log("    that, i.e. below 1e-8.  That is the kernel's QED uncertainty.")
    return out


# -- (e) the direct three-body Monte Carlo ---------------------------------
def _dalitz_vectors(M, m, z, s23):
    """Explicit four-vectors ``(p_plus, p_minus, k)`` in the parent rest frame.

    Built from the invariants only: ``s12 = z M^2 = (p_+ + p_-)^2`` and
    ``s23 = (p_- + k)^2``.  The photon is put along ``+z`` (the orientation is
    integrated over trivially and drops out of a Lorentz scalar).  Returns the
    vectors in the ``(E, px, py, pz)`` convention with metric ``(+,-,-,-)``.
    """
    s12 = z * M * M
    s13 = M * M + 2.0 * m * m - s12 - s23
    eg = (M * M - s12) / (2.0 * M)
    ep = (M * M + m * m - s23) / (2.0 * M)      # E of pi+  (recoiling vs s23)
    em = (M * M + m * m - s13) / (2.0 * M)
    pp = np.sqrt(np.maximum(ep * ep - m * m, 0.0))
    # p_+ . k = (s13 - m^2)/2 = ep eg - |p_+| eg cos
    cos = (ep * eg - 0.5 * (s13 - m * m)) / np.maximum(pp * eg, 1e-300)
    cos = np.clip(cos, -1.0, 1.0)
    sin = np.sqrt(np.maximum(1.0 - cos * cos, 0.0))
    zero = np.zeros_like(ep)
    k4 = np.stack([eg * np.ones_like(ep), zero, zero, eg * np.ones_like(ep)], -1)
    p4p = np.stack([ep, pp * sin, zero, pp * cos], -1)
    P = np.stack([M * np.ones_like(ep), zero, zero, zero], -1)
    p4m = P - k4 - p4p
    return p4p, p4m, k4, em


def _dot(a, b):
    return (a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]
            - a[..., 2] * b[..., 2] - a[..., 3] * b[..., 3])


def sum_m2_over_g2(pp, pm, k, m):
    """``sum_pol |M|^2 / (e^2 |G|^2) = -J^2``, eq. (S2), from four-vectors."""
    dp, dm_ = _dot(pp, k), _dot(pm, k)
    return (2.0 * _dot(pp, pm) / (dp * dm_)
            - m * m / (dp * dp) - m * m / (dm_ * dm_))


def v_mc(k, log, zs=(0.95, 0.70, 0.40), n=2_000_000, seed=20260917):
    """(e) the closed-form angular integral against a direct 3-body MC.

    At fixed ``z``, the Dalitz variable ``s23 = (p_- + k)^2`` is uniform over
    ``[s23-, s23+]`` in the flat three-body phase space
    ``dGamma = |M|^2 ds12 ds23/(32 (2 pi)^3 M^3)``, so

        (1/Gamma_0) dGamma/dz = (alpha/(4 pi beta_0)) int ds23 (-J^2)

    with ``Gamma_0 = |G|^2 beta_0/(16 pi M)``.  The four-vectors are rebuilt
    explicitly from ``(z, s23)`` and ``-J^2`` is evaluated from their dot
    products -- no reuse of the closed form (S3), which is what makes this an
    independent check of the algebra AND of the phase-space normalisation.
    """
    log("\n(e) DIRECT THREE-BODY MONTE CARLO of sum|M|^2 from four-vectors.")
    M, m = k.m, k.mch
    rng = np.random.default_rng(seed)
    out = {"z": [], "mc": [], "closed": [], "ratio": [], "err": []}
    log("        z      R1 (closed form)      R1 (MC)              ratio - 1"
        "        MC error")
    for z in zs:
        s12 = z * M * M
        E, w = 0.5 * math.sqrt(s12), (M * M - s12) / (2.0 * math.sqrt(s12))
        q = E * math.sqrt(1.0 - 4.0 * m * m / s12)
        s23lo, s23hi = m * m + 2.0 * w * (E - q), m * m + 2.0 * w * (E + q)
        s23 = rng.uniform(s23lo, s23hi, n)
        pp, pm, kk, em = _dalitz_vectors(M, m, z, s23)
        # the vectors must be on shell and add up: this is the generator's gate
        bad = max(float(np.max(np.abs(_dot(pm, pm) / (m * m) - 1.0))),
                  float(np.max(np.abs(pm[..., 0] / em - 1.0))))
        t = sum_m2_over_g2(pp, pm, kk, m)
        pre = fa.ALPHA / (4.0 * math.pi * k.beta0) * (s23hi - s23lo)
        mc, err = pre * float(t.mean()), pre * float(t.std() / math.sqrt(n))
        cf = float(k.r1(np.array([z]))[0])
        log(f"    {z:6.3f}   {cf:.12e}   {mc:.12e}   {mc / cf - 1:+.2e}"
            f"      {err / cf:.1e}   (p_-^2/m^2 - 1 < {bad:.1e})")
        for key, v in zip(("z", "mc", "closed", "ratio", "err"),
                          (z, mc, cf, mc / cf - 1.0, err / cf)):
            out[key].append(float(v))
    # gauge invariance of the current, and (S3) against (S2)
    z = 0.70
    s12 = z * M * M
    E, w = 0.5 * math.sqrt(s12), (M * M - s12) / (2.0 * math.sqrt(s12))
    q = E * math.sqrt(1.0 - 4.0 * m * m / s12)
    s23 = rng.uniform(m * m + 2.0 * w * (E - q), m * m + 2.0 * w * (E + q), 5000)
    pp, pm, kk, _ = _dalitz_vectors(M, m, z, s23)
    J = (pp / _dot(pp, kk)[..., None] - pm / _dot(pm, kk)[..., None])
    kJ = np.abs(_dot(kk, J)) * float(np.mean(_dot(pp, kk)))
    log(f"    gauge invariance  max |k.J| (scaled) = {float(kJ.max()):.2e}")
    # (S3): 4 beta^2 sin^2/(w^2 (1-beta^2 cos^2)^2), cos from s23
    bmu = q / E
    cos = (E - 0.5 * (s23 - m * m) / w) / q
    s3 = 4.0 * bmu ** 2 * (1.0 - cos ** 2) / (w * w * (1.0 - bmu ** 2 * cos ** 2) ** 2)
    s2 = sum_m2_over_g2(pp, pm, kk, m)
    log(f"    closed form (S3) vs (S2) from vectors: max rel "
        f"{float(np.max(np.abs(s3 / s2 - 1.0))):.2e}")
    out["gauge"] = float(kJ.max())
    out["s3_vs_s2"] = float(np.max(np.abs(s3 / s2 - 1.0)))
    return out


def v_coulomb(k, log, windows=WINDOWS):
    """The pi+ pi- Sommerfeld factor: a normalisation, not a shape."""
    log("\n(f) COULOMB / VIRTUAL.  Virtual corrections are ~delta(1-z) at")
    log("    O(alpha) and cancel in a normalised kernel exactly.  The")
    log("    attractive pi+ pi- Sommerfeld factor does depend on z through")
    log("    beta(z), so it is measured rather than assumed:")
    s1 = float(k._coulomb(np.array([1.0]))[0])
    for z in (1.0, 0.99, 0.95, 0.90, 0.799, 0.5, k.r + 1e-6):
        s = float(k._coulomb(np.array([z]))[0])
        log(f"      z = {z:<8.5f}  S = {s:.9f}   S/S(1) - 1 = {s / s1 - 1:+.3e}"
            f"   (dm = {k.m * (math.sqrt(z) - 1) * 1e3:+8.2f} MeV)")
    kc, wc, ubc, _ = build(k.m, k.mch, k.variant, coulomb=True, log=lambda *_: None)
    _, w0, ub0, _ = build(k.m, k.mch, k.variant, coulomb=False,
                          log=lambda *_: None)
    log("      window      <dm|in> no Coulomb   with Coulomb    difference")
    out = {"S1": s1, "windows": {}}
    for hw in windows:
        d0, dc = k.m * np.expm1(-ub0), k.m * np.expm1(-ubc)
        m0, mc = np.abs(d0) <= hw, np.abs(dc) <= hw
        a = float(np.sum(w0[m0] * d0[m0]) / np.sum(w0[m0]))
        b = float(np.sum(wc[mc] * dc[mc]) / np.sum(wc[mc]))
        log(f"      +-{hw * 1e3:5.1f} MeV  {a * 1e3:+.9f} MeV   "
            f"{b * 1e3:+.9f} MeV  {(b - a) / k.m:+.2e} of the scale")
        out["windows"][f"{hw:g}"] = dict(no=a, yes=b, rel=(b - a) / k.m)
    return out


def v_bounds(k, w, ub, log, windows=WINDOWS):
    """Direct emission and the e+e- conversion channel, bounded per window.

    Both are bounded by MEASUREMENT times a window-geometry argument, with no
    model of either channel: what a window-conditional mean can be moved by is
    the weight that falls INSIDE the window times the halfwidth, and a decay
    that loses more than the halfwidth leaves the window instead of biasing it.
    """
    log("\n(g) DIRECT EMISSION and PAIR CONVERSION, bounded per window.")
    dm = k.m * np.expm1(-ub)
    egmax = float(k.egamma(k.r))
    out = {"windows": {}}
    log(f"    DE: < {PDG_DE_LIMIT_50:.1e} of Gamma(pi+pi-) above E* = 50 MeV")
    log(f"        (TAUREG 76, 90 % CL) = {PDG_DE_LIMIT_50 / PDG_IB_50[0] * 100:.1f}"
        " % of the IB rate there.  Its spectrum rises")
    log("        as E*^3, so the fraction of it below a window edge E*_w is")
    log("        (E*_w^4)/(E*_max^4 - (50 MeV)^4), and a FLAT spectrum -- "
        "deliberately")
    log("        wrong, as a ceiling -- would give E*_w/(E*_max - 50 MeV).")
    log(f"    pairs: P = B(pi+pi-e+e-)/B(pi+pi-) = {PAIR_RATE:.3e}, a real pair")
    log("        replacing a real photon.")
    log("      window       pair |d<dm>|            DE |d<dm>|, E*^3     "
        "DE, flat spectrum")
    for hw in windows:
        pin = float(np.sum(w[np.abs(dm) <= hw]))
        egw = 0.5 * k.m * (1.0 - (1.0 - hw / k.m) ** 2)
        pair = PAIR_RATE * hw / pin
        f_e3 = egw ** 4 / (egmax ** 4 - 0.050 ** 4)
        f_flat = egw / (egmax - 0.050)
        de3, dfl = PDG_DE_LIMIT_50 * f_e3 * hw / pin, \
            PDG_DE_LIMIT_50 * f_flat * hw / pin
        log(f"      +-{hw * 1e3:5.1f} MeV  <= {pair * 1e3:.2e} MeV "
            f"({pair / k.m:.1e})  <= {de3 * 1e3:.2e} MeV ({de3 / k.m:.1e})"
            f"  <= {dfl * 1e3:.2e} MeV ({dfl / k.m:.1e})")
        out["windows"][f"{hw:g}"] = dict(
            pair_bound_dm=pair, pair_bound_rel=pair / k.m,
            de_e3_rel=de3 / k.m, de_flat_rel=dfl / k.m,
            de_frac_in_e3=f_e3, de_frac_in_flat=f_flat)
    return out


def v_literature(k, log):
    """The E*_gamma spectrum against the K -> pi pi gamma IB form."""
    log("\n(h) THE E*_gamma SPECTRUM, in the form the K -> pi pi gamma")
    log("    literature writes it (NA48/KTeV/KLOE):")
    log("      dGamma_IB/dE* = Gamma_0 (alpha/pi) (beta/beta_0) Bcal(beta)")
    log("                      x (1 - 2E*/M) / E* ,   beta = beta(m_pipi) .")
    log("    The pure eikonal keeps only the 1/E* and Bcal(beta_0); the ratio")
    log("    to it is what a soft approximation would get wrong:")
    log("      E* [MeV]   dG/dE* / Gamma_0 [1/GeV]   /(eikonal)   beta(z)")
    out = {"egamma": [], "ratio": []}
    for eg in (1e-3, 5e-3, 0.010, 0.020, 0.050, 0.100, 0.150, 0.170):
        z = 1.0 - 2.0 * eg / k.m
        if z <= k.r:
            continue
        d = float(k.r1_egamma(np.array([eg]))[0])
        eik = k.beta / eg
        log(f"      {eg * 1e3:7.2f}    {d:.6e}              {d / eik:.6f}"
            f"     {float(k.beta_of_z(np.array([z]))[0]):.6f}")
        out["egamma"].append(eg)
        out["ratio"].append(d / eik)
    return out


def v_cf(k, w, ub, var, log, n_fine=20000, ng=16, tmax=6000.0, dt=0.05,
         tprobe=(1.0, 100.0, 767.0, 2000.0, 3946.0, 6000.0)):
    """(i) the CF tabulation: cell discretisation and linear interpolation.

    Two errors, both measured rather than asserted.  The CELL error is the
    residual of the per-cell ``O(var^2)`` expansion `cf_from_cells` makes: it
    is bounded by rebuilding the kernel with three times as many panels and
    comparing at the probe frequencies.  The INTERPOLATION error is what
    ``MassCFTerm._interp_phik`` adds when it reads the uniform grid between
    nodes: it is ``dt^2 |phi''|/8 <= dt^2 E[dm^2]/8``, checked at the midpoints.
    """
    log("\n(i) CF DISCRETISATION.")
    npt = int(round(tmax / dt)) + 1
    t, re, im = cf_from_cells(w, ub, var, k.m, tmax, npt, log=lambda *_: None)
    c = k.cells(n_fine=3 * n_fine, ng=ng)
    g = c[0] > 0
    w2, m1, m2 = c[0][g], c[1][g], c[2][g]
    ub2 = m1 / w2
    var2 = np.maximum(m2 / w2 - ub2 * ub2, 0.0)
    w2 = w2 / w2.sum()
    tp = np.array([x for x in tprobe if x <= tmax])
    # the reference is `cf_from_cells`' integrand at the probes only, since
    # that routine wants a uniform grid and only six frequencies are needed
    g_ = np.expm1(-ub2)
    ph = k.m * np.outer(tp, g_)
    f = np.exp(1j * ph)
    corr = 1.0 + 0.5 * var2[None, :] * (
        1j * tp[:, None] * k.m * (1.0 + g_[None, :])
        - (tp[:, None] * k.m * (1.0 + g_[None, :])) ** 2)
    ref = (f * corr) @ w2
    log(f"    cells {len(w)} (n_fine={n_fine}) against {len(w2)} "
        f"(n_fine={3 * n_fine}); grid {npt} points, dt = {dt}")
    worst_cell = 0.0
    for j, tv in enumerate(tp):
        tab = complex(np.interp(tv, t, re), np.interp(tv, t, im))
        d = abs(tab - ref[j])
        worst_cell = max(worst_cell, d)
        log(f"      t = {tv:7.1f}  phi = {ref[j].real:+.9f}{ref[j].imag:+.9f}i"
            f"   |table - fine| = {d:.2e}")
    # interpolation: midpoints against the exact cell sum
    mid = 0.5 * (t[:-1] + t[1:])
    sel = np.linspace(0, len(mid) - 1, 400).astype(int)
    tm = mid[sel]
    gg = np.expm1(-ub)
    fm = np.exp(1j * k.m * np.outer(tm, gg))
    cm = 1.0 + 0.5 * var[None, :] * (
        1j * tm[:, None] * k.m * (1.0 + gg[None, :])
        - (tm[:, None] * k.m * (1.0 + gg[None, :])) ** 2)
    ex = (fm * cm) @ w
    ap = np.interp(tm, t, re) + 1j * np.interp(tm, t, im)
    worst_int = float(np.max(np.abs(ap - ex)))
    log(f"    linear interpolation at dt = {dt}: worst |interp - exact| = "
        f"{worst_int:.2e}  (bound dt^2 E[dm^2]/8 = "
        f"{dt * dt * k.m ** 2 * k.moments()['u2'] / 8.0:.1e})")
    return dict(worst_cell=float(worst_cell), worst_interp=worst_int,
                tmax=tmax, dt=dt, npoints=npt)


def validate(mass=MKS, mch=MPI, variant="exp1", n_fine=20000, ng=16,
             mc_n=2_000_000, log=print):
    k, w, ub, var = build(mass, mch, variant, log=log)
    res = {"kernel": dict(M=mass, m=mch, variant=variant, r=k.r,
                          beta0=k.beta0, L=k.L, b=k.beta, C=k._C)}
    res["report"] = report(k, w, ub, mass, log)
    res["soft"] = v_soft(k, log)
    res["norm"] = v_norm(k, log, n_fine, ng)
    res["rate"] = v_rate(k, log)
    res["expansion"] = v_expansion(k, log)
    res["mc"] = v_mc(k, log, n=mc_n)
    res["coulomb"] = v_coulomb(k, log)
    res["bounds"] = v_bounds(k, w, ub, log)
    res["literature"] = v_literature(k, log)
    res["cf"] = v_cf(k, w, ub, var, log, n_fine, ng)
    res["resum"] = v_resum(k, w, ub, log)
    return res


# --------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["analytic", "validate"])
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--mass", type=float, default=MKS,
                   help="the term's m_ref, i.e. the delta the kernel replaces")
    p.add_argument("--mch", type=float, default=MPI,
                   help="charged-daughter mass; the default is the pion")
    p.add_argument("--variant", default="exp1", choices=["exp1"],
                   help="exp1, the exponentiated exact eikonal, is the only "
                        "kernel this writes: a FIXED-ORDER kernel carries a "
                        "delta(1-z) whose weight depends on the IR cutoff, so "
                        "renormalising the bare R1 to one would be a different "
                        "and wrong object. The cutoff-independent fixed-order "
                        "window mean is what `validate` (j) computes")
    p.add_argument("--coulomb", action="store_true",
                   help="fold in the pi+ pi- Sommerfeld factor S(z)/S(1)")
    p.add_argument("--tmax", type=float, default=6000.0,
                   help="upper end of the CF tabulation [1/GeV]; must exceed "
                        "max(tgrid)/min(sigma) of the term -- 7.8926/0.002 = "
                        "3946 for a 2 MeV K_S resolution -- or `_build_norm` "
                        "refuses the card")
    p.add_argument("--npoints", type=int, default=120001,
                   help="uniform tabulation points on [0, tmax]; the default "
                        "is dt = 0.05, where linear interpolation costs "
                        "dt^2 E[dm^2]/8 = 1.1e-8")
    p.add_argument("--truncate", type=float, default=0.0,
                   help="restrict the kernel to |dm| <= this and renormalise, "
                        "to match a cache built with a gen-mass acceptance")
    p.add_argument("--n-fine", type=int, default=20000)
    p.add_argument("--ng", type=int, default=16)
    p.add_argument("--mc-n", type=int, default=2_000_000,
                   help="validate: samples per z of the three-body MC")
    p.add_argument("--report", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    log = print
    if a.mode == "validate":
        res = validate(a.mass, a.mch, a.variant, a.n_fine, a.ng, a.mc_n, log)
        if a.output:
            with open(a.output, "w") as f:
                json.dump(res, f, indent=1, default=float)
            log(f"\n  -> {a.output}")
        return

    k, w, ub, var = build(a.mass, a.mch, a.variant, a.coulomb,
                          a.n_fine, a.ng, log)
    if a.truncate > 0:
        dmc = a.mass * np.expm1(-ub)
        m = np.abs(dmc) <= a.truncate
        w, ub, var = w[m] / w[m].sum(), ub[m], var[m]
        log(f"  truncated to |dm| <= {a.truncate} GeV and renormalised: "
            f"{int(m.sum())} of {len(m)} cells")
    stats = report(k, w, ub, a.mass, log) if a.report else {}
    t, re, im = cf_from_cells(w, ub, var, a.mass, a.tmax, a.npoints, log)

    log(f"\n  phi_K(0) = {re[0]:.12f}{im[0]:+.3e}i   "
        f"|phi_K(tmax)| = {math.hypot(re[-1], im[-1]):.6f}")
    dphi = (im[1] - im[0]) / (t[1] - t[0])
    log(f"  d Im phi_K/dt at 0 = <dm> = {dphi * 1e3:+.6f} MeV "
        f"= {dphi / a.mass:+.6e} relative")

    meta = {"mode": "analytic", "channel": "KS -> pi+ pi- (gamma)",
            "mass": a.mass, "mch": a.mch, "variant": a.variant,
            "coulomb": bool(a.coulomb), "truncate": a.truncate,
            "tmax": a.tmax, "npoints": a.npoints,
            "zmin": k.r, "beta0": k.beta0, "L": k.L, "b_soft": float(k.beta),
            "C": float(k._C), "int_h": float(k._int_h),
            "mean_dm_cf": float(dphi)}
    meta.update({f"stat_{key}": v for key, v in stats.items()})
    if a.output:
        os.makedirs(os.path.dirname(os.path.abspath(a.output)) or ".",
                    exist_ok=True)
        np.savez_compressed(a.output, phik_t=t, phik_re=re, phik_im=im,
                            u=ub, w=w, var=var, meta=json.dumps(meta))
        log(f"  -> {a.output}")


if __name__ == "__main__":
    main()
