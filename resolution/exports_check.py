#!/usr/bin/env python3
"""Global-fit EXPORTS under the IRLS/Fisher-scoring noise model.

WHAT THIS IS FOR (CRITIQUE_260813 point 2). CVH exports, per track, the
reduced gradient g = dL/da, the reduced Hessian H = d2L/da2 and the Jacobian
dx_ref/da with respect to the GLOBAL parameters (alignment, material, field).
Those feed a Millepede-style marginalised global fit. If they are taken from
the converged IRLS SURROGATE Gaussian problem rather than from the TRUE
likelihood they omit terms, and no momentum closure test would reveal it.

This script measures the omitted terms on a REAL block.

THE MODEL PARAMETER USED THROUGHOUT is the material / energy-loss scale, the
`dxival` of ResidualGlobalCorrectionMakerG4e (which, with doRes off, is also
`dionival` and `dmsval`, i.e. it scales the noise as well as the mean --
Geant4ePropagator.cc:1063 `ionifact = exp(dioni)*matStepFact`). For a compound
Poisson a pure rate scale a gives, EXACTLY,

    K(theta; a) = e^a K(theta)        (both centred and uncentred)
    mean loss    m(a) = e^a m

so every shape derivative is available in closed form with no new machinery.
This is the ONE global parameter with a first-order shape derivative; see the
notes file for why alignment and B-field have none.

NUMERICS RULES OBSERVED (they cost this study four wrong answers):
 * every expectation is taken against the EXACT FFT-inverted density, never
   against the saddlepoint density (NOTES_XXII: the SPA density is 2-8x low
   below the mode and up to 140x in the deep tail).
 * p, p', p'', d_a p, d_a p' all come from transforms of the SAME phi with
   different multipliers -- no numerical differencing of a density or of its
   log anywhere.
 * the t grid and the z grid are matched PER BLOCK; floors are RELATIVE.
 * psi is not monotonic; the profile minimisation is a bracketed scan +
   golden refinement, never an interpolated root inversion.

usage:
  python exports_check.py --model <model.root> [--plane 18] [--parts A,B,C,D]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cf_propagation_test as cpt                                     # noqa: E402
from cf_propagation_test import FUNCTIONALS, model_variance, step_transports  # noqa: E402
import cgf_channels as cc                                             # noqa: E402
from cgf_saddlepoint import ioni_cgf_derivs                           # noqa: E402


# ============================================================ block builders ==
def ioni_steps_of_legs(legs, k, avec, sigma, jsel):
    """collect_ioni_steps restricted to a subset of legs, transports to plane k
    unchanged. Mirrors cgf_channels.collect_ioni_steps exactly."""
    _, A_ioni, _ = step_transports(legs, k)
    out = []
    for j in jsel:
        leg = legs[j]
        if not len(leg["ioni"]):
            continue
        q = np.sign(leg["refqop"]) or 1.0
        w = q * np.einsum("i,sij->sj", avec, A_ioni[j])[:, 0] / sigma
        st = leg["ioni"].copy().astype(np.float64)
        st[:, 10] *= w
        out.append(st)
    return np.concatenate(out) if out else np.zeros((0, 11))


def mean_loss_z(steps):
    """Mean of the (uncentred) weighted loss of a pooled ionization block, in
    z units. This is the lever arm m_b = -dr_b/da of a rate scale.

    Channels, from the Urban record layout used by ioni_cgf_derivs:
      excitations   a1 at fixed e1*gam, a2 at fixed e2*gam
      delta rays    a3 collisions with 1/E^2 on [e0*gam, tmax*gam];
                    E[E] = e0 ln(w)/(1 - 1/w), w = tmax/e0
    Regime-0 (already-Gaussian) steps carry their mean in the reference, not
    in the fluctuation, so they contribute nothing here.
    """
    if not len(steps):
        return 0.0
    reg = steps[:, 0]
    gs = steps[:, 10].astype(np.float64) * 1e-3
    mu = reg != 0
    if not mu.any():
        return 0.0
    gam = steps[mu, 9]
    gsu = gs[mu]
    m = 0.0
    for ja, je in ((2, 3), (4, 5)):
        aj = steps[mu, ja]
        ej = steps[mu, je] * gam
        act = (aj > 0.0) & (ej > 0.0)
        if act.any():
            m += float(np.sum(aj[act] * gsu[act] * ej[act]))
    a3 = steps[mu, 6]
    e0 = steps[mu, 7] * gam
    tmax = steps[mu, 8] * gam
    act = (a3 > 0.0) & (tmax > e0) & (e0 > 0.0)
    if act.any():
        w = tmax[act] / e0[act]
        c = gsu[act] * e0[act]
        m += float(np.sum(a3[act] * c * np.log(w) / (1.0 - 1.0 / w)))
    return float(m)


# ================================================= exact density and scores ==
def _reach(S_of_t, lncut=-60.0, lo=1e-3, hi=1e8, n=500):
    """t at which Re S = lncut, bisected -- the per-block grid match."""
    t = np.geomspace(lo, hi, n)
    S = S_of_t(t).real
    below = np.where(S < lncut)[0]
    if not len(below):
        return float(hi)
    if int(below[0]) == 0:
        return float(lo)
    i = int(below[0])
    a, b = t[i - 1], t[i]
    for _ in range(6):
        m = np.sqrt(a * b)
        if float(S_of_t(np.array([m])).real[0]) < lncut:
            b = m
        else:
            a = m
    return float(b)


def _invert(S, tau, mult, nt=1 << 17, npad=32):
    """Real inversion of mult(t)*exp(S(t)) on the uniform grid matched to tau.

    mult is evaluated on the UNIFORM grid; pass a callable of (tu, Su).
    Returns (z, values) sorted in z.
    """
    tu = np.linspace(0.0, tau[-1], nt)
    Su = np.interp(tu, tau, S.real) + 1j * np.interp(tu, tau, S.imag)
    with np.errstate(over="ignore"):
        phi = np.exp(Su)
    dt = tu[1] - tu[0]
    wgt = np.full(nt, dt)
    wgt[0] = 0.5 * dt
    z = 2 * np.pi * np.fft.fftfreq(nt * npad, d=dt)
    o = np.argsort(z)
    out = []
    for m in mult:
        c = np.zeros(nt * npad, dtype=np.complex128)
        c[:nt] = phi * wgt * m(tu, Su)
        out.append((np.fft.fft(c).real / np.pi)[o])
    return z[o], out


def exact_block(S_of_t, scale=1.0, nt=1 << 17, npad=32, lncut=-60.0,
                floor=1e-9, tau=None, sl=None):
    """Everything the exports need for ONE block, all by exact FFT inversion.

    S_of_t(t) is the CENTRED CF exponent at rate 1. `scale` = e^a multiplies it
    (the rate scale), so S(t; a) = scale * S_0(t).

    Returns dict with, on the retained (relative-floor, contiguous) support:
        z, p, psi, dpsi, phi, dphi
    where  psi = -dln p/dr,  dpsi = -d2 ln p/dr2,
           phi = -d ln p/da |_r,  dphi = d psi/da |_r = d phi/dr.
    d/da at fixed r uses d_a S = S (rate scale), hence d_a phi_CF = S phi_CF.
    """
    if tau is None:
        treach = _reach(lambda t: scale * S_of_t(t), lncut=lncut)
        tau = np.concatenate([[0.0], np.geomspace(1e-4, 1.3 * treach, 8000)])
    S = scale * S_of_t(tau)
    z, (p, p1, p2, pa, pa1) = _invert(
        S, tau,
        [lambda t, s: 1.0,
         lambda t, s: -1j * t,
         lambda t, s: (-1j * t) ** 2,
         lambda t, s: s,
         lambda t, s: s * (-1j * t)],
        nt=nt, npad=npad)
    # contiguous relative-floor support around the mode. `sl` overrides it so
    # that a finite difference in `scale` can be taken on an IDENTICAL grid --
    # otherwise the support boundary moves and the difference picks up the
    # boundary, not the physics.
    if sl is None:
        i0 = int(np.nanargmax(p))
        thr = floor * p[i0]
        lo, hi = i0, i0
        while lo > 0 and p[lo - 1] > thr:
            lo -= 1
        while hi < len(p) - 1 and p[hi + 1] > thr:
            hi += 1
        s = slice(lo, hi + 1)
    else:
        s = sl
    z, p, p1, p2, pa, pa1 = z[s], p[s], p1[s], p2[s], pa[s], pa1[s]
    mass = float(np.trapezoid(p, z))
    psi = -p1 / p
    dpsi = -(p2 / p - (p1 / p) ** 2)
    phi = -pa / p
    dphi = -(pa1 / p - pa * p1 / p ** 2)
    return dict(z=z, p=p, lnp=np.log(p), psi=psi, dpsi=dpsi, phi=phi,
                dphi=dphi, mass=mass, dz=float(z[1] - z[0]),
                tmax=float(tau[-1]), tau=tau, sl=s)


def E(bl, f):
    """Expectation against the exact density on its retained support."""
    return float(np.trapezoid(f * bl["p"], bl["z"]) / bl["mass"])


# ========================================================= saddlepoint forms ==
def spa_scores(steps, r):
    """(theta_hat, psi, phi) from the CLOSED FORM, for the ionization channel.

    psi  = theta + K'''/(2 K''^2)                             (NOTES XIII)
    phi  = 1/2 - K(theta) - r (psi - theta)                   (derived here)

    The phi form is d/da of the saddlepoint log-density at fixed r under a rate
    scale, using d_a K = K, d_a K' = K', d_a K'' = K'' and K'(theta)=r:
        d_a ln p|_r = K - 1/2 + r K'''/(2 K''^2) = K - 1/2 + r (psi - theta).
    Gaussian check: K = r^2/2s2, K'''=0 -> d_a ln p = r^2/(2 s2) - 1/2. Correct.
    """
    from cgf_irls import solve_theta, make_block
    blk = make_block(steps, nfisher=200)
    th = np.array([solve_theta(blk, float(x)) for x in np.atleast_1d(r)])
    K, K1, K2, K3 = ioni_cgf_derivs(steps, th, order=3)
    psi = th + 0.5 * K3 / K2 ** 2
    phi = 0.5 - K - np.atleast_1d(r) * (psi - th)
    return th, psi, phi


# ================================================================== part A ====
def part_A(legs, k, avec, sigma, args):
    print("=" * 92)
    print("A. THE SHAPE SCORE phi ON A REAL BLOCK, AND THE BARTLETT IDENTITIES")
    print("=" * 92)
    print("""
   psi(r) = -dln p/dr        the residual score  (what the surrogate reproduces)
   phi(r) = -dln p/da |_r    the SHAPE score     (what the surrogate DROPS)
   a = material rate scale, so K(theta;a) = e^a K(theta) exactly.
""")
    rows = []
    for chans in (("ioni",), ("ioni", "ms", "rad")):
        def S_of_t(t, chans=chans):
            return cc.block_cf_exponent(legs, k, avec, sigma, t, channels=chans)
        bl = exact_block(S_of_t, nt=args.nt, npad=args.npad)
        Epsi = E(bl, bl["psi"])
        Ephi = E(bl, bl["phi"])
        I = E(bl, bl["psi"] ** 2)
        Ipa = E(bl, bl["psi"] * bl["phi"])
        Ia = E(bl, bl["phi"] ** 2)
        Edpsi = E(bl, bl["dpsi"])
        Edphi = E(bl, bl["dphi"])
        Epsi3 = E(bl, bl["psi"] ** 3)
        rows.append((chans, bl, dict(Epsi=Epsi, Ephi=Ephi, I=I, Ipa=Ipa, Ia=Ia,
                                     Edpsi=Edpsi, Edphi=Edphi, Epsi3=Epsi3)))
        tag = "+".join(chans)
        print(f"  block = plane {k}, channels {tag:14s}  "
              f"mass on support = {bl['mass']:.6f}, dz = {bl['dz']:.3e}, "
              f"z in [{bl['z'][0]:.3g}, {bl['z'][-1]:.3g}]")
        print(f"      E[psi]  = {Epsi:+.3e}   (must be 0)")
        print(f"      E[phi]  = {Ephi:+.3e}   (must be 0 -- d_a of the mass)")
        print(f"      I  = E[psi^2]  = {I:12.6f}   1/I = {1/I:.6f}")
        print(f"         E[dpsi/dr]  = {Edpsi:12.6f}   ratio {Edpsi/I:.6f}"
              "   <- 1st information identity")
        print(f"      E[psi phi]     = {Ipa:12.6f}")
        print(f"         E[dphi/dr]  = {Edphi:12.6f}   ratio "
              f"{Edphi/Ipa if Ipa else np.nan:.6f}   <- 2nd (Bartlett)")
        print(f"      E[phi^2]       = {Ia:12.6f}     (the shape channel's own "
              "information)")
        print(f"      E[psi^3]       = {Epsi3:+12.4f}    (0 for a Gaussian; "
              "sets the profile-score bias)")
        print()

    # SPA closed form vs exact, ionization channel only (the closed form above
    # is written for that channel).
    steps = cc.collect_ioni_steps(legs, k, avec, sigma)
    blI = rows[0][1]
    rprobe = np.array([-3.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    rprobe = rprobe[(rprobe > blI["z"][0]) & (rprobe < blI["z"][-1])]
    th, psi_s, phi_s = spa_scores(steps, rprobe)
    psi_e = np.interp(rprobe, blI["z"], blI["psi"])
    phi_e = np.interp(rprobe, blI["z"], blI["phi"])
    print("  SADDLEPOINT CLOSED FORM vs EXACT INVERSION (ionization channel):")
    print(f"  {'r':>7} {'theta_hat':>12} {'psi SPA':>11} {'psi exact':>11} "
          f"{'ratio':>8} | {'phi SPA':>11} {'phi exact':>11} {'ratio':>8}")
    for i, r in enumerate(rprobe):
        print(f"  {r:7.2f} {th[i]:12.4e} {psi_s[i]:11.5f} {psi_e[i]:11.5f} "
              f"{psi_s[i]/psi_e[i] if psi_e[i] else np.nan:8.4f} | "
              f"{phi_s[i]:11.5f} {phi_e[i]:11.5f} "
              f"{phi_s[i]/phi_e[i] if phi_e[i] else np.nan:8.4f}")
    print()
    return rows


# ================================================================== part B ====
def build_multiblock(legs, k, avec, sigma, minsteps=8, nt=1 << 16, npad=16):
    """Pooled per-leg ionization blocks, >= minsteps steps each (NOTES XXI 5b:
    the naive per-leg construction is degenerate because 4 legs have 1-2 steps,
    and the real fit pools per module / per global parameter anyway)."""
    from cf_track_resolution import ioni_step_exponent
    groups, cur = [], []
    for j in range(k + 1):
        cur.append(j)
        n = sum(len(legs[i]["ioni"]) for i in cur)
        if n >= minsteps:
            groups.append(cur)
            cur = []
    if cur:
        if groups:
            groups[-1].extend(cur)
        else:
            groups = [cur]
    blocks = []
    for g in groups:
        st = ioni_steps_of_legs(legs, k, avec, sigma, g)
        if not len(st):
            continue

        def S_of_t(t, st=st):
            return ioni_step_exponent(st, 1.0, t)
        bl = exact_block(S_of_t, nt=nt, npad=npad)
        bl["steps"] = st
        bl["legs"] = g
        bl["m"] = mean_loss_z(st)
        bl["S_of_t"] = S_of_t
        bl["nt"], bl["npad"] = nt, npad
        bl["I"] = E(bl, bl["psi"] ** 2)
        blocks.append(bl)
    return blocks


def rescaled_blocks(blocks, a):
    """The SAME blocks with the CGF rate scaled by e^a, on IDENTICAL tau and
    support grids so a finite difference in a is clean."""
    out = []
    for b in blocks:
        nb = exact_block(b["S_of_t"], scale=np.exp(a), nt=b["nt"],
                         npad=b["npad"], tau=b["tau"], sl=b["sl"])
        # m stays at its a=0 value: profile_c's mean channel is the ACCUMULATED
        # shift (e^a - 1) m(0), so d r/d a = -e^a m(0) -> -m(0) at a = 0.
        nb["m"] = b["m"]
        nb["steps"], nb["legs"] = b["steps"], b["legs"]
        out.append(nb)
    return out


def _interp(bl, key, r):
    """Linear interpolation of a tabulated per-block function at r.

    The FFT z grid is UNIFORM, so the index is arithmetic -- no search. This
    matters: the naive np.interp version made the profile scan O(minutes).
    """
    z = bl["z"]
    dz = bl["dz"]
    t = (r - z[0]) / dz
    if not np.isfinite(t) or t < 0.0 or t > len(z) - 1:
        return np.nan
    i = int(t)
    if i >= len(z) - 1:
        return float(bl[key][-1])
    f = t - i
    y = bl[key]
    return float(y[i] * (1.0 - f) + y[i + 1] * f)


def profile_c(blocks, zs, a, ngrid=4001, nbis=80):
    """argmin_c sum_b -ln p_b(z_b - c - (e^a-1) m_b ; a).

    Two stages, deliberately:
      1. a COARSE SCAN of F on the feasible domain to identify the basin. psi
         has multiple zero crossings (NOTES XXI 4a), so the root cannot be
         found by inversion; the basin has to be located by the objective.
      2. inside the bracket, BISECTION on the stationarity condition
         G(c) = sum_b psi_b(r_b) = 0, using the exactly-tabulated psi.
    Stage 2 is what makes the finite difference in `a` usable: golden-section
    on a piecewise-LINEAR interpolant of ln p only locates chat to O(dz), and
    dz = 2.8e-3 against a signal of 5e-4 over the FD step -- the scan-only
    version of this routine gave a 5-16 % noise floor on dchat/da.

    `blocks` may be a list of (block_at_this_a) dicts; the mean channel uses
    each block's m.
    """
    ea = np.exp(a)
    off = np.array([z - (ea - 1.0) * b["m"] for z, b in zip(zs, blocks)])

    def F(c):
        tot = 0.0
        for o, b in zip(off, blocks):
            lp = _interp(b, "lnp", o - c)
            if not np.isfinite(lp):
                return np.inf
            tot -= lp
        return tot

    def G(c):
        tot = 0.0
        for o, b in zip(off, blocks):
            v = _interp(b, "psi", o - c)
            if not np.isfinite(v):
                return np.nan
            tot += v
        return tot

    lo = max(o - b["z"][-1] for o, b in zip(off, blocks))
    hi = min(o - b["z"][0] for o, b in zip(off, blocks))
    if not (hi > lo):
        raise RuntimeError("empty feasible domain in c")
    pad = 1e-6 * (hi - lo)
    cs = np.linspace(lo + pad, hi - pad, ngrid)
    Fs = np.array([F(c) for c in cs])
    i = int(np.nanargmin(Fs))
    lo2 = cs[max(i - 1, 0)]
    hi2 = cs[min(i + 1, ngrid - 1)]
    glo, ghi = G(lo2), G(hi2)
    if not (np.isfinite(glo) and np.isfinite(ghi) and glo * ghi <= 0.0):
        # no sign change in the bracket (boundary minimum): fall back to the
        # scan node and say so by returning it unrefined
        return float(cs[i])
    for _ in range(nbis):
        mid = 0.5 * (lo2 + hi2)
        gm = G(mid)
        if not np.isfinite(gm):
            break
        if gm * glo <= 0.0:
            hi2, ghi = mid, gm
        else:
            lo2, glo = mid, gm
        if hi2 - lo2 < 1e-14 * max(1.0, abs(mid)):
            break
    return 0.5 * (lo2 + hi2)


def part_B(blocks, args):
    print("=" * 92)
    print("B. THE EXPORTED JACOBIAN  dxhat/da :  FD TRUTH vs EXACT IMPLICIT "
          "vs SURROGATE")
    print("=" * 92)
    print(f"""
   {len(blocks)} pooled ionization blocks from one real track, one shared track
   parameter c (the CVH per-track state stands in for it), one global parameter
   a = material rate scale entering BOTH channels:
       r_b(c,a) = z_b - c - (e^a - 1) m_b     (mean channel, lever arm m_b)
       K_b(theta;a) = e^a K_b(theta)          (shape channel)

   truth      : central finite difference of the profiled chat(a)
   implicit   : -(sum psi'_b)^-1 (sum m_b psi'_b - sum dphi_b)   [TRUE score]
   surrogate  : -(sum I_b)^-1 (sum m_b I_b)                      [Fisher weights,
                                                                  no shape term]
""")
    print(f"  {'blk':>4} {'legs':>7} {'nstep':>6} {'m_b [z]':>11} "
          f"{'1/I_b':>10} {'kappa2_b':>11}")
    for i, b in enumerate(blocks):
        _, _, k2 = ioni_cgf_derivs(b["steps"], 0.0)
        print(f"  {i:4d} {str(b['legs'][0])+'-'+str(b['legs'][-1]):>7} "
              f"{len(b['steps']):6d} {b['m']:11.4f} {1/b['I']:10.4f} "
              f"{float(k2[0]):11.2f}")
    print()

    rng = np.random.default_rng(20260814)
    cases = []
    # observations: each block at its own mode (the closure case), and three
    # noise realisations drawn from the exact per-block densities
    modes = [float(b["z"][int(np.argmax(b["p"]))]) for b in blocks]
    cases.append(("all at mode", np.array(modes)))
    for it in range(args.ncase):
        zs = []
        for b in blocks:
            cdf = np.concatenate([[0.0], np.cumsum(
                0.5 * (b["p"][1:] + b["p"][:-1]) * np.diff(b["z"]))])
            cdf /= cdf[-1]
            zs.append(float(np.interp(rng.random(), cdf, b["z"])))
        cases.append((f"sample {it}", np.array(zs)))

    h = args.hfd
    print(f"  rebuilding the {len(blocks)} blocks at rate scale e^(+-{h}) on "
          f"IDENTICAL tau/support grids for the finite difference ...")
    bp = rescaled_blocks(blocks, +h)
    bm = rescaled_blocks(blocks, -h)
    print(f"  {'case':>12} {'chat(0)':>10} | {'dchat/da':>12} {'implicit':>12} "
          f"{'surrogate':>12} | {'imp/truth':>10} {'surr/truth':>11}")
    print("  " + "-" * 92)
    out = []
    for name, zs in cases:
        c0 = profile_c(blocks, zs, 0.0)
        cp = profile_c(bp, zs, +h)
        cm = profile_c(bm, zs, -h)
        dtrue = (cp - cm) / (2 * h)
        rb = np.array([z - c0 for z in zs])
        dpsi = np.array([_interp(b, "dpsi", r) for b, r in zip(blocks, rb)])
        dphi = np.array([_interp(b, "dphi", r) for b, r in zip(blocks, rb)])
        mb = np.array([b["m"] for b in blocks])
        Ib = np.array([b["I"] for b in blocks])
        dimp = -(np.sum(mb * dpsi) - np.sum(dphi)) / np.sum(dpsi)
        dsur = -np.sum(mb * Ib) / np.sum(Ib)
        # the two pieces of the discrepancy, separately
        dnoshape = -np.sum(mb * dpsi) / np.sum(dpsi)     # psi' but no shape term
        out.append((name, c0, dtrue, dimp, dsur, dnoshape,
                    np.sum(dpsi), np.sum(Ib), np.sum(dphi)))
        print(f"  {name:>12} {c0:10.5f} | {dtrue:12.6f} {dimp:12.6f} "
              f"{dsur:12.6f} | {dimp/dtrue:10.5f} {dsur/dtrue:11.5f}")
    print()
    print("  decomposition of the surrogate error (same cases):")
    print(f"  {'case':>12} {'sum psip':>12} {'sum I':>12} {'psip/I':>9} "
          f"{'sum dphi':>12} | {'obs-curv only':>14} {'+shape':>10}")
    for (name, c0, dt, di, ds, dn, sp, si, sf) in out:
        print(f"  {name:>12} {sp:12.4f} {si:12.4f} {sp/si:9.4f} {sf:12.4f} | "
              f"{dn/dt:14.5f} {di/dt:10.5f}")
    print()

    # ---- the two error channels, separated and quantified ------------------
    print("  " + "=" * 88)
    print("  THE TWO ERROR CHANNELS, SEPARATED")
    print("  " + "=" * 88)
    print("""
   channel 1 (ALL global parameters): observed curvature psi'(r) replaced by
       the Fisher information I. Unbiased in expectation (E[psi'] = I), so this
       shows up as per-track SCATTER, not as a shift.
       measured as   surrogate / (implicit with the shape term switched off)
   channel 2 (only parameters that change the noise SHAPE, i.e. material /
       energy loss / the resolution parmtypes): the dropped -sum dphi term.
       measured as   (implicit, no shape) / (implicit, with shape)
""")
    r1 = np.array([ds / dn for (_, _, _, _, ds, dn, _, _, _) in out])
    r2 = np.array([dn / di for (_, _, _, di, _, dn, _, _, _) in out])
    print(f"   channel 1  surrogate/no-shape :  mean {r1.mean():.4f}  "
          f"rms {r1.std():.4f}   range [{r1.min():.4f}, {r1.max():.4f}]")
    print(f"   channel 2  no-shape/with-shape:  mean {r2.mean():.4f}  "
          f"rms {r2.std():.4f}   range [{r2.min():.4f}, {r2.max():.4f}]")
    print()

    # ---- per-block (per-module-like) lever arms ---------------------------
    print("  PER-BLOCK global parameter (what a per-module material parameter "
          "actually looks\n  like): a_b affects block b only, so")
    print("      truth_b     = -(m_b psi'_b - dphi_b) / sum_j psi'_j")
    print("      surrogate_b = -(m_b I_b)             / sum_j I_j")
    zs = cases[1][1]
    c0 = profile_c(blocks, zs, 0.0)
    rb = np.array([z - c0 for z in zs])
    dpsi = np.array([_interp(b, "dpsi", r) for b, r in zip(blocks, rb)])
    dphi = np.array([_interp(b, "dphi", r) for b, r in zip(blocks, rb)])
    mb = np.array([b["m"] for b in blocks])
    Ib = np.array([b["I"] for b in blocks])
    tb = -(mb * dpsi - dphi) / np.sum(dpsi)
    sb = -(mb * Ib) / np.sum(Ib)
    nb = -(mb * dpsi) / np.sum(dpsi)
    print(f"  {'blk':>4} {'m_b':>9} {'psi_b(r)':>10} {'psi′_b':>11} "
          f"{'I_b':>11} {'dphi_b':>11} | {'truth_b':>11} {'surr_b':>11} "
          f"{'surr/truth':>11}")
    psb = np.array([_interp(b, "psi", r) for b, r in zip(blocks, rb)])
    for i in range(len(blocks)):
        print(f"  {i:4d} {mb[i]:9.4f} {psb[i]:10.4f} {dpsi[i]:11.2f} "
              f"{Ib[i]:11.2f} {dphi[i]:11.2f} | {tb[i]:11.3e} {sb[i]:11.3e} "
              f"{sb[i]/tb[i] if tb[i] else np.nan:11.4f}")
    print(f"\n   |surr/truth - 1| over blocks: median "
          f"{np.median(np.abs(sb/tb - 1)):.3f}, max "
          f"{np.max(np.abs(sb/tb - 1)):.3f}")
    print()
    return out


# ================================================================== part C ====
def part_C(blocks, rowsA):
    print("=" * 92)
    print("C. IS THE EXPECTED (FISHER) HESSIAN POSITIVE DEFINITE? "
          "-- the marginalisation")
    print("=" * 92)
    print("""
   Per block the score w.r.t. (c, a) is  u_b = (dlnp/dc, dlnp/da) = (psi_b,
   -phi_b), so the Fisher matrix  sum_b E[u u^T]  is a GRAM MATRIX and is PSD
   by construction, for ANY density -- exactly the property that
   ResidualGlobalCorrectionMakerG4e already relies on for the resolution block
   (hess(i,j) = tr(dV_i R dV_j R), a Gram matrix in the R metric; the
   data-dependent corrections sit in the disabled `if (false)` at line 4142).
   The Schur complement of a PSD matrix with PD (1,1) block is PSD, so
   Millepede marginalisation survives. Below: the numbers on the real blocks.
""")
    mb = np.array([b["m"] for b in blocks])
    Ib = np.array([E(b, b["psi"] ** 2) for b in blocks])
    Ipa = np.array([E(b, b["psi"] * b["phi"]) for b in blocks])
    Ia = np.array([E(b, b["phi"] ** 2) for b in blocks])
    # Fisher in (c, a): d r_b/dc = -1, d r_b/da = -m_b, plus the shape channel
    # score in a is +phi.  s_b = (-psi, -m psi + phi)
    Fcc = float(np.sum(Ib))
    Fca = float(np.sum(mb * Ib - Ipa))
    Faa = float(np.sum(mb ** 2 * Ib - 2 * mb * Ipa + Ia))
    Fm = np.array([[Fcc, Fca], [Fca, Faa]])
    ev = np.linalg.eigvalsh(Fm)
    schur = Faa - Fca ** 2 / Fcc
    schur_nosh = (float(np.sum(mb ** 2 * Ib))
                  - float(np.sum(mb * Ib)) ** 2 / Fcc)
    print(f"   Fisher(c,c) = {Fcc:14.5f}")
    print(f"   Fisher(c,a) = {Fca:14.5f}")
    print(f"   Fisher(a,a) = {Faa:14.5f}")
    print(f"   eigenvalues = {ev[0]:.6e}, {ev[1]:.6e}   -> "
          f"{'PD' if ev[0] > 0 else 'NOT PD'}")
    print(f"   Schur complement (the marginalised information in a):")
    print(f"      with shape channel     {schur:14.6f}")
    print(f"      surrogate (r-channel)  {schur_nosh:14.6f}   "
          f"ratio {schur/schur_nosh if schur_nosh else np.nan:.4f}")
    print()
    print("   per-block eigenvalues of the 2x2 E[u u^T] (all must be >= 0):")
    print(f"   {'blk':>4} {'I=E[psi^2]':>12} {'E[psi phi]':>12} "
          f"{'E[phi^2]':>12} {'lam_min':>12} {'corr':>8}")
    worst = np.inf
    for i, b in enumerate(blocks):
        M = np.array([[Ib[i], -Ipa[i]], [-Ipa[i], Ia[i]]])
        e = np.linalg.eigvalsh(M)
        worst = min(worst, e[0])
        corr = Ipa[i] / np.sqrt(Ib[i] * Ia[i]) if Ib[i] * Ia[i] > 0 else np.nan
        print(f"   {i:4d} {Ib[i]:12.5f} {Ipa[i]:12.5f} {Ia[i]:12.5f} "
              f"{e[0]:12.4e} {corr:8.4f}")
    print(f"\n   smallest per-block eigenvalue over all blocks: {worst:.4e}")
    print("   (a Gram matrix cannot be indefinite; a negative value here would "
          "be a\n    quadrature failure, not a property of the model)")
    print()
    # observed Hessian, for contrast
    print("   CONTRAST -- the OBSERVED curvature psi'(r) over the blocks' own "
          "support:")
    for i, b in enumerate(blocks[:4]):
        f = b["dpsi"]
        neg = float(np.trapezoid(b["p"][f < 0], b["z"][f < 0]) / b["mass"]) \
            if np.any(f < 0) else 0.0
        print(f"   blk {i}: psi' in [{np.min(f):.3f}, {np.max(f):.3f}], "
              f"I = {Ib[i]:.4f}, mass with psi' < 0 = {neg*100:.1f} %")
    print("   -> the observed Hessian is INDEFINITE block by block; only the "
          "expectation\n      is safe to marginalise. This is the whole reason "
          "the `if (false)` is off.")
    print()

    # ---- realisation-INDEPENDENT statement of the shape-term bias ----------
    print("   EXPECTED per-block Jacobian bias (no Monte Carlo, no "
          "realisation):")
    print("      E[truth_b] / surrogate_b = 1 - E[psi phi]_b / (m_b I_b)")
    print("   because E[psi'_b] = I_b and E[dphi_b] = E[psi phi]_b (Bartlett).")
    print("   This is the part of the surrogate error that does NOT average "
          "away.")
    print(f"   {'blk':>4} {'m_b':>9} {'m_b I_b':>12} {'E[psi phi]_b':>13} "
          f"{'E[t]/surr':>11} {'surr/E[t]':>11}")
    rat = []
    for i, b in enumerate(blocks):
        den = mb[i] * Ib[i]
        rr = 1.0 - Ipa[i] / den if den else np.nan
        rat.append(rr)
        print(f"   {i:4d} {mb[i]:9.4f} {den:12.2f} {Ipa[i]:13.2f} "
              f"{rr:11.4f} {1/rr if rr else np.nan:11.4f}")
    rat = np.array(rat)
    print(f"\n   surrogate / expected-truth over the 17 blocks: "
          f"mean {np.mean(1/rat):.4f}, range "
          f"[{np.min(1/rat):.4f}, {np.max(1/rat):.4f}]")
    print("   i.e. a COHERENT overestimate on every block, not a scatter.")
    print()


# ================================================================== part D ====
def part_D(blocks, args):
    print("=" * 92)
    print("D. WHAT THE DROPPED TERM DOES TO THE GLOBAL PARAMETER "
          "(Monte Carlo)")
    print("=" * 92)
    print(f"""
   {args.ntoy} pseudo-tracks. Each block's z_b is drawn from its own EXACT
   density; chat is profiled out; the exported gradient in a is evaluated two
   ways at the converged point:
       g_true = sum_b [ -m_b psi_b(r_b) + phi_b(r_b) ]
       g_surr = sum_b [ -m_b psi_b(r_b) ]              (the surrogate route)
   A nonzero mean of either is a bias in the global parameter of
   dA = -<g>/Fisher_a.  Under the CORRECT model E[phi]=0 pointwise, so the
   whole of <g_surr> - <g_true> is the Neyman-Scott profiling effect; the
   MISSPECIFIED row perturbs the generating rate away from the fitted one.
""")
    rng = np.random.default_rng(7)
    mb = np.array([b["m"] for b in blocks])
    Ib = np.array([b["I"] for b in blocks])
    Ipa = np.array([E(b, b["psi"] * b["phi"]) for b in blocks])
    Ia = np.array([E(b, b["phi"] ** 2) for b in blocks])
    Fcc = float(np.sum(Ib))
    Fca = float(np.sum(mb * Ib - Ipa))
    Faa = float(np.sum(mb ** 2 * Ib - 2 * mb * Ipa + Ia))
    Fa = Faa - Fca ** 2 / Fcc
    def _cdfs(bls):
        out = []
        for b in bls:
            c = np.concatenate([[0.0], np.cumsum(
                0.5 * (b["p"][1:] + b["p"][:-1]) * np.diff(b["z"]))])
            out.append(c / c[-1])
        return out

    # the generating models: the fitted one, and MISSPECIFIED ones whose true
    # delta-ray rate is 1 % / 5 % higher. The misspecified samples are drawn
    # from the genuinely rescaled exact density (rescaled_blocks), not from a
    # scaled deviate -- the shape is exactly what is being probed. In every
    # case the FIT uses the a = 0 blocks.
    gens = [("correct model", blocks, 0.0)]
    for f in (0.01, 0.05):
        gens.append((f"gen rate +{100*f:.0f} %", rescaled_blocks(blocks,
                                                                np.log(1 + f)),
                     np.log(1 + f)))

    summary = {}
    for label, genbl, ag in gens:
        cdfs = _cdfs(genbl)
        shift = np.exp(ag) - 1.0
        gt, gs, gt0, gs0 = [], [], [], []
        for _ in range(args.ntoy):
            zs = np.array([float(np.interp(rng.random(), c, b["z"]))
                           + shift * b["m"] for b, c in zip(genbl, cdfs)])
            # CONTROL: the same gradient at the TRUE c (no profiling). Under
            # the correct model E[psi] = E[phi] = 0 pointwise, so both must
            # come out zero -- this is what separates a sampler/table bug from
            # the profiling (Neyman-Scott) effect below.
            psi0 = np.array([_interp(b, "psi", r) for b, r in zip(blocks, zs)])
            phi0 = np.array([_interp(b, "phi", r) for b, r in zip(blocks, zs)])
            if np.all(np.isfinite(psi0)) and np.all(np.isfinite(phi0)):
                gt0.append(float(np.sum(-mb * psi0 + phi0)))
                gs0.append(float(np.sum(-mb * psi0)))
            try:
                c0 = profile_c(blocks, zs, 0.0, ngrid=1201)
            except RuntimeError:
                continue
            rb = zs - c0
            psi = np.array([_interp(b, "psi", r) for b, r in zip(blocks, rb)])
            phi = np.array([_interp(b, "phi", r) for b, r in zip(blocks, rb)])
            if not (np.all(np.isfinite(psi)) and np.all(np.isfinite(phi))):
                continue
            gt.append(float(np.sum(-mb * psi + phi)))
            gs.append(float(np.sum(-mb * psi)))
        gt, gs = np.array(gt), np.array(gs)
        gt0, gs0 = np.array(gt0), np.array(gs0)
        n = len(gt)
        print(f"   {label:16s}  {n} usable toys")
        print(f"      CONTROL, c fixed at truth (no profiling):")
        for nm, g in (("true ", gt0), ("surr ", gs0)):
            m, e = float(np.mean(g)), float(np.std(g) / np.sqrt(len(g)))
            print(f"         <g {nm}> = {m:+10.4f} +- {e:8.4f} "
                  f"({abs(m)/e:5.1f} sigma)")
        print(f"      PROFILED over chat:")
        for nm, g in (("true ", gt), ("surr ", gs)):
            m, e = float(np.mean(g)), float(np.std(g) / np.sqrt(n))
            print(f"         <g {nm}> = {m:+10.4f} +- {e:8.4f} "
                  f"({abs(m)/e:5.1f} sigma)  var/track = {np.var(g):9.1f}"
                  f"   -> dA = {-m/Fa:+.3e}")
        d = gt - gs
        print(f"      dropped term <phi> = {np.mean(d):+10.4f} +- "
              f"{np.std(d)/np.sqrt(n):.4f}   rms {np.std(d):8.4f}   "
              f"(rms rel. to kept: {np.std(d)/np.std(gs):.3f})")
        summary[label] = (ag, float(np.mean(gt)), float(np.mean(gs)),
                          float(np.var(gt)), float(np.var(gs)))
        print()

    print("   SENSITIVITY AND THE SANDWICH. For a correct score, "
          "-d<g>/da = var(g) = Fisher.")
    print(f"   {'route':>10} {'-d<g>/da (1%)':>15} {'-d<g>/da (5%)':>15} "
          f"{'var(g)':>12} {'Fisher_a':>12}")
    a0, gt0m, gs0m, vt, vs = summary["correct model"]
    for nm, i in (("true", 1), ("surrogate", 2)):
        s1 = -(summary["gen rate +1 %"][i] - summary["correct model"][i]) \
            / summary["gen rate +1 %"][0]
        s5 = -(summary["gen rate +5 %"][i] - summary["correct model"][i]) \
            / summary["gen rate +5 %"][0]
        v = vt if i == 1 else vs
        print(f"   {nm:>10} {s1:15.1f} {s5:15.1f} {v:12.1f} "
              f"{Fa if i == 1 else np.nan:12.1f}")
    print()
    print(f"   Fisher_a (marginalised, with shape channel) = {Fa:.4f} per "
          f"track; sigma(A) per track = {1/np.sqrt(Fa):.4f}")
    print()


# ==================================================================== main ====
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="/ceph/submit/data/user/d/david_w/ZMass/"
                                       "cvh/cleanprop/model/model_mu_pt3_eta0.30.root")
    ap.add_argument("--plane", type=int, default=None)
    ap.add_argument("--func", default="qop")
    ap.add_argument("--parts", default="A,B,C,D")
    ap.add_argument("--nt", type=int, default=1 << 17)
    ap.add_argument("--npad", type=int, default=32)
    ap.add_argument("--minsteps", type=int, default=8)
    ap.add_argument("--ncase", type=int, default=3)
    ap.add_argument("--hfd", type=float, default=1e-3)
    ap.add_argument("--ntoy", type=int, default=400)
    args = ap.parse_args()

    legs = cpt.load_model(args.model)
    k = args.plane if args.plane is not None else len(legs) - 1
    avec = FUNCTIONALS[args.func]
    var, _, _ = model_variance(legs, k, avec)
    sigma = float(np.sqrt(var))
    print(f"model {os.path.basename(args.model)}  plane {k}  func {args.func}  "
          f"sigma = {sigma:.6e}\n")

    parts = set(args.parts.split(","))
    rowsA = None
    if "A" in parts:
        rowsA = part_A(legs, k, avec, sigma, args)
    blocks = None
    if parts & {"B", "C", "D"}:
        blocks = build_multiblock(legs, k, avec, sigma, minsteps=args.minsteps)
        print(f"  built {len(blocks)} pooled blocks\n")
    if "B" in parts:
        part_B(blocks, args)
    if "C" in parts:
        part_C(blocks, rowsA)
    if "D" in parts:
        part_D(blocks, args)


if __name__ == "__main__":
    main()
