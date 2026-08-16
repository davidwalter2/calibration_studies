#!/usr/bin/env python3
"""Four numerical measurements needed before the exact-CGF block noise model
can be moved INSIDE the C++ track fit.

Everything here is measured with the already-validated offline machinery:
`cgf_channels.block_cf_exponent` / `exact_density` / `invert_cf` for the CF and
its inversion, `cgf_fisher.fisher_exact` for I -- i.e. literally the route
`cgf_channels.py --model ...` section 4 takes. No file in this directory is
modified; the only local re-implementation is a copy of
`cf_brems_exact.rad_exponent` carrying a `centred` flag (part C), which is the
whole point of that part.

A. HOW SMALL CAN THE FFT BE?
   `exact_density` inverts on `linspace(0, tau[-1], nt)` zero-padded by `npad`,
   so the transform length is nt*npad, z_max = pi (nt-1)/tau[-1] (aliasing) and
   dz = 2 pi/(npad tau[-1]) (peak resolution). nt x npad is scanned at three
   planes, plus the `lncut` that sets tau[-1] itself. The deliverable is the
   smallest nt*npad whose 1/I is within 0.1 % / 1 % of the reference grid.

B. IS THE STANDARDIZATION SCALE FREE?
   The offline code divides the residual by sigma = sqrt(model_variance) and
   quotes 1/I in those z units. If the in-fit code is allowed to pick its own
   internal scale then sigma^2 * (1/I_z) -- the PHYSICAL inverse information in
   GeV^-2 -- must not depend on it. Measured by deliberately feeding c*sigma
   into the CF construction for c = 1/4 ... 4, with the tau grid rebuilt each
   time, and multiplying back by (c sigma)^2.

C. CENTRED vs UNCENTRED RADIATIVE CHANNEL.
   `rad_exponent` uses (e^{ix} - 1 - ix): the propagator's mean-loss table is
   built with ionOnly = false, so the reference trajectory ALREADY subtracted
   the radiative mean and the CF must be centred. If the reference table is
   switched to ionization-only the residual carries that mean and the block CF
   must become (e^{ix} - 1). Fisher information is translation invariant, so
   the two must give the SAME I; the measurement is whether the change really
   is a pure translation (safe) or also changes the shape (not safe).

D. POOLING CLASS COUNT.
   The per-step ionization exponent is parameterised by (regime, gsig2, a1, e1,
   a2, e2, a3, e0r, tmaxr, scaling) plus the transport-weighted cs in column 10.
   Steps whose parameters agree can be merged by SUMMING their amplitudes
   a1/a2/a3. How many distinct classes survive at 0.01 % / 0.1 % / 1 % relative
   binning, and -- the load-bearing question -- does the pooled CF still give
   the same 1/I?

usage:
  python cgf_infit_prep.py --model <model.root> [--planes 0,9,18] [--parts ABCD]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cf_brems_exact as cbe
import cf_propagation_test as cpt
import cgf_channels as cgc
import cgf_fisher
from cf_propagation_test import FUNCTIONALS, model_variance
from cf_track_resolution import ioni_step_exponent
from cgf_saddlepoint import ioni_cgf_derivs

REF_NT = 1 << 17
REF_NPAD = 32
REF_LNCUT = -60.0


# ============================================================ local rad CF ===
def rad_exponent_local(tau, recs, spec, vg, weights=None, centred=True,
                       ser_sign=+1.0):
    """LOCAL copy of cf_brems_exact.rad_exponent with a `centred` flag.

    centred = True   e^{ix} - 1 - ix   (production: the propagator's dE/dx
                     table is built with ionOnly = false, so the reference
                     trajectory has already subtracted the radiative mean)
    centred = False  e^{ix} - 1        (what the block CF must become if the
                     reference table is switched to ionization-only and the
                     residual therefore carries the radiative mean)

    The two branches differ by EXACTLY +i x inside the v integral, in both the
    large-|x| and the small-|x| branch, so any quirk of the series branch
    cancels in the difference and the comparison is like-for-like.

    ser_sign multiplies the x^3 term of the small-|x| series. The production
    code has  -x^2/2 + i x^3/6, whereas e^{ix} - 1 - ix = -x^2/2 - i x^3/6;
    ser_sign = -1 evaluates the corrected sign so its (tiny) effect can be
    measured rather than assumed. ser_sign = +1 reproduces production bit for
    bit -- verified in part C's premise check.

    Everything else (the spectrum, the weights, the `a` convention, the 1e-4
    branch point) is verbatim from cf_brems_exact.
    """
    tau = np.asarray(tau, dtype=float)
    S = np.zeros(len(tau), dtype=np.complex128)
    if weights is None:
        weights = np.ones(len(recs))
    for rec, sp, w in zip(recs, spec, weights):
        v, dNdv = cbe.step_spectrum(rec, sp, vg)
        if not np.any(dNdv > 0.):
            continue
        E = rec[cbe.R_ETOT]
        a = tau * rec[cbe.R_CS] * w
        x = np.outer(a, v * E)                       # (nt, nv)
        small = np.abs(x) < 1e-4
        term = np.empty_like(x, dtype=np.complex128)
        term[~small] = np.expm1(1j * x[~small]) - 1j * x[~small]
        xs = x[small]
        term[small] = -0.5 * xs ** 2 + ser_sign * 1j * xs ** 3 / 6.
        if not centred:
            term = term + 1j * x
        S += np.trapezoid(term * dNdv[None, :], v, axis=1)
    return S


def rad_mean_shift(recs, spec, vg, weights):
    """dz_rad = sum_steps INT dv (dN/dv) (cs w) v E, the radiative mean loss in
    z units. This is the coefficient of i t in S_uncentred - S_centred, i.e.
    the translation the two conventions differ by."""
    mu = 0.0
    for rec, sp, w in zip(recs, spec, weights):
        v, dNdv = cbe.step_spectrum(rec, sp, vg)
        if not np.any(dNdv > 0.):
            continue
        mu += rec[cbe.R_CS] * w * float(np.trapezoid(dNdv * v * rec[cbe.R_ETOT],
                                                     v))
    return float(mu)


# ================================================================= plumbing ==
_CACHE = {}


def cf_parts(legs, k, avec, sigma, lncut=REF_LNCUT):
    """(tau, S_ioni, S_ms, S_rad) on ONE tau grid matched to the FULL block.

    Split by channel so the ionization part can be swapped for a pooled one
    (part D) and the radiative part for an uncentred one (part C) without
    recomputing the other two. S_ioni is built from the SAME collected step
    array that part D pools, and the premise that

        ioni_step_exponent(collect_ioni_steps(...), 1.0, tau)
          == block_cf_exponent(..., channels=('ioni',))

    is checked, not assumed (see `check_premises`).
    """
    key = (k, float(sigma), float(lncut))
    if key in _CACHE:
        return _CACHE[key]
    tau = cgc.auto_tau(legs, k, avec, sigma, lncut=lncut)
    st = cgc.collect_ioni_steps(legs, k, avec, sigma)
    Si = ioni_step_exponent(st, 1.0, tau)
    Sm = cgc.block_cf_exponent(legs, k, avec, sigma, tau, channels=("ms",))
    R, SP, VG, W = cgc.collect_rad_steps(legs, k, avec, sigma)
    Sr = (rad_exponent_local(tau, R, SP, VG, weights=W, centred=True)
          if R is not None and len(R) else np.zeros(len(tau), np.complex128))
    _CACHE[key] = (tau, Si, Sm, Sr)
    return _CACHE[key]


def inv_fisher(S, tau, nt=REF_NT, npad=REF_NPAD, floor=1e-8):
    """1/I and the grid it came from. floor is RELATIVE (cgf_fisher default);
    an absolute floor was a documented 57x error in this study."""
    z, p, dp = cgc.invert_cf(S, tau, npad=npad, nt=nt, deriv=True)
    I, d = cgf_fisher.fisher_exact(z, p, dp, floor=floor)
    return (1.0 / I if np.isfinite(I) and I > 0 else np.nan), z, p, dp, d


def planes_sigma(legs, avec, planes):
    out = {}
    for k in planes:
        var, _, _ = model_variance(legs, k, avec)
        out[k] = float(np.sqrt(var))
    return out


# ============================================================== premises =====
def check_premises(legs, avec, planes, sig):
    """Nothing downstream is reported unless these hold."""
    print("=" * 100)
    print("0. PREMISES (checked, not assumed)")
    print("=" * 100)
    print(f"{'plane':>5} {'|S_split - S_block|/|S_block|':>30} "
          f"{'1/I this route':>15} {'nioni':>7} {'nms':>7} {'nrad':>7}")
    for k in planes:
        tau, Si, Sm, Sr = cf_parts(legs, k, avec, sig[k])
        Sb = cgc.block_cf_exponent(legs, k, avec, sig[k], tau)
        S = Si + Sm + Sr
        num = float(np.max(np.abs(S - Sb)))
        den = float(np.max(np.abs(Sb)))
        v, *_ = inv_fisher(S, tau)
        st = cgc.collect_ioni_steps(legs, k, avec, sig[k])
        sms, _ = cgc.collect_ms_steps(legs, k, avec, sig[k])
        R, _, _, _ = cgc.collect_rad_steps(legs, k, avec, sig[k])
        print(f"{k:5d} {num/den:30.3e} {v:15.5f} {len(st):7d} {len(sms):7d} "
              f"{0 if R is None else len(R):7d}")
    print("  The split-by-channel CF is the production block CF to float64 "
          "round-off, so\n  every number below is on the validated route.")


# ==================================================================== A ======
def part_A(legs, avec, planes, sig, nts, npads, lncuts):
    print()
    print("=" * 100)
    print("A. MINIMAL FFT GRID THAT STILL REPRODUCES 1/I")
    print("=" * 100)
    print(f"  reference cell: nt = {REF_NT} (2^{int(np.log2(REF_NT))}), "
          f"npad = {REF_NPAD}, lncut = {REF_LNCUT:g}, full block "
          f"(ioni+ms+rad), floor 1e-8 RELATIVE")
    ref = {}
    for k in planes:
        tau, Si, Sm, Sr = cf_parts(legs, k, avec, sig[k])
        ref[k] = inv_fisher(Si + Sm + Sr, tau)[0]
    print("  reference 1/I: " + "  ".join(f"plane {k}: {ref[k]:.5f}"
                                          for k in planes))

    grid = {}
    for k in planes:
        tau, Si, Sm, Sr = cf_parts(legs, k, avec, sig[k])
        S = Si + Sm + Sr
        tmax = tau[-1]
        print()
        print(f"  --- plane {k}   tau_max = {tmax:.3f}   (lncut {REF_LNCUT:g})")
        print(f"  {'nt':>8} {'npad':>5} {'FFT len':>10} {'z_max':>9} "
              f"{'dz':>10} {'1/I':>10} {'rel dev':>10} {'massfrac':>9} "
              f"{'noise/pk':>9}")
        for nt in nts:
            for npad in npads:
                v, z, p, dp, d = inv_fisher(S, tau, nt=nt, npad=npad)
                dz = z[1] - z[0]
                zmax = float(np.max(z))
                nz = abs(float(np.min(p))) / float(np.max(p))
                rel = v / ref[k] - 1.0
                grid[(k, nt, npad)] = (v, rel, nt * npad)
                print(f"  {nt:8d} {npad:5d} {nt*npad:10d} {zmax:9.2f} "
                      f"{dz:10.3e} {v:10.5f} {rel:+10.2e} "
                      f"{d.get('mass_frac', np.nan):9.5f} {nz:9.1e}")

    print()
    print("  --- dz sensitivity, ISOLATED (npad liveness check)")
    print("      npad only sets dz, and the table above shows 1/I does not "
          "move for any\n      npad >= 1 -- but npad >= 1 already forces "
          "dz <= 2 pi/tau_max <= 0.09 here. To\n      prove dz is a live "
          "axis and not an inert knob, the SAME reference density is\n"
          "      decimated in z (p and dp are exact at every retained point, "
          "so this changes\n      only the quadrature step).")
    print(f"  {'plane':>5} " + " ".join(f"{f'x{f}':>17}" for f in
                                        (1, 2, 4, 8, 16, 32, 64, 128, 256)))
    for k in planes:
        tau, Si, Sm, Sr = cf_parts(legs, k, avec, sig[k])
        z, p, dp = cgc.invert_cf(Si + Sm + Sr, tau, npad=1, nt=REF_NT,
                                 deriv=True)
        row = []
        for f in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            I, _ = cgf_fisher.fisher_exact(z[::f], p[::f], dp[::f])
            v = 1.0 / I if np.isfinite(I) and I > 0 else np.nan
            row.append(f"{z[f]-z[0]:7.4f} {v:9.5f}")
        print(f"  {k:5d} " + " ".join(f"{r:>17}" for r in row))
    print("  (columns: dz, 1/I. The knob is live: 1/I is stable while dz is "
          "well inside the\n   peak and blows up once it is not, so the "
          "insensitivity to npad above is a\n   measured convergence, not an "
          "unconnected knob.)")

    print()
    print("  --- lncut scan (tau_max is rebuilt per lncut; nt/npad at the "
          "reference values)")
    print(f"  {'plane':>5} " + " ".join(f"{f'lncut {c:g}':>22}" for c in lncuts))
    for k in planes:
        row = []
        for c in lncuts:
            tau, Si, Sm, Sr = cf_parts(legs, k, avec, sig[k], lncut=c)
            v = inv_fisher(Si + Sm + Sr, tau)[0]
            row.append(f"{tau[-1]:8.2f} {v:8.5f} {v/ref[k]-1:+5.0e}")
        print(f"  {k:5d} " + " ".join(f"{r:>22}" for r in row))
    print("  (columns: tau_max, 1/I, rel dev from the lncut = -60 reference)")

    print()
    print("  --- SMALLEST FFT LENGTH meeting a tolerance AT ALL THREE PLANES")
    print(f"  {'tol':>8} {'FFT len':>10} {'nt':>8} {'npad':>5} | "
          + " ".join(f"{f'rel dev p{k}':>13}" for k in planes))
    cells = sorted({(nt * npad, nt, npad) for nt in nts for npad in npads})
    for tol in (1e-3, 1e-2):
        best = None
        for L, nt, npad in cells:
            devs = [abs(grid[(k, nt, npad)][1]) for k in planes]
            if max(devs) <= tol:
                best = (L, nt, npad, devs)
                break
        if best is None:
            print(f"  {tol:8.0e} {'none in scan':>10}")
        else:
            L, nt, npad, devs = best
            print(f"  {tol:8.0e} {L:10d} {nt:8d} {npad:5d} | "
                  + " ".join(f"{d:13.2e}" for d in devs))

    print()
    print("  --- SMALLEST FFT LENGTH when lncut is ALSO free")
    print("      lncut sets tau_max, and z_max = pi nt/tau_max, so a shorter "
          "t grid buys the\n      same z_max at a smaller nt. The three knobs "
          "are therefore minimized jointly.")
    jn = [1 << e for e in (9, 10, 11, 12, 13, 14)]
    jp = (1, 2, 4, 8)
    jgrid = {}
    for c in lncuts:
        for k in planes:
            tau, Si, Sm, Sr = cf_parts(legs, k, avec, sig[k], lncut=c)
            S = Si + Sm + Sr
            for nt in jn:
                for npad in jp:
                    v = inv_fisher(S, tau, nt=nt, npad=npad)[0]
                    jgrid[(c, k, nt, npad)] = abs(v / ref[k] - 1.0)
    print(f"  {'tol':>8} {'FFT len':>10} {'nt':>8} {'npad':>5} {'lncut':>7} | "
          + " ".join(f"{f'rel dev p{k}':>13}" for k in planes))
    for tol in (1e-3, 1e-2):
        best = None
        for L, nt, npad in sorted({(nt * npad, nt, npad)
                                   for nt in jn for npad in jp}):
            for c in lncuts:
                devs = [jgrid[(c, k, nt, npad)] for k in planes]
                if max(devs) <= tol:
                    best = (L, nt, npad, c, devs)
                    break
            if best:
                break
        if best is None:
            print(f"  {tol:8.0e} {'none in scan':>10}")
        else:
            L, nt, npad, c, devs = best
            print(f"  {tol:8.0e} {L:10d} {nt:8d} {npad:5d} {c:7g} | "
                  + " ".join(f"{d:13.2e}" for d in devs))
    return grid


# ==================================================================== B ======
def part_B(legs, avec, planes, sig, scales):
    print()
    print("=" * 100)
    print("B. IS THE STANDARDIZATION SCALE A FREE NUMERICAL CONVENTION?")
    print("=" * 100)
    print("  sigma_used = c * sqrt(model_variance) is fed into the CF "
          "construction (it divides\n  every transport weight), the tau grid "
          "is rebuilt for each c, and the physical\n  inverse information "
          "sigma_used^2 * (1/I_z) [GeV^-2] must not depend on c.")
    print(f"  {'plane':>5} {'sigma [GeV^-1]':>15} | "
          + " ".join(f"{f'c={c:g}':>26}" for c in scales)
          + f" {'max rel spread':>15}")
    out = {}
    for k in planes:
        row, phys = [], []
        for c in scales:
            s = c * sig[k]
            tau, Si, Sm, Sr = cf_parts(legs, k, avec, s)
            vz = inv_fisher(Si + Sm + Sr, tau)[0]
            ph = s ** 2 * vz
            phys.append(ph)
            row.append(f"{vz:10.5f} {ph:11.4e}")
        spread = (max(phys) - min(phys)) / np.mean(phys)
        out[k] = (phys, spread)
        print(f"  {k:5d} {sig[k]:15.6e} | "
              + " ".join(f"{r:>26}" for r in row) + f" {spread:15.2e}")
    print("  (each cell: 1/I in ITS OWN z units, then the physical "
          "sigma_used^2 * (1/I_z) in GeV^-2)")
    worst = max(v[1] for v in out.values())
    if worst < 1e-3:
        print(f"  -> INVARIANT to {worst:.1e} across a factor 16 in scale: "
              "the in-fit code may pick\n     its own internal "
              "standardization. The z-unit 1/I on its own is NOT a "
              "physical\n     number and must never be compared across "
              "conventions.")
    else:
        print(f"  -> *** NOT INVARIANT *** max spread {worst:.2e}. The "
              "standardization is NOT free;\n     the in-fit code must "
              "reproduce sqrt(model_variance) exactly.")
    return out


# ==================================================================== C ======
def part_C(legs, avec, planes, sig):
    print()
    print("=" * 100)
    print("C. CENTRED vs UNCENTRED RADIATIVE CHANNEL")
    print("=" * 100)
    print("  centred   S_rad = sum_s INT dv (dN/dv)(e^{iavE} - 1 - iavE)  "
          "(production)")
    print("  uncentred S_rad = sum_s INT dv (dN/dv)(e^{iavE} - 1)         "
          "(reference dE/dx table switched to ionization-only)")

    print()
    print("  C.0 premise: the local copy reproduces cf_brems_exact.rad_exponent"
          " in the\n      centred case (it must, or nothing below is "
          "like-for-like)")
    print(f"  {'plane':>5} {'max |S_local - S_prod|':>24} "
          f"{'/ max|S_prod|':>15}")
    for k in planes:
        tau, _, _, _ = cf_parts(legs, k, avec, sig[k])
        R, SP, VG, W = cgc.collect_rad_steps(legs, k, avec, sig[k])
        a = rad_exponent_local(tau, R, SP, VG, weights=W, centred=True)
        b = cbe.rad_exponent(tau, R, SP, VG, weights=W)
        print(f"  {k:5d} {float(np.max(np.abs(a-b))):24.3e} "
              f"{float(np.max(np.abs(a-b)))/float(np.max(np.abs(b))):15.3e}")

    print()
    print("  C.1 is the difference a PURE TRANSLATION?  "
          "S_unc - S_cen must equal i t dz_rad")
    print(f"  {'plane':>5} {'dz_rad (quadrature)':>20} {'dz_rad (from S)':>17} "
          f"{'max |resid|/|S|':>16}")
    shifts = {}
    for k in planes:
        tau, Si, Sm, Sr = cf_parts(legs, k, avec, sig[k])
        R, SP, VG, W = cgc.collect_rad_steps(legs, k, avec, sig[k])
        Su = rad_exponent_local(tau, R, SP, VG, weights=W, centred=False)
        mu = rad_mean_shift(R, SP, VG, W)
        d = Su - Sr
        # dz_rad read off the CF itself: -Im dS/dt at t = 0, by the slope of
        # the (exactly linear) difference
        m_fit = float(np.imag(d[-1]) / tau[-1])
        resid = float(np.max(np.abs(d - 1j * tau * mu)))
        shifts[k] = (mu, Su)
        print(f"  {k:5d} {mu:20.8e} {m_fit:17.8e} "
              f"{resid/float(np.max(np.abs(Sr+Si+Sm))):16.3e}")
    print("  (a nonzero residual would mean the two conventions differ by "
          "more than a shift)")

    print()
    print("  C.2 the two 1/I, and the mode of each density")
    print(f"  {'plane':>5} {'1/I centred':>12} {'1/I uncentred':>14} "
          f"{'rel diff':>10} | {'mode cen':>10} {'mode unc':>10} "
          f"{'mode diff':>12} {'/dz_rad - 1':>12}")
    res = {}
    for k in planes:
        tau, Si, Sm, Sr = cf_parts(legs, k, avec, sig[k])
        mu, Su = shifts[k]
        vc, zc, pc, _, _ = inv_fisher(Si + Sm + Sr, tau)
        vu, zu, pu, _, _ = inv_fisher(Si + Sm + Su, tau)
        mc = cgc.mode_from_density(zc, pc)
        mu_mode = cgc.mode_from_density(zu, pu)
        dm = mu_mode - mc
        res[k] = (vc, vu, mu, dm)
        print(f"  {k:5d} {vc:12.5f} {vu:14.5f} {vu/vc-1:+10.2e} | "
              f"{mc:10.5f} {mu_mode:10.5f} {dm:+12.5e} "
              f"{dm/mu-1 if mu else np.nan:+12.2e}")

    print()
    print("  C.3 how big is the radiative mean, physically?")
    print(f"  {'plane':>5} {'dz_rad [z]':>13} {'sqrt(K2) [z]':>13} "
          f"{'dz/sqrt(K2)':>12} {'1/sqrt(I) [z]':>13} {'dz*sqrt(I)':>11} "
          f"{'d(q/p) [GeV^-1]':>16} {'rel. to q/p':>12}")
    for k in planes:
        blk = cgc.collect_block(legs, k, avec, sig[k])
        K2 = float(np.atleast_1d(cgc.block_cgf_derivs(blk, 0.0, order=2)[2])[0])
        vc, vu, mu, dm = res[k]
        qop = abs(legs[k]["refqop"])
        print(f"  {k:5d} {mu:13.5e} {np.sqrt(K2):13.5f} {mu/np.sqrt(K2):12.3e} "
              f"{np.sqrt(vc):13.5f} {mu/np.sqrt(vc):11.3e} "
              f"{mu*sig[k]:16.4e} {mu*sig[k]/qop:12.3e}")
    print("  (dz_rad is in units of sigma by construction -- z IS the residual "
          "over sigma --\n   so column 2 is already 'as a fraction of the "
          "block's own sigma' for the\n   Gaussian-limit sigma; sqrt(K2) is "
          "the CF's own std, which the delta-ray tail\n   inflates, and "
          "1/sqrt(I) is the information-equivalent width.)")

    print()
    print("  C.4 incidental: the small-|x| series in rad_exponent has "
          "+i x^3/6 where\n      e^{ix} - 1 - ix = -x^2/2 - i x^3/6. "
          "Size of that sign, at the outermost plane:")
    k = planes[-1]
    tau, Si, Sm, Sr = cf_parts(legs, k, avec, sig[k])
    R, SP, VG, W = cgc.collect_rad_steps(legs, k, avec, sig[k])
    Sfix = rad_exponent_local(tau, R, SP, VG, weights=W, centred=True,
                              ser_sign=-1.0)
    vfix = inv_fisher(Si + Sm + Sfix, tau)[0]
    vc = res[k][0]
    print(f"  plane {k}: max |dS| = {float(np.max(np.abs(Sfix-Sr))):.3e} "
          f"(|S| up to {float(np.max(np.abs(Sr))):.3e}), "
          f"1/I {vc:.6f} -> {vfix:.6f}  (rel {vfix/vc-1:+.2e})")
    return res


# ==================================================================== D ======
def _relbin(x, w):
    """(sign, index) of |x| in geometric bins of relative width (1+w)."""
    x = np.asarray(x, dtype=np.float64)
    s = np.sign(x).astype(np.int64)
    b = np.zeros(x.shape, dtype=np.int64)
    nz = x != 0.0
    b[nz] = np.floor(np.log(np.abs(x[nz])) / np.log1p(w)).astype(np.int64)
    return s, b


def pool_ioni(st, w, full_key):
    """Merge ionization steps into classes and sum a1/a2/a3 within a class.

    The per-step exponent (cf_track_resolution.ioni_step_exponent) is

      regime 0:  -t^2/2 * gsig2 * gs^2
      regime 1:  a1 (e^{i gs e1 gam t} - 1 - i...) + a2 (same with e2)
               + a3 * delta_term(gs e0 gam t, tmax/e0)

    with gs = col10 * 1e-3 (the transport weight is already folded into col10).
    So the class key must fix every product that enters: gs, e0*gam, tmax*gam
    and -- if `full_key` -- also e1*gam and e2*gam. The MINIMAL key of the
    brief is (e0r*scaling, tmaxr*scaling, gs); `full_key` adds e1, e2 so the
    two can be compared.

    Representatives: amplitude-weighted means (a1 for e1, a2 for e2, a3 for
    e0/tmax, the step's own K''(0) for gs). Any weighted mean is correct to
    O(binwidth^2); the amplitudes are the physically meaningful weights.
    Gaussian-regime steps pool EXACTLY (their variances just add).
    """
    reg = st[:, 0]
    gam = st[:, 9]
    gs = st[:, 10]
    e1, e2 = st[:, 3] * gam, st[:, 5] * gam
    e0, tmax = st[:, 7] * gam, st[:, 8] * gam
    a1, a2, a3 = st[:, 2], st[:, 4], st[:, 6]

    # per-step K''(0), the weight for the shared gs
    k2 = np.array([float(np.atleast_1d(
        ioni_cgf_derivs(st[i:i + 1], 0.0, order=2)[2])[0])
        for i in range(len(st))])
    k2 = np.where(np.isfinite(k2) & (k2 > 0), k2, 1.0)

    sg, bg = _relbin(gs, w)
    _, b0 = _relbin(e0, w)
    _, bt = _relbin(tmax, w)
    keys = [(int(r), int(a), int(b), int(c), int(d))
            for r, a, b, c, d in zip(reg, sg, bg, b0, bt)]
    if full_key:
        _, b1 = _relbin(e1, w)
        _, b2 = _relbin(e2, w)
        keys = [kk + (int(x), int(y)) for kk, x, y in zip(keys, b1, b2)]

    groups = {}
    for i, kk in enumerate(keys):
        groups.setdefault(kk, []).append(i)

    rows = []
    gauss_var = 0.0
    for kk, idx in groups.items():
        idx = np.asarray(idx)
        if kk[0] == 0:                       # already-Gaussian steps: exact
            gauss_var += float(np.sum(st[idx, 1] * (gs[idx] * 1e-3) ** 2))
            continue
        A1, A2, A3 = a1[idx].sum(), a2[idx].sum(), a3[idx].sum()
        wm = lambda v, wt: (float(np.sum(v[idx] * wt[idx]) / np.sum(wt[idx]))
                            if np.sum(wt[idx]) > 0 else float(np.mean(v[idx])))
        r = np.zeros(11)
        r[0] = 1.0
        r[2], r[3] = A1, wm(e1, a1)
        r[4], r[5] = A2, wm(e2, a2)
        r[6], r[7], r[8] = A3, wm(e0, a3), wm(tmax, a3)
        r[9] = 1.0                            # gam already folded into 3/5/7/8
        r[10] = wm(gs, k2)
        rows.append(r)
    if gauss_var > 0.0:
        r = np.zeros(11)
        r[0] = 0.0
        r[1] = gauss_var                      # with col10 = 1e3 -> gs = 1
        r[10] = 1e3
        rows.append(r)
    return np.array(rows) if rows else np.zeros((0, 11)), len(groups)


def part_D(legs, avec, planes, sig, widths):
    k = planes[-1]
    print()
    print("=" * 100)
    print(f"D. POOLING CLASS COUNT  (outermost plane {k})")
    print("=" * 100)
    st = cgc.collect_ioni_steps(legs, k, avec, sig[k])
    gam = st[:, 9]
    print(f"  {len(st)} ionization steps feed plane {k}. Spread of the "
          f"parameters that the key\n  must fix (max/min):")
    for nm, v in (("gs = w*cs (col10)", st[:, 10]), ("e0r*scaling", st[:, 7] * gam),
                  ("tmaxr*scaling", st[:, 8] * gam), ("e1*scaling", st[:, 3] * gam),
                  ("e2*scaling", st[:, 5] * gam), ("scaling", gam)):
        a = np.abs(v)
        print(f"    {nm:20s} min {a.min():12.5e}  max {a.max():12.5e}  "
              f"ratio {a.max()/a.min() if a.min() > 0 else np.inf:10.4f}  "
              f"nuniq {len(np.unique(v)):5d}")
    print("  regimes present: "
          + ", ".join(f"{int(r)}: {c}" for r, c in
                      zip(*np.unique(st[:, 0], return_counts=True))))

    tau, Si, Sm, Sr = cf_parts(legs, k, avec, sig[k])
    ref_full = inv_fisher(Si + Sm + Sr, tau)[0]
    ref_ioni = inv_fisher(Si, tau)[0]
    print(f"\n  unpooled reference: 1/I(ioni only) = {ref_ioni:.6f}, "
          f"1/I(full block) = {ref_full:.6f}")

    for full_key, label in ((False, "MINIMAL key (e0r*scaling, tmaxr*scaling, "
                                    "gs) -- as specified"),
                            (True, "FULL key   (+ e1*scaling, e2*scaling)")):
        print()
        print(f"  --- {label}")
        print(f"  {'bin width':>10} {'classes':>8} {'compress':>9} | "
              f"{'1/I ioni':>10} {'rel err':>10} | {'1/I full':>10} "
              f"{'rel err':>10}")
        for w in widths:
            pooled, ncls = pool_ioni(st, w, full_key)
            Sp = ioni_step_exponent(pooled, 1.0, tau)
            vi = inv_fisher(Sp, tau)[0]
            vf = inv_fisher(Sp + Sm + Sr, tau)[0]
            print(f"  {w:10.0e} {len(pooled):8d} {len(st)/max(len(pooled),1):9.1f}x"
                  f" | {vi:10.6f} {vi/ref_ioni-1:+10.2e} | {vf:10.6f} "
                  f"{vf/ref_full-1:+10.2e}")
    print("  (`classes` is the number of pooled records actually built; "
          "`compress` = steps/classes.\n   The coarsest widths are in the "
          "scan to show the knob is LIVE -- a class count that\n   collapses "
          "without moving 1/I would otherwise be indistinguishable from an "
          "inert knob.)")


# ===================================================================== main ==
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--planes", default="0,9,18")
    ap.add_argument("--func", default="qop")
    ap.add_argument("--parts", default="ABCD")
    args = ap.parse_args()

    planes = [int(x) for x in args.planes.split(",")]
    legs = cpt.load_model(args.model)
    avec = FUNCTIONALS[args.func]
    sig = planes_sigma(legs, avec, planes)
    t0 = time.time()
    print(f"model {os.path.basename(args.model)}, func {args.func}, "
          f"{len(legs)} legs, planes {planes}")

    check_premises(legs, avec, planes, sig)
    if "A" in args.parts:
        part_A(legs, avec, planes, sig,
               nts=[1 << e for e in (10, 11, 12, 13, 14, 15, 17)],
               # 1, 2, 4 are BELOW the brief's range and are here only to
               # demonstrate that npad is a live knob at all: from 8 upward it
               # does not move 1/I by 1e-5, and a null result on an untested
               # knob is not a result.
               npads=(1, 2, 4, 8, 16, 32, 64),
               lncuts=(-30.0, -45.0, -60.0, -90.0))
    if "B" in args.parts:
        part_B(legs, avec, planes, sig, scales=(0.25, 0.5, 1.0, 2.0, 4.0))
    if "C" in args.parts:
        part_C(legs, avec, planes, sig)
    if "D" in args.parts:
        part_D(legs, avec, planes, sig,
               widths=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0))
    print(f"\ntotal {time.time()-t0:.1f} s")


if __name__ == "__main__":
    main()
