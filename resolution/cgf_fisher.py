#!/usr/bin/env python3
"""Fisher information of an ionization block, two-sided, closed form vs exact.

WHY IT IS THE QUANTITY OF INTEREST. The CVH fit weights each block by an
inverse VARIANCE, which for a 1/E^2 delta-ray spectrum is tail-dominated and
only exists because the spectrum is truncated at alpha = 0.999. The
statistically correct weight is the INFORMATION

    I = int psi(r)^2 p(r) dr,     psi = -dln p/dr

which is finite without any truncation, and gives sigma2_eff = 1/I for a
Fisher-scoring surrogate that is positive by construction -- including on the
38 % of residuals where the observed (Newton) curvature is negative.

TWO ROUTES ARE COMPUTED AND COMPARED.

 A. CLOSED FORM (saddlepoint), parameterized by theta so that no root finding
    is needed: r = K'(theta), dr = K'' dtheta, p = e^{K - theta K'}/sqrt(2 pi K''),
    psi = theta + K'''/(2 K''^2), so

        I = int psi^2 e^{K - theta K'} sqrt(K''/2 pi) dtheta.

    theta now runs over BOTH signs. It could not before: _delta_derivs clipped
    e^{bw} and returned a wrong finite number, which made one half-line look
    divergent (NOTES 2026-08-13 XVII/XVIII). The CGF is entire -- all jumps are
    bounded -- so there is no divergent side, only an overflow bound, and
    K''' / K'''' are now analytic rather than finite-differenced.

 B. EXACT, by inverting the ionization-only characteristic function. p AND p'
    are both obtained by inversion (p' from the -i t phi transform), so no
    finite differencing of a log density enters, and the transform is done with
    an FFT, which is EXACT for the periodized density: the only error is
    aliasing from |z| > z_max, kept at ~1e-7 of the peak.

    I = int (p')^2 / p dz over {p > floor * max p}, floor RELATIVE (an absolute
    floor was the source of a 57x error, NOTES 2026-08-13 XV).

usage:
  python cgf_fisher.py --model M.root [--planes 0,9,18] [--func qop]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cf_propagation_test import load_model, model_variance, FUNCTIONALS
from cf_track_resolution import ioni_step_exponent
from cgf_phase0_validate import collect_ioni_steps
from cgf_saddlepoint import (ioni_cgf_derivs, theta_grid, heavy_tail_sign,
                             theta_overflow_limit)


# ----------------------------------------------------------------- route A
def fisher_spa(steps, n=4000, two_sided=True, lo=1e-8):
    """(I, diagnostics) from the saddlepoint density along the theta curve."""
    th = theta_grid(steps, n=n, lo=lo, two_sided=two_sided)
    K, K1, K2, K3, K4 = ioni_cgf_derivs(steps, th, order=4)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ok = (np.isfinite(K) & np.isfinite(K1) & np.isfinite(K2)
              & np.isfinite(K3) & (K2 > 0))
        lp = np.where(ok, K - th * K1 - 0.5 * np.log(2.0 * np.pi * K2), -np.inf)
        # K3/K2**2 must be taken ONE K2 at a time: K2 -> 0 at the bounded hard
        # edge (K2**2 underflows to 0 -> inf) and K2 ~ 1e300 far out in the
        # tail (K2**2 overflows to inf -> 0 or nan). Neither is a real value.
        psi = np.where(ok, th + 0.5 * (K3 / K2) / K2, 0.0)
        # p dr = e^{lp} K'' dtheta
        meas = np.where(ok, np.exp(lp) * K2, 0.0)
    # DOMAIN CUT at the turn-up of ln p_SPA on the bounded side. p_SPA =
    # e^{K-theta r}/sqrt(2 pi K''), and K'' -> 0 at the hard edge, so the
    # prefactor -0.5 ln(2 pi K'') diverges and p_SPA eventually RISES again.
    # That is a genuine property of the saddlepoint density, not a numerical
    # artefact, and past the turn-up psi^2 * meas is pure garbage (denormal
    # measure times a diverging score). Everything beyond it is dropped.
    sgn = heavy_tail_sign(steps)
    bnd = (sgn * th > 0) & ok
    if bnd.any():
        idx = np.where(bnd)[0]
        j = idx[int(np.argmin(lp[idx]))]
        beyond = (sgn * th > sgn * th[j])
        ok = ok & ~beyond
        meas = np.where(beyond, 0.0, meas)
    meas = np.where(np.isfinite(meas), meas, 0.0)
    # psi^2 overflows far out in the tail, where meas has underflowed to 0 and
    # the product is 0*inf = nan. Evaluate only on the support, and assert that
    # nothing non-finite survives THERE -- silently nan-ing the integrand is
    # exactly the class of bug this module exists to remove.
    sel = meas > 0.0
    integ = np.zeros_like(meas)
    with np.errstate(over="ignore", invalid="ignore"):
        integ[sel] = psi[sel] ** 2 * meas[sel]
    nbad = int(np.sum(~np.isfinite(integ[sel])))
    if nbad:
        raise FloatingPointError(
            f"{nbad} non-finite integrand values on the support")
    Z = float(np.trapezoid(meas, th))
    I = float(np.trapezoid(integ, th))
    rr = np.where(ok, K1, np.nan)
    m = meas > 0
    # normalize by the saddlepoint mass, exactly as the exact route divides by
    # the mass on its support -- otherwise the two are not comparable
    return (I / Z if Z > 0 else np.nan), dict(Z=Z, I_raw=I,
                   rmin=float(np.nanmin(rr[m])) if m.any() else np.nan,
                   rmax=float(np.nanmax(rr[m])) if m.any() else np.nan,
                   nth=int(ok.sum()), ntot=len(th),
                   thmin=float(th[m].min()) if m.any() else np.nan,
                   thmax=float(th[m].max()) if m.any() else np.nan)


# ----------------------------------------------------------------- route B
def _phi_cut(steps, t, lncut=-60.0):
    """phi(t) on a uniform grid, evaluated only where it is not negligible.

    |phi| = e^{Re S}; the contribution of t > t_cut to p(z) is bounded by
    (1/pi) int |phi| dt, so a cut at Re S = -60 (|phi| = 9e-27) is far below
    the 1e-12 relative floors used later, while Re S = -745 (true underflow)
    would cost 5x more exponent evaluations. Returns (phi, n_eval).
    """
    n = len(t)
    # bisect on the first index whose exponent has underflowed
    lo, hi = 0, n - 1
    S_hi = ioni_step_exponent(steps, 1.0, np.array([t[hi]]))[0]
    if S_hi.real > lncut:
        ne = n
    else:
        while hi - lo > 1:
            mid = (lo + hi) // 2
            s = ioni_step_exponent(steps, 1.0, np.array([t[mid]]))[0]
            if s.real > lncut:
                lo = mid
            else:
                hi = mid
        ne = hi + 1
    phi = np.zeros(n, dtype=np.complex128)
    # chunk so the (nsteps, ntau) temporaries stay small
    cs = 4000
    for a in range(0, ne, cs):
        b = min(a + cs, ne)
        phi[a:b] = np.exp(ioni_step_exponent(steps, 1.0, t[a:b]))
    return phi, ne


def exact_density_fft(steps, taumax=2000.0, M=1 << 19, tcut=None, pre=None):
    """(z, p, dp/dz) for the ionization-only block, by FFT inversion.

    p(z) = (1/pi) Re int_0^inf phi(t) e^{-i t z} dt is discretized on t_j = j dt
    with dt = taumax/M; the sum is then a DFT and z_k = 2 pi fftfreq(M, dt), so
    dz = 2 pi / taumax and z_max = pi / dt. The derivative comes from the same
    transform with phi -> -i t phi, which avoids differencing p numerically.

    pre = (t, phi, n_eval) reuses an already-computed phi (the t_cut scan only
    zeroes it, it does not change it).
    """
    if pre is None:
        dt = taumax / M
        t = np.arange(M) * dt
        phi, ne = _phi_cut(steps, t)
    else:
        t, phi, ne = pre
        dt = t[1] - t[0]
        M = len(t)
    if tcut is not None:
        phi = np.where(t <= tcut, phi, 0.0)
    wgt = np.full(M, dt)
    wgt[0] = 0.5 * dt                        # trapezoid, phi(0) = 1
    c = phi * wgt
    p = np.fft.fft(c).real / np.pi
    dp = np.fft.fft(c * (-1j * t)).real / np.pi
    z = 2.0 * np.pi * np.fft.fftfreq(M, d=dt)
    o = np.argsort(z)
    return z[o], p[o], dp[o], ne


def phi_uniform(steps, taumax=2000.0, M=1 << 19):
    """(t, phi, n_eval) on the uniform FFT grid, cached by the caller."""
    dt = taumax / M
    t = np.arange(M) * dt
    phi, ne = _phi_cut(steps, t)
    return t, phi, ne


def _support(p, floor):
    """Largest CONTIGUOUS run with p > floor * max(p), containing the mode.

    Contiguity matters: np.trapezoid over a masked, non-contiguous z array
    silently stitches the gaps and integrates over intervals that are not in
    the support at all (it produced a 7 % outlier at plane 9, floor 1e-6).
    """
    pm = np.nanmax(p)
    i0 = int(np.nanargmax(p))
    thr = floor * pm
    lo = i0
    while lo > 0 and p[lo - 1] > thr:
        lo -= 1
    hi = i0
    while hi < len(p) - 1 and p[hi + 1] > thr:
        hi += 1
    return slice(lo, hi + 1)


def fisher_exact(z, p, dp, floor=1e-8):
    """I = int (p')^2/p dz on the support, normalized by the mass there.

    RELATIVE floor and the contiguous support it selects ARE the matched grid:
    an absolute floor on an oversized grid is what produced the 57x error of
    NOTES XV.
    """
    s = _support(p, floor)
    if s.stop - s.start < 10:
        return np.nan, dict(mass=np.nan)
    I = float(np.trapezoid(dp[s] ** 2 / p[s], z[s]))
    mass = float(np.trapezoid(p[s], z[s]))
    tot = float(np.trapezoid(np.maximum(p, 0.0), z))
    return I / mass, dict(mass=mass, mass_frac=mass / tot,
                          zlo=float(z[s][0]), zhi=float(z[s][-1]),
                          zmode=float(z[int(np.nanargmax(p))]))


def fisher_exact_fd(z, p, floor=1e-8):
    """The NOTES XV/XVII recipe: I from central differences of ln p, kept as a
    like-for-like cross-check of the reference values."""
    s = _support(p, floor)
    h = z[1] - z[0]
    lp = np.log(np.maximum(p, 1e-300))
    psi = np.full_like(z, np.nan)
    psi[1:-1] = -(lp[2:] - lp[:-2]) / (2 * h)
    ss = slice(max(s.start, 1), min(s.stop, len(z) - 1))
    I = float(np.trapezoid(psi[ss] ** 2 * p[ss], z[ss]))
    mass = float(np.trapezoid(p[ss], z[ss]))
    return I / mass


def _selftest(verbose=True):
    """End-to-end check of the FFT route on a case with a known answer.

    A unit Gaussian has phi(t) = e^{-t^2/2}, p' = -z p and I = 1 exactly, so
    this tests the transform, the weights, the derivative transform, the
    support selection and the normalization together. A pure-imaginary shift
    checks that a non-symmetric phi is handled with the right sign convention.
    """
    taumax, M = 200.0, 1 << 16
    dt = taumax / M
    t = np.arange(M) * dt
    z = 2.0 * np.pi * np.fft.fftfreq(M, d=dt)
    o = np.argsort(z)
    out = []
    for mu, sg in ((0.0, 1.0), (0.7, 0.35)):
        phi = np.exp(1j * mu * t - 0.5 * sg ** 2 * t ** 2)
        wgt = np.full(M, dt)
        wgt[0] = 0.5 * dt
        c = phi * wgt
        p = (np.fft.fft(c).real / np.pi)[o]
        dp = (np.fft.fft(c * (-1j * t)).real / np.pi)[o]
        zz = z[o]
        I, d = fisher_exact(zz, p, dp, floor=1e-10)
        pex = np.exp(-0.5 * ((zz - mu) / sg) ** 2) / (sg * np.sqrt(2 * np.pi))
        emax = float(np.max(np.abs(p - pex)))
        out.append((mu, sg, I, I * sg ** 2, emax, d["mass_frac"]))
        if verbose:
            print(f"  N(mu={mu}, sigma={sg}): I = {I:.8f}, I*sigma^2 = "
                  f"{I*sg**2:.8f} (exact 1), max|p - p_exact| = {emax:.2e}, "
                  f"mass {d['mass_frac']:.6f}")
    for mu, sg, I, Is2, emax, mf in out:
        assert abs(Is2 - 1.0) < 1e-6, f"I*sigma^2 = {Is2}"
        assert emax < 1e-9, f"density error {emax}"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--planes", default="0,9,18")
    ap.add_argument("--func", default="qop")
    ap.add_argument("--taumax", type=float, default=2000.0)
    ap.add_argument("--logm", type=int, default=19)
    ap.add_argument("--nth", type=int, default=4000)
    args = ap.parse_args()

    legs = load_model(args.model)
    avec = FUNCTIONALS[args.func]
    planes = [int(x) for x in args.planes.split(",")]
    M = 1 << args.logm

    allsteps = {}
    for k in planes:
        var, _, _ = model_variance(legs, k, avec)
        sig = float(np.sqrt(var))
        allsteps[k] = (collect_ioni_steps(legs, k, avec, sig), sig)

    print("selftest of the FFT inversion + Fisher integral:")
    _selftest()
    print()
    print(f"model {os.path.basename(args.model)}, func {args.func}, "
          f"{len(legs)} planes")
    print(f"FFT inversion: taumax {args.taumax:.0f}, M = 2^{args.logm}, "
          f"dz = {2*np.pi/args.taumax:.2e}, z_max = {np.pi*M/args.taumax:.0f}")
    print()

    print("=" * 104)
    print("1. TWO-SIDED closed form vs one-sided, against the exact inversion")
    print("=" * 104)
    # the values quoted as the reference in NOTES 2026-08-13 XVII/XVIII
    NOTES_REF = {0: 0.0628, 9: 0.7696, 18: 1.8731}
    print(f"{'plane':>5} {'kappa2':>9} | {'1/I 1-sided':>11} {'1/I 2-sided':>11} "
          f"{'1/I exact':>10} {'2s/exact':>9} {'1s/exact':>9} | "
          f"{'NOTES XVIII':>11} {'2s/notes':>9}")
    print("-" * 104)
    cache = {}
    pcache = {}
    for k in planes:
        steps, sig = allsteps[k]
        _, _, k2 = ioni_cgf_derivs(steps, 0.0)
        kap2 = float(np.atleast_1d(k2)[0])
        I1, d1 = fisher_spa(steps, n=args.nth, two_sided=False)
        I2, d2 = fisher_spa(steps, n=args.nth, two_sided=True)
        pcache[k] = phi_uniform(steps, args.taumax, M)
        z, p, dp, ne = exact_density_fft(steps, args.taumax, M, pre=pcache[k])
        cache[k] = (z, p, dp, ne)
        Ie, de = fisher_exact(z, p, dp, floor=1e-8)
        nr = NOTES_REF.get(k)
        print(f"{k:5d} {kap2:9.2f} | {1/I1:11.4f} {1/I2:11.4f} {1/Ie:10.4f} "
              f"{Ie/I2:9.3f} {Ie/I1:9.3f} | " +
              (f"{nr:11.4f} {nr*I2:9.3f}" if nr else " " * 21))
    print("\n  (ratio > 1 means the closed form UNDERestimates I, i.e. "
          "overestimates 1/I)")

    # the Gaussian selftest checks the transform; this checks it on the ACTUAL
    # phi, against a direct trapezoid quadrature at a handful of z
    print("\n  FFT vs direct quadrature of the same phi (relative):")
    for k in planes:
        t, phi, ne = pcache[k]
        z, p, dp, _ = cache[k]
        dtt = t[1] - t[0]
        wgt = np.full(len(t), dtt)
        wgt[0] = 0.5 * dtt
        c = phi * wgt
        errs = []
        for zq in (-3.0, -0.5, 0.0, 1.0, 4.0):
            i = int(np.argmin(np.abs(z - zq)))
            direct = float(np.sum((c * np.exp(-1j * t * z[i])).real) / np.pi)
            errs.append(abs(direct - p[i]) / max(abs(p[i]), 1e-300))
        print(f"    plane {k:2d}: max rel diff {max(errs):.2e}")

    print()
    print("=" * 104)
    print("2. Exact inversion: is it converged? RELATIVE floor scan, and the "
          "two things\n   'tau' conflates -- integration RANGE vs t-grid "
          "RESOLUTION.")
    print("=" * 104)
    print(f"{'plane':>5} | " + " ".join(f"{f'floor 1e-{e}':>11}"
                                        for e in (4, 6, 8, 10, 12)) +
          f" | {'mass frac':>9} {'I by dlnp FD':>13}")
    print("-" * 104)
    for k in planes:
        z, p, dp, ne = cache[k]
        row = []
        for e in (4, 6, 8, 10, 12):
            Ie, de = fisher_exact(z, p, dp, floor=10.0 ** (-e))
            row.append(1.0 / Ie)
        _, de = fisher_exact(z, p, dp, floor=1e-8)
        Ifd = fisher_exact_fd(z, p, floor=1e-8)
        print(f"{k:5d} | " + " ".join(f"{v:11.4f}" for v in row) +
              f" | {de['mass_frac']:9.5f} {1/Ifd:13.4f}")

    print()
    print(f"{'plane':>5} | integration range t_cut (dt and dz FIXED)")
    print(f"{'':>5} | " + " ".join(f"{f't<{c:g}':>11}"
                                   for c in (10, 30, 60, 120, 600, 2000)) +
          f" {'n_eval':>8}")
    print("-" * 104)
    for k in planes:
        steps, sig = allsteps[k]
        z, p, dp, ne = cache[k]
        row = []
        for c in (10, 30, 60, 120, 600, 2000):
            zz, pp, dpp, _ = exact_density_fft(steps, args.taumax, M, tcut=c,
                                               pre=pcache[k])
            Ie, _ = fisher_exact(zz, pp, dpp, floor=1e-8)
            row.append(1.0 / Ie)
        print(f"{k:5d} | " + " ".join(f"{v:11.4f}" for v in row) +
              f" {ne:8d}")
    print("  n_eval = number of t samples before |phi| underflows; beyond it "
          "the integrand is exactly 0.")

    print()
    print(f"{'plane':>5} | z-grid resolution dz = 2 pi / taumax "
          f"(z_max held at ~{np.pi*M/args.taumax:.0f})")
    print(f"{'':>5} | " + " ".join(f"{f'tau={T:g}':>11}"
                                   for T in (150, 600, 2000, 6000)))
    print("-" * 104)
    for k in planes:
        steps, sig = allsteps[k]
        row = []
        for T in (150.0, 600.0, 2000.0, 6000.0):
            MM = int(2 ** np.round(np.log2(M * T / args.taumax)))
            zz, pp, dpp, _ = exact_density_fft(steps, T, MM)
            Ie, _ = fisher_exact(zz, pp, dpp, floor=1e-8)
            row.append(1.0 / Ie)
        print(f"{k:5d} | " + " ".join(f"{v:11.4f}" for v in row))

    print()
    print(f"{'plane':>5} | EMULATION of the NOTES XV/XVII recipe: ntau = "
          f"20000 uniform on [0, taumax],")
    print(f"{'':>5} |   i.e. dt = taumax/20000 GROWS with taumax. The trapezoid"
          f" sum over a uniform t grid")
    print(f"{'':>5} |   is exactly the periodized density with period 2 pi/dt, "
          f"so a coarse dt folds")
    print(f"{'':>5} |   p(z +- 2 pi/dt) back into the range.")
    print(f"{'':>5} | " + " ".join(f"{f'tau={T:g}':>11}"
                                   for T in (40, 150, 600, 2000)) +
          f"   {'| NOTES XVII quoted':>20}")
    print("-" * 104)
    quoted = {0: (0.0604, 0.0622, 0.0623, 0.0628),
              9: (0.7219, 0.7226, 0.7314, 0.7696),
              18: (1.6599, 1.6626, 1.7093, 1.8731)}
    for k in planes:
        steps, sig = allsteps[k]
        row = []
        for T in (40.0, 150.0, 600.0, 2000.0):
            zz, pp, dpp, _ = exact_density_fft(steps, T, 20000)
            Ie, _ = fisher_exact(zz, pp, dpp, floor=1e-8)
            row.append(1.0 / Ie)
        q = quoted.get(k)
        print(f"{k:5d} | " + " ".join(f"{v:11.4f}" for v in row) +
              ("   | " + " ".join(f"{v:8.4f}" for v in q) if q else ""))
    print("  If the emulated row drifts like the quoted one, the '9 % tau "
          "drift' of NOTES XVII was\n  the t-grid, not the integration range: "
          "the converged values are the ones above.")

    print()
    print("=" * 104)
    print("3. Closed-form diagnostics (two-sided)")
    print("=" * 104)
    print(f"{'plane':>5} {'sgn':>4} {'th_overflow':>12} {'th used':>22} "
          f"{'r used':>26} {'Z (SPA norm)':>13}")
    print("-" * 104)
    for k in planes:
        steps, sig = allsteps[k]
        I2, d2 = fisher_spa(steps, n=args.nth, two_sided=True)
        print(f"{k:5d} {heavy_tail_sign(steps):+4.0f} "
              f"{theta_overflow_limit(steps):12.4e} "
              f"[{d2['thmin']:+.2e},{d2['thmax']:+.2e}] "
              f"[{d2['rmin']:+.4g},{d2['rmax']:+.4g}]".rjust(26) +
              f" {d2['Z']:13.5f}")
    print("  Z = int p_SPA dr. It is NOT 1: the saddlepoint density is "
          "unnormalized, and\n  I is quoted after dividing by it. What is left "
          "of the 2-sided/exact gap tracks\n  (Z - 1) and the number of pooled "
          "steps, i.e. it is the saddlepoint's own\n  O(1/n) error, not the "
          "theta coverage.")

    print()
    print(f"{'plane':>5} | theta-quadrature convergence of the two-sided 1/I")
    print(f"{'':>5} | " + " ".join(f"{c:>12}" for c in
                                   ("n=1000", "n=4000", "n=16000",
                                    "lo=1e-10", "lo=1e-6")))
    print("-" * 104)
    for k in planes:
        steps, sig = allsteps[k]
        row = []
        for n, lo in ((1000, 1e-8), (4000, 1e-8), (16000, 1e-8),
                      (4000, 1e-10), (4000, 1e-6)):
            I, _ = fisher_spa(steps, n=n, two_sided=True, lo=lo)
            row.append(1.0 / I)
        print(f"{k:5d} | " + " ".join(f"{v:12.4f}" for v in row))


if __name__ == "__main__":
    main()
