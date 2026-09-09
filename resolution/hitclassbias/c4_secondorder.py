#!/usr/bin/env python3
"""THE SECOND-ORDER (Box) BIAS OF THE CONVERGED GAUSS-NEWTON ESTIMATOR,
computed from the exported per-track quantities with NO free parameter, and
compared against the measured charge-even shift.

THE ARGUMENT.
For chi2(theta) = r(theta)^T W r(theta) with r = m - f(theta), a Gauss-Newton
step from a point displaced by `d` from the minimum lands exactly at the
minimum IF f is linear in theta. It does not if f is curved, and the leftover
is second order in `d`:

    d2 = a d1 + (K/2) d1^2 + O(d1^3)

with `K` the effective curvature of the model along the q/p direction (units
GeV, since d is in GeV^-1). BOTH `d1` and `d2` are exported per track:

    d1 = refParms_iter0[0] - trackParms[0]      (seed -> first GN point)
    d2 = refParms[0]       - refParms_iter0[0]  (the rest of the correction)

The SAME curvature acts on the estimator's OWN statistical fluctuation, whose
scale is sigma = sqrt(refCov[0]) rather than the seed offset. The sign is NOT
the same, and the derivation is worth writing out because getting it wrong
flips the answer. With S = sum_i (m_i - f_i)^2, B = sum f'^2, C = sum f' f'',
stationarity of S at theta-hat = theta* + delta gives

    0 = sum f' eps - delta B + delta sum f'' eps - (3/2) delta^2 C + ...
    delta_1 = sum f' eps / B,
    delta_2 = [delta_1 sum f'' eps - (3/2) delta_1^2 C] / B,
    E[delta_1 sum f'' eps] = sigma^2 C / B,   E[delta_1^2] = sigma^2 / B,
    E[delta_2] = (sigma^2 C/B - (3/2) sigma^2 C/B)/B = -(sigma_theta^2/2)(C/B),

while the NOISELESS two-step gives d2 = +(C/2B) d1^2, i.e. K = C/B. So

    <Delta(q/p)> = -(K/2) sigma^2      ->   <z> = -(K/2) sigma,

with the MINUS sign -- this is Box's (1971) formula, and the two-step
regression measures exactly the C/B it needs. No free parameter.

WHY THE LINEAR TERM MUST BE IN THE FIT. The seed is the generalTracks Kalman
fit, which shares its hits with CVH, so d1 is correlated with the measurement
noise that also sets d2 -- this is the conditioning trap of sec. 3 of the
2026-09-08 note (corr(seed->final SIGNED, x) = +0.113). That contamination is
LINEAR in d1 to leading order, so it is absorbed by `a` and the QUADRATIC
coefficient is what carries the curvature. This is a mitigation, not a proof:
a d1^2-correlated noise term would still leak.

THE CHARGE STRUCTURE, which is the sharp part.
Under the mirror map (reflection through a plane containing the beam axis) a
charge-+ track maps to a charge-- track and q/p -> -(q/p), so
Delta(q/p) -> -Delta(q/p): ANY estimator effect that respects the mirror gives
a charge-ODD <z>, i.e. a momentum SCALE bias, and ZERO charge-even part. The
charge-EVEN <z> is an additive, charge-independent curvature offset -- a
sagitta-like bias -- and it can only be sourced by something CHIRAL: the
module layout (tilted BPix ladders, the FPix turbine, stereo angles) or a
hit-level location bias with a fixed azimuthal sense (Lorentz drift). So the
second-order mechanism predicts the bias in the charge-ODD channel first, and
its charge-EVEN part only at the level of the layout chirality. K is therefore
fitted SEPARATELY per charge and both combinations are formed.
"""
import argparse

import numpy as np

import conv_common as cc


def fit_quad(d1, d2, w=None):
    """Weighted LS of d2 = b0 + a d1 + (K/2) d1^2; returns (b0, a, K) + errors."""
    n = len(d1)
    if n < 200:
        return None
    X = np.stack([np.ones(n), d1, d1 ** 2], axis=1)
    if w is None:
        w = np.ones(n)
    XtW = X.T * w
    A = XtW @ X
    b = XtW @ d2
    try:
        Ai = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return None
    p = Ai @ b
    resid = d2 - X @ p
    s2 = float((w * resid ** 2).sum() / max(n - 3, 1))
    cov = Ai * s2
    return p[0], p[1], 2 * p[2], np.sqrt(np.diag(cov)) * np.array([1, 1, 2])


def run(V, rng, ntrim=0.99):
    print(f"\n{'='*78}\n=== SECOND-ORDER GN BIAS -- {V.name}\n{'='*78}")
    d1 = (V.d["qop_it0"] - V.d["qop_seed"]).astype(np.float64)
    d2 = (V.d["qop_ref"] - V.d["qop_it0"]).astype(np.float64)
    sig = V.sigma
    g = V.good & np.isfinite(d1) & np.isfinite(d2)
    # trim the extreme 1 % in |d1| so a handful of pathological seeds cannot
    # drive a quadratic coefficient
    t1 = np.percentile(np.abs(d1[g]), 100 * ntrim)
    t2 = np.percentile(np.abs(d2[g]), 100 * ntrim)
    g &= (np.abs(d1) < t1) & (np.abs(d2) < t2)
    print(f"tracks used {int(g.sum())} of {int(V.good.sum())} "
          f"(|d1| < {t1:.3e}, |d2| < {t2:.3e} GeV^-1)")
    print(f"scales:  rms(d1) {d1[g].std():.4e}   rms(d2) {d2[g].std():.4e}   "
          f"<sigma> {sig[g].mean():.4e}   rms(d1)/<sigma> "
          f"{d1[g].std()/sig[g].mean():.2f}")

    print("\n--- d2 = b0 + a d1 + (K/2) d1^2, fitted per charge ---")
    out = {}
    for lab, qs in (("q = +1", +1), ("q = -1", -1)):
        m = g & (V.q == qs)
        r = fit_quad(d1[m], d2[m])
        if r is None:
            print(f"  {lab}: too few tracks")
            continue
        b0, a, K, e = r
        s = sig[m].mean()
        pred = -0.5 * K * s     # Box: <z> = -(K/2) sigma
        # error on the prediction from the K error alone
        epred = 0.5 * e[2] * s
        out[qs] = (K, e[2], pred, epred, s)
        print(f"  {lab}  n={int(m.sum()):7d}   b0 {b0:+.3e}   a {a:+.5f}+-{e[1]:.5f}"
              f"   K {K:+.4e}+-{e[2]:.4e} GeV")
        print(f"          -> predicted <z> = -(K/2) sigma = "
              f"{pred*1e3:+8.3f} +- {epred*1e3:5.3f} e-3   "
              f"(<sigma> {s:.3e})")
    if len(out) == 2:
        kp, ekp, pp, epp, _ = out[+1]
        km, ekm, pm, epm, _ = out[-1]
        ev = 0.5 * (pp + pm)
        eev = 0.5 * np.hypot(epp, epm)
        od = 0.5 * (pp - pm)
        eod = eev
        print(f"\n  PREDICTED charge-EVEN <z> = {ev*1e3:+8.3f} +- {eev*1e3:5.3f} e-3")
        print(f"  PREDICTED charge-ODD  <z> = {od*1e3:+8.3f} +- {eod*1e3:5.3f} e-3")
        print(f"  (K_+ {kp:+.4e}, K_- {km:+.4e}; the mirror argument says "
              f"K_- = -K_+,")
        print(f"   i.e. a PURE charge-odd prediction, and K_+ + K_- = "
              f"{kp+km:+.3e} +- {np.hypot(ekp,ekm):.3e} measures the chiral part)")

    # <z> = -<K sigma>/2, NOT -<K><sigma>/2: K and sigma are correlated across
    # tracks (K runs 1.5e+01 in the barrel to 4e+01 in the middle band). K is
    # therefore refitted in quintiles of sigma and the prediction assembled
    # bin by bin.
    print("\n--- the same prediction with K refitted in quintiles of sigma ---")
    print(f"  {'charge':8s}{'sigma bin':22s}{'<sigma>':>11s}{'K':>13s}"
          f"{'-(K/2)<sigma>':>15s}{'weight':>9s}")
    predq = {}
    for qs in (+1, -1):
        mq = g & (V.q == qs)
        qe = np.percentile(sig[mq], [0, 20, 40, 60, 80, 100])
        qe[-1] += 1e-12
        tot, wsum = 0., 0.
        for i in range(5):
            mm = mq & (sig >= qe[i]) & (sig < qe[i + 1])
            r = fit_quad(d1[mm], d2[mm])
            if r is None:
                continue
            sb = sig[mm].mean()
            pb = -0.5 * r[2] * sb
            w = int(mm.sum())
            tot += pb * w
            wsum += w
            print(f"  {qs:+8d}{f'{qe[i]:.3e}-{qe[i+1]:.3e}':22s}{sb:11.3e}"
                  f"{r[2]:+13.4e}{pb*1e3:+15.3f}{w:9d}")
        predq[qs] = tot / max(wsum, 1)
        print(f"  {'':8s}{'-> combined':22s}{'':11s}{'':13s}"
              f"{predq[qs]*1e3:+15.3f}")
    if len(predq) == 2:
        print(f"  sigma-binned PREDICTED charge-EVEN <z> = "
              f"{0.5*(predq[1]+predq[-1])*1e3:+8.3f} e-3")
        print(f"  sigma-binned PREDICTED charge-ODD  <z> = "
              f"{0.5*(predq[1]-predq[-1])*1e3:+8.3f} e-3")

    print("\n--- the MEASURED moments on the same tracks, for comparison ---")
    for nm, fn in (("even", cc.even_mean), ("odd", cc.odd_mean)):
        v, e, n = fn(V.x, V.q, g, rng)
        vz, ez, _ = fn(V.z, V.q, g, rng)
        print(f"  MEASURED charge-{nm.upper():5s} <x> = {v*1e3:+8.3f}+-{e*1e3:5.3f}"
              f" e-3    (<z> = {vz*1e3:+8.3f}+-{ez*1e3:5.3f} e-3)")

    print("\n--- per eta band: K, the prediction, and the measurement ---")
    print(f"  {'band':10s}{'K_+':>13s}{'K_-':>13s}{'pred even':>13s}"
          f"{'pred odd':>12s}{'meas even':>16s}{'meas odd':>16s}")
    for ib, nm in enumerate(cc.BANDS_SHORT):
        mb = g & (V.band == ib)
        pr = {}
        for qs in (+1, -1):
            r = fit_quad(d1[mb & (V.q == qs)], d2[mb & (V.q == qs)])
            pr[qs] = (r[2], -0.5 * r[2] * sig[mb & (V.q == qs)].mean()) if r else (np.nan, np.nan)
        pe = 0.5 * (pr[+1][1] + pr[-1][1])
        po = 0.5 * (pr[+1][1] - pr[-1][1])
        me, mee, _ = cc.even_mean(V.x, V.q, mb, rng)
        mo, moe, _ = cc.odd_mean(V.x, V.q, mb, rng)
        print(f"  {nm:10s}{pr[+1][0]:+13.3e}{pr[-1][0]:+13.3e}"
              f"{pe*1e3:+13.2f}{po*1e3:+12.2f}"
              f"{me*1e3:+11.2f}+-{mee*1e3:4.2f}{mo*1e3:+11.2f}+-{moe*1e3:4.2f}")

    print("\n--- the same, inside the OUT (bulk) and IN populations ---")
    t = np.percentile(V.dq_seed[V.good], 90.)
    for lab, cut in (("OUT (bulk 90%)", V.dq_seed <= t), ("IN  (top 10%)", V.dq_seed > t)):
        mb = g & cut
        pr = {}
        for qs in (+1, -1):
            r = fit_quad(d1[mb & (V.q == qs)], d2[mb & (V.q == qs)])
            pr[qs] = -0.5 * r[2] * sig[mb & (V.q == qs)].mean() if r else np.nan
        me, mee, n = cc.even_mean(V.x, V.q, mb, rng)
        mo, moe, _ = cc.odd_mean(V.x, V.q, mb, rng)
        print(f"  {lab:16s} n={n:7d}  pred even {0.5*(pr[1]+pr[-1])*1e3:+7.2f}"
              f"  pred odd {0.5*(pr[1]-pr[-1])*1e3:+7.2f}"
              f"   meas even {me*1e3:+7.2f}+-{mee*1e3:4.2f}"
              f"   meas odd {mo*1e3:+7.2f}+-{moe*1e3:4.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    a = ap.parse_args()
    rng = np.random.default_rng(404)
    for p in a.npz:
        run(cc.Var(p), rng)


if __name__ == "__main__":
    main()
