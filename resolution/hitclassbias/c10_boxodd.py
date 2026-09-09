#!/usr/bin/env python3
"""THE SECOND-ORDER (Box) BIAS IN THE CHARGE-ODD CHANNEL: is it the residual
the CF ionisation+radiation model does not account for, and what does it do at
Z momenta?

WHY THE CHARGE-ODD CHANNEL IS THE ONE THAT MATTERS. Under the mirror map a `+`
track maps to a `-` track and `q/p -> -(q/p)`, so a charge-ODD pull shift is a
charge-INDEPENDENT momentum shift -- a SCALE bias. In a dimuon mass it ADDS
(both legs shift the same way, `dm/m = dp/p`), unlike the charge-EVEN sagitta
bias, which is opposite on the two legs and cancels to first order. So this is
the one track-level estimator effect that reaches m_Z.

THE PREDICTION, with no free parameter. From `c4_secondorder.py`: with
`d1 = refParms_iter0[0] - trackParms[0]` and `d2 = refParms[0] -
refParms_iter0[0]`, the noiseless two-step gives `d2 = (K/2) d1^2` with
`K = C/B = sum f' f'' / sum f'^2`, and Box's bias of the converged estimator is

    <Delta(q/p)> = -(K/2) sigma^2   ->   <z> = -(K/2) sigma .

K is NOT a single number -- it runs from +15 (barrel) to +40 (middle) GeV and
falls with sigma -- so it is refitted here on a (eta band x sigma quintile)
grid and the prediction is assembled PER TRACK,

    b_i = -(K(band_i, sigma-bin_i) / 2) * sigma_i ,

which is charge-ODD by construction (`K_- = -K_+` to 2 %, the mirror map).
That per-track prediction is then (a) compared with the measured charge-odd
moment, (b) regressed against it (slope 1 = validated), and (c) evaluated at Z
kinematics.

THE LEVER ARM. The 20-60 GeV gun alone cannot validate an O(sigma) prediction:
its sigma_rel spans only 1.24 in gen pT. The LOW-pT gun
(`mugun_lowpt_260903x_m0`, the same geometry and field) has a completely
different sigma and a different K, so running the identical prediction there
is a genuine out-of-sample test.
"""
import argparse

import numpy as np

import conv_common as cc

U = 0.05                       # the oddmoment study's probe
CM = (1. / (1. + 2 * U)) ** 1.5   # a mean shift m enters the probe as CM * m


def fit_quad(d1, d2):
    n = len(d1)
    if n < 200:
        return None
    X = np.stack([np.ones(n), d1, d1 ** 2], axis=1)
    A = X.T @ X
    p = np.linalg.solve(A, X.T @ d2)
    r = d2 - X @ p
    s2 = float(r @ r) / max(n - 3, 1)
    cov = np.linalg.inv(A) * s2
    return 2 * p[2], 2 * np.sqrt(cov[2, 2])


def per_track_box(V, nsig=5):
    """b_i = -(K/2) sigma_i with K refit on a (band x sigma quintile) grid."""
    d1 = (V.d["qop_it0"] - V.d["qop_seed"]).astype(np.float64)
    d2 = (V.d["qop_ref"] - V.d["qop_it0"]).astype(np.float64)
    g = V.good & np.isfinite(d1) & np.isfinite(d2)
    g &= (np.abs(d1) < np.percentile(np.abs(d1[g]), 99))
    g &= (np.abs(d2) < np.percentile(np.abs(d2[g]), 99))
    b = np.full(len(V), np.nan)
    Ktab = {}
    for ib in range(3):
        for qq in (+1, -1):
            m0 = g & (V.band == ib) & (V.q == qq)
            if m0.sum() < 1000:
                continue
            qs = np.percentile(V.sigma[m0], np.linspace(0, 100, nsig + 1))
            qs[-1] += 1e-12
            for i in range(nsig):
                m = m0 & (V.sigma >= qs[i]) & (V.sigma < qs[i + 1])
                r = fit_quad(d1[m], d2[m])
                if r is None:
                    continue
                Ktab[(ib, qq, i)] = r
                b[m] = -0.5 * r[0] * V.sigma[m]
    return b, g, Ktab


def moments(x, q, m, probe=None):
    """charge-odd mean (and its error) of x, optionally through the probe."""
    y = x if probe is None else x * np.exp(-probe * x ** 2)
    v, e = [], []
    for sg in (+1, -1):
        s = m & (q == sg)
        v.append(y[s].mean())
        e.append(y[s].std() / np.sqrt(s.sum()))
    return 0.5 * (v[0] - v[1]), 0.5 * float(np.hypot(*e))


KFIT = []


def run(V, tag, args):
    b, g, Ktab = per_track_box(V)
    ok = g & np.isfinite(b) & (np.abs(V.x) < args.clip)
    print(f"\n{'='*78}\n=== {tag}   {int(ok.sum())} tracks\n{'='*78}")
    print("  K on the (band x sigma quintile) grid, GeV  [q=+1 rows; "
          "q=-1 is -K to 2 %]")
    for ib, nm in enumerate(cc.BANDS_SHORT):
        row = "  ".join(f"{Ktab[(ib, 1, i)][0]:+7.2f}"
                        for i in range(5) if (ib, 1, i) in Ktab)
        print(f"    {nm:8s} {row}")
    pb = float(np.mean(b[ok] * V.q[ok] * V.q[ok]))   # b is already signed
    # charge-odd projection of the prediction, exactly as for the data
    pv, pe = moments(b, V.q, ok)
    print(f"\n  PREDICTED charge-odd <z> = {pv*1e3:+7.3f} e-3   "
          f"(<b> over all tracks {pb*1e3:+7.3f} e-3)")
    mv, me = moments(V.x, V.q, ok)
    qv, qe = moments(V.x, V.q, ok, probe=U)
    print(f"  MEASURED  charge-odd <x> = {mv*1e3:+7.3f}+-{me*1e3:5.3f} e-3"
          f"   (probe u={U}: {qv*1e3:+7.3f}+-{qe*1e3:5.3f} e-3)")
    print(f"  prediction in probe space (x{CM:.4f}) = {pv*CM*1e3:+7.3f} e-3")

    print("\n  --- per eta band, 1e-3 ---")
    print(f"  {'band':10s}{'pred <z>':>12s}{'meas <x>':>18s}"
          f"{'meas probe':>18s}{'pull (mean)':>13s}")
    for ib, nm in enumerate(cc.BANDS_SHORT):
        m = ok & (V.band == ib)
        p_, _ = moments(b, V.q, m)
        v_, e_ = moments(V.x, V.q, m)
        w_, f_ = moments(V.x, V.q, m, probe=U)
        print(f"  {nm:10s}{p_*1e3:+12.3f}{v_*1e3:+13.3f}+-{e_*1e3:4.2f}"
              f"{w_*1e3:+13.3f}+-{f_*1e3:4.2f}{(v_-p_)/e_:+13.2f}")

    print("\n  --- the VALIDATION: regress the measured pull on the "
          "per-track prediction ---")
    print("  slope 1 = the Box bias is exactly what the data carry; "
          "0 = it is absent.")
    y = V.x[ok]
    bb = b[ok]
    sl = float((bb * y).sum() / (bb * bb).sum())
    esl = float(np.sqrt(((y - sl * bb) ** 2).sum() / (bb * bb).sum()
                        / max(len(y) - 1, 1)))
    print(f"    slope = {sl:+.3f} +- {esl:.3f}")
    # The shape handle. Binning on the FITTED sigma is the conditioning trap
    # of this campaign (`sigma` is the pull's own denominator, and selecting on
    # it manufactures a charge-ODD trend of tens of e-3 -- measured: -68 -> +44
    # across sigma quintiles, which is the artefact, not the physics). Bins are
    # therefore in GEN pT x GEN |eta|, which is exogenous, and the PREDICTION
    # is averaged in the same cells.
    print("\n  --- the shape, on an EXOGENOUS gen (pT, |eta|) grid ---")
    gp = V.d["genpt"].astype(float)
    ge = np.abs(V.d["geneta"].astype(float))
    pe = np.percentile(gp[ok], [0, 33.3, 66.7, 100])
    ee = np.percentile(ge[ok], [0, 33.3, 66.7, 100])
    pe[-1] += 1e-9
    ee[-1] += 1e-9
    print(f"  {'cell':30s}{'<sigma>':>11s}{'pred':>10s}{'measured':>17s}")
    cells = []
    for i in range(3):
        for j in range(3):
            m = ok & (gp >= pe[i]) & (gp < pe[i + 1]) & (ge >= ee[j]) & (ge < ee[j + 1])
            if m.sum() < 2000:
                continue
            p_, _ = moments(b, V.q, m)
            v_, e_ = moments(V.x, V.q, m)
            cells.append((p_, v_, e_))
            print(f"  pT {pe[i]:5.1f}-{pe[i+1]:5.1f} |eta| {ee[j]:.2f}-{ee[j+1]:.2f}"
                  f"{V.sigma[m].mean():11.3e}{p_*1e3:+10.2f}"
                  f"{v_*1e3:+12.2f}+-{e_*1e3:4.2f}")
    P = np.array([c[0] for c in cells])
    M = np.array([c[1] for c in cells])
    E = np.array([c[2] for c in cells])
    w = 1. / E ** 2
    k = float((w * P * M).sum() / (w * P * P).sum())
    ek = float(1. / np.sqrt((w * P * P).sum()))
    KFIT.append((k, ek))
    print(f"    cell-wise scale of the prediction: k = {k:+.2f} +- {ek:.2f}"
          f"   (1 = validated)   chi2(k=1) = "
          f"{float((w*(M-P)**2).sum()):.1f}/{len(cells)}"
          f"   chi2(k=0) = {float((w*M**2).sum()):.1f}/{len(cells)}")

    print("\n  --- what it does to a MOMENTUM SCALE ---")
    print("  <z>_odd * sigma_rel = Delta(1/p)/(1/p), so the MOMENTUM bias is")
    print("  dp/p = -<z>_odd sigma_rel, charge-INDEPENDENT; in a dimuon mass")
    print("  dm/m = dp/p -- it ADDS on both legs, unlike the sagitta term.")
    print(f"  {'selection':26s}{'<sigma_rel>':>12s}{'<p>':>8s}"
          f"{'dp/p':>13s}{'dm at m_Z':>14s}")
    sel_list = [(nm, ok & (V.band == i)) for i, nm in enumerate(cc.BANDS_SHORT)]
    sel_list.append(("Z-like: pT 40-50 |eta|<0.4",
                     ok & (gp >= 40) & (gp < 50) & (ge < 0.4)))
    sel_list.append(("Z-like: pT 40-50 all eta",
                     ok & (gp >= 40) & (gp < 50)))
    for nm, m in sel_list:
        if m.sum() < 500:
            continue
        beta, _ = moments(b * V.sigrel, V.q, m)
        print(f"  {nm:26s}{V.sigrel[m].mean():12.5f}{V.p[m].mean():8.1f}"
              f"{-beta*1e6:+10.2f}e-6{-beta*91188*1e3:+11.1f} keV")
    return Ktab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--clip", type=float, default=10.)
    a = ap.parse_args()
    for p in a.npz:
        run(cc.Var(p), p, a)
    if len(KFIT) > 1:
        w = np.array([1. / e ** 2 for _, e in KFIT])
        kk = np.array([k for k, _ in KFIT])
        mu = float((kk * w).sum() / w.sum())
        em = float(1. / np.sqrt(w.sum()))
        print(f"\n{'='*78}\n=== COMBINED over the two guns: the per-track Box "
              f"prediction carries a scale\n    k = {mu:+.2f} +- {em:.2f}   "
              f"(1 = the data carry exactly the predicted bias; "
              f"{abs(mu)/em:.1f} sigma from 0)\n{'='*78}")


if __name__ == "__main__":
    main()
