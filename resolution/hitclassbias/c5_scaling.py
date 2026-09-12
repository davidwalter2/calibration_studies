#!/usr/bin/env python3
"""WHAT SHAPE IS THE CHARGE-EVEN SHIFT? A per-track regression that separates a
SECOND-ORDER estimator bias from a FIXED additive curvature offset.

THE THREE HYPOTHESES, written as what each does to Delta(q/p) and hence to the
pull x = Delta(q/p)/sigma. Write sigma_rel = sigma * p (the RELATIVE curvature
error) and remember sigma = sigma_rel / p.

  M1  SECOND-ORDER (Box) bias of the converged Gauss-Newton estimator.
      A nonlinear least-squares estimator is biased at O(sigma^2) in the
      parameter, and the dimensionless statement is that the RELATIVE bias is
      c times the square of the RELATIVE error:
          Delta(q/p) / |q/p| = c sigma_rel^2   ->   <x> = c sigma_rel.
      c is dimensionless and O(1) is its natural size.

  M2  FIXED ADDITIVE CURVATURE OFFSET in q/p -- a hit-location bias, a
      residual misalignment, a field error: something that displaces the
      measured sagitta by a fixed amount independent of the momentum.
          Delta(q/p) = D   ->   <x> = D / sigma.

  M2T The same, but fixed in q/pT, which is what a sagitta-like bias in the
      BENDING PLANE actually is (Delta(q/p) = Delta(q/pT)/cosh eta):
          <x> = D_T / (sigma cosh eta).

  M3  A constant offset OF THE PULL itself -- no physical shape, the null
      shape the per-band tables assume.
          <x> = const.

WHY A REGRESSION AND NOT A BINNED FIT. Binning on the fitted sigma is a
conditioning trap: `sigma` is the pull's own denominator. Here every regressor is
built from `sigma` and the RECO p, which are the same reconstructed quantities
-- but the regression is against the CHARGE-EVEN projection, and the leading
selection artefact (`sigma = sigma_bar (1 + a q x)`) is charge-ODD, so it
enters the odd columns and not the even ones. Every regressor is therefore
entered TWICE, once as `f` (charge-even) and once as `q f` (charge-odd), and
the odd columns are the control: a trap that leaks shows up there first.
The binned 2-D GEN pT x GEN |eta| table is printed as well, because it uses
purely exogenous binning and must tell the same story.
"""
import argparse

import numpy as np

import conv_common as cc

DSCALE = 1e-4      # so a coefficient of 1 means D = 1e-4 GeV^-1


def design(V, m):
    """The four charge-even regressors, per track."""
    return {
        "M1  c sigma_rel  (2nd order)": V.sigrel[m],
        "M2  D/sigma      (fixed dq/p)": DSCALE / V.sigma[m],
        "M2T D_T/(s cosh eta) (sagitta)": DSCALE / (V.sigma[m] * np.cosh(V.eta[m])),
        "M3  constant pull": np.ones(int(m.sum())),
    }


def wls(y, X):
    A = X.T @ X
    b = X.T @ y
    Ai = np.linalg.inv(A)
    p = Ai @ b
    r = y - X @ p
    s2 = float(r @ r) / (len(y) - X.shape[1])
    return p, np.sqrt(np.diag(Ai) * s2), float(r @ r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--clip", type=float, default=10.,
                    help="|x| clip; the trim scan says the shift is a CORE "
                         "shift so this is safe and it stabilises the LS")
    ap.add_argument("--sigrelmax", type=float, default=0.1,
                    help="drop the 0.5 %% of tracks with sigma_rel above this; "
                         "they are the very forward, badly measured tail and "
                         "they would otherwise carry all the leverage of the "
                         "M1 regressor, which is sigma_rel itself")
    ap.add_argument("--npt", type=int, default=4)
    ap.add_argument("--neta", type=int, default=6)
    a = ap.parse_args()
    rng = np.random.default_rng(515)
    for path in a.npz:
        V = cc.Var(path)
        m = V.good & (np.abs(V.x) < a.clip) & (V.sigrel < a.sigrelmax)
        print(f"\n{'='*78}\n=== SHAPE OF THE CHARGE-EVEN SHIFT -- {V.name}\n"
              f"    {int(m.sum())} tracks of {len(V)} (|x| < {a.clip})\n{'='*78}")
        y = V.x[m]
        q = V.q[m]
        reg = design(V, m)
        print(f"\n  regressor ranges: sigma_rel {V.sigrel[m].min():.5f}-"
              f"{V.sigrel[m].max():.5f}, sigma {V.sigma[m].min():.3e}-"
              f"{V.sigma[m].max():.3e} GeV^-1")

        print("\n  --- ONE SHAPE AT A TIME (with its charge-odd twin as the "
              "control) ---")
        print(f"  {'model':34s}{'even coeff':>22s}{'odd coeff':>22s}"
              f"{'dchi2 even|odd':>15s}")
        chi0 = float(y @ y)
        singles = {}
        for nm, f in reg.items():
            X = np.stack([f, q * f], axis=1)
            p, e, chi = wls(y, X)
            # the improvement the EVEN column buys with the odd one already in
            _, _, chi_oddonly = wls(y, (q * f)[:, None])
            singles[nm] = (p[0], e[0], chi_oddonly - chi)
            print(f"  {nm:34s}{p[0]:+13.5g}+-{e[0]:8.2g}"
                  f"{p[1]:+13.5g}+-{e[1]:8.2g}{chi_oddonly-chi:15.2f}")

        print("\n  --- ALL FOUR TOGETHER (even + odd columns) ---")
        F = np.stack(list(reg.values()), axis=1)
        X = np.concatenate([F, F * q[:, None]], axis=1)
        p, e, chi = wls(y, X)
        for i, nm in enumerate(reg):
            print(f"  {nm:34s}{p[i]:+13.5g}+-{e[i]:8.2g}"
                  f"{p[i+len(reg)]:+13.5g}+-{e[i+len(reg)]:8.2g}")
        print(f"  joint dchi2 vs no effect = {chi0-chi:.2f} for 8 parameters")

        print("\n  --- what each single fit MEANS physically ---")
        c1 = singles["M1  c sigma_rel  (2nd order)"]
        d2 = singles["M2  D/sigma      (fixed dq/p)"]
        d2t = singles["M2T D_T/(s cosh eta) (sagitta)"]
        c3 = singles["M3  constant pull"]
        print(f"    M1  relative curvature bias = {c1[0]:+.4f} sigma_rel^2 "
              f"({abs(c1[0]/c1[1]):.1f} sigma); at sigma_rel = 0.018 that is "
              f"{c1[0]*0.018**2*1e6:+.3f}e-6 relative")
        print(f"    M2  Delta(q/p)  = {d2[0]*DSCALE*1e6:+.4f}e-6 GeV^-1  "
              f"({abs(d2[0]/d2[1]):.1f} sigma)")
        print(f"    M2T Delta(q/pT) = {d2t[0]*DSCALE*1e6:+.4f}e-6 GeV^-1 "
              f"({abs(d2t[0]/d2t[1]):.1f} sigma)  "
              f"[the AN's |M| bound is 1e-4 GeV^-1]")
        print(f"    M3  <x> = {c3[0]*1e3:+.3f}e-3 ({abs(c3[0]/c3[1]):.1f} sigma)")

        # --- the exogenous 2-D grid, as the cross-check --------------------
        gp = V.d["genpt"].astype(float)
        ge = np.abs(V.d["geneta"].astype(float))
        pe = np.percentile(gp[V.good], np.linspace(0, 100, a.npt + 1))
        ee = np.percentile(ge[V.good], np.linspace(0, 100, a.neta + 1))
        pe[-1] += 1e-9
        ee[-1] += 1e-9
        rows = []
        for i in range(a.npt):
            for j in range(a.neta):
                mm = (V.good & (gp >= pe[i]) & (gp < pe[i + 1])
                      & (ge >= ee[j]) & (ge < ee[j + 1]))
                v, er, n = cc.even_mean(V.x, V.q, mm, rng, 200)
                if not np.isfinite(v):
                    continue
                rows.append((V.sigrel[mm].mean(), V.sigma[mm].mean(),
                             np.mean(1. / (V.sigma[mm] * np.cosh(V.eta[mm]))),
                             v, er, n,
                             f"pT {pe[i]:.0f}-{pe[i+1]:.0f} "
                             f"|eta| {ee[j]:.2f}-{ee[j+1]:.2f}"))
        rows.sort()
        sr = np.array([r[0] for r in rows])
        si = np.array([r[1] for r in rows])
        sc = np.array([r[2] for r in rows])
        ev = np.array([r[3] for r in rows])
        er = np.array([r[4] for r in rows])
        print(f"\n  --- exogenous GEN pT x GEN |eta| grid, {len(rows)} cells, "
              f"sigma_rel lever arm {sr.max()/sr.min():.2f} ---")
        print(f"  {'cell':34s}{'<sigma_rel>':>12s}{'<x>_even (1e-3)':>20s}")
        for r in rows:
            print(f"  {r[6]:34s}{r[0]:12.5f}{r[3]*1e3:+14.2f}+-{r[4]*1e3:4.2f}")
        w = 1. / er ** 2
        null = float((w * ev ** 2).sum())
        print(f"\n  chi2(no effect) = {null:.2f} / {len(rows)}")
        for nm, X in (("M1  c sigma_rel", sr),
                      ("M2  D/sigma", DSCALE / si),
                      ("M2T D_T/(s cosh eta)", DSCALE * sc),
                      ("M3  constant", np.ones_like(sr))):
            k = (w * X * ev).sum() / (w * X * X).sum()
            ek = 1. / np.sqrt((w * X * X).sum())
            c2 = float((w * (ev - k * X) ** 2).sum())
            print(f"    {nm:24s} k = {k:+.5g} +- {ek:.5g} "
                  f"({abs(k)/ek:4.1f} sigma)  chi2 {c2:7.2f}/{len(rows)-1}"
                  f"  dchi2 {c2-null:+7.2f}")


if __name__ == "__main__":
    main()
