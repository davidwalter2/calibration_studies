#!/usr/bin/env python3
"""Toy of the CVH quadratic term's material parameter, to settle three
mechanisms by construction rather than by assertion.

THE STRUCTURE BEING REPRODUCED.  In the CVH fit the material-group parameter
`k_g` enters the design matrix through ONE column, `transportJacobianBxByBzD`'s
`dxi` column, whose only non-zero row is `dqopdxi = -dEdx * s * ...`
(`Geant4ePropagator.cc` ~2967; `dlamdxi = dphidxi = dxtdxi = dytdxi = 0`).  It
is therefore a MEAN-only derivative on the process-noise (material) constraint
rows, weighted by the CURRENT Q.  `exp(k_g)` also multiplies the step's MS and
ionization VARIANCES (`Geant4ePropagator.cc:1217-1265`), but that scaling is
never differentiated -- "the fit never differentiates through the weights".

The toy is that model, minimally:

    state      u_i = q/p at plane i,          i = 0 .. N-1
    process    u_{i+1} = u_i + delta_i        delta_i ~ (mean m_i, straggling)
    hits       y_i    = u_i + eps_i,          eps_i ~ N(0, sigma_h^2)
    model      u_{i+1} - u_i - m_i A(k) = 0   with weight 1/v_i  (v_i FIXED)
    chi2(u,k)  = sum (y_i-u_i)^2/sigma_h^2 + sum (u_{i+1}-u_i-m_i A(k))^2/v_i
    A(k)       = 1 + k   (the linearization of exp(k) at k = 0)

TESTS
  A  true straggling variance = r * v_i, mean correct
     -> E[k] = 0 for every r; only the ERROR is wrong, by sqrt(r).
        (so a wrong Q CANNOT produce a shifted k in this functional)
  B  true mean = rho * m_i, variance correct
     -> E[k] = rho - 1 (= ln rho to first order), exactly.
  C  correct mean, SKEWED (Moyal ~ Landau) straggling, plus the production's
     own `chi2/ndof < c` selection
     -> E[k] < 0, growing with the skewness and tightening with c: the mean is
        right but the SELECTED sample's is not.
  C0 the same with a SYMMETRIC straggling of the same width -> E[k] = 0.
"""
import argparse

import numpy as np


def build(N, m, v, sh2):
    """Design matrix and weights of the linear model with unknowns (u_0..u_{N-1}, k)."""
    nrow = N + (N - 1)
    J = np.zeros((nrow, N + 1))
    W = np.zeros(nrow)
    for i in range(N):                      # hit rows:  y_i - u_i
        J[i, i] = 1.0
        W[i] = 1.0 / sh2
    for i in range(N - 1):                  # material rows: u_{i+1}-u_i-m_i(1+k)
        r = N + i
        J[r, i] = -1.0
        J[r, i + 1] = 1.0
        J[r, N] = -m[i]
        W[r] = 1.0 / v[i]
    return J, W


def run(ntrial, N, m, v, sh2, rvar=1.0, rho=1.0, kind="gauss", chi2cut=0.0,
        seed=0):
    rng = np.random.default_rng(seed)
    J, W = build(N, m, v, sh2)
    JW = J.T * W
    A = JW @ J
    Ainv = np.linalg.inv(A)
    ks, c2s = [], []
    nrow = J.shape[0]
    sd = np.sqrt(rvar * v)
    for _ in range(ntrial):
        if kind == "gauss":
            d = m * rho + rng.normal(0.0, 1.0, N - 1) * sd
        elif kind == "moyal":
            # Moyal: mean ln2+gamma = 1.27036, var pi^2/2, skew 1.5351.
            # standardized and rescaled so the mean is EXACTLY m*rho and the
            # sd is EXACTLY sd -- only the third and higher cumulants change.
            z = rng.standard_normal(N - 1)
            uu = rng.random(N - 1)
            _ = z
            # inverse-CDF sampling of Moyal:  F(x) = erfc(exp(-x/2)/sqrt2)
            from scipy.special import erfcinv
            x = -np.log(2.0 * erfcinv(uu) ** 2)   # inverse CDF of Moyal
            x = (x - 1.2703628) / np.sqrt(np.pi ** 2 / 2.0)
            d = m * rho + x * sd
        elif kind == "sym":
            # symmetric heavy-tailed control with the same sd (Student-t, 5 dof)
            x = rng.standard_t(5, N - 1)
            x = x / np.sqrt(5.0 / 3.0)
            d = m * rho + x * sd
        else:
            raise ValueError(kind)
        u = np.concatenate([[0.0], np.cumsum(d)])
        y = u + rng.normal(0.0, np.sqrt(sh2), N)
        b = np.zeros(nrow)
        b[:N] = y
        b[N:] = m * 0.0                      # residual of the model at k=0 is
        # (u_{i+1}-u_i-m_i); the constant -m_i goes into the rhs:
        b[N:] = m
        # solve  min || J x - b ||_W^2  with b_hit = y and b_mat = m
        th = Ainv @ (JW @ b)
        rres = J @ th - b
        c2 = float(rres @ (W * rres))
        ndof = nrow - (N + 1)
        ks.append(th[N])
        c2s.append(c2 / max(ndof, 1))
    ks = np.asarray(ks)
    c2s = np.asarray(c2s)
    sel = c2s < chi2cut if chi2cut > 0 else np.ones(len(ks), bool)
    return ks, c2s, sel, np.sqrt(Ainv[N, N])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--ntrial", type=int, default=40000)
    ap.add_argument("-N", "--nplane", type=int, default=12)
    ap.add_argument("--dE", type=float, default=0.02,
                    help="mean q/p step per plane, in units of the hit sigma")
    ap.add_argument("--strag", type=float, default=1.0,
                    help="straggling sd of a step, in units of the mean step")
    ap.add_argument("--sigh", type=float, default=1.0)
    args = ap.parse_args()

    N = args.nplane
    m = np.full(N - 1, args.dE)
    v = (args.strag * m) ** 2
    sh2 = args.sigh ** 2

    print(f"toy: N={N} planes, mean step m={args.dE} (hit sigma units), "
          f"straggling sd/mean={args.strag}\n")

    print("A. wrong Q (true straggling variance = r * model v), mean correct")
    print(f"   {'r':>6s} {'<k>':>11s} {'err(<k>)':>10s} {'rms(k)':>10s} "
          f"{'Fisher sig':>11s} {'rms/Fisher':>11s}")
    for ir, r in enumerate((0.25, 0.5, 1.0, 2.0, 4.0)):
        ks, c2, sel, sig = run(args.ntrial, N, m, v, sh2, rvar=r, seed=101 + ir)
        print(f"   {r:6.2f} {ks.mean():11.5f} {ks.std()/np.sqrt(len(ks)):10.5f} "
              f"{ks.std():10.5f} {sig:11.5f} {ks.std()/sig:11.3f}")

    print("\nB. wrong MEAN (true mean = rho * model m), variance correct")
    print(f"   {'rho':>6s} {'<k>':>11s} {'err':>11s} {'expected':>11s}")
    for ir, rho in enumerate((0.6, 0.8, 1.0, 1.2)):
        ks, c2, sel, sig = run(args.ntrial, N, m, v, sh2, rho=rho, seed=201 + ir)
        _ = sig
        print(f"   {rho:6.2f} {ks.mean():11.5f} +- {ks.std()/np.sqrt(len(ks)):8.5f}  {rho-1.0:11.5f}")

    print("\nC. correct mean, SKEWED (Moyal) straggling + a chi2/ndof selection")
    print(f"   {'strag':>6s} {'chi2cut':>8s} {'keep':>7s} {'<k> moyal':>11s} "
          f"{'<k> sym-t5':>11s} {'<k> gauss':>11s}")
    for st in (0.5, 1.0, 2.0, 4.0):
        vv = (st * m) ** 2
        for cut in (0.0, 3.0, 1.5):
            out = []
            for ik, kind in enumerate(("moyal", "sym", "gauss")):
                ks, c2, sel, sig = run(args.ntrial, N, m, vv, sh2, kind=kind,
                                       chi2cut=cut, seed=301 + ik)
                out.append((ks[sel].mean() if sel.any() else np.nan,
                            sel.mean(),
                            ks[sel].std() / np.sqrt(max(sel.sum(), 1))))
            print(f"   {st:6.2f} {cut:8.2f} {out[0][1]*100:6.1f}% "
                  f"{out[0][0]:11.5f} {out[1][0]:11.5f} {out[2][0]:11.5f}"
                  f"   (mc err {out[0][2]:.5f})")


if __name__ == "__main__":
    main()
