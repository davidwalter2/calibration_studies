#!/usr/bin/env python3
"""IS THE ODD-IN-CHARGE PULL ASYMMETRY A PROPERTY OF THE MOMENTUM, OR OF 1/sigma?

z = (q/p_fit - q/p_gen)/sigma with sigma = sqrt(refCov[0,0]) -- the CVH fit's
OWN covariance, assembled at the CONVERGED state.  The process noise that
dominates that covariance scales with the momentum (MS and radiative:
sigma_qop ~ kappa; ionization straggling: sigma_qop ~ kappa^2), so sigma is a
monotone function of the FITTED |q/p| -- i.e. of the very fluctuation that the
pull is measuring.

Write eps = q/p_fit - q/p_gen = sigma0 x with x symmetric and unbiased, and
sigma = sigma0 (1 + a q x) with

    a = (d ln sigma / d ln kappa) * sigma_rel ,   sigma_rel = sigma / kappa .

Then  z = x/(1 + a q x) ~= x - a q x^2  and

    <q z>       = -a           NOT ZERO, charge-odd, ~ -sigma_rel*(1-vgf)
    <q eps/kappa> = 0          the momentum itself stays unbiased
    E[q z | sigma] increases with sigma   (binning on sigma bins on q x)

so a division by a self-consistent sigma manufactures exactly the observed
signature.  Three tests here, all on the existing caches:

  T1  <q z> vs <q eps/kappa>, same tracks, same trim.
  T2  z rebuilt with a TRUTH-ONLY sigma_bar(gen p, |eta|, nvalidhits) that
      cannot know the fluctuation.  The anomaly must collapse.
  T3  the coefficient a measured directly, as the within-cell regression slope
      of ln sigma on q z, against the observed -<q z>.
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.dirname(_HERE)
sys.path.insert(0, _RES)
sys.path.insert(0, _HERE)


def cellid(*cols):
    """Dense id from a list of integer columns."""
    out = np.zeros(len(cols[0]), dtype=np.int64)
    for c in cols:
        out = out * (int(c.max()) + 1) + c
    _, inv = np.unique(out, return_inverse=True)
    return inv


def cellmean(val, cid, ncell, leave_one_out=False):
    s = np.bincount(cid, weights=val, minlength=ncell)
    n = np.bincount(cid, minlength=ncell).astype(np.float64)
    full = (s / np.maximum(n, 1.))[cid]
    if not leave_one_out:
        return full
    # leave-one-out where the cell has >= 2 entries; the plain cell mean
    # otherwise (a singleton cell carries no self-correlation to remove
    # beyond the whole of itself, and there are O(10) of them)
    loo = (s[cid] - val) / np.maximum(n[cid] - 1., 1.)
    return np.where(n[cid] >= 2., loo, full)


def tmean(v, msk):
    return float(v[msk].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--trim", type=float, default=5.0)
    ap.add_argument("--nboot", type=int, default=200)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    d = np.load(a.cache)
    z = d["z"]; sig = d["sigma"]
    q = np.sign(d["charge"]).astype(np.float64)
    eta = d["eta"].astype(np.float64)
    gpt = d["genpt"]; nvh = d["nvalidhits"]; vgf = d["vgf"]
    pgen = gpt * np.cosh(eta)
    kap = 1.0 / pgen
    eps = z * sig
    srel = sig / kap
    n = len(z)
    L = []; P = L.append
    P(f"# {a.label or os.path.basename(a.cache)}   n = {n}   trim |z| < {a.trim}")
    P(f"# <sigma_rel> = {srel.mean():.5f}   median {np.median(srel):.5f}   "
      f"<vgf> = {vgf.mean():.4f}   median {np.median(vgf):.4f}")

    msk = np.abs(z) < a.trim
    rng = np.random.default_rng(20260905)

    def bootse(v, m):
        idx = np.flatnonzero(m)
        s = np.array([v[idx[rng.integers(0, len(idx), len(idx))]].mean()
                      for _ in range(a.nboot)])
        return float(s.std(ddof=1))

    # ---------------------------------------------------------------- T1
    P("")
    P("## T1  the pull mean vs the momentum mean, SAME tracks")
    P("| statistic | value | boot err |")
    P("|---|---|---|")
    A = tmean(q * z, msk); eA = bootse(q * z, msk)
    R = tmean(q * eps / kap, msk); eR = bootse(q * eps / kap, msk)
    P(f"| A = <q z>  (pull units) | {A:+.5f} | {eA:.5f} |")
    P(f"| <q eps/kappa> = <p_gen/p_fit> - 1  (relative) | {R:+.3e} | {eR:.3e} |")
    P(f"| A x <sigma_rel>  (what A would mean if uncorrelated) | "
      f"{A*srel[msk].mean():+.3e} | - |")
    P(f"| Cov(q z, sigma_rel) | {np.cov(q[msk]*z[msk], srel[msk])[0,1]:+.3e} | - |")
    P(f"| a from the identity  Cov/<sigma_rel> | "
      f"{np.cov(q[msk]*z[msk], srel[msk])[0,1]/srel[msk].mean():+.5f} | - |")

    # ---------------------------------------------------------------- T2
    P("")
    P("## T2  the pull rebuilt on a TRUTH-ONLY sigma_bar")
    P("sigma_bar = leave-one-out mean of sigma in cells of (gen p, |eta|, "
      "nvalidhits); it cannot know the track's own fluctuation.")
    P("| cells | n/cell | <q z> | boot | <q z_bar> | boot | <q eps/kappa> |")
    P("|---|---|---|---|---|---|---|")
    for npbin, netabin in ((12, 12), (20, 20), (30, 24)):
        ip = np.clip(np.digitize(np.log(pgen),
                     np.quantile(np.log(pgen), np.linspace(0, 1, npbin + 1))[1:-1]),
                     0, npbin - 1)
        ie = np.clip(np.digitize(np.abs(eta),
                     np.quantile(np.abs(eta), np.linspace(0, 1, netabin + 1))[1:-1]),
                     0, netabin - 1)
        ih = np.clip(nvh.astype(np.int64) - int(nvh.min()), 0, 30)
        cid = cellid(ip.astype(np.int64), ie.astype(np.int64), ih)
        nc = int(cid.max()) + 1
        sbar = cellmean(sig, cid, nc, leave_one_out=True)
        zb = eps / sbar
        mb = np.abs(z) < a.trim          # SAME tracks as A, by construction
        Ab = tmean(q * zb, mb); eAb = bootse(q * zb, mb)
        P(f"| {nc} | {n/nc:.0f} | {A:+.5f} | {eA:.5f} | {Ab:+.5f} | {eAb:.5f} "
          f"| {R:+.3e} |")

    # ---------------------------------------------------------------- T3
    P("")
    P("## T3  the coefficient a = d ln sigma / d(q z), measured in-cell")
    P("Within a (gen p, |eta|, nvalidhits) cell sigma still varies; the part "
      "that varies WITH q z is the self-consistency. Slope of ln sigma on q z, "
      "both centred in the cell.")
    P("| cells | n/cell | a_meas = slope | -a_meas | observed A | ratio A/(-a) |")
    P("|---|---|---|---|---|---|")
    for npbin, netabin in ((12, 12), (20, 20), (30, 24)):
        ip = np.clip(np.digitize(np.log(pgen),
                     np.quantile(np.log(pgen), np.linspace(0, 1, npbin + 1))[1:-1]),
                     0, npbin - 1)
        ie = np.clip(np.digitize(np.abs(eta),
                     np.quantile(np.abs(eta), np.linspace(0, 1, netabin + 1))[1:-1]),
                     0, netabin - 1)
        ih = np.clip(nvh.astype(np.int64) - int(nvh.min()), 0, 30)
        cid = cellid(ip.astype(np.int64), ie.astype(np.int64), ih)
        nc = int(cid.max()) + 1
        ls = np.log(sig); y = q * z
        lsc = ls - cellmean(ls, cid, nc)
        yc = y - cellmean(y, cid, nc)
        sel = msk
        slope = float((lsc[sel] * yc[sel]).sum() / (yc[sel] ** 2).sum())
        P(f"| {nc} | {n/nc:.0f} | {slope:+.5f} | {-slope:+.5f} | {A:+.5f} "
          f"| {A/(-slope) if slope else np.nan:.3f} |")

    # ---------------------------------------------------------------- T4
    P("")
    P("## T4  the WHOLE shape under the truth-only sigma_bar")
    ip = np.clip(np.digitize(np.log(pgen),
                 np.quantile(np.log(pgen), np.linspace(0, 1, 21))[1:-1]), 0, 19)
    ie = np.clip(np.digitize(np.abs(eta),
                 np.quantile(np.abs(eta), np.linspace(0, 1, 21))[1:-1]), 0, 19)
    ih = np.clip(nvh.astype(np.int64) - int(nvh.min()), 0, 30)
    cid = cellid(ip.astype(np.int64), ie.astype(np.int64), ih)
    nc = int(cid.max()) + 1
    sbar = cellmean(sig, cid, nc, leave_one_out=True)
    zb = eps / sbar
    P(f"(20 x 20 x nhits = {nc} cells, {n/nc:.0f} tracks/cell; "
      f"<sigma/sigma_bar> = {(sig/sbar).mean():.4f}, "
      f"rms {(sig/sbar).std():.4f})")
    P("| statistic | A(z) | A(z_bar) | S(z) | S(z_bar) |")
    P("|---|---|---|---|---|")
    def AS_of(v, sel):
        vp = v[sel & (q > 0)]; vm = v[sel & (q < 0)]
        return 0.5 * (vp.mean() - vm.mean()), 0.5 * (vp.mean() + vm.mean())
    for nm, T in (("mean", None), ("trim1", 1.), ("trim2", 2.),
                  ("trim3", 3.), ("trim5", 5.), ("trim10", 10.)):
        s1 = np.ones(n, bool) if T is None else (np.abs(z) < T)
        s2 = np.ones(n, bool) if T is None else (np.abs(zb) < T)
        A1, S1 = AS_of(z, s1); A2, S2 = AS_of(zb, s2)
        P(f"| {nm} | {A1:+.5f} | {A2:+.5f} | {S1:+.5f} | {S2:+.5f} |")
    for u in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
        A1, S1 = AS_of(z * np.exp(-u * z ** 2), np.ones(n, bool))
        A2, S2 = AS_of(zb * np.exp(-u * zb ** 2), np.ones(n, bool))
        P(f"| <z e^-{u:g}z^2> | {A1:+.5f} | {A2:+.5f} | {S1:+.5f} | {S2:+.5f} |")

    # -------------------------------------------------------- prediction
    P("")
    P("## the closed-form prediction  A = -<sigma_rel * d ln sigma / d ln kappa>")
    P("d ln sigma/d ln kappa = f_b + 2 f_c with f_b the share of sigma^2 that "
      "scales as kappa^2 (MS, radiative) and f_c the share that scales as "
      "kappa^4 (ionization straggling); 1 - vgf = f_b + f_c is measured.")
    lo = -(srel * (1. - vgf))[msk].mean()
    hi = -2. * (srel * (1. - vgf))[msk].mean()
    P(f"| bound | value |")
    P(f"|---|---|")
    P(f"| -<sigma_rel (1-vgf)>   (f_c = 0) | {lo:+.5f} |")
    P(f"| -2<sigma_rel (1-vgf)>  (f_b = 0) | {hi:+.5f} |")
    P(f"| observed A | {A:+.5f} +- {eA:.5f} |")

    txt = "\n".join(L)
    print(txt)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
