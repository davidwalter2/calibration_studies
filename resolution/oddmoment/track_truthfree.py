#!/usr/bin/env python3
"""The TRUTH-FREE version of the single-track pull correction.

`oddmoment/sigma_pull.py` T2 removes the self-consistency of sigma by replacing
it with a sigma_bar built from GEN quantities -- a diagnostic, not something a
data analysis can do.  The same inversion needs no truth at all:

    sigma_i = sigma_bar_i + a_i q_i eps_i        (definition of a_i)
  =>sigma_bar_i = sigma_i - a_i q_i eps_i = sigma_i (1 - a_i q_i z_i)
  =>x_i = eps_i/sigma_bar_i = z_i / (1 - a_i q_i z_i)

with a_i = sigma_rel,i * d ln sigma/d ln kappa_i built from OBSERVED quantities:
sigma_rel,i = sigma_i * p_fit,i (p from the refit itself) and
d ln sigma/d ln kappa = f_ms + 2 f_ioni ~= 1 - vgf_i (vgf is cached; f_ioni is
per-mille and is only separable when the per-family shares are extracted).

x is then the variable the CF model actually describes, so the closure is
`model vs x` with the model untouched.  --a-scale rescales a_i to expose how
much of the closure depends on the approximation d ln sigma/d ln kappa = 1-vgf.
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.dirname(_HERE)
sys.path.insert(0, _RES)
sys.path.insert(0, _HERE)
import decomp as D                                            # noqa: E402
import sigma_pull as SP                                       # noqa: E402

TRIMS = (1., 2., 3., 5., 10.)
PROBES = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--krad", type=float, default=1.0)
    ap.add_argument("--a-scale", type=float, default=1.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    args = D.Args(); args.krad = a.krad
    d = D.load(a.cache)
    z = d["z"]; sig = d["sigma"]; vgf = d["vgf"]
    q = np.sign(d["charge"]).astype(np.float64)
    eta = d["eta"].astype(np.float64)
    pfit = d["trackpt"] * np.cosh(eta)              # OBSERVED momentum
    n = len(z)
    ai = sig * pfit * (1. - vgf) * a.a_scale        # truth-free a_i
    den = 1. - ai * q * z
    bad = den < 0.3
    den = np.maximum(den, 0.3)
    x = z / den

    # the truth-referenced sigma_bar of T2, for comparison
    pgen = d["genpt"] * np.cosh(eta)
    cid = SP.cellid(
        np.clip(np.digitize(np.log(pgen),
                np.quantile(np.log(pgen), np.linspace(0, 1, 21))[1:-1]),
                0, 19).astype(np.int64),
        np.clip(np.digitize(np.abs(eta),
                np.quantile(np.abs(eta), np.linspace(0, 1, 21))[1:-1]),
                0, 19).astype(np.int64),
        np.clip(d["nvalidhits"].astype(np.int64)
                - int(d["nvalidhits"].min()), 0, 30))
    nc = int(cid.max()) + 1
    sb = SP.cellmean(sig, cid, nc, leave_one_out=True)
    zb = z * sig / sb

    Om, Em, phis, cnt = D.model_per_track(
        d, args, PROBES, pool_id=np.where(q > 0, 0, 1), npool=2)

    L = []; P = L.append
    P(f"# {a.label or os.path.basename(a.cache)}   n = {n}   "
      f"a_scale = {a.a_scale}")
    P(f"# truth-free a_i = sigma p_fit (1 - vgf): median {np.median(ai):.5f}, "
      f"mean {ai.mean():.5f};  {int(bad.sum())} candidates ({100.*bad.mean():.3f} %) "
      f"hit the 1 - a q z > 0.3 guard")
    P("")
    P("## A = <q z> in the three variables, against the CF model")
    P("| statistic | A(z) fit sigma | A(x) TRUTH-FREE | A(z_bar) truth cells | A model |")
    P("|---|---|---|---|---|")
    for nm, T in (("mean", None),) + tuple((f"trim{t:g}", t) for t in TRIMS):
        s0 = np.ones(n, bool) if T is None else (np.abs(z) < T)
        A0 = float((q * z)[s0].mean())
        A1 = float((q * x)[s0].mean())
        A2 = float((q * zb)[s0].mean())
        # model: trimmed mean of the pooled density
        zg = np.linspace(-12, 12, 4801)
        pf = 0.5 * (phis[0] + np.conj(phis[1]))
        pmod = D.density(pf, zg)
        if T is None:
            Am = float(np.trapezoid(zg * pmod, zg))
        else:
            sl = np.abs(zg) < T
            Am = float(np.trapezoid(zg[sl] * pmod[sl], zg[sl])
                       / np.trapezoid(pmod[sl], zg[sl]))
        e = float((q * x)[s0].std(ddof=1) / np.sqrt(s0.sum()))
        P(f"| {nm} | {A0:+.5f} | {A1:+.5f} +- {e:.5f} | {A2:+.5f} | {Am:+.5f} |")
    for iu, u in enumerate(PROBES):
        A0 = float((q * z * np.exp(-u * z ** 2)).mean())
        A1 = float((q * x * np.exp(-u * x ** 2)).mean())
        A2 = float((q * zb * np.exp(-u * zb ** 2)).mean())
        Am = 0.5 * (Om[q > 0, iu].mean() - Om[q < 0, iu].mean())
        e = float((q * x * np.exp(-u * x ** 2)).std(ddof=1) / np.sqrt(n))
        P(f"| <z e^-{u:g}z^2> | {A0:+.5f} | {A1:+.5f} +- {e:.5f} | {A2:+.5f} "
          f"| {Am:+.5f} |")
    txt = "\n".join(L)
    print(txt)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w").write(txt + "\n")
        np.savez_compressed(a.out.replace(".txt", ".npz"),
                            ai=ai.astype(np.float32), x=x.astype(np.float32))


if __name__ == "__main__":
    main()
