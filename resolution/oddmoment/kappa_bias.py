#!/usr/bin/env python3
"""Mechanism 3: the second-order estimator bias of the GN track fit in q/p.

The quantity the mass needs is the relative bias of the CURVATURE MAGNITUDE,

    b = E[q (q/p_fit - q/p_gen)] / |q/p_gen| = E[|kappa_fit|]/|kappa_gen| - 1
      = <p_gen/p_fit> - 1                          (identically)

because ln m = 1/2 (ln p1 + ln p2) + g_ang and

    E[d ln|kappa|] = b - 1/2 sigma_rel^2 ,

so the mass picks up  alpha_3 = +1/2 sum_l b_l   from the LINEAR part of the
curvature bias (the -1/2 sigma_rel^2 piece is the Jensen term and is counted
separately).  Note the sign: a curvature over-estimate (b > 0) means p is
UNDER-estimated, so the mass comes out low: alpha_3 = -b for equal legs.

b is a raw momentum quantity: it is NOT touched by the pull-normalisation
defect (no 1/sigma anywhere).  It is however tail-sensitive, so both the
untrimmed mean (unbiased, noisy) and trimmed versions are quoted.
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
import decomp as D                                            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=300)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rng = np.random.default_rng(31)
    L = []; P = L.append
    P("## b = <p_gen/p_fit> - 1 (relative bias of |q/p|), single-track CVH fits")
    P("| sample | selection | n | b [1e-4] | boot | alpha_3 = -b [1e-3] |")
    P("|---|---|---|---|---|---|")
    for tag, p in (("mu gun pT 20-60", "runs/cf_trackres_mugun_ul16_260903x_m0_k0.npz"),
                   ("mu gun pT 2-20", "runs/cf_trackres_mugun_lowpt_260905d_m0_k0.npz")):
        d = D.load(p, need_model=False)
        z = d["z"]; sig = d["sigma"]
        q = np.sign(d["charge"]).astype(np.float64)
        eta = d["eta"].astype(np.float64)
        pgen = d["genpt"] * np.cosh(eta)
        rel = q * z * sig * pgen
        def bse(v):
            n = len(v)
            return float(np.array([v[rng.integers(0, n, n)].mean()
                                   for _ in range(a.nboot)]).std(ddof=1))
        for nm, msk in (("all", np.ones(len(z), bool)),
                        ("|z|<10", np.abs(z) < 10.),
                        ("|z|<5", np.abs(z) < 5.)):
            v = rel[msk]
            P(f"| {tag} | {nm} | {int(msk.sum())} | {1e4*v.mean():+.3f} "
              f"| {1e4*bse(v):.3f} | {-1e3*v.mean():+.4f} |")
        # differential in gen pT
        P("")
        P(f"### {tag}, in gen pT quintiles (untrimmed)")
        P("| gen pT | n | b [1e-4] | boot |")
        P("|---|---|---|---|")
        e = np.quantile(d["genpt"], np.linspace(0, 1, 6))
        b = np.clip(np.digitize(d["genpt"], e[1:-1]), 0, 4)
        for k in range(5):
            v = rel[b == k]
            P(f"| {e[k]:.2f}..{e[k+1]:.2f} | {len(v)} | {1e4*v.mean():+.3f} "
              f"| {1e4*bse(v):.3f} |")
        P("")
    P("The J/psi legs live at p ~ 3-15 GeV, i.e. the pT 2-20 sample: that row "
      "is the one that bounds alpha_3 for the candidate fits.  The single-track "
      "and two-track estimators are not identical, so this is a BOUND, not a "
      "subtraction.")
    txt = "\n".join(L)
    print(txt)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
