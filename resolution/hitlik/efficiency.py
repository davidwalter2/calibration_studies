#!/usr/bin/env python3
"""Does the full PDF actually CONSTRAIN better?  The sandwich, not the claim.

The information ratio `report.py` prints compares the CF's Fisher information
with the Gaussian MODEL's Fisher information -- i.e. the error the Gaussian
term CLAIMS.  That is not the decision-relevant number.  Under heavy-tailed
data the Gaussian-likelihood estimator of a width is a weighted sum of `z^2`,
whose variance is governed by the FOURTH moment of the real residual, not the
second; for Moliere/Landau that fourth moment is large and cut-dependent.  So
the Gaussian's ACTUAL error is the sandwich

    S_G = (H_G + P)^-1  J_G  (H_G + P)^-1 ,

with `H_G` the Gaussian term's own curvature ON THE REAL DATA, `J_G` its
score covariance ON THE REAL DATA (both at MC truth), and `P` the prior
curvature both terms carry in every fit.  `sqrt(diag S_G)` is what the
Gaussian estimate would actually scatter by; `sqrt(diag (H_G+P)^-1)` is what
the fit would report.

The CF arm gets the same treatment.  If the CF model describes the data then
`H_CF ~ J_CF` and its sandwich equals its quoted error -- which is itself a
test of the model, printed as `S/Q`.

THE ANSWER is

    efficiency = sigma_Gauss,actual^2 / sigma_CF,actual^2

    > 1  the full PDF constrains better by that factor in variance.

An empirical version of the same is available two ways: the one-step
BOOTSTRAP over the stored per-batch gradients (algebraically the sandwich at
one Newton step, so it validates the linearisation and the batch
independence, not the sandwich itself), and -- with `--subfits` -- the spread
of REAL rabbit fits over DISJOINT subsamples, which assumes nothing.
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.dirname(_HERE)
for _p in (_HERE, _RES, os.path.join(_RES, "matres")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def psd_inv(A, rtol=1e-12):
    w, V = np.linalg.eigh(A)
    keep = w > max(rtol * w.max(), 0.0)
    return (V[:, keep] / w[keep]) @ V[:, keep].T, int(keep.sum()), w


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fisher", required=True,
                   help="a fisher_cmp.py output carrying BOTH H_* and J_*")
    p.add_argument("--cset", default="0123")
    p.add_argument("--arms", nargs="+", default=["cf", "gauss", "gaussq"])
    p.add_argument("--ref", default="cf")
    p.add_argument("--hit-prior", type=float, default=1.0)
    p.add_argument("--prior-power", type=float, default=1.0,
                   help="exponent on the material tier prior: 2 for a "
                        "WHITENED card, where 1 prior sigma is gprior**2 in "
                        "card units (1 = the historical, too-loose, value)")
    p.add_argument("--nboot", type=int, default=2000)
    p.add_argument("--ntrk", type=int, default=20000)
    p.add_argument("--scale-to", type=int, default=0,
                   help="also quote every sigma scaled to this many tracks")
    p.add_argument("--groups", default="/work/submit/david_w/ZMass/"
                   "CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/data/"
                   "materialGroups50.txt")
    p.add_argument("--top", type=int, default=14)
    p.add_argument("-o", "--output", default=None)
    a = p.parse_args()

    d = np.load(a.fisher, allow_pickle=True)
    params = [str(s) for s in d["params"]]
    npar = len(params)
    import groups as G
    gnames, gpri = G.group_param_names(42, a.groups)
    prior_of = dict(zip(gnames, gpri))
    # THE PRIOR MUST BE IN CARD UNITS.  A whitened card carries the material
    # parameter in units of its own tier prior, so 1 prior sigma is
    # `gprior**2` in card units, not `gprior` (`make_*_card.py`:
    # `gprior_card = gpriors * gscale`).  With `--prior-power 1` the prior
    # matrix `P` is 1/gprior too LOOSE -- typically a factor 20 -- which
    # leaves the "marginal" numbers effectively prior-free and inflates the
    # material efficiency ratios.  Default 1 preserves every number produced
    # before 2026-09-11; pass 2 for a whitened card.
    pv = np.array([prior_of.get(q, np.nan) ** a.prior_power
                   if q in prior_of else
                   (a.hit_prior if q.startswith("hitres_") else np.inf)
                   for q in params])
    P = np.diag(np.where(np.isfinite(pv), 1.0 / np.maximum(pv, 1e-300) ** 2, 0.0))

    res = {}
    S = {}
    print("=" * 108)
    print("THE GAUSSIAN TERM'S ACTUAL ERROR (sandwich) vs THE ONE IT QUOTES")
    print(f"   {a.ntrk} tracks, components '{a.cset}', at MC truth, with the "
          f"parmtype-15 tier priors and {a.hit_prior:g} on every hit class")
    print("=" * 108)
    for arm in a.arms:
        H = np.asarray(d[f"H_{arm}_{a.cset}"], float)
        J = np.asarray(d[f"J_{arm}_{a.cset}"], float)
        if not np.any(H):
            sys.exit(f"H_{arm}_{a.cset} is empty -- rerun fisher_cmp.py "
                     "WITHOUT --no-hessian")
        A = H + P
        C, rk, w = psd_inv(A)
        sq = np.sqrt(np.maximum(np.diag(C), 0.0))          # quoted
        Sw = C @ J @ C
        sa = np.sqrt(np.maximum(np.diag(Sw), 0.0))         # actual
        # one-step bootstrap over the stored per-batch gradients
        sb = np.full(npar, np.nan)
        if f"G_{arm}_{a.cset}" in d.files and a.nboot:
            Gm = np.asarray(d[f"G_{arm}_{a.cset}"], float)
            M = Gm.shape[0]
            rng = np.random.default_rng(20260910)
            cnt = rng.multinomial(M, np.full(M, 1.0 / M), size=a.nboot)
            gb = cnt @ Gm                                   # (nboot, npar)
            th = -(gb - Gm.sum(axis=0)) @ C.T
            sb = th.std(axis=0, ddof=1)
        # PRIOR-FREE, standalone (every other parameter fixed): quoted
        # 1/sqrt(H_pp), actual sqrt(J_pp)/H_pp.  Well defined for every
        # parameter and free of the prior, which otherwise compresses BOTH
        # sandwich and quoted for a group the term barely measures.
        hd = np.maximum(np.diag(H), 1e-300)
        aq = 1.0 / np.sqrt(hd)
        aa = np.sqrt(np.maximum(np.diag(J), 0.0)) / hd
        S[arm] = dict(quoted=sq, actual=sa, boot=sb, rank=rk, aq=aq, aa=aa,
                      eigmin=float(w.min()), eigmax=float(w.max()))
        res[f"sigma_alone_quoted_{arm}"] = aq
        res[f"sigma_alone_actual_{arm}"] = aa
        res[f"sigma_quoted_{arm}"] = sq
        res[f"sigma_actual_{arm}"] = sa
        res[f"sigma_boot_{arm}"] = sb
        print(f"   {arm:<8} H+P rank {rk}/{npar}, eig {w.min():+.3e} .. "
              f"{w.max():.3e};  median sandwich/quoted over the informative "
              f"parameters: ", end="")
        infm = np.isfinite(pv) & (sq < 0.98 * pv)
        print(f"{np.median((sa / sq)[infm]):.3f}"
              f"   (bootstrap/sandwich {np.median((sb / sa)[infm]):.4f})")

    ref = a.ref
    infm = np.isfinite(pv) & (S[ref]["quoted"] < 0.98 * pv)
    order = np.argsort(S[ref]["quoted"] / np.where(np.isfinite(pv), pv, 1.0))
    mats = [i for i in order if params[i].startswith("material_") and infm[i]]
    hits = [i for i in order if params[i].startswith("hitres_")]
    sc = (np.sqrt(a.ntrk / a.scale_to) if a.scale_to else 1.0)

    for lab, idxs in (("MATERIAL GROUPS the term constrains", mats),
                      ("HIT CLASSES", hits)):
        print()
        print("-" * 108)
        print(lab + (f"   (sigma also scaled to {a.scale_to} tracks)"
                     if a.scale_to else ""))
        print("-" * 108)
        hdr = f"{'parameter':<26}"
        for arm in a.arms:
            hdr += f"{arm + ' quoted':>14}{arm + ' ACTUAL':>14}{'S/Q':>7}"
        hdr += f"{'EFF marg':>10}{'EFF alone':>11}"
        print(hdr)
        for i in idxs[: a.top]:
            row = f"{params[i]:<26}"
            for arm in a.arms:
                row += (f"{S[arm]['quoted'][i]*sc:>14.4f}"
                        f"{S[arm]['actual'][i]*sc:>14.4f}"
                        f"{S[arm]['actual'][i]/max(S[arm]['quoted'][i],1e-300):>7.2f}")
            eff = (S["gaussq"]["actual"][i] / S[ref]["actual"][i]) ** 2 \
                if "gaussq" in S else np.nan
            effa = (S["gaussq"]["aa"][i] / S[ref]["aa"][i]) ** 2 \
                if "gaussq" in S else np.nan
            row += f"{eff:>10.3f}{effa:>11.3f}"
            print(row)
        if "gaussq" in S and len(idxs):
            e = np.array([(S["gaussq"]["actual"][i] / S[ref]["actual"][i]) ** 2
                          for i in idxs])
            print(f"   EFFICIENCY sigma^2(Gauss, ACTUAL) / sigma^2(CF, actual):"
                  f"  median {np.median(e):.3f}   "
                  f"16-84 % {np.percentile(e,16):.3f}-{np.percentile(e,84):.3f}"
                  f"   min {e.min():.3f}  max {e.max():.3f}")
            ea = np.array([(S["gaussq"]["aa"][i] / S[ref]["aa"][i]) ** 2
                           for i in idxs])
            print(f"   PRIOR-FREE (standalone) efficiency:            "
                  f"  median {np.median(ea):.3f}   "
                  f"16-84 % {np.percentile(ea,16):.3f}-"
                  f"{np.percentile(ea,84):.3f}")
            q = np.array([(S["gaussq"]["quoted"][i] / S[ref]["quoted"][i]) ** 2
                          for i in idxs])
            print(f"   (the QUOTED-vs-QUOTED ratio, i.e. what the Gaussian "
                  f"CLAIMS: median {np.median(q):.3f})")
            sq_cf = np.array([S[ref]["actual"][i] / S[ref]["quoted"][i]
                              for i in idxs])
            sq_g = np.array([S["gaussq"]["actual"][i] / S["gaussq"]["quoted"][i]
                             for i in idxs])
            aq_cf = np.array([S[ref]["aa"][i] / S[ref]["aq"][i] for i in idxs])
            aq_g = np.array([S["gaussq"]["aa"][i] / S["gaussq"]["aq"][i]
                             for i in idxs])
            print(f"   sandwich/quoted, PRIOR-FREE: CF {np.median(aq_cf):.3f} "
                  f"(1.000 = the model is right), Gaussian "
                  f"{np.median(aq_g):.3f}")
            res["effalone_" + lab.split()[0].lower()] = ea
            res["efficiency_" + lab.split()[0].lower()] = e

    print()
    print("READING: EFFICIENCY > 1 means the full PDF constrains that "
          "parameter better,\n by that factor in VARIANCE, than the Gaussian "
          "chi2 actually does -- as opposed\n to the factor the Gaussian "
          "claims.  `S/Q` is each arm's own sandwich-over-quoted:\n it is ~1 "
          "when the model describes the data and > 1 when it does not.")

    if a.output:
        res["params"] = np.array(params, dtype=object)
        res["prior"] = pv
        np.savez_compressed(a.output, **res)
        print(f"\nwrote {a.output}")


if __name__ == "__main__":
    main()
