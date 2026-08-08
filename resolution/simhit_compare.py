#!/usr/bin/env python3
"""Sim-hit vs reco-hit refit: does the momentum SCALE offset live in the hits?

The single-track closures were all run on RECONSTRUCTED hits, so every number
folded hit reconstruction and transport together. `fitSimHitPositions=True`
substitutes the PSimHit truth positions and leaves the covariances unchanged,
which makes this a controlled ablation: SAME estimator, one input changed.

Two things are read off, and they answer different questions.

  WIDTH.  sigma is unchanged (covariances untouched), so
              var(simhit) / var(reco) = 1 - f_hit
          gives the hit share of the q/p variance DIRECTLY. The CF infers the
          same quantity indirectly through vgf, so this is a cross-check of the
          formalism, not just of the numbers.

  SCALE.  Sim positions remove hit NOISE and hit BIAS (Lorentz angle, charge
          sharing, edge effects) while leaving transport untouched. So:
              offset survives  -> it is transport / field
              offset vanishes  -> it is hit reconstruction
          Those have completely different fixes, which is why this is worth a
          dedicated production.

Deliberately NOT done: shrinking the hit covariances to match the substituted
positions. That would change the ESTIMATOR (the weighting of hit terms against
MS kinks) as well as the data, and a shift could then come from either -- the
attribution would be lost. The resulting chi2/ndof ~ 0.12 is the expected
signature of the ablation, not a defect.

Estimator is the MEDIAN throughout: Dk/k = q z sigma p carries a factor p, so
the tail's lever grows with momentum and the mean is tail-dominated even for
muons (mean and median differ in SIGN on these samples).

usage:
  python simhit_compare.py --tag mugun_lowpt [--nfiles 0] [--maxchi2 0]
"""
import argparse
import glob
import os

import numpy as np
import uproot

CEPH = "/ceph/submit/data/user/d/david_w/ZMass/cvh"
BRANCHES = ["refParms", "refCov", "genParms", "chisqval", "ndof", "nValidHits", "genPt"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="mugun_lowpt")
    p.add_argument("--nfiles", type=int, default=0, help="0 = all")
    p.add_argument("--maxchi2", type=float, default=0., help="cut on chisqval/nValidHits; 0 = none")
    p.add_argument("--nboot", type=int, default=200)
    return p.parse_args()


def load(tag, nfiles):
    fs = sorted(glob.glob(f"{CEPH}/resolution_trackres_{tag}/task_*/globalcor_resclosure_0.root"))
    if nfiles:
        fs = fs[:nfiles]
    qp, qg, c0, cv, nd, nv, pt = [], [], [], [], [], [], []
    for fn in fs:
        try:
            t = uproot.open(fn)["tree"]
            a = t.arrays(BRANCHES, library="np")
        except Exception:
            continue
        n = len(a["refCov"])
        if n == 0:
            continue
        qp.append(np.array([a["refParms"][i][0] for i in range(n)]))
        qg.append(np.array([a["genParms"][i][0] for i in range(n)]))
        c0.append(np.array([a["refCov"][i][0] for i in range(n)]))
        cv.append(np.asarray(a["chisqval"], float))
        nd.append(np.asarray(a["ndof"], float))
        nv.append(np.asarray(a["nValidHits"], float))
        pt.append(np.asarray(a["genPt"], float))
    if not qp:
        return None
    d = dict(qp=np.concatenate(qp), qg=np.concatenate(qg), c0=np.concatenate(c0),
             cv=np.concatenate(cv), nd=np.concatenate(nd),
             nv=np.concatenate(nv), pt=np.concatenate(pt))
    g = (d["qg"] != 0) & (d["c0"] > 0) & np.isfinite(d["qp"])
    return {k: v[g] for k, v in d.items()}, len(fs)


def med_err(x, nboot, rng):
    m = float(np.median(x))
    bs = np.array([np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(nboot)])
    return m, float(bs.std())


def main():
    args = parse_args()
    rng = np.random.default_rng(20260808)
    out = {}
    for lab, tag in (("RECO hits", args.tag), ("SIM hits ", f"{args.tag}_simhit")):
        r = load(tag, args.nfiles)
        if r is None:
            print(f"{lab}: no files for {tag}")
            return
        d, nf = r
        if args.maxchi2 > 0:
            s = d["cv"] / np.maximum(d["nv"], 1.) < args.maxchi2
            d = {k: v[s] for k, v in d.items()}
        sig = np.sqrt(d["c0"])
        z = (d["qp"] - d["qg"]) / sig
        dkk = (d["qp"] - d["qg"]) / d["qg"]          # = -Dp/p
        lo, hi = np.percentile(z, [15.865, 84.135])
        rob = 0.5 * (hi - lo)
        mdk, edk = med_err(dkk, args.nboot, rng)
        out[lab] = dict(n=len(z), nf=nf, rob=rob, mz=float(np.median(z)),
                        mdk=mdk, edk=edk,
                        chi2=float(np.median(d["cv"] / np.maximum(d["nd"], 1.))))
        o = out[lab]
        print(f"{lab}  files={o['nf']:3d} n={o['n']:7d}  rob68(z)={o['rob']:.4f}  "
              f"med(z)={o['mz']:+.4f}  CVH chi2/ndof={o['chi2']:.4f}  "
              f"median Dk/k = {o['mdk']*1e4:+8.3f} +- {o['edk']*1e4:.3f}  x1e-4")

    a, b = out["RECO hits"], out["SIM hits "]
    ratio = (b["rob"] / a["rob"]) ** 2
    print(f"\n  var(simhit)/var(reco) = {ratio:.4f}  ->  HIT SHARE of the q/p "
          f"variance f_hit = {1 - ratio:.4f}")
    d = b["mdk"] - a["mdk"]
    e = np.hypot(a["edk"], b["edk"])
    print(f"  scale shift (SIM - RECO) = {d*1e4:+.3f} +- {e*1e4:.3f} x1e-4"
          f"   ({abs(d)/e:.1f} sigma)")
    if abs(d) > 3 * e:
        print("  => the offset MOVES with the hits: hit reconstruction (CPE / "
              "Lorentz angle / charge sharing), not transport.")
    else:
        print("  => the offset does NOT move: it is transport / field, not hits.")


if __name__ == "__main__":
    main()
