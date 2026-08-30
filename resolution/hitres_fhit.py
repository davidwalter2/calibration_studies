#!/usr/bin/env python3
"""Does the per-hit pull EXPLAIN the track-level hit-share excess?

Two independent measurements of the same thing exist now, at different levels:

  TRACK level (the sim-position ablation): substituting truth positions leaves
    sigma untouched, so var(sim)/var(reco) = 1 - f_hit reads the hit share of
    the fitted q/p variance directly. Measured f_hit = 0.1248 (mu, pT 2-20)
    and 0.2645 (mu, pT 20-60).

  HIT level (this study): the pull core width per hit class, i.e. how much
    bigger the true hit error is than the assigned one, class by class.

The bridge is the fit's OWN influence weights. `resinfvarv` is
w_b^T dV_b w_b -- the contribution of resolution block b to Var(q/p) -- and
`resinfcov` = sum_b v_b = refCov(0,0) exactly. So the model's hit share is
sum over the parmtype-8/9 blocks of v_b, over refCov(0,0); and if the true
variance of block b is c_b times the assigned one, the corrected share is
sum c_b v_b / (sum c_b v_b + sum_{non-hit} v_b).

Feeding the measured c_b = core^2 (1.166 strip, 0.872 pixel x, 0.877 pixel y)
turns the hit-level measurement into a PREDICTION of the track-level number.
Agreement would mean the pull study accounts for the whole excess; a gap
would mean part of it lives somewhere the pull cannot see.

usage:
  python hitres_fhit.py --tag mugun_lowpt --cstrip 1.166 --cpixx 0.872 --cpixy 0.877
"""
import argparse
import glob
import os
import sys

import numpy as np


CEPH = "/ceph/submit/data/user/d/david_w/ZMass/cvh"


def run_tree_map(fn):
    """iidx -> (parmtype, subdet) from the per-run parameter tree."""
    import uproot
    f = uproot.open(fn)
    key = next((k for k in f.keys() if k.split(";")[0] == "runtree"), None)
    if key is None:
        raise SystemExit(f"{fn} has no runtree (fillRunTree must be on)")
    a = f[key].arrays(["iidx", "parmtype", "subdet"], library="np")
    m = {}
    for i, p, s in zip(a["iidx"], a["parmtype"], a["subdet"]):
        m[int(i)] = (int(p), int(s))
    return m


def main():
    import uproot
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="mugun_lowpt")
    p.add_argument("--subdir", default="resolution_trackres")
    p.add_argument("--nfiles", type=int, default=20)
    p.add_argument("--cstrip", type=float, default=1.166)
    p.add_argument("--cpixx", type=float, default=0.872)
    p.add_argument("--cpixy", type=float, default=0.877)
    args = p.parse_args()

    fs = [f for f in sorted(glob.glob(
        f"{CEPH}/{args.subdir}_{args.tag}/task_*/globalcor_resclosure_0.root"))
        if os.path.exists(os.path.join(os.path.dirname(f), ".complete"))]
    if args.nfiles:
        fs = fs[:args.nfiles]
    if not fs:
        raise SystemExit("no complete files")

    pm = run_tree_map(fs[0])
    tot = {"hitx": 0.0, "hity": 0.0, "other": 0.0}
    totc = 0.0
    ntrk = 0
    fh_obs, fh_cor = [], []
    for fn in fs:
        t = uproot.open(fn)["tree"]
        have = set(t.keys())
        for b in ("reseigidx", "resinfvarv", "resinfcov"):
            if b not in have:
                raise SystemExit(f"{fn} lacks {b}: the arm must run "
                                 f"doRes=True with fillGrads=True")
        a = t.arrays(["reseigidx", "resinfvarv", "resinfcov"], library="np")
        for idxs, vs, cov in zip(a["reseigidx"], a["resinfvarv"], a["resinfcov"]):
            if cov <= 0 or len(idxs) == 0:
                continue
            ntrk += 1
            hx = hy = ot = 0.0
            chx = chy = 0.0
            for i, v in zip(idxs, vs):
                pt, sd = pm.get(int(i), (-1, -1))
                if pt == 8:
                    hx += v
                    chx += v * (args.cpixx if sd <= 2 else args.cstrip)
                elif pt == 9:
                    hy += v
                    chy += v * args.cpixy
                else:
                    ot += v
            tot["hitx"] += hx; tot["hity"] += hy; tot["other"] += ot
            totc += cov
            s = hx + hy + ot
            if s > 0:
                fh_obs.append((hx + hy) / s)
                fh_cor.append((chx + chy) / (chx + chy + ot))
    fh_obs = np.array(fh_obs); fh_cor = np.array(fh_cor)
    print(f"{args.subdir}_{args.tag}: {len(fs)} files, {ntrk} tracks")
    print(f"  sum(v_b)/sum(refCov00) = {(tot['hitx']+tot['hity']+tot['other'])/totc:.6f}"
          f"   (identity check; 1.000000 means the blocks are complete)")
    print(f"  model hit share f_hit          = {np.mean(fh_obs):.4f}"
          f"  (median {np.median(fh_obs):.4f})")
    print(f"  with the measured c_b applied  = {np.mean(fh_cor):.4f}"
          f"  (median {np.median(fh_cor):.4f})")
    print(f"    c_strip = {args.cstrip}   c_pixel_x = {args.cpixx}"
          f"   c_pixel_y = {args.cpixy}")
    print(f"  ratio corrected/model          = {np.mean(fh_cor)/np.mean(fh_obs):.4f}")


if __name__ == "__main__":
    main()
