#!/usr/bin/env python3
"""Two measurements on the MS-ablated gun samples, from ONE sim-hit refit.

(A) theta_c -- the hit-resolution truth ratio Var(dxrecsim/dxerr). The CF says
    the A1/A2 excess (obs/model 1.135 in width) must live in the hit term,
    because hits are 93% of the model variance there and sum_b v_b = refCov00
    exactly. Prediction to falsify: theta_c(A1) ~ 1.27 against 0.983 nominal.

(B) the delta-ray recoil block. In A2 `msc` and `CoulombScat` are off, so the
    ONLY thing deflecting the muon is delta emission. simlocaldxdz -
    simlocaldxdzprop is the true track's accumulated deflection from the
    gen-propagated helix, so A2/nominal of that IS delta/MS.
    PRE-REGISTERED (before the refit landed, from real msmoliv records):
       variance ratio 0.0575  (uncapped -- the sim has no 50 MeV cap)
       rob68 ratio    0.118   (the delta core is narrow; its variance is in
                               rare hard kicks, so the two differ by ~2x)
    Both must hold or the block's rate / 1/T^2 spectrum is wrong.
"""
import argparse, glob, os, datetime
import numpy as np
import uproot

SENT = -98.0
SAMPLES = {"nominal": "hitres_mugun_lowpt",
           "A1 noms": "hitres_mugun_lowpt_noms",
           "A2 nomsrad": "hitres_mugun_lowpt_nomsrad"}
CEPH = "/ceph/submit/data/user/d/david_w/ZMass/cvh"


def rob(x, f=0.68):
    """half-width of the central f interval"""
    if len(x) < 50:
        return np.nan
    lo, hi = np.percentile(x, [50 - 50 * f, 50 + 50 * f])
    return 0.5 * (hi - lo)


def load(tag, nfiles, branches):
    fs = sorted(glob.glob(f"{CEPH}/{tag}/task_*/globalcor_resclosure_0.root"))[:nfiles]
    out = {b: [] for b in branches}
    nread = 0
    for fn in fs:
        try:
            t = uproot.open(fn)["tree"]
            have = [b for b in branches if b in t.keys()]
            if len(have) != len(branches):
                continue
            a = t.arrays(branches, library="np")
        except Exception:
            continue
        for b in branches:
            out[b].append(np.concatenate([np.asarray(v).ravel() for v in a[b]])
                          if a[b].dtype == object else np.asarray(a[b]).ravel())
        nread += 1
    if not nread:
        return None, 0
    return {b: np.concatenate(v) for b, v in out.items()}, nread


def part_a(nfiles):
    print("(A) theta_c = Var(dxrecsim / dxerr)\n")
    print(f"  {'sample':<12} {'nfile':>5} {'nhit':>9} {'rob68 pull':>11} "
          f"{'theta_c rob':>12} {'theta_c var':>12}")
    res = {}
    for lab, tag in SAMPLES.items():
        d, nf = load(tag, nfiles, ["dxrecsim", "dxerr"])
        if d is None:
            print(f"  {lab:<12} {'-':>5}  (no files yet at {tag})")
            continue
        m = (d["dxrecsim"] > SENT) & (d["dxerr"] > 0)
        p = d["dxrecsim"][m] / d["dxerr"][m]
        r = rob(p) / 0.9944578832097535           # rob68 -> Gaussian sigma
        # variance on a 5-sigma trim: the raw variance is tail-dominated and
        # not comparable between samples with different tail content
        tr = p[np.abs(p) < 5 * r]
        print(f"  {lab:<12} {nf:5d} {m.sum():9d} {rob(p):11.4f} "
              f"{r**2:12.4f} {tr.var():12.4f}")
        res[lab] = r ** 2
    if "nominal" in res:
        print(f"\n  nominal theta_c should reproduce the section-14 value 0.9828")
        for k in ("A1 noms", "A2 nomsrad"):
            if k in res:
                print(f"  {k:<12} / nominal = {res[k]/res['nominal']:.4f}   "
                      f"(1.29 would explain the whole CF excess)")
    return res


def part_b(nfiles):
    print("\n(B) delta-ray recoil: accumulated deflection simlocaldxdz - simlocaldxdzprop\n")
    # simlocaldxdzprop / simlocalxprop are NOT filled in this configuration
    # (all -99), so use dxsimgen: the sim hit minus the GEN-PROPAGATED position,
    # i.e. the same accumulated deflection expressed as a displacement. MS and
    # delta kicks happen at the same planes with the same lever arms, so the
    # A2/nominal ratio is still exactly delta/MS.
    br = ["dxsimgen"]
    print(f"  {'sample':<12} {'nhit':>9} {'rob68':>11} {'trim var':>12}")
    got = {}
    for lab, tag in SAMPLES.items():
        d, nf = load(tag, nfiles, br)
        if d is None:
            print(f"  {lab:<12}  (no files yet)")
            continue
        m = d["dxsimgen"] > SENT
        dd = d["dxsimgen"][m]
        r = rob(dd)
        tr = dd[np.abs(dd) < 5 * r / 0.9945]
        print(f"  {lab:<12} {m.sum():9d} {r:11.4e} {tr.var():12.4e}")
        got[lab] = (r, tr.var())
    if "nominal" in got:
        print(f"\n  {'ratio to nominal':<20} {'rob68':>9} {'variance':>10}   PREDICTED")
        for k in ("A1 noms", "A2 nomsrad"):
            if k in got:
                print(f"  {k:<20} {got[k][0]/got['nominal'][0]:9.4f} "
                      f"{got[k][1]/got['nominal'][1]:10.4f}   0.118 / 0.0575")
    return got


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nfiles", type=int, default=80)
    ap.add_argument("--part", choices=["a", "b", "both"], default="both")
    a = ap.parse_args()
    od = datetime.date.today().strftime("%y%m%d")
    print(f"# hit-resolution ablation, {od}, up to {a.nfiles} files/sample\n")
    if a.part in ("a", "both"):
        part_a(a.nfiles)
    if a.part in ("b", "both"):
        part_b(a.nfiles)
