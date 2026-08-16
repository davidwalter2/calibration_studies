#!/usr/bin/env python3
"""Per-material-group (or nested-cylinder) influence weights of the CVH fit.

Closing measurement for NOTES_TRANSMISSION.md. The 2026-08-13 result was the
GLOBAL coherent transmission

    T_sys = d(p_fit at PCA) / d(assumed total energy loss) = 0.6127 +- 0.0003,

i.e. the material-weighted mean of the fit's per-step influence weights w_i.
That single number does not say how the weights are DISTRIBUTED, and the
distribution is what decides whether the mode/median reference fraction of the
UNWEIGHTED total loss (what `gate_refs.json` holds) is the right comparand for
ds_req, which is the gap of the WEIGHTED loss sum.

Here the shift is applied to one material group at a time (mean only, Q
untouched -- see MaterialGroupModel.cc), so per track

    dEref_g = applied loss in group g,   dp_fit_g = the fit's response

and w_g = dp_fit_g / dEref_g is that group's influence weight. Two closures:

  * Sum rule:  sum_g (loss_g * w_g) / sum_g loss_g  must reproduce the global
    T_sys measured independently from CVH_DEDX_SCALE. If it does not, the
    decomposition is wrong (overlapping groups, unclassified steps, ...).
  * Innermost material must give w = 1 (everything downstream) and outermost
    w = 0 (nothing downstream).

usage:
  python matgroup_weights.py [--tag 260814] [--nom-tag 260813] [--ntasks 8]
  python matgroup_weights.py --mode cyl --tag 260814cyl
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import uproot

CEPH = "/ceph/submit/data/user/d/david_w/ZMass/cvh"
RULES = ("/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/"
         "Analysis/HitAnalyzer/data/materialGroups50.txt")

BRANCHES = ["run", "lumi", "event",
            "Muplus_pt", "Muplus_eta", "Muminus_pt", "Muminus_eta",
            "Muplusgen_pt", "Muplusgen_eta", "Muminusgen_pt", "Muminusgen_eta",
            "Muplus_dEref", "Muminus_dEref"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", default="group", choices=("group", "cyl"))
    p.add_argument("--tag", default="260814")
    p.add_argument("--nom-tag", default="260813")
    p.add_argument("--ntasks", type=int, default=8)
    p.add_argument("--eps", type=float, default=0.693147)
    p.add_argument("--out", default="runs/matgroup_weights.json")
    return p.parse_args()


def group_names():
    names = {0: "other"}
    with open(RULES) as fh:
        for line in fh:
            f = line.split()
            if f and f[0] == "RULE":
                names[int(f[1])] = f[2]
    return names


def load(pattern, ntasks):
    fs = sorted(glob.glob(pattern))
    fs = [f for f in fs if os.path.exists(os.path.join(os.path.dirname(f), ".complete"))]
    fs = fs[:ntasks]
    if not fs:
        return None
    cols = {k: [] for k in BRANCHES}
    for fn in fs:
        try:
            a = uproot.open(fn)["tree"].arrays(BRANCHES, library="np")
        except Exception as exc:                          # noqa: BLE001
            print(f"  [warn] {fn}: {exc}")
            continue
        for k in BRANCHES:
            cols[k].append(np.asarray(a[k], dtype=np.float64))
    if not cols["run"]:
        return None
    d = {k: np.concatenate(v) for k, v in cols.items()}
    d["key"] = np.array([f"{int(r)}:{int(l)}:{int(e)}:{g:.5f}"
                         for r, l, e, g in zip(d["run"], d["lumi"], d["event"],
                                               d["Muplusgen_pt"])])
    return d


def legs(d, idx):
    p = np.concatenate([d["Muplus_pt"][idx] * np.cosh(d["Muplus_eta"][idx]),
                        d["Muminus_pt"][idx] * np.cosh(d["Muminus_eta"][idx])])
    e = np.concatenate([d["Muplus_dEref"][idx], d["Muminus_dEref"][idx]])
    pg = np.concatenate([d["Muplusgen_pt"][idx] * np.cosh(d["Muplusgen_eta"][idx]),
                         d["Muminusgen_pt"][idx] * np.cosh(d["Muminusgen_eta"][idx])])
    eta = np.concatenate([np.abs(d["Muplusgen_eta"][idx]), np.abs(d["Muminusgen_eta"][idx])])
    return p, e, pg, eta


def robust_ratio(dp, de, floor):
    """Median and population (ratio-of-sums) weight, on legs that actually
    traverse the probed material. `floor` is RELATIVE to the probe's own
    typical size, never absolute: groups differ by three orders of magnitude
    in thickness and a fixed floor would silently empty the thin ones."""
    m = np.isfinite(dp) & np.isfinite(de) & (de > floor)
    if m.sum() < 50:
        return np.nan, np.nan, np.nan, int(m.sum())
    r = dp[m] / de[m]
    iqr = np.quantile(r, 0.84) - np.quantile(r, 0.16)
    keep = np.abs(r - np.median(r)) < 5.0 * max(iqr, 1e-3)
    med = float(np.median(r[keep]))
    pop = float(dp[m][keep].sum() / de[m][keep].sum())
    # bootstrap the median
    rng = np.random.default_rng(3)
    rr = r[keep]
    bs = np.median(rr[rng.integers(0, len(rr), (200, len(rr)))], axis=1)
    return med, pop, float(bs.std()), int(keep.sum())


def main():
    args = parse_args()
    names = group_names()

    nom = load(f"{CEPH}/resolution_transmission_{args.nom_tag}_s1000/task_*/globalcor_0.root",
               args.ntasks)
    if nom is None:
        raise SystemExit("no nominal sample")
    print(f"nominal: n={len(nom['key'])} candidates ({args.ntasks} tasks)")

    if args.mode == "group":
        dirs = sorted(glob.glob(f"{CEPH}/resolution_matgroup_{args.tag}_g??"))
        keyof = lambda d: int(d[-2:])                     # noqa: E731
    else:
        dirs = sorted(glob.glob(f"{CEPH}/resolution_cylprobe_{args.tag}_u???"))
        keyof = lambda d: int(d[-3:])                     # noqa: E731

    scale = np.expm1(args.eps)    # applied loss = (e^eps - 1) * nominal loss
    print(f"probe eps = {args.eps:.6f}  (applied loss = {scale:.4f} x nominal "
          f"loss of the probed material)\n")

    rows = []
    hdr = (f"{'id':>4} {'name':>18} {'n':>7} {'loss [MeV]':>11} {'frac':>7} "
           f"{'w (med)':>9} {'err':>7} {'w (pop)':>9} {'contrib':>9}")
    print(hdr)
    print("-" * len(hdr))
    tot_loss = np.median(np.concatenate(
        [nom["Muplus_dEref"], nom["Muminus_dEref"]]))
    for dn in dirs:
        gid = keyof(dn)
        pr = load(f"{dn}/task_*/globalcor_0.root", args.ntasks)
        if pr is None:
            print(f"{gid:4d} {names.get(gid,'?'):>18}   [no complete tasks]")
            continue
        common = np.array(sorted(set(nom["key"]) & set(pr["key"])))
        if len(common) < 200:
            print(f"{gid:4d} {names.get(gid,'?'):>18}   [only {len(common)} matched]")
            continue
        pn = {k: i for i, k in enumerate(nom["key"])}
        pp = {k: i for i, k in enumerate(pr["key"])}
        i = np.array([pn[k] for k in common])
        j = np.array([pp[k] for k in common])
        p0, e0, pg, eta = legs(nom, i)
        p1, e1, _, _ = legs(pr, j)
        de = e1 - e0
        dp = p1 - p0
        # RELATIVE floor: 1% of this probe's own median positive shift
        pos = de[de > 0]
        floor = 0.01 * np.median(pos) if len(pos) > 50 else 1e-7
        med, pop, err, n = robust_ratio(dp, de, floor)
        loss = float(np.median(pos)) / scale if len(pos) else 0.0   # MeV-scale (GeV)
        frac = float(np.sum(de[np.isfinite(de)]) / scale) / float(np.sum(e0))
        # material-weighted contribution to the global T_sys.
        #
        # It MUST use the clipped w, not a raw sum of dp over all legs: ~0.8 %
        # of legs land in a different local minimum under the perturbed
        # reference and give |dp/de| up to 1300 (measured). Summing those
        # unclipped inflates the sum rule to 0.671 against a global T_sys of
        # 0.593 -- a 13 % "violation" that is pure outlier contamination and
        # not a property of the decomposition. The global number this is
        # compared against is itself clipped the same way.
        contrib = frac * pop if np.isfinite(pop) else np.nan
        rows.append(dict(id=gid, name=names.get(gid, str(gid)), n=n,
                         loss_mev=loss * 1e3, frac=frac, w_med=med, w_err=err,
                         w_pop=pop, contrib=contrib))
        print(f"{gid:4d} {names.get(gid,'?'):>18} {n:7d} {loss*1e3:11.4f} "
              f"{frac:7.4f} {med:9.4f} {err:7.4f} {pop:9.4f} {contrib:9.5f}")

    print("-" * len(hdr))
    fs = np.array([r["frac"] for r in rows])
    cs = np.array([r["contrib"] for r in rows])
    cs = np.where(np.isfinite(cs), cs, 0.0)
    print(f"{'SUM':>4} {'':>18} {'':>7} {tot_loss*1e3:11.4f} {fs.sum():7.4f} "
          f"{'':>9} {'':>7} {'':>9} {cs.sum():9.5f}")
    print(f"\nSUM RULE: sum_g frac_g = {fs.sum():.4f}  (must be 1 if the groups "
          f"tile all traversed material)")
    print(f"          sum_g frac_g*w_pop_g = {cs.sum():.4f}  (must equal the "
          f"global T_sys measured from CVH_DEDX_SCALE on the SAME tasks and "
          f"with the SAME clipping: T_pop = 0.5932)")
    if fs.sum() > 0:
        print(f"          ratio sum(contrib)/sum(frac) = {cs.sum()/fs.sum():.4f} "
              f"= material-weighted mean w")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
