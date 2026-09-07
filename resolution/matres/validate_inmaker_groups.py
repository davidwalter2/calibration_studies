#!/usr/bin/env python3
"""Audit a `cf_inmaker.py pairs --groups` cache against its own flat exponents.

The per-group and the flat exponents are the SAME per-step sums associated
differently, so summing the group rows of a candidate must reproduce its flat
row to float32 round-off. That is the one check that says the reader indexed
the maker's row-major `(group, tau)` block correctly; nothing else in the chain
would notice a transposition or an off-by-one in the CSR pointer.

Three things are measured and one is reported:

1. CLOSURE, per family: `max |sum_g S_g - S_flat|` and the same relative to
   `max |S_flat|`. The float32 storage floor is `eps_32 * max|S| * sqrt(ngrp)`,
   which is quoted alongside so "small" is not a judgement call.
2. THE HIT SHARES: `vg_other + sum_c v_c == vgf` is a tautology once
   `vg_other` is DEFINED as the remainder, so what is actually checked is that
   the remainder is non-negative and physically sized -- a negative one would
   mean the hit-class variances overshoot the total Gaussian share.
3. GROUP OCCUPANCY: how many of the parmtype-15 groups a candidate touches and
   which groups almost nobody touches, which is what decides whether all of
   them are constrainable.

usage:
  source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
  python3 validate_inmaker_groups.py --cache ../../fullscale/runs/gpairs_smoke2.npz
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FAMS = (("grp_ms", "Sms"),
        ("grp_io_re", "Sio_re"),
        ("grp_io_im", "Sio_im"),
        ("grp_rad_re", "Srad_re"),
        ("grp_rad_im", "Srad_im"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--maxn", type=int, default=0,
                   help="0 = every candidate in the cache")
    a = p.parse_args()
    d = np.load(a.cache, allow_pickle=True)
    keys = set(d.files)
    for k in ("grp_ptr", "grp_id", "hit_ptr", "hit_cls", "hit_v", "vg_other",
              "group_names", "hit_classes", "families"):
        if k not in keys:
            raise SystemExit(f"{a.cache} has no `{k}`: it was not written with "
                             f"`cf_inmaker.py pairs --groups`")
    ptr = d["grp_ptr"].astype(np.int64)
    gid = d["grp_id"].astype(np.int64)
    n = len(ptr) - 1
    N = n if not a.maxn else min(a.maxn, n)
    gnames = [str(x) for x in d["group_names"]]
    ngroups = len(gnames)
    nt = len(d["tgrid"])
    print(f"{a.cache}")
    print(f"  {n} candidates, {nt} tau points, {ngroups} material groups, "
          f"{len(d['hit_classes'])} hit classes; auditing {N}")

    # ---- 1. closure ----------------------------------------------------
    print("\n=== 1. sum_g S_g  vs  the flat exponent ===")
    print(f"  {'family':<10s} {'max |sum_g - flat|':>20s} {'rel to max|S|':>15s} "
          f"{'float32 floor':>15s} {'verdict':>10s}")
    seg = np.repeat(np.arange(N), np.diff(ptr)[:N])
    rows = slice(0, int(ptr[N]))
    ok = True
    for gk, fk in FAMS:
        if "S" + gk not in keys or fk not in keys:
            print(f"  {gk:<10s} absent")
            continue
        Sg = np.asarray(d["S" + gk][rows], dtype=np.float64)
        acc = np.zeros((N, nt))
        np.add.at(acc, seg, Sg)
        flat = np.asarray(d[fk][:N], dtype=np.float64)
        dev = np.abs(acc - flat)
        scale = max(np.abs(flat).max(), 1e-300)
        mult = np.sqrt(np.diff(ptr)[:N].max())
        floor = np.finfo(np.float32).eps * scale * mult
        good = dev.max() <= 20.0 * floor
        ok &= good
        print(f"  {gk:<10s} {dev.max():20.4e} {dev.max()/scale:15.4e} "
              f"{floor:15.4e} {'ok' if good else 'FAIL':>10s}")

    if "grp_closure" in keys:
        c = np.asarray(d["grp_closure"][:N], dtype=np.float64)
        print(f"  the MAKER's own `cfmass_grp_closure` (same quantity, "
              f"computed in float64 before storage): max {np.nanmax(c):.4e}, "
              f"median {np.nanmedian(c):.4e}")

    # ---- 2. hit shares --------------------------------------------------
    print("\n=== 2. the Gaussian hit shares ===")
    hp = d["hit_ptr"].astype(np.int64)
    hv = np.asarray(d["hit_v"], dtype=np.float64)
    vgf = np.asarray(d["vgf"], dtype=np.float64)
    vgo = np.asarray(d["vg_other"], dtype=np.float64)
    ssum = np.zeros(n)
    if hv.size:
        np.add.at(ssum, np.repeat(np.arange(n), np.diff(hp)), hv)
    tot = vgo + ssum
    print(f"  max |vg_other + sum_c v_c - vgf| = {np.abs(tot - vgf).max():.4e} "
          f"(a tautology: vg_other is DEFINED as the remainder -- this only "
          f"catches a mis-assembled CSR)")
    frac = np.divide(ssum, np.maximum(vgf, 1e-300))
    print(f"  sum_c v_c / vgf : median {np.median(frac):.4f}, "
          f"q01 {np.quantile(frac,0.01):.4f}, q99 {np.quantile(frac,0.99):.4f}")
    nneg = int((vgo < 0).sum())
    print(f"  vg_other < 0 : {nneg} of {n} ({100.0*nneg/max(n,1):.4f} %)"
          + ("  <-- the hit classes overshoot the total Gaussian share"
             if nneg else "  (none, as it must be)"))
    print(f"  hit rows/candidate: mean {np.diff(hp).mean():.2f}, "
          f"min {np.diff(hp).min()}, max {np.diff(hp).max()}")

    # ---- 3. occupancy ----------------------------------------------------
    print("\n=== 3. material-group occupancy ===")
    mult = np.diff(ptr)
    print(f"  groups/candidate: mean {mult.mean():.2f}, median "
          f"{np.median(mult):.0f}, p1 {np.percentile(mult,1):.0f}, "
          f"p99 {np.percentile(mult,99):.0f}, max {mult.max()} of {ngroups}")
    cnt = np.bincount(gid, minlength=ngroups)
    occ = cnt / max(n, 1)
    order = np.argsort(occ)
    print(f"  the 8 LEAST occupied groups (fraction of candidates that touch "
          f"them at all):")
    for g in order[:8]:
        print(f"    {g:3d} {gnames[g]:<32s} {occ[g]:.6f}")
    dead = [g for g in range(ngroups) if cnt[g] == 0]
    print(f"  groups touched by NOBODY: {len(dead)}"
          + (": " + ", ".join(gnames[g] for g in dead) if dead else ""))
    thin = [g for g in range(ngroups) if 0 < occ[g] < 0.01]
    print(f"  groups touched by < 1 % of candidates: {len(thin)}"
          + (": " + ", ".join(gnames[g] for g in thin) if thin else ""))

    print("\n" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
