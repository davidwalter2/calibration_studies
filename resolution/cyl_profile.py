#!/usr/bin/env python3
"""Differential influence-weight profile w(u) from the nested-cylinder probe.

The cylinders r < 120u, |z| < 300u are nested, so the probe at u measures the
CUMULATIVE quantities

    M(u)      = sum_{steps inside u} mu_i        (from the dEref response)
    Lambda(u) = sum_{steps inside u} w_i mu_i    (from the p_fit response)

and the shell between two adjacent cylinders gives the differential weight

    w_j = [Lambda(u_{j+1}) - Lambda(u_j)] / [M(u_{j+1}) - M(u_j)]

computed per leg as a difference of paired quantities, so it needs no model.
The outermost cylinder (u = 1.4, everything) must reproduce the global
T_sys measured independently from CVH_DEDX_SCALE -- that is the closure.

Output feeds weighted_gap.py, which maps a cleanprop plane at (refglobr,
refglobz) onto u = max(r/120, |z|/300).

usage: python cyl_profile.py [--tag 260814cyl] [--ntasks 8]
"""
import argparse
import glob
import json
import os

import numpy as np
import uproot

CEPH = "/ceph/submit/data/user/d/david_w/ZMass/cvh"
RMAX, ZMAX = 120.0, 300.0

BRANCHES = ["run", "lumi", "event",
            "Muplus_pt", "Muplus_eta", "Muminus_pt", "Muminus_eta",
            "Muplusgen_pt", "Muplusgen_eta", "Muminusgen_pt", "Muminusgen_eta",
            "Muplus_dEref", "Muminus_dEref"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="260814cyl")
    p.add_argument("--nom-tag", default="260813")
    p.add_argument("--ntasks", type=int, default=8)
    p.add_argument("--eps", type=float, default=0.693147)
    p.add_argument("--etamax", type=float, default=None,
                   help="restrict to |eta| below this (for the eta dependence)")
    p.add_argument("--out", default="runs/cyl_weights.json")
    return p.parse_args()


def load(pattern, ntasks):
    fs = [f for f in sorted(glob.glob(pattern))
          if os.path.exists(os.path.join(os.path.dirname(f), ".complete"))][:ntasks]
    if not fs:
        return None
    cols = {k: [] for k in BRANCHES}
    for fn in fs:
        a = uproot.open(fn)["tree"].arrays(BRANCHES, library="np")
        for k in BRANCHES:
            cols[k].append(np.asarray(a[k], dtype=np.float64))
    d = {k: np.concatenate(v) for k, v in cols.items()}
    d["key"] = np.array([f"{int(r)}:{int(l)}:{int(e)}:{g:.5f}"
                         for r, l, e, g in zip(d["run"], d["lumi"], d["event"],
                                               d["Muplusgen_pt"])])
    return d


def legs(d, idx):
    p = np.concatenate([d["Muplus_pt"][idx] * np.cosh(d["Muplus_eta"][idx]),
                        d["Muminus_pt"][idx] * np.cosh(d["Muminus_eta"][idx])])
    e = np.concatenate([d["Muplus_dEref"][idx], d["Muminus_dEref"][idx]])
    eta = np.concatenate([np.abs(d["Muplusgen_eta"][idx]),
                          np.abs(d["Muminusgen_eta"][idx])])
    pg = np.concatenate([d["Muplusgen_pt"][idx] * np.cosh(d["Muplusgen_eta"][idx]),
                         d["Muminusgen_pt"][idx] * np.cosh(d["Muminusgen_eta"][idx])])
    return p, e, eta, pg


def shell_weight(dp, de, nboot, rng):
    """Ratio-of-sums on legs that actually cross the shell, after clipping.

    The clip is mandatory and RELATIVE: ~0.8 % of legs land in a different
    local minimum under the perturbed reference and reach |dp/de| ~ 1e3, which
    is enough to move a ratio of sums by 10 % (measured 2026-08-14)."""
    floor = 0.01 * np.median(de[de > 0]) if np.any(de > 0) else 1e-7
    m = np.isfinite(dp) & np.isfinite(de) & (de > floor)
    if m.sum() < 50:
        return np.nan, np.nan, 0
    r = dp[m] / de[m]
    iqr = np.quantile(r, 0.84) - np.quantile(r, 0.16)
    keep = np.abs(r - np.median(r)) < 5.0 * max(iqr, 1e-3)
    a, b = dp[m][keep], de[m][keep]
    w = float(a.sum() / b.sum())
    idx = rng.integers(0, len(a), (nboot, len(a)))
    bs = a[idx].sum(axis=1) / b[idx].sum(axis=1)
    return w, float(np.std(bs)), int(keep.sum())


def main():
    args = parse_args()
    rng = np.random.default_rng(23)
    scale = np.expm1(args.eps)

    nom = load(f"{CEPH}/resolution_transmission_{args.nom_tag}_s1000/task_*/globalcor_0.root",
               args.ntasks)
    dirs = sorted(glob.glob(f"{CEPH}/resolution_cylprobe_{args.tag}_u*"))
    us, data = [], {}
    for dn in dirs:
        u = int(dn.split("_u")[-1]) / 1000.0
        if u <= 0:
            continue           # the R=0 inertness run
        d = load(f"{dn}/task_*/globalcor_0.root", args.ntasks)
        if d is not None:
            us.append(u)
            data[u] = d
    us = sorted(us)
    print(f"nominal n={len(nom['key'])}; cylinders u = "
          + ", ".join(f"{u:.3f}" for u in us))

    common = set(nom["key"])
    for u in us:
        common &= set(data[u]["key"])
    common = np.array(sorted(common))
    print(f"key-matched candidates common to all: {len(common)}\n")

    pn = {k: i for i, k in enumerate(nom["key"])}
    i0 = np.array([pn[k] for k in common])
    p0, e0, eta, pg = legs(nom, i0)
    sel = np.ones(len(p0), bool)
    if args.etamax is not None:
        sel = eta < args.etamax
        print(f"|eta| < {args.etamax}: {sel.sum()} / {len(sel)} legs\n")

    P, E = {}, {}
    for u in us:
        pp = {k: i for i, k in enumerate(data[u]["key"])}
        iu = np.array([pp[k] for k in common])
        P[u], E[u], _, _ = legs(data[u], iu)

    # cumulative closure
    print("CUMULATIVE (each cylinder against nominal):")
    print(f"{'u':>7} {'r<[cm]':>8} {'|z|<[cm]':>9} {'M(u) [MeV]':>11} {'M/Mtot':>8} "
          f"{'Lambda/M':>9} {'err':>7} {'n':>7}")
    Mtot = float(np.sum(e0[sel]))
    cum = {}
    for u in us:
        de = (E[u] - e0)[sel]
        dp = (P[u] - p0)[sel]
        w, we, n = shell_weight(dp, de, 200, rng)
        M = float(np.sum(de[np.isfinite(de)]) / scale)
        cum[u] = (M, w)
        print(f"{u:7.3f} {RMAX*u:8.1f} {ZMAX*u:9.1f} {1e3*M/max(sel.sum(),1):11.4f} "
              f"{M/Mtot:8.4f} {w:9.4f} {we:7.4f} {n:7d}")

    # differential shells
    print(f"\nDIFFERENTIAL shells (paired differences, per leg):")
    print(f"{'u_lo':>7} {'u_hi':>7} {'r range [cm]':>16} {'dM/Mtot':>9} "
          f"{'w':>9} {'err':>7} {'n':>7}")
    prof = []
    prev = None
    for u in us:
        if prev is None:
            dp = (P[u] - p0)[sel]
            de = (E[u] - e0)[sel]
            ulo = 0.0
        else:
            dp = (P[u] - P[prev])[sel]
            de = (E[u] - E[prev])[sel]
            ulo = prev
        w, we, n = shell_weight(dp, de, 200, rng)
        dM = float(np.sum(de[np.isfinite(de)]) / scale)
        prof.append(dict(u_lo=ulo, u_hi=u, u_mid=0.5 * (ulo + u), w=w, w_err=we,
                         dM_frac=dM / Mtot, n=n))
        print(f"{ulo:7.3f} {u:7.3f} {RMAX*ulo:7.1f} - {RMAX*u:6.1f} {dM/Mtot:9.4f} "
              f"{w:9.4f} {we:7.4f} {n:7d}")
        prev = u

    tot = sum(p["dM_frac"] * p["w"] for p in prof if np.isfinite(p["w"]))
    frac = sum(p["dM_frac"] for p in prof if np.isfinite(p["w"]))
    print(f"\nCLOSURE: sum_shells dM_frac = {frac:.4f} (must be 1)")
    print(f"         sum_shells dM_frac * w = {tot:.4f}  vs the outermost "
          f"cylinder's Lambda/M = {cum[us[-1]][1]:.4f}")
    print(f"         and vs the global T_pop from CVH_DEDX_SCALE = 0.5932")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(prof, fh, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
