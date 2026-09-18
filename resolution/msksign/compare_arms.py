#!/usr/bin/env python3
"""Candidate-by-candidate comparison of two CVH two-track outputs that differ
by ONE switch, for the 2026-09-18 defect impact assessment.

Candidates are matched on (run, lumi, event) plus the ordinal within the
event; the match is verified on a quantity the switch cannot move (the input
track phi of the plus leg is not available, so the gen mass and the number of
hits are used as the witness).

  usage: compare_arms.py <A.root> <B.root> [--label-a ...] [--label-b ...]
         [--sigma-cols]  [--json out.json]

Reports, per branch group: the median and the 95th percentile of |B - A|,
relative where a scale exists, and in units of the per-candidate sigma where
one is exported.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import uproot


SCALARS = [
    "Jpsi_mass", "Jpsi_sigmamass", "Jpsi_massErr", "Jpsicons_mass",
    "Jpsikin_mass", "Jpsitrk_mass", "Jpsi_mass_unc",
    "Jpsi_x", "Jpsi_y", "Jpsi_z",
    "Jpsi_qoprefplus", "Jpsi_qoprefminus",
    "Jpsi_sigmarelplus", "Jpsi_sigmarelminus", "Jpsi_rhomom", "Jpsi_fang",
    "chisqval", "ndof", "deltachisqval",
    "niter", "niter_cons0", "edmval", "edmval_cons0", "edmvalref",
    "Jpsi_vtxres", "Jpsi_vtxsig", "Jpsi_vtxz", "Jpsi_vtxdchi2",
    "Jpsi_bsres", "Jpsi_bssig", "Jpsi_bsz", "Jpsi_bschi2", "Jpsi_bschi2fit",
    "Jpsi_bschi20", "Jpsi_bsvchk",
    "Muplus_pt", "Muminus_pt", "Muplus_eta", "Muminus_eta",
    "gradmax", "hessmax",
]
VECTORS = [
    "Muplus_refParms", "Muminus_refParms", "Jpsi_covvtx", "Jpsi_covrefmom",
    "Jpsi_bscov",
    "cfmass_ms", "cfmass_ioni_re", "cfmass_ioni_im",
    "cfmass_rad_re", "cfmass_rad_im", "cfmass_del", "cfmass_hitv",
    "hitdiag_dx", "hitdiag_dy", "hitdiag_exx", "hitdiag_eyy",
]
KEYS = ["run", "lumi", "event"]


def onefile(pat):
    """A comma-separated list of directories or files; every ROOT file found is
    used, so several arms of the same configuration can be pooled."""
    out = []
    for part in pat.split(","):
        part = part.strip()
        if not part:
            continue
        fs = sorted(glob.glob(os.path.join(part, "*.root"))) if os.path.isdir(part) else [part]
        out += [f for f in fs if os.path.getsize(f) > 0]
    if not out:
        sys.exit(f"no ROOT file under {pat}")
    return out


def load(paths, names):
    if isinstance(paths, str):
        paths = [paths]
    parts, keys, n = [], None, 0
    for path in paths:
        t = uproot.open(path)["tree"]
        k = set(t.keys())
        keys = k if keys is None else (keys & k)
        parts.append(t)
        n += t.num_entries
    have = [x for x in names if x in keys]
    merged = {}
    for t in parts:
        a = t.arrays(have, library="np")
        for x in have:
            merged.setdefault(x, []).append(a[x])
    return {x: np.concatenate(v) for x, v in merged.items()}, keys, n


def ordinals(run, lumi, event):
    """ordinal of each entry within its (run, lumi, event)"""
    key = np.stack([run, lumi, event], axis=1)
    out = np.zeros(len(run), dtype=np.int64)
    seen = {}
    for i, k in enumerate(map(tuple, key)):
        out[i] = seen.get(k, 0)
        seen[k] = out[i] + 1
    return [(*k, o) for k, o in zip(map(tuple, key), out)]


def stats(d, ref=None):
    """|B-A| summary; `ref` (same shape as d) turns it into a relative one.
    d and ref are masked TOGETHER so the relative numbers stay paired."""
    d = np.asarray(d, dtype=float)
    r = np.asarray(ref, dtype=float) if ref is not None else None
    good = np.isfinite(d) if r is None else (np.isfinite(d) & np.isfinite(r))
    d = d[good]
    if d.size == 0:
        return None
    a = np.abs(d)
    out = {"n": int(d.size), "nonzero": int(np.count_nonzero(a)),
           "med_abs": float(np.median(a)), "p95_abs": float(np.percentile(a, 95)),
           "max_abs": float(a.max()), "mean": float(d.mean()),
           "sem": float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else 0.0}
    if r is not None:
        r = r[good]
        nz = np.abs(r) > 0
        rel = d[nz] / r[nz]
        if rel.size:
            out["med_rel"] = float(np.median(np.abs(rel)))
            out["p95_rel"] = float(np.percentile(np.abs(rel), 95))
            out["mean_rel"] = float(rel.mean())
            out["sem_rel"] = float(rel.std(ddof=1) / np.sqrt(rel.size)) if rel.size > 1 else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    fa, fb = onefile(args.a), onefile(args.b)
    want = KEYS + SCALARS + VECTORS
    A, ka, na = load(fa, want)
    B, kb, nb = load(fb, want)
    print(f"{args.label_a}: {len(fa)} file(s), {na} candidates")
    print(f"{args.label_b}: {len(fb)} file(s), {nb} candidates")

    oa = ordinals(A["run"], A["lumi"], A["event"])
    ob = ordinals(B["run"], B["lumi"], B["event"])
    ia = {k: i for i, k in enumerate(oa)}
    common = [(ia[k], j) for j, k in enumerate(ob) if k in ia]
    if not common:
        sys.exit("no common candidates")
    si = np.array([c[0] for c in common])
    sj = np.array([c[1] for c in common])
    print(f"matched {len(si)} candidates "
          f"({na - len(si)} only in {args.label_a}, {nb - len(sj)} only in {args.label_b})")
    print()

    res = {}
    # per-candidate sigma for the mass, used as the natural yardstick
    sig = A["Jpsi_sigmamass"][si].astype(float) if "Jpsi_sigmamass" in A else None

    hdr = (f"{'branch':<24} {'n':>6} {'ndiff':>7} {'med|d|':>12} {'p95|d|':>12} {'med rel':>11}"
           f" {'p95 rel':>11} {'mean d':>13} {'+-':>12} {'mean rel':>12} {'+-':>11}")
    print(hdr)
    print("-" * len(hdr))
    for nm in SCALARS:
        if nm not in A or nm not in B:
            continue
        a = np.asarray(A[nm][si], dtype=float)
        b = np.asarray(B[nm][sj], dtype=float)
        s = stats(b - a, ref=a)
        if s is None:
            continue
        res[nm] = s
        print(f"{nm:<24} {s['n']:>6} {s['nonzero']:>7} {s['med_abs']:>12.4e} {s['p95_abs']:>12.4e}"
              f" {s.get('med_rel', float('nan')):>11.3e} {s.get('p95_rel', float('nan')):>11.3e}"
              f" {s['mean']:>13.5e} {s['sem']:>12.3e}"
              f" {s.get('mean_rel', float('nan')):>12.4e} {s.get('sem_rel', float('nan')):>11.3e}")
        if nm == "Jpsi_mass" and sig is not None:
            z = (b - a) / np.where(sig > 0, sig, np.nan)
            z = z[np.isfinite(z)]
            print(f"{'  ^ in units of sigma_m':<24} {len(z):>6} {'':>7} {np.median(np.abs(z)):>12.4e}"
                  f" {np.percentile(np.abs(z), 95):>12.4e} {'':>11} {'':>11}"
                  f" {z.mean():>13.5e} {z.std(ddof=1)/np.sqrt(len(z)):>12.3e}")
            res["Jpsi_mass_sigma_units"] = {"mean": float(z.mean()),
                                            "sem": float(z.std(ddof=1)/np.sqrt(len(z))),
                                            "med_abs": float(np.median(np.abs(z)))}
    print()
    for nm in VECTORS:
        if nm not in A or nm not in B:
            continue
        da, db = [], []
        for i, j in zip(si, sj):
            x, y = np.asarray(A[nm][i], dtype=float), np.asarray(B[nm][j], dtype=float)
            if x.shape != y.shape:
                continue
            da.append(x)
            db.append(y)
        if not da:
            continue
        x = np.concatenate(da)
        y = np.concatenate(db)
        s = stats(y - x, ref=x)
        res[nm] = s
        print(f"{nm:<24} {s['n']:>6} {s['nonzero']:>7} {s['med_abs']:>12.4e} {s['p95_abs']:>12.4e}"
              f" {s.get('med_rel', float('nan')):>11.3e} {s.get('p95_rel', float('nan')):>11.3e}"
              f" {s['mean']:>13.5e} {s['sem']:>12.3e}"
              f" {s.get('mean_rel', float('nan')):>12.4e} {s.get('sem_rel', float('nan')):>11.3e}")

    # the two leg components separately, since (q/p, lambda) is the point
    for leg in ("Muplus", "Muminus"):
        nm = f"{leg}_refParms"
        if nm not in A or nm not in B:
            continue
        x = np.stack([np.asarray(A[nm][i], dtype=float) for i in si])
        y = np.stack([np.asarray(B[nm][j], dtype=float) for j in sj])
        for c, cn in enumerate(("qop", "lambda", "phi")):
            d = y[:, c] - x[:, c]
            s = stats(d, ref=x[:, c])
            res[f"{nm}_{cn}"] = s
            print(f"{nm + '.' + cn:<24} {s['n']:>6} {s['nonzero']:>7} {s['med_abs']:>12.4e}"
                  f" {s['p95_abs']:>12.4e} {s.get('med_rel', float('nan')):>11.3e}"
                  f" {s.get('p95_rel', float('nan')):>11.3e} {s['mean']:>13.5e} {s['sem']:>12.3e}"
              f" {s.get('mean_rel', float('nan')):>12.4e} {s.get('sem_rel', float('nan')):>11.3e}")
        # in units of the exported per-leg reference sigma, when present
        sg = "Jpsi_sigmarelplus" if leg == "Muplus" else "Jpsi_sigmarelminus"
        if sg in A:
            sr = np.asarray(A[sg][si], dtype=float) * np.abs(x[:, 0])
            z = (y[:, 0] - x[:, 0]) / np.where(sr > 0, sr, np.nan)
            z = z[np.isfinite(z)]
            if z.size:
                print(f"{nm + '.qop [sigma]':<24} {z.size:>6} {'':>7} {np.median(np.abs(z)):>12.4e}"
                      f" {np.percentile(np.abs(z), 95):>12.4e} {'':>11} {'':>11}"
                      f" {z.mean():>13.5e} {z.std(ddof=1)/np.sqrt(z.size):>12.3e}")
                res[f"{nm}_qop_sigma_units"] = {"mean": float(z.mean()),
                                                "sem": float(z.std(ddof=1)/np.sqrt(z.size))}

    print()
    print("branch-set difference:", sorted(ka ^ kb) or "none")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
