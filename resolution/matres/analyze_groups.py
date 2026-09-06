#!/usr/bin/env python3
"""Storage, pruning and leverage of the per-material-group CF exponents.

Answers, from a ``matres/extract_groups.py`` npz:

* how many material groups a candidate touches, and which;
* what the per-group exponents cost per candidate -- raw, after pruning, and
  in a rank-R PCA of the tau axis (the ``cfcompress`` construction, applied
  per family across all (candidate, group) rows);
* which groups actually carry the resolution (the leverage table), so the
  degeneracy structure of the later fit can be read against something;
* the Fisher information the MASS term alone has on each group amount, in the
  Gaussian limit -- the cheap forecast of which groups the mass term can
  separate before any fit is run.

Usage::

    python analyze_groups.py -i runs/matres/gun_groups.npz [--rank 16] [--plot]
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", required=True)
    p.add_argument("--rank", type=int, nargs="*", default=[4, 8, 16, 32])
    p.add_argument("--prune", type=float, nargs="*",
                   default=[0.0, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1])
    p.add_argument("--maxn", type=int, default=0,
                   help="use only N candidates for the PCA (it is O(nnz nt^2))")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--outdir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    d = np.load(args.input, allow_pickle=False)
    keys = set(d.files)
    fams = [str(x) for x in d["families"]]
    gnames = [str(x) for x in d["group_names"]]
    cnames = [str(x) for x in d["hit_classes"]]
    ptr = d["grp_ptr"].astype(np.int64)
    gid = d["grp_id"].astype(np.int64)
    tg = d["tgrid"]
    n = len(ptr) - 1
    nt = len(tg)
    mult = np.diff(ptr)
    print(f"input          {args.input}")
    print(f"candidates     {n}")
    print(f"tau grid       {nt} points, max {tg[-1]:.4f}")
    print(f"families       {fams}")
    print(f"groups         {len(gnames)}  |  hit classes {len(cnames)}")
    print(f"functional     {str(d['functional'])}  hitmode {str(d['hitmode'])}")

    print("\n=== 1. group multiplicity per candidate ===")
    print(f"  mean {mult.mean():.2f}  median {np.median(mult):.0f}  "
          f"min {mult.min()}  p1 {np.percentile(mult,1):.0f}  "
          f"p99 {np.percentile(mult,99):.0f}  max {mult.max()}")
    occ = np.bincount(gid, minlength=len(gnames))
    print("  group occupancy (fraction of candidates that touch it):")
    for g in np.argsort(-occ):
        if occ[g] == 0:
            continue
        print(f"    {g:3d} {gnames[g]:<24} {100.*occ[g]/n:6.2f} %")
    dead = [gnames[g] for g in range(len(gnames)) if occ[g] == 0]
    if dead:
        print(f"  NEVER touched ({len(dead)}): " + ", ".join(dead))

    print("\n=== 2. storage per candidate ===")
    nnz = len(gid)
    raw = nnz * nt * 4 * len(fams) / n
    flat = len(fams) * nt * 4
    print(f"  flat (per-family only)      {flat/1024:8.2f} kB")
    print(f"  per group, raw              {raw/1024:8.2f} kB   "
          f"(x{raw/flat:.1f})")
    idx_b = (nnz * 2 + n * 8) / n  # int16 gid + int64 ptr
    print(f"  + CSR index                 {idx_b/1024:8.3f} kB")

    # amplitude of every row (the max over tau and families)
    amp = np.zeros(nnz)
    for f in fams:
        amp = np.maximum(amp, np.abs(d["S" + f]).max(axis=1))
    seg = np.repeat(np.arange(n), mult)
    top = np.zeros(n)
    np.maximum.at(top, seg, amp)
    print("\n  pruning (fold rows below frac x the candidate's max |S| into a"
          " fixed baseline):")
    print(f"  {'frac':>8} {'rows kept':>11} {'groups/cand':>12} "
          f"{'kB/cand':>9} {'max |dS| folded':>16}")
    for fr in args.prune:
        keep = amp >= fr * top[seg]
        k = int(keep.sum())
        kb = (k * nt * 4 * len(fams) / n
              + (0 if fr == 0.0 else len(fams) * nt * 4)) / 1024
        # the largest single-candidate folded exponent
        fold = np.zeros(n)
        if (~keep).any():
            np.add.at(fold, seg[~keep], amp[~keep])
        print(f"  {fr:8.4g} {100.*k/nnz:10.2f}% {k/n:12.2f} {kb:9.2f} "
              f"{fold.max():16.3e}")

    print("\n=== 3. rank-R PCA of the tau axis (per family, over all rows) ===")
    sub = slice(None)
    if args.maxn and args.maxn < n:
        sub = slice(0, int(ptr[args.maxn]))
        print(f"  (PCA on the first {args.maxn} candidates)")
    print(f"  {'family':<10} " + " ".join(f"R={r:<3d} relerr" for r in args.rank))
    tot_R = {r: 0 for r in args.rank}
    for f in fams:
        A = d["S" + f][sub].astype(np.float64)
        if not A.size or not np.any(A):
            print(f"  {f:<10} (identically zero)")
            continue
        # economical SVD of the (rows, nt) matrix
        _, s, _ = np.linalg.svd(A, full_matrices=False)
        e2 = np.cumsum(s ** 2)
        tot = e2[-1]
        cells = []
        for r in args.rank:
            rr = min(r, len(s))
            cells.append(f"{np.sqrt(max(tot - e2[rr - 1], 0.0)/tot):10.3e}")
            tot_R[r] += 1
        print(f"  {f:<10} " + " ".join(cells))
    for r in args.rank:
        b = nnz * r * 4 * len(fams) / n
        print(f"  rank {r:<3d} storage {b/1024:8.2f} kB/candidate "
              f"(x{b/flat:.1f} the flat cache)")

    print("\n=== 4. per-group leverage ===")
    lev = np.zeros(len(gnames))
    np.add.at(lev, gid, amp)
    tot = lev.sum() or 1.0
    print(f"  {'group':<26} {'share of sum |S|':>17} {'occupancy':>10} "
          f"{'mean |S| when present':>22}")
    for g in np.argsort(-lev):
        if occ[g] == 0:
            continue
        print(f"  {gnames[g]:<26} {100.*lev[g]/tot:16.2f} % "
              f"{100.*occ[g]/n:9.2f} % {lev[g]/occ[g]:22.4f}")

    print("\n=== 5. Gaussian-limit Fisher forecast on the group amounts ===")
    # In the Gaussian limit the candidate's standardized variance is
    #   V_i(k) = vg_other_i + sum_c v_ci + sum_g A(k_g) V_{i,g},
    # with V_{i,g} = -2 d^2/dtau^2 Re S_{i,g}|_0 read off the small-tau
    # curvature.  For a Gaussian of variance V the Fisher information in
    # log V is 1/2, so  I_gh = 0.5 sum_i (V_ig/V_i)(V_ih/V_i)  at k = 0.
    rec = [f for f in fams if not f.endswith("_im")]
    it = np.searchsorted(tg, min(1.0, tg[-1] * 0.25))
    it = max(it, 2)
    curv = np.zeros(nnz)
    for f in rec:
        A = d["S" + f][:, it].astype(np.float64)
        curv += -2.0 * A / (tg[it] ** 2)
    Vg = np.maximum(curv, 0.0)
    V = np.array(d["vgf"], np.float64)
    np.add.at(V, seg, Vg)
    ng = len(gnames)
    M = np.zeros((n, ng))
    np.add.at(M, (seg, gid), Vg / V[seg])
    I = 0.5 * (M.T @ M)
    live = np.where(occ > 0)[0]
    Il = I[np.ix_(live, live)]
    w = np.linalg.eigvalsh(Il)
    cov = np.linalg.pinv(Il, rcond=1e-12)
    err = np.sqrt(np.abs(np.diag(cov)))
    print(f"  {len(live)} live groups; eigenvalues {w.min():.4g} .. "
          f"{w.max():.4g}, cond {w.max()/max(w.min(),1e-300):.3e}, "
          f"rank(1e-10) {int((w > 1e-10*w.max()).sum())}")
    print(f"  {'group':<26} {'sigma(k_g) marginal':>20} {'standalone':>12}")
    for j, g in enumerate(np.argsort(-np.diag(Il))):
        gi = live[g] if False else g
    order = np.argsort(err)
    for j in order:
        g = live[j]
        print(f"  {gnames[g]:<26} {err[j]:20.4f} "
              f"{1.0/np.sqrt(max(Il[j,j],1e-300)):12.4f}")
    ratio = err / np.array([1.0 / np.sqrt(max(Il[j, j], 1e-300))
                            for j in range(len(live))])
    print(f"  marginal/standalone inflation: median {np.median(ratio):.1f}x, "
          f"max {ratio.max():.1f}x -- the size of the degeneracy")
    # near-null directions
    wv, Vv = np.linalg.eigh(Il)
    print("  softest directions of the mass-term Fisher matrix:")
    for k in range(min(4, len(wv))):
        v = Vv[:, k]
        o = np.argsort(-np.abs(v))[:5]
        print(f"    lambda/lambda_max {wv[k]/wv[-1]:.3e}: " + "  ".join(
            f"{gnames[live[i]]} {v[i]:+.3f}" for i in o))

    if "hit_ptr" in keys and int(d["hit_ptr"][-1]) > 0:
        print("\n=== 6. hit classes ===")
        hp = d["hit_ptr"].astype(np.int64)
        hc = d["hit_cls"].astype(np.int64)
        hv = np.asarray(d["hit_v"], np.float64)
        hocc = np.bincount(hc, minlength=len(cnames))
        hsum = np.zeros(len(cnames))
        np.add.at(hsum, hc, hv)
        print(f"  {np.diff(hp).mean():.2f} class rows per candidate")
        print(f"  {'class':<16} {'occupancy':>10} {'share of v_hit':>16}")
        tot = hsum.sum() or 1.0
        for c in np.argsort(-hsum):
            if hocc[c] == 0:
                continue
            print(f"  {cnames[c]:<16} {100.*hocc[c]/n:9.2f} % "
                  f"{100.*hsum[c]/tot:15.2f} %")
        vo = np.asarray(d["vg_other"], np.float64)
        print(f"  v_other / vgf: mean {np.mean(vo/np.maximum(d['vgf'],1e-30)):.4f} "
              "(the Gaussian share no hit parameter scales)")


if __name__ == "__main__":
    main()
