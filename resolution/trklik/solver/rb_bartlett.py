"""Bartlett's first identity for the conditional-Kalman recursion: at the
truth the score of an exact likelihood has mean zero, E[d ln L / d x_ref] = 0,
whatever the noise law -- a test of the likelihood itself, free of the
nuisance elimination the closure (gate A3) adds on top.

For tail-toy replicas of real frozen models (`tail_toy.py`'s noise and
seeds), the score of arm R (`rbgrid.RBGridLikelihood`) and of the Gaussian
marginal G at x_ref = 0, whitened by the Gaussian Fisher matrix A,
w = A^-1/2 g (unit variance per component under G's own model), and its
component along the mass direction, s_m = J A^-1 g / sqrt(J A^-1 J). The
second identity, E[g g^T] = E[-H], is reported as the ratio of the mass
direction's score variance to its mean observed information.

    python solver/rb_bartlett.py 'GLOB' --models-per-file 16 --replicas 10 \
        --jobs 64 --out solver/runs/rbgrid/bartlett.npz
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import glob
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
warnings.filterwarnings("ignore")


def run_model(task):
    path, entry, args, seed = task
    import solver.record as R
    from solver.marginal import MarginalLikelihood, sample_noise
    from solver.mass import MassConstraint
    from solver.rbgrid import LatticeConfig, RBGridLikelihood
    import uproot
    out = dict(path=path, entry=entry, err=None, rows=[])
    try:
        with uproot.open(path) as fh:
            t = fh["tree"]
            ms0 = np.asarray(t["msmoliv"].array(entry_start=0, entry_stop=1, library="np")[0])
            nms0 = np.asarray(t["fm_nmssteps"].array(entry_start=0, entry_stop=1, library="np")[0])
        st = int(len(ms0) // max(int(np.sum(nms0)), 1))
        if st in (11, 13):
            R.MS_STRIDE = st
        cand = next(iter(R.iter_candidates(path, entries={entry}, need_steps=True, extra=[])))
        gen = MarginalLikelihood(cand, gaussian=False)
        mG = MarginalLikelihood(cand, gaussian=True)
        cfg = LatticeConfig(h=args.h, rmax=args.rmax, lookahead=args.lookahead)
        mR = RBGridLikelihood(cand, gaussian=False, cfg=cfg,
                              ml=MarginalLikelihood(cand, gaussian=False), run=False)
        x0 = np.zeros(mR.nfree)
        A = mR.fisher()
        w, V = np.linalg.eigh(A)
        Aih = (V / np.sqrt(w)) @ V.T                     # A^-1/2
        J = MassConstraint(mR).jac(x0)
        AiJ = np.linalg.solve(A, J)
        nJ = float(np.sqrt(J @ AiJ))
        rng = np.random.default_rng(seed)
        for r in range(args.replicas):
            z = sample_noise(gen, rng)
            mR.set_data(-z)
            _, gR, HR = mR.value_grad_hess(x0)
            mG.red.c0 = -z
            mG._t = np.zeros(mG.red.nrows)
            _, gG, HG = mG.value_grad_hess(x0)
            # value_grad_hess returns the gradient of -ln L: the score is -g
            sR, sG = -np.asarray(gR), -np.asarray(gG)
            out["rows"].append(dict(
                wR=Aih @ sR, wG=Aih @ sG, mR=float(AiJ @ sR) / nJ, mG=float(AiJ @ sG) / nJ,
                iR=float(AiJ @ HR @ AiJ) / nJ ** 2, iG=float(AiJ @ HG @ AiJ) / nJ ** 2,
                neg=max(d.get("neg", 0.0) for d in mR.diag),
                guard=sum(d.get("trimguard", 0) for d in mR.diag)))
    except Exception:                                             # noqa: BLE001
        import traceback
        out["err"] = traceback.format_exc()[-800:]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files")
    ap.add_argument("--models-per-file", type=int, default=16)
    ap.add_argument("--file-start", type=int, default=0)
    ap.add_argument("--file-stop", type=int, default=None)
    ap.add_argument("--replicas", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--h", type=float, default=0.3)
    ap.add_argument("--rmax", type=int, default=3)
    ap.add_argument("--lookahead", default="kicks", choices=("kicks", "flat", "none"))
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import uproot
    files = sorted(glob.glob(args.files))
    tasks, k = [], 0
    stop = args.file_stop if args.file_stop is not None else len(files)
    for fi, f in enumerate(files):
        if not (args.file_start <= fi < stop):
            k += args.models_per_file
            continue
        with uproot.open(f) as fh:
            n = fh["tree"].num_entries
        for e in np.linspace(0, n - 1, args.models_per_file).astype(int):
            tasks.append((f, int(e), args, args.seed + 7919 * k))
            k += 1
    print(f"{len(tasks)} models x {args.replicas} replicas, {args.jobs} workers", flush=True)
    t0 = time.perf_counter()
    with Pool(args.jobs) as p:
        res = list(p.imap_unordered(run_model, tasks))
    errs = [r for r in res if r["err"]]
    rows = [x for r in res for x in r["rows"]]
    print(f"done in {time.perf_counter() - t0:.0f} s; {len(errs)} models failed")
    for r in errs[:3]:
        print("  ", r["err"].splitlines()[-1])
    wR = np.array([x["wR"] for x in rows])
    wG = np.array([x["wG"] for x in rows])
    mR = np.array([x["mR"] for x in rows])
    mG = np.array([x["mG"] for x in rows])
    iR = np.array([x["iR"] for x in rows])
    iG = np.array([x["iG"] for x in rows])
    n = len(rows)
    f = lambda v: f"{v.mean():+.4f} +- {v.std(ddof=1) / np.sqrt(len(v)):.4f}"  # noqa: E731
    print(f"\n{n} replicas; score at the truth, whitened by the Gaussian Fisher matrix")
    print("   component   R                  G                  R - G (paired)")
    for c in range(wR.shape[1]):
        print(f"   {c:9d}   {f(wR[:, c])}   {f(wG[:, c])}   {f(wR[:, c] - wG[:, c])}")
    print(f"   mass dir.   {f(mR)}   {f(mG)}   {f(mR - mG)}")
    print(f"second identity along the mass direction: var(score) / mean(observed information): "
          f"R {mR.var() / iR.mean():.4f}, G {mG.var() / iG.mean():.4f}")
    guard = sum(x["guard"] for x in rows)
    neg = np.array([x["neg"] for x in rows])
    print(f"trim guards {guard}; negative share median {np.median(neg):.1e}, max {neg.max():.2f}")
    if args.out:
        np.savez(args.out, wR=wR, wG=wG, mR=mR, mG=mG, iR=iR, iG=iG, neg=neg,
                 paths=np.array([r["path"] for r in res for _ in r["rows"]]),
                 entries=np.array([r["entry"] for r in res for _ in r["rows"]]))
        print("wrote", args.out)


if __name__ == "__main__":
    sys.exit(main())
