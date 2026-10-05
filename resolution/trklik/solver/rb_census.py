"""Gate A0 of the conditional-Kalman recursion: the dimension census.

For every leg of a sample of real frozen models: run the Gaussian filter of
the recursion (process noise = the Gaussian remainder `Q^G`, no kicks) and, at
every cut k between hit k and hit k+1,

    d_kj  = Psi_kj u_j        the filtered shift at k per unit kick at j <= k
                              (D_{j|j-1} = u_j, D = A D, D_{i+1|i} = Phi D)
    J_k   = sum_{i>k} M_ik^T H_i^T S_i^-1 H_i M_ik
                              what the FUTURE innovations know about D_k
                              (M_ik maps D_k to D_{i|i-1}, no further kicks)
    C_k   = sum_{j<=k} s_j^2 d_kj d_kj^T     the past kicks' spread at k
    W_k   = J_k^1/2 C_k J_k^1/2              eigenvalues e_1 >= ... >= e_5

`e_r` is the variance, in units of the future information, that the past
kicks put along the r-th direction: the directions a lattice for the message
mu_k(D) must carry.  The kick scale `s_j` is either the CORE width
(`sqrt(lambda_j)`, the fit's alpha = 0.999-truncated variance along u) or a
BIG-JUMP scale: the kick |x| above which the block's expected number of
knock-ons is `p` (1e-2, 1e-4), from the Urban intensity of the step records.
The same is reported in eta space, the eigenvalues of `s Lambda s`,
`Lambda = D^T Sigma_G^-1 D`, `D` the kicks' columns in the leg's rows.

    python solver/rb_census.py 'GLOB' --models-per-file 2 --max-files 160 \
        --jobs 64 --out solver/runs/rbgrid/census.npz
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import glob
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

PJ = (1e-2, 1e-4, 1e-6)


def jump_scale(urban, p):
    """The kick |x| above which the block's expected knock-on count is `p`
    (the exact spin-1/2 intensity of `UrbanCGF.set_cut`), bisected on |x|
    directly so that steps with different conversion factors are pooled."""
    st = urban.steps
    m = st[:, 0] >= 2
    if not m.any():
        return 0.0
    gam = st[m, 9]
    xi = st[m, 6] * gam
    e0 = st[m, 7] * gam
    tmax = st[m, 8] * gam
    b2 = st[m, 11]
    et = st[m, 12]
    half = (st[m, 0] == 2).astype(float)
    c = np.abs(urban.wstd * st[m, 10] * 1e-3)

    def count(X):
        Tc = X / np.maximum(c, 1e-300)
        hi = np.maximum(tmax, Tc)
        lo = np.minimum(np.maximum(Tc, e0), hi)
        return float(np.sum(xi * ((1.0 / lo - 1.0 / hi) - (b2 / tmax) * np.log(hi / lo)
                                  + half * (hi - lo) / (2 * et ** 2))))
    lo_, hi_ = 1e-30, float(np.max(c * tmax))
    if count(hi_ * (1 - 1e-12)) > p:
        return hi_
    if count(lo_) < p:
        return 0.0
    for _ in range(300):
        mid = np.sqrt(lo_ * hi_)
        if count(mid) > p:
            lo_ = mid
        else:
            hi_ = mid
        if hi_ / lo_ < 1 + 1e-9:
            break
    return float(np.sqrt(lo_ * hi_))


def census_leg(leg, y, x):
    from solver.rbgrid import kalman
    kf = kalman(leg, y, x)
    n = leg.n
    # forward responses: d[k][j] = filtered shift at k per unit kick at j
    d = [[None] * n for _ in range(n)]
    for j in range(n):
        D = leg.u[j].copy()
        for k in range(j, n):
            if k > j:
                D = leg.Phi[k] @ D
            # D is D_{k|k-1}; filtered:
            Df = kf["A"][k] @ D
            d[k][j] = Df
            D = Df
    # future information on the filtered shift at k
    J = [np.zeros((5, 5)) for _ in range(n)]
    for k in range(n):
        M = np.eye(5)
        Jk = np.zeros((5, 5))
        for i in range(k + 1, n):
            M = leg.Phi[i] @ M                  # -> D_{i|i-1}
            Hi = leg.H[i]
            if len(Hi):
                HM = Hi @ M
                Jk += HM.T @ np.linalg.solve(kf["S"][i], HM)
            M = kf["A"][i] @ M                  # -> D_i
        J[k] = 0.5 * (Jk + Jk.T)
    return kf, d, J


def backward_census(leg, y, scales):
    """The same census in the frame of the BACKWARD recursion (outer hits
    first): at every kick op t, the spread the processed kicks put on the
    current shift, in units of the REMAINING information J_t (the inner
    weights plus the final Gaussian, all mapped to the frame of op t).
    Returns {scale name: (nkick, 5) eigenvalues} and the per-op dims."""
    from solver.rbgrid import backward_filter
    ops, fin = backward_filter(leg, y)
    nops = len(ops)
    # remaining information before each op (sensitivity of everything after
    # op t to the shift entering op t), computed from the end
    Jrem = [None] * (nops + 1)
    J = fin[0].copy()
    Jrem[nops] = J.copy()
    for t in range(nops - 1, -1, -1):
        o = ops[t]
        if o["kind"] == "map":
            J = o["M"].T @ J @ o["M"]
        elif o["kind"] == "meas":
            J = o["R"].T @ J @ o["R"] + o["W"]
        Jrem[t] = 0.5 * (J + J.T)
    out = {k: [] for k in scales}
    vecs = []                                    # processed kick vectors, current frame
    for t, o in enumerate(ops):
        if o["kind"] == "kick":
            j = o["block"]
            vecs = [v for v in vecs] + [(-o["u"], j)]
            Jt = Jrem[t + 1]                     # what remains after this kick
            w, V = np.linalg.eigh(Jt)
            Jh = (V * np.sqrt(np.maximum(w, 0.0))) @ V.T
            for k, s in scales.items():
                C = np.zeros((5, 5))
                for v, jj in vecs:
                    vv = Jh @ v
                    C += s[jj] ** 2 * np.outer(vv, vv)
                out[k].append(np.sort(np.maximum(np.linalg.eigvalsh(0.5 * (C + C.T)), 0.0))[::-1])
        elif o["kind"] == "map":
            vecs = [(o["M"] @ v, jj) for v, jj in vecs]
        elif o["kind"] == "meas":
            vecs = [(o["R"] @ v, jj) for v, jj in vecs]
    return {k: np.array(v) for k, v in out.items()}


def run_model(task):
    path, entry, args = task
    import solver.record as R
    from solver.marginal import MarginalLikelihood
    from solver.rbgrid import build_legs
    out = dict(path=path, entry=entry, err=None, legs=[])
    t0 = time.perf_counter()
    try:
        R.MS_STRIDE = args.stride
        cand = next(iter(R.iter_candidates(path, entries={entry}, need_steps=True, extra=[])))
        ml = MarginalLikelihood(cand, gaussian=False)
        legs = build_legs(ml)
        y = -ml.red.c0
        x = np.zeros(10)
        Sg = None
        for leg in legs:
            kf, d, J = census_leg(leg, y, x)
            n = leg.n
            scales = {"core": np.sqrt(np.maximum(leg.lam, 0.0))}
            for p in PJ:
                scales[f"J{p:g}"] = np.array([jump_scale(cg, p) for cg in leg.cgf])
            # the posterior q/p sigma at each hit (filtered), for the kick sizes
            sq = np.sqrt(np.array([kf["Pfilt"][k][0, 0] for k in range(n)]))
            rec = dict(n=n, nmeas=leg.nmeas, sig_qop_end=float(sq[-1]),
                       sig_qop_first=float(sq[0]), scales={k: v for k, v in scales.items()})
            for sname, s in scales.items():
                ev = np.zeros((n, 5))
                lead = np.zeros((n, 5))         # bending share of the r-th eigvec
                for k in range(n):
                    C = np.zeros((5, 5))
                    for j in range(k + 1):
                        C += s[j] ** 2 * np.outer(d[k][j], d[k][j])
                    wJ, VJ = np.linalg.eigh(J[k])
                    wJ = np.maximum(wJ, 0.0)
                    Jh = (VJ * np.sqrt(wJ)) @ VJ.T
                    Wk = Jh @ C @ Jh
                    w, Vw = np.linalg.eigh(0.5 * (Wk + Wk.T))
                    o = np.argsort(w)[::-1]
                    ev[k] = np.maximum(w[o], 0.0)
                    # the eigenvector in D space (J^-1/2 V), its non-bending share
                    Jih = (VJ * np.where(wJ > 1e-30 * max(wJ.max(), 1e-300),
                                         1.0 / np.sqrt(np.maximum(wJ, 1e-300)), 0.0)) @ VJ.T
                    for r in range(5):
                        vD = Jih @ Vw[:, o[r]]
                        nr = np.linalg.norm(vD)
                        lead[k, r] = (float(np.hypot(vD[1], vD[4]) / nr) if nr > 0 else np.nan)
                rec["ev_" + sname] = ev
                rec["nb_" + sname] = lead
            # eta space: Lambda = D^T Sigma_G^-1 D on the leg's own rows
            rows = np.concatenate([r for r in leg.rows if len(r)])
            if Sg is None:
                red = ml.red
                Sg = red.Vhit.copy()
                for Gb, b in zip(red.Gb, ml.blocks):
                    Sg += Gb @ b.Cres @ Gb.T
            Sl = Sg[np.ix_(rows, rows)]
            Dm = np.column_stack([ml.red.Gb[i][rows] @ ml.blocks[i].u for i in leg.blk])
            Lam = Dm.T @ np.linalg.solve(Sl, Dm)
            for sname, s in scales.items():
                M = (s[:, None] * Lam) * s[None, :]
                rec["eta_" + sname] = np.sort(np.maximum(np.linalg.eigvalsh(0.5 * (M + M.T)), 0.0))[::-1]
            bc = backward_census(leg, y, scales)
            for sname in scales:
                rec["bev_" + sname] = bc[sname]
            out["legs"].append(rec)
        out["sig_rel"] = None
    except Exception as exc:                                       # noqa: BLE001
        import traceback
        out["err"] = f"{type(exc).__name__}: {exc} | {traceback.format_exc()[-400:]}"
    out["t"] = time.perf_counter() - t0
    return out


def summarise(rows, thresholds=(1e-8, 1e-6, 1e-4, 1e-2, 1.0)):
    good = [r for r in rows if not r["err"]]
    legs = [lg for r in good for lg in r["legs"]]
    print(f"{len(good)} models, {len(legs)} legs; hits per leg median "
          f"{np.median([lg['n'] for lg in legs]):.0f}")
    names = ["core"] + [f"J{p:g}" for p in PJ]
    for sname in names:
        print(f"\n== kick scale '{sname}': forward-message census (max over the cuts of a leg)")
        evs = [lg["ev_" + sname] for lg in legs]
        top = np.array([e.max(axis=0) for e in evs])          # (nlegs, 5) max over k
        for r in range(5):
            q = np.quantile(top[:, r], [0.5, 0.9, 0.99, 1.0])
            print(f"   e_{r + 1}: median {q[0]:.2e}  p90 {q[1]:.2e}  p99 {q[2]:.2e}  max {q[3]:.2e}")
        for th in thresholds:
            nd = np.array([int(np.max(np.sum(e > th, axis=1))) for e in evs])
            print(f"   dims with e > {th:g}: " + "  ".join(
                f"{k}:{np.mean(nd == k) * 100:.1f}%" for k in range(6)))
        for r in (1, 2, 3):
            leak = np.array([np.max(np.sum(e[:, r:], axis=1)) for e in evs])
            q = np.quantile(leak, [0.5, 0.9, 0.99, 1.0])
            print(f"   leakage beyond r={r} (max over cuts of sum_{{i>r}} e_i): median {q[0]:.2e} "
                  f"p90 {q[1]:.2e} p99 {q[2]:.2e} max {q[3]:.2e}")
        nb = np.array([np.nanmax(lg["nb_" + sname][:, :3]) for lg in legs])
        q = np.quantile(nb, [0.5, 0.9, 0.99, 1.0])
        print(f"   non-bending share (lambda, x4) of the leading 3 directions: median {q[0]:.2e} "
              f"p90 {q[1]:.2e} max {q[3]:.2e}")
        eta = [lg["eta_" + sname] for lg in legs]
        for th in thresholds:
            nd = np.array([int(np.sum(e > th)) for e in eta])
            print(f"   eta space, eigenvalues of s Lambda s above {th:g}: median {np.median(nd):.0f}, "
                  f"p90 {np.quantile(nd, 0.9):.0f}, max {nd.max()}  (of median "
                  f"{np.median([len(e) for e in eta]):.0f} kicks)")
    for sname in names:
        print(f"\n== kick scale '{sname}': BACKWARD census, remaining-information frame "
              f"(max over the kick ops of a leg)")
        evs = [lg["bev_" + sname] for lg in legs]
        top = np.array([e.max(axis=0) for e in evs])
        for r in range(5):
            q = np.quantile(top[:, r], [0.5, 0.9, 0.99, 1.0])
            print(f"   e_{r + 1}: median {q[0]:.2e}  p90 {q[1]:.2e}  p99 {q[2]:.2e}  max {q[3]:.2e}")
        for th in thresholds:
            nd = np.array([int(np.max(np.sum(e > th, axis=1))) for e in evs])
            print(f"   dims with e > {th:g}: " + "  ".join(
                f"{k}:{np.mean(nd == k) * 100:.1f}%" for k in range(6)))
        for r in (1, 2, 3):
            leak = np.array([np.max(np.sum(e[:, r:], axis=1)) for e in evs])
            q = np.quantile(leak, [0.5, 0.9, 0.99, 1.0])
            print(f"   leakage beyond r={r}: median {q[0]:.2e} p90 {q[1]:.2e} p99 {q[2]:.2e} "
                  f"max {q[3]:.2e}")
    # the kick sizes in units of the leg's final q/p sigma
    print("\n== kick scales against the leg's final q/p resolution (filtered at the last hit)")
    for sname in names:
        ratio = np.array([np.max(lg["scales"][sname]) / lg["sig_qop_end"] for lg in legs])
        q = np.quantile(ratio, [0.5, 0.9, 0.99, 1.0])
        print(f"   max_j s_j / sigma_qop: {sname:6s} median {q[0]:.3g}  p90 {q[1]:.3g}  "
              f"p99 {q[2]:.3g}  max {q[3]:.3g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files")
    ap.add_argument("--models-per-file", type=int, default=2)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--stride", type=int, default=13)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import uproot
    files = sorted(glob.glob(args.files))[: args.max_files]
    tasks = []
    for f in files:
        with uproot.open(f) as fh:
            n = fh["tree"].num_entries
        for e in np.linspace(0, n - 1, args.models_per_file).astype(int):
            tasks.append((f, int(e), args))
    print(f"{len(tasks)} models, {args.jobs} workers", flush=True)
    t0 = time.perf_counter()
    with Pool(args.jobs) as p:
        rows = list(p.imap_unordered(run_model, tasks))
    print(f"done in {time.perf_counter() - t0:.0f} s; {sum(1 for r in rows if r['err'])} failed")
    for r in [r for r in rows if r["err"]][:3]:
        print("  ", r["err"][:300])
    summarise(rows)
    if args.out:
        import pickle
        with open(args.out, "wb") as fh:
            pickle.dump(rows, fh)
        print("wrote", args.out)


if __name__ == "__main__":
    sys.exit(main())
