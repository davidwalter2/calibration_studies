"""Gate A2: the conditional-Kalman recursion against an independent Monte Carlo
of each leg's likelihood, on toy replicas of real frozen models (noise drawn
from the model the recursion assumes, as `tail_toy.py` draws it), including
large-loss replicas, at the points of the mass scan on both sides of the truth.

For every replica: the recursion's unconstrained solution and the constrained
solutions at m_true (1 + k h_scan), k in `--scan`; at each of those states the
recursion's ln L_leg and the Rao-Blackwellised Monte Carlo (`rb_mc.LegMC`:
eta from the exact Urban compound processes, the Gaussian parts integrated
exactly, one block's kick integrated exactly through its Gaussian-smoothed
density -- the block giving the smallest error).

    python solver/rb_gate_a2.py 'GLOB' --models 24 --replicas 4 --jobs 24 \
        --nmc 200000 --out solver/runs/rbgrid/a2.npz
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


def run(task):
    path, entry, args, seed = task
    import solver.record as R
    from solver.marginal import MarginalLikelihood, sample_urban_leg
    from solver.mass import MassConstraint
    from solver.minimise import minimise
    from solver.profile import constrained_minimise
    from solver import rbgrid as RB
    from solver.rb_mc import LegMC, sample_urban_many
    out = dict(path=path, entry=entry, err=None, rows=[])
    t0 = time.perf_counter()
    try:
        R.MS_STRIDE = 13
        cand = next(iter(R.iter_candidates(path, entries={entry}, need_steps=True, extra=[])))
        gen = MarginalLikelihood(cand, gaussian=False)
        cfg = RB.LatticeConfig(trim=args.trim, rbox=args.rbox, h=args.h, rmax=args.rmax,
                               floor=args.floor, order=args.order)
        rb = RB.RBGridLikelihood(cand, gaussian=False, cfg=cfg,
                                 ml=MarginalLikelihood(cand, gaussian=False), run=False)
        cons = MassConstraint(rb)
        mtrue = cons.value(np.zeros(rb.nfree))
        rng = np.random.default_rng(seed)
        red = gen.red
        roots = []
        for b in gen.blocks:
            w, V = np.linalg.eigh(b.Cres)
            roots.append(V * np.sqrt(np.maximum(w, 0.0)))
        wv, Vv = np.linalg.eigh(red.Vhit)
        vroot = Vv * np.sqrt(np.maximum(wv, 0.0))
        lamv = np.array([b.lam for b in gen.blocks])
        # replicas from the model (the toy's own construction), keeping the kicks
        pool = []
        for _ in range(args.pool):
            eta = np.array([sample_urban_leg(b.urban.steps, b.urban.wstd, rng) for b in gen.blocks])
            n = np.concatenate([eta[i] * gen.blocks[i].u + roots[i] @ rng.normal(size=5)
                                for i in range(len(gen.blocks))])
            z = red.G @ n + vroot @ rng.normal(size=red.nrows)
            pool.append((float(np.max(np.abs(eta) / np.sqrt(lamv))), z))
        order = np.argsort([-p[0] for p in pool])
        pick = list(order[: args.big]) + [int(i) for i in order[len(order) // 2:
                                                              len(order) // 2 + args.replicas - args.big]]
        # the prior draws, once per model (common random numbers)
        mcs = [LegMC(gen, leg) for leg in rb.legs]
        etas = [np.column_stack([sample_urban_many(lw.steps, lw.wstd, rng, args.nmc)
                                 for lw in mc.laws]) for mc in mcs]
        for ip in pick:
            kick, z = pool[ip]
            ta = time.perf_counter()
            rb.set_data(-z)
            tr = time.perf_counter() - ta
            res = minimise(rb, tol=1e-9, maxiter=60)
            states = [("mle", res.delta)]
            for k in args.scan:
                mt = mtrue * (1.0 + k * args.hscan)
                cr = constrained_minimise(rb, cons, mt, delta0=res.delta, tol=1e-9, maxiter=80)
                states.append((f"m{k:+g}", cr.delta))
            x10s = [rb.embed(s) for _, s in states]
            for il, (leg, mc, eta) in enumerate(zip(rb.legs, mcs, etas)):
                mc.red.c0 = -z
                lat = [float(rb.mix[il].logL(leg.T0 @ x, 0)[0]) for x in x10s]
                plain = [mc.logL_prior(x, eta) for x in x10s]
                best = (max(p[1] for p in plain), -1, [(p[0], p[1], p[2], 0.0) for p in plain])
                if min(p[2] for p in plain) < args.ess_rb:
                    # Rao-Blackwellise the block that carries the weight
                    rw0 = np.linalg.solve(mc.L, mc.rho(x10s[0]))
                    q0 = rw0[None, :] - eta @ mc.Dw.T
                    lf0 = -0.5 * np.sum(q0 * q0, axis=1)
                    top = np.argsort(lf0)[::-1][:200]
                    lamd = np.sum(mc.Dw * mc.Dw, axis=0)
                    resp = np.abs(eta[top]) * np.sqrt(lamd)[None, :]
                    cands = list(dict.fromkeys(np.argsort(-resp.max(axis=0))[:args.rbk].tolist()))
                    for kb in cands:
                        lamk = float(lamd[kb])
                        other = eta.copy()
                        other[:, kb] = 0.0
                        Dq = other @ mc.Dw.T
                        mloc = []
                        for x in x10s:
                            rwx = np.linalg.solve(mc.L, mc.rho(x))
                            mloc.append(((rwx[None, :] - Dq) @ mc.Dw[:, kb]) / lamk)
                        mloc = np.concatenate(mloc)
                        mlo, mhi = np.quantile(mloc, [1e-5, 1 - 1e-5])
                        hk = mc.smoothed_density(kb, lamk, float(mlo), float(mhi))
                        rr = mc.logL_rb(x10s, eta, kb, hk, lamk)
                        sem = max(r[1] for r in rr)
                        if sem < best[0]:
                            best = (sem, kb, rr)
                for (lab, s), lv, rr, pl in zip(states, lat, best[2], plain):
                    out["rows"].append(dict(
                        model=(path, entry), replica=int(ip), kick=kick, leg=il, state=lab,
                        lattice=lv, mc=rr[0], mc_se=rr[1], mc_ess=rr[2], mc_out=rr[3],
                        rb_block=best[1], plain=pl[0], plain_se=pl[1], plain_ess=pl[2],
                        trec=tr, diag={k: v for k, v in rb.diag[il].items() if k != "shape"}))
    except Exception as exc:                                      # noqa: BLE001
        import traceback
        out["err"] = f"{type(exc).__name__}: {exc} | {traceback.format_exc()[-600:]}"
    out["t"] = time.perf_counter() - t0
    return out


def summarise(rows, se_max=2e-3):
    """Lattice against Monte Carlo: ln L_leg at every state, and the scan's
    shape (each state relative to the replica's unconstrained solution, the
    same draws, which is what the mass profile uses).  Rows whose MC standard
    error exceeds `se_max` do not resolve the comparison and are counted
    apart."""
    d = np.array([r["lattice"] - r["mc"] for r in rows])
    se = np.array([r["mc_se"] for r in rows])
    kick = np.array([r["kick"] for r in rows])
    side = np.array([0 if r["state"] == "mle" else (1 if r["state"].startswith("m+") else -1)
                     for r in rows])
    ok = se < se_max
    q = lambda a: (f"median {np.median(a):.1e} p90 {np.quantile(a, 0.9):.1e} "  # noqa: E731
                   f"max {np.max(a):.1e}")
    print(f"MC standard error: median {np.median(se):.1e}; resolved (se < {se_max:g}): "
          f"{int(ok.sum())}/{len(rows)}")
    pull = d / np.maximum(se, 1e-300)
    print(f"ln L_leg, lattice - MC, resolved: |d| {q(np.abs(d[ok]))}; |pull| {q(np.abs(pull[ok]))}; "
          f"mean pull {np.mean(pull[ok]):+.2f}")
    kmed = np.median(kick)
    for lab, m in (("ordinary", kick < kmed), ("large-loss", kick >= kmed)):
        m = m & ok
        if m.any():
            print(f"  {lab:10s} ({int(m.sum())} rows, kick/core max {np.max(kick[m]):.0f}): "
                  f"|d| {q(np.abs(d[m]))}; |pull| {q(np.abs(pull[m]))}")
    for lab, sd in (("below the truth", -1), ("above the truth", 1)):
        m = ok & (side == sd)
        if m.any():
            print(f"  {lab}: |pull| {q(np.abs(pull[m]))}, mean pull {np.mean(pull[m]):+.2f}")
    key = {}
    for r in rows:
        key.setdefault((r["model"], r["replica"], r["leg"]), {})[r["state"]] = r
    dd, ee = [], []
    for st in key.values():
        if "mle" not in st:
            continue
        for lab, r in st.items():
            if lab == "mle" or not (r["mc_se"] < se_max and st["mle"]["mc_se"] < se_max):
                continue
            dd.append((r["lattice"] - st["mle"]["lattice"]) - (r["mc"] - st["mle"]["mc"]))
            ee.append(np.hypot(r["mc_se"], st["mle"]["mc_se"]))
    if dd:
        dd, ee = np.array(dd), np.array(ee)
        print(f"scan shape, lattice - MC: |d| {q(np.abs(dd))}; |d|/se {q(np.abs(dd) / ee)} "
              f"(the draws are shared, so se is an upper bound)")
    diag = np.array([r["diag"].get("dropped", 0.0) for r in rows])
    for lo, hi in ((0, 0.01), (0.01, 0.05), (0.05, 1.0)):
        m = ok & (diag >= lo) & (diag < hi)
        if m.any():
            print(f"  dropped in [{lo:g}, {hi:g}): {int(m.sum())} rows, |d| median "
                  f"{np.median(np.abs(d[m])):.1e}, |pull| median {np.median(np.abs(pull[m])):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files")
    ap.add_argument("--models", type=int, default=12)
    ap.add_argument("--replicas", type=int, default=4)
    ap.add_argument("--big", type=int, default=2, help="of which the largest-loss ones")
    ap.add_argument("--pool", type=int, default=200)
    ap.add_argument("--scan", type=float, nargs="*", default=[-2, -1, 1, 2])
    ap.add_argument("--hscan", type=float, default=5e-3)
    ap.add_argument("--nmc", type=int, default=200000)
    ap.add_argument("--rbk", type=int, default=3, help="blocks tried for the Rao-Blackwellisation")
    ap.add_argument("--trim", type=float, default=25.0)
    ap.add_argument("--rbox", type=float, default=20.0)
    ap.add_argument("--h", type=float, default=0.3)
    ap.add_argument("--rmax", type=int, default=3)
    ap.add_argument("--floor", type=float, default=1e-11)
    ap.add_argument("--ess-rb", dest="ess_rb", type=float, default=2000.0)
    ap.add_argument("--order", type=int, default=9)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import uproot
    files = sorted(glob.glob(args.files))
    tasks = []
    per = max(1, int(np.ceil(args.models / len(files))))
    for f in files:
        with uproot.open(f) as fh:
            n = fh["tree"].num_entries
        for e in np.linspace(0, n - 1, per + 2)[1:-1].astype(int):
            tasks.append((f, int(e), args, args.seed + 7919 * len(tasks)))
            if len(tasks) >= args.models:
                break
        if len(tasks) >= args.models:
            break
    print(f"{len(tasks)} models x {args.replicas} replicas ({args.big} large-loss), "
          f"{args.nmc} MC draws, {args.jobs} workers", flush=True)
    t0 = time.perf_counter()
    rows, errs = [], []
    with Pool(args.jobs) as p:
        for r in p.imap_unordered(run, tasks):
            if r["err"]:
                errs.append(r["err"])
                print("ERR", r["err"][:400], flush=True)
            rows.extend(r["rows"])
            print(f"  model done ({len(rows)} rows, {time.perf_counter() - t0:.0f} s)", flush=True)
    print(f"\n{len(rows)} comparisons; failed models {len(errs)}")
    summarise(rows)
    if args.out:
        import pickle
        with open(args.out, "wb") as fh:
            pickle.dump(dict(rows=rows, errs=errs, args=vars(args)), fh)
        print("wrote", args.out)


if __name__ == "__main__":
    sys.exit(main())
