"""Lattice convergence of the conditional-Kalman recursion on real frozen models:
the same toy replicas and the same scan states (the unconstrained solution and
constrained solutions at m_hat + k sigma_m, k in --scan), ln L under a list of
lattice settings, reported against the first one (the reference: finer,
wider, four axes).  What the closure
uses are the differences along the scan, so those are reported with the
common offset removed.

    python solver/rb_converge.py 'GLOB' --models 8 --replicas 3 --jobs 8
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

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

SETTINGS = {
    "ref":     dict(h=0.2, rbox=30, trim=30, floor=1e-13, order=9, rmax=4),
    "prod":    dict(h=0.3, rbox=20, trim=25, floor=1e-11, order=9),
    "h0.2":    dict(h=0.2, rbox=20, trim=25, floor=1e-11, order=9),
    "h0.4":    dict(h=0.4, rbox=20, trim=25, floor=1e-11, order=9),
    "rmax4":   dict(h=0.3, rbox=20, trim=25, floor=1e-11, order=9, rmax=4),
    "mm":      dict(h=0.3, rbox=20, trim=25, floor=1e-11, order=9, moment_match=True),
    "o7":      dict(h=0.3, rbox=20, trim=25, floor=1e-11, order=7),
    "rb12":    dict(h=0.3, rbox=12, trim=25, floor=1e-11, order=9),
    "fl1e-8":  dict(h=0.3, rbox=20, trim=18, floor=1e-8, order=9),
    "r2":      dict(h=0.3, rbox=20, trim=25, floor=1e-11, order=9, rmax=2),
}


def run(task):
    path, entry, args, seed = task
    import solver.record as R
    from solver.marginal import MarginalLikelihood, sample_noise
    from solver.mass import MassConstraint
    from solver.minimise import minimise
    from solver.profile import constrained_minimise
    from solver import rbgrid as RB
    out = dict(rows=[], err=None)
    try:
        R.MS_STRIDE = 13
        cand = next(iter(R.iter_candidates(path, entries={entry}, need_steps=True, extra=[])))
        gen = MarginalLikelihood(cand, gaussian=False)
        rb = RB.RBGridLikelihood(cand, gaussian=False, ml=MarginalLikelihood(cand, gaussian=False),
                                 cfg=RB.LatticeConfig(**SETTINGS[args.settings[0]]),
                                 run=False)
        rng = np.random.default_rng(seed)
        cons = MassConstraint(rb)
        for r in range(args.replicas):
            z = sample_noise(gen, rng)
            xs, rows = None, {}
            for name in args.settings:
                kw = dict(SETTINGS[name])
                rb.cfg = RB.LatticeConfig(**kw)
                t0 = time.perf_counter()
                rb.set_data(-z)
                trec = time.perf_counter() - t0
                if xs is None:
                    res = minimise(rb, tol=1e-9, maxiter=60)
                    J = cons.jac(res.delta)
                    sig = float(np.sqrt(J @ np.linalg.solve(res.hess, J)))
                    m0 = cons.value(res.delta)
                    xs = [res.delta]
                    for k in args.scan:
                        cr = constrained_minimise(rb, cons, m0 + k * sig, delta0=res.delta,
                                                  tol=1e-9, maxiter=80)
                        xs.append(cr.delta)
                vals = np.array([rb.value(x) for x in xs])
                grads = np.array([rb.value_grad(x)[1] for x in xs])
                rows[name] = dict(vals=vals, grads=grads, trec=trec,
                                  maxN=max(d.get("maxN", 0) for d in rb.diag),
                                  dropped=max(d.get("dropped", 0.0) for d in rb.diag))
            out["rows"].append(dict(model=(path, entry), replica=r, sd=np.sqrt(np.diag(np.linalg.inv(rb._A))),
                                    rows=rows))
    except Exception as exc:                                      # noqa: BLE001
        import traceback
        out["err"] = f"{type(exc).__name__}: {exc} | {traceback.format_exc()[-500:]}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files")
    ap.add_argument("--models", type=int, default=8)
    ap.add_argument("--replicas", type=int, default=3)
    ap.add_argument("--settings", nargs="*", default=list(SETTINGS))
    ap.add_argument("--scan", type=float, nargs="*", default=[-3, -1, 1, 3])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import uproot
    files = sorted(glob.glob(args.files))
    tasks = []
    for f in files:
        with uproot.open(f) as fh:
            n = fh["tree"].num_entries
        tasks.append((f, int(n // 2), args, args.seed + len(tasks)))
        if len(tasks) >= args.models:
            break
    with Pool(args.jobs) as p:
        res = list(p.imap_unordered(run, tasks))
    for o in res:
        if o["err"]:
            print("ERR", o["err"][:400])
    reps = [r for o in res for r in o["rows"]]
    ref = args.settings[0]
    print(f"{len(reps)} replicas; reference '{ref}'; scan k = {args.scan} sigma_m")
    print(f"{'setting':8s} {'rec s':>7s} {'maxN':>9s}  |scan-shape d lnL| median / max     "
          f"|grad diff| (sigma) max   dropped max")
    for name in args.settings:
        dd, gg, tt, nn, dr = [], [], [], [], []
        for r in reps:
            a, b = r["rows"][name], r["rows"][ref]
            d = a["vals"] - b["vals"]
            dd.append(np.abs(d[1:] - d[0]))
            gg.append(np.max(np.abs((a["grads"] - b["grads"]) * r["sd"][None, :])))
            tt.append(a["trec"])
            nn.append(a["maxN"])
            dr.append(a["dropped"])
        dd = np.concatenate(dd)
        print(f"{name:8s} {np.median(tt):7.2f} {int(np.median(nn)):9d}  {np.median(dd):.1e} / {np.max(dd):.1e}"
              f"                {np.max(gg):.1e}           {np.max(dr):.2e}")
    if args.out:
        import pickle
        with open(args.out, "wb") as fh:
            pickle.dump(reps, fh)


if __name__ == "__main__":
    sys.exit(main())
