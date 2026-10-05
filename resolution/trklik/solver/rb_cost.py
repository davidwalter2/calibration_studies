"""Gate A4 of the conditional-Kalman recursion: the cost per candidate.

Single-threaded timings on real frozen models (one model per file, spread
over the files), split into the per-model setup (the leg models, the kick
laws, the first recursion with the Levy-kernel builds), the recursion for new
data (`set_data`, what a data candidate pays once), the unconstrained
minimisation and the modified profile at the toy's three mass points, the
number of objective evaluations, and the cost of one evaluation.  A cProfile
of one recursion gives the split between the numerical kernels.

    python solver/rb_cost.py 'GLOB' --models 24 --replicas 3 --jobs 12 \
        --out solver/runs/rbgrid/a4_cost.pkl
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# the script's own directory holds solver/profile.py, which would shadow the
# standard library's `profile` that cProfile imports
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
sys.path.insert(0, os.path.dirname(HERE))

import argparse
import cProfile
import glob
import io
import pickle
import pstats
import time
import warnings
from multiprocessing import Pool

import numpy as np


class Counted:
    """Count the objective evaluations a minimiser/profile makes."""

    def __init__(self, m):
        self.m = m
        self.n = 0

    def __getattr__(self, k):
        return getattr(self.m, k)

    def value(self, x):
        self.n += 1
        return self.m.value(x)

    def value_grad(self, x):
        self.n += 1
        return self.m.value_grad(x)

    def value_grad_hess(self, x, *a, **k):
        self.n += 1
        return self.m.value_grad_hess(x, *a, **k)

    def fisher_at(self, d):
        self.n += 1
        return self.m.fisher_at(d)

    def mixed_hessian(self, d, hess=None):
        if hess is None:
            self.n += 1
        return self.m.mixed_hessian(d, hess)


def run_model(task):
    path, entry, args, seed = task
    warnings.filterwarnings("ignore")
    import solver.record as R
    from solver.marginal import MarginalLikelihood, sample_noise
    from solver.mass import MassConstraint
    from solver.minimise import minimise
    from solver.profile import mass_profile
    from solver.rbgrid import LatticeConfig, RBGridLikelihood
    import uproot
    out = dict(path=path, entry=entry, err=None, rep=[])
    try:
        with uproot.open(path) as fh:
            t = fh["tree"]
            ms0 = np.asarray(t["msmoliv"].array(entry_start=0, entry_stop=1, library="np")[0])
            nms0 = np.asarray(t["fm_nmssteps"].array(entry_start=0, entry_stop=1, library="np")[0])
        st = int(len(ms0) // max(int(np.sum(nms0)), 1))
        if st in (11, 13):
            R.MS_STRIDE = st
        cand = next(iter(R.iter_candidates(path, entries={entry}, need_steps=True, extra=[])))
        cfg = LatticeConfig(h=args.h, rbox=args.rbox, trim=args.trim, rmax=args.rmax,
                            order=args.order, floor=args.floor)
        # the first construction in a process builds the module-level electron
        # tables of cf_ms_exact (a one-time cost per process, not per candidate)
        tw = time.process_time()
        MarginalLikelihood(cand, gaussian=False)
        out["t_import"] = time.process_time() - tw
        t0 = time.process_time()
        ml = MarginalLikelihood(cand, gaussian=False)
        t1 = time.process_time()
        mR = RBGridLikelihood(cand, gaussian=False, cfg=cfg, ml=ml, run=False)
        cons = MassConstraint(mR)
        mtrue = cons.value(np.zeros(mR.nfree))
        t2 = time.process_time()
        gen = MarginalLikelihood(cand, gaussian=False)
        rng = np.random.default_rng(seed)
        z0 = sample_noise(gen, rng)
        mR.set_data(-z0)                      # the first recursion builds the kernels
        t3 = time.process_time()
        out.update(t_ml=t1 - t0, t_legs=t2 - t1, t_first=t3 - t2,
                   nhits=int(cand.nhits), nkick=[int(len(leg.lam)) for leg in mR.legs])
        hstep = args.hscan
        masses = mtrue * (1.0 + np.array([-hstep, 0.0, hstep]))
        if args.profile_one:
            pr = cProfile.Profile()
            pr.enable()
            mR.set_data(-z0)
            pr.disable()
            s = io.StringIO()
            pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(14)
            out["prof"] = s.getvalue()
        for r in range(args.replicas):
            z = sample_noise(gen, rng)
            a = time.process_time()
            mR.set_data(-z)
            b = time.process_time()
            mc = Counted(mR)
            res = minimise(mc, tol=args.tol, maxiter=60)
            c = time.process_time()
            n_min = mc.n
            mc.n = 0
            prf = mass_profile(mc, masses, cons=cons, unconstrained=res, form="modprofile",
                               tol=args.tol, maxiter=60)
            d = time.process_time()
            n_prof = mc.n
            x = res.delta
            e0 = time.process_time()
            for _ in range(5):
                mR.value_grad_hess(x)
            e1 = time.process_time()
            out["rep"].append(dict(t_rec=b - a, t_min=c - b, t_prof=d - c, n_min=n_min,
                                   n_prof=n_prof, t_eval=(e1 - e0) / 5,
                                   maxN=[int(dg.get("maxN", 0)) for dg in mR.diag],
                                   shape=[dg.get("shape") for dg in mR.diag],
                                   ok=bool(res.converged) and bool(np.all(prf["converged"]))))
    except Exception as exc:  # noqa: BLE001
        import traceback
        out["err"] = traceback.format_exc()
    return out


def summarise(res, args):
    good = [r for r in res if r["err"] is None]
    print(f"{len(good)}/{len(res)} models; settings h={args.h} rbox={args.rbox} "
          f"trim={args.trim} rmax={args.rmax} order={args.order}")
    for r in res:
        if r["err"] is not None:
            print("FAILED", r["path"], r["entry"], r["err"].splitlines()[-1])
    q = lambda a: (np.median(a), np.percentile(a, 90), np.mean(a), np.max(a))  # noqa: E731

    def line(name, a, unit="s"):
        m, p90, mean, mx = q(np.asarray(a, float))
        print(f"  {name:28s} median {m:8.3f} {unit}  p90 {p90:8.3f}  mean {mean:8.3f}  max {mx:8.3f}")

    line("once per process: tables", [r["t_import"] for r in good])
    line("setup: MarginalLikelihood", [r["t_ml"] for r in good])
    line("setup: legs + laws", [r["t_legs"] for r in good])
    line("setup: first recursion", [r["t_first"] for r in good])
    reps = [p for r in good for p in r["rep"]]
    line("recursion (set_data)", [p["t_rec"] for p in reps])
    line("minimise", [p["t_min"] for p in reps])
    line("profile, 3 mass points", [p["t_prof"] for p in reps])
    line("evaluations: minimise", [p["n_min"] for p in reps], "")
    line("evaluations: profile", [p["n_prof"] for p in reps], "")
    line("one evaluation", [1e3 * p["t_eval"] for p in reps], "ms")
    line("lattice atoms (max leg)", [1e-6 * max(p["maxN"]) for p in reps], "M")
    tot = [p["t_rec"] + p["t_min"] + p["t_prof"] for p in reps]
    line("per replica total", tot)
    first = [r["t_ml"] + r["t_legs"] + r["t_first"] for r in good]
    one = [f + p["t_min"] + p["t_prof"] for f, r in zip(first, good) for p in r["rep"][:1]]
    line("per data candidate (1 fit)", one)
    print(f"  not converged: {sum(not p['ok'] for p in reps)} of {len(reps)}")
    for r in good[:1]:
        if "prof" in r:
            print(r["prof"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glob")
    ap.add_argument("--models", type=int, default=24)
    ap.add_argument("--replicas", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=4711)
    ap.add_argument("--h", type=float, default=0.3)
    ap.add_argument("--rbox", type=float, default=20.0)
    ap.add_argument("--trim", type=float, default=25.0)
    ap.add_argument("--rmax", type=int, default=3)
    ap.add_argument("--order", type=int, default=9)
    ap.add_argument("--floor", type=float, default=1e-11)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--hscan", type=float, default=1.5e-4)
    ap.add_argument("--profile-one", dest="profile_one", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    files = sorted(glob.glob(args.glob))
    step = max(len(files) // args.models, 1)
    tasks = []
    import uproot
    for k, f in enumerate(files[::step][: args.models]):
        with uproot.open(f) as fh:
            n = fh["tree"].num_entries
        e = int(np.random.default_rng(args.seed + k).integers(0, n))
        tasks.append((f, e, args, args.seed + 7919 * k))
    with Pool(args.jobs) as p:
        res = p.map(run_model, tasks, chunksize=1)
    summarise(res, args)
    if args.out:
        with open(args.out, "wb") as fh:
            pickle.dump(dict(res=res, args=vars(args)), fh)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
