"""Gate A1 of the conditional-Kalman recursion: the Gaussian limit, exact.

Three settings, all with an exact answer (the Kalman filter of `rbgrid.kalman`
with the kicks' variances added to the process noise, which reproduces the
GLS marginal of `marginal.py` to 1e-15):

  gauss   every Urban law replaced by the Gaussian of the toy's Gaussian arm
          (variance lambda = the exported fm_QI eigenvalue): the kicks go to the
          analytic kernel and the lattice stays a single atom -- this checks the
          backward filter, the weights, the kernel and the mixture;
  split   the same kicks with half of each variance forced onto the LATTICE
          (a sub-lattice spike, re-gridded at every later kick);
  wide    Gaussian kicks of the big-jump scale (the knock-on count above |x| is
          `--wide-p`), wholly on the lattice: resolved rays, re-framing, the
          region-of-interest truncation.

Value, gradient (sigma units) and Hessian of -ln L in x_ref at the MLE and at
points along the mass direction and at random, per candidate.

    python solver/rb_gate_a1.py 'GLOB' --models 20 --jobs 20
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


class ExactGauss:
    """-ln L of the Gaussian model with kick variances `var[leg][i]` added along
    u_i, by the Kalman filter; gradient and Hessian by its linearity in x."""

    def __init__(self, rb, var):
        from solver.rbgrid import kalman
        self.rb, self.var, self.kalman = rb, var, kalman

    def value(self, x_free):
        y = -self.rb.red.c0
        x = self.rb.embed(x_free)
        return -sum(self.kalman(leg, y, x, extra_var=np.asarray(v))["ll"]
                    for leg, v in zip(self.rb.legs, self.var))

    def vgh(self, x_free, h=None):
        # the log-likelihood is exactly quadratic in x: central differences
        # with a step of a fraction of the Gaussian sigma are exact up to
        # round-off; the Hessian from the quadratic's own second differences
        A = self.rb._A
        sd = np.sqrt(np.diag(np.linalg.inv(A)))
        n = len(x_free)
        f0 = self.value(x_free)
        g = np.zeros(n)
        H = np.zeros((n, n))
        e = np.eye(n)
        for i in range(n):
            hi = 0.5 * sd[i]
            fp, fm = self.value(x_free + hi * e[i]), self.value(x_free - hi * e[i])
            g[i] = (fp - fm) / (2 * hi)
            H[i, i] = (fp - 2 * f0 + fm) / hi ** 2
        for i in range(n):
            for j in range(i + 1, n):
                hi, hj = 0.5 * sd[i], 0.5 * sd[j]
                fpp = self.value(x_free + hi * e[i] + hj * e[j])
                fmm = self.value(x_free - hi * e[i] - hj * e[j])
                fpm = self.value(x_free + hi * e[i] - hj * e[j])
                fmp = self.value(x_free - hi * e[i] + hj * e[j])
                H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * hi * hj)
        return f0, g, H


def run(task):
    path, entry, args = task
    import solver.record as R
    from solver.marginal import MarginalLikelihood
    from solver.mass import MassConstraint
    from solver.minimise import minimise
    from solver.profile import constrained_minimise
    from solver import rbgrid as RB
    from solver.rb_census import jump_scale
    out = dict(path=path, entry=entry, err=None, rows=[])
    try:
        R.MS_STRIDE = 13
        cand = next(iter(R.iter_candidates(path, entries={entry}, need_steps=True, extra=[])))
        legsU = RB.build_legs(MarginalLikelihood(cand, gaussian=False))   # the Urban laws
        for mode in args.modes:
            cfg = RB.LatticeConfig(h=args.h, rbox=args.rbox, trim=args.trim, rmax=args.rmax,
                                   floor=args.floor, order=args.order, rtol=args.rtol)
            rb = RB.RBGridLikelihood(cand, gaussian=True, cfg=cfg,
                                     ml=MarginalLikelihood(cand, gaussian=True), run=False)
            var = []
            for leg, laws in zip(rb.legs, rb.laws):
                vl = []
                for k, law in enumerate(laws):
                    if mode == "wide":
                        v = jump_scale(legsU[leg.leg].cgf[k], args.wide_p) ** 2
                        law.var = v
                        law.split = 0.0
                    elif mode == "split":
                        law.split = 0.5
                        v = law.var
                    else:
                        v = law.var
                    law._core = None
                    vl.append(v)
                var.append(vl)
            ex = ExactGauss(rb, var)
            t0 = time.perf_counter()
            rb.set_data(rb.red.c0)
            trec = time.perf_counter() - t0
            res = minimise(rb, tol=1e-10, maxiter=60)
            cons = MassConstraint(rb)
            Hm = res.hess
            J = cons.jac(res.delta)
            sig = float(np.sqrt(J @ np.linalg.solve(Hm, J)))
            m0 = cons.value(res.delta)
            pts = [("mle", res.delta)]
            for k in (-3, -1, 1, 3):
                cr = constrained_minimise(rb, cons, m0 + k * sig, delta0=res.delta, tol=1e-10,
                                          maxiter=80)
                pts.append((f"m{k:+d}", cr.delta))
            rng = np.random.default_rng(entry)
            C = np.linalg.inv(rb._A)
            for k in range(2):
                pts.append((f"rnd{k}", res.delta + rng.multivariate_normal(np.zeros(rb.nfree), C)))
            sd = np.sqrt(np.diag(C))
            for lab, x in pts:
                v1, g1, H1 = rb.value_grad_hess(x)
                v0, g0, H0 = ex.vgh(x)
                out["rows"].append(dict(mode=mode, pt=lab, dv=float(v1 - v0), v=float(v0),
                                        dg=float(np.max(np.abs((g1 - g0) * sd))),
                                        dH=float(np.max(np.abs(H1 - H0)) / np.max(np.abs(H0))),
                                        trec=trec, maxN=max(d.get("maxN", 0) for d in rb.diag),
                                        r=max(d.get("r", 0) for d in rb.diag)))
    except Exception as exc:                                      # noqa: BLE001
        import traceback
        out["err"] = f"{type(exc).__name__}: {exc} | {traceback.format_exc()[-600:]}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files")
    ap.add_argument("--models", type=int, default=10)
    ap.add_argument("--modes", nargs="*", default=["gauss", "split", "wide"])
    ap.add_argument("--wide-p", dest="wide_p", type=float, default=0.05)
    ap.add_argument("--h", type=float, default=0.3)
    ap.add_argument("--rbox", type=float, default=20.0)
    ap.add_argument("--trim", type=float, default=25.0)
    ap.add_argument("--rmax", type=int, default=3)
    ap.add_argument("--floor", type=float, default=1e-11)
    ap.add_argument("--order", type=int, default=9)
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import uproot
    files = sorted(glob.glob(args.files))
    tasks = []
    for f in files:
        with uproot.open(f) as fh:
            n = fh["tree"].num_entries
        tasks.append((f, int(n // 3), args))
        if len(tasks) >= args.models:
            break
    with Pool(args.jobs) as p:
        res = list(p.imap_unordered(run, tasks))
    rows = [r for o in res for r in o["rows"]]
    for o in res:
        if o["err"]:
            print("ERR", o["err"][:500])
    print(f"{len(rows)} points on {len(res)} models; h {args.h} rbox {args.rbox} trim {args.trim} "
          f"rmax {args.rmax} floor {args.floor} order {args.order} rtol {args.rtol:g}")
    for mode in args.modes:
        rr = [r for r in rows if r["mode"] == mode]
        if not rr:
            continue
        for sel, lab in ((lambda r: r["pt"] in ("mle", "m-1", "m+1"), "mode +-1 sigma"),
                         (lambda r: r["pt"] in ("m-3", "m+3"), "+-3 sigma"),
                         (lambda r: r["pt"].startswith("rnd"), "random 1-sigma")):
            q = [r for r in rr if sel(r)]
            if not q:
                continue
            dv = np.array([abs(r["dv"]) for r in q])
            print(f"  {mode:6s} {lab:15s}: |d ln L| median {np.median(dv):.1e} max {dv.max():.1e}; "
                  f"grad (sigma) max {max(r['dg'] for r in q):.1e}; Hessian rel max "
                  f"{max(r['dH'] for r in q):.1e}")
        print(f"         recursion {np.median([r['trec'] for r in rr]):.2f} s median, lattice "
              f"maxN median {np.median([r['maxN'] for r in rr]):.0f}, r max {max(r['r'] for r in rr)}")
    if args.out:
        import pickle
        with open(args.out, "wb") as fh:
            pickle.dump(rows, fh)


if __name__ == "__main__":
    sys.exit(main())
