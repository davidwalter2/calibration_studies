#!/usr/bin/env python3
"""Fit the Z channel datacard, and project its sensitivity to full statistics.

Reads the datacard written by ``make_z_card.py`` back through
``rabbit.unbinned.read_unbinned_terms_from_h5`` -- so this also exercises the
on-disk round trip -- minimises ``-sum log L_i / Z_i`` (plus the declared
Gaussian priors, which the rabbit ``Fitter`` would otherwise apply) with
scipy ``trust-exact`` on TF value/gradient/Hessian, and reports:

* the fitted parameters, their errors and the correlation matrix;
* the projected errors at ``--project N`` candidates, obtained by scaling the
  Hessian by ``N / n``.  The Hessian is evaluated at the *reference* point as
  well as at the minimum: at reference it is the expected (Asimov)
  information, which is the meaningful thing to extrapolate from a sample too
  small to locate the minimum precisely.
* the same with the resolution scales fixed, so the cost of profiling them is
  explicit.

Usage::

    python fit_z.py --card data/zcard_smoke.hdf5 --project 3.9e6
"""

import argparse
import json
import time

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize

DTYPE = tf.float64


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--card", required=True, help="datacard from make_z_card.py")
    p.add_argument("--term", default=None, help="term name (default: the only one)")
    p.add_argument("--project", type=float, nargs="+", default=[3.9e6],
                   help="candidate counts to project the covariance to")
    p.add_argument("--fix", nargs="*", default=[],
                   help="parameters to hold at their starting value")
    p.add_argument("--no-fit", action="store_true",
                   help="only the reference-point information, no minimisation")
    p.add_argument("--scan", nargs="*", default=[],
                   help="NAME:LO:HI:N -- 1D NLL scan (profiled over the rest)")
    return p.parse_args()


class Objective:
    """NLL of one unbinned term plus its declared Gaussian priors."""

    def __init__(self, term, free=None):
        self.term = term
        self.names = list(term.param_names)
        self.x0 = np.asarray(term.param_defaults, dtype=np.float64)
        self.prior_sigma = np.asarray(term.param_prior_sigmas, dtype=np.float64)
        self.prior_mean = np.asarray(term.param_prior_means, dtype=np.float64)
        self.free = list(range(len(self.names))) if free is None else list(free)
        self.freenames = [self.names[i] for i in self.free]
        self._fixed = tf.constant(self.x0, DTYPE)
        self._idx = tf.constant(np.array(self.free)[:, None], tf.int32)
        pm = np.isfinite(self.prior_sigma)
        self._pmask = tf.constant(pm.astype(np.float64), DTYPE)
        self._psig = tf.constant(np.where(pm, self.prior_sigma, 1.0), DTYPE)
        self._pmean = tf.constant(self.prior_mean, DTYPE)
        self.nbad = 0

    def full(self, xf):
        return tf.tensor_scatter_nd_update(self._fixed, self._idx, xf)

    def __call__(self, xf):
        x = self.full(xf)
        nll = self.term.nll(x)
        r = (x - self._pmean) / self._psig
        return nll + 0.5 * tf.reduce_sum(self._pmask * r * r)

    def value_grad(self, xf):
        v = tf.constant(xf, DTYPE)
        with tf.GradientTape() as t:
            t.watch(v)
            f = self(v)
        val = f.numpy()
        grad = t.gradient(f, v).numpy()
        if not (np.isfinite(val) and np.all(np.isfinite(grad))):
            # a resolution scale wandered far enough that the density (or the
            # window integral) went non-positive; steer back rather than crash
            self.nbad += 1
            return 1e30, np.where(np.isfinite(grad), grad, 0.0) + 1e6 * (
                xf - self.x0[self.free])
        return val, grad

    def hess(self, xf):
        v = tf.constant(xf, DTYPE)
        with tf.GradientTape() as t2:
            t2.watch(v)
            with tf.GradientTape() as t1:
                t1.watch(v)
                f = self(v)
            g = t1.gradient(f, v)
        H = t2.jacobian(g, v).numpy()
        if not np.all(np.isfinite(H)):
            return np.eye(len(xf)) * 1e6
        return H


def report_cov(names, H, n, projections, label):
    C = np.linalg.inv(H)
    err = np.sqrt(np.diag(C))
    corr = C / np.outer(err, err)
    print(f"\n  --- {label} (n = {n}) ---")
    w = max(len(s) for s in names) + 1
    print(f"    {'parameter':>{w}s} {'error':>12s}" +
          "".join(f" {f'N={p:.3g}':>12s}" for p in projections))
    for i, nm in enumerate(names):
        row = "".join(f" {err[i]*np.sqrt(n/p):12.4f}" for p in projections)
        print(f"    {nm:>{w}s} {err[i]:12.4f}{row}")
    print(f"    correlations:")
    print(f"    {'':>{w}s} " + " ".join(f"{s:>9s}" for s in names))
    for i, nm in enumerate(names):
        print(f"    {nm:>{w}s} " + " ".join(f"{corr[i,j]:9.4f}" for j in
                                            range(len(names))))
    return err, corr


def main():
    import h5py

    from rabbit import unbinned

    args = parse_args()
    with h5py.File(args.card, "r") as f:
        terms = unbinned.read_unbinned_terms_from_h5(f["unbinned_terms"])
    term = terms[0] if args.term is None else \
        next(t for t in terms if t.name == args.term)
    cfg = term.config()
    print(f"[fit_z] term '{term.name}': {term.n} candidates, nt = {term.nt}, "
          f"params {list(term.param_names)}")
    print(f"        kernel {cfg['kernel']['type']}"
          + (f" ({cfg['kernel']['provider']['type']}, window "
             f"{cfg['kernel']['provider']['window']})"
             if "provider" in cfg["kernel"] else ""))
    print(f"        norm_window {cfg['norm_window']}, "
          f"{cfg['norm_nodes']} nodes, background {cfg['background']['type']}")

    names = list(term.param_names)
    fixed = set(args.fix)
    free = [i for i, nm in enumerate(names) if nm not in fixed]
    obj = Objective(term, free)
    x0 = obj.x0[free]

    t0 = time.time()
    f0, g0 = obj.value_grad(x0)
    H0 = obj.hess(x0)
    print(f"\n  reference point: NLL = {f0:.6f}, |grad|inf = "
          f"{np.max(np.abs(g0)):.4g}  [{time.time()-t0:.1f} s]")
    ev = np.linalg.eigvalsh(H0)
    print(f"  Hessian eigenvalues {np.array2string(ev, precision=4)}"
          f"  (cond {ev.max()/max(ev.min(), 1e-300):.3g})")
    report_cov(obj.freenames, H0, term.n, args.project,
               "expected (Asimov) errors from the reference-point information")

    if args.no_fit:
        return

    t0 = time.time()
    res = minimize(obj.value_grad, x0, jac=True, hess=obj.hess,
                   method="trust-exact")
    H = obj.hess(res.x)
    print(f"\n  fit: {res.nit} iterations, {time.time()-t0:.0f} s, "
          f"NLL = {res.fun:.6f}, |grad|inf = {np.max(np.abs(res.jac)):.3g}"
          + (f", {obj.nbad} non-finite evaluations steered back" if obj.nbad else ""))
    err, corr = report_cov(obj.freenames, H, term.n, args.project,
                           "fitted values / observed information")
    print("\n    fitted values:")
    for nm, v, e in zip(obj.freenames, res.x, err):
        print(f"      {nm:>10s} = {v:+12.5f} +- {e:.5f}")

    for spec in args.scan:
        nm, lo, hi, npt = spec.split(":")
        i = obj.freenames.index(nm)
        others = [j for j in range(len(obj.freenames)) if j != i]
        print(f"\n  --- profile scan of {nm} ---")
        for v in np.linspace(float(lo), float(hi), int(npt)):
            sub = Objective(term, [free[j] for j in others])
            sub._fixed = tf.tensor_scatter_nd_update(
                sub._fixed, tf.constant([[free[i]]], tf.int32),
                tf.constant([v], DTYPE))
            r = minimize(sub.value_grad, res.x[others], jac=True,
                         hess=sub.hess, method="trust-exact")
            print(f"      {nm} = {v:+10.4f}   2*dNLL = "
                  f"{2*(r.fun - res.fun):10.5f}")


if __name__ == "__main__":
    main()
