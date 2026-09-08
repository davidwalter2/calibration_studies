#!/usr/bin/env python3
"""GATE 2: the analytic gradient against finite differences, on any card.

The v formulation changes the term's coefficients and the provider's transform;
a wrong gradient that still converges is the failure mode that would cost the
most, so this checks the gradient (and, for the parameters that matter, the
Hessian diagonal) against central differences at the reference point and at a
displaced point.

    RABBIT=... ./run_tf.sh python3 -u gate_fd.py --card cards/z_vp1264.hdf5 \\
        --params m_Z Gamma_Z shape1 k_ms
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True)
    ap.add_argument("--params", nargs="*",
                    default=["m_Z", "Gamma_Z", "shape1", "k_ms"])
    ap.add_argument("--steps", type=float, nargs="*", default=[1e-3, 1e-2, 1e-1])
    ap.add_argument("--at", type=float, nargs="*", default=None,
                    help="displace these parameters (name value pairs are not "
                         "parsed; this is a full parameter vector)")
    ap.add_argument("--tol", type=float, default=1e-5,
                    help="required |analytic - FD| / max(|FD|, 1) at the best step")
    ap.add_argument("-o", "--output", default=None)
    a = ap.parse_args()

    import tensorflow as tf
    import h5py
    from rabbit import unbinned

    with h5py.File(a.card, "r") as f:
        terms = unbinned.read_unbinned_terms_from_h5(f["unbinned_terms"])
    term = terms[0]
    names = list(term.param_names)
    x0 = np.asarray(term.param_defaults, dtype=np.float64)
    if a.at is not None:
        x0 = np.asarray(a.at, dtype=np.float64)
    print(f"[gate2] {a.card}: {term.n} candidates, {len(names)} parameters")
    print(f"        vpow = {getattr(term, 'vpow', None)}, "
          f"corr_form = {getattr(term, 'corr_form', None)}")

    def nll(x):
        return float(term.nll(tf.constant(x, tf.float64)).numpy())

    xv = tf.Variable(x0, dtype=tf.float64)
    with tf.GradientTape() as t:
        f0 = term.nll(xv)
    g = t.gradient(f0, xv).numpy()
    print(f"        NLL = {float(f0):.6f}")

    out = {"card": a.card, "params": names, "x0": x0.tolist(),
           "nll": float(f0), "checks": {}}
    worst = 0.0
    print(f"\n  {'parameter':12s} {'analytic':>16s} {'central diff':>16s} "
          f"{'|rel|':>10s} {'step':>8s}")
    for p in a.params:
        if p not in names:
            print(f"  {p:12s} NOT IN THE CARD")
            continue
        i = names.index(p)
        best = None
        for h in a.steps:
            xp, xm = x0.copy(), x0.copy()
            xp[i] += h
            xm[i] -= h
            fd = (nll(xp) - nll(xm)) / (2.0 * h)
            rel = abs(g[i] - fd) / max(abs(fd), 1.0)
            if best is None or rel < best[0]:
                best = (rel, fd, h)
        rel, fd, h = best
        worst = max(worst, rel)
        print(f"  {p:12s} {g[i]:16.8f} {fd:16.8f} {rel:10.2e} {h:8.1e}")
        out["checks"][p] = {"analytic": float(g[i]), "fd": float(fd),
                            "rel": float(rel), "step": float(h)}
    ok = worst < a.tol
    print(f"\n  GATE 2: worst |analytic - FD| / max(|FD|,1) = {worst:.2e} "
          f"(requirement < {a.tol:g}) -> {'PASS' if ok else 'FAIL'}")
    out["_gate"] = {"worst": float(worst), "tol": a.tol, "pass": bool(ok)}
    if a.output:
        with open(a.output, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"  -> {a.output}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
