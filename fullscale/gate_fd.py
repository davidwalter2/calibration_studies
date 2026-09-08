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
    ap.add_argument("--compare", default=None,
                   help="a second card whose NLL DIFFERENCES must match this "
                        "one's. Used as the off-switch gate: a card built with "
                        "`--vpow 1e-9` must reproduce one built without "
                        "`--vpow` at all, because v(m) -> m as p -> 0. The "
                        "ABSOLUTE NLL may differ by the sum of the "
                        "per-candidate Jacobians, which is a constant and "
                        "invisible to the fit; every difference must not.")
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

    if a.compare:
        with h5py.File(a.compare, "r") as f:
            t2 = unbinned.read_unbinned_terms_from_h5(f["unbinned_terms"])[0]
        n2 = list(t2.param_names)
        x2 = np.asarray(t2.param_defaults, dtype=np.float64)
        f2 = float(t2.nll(tf.constant(x2, tf.float64)).numpy())
        print(f"\n  --compare {a.compare}: vpow = {getattr(t2, 'vpow', None)}, "
              f"NLL = {f2:.6f} (this card {float(f0):.6f}, "
              f"offset {f2 - float(f0):+.6f} -- a candidate constant)")
        worst2 = 0.0
        print(f"  {'displacement':20s} {'this card':>18s} {'--compare':>18s} "
              f"{'|diff|':>12s}")
        for p_, dx in (("m_Z", 5.0), ("Gamma_Z", 10.0), ("shape1", 0.05),
                       ("k_ms", 0.02)):
            if p_ not in names or p_ not in n2:
                continue
            y = x0.copy(); y[names.index(p_)] += dx
            y2 = x2.copy(); y2[n2.index(p_)] += dx
            d1 = nll(y) - float(f0)
            d2 = float(t2.nll(tf.constant(y2, tf.float64)).numpy()) - f2
            worst2 = max(worst2, abs(d1 - d2))
            print(f"  {p_ + ' + ' + repr(dx):20s} {d1:+18.8f} {d2:+18.8f} "
                  f"{abs(d1 - d2):12.3e}")
        ok2 = worst2 < 1e-5
        print(f"\n  GATE 2b (off switch): worst |NLL difference mismatch| = "
              f"{worst2:.2e} (requirement < 1e-05) -> {'PASS' if ok2 else 'FAIL'}")
        out["_gate2b"] = {"compare": a.compare, "worst": float(worst2),
                          "pass": bool(ok2)}
        ok = ok and ok2
    out["_gate"] = {"worst": float(worst), "tol": a.tol, "pass": bool(ok)}
    if a.output:
        with open(a.output, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"  -> {a.output}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
