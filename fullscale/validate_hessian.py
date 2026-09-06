#!/usr/bin/env python3
"""Gate the chunk-accumulated Hessian against the monolithic one.

`chunkfit.ChunkedObjective` claims to compute the SAME value, gradient and
Hessian as `zchannel/fit_z.Objective`, which differentiates the whole sample at
once. That claim is the licence to run a 3.7 M-candidate fit at all, so it is
measured rather than asserted: on a card small enough for the monolithic
construction to fit in memory, the two must agree at double round-off, at the
reference point AND at a displaced point (a Hessian that agrees only where the
gradient vanishes proves nothing).

It also times and sizes the two Hessian modes (`pfor` and the forward-over-
reverse `hvp`) against chunk size, which is the knob that trades the peak
against the number of times the lineshape provider's FFT and FSR fold are
recomputed -- they are candidate-INDEPENDENT, so chunking pays for them once
per chunk.

usage:
    python validate_hessian.py --card cards/z_n300k.hdf5 --fix k_hit k_ms k_ioni k_rad
"""
import argparse
import os
import resource
import sys
import time

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from chunkfit import ChunkedObjective  # noqa: E402

DTYPE = tf.float64


def monolithic(term, free, x):
    """value, gradient, Hessian with the whole sample on one tape."""
    v = tf.constant(x, DTYPE)
    fixed = tf.constant(np.asarray(term.param_defaults, np.float64), DTYPE)
    idx = tf.constant(np.array(free, dtype=np.int64)[:, None], tf.int32)
    ps = np.asarray(term.param_prior_sigmas, np.float64)
    pm_ = np.isfinite(ps)
    pmask = tf.constant(pm_.astype(np.float64), DTYPE)
    psig = tf.constant(np.where(pm_, ps, 1.0), DTYPE)
    pmean = tf.constant(np.asarray(term.param_prior_means, np.float64), DTYPE)

    def f(vv):
        xx = tf.tensor_scatter_nd_update(fixed, idx, vv)
        r = (xx - pmean) / psig
        return term.nll(xx) + 0.5 * tf.reduce_sum(pmask * r * r)

    with tf.GradientTape() as t2:
        t2.watch(v)
        with tf.GradientTape() as t1:
            t1.watch(v)
            val = f(v)
        g = t1.gradient(val, v)
    H = t2.jacobian(g, v)
    return float(val.numpy()), g.numpy(), H.numpy()


def main():
    import h5py
    from rabbit import unbinned

    p = argparse.ArgumentParser()
    p.add_argument("--card", required=True)
    p.add_argument("--fix", nargs="*", default=[])
    p.add_argument("--chunks", type=int, nargs="*",
                   default=[32768, 131072, 524288])
    p.add_argument("--displace", type=float, default=0.7,
                   help="displacement, in units of the reference-point error, "
                        "of the second test point")
    a = p.parse_args()

    with h5py.File(a.card, "r") as fh:
        term = unbinned.read_unbinned_terms_from_h5(fh["unbinned_terms"])[0]
    names = list(term.param_names)
    free = [i for i, nm in enumerate(names) if nm not in set(a.fix)]
    x0 = np.asarray(term.param_defaults, np.float64)[free]
    print(f"[validate] {term.n} candidates, free {[names[i] for i in free]}")

    t0 = time.time()
    v_m, g_m, H_m = monolithic(term, free, x0)
    t_m = time.time() - t0
    rss_m = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 ** 2
    print(f"  monolithic: NLL {v_m:.9f}  [{t_m:.1f} s, peak RSS {rss_m:.1f} GB]")

    # a displaced point, so the Hessian is compared where the gradient is not 0
    err = np.sqrt(np.diag(np.linalg.inv(H_m)))
    rng = np.random.default_rng(7)
    x1 = x0 + a.displace * err * rng.standard_normal(len(free))
    v_m1, g_m1, H_m1 = monolithic(term, free, x1)

    print(f"\n  {'chunk':>8s} {'mode':>5s} {'nchunk':>7s} {'dNLL':>11s} "
          f"{'dgrad':>11s} {'dHess':>11s} {'t_grad':>8s} {'t_hess':>8s} {'RSS':>8s}")
    for ch in a.chunks:
        for mode in ("pfor", "hvp"):
            obj = ChunkedObjective([term], free, hess_mode=mode, chunk=ch)
            t0 = time.time()
            v_c, g_c = obj.value_grad(x0)
            tg = time.time() - t0
            t0 = time.time()
            H_c = obj.hess(x0)
            th = time.time() - t0
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 ** 2
            dv = abs(v_c - v_m) / max(abs(v_m), 1e-300)
            dg = np.max(np.abs(g_c - g_m)) / max(np.max(np.abs(g_m)), 1e-300)
            dH = np.max(np.abs(H_c - H_m)) / max(np.max(np.abs(H_m)), 1e-300)
            print(f"  {ch:8d} {mode:>5s} {obj.nchunk:7d} {dv:11.3e} {dg:11.3e} "
                  f"{dH:11.3e} {tg:8.1f} {th:8.1f} {rss:7.1f}G")
            if ch == a.chunks[0]:
                v_c1, g_c1 = obj.value_grad(x1)
                H_c1 = obj.hess(x1)
                print(f"           displaced: dNLL "
                      f"{abs(v_c1-v_m1)/max(abs(v_m1),1e-300):.3e}  dgrad "
                      f"{np.max(np.abs(g_c1-g_m1))/max(np.max(np.abs(g_m1)),1e-300):.3e}"
                      f"  dHess "
                      f"{np.max(np.abs(H_c1-H_m1))/max(np.max(np.abs(H_m1)),1e-300):.3e}")
            del obj

    # the sandwich's own gate (it asserts internally that its per-candidate
    # vector reproduces the term's `_mix`)
    obj = ChunkedObjective([term], free, hess_mode="pfor", chunk=a.chunks[0])
    t0 = time.time()
    J = obj.sandwich(x0)
    print(f"\n  sandwich meat in {time.time()-t0:.1f} s; "
          f"sqrt(diag(J)) / sqrt(diag(2H)) = "
          f"{np.array2string(np.sqrt(np.diag(J) / np.maximum(2*np.diag(H_m), 1e-300)), precision=4)}")
    C = np.linalg.inv(H_m)
    print(f"  error ratio sandwich/inverse-Hessian = "
          f"{np.array2string(np.sqrt(np.diag(C @ J @ C)) / np.sqrt(np.diag(C)), precision=4)}")


if __name__ == "__main__":
    main()
