#!/usr/bin/env python3
"""The device chunk loop reproduces the host one, and the native minimizers
reproduce scipy, on a real card.

Run as a script (not pytest), like the other fullscale checks::

    RABBIT=../../rabbit-native ./run_tf.sh python3 -u test_devobj.py \
        --card cards/smoke_zls.hdf5 --chunk 8192

Checks
------
1. value / gradient / Hessian-vector product of `devobj.DeviceChunkedObjective`
   against `chunkfit.ChunkedObjective` at three parameter points, to float
   round-off. Nothing is approximated, so a difference means a bug.
2. the dense Hessian built from device HVPs against the host `pfor` Hessian.
3. `nchunk` invariance: the same numbers for two different chunk sizes (the
   traced graph has to handle the short last chunk).
4. a full minimisation with each native method against the scipy reference:
   the minima must agree to 1e-6 relative on every parameter.
"""
import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from chunkfit import ChunkedObjective  # noqa: E402
from devobj import DeviceChunkedObjective, check_against_host  # noqa: E402


def load(card, name=None, chunk=None):
    import h5py

    from rabbit import unbinned

    with h5py.File(card, "r") as f:
        terms = unbinned.read_unbinned_terms_from_h5(f["unbinned_terms"])
    t = terms[0] if name is None else next(x for x in terms if x.name == name)
    return t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--card", required=True)
    p.add_argument("--term", default=None)
    p.add_argument("--chunk", type=int, default=0)
    p.add_argument("--chunk2", type=int, default=0, help="second chunking for check 3")
    p.add_argument("--fix", nargs="*", default=["k_hit", "k_ms", "k_ioni", "k_rad"])
    p.add_argument("--no-fit", action="store_true")
    p.add_argument("--rtol", type=float, default=1e-9)
    args = p.parse_args()

    t0 = time.time()
    term = load(args.card, args.term)
    names = list(term.param_names)
    free = [i for i, nm in enumerate(names) if nm not in set(args.fix)]
    print(f"[test_devobj] {term.name}: n {term.n}, params {names}")
    print(f"              free {[names[i] for i in free]}, loaded {time.time()-t0:.0f} s")

    host = ChunkedObjective([term], free, hess_mode="pfor", chunk=args.chunk or None)
    dev = DeviceChunkedObjective([term], free, chunk=args.chunk or None)
    x0 = host.x0[free]
    print(f"              {host.nchunk} chunks of {term.chunk}")

    print("\n=== 1. value / grad / hessp against the host loop ===")
    rng = np.random.default_rng(0)
    for k in range(3):
        x = x0 if k == 0 else x0 + 0.3 * rng.normal(size=len(free))
        print(f"  point {k}: {np.array2string(x, precision=4)}")
        check_against_host(dev, host, x, rtol=args.rtol)

    print("\n=== 2. dense Hessian: device HVP columns vs the host pfor jacobian ===")
    t0 = time.time()
    Hh = host.hess(x0, mode="pfor")
    th = time.time() - t0
    t0 = time.time()
    Hd = dev.hess(x0)
    td = time.time() - t0
    d = np.max(np.abs(Hd - Hh)) / max(np.max(np.abs(Hh)), 1.0)
    print(f"  host pfor {th:.2f} s, device hvp {td:.2f} s, max rel diff {d:.3e}")
    assert d < 1e-8, f"Hessians disagree at {d:.3e}"

    if args.chunk2:
        print(f"\n=== 3. chunk invariance ({args.chunk or term.chunk} vs {args.chunk2}) ===")
        term2 = load(args.card, args.term)
        dev2 = DeviceChunkedObjective([term2], free, chunk=args.chunk2)
        v1, g1 = dev.value_grad(x0)
        v2, g2 = dev2.value_grad(x0)
        dv = abs(v2 - v1) / max(abs(v1), 1.0)
        dg = np.max(np.abs(g2 - g1)) / max(np.max(np.abs(g1)), 1.0)
        print(f"  {dev.nchunk} vs {dev2.nchunk} chunks: value {dv:.3e}, grad {dg:.3e}")
        assert max(dv, dg) < 1e-9
        del dev2, term2

    if args.no_fit:
        print("\nALL CHECKS PASSED (no fit requested)")
        return

    print("\n=== 4. native minimizers vs scipy, same start ===")
    from scipy.optimize import minimize

    from rabbit.minimizer import (
        minimize_trust_exact,
        minimize_trust_krylov,
        minimize_trust_ncg,
    )

    t0 = time.time()
    ref = minimize(
        host.value_grad,
        x0,
        jac=True,
        hess=host.hess,
        method="trust-exact",
        options={"maxiter": 200, "gtol": 1e-8},
    )
    tref = time.time() - t0
    print(
        f"  scipy trust-exact (host): {ref.nit} it, {tref:.1f} s, "
        f"NLL {ref.fun:.9f}, |g|inf {np.max(np.abs(ref.jac)):.3g}"
    )

    ok = True
    for method, fn in (
        ("tf-trust-krylov", minimize_trust_krylov),
        ("tf-trust-ncg", minimize_trust_ncg),
        ("tf-trust-exact", minimize_trust_exact),
    ):
        t0 = time.time()
        if method == "tf-trust-exact":
            r = fn(dev.native_fun, dev.native_closure_hess, x0, gtol=1e-8, maxiter=200)
        else:
            r = fn(
                dev.native_fun,
                dev.native_closure,
                dev.native_hessp(),
                dev.native_set_point,
                x0,
                gtol=1e-8,
                maxiter=200,
            )
        dt = time.time() - t0
        dx = np.max(np.abs(r.x - ref.x)) / max(np.max(np.abs(ref.x)), 1.0)
        dn = abs(r.fun - ref.fun) / max(abs(ref.fun), 1.0)
        flag = "OK" if (dx < 1e-6 and dn < 1e-10) else "MISMATCH"
        ok &= flag == "OK"
        print(
            f"  {method:16s} {r.nit:3d} it, {dt:7.1f} s, NLL {r.fun:.9f}, "
            f"dx {dx:.2e}, dNLL {dn:.2e}   {flag}"
        )
        print(f"      {np.array2string(r.x, precision=6)}")
    print(f"      {np.array2string(ref.x, precision=6)}  (scipy)")
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
