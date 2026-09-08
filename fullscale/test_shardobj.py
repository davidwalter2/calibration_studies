#!/usr/bin/env python3
"""Candidate sharding reproduces the single-device objective, exactly.

Sharding splits a SUM, so it is an identity, not an approximation: the test
is that value, gradient and Hessian-vector product agree to float round-off
with the unsharded device objective on the same card, for 1, 2 and 3 shards.

Two logical CPU devices stand in for two GPUs when no GPU is present -- the
placement, the per-shard tapes and the cross-device add are all exercised;
only the bandwidth is different.

    RABBIT=../../rabbit-native ./run_tf.sh python3 -u test_shardobj.py \
        --card cards/smoke_zls.hdf5 --chunk 4096 --shards 1 2 3
"""
import argparse
import os
import sys

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--card", required=True)
    p.add_argument("--term", default=None)
    p.add_argument("--chunk", type=int, default=0)
    p.add_argument("--shards", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--fix", nargs="*", default=["k_hit", "k_ms", "k_ioni", "k_rad"])
    p.add_argument("--rtol", type=float, default=1e-10)
    args = p.parse_args()

    kind = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
    if kind == "CPU":
        phys = tf.config.list_physical_devices("CPU")[0]
        tf.config.set_logical_device_configuration(
            phys, [tf.config.LogicalDeviceConfiguration() for _ in range(4)]
        )
    print("sharding over", kind, tf.config.list_logical_devices(kind))

    import h5py

    from rabbit import unbinned

    from devobj import DeviceChunkedObjective
    from shardobj import ShardedChunkedObjective

    with h5py.File(args.card, "r") as f:
        terms = unbinned.read_unbinned_terms_from_h5(f["unbinned_terms"])
    term = terms[0] if args.term is None else next(t for t in terms if t.name == args.term)
    names = list(term.param_names)
    free = [i for i, nm in enumerate(names) if nm not in set(args.fix)]
    print(f"[test_shardobj] {term.name}: n {term.n}, free {[names[i] for i in free]}")

    ref = DeviceChunkedObjective([term], free, chunk=args.chunk or None)
    x = ref.x0[free] + 0.05 * np.linspace(1, -1, len(free))
    pdir = np.linspace(1.0, -0.5, len(free))
    v0, g0 = ref.value_grad(x)
    h0 = ref.hessp(x, pdir)
    print(f"  reference (1 device, {ref.nchunk} chunks): NLL {v0:.9f}")

    ok = True
    for k in args.shards:
        obj = ShardedChunkedObjective([term], free, chunk=args.chunk or None,
                                      devices=k, kind=kind,
                                      log=lambda m: print("   " + m))
        v, g = obj.value_grad(x)
        h = obj.hessp(x, pdir)
        dv = abs(v - v0) / max(abs(v0), 1.0)
        dg = np.max(np.abs(g - g0)) / max(np.max(np.abs(g0)), 1.0)
        dh = np.max(np.abs(h - h0)) / max(np.max(np.abs(h0)), 1.0)
        flag = "OK" if max(dv, dg, dh) < args.rtol else "MISMATCH"
        ok &= flag == "OK"
        print(f"  {k} shard(s): value {dv:.3e}, grad {dg:.3e}, hessp {dh:.3e}   {flag}")
        del obj
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
