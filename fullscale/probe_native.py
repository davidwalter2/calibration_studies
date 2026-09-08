#!/usr/bin/env python3
"""Chunk-size scan and host-vs-device cross-check for the device chunk loop.

What it answers, on the real card and the real GPU:

* does `devobj.DeviceChunkedObjective` reproduce `chunkfit.ChunkedObjective`
  (it must, to round-off -- it calls the same `_chunk_nll`);
* what a value+gradient and a Hessian-vector product cost on each path;
* how both scale with `--chunk`, which is the memory / dispatch trade-off:
  the host path pays a python round trip and a device sync per chunk, so it
  wants few large chunks; the device path pays neither, so its optimum is set
  by the arithmetic alone;
* what the GPU is actually doing while each runs.

usage:
    python3 probe_native.py --card cards/z_n300k.hdf5 \
        --chunks 8192 32768 131072 262144
"""
import argparse
import os
import resource
import subprocess
import sys
import time

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import minimize_driver as md  # noqa: E402
from chunkfit import ChunkedObjective  # noqa: E402
from devobj import DeviceChunkedObjective, check_against_host  # noqa: E402


def gpu_mem_mb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        return max(float(v) for v in out.split())
    except Exception:
        return float("nan")


def timeit(fn, n=3):
    """Best of ``n`` after one warm-up (the first call traces the graph)."""
    t0 = time.time()
    fn()
    t_first = time.time() - t0
    best = np.inf
    for _ in range(n):
        t0 = time.time()
        fn()
        best = min(best, time.time() - t0)
    return t_first, best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--card", required=True)
    p.add_argument("--term", default=None)
    p.add_argument("--chunks", type=int, nargs="+", default=[32768])
    p.add_argument("--fix", nargs="*", default=["k_hit", "k_ms", "k_ioni", "k_rad"])
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--no-host", action="store_true", help="skip the host timings")
    p.add_argument("--no-check", action="store_true")
    args = p.parse_args()

    import h5py

    from rabbit import unbinned

    print("GPUs:", tf.config.list_physical_devices("GPU"))
    t0 = time.time()
    with h5py.File(args.card, "r") as f:
        terms = unbinned.read_unbinned_terms_from_h5(f["unbinned_terms"])
    term = terms[0] if args.term is None else next(t for t in terms if t.name == args.term)
    names = list(term.param_names)
    free = [i for i, nm in enumerate(names) if nm not in set(args.fix)]
    print(
        f"[probe] {term.name}: n {term.n}, nt {term.nt} "
        f"(integration {getattr(term, 'nt_int', term.nt)}), "
        f"card chunk {term.chunk}, loaded in {time.time()-t0:.0f} s"
    )
    print(f"        {len(free)} free: {[names[i] for i in free]}")

    x0 = np.asarray(term.param_defaults, np.float64)[free]
    pdir = np.linspace(1.0, -0.5, len(free))
    rows = []
    for ch in args.chunks:
        print(f"\n==== chunk {ch} ====")
        row = {"chunk": ch}
        base = gpu_mem_mb()
        try:
            dev = DeviceChunkedObjective([term], free, chunk=ch)
        except Exception as ex:
            print(f"  device objective failed: {type(ex).__name__}: {ex}")
            continue
        row["nchunk"] = dev.nchunk
        with md.GpuMonitor(0.2) as g:
            tf_first, t_vg = timeit(lambda: dev.value_grad(x0), args.repeat)
            _, t_hp = timeit(lambda: dev.hessp(x0, pdir), args.repeat)
        s = g.summary() or {}
        row.update(
            dev_trace=tf_first,
            dev_vg=t_vg,
            dev_hessp=t_hp,
            dev_util=s.get("util_mean", float("nan")),
            dev_utilp90=s.get("util_p90", float("nan")),
            dev_gpu_mb=(s.get("mem_max_mb", float("nan")) - base),
        )
        print(
            f"  device : {dev.nchunk:4d} chunks | trace {tf_first:7.1f} s | "
            f"value+grad {t_vg:7.3f} s | hessp {t_hp:7.3f} s | "
            f"GPU {row['dev_util']:5.1f} % (p90 {row['dev_utilp90']:5.1f} %) | "
            f"+{row['dev_gpu_mb']/1024:5.2f} GB"
        )
        if not args.no_host:
            host = ChunkedObjective([term], free, hess_mode="hvp", chunk=ch)
            if not args.no_check:
                check_against_host(dev, host, x0, log=lambda m: print("  " + m))
            with md.GpuMonitor(0.2) as g:
                _, h_vg = timeit(lambda: host.value_grad(x0), args.repeat)
                _, h_hp = timeit(lambda: host.hessp(x0, pdir), args.repeat)
            s = g.summary() or {}
            row.update(
                host_vg=h_vg,
                host_hessp=h_hp,
                host_util=s.get("util_mean", float("nan")),
                host_utilp90=s.get("util_p90", float("nan")),
            )
            print(
                f"  host   : {host.nchunk:4d} chunks |               | "
                f"value+grad {h_vg:7.3f} s | hessp {h_hp:7.3f} s | "
                f"GPU {row['host_util']:5.1f} % (p90 {row['host_utilp90']:5.1f} %)"
            )
            print(
                f"  SPEEDUP: value+grad {h_vg/t_vg:6.1f} x, "
                f"hessp {h_hp/t_hp:6.1f} x"
            )
            del host
        row["rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0**2
        rows.append(row)
        del dev

    print("\n==== summary ====")
    hdr = (
        f"{'chunk':>8s} {'nchunk':>7s} {'trace':>8s} {'dev v+g':>9s} "
        f"{'dev hvp':>9s} {'host v+g':>9s} {'host hvp':>9s} "
        f"{'x v+g':>7s} {'x hvp':>7s} {'GPU dev':>8s} {'GPU host':>9s} {'RSS':>7s}"
    )
    print(hdr)
    for r in rows:
        hv = r.get("host_vg", float("nan"))
        hh = r.get("host_hessp", float("nan"))
        print(
            f"{r['chunk']:8d} {r['nchunk']:7d} {r['dev_trace']:8.1f} "
            f"{r['dev_vg']:9.3f} {r['dev_hessp']:9.3f} {hv:9.3f} {hh:9.3f} "
            f"{hv/r['dev_vg']:7.1f} {hh/r['dev_hessp']:7.1f} "
            f"{r['dev_util']:7.1f}% {r.get('host_util', float('nan')):8.1f}% "
            f"{r['rss_gb']:6.1f}G"
        )


if __name__ == "__main__":
    main()
