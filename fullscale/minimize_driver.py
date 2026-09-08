#!/usr/bin/env python3
"""One minimiser entry point for `fit.py` and `fit_joint.py`.

Two axes, deliberately orthogonal:

**engine** -- who evaluates the objective.
  ``host``    `chunkfit.ChunkedObjective`: a python loop over candidate
              chunks, eager TF, `.numpy()` per chunk. The reference.
  ``device``  `devobj.DeviceChunkedObjective`: the same arithmetic with the
              chunk loop as a `tf.while_loop` inside a `tf.function`.

**method** -- who decides the steps.
  ``trust-exact`` / ``trust-krylov`` / ``trust-ncg``
              `scipy.optimize.minimize`. The reference.
  ``tf-trust-exact`` / ``tf-trust-ncg`` / ``tf-trust-krylov``
              `rabbit.minimizer` (PR #153): the same trust-region outer loop,
              with the subproblem -- the Cholesky factorizations of
              trust-exact, the whole CG / Lanczos inner loop of the other two
              -- kept on the TF device. A Krylov solve is then ONE graph
              dispatch instead of one python round trip (x assignment, two
              numpy conversions, a forced device sync) per Hessian-vector
              product.

The native methods need a *graph-compatible* `hessp`, which only the device
engine has, so ``--method tf-*`` implies ``--engine device``.

Snapshots (PR #155) are wired here rather than in each driver: the parameter
vector is written after every accepted iteration, on SIGTERM/SIGINT (a slurm
wall-clock kill, a Ctrl-C) and on failure, so an interrupted fit is a fit that
resumes rather than a fit that is lost.
"""
import json
import os
import subprocess
import threading
import time

import numpy as np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
SCIPY_METHODS = ("trust-exact", "trust-krylov", "trust-ncg")
NATIVE_METHODS = ("tf-trust-exact", "tf-trust-ncg", "tf-trust-krylov")
ALL_METHODS = SCIPY_METHODS + NATIVE_METHODS


def add_arguments(p, default_method="trust-exact"):
    """The engine / method / snapshot flags shared by both drivers."""
    p.add_argument(
        "--method",
        choices=list(ALL_METHODS),
        default=default_method,
        help="`trust-exact` builds the FULL Hessian every iteration; the "
        "Krylov methods ask for Hessian-VECTOR products instead. The `tf-` "
        "prefixed ones are rabbit's NATIVE TensorFlow trust-region minimizers "
        "(rabbit/minimizer/, PR #153): same iterates, but the subproblem stays "
        "on the device instead of round-tripping through numpy/LAPACK once per "
        "Hessian-vector product. They require --engine device.",
    )
    p.add_argument(
        "--engine",
        choices=["host", "device", "auto"],
        default="auto",
        help="who evaluates the objective. `host` is `chunkfit."
        "ChunkedObjective` (python loop over chunks, eager TF, a device sync "
        "per chunk -- 8-9 %% GPU utilisation on the full-scale fits); `device` "
        "is `devobj.DeviceChunkedObjective`, the same arithmetic with the "
        "chunk loop as a tf.while_loop inside a tf.function. `auto` picks "
        "device for a `tf-` method and host otherwise (the reference path).",
    )
    p.add_argument(
        "--check-device",
        action="store_true",
        help="before fitting, assert the device objective reproduces the host "
        "one (value, gradient, one Hessian-vector product) to round-off. Costs "
        "one host gradient + one host HVP.",
    )
    p.add_argument(
        "--snapshot-file",
        default=None,
        help="write the parameter vector here after every accepted iteration, "
        "on SIGTERM/SIGINT and on failure. Resume with --resume. Defaults to "
        "<output>.snapshot.hdf5 when -o is given and --snapshot-interval > 0.",
    )
    p.add_argument(
        "--snapshot-interval",
        type=float,
        default=0.0,
        help="hours between periodic snapshots (0 disables the periodic ones; "
        "the interrupt / failure / convergence snapshots do not depend on it).",
    )
    p.add_argument(
        "--resume",
        default=None,
        help="seed the fit from a snapshot written by --snapshot-file. Names "
        "are matched, so a snapshot of a different free set still seeds the "
        "parameters it has in common.",
    )
    p.add_argument(
        "--gpu-monitor",
        type=float,
        default=0.0,
        help="sample `nvidia-smi --query-gpu=utilization.gpu,memory.used` every "
        "this many seconds during the fit and report the distribution. 0 off.",
    )
    return p


def resolve_engine(args, log=print):
    engine = args.engine
    if engine == "auto":
        engine = "device" if args.method in NATIVE_METHODS else "host"
    if args.method in NATIVE_METHODS and engine != "device":
        raise SystemExit(
            f"--method {args.method} needs a graph-compatible hessp, which "
            "only --engine device provides"
        )
    log(f"  minimiser: method {args.method}, engine {engine}")
    return engine


# ---------------------------------------------------------------------------
# GPU sampling
# ---------------------------------------------------------------------------
class GpuMonitor:
    """Sample GPU utilisation and memory in a thread, for the report.

    `nvidia-smi dmon` would do the same, but a thread here has the fit's own
    start/stop times, so the numbers describe the fit rather than the job.
    """

    def __init__(self, period=1.0):
        self.period = float(period)
        self.util = []
        self.mem = []
        self._stop = threading.Event()
        self._thread = None
        self.available = period > 0

    def _run(self):
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.wait(self.period):
            try:
                out = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=10
                ).stdout
            except Exception:
                return
            for line in out.strip().splitlines():
                try:
                    u, m = (float(v) for v in line.split(","))
                except ValueError:
                    continue
                self.util.append(u)
                self.mem.append(m)

    def __enter__(self):
        if self.available:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return False

    def summary(self):
        if not self.util:
            return None
        u = np.asarray(self.util)
        m = np.asarray(self.mem)
        return {
            "n_samples": int(u.size),
            "util_mean": float(u.mean()),
            "util_median": float(np.median(u)),
            "util_p90": float(np.percentile(u, 90)),
            "util_max": float(u.max()),
            "mem_max_mb": float(m.max()),
            "mem_mean_mb": float(m.mean()),
        }


# ---------------------------------------------------------------------------
# snapshots
# ---------------------------------------------------------------------------
def make_snapshotter(args, names, log=print):
    """A `rabbit.snapshot.Snapshotter` over the FREE parameter names, or None."""
    path = args.snapshot_file
    if path is None and args.snapshot_interval > 0 and getattr(args, "output", None):
        path = os.path.splitext(args.output)[0] + ".snapshot.hdf5"
    if path is None:
        return None
    try:
        from rabbit.snapshot import Snapshotter
    except ImportError:
        log("  snapshots requested but this rabbit has no rabbit/snapshot.py")
        return None
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    log(f"  snapshots -> {path} (every {args.snapshot_interval} h, plus "
        f"interrupt / failure / convergence)")
    return Snapshotter(path, np.asarray(names), interval_hours=args.snapshot_interval)


def load_snapshot(path, freenames, x0, log=print):
    """Overwrite the entries of ``x0`` whose names the snapshot carries."""
    import h5py

    with h5py.File(path, "r") as f:
        parms = [str(s) for s in np.asarray(f["parms"]).astype(str)]
        x = np.asarray(f["x"])
        reason = f.attrs.get("reason", "?")
    have = dict(zip(parms, x))
    moved = [nm for nm in freenames if nm in have]
    x0 = np.asarray(x0, np.float64).copy()
    for i, nm in enumerate(freenames):
        if nm in have:
            x0[i] = float(have[nm])
    log(f"  resumed from {path} (written at '{reason}'): {len(moved)} of "
        f"{len(freenames)} parameters seeded")
    return x0


class _Callback:
    """Per-iteration bookkeeping shared by scipy and the native loop.

    scipy's trust-region methods call ``callback(xk)``; `rabbit.minimizer`
    calls ``callback(OptimizeResult(x=..., fun=...))``. Both are accepted.
    """

    def __init__(self, snapshotter=None, log=print, every=0):
        self.snap = snapshotter
        self.log = log
        self.every = int(every)
        self.nit = 0
        self.t0 = time.time()
        self.history = []

    def __call__(self, arg, *rest):
        x = getattr(arg, "x", arg)
        fun = getattr(arg, "fun", None)
        self.nit += 1
        el = time.time() - self.t0
        self.history.append((self.nit, el, None if fun is None else float(fun)))
        if self.every and self.nit % self.every == 0:
            self.log(
                f"    it {self.nit:4d}  {el:8.1f} s"
                + (f"  NLL {fun:.6f}" if fun is not None else "")
            )
        if self.snap is not None:
            self.snap.maybe_save(np.asarray(x), el, nit=self.nit)


# ---------------------------------------------------------------------------
# the entry point
# ---------------------------------------------------------------------------
def minimize(obj, x0, args, snapshotter=None, log=print, log_every=1):
    """Run ``args.method`` on ``obj`` from ``x0``; return the OptimizeResult.

    ``obj`` must have ``value_grad`` and ``hess``/``hessp`` (host or device);
    a native method additionally needs the device engine's ``native_*``.
    """
    cb = _Callback(snapshotter, log=log, every=log_every)
    if snapshotter is not None:
        snapshotter.update(np.asarray(x0))

    def _run():
        if args.method in SCIPY_METHODS:
            import scipy.optimize

            opts = {"maxiter": args.maxiter, "gtol": args.gtol}
            if args.method == "trust-exact":
                return scipy.optimize.minimize(
                    obj.value_grad, x0, jac=True, hess=obj.hess,
                    method="trust-exact", options=opts, callback=cb,
                )
            return scipy.optimize.minimize(
                obj.value_grad, x0, jac=True, hessp=obj.hessp,
                method=args.method, options=opts, callback=cb,
            )

        from rabbit.minimizer import (
            minimize_trust_exact,
            minimize_trust_krylov,
            minimize_trust_ncg,
        )

        if args.method == "tf-trust-exact":
            return minimize_trust_exact(
                obj.native_fun, obj.native_closure_hess, x0,
                gtol=args.gtol, maxiter=args.maxiter, callback=cb,
            )
        fn = (
            minimize_trust_ncg
            if args.method == "tf-trust-ncg"
            else minimize_trust_krylov
        )
        return fn(
            obj.native_fun, obj.native_closure, obj.native_hessp(),
            obj.native_set_point, x0,
            gtol=args.gtol, maxiter=args.maxiter, callback=cb,
        )

    try:
        from rabbit.snapshot import snapshot_on_signal
    except ImportError:
        snapshot_on_signal = None

    if snapshotter is None or snapshot_on_signal is None:
        try:
            res = _run()
        except Exception as ex:
            if snapshotter is not None:
                snapshotter.save_latest("minimizer-failed", error=str(ex)[:200])
            raise
    else:
        with snapshot_on_signal(snapshotter):
            try:
                res = _run()
            except Exception as ex:
                snapshotter.save_latest("minimizer-failed", error=str(ex)[:200])
                raise
        snapshotter.save(np.asarray(res.x), "converged")
    res.callback = cb
    return res


def timing_report(obj, res, t_fit, gpu=None, log=print):
    """One dict summarising the run, for the json and the report table."""
    out = {
        "nit": int(getattr(res, "nit", -1)),
        "nfev": int(getattr(res, "nfev", -1)),
        "nhev": int(getattr(res, "nhev", -1)),
        "t_fit": float(t_fit),
        "status": int(getattr(res, "status", -1)),
        "message": str(getattr(res, "message", "")),
    }
    calls = getattr(obj, "ncall", None)
    if calls is not None:
        out["ncall"] = dict(calls)
        out["t_per_grad"] = (
            t_fit / calls["grad"] if calls.get("grad") else float("nan")
        )
    if gpu is not None:
        s = gpu.summary()
        if s:
            out["gpu"] = s
            log(
                f"  GPU over the fit: mean {s['util_mean']:.1f} %, median "
                f"{s['util_median']:.1f} %, p90 {s['util_p90']:.1f} %, peak mem "
                f"{s['mem_max_mb']/1024:.1f} GB ({s['n_samples']} samples)"
            )
    log(
        f"  minimiser: {out['nit']} iterations, {t_fit:.1f} s"
        + (f", {out['ncall']}" if "ncall" in out else "")
    )
    return out


def dump_json(path, payload, log=print):
    if not path:
        return
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1, default=str)
    log(f"\n  -> {path}")
