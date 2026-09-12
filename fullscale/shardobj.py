#!/usr/bin/env python3
"""The device chunk loop split over SEVERAL GPUs, sharded by CANDIDATE.

rabbit PR #154 (`rabbit/sharding.py`) shards the likelihood over **bins**:
`ShardIndataView` slices `norm` / `sumw` / `sumw2` / `logk` on the bins axis,
one tape per shard on a device-local copy of `x`, and the only things that
cross devices per evaluation are the parameter vector out and one `[nparams]`
partial back. None of that touches unbinned terms, because an unbinned term
has no bins -- it is a sum over candidates, and the shard views carry no
candidate axis. A sharded fit therefore evaluates unbinned terms in the
*global*, unsharded term, which is correct but puts their per-candidate
tensors on one device.

This module applies exactly PR #154's structure to the axis an unbinned term
does have. The two rules it establishes hold verbatim:

1. **backward placement must be explicit** -- one `GradientTape` per shard,
   differentiating w.r.t. a device-local copy of the parameter vector, so each
   shard's backward subgraph is anchored to its own device;
2. the per-shard functions are traced separately and the combiner stays thin.

What made it a contained change is `MassCFTerm.candidate_slice(a, b, device)`
(rabbit, this branch): the unbinned analogue of `ShardIndataView`. It detects
the per-candidate tensors rather than listing them -- every tf attribute whose
leading dimension is `n`, plus the `(n, nt)` family blocks -- and
re-materialises them on the shard's device. A hand-written list would stop
covering a tensor added later, and that failure mode is a wrong answer, not an
error.

What is NOT shared: the priors and any external quadratic are parameter-level,
so they are evaluated once in a "global" piece rather than once per shard.

usage:
    obj = ShardedChunkedObjective(terms, free=..., devices=2, chunk=...)
    val, grad = obj.value_grad(x)
"""
import numpy as np
import tensorflow as tf

from devobj import DeviceChunkedObjective

DTYPE = tf.float64


def select_devices(n, kind="GPU"):
    """The logical names of ``n`` devices, or ``None`` if there are not that many.

    ``kind="CPU"`` is for the unit test: TF will happily make several logical
    CPU devices out of one physical one, which exercises every placement and
    cross-device add without needing two GPUs.
    """
    devs = tf.config.list_logical_devices(kind)
    if len(devs) < n:
        return None
    return [d.name for d in devs[:n]]


def shard_edges(n, k):
    """Contiguous, balanced candidate ranges."""
    e = np.linspace(0, n, k + 1).astype(np.int64)
    return [(int(e[i]), int(e[i + 1])) for i in range(k)]


class ShardedChunkedObjective(DeviceChunkedObjective):
    """`DeviceChunkedObjective` whose chunks are split across GPUs.

    Parameters
    ----------
    devices : int or list[str]
        Number of GPUs, or explicit logical device names.

    The shard objectives are `DeviceChunkedObjective`s over the sliced terms
    with their priors turned off; this object keeps the priors and the external
    quadratic and adds the shards' partials.
    """

    def __init__(self, terms, free=None, chunk=None, log=print, external=None,
                 devices=2, kind="GPU"):
        if isinstance(devices, int):
            names = select_devices(devices, kind)
            if names is None:
                raise RuntimeError(
                    f"{devices} {kind}s requested, "
                    f"{len(tf.config.list_logical_devices(kind))} visible"
                )
        else:
            names = list(devices)
        # the parent builds the un-sharded graphs; they are never called here,
        # but everything the API needs (names, free set, priors, the selection
        # matrices, x0) comes from it and stays authoritative
        super().__init__(terms, free=free, chunk=chunk, log=log, external=external)
        self.devices = names
        self.shards = []
        for k, dev in enumerate(names):
            with tf.device(dev):
                sterms = []
                for t in self.terms:
                    a, b = shard_edges(t.n, len(names))[k]
                    sterms.append(t.candidate_slice(a, b, device=dev,
                                                    chunk=chunk or t.chunk))
                sh = DeviceChunkedObjective(sterms, free=self.free, log=lambda *a: None)
                # the priors belong to the whole fit, not to each shard
                sh._pmask = tf.zeros_like(sh._pmask)
                sh._pmask_free = tf.zeros_like(sh._pmask_free)
            self.shards.append((dev, sh))
        log(f"  sharded over {len(names)} device(s): "
            + ", ".join(f"{d} n={sum(t.n for t in sh.terms)}"
                        for d, sh in self.shards))
        self._f_value = tf.function(self._sharded_value)
        self._f_value_grad = tf.function(self._sharded_value_grad)
        self._f_hessp_at = tf.function(self._hessp_at_impl)

    # ------------------------------------------------------------------
    def _sharded_value(self, xf):
        parts = [self._prior(self.full(xf))]
        for dev, sh in self.shards:
            with tf.device(dev):
                parts.append(sh._value_impl(tf.identity(xf)))
        return tf.add_n(parts)

    def _sharded_value_grad(self, xf):
        with tf.GradientTape() as tape:
            tape.watch(xf)
            fp = self._prior(self.full(xf))
        gp = tape.gradient(fp, xf)
        vs, gs = [fp], [tf.zeros_like(xf) if gp is None else gp]
        for dev, sh in self.shards:
            with tf.device(dev):
                v, g = sh._value_grad_impl(tf.identity(xf))
            vs.append(v)
            gs.append(g)
        return tf.add_n(vs), tf.add_n(gs)

    def _hessp_at_impl(self, xf, p):
        out = self._pmask_free / self._psig_free**2 * p
        if self._ext_H is not None:
            out = out + tf.linalg.matvec(self._ext_H, p)
        parts = [out]
        for dev, sh in self.shards:
            with tf.device(dev):
                parts.append(sh._hessp_at_impl(tf.identity(xf), tf.identity(p)))
        return tf.add_n(parts)

    # the sandwich and the host-loop pieces are inherited and would use the
    # UN-sliced terms; that is correct (they are single-device by design) but
    # it means the sandwich is not sharded.
