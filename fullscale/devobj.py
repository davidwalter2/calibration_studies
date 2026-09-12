#!/usr/bin/env python3
"""The chunked unbinned objective with the CHUNK LOOP ON THE DEVICE.

WHY THIS EXISTS.  `chunkfit.ChunkedObjective` evaluates the same NLL as
`rabbit.unbinned` but chunk by chunk, so the peak memory is set by ONE chunk
instead of the whole sample.  It does that with a **python** loop, in eager
mode, and it converts to numpy at the end of every chunk::

    for it, t, ci in self._pieces():
        with tf.GradientTape() as tape: ...
        val += float(f.numpy())          # device -> host sync, per chunk
        grad += tape.gradient(f, v).numpy()

Every TF op in the chunk is dispatched individually from python, and the two
`.numpy()` calls force a device sync that stops any overlap between the host
building chunk `c+1` and the device finishing chunk `c`.  On the full-scale
fits that costs an order of magnitude: **8-9 % GPU utilisation** on an H200.
The device is idle almost all the time; the fit is bound by python.

This module evaluates the SAME objective -- it subclasses
`ChunkedObjective` and calls its `_chunk_nll`, so the arithmetic is literally
the same method -- with the loop expressed as a `tf.while_loop` inside a
`tf.function`:

* the whole loop is ONE graph dispatch per value / gradient / Hessian-vector
  product, not `nchunk` python round trips;
* the graph is traced ONCE (the chunk index is a traced int32 tensor, and
  `rabbit.unbinned.ChunkTable` turns it into a dynamic `strided_slice`), so
  the last, short chunk needs no second trace;
* `parallel_iterations=1` keeps exactly one chunk live, which is the whole
  point of chunking -- a `tf.function` with an unrolled python loop would let
  the executor start all `nchunk` chunks at once and blow the memory budget;
* the gradient is taken INSIDE the loop body and accumulated, so no chunk's
  forward tape outlives its iteration (differentiating an unrolled loop from
  the outside would keep all of them).

The Hessian-vector product is forward-over-reverse in the same body, which is
what `rabbit.minimizer`'s native trust-region methods ask for: `hessp` there is
called from inside the solver's own compiled CG / Lanczos loop, so it has to be
graph-compatible.  :meth:`DeviceChunkedObjective.native_hessp` is exactly that
-- a pure-TF callable reading the pinned point out of a `tf.Variable`.

usage:
    from devobj import DeviceChunkedObjective
    obj = DeviceChunkedObjective(terms, free=..., chunk=...)
    val, grad = obj.value_grad(x)          # numpy, drop-in for ChunkedObjective
    hv = obj.hessp(x, p)
    # or, for rabbit.minimizer:
    minimize_trust_krylov(obj.native_fun, obj.native_closure,
                          obj.native_hessp(), obj.native_set_point, x0)
"""
import numpy as np
import tensorflow as tf

from chunkfit import ChunkedObjective

DTYPE = tf.float64


class DeviceChunkedObjective(ChunkedObjective):
    """`ChunkedObjective` whose candidate loop runs as a `tf.while_loop`.

    Parameters are the parent's, plus

    external : object or None
        Anything with a ``Hfree`` (nfree x nfree) matrix and a
        ``value_grad(xf)`` -- `fit_joint.ExternalQuadratic`.  Kept here rather
        than bolted on outside so the quadratic's contribution to the HVP is a
        tf matvec inside the same graph, instead of a numpy one that would
        force the Krylov loop back onto the host.
    """

    def __init__(self, terms, free=None, chunk=None, log=print, external=None):
        try:
            from rabbit.unbinned import ChunkTable  # noqa: F401
        except ImportError:
            raise ImportError(
                "the device chunk loop needs rabbit.unbinned.ChunkTable (the "
                "native-minimizer merge, branch material-resolution-native); "
                "point RABBIT= at that worktree or use ChunkedObjective"
            )
        super().__init__(terms, free=free, hess_mode="hvp", chunk=chunk, log=log)
        bad = [
            t.name
            for t in self.terms
            if not getattr(t, "graph_chunkable", False)
            or not isinstance(t._chunks, ChunkTable)
        ]
        if bad:
            raise ValueError(
                f"terms {bad} cannot run the device chunk loop (their per-chunk "
                "slices need a python chunk index); use ChunkedObjective"
            )
        self.external = external
        self._ext_H = None
        if external is not None:
            self._ext_H = tf.constant(np.asarray(external.Hfree, np.float64), DTYPE)
        n = len(self.free)
        # the linearization point the graph-compatible HVP is taken at. A
        # Variable, not a constant: the native minimizers re-pin it between
        # solves (`set_point`) without retracing the solver's graph.
        self._xv = tf.Variable(np.zeros(n), dtype=DTYPE, trainable=False)
        self._nchunks = [int(t.nchunk) for t in self.terms]
        self._f_value = tf.function(self._value_impl)
        self._f_value_grad = tf.function(self._value_grad_impl)
        self._f_hessp_at = tf.function(self._hessp_at_impl)
        # the Gaussian priors restricted to the free subspace, once
        self._pmask_free = tf.constant(self._pmask.numpy()[self.free], DTYPE)
        self._psig_free = tf.constant(self._psig.numpy()[self.free], DTYPE)
        self.ncall = {"value": 0, "grad": 0, "hessp": 0}

    def attach_external(self, external):
        """Hand over the external quadratic AFTER construction.

        `fit_joint.ExternalQuadratic` needs the parameter names and the free
        set, which come from this object, so it cannot be built before it.
        The graphs are rebuilt because `_ext_H` is captured by the traced HVP.
        """
        self.external = external
        self._ext_H = tf.constant(np.asarray(external.Hfree, np.float64), DTYPE)
        self._f_value = tf.function(self._value_impl)
        self._f_value_grad = tf.function(self._value_grad_impl)
        self._f_hessp_at = tf.function(self._hessp_at_impl)
        return self

    # ------------------------------------------------------------------
    # the graph bodies
    # ------------------------------------------------------------------
    def _term_loop(self, it, xf, init, body_piece):
        """`tf.while_loop` over the chunks of term ``it``.

        ``body_piece(ci, xf) -> tuple`` returns this chunk's contribution to
        each accumulator; they are summed.  ``parallel_iterations=1`` is not a
        tuning knob -- it is what bounds the live memory to one chunk.
        """
        nch = self._nchunks[it]
        if nch == 0:
            return init

        def cond(ci, *acc):
            return ci < nch

        def body(ci, *acc):
            piece = body_piece(ci, xf)
            return (ci + 1,) + tuple(a + b for a, b in zip(acc, piece))

        out = tf.while_loop(
            cond,
            body,
            (tf.constant(0, tf.int32),) + tuple(init),
            parallel_iterations=1,
            swap_memory=False,
            maximum_iterations=nch,
        )
        return out[1:]

    def _chunk_value(self, it, ci, xf):
        t = self.terms[it]
        x = self.full(xf)
        vals = t._values(tf.linalg.matvec(self._sel[it], x))
        return self._chunk_nll(t, vals, ci)

    def _value_impl(self, xf):
        tot = self._prior(self.full(xf))
        for it in range(len(self.terms)):
            (v,) = self._term_loop(
                it,
                xf,
                (tf.zeros([], DTYPE),),
                lambda ci, x, it=it: (self._chunk_value(it, ci, x),),
            )
            tot = tot + v
        return tot

    def _value_grad_impl(self, xf):
        zero_g = tf.zeros_like(xf)
        with tf.GradientTape() as tape:
            tape.watch(xf)
            fp = self._prior(self.full(xf))
        gp = tape.gradient(fp, xf)
        tot, grad = fp, (zero_g if gp is None else gp)

        for it in range(len(self.terms)):

            def piece(ci, x, it=it):
                with tf.GradientTape() as tp:
                    tp.watch(x)
                    f = self._chunk_value(it, ci, x)
                g = tp.gradient(f, x)
                return f, (tf.zeros_like(x) if g is None else g)

            v, g = self._term_loop(it, xf, (tf.zeros([], DTYPE), zero_g), piece)
            tot, grad = tot + v, grad + g
        return tot, grad

    def _hessp_at_impl(self, xf, p):
        """``H @ p`` at ``xf``, forward-over-reverse, one chunk at a time."""
        # priors: analytic diagonal, and the external quadratic is exactly its
        # own Hessian -- neither needs a tape
        out = self._pmask_free / self._psig_free**2 * p
        if self._ext_H is not None:
            out = out + tf.linalg.matvec(self._ext_H, p)

        for it in range(len(self.terms)):

            def piece(ci, x, it=it):
                with tf.autodiff.ForwardAccumulator(x, p) as acc:
                    with tf.GradientTape() as tp:
                        tp.watch(x)
                        f = self._chunk_value(it, ci, x)
                    g = tp.gradient(f, x)
                    g = tf.zeros_like(x) if g is None else g
                hv = acc.jvp(g)
                return (tf.zeros_like(x) if hv is None else hv,)

            (hv,) = self._term_loop(it, xf, (tf.zeros_like(xf),), piece)
            out = out + hv
        return out

    # ------------------------------------------------------------------
    # numpy-facing API -- drop-in for ChunkedObjective
    # ------------------------------------------------------------------
    def _tf(self, xf):
        return tf.constant(np.asarray(xf, np.float64), DTYPE)

    def value(self, xf):
        self.ncall["value"] += 1
        v = float(self._f_value(self._tf(xf)).numpy())
        if self.external is not None:
            v += self.external.value_grad(xf)[0]
        return v

    def value_grad(self, xf):
        self.ncall["grad"] += 1
        v, g = self._f_value_grad(self._tf(xf))
        val, grad = float(v.numpy()), g.numpy()
        if self.external is not None:
            ev, eg = self.external.value_grad(xf)
            val, grad = val + ev, grad + eg
        if not (np.isfinite(val) and np.all(np.isfinite(grad))):
            # same steering the host objective does: a scale wandered far
            # enough that the density (or the window integral) went
            # non-positive. Push back towards the starting point rather than
            # handing the minimizer a NaN.
            self.nbad += 1
            return 1e30, np.where(np.isfinite(grad), grad, 0.0) + 1e6 * (
                np.asarray(xf) - self.x0[self.free]
            )
        return val, grad

    def hessp(self, xf, p):
        self.ncall["hessp"] += 1
        out = self._f_hessp_at(self._tf(xf), self._tf(p)).numpy()
        if not np.all(np.isfinite(out)):
            return np.zeros_like(out)
        return out

    def hess(self, xf, mode=None):
        """The dense Hessian as ``nfree`` Hessian-vector products.

        The host objective's `pfor` mode keeps `nfree` copies of a chunk's tape
        live and is what costs 390 s (and 99x the memory) on the full-scale
        card. Here each column is one graph call over the chunks at the memory
        of a single gradient, so the whole matrix costs `nfree` chunk passes --
        seconds, not minutes, and flat in memory.
        """
        n = len(self.free)
        v = self._tf(xf)
        H = np.empty((n, n))
        eye = np.eye(n)
        for j in range(n):
            H[:, j] = self._f_hessp_at(v, tf.constant(eye[j], DTYPE)).numpy()
        if not np.all(np.isfinite(H)):
            return np.eye(n) * 1e6
        return 0.5 * (H + H.T)

    # ------------------------------------------------------------------
    # rabbit.minimizer-facing API
    # ------------------------------------------------------------------
    def native_fun(self, x_np):
        """Objective only, used by the trust-region loop to judge proposals."""
        v = self.value(x_np)
        return 1e30 if not np.isfinite(v) else v

    def native_closure(self, x_np):
        """``(value, grad)`` with the gradient left as a tf tensor.

        `minimize_trust_ncg` / `minimize_trust_krylov` want it that way: the
        gradient goes straight into the on-device Krylov solve without a host
        round trip.
        """
        self.native_set_point(x_np)
        self.ncall["grad"] += 1
        v, g = self._f_value_grad(self._xv)
        val = float(v.numpy())
        if self.external is not None:
            ev, eg = self.external.value_grad(x_np)
            val += ev
            g = g + tf.constant(eg, DTYPE)
        if not np.isfinite(val):
            self.nbad += 1
            val = 1e30
        return val, g

    def native_closure_hess(self, x_np):
        """``(value, grad, hess)`` for `tf-trust-exact`."""
        val, g = self.native_closure(x_np)
        return val, g, tf.constant(self.hess(x_np), DTYPE)

    def native_set_point(self, x_np):
        self._xv.assign(np.asarray(x_np, np.float64))

    def native_hessp(self):
        """A graph-compatible ``v -> H @ v`` at the pinned point.

        Returned as a closure (not a bound method) because that is the shape
        `rabbit.minimizer` expects, and because the solver traces it once into
        its own CG / Lanczos graph -- the pinned point has to be read from a
        Variable for that trace to stay valid across re-pinnings.
        """

        def hessp(v):
            # NOT self._f_hessp_at: this is traced INTO the solver's own CG /
            # Lanczos graph, so it must build ops rather than call a graph.
            return self._hessp_at_impl(self._xv, v)

        return hessp


# ---------------------------------------------------------------------------
def check_against_host(dev, host, x, log=print, rtol=1e-10):
    """Assert the device objective reproduces the host one at ``x``.

    Value, gradient and one Hessian-vector product. Nothing here is an
    approximation, so the tolerance is float round-off over a sum of `nchunk`
    terms, not a physics tolerance.
    """
    p = np.linspace(1.0, -0.5, len(x))
    vh, gh = host.value_grad(x)
    vd, gd = dev.value_grad(x)
    hh = host.hessp(x, p)
    hd = dev.hessp(x, p)
    dv = abs(vd - vh) / max(abs(vh), 1.0)
    dg = np.max(np.abs(gd - gh)) / max(np.max(np.abs(gh)), 1.0)
    dh = np.max(np.abs(hd - hh)) / max(np.max(np.abs(hh)), 1.0)
    log(f"  device vs host: value {dv:.3e}, grad {dg:.3e}, hessp {dh:.3e}")
    worst = max(dv, dg, dh)
    if worst > rtol:
        raise RuntimeError(
            f"the device chunk loop does not reproduce the host one "
            f"(worst relative difference {worst:.3e} > {rtol:.1e})"
        )
    return dict(value=dv, grad=dg, hessp=dh)
