#!/usr/bin/env python3
"""Value / gradient / Hessian of unbinned terms, ACCUMULATED OVER CHUNKS.

WHY THIS EXISTS. `rabbit_fit.py` and `zchannel/fit_z.py` both build the
Hessian by differentiating the WHOLE sample's NLL twice: rabbit with
`tf.vectorized_map` (pfor) over the parameters, `fit_z.py` with
`GradientTape.jacobian`. Either way the forward tape of every candidate is
live at once, and the peak is `n x nt_int x nparams x sizeof(float64)` times a
small constant. Measured: **68 GB at 300 k candidates and 5 parameters**
(engaging/README.md sec. 5), which is why the 300 k J/psi card OOMs on a 46 GB
L40S. At the 3.9 M candidates of the DY production and 8-10 parameters the
same construction asks for ~1 TB and cannot be run anywhere.

It does not have to be that way. The NLL is a SUM over candidates,

    F(theta) = sum_c f_c(theta) + priors ,

so its gradient and its Hessian are the same sums, and a chunk's tape can be
released before the next chunk is taped. This module evaluates exactly the
same objective as `fit_z.Objective` -- the same `MassCFTerm` methods, in the
same order -- but chunk by chunk, so the peak is set by ONE chunk. Nothing is
approximated: `--check` asserts agreement with the monolithic version to
floating-point round-off.

Two Hessian modes:

`pfor` (default)
    per chunk, `t2.jacobian(g, v)` -- the monolithic construction restricted
    to a chunk. Peak ~ `chunk x nt_int x nparams`. Fastest when it fits.
`hvp`
    per chunk and per parameter j, a forward-over-reverse Hessian-vector
    product with `tf.autodiff.ForwardAccumulator`. Peak ~ 2 x ONE gradient,
    i.e. INDEPENDENT of nparams, at nparams times the passes. This is the mode
    that keeps working when the parameter vector grows to the ~92 globals of
    the joint fit.

Both are exact; `--check` compares them.

`hessp(x, p)` is ONE such product with an arbitrary tangent, which is what a
Krylov trust-region minimiser (`trust-krylov`, `trust-ncg`) asks for instead of
the matrix. It matters: at 99 free parameters `pfor` OOMs an H200 at
`chunk 32768` and, where it fits, `trust-exact` needs one FULL Hessian per
iteration -- ~10 000 s each on the joint card, ~105 h for the 38 iterations the
Z-alone fit took. A Krylov step needs O(10) `hessp` calls, each ~2 gradients and
independent of the parameter count.

usage:
    from chunkfit import ChunkedObjective
    obj = ChunkedObjective(terms, free=..., hess_mode="pfor")
    val, grad = obj.value_grad(x)
    H = obj.hess(x)
"""
import time

import numpy as np
import tensorflow as tf

DTYPE = tf.float64


class ChunkedObjective:
    """NLL of one or more unbinned terms plus their declared Gaussian priors.

    Parameters
    ----------
    terms : list
        `rabbit.unbinned.UnbinnedTerm` instances. They must share one
        parameter vector: `param_names` of the FIRST term defines the order,
        and every other term's parameters must be a subset of it (they are
        gathered by name).
    free : list[int] or None
        Indices of the parameters to vary; the rest stay at their default.
    hess_mode : {"pfor", "hvp"}
    chunk : int or None
        Override the terms' own chunk size. A term's `_chunks` list is
        rebuilt, which is safe: it is derived data, not state.
    """

    def __init__(self, terms, free=None, hess_mode="pfor", chunk=None,
                 log=print):
        self.terms = list(terms)
        self.log = log
        t0 = self.terms[0]
        self.names = list(t0.param_names)
        self.x0 = np.asarray(t0.param_defaults, dtype=np.float64)
        self.prior_sigma = np.asarray(t0.param_prior_sigmas, dtype=np.float64)
        self.prior_mean = np.asarray(t0.param_prior_means, dtype=np.float64)
        for t in self.terms[1:]:
            missing = set(t.param_names) - set(self.names)
            if missing:
                raise ValueError(
                    f"term '{t.name}' has parameters {sorted(missing)} that the "
                    f"first term does not declare; build a joint declaration first"
                )
        # Per-term selection of the shared vector into the term's own order,
        # as a DENSE 0/1 matrix rather than `tf.gather`. A gather's gradient is
        # an `IndexedSlices`, and `tf.autodiff.ForwardAccumulator` cannot
        # differentiate through one (`AttributeError: 'IndexedSlices' object
        # has no attribute '_id'`), which kills the HVP Hessian and the
        # sandwich. The matrices are nparams x nparams -- a few hundred floats.
        self._sel = []
        for t in self.terms:
            S = np.zeros((len(t.param_names), len(self.names)))
            for r, p in enumerate(t.param_names):
                S[r, self.names.index(p)] = 1.0
            self._sel.append(tf.constant(S, DTYPE))
        if chunk:
            for t in self.terms:
                self._rechunk(t, int(chunk))
        self.free = list(range(len(self.names))) if free is None else list(free)
        self.freenames = [self.names[i] for i in self.free]
        self.hess_mode = hess_mode
        # `full` as an affine map, for the same reason: the gradient of
        # `tensor_scatter_nd_update` w.r.t. its updates is also a gather.
        base = self.x0.copy()
        base[self.free] = 0.0
        self._base = tf.constant(base, DTYPE)
        E = np.zeros((len(self.names), len(self.free)))
        for c, i in enumerate(self.free):
            E[i, c] = 1.0
        self._embed = tf.constant(E, DTYPE)
        pm = np.isfinite(self.prior_sigma)
        self._pmask = tf.constant(pm.astype(np.float64), DTYPE)
        self._psig = tf.constant(np.where(pm, self.prior_sigma, 1.0), DTYPE)
        self._pmean = tf.constant(self.prior_mean, DTYPE)
        self.nbad = 0
        self.nchunk = sum(t.nchunk for t in self.terms)

    # ------------------------------------------------------------------
    @staticmethod
    def _rechunk(term, chunk):
        term.chunk = int(min(chunk, term.n)) if term.n else 1
        term.nchunk = int(np.ceil(term.n / term.chunk)) if term.n else 0
        term._chunks = [
            (i * term.chunk, min(term.n, (i + 1) * term.chunk))
            for i in range(term.nchunk)
        ]

    def full(self, xf):
        """The whole parameter vector, with the free entries embedded in."""
        return self._base + tf.linalg.matvec(self._embed, xf)

    def _prior(self, x):
        r = (x - self._pmean) / self._psig
        return 0.5 * tf.reduce_sum(self._pmask * r * r)

    def _chunk_nll(self, term, values, ci):
        """One chunk's contribution -- `MassCFTerm.nll`'s loop body, verbatim."""
        li = term._chunk_li(values, ci)
        if term._norm is not None:
            z = term._norm_z(values)
            lo, hi = term._chunks[ci]
            li = li / tf.gather(z, term._norm_class[lo:hi])
        return term._mix(values, li, ci)

    @staticmethod
    def _chunk_logl_vec(term, values, ci, li):
        """Per-candidate ``w_i log L_i`` of chunk ``ci``.

        This is ``MassCFTerm._mix`` WITHOUT its final ``-reduce_sum``, which is
        what a sandwich needs: the score of each candidate separately. The
        arithmetic is transcribed rather than refactored into rabbit, and
        :meth:`sandwich` asserts ``-sum(vec) == _mix(...)`` on the first chunk,
        so a drift between the two is caught rather than silently believed.
        """
        lo, hi = term._chunks[ci]
        if term.floor == "clip":
            lp = tf.maximum(li, term.npdt(0.0))
        elif term.floor == "softplus":
            sc = term.npdt(term.floor_scale)
            lp = sc * tf.math.softplus(li / sc)
        else:
            lp = li
        if term.bkg_frac_param is not None:
            fb = values[term.bkg_frac_param] * term.npdt(term.bkg_frac_unit)
        elif term.bkg_frac:
            fb = tf.constant(term.bkg_frac, term.dtype)
        else:
            fb = None
        if fb is None:
            total = lp
        else:
            bkg = term.background.pdf(
                values, term.mobs[lo:hi] + term.npdt(term.m_ref))
            total = (tf.constant(1.0, term.dtype) - fb) * lp + fb * bkg
        logl = tf.math.log(total)
        lj = getattr(term, "_chunk_logjac", lambda *a: None)(values, ci)
        if lj is not None:
            logl = logl + lj
        if term.weights is not None:
            logl = term.weights[lo:hi] * logl
        return logl

    def sandwich(self, xf, check=True):
        """Robust covariance meat ``J = sum_i (w_i s_i)(w_i s_i)^T``.

        The inverse Hessian is the covariance only for an unweighted sample
        with a correctly specified model. Here neither holds: the MiNNLO
        weights alone take ``N_eff/N`` to ~0.68, i.e. every error up by 1.21.
        ``J`` is built with FORWARD-mode autodiff -- one
        ``ForwardAccumulator`` pass per parameter gives the whole chunk's
        column ``ds_i/dtheta_j`` at the memory of a single forward evaluation,
        against the ``(chunk, nparams)`` reverse-mode jacobian that would
        otherwise be materialised.
        """
        v = tf.constant(np.asarray(xf, np.float64), DTYPE)
        n = len(self.free)
        J = np.zeros((n, n))
        eye = np.eye(n)
        first = check
        for it, t, ci in self._pieces():
            cols = []
            for j in range(n):
                with tf.autodiff.ForwardAccumulator(
                        v, tf.constant(eye[j], DTYPE)) as acc:
                    x = self.full(v)
                    vals = t._values(tf.linalg.matvec(self._sel[it], x))
                    li = t._chunk_li(vals, ci)
                    if t._norm is not None:
                        z = t._norm_z(vals)
                        lo, hi = t._chunks[ci]
                        li = li / tf.gather(z, t._norm_class[lo:hi])
                    vec = self._chunk_logl_vec(t, vals, ci, li)
                d = acc.jvp(vec)
                cols.append(np.zeros(int(vec.shape[0])) if d is None else d.numpy())
            if first:
                ref = float(self._chunk_nll(t, t._values(
                    tf.linalg.matvec(self._sel[it], self.full(v))), ci).numpy())
                got = -float(np.sum(vec.numpy()))
                if not np.isclose(got, ref, rtol=1e-12, atol=1e-9):
                    raise RuntimeError(
                        f"the per-candidate log-likelihood vector does not "
                        f"reproduce the term's own _mix: {got} vs {ref}")
                first = False
            S = np.stack(cols, axis=1)          # (chunk, nparams)
            J += S.T @ S
        return J

    def _pieces(self):
        """(term, chunk index) for every piece of the objective."""
        for it, t in enumerate(self.terms):
            for ci in range(t.nchunk):
                yield it, t, ci

    # ------------------------------------------------------------------
    def value(self, xf):
        x = self.full(tf.constant(xf, DTYPE) if not tf.is_tensor(xf) else xf)
        tot = self._prior(x)
        for it, t, ci in self._pieces():
            vals = t._values(tf.linalg.matvec(self._sel[it], x))
            tot = tot + self._chunk_nll(t, vals, ci)
        return float(tot.numpy())

    def value_grad(self, xf):
        v = tf.constant(np.asarray(xf, np.float64), DTYPE)
        val = 0.0
        grad = np.zeros(len(self.free))
        with tf.GradientTape() as tape:
            tape.watch(v)
            f = self._prior(self.full(v))
        val += float(f.numpy())
        g = tape.gradient(f, v)
        if g is not None:
            grad += g.numpy()
        for it, t, ci in self._pieces():
            with tf.GradientTape() as tape:
                tape.watch(v)
                x = self.full(v)
                vals = t._values(tf.linalg.matvec(self._sel[it], x))
                f = self._chunk_nll(t, vals, ci)
            val += float(f.numpy())
            g = tape.gradient(f, v)
            grad += np.zeros(len(self.free)) if g is None else g.numpy()
        if not (np.isfinite(val) and np.all(np.isfinite(grad))):
            # a scale wandered far enough that the density (or the window
            # integral) went non-positive; steer back rather than crash
            self.nbad += 1
            return 1e30, np.where(np.isfinite(grad), grad, 0.0) + 1e6 * (
                np.asarray(xf) - self.x0[self.free])
        return val, grad

    # ------------------------------------------------------------------
    def _hess_piece_pfor(self, v, it, t, ci):
        with tf.GradientTape() as t2:
            t2.watch(v)
            with tf.GradientTape() as t1:
                t1.watch(v)
                x = self.full(v)
                vals = t._values(tf.linalg.matvec(self._sel[it], x))
                f = self._chunk_nll(t, vals, ci)
            g = t1.gradient(f, v)
        return t2.jacobian(g, v).numpy()

    def _hess_piece_hvp(self, v, it, t, ci):
        """Forward-over-reverse: nparams HVPs, each ~2x a gradient in memory."""
        n = int(v.shape[0])
        H = np.empty((n, n))
        eye = np.eye(n)
        for j in range(n):
            tangent = tf.constant(eye[j], DTYPE)
            with tf.autodiff.ForwardAccumulator(v, tangent) as acc:
                with tf.GradientTape() as tape:
                    tape.watch(v)
                    x = self.full(v)
                    vals = t._values(tf.linalg.matvec(self._sel[it], x))
                    f = self._chunk_nll(t, vals, ci)
                g = tape.gradient(f, v)
            hv = acc.jvp(g)
            H[:, j] = np.zeros(n) if hv is None else hv.numpy()
        return H

    def hessp(self, xf, p):
        """ONE Hessian-vector product ``H @ p``, accumulated over chunks.

        WHY THIS EXISTS.  `trust-exact` needs the FULL Hessian at every
        iteration, and over the joint fit's 99 free parameters that is 99
        columns of tape per chunk: measured, it OOMs an H200 at `chunk 32768`
        and costs ~10 000 s per Hessian there even when it fits, i.e. ~105 h for
        the 38 iterations the Z-alone fit took.  A Krylov trust-region method
        (`trust-krylov`, `trust-ncg`) never forms the Hessian: it asks for
        O(10) products of it with a vector, each ~2 gradients and INDEPENDENT of
        the parameter count.  That is the difference between a fit that runs and
        one that does not.

        Exactly the same arithmetic as one column of :meth:`_hess_piece_hvp`,
        with an arbitrary tangent instead of a unit vector, plus the analytic
        prior diagonal.
        """
        v = tf.constant(np.asarray(xf, np.float64), DTYPE)
        pv = np.asarray(p, np.float64)
        tangent = tf.constant(pv, DTYPE)
        out = (self._pmask.numpy() / self._psig.numpy() ** 2)[self.free] * pv
        for it, t, ci in self._pieces():
            with tf.autodiff.ForwardAccumulator(v, tangent) as acc:
                with tf.GradientTape() as tape:
                    tape.watch(v)
                    x = self.full(v)
                    vals = t._values(tf.linalg.matvec(self._sel[it], x))
                    f = self._chunk_nll(t, vals, ci)
                g = tape.gradient(f, v)
            hv = acc.jvp(g)
            if hv is not None:
                out = out + hv.numpy()
        if not np.all(np.isfinite(out)):
            return np.zeros_like(out)
        return out

    def hess(self, xf, mode=None):
        v = tf.constant(np.asarray(xf, np.float64), DTYPE)
        n = len(self.free)
        mode = mode or self.hess_mode
        piece = (self._hess_piece_pfor if mode == "pfor"
                 else self._hess_piece_hvp)
        # priors: exact, and cheap
        H = np.diag(
            (self._pmask.numpy() / self._psig.numpy() ** 2)[self.free])
        for it, t, ci in self._pieces():
            H = H + piece(v, it, t, ci)
        if not np.all(np.isfinite(H)):
            return np.eye(n) * 1e6
        return 0.5 * (H + H.T)

    # ------------------------------------------------------------------
    def timed(self, xf, label=""):
        """Value / grad / Hessian with wall time and peak RSS, for the report."""
        import resource
        out = {}
        t0 = time.time()
        out["nll"] = self.value(xf)
        out["t_nll"] = time.time() - t0
        t0 = time.time()
        _, g = self.value_grad(xf)
        out["t_grad"] = time.time() - t0
        out["gradmax"] = float(np.max(np.abs(g)))
        t0 = time.time()
        H = self.hess(xf)
        out["t_hess"] = time.time() - t0
        out["rss_gb"] = resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss / 1024.0 ** 2
        out["H"] = H
        if label:
            self.log(f"  [{label}] NLL {out['nll']:.6f}  "
                     f"t: nll {out['t_nll']:.2f}s grad {out['t_grad']:.2f}s "
                     f"hess {out['t_hess']:.2f}s  peak RSS {out['rss_gb']:.1f} GB "
                     f"({self.nchunk} chunks, {n_free(self)} free params)")
        return out


def n_free(obj):
    return len(obj.free)


def report_cov(names, H, n, projections, label, log=print):
    """Errors and correlations, plus the sqrt(N) projection to full statistics."""
    C = np.linalg.inv(H)
    err = np.sqrt(np.diag(C))
    corr = C / np.outer(err, err)
    log(f"\n  --- {label} (n = {n}) ---")
    w = max(len(s) for s in names) + 1
    log(f"    {'parameter':>{w}s} {'error':>12s}"
        + "".join(f" {f'N={p:.3g}':>12s}" for p in projections))
    for i, nm in enumerate(names):
        row = "".join(f" {err[i]*np.sqrt(n/p):12.4f}" for p in projections)
        log(f"    {nm:>{w}s} {err[i]:12.4f}{row}")
    log("    correlations:")
    log(f"    {'':>{w}s} " + " ".join(f"{s:>9s}" for s in names))
    for i, nm in enumerate(names):
        log(f"    {nm:>{w}s} " + " ".join(
            f"{corr[i, j]:9.4f}" for j in range(len(names))))
    return err, corr
