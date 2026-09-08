#!/usr/bin/env python3
"""GATE: the dense `D` representation is the sparse one, exactly.

`tf.sparse.sparse_dense_matmul` has no deterministic GPU kernel, so a card
carrying a per-candidate sparse `D` dies at the first loss evaluation under
`rabbit_fit.py`'s `enable_op_determinism()`. `JacChunkTable` now stores `D`
dense when it is denser than ~1/3 (the joint cards are 79-82 % dense, where
COO is the LARGER representation as well as the undifferentiable-on-GPU one).

A change of representation must be a change of nothing else. This builds ONE
`MassCFTerm` twice from the same arrays, `dense=False` and `dense=True`, and
compares the value, the gradient and a Hessian-vector product on both the
eager and the graph chunk loop.

    python3 gate_jacdense.py [--n 20000] [--njac 20]
"""
import argparse

import numpy as np
import tensorflow as tf


def load(card, dense):
    """Every unbinned term of `card`, with `D` stored dense or sparse."""
    import h5py
    try:
        import hdf5plugin  # noqa: F401
    except ImportError:
        pass
    from rabbit import unbinned

    orig = unbinned.JacChunkTable.__init__

    def patched(self, blocks, chunks, njac, dense=None):
        orig(self, blocks, chunks, njac, dense=_FORCE[0])

    _FORCE = [dense]
    unbinned.JacChunkTable.__init__ = patched
    try:
        with h5py.File(card, "r") as f:
            terms = unbinned.read_unbinned_terms_from_h5(f["unbinned_terms"])
    finally:
        unbinned.JacChunkTable.__init__ = orig
    return terms


def probe(term, x, v, mode):
    term.chunk_mode = mode
    names = list(term.param_names)
    # `term.nll` indexes its argument POSITIONALLY, in `param_names` order,
    # and its `tf.custom_gradient` VJP returns ONE tensor -- so the argument
    # has to be one tensor, not a list of scalars
    xv = tf.Variable([float(x.get(p, 0.0)) for p in names], dtype=tf.float64)
    vv = tf.constant([float(v.get(p, 0.0)) for p in names], dtype=tf.float64)

    @tf.function
    def loss():
        return term.nll(xv)

    @tf.function
    def grad():
        with tf.GradientTape() as t:
            L = term.nll(xv)
        return L, t.gradient(L, xv)

    @tf.function
    def hvp():
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                L = term.nll(xv)
            g = t1.gradient(L, xv)
            gv = tf.reduce_sum(g * vv)
        return t2.gradient(gv, xv)

    L, g = grad()
    return float(loss()), np.asarray(g), np.asarray(hvp())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default="cards/joint_smoke.hdf5")
    a = ap.parse_args()

    ts = load(a.card, False)
    td = load(a.card, True)
    rng = np.random.default_rng(1)
    ok = True
    for t_s, t_d in zip(ts, td):
        if getattr(t_s, "_jac_chunks", None) is None:
            print(f"== {t_s.name}: no D, skipped")
            continue
        print(f"== {t_s.name}: D {t_s._jac_chunks.density:.1%} dense over "
              f"{t_s.n} x {t_s._jac_chunks.njac}; "
              f"is_dense {t_s._jac_chunks.is_dense} vs {t_d._jac_chunks.is_dense}")
        assert not t_s._jac_chunks.is_dense and t_d._jac_chunks.is_dense
        names = list(t_s.param_names)
        # AT the term's OWN defaults. Not at zero: the resolution knobs
        # default to 1 and a zero `k_hit` divides by zero, and the 92
        # calibration parameters are whitened card units where O(0.05) is
        # already outside the region the corrections are linearised on. The
        # gate is about the REPRESENTATION of D, so it needs a point where
        # both paths are finite and a direction that moves them.
        x = dict(zip(names, np.asarray(t_s.param_defaults, dtype=float)))
        v = {p: float(rng.normal(0.0, 1e-3)) for p in names}
        for mode in ("eager", "graph"):
            Ls, gs, hs = probe(t_s, x, v, mode)
            Ld, gd, hd = probe(t_d, x, v, mode)
            dL = abs(Ld - Ls) / max(abs(Ls), 1e-300)
            dg = np.max(np.abs(gd - gs)) / max(np.max(np.abs(gs)), 1e-300)
            dh = np.max(np.abs(hd - hs)) / max(np.max(np.abs(hs)), 1e-300)
            print(f"   {mode:6s} loop:  NLL {Ls:.9f}  value {dL:.3e}"
                  f"   gradient {dg:.3e}   HVP {dh:.3e}")
            ok &= np.isfinite(Ls) and dL < 1e-13 and dg < 1e-10 and dh < 1e-10
    print("\nGATE " + ("PASSES" if ok else "FAILS"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
