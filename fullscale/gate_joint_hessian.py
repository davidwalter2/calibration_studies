#!/usr/bin/env python3
"""GATE: the Fitter's Hessian route, on a card with MORE THAN ONE term.

`test_rabbit_path.py`'s check 4 compares the Fitter's Hessian BLOCK on one
unbinned term's parameters against that term's standalone
`ChunkedObjective.hess`. That is a correct test of a card whose only content
is that term -- `cards/z_full380_fl.hdf5`, which is what it was written for --
and it is STRUCTURALLY WRONG on a joint card: the block also carries the other
mass term's curvature (the two share 110 parameters) and the `hitchi2`
external quadratic's, neither of which is in the standalone objective. On the
phase-3 smoke card it reports a "mismatch" of 7.2e5, and the reason is visible
in its own output -- the Fitter Hessian's largest eigenvalue, 7.243e13, is to
four digits the largest eigenvalue of the external quadratic alone.

So the Hessian has to be checked WITHOUT assuming what is in it. This does it
two ways, neither needing a full Hessian:

1. `H p` from `loss_val_grad_hessp` against the CENTRAL DIFFERENCE of the
   Fitter's own gradient along `p`. That is the whole route -- every term, the
   external quadratic, the constraints -- against the definition, and it is
   two gradient evaluations rather than 117 HVPs.
2. the DECOMPOSITION: `H p` against the sum of the per-term HVPs (each from
   the term's own `nll`, on its own sub-vector) plus the external quadratic's
   `K p` plus the Gaussian constraints' `p / sigma^2`. This says not just that
   the Hessian is right but that it is made of the pieces it should be.

usage:
    python3 gate_joint_hessian.py --card cards/joint_mat_smoke.hdf5 \\
        --model "ExternalParams bundle:global_params" \\
        --quad runs/quad_jpsiv2_ok.npz runs/quad_dyv2.npz \\
        --groups .../materialGroups50.txt
"""
import argparse
import os
import sys

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.join(os.path.dirname(HERE), "resolution")
for _p in (HERE, _RES, os.path.join(_RES, "globalfit")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class Options:
    def __init__(self, **kw):
        d = dict(earlyStopping=-1, noBinByBinStat=False, binByBinStatMode="lite",
                 binByBinStatType="automatic", covarianceFit=False,
                 chisqFit=False, diagnostics=False,
                 minimizerMethod="trust-exact",
                 prefitUnconstrainedNuisanceUncertainty=0.0,
                 freezeParameters=[], setConstraintMinimum=[], unblind=[],
                 blindingGroup=[], unbinnedChunk=0, unbinnedChunkMode="graph")
        d.update(kw)
        for k, v in d.items():
            setattr(self, k, v)


def term_hvp(term, x_sub, p_sub):
    @tf.function
    def _h(xt, pt):
        with tf.GradientTape() as t2:
            t2.watch(xt)
            with tf.GradientTape() as t1:
                t1.watch(xt)
                v = term.nll(xt)
            g = t1.gradient(v, xt)
        hv = t2.gradient(g, xt, output_gradients=pt)
        return tf.zeros_like(xt) if hv is None else hv
    return _h(tf.constant(x_sub), tf.constant(p_sub)).numpy()


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--card", required=True)
    p.add_argument("--model", default="ExternalParams bundle:global_params")
    p.add_argument("--quad", nargs="*", default=[],
                   help="the same extraction(s) the card's external term was "
                        "built from; without them check 2 is skipped")
    p.add_argument("--groups", default=None)
    p.add_argument("--coeffs", default=None)
    p.add_argument("--eps", type=float, nargs="+", default=[1e-5, 1e-6, 1e-7])
    p.add_argument("--seed", type=int, default=3)
    a = p.parse_args()

    from rabbit import fitter as fitter_mod
    from rabbit import inputdata
    from rabbit.param_models.helpers import load_model

    indata = inputdata.FitInputData(a.card)
    terms = list(indata.unbinned_terms)
    spec = a.model.split()
    pm = load_model(spec[0], indata, *spec[1:])
    f = fitter_mod.Fitter(indata, pm, Options())
    f.set_nobs(f.indata.data_obs)
    names = list(np.asarray(f.parms).astype(str))
    n = len(names)
    print(f"[gate] {a.card}: {len(terms)} unbinned term(s), {n} fit parameters")

    rng = np.random.default_rng(a.seed)
    x0 = f.x.numpy()
    xv = x0 + 1e-3 * rng.normal(size=n)
    pdir = rng.normal(size=n)
    pdir /= np.linalg.norm(pdir)
    f.x.assign(tf.constant(xv, f.x.dtype))
    _, g0, hp = f.loss_val_grad_hessp(tf.constant(pdir, f.x.dtype))
    hp = hp.numpy()
    print(f"  |H p| = {np.abs(hp).max():.6e},  |g| = {np.abs(g0.numpy()).max():.6e}")

    # ---- 1. against the central difference of the Fitter's own gradient ----
    print("\n=== 1. H p vs the central difference of the Fitter's gradient ===")
    best = np.inf
    for eps in a.eps:
        f.x.assign(tf.constant(xv + eps * pdir, f.x.dtype))
        _, gp = f.loss_val_grad()
        f.x.assign(tf.constant(xv - eps * pdir, f.x.dtype))
        _, gm = f.loss_val_grad()
        fd = (gp.numpy() - gm.numpy()) / (2 * eps)
        rel = np.max(np.abs(hp - fd)) / max(np.abs(fd).max(), 1e-300)
        best = min(best, rel)
        print(f"  eps {eps:.0e}: max |H p - dg/de| / max|dg/de| = {rel:.3e}")
    f.x.assign(tf.constant(xv, f.x.dtype))
    ok1 = best < 1e-5
    print(f"  best over eps: {best:.3e}   {'OK' if ok1 else 'MISMATCH'}"
          "   (the floor is the FD's own truncation + round-off)")

    # ---- 2. the decomposition ---------------------------------------------
    print("\n=== 2. H p vs sum(term HVPs) + K p + p/sigma^2 ===")
    ok2 = True
    acc = np.zeros(n)
    for t in terms:
        idx = np.array([names.index(q) for q in t.param_names])
        acc[idx] += term_hvp(t, xv[idx], pdir[idx])
        print(f"  term '{t.name}': {len(idx)} parameters")
    if a.quad:
        import make_global_term as mgt
        import make_joint_card as mjc
        G, K, J, cat, _ = mjc.sum_quadratic(a.quad, lambda *x: None)
        gnames, _ = mgt.name_params(cat["parmtype"], cat["subidx"], a.groups,
                                    0.0, 1.0)
        ps, _ = mgt.param_scales(cat["parmtype"], cat["subidx"], a.coeffs,
                                 a.groups)
        inv = 1.0 / np.maximum(ps, 1e-300)
        Kw = 0.5 * (K * np.outer(inv, inv))     # chi2 -> NLL, as the card writes it
        j = np.array([names.index(q) for q in gnames])
        acc[j] += Kw @ pdir[j]
        print(f"  external 'hitchi2': {len(j)} parameters, "
              f"max|K/2| {np.abs(Kw).max():.4e}")
    else:
        print("  external 'hitchi2': SKIPPED (--quad not given)")
        ok2 = None
    sig = np.asarray(pm.prior_sigmas, np.float64)
    m = np.isfinite(sig) & (sig > 0)
    acc[m] += pdir[m] / sig[m] ** 2
    print(f"  constraints: {int(m.sum())} parameter(s) with a Gaussian prior")
    if ok2 is not None:
        rel = np.max(np.abs(hp - acc)) / max(np.abs(hp).max(), 1e-300)
        j = int(np.argmax(np.abs(hp - acc)))
        ok2 = rel < 1e-9
        print(f"  max |H p - (sum of the pieces)| / max|H p| = {rel:.3e}   "
              f"{'OK' if ok2 else 'MISMATCH'}")
        print(f"      worst component: {names[j]}  {hp[j]:.8g} vs {acc[j]:.8g}")

    good = ok1 and (ok2 is not False)
    print("\n" + ("GATE PASSED" if good else "GATE FAILED"))
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
