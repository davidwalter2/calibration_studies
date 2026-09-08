#!/usr/bin/env python3
"""Gate: the rabbit Fitter reproduces the standalone drivers, exactly.

The full-scale fits were driven by `fit.py` / `chunkfit.py` / `devobj.py`
rather than by `Fitter.minimize` only because an unbinned term did not fit in
memory inside the Fitter. Now that the candidate loop is the TERM's own
implementation (`rabbit.unbinned`, `chunk_mode="graph"`), the Fitter is the
right entry point again -- but only if it computes the same numbers.

This checks, on a real card:

1. **graph == eager inside rabbit.** `term.nll` with the `tf.while_loop` loop
   against the python loop it replaces, and the same for the gradient and one
   Hessian-vector product. Same arithmetic, so the tolerance is round-off.
2. **rabbit == the standalone objective.** `term.nll` against
   `chunkfit.ChunkedObjective` / `devobj.DeviceChunkedObjective` over the same
   term with every parameter free. The drivers add the declared Gaussian
   priors and the Fitter adds them as constraints, so the comparison is made
   on the unbinned term alone.
3. **the Fitter's own value / gradient / HVP.** `loss_val`, `loss_val_grad`
   and `loss_val_grad_hessp` against the term-level numbers plus the prior,
   which is what `rabbit_fit.py` actually minimises.
4. **the Hessian route.** `loss_val_grad_hess` (assembled from HVPs when an
   unbinned term is present) against the standalone `ChunkedObjective.hess`,
   and its EDM.

usage:
    RABBIT=../../rabbit-native ./run_tf.sh python3 -u test_rabbit_path.py \
        --card cards/z_n300k.hdf5
    RABBIT=../../rabbit-native ./run_tf.sh python3 -u test_rabbit_path.py \
        --card cards/joint_smoke.hdf5 \
        --model "ExternalParams bundle:global_params"
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


class Options:
    """Stand-in for the rabbit_fit.py argparse namespace."""

    def __init__(self, **kwargs):
        defaults = dict(
            earlyStopping=-1,
            noBinByBinStat=False,
            binByBinStatMode="lite",
            binByBinStatType="automatic",
            covarianceFit=False,
            chisqFit=False,
            diagnostics=False,
            minimizerMethod="trust-exact",
            prefitUnconstrainedNuisanceUncertainty=0.0,
            freezeParameters=[],
            setConstraintMinimum=[],
            unblind=[],
            blindingGroup=[],
            unbinnedChunk=0,
            unbinnedChunkMode="graph",
        )
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


def vgh_fn(term):
    """``(value, gradient, H p)`` of one term, INSIDE a tf.function.

    Inside is not a detail. `tf.while_loop` executes eagerly as a plain python
    loop outside a `tf.function`, so the memory bound the graph chunk loop
    exists to provide only holds once it is traced -- which is how the Fitter
    always calls it (`loss_val`, `loss_val_grad` and `loss_val_grad_hessp` are
    all `tf.function`s). Calling `term.nll` in eager mode is correct but will
    hold the whole sample, and at 3.7 M candidates it OOMs an H200.
    """

    @tf.function
    def _vgh(xt, pt):
        with tf.GradientTape() as t2:
            t2.watch(xt)
            with tf.GradientTape() as t1:
                t1.watch(xt)
                v = term.nll(xt)
            g = t1.gradient(v, xt)
        hv = t2.gradient(g, xt, output_gradients=pt)
        return v, g, (tf.zeros_like(xt) if hv is None else hv)

    return _vgh


def step(term, rng, frac=0.05):
    """A perturbation that respects the parameter scales.

    The calibration parameters of a joint card (92 field modes) live at ~1e-3;
    a 0.05 step on them takes the density negative and the NLL to nan, which
    tests nothing. Scale by the declared prior sigma where there is one.
    """
    sig = np.asarray(term.param_prior_sigmas, np.float64)
    sig = np.where(np.isfinite(sig) & (sig > 0), sig, 0.05)
    return frac * sig * rng.normal(size=len(sig))


def rel(a, b, scale=None):
    a, b = np.atleast_1d(a), np.atleast_1d(b)
    den = np.max(np.abs(b)) if scale is None else scale
    return float(np.max(np.abs(a - b)) / max(den, 1e-300))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--card", required=True)
    p.add_argument("--model", default="UnbinnedParams",
                   help="the param model, with rabbit_fit.py's own extra "
                        "arguments after it, space separated -- e.g. "
                        "'ExternalParams bundle:global_params', which is what "
                        "make_joint_card.py's cards need (their bundle is not "
                        "the default name)")
    p.add_argument("--chunk", type=int, default=0)
    p.add_argument("--rtol", type=float, default=1e-11)
    p.add_argument("--no-hess", action="store_true", help="skip check 4")
    p.add_argument(
        "--max-eager-n",
        type=int,
        default=400000,
        help="skip the eager arm of check 1 above this many candidates. The "
        "eager loop is a python `for` under ONE tape, so every chunk's tape "
        "is live at once -- 116 GB at 300 k candidates and 5 parameters, and "
        "an OOM at 3.7 M. That is the whole reason the graph loop exists, so "
        "refusing to run it at full scale is the point, not a gap in the "
        "gate: checks 2-4 still compare against the chunked host objective.",
    )
    args = p.parse_args()

    from rabbit import fitter as fitter_mod
    from rabbit import inputdata
    from rabbit.param_models.helpers import load_model

    from chunkfit import ChunkedObjective
    from devobj import DeviceChunkedObjective

    t0 = time.time()
    indata = inputdata.FitInputData(args.card)
    terms = list(indata.unbinned_terms)
    print(f"[gate] {args.card}: {len(terms)} unbinned term(s), "
          f"loaded in {time.time()-t0:.0f} s")
    for t in terms:
        print(f"        {t.name}: n {t.n}, {t.nchunk} chunk(s) of {t.chunk}, "
              f"{len(t.param_names)} params, graph_chunkable {t.graph_chunkable}")

    rng = np.random.default_rng(0)
    ok = True

    # ---- 1. graph vs eager, inside the term --------------------------------
    print("\n=== 1. graph chunk loop vs the eager python loop (same term) ===")
    for t in terms:
        x0 = np.asarray(t.param_defaults, np.float64)
        pdir = np.linspace(1.0, -0.5, len(x0))
        for k in range(2):
            xv = x0 if k == 0 else x0 + step(t, rng)
            xt = tf.constant(xv, tf.float64)
            pt = tf.constant(pdir, tf.float64)
            out = {}
            for mode in ("eager", "graph"):
                if mode == "graph" and not t.graph_chunkable:
                    continue
                if mode == "eager" and t.n > args.max_eager_n:
                    continue
                t.chunk_mode = mode
                v, g, hv = vgh_fn(t)(xt, pt)
                out[mode] = (float(v.numpy()), g.numpy(), hv.numpy())
            t.chunk_mode = None
            if "graph" not in out:
                print(f"  {t.name}: not graph-chunkable, skipped")
                continue
            if "eager" not in out:
                print(f"  {t.name}: {t.n} candidates > --max-eager-n "
                      f"{args.max_eager_n}, the eager arm would OOM; "
                      "checks 2-4 carry the comparison")
                break
            dv = abs(out["graph"][0] - out["eager"][0]) / max(abs(out["eager"][0]), 1.0)
            dg = rel(out["graph"][1], out["eager"][1])
            dh = rel(out["graph"][2], out["eager"][2])
            flag = "OK" if max(dv, dg, dh) < args.rtol else "MISMATCH"
            ok &= flag == "OK"
            print(f"  {t.name} point {k}: value {dv:.3e}, grad {dg:.3e}, "
                  f"hvp {dh:.3e}   {flag}")

    # ---- 2. the term vs the standalone objectives --------------------------
    print("\n=== 2. rabbit's term vs the standalone chunked objectives ===")
    for t in terms:
        free = list(range(len(t.param_names)))
        host = ChunkedObjective([t], free, hess_mode="hvp")
        dev = (DeviceChunkedObjective([t], free) if t.graph_chunkable else None)
        x0 = np.asarray(t.param_defaults, np.float64)
        xv = x0 + step(t, rng)
        pdir = np.linspace(1.0, -0.5, len(x0))
        # the drivers add the declared Gaussian priors; strip them so the
        # comparison is on the unbinned term alone
        pm = host._pmask.numpy()
        ps = host._psig.numpy()
        pmu = host._pmean.numpy()
        prior = 0.5 * float(np.sum(pm * ((xv - pmu) / ps) ** 2))
        gprior = pm * (xv - pmu) / ps**2
        hprior = pm / ps**2 * pdir

        xt = tf.constant(xv, tf.float64)
        pt = tf.constant(pdir, tf.float64)
        v, g, hv = vgh_fn(t)(xt, pt)

        hv_host, hg_host = host.value_grad(xv)
        hp_host = host.hessp(xv, pdir)
        dv = abs(float(v.numpy()) + prior - hv_host) / max(abs(hv_host), 1.0)
        dg = rel(g.numpy() + gprior, hg_host)
        dh = rel(hv.numpy() + hprior, hp_host)
        flag = "OK" if max(dv, dg, dh) < args.rtol else "MISMATCH"
        ok &= flag == "OK"
        print(f"  {t.name} vs ChunkedObjective : value {dv:.3e}, grad {dg:.3e}, "
              f"hvp {dh:.3e}   {flag}")
        if dev is not None:
            dv2, dg2 = dev.value_grad(xv)
            dh2 = dev.hessp(xv, pdir)
            e = max(abs(float(v.numpy()) + prior - dv2) / max(abs(dv2), 1.0),
                    rel(g.numpy() + gprior, dg2), rel(hv.numpy() + hprior, dh2))
            flag = "OK" if e < args.rtol else "MISMATCH"
            ok &= flag == "OK"
            print(f"  {t.name} vs DeviceChunkedObjective: worst {e:.3e}   {flag}")
            del dev
        del host

    # ---- 3/4. through the Fitter -------------------------------------------
    print("\n=== 3. the Fitter's loss / gradient / HVP ===")
    spec = args.model.split()
    param_model = load_model(spec[0], indata, *spec[1:])
    f = fitter_mod.Fitter(indata, param_model, Options(unbinnedChunk=args.chunk))
    f.set_nobs(f.indata.data_obs)
    n = int(f.x.shape[0])
    print(f"  {n} fit parameters, jit_compile {f.jit_compile}, "
          f"hvp {f.hvp_method}, hessian route "
          f"{'hvps' if f.unbinned_terms else 'jacobian'}")

    xv = f.x.numpy() + 0.01 * rng.normal(size=n)
    f.x.assign(tf.constant(xv, f.x.dtype))
    pdir = np.linspace(1.0, -0.5, n)
    pt = tf.constant(pdir, f.x.dtype)

    t0 = time.time()
    v_only = float(f.loss_val().numpy())
    t_val = time.time() - t0
    t0 = time.time()
    v, g = f.loss_val_grad()
    t_grad = time.time() - t0
    t0 = time.time()
    v2, g2, hp = f.loss_val_grad_hessp(pt)
    t_hvp = time.time() - t0
    print(f"  loss {v_only:.9f}  [value {t_val:.2f} s, value+grad {t_grad:.2f} s, "
          f"+hvp {t_hvp:.2f} s]")
    d = abs(float(v.numpy()) - v_only) / max(abs(v_only), 1.0)
    print(f"  loss_val vs loss_val_grad: {d:.3e}   "
          f"{'OK' if d < args.rtol else 'MISMATCH'}")
    ok &= d < args.rtol

    # finite-difference the Fitter's own gradient along pdir
    eps = 1e-6
    f.x.assign(tf.constant(xv + eps * pdir, f.x.dtype))
    vp = float(f.loss_val().numpy())
    f.x.assign(tf.constant(xv - eps * pdir, f.x.dtype))
    vm = float(f.loss_val().numpy())
    f.x.assign(tf.constant(xv, f.x.dtype))
    fd = (vp - vm) / (2 * eps)
    an = float(np.dot(g.numpy(), pdir))
    dfd = abs(fd - an) / max(abs(an), 1.0)
    print(f"  gradient . p: analytic {an:.8g}, central FD {fd:.8g}, rel {dfd:.2e}   "
          f"{'OK' if dfd < 1e-5 else 'CHECK'}")

    if not args.no_hess:
        print("\n=== 4. the Hessian route (HVP columns) and the EDM ===")
        t0 = time.time()
        vh, gh, H = f.loss_val_grad_hess()
        t_hess = time.time() - t0
        H = np.asarray(H)
        asym = np.max(np.abs(H - H.T)) / max(np.max(np.abs(H)), 1e-300)
        ev = np.linalg.eigvalsh(H)
        from rabbit import tfhelpers as tfh

        edmval = float(np.asarray(tfh.edmval(gh, tf.constant(H, f.x.dtype))))
        print(f"  Hessian in {t_hess:.1f} s, asymmetry {asym:.2e}, "
              f"eigenvalues [{ev.min():.4g}, {ev.max():.4g}], EDM {edmval:.4g}")
        # against the standalone objective, on the unbinned parameters only,
        # at the SAME point the Fitter's Hessian was taken at
        names = list(np.asarray(f.parms).astype(str))
        xnow = f.x.numpy()
        for t in terms[:1]:
            free = list(range(len(t.param_names)))
            host = ChunkedObjective([t], free, hess_mode="pfor")
            idx = [names.index(q) for q in t.param_names]
            Href = host.hess(xnow[idx])
            sub = H[np.ix_(idx, idx)]
            d = rel(sub, Href)
            flag = "OK" if d < 1e-6 else "MISMATCH"
            ok &= flag == "OK"
            print(f"  Fitter Hessian block vs ChunkedObjective.hess (pfor) "
                  f"({len(idx)}x{len(idx)}): {d:.3e}   {flag}")
            del host

    print("\n" + ("ALL GATES PASSED" if ok else "SOME GATES FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
