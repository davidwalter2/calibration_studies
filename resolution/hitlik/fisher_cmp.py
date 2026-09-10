#!/usr/bin/env python3
"""Fisher information of the residual-vector likelihood: CF vs Gaussian.

For each ARM (``cf`` / ``gauss`` / ``gaussq``, see ``hitlik_term``) and each
COMPONENT subset, build the term over the same tracks and compute the observed
information at the MC truth ``theta = 0``,

    H_pq = d^2 (-log L) / d theta_p d theta_q ,

by ``nparams`` reverse-over-reverse Hessian-vector products (the same tape path
rabbit's fitter uses, so the chunked graph loop is honoured and the memory
stays bounded to one chunk).

Reported per parameter:

    sigma_marg  = sqrt( (H^-1)_pp )      -- with every other parameter free
    sigma_alone = 1 / sqrt( H_pp )       -- with every other parameter fixed

and the INFORMATION RATIO between two arms, ``(sigma_B / sigma_A)^2``.

Also reported, because the answer depends on them:

* ``Var(z_k)`` of the DATA per component -- the pull variance the arms are
  being asked to describe;
* the model variance of each arm per component, ``sum_g kappa2 + vgf``, which
  is 1 by construction for ``gaussq`` (it IS the fit's own error model) and
  larger for ``cf``/``gauss`` by the Rossi-vs-Moliere gap.

usage:
    fisher_cmp.py --npz runs/hitlik/mugun20k.npz --max-tracks 20000 \
        --arms cf gauss gaussq --comps 0 01234 -o runs/hitlik/fisher.npz
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import hitlik_term as HT  # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", required=True)
    p.add_argument("--max-tracks", type=int, default=0)
    p.add_argument("--max-chi2-ndof", type=float, default=0.0)
    p.add_argument("--max-inflat", type=float, default=1e4,
                   help="drop tracks whose Cholesky variance inflation "
                        "V_kk/d_k exceeds this in any used component -- a "
                        "guard on the FIT COVARIANCE, not on the residual")
    p.add_argument("--arms", nargs="+", default=list(HT.ARMS))
    p.add_argument("--comps", nargs="+", default=["0", "01234"],
                   help="component subsets, each a string of digits")
    p.add_argument("--prune-frac", type=float, default=0.001)
    p.add_argument("--no-hits", action="store_true")
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--upsample", type=int, default=1)
    p.add_argument("--no-hessian", action="store_true",
                   help="skip the observed Hessian (nparams HVPs); the score "
                        "covariance is the estimator the comparison uses")
    p.add_argument("--nbatch", type=int, default=400,
                   help="batches for the score-covariance Fisher estimator")
    p.add_argument("-o", "--output", required=True)
    return p.parse_args()


def hessian(term, x0):
    """H at x0 by nparams reverse-over-reverse HVPs."""
    import tensorflow as tf
    npar = len(term.param_names)
    H = np.zeros((npar, npar))
    x = tf.Variable(np.asarray(x0, np.float64), dtype=tf.float64)
    eye = np.eye(npar)
    for j in range(npar):
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                v = term.nll(x)
            g = t1.gradient(v, x)
        col = t2.gradient(g, x, output_gradients=tf.constant(eye[j]))
        H[j] = np.asarray(col.numpy())
    return 0.5 * (H + H.T), float(v.numpy()), np.asarray(g.numpy())


def score_cov(term, x0, nbatch=400):
    """The EXPECTED Fisher information, as the score covariance.

    The observed Hessian at MC truth is INDEFINITE here (theta = 0 is not the
    minimum: the model differs from the fit's own error at the few-per-cent
    level), so `(H^-1)_pp` is not an error and the arms cannot be compared
    with it.  The score covariance is PSD by construction and is the Fisher
    information at the true parameter (and the sandwich "meat" otherwise),
    which is exactly "how much does this data constrain this parameter".

    Estimated by BATCH MEANS: with `g_m` the gradient of batch `m` of `n`
    rows, `Cov(g_m) = n Sigma_s` and the whole-sample information is
    `I = M n Sigma_s = M x (sample covariance of the g_m)`.  One traced
    `_chunk_contribution` graph serves every batch, so the total work is ONE
    gradient pass over the sample.
    """
    import tensorflow as tf
    npar = len(term.param_names)
    keep = term.chunk
    n = int(np.ceil(term.n / nbatch))
    term.rechunk(max(1, n))
    M = term.nchunk

    @tf.function(reduce_retracing=True)
    def gchunk(p, ci):
        with tf.GradientTape() as t:
            t.watch(p)
            f = term._chunk_contribution(p, ci)
        g = t.gradient(f, p)
        return tf.zeros_like(p) if g is None else g

    x = tf.constant(np.asarray(x0, np.float64))
    G = np.zeros((M, npar))
    for m in range(M):
        G[m] = np.asarray(gchunk(x, tf.constant(m, tf.int32)).numpy())
    term.rechunk(keep)
    gb = G.mean(axis=0)
    D = G - gb
    return (M / (M - 1.0)) * (D.T @ D), G.sum(axis=0), M


def model_variance(sel, arm):
    """sum_g kappa2_g + vgf per row, for the arm's own families."""
    tg = sel["tgrid"]
    n = len(sel["z"])
    seg = sel["grp_seg"]
    tot = np.zeros(n)
    for f in HT.FAMS:
        if f.endswith("_im"):
            continue
        if arm == "cf" or arm == "gauss":
            k2 = HT.kappa2_from_grid(sel["S" + f].astype(np.float64), tg)
        elif arm == "gaussq":
            k2 = (sel["vQms"].astype(np.float64) if f == "ms"
                  else sel["vQio"].astype(np.float64) if f == "io_re"
                  else np.zeros(len(seg)))
        np.add.at(tot, seg, k2)
    return tot + sel["vgf"]


def main():
    args = parse_args()
    import tensorflow as tf
    tf.config.optimizer.set_jit(False)

    res = {}
    meta_all = {}
    sel_cache = {}
    for cs in args.comps:
        comps = [int(c) for c in cs]
        sel = HT.load(args.npz, max_tracks=args.max_tracks, comps=comps,
                      max_chi2_ndof=args.max_chi2_ndof, max_inflat=args.max_inflat)
        sel_cache[cs] = sel
        log(f"comps {cs}: {sel['ntrk']} tracks x {len(comps)} components = "
            f"{len(sel['z'])} rows, {len(sel['grp_id'])} group rows")
        # data pull variance per component
        for k in comps:
            m = sel["comp"] == k
            z = sel["z"][m]
            res[f"varz_c{k}"] = float(np.var(z))
            res[f"meanz_c{k}"] = float(np.mean(z))
            res[f"n_c{k}"] = int(m.sum())
        for arm in args.arms:
            mv = model_variance(sel, arm)
            res[f"modelvar_{arm}_{cs}"] = float(np.mean(mv))
            term, data, meta = HT.build(
                sel, arm=arm, prune_frac=args.prune_frac,
                groups_file=sel["groups_file"], chunk=args.chunk,
                no_hits=args.no_hits)
            if args.upsample > 1:
                pass  # upsample is a constructor arg; kept 1 for the Hessian
            npar = len(term.param_names)
            t0 = time.time()
            if args.no_hessian:
                import tensorflow as tf
                x = tf.Variable(np.zeros(npar), dtype=tf.float64)
                with tf.GradientTape() as tp:
                    vv = term.nll(x)
                g = np.asarray(tp.gradient(vv, x).numpy())
                v = float(vv.numpy())
                H = np.zeros((npar, npar))
                log(f"  arm {arm}: NLL(0) = {v:.6f}, max|grad| = "
                    f"{np.abs(g).max():.4g} (Hessian skipped)")
            else:
                H, v, g = hessian(term, np.zeros(npar))
                log(f"  arm {arm}: NLL(0) = {v:.6f}, max|grad| = "
                    f"{np.abs(g).max():.4g}, H in {time.time()-t0:.0f} s "
                    f"({npar} params)")
            t0 = time.time()
            J, gtot, M = score_cov(term, np.zeros(npar), args.nbatch)
            wj = np.linalg.eigvalsh(J)
            log(f"     score covariance ({M} batches) in {time.time()-t0:.0f} "
                f"s: eig min {wj.min():.3e} max {wj.max():.3e}, "
                f"|grad check| {np.abs(gtot - g).max():.3e}")
            res[f"J_{arm}_{cs}"] = J
            res[f"H_{arm}_{cs}"] = H
            res[f"nll0_{arm}_{cs}"] = v
            res[f"grad0_{arm}_{cs}"] = g
            res[f"modelvar_rows_{arm}_{cs}"] = np.float32(mv.mean())
            meta_all[f"{arm}_{cs}"] = dict(
                params=list(term.param_names), n=int(meta["n"]),
                ntrk=int(sel["ntrk"]), ndrop=int(meta["ndrop"]),
                nrows=int(meta["nrows"]))
            del term
    res["meta"] = json.dumps(meta_all)
    res["params"] = np.array(meta_all[list(meta_all)[0]]["params"], dtype=object)
    res["argv"] = json.dumps(vars(args))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".",
                exist_ok=True)
    np.savez_compressed(args.output, **res)
    log(f"wrote {args.output}")


if __name__ == "__main__":
    main()
