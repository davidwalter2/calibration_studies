#!/usr/bin/env python3
"""H and J of the vertex / mass / joint CF term, in the layout
`hitlik/efficiency.py` reads (`H_<arm>_<channel>`, `J_<arm>_<channel>`,
`G_<arm>_<channel>`, `params`).

`H` is the curvature the term CLAIMS; `J` is the score covariance, so the
SANDWICH `(H+P)^-1 J (H+P)^-1` is the variance the estimator actually has.
The two `hessian` / `score_cov` routines are imported from
`hitlik/fisher_cmp.py` -- the same code that produced the per-hit numbers.

For the JOINT channel the per-batch gradients of the two terms are SUMMED
batch by batch over the SAME candidates, so `J` carries the within-candidate
correlation between the vertex and the mass score.  That is the only way the
over-counting can show up (a `sandwich/quoted` above 1 on the joint that
neither term shows alone).

usage:
  ./run_tf.sh python3 fisher_vtx.py --vtx-npz v.npz --mass-npz m.npz \
      --groups .../materialGroups50.txt --channels vtx mass joint \
      --arms cf gauss gaussq --maxn 8000 -o runs/vtxres/fisherHJ.npz
"""
import argparse, os, sys, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.dirname(_HERE)
for _p in (_HERE, _RES, os.path.join(_RES, "matres"), os.path.join(_RES, "hitlik"),
           os.path.join(_RES, "globalfit")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fisher_cmp as FC        # noqa: E402  (hessian, score_cov)
import make_vtx_card as MVC    # noqa: E402
import groups as G             # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class _A:
    """The subset of make_vtx_card's argparse namespace `build_term` uses."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vtx-npz", default=None)
    p.add_argument("--mass-npz", default=None)
    p.add_argument("--groups", required=True)
    p.add_argument("--channels", nargs="+", default=["vtx", "mass", "joint"])
    p.add_argument("--arms", nargs="+", default=["cf", "gauss", "gaussq"])
    p.add_argument("--maxn", type=int, default=8000)
    p.add_argument("--max-chi2-ndof", type=float, default=3.0)
    p.add_argument("--prune-frac", type=float, default=1e-3)
    p.add_argument("--no-hits", action="store_true")
    p.add_argument("--nbatch", type=int, default=200)
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--no-hessian", action="store_true")
    p.add_argument("--max-abs-z", type=float, default=40.0)
    p.add_argument("-o", "--output", required=True)
    a = p.parse_args()

    import tensorflow as tf
    tf.config.optimizer.set_jit(False)

    gmap, _ = G.read_groups(a.groups)
    ngroups = (max(gmap) + 1) if gmap else 0
    gparams, gpriors = G.group_param_names(ngroups, a.groups)
    group_units = 1.0 / np.maximum(gpriors, 1e-300)   # whitened, as the cards
    import hitres_classes
    hparams = [f"hitres_{c}" for c in hitres_classes.CLASSES]

    ba = _A(max_chi2_ndof=a.max_chi2_ndof, maxn=a.maxn, prune_frac=a.prune_frac,
            amount_mode="exp", hit_mode="linear", no_hits=a.no_hits,
            floor="softplus", chunk=a.chunk, vtx_window=0.0,
            max_abs_z=a.max_abs_z)

    res, params = {}, None
    idx = None
    for ch in a.channels:
        need = ({"vtx": [("vtx", a.vtx_npz)], "mass": [("mass", a.mass_npz)],
                 "joint": [("vtx", a.vtx_npz), ("mass", a.mass_npz)]})[ch]
        if any(x[1] is None for x in need):
            log(f"skip channel {ch}: missing npz")
            continue
        for arm in a.arms:
            terms = []
            lidx = idx
            for nm, npz in need:
                t, _, lidx = MVC.build_term(nm, npz, arm, ba, group_units,
                                            gparams, hparams, ngroups, {}, {},
                                            idx=lidx)
                terms.append(t)
            if idx is None:
                idx = lidx
            names = terms[0].param_names
            for t in terms[1:]:
                if list(t.param_names) != list(names):
                    sys.exit("the two terms do not share a parameter vector")
            if params is None:
                params = list(names)
            npar = len(names)
            x0 = np.zeros(npar)
            # ---- H -------------------------------------------------------
            H = np.zeros((npar, npar))
            if not a.no_hessian:
                t0 = time.time()
                for t in terms:
                    Hi, v, g = FC.hessian(t, x0)
                    H += Hi
                log(f"{ch}/{arm}: H in {time.time()-t0:.0f} s, "
                    f"diag {np.diag(H).min():.3g} .. {np.diag(H).max():.3g}")
            # ---- J, batch means over the SAME candidate batches ----------
            Gsum = None
            gtot = np.zeros(npar)
            for t in terms:
                _, gt, M, Gm = FC.score_cov(t, x0, nbatch=a.nbatch)
                Gsum = Gm if Gsum is None else Gsum + Gm
                gtot = gtot + gt
            M = Gsum.shape[0]
            gb = Gsum.mean(axis=0)
            D = Gsum - gb
            J = (M / (M - 1.0)) * (D.T @ D)
            log(f"{ch}/{arm}: J from {M} batches, "
                f"|sum_m g_m - g| = {np.abs(Gsum.sum(axis=0)-gtot).max():.3g}")
            res[f"H_{arm}_{ch}"] = H
            res[f"J_{arm}_{ch}"] = J
            res[f"G_{arm}_{ch}"] = Gsum
            res[f"n_{ch}"] = terms[0].n
    res["params"] = np.array(params)
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    np.savez_compressed(a.output, **res)
    log(f"-> {a.output}")


if __name__ == "__main__":
    main()
