#!/usr/bin/env python3
"""H and J of the J/psi-gun MASS term, added to the per-hit residual term's.

WHY.  The working group asked whether the mass term and the hit residuals are
correlated and how that is accounted for.  Two different questions hide in
that, and they need two different objects:

* SAME tracks.  The truth-referenced q/p pull -- the variable the mass
  functional is built from -- and the per-hit innovations live on the same
  tracks.  They are EXACTLY uncorrelated (`F^T R = 0`, gate 4), but they are
  not independent: the composite likelihood that multiplies their marginals
  drops their joint cumulants.  That is measured by the sandwich on the
  `all` component set (both on the same tracks) -- `fisher_cmp.py --comps all`
  plus `efficiency.py --cset all`.  A `sandwich/quoted` above 1 on the joint
  that neither arm shows alone is the signature of over-counting.

* DISJOINT samples.  hitlik's `joint` is the residual term plus the J/psi-gun
  MASS term, and those are different events.  Then the score covariance is
  additive exactly, `J = J_res + J_mass` and `H = H_res + H_mass`, and there
  is nothing to over-count.  This script forms that sum, so `efficiency.py`
  can be run on the joint the same way as on either arm alone.

The mass term is built by `make_hitlik_card.build_mass_term`, i.e. the same
object the `ph_resmass` card carries, with the SAME material parameters and
the same card units, and its H and J are computed by the SAME two functions
`fisher_cmp.py` uses.  Parameters are matched BY NAME into the residual term's
60-parameter space; the hit classes get zero rows from the mass term (the
two-track maker exports no parmtype-8/9 blocks, so the mass term does not see
them), which is why the joint must be inverted with the priors -- as
`efficiency.py` does.

usage:
  fisher_joint.py --fisher runs/perhit/fisherHJ.npz --cset hit \
      --mass-npz runs/matres/gun_groups_probe.npz -o runs/perhit/fisherHJ_joint.npz
  efficiency.py --fisher runs/perhit/fisherHJ_joint.npz --cset hitmass \
      --arms cf gaussq --ref cf
"""

import argparse
import json
import os
import sys
import time
import types

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_HL = os.path.dirname(_HERE)
for _p in (_HERE, _HL, os.path.dirname(_HL), os.path.join(os.path.dirname(_HL), "matres")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fisher_cmp as FC          # noqa: E402  (hessian, score_cov)
import make_hitlik_card as MK    # noqa: E402  (build_mass_term)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fisher", required=True,
                   help="the residual term's fisher_cmp.py output")
    p.add_argument("--cset", default="hit",
                   help="component set of the residual arms to add to")
    p.add_argument("--arms", nargs="+", default=["cf", "gaussq"])
    p.add_argument("--mass-npz", default="/work/submit/david_w/ZMass/"
                   "calibration_studies/resolution/runs/matres/"
                   "gun_groups_probe.npz")
    p.add_argument("--mass-max-chi2-ndof", type=float, default=3.0)
    p.add_argument("--mass-max-cands", type=int, default=0)
    p.add_argument("--groups", default="/work/submit/david_w/ZMass/"
                   "CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/data/"
                   "materialGroups50.txt")
    p.add_argument("--prune-frac", type=float, default=0.001)
    p.add_argument("--amount-mode", default="exp")
    p.add_argument("--hit-mode", default="linear")
    p.add_argument("--floor", type=float, default=1e-12)
    p.add_argument("--chunk", type=int, default=8192)
    p.add_argument("--with-alpha", action="store_true")
    p.add_argument("--nbatch", type=int, default=200)
    p.add_argument("--out-cset", default="hitmass")
    p.add_argument("--mass-cache", default=None,
                   help="npz holding the mass term's own H, J and parameter "
                        "names.  Written if absent, read if present -- the "
                        "mass Hessian is 42 HVPs over 24 k candidates and is "
                        "independent of the residual arms, so it is computed "
                        "once and reused.")
    p.add_argument("--mass-only", action="store_true",
                   help="build the mass cache and stop")
    p.add_argument("-o", "--output", required=True)
    a = p.parse_args()

    d = np.load(a.fisher, allow_pickle=True)
    params = [str(s) for s in d["params"]]
    npar = len(params)
    idx_of = {q: i for i, q in enumerate(params)}
    log(f"residual fisher: {npar} parameters, cset '{a.cset}'")

    # the card units: exactly make_hitlik_card's, so the mass term floats the
    # same variable as the residual term.
    import groups as G
    ngroups = sum(1 for q in params if q.startswith("material_"))
    gparams_grp, gpriors_grp = G.group_param_names(ngroups, a.groups)
    group_units = 1.0 / np.maximum(gpriors_grp, 1e-300)

    if a.mass_cache and os.path.exists(a.mass_cache) and not a.mass_only:
        mc = np.load(a.mass_cache, allow_pickle=True)
        Hm, Jm = np.asarray(mc["H"], float), np.asarray(mc["J"], float)
        mnames = [str(x) for x in mc["names"]]
        ncand = int(mc["ncand"])
        log(f"mass H/J from cache {a.mass_cache}: {len(mnames)} params, "
            f"{ncand} candidates")
        return _finish(a, d, params, npar, idx_of, Hm, Jm, mnames, ncand)

    margs = types.SimpleNamespace(
        mass_npz=a.mass_npz, mass_max_chi2_ndof=a.mass_max_chi2_ndof,
        mass_max_cands=a.mass_max_cands, prune_frac=a.prune_frac,
        amount_mode=a.amount_mode, hit_mode=a.hit_mode, floor=a.floor,
        chunk=a.chunk, with_alpha=a.with_alpha)
    mterm, _ = MK.build_mass_term(margs, a.groups, ngroups, group_units,
                                  gparams_grp, None, log)
    mnames = list(mterm.param_names)
    nm = len(mnames)

    t0 = time.time()
    Hm, v0, g0 = FC.hessian(mterm, np.zeros(nm))
    log(f"mass H in {time.time()-t0:.0f} s: NLL(0) = {v0:.6f}, "
        f"max|grad| = {np.abs(g0).max():.4g}")
    t0 = time.time()
    Jm, gtot, M, Gm = FC.score_cov(mterm, np.zeros(nm), a.nbatch)
    log(f"mass J ({M} batches) in {time.time()-t0:.0f} s: "
        f"|grad check| {np.abs(gtot - g0).max():.3e}")

    if a.mass_cache:
        np.savez_compressed(a.mass_cache, H=Hm, J=Jm,
                            names=np.array(mnames, dtype=object),
                            ncand=int(mterm.n))
        log(f"wrote mass cache {a.mass_cache}")
    if a.mass_only:
        return
    return _finish(a, d, params, npar, idx_of, Hm, Jm, mnames, int(mterm.n))


def _finish(a, d, params, npar, idx_of, Hm, Jm, mnames, ncand):
    # embed by NAME
    col = np.array([idx_of.get(q, -1) for q in mnames])
    miss = [q for q, c in zip(mnames, col) if c < 0]
    if miss:
        log(f"  NOT in the residual parameter set (dropped): {miss}")
    keep = col >= 0
    ci = col[keep]
    HM = np.zeros((npar, npar))
    JM = np.zeros((npar, npar))
    HM[np.ix_(ci, ci)] = Hm[np.ix_(np.where(keep)[0], np.where(keep)[0])]
    JM[np.ix_(ci, ci)] = Jm[np.ix_(np.where(keep)[0], np.where(keep)[0])]

    res = {"params": np.array(params, dtype=object)}
    meta = {}
    for arm in a.arms:
        H = np.asarray(d[f"H_{arm}_{a.cset}"], float)
        J = np.asarray(d[f"J_{arm}_{a.cset}"], float)
        res[f"H_{arm}_{a.out_cset}"] = H + HM
        res[f"J_{arm}_{a.out_cset}"] = J + JM
        res[f"H_{arm}_mass"] = HM
        res[f"J_{arm}_mass"] = JM
        res[f"H_{arm}_res"] = H
        res[f"J_{arm}_res"] = J
        meta[f"{arm}_{a.out_cset}"] = dict(params=params, ncand=ncand)
    res["meta"] = json.dumps(meta)
    res["argv"] = json.dumps(vars(a))
    np.savez_compressed(a.output, **res)
    log(f"wrote {a.output}  (csets: {a.out_cset}, mass, res)")


if __name__ == "__main__":
    main()
