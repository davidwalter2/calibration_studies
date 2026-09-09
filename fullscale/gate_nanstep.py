#!/usr/bin/env python3
"""Where is the NaN born? Localise a `trust-exact` failure to a point, a term,
a class and a candidate.

Two fits died with `Minimizer raised: array must not contain infs or NaNs`.
That string is `scipy.linalg.norm(self.hess, np.inf)` inside
`IterativeSubproblem.__init__` (scipy checks finiteness there), i.e. scipy
never even got to the factorisation: the Hessian handed to it already had a
NaN.  Since the subproblem is CONSTRUCTED at every trial point -- before the
reduction-ratio test that would have rejected it -- one bad trial point is a
hard abort, and `cb.xval` (the snapshot) is still the last ACCEPTED point,
which is why the snapshot looks like the minimiser never moved.

What this does, on the real card and the real Fitter:

1. evaluate value / gradient / Hessian at the START point and say whether each
   is finite (this alone separates `P2X`, whose first printed diagnostic is
   already `nan`, from `SVetaEslo`, whose first is fine);
2. per TERM and per PIECE at that point: each unbinned term's own `nll`, the
   external terms, the constraints -- so a NaN at the start point is
   attributed without guessing;
3. per CANDIDATE and per CLASS inside a term: the truncation normalisation
   `Z_c` and the per-candidate density `L_i` before the log, which are the
   only two places `log` can be handed a non-positive number;
4. run scipy `trust-exact` itself, with the Fitter's own callbacks and NO
   preconditioning (rabbit's default), catch the abort and keep the point it
   aborted at; then BISECT `x0 -> x_bad` for the first parameter vector at
   which the density goes non-positive, and print the step in the parameter
   basis so the responsible direction is named rather than assumed.

usage:
    RABBIT=../../rabbit-vmass ./run_tf.sh python3 -u gate_nanstep.py \
        --card cards/z_V_etaE_slo.hdf5 --freeze k_hit k_ms k_ioni k_rad
    ... --card cards/joint_ok_full.hdf5 --model "ExternalParams bundle:global_params"
"""
import argparse
import os
import sys
import time

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


def fin(a):
    a = np.asarray(a)
    n = int(np.sum(~np.isfinite(a)))
    return f"finite" if n == 0 else f"NON-FINITE ({n} of {a.size})"


# ---------------------------------------------------------------- per term ---
def term_report(term, xfull, names, prefix="   "):
    """`nll`, the class normalisation `Z_c` and the per-candidate density.

    Eager and WITHOUT a tape: the point is to look at the intermediate
    quantities, not to differentiate them, so the whole sample can be walked
    one chunk at a time at no memory cost.
    """
    idx = [names.index(q) for q in term.param_names]
    p = tf.constant(xfull[idx], tf.float64)
    values = term._values(p)
    out = {}
    z = None
    if getattr(term, "_norm", None) is not None:
        z = np.asarray(term._norm_z(values))
        out["z_min"] = float(np.nanmin(z))
        out["z_nonpos"] = int(np.sum(~(z > 0)))
        out["z_nan"] = int(np.sum(~np.isfinite(z)))
    li_min = np.inf
    n_nonpos = 0
    n_nan = 0
    bad = []
    tot = 0.0
    zc = None if z is None else tf.constant(z, tf.float64)
    for ci in range(term.nchunk):
        lo, hi = term._chunks[ci]
        li = np.asarray(term._chunk_li(values, ci))
        if zc is not None:
            li = li / np.asarray(tf.gather(zc, term._norm_class[lo:hi]))
        li_min = min(li_min, float(np.nanmin(li)))
        m = ~(li > 0)
        n_nonpos += int(np.sum(m))
        n_nan += int(np.sum(~np.isfinite(li)))
        if np.any(m) and len(bad) < 20:
            for j in np.nonzero(m)[0][: 20 - len(bad)]:
                bad.append((lo + int(j), float(li[j])))
        with np.errstate(divide="ignore", invalid="ignore"):
            tot += float(np.nansum(np.log(np.where(li > 0, li, np.nan))))
    out.update(li_min=li_min, n_nonpos=n_nonpos, n_nan=n_nan, bad=bad)
    out["nll_eager"] = -tot
    return out


def describe_bad(term, bad):
    """What is special about the candidates whose density went non-positive."""
    if not bad:
        return
    i = np.array([b[0] for b in bad])
    sig = np.asarray(term.sigma)[i]
    mob = np.asarray(term.mobs)[i] + float(term.m_ref)
    cls = (np.asarray(term._norm_class)[i]
           if getattr(term, "_norm_class", None) is not None else np.full(len(i), -1))
    g = (np.asarray(term._fl_g)[i] if getattr(term, "_fl_g", None) is not None
         else np.full(len(i), np.nan))
    a = (np.asarray(term._fl_a)[i] if getattr(term, "_fl_a", None) is not None
         else np.full(len(i), np.nan))
    print(f"      {'i':>9s} {'m':>10s} {'sigma':>10s} {'class':>6s} "
          f"{'a_i':>10s} {'g_i':>10s} {'L_i':>12s}")
    for k in range(len(i)):
        print(f"      {i[k]:9d} {mob[k]:10.4f} {sig[k]:10.5f} {cls[k]:6d} "
              f"{a[k]:10.4g} {g[k]:10.4g} {bad[k][1]:12.4g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True)
    ap.add_argument("--model", default="UnbinnedParams")
    ap.add_argument("--freeze", nargs="*", default=["k_hit", "k_ms", "k_ioni", "k_rad"])
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--maxiter", type=int, default=3,
                    help="stop the real minimiser after this many ACCEPTED steps")
    ap.add_argument("--no-min", action="store_true", help="stop after the start point")
    ap.add_argument("--quad", nargs="*", default=None)
    ap.add_argument("--groups", default=None)
    a = ap.parse_args()

    from rabbit import fitter as fitter_mod
    from rabbit import inputdata, tfhelpers as tfh
    from rabbit.param_models.helpers import load_model

    t0 = time.time()
    indata = inputdata.FitInputData(a.card)
    terms = list(indata.unbinned_terms)
    print(f"[nanstep] {a.card}: {len(terms)} unbinned term(s), "
          f"{len(getattr(indata, 'external_terms', []) or [])} external, "
          f"loaded in {time.time()-t0:.0f} s")
    spec = a.model.split()
    pm = load_model(spec[0], indata, *spec[1:])
    f = fitter_mod.Fitter(indata, pm, Options(unbinnedChunk=a.chunk,
                                              freezeParameters=list(a.freeze)))
    f.set_nobs(f.indata.data_obs)
    names = list(np.asarray(f.parms).astype(str))
    free = [i for i in range(len(names)) if i in set(np.asarray(f.floating_indices))]
    x0 = f.x.numpy().copy()
    print(f"  {len(names)} parameters, {len(free)} floating, "
          f"frozen {sorted(set(names[i] for i in range(len(names)) if i not in free))}")

    # ---- 1. the start point -------------------------------------------------
    def at(x, label, per_term=True):
        f.x.assign(tf.constant(x, f.x.dtype))
        v, g, H = f.loss_val_grad_hess()
        v = float(v.numpy()); g = np.asarray(g); H = np.asarray(H)
        print(f"\n--- {label}")
        print(f"  loss {v!r}  grad {fin(g)}  hess {fin(H)}")
        if np.all(np.isfinite(H)) and np.all(np.isfinite(g)):
            sg, sh = f._floating_block(tf.constant(g), tf.constant(H))
            print(f"  cond {float(np.asarray(tfh.cond_number(sh))):.6g}  "
                  f"edm {float(np.asarray(tfh.edmval(sg, sh))):.6g}")
        if per_term:
            for t in terms:
                r = term_report(t, x, names)
                print(f"  term '{t.name}': nll {r['nll_eager']:.6f}  "
                      f"min L_i {r['li_min']:.6g}  non-positive {r['n_nonpos']}  "
                      f"nan {r['n_nan']}"
                      + (f"  min Z_c {r['z_min']:.6g} (non-positive {r['z_nonpos']}, "
                         f"nan {r['z_nan']})" if "z_min" in r else ""))
                describe_bad(t, r["bad"])
            for et in (getattr(indata, "external_terms", []) or []):
                try:
                    ev = float(np.asarray(et.nll(f.x)))
                except Exception as ex:
                    ev = f"<{ex}>"
                print(f"  external '{getattr(et,'name','?')}': nll {ev}")
        return v, g, H

    v0, g0, H0 = at(x0, "START POINT (parameter defaults)")
    if a.no_min or not np.all(np.isfinite(H0)):
        if not np.all(np.isfinite(H0)):
            j = np.nonzero(~np.isfinite(H0).all(axis=1))[0]
            print(f"\n  NON-FINITE Hessian ROWS at the START point: "
                  f"{[names[k] for k in j[:20]]}")
            j = np.nonzero(~np.isfinite(g0))[0]
            print(f"  NON-FINITE gradient components: {[names[k] for k in j[:20]]}")
        return 0 if np.all(np.isfinite(H0)) else 1

    # ---- 2. the real minimiser, instrumented -------------------------------
    import scipy.optimize

    seen = {"last_hess_x": None, "n": 0}

    def sloss(y):
        f.x.assign(tf.constant(y, f.x.dtype))
        val, grad = f.loss_val_grad()
        return float(val.numpy()), np.asarray(grad)

    def shess(y):
        seen["last_hess_x"] = np.array(y, dtype=np.float64)
        seen["n"] += 1
        f.x.assign(tf.constant(y, f.x.dtype))
        val, grad, hess = f.loss_val_grad_hess()
        H = np.asarray(f.hess_for_minimizer(hess))
        nb = int(np.sum(~np.isfinite(H)))
        print(f"  [hess {seen['n']:2d}] |x-x0| {np.linalg.norm(y-x0):.6g}  "
              f"loss {float(val.numpy()):.6f}  {'OK' if nb==0 else f'{nb} NON-FINITE'}")
        return H

    accepted = []

    def cb(xk, *rest):
        accepted.append(np.array(xk, dtype=np.float64))
        print(f"  [accepted step {len(accepted)}] |x-x0| = "
              f"{np.linalg.norm(xk-x0):.6g}")
        if len(accepted) >= a.maxiter:
            raise StopIteration

    print("\n--- running scipy trust-exact (no preconditioning, as rabbit does)")
    try:
        scipy.optimize.minimize(sloss, x0, method="trust-exact", jac=True,
                                hess=shess, tol=0.0, callback=cb)
        print("  minimiser returned without raising")
    except StopIteration:
        print("  stopped by the callback (maxiter)")
    except Exception as ex:
        print(f"  MINIMIZER RAISED: {type(ex).__name__}: {ex}")

    xb = seen["last_hess_x"]
    if xb is None or np.allclose(xb, x0):
        print("  the failure is not at a trial point away from the start")
        return 0
    d = xb - x0
    print(f"\n--- the trial point it aborted at: |dx| = {np.linalg.norm(d):.6g}")
    ordr = np.argsort(-np.abs(d))
    for k in ordr[:12]:
        print(f"      {names[k]:16s} {x0[k]:+12.6g} -> {xb[k]:+12.6g}  "
              f"(d = {d[k]:+.6g})")

    # ---- 3. bisect x0 -> x_bad ---------------------------------------------
    def bad_at(t):
        x = x0 + t * d
        for tm in terms:
            r = term_report(tm, x, names)
            if r["n_nonpos"] or r["n_nan"] or r.get("z_nonpos", 0):
                return tm, r
        return None, None

    lo_t, hi_t = 0.0, 1.0
    tm, r = bad_at(1.0)
    if tm is None:
        print("  the density is positive everywhere on the whole step -- the "
              "NaN is NOT a non-positive density; look at the Hessian pieces")
        at(xb, "THE ABORT POINT")
        return 0
    for _ in range(24):
        mid = 0.5 * (lo_t + hi_t)
        t2, r2 = bad_at(mid)
        if t2 is None:
            lo_t = mid
        else:
            hi_t, tm, r = mid, t2, r2
    print(f"\n--- FIRST non-positive density along the step at t = {hi_t:.6g} "
          f"(last good {lo_t:.6g}), term '{tm.name}'")
    print(f"    min L_i {r['li_min']:.6g}, {r['n_nonpos']} non-positive, "
          f"{r['n_nan']} nan"
          + (f", min Z_c {r['z_min']:.6g} ({r['z_nonpos']} non-positive)"
             if "z_min" in r else ""))
    describe_bad(tm, r["bad"])
    xf = x0 + hi_t * d
    for k in ordr[:8]:
        print(f"      {names[k]:16s} {x0[k]:+12.6g} -> {xf[k]:+12.6g}")

    # ---- 4. which direction alone does it ----------------------------------
    print("\n--- one coordinate at a time, at the FULL step size")
    for k in ordr[:10]:
        if abs(d[k]) < 1e-14:
            continue
        x = x0.copy()
        x[k] = xb[k]
        worst = None
        for tm2 in terms:
            r2 = term_report(tm2, x, names)
            if worst is None or r2["li_min"] < worst[1]["li_min"]:
                worst = (tm2, r2)
        r2 = worst[1]
        print(f"      {names[k]:16s} d {d[k]:+.6g}: min L_i {r2['li_min']:.6g}, "
              f"{r2['n_nonpos']} non-positive, {r2['n_nan']} nan"
              + (f", min Z_c {r2['z_min']:.6g}" if "z_min" in r2 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
