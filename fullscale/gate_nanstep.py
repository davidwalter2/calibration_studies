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
    ap.add_argument(
        "--scan", nargs="*", type=float, default=None,
        help="skip the Hessian entirely and walk each FLOATING parameter "
        "alone to these offsets, reporting the density. A trust-region step "
        "of the default initial radius 1.0 has every component bounded by 1, "
        "so the coordinate walk brackets what the first step can reach -- and "
        "it costs one eager pass per point instead of one Hessian.")
    ap.add_argument(
        "--amax-scan", nargs="*", type=float, default=None,
        help="ladder of `corr_a_max` values (0 = unbounded, today's default). "
        "For each rung, per unbinned term: how many candidates the bound "
        "reaches, how many modelled densities are non-positive, and the "
        "minimum density -- at the default parameter point AND at the four "
        "displaced points the `corr_coeff_max` scan used (m_Z +-30 MeV, "
        "Gamma_Z +-60 MeV). The acceptance is the largest value leaving ZERO "
        "non-positive densities anywhere. Also reports whether the bounded "
        "population is the same one `corr_coeff_max` already reaches, and the "
        "mass weight (1/sigma^2, relative to the median) of the candidates "
        "that go non-positive unbounded.")
    ap.add_argument(
        "--scan-combo", nargs="*", default=None,
        help="extra directions to scan with --scan, as signed parameter sums, "
        "e.g. 'k_hit+k_ms+k_ioni+k_rad' or 'k_hit-k_ms+k_ioni-k_rad'. The "
        "direction is normalised to unit length, so the offset is the "
        "displacement in the same units as the trust radius. These are the "
        "directions a trust-region hard case actually walks along: the "
        "smallest-curvature eigenvector of a degenerate block is an arbitrary "
        "vector inside it, not a coordinate axis.")
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

    if a.amax_scan is not None:
        pts = [("default", {})]
        for nm, d in (("m_Z", 30.0), ("Gamma_Z", 60.0)):
            if nm in names:
                pts += [(f"{nm}{d:+.0f}", {nm: +d}), (f"{nm}{-d:+.0f}", {nm: -d})]
        # the populations, and the weight of what goes wrong, ONCE
        for t in terms:
            if getattr(t, "_fl_a", None) is None or getattr(t, "_fl_g", None) is None:
                print(f"  term '{t.name}': no fluctuation block, nothing to bound")
                continue
            av = np.abs(np.asarray(t._fl_a))
            gv = np.abs(np.asarray(t._fl_g))
            sg = np.asarray(t.sigma)
            w = 1.0 / sg**2
            wmed = float(np.median(w))
            r0 = term_report(t, x0, names)
            bad = np.array([b[0] for b in r0["bad"]], dtype=int)
            print(f"\n  term '{t.name}': n {t.n}, corr_coeff_max "
                  f"{t.corr_coeff_max:g}, corr_a_max {t.corr_a_max:g}")
            print(f"    |a| quantiles 50/90/99/99.9/max: "
                  + " ".join(f"{q:.4g}" for q in np.percentile(
                      av, [50, 90, 99, 99.9, 100])))
            print(f"    |g| >= corr_coeff_max: "
                  f"{int(np.sum(gv >= t.corr_coeff_max)) if t.corr_coeff_max else 0}")
            if bad.size:
                print(f"    non-positive at the default point: {bad.size}, "
                      f"their |a| {np.array2string(av[bad], precision=4)}, "
                      f"|g| {np.array2string(gv[bad], precision=4)}, "
                      f"mass weight 1/sigma^2 relative to the median "
                      f"{np.array2string(w[bad] / wmed, precision=4)}")
        hdr = (f"\n  {'a_max':>8s} {'term':10s} {'point':12s} {'bounded':>9s} "
               f"{'also |g|':>9s} {'non-pos':>8s} {'min L_i':>12s} {'nll':>16s}")
        print(hdr)
        for amax in a.amax_scan:
            for t in terms:
                if hasattr(t, "set_corr_bounds"):
                    t.set_corr_bounds(a_max=amax)
                av = (np.abs(np.asarray(t._fl_a))
                      if getattr(t, "_fl_a", None) is not None else None)
                gv = (np.abs(np.asarray(t._fl_g))
                      if getattr(t, "_fl_g", None) is not None else None)
                # what the bound REACHED: recomputed against the unbounded a is
                # not available after clipping, so count the saturated entries
                nb = (0 if not amax or av is None
                      else int(np.sum(av >= amax * (1 - 1e-12))))
                nboth = (
                    0
                    if not amax or av is None or gv is None or not t.corr_coeff_max
                    else int(np.sum((av >= amax * (1 - 1e-12))
                                    & (gv >= t.corr_coeff_max * (1 - 1e-12))))
                )
                for label, over in pts:
                    x = x0.copy()
                    for k, v in over.items():
                        x[names.index(k)] = x0[names.index(k)] + v
                    r = term_report(t, x, names)
                    print(f"  {amax:8.4g} {t.name:10s} {label:12s} {nb:9d} "
                          f"{nboth:9d} {r['n_nonpos']:8d} {r['li_min']:12.4g} "
                          f"{r['nll_eager']:16.4f}")
        return 0

    if a.scan is not None:
        print("\n--- START POINT, density only (no Hessian)")
        for t in terms:
            r = term_report(t, x0, names)
            print(f"  term '{t.name}': nll {r['nll_eager']:.6f}  "
                  f"min L_i {r['li_min']:.6g}  non-positive {r['n_nonpos']}  "
                  f"nan {r['n_nan']}"
                  + (f"  min Z_c {r['z_min']:.6g}" if "z_min" in r else ""))
        offs = a.scan or [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0]
        print("\n--- one FLOATING parameter at a time, |d| bracketing the "
              "initial trust radius 1.0")
        hdr = (f"  {'parameter':16s} {'d':>8s} {'nll':>16s} {'min L_i':>12s} "
               f"{'non-pos':>8s} {'nan':>6s} {'min Z_c':>12s}")
        print(hdr)
        dirs = []
        for k in free:
            u = np.zeros(len(names))
            u[k] = 1.0
            dirs.append((names[k], u))
        for spec in (a.scan_combo or []):
            u = np.zeros(len(names))
            tok = spec.replace("-", "+-").split("+")
            for t in tok:
                t = t.strip()
                if not t:
                    continue
                sgn, nm = (-1.0, t[1:]) if t.startswith("-") else (1.0, t)
                u[names.index(nm)] = sgn
            u /= max(np.linalg.norm(u), 1e-300)
            dirs.append((spec, u))
        for label, u in dirs:
            for d in offs:
                x = x0 + d * u
                for t in terms:
                    r = term_report(t, x, names)
                    print(f"  {label:16s} {d:+8.3f} {r['nll_eager']:16.4f} "
                          f"{r['li_min']:12.4g} {r['n_nonpos']:8d} "
                          f"{r['n_nan']:6d} "
                          f"{r.get('z_min', float('nan')):12.4g}")
                    if r["n_nonpos"] or r["n_nan"] or r.get("z_nonpos", 0):
                        describe_bad(t, r["bad"])
        return 0

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
