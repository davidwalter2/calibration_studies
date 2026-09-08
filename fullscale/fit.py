#!/usr/bin/env python3
"""Fit a full-scale unbinned card with a CHUNK-ACCUMULATED Hessian.

`zchannel/fit_z.py` and `rabbit_fit.py` both differentiate the whole sample's
NLL twice with the entire forward tape live -- 68 GB at 300 k candidates and 5
parameters, i.e. ~1 TB at the 3.7 M of the DY production. This driver is the
same objective, evaluated chunk by chunk (`chunkfit.ChunkedObjective`), so the
peak is set by ONE chunk and the fit runs anywhere.

It also quotes the SANDWICH covariance. The sample is MiNNLO-weighted; the
inverse Hessian is not its covariance and understates every error by
sqrt(N/N_eff) ~ 1.21 before any model mis-specification is considered.

TRUTH. POWHEG converts the PDG inputs to the constant-width scheme
unconditionally and generated

    m_Z    = 91.153509740726733 GeV
    Gamma_Z = 2.4932018986110700 GeV

(`zchannel/data/generator_settings_DYJetsToMuMu_powhegMiNNLO_UL16.md`). The
provider's `width_scheme="fixed"` references are MZ_FIXED / GZ_FIXED, so the
fitted offsets are compared against those two numbers, not against the PDG.

usage:
    python fit.py --card cards/zcard_dyv2.hdf5 --fix k_hit k_ms k_ioni k_rad
"""
import argparse
import json
import os
import sys
import time

import numpy as np

# the parameters whose convergence decides whether a result may be quoted
POI_GATE = ("m_Z", "Gamma_Z", "alpha", "k_ms", "k_hit", "k_ioni", "k_rad")
import tensorflow as tf
# scipy.optimize.minimize is reached through minimize_driver (see --method)

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import minimize_driver as md  # noqa: E402
from chunkfit import ChunkedObjective, report_cov  # noqa: E402
from devobj import DeviceChunkedObjective, check_against_host  # noqa: E402

# the generator's own EW inputs, constant-width scheme
GEN_MZ = 91.153509740726733
GEN_GZ = 2.4932018986110700
DTYPE = tf.float64


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--card", required=True)
    p.add_argument("--term", default=None)
    p.add_argument("--fix", nargs="*", default=[],
                   help="parameters held at their declared starting value. "
                        "The MC-truth resolution is `--fix k_hit k_ms k_ioni "
                        "k_rad`: the in-maker exponents are computed from the "
                        "SIM's own material, so 1 IS the truth.")
    p.add_argument("--project", type=float, nargs="+", default=[],
                   help="candidate counts to project the covariance to "
                        "(default: the term's own n)")
    p.add_argument("--chunk", type=int, default=0,
                   help="override the card's chunk size")
    p.add_argument("--hess-mode", choices=["pfor", "hvp"], default="pfor")
    p.add_argument("--ares", choices=["card", "on", "off"], default="card",
                   help="override the card's self-consistent-resolution "
                        "switch. `a_res` itself is a stored array, so the ON "
                        "and OFF variants are the SAME card and the same "
                        "candidates -- the only difference is the switch.")
    p.add_argument("--jensen", choices=["card", "exact", "shift", "off"],
                   default="card",
                   help="override the card's Jensen mode; same reasoning")
    p.add_argument("--corr-clip", type=float, default=None,
                   help="override the domain of BOTH corrections, in units of "
                        "sigma_i. It is a scalar attribute of the term, so the "
                        "whole scan runs off one card.")
    p.add_argument("--no-fit", action="store_true")
    p.add_argument("--no-sandwich", action="store_true")
    md.add_arguments(p)
    p.add_argument("--maxiter", type=int, default=200)
    p.add_argument("--start-from", default=None,
                   help="a previous fit.py json whose fitted values seed this "
                        "one. The full-scale Hessian is the expensive object, "
                        "so the cheap route to a full-scale minimum is to "
                        "locate it on a subsample and take two or three Newton "
                        "steps here; the likelihood is quadratic to well "
                        "inside the subsample's own error.")
    p.add_argument("--gtol", type=float, default=1e-6)
    p.add_argument("--edm-tol", type=float, default=1e-3,
                   help="THE convergence criterion: rabbit's own EDM, "
                        "`0.5 g^T H^-1 g` (rabbit.tfhelpers.edmval), the NLL "
                        "still to be gained WITH the parameter correlations. "
                        "rabbit's fitter has no edmtol -- it runs to gtol=0, "
                        "terminating when the quadratic model predicts no "
                        "further improvement, and reports EDM as a diagnostic "
                        "-- so this threshold is ours and is chosen to mean "
                        "something: for a quadratic a displacement of d sigma "
                        "in the worst direction costs d^2/2, so EDM < 1e-3 is "
                        "d < 0.045 sigma. `|grad|inf` is NOT a convergence "
                        "test here (condition number 3e12 measured), and the DIAGONAL "
                        "Newton step is only a proxy: it ignores the POI-shape "
                        "correlations, which is where the displacement is.")
    p.add_argument("-o", "--output", default=None, help="json result")
    p.add_argument("--label", default="")
    return p.parse_args()


def truth_offsets(term):
    """(m_Z, Gamma_Z) truth in the card's own fitted units, or NaN."""
    cfg = term.config()
    prov = cfg.get("kernel", {}).get("provider", {})
    if prov.get("type") != "zgamma":
        return {}
    out = {}
    if prov.get("mz_param"):
        out[prov["mz_param"]] = (GEN_MZ - prov["mz_ref"]) / prov["mz_unit"]
    if prov.get("gz_param"):
        out[prov["gz_param"]] = (GEN_GZ - prov["gz_ref"]) / prov["gz_unit"]
    return out


def main():
    import h5py
    import resource

    from rabbit import unbinned

    args = parse_args()
    t0 = time.time()
    with h5py.File(args.card, "r") as f:
        terms = unbinned.read_unbinned_terms_from_h5(f["unbinned_terms"])
    term = terms[0] if args.term is None else \
        next(t for t in terms if t.name == args.term)
    tload = time.time() - t0
    cfg = term.config()
    print(f"[fit] {args.label or term.name}: {term.n} candidates, "
          f"nt {term.nt} (integration {getattr(term, 'nt_int', term.nt)}), "
          f"loaded in {tload:.0f} s")
    print(f"      params {list(term.param_names)}")
    print(f"      norm_window {cfg.get('norm_window')}, "
          f"upsample {cfg.get('upsample')}, "
          f"self_consistent_sigma {cfg.get('self_consistent_sigma')}, "
          f"jensen {cfg.get('jensen_mode')}, "
          f"corr_form {cfg.get('corr_form', 'residual')}, "
          f"background {cfg['background']['type']}")

    # ---- fit-time model switches (one card, every variant) --------------
    # In the FLUCTUATION form the two corrections are baked into
    # per-candidate constants at construction, so the flags cannot simply be
    # flipped: `set_corrections` rebuilds them.  On a rabbit that predates it
    # the old in-place flip is still correct (residual form only).
    if args.ares != "card" or args.jensen != "card":
        want_a = None if args.ares == "card" else (args.ares == "on")
        want_j = None if args.jensen == "card" else args.jensen
        if want_a and getattr(term, "a_res", None) is None:
            raise SystemExit("--ares on, but the card stores no a_res")
        if want_j not in (None, "off") and getattr(term, "jensen_s2", None) is None:
            raise SystemExit(f"--jensen {args.jensen}, but the card stores no "
                             f"jensen_s2")
        if hasattr(term, "set_corrections"):
            term.set_corrections(self_consistent_sigma=want_a, jensen_mode=want_j)
        else:
            if want_a is not None:
                term.self_consistent_sigma = want_a
                term._dyn_sigma = bool(
                    want_a and term.a_res is not None
                    and np.any(term._a_res_np != 0.0))
            if want_j is not None:
                term.jensen_mode = want_j
                term._jensen = (
                    want_j != "off" and term.jensen_s2 is not None
                    and term.jensen_scale != 0.0
                    and bool(np.any(term._jensen_s2_np != 0.0)))
        print(f"      OVERRIDE ares -> {args.ares}, jensen -> {args.jensen} "
              f"(active: ares {getattr(term, '_dyn_sigma', False)}, "
              f"jensen {getattr(term, '_jensen', False)}, "
              f"fluctuation {getattr(term, '_fluct_active', False)})")

    if args.corr_clip is not None:
        if not hasattr(term, "corr_clip"):
            raise SystemExit("this rabbit's MassCFTerm has no `corr_clip`")
        if getattr(term, "corr_form", "residual") != "residual":
            raise SystemExit(
                "--corr-clip has no meaning on a fluctuation-form card: that "
                "form has no residual-valued argument to clip")
        term.corr_clip = float(args.corr_clip)
        print(f"      OVERRIDE corr_clip -> {term.corr_clip}")

    names = list(term.param_names)
    fixed = set(args.fix)
    unknown = fixed - set(names)
    if unknown:
        raise SystemExit(f"--fix names not in the card: {sorted(unknown)}")
    free = [i for i, nm in enumerate(names) if nm not in fixed]
    engine = md.resolve_engine(args)
    if engine == "device" and getattr(args, "devices", 1) > 1:
        from shardobj import ShardedChunkedObjective
        obj = ShardedChunkedObjective([term], free, chunk=args.chunk or None,
                                      devices=args.devices)
    elif engine == "device":
        obj = DeviceChunkedObjective([term], free, chunk=args.chunk or None)
    else:
        obj = ChunkedObjective([term], free, hess_mode=args.hess_mode,
                               chunk=args.chunk or None)
    x0 = obj.x0[free]
    if args.start_from:
        with open(args.start_from) as fh:
            prev = json.load(fh)
        pv = dict(zip(prev["params"], prev.get("fitted", prev["params"])))
        moved = []
        for i, nm in enumerate(obj.freenames):
            if nm in pv:
                x0[i] = float(pv[nm])
                moved.append(nm)
        print(f"      seeded from {args.start_from}: {moved}")
    resume_radius = None
    if args.resume:
        x0, resume_radius = md.load_snapshot(args.resume, obj.freenames, x0,
                                             log=lambda m: print("     " + m))
    print(f"      {len(free)} free, {len(fixed)} fixed{' ' + str(sorted(fixed)) if fixed else ''};"
          f" {obj.nchunk} chunks of {term.chunk}")

    if args.check_device and engine == "device":
        ref = ChunkedObjective([term], free, hess_mode="hvp",
                              chunk=args.chunk or None)
        check_against_host(obj, ref, x0, log=lambda m: print("   " + m))
        del ref

    truth = truth_offsets(term)
    proj = args.project or [float(term.n)]

    t0 = time.time()
    f0, g0 = obj.value_grad(x0)
    t_grad = time.time() - t0
    t0 = time.time()
    H0 = obj.hess(x0)
    t_hess = time.time() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 ** 2
    print(f"\n  reference point: NLL {f0:.6f}, |grad|inf {np.max(np.abs(g0)):.4g}"
          f"   [value+grad {t_grad:.1f} s, Hessian {t_hess:.1f} s ({args.hess_mode}), "
          f"peak RSS {rss:.1f} GB]")
    ev = np.linalg.eigvalsh(H0)
    print(f"  Hessian eigenvalues {np.array2string(ev, precision=4)} "
          f"(cond {ev.max()/max(ev.min(), 1e-300):.3g})")
    report_cov(obj.freenames, H0, term.n, proj,
               "expected (Asimov) errors from the reference-point information")

    res = {"card": os.path.abspath(args.card), "n": int(term.n),
           "ares_active": bool(getattr(term, "_dyn_sigma", False)),
           "jensen_active": bool(getattr(term, "_jensen", False)),
           "jensen_mode": getattr(term, "jensen_mode", "n/a"),
           "corr_clip": float(getattr(term, "corr_clip", 0.0)),
           "corr_form": getattr(term, "corr_form", "residual"),
           "fluct_active": bool(getattr(term, "_fluct_active", False)),
           "label": args.label, "params": obj.freenames,
           "fixed": sorted(fixed), "nll0": f0,
           "t_load": tload, "t_grad": t_grad, "t_hess": t_hess,
           "rss_gb": rss, "config": cfg, "truth": truth,
           "asimov_err": np.sqrt(np.diag(np.linalg.inv(H0))).tolist()}

    if not args.no_fit:
        # A Krylov trust-region step never forms the Hessian: it asks for O(10)
        # Hessian-VECTOR products, each ~2 gradients and INDEPENDENT of the
        # parameter count.  `trust-exact` needs the full matrix at EVERY
        # iteration, which at 99 free parameters OOMs an H200 at chunk 32768 and
        # is ~105 h where it fits.  One full Hessian is still built AFTER the
        # fit, for the covariance.  The `tf-` methods keep the trust-region
        # subproblem on the device as well (rabbit/minimizer/, PR #153).
        snap = md.make_snapshotter(args, obj.freenames,
                                   log=lambda m: print("   " + m))
        t0 = time.time()
        with md.GpuMonitor(args.gpu_monitor) as gpu:
            r = md.minimize(obj, x0, args, snapshotter=snap,
                            trust_radius=resume_radius,
                            log=lambda m: print("   " + m))
        t_fit = time.time() - t0
        res["minimizer"] = md.timing_report(obj, r, t_fit, gpu=gpu,
                                            log=lambda m: print("   " + m))
        res["method"], res["engine"] = args.method, engine
        H = obj.hess(r.x)
        C = np.linalg.inv(H)
        err = np.sqrt(np.diag(C))
        print(f"\n  fit: {r.nit} iterations, {t_fit:.0f} s, NLL {r.fun:.6f}, "
              f"|grad|inf {np.max(np.abs(r.jac)):.3g}"
              + (f", {obj.nbad} non-finite steered back" if obj.nbad else ""))
        errs = err
        if not args.no_sandwich:
            t0 = time.time()
            J = obj.sandwich(r.x)
            t_sw = time.time() - t0
            Csw = C @ J @ C
            errs = np.sqrt(np.diag(Csw))
            print(f"  sandwich in {t_sw:.0f} s: error ratio "
                  f"{np.array2string(errs/err, precision=4)}")
            res["t_sandwich"] = t_sw
            res["sandwich_err"] = errs.tolist()
            res["sandwich_ratio"] = (errs / err).tolist()
        report_cov(obj.freenames, H, term.n, proj,
                   "fitted values / observed information")
        print("\n    fitted values (sandwich errors where available):")
        w = max(len(s) for s in obj.freenames) + 1
        print(f"      {'parameter':>{w}s} {'fitted':>13s} {'error':>11s} "
              f"{'truth':>11s} {'fit-truth':>12s} {'pull':>7s}")
        for nm, v, e in zip(obj.freenames, r.x, errs):
            tv = truth.get(nm, np.nan)
            dv = v - tv
            print(f"      {nm:>{w}s} {v:+13.5f} {e:11.5f} "
                  + (f"{tv:+11.5f} {dv:+12.5f} {dv/e:7.2f}" if np.isfinite(tv)
                     else f"{'-':>11s} {'-':>12s} {'-':>7s}"))
        # ---- THE CONVERGENCE GATE ------------------------------------
        # `|grad|inf` is NOT a convergence test for this objective. Its Hessian
        # eigenvalues span 0.05 to 1e5 -- the soft ones are `1/sigma^2` for
        # `m_Z` and `Gamma_Z`, the stiff ones the `K(m)` shape coefficients --
        # so a stopping rule on the unscaled gradient infinity norm stops when
        # the STIFFEST direction is converged and says nothing about the
        # softest. Measured: a full-statistics fit stopped at
        # `|grad|inf = 1.42`, ALL of it `shape5` (converged to 0.003 sigma),
        # with `m_Z` 0.98 sigma and `Gamma_Z` 2.06 sigma from their minimum and
        # both POIs still within 1e-4 of their STARTING values. It looked like
        # a perfect closure and it was a fit that never took a POI step.
        #
        # What is reported instead is the diagonal Newton step at the stop, in
        # units of each parameter's OWN error: `g_i sigma_i`, since
        # `H_ii ~ 1/sigma_i^2`. It is a LOWER bound on the true displacement
        # where the POIs correlate with the shapes, which is the safe direction.
        # THE criterion is rabbit's own EDM, `0.5 g^T H^-1 g`
        # (`rabbit.tfhelpers.edmval`): the amount of NLL still to be gained,
        # WITH the parameter correlations, which is exactly what a diagonal
        # proxy throws away. `C` is already inverted for the errors, so it is
        # free. `Delta = -H^-1 g` in units of each error is kept as the
        # interpretable displacement column, not as the test.
        gj_ = np.asarray(r.jac)
        delta = -(C @ gj_)
        edm = float(0.5 * gj_ @ (C @ gj_))
        step = (delta / errs).tolist()
        dstep = (gj_ * errs).tolist()          # the OLD diagonal proxy, for the record
        pois = [nm for nm in obj.freenames if nm in POI_GATE]
        worst = max((abs(step[obj.freenames.index(nm)]) for nm in pois),
                    default=0.0)
        converged = bool(edm < args.edm_tol)
        print(f"\n    convergence: EDM = 0.5 g^T H^-1 g = {edm:.4g} "
              f"(requirement < {args.edm_tol:g}) -> "
              f"{'CONVERGED' if converged else 'NOT CONVERGED'}")
        print(f"      |grad|inf = {np.max(np.abs(gj_)):.3g} is NOT the test: "
              f"the Hessian is ill-conditioned here (measured cond 3.4e12)")
        print(f"      {'parameter':>{w}s} {'gradient':>13s} "
              f"{'FULL step [sig]':>16s} {'diag proxy':>12s}")
        for nm, g_, sj, dj in zip(obj.freenames, gj_, step, dstep):
            mark = "  <-- POI" if nm in POI_GATE else ""
            print(f"      {nm:>{w}s} {g_:+13.5g} {sj:+16.5f} {dj:+12.5f}{mark}")
        print(f"      worst POI displacement {worst:.4f} sigma "
              f"(the diagonal proxy said "
              f"{max((abs(dstep[obj.freenames.index(nm)]) for nm in pois), default=0.0):.4f})")
        if not converged:
            print("      *** THIS RESULT MUST NOT BE QUOTED. Re-run with "
                  "--start-from this json and --method trust-exact. ***")
        res.update({"fitted": r.x.tolist(), "err": err.tolist(),
                    "grad": np.asarray(r.jac).tolist(),
                    "newton_step_sigma": step,
                    "diag_step_sigma": dstep,
                    "worst_poi_step_sigma": float(worst),
                    "edm": edm,
                    "edm_tol": float(args.edm_tol),
                    "converged": converged,
                    "nll": float(r.fun), "nit": int(r.nit), "t_fit": t_fit,
                    "gradmax": float(np.max(np.abs(r.jac))),
                    "corr": (C / np.outer(err, err)).tolist(),
                    "projections": {str(p): (err * np.sqrt(term.n / p)).tolist()
                                    for p in proj}})
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".",
                    exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(res, fh, indent=1, default=str)
        print(f"\n  -> {args.output}")


if __name__ == "__main__":
    main()
