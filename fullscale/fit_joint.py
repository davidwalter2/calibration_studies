#!/usr/bin/env python3
"""PHASE 2 -- fit the JOINT card: two mass terms plus the hit-chi2 quadratic.

`fit.py` cannot do this. It takes `terms[0]` (or `--term`) and nothing else, so
on a joint card it silently drops the J/psi mass term AND the external
quadratic term -- which is precisely the hit-chi2 information the joint fit
exists to combine with the two mass constraints. This driver evaluates all
three over ONE parameter vector.

WHAT IS ADDED, AND IN WHOSE CONVENTION
--------------------------------------
The unbinned part is `chunkfit.ChunkedObjective`, unchanged: the same chunked
value / gradient / Hessian / sandwich `fit.py` uses, with several terms instead
of one. `ChunkedObjective` requires the FIRST term's parameter list to be a
superset of every other's, so the terms are ordered by parameter count and the
subset property is CHECKED, not assumed.

The external term is `rabbit.external_likelihood`'s stored form, evaluated
exactly as `compute_external_nll` does:

    NLL_ext = const [+ lognorm, full NLL only] + g^T x_sub + 0.5 x_sub^T H x_sub

with `x_sub = x[indices]`, `H` symmetric. `make_joint_card.py` writes
`g = G/2`, `H = K/2` -- the chi2 -> NLL factor of 1/2 is already inside the
stored arrays, and this driver must not apply it again. Being a quadratic form
in 92 parameters, its value, gradient (`g + H x_sub`) and Hessian (`H`) are
closed-form; they are added to the chunked ones in the SAME free-parameter
subspace and order that `ChunkedObjective` uses (`x = base + E xf`).

THE SANDWICH, AND THE ONE THING IT CANNOT SEE
---------------------------------------------
`globalfit/extract.py` accumulates `jsand = sum_i G_i G_i^T` over the quadratic
term's own candidates, in chi2 units, and `make_joint_card.py` carries it into
the card's `global_index_map` auxiliary in the card's (whitened) units. The NLL
score of one such candidate is `s_i = G_i/2`, so the meat is `jsand/4`. That is
added to the unbinned meat when `--ext-sandwich` is on (the default).

What NOBODY has is the CROSS term. A J/psi candidate contributes to the mass
term AND to the quadratic term, so its two scores are correlated, and the
correct meat has a cross block that neither extraction stored (each accumulates
only its own outer products). Adding the two meats therefore treats the two
constraints as independent, which they are not. The driver says so in its
output rather than quoting a robust error that quietly assumes it away.

usage:
    ./run_tf.sh python3 fit_joint.py --card cards/joint_v2.hdf5 \
        --fix k_hit k_ms k_ioni k_rad --hess-mode hvp -o results/fit_joint.json
    ./run_tf.sh python3 fit_joint.py --card cards/joint_smoke.hdf5 --selftest
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from chunkfit import ChunkedObjective, report_cov  # noqa: E402
from fit import GEN_MZ, GEN_GZ, truth_offsets      # noqa: E402

DTYPE = tf.float64


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--card", required=True)
    p.add_argument("--terms", nargs="*", default=None,
                   help="unbinned terms to include (default: all). "
                        "`--terms zmass` is the Z leg of a joint card.")
    p.add_argument("--fix", nargs="*", default=[],
                   help="parameters held at their declared starting value")
    p.add_argument("--fix-globals", action="store_true",
                   help="freeze all 92 calibration parameters. With "
                        "`--terms zmass` this is the plumbing configuration: "
                        "the external term becomes a constant and the "
                        "objective is exactly fit.py's on the Z term.")
    p.add_argument("--free", nargs="*", default=None,
                   help="the ONLY parameters left free; everything else is "
                        "fixed. Overrides --fix / --fix-globals.")
    p.add_argument("--no-external", action="store_true",
                   help="drop the hit-chi2 term (diagnostic)")
    p.add_argument("--ext-sandwich", action="store_true", default=True,
                   help="add jsand/4 to the sandwich meat")
    p.add_argument("--no-ext-sandwich", dest="ext_sandwich",
                   action="store_false")
    p.add_argument("--chunk", type=int, default=0,
                   help="override the card's chunk size. REFUSED on a card "
                        "whose terms carry a sparse D -- see the error text.")
    p.add_argument("--hess-mode", choices=["pfor", "hvp"], default="hvp",
                   help="`hvp` is the default for MEMORY, not for speed: on "
                        "the 60 k + 60 k smoke card with 99 free parameters "
                        "pfor took 336 s and hvp 1453 s (agreeing to 1.7e-21), "
                        "but pfor holds 99 columns of a (chunk, nt_int) tape "
                        "at once while hvp holds ~2 gradients regardless of "
                        "the parameter count. Use pfor whenever it fits.")
    p.add_argument("--no-fit", action="store_true")
    p.add_argument("--no-sandwich", action="store_true")
    p.add_argument("--selftest", action="store_true",
                   help="the three plumbing gates; implies --no-fit")
    p.add_argument("--fd-params", nargs="*", default=None,
                   help="parameters to finite-difference in --selftest")
    p.add_argument("--maxiter", type=int, default=200)
    p.add_argument("--gtol", type=float, default=1e-6)
    p.add_argument("--start-from", default=None)
    p.add_argument("--project", type=float, nargs="+", default=[])
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--label", default="")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
class ExternalQuadratic:
    """`g^T x_sub + 0.5 x_sub^T H x_sub + const`, in the free subspace.

    Built once from the card. `H` does not depend on the parameters, so the
    free-subspace Hessian is precomputed and the per-evaluation cost is one
    92-vector matvec.
    """

    def __init__(self, raw, names, free, base, log=print):
        from rabbit.external_likelihood import gaussian_scalars

        self.name = raw["name"]
        params = [str(s) for s in np.asarray(raw["params"]).astype(str)]
        idx = np.empty(len(params), dtype=np.int64)
        for i, p in enumerate(params):
            if p not in names:
                raise SystemExit(
                    f"external term '{self.name}' parameter '{p}' is not "
                    f"declared by any unbinned term of this card")
            idx[i] = names.index(p)
        self.params = params
        self.idx = idx
        self.g = (np.zeros(len(params)) if raw["grad_values"] is None
                  else np.asarray(raw["grad_values"], dtype=np.float64).ravel())
        if raw["hess_dense"] is not None:
            H = np.asarray(raw["hess_dense"], dtype=np.float64)
        elif raw["hess_sparse"] is not None:
            H = np.asarray(tf.sparse.to_dense(raw["hess_sparse"]).numpy(),
                           dtype=np.float64)
        else:
            H = np.zeros((len(params), len(params)))
        self.H = 0.5 * (H + H.T)          # the loss only ever sees this part
        const, lognorm = raw.get("const"), raw.get("lognorm")
        if const is None or lognorm is None:
            d_const, d_lognorm = gaussian_scalars(self.g, self.H, self.name)
            const = d_const if const is None else const
            lognorm = d_lognorm if lognorm is None else lognorm
        self.const, self.lognorm = float(const), float(lognorm)

        self.free = list(free)
        self.base = np.asarray(base, dtype=np.float64)
        # column of each external parameter in the free vector, or -1
        fpos = {p: c for c, p in enumerate(self.free)}
        self.col = np.array([fpos.get(int(i), -1) for i in idx], dtype=np.int64)
        self.nfreeext = int((self.col >= 0).sum())
        # free-subspace Hessian, once
        n = len(self.free)
        Hf = np.zeros((n, n))
        k = np.flatnonzero(self.col >= 0)
        if len(k):
            Hf[np.ix_(self.col[k], self.col[k])] = self.H[np.ix_(k, k)]
        self.Hfree = Hf
        log(f"  external term '{self.name}': {len(params)} parameters, "
            f"{self.nfreeext} of them free; const {self.const:.6g}, "
            f"lognorm {self.lognorm:.6g}")

    def _xsub(self, xf):
        x = self.base.copy()
        x[self.free] = np.asarray(xf, dtype=np.float64)
        return x[self.idx]

    def value_grad(self, xf):
        xs = self._xsub(xf)
        Hx = self.H @ xs
        val = self.const + float(self.g @ xs) + 0.5 * float(xs @ Hx)
        d = self.g + Hx
        grad = np.zeros(len(self.free))
        k = np.flatnonzero(self.col >= 0)
        if len(k):
            grad[self.col[k]] = d[k]
        return val, grad


class JointObjective:
    """`ChunkedObjective` over several mass terms + the external quadratic."""

    def __init__(self, terms, external, free=None, hess_mode="hvp",
                 chunk=None, log=print):
        self.inner = ChunkedObjective(terms, free, hess_mode=hess_mode,
                                      chunk=chunk, log=log)
        self.names = self.inner.names
        self.free = self.inner.free
        self.freenames = self.inner.freenames
        self.x0 = self.inner.x0
        self.nchunk = self.inner.nchunk
        # `x = x0 everywhere, with the free entries replaced by xf` -- the
        # same map `ChunkedObjective.full` applies, in numpy
        self.ext = None
        if external is not None:
            self.ext = ExternalQuadratic(external, self.names, self.free,
                                         self.x0.copy(), log=log)
        self.nbad_ext = 0

    # -- the three things scipy and the report need ----------------------
    def value_grad(self, xf):
        v, g = self.inner.value_grad(xf)
        if self.ext is not None:
            ev, eg = self.ext.value_grad(xf)
            if v >= 1e29:          # the inner objective steered back
                return v, g
            v = v + ev
            g = g + eg
        return v, g

    def hess(self, xf, mode=None):
        H = self.inner.hess(xf, mode=mode)
        if self.ext is not None:
            H = H + self.ext.Hfree
        return H

    def sandwich(self, xf, check=True, jsand=None):
        J = self.inner.sandwich(xf, check=check)
        if jsand is not None:
            J = J + jsand
        return J

    @property
    def nbad(self):
        return self.inner.nbad


# ---------------------------------------------------------------------------
def load(args, log=print):
    """terms (ordered), the raw external term, and the aux bundles."""
    import h5py
    from rabbit import unbinned
    from rabbit.auxiliary import read_auxiliary_from_h5
    from rabbit.external_likelihood import read_external_terms_from_h5

    with h5py.File(args.card, "r") as f:
        terms = unbinned.read_unbinned_terms_from_h5(f["unbinned_terms"])
        ext = read_external_terms_from_h5(f.get("external_terms"))
        aux = read_auxiliary_from_h5(f.get("auxiliary"))
    if args.terms:
        keep = set(args.terms)
        unknown = keep - {t.name for t in terms}
        if unknown:
            raise SystemExit(f"--terms {sorted(unknown)} not in the card "
                             f"(it has {[t.name for t in terms]})")
        terms = [t for t in terms if t.name in keep]
    if not terms:
        raise SystemExit("no unbinned terms selected")
    # ChunkedObjective wants the SUPERSET first
    terms.sort(key=lambda t: -len(t.param_names))
    head = set(terms[0].param_names)
    for t in terms[1:]:
        missing = sorted(set(t.param_names) - head)
        if missing:
            raise SystemExit(
                f"term '{t.name}' declares {missing} which term "
                f"'{terms[0].name}' does not; the card cannot be fitted over "
                f"one parameter vector as written")
    external = None
    if ext and not args.no_external:
        if len(ext) > 1:
            raise SystemExit(f"more than one external term: "
                             f"{[e['name'] for e in ext]}; not handled")
        external = ext[0]
    return terms, external, aux


def ext_sandwich_block(aux, external, names, free, log=print):
    """`jsand/4` in the free subspace, or None with a stated reason."""
    if external is None:
        return None, "no external term"
    gim = aux.get("global_index_map")
    if gim is None or "jsand" not in gim:
        return None, "the card carries no `jsand`"
    J = np.asarray(gim["jsand"], dtype=np.float64)
    if not np.any(J):
        return None, "`jsand` is all zero (the extraction predates it)"
    gp = [str(s) for s in gim["params"]]
    if gp != list(external["params"].astype(str)):
        # reorder jsand into the external term's parameter order
        pos = {p: i for i, p in enumerate(gp)}
        try:
            k = np.array([pos[p] for p in external["params"].astype(str)])
        except KeyError as e:
            return None, f"`jsand` has no row for {e}"
        J = J[np.ix_(k, k)]
    idx = [names.index(p) for p in external["params"].astype(str)]
    fpos = {p: c for c, p in enumerate(free)}
    col = np.array([fpos.get(i, -1) for i in idx])
    n = len(free)
    out = np.zeros((n, n))
    k = np.flatnonzero(col >= 0)
    if not len(k):
        return None, "none of its parameters is free"
    # chi2 -> NLL: the score is G_i/2, so the meat is jsand/4
    out[np.ix_(col[k], col[k])] = 0.25 * J[np.ix_(k, k)]
    log(f"  external sandwich meat: jsand/4 over {len(k)} free parameters "
        f"(trace {np.trace(out):.6g})")
    return out, None


def selftest(obj, terms, external, args, log=print):
    """The three plumbing gates."""
    log("\n=== selftest ===")
    x0 = obj.x0[obj.free]
    ok = True

    # (a) with the external parameters fixed the term is a CONSTANT, and the
    #     rest is exactly what fit.py evaluates on the same unbinned terms
    inner = ChunkedObjective(terms, obj.free, hess_mode="pfor", log=lambda *a: None)
    vi, gi = inner.value_grad(x0)
    vj, gj = obj.value_grad(x0)
    if obj.ext is not None and obj.ext.nfreeext == 0:
        dv = vj - vi
        c = obj.ext.const + float(obj.ext.g @ obj.ext._xsub(x0)) + 0.5 * float(
            obj.ext._xsub(x0) @ (obj.ext.H @ obj.ext._xsub(x0)))
        rel = abs(dv - c) / max(abs(c), 1.0)
        dg = float(np.max(np.abs(gj - gi)))
        log(f"  (a) globals fixed: NLL_joint - NLL_unbinned = {dv:.12g}, "
            f"the external constant = {c:.12g}, rel {rel:.3e}; "
            f"max |grad difference| {dg:.3e}")
        ok &= rel < 1e-12 and dg == 0.0
    else:
        log("  (a) SKIPPED: run with --fix-globals (and --terms zmass) to "
            "compare against fit.py's objective")

    # (b) analytic gradient vs central finite differences
    picks = args.fd_params
    if not picks:
        picks = [nm for nm in ("m_Z", "Gamma_Z", "k_hit") if nm in obj.freenames]
        picks += [nm for nm in obj.freenames
                  if nm.startswith(("bfield_", "material_"))][:2]
        picks = [p for p in picks if p in obj.freenames][:5]
    # A central difference of an NLL of size |F| taken with step h carries a
    # cancellation floor of ~eps |F| / h, which for |F| ~ 1e5 and h ~ 1e-5 is
    # 2e-6 -- larger than the gradient itself for a parameter as flat as m_Z.
    # The gate is therefore |analytic - fd| against max(1e-5 |fd|, that floor),
    # not a bare relative tolerance.
    log(f"  (b) finite differences on {picks}")
    eps = np.finfo(np.float64).eps
    for nm in picks:
        j = obj.freenames.index(nm)
        h = 1e-5 * max(abs(x0[j]), 1.0)
        xp, xm = x0.copy(), x0.copy()
        xp[j] += h
        xm[j] -= h
        fd = (obj.value_grad(xp)[0] - obj.value_grad(xm)[0]) / (2 * h)
        an = gj[j]
        d = abs(an - fd)
        floor = 4.0 * eps * max(abs(vj), 1.0) / h
        tol = max(1e-5 * abs(fd), floor)
        log(f"      {nm:<22s} analytic {an:+16.8f}  fd {fd:+16.8f}  "
            f"|diff| {d:.3e}  tol {tol:.3e} "
            f"({'rel' if 1e-5*abs(fd) > floor else 'fd noise floor'})")
        ok &= d <= tol

    # (c) pfor vs hvp
    t0 = time.time()
    Hp = obj.hess(x0, mode="pfor")
    tp = time.time() - t0
    t0 = time.time()
    Hh = obj.hess(x0, mode="hvp")
    th = time.time() - t0
    d = float(np.max(np.abs(Hp - Hh)) / max(np.max(np.abs(Hp)), 1e-300))
    log(f"  (c) Hessian pfor ({tp:.1f} s) vs hvp ({th:.1f} s): "
        f"max relative difference {d:.3e}")
    ok &= d < 1e-10
    log(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def main(argv=None):
    import resource

    args = parse_args(argv)
    t0 = time.time()
    terms, external, aux = load(args)
    tload = time.time() - t0
    names = list(terms[0].param_names)
    ntot = sum(t.n for t in terms)
    print(f"[fit_joint] {args.label or 'joint'}: "
          f"{len(terms)} mass term(s) {[t.name for t in terms]}, "
          f"{ntot} candidates, loaded in {tload:.0f} s")
    for t in terms:
        c = t.config()
        print(f"      {t.name}: {t.n} candidates, nt {t.nt} "
              f"(integration {getattr(t, 'nt_int', t.nt)}), "
              f"{len(t.param_names)} params, chunk {t.chunk}, "
              f"kernel {c['kernel']['type']}, "
              f"corr_form {c.get('corr_form', 'residual')}, "
              f"jensen {c.get('jensen_mode')}, "
              f"scale_param {c.get('scale_param')}")

    if args.chunk:
        withD = [t.name for t in terms if getattr(t, "_jac_chunks", None)]
        if withD:
            raise SystemExit(
                f"--chunk is refused on this card: term(s) {withD} carry a "
                "sparse D, and ChunkedObjective(chunk=) rebuilds a term's "
                "`_chunks` but NOT its `_jac_chunks`, so the per-chunk "
                "Jacobian would keep the size the card was WRITTEN at and "
                "`_chunk_residual` would raise `Incompatible shapes`. Rebuild "
                "the card with make_joint_card.py --chunk instead.")

    # ---- the free set ----------------------------------------------------
    globals_ = []
    gim = aux.get("global_index_map")
    if gim is not None:
        globals_ = [str(s) for s in gim["params"]]
    if args.free is not None:
        unknown = set(args.free) - set(names)
        if unknown:
            raise SystemExit(f"--free names not in the card: {sorted(unknown)}")
        fixed = set(names) - set(args.free)
    else:
        fixed = set(args.fix)
        if args.fix_globals:
            if not globals_:
                raise SystemExit("--fix-globals but the card has no "
                                 "`global_index_map` auxiliary")
            fixed |= {g for g in globals_ if g in set(names)}
        unknown = set(args.fix) - set(names)
        if unknown:
            raise SystemExit(f"--fix names not in the card: {sorted(unknown)}")
    free = [i for i, nm in enumerate(names) if nm not in fixed]
    if not free:
        raise SystemExit("every parameter is fixed")

    obj = JointObjective(terms, external, free, hess_mode=args.hess_mode,
                         chunk=None)
    nglob_free = sum(1 for nm in obj.freenames if nm in set(globals_))
    print(f"      {len(free)} free ({nglob_free} of them calibration "
          f"parameters), {len(fixed)} fixed; {obj.nchunk} chunks")

    if args.selftest:
        ok = selftest(obj, terms, external, args)
        return 0 if ok else 1

    # ---- reference point --------------------------------------------------
    x0 = obj.x0[free]
    if args.start_from:
        with open(args.start_from) as fh:
            prev = json.load(fh)
        pv = dict(zip(prev["params"], prev.get("fitted", [])))
        moved = [nm for nm in obj.freenames if nm in pv]
        for i, nm in enumerate(obj.freenames):
            if nm in pv:
                x0[i] = float(pv[nm])
        print(f"      seeded from {args.start_from}: {len(moved)} parameters")

    truth = {}
    for t in terms:
        truth.update(truth_offsets(t))
    if gim is not None and "injected" in gim:
        inj = np.asarray(gim["injected"], dtype=np.float64)
        for nm, v in zip(globals_, inj):
            if nm in names:
                truth[nm] = float(v)
    proj = args.project or [float(ntot)]

    t0 = time.time()
    f0, g0 = obj.value_grad(x0)
    t_grad = time.time() - t0
    t0 = time.time()
    H0 = obj.hess(x0)
    t_hess = time.time() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 ** 2
    print(f"\n  reference point: NLL {f0:.6f}, |grad|inf {np.max(np.abs(g0)):.4g}"
          f"   [value+grad {t_grad:.1f} s, Hessian {t_hess:.1f} s "
          f"({args.hess_mode}), peak RSS {rss:.1f} GB]")
    ev = np.linalg.eigvalsh(H0)
    print(f"  Hessian eigenvalues [{ev[0]:.4g}, {ev[-1]:.4g}], "
          f"cond {ev[-1]/max(ev[0], 1e-300):.4g}, "
          f"{int((ev <= 0).sum())} non-positive")
    small = [nm for nm, e in zip(obj.freenames, np.sqrt(np.abs(np.diag(H0))))
             if e == 0.0]
    if small:
        print(f"  parameters with ZERO information: {small[:8]}"
              + (" ..." if len(small) > 8 else ""))

    res = {"card": os.path.abspath(args.card), "n": int(ntot),
           "terms": [t.name for t in terms],
           "external": None if external is None else external["name"],
           "corr_form": terms[0].config().get("corr_form", "residual"),
           "label": args.label, "params": obj.freenames,
           "fixed": sorted(fixed), "nll0": f0,
           "t_load": tload, "t_grad": t_grad, "t_hess": t_hess,
           "rss_gb": rss, "truth": truth,
           "hess_mode": args.hess_mode}
    try:
        res["asimov_err"] = np.sqrt(
            np.diag(np.linalg.inv(H0))).tolist()
    except np.linalg.LinAlgError:
        res["asimov_err"] = None
    if len(free) <= 20:
        report_cov(obj.freenames, H0, ntot, proj,
                   "expected (Asimov) errors from the reference-point information")

    if args.no_fit:
        if args.output:
            with open(args.output, "w") as fh:
                json.dump(res, fh, indent=1, default=str)
            print(f"\n  -> {args.output}")
        return 0

    # ---- fit --------------------------------------------------------------
    t0 = time.time()
    r = minimize(obj.value_grad, x0, jac=True, hess=obj.hess,
                 method="trust-exact",
                 options={"maxiter": args.maxiter, "gtol": args.gtol})
    t_fit = time.time() - t0
    H = obj.hess(r.x)
    C = np.linalg.inv(H)
    err = np.sqrt(np.diag(C))
    print(f"\n  fit: {r.nit} iterations, {t_fit:.0f} s, NLL {r.fun:.6f}, "
          f"|grad|inf {np.max(np.abs(r.jac)):.3g}"
          + (f", {obj.nbad} non-finite steered back" if obj.nbad else ""))
    errs = err
    if not args.no_sandwich:
        jext, why = (ext_sandwich_block(aux, external, names, free)
                     if args.ext_sandwich else (None, "--no-ext-sandwich"))
        if jext is None:
            print(f"  external sandwich meat NOT added: {why}")
        t0 = time.time()
        J = obj.sandwich(r.x, jsand=jext)
        t_sw = time.time() - t0
        Csw = C @ J @ C
        errs = np.sqrt(np.diag(Csw))
        print(f"  sandwich in {t_sw:.0f} s: error ratio "
              f"[{np.min(errs/err):.4f}, {np.max(errs/err):.4f}] "
              f"(median {np.median(errs/err):.4f})")
        print("  CAVEAT: the mass and hit-chi2 scores of the SAME candidate "
              "are correlated and no extraction stored that cross block, so "
              "the two meats are added as if independent.")
        res["t_sandwich"] = t_sw
        res["sandwich_err"] = errs.tolist()
        res["sandwich_ratio"] = (errs / err).tolist()
        res["ext_sandwich"] = jext is not None

    # ---- report -----------------------------------------------------------
    corr = C / np.outer(err, err)
    gset = set(globals_)
    phys = [i for i, nm in enumerate(obj.freenames) if nm not in gset]
    if phys:
        print("\n    physics parameters (sandwich errors where available):")
        w = max(len(obj.freenames[i]) for i in phys) + 1
        print(f"      {'parameter':>{w}s} {'fitted':>13s} {'error':>11s} "
              f"{'truth':>11s} {'fit-truth':>12s} {'pull':>7s}")
        for i in phys:
            nm, v, e = obj.freenames[i], r.x[i], errs[i]
            tv = truth.get(nm, np.nan)
            dv = v - tv
            print(f"      {nm:>{w}s} {v:+13.5f} {e:11.5f} "
                  + (f"{tv:+11.5f} {dv:+12.5f} {dv/e:7.2f}" if np.isfinite(tv)
                     else f"{'-':>11s} {'-':>12s} {'-':>7s}"))

    gi_ = [i for i, nm in enumerate(obj.freenames) if nm in gset]
    if gi_:
        pulls = np.array([(r.x[i] - truth.get(obj.freenames[i], 0.0)) / errs[i]
                          for i in gi_])
        big = [(obj.freenames[i], p) for i, p in zip(gi_, pulls) if abs(p) > 3]
        print(f"\n    calibration parameters ({len(gi_)} free): pull mean "
              f"{pulls.mean():+.3f}, RMS {pulls.std(ddof=0):.3f}, "
              f"max |pull| {np.max(np.abs(pulls)):.2f}"
              f" ({obj.freenames[gi_[int(np.argmax(np.abs(pulls)))]]})")
        print(f"    |pull| > 3: {len(big)}"
              + ("  " + ", ".join(f"{n} {p:+.1f}" for n, p in big[:8])
                 if big else ""))
        res["global_pulls"] = {obj.freenames[i]: float(p)
                               for i, p in zip(gi_, pulls)}
        for poi in ("m_Z", "Gamma_Z"):
            if poi not in obj.freenames:
                continue
            ip = obj.freenames.index(poi)
            for tag, pref in (("field", "bfield_"), ("material", "material_")):
                blk = [i for i in gi_ if obj.freenames[i].startswith(pref)]
                if not blk:
                    continue
                rho = np.array([corr[ip, i] for i in blk])
                k = int(np.argmax(np.abs(rho)))
                print(f"    rho({poi}, {tag}): max |rho| {np.abs(rho).max():.4f} "
                      f"({obj.freenames[blk[k]]}), RMS {np.sqrt((rho**2).mean()):.4f}")
                res[f"rho_{poi}_{tag}"] = {
                    "max_abs": float(np.abs(rho).max()),
                    "argmax": obj.freenames[blk[k]],
                    "rms": float(np.sqrt((rho ** 2).mean()))}

    res.update({"fitted": r.x.tolist(), "err": err.tolist(),
                "nll": float(r.fun), "nit": int(r.nit), "t_fit": t_fit,
                "gradmax": float(np.max(np.abs(r.jac))),
                "corr": corr.tolist() if len(free) <= 200 else None,
                "projections": {str(p): (err * np.sqrt(ntot / p)).tolist()
                                for p in proj}})
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".",
                    exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(res, fh, indent=1, default=str)
        print(f"\n  -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
