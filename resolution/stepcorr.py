#!/usr/bin/env python3
"""Step-to-step correlation: does the realised energy loss feed back into the
straggling of later steps, and how much of the clean-propagation non-closure
does that feedback carry?

THE HYPOTHESIS
--------------
`cf_propagation_test.model_phi` builds the block characteristic function as a
SUM of per-step exponents,

    S_block(t) = sum_s S_s(w_s t),

i.e. it asserts that the per-step fluctuations are INDEPENDENT, and it
evaluates every step's Urban / delta-ray parameters on the REFERENCE
trajectory.  In Geant4 the energy lost in step i lowers the momentum entering
step i+1, so the straggling of the later steps depends on what already
happened.  An additive exponent cannot represent that.

What the model already gets right, and what this note must not re-discover:

  * the MEAN degradation along the track -- every step's record is written at
    that step's own reference momentum, so d(dE/dx)/ds is in the record;
  * the LINEAR response of a later step's mean loss to an earlier fluctuation
    -- this is in the transport Jacobian.  `F[0,0] = 1.0012` per leg in the
    pT = 3 toy is exactly d(qop_out)/d(qop_in): a track that is already low in
    momentum loses faster, and the block CF transports its own earlier steps
    through that factor.  Over 13 legs it compounds to 1.6 %.

So what is missing is the response of the FLUCTUATION -- xi, Tmax and the
shape of the knock-on spectrum evaluated at the realised rather than the
reference momentum -- plus the nonlinear part of the mean response.
`xi ~ 1/beta^2` is flat at these energies; `Tmax ~ gamma^2` is not, and Tmax is
what the exact-delta correction (NOTES_DELTASPEC) is sensitive to.

THE DECOMPOSITION EVERYTHING HERE IS BUILT ON
---------------------------------------------
Write the local 5D state at plane m as x_m and the propagator's reference as
xref_m, and define the per-plane increment

    D_m = (x_m - xref_m) - F_m (x_{m-1} - xref_{m-1}),      D_0 = x_0 - xref_0

with F_m the exported per-leg transport Jacobian.  Then, telescoping,

    x_K - xref_K = sum_{m<=K} T_{m->K} D_m,   T_{m->K} = F_K F_{K-1} ... F_{m+1}

is an ALGEBRAIC IDENTITY -- it holds event by event with no physics assumption,
because D_m is *defined* as the residue of the linear transport.  D_m is
therefore exactly the object the model calls "the noise of leg m", and the
model's claim is precisely that the D_m are mutually independent with the
sub-block CF as marginal.

That gives three tests, in increasing strength:

  `corr`    Cov(D_i, D_j), i != j.  Zero under the model.  Model-free.
  `cond`    the task's primary test: bin events by the realised accumulated
            loss at an intermediate plane k, then measure the closure of the
            REMAINING increment over planes k+1..N in each bin.  Under
            independence every bin closes identically; the model half of the
            closure is bin-independent BY CONSTRUCTION (it is a function of the
            reference trajectory alone), so the span across bins is a pure
            data statement that no normalization choice can manufacture.
  `shuffle` the decisive one: permute the D_m across events, independently per
            m, and rebuild x_K - xref_K from the identity above.  The surrogate
            sample has the SAME per-plane marginals and ZERO cross-plane
            correlation -- it is the sample the model believes in.  The
            difference between the real and surrogate closure functionals is
            the entire contribution of step-to-step correlation to the
            observable, measured with no model, no CF and no inversion.

`shuffle` decorrelates across PLANE boundaries.  Correlation between Geant4
steps INSIDE one leg is not probed by it; see `--split` for the scaling
argument that bounds it.

SUBCOMMANDS
    check     premise checks: the telescoping identity, the leg-masking
              identity, the F structure, per-bin acceptance, and the
              label-permutation null of the whole `cond` machinery
    corr      Cov/Corr(D_i, D_j) and the conditional means
    cond      the conditional-subsample closure, per bin, with errors
    shuffle   real vs independent-increment surrogate, per plane, per probe
    plots     figures for the note
"""

import argparse
import datetime
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "cleanprop"))

import cf_propagation_test as cpt                                # noqa: E402
import fisher_norm as fn                                         # noqa: E402
from cf_propagation_test import FUNCTIONALS                      # noqa: E402
from wums import logging as _wums_logging                        # noqa: E402

logger = _wums_logging.child_logger(__name__)

SCRATCH = fn.SCRATCH
OUT = os.path.join(SCRATCH, "sc")
UCURVE = fn.UCURVE
STATE = ("qop", "dxdz", "dydz", "locx", "locy")   # the local 5D order

TODAY = datetime.date.today().strftime("%y%m%d")
PLOTDIR = os.path.expanduser(f"~/public_html/ZMass/resolution/{TODAY}_stepcorr")


# ==========================================================================
# samples
#
# Every one of these is ARCHIVED and is reused byte-identically; nothing here
# runs a simulation.  `pt3`/`pt40` are the 400k NOTES_TOY_PT40 NSUB=1 samples
# the task names; `pt3_cut1e4` / `pt40_cut001` are the 200k physically faithful
# (small production cut) configurations from NOTES_TAILHUNT, kept because the
# default-cut toy hands 12-26 % of the primary's loss to explicit secondaries
# and the delta-ray feedback is exactly what this note is about; `real` is the
# 400k CMS-tracker sample.
# ==========================================================================

SAMPLES = {
    "pt3":        dict(kind="toy", pt="pt3",
                       model=f"{SCRATCH}/tp/pt3_K1_model.root",
                       sim=f"{SCRATCH}/tp/pt3_K1_sim.root",
                       label="layered toy pT=3, NSUB=1, default cut, 400k"),
    "pt40":       dict(kind="toy", pt="pt40",
                       model=f"{SCRATCH}/tp/pt40_K1_model.root",
                       sim=f"{SCRATCH}/tp/pt40_K1_sim.root",
                       label="layered toy pT=40, NSUB=1, default cut, 400k"),
    "pt3_cut1e4": dict(kind="toy", pt="pt3",
                       model=f"{SCRATCH}/tp/pt3_K1_model.root",
                       sim=f"{SCRATCH}/th/pt3_cut1e4_s*_sim.root",
                       label="layered toy pT=3, cut 9.49 keV, 200k"),
    "pt40_cut001": dict(kind="toy", pt="pt40",
                        model=f"{SCRATCH}/tp/pt40_K1_model.root",
                        sim=f"{SCRATCH}/th/pt40_cut001_s*_sim.root",
                        label="layered toy pT=40, cut 0.296 MeV, 200k"),
    # `real` is the ARCHIVED NOTES_RADOFF radiation-on sample (200k) -- the one
    # NOTES_DELTASPEC 5.1's per-plane radial-growth table is built on.
    # `real400` is the NOTES_TAILHUNT 400k default-cut production, which is what
    # the published `REAL pT=3, all 19 planes` closure row uses.
    "real":       dict(kind="real", pt="pt3", which="ARCHIVED",
                       model=None, sim=None,
                       label="real CMS tracker, pT=3 eta=0.30 phi=0.70, 200k "
                             "(archived radiation-on)"),
    "real400":    dict(kind="real", pt="pt3", which="real_base",
                       model=None, sim=None,
                       label="real CMS tracker, pT=3 eta=0.30 phi=0.70, 400k "
                             "(default production cuts)"),
    "real_cut01mm": dict(kind="real", pt="pt3", which="real_cut01mm",
                         model=None, sim=None,
                         label="real CMS tracker, pT=3, flat 0.01 cm cut, 400k"),
}

_S, _M = {}, {}


def model_of(tag):
    if tag not in _M:
        s = SAMPLES[tag]
        if s["kind"] == "real":
            import real_probe as rp
            _M[tag] = fn.load(rp.MODEL)
        else:
            _M[tag] = fn.load(s["model"])
    return _M[tag]


def sim_of(tag):
    if tag not in _S:
        s = SAMPLES[tag]
        if s["kind"] == "real":
            import real_probe as rp
            path = (rp.ARCHIVED_SIM if s["which"] == "ARCHIVED"
                    else rp.sim_glob(s["which"]))
            _S[tag] = cpt.load_sim(path, acceptance="perplane")
        else:
            import tail_probe as tp
            from toy_loader import load_toy_sim
            ns = tp._planes_ns(f"{s['pt']}_base")
            _S[tag] = load_toy_sim(s["sim"], ns["origin"], ns["normal"],
                                   ns["uaxis"])
    return _S[tag]


# ==========================================================================
# the state array and the increment decomposition
# ==========================================================================

def state_array(sim, legs, nplane=None):
    """(X, Xref, valid): X[e,k,5] local state, Xref[k,5] reference, in the
    order the a-vectors of `FUNCTIONALS` assume.

    `dydz` and `locy` are not in REF_BRANCH because no functional uses them;
    they are carried anyway so the 5D transport is the real one and not a
    2-of-5 projection of it.
    """
    n = min(sim["valid"].shape[1], len(legs))
    if nplane is not None:
        n = min(n, nplane)
    X = np.stack([sim[c][:, :n] for c in STATE], axis=2)
    Xref = np.array([[legs[k]["refqop"], legs[k]["refdxdz"], legs[k]["refdydz"],
                      legs[k]["reflocx"], legs[k]["reflocy"]] for k in range(n)])
    valid = sim["valid"][:, :n] & np.all(np.isfinite(X), axis=2)
    return X, Xref, valid


def transports(legs, n):
    """T[m, K] = F_K F_{K-1} ... F_{m+1} for 0 <= m <= K < n; T[K,K] = I."""
    T = np.zeros((n, n, 5, 5))
    for K in range(n):
        T[K, K] = np.eye(5)
        for m in range(K - 1, -1, -1):
            T[m, K] = T[m + 1, K] @ legs[m + 1]["F"]
    return T


def increments(X, Xref, legs, n):
    """D[e,m,5], the per-plane increment of the identity in the docstring."""
    R = X - Xref[None, :, :]
    D = np.empty_like(R)
    D[:, 0, :] = R[:, 0, :]
    for m in range(1, n):
        D[:, m, :] = R[:, m, :] - R[:, m - 1, :] @ legs[m]["F"].T
    return D


# ==========================================================================
# the sub-block model
#
# `_sub_legs(legs, jmin)` returns the SAME leg list with the physics records
# (ionization, MS, radiative, and the Gaussian Q/dQMS/dQI) of every leg below
# jmin emptied, and everything that defines the reference path -- F, stepjacc,
# refqop, refp -- untouched.  Every downstream consumer
# (`step_transports`, `model_phi`, `cgf_channels.block_cf_exponent`,
# `model_variance`, `fisher_norm.plane_scales`) then computes the sub-block
# quantity with no code change, because each of them guards on `len(leg[...])`.
#
# That the masking does exactly this is not assumed: `check` verifies
#     S_full(K) == S_{0..j0}(K) + S_{j0+1..}(K)
# to machine precision on the real exponents, which is the only property the
# rest of the module uses.
#
# These legs deliberately carry NO file provenance, so `fisher_norm`'s on-disk
# scale cache skips them (it logs that it does).  A sub-block must never be
# able to collide with a cached full-block 1/I.
# ==========================================================================

def _sub_legs(legs, jmin):
    if jmin <= 0:
        return legs
    out = []
    for j, leg in enumerate(legs):
        if j >= jmin:
            out.append(leg)
            continue
        d = dict(leg)
        d["ioni"] = np.zeros((0, leg["ioni"].shape[1] if leg["ioni"].size else 11))
        d["ms"] = np.zeros((0, leg["ms"].shape[1] if leg["ms"].size else 10))
        d["rad"] = None
        d["radspec"] = None
        d["radvgrid"] = None
        for q in ("Q", "dQMS", "dQI"):
            d[q] = np.zeros((5, 5))
        out.append(d)
    return out


_SUBSCALE = {}


def sub_scale(legs, jmin, K, func):
    """(sigma, 1/I, s_F) of the increment block (legs jmin..K) at plane K.

    Same pipeline as `fisher_norm.plane_scales`: sigma from the propagator's
    own Q, 1/I from the exact FFT inversion of the block CF on a grid matched
    to THAT block.  Matching the grid to the thing being inverted is the rule
    this study keeps relearning, and a sub-block's CF decays at a different t
    from the full block's, so `auto_tau` is re-run rather than reused.
    """
    key = (id(legs), int(jmin), int(K), func)
    if key not in _SUBSCALE:
        import cgf_channels as cc
        sub = _sub_legs(legs, jmin)
        a = FUNCTIONALS[func]
        sig = float(np.sqrt(cpt.model_variance(sub, K, a)[0]))
        z, p, dp = cc.exact_density(sub, K, a, sig, channels=("ioni", "ms", "rad"))
        I, _ = cc.fisher_exact(z, p, dp, floor=1e-8)
        _SUBSCALE[key] = (sig, 1.0 / I, sig * np.sqrt(1.0 / I))
    return _SUBSCALE[key]


def sub_model_probe(legs, jmin, K, func, s, probes=UCURVE):
    """<e^{-u d^2}>_model for the increment block, d standardized by `s`."""
    sub = _sub_legs(legs, jmin)
    tau = fn.closure_tau(float(np.max(probes)))
    phi = cpt.model_phi(sub, K, FUNCTIONALS[func], float(s), tau)
    return np.array([cpt.weier_scalar(phi, u, tau) for u in probes])


# ==========================================================================
# errors
#
# The plane mean's error is NOT err_plane/sqrt(nplane): every plane is
# evaluated on the same events (NOTES_GEOMCLOSURE, "The error bar, and why the
# old one was wrong" -- the naive form is 2.8x too small).  The estimator is
# the error of the per-EVENT plane average, which carries the full covariance.
# ==========================================================================

def per_event_mean_err(E, mask):
    """(mean over planes, its correlated error) from E[u, ev, plane].

    `mask[ev, plane]` says where the event is usable.  Each plane's term is
    reweighted by N/n_k so it stays an unbiased estimate of that plane's own
    mean when some events are missing, exactly as `geom_closure.closure_rows`.
    """
    nu, nev, nk = E.shape
    acc = np.zeros((nu, nev))
    nkeep = 0
    for k in range(nk):
        m = mask[:, k]
        n = int(m.sum())
        if n < 100:
            continue
        acc[:, m] += E[:, m, k] * (nev / n)
        nkeep += 1
    if nkeep == 0:
        return np.full(nu, np.nan), np.full(nu, np.nan)
    acc /= nkeep
    return acc.mean(axis=1), acc.std(axis=1) / np.sqrt(nev)


def fmt(vals, prec=5, w=10, sign=True):
    f = "+" if sign else ""
    return "".join(f"{v:{f}{w}.{prec}f}" for v in vals)


def hdr(probes=UCURVE, w=10):
    return "".join(f"{u:>{w}.3g}" for u in probes)


# ==========================================================================
# subcommand: check -- every premise, before any number is interpreted
# ==========================================================================

def cmd_check(args):
    import cgf_channels as cc
    print("=" * 100)
    print("PREMISE CHECKS")
    print("=" * 100)
    for tag in args.tags:
        legs, sim = model_of(tag), sim_of(tag)
        n = min(sim["valid"].shape[1], len(legs))
        X, Xref, valid = state_array(sim, legs)
        D = increments(X, Xref, legs, n)
        T = transports(legs, n)
        print(f"\n### {tag}  --  {SAMPLES[tag]['label']}")
        print(f"  {n} planes, {X.shape[0]} events")

        # ---- 1. the telescoping identity, event by event ------------------
        print("\n  1. telescoping identity   x_K - xref_K == sum_m T_{m->K} D_m")
        print(f"     {'K':>3} {'func':>5} {'max |lhs-rhs|':>15} {'/ rms(resid)':>14}")
        for K in (n // 2, n - 1):
            ok = np.all(valid[:, :K + 1], axis=1)
            for func in ("qop", "locx"):
                a = FUNCTIONALS[func]
                lhs = (X[ok, K, :] - Xref[K]) @ a
                rhs = np.einsum("i,mij,emj->e", a, T[:K + 1, K], D[ok, :K + 1, :])
                err = np.max(np.abs(lhs - rhs))
                print(f"     {K:>3} {func:>5} {err:15.3e} {err/np.std(lhs):14.3e}")

        # ---- 2. the leg-masking identity on the CF exponent ----------------
        print("\n  2. leg masking   S_full(K) == S_{0..j0}(K) + S_{j0+1..}(K)")
        K = n - 1
        j0 = n // 2
        for func in ("qop", "locx"):
            a = FUNCTIONALS[func]
            sig = float(np.sqrt(cpt.model_variance(legs, K, a)[0]))
            tau = fn.closure_tau(10.0)
            Sf = cc.block_cf_exponent(legs, K, a, sig, tau)
            # "legs 0..j0" is the complement mask: keep the low legs, empty the
            # high ones.  Built by the same routine applied to a reversed
            # selection so that only one masking implementation is on trial.
            lo = _sub_legs_only(legs, 0, j0)
            hi = _sub_legs(legs, j0 + 1)
            Sl = cc.block_cf_exponent(lo, K, a, sig, tau)
            Sh = cc.block_cf_exponent(hi, K, a, sig, tau)
            d = np.max(np.abs(Sf - (Sl + Sh)))
            print(f"     {func:>5}  max|dS| = {d:.3e}   (|S| up to {np.max(np.abs(Sf)):.3e})")

        # ---- 3. what the transport already contains -----------------------
        print("\n  3. the transport F already carries the LINEAR mean feedback")
        f00 = np.array([legs[j]["F"][0, 0] for j in range(1, n)])
        print(f"     F[0,0] per leg: {f00.min():.6f} .. {f00.max():.6f}, "
              f"product over legs 1..{n-1} = {np.prod(f00):.6f}")
        row = np.abs(np.array([legs[j]["F"][0, 1:] for j in range(1, n)])).max(axis=0)
        sx = np.nanstd(X[:, :, 3])
        print(f"     max |F[0,1:]| = {np.array2string(row, precision=2)}; "
              f"times rms(locx)={sx:.4f} cm -> {row[2]*sx:.2e} in qop, "
              f"i.e. {row[2]*sx/np.nanstd(X[:,n-1,0]-Xref[n-1,0]):.1e} of the qop residual")

        # ---- 4. acceptance ------------------------------------------------
        print("\n  4. acceptance per plane")
        v = valid.sum(axis=0) / valid.shape[0]
        print(f"     min {v.min()*100:.3f} %  max {v.max()*100:.3f} %  "
              f"all-plane {np.all(valid,axis=1).mean()*100:.3f} %")
    print()


def _sub_legs_only(legs, jlo, jhi):
    """Complement of `_sub_legs`: keep the physics of legs jlo..jhi, empty the
    rest.  Used ONLY by `check`, so that the masking used everywhere else is
    tested against an independently written mask rather than against itself."""
    out = []
    for j, leg in enumerate(legs):
        if jlo <= j <= jhi:
            out.append(leg)
            continue
        d = dict(leg)
        d["ioni"] = np.zeros((0, leg["ioni"].shape[1] if leg["ioni"].size else 11))
        d["ms"] = np.zeros((0, leg["ms"].shape[1] if leg["ms"].size else 10))
        d["rad"] = d["radspec"] = d["radvgrid"] = None
        for q in ("Q", "dQMS", "dQI"):
            d[q] = np.zeros((5, 5))
        out.append(d)
    return out


# ==========================================================================
# subcommand: corr -- the direct measurement, no model anywhere
# ==========================================================================

def _rob(x, axis=0):
    """Half the 16-84 inter-quantile range: the width estimator that survives
    a distribution whose rms is set by two events in 400 000 (measured on the
    pT = 3 toy: rms(D_m) = 5.5e-4 against rob = 2.6e-5, a factor 21, and the
    Pearson correlation of D_6 with D_13 is 0.76 because ONE event lost 1.8 GeV
    between planes 5 and 6 and another 1.0 GeV between 12 and 13)."""
    q = np.quantile(x, [0.16, 0.84], axis=axis)
    return 0.5 * (q[1] - q[0])


def cmd_corr(args):
    from scipy.stats import rankdata
    print("=" * 110)
    print("DIRECT STEP-TO-STEP CORRELATION   Corr(D_i, D_j),  ZERO UNDER THE MODEL")
    print("=" * 110)
    print("D_m = (x_m - xref_m) - F_m (x_{m-1} - xref_{m-1}) is the residue of the\n"
          "linear transport, i.e. EXACTLY the object the block CF adds as an\n"
          "independent per-leg noise.  Nothing below uses the model CF.\n"
          "\n"
          "D_m is a compound-Poisson variable with a 1/T^2 tail, so its PEARSON\n"
          "correlation is an estimator of the tail and not of the dependence:\n"
          "one 1.8 GeV catastrophic event in 400k sets Corr(D_6,D_13) = 0.76.\n"
          "Three estimators are therefore quoted -- Pearson (reported, not\n"
          "interpreted), Spearman rank, and the correlation of the BOUNDED\n"
          "functional exp(-u D^2) that the closure actually integrates.\n")
    rng = np.random.default_rng(args.seed)
    for tag in args.tags:
        legs, sim = model_of(tag), sim_of(tag)
        n = min(sim["valid"].shape[1], len(legs))
        X, Xref, valid = state_array(sim, legs)
        ok = np.all(valid, axis=1)
        D = increments(X[ok], Xref, legs, n)
        for func in args.funcs:
            a = FUNCTIONALS[func]
            d = D @ a                                    # (nev, nplane)
            nev = d.shape[0]
            off = ~np.eye(n, dtype=bool)
            print(f"\n### {tag}   {func}   {nev} events on all {n} planes")
            print(f"  per-plane widths: rob(16-84) {_rob(d).min():.3e}"
                  f"..{_rob(d).max():.3e}   rms {d.std(axis=0).min():.3e}"
                  f"..{d.std(axis=0).max():.3e}   "
                  f"rms/rob up to {np.max(d.std(axis=0)/_rob(d)):.1f}")

            C = np.corrcoef(d.T)
            print(f"\n  [1] Pearson    off-diag mean {C[off].mean():+.5f}  "
                  f"min {C[off].min():+.5f}  max {C[off].max():+.5f}   "
                  f"(tail-dominated, see above)")

            r = np.apply_along_axis(rankdata, 0, d) / nev
            Cs = np.corrcoef(r.T)
            boots = np.empty((args.nboot, n, n))
            for b in range(args.nboot):
                idx = rng.integers(0, nev, nev)
                boots[b] = np.corrcoef(r[idx].T)
            Es = boots.std(axis=0)
            pull = Cs[off] / np.where(Es[off] > 0, Es[off], np.nan)
            print(f"  [2] Spearman   off-diag mean {Cs[off].mean():+.5f}  "
                  f"min {Cs[off].min():+.5f}  max {Cs[off].max():+.5f}   "
                  f"typ. error {np.median(Es[off]):.5f}")
            print(f"      pulls  |pull| median {np.nanmedian(np.abs(pull)):.2f}"
                  f"  max {np.nanmax(np.abs(pull)):.2f}"
                  f"  n(|pull|>3) = {int(np.nansum(np.abs(pull) > 3))}"
                  f" of {off.sum()}   (chance at 3 sigma: {0.0027*off.sum():.1f})")
            if args.verbose:
                print("      Spearman matrix (x1e3)")
                print("       " + "".join(f"{j:>7d}" for j in range(n)))
                for i in range(n):
                    print(f"    {i:>3}" + "".join(
                        f"{1e3*Cs[i, j]:>7.2f}" if i != j else f"{'--':>7}"
                        for j in range(n)))

            # [3] the bounded functional the closure actually integrates
            sF = _rob(d)
            print("\n  [3] Corr(exp(-u D_i^2), exp(-u D_j^2)) -- the object the "
                  "closure integrates, D standardized by rob")
            print(f"      {'u':>8} {'mean':>10} {'min':>10} {'max':>10} "
                  f"{'error':>10} {'max|pull|':>10}")
            for u in (0.01, 0.1, 1.0, 10.0):
                g = np.exp(-u * (d / sF) ** 2)
                Cg = np.corrcoef(g.T)
                bb = np.empty((min(args.nboot, 40), n, n))
                for b in range(bb.shape[0]):
                    idx = rng.integers(0, nev, nev)
                    bb[b] = np.corrcoef(g[idx].T)
                Eg = bb.std(axis=0)
                pl = Cg[off] / np.where(Eg[off] > 0, Eg[off], np.nan)
                print(f"      {u:>8.2g} {Cg[off].mean():+10.5f} "
                      f"{Cg[off].min():+10.5f} {Cg[off].max():+10.5f} "
                      f"{np.median(Eg[off]):10.5f} "
                      f"{np.nanmax(np.abs(pl)):10.2f}")

            # ---- the physically readable form: conditional moments --------
            k = args.kcond if args.kcond is not None else n // 2
            dp = 1e3 * (sim["pabs"][ok, k] - legs[k]["refp"])   # MeV, signed
            qs = np.quantile(dp, [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
            bins = _qbins(dp, qs)
            js = list(range(k + 1, n))
            base_m = np.median(d[:, js], axis=0)
            base_s = _rob(d[:, js])
            print(f"\n  [4] conditional moments of the LATER increments, binned "
                  f"on the realised momentum\n      deficit at plane k = {k} "
                  f"(dp = p_real - p_ref, MeV).  Widths are rob(16-84), so a\n"
                  f"      handful of catastrophic events cannot set them.")
            print("\n      [median(D_j|bin) - median(D_j)] / rob(D_j)"
                  "   -- ZERO in every bin under the model:")
            print(f"      {'bin [MeV]':>18} {'n':>8} {'<dp>':>9}" +
                  "".join(f"{'D%d' % j:>9}" for j in js))
            for lab, m in bins:
                print(f"      {lab:>18} {int(m.sum()):>8} {dp[m].mean():+9.2f}" +
                      "".join(f"{v:+9.4f}" for v in
                              (np.median(d[m][:, js], axis=0) - base_m) / base_s))
            print("\n      rob(D_j|bin) / rob(D_j)"
                  "   -- ONE in every bin under the model:")
            print(f"      {'bin [MeV]':>18} {'n':>8} {'<dp>':>9}" +
                  "".join(f"{'D%d' % j:>9}" for j in js))
            for lab, m in bins:
                print(f"      {lab:>18} {int(m.sum()):>8} {dp[m].mean():+9.2f}" +
                      "".join(f"{v:9.4f}" for v in
                              _rob(d[m][:, js]) / base_s))
            print("\n      P(|D_j| > 5 rob | bin) / P(|D_j| > 5 rob)"
                  "   -- the TAIL rate, ONE under the model:")
            print(f"      {'bin [MeV]':>18} {'n':>8} {'<dp>':>9}" +
                  "".join(f"{'D%d' % j:>9}" for j in js))
            ref = (np.abs(d[:, js]) > 5 * base_s).mean(axis=0)
            for lab, m in bins:
                v = (np.abs(d[m][:, js]) > 5 * base_s).mean(axis=0) / ref
                e = np.sqrt((np.abs(d[m][:, js]) > 5 * base_s).mean(axis=0)
                            / max(int(m.sum()), 1)) / ref
                print(f"      {lab:>18} {int(m.sum()):>8} {dp[m].mean():+9.2f}" +
                      "".join(f"{a_:9.4f}" for a_ in v))
                print(f"      {'   +-':>18} {'':>8} {'':>9}" +
                      "".join(f"{a_:9.4f}" for a_ in e))
    print()


def _qbins(c, edges):
    """[(label, mask)] for the quantile edges `edges`, last bin closed."""
    out = []
    for b in range(len(edges) - 1):
        if b == len(edges) - 2:
            m = (c >= edges[b]) & (c <= edges[b + 1])
        else:
            m = (c >= edges[b]) & (c < edges[b + 1])
        out.append((f"[{edges[b]:+.1f},{edges[b+1]:+.1f})", m))
    return out


# ==========================================================================
# subcommand: cond -- THE PRIMARY TEST
#
# Bin on the realised accumulated loss at plane k; measure the closure of the
# REMAINING increment over planes k+1..N in each bin.
#
# The comparison across bins is like-for-like because the MODEL half is
# literally the same number in every bin: the block CF of legs k+1..j and its
# Fisher scale are functions of the deterministic reference trajectory alone
# and know nothing about the conditioning.  Conditioning selects on momentum
# and hence on Tmax and on 1/I *of the truth*, which is the effect; it cannot
# move the model, so the SPAN across bins is a pure data statement.  (Had each
# bin instead been standardized by its own empirical width, the effect would
# have been divided out -- that is the naive version the task warns about.)
# ==========================================================================

_COND_CTX = None


def _cond_model_one(j):
    legs, kc, func, probes = _COND_CTX
    s = sub_scale(legs, kc + 1, j, func)[2]
    return s, sub_model_probe(legs, kc + 1, j, func, s, probes)


def cond_one(tag, kc, func, edges, probes=UCURVE, condvar="dp", nperm=0,
             seed=0):
    """The conditional-subsample closure at conditioning plane `kc`.

    Returns dict with per-bin plane-mean closure curves and their correlated
    errors, the per-plane detail, and the bin bookkeeping.
    """
    legs, sim = model_of(tag), sim_of(tag)
    n = min(sim["valid"].shape[1], len(legs))
    X, Xref, valid = state_array(sim, legs)
    T = transports(legs, n)
    a = FUNCTIONALS[func]
    js = list(range(kc + 1, n))

    global _COND_CTX
    _COND_CTX = (legs, kc, func, probes)
    got = fn.pmap(_cond_model_one, js)
    sF = {j: g[0] for j, g in zip(js, got)}
    mod = {j: g[1] for j, g in zip(js, got)}

    # the conditioning variable, and the events that have it
    have_k = valid[:, kc]
    if condvar == "dp":
        c = 1e3 * (sim["pabs"][:, kc] - legs[kc]["refp"])       # MeV
        cname = "dp = p_real - p_ref [MeV]"
    elif condvar == "dxdz":            # CONTROL: an MS variable, not a loss
        c = (X[:, kc, 1] - Xref[kc, 1]) / np.nanstd(X[have_k, kc, 1])
        cname = "dxdz residual / rms  (CONTROL)"
    else:
        raise ValueError(condvar)
    if nperm:
        # CONTROL: destroy the conditioning while keeping the bin sizes.
        # Permuted WITHIN the have_k subset -- permuting the whole array would
        # scatter the NaNs of the events that never crossed plane kc onto
        # events that did, and the quantiles would come back NaN.
        rng = np.random.default_rng(seed + nperm)
        idx = np.flatnonzero(have_k)
        c = c.copy()
        c[idx] = c[idx][rng.permutation(len(idx))]
        cname += "  [LABELS PERMUTED]"

    qs = np.quantile(c[have_k], edges)
    qs[0], qs[-1] = -np.inf, np.inf
    bins = _qbins(c, qs)

    # the standardized increment, per target plane
    nev = X.shape[0]
    Z = np.full((nev, len(js)), np.nan)
    for i, j in enumerate(js):
        w = T[kc, j].T @ a
        d = (X[:, j, :] - Xref[j]) @ a - (X[:, kc, :] - Xref[kc]) @ w
        Z[:, i] = d / sF[j]
    okj = valid[:, js] & have_k[:, None] & np.isfinite(Z)

    E = np.exp(-np.asarray(probes)[:, None, None] * np.nan_to_num(Z)[None] ** 2)
    E[:, ~okj] = 0.0
    M = np.array([mod[j] for j in js]).T                      # (nu, nplane)

    out = dict(tag=tag, kc=kc, func=func, js=js, sF=sF, model=M, probes=probes,
               cname=cname, bins=[], n=nev)
    for lab, m in bins:
        mm = m & have_k
        dm, de = per_event_mean_err(E[:, mm, :], okj[mm])
        # the model's plane mean over the same planes that were kept
        keep = [i for i in range(len(js)) if okj[mm][:, i].sum() >= 100]
        mmod = M[:, keep].mean(axis=1)
        perpl = np.array([E[:, mm, i][:, okj[mm][:, i]].mean(axis=1) - M[:, i]
                          for i in keep]).T
        out["bins"].append(dict(label=lab, n=int(mm.sum()),
                                cmean=float(np.mean(c[mm])),
                                data=dm, err=de, model=mmod,
                                closure=dm - mmod, perplane=perpl,
                                keep=[js[i] for i in keep]))
    return out


def cmd_cond(args):
    edges = [float(x) for x in args.edges.split(",")]
    print("=" * 110)
    print("CONDITIONAL-SUBSAMPLE CLOSURE  --  THE PRIMARY TEST")
    print("=" * 110)
    print("Bin on the realised accumulated loss at plane k; close over the "
          "REMAINING planes.\nThe model half is bin-independent by "
          "construction, so the SPAN across bins is the effect.\n")
    for tag in args.tags:
        legs, sim = model_of(tag), sim_of(tag)
        n = min(sim["valid"].shape[1], len(legs))
        ks = args.k if args.k else [max(1, n // 4), n // 2, (3 * n) // 4]
        for func in args.funcs:
            for kc in ks:
                if kc >= n - 1:
                    continue
                r = cond_one(tag, kc, func, edges, condvar=args.condvar,
                             nperm=args.permute, seed=args.seed)
                _print_cond(r, tag, n)
    print()


def _print_cond(r, tag, n):
    kc, func, probes = r["kc"], r["func"], r["probes"]
    print(f"### {tag}   {func}   k = {kc}  ->  closure over planes "
          f"{r['js'][0]}..{r['js'][-1]}   [{r['cname']}]")
    print(f"  {'bin':>20} {'n':>8} {'<c>':>9}" + hdr(probes) + f"{'rms':>10}")
    print(f"  {'MODEL <e^-uz2>':>20} {'':>8} {'':>9}" + fmt(r["model"].mean(axis=1) if r["model"].ndim > 1 else r["model"], sign=False))
    cl = []
    for b in r["bins"]:
        cl.append(b["closure"])
        print(f"  {b['label']:>20} {b['n']:>8} {b['cmean']:+9.2f}"
              + fmt(b["closure"]) + f"{np.sqrt(np.mean(b['closure']**2)):10.5f}")
        print(f"  {'   +-':>20} {'':>8} {'':>9}" + fmt(b["err"], sign=False))
    cl = np.array(cl)
    errs = np.array([b["err"] for b in r["bins"]])
    span = cl.max(axis=0) - cl.min(axis=0)
    espan = np.sqrt(errs[np.argmax(cl, axis=0), np.arange(cl.shape[1])] ** 2
                    + errs[np.argmin(cl, axis=0), np.arange(cl.shape[1])] ** 2)
    print(f"  {'SPAN (max-min)':>20} {'':>8} {'':>9}" + fmt(span, sign=False))
    print(f"  {'   +-':>20} {'':>8} {'':>9}" + fmt(espan, sign=False))
    print(f"  {'   pull':>20} {'':>8} {'':>9}"
          + fmt(span / espan, prec=2, sign=False))
    # PRE-SPECIFIED contrasts.  max-min over eight bins has an inflated
    # expectation under the null -- the label-permutation control returns
    # spans whose "pull" is 1.0-2.6 by construction -- so the two contrasts
    # that were fixed before looking are quoted alongside it.  They are
    # ordinary two-sample differences on disjoint event sets and their pulls
    # mean what they say.
    d1 = cl[0] - cl[-1]
    e1 = np.sqrt(errs[0] ** 2 + errs[-1] ** 2)
    print(f"  {'lowest - highest':>20} {'':>8} {'':>9}" + fmt(d1))
    print(f"  {'   pull':>20} {'':>8} {'':>9}" + fmt(d1 / e1, prec=2))
    w = np.array([b["n"] for b in r["bins"]], dtype=float)
    rest = (cl[1:] * w[1:, None]).sum(axis=0) / w[1:].sum()
    erest = np.sqrt((errs[1:] ** 2 * w[1:, None] ** 2).sum(axis=0)) / w[1:].sum()
    d2 = cl[0] - rest
    e2 = np.sqrt(errs[0] ** 2 + erest ** 2)
    print(f"  {'lowest - all others':>20} {'':>8} {'':>9}" + fmt(d2))
    print(f"  {'   pull':>20} {'':>8} {'':>9}" + fmt(d2 / e2, prec=2))
    print()


# ==========================================================================
# subcommand: shuffle -- the model-free size of the whole effect
#
# The surrogate replaces the joint law of (D_0 .. D_K) by the PRODUCT of its
# own marginals, event by event, using nothing but a permutation.  Marginals
# are preserved EXACTLY (each D_m is the same multiset of values), so the
# difference between the real and surrogate closure functionals is entirely
# the cross-plane dependence -- no CF, no inversion, no fitted parameter, and
# no sensitivity to the normalization, which cancels in the difference because
# both halves use the same s_F.
#
# `gsize` groups consecutive planes and permutes the GROUP sums, keeping the
# correlation inside a group.  gsize = 1 removes every cross-plane
# correlation; gsize >= nplane removes none and must return exactly zero,
# which is the built-in null of the machinery.
# ==========================================================================

def _groups(nplane, gsize=None, split=None):
    if split is not None:
        return [list(range(0, split + 1)), list(range(split + 1, nplane))]
    g = int(gsize or 1)
    return [list(range(i, min(i + g, nplane))) for i in range(0, nplane, g)]


def _perm_within(s, key, rng):
    """`s` permuted WITHIN each stratum of `key`, without replacement.

    Both index lists enumerate the events stratum by stratum in the same
    stratum order -- `argsort(key)` in the original within-stratum order,
    `lexsort((random, key))` in a random one -- so the assignment moves values
    only between events that share a stratum.
    """
    dst = np.argsort(key, kind="stable")
    src = np.lexsort((rng.random(len(s)), key))
    out = np.empty_like(s)
    out[dst] = s[src]
    return out


def _surrogate(contrib, groups, rng, strata=None):
    """Sum of the group contributions with each GROUP independently permuted.

    With `strata` (an int code per event and per plane) the permutation is
    restricted to events in the same stratum, so any dependence MEDIATED BY
    the stratum variable survives into the surrogate and only the rest is
    destroyed.  Stratifying on the crossing position is how the geometric
    channel (multiple scattering displaces the track -> it meets a different
    amount of material -> its later energy loss changes) is held fixed while
    the momentum-feedback channel is removed.
    """
    n = contrib.shape[0]
    z = np.zeros(n)
    for g in groups:
        g = [m for m in g if m < contrib.shape[1]]
        if not g:
            continue
        s = contrib[:, g].sum(axis=1)
        if strata is None:
            z += s[rng.permutation(n)]
        else:
            z += _perm_within(s, strata[:, g[0]], rng)
    return z


def _strata_codes(X, Xref, valid, mode, nbin=8):
    """Integer stratum code per (event, plane) from the crossing position.

    `locy`  -- the NON-BENDING local coordinate only.  Preferred, because it is
               the one the material inhomogeneity is a function of (measured:
               Spearman(D_17, locy_17) = +0.92 on the real tracker) AND it is
               not moved by the momentum: a track that lost energy bends in
               locx, not in locy, so stratifying on locy cannot absorb any of
               the momentum-feedback channel.
    `locxy` -- a 2D nbin x nbin quantile grid in (locx, locy).  Catches the
               module-edge structure in the bending direction as well
               (Spearman(D_18, |locx_18|) = -0.52) but DOES absorb part of the
               momentum channel, because a 0.7 % momentum change moves locx by
               ~1/3 of its residual width at r = 95 cm.
    """
    nev, nplane, _ = X.shape
    cols = {"locy": (4,), "locxy": (3, 4)}[mode]
    code = np.zeros((nev, nplane), dtype=np.int64)
    for m in range(nplane):
        v = valid[:, m]
        for c in cols:
            x = X[:, m, c]
            q = np.quantile(x[v], np.linspace(0, 1, nbin + 1)[1:-1])
            code[:, m] = code[:, m] * nbin + np.searchsorted(q, x)
    return code


_SH_CTX = None


def _shuffle_one_plane(K):
    (X, Xref, D, T, valid, a, sF, legs, probes, gsize, split, nboot,
     seed, bidx, codes) = _SH_CTX
    ok = np.all(valid[:, :K + 1], axis=1)
    nev = int(ok.sum())
    if nev < 1000:
        return None
    contrib = np.einsum("i,mij,emj->em", a, T[:K + 1, K], D[ok, :K + 1, :]) / sF[K]
    st = None if codes is None else codes[ok]
    z = contrib.sum(axis=1)
    groups = _groups(K + 1, gsize, split)
    u = np.asarray(probes)[:, None]
    Ereal = np.exp(-u * z[None] ** 2).mean(axis=1)
    rng = np.random.default_rng(seed + 1000 * K)
    nperm = 8
    zs = [_surrogate(contrib, groups, rng, st) for _ in range(nperm)]
    surr = np.array([np.exp(-u * s[None] ** 2).mean(axis=1) for s in zs])
    # Bootstrap the DIFFERENCE with a fresh permutation AND a fresh event
    # resample per replicate.  The replicate index sets `bidx` are drawn ONCE
    # in the parent and shared by every plane, so the per-replicate PLANE MEAN
    # is a resample of whole EVENTS -- which is what the correlated error bar
    # requires (planes share events; the naive per-plane/sqrt(nplane) form is
    # 2.8x too small, NOTES_GEOMCLOSURE).
    #
    # `bidx` indexes the full event list; each plane keeps the drawn events
    # that are usable on it, so a plane with lower acceptance simply gets a
    # smaller replicate rather than a different set of events.
    pos = np.full(len(ok), -1)
    pos[ok] = np.arange(nev)
    dif = np.zeros((max(nboot, 1), len(probes)))
    for b in range(nboot):
        sel = pos[bidx[b]]
        sel = sel[sel >= 0]
        cb = contrib[sel]
        dif[b] = (np.exp(-u * cb.sum(axis=1)[None] ** 2).mean(axis=1)
                  - np.exp(-u * _surrogate(cb, groups, rng,
                                            None if st is None else st[sel]
                                            )[None] ** 2).mean(axis=1))
    return dict(K=K, n=nev, real=Ereal, surr=surr.mean(axis=0),
                surr_sd=surr.std(axis=0), delta=Ereal - surr.mean(axis=0),
                boot=dif, ngroup=len(groups),
                rob_real=_rob(z), rob_surr=float(np.mean([_rob(s) for s in zs])),
                rms_real=float(z.std()),
                rms_surr=float(np.mean([s.std() for s in zs])),
                delta_err=dif.std(axis=0) if nboot else np.zeros(len(probes)))


def inject_feedback(D, a, lam, rob):
    """Multiply every leg's noise by (1 + lam * u), u = the standardized
    realised loss accumulated BEFORE that leg.

    This is the mechanism under test, put in by hand with a knob: the
    straggling of leg m is made to depend on what the track has already lost.
    It exists so that the `shuffle` null can be given a SENSITIVITY, i.e. so
    that "no effect" can be separated from "no sensitivity".  `u` is clipped at
    +-3 so that the two catastrophic events in 400 000 (one loses 1.8 GeV
    between planes 5 and 6) cannot define the injection.

    The sign convention is the physical one: u > 0 means the track has lost
    MORE than the reference, and lam > 0 then widens the later fluctuations,
    which is what the conversion factor cs = E/p^3 does when p falls.
    """
    if not lam:
        return D
    D = D.copy()
    nplane = D.shape[1]
    acc = np.zeros(len(D))
    for m in range(nplane):
        if m:
            u = np.clip(-acc / rob[m - 1], -3.0, 3.0)
            D[:, m, :] *= (1.0 + lam * u)[:, None]
        acc = acc + D[:, m, :] @ a          # unweighted: F ~ 1 for qop
    return D


def shuffle_all(tag, func, probes=UCURVE, gsize=1, split=None, nboot=40,
                seed=0, nplane=None, strat=None, nstrat=8, inject=0.0):
    legs, sim = model_of(tag), sim_of(tag)
    n = min(sim["valid"].shape[1], len(legs))
    if nplane:
        n = min(n, nplane)
    X, Xref, valid = state_array(sim, legs, nplane=n)
    D = increments(X, Xref, legs, n)
    T = transports(legs, n)
    a = FUNCTIONALS[func]
    if inject:
        ok0 = np.all(valid, axis=1)
        acc = np.cumsum((D[ok0] @ a), axis=1)
        rob = np.array([_rob(acc[:, m]) for m in range(n)])
        D = inject_feedback(D, a, inject, rob)
    mp = (SAMPLES[tag]["model"] if SAMPLES[tag]["kind"] == "toy"
          else __import__("real_probe").MODEL)
    sc = fn.plane_scales(legs, func, tag=mp, nplane=n)
    nev = X.shape[0]
    brng = np.random.default_rng(seed + 7717)
    bidx = brng.integers(0, nev, (max(nboot, 1), nev))
    codes = (None if not strat
             else _strata_codes(X, Xref, valid, strat, nstrat))
    global _SH_CTX
    _SH_CTX = (X, Xref, D, T, valid, a, sc["sF"], legs, probes, gsize, split,
               nboot, seed, bidx, codes)
    res = [r for r in fn.pmap(_shuffle_one_plane, range(n)) if r is not None]
    # the model half, once per plane, for the absolute closure
    tau = fn.closure_tau(float(np.max(probes)))
    global _MOD_CTX
    _MOD_CTX = (legs, func, sc["sF"], probes, tau)
    mods = dict(zip([r["K"] for r in res],
                    fn.pmap(_full_model_one, [r["K"] for r in res])))
    for r in res:
        r["model"] = mods[r["K"]]
        r["clos_real"] = r["real"] - r["model"]
        r["clos_surr"] = r["surr"] - r["model"]
    return res, sc


_MOD_CTX = None


def _full_model_one(K):
    legs, func, sF, probes, tau = _MOD_CTX
    phi = cpt.model_phi(legs, K, FUNCTIONALS[func], float(sF[K]), tau)
    return np.array([cpt.weier_scalar(phi, u, tau) for u in probes])


def cmd_shuffle(args):
    print("=" * 110)
    print("REAL SAMPLE vs INDEPENDENT-INCREMENT SURROGATE")
    print("=" * 110)
    print("The surrogate has the SAME per-plane marginals and zero cross-plane\n"
          "correlation.  delta = real - surrogate is the entire step-to-step\n"
          "correlation contribution to <e^{-u z^2}>; the model half cancels in it.\n")
    for tag in args.tags:
        for func in args.funcs:
            res, sc = shuffle_all(tag, func, gsize=args.gsize, split=args.split,
                                  nboot=args.nboot, seed=args.seed,
                                  nplane=args.nplane, strat=args.strat,
                                  nstrat=args.nstrat, inject=args.inject)
            g = (f"split at {args.split}" if args.split is not None
                 else f"group size {args.gsize}")
            if args.strat:
                g += f", stratified on {args.strat} ({args.nstrat} bins/dim)"
            if args.inject:
                g += f", INJECTED feedback lambda = {args.inject}"
            print(f"### {tag}   {func}   [{g}]   {SAMPLES[tag]['label']}")
            print("  widths of the standardized residual, real vs surrogate "
                  "(the shuffle IS doing something):")
            print(f"  {'K':>4} {'ngroup':>7} {'rob real':>11} {'rob surr':>11}"
                  f" {'ratio':>9} {'rms real':>11} {'rms surr':>11} {'ratio':>9}")
            for r in res:
                print(f"  {r['K']:>4} {r['ngroup']:>7} {r['rob_real']:11.6f}"
                      f" {r['rob_surr']:11.6f} "
                      f"{r['rob_real']/r['rob_surr']:9.5f}"
                      f" {r['rms_real']:11.4f} {r['rms_surr']:11.4f}"
                      f" {r['rms_real']/r['rms_surr']:9.5f}")
            print()
            print(f"  {'K':>3}{'n':>9}  {'quantity':<14}" + hdr())
            for r in res:
                print(f"  {r['K']:>3}{r['n']:>9}  {'closure real':<14}"
                      + fmt(r["clos_real"]))
                print(f"  {'':>3}{'':>9}  {'closure surr':<14}"
                      + fmt(r["clos_surr"]))
                print(f"  {'':>3}{'':>9}  {'delta':<14}" + fmt(r["delta"]))
                print(f"  {'':>3}{'':>9}  {'   +-':<14}"
                      + fmt(r["delta_err"], sign=False))
            R = np.array([r["clos_real"] for r in res])
            S = np.array([r["clos_surr"] for r in res])
            Dl = np.array([r["delta"] for r in res])
            # the plane MEAN's error, from the shared per-event bootstrap:
            # each replicate is one resample of whole events applied to every
            # plane, so the plane mean of a replicate carries the full
            # plane-to-plane covariance.
            B = np.array([r["boot"] for r in res])           # (nplane, nboot, nu)
            emean = B.mean(axis=0).std(axis=0)
            print(f"  {'':>3}{'':>9}  {'-' * 100}")
            print(f"  {'':>3}{'':>9}  {'MEAN real':<14}" + fmt(R.mean(axis=0))
                  + f"   rms {np.sqrt(np.mean(R.mean(axis=0)**2)):.5f}")
            print(f"  {'':>3}{'':>9}  {'MEAN surr':<14}" + fmt(S.mean(axis=0))
                  + f"   rms {np.sqrt(np.mean(S.mean(axis=0)**2)):.5f}")
            print(f"  {'':>3}{'':>9}  {'MEAN delta':<14}" + fmt(Dl.mean(axis=0)))
            print(f"  {'':>3}{'':>9}  {'   +- (corr)':<14}"
                  + fmt(emean, sign=False))
            print(f"  {'':>3}{'':>9}  {'   pull':<14}"
                  + fmt(Dl.mean(axis=0) / emean, prec=2))
            print(f"  {'':>3}{'':>9}  {'delta/real [%]':<14}"
                  + fmt(100 * Dl.mean(axis=0) / R.mean(axis=0), prec=2))
            print()


# ==========================================================================
# driver
# ==========================================================================

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)

    def common(q, default_funcs=("qop",)):
        # No `choices=`: argparse validates the whole DEFAULT LIST against
        # `choices` when nargs="*" consumes nothing, so an empty default is
        # itself reported as an invalid choice. Validated by hand below.
        q.add_argument("tags", nargs="*", default=[])
        q.add_argument("--funcs", nargs="+", default=list(default_funcs))
        q.add_argument("--seed", type=int, default=0)
        return q

    q = s.add_parser("check", help="premise checks")
    common(q)
    q.set_defaults(fn=cmd_check)

    q = s.add_parser("corr", help="Corr(D_i, D_j), model-free")
    common(q)
    q.add_argument("--nboot", type=int, default=100)
    q.add_argument("--kcond", type=int, default=None)
    q.add_argument("--verbose", action="store_true")
    q.set_defaults(fn=cmd_corr)

    q = s.add_parser("cond", help="the conditional-subsample closure")
    common(q)
    q.add_argument("-k", type=int, nargs="+", default=None,
                   help="conditioning planes (default: n/4, n/2, 3n/4)")
    q.add_argument("--edges", default="0,0.02,0.1,0.3,0.5,0.7,0.9,0.98,1",
                   help="quantile edges of the conditioning variable")
    q.add_argument("--condvar", default="dp", choices=("dp", "dxdz"))
    q.add_argument("--permute", type=int, default=0,
                   help="!=0: permute the bin labels (the null of the machinery)")
    q.set_defaults(fn=cmd_cond)

    q = s.add_parser("shuffle", help="real vs independent-increment surrogate")
    common(q)
    q.add_argument("--gsize", type=int, default=1)
    q.add_argument("--split", type=int, default=None)
    q.add_argument("--nboot", type=int, default=40)
    q.add_argument("--nplane", type=int, default=None)
    q.add_argument("--strat", default=None, choices=("locy", "locxy"),
                   help="permute WITHIN strata of the crossing position, so "
                        "the geometric (material-sampling) channel survives "
                        "into the surrogate and only the rest is removed")
    q.add_argument("--nstrat", type=int, default=8)
    q.add_argument("--inject", type=float, default=0.0,
                   help="inject a known loss->straggling feedback of this "
                        "strength before shuffling (sensitivity calibration)")
    q.set_defaults(fn=cmd_shuffle)

    q = s.add_parser("plots", help="figures for the note")
    common(q)
    q.set_defaults(fn=cmd_plots)

    a = p.parse_args()
    if not a.tags:
        a.tags = ["pt3"]
    bad = [t for t in a.tags if t not in SAMPLES]
    if bad:
        p.error(f"unknown sample(s) {bad}; choose from {list(SAMPLES)}")
    os.makedirs(OUT, exist_ok=True)
    a.fn(a)




# ==========================================================================
# subcommand: plots
# ==========================================================================

def cmd_plots(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.style.use(hep.style.ROOT)
    os.makedirs(PLOTDIR, exist_ok=True)
    MITRED = "#A31F34"
    u = np.asarray(UCURVE)

    # ---- 1. the conditional-subsample closure, per bin --------------------
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.6), constrained_layout=True)
    for ax, (tag, kc) in zip(axs, (("pt3", 7), ("real400", 9))):
        r = cond_one(tag, kc, "qop", [float(x) for x in
                                      "0,0.02,0.1,0.3,0.5,0.7,0.9,0.98,1".split(",")])
        nb = len(r["bins"])
        cm = plt.cm.coolwarm(np.linspace(0, 1, nb))
        for i, b in enumerate(r["bins"]):
            lw = 2.6 if i in (0, nb - 1) else 1.2
            ax.errorbar(u, b["closure"], yerr=b["err"], color=cm[i], lw=lw,
                        marker="o", ms=4,
                        label=rf"$\langle\delta p\rangle$ = {b['cmean']:+.1f} MeV")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("$u$")
        ax.set_title(f"{tag},  k = {kc}", fontsize=15)
        ax.legend(fontsize=9, ncol=2)
    axs[0].set_ylabel(r"data $-$ model,  $\langle e^{-u d^2}\rangle$")
    fig.savefig(os.path.join(PLOTDIR, "cond_bins.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)

    # ---- 2. real vs surrogate, per plane ----------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.6), constrained_layout=True)
    jobs = [("pt3", None, "toy pT=3, unstratified"),
            ("real400", None, "real tracker, unstratified"),
            ("real400", "locy", "real tracker, stratified on locy")]
    for ax, (tag, st, ttl) in zip(axs, jobs):
        res, sc_ = shuffle_all(tag, "qop", nboot=0, strat=st)
        cm = plt.cm.viridis(np.linspace(0, 0.92, len(res)))
        for i, r in enumerate(res):
            ax.plot(u, r["delta"], color=cm[i], lw=1.4, marker="o", ms=3)
        ax.plot(u, np.mean([r["delta"] for r in res], axis=0), color=MITRED,
                lw=3.0, label="plane mean")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("$u$")
        ax.set_title(ttl, fontsize=14)
        ax.legend(fontsize=11)
    axs[0].set_ylabel(r"real $-$ surrogate,  $\langle e^{-u z^2}\rangle$")
    fig.savefig(os.path.join(PLOTDIR, "shuffle_delta.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)

    # ---- 3. the two channels, side by side --------------------------------
    from scipy.stats import rankdata
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
    for tag, col, mk in (("pt3", "k", "o"), ("real400", MITRED, "s")):
        legs, sim = model_of(tag), sim_of(tag)
        n = min(sim["valid"].shape[1], len(legs))
        X, Xref, valid = state_array(sim, legs)
        ok = np.all(valid, axis=1)
        D = increments(X[ok], Xref, legs, n)
        d = D @ FUNCTIONALS["qop"]
        ly = X[ok][:, :, 4]
        sp = [np.corrcoef(rankdata(d[:, m]), rankdata(ly[:, m]))[0, 1]
              for m in range(n)]
        axs[0].plot(range(n), sp, mk + "-", color=col, ms=6, lw=1.6, label=tag)
        acc = np.cumsum(d, axis=1)
        lam = []
        for m in range(1, n):
            rr = _rob(acc[:, m - 1])
            uu = np.clip(-acc[:, m - 1] / rr, -3, 3)
            q = np.quantile(uu, np.linspace(0, 1, 11))
            xs, ys = [], []
            base = _rob(d[:, m])
            for b in range(10):
                msk = (uu >= q[b]) & (uu < q[b + 1] if b < 9 else uu <= q[b + 1])
                if msk.sum() < 500:
                    continue
                xs.append(uu[msk].mean())
                ys.append(_rob(d[msk, m]) / base)
            lam.append(np.polyfit(xs, ys, 1)[0])
        axs[1].plot(range(1, n), lam, mk + "-", color=col, ms=6, lw=1.6,
                    label=tag)
    axs[0].axhline(0, color="k", lw=0.8)
    axs[0].set_xlabel("plane $m$")
    axs[0].set_ylabel(r"$\rho_s(D_m,\ \mathrm{loc}y_m)$")
    axs[0].set_title("geometric channel: material vs crossing point",
                     fontsize=14)
    axs[0].legend(fontsize=12)
    axs[1].axhline(0, color="k", lw=0.8)
    axs[1].set_xlabel("leg $m$")
    axs[1].set_ylabel(r"$\lambda_m$")
    axs[1].set_title(r"momentum feedback: d ln rob($D_m$) / d(loss)",
                     fontsize=14)
    axs[1].legend(fontsize=12)
    fig.savefig(os.path.join(PLOTDIR, "channels.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {PLOTDIR}/{{cond_bins,shuffle_delta,channels}}.png")

if __name__ == "__main__":
    main()
