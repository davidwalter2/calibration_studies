#!/usr/bin/env python3
"""Standardize the clean-propagation closure test by the FISHER scale
sqrt(1/I) instead of the model's alpha-truncated sigma.

WHY
---
The closure statistic is  <e^{-u z^2}>_data - <e^{-u z^2}>_model  with
z = (data - reference)/s.  Today s = sqrt(model_variance(...)), which is built
from the ALPHA-TRUNCATED ionization variance.  NOTES_TCUT established that the
model's characteristic function is bit-identical across truncation conventions
-- alpha enters the test ONLY through s -- while s itself moves by 20-64 %
(up to x2.8 on a single plane) under a 10x change of `StepLengthLimit`.  So the
test is quoted in an arbitrary, step-dependent unit.

    s_F = sigma * sqrt(1/I_z),      I_z = Fisher information of the block in
                                    units of the standardized variable z

is convention-free: I is a functional of the model's own (truncation-free)
density, needs no cut, and is the scale a Fisher-scoring fit would use.
1/I_z is taken from the EXACT FFT inversion of the model CF -- never from the
saddlepoint, which NOTES_XXII showed is 5-38 % wrong and unpredictably so.

WHAT IS AND IS NOT NEW
----------------------
Changing the standardization is EXACTLY a relabelling of the probe axis:

    closure_F(u) == closure_sigma(u * I_z)      identically,

so it cannot create or destroy a data-model difference.  What it buys is that
the u axis stops moving when a model convention changes.  Both routes are
implemented independently here (`closure` computes the model CF at the new
scale from scratch; `--check-identity` compares it with the relabelled old
one) so the identity is verified rather than assumed.

SUBCOMMANDS
-----------
    fisher     1/I, sigma and s_F per plane + the convergence knobs
    closure    the renormalized closure, both normalizations, u-scan
    stepscan   THE CONTROL: stepLength 10 / 1 / 0.1 mm under both norms
    thetacut   MS angular-regulator sensitivity of I for the position functional
    gshape     the cf_ms_exact j0(x)-1 fix, closure with and without

Model files live in the session scratchpad; see --help for the defaults.
"""

import argparse
import os
import sys

import numpy as np
from scipy.special import j0

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "cleanprop"))

import cf_ms_exact                                              # noqa: E402
import cgf_channels as cc                                       # noqa: E402
import hbasis                                                   # noqa: E402
from cf_ms_exact import moliere_params                          # noqa: E402
from cf_propagation_test import (FUNCTIONALS, REF_BRANCH, SIM_BRANCH,  # noqa: E402
                                 load_model, model_phi, model_variance,
                                 weier_scalar)
from wums import logging as _wums_logging                       # noqa: E402

logger = _wums_logging.child_logger(__name__)

SCRATCH = ("/tmp/claude-125124/-work-submit-david-w-ZMass/"
           "40621a07-b09b-46d9-b7b3-4189252bcf3e/scratchpad")
PLANES = ("/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/"
          "Analysis/HitAnalyzer/test/toyPlanes_pt3.py")

# Probe grid for the closure curve. The three headline probes of every earlier
# table (0.01, 0.1, 1) are members, so the curve can be read against them.
UCURVE = np.array([1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
UHEAD = (0.01, 0.1, 1.0)
IHEAD = [int(np.argmin(np.abs(UCURVE - u))) for u in UHEAD]   # UHEAD c UCURVE

_MODEL_CACHE = {}
_SIM_CACHE = {}
_SCALE_CACHE = {}


# ==========================================================================
# per-plane parallelism
#
# Every plane's model quantity (1/I, the model CF, the Weierstrass transform)
# is a function of the LEGS ALONE -- the deterministic reference path -- so the
# planes do not talk to each other and the loop over them is embarrassingly
# parallel. It is also the whole cost: profiled 2026-08-15, `geom_closure
# closure` spends 62 % of 477 s inside `cf_brems_exact.rad_exponent`, called
# once per (plane, leg).
#
# fork + return only the SMALL result. The children inherit `legs`/`sim`
# copy-on-write, so nothing large is pickled in either direction (a worker
# returns 6 floats per plane for the scales, nprobes floats for the closure).
# That matters for correctness as much as for speed: the arithmetic in the
# child is bit-for-bit the arithmetic the serial loop did, because it is the
# same code on the same inherited arrays, and float64 survives pickling
# exactly. Verified plane-by-plane against the serial path.
#
# RES_NPROC=1 (or 0) forces the serial loop, which is also the automatic
# fallback if a pool cannot be created (e.g. already inside a daemon worker).
# ==========================================================================

def nproc_default(n):
    """Workers to use for an n-item plane loop."""
    try:
        env = int(os.environ.get("RES_NPROC", "0"))
    except ValueError:
        env = 0
    if env > 0:
        return min(env, n)
    return min(n, len(os.sched_getaffinity(0)))


def pmap(func, items, nproc=None):
    """`[func(x) for x in items]`, evaluated in forked workers, order kept.

    Falls back to the plain list comprehension for a single item, when
    RES_NPROC forces it, or if a pool cannot be started.
    """
    items = list(items)
    n = len(items)
    if n <= 1:
        return [func(x) for x in items]
    npr = nproc_default(n) if nproc is None else min(nproc, n)
    if npr <= 1:
        return [func(x) for x in items]
    import multiprocessing as mp
    try:
        pool = mp.get_context("fork").Pool(processes=npr)
    except Exception as exc:                                  # pragma: no cover
        # Only a failure to START a pool falls back (e.g. already inside a
        # daemonic worker). An exception raised by `func` itself must
        # propagate, not silently trigger a serial re-run of the same work.
        import warnings
        warnings.warn(f"pmap: falling back to serial ({exc})")
        return [func(x) for x in items]
    with pool:
        return pool.map(func, items, chunksize=1)


def _path(p):
    return p if os.path.sep in p else os.path.join(SCRATCH, p)


def load(path):
    path = _path(path)
    if path not in _MODEL_CACHE:
        legs = load_model(path)
        _MODEL_CACHE[path] = legs
        # Provenance for the on-disk scale cache. `_MODEL_CACHE` holds the only
        # strong reference that matters, so `id(legs)` stays unique for the life
        # of the process and cannot be recycled onto a different model.
        _PROVENANCE[id(legs)] = _file_identity(path)
        return _MODEL_CACHE[path]
    # A HIT is re-verified against the file's content, because the key is only
    # the PATH and a model file can be re-exported under the same name inside
    # one session (`export --force` does exactly that). Serving the previous
    # export's legs from a stale path key would be silent and total. ~1 ms.
    now = _file_identity(path)
    was = _PROVENANCE.get(id(_MODEL_CACHE[path]))
    if was is not None and now[3] != was[3]:
        raise RuntimeError(
            f"model cache STALE for {path}\n"
            f"  cached sha256 {was[3]}\n  on-disk sha256 {now[3]}\n"
            f"The file was rewritten after it was loaded. Reload in a fresh "
            f"process rather than reusing this one.")
    return _MODEL_CACHE[path]


_PROVENANCE = {}


def _file_identity(path):
    """(abspath, size, mtime_ns, sha256) of a model file.

    Content hash, not just mtime: a regenerated model with the same size and a
    restored timestamp would otherwise reuse another model's `1/I`. The files
    are 90 kB - 650 kB, so hashing costs ~1 ms.
    """
    import hashlib
    st = os.stat(path)
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return (os.path.abspath(path), st.st_size, st.st_mtime_ns, h.hexdigest())


def load_sim(path, planes=PLANES):
    from toy_loader import load_toy_sim
    path = _path(path)
    if path not in _SIM_CACHE:
        ns = {}
        exec(open(planes).read(), ns)
        _SIM_CACHE[path] = load_toy_sim(path, ns["origin"], ns["normal"],
                                        ns["uaxis"])
    return _SIM_CACHE[path]


# ==========================================================================
# tau grid for the closure transform
# ==========================================================================

def closure_tau(umax, n=3000, tlo=1e-4):
    """Log t grid long enough for the Weierstrass weight of the largest probe.

    The weight is e^{-t^2/(4u)}, so t_max = sqrt(4 u ln(1/eps)) with
    eps = 1e-22 gives sqrt(4*u*50.6) = 14.2 sqrt(u).  A floor of 40 keeps the
    grid at least as long as the historical TAU, so the reproduction of the
    published numbers is like-for-like.  MATCH THE GRID TO THE THING BEING
    INTEGRATED -- the rule this study keeps relearning.
    """
    return np.concatenate([[0.0], np.geomspace(tlo, max(40.0,
                                                        14.2 * np.sqrt(umax)),
                                               n)])


# ==========================================================================
# scales: sigma, 1/I, s_F
# ==========================================================================

def plane_scales(legs, func, floor=1e-8, nt=1 << 17, npad=32, lncut=-60.0,
                 channels=("ioni", "ms", "rad"), nplane=None, tag=""):
    """Per plane: sigma (the fit's own width), 1/I in z^2 units, and the
    absolute Fisher scale s_F = sigma sqrt(1/I).

    I is the Fisher information of the MODEL's own density, by exact FFT
    inversion of the block CF on a grid matched to that block (cgf_channels
    `exact_density` -> `fisher_exact`, relative floor).  It is the same
    pipeline that produced the qop reference values 0.0634 / 0.7356 / 1.6908
    on realgeo_sl10.0; `fisher --ref` re-checks that.
    """
    ident = scale_identity(legs, func, floor, nt, npad, lncut, channels,
                           nplane)
    # In-process key is the model's CONTENT identity, not the caller's `tag`.
    # `tag` was free text: two different models handed the same tag would have
    # silently shared a `1/I`. Nothing in the tree does that today (every caller
    # passes a per-model tag), so this is a hazard removed, not a number
    # changed -- and the full-precision comparison confirms it.
    key = ident["key"]
    if key in _SCALE_CACHE:
        return _SCALE_CACHE[key]
    out = _scale_cache_load(ident, tag)
    if out is None:
        # PER-PLANE a-vectors. With hbasis.USE_H off this is the single
        # FUNCTIONALS[func] object repeated, so the legacy path is unchanged
        # bit for bit; with it on, plane k gets H_k^T e_i so that sigma is the
        # width of the LOCAL component the sim residual actually reports.
        avec = hbasis.avecs(legs, func)
        n = len(legs) if nplane is None else min(nplane, len(legs))
        out = dict(sigma=np.zeros(n), invI=np.zeros(n), sF=np.zeros(n),
                   mass=np.zeros(n), zlo=np.zeros(n), zhi=np.zeros(n))
        global _PS_CTX
        _PS_CTX = (legs, avec, channels, nt, npad, lncut, floor)
        # Build any nuclear-elastic kernels HERE, in the parent: the children
        # inherit them through the fork, so the pool never stampedes the
        # driver (and cannot observe a half-written cache file).
        import cf_nucel_exact as _cnu
        _cnu.warm(legs)
        res = pmap(_plane_scale_one, range(n))
        for k, (sig, invI, mass, zlo, zhi) in enumerate(res):
            out["sigma"][k] = sig
            out["invI"][k] = invI
            out["sF"][k] = sig * np.sqrt(invI)
            out["mass"][k] = mass
            out["zlo"][k] = zlo
            out["zhi"][k] = zhi
        _scale_cache_store(ident, out, tag)
    _SCALE_CACHE[key] = out
    return out


# ==========================================================================
# on-disk cache for the per-plane scales
#
# 1/I is event-independent: a per-plane scalar per (model, functional,
# grid/channel settings). It is also 13.6 s of a real-geometry run. So it is
# keyed to disk -- but the whole value of the number depends on the key being
# unable to go stale, because a cache that silently returns the wrong 1/I is
# strictly worse than no cache. The key therefore covers, and the stored file
# re-verifies on load:
#
#   * the MODEL FILE's content sha256 (plus path/size/mtime, informational) --
#     a regenerated model with a restored timestamp still misses;
#   * the DIRECTION (`func`, i.e. the a-vector) and the CHANNEL selection
#     (`channels` -- the ioni/ms/rad switch that `ionOnly` studies move);
#   * every grid parameter: `floor`, `nt`, `npad`, `lncut`, `nplane`;
#   * the module-global physics switches `J0M1_GUARD`, `MS_NSUB`, `KMS_SCALE`;
#   * a CODE FINGERPRINT -- the sha256 of the source of every module that
#     contributes to the CF or the inversion. Editing any of them changes the
#     key, so a code change cannot be served from cache.
#
# A file whose stored components disagree with the current ones is a HARD
# ERROR, not a silent recompute: it means two different states hashed to the
# same key, which is a bug in the key and must be seen. RES_CACHE=0 disables
# the disk layer entirely; RES_CACHE_DIR relocates it.
# ==========================================================================

_CACHE_MODULES = ("cf_propagation_test", "cf_brems_exact", "cf_track_resolution",
                  "cf_ms_exact", "cf_nucel_exact", "cgf_channels", "fisher_norm",
                  # hbasis picks the a-vector basis and curv2local builds H;
                  # both change sigma, so both belong in the code fingerprint
                  "hbasis", "curv2local")
_CODE_FP = None
SCALE_CACHE_STATS = {"hit": 0, "miss": 0, "store": 0, "disabled": 0}


def _code_fingerprint():
    """sha256 over the source of every module the scales depend on."""
    global _CODE_FP
    if _CODE_FP is None:
        import hashlib
        h = hashlib.sha256()
        here = os.path.dirname(os.path.abspath(__file__))
        for m in _CACHE_MODULES:
            with open(os.path.join(here, m + ".py"), "rb") as fh:
                h.update(fh.read())
        _CODE_FP = h.hexdigest()
    return _CODE_FP


def scale_identity(legs, func, floor, nt, npad, lncut, channels, nplane):
    """Everything that would invalidate a cached `plane_scales` result."""
    import hashlib
    import cf_propagation_test as _cpt
    import cf_track_resolution as _ctr
    import cf_nucel_exact as _cnu
    prov = _PROVENANCE.get(id(legs))
    comp = dict(
        model=prov[3] if prov else None,
        model_path=prov[0] if prov else None,
        nlegs=len(legs), func=func, floor=repr(floor), nt=nt, npad=npad,
        lncut=repr(lncut), channels=",".join(channels), nplane=repr(nplane),
        # THE PHYSICS KNOBS, from each module's own registry rather than from a
        # hand-written list.  The hand-written list carried J0M1_GUARD,
        # MS_NSUB, KMS_SCALE, IONI_A3_SCALE, IONI_TMAX_SCALE and
        # IONI_EXC_SCALE, and therefore silently omitted IONI_KOKOULIN (and
        # its TCUT/NBIN), RAD_CHANNEL, G4_FF_SQUARED and G4_SCREEN_F -- all of
        # which change the CF that 1/I is built from, i.e. all of which
        # silently rescale the u axis of every closure curve.  Registry-driven
        # so that the omission cannot recur; `barkas_probe.py guards` fails if
        # a module grows a knob that is in neither PHYSICS_GLOBALS nor
        # _NOT_PHYSICS.
        knobs_cpt=repr(_cpt.physics_state()),
        knobs_ctr=repr(_ctr.physics_state()),
        knobs_ms=repr(cf_ms_exact.physics_state()),
        # The nuclear elastic channel enters block_cf_exponent, so it changes
        # the CF that 1/I is built from and therefore rescales the u axis --
        # the same way RAD_CHANNEL does.  It has to be in this hash for the
        # same reason all the others are.
        knobs_nucel=repr(_cnu.physics_state()),
        # USE_H switches the a-vector between the curvilinear FUNCTIONALS
        # vector and H_k^T e_i, i.e. it changes sigma on every plane. Without
        # it here a legacy-basis s_F would be served to an H-basis call.
        knobs_h=repr(hbasis.physics_state()),
        code=_code_fingerprint())
    canon = "\n".join(f"{k}={comp[k]}" for k in sorted(comp))
    comp["_canon"] = canon
    comp["_sha"] = hashlib.sha256(canon.encode()).hexdigest()
    # In-process key: the content identity when we have provenance, otherwise
    # the object identity, which is unique for as long as the caller holds it.
    comp["key"] = (comp["_sha"] if prov is not None
                   else ("noprov", id(legs), comp["_sha"]))
    return comp


def _cache_dir():
    d = os.environ.get("RES_CACHE_DIR") or os.path.expanduser(
        "~/.cache/cvh_resolution/plane_scales")
    return d


def _cache_enabled():
    return os.environ.get("RES_CACHE", "1") not in ("0", "no", "off")


def _scale_cache_load(ident, tag):
    if not _cache_enabled():
        SCALE_CACHE_STATS["disabled"] += 1
        return None
    if ident["model"] is None:
        SCALE_CACHE_STATS["miss"] += 1
        logger.info(f"scale cache: SKIP ({tag or ident['func']}) -- these legs "
                    f"have no file provenance, load them through "
                    f"fisher_norm.load() to make them cacheable")
        return None
    p = os.path.join(_cache_dir(), ident["_sha"] + ".npz")
    if not os.path.exists(p):
        SCALE_CACHE_STATS["miss"] += 1
        logger.info(f"scale cache: MISS ({tag or ''} {ident['func']}) "
                    f"{ident['_sha'][:12]} -- computing")
        return None
    d = np.load(p, allow_pickle=False)
    stored = str(d["canon"])
    if stored != ident["_canon"]:
        # Two different states reached the same file name. Never silently
        # recompute past this: it means the key is broken.
        raise RuntimeError(
            f"scale cache COLLISION at {p}\nstored:\n{stored}\n"
            f"wanted:\n{ident['_canon']}\n"
            f"The key is wrong -- fix it rather than deleting the file.")
    SCALE_CACHE_STATS["hit"] += 1
    logger.info(f"scale cache: HIT  ({tag or ''} {ident['func']}) "
                f"{ident['_sha'][:12]}")
    return {k: d[k] for k in ("sigma", "invI", "sF", "mass", "zlo", "zhi")}


def _scale_cache_store(ident, out, tag):
    if not _cache_enabled() or ident["model"] is None:
        return
    d = _cache_dir()
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, ident["_sha"] + ".npz")
    # Write through a file OBJECT: np.savez appends '.npz' to a path that lacks
    # it, so a '<sha>.npz.<pid>.tmp' path silently becomes '....tmp.npz' and the
    # os.replace below then fails on a name that does not exist.
    tmp = p + f".{os.getpid()}.tmp"
    with open(tmp, "wb") as fh:
        np.savez(fh, canon=np.array(ident["_canon"]), **out)
    os.replace(tmp, p)                       # atomic; concurrent runs are safe
    SCALE_CACHE_STATS["store"] += 1
    logger.info(f"scale cache: STORE ({tag or ''} {ident['func']}) "
                f"{ident['_sha'][:12]} -> {p}")


def prewarm_scales(jobs, **kw):
    """Compute `plane_scales` for MANY (legs, func) at once, in ONE flat pool.

    `jobs` is a list of `(legs, func, tag)`. Per-plane parallelism caps at the
    number of planes (14-19); a campaign is configurations x functionals x
    planes, and those are all independent, so the natural unit is the campaign.
    This flattens every (job, plane) pair into a single `pmap`, which is the
    difference between six sequential 14-19-way pools and one ~100-way pool on
    a 768-core machine.

    Results land in the in-process cache (and on disk), so the ordinary
    `plane_scales` calls that follow are hits. Each task runs exactly the code
    the serial path runs, so the values are bit-identical -- this is scheduling,
    not arithmetic. Jobs already cached are skipped, not recomputed.
    """
    floor = kw.get("floor", 1e-8)
    nt = kw.get("nt", 1 << 17)
    npad = kw.get("npad", 32)
    lncut = kw.get("lncut", -60.0)
    channels = kw.get("channels", ("ioni", "ms", "rad"))
    nplane = kw.get("nplane", None)

    todo, idents = [], []
    for legs, func, tag in jobs:
        ident = scale_identity(legs, func, floor, nt, npad, lncut, channels,
                               nplane)
        if ident["key"] in _SCALE_CACHE:
            continue
        out = _scale_cache_load(ident, tag)
        if out is not None:
            _SCALE_CACHE[ident["key"]] = out
            continue
        todo.append((legs, func, tag))
        idents.append(ident)
    if not todo:
        return 0

    global _PS_JOBS
    _PS_JOBS = [(legs, hbasis.avecs(legs, func), channels, nt, npad, lncut,
                 floor,
                 len(legs) if nplane is None else min(nplane, len(legs)))
                for legs, func, _ in todo]
    tasks = [(j, k) for j, spec in enumerate(_PS_JOBS) for k in range(spec[7])]
    res = pmap(_plane_scale_job_one, tasks)

    it = iter(res)
    for j, ((legs, func, tag), ident) in enumerate(zip(todo, idents)):
        n = _PS_JOBS[j][7]
        out = dict(sigma=np.zeros(n), invI=np.zeros(n), sF=np.zeros(n),
                   mass=np.zeros(n), zlo=np.zeros(n), zhi=np.zeros(n))
        for k in range(n):
            sig, invI, mass, zlo, zhi = next(it)
            out["sigma"][k] = sig
            out["invI"][k] = invI
            out["sF"][k] = sig * np.sqrt(invI)
            out["mass"][k] = mass
            out["zlo"][k] = zlo
            out["zhi"][k] = zhi
        _scale_cache_store(ident, out, tag)
        _SCALE_CACHE[ident["key"]] = out
    logger.info(f"prewarm_scales: {len(todo)} (model, functional) sets, "
                f"{len(tasks)} planes, {nproc_default(len(tasks))} workers")
    return len(tasks)


_PS_JOBS = None


def _plane_scale_job_one(jk):
    j, k = jk
    legs, avecs, channels, nt, npad, lncut, floor, _ = _PS_JOBS[j]
    avec = avecs[k]
    sig = float(np.sqrt(model_variance(legs, k, avec)[0]))
    z, p, dp = cc.exact_density(legs, k, avec, sig, channels=channels,
                                nt=nt, npad=npad, lncut=lncut)
    I, info = cc.fisher_exact(z, p, dp, floor=floor)
    return (sig, 1.0 / I, info.get("mass_frac", np.nan),
            info.get("zlo", np.nan), info.get("zhi", np.nan))


_PS_CTX = None


def _plane_scale_one(k):
    """One plane's (sigma, 1/I, mass, zlo, zhi). Reads `_PS_CTX`, which the
    parent sets before forking -- see `pmap`. Module level so it is picklable
    by reference; the arrays themselves are never pickled."""
    legs, avecs, channels, nt, npad, lncut, floor = _PS_CTX
    avec = avecs[k]
    sig = float(np.sqrt(model_variance(legs, k, avec)[0]))
    z, p, dp = cc.exact_density(legs, k, avec, sig, channels=channels,
                                nt=nt, npad=npad, lncut=lncut)
    I, info = cc.fisher_exact(z, p, dp, floor=floor)
    return (sig, 1.0 / I, info.get("mass_frac", np.nan),
            info.get("zlo", np.nan), info.get("zhi", np.nan))


# ==========================================================================
# closure
# ==========================================================================

def closure_plane(legs, sim, func, k, s, probes, tau=None):
    """(data - model) of <e^{-u z^2}> on ONE plane, z = (sim - ref)/s.

    `s` is an ABSOLUTE width; the model CF is rebuilt at that scale, so
    nothing here knows which convention produced it.
    """
    probes = np.atleast_1d(np.asarray(probes, dtype=float))
    if tau is None:
        tau = closure_tau(float(probes.max()))
    good = sim["valid"][:, k] & np.isfinite(sim[SIM_BRANCH[func]][:, k])
    if good.sum() < 100:
        return None
    s = float(s)
    z = (sim[SIM_BRANCH[func]][good, k] - legs[k][REF_BRANCH[func]]) / s
    phi = model_phi(legs, k, FUNCTIONALS[func], s, tau)
    return np.array([float(np.mean(np.exp(-u * z ** 2)))
                     - weier_scalar(phi, u, tau) for u in probes])


def closure(legs, sim, func, scale, probes, tau=None):
    """`closure_plane` over every plane the sim populates."""
    nl = min(sim["valid"].shape[1], len(legs), len(scale))
    global _CP_CTX
    _CP_CTX = (legs, sim, func, scale, probes, tau)
    res = pmap(_closure_plane_one, range(nl))
    rows = [r for r in res if r is not None]
    ks = [k for k, r in enumerate(res) if r is not None]
    return np.array(rows), np.array(ks)


_CP_CTX = None


def _closure_plane_one(k):
    legs, sim, func, scale, probes, tau = _CP_CTX
    return closure_plane(legs, sim, func, k, scale[k], probes, tau=tau)


def _fmt_row(label, vals, width=10, prec=5):
    return f"{label:<26s}" + "".join(f"{v:+{width}.{prec}f}" for v in vals)


# ==========================================================================
# subcommand: fisher
# ==========================================================================

def cmd_fisher(args):
    if args.ref:
        print("REFERENCE CHECK -- realgeo_sl10.0 q/p, planes 0/9/18")
        print("targets from NOTES_XXII 8.5: 0.0634 / 0.7356 / 1.6908\n")
        rg = load("realgeo_sl10.0.root")
        for k in (0, 9, 18):
            sig = float(np.sqrt(model_variance(rg, k, FUNCTIONALS["qop"])[0]))
            z, p, dp = cc.exact_density(rg, k, FUNCTIONALS["qop"], sig)
            I, info = cc.fisher_exact(z, p, dp)
            print(f"   plane {k:2d}   sigma = {sig:.4e}   "
                  f"1/I = {1.0 / I:.4f}   mass = {info['mass_frac']:.6f}")
        print()

    for mod in args.models:
        legs = load(mod)
        for func in args.funcs:
            sc = plane_scales(legs, func, floor=args.floor, tag=mod)
            print(f"=== {mod}   {func}   ({len(sc['sigma'])} planes)")
            print(f"{'k':>3} {'sigma':>12} {'1/I [z^2]':>11} "
                  f"{'sqrt(1/I)':>10} {'s_F':>12} {'mass':>10} "
                  f"{'z support':>20}")
            for k in range(len(sc["sigma"])):
                print(f"{k:3d} {sc['sigma'][k]:12.4e} {sc['invI'][k]:11.4f} "
                      f"{np.sqrt(sc['invI'][k]):10.4f} {sc['sF'][k]:12.4e} "
                      f"{sc['mass'][k]:10.6f} "
                      f"[{sc['zlo'][k]:8.1f},{sc['zhi'][k]:7.1f}]")
            g = float(np.exp(np.mean(np.log(sc["invI"]))))
            print(f"    geometric mean 1/I = {g:.4f}  "
                  f"(=> u_Fisher ~ {g:.3f} x u_sigma probes the same region "
                  f"on average)\n")

    if args.knobs:
        print("CONVERGENCE KNOBS (the rule: relative floors, matched grids)\n")
        legs = load(args.models[0])
        for func in args.funcs:
            print(f"--- {args.models[0]}  {func}   1/I")
            hdr = ["floor 1e-4", "1e-6", "1e-8", "1e-10", "1e-12",
                   "nt/4", "npad x4", "lncut -30", "lncut -120"]
            print(f"{'k':>3} " + "".join(f"{h:>12s}" for h in hdr))
            for k in args.knobplanes:
                if k >= len(legs):
                    continue
                vals = []
                for f in (1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
                    vals.append(plane_scales(legs, func, floor=f,
                                             tag=args.models[0])["invI"][k])
                vals.append(plane_scales(legs, func, nt=1 << 15,
                                         tag=args.models[0])["invI"][k])
                vals.append(plane_scales(legs, func, npad=128,
                                         tag=args.models[0])["invI"][k])
                vals.append(plane_scales(legs, func, lncut=-30.0,
                                         tag=args.models[0])["invI"][k])
                vals.append(plane_scales(legs, func, lncut=-120.0,
                                         tag=args.models[0])["invI"][k])
                print(f"{k:3d} " + "".join(f"{v:12.5f}" for v in vals))
            print()


# ==========================================================================
# subcommand: closure
# ==========================================================================

def cmd_closure(args):
    sim = load_sim(args.sim, args.planes)
    for mod in args.models:
        legs = load(mod)
        for func in args.funcs:
            sc = plane_scales(legs, func, tag=mod)
            print(f"=== {mod}   {func}   "
                  f"{sim['ntot']} events, {sim['nkept']} complete\n")

            full = {}
            for norm, scale in (("sigma", sc["sigma"]), ("Fisher", sc["sF"])):
                rows, ks = closure(legs, sim, func, scale, UCURVE)
                full[norm] = rows
                print(f"  norm = {norm:<7s}  "
                      f"u in units of this normalization")
                print(f"  {'k':>3} {'scale':>12} " +
                      "".join(f"{'u=' + str(u):>12s}" for u in UHEAD))
                for i, k in enumerate(ks):
                    print(f"  {k:3d} {scale[k]:12.4e} " +
                          "".join(f"{v:+12.5f}" for v in rows[i, IHEAD]))
                print(f"  {'mean':>3} {'':>12} " +
                      "".join(f"{v:+12.5f}" for v in rows.mean(axis=0)[IHEAD]))
                print(f"  {'rms':>3} {'':>12} " +
                      "".join(f"{v:12.5f}" for v in rows.std(axis=0)[IHEAD]))
                print()

            # the u-curve, both normalizations, plane-averaged
            print("  plane-averaged closure vs u (mean over planes)")
            print(f"  {'norm':>8} " + "".join(f"{u:>10.3g}" for u in UCURVE))
            for norm in ("sigma", "Fisher"):
                print(f"  {norm:>8} " +
                      "".join(f"{v:+10.5f}" for v in full[norm].mean(axis=0)))
            print()

            if args.check_identity:
                # closure_F(u) must equal closure_sigma(u * I_z) IDENTICALLY:
                # the two differ only by the substitution z -> z sqrt(I).
                # Computed independently (own scale, own CF, own tau) so this
                # is a check of the implementation, not an algebraic tautology.
                print("  IDENTITY CHECK  closure_F(u) == "
                      "closure_sigma(u * I_z)")
                print(f"  {'k':>3} " +
                      "".join(f"{'u=' + str(u):>26s}" for u in UHEAD))
                worst = 0.0
                for k in range(len(sc["sigma"])):
                    cells = []
                    for u in UHEAD:
                        rF = closure_plane(legs, sim, func, k, sc["sF"][k], [u])
                        rS = closure_plane(legs, sim, func, k,
                                           sc["sigma"][k],
                                           [u / sc["invI"][k]])
                        if rF is None:
                            continue
                        worst = max(worst, abs(rF[0] - rS[0]))
                        cells.append(f"{rF[0]:+12.6f}/{rS[0]:+12.6f}")
                    if cells:
                        print(f"  {k:3d} " + "".join(f"{c:>26s}"
                                                     for c in cells))
                print(f"  worst |difference| = {worst:.2e}\n")


# ==========================================================================
# subcommand: stepscan  -- THE CONTROL
# ==========================================================================

def cmd_stepscan(args):
    sim = load_sim(args.sim, args.planes)
    print("STEP-LIMIT SCAN.  Only the MODEL changes across rows: the sim is one\n"
          "file for all three, so this isolates the model's own step structure.\n")

    # KNOB LIVENESS + common reference, before anything is interpreted.
    ref = args.models[0]
    r0 = load(ref)
    print(f"{'model':<22} {'ms steps':>9} {'ioni steps':>11} "
          f"{'max |d refqop|/qop':>19} {'max |d reflocx| [um]':>21}")
    for mod in args.models:
        m = load(mod)
        dq = max(abs(m[k]["refqop"] / r0[k]["refqop"] - 1.0)
                 for k in range(len(r0)))
        dx = max(abs(m[k]["reflocx"] - r0[k]["reflocx"])
                 for k in range(len(r0))) * 1e4
        print(f"{mod:<22} {sum(len(l['ms']) for l in m):9d} "
              f"{sum(len(l['ioni']) for l in m):11d} {dq:19.2e} "
              f"{dx:21.2e}")
    print("  (the reference trajectory is common to all three; only the noise\n"
          "   bookkeeping is re-stepped, so the closure is a like-for-like "
          "test)\n")

    for func in args.funcs:
        scales = {}
        for mod in args.models:
            scales[mod] = plane_scales(load(mod), func, tag=mod)

        ref = args.models[0]
        print(f"=== {func}   scale ratios against {ref} (per plane, "
              f"min / median / max)")
        print(f"{'model':<22} {'sigma ratio':>28} {'s_F ratio':>28}")
        for mod in args.models:
            r1 = scales[mod]["sigma"] / scales[ref]["sigma"]
            r2 = scales[mod]["sF"] / scales[ref]["sF"]
            print(f"{mod:<22} "
                  f"{r1.min():8.4f} {np.median(r1):9.4f} {r1.max():9.4f} "
                  f"{r2.min():8.4f} {np.median(r2):9.4f} {r2.max():9.4f}")
        print()

        full = {}
        for norm in ("sigma", "Fisher"):
            key = "sigma" if norm == "sigma" else "sF"
            for mod in args.models:
                full[(norm, mod)] = closure(load(mod), sim, func,
                                            scales[mod][key], UCURVE)[0]

        for norm in ("sigma", "Fisher"):
            print(f"--- closure, u in units of the {norm} scale "
                  f"(mean over planes, rms in brackets)")
            print(f"{'model':<22} " +
                  "".join(f"{'u=' + str(u):>20s}" for u in UHEAD))
            for mod in args.models:
                rows = full[(norm, mod)]
                m, s = rows.mean(axis=0)[IHEAD], rows.std(axis=0)[IHEAD]
                print(f"{mod:<22} " +
                      "".join(f"{mi:+12.5f} ({si:5.5f})"
                              for mi, si in zip(m, s)))
            print()

        print(f"--- closure curve vs u, {func}, "
              f"u in units of each normalization")
        for norm in ("sigma", "Fisher"):
            print(f"  norm = {norm}")
            print(f"  {'model':<20} " +
                  "".join(f"{u:>10.3g}" for u in UCURVE))
            for mod in args.models:
                print(f"  {mod:<20} " +
                      "".join(f"{v:+10.5f}"
                              for v in full[(norm, mod)].mean(axis=0)))
            print()


# ==========================================================================
# MS angular regulator: an independent, cut-able CF for the MS channel
# ==========================================================================

def j0m1(x):
    """J0(x) - 1, stable at small x.

    j0(x) - 1 evaluated naively loses ALL precision below |x| ~ 1e-8 (both
    terms are ~1, the difference is -x^2/4).  This is the bug NOTES_XXII 8.4
    found in cf_ms_exact's G table; the same trap is avoided here.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty(x.shape, dtype=np.float64)
    small = np.abs(x) < 1e-2
    x2 = x[small] ** 2
    out[small] = -0.25 * x2 * (1.0 - x2 / 16.0 + x2 * x2 / 576.0)
    out[~small] = j0(x[~small]) - 1.0
    return out


_GT = np.concatenate([[0.0], np.geomspace(1e-9, 1e5, 2400)])
_GROW_CACHE = {}


def g_row(ymax, ycut, ny2=2400, ff_pow=4):
    """G(tau; ymax, ycut) = Int_0^{ycut^2} (J0(tau y) - 1) dy^2 /
                            (1+y^2)^2 / (1 + y^2/ymax^2)^ff_pow
    on the log-tau grid `_GT`.

    Independent of cf_ms_exact's table: stable j0-1, a finer y^2 grid
    (2400 vs 360 nodes) and Gauss-Legendre-free log-trapezoid in ln y^2 so
    the quadrature error is not the +2.0e-3 plateau bias of NOTES_XXII V7.
    `ycut` is the hard angular cutoff expressed in units of chi_a.
    """
    key = (round(np.log(ymax), 6), round(np.log(ycut), 6), ny2, ff_pow)
    if key in _GROW_CACHE:
        return _GROW_CACHE[key]
    hi = min(ycut ** 2, 1e13)
    ly = np.linspace(np.log(1e-8), np.log(hi), ny2)
    y2 = np.exp(ly)
    w = y2 / (1.0 + y2) ** 2 / (1.0 + y2 / ymax ** 2) ** ff_pow   # dy^2 = y^2 dln
    arg = np.outer(_GT, np.sqrt(y2))
    row = np.trapezoid(j0m1(arg) * w[None, :], ly, axis=1)
    _GROW_CACHE[key] = row
    return row


def _snap_ymax(ymax):
    """The 13-row ymax snap that `cf_track_resolution.ms_step_exponent`
    performs (it calls gshape with one of `_YMAXG`'s values, not with the
    step's own ymax).  Mirrored so that theta_cut = inf CONTINUES the CF the
    production code actually evaluates; without it the two differ by 5 % in
    the MS variance for this geometry, because G(tau->0) ~ ln(ymax^2)."""
    lg = np.log(np.clip(ymax, 1e1, 1e7))
    r = np.round((lg - np.log(1e1)) / (np.log(1e7 / 1e1) / 12))
    return np.exp(np.log(1e1) + r * (np.log(1e7 / 1e1) / 12))


def ms_step_exponent_cut(steps, wstd, tau, theta_cut=np.inf, ny2=2400,
                         snap=True):
    """The MS log-CF exponent of `steps` with an explicit maximum single
    deflection `theta_cut` [rad].  theta_cut = inf reproduces the production
    spectrum (whose only angular termination is the nuclear form factor)."""
    ok = steps[:, 5] > 0.0
    if not ok.any():
        return np.zeros(len(tau))
    st = steps[ok]
    if st.shape[1] >= 10:
        prm = np.array([moliere_params(*s[:5], s[7], s[8]) for s in st])
    else:
        prm = np.array([moliere_params(*s[:5]) for s in st])
    chic2, chia2, thff2 = prm[:, 0], prm[:, 1], prm[:, 2]
    act = (chic2 > 0.0) & (chia2 > 0.0)
    if not act.any():
        return np.zeros(len(tau))
    chic2, chia2, thff2 = chic2[act], chia2[act], thff2[act]
    ymax = np.sqrt(thff2 / chia2)
    if snap:
        ymax = _snap_ymax(ymax)
    ycut = (np.inf if not np.isfinite(theta_cut)
            else theta_cut / np.sqrt(chia2))
    ycut = np.broadcast_to(np.asarray(ycut, dtype=float), ymax.shape)
    # bucket in 0.02-decade bins so the table is reused; both axes enter only
    # through logs, so this is a controlled interpolation error (checked by
    # halving the bin width).
    kmax = np.round(np.log10(ymax) / 0.02).astype(int)
    kcut = np.round(np.log10(np.minimum(ycut, 1e7)) / 0.02).astype(int)
    S = np.zeros(len(tau))
    for pair in set(zip(kmax.tolist(), kcut.tolist())):
        m = (kmax == pair[0]) & (kcut == pair[1])
        row = g_row(10.0 ** (0.02 * pair[0]), 10.0 ** (0.02 * pair[1]),
                    ny2=ny2)
        for i in np.where(m)[0]:
            a = np.sqrt(chia2[i]) * wstd * np.abs(tau)
            g = np.interp(a, _GT, row, left=row[0], right=row[-1])
            S += (chic2[i] / chia2[i]) * g
    return S


def ms_cf_exponent_block(legs, k, avec, sigma, tau, theta_cut=np.inf,
                         ny2=2400, snap=True):
    """MS-channel exponent of the whole block, with the same sub-step
    quadrature and weights as cf_propagation_test.model_phi."""
    import cf_propagation_test as cpt
    A_ms, _, A_ms_start = cpt.step_transports(legs, k)
    S = np.zeros(len(tau))
    for j in range(k + 1):
        leg = legs[j]
        if not len(leg["ms"]):
            continue
        wv = np.einsum("i,sij->sj", avec, A_ms[j])
        wv0 = np.einsum("i,sij->sj", avec, A_ms_start[j])
        coslam = leg["refpt"] / leg["refp"] if leg["refp"] > 0 else 1.0

        def _weff(v):
            return np.sqrt(v[:, 1] ** 2
                           + (v[:, 2] / max(coslam, 1e-3)) ** 2) / sigma
        weff, weff0 = _weff(wv), _weff(wv0)
        for s in range(len(leg["ms"])):
            if weff[s] <= 0.0 and weff0[s] <= 0.0:
                continue
            rec = leg["ms"][s:s + 1].copy()
            rec[:, 2] /= cpt.MS_NSUB
            for i in range(cpt.MS_NSUB):
                f = (i + 0.5) / cpt.MS_NSUB
                w = weff0[s] + f * (weff[s] - weff0[s])
                if w > 0.0:
                    S += ms_step_exponent_cut(rec, w, tau,
                                              theta_cut=theta_cut, ny2=ny2,
                                              snap=snap)
    return S


def cmd_thetacut(args):
    """Task 3: does the MS angular regulator move I for the POSITION
    functional, where MS is ~100 % of the variance?"""
    legs = load(args.model)
    func = args.func
    avec = FUNCTIONALS[func]
    print(f"MS ANGULAR REGULATOR SENSITIVITY -- {args.model}, {func}\n")

    # what the production spectrum's own angular scales are
    st = np.vstack([l["ms"] for l in legs if len(l["ms"])])
    prm = np.array([moliere_params(*s[:5], s[7], s[8]) if st.shape[1] >= 10
                    else moliere_params(*s[:5]) for s in st])
    chia = np.sqrt(prm[:, 1])
    thff = np.sqrt(prm[:, 2])
    print(f"  screening angle chi_a      : {chia.min():.3e} - "
          f"{chia.max():.3e} rad")
    print(f"  form-factor angle theta_FF : {thff.min():.3e} - "
          f"{thff.max():.3e} rad")
    print(f"  y^2 grid top of the production table = 1e13 => "
          f"theta_max = {np.sqrt(1e13) * chia.max():.3g} rad\n")

    for k in args.planes:
        if k >= len(legs):
            continue
        sig = float(np.sqrt(model_variance(legs, k, avec)[0]))
        tau = cc.auto_tau(legs, k, avec, sig)
        # non-MS channels once
        Sother = cc.block_cf_exponent(legs, k, avec, sig, tau,
                                      channels=("ioni", "rad"))
        print(f"--- plane {k}   sigma = {sig:.4e}")
        print(f"{'theta_cut [rad]':>16} {'1/I':>10} {'ratio':>9} "
              f"{'mass':>10} {'k2/k2(inf)':>11}")
        base = None
        k2ref = None
        for tc in args.cuts:
            Sms = ms_cf_exponent_block(legs, k, avec, sig, tau, theta_cut=tc,
                                       ny2=args.ny2)
            z, p, dp = cc.invert_cf(Sother + Sms, tau, npad=32, nt=1 << 17,
                                    deriv=True)
            I, info = cc.fisher_exact(z, p, dp, floor=args.floor)
            # kappa2 of the MS channel from the small-t curvature of S
            k2 = -2.0 * np.interp(1e-3, tau, Sms) / 1e-6
            if base is None:
                base, k2ref = 1.0 / I, k2
            print(f"{tc:16.4g} {1.0 / I:10.4f} {(1.0 / I) / base:9.4f} "
                  f"{info.get('mass_frac', np.nan):10.6f} "
                  f"{k2 / k2ref:11.5f}")
        print()


# ==========================================================================
# subcommand: gshape  -- task 5, the j0(x)-1 fix
# ==========================================================================

def cmd_gshape(args):
    sim = load_sim(args.sim, args.planes)
    print("cf_ms_exact G-table j0(x)-1 GUARD: A/B\n")

    # 1. how much does the table itself move?
    cf_ms_exact.set_j0_guard(False)
    Gold = cf_ms_exact._G2D.copy()
    cf_ms_exact.set_j0_guard(True)
    Gnew = cf_ms_exact._G2D.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(Gnew != 0.0, Gold / Gnew - 1.0, np.nan)
    print("  table (old/new - 1), by ymax row, at three tau decades")
    tsel = [np.argmin(np.abs(cf_ms_exact._GTAU - t))
            for t in (1e-8, 1e-4, 1e0)]
    print(f"  {'ymax':>10} " + "".join(f"{'tau=' + f'{cf_ms_exact._GTAU[i]:.0e}':>14}"
                                       for i in tsel))
    for iy in range(len(cf_ms_exact._YMAXG)):
        print(f"  {cf_ms_exact._YMAXG[iy]:10.3g} " +
              "".join(f"{rel[iy, i]:14.3e}" for i in tsel))
    print()

    # 2. closure with and without, both normalizations
    for mod in args.models:
        for func in args.funcs:
            print(f"=== {mod}  {func}")
            print(f"  {'guard':<8} {'norm':<8} {'<s>/<s(guard off)>':>20} " +
                  "".join(f"{'u=' + str(u):>14s}" for u in UHEAD))
            base = {}
            for guard in (False, True):
                cf_ms_exact.set_j0_guard(guard)
                _SCALE_CACHE.clear()
                legs = load(mod)
                sc = plane_scales(legs, func, tag=f"{mod}|g{guard}")
                for norm, key in (("sigma", "sigma"), ("Fisher", "sF")):
                    if not guard:
                        base[norm] = sc[key].copy()
                    rows, _ = closure(legs, sim, func, sc[key], UHEAD)
                    r = float(np.mean(sc[key] / base[norm]))
                    print(f"  {str(guard):<8} {norm:<8} {r:20.6f} " +
                          "".join(f"{v:+14.5f}" for v in rows.mean(axis=0)))
            print()
    cf_ms_exact.set_j0_guard(True)


# ==========================================================================

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q, models=("mod_sl10.0.root",), funcs=("qop", "locx")):
        q.add_argument("--models", nargs="+", default=list(models))
        q.add_argument("--funcs", nargs="+", default=list(funcs))
        q.add_argument("--sim", default="sim_homo_scan.root")
        q.add_argument("--planes", default=PLANES)

    q = sub.add_parser("fisher")
    common(q)
    q.add_argument("--floor", type=float, default=1e-8)
    q.add_argument("--ref", action="store_true")
    q.add_argument("--knobs", action="store_true")
    q.add_argument("--knobplanes", type=int, nargs="+", default=[0, 6, 13])
    q.set_defaults(fn=cmd_fisher)

    q = sub.add_parser("closure")
    common(q)
    q.add_argument("--check-identity", action="store_true")
    q.set_defaults(fn=cmd_closure)

    q = sub.add_parser("stepscan")
    common(q, models=("mod_sl10.0.root", "mod_sl1.0.root", "mod_sl0.1.root"))
    q.set_defaults(fn=cmd_stepscan)

    q = sub.add_parser("thetacut")
    q.add_argument("--model", default="mod_sl10.0.root")
    q.add_argument("--func", default="locx")
    q.add_argument("--planes", type=int, nargs="+", default=[0, 6, 13])
    q.add_argument("--cuts", type=float, nargs="+",
                   default=[np.inf, np.pi, 1.0, 0.3, 0.1, 0.03, 0.01, 0.003])
    q.add_argument("--floor", type=float, default=1e-8)
    q.add_argument("--ny2", type=int, default=2400)
    q.set_defaults(fn=cmd_thetacut)

    q = sub.add_parser("gshape")
    common(q, funcs=("locx", "qop"))
    q.set_defaults(fn=cmd_gshape)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
