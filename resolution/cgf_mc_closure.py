#!/usr/bin/env python3
"""Does the CGF weight fit the momentum BETTER than the vanilla one?

Everything measured about the CGF so far is a DIFFERENCE from the previous
estimator -- how far the fitted q/p moved, how the convergence changed, how
little the answer depends on a convention that used to be there. None of it
says the answer is closer to the truth, because on data there is no truth.

This runs the same gen-matched MC tracks through both builds and compares
(q/p_fit - q/p_gen): its BIAS and its WIDTH. That is the only statement that
matters for a momentum-scale calibration, and it is the one the workstream did
not have.

Two areas, because the two estimators are two builds:
  vanilla  CMSSW_15_0_19_patch2       at 663639e (the commit before the CGF
                                      default), so everything else -- the
                                      dEdxlast fix, switches-as-configuration,
                                      the default-ON energy-loss corrections --
                                      is IDENTICAL and only the weight differs.
  cgf      CMSSW_15_0_19_patch2_dev   at 9a7c692 (CgfQoPMode = 1, no alpha).

Robust statistics throughout: the pull distribution has heavy tails by
construction (that is the whole subject), so a mean and an rms would describe
the few catastrophic tracks rather than the estimator. Bias is the median and
a 10 % trimmed mean; width is the 68 % interquantile half-width.

usage:
  python cgf_mc_closure.py run   [--nev 400]
  python cgf_mc_closure.py cmp
"""
import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cf_track_resolution as ctr                                # noqa: E402
from wums import logging as _wums_logging                        # noqa: E402

logger = _wums_logging.child_logger(__name__)

AREAS = {
    "vanilla": "/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2",
    "cgf": "/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev",
}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "mcclosure")

# Two momentum regimes, because the block being replaced is the IONIZATION
# process noise and its weight in the fit rises as the momentum falls: on
# Pt8toInf the substitution moved the fitted q/p by rms 2.1e-6, on J/psi DATA
# at pT 3-10 by 1.3e-5. If the CGF has an accuracy benefit anywhere it is at
# low momentum, so the two samples are the test and its control.
SAMPLES = {
    "pt8": ("/ceph/submit/data/group/cms/store/mc/RunIISummer20UL16RECO/"
            "JPsiToMuMu_Pt8toInf-pythia8/ALCARECO/"
            "TkAlJpsiMuMu-106X_mcRun2_asymptotic_v13_ext1-v3/30000/"
            "0109866D-FAAB-7844-A4D3-98FC9D81106E.root"),
    "pt0": ("/ceph/submit/data/group/cms/store/mc/RunIISummer20UL16RECO/"
            "JPsiToMuMu_Pt0to8-pythia8/ALCARECO/"
            "TkAlJpsiMuMu-106X_mcRun2_asymptotic_v13_ext1-v3/2550000/"
            "6ED536C9-824B-CC4A-BD89-E5B65689CACA.root"),
}
FIELD = ("/work/submit/david_w/ZMass/mfs/data/fitresults/"
         "polyfit3d_full_coeffs_lmax18_cmsswnorm.txt")

_KEEP = ("HOME", "USER", "LOGNAME", "SHELL", "TERM", "HOSTNAME", "TMPDIR",
         "X509_USER_PROXY", "KRB5CCNAME")


def sample_files(sample, n=None):
    """The sample's files, sorted, so chunk i is reproducibly the same file."""
    import glob
    d = os.path.dirname(SAMPLES[sample])
    fs = sorted(glob.glob(os.path.join(d, "*.root")))
    return fs if n is None else fs[:n]


def outfile(arm, sample="pt8", ms=None, chunk=None, mode=None, damp=None, tag=""):
    sub = arm if sample == "pt8" else f"{sample}_{arm}"
    if ms is not None:
        sub = f"{sub}_ms{str(ms).replace('.', '')}"
    if mode is not None:
        sub = f"{sub}_m{mode}"
    if damp is not None:
        sub = f"{sub}_d{str(damp).replace('.', '')}"
    if tag:
        # A campaign tag. The directory name does NOT otherwise encode the
        # event count, so a second campaign at a different --nev would land on
        # the first one's outputs and be skipped as "already done" -- silently
        # mixing two samples, or silently reusing the wrong one.
        sub = f"{sub}_{tag}"
    if chunk is not None:
        sub = f"{sub}_c{chunk:02d}"
    return os.path.join(OUT, sub, "globalcor_singlemc_0.root")


def run_one(arm, nev, force=False, sample="pt8", ms=None, chunk=None, mode=None, damp=None, tag=""):
    """`ms` scales Q's MS variance through CVH_MS_SCALE.

    That knob is weight-ONLY -- multiple scattering is symmetric, so it cannot
    move the reference trajectory -- which is what makes this a clean test of
    the MS weight against truth, with no model port required. The fit is ~5000x
    more leveraged on this weight than on the ionization one (NOTES_CGFFIT
    s71), and the two available corrections to it point in OPPOSITE
    directions: Rossi's core width is 14 % below the modelled second moment
    (scale up), while the angular directions' Fisher information is 0.63-0.98
    of the current width (scale down). Only truth can say which helps."""
    wd = os.path.dirname(outfile(arm, sample, ms, chunk, mode, damp, tag))
    if os.path.exists(os.path.join(wd, "wall.txt")) and not force:
        logger.info(f"[{arm}] exists, skipping")
        return 0
    os.makedirs(wd, exist_ok=True)

    # ONE JOB PER OUTPUT DIRECTORY, ENFORCED.
    #
    # Two launches of the same arm write the same globalcor file and the result
    # is a race whose output looks perfectly normal -- a readable ROOT file, a
    # `wall.txt` from whichever finished first, and physics numbers that are
    # partly one job and partly another. It happened twice here (the pt0
    # vanilla arm, then every 0.87/0.75 chunk) because a launcher that appears
    # to have died may simply be two minutes into Geant4 initialisation, and
    # "no directory yet" is not evidence of "no job".
    #
    # The lock records the PID; a stale lock from a killed job is reclaimed,
    # a live one refuses. Cheap, and it removes an entire class of silent
    # corruption from this harness.
    lock = os.path.join(wd, ".running")
    if os.path.exists(lock):
        try:
            old_pid = int(open(lock).read().strip())
            os.kill(old_pid, 0)          # raises unless the process is alive
            logger.error(f"[{arm}] {wd} already has a live job (pid {old_pid}); refusing")
            return 1
        except (ValueError, ProcessLookupError):
            logger.warning(f"[{arm}] reclaiming stale lock in {wd}")
        except PermissionError:
            logger.error(f"[{arm}] {wd} locked by another user's pid; refusing")
            return 1
    open(lock, "w").write(f"{os.getpid()}\n")

    area = AREAS[arm]
    # NO CGF OPTIONS ON EITHER SIDE. Each build runs its own default, which is
    # the comparison: vanilla's default is the alpha-truncated Gaussian weight,
    # the dev build's is the Fisher weight. Passing CgfQoPMode explicitly would
    # not even parse against the vanilla driver.
    src = SAMPLES[sample] if chunk is None else sample_files(sample)[chunk]
    extra = (f"input={src} nEvents={nev} scalarPot3DInitFile={FIELD} "
             f"doKinkFinder=False")
    if mode is not None:
        # `CgfQoPMode` is a ParameterSet parameter, so it goes on the command
        # line rather than into the environment. Mode 3 adds the IRLS
        # re-centring to the weight -- the only piece of this programme that
        # can move the estimator's CENTRE, which is why it is worth a
        # truth-based arm of its own.
        extra = f"{extra} CgfQoPMode={mode}"
    if damp is not None:
        extra = f"{extra} CgfRecentreDamping={damp}"
    cmd = ("source /cvmfs/cms.cern.ch/cmsset_default.sh >/dev/null 2>&1 && "
           f"cd {area}/src && eval $(scramv1 runtime -sh) && cd {wd} && "
           f"exec {area}/src/Analysis/HitAnalyzer/test/cmsswlock.sh run cmsRun "
           f"{area}/src/Analysis/HitAnalyzer/test/runCvhSingleTrackMC.py {extra}")
    env = {k: os.environ[k] for k in _KEEP if k in os.environ}
    env["PATH"] = "/usr/local/bin:/usr/bin:/bin"
    if ms is not None:
        env["CVH_MS_SCALE"] = str(ms)
    logger.info(f"[{arm}] {area}")
    with open(wd + ".log", "w") as fh:
        fh.write(f"### area: {area}\n### cmd: {cmd}\n\n")
        fh.flush()
        p = subprocess.run(["bash", "-c", cmd], stdout=fh, stderr=subprocess.STDOUT, env=env)
    if p.returncode == 0:
        open(os.path.join(wd, "wall.txt"), "w").write("done\n")
    try:
        os.remove(lock)
    except OSError:
        pass
    logger.info(f"[{arm}] rc={p.returncode}")
    return p.returncode


def cmd_run(args):
    for arm in args.arms:
        for ms in (args.ms or [None]):
            for ch in (args.chunks or [None]):
                if run_one(arm, args.nev, args.force, args.sample, ms, ch, args.mode, args.damp, args.tag):
                    raise SystemExit(f"{arm} ms={ms} chunk={ch} failed; see {OUT}/")


_KEY = ["run", "lumi", "event", "trackPt", "trackEta", "trackPhi", "trackCharge"]


def _load(arm, sample="pt8", ms=None, chunks=None, mode=None, damp=None, tag=""):
    """One arm, optionally concatenated over input-file chunks."""
    import uproot
    if chunks:
        import collections
        outs = collections.defaultdict(list)
        for c in chunks:
            f = outfile(arm, sample, ms, c, mode, damp, tag)
            if not os.path.exists(os.path.join(os.path.dirname(f), "wall.txt")):
                logger.warning(f"chunk {c} of ms={ms} incomplete; skipped")
                continue
            a = _load_one(uproot.open(f)["tree"])
            for k, v in a.items():
                outs[k].append(v)
        return {k: np.concatenate(v) for k, v in outs.items()}
    t = uproot.open(outfile(arm, sample, ms, None, mode, damp, tag))["tree"]
    return _load_one(t)


def _load_one(t):
    have = set(t.keys())
    keys = [k for k in _KEY + ["refParms", "genParms", "niter", "edmvalref",
                               "genPt", "genEta"] if k in have]
    return t.arrays(keys, library="np")


def _robust(d):
    """(median, 10 % trimmed mean, 68 % half-width, n)."""
    d = np.sort(np.asarray(d, float))
    n = len(d)
    if n < 4:
        return (np.nan,) * 3 + (n,)
    lo, hi = int(0.1 * n), max(int(0.9 * n), int(0.1 * n) + 1)
    q16, q84 = np.percentile(d, [15.865, 84.135])
    return float(np.median(d)), float(d[lo:hi].mean()), float(0.5 * (q84 - q16)), n


def cmd_cmp(args):
    A = {arm: _load(arm, args.sample) for arm in args.arms}
    key = {arm: list(zip(*[A[arm][k] for k in _KEY])) for arm in A}
    common = set(key[args.arms[0]])
    for arm in args.arms[1:]:
        common &= set(key[arm])
    logger.info(f"tracks common to all arms: {len(common)}")

    print()
    print("=" * 92)
    print("(q/p_fit - q/p_gen) / |q/p_gen|   -- gen-matched J/psi MC, same tracks")
    print("=" * 92)
    print(f"{'arm':>9} {'n':>6} {'BIAS median':>14} {'trim-mean':>13} "
          f"{'WIDTH 68%':>12} {'n(fail)':>8}")
    res = {}
    for arm in args.arms:
        a = A[arm]
        idx = [i for i, k in enumerate(key[arm]) if k in common]
        q = np.array([p[0] for p in a["refParms"]])[idx]
        g = np.array([p[0] for p in a["genParms"]])[idx]
        ok = np.isfinite(q) & np.isfinite(g) & (np.abs(g) > 0)
        d = (q[ok] - g[ok]) / np.abs(g[ok])
        # the fit's own failures, reported rather than silently cut
        nit = np.asarray(a["niter"], float)[idx]
        res[arm] = d
        med, trim, wid, n = _robust(d)
        print(f"{arm:>9} {n:6d} {med:14.4e} {trim:13.4e} {wid:12.4e} "
              f"{int((nit >= 60).sum()):8d}")

    if len(args.arms) != 2:
        return
    a, b = args.arms
    print()
    print(f"{b} - {a}:")
    for name, f in (("bias (median)", lambda x: _robust(x)[0]),
                    ("bias (trim-mean)", lambda x: _robust(x)[1]),
                    ("width (68%)", lambda x: _robust(x)[2])):
        va, vb = f(res[a]), f(res[b])
        rel = (vb - va) / abs(va) if va else float("nan")
        print(f"  {name:18s} {va:12.4e} -> {vb:12.4e}   ({rel:+.2%})")

    # ------------------------------------------------------------------
    # THE PAIRED ANALYSIS, WHICH IS THE ONLY ONE WITH POWER HERE.
    #
    # The two arms are compared UNPAIRED above, and at this sample size that
    # comparison cannot answer the question: the residual width is ~1e-2, so
    # the uncertainty on either arm's bias is ~1e-2/sqrt(N) ~ 4e-4, forty
    # times larger than the difference being looked for. Quoting "the bias
    # improved" from those two numbers would be reading noise.
    #
    # Paired, it is tractable. Write r_b = r_a + delta. Then
    #
    #     Var(r_b) - Var(r_a) = 2 Cov(r_a, delta) + Var(delta),
    #
    # and BOTH terms are measurable because `delta` is small and known per
    # track. Var(delta) is positive by construction -- ANY perturbation of an
    # estimator, right or wrong, widens it at second order -- so the estimator
    # only improves if 2 Cov(r_a, delta) is negative AND beats it. That is the
    # statement to test, and it is what a per-track |residual| comparison
    # cannot say, because a random perturbation makes |r| grow on average too.
    ka = {k: i for i, k in enumerate(key[a])}
    kb = {k: i for i, k in enumerate(key[b])}
    ck = [k for k in common if k in ka and k in kb]
    qa = np.array([A[a]["refParms"][ka[k]][0] for k in ck])
    ga = np.array([A[a]["genParms"][ka[k]][0] for k in ck])
    qb = np.array([A[b]["refParms"][kb[k]][0] for k in ck])
    gb = np.array([A[b]["genParms"][kb[k]][0] for k in ck])
    m = (np.isfinite(qa) & np.isfinite(qb) & np.isfinite(ga) & (np.abs(ga) > 0)
         & (np.abs(ga - gb) < 1e-12 * np.abs(ga)))   # same gen track both sides
    ra, rb = (qa[m] - ga[m]) / np.abs(ga[m]), (qb[m] - gb[m]) / np.abs(gb[m])
    d = rb - ra
    n = len(d)
    # trim the catastrophic tail on the PAIR, symmetric in the two arms
    keep = np.abs(ra) < 5 * _robust(ra)[2]
    rat, dt = ra[keep], d[keep]
    cov = float(np.mean(rat * dt) - np.mean(rat) * np.mean(dt))
    vard = float(np.var(dt))
    dvar = 2 * cov + vard
    # uncertainty on Cov by bootstrap -- the distribution is heavy-tailed and
    # a Gaussian error formula would understate it
    rng = np.random.default_rng(12345)
    bs = np.array([np.cov(rat[i], dt[i])[0, 1]
                   for i in (rng.integers(0, len(rat), len(rat)) for _ in range(400))])
    print()
    print(f"  paired, {n} tracks ({len(rat)} after a 5-sigma trim on {a}):")
    print(f"    median(delta)              {np.median(d):12.4e}")
    print(f"    Var(delta)                 {vard:12.4e}   (>= 0 always)")
    print(f"    2 Cov(r_{a[:3]}, delta)         {2 * cov:12.4e} +- {2 * bs.std():.2e}")
    print(f"    => Var(r_{b[:3]}) - Var(r_{a[:3]})   {dvar:12.4e}"
          f"   ({dvar / np.var(rat):+.3%} of the variance)")
    print(f"    sigma_{a[:3]} = {np.sqrt(np.var(rat)):.4e}   "
          f"sigma_{b[:3]} = {np.sqrt(np.var(rat + dt)):.4e}")


def cmd_mscan(args):
    """Residual against TRUTH as a function of the MS weight scale.

    The baseline arm (no CVH_MS_SCALE) is scale 1.0. For each scanned value the
    table gives the bias and the width against gen, and -- paired on the same
    tracks, which is where the power is -- the change in the residual VARIANCE
    with a bootstrap error. A minimum away from 1.0 means the fit's MS weight
    is mis-set, and its position says by how much and in which direction."""
    base = _load(args.arms[0], args.sample, None, args.chunks)
    rows = [(1.0, base)]
    for ms in args.ms:
        try:
            rows.append((ms, _load(args.arms[0], args.sample, ms, args.chunks)))
        except Exception as e:
            logger.warning(f"ms={ms}: {e}")

    def resid(a, idx):
        q = np.array([a["refParms"][i][0] for i in idx])
        g = np.array([a["genParms"][i][0] for i in idx])
        # gen q/p can be exactly 0 for an unmatched track, and 1/0 -> inf
        # corrupts every percentile downstream. Mask rather than clip: an
        # unmatched track has no truth to be compared against.
        bad = ~np.isfinite(q) | ~np.isfinite(g) | (g == 0)
        r = np.where(bad, np.nan, (q - g) / np.abs(np.where(g == 0, 1.0, g)))
        return r

    key = {ms: list(zip(*[a[k] for k in _KEY])) for ms, a in rows}
    common = set(key[1.0])
    for ms, _ in rows[1:]:
        common &= set(key[ms])
    logger.info(f"tracks common to all arms: {len(common)}")

    idx = {ms: [i for i, k in enumerate(key[ms]) if k in common] for ms, _ in rows}
    # order the index lists identically so the comparison is track by track
    order = {ms: sorted(idx[ms], key=lambda i: key[ms][i]) for ms, _ in rows}
    r0 = resid(rows[0][1], order[1.0])
    good = np.isfinite(r0)
    for ms, a in rows[1:]:
        good &= np.isfinite(resid(a, order[ms]))
    keep = good & (np.abs(np.nan_to_num(r0)) < 5 * _robust(r0[good])[2])

    print()
    print("=" * 96)
    print(f"MS WEIGHT SCALE vs TRUTH   sample={args.sample}   "
          f"{int(keep.sum())} tracks (5-sigma trimmed)")
    print("=" * 96)
    print(f"{'MS scale':>9} {'bias median':>14} {'d(bias)':>12} {'width 68%':>12} "
          f"{'d(width)':>12} {'dVar/Var':>11}")
    print(f"{'':9} {'':14} {'[sigma]':>12} {'':12} {'[sigma]':>12} {'[sigma]':>11}")
    rng = np.random.default_rng(20260824)
    NB = 600
    boot = [rng.integers(0, int(keep.sum()), int(keep.sum())) for _ in range(NB)]
    r0k = None
    for ms, a in rows:
        r = resid(a, order[ms])[keep]
        med, trim, wid, n = _robust(r)
        if ms == 1.0:
            r0k = r
            print(f"{ms:9.2f} {med:14.4e} {'--':>12} {wid:12.4e} {'--':>12} {'--':>11}")
            continue
        # EVERY error bar here is PAIRED: each bootstrap replica resamples
        # TRACKS and recomputes both arms on the same replica, so the enormous
        # common part of the residual cancels and what is left is the effect of
        # the weight. Unpaired errors would be ~1e-3 and would hide everything.
        dmed = np.median(r) - np.median(r0k)
        dwid = _robust(r)[2] - _robust(r0k)[2]
        dvar = float(np.var(r) - np.var(r0k))
        bmed = np.array([np.median(r[i]) - np.median(r0k[i]) for i in boot])
        bwid = np.array([_robust(r[i])[2] - _robust(r0k[i])[2] for i in boot])
        bvar = np.array([np.var(r[i]) - np.var(r0k[i]) for i in boot])
        print(f"{ms:9.2f} {med:14.4e} {dmed / max(bmed.std(), 1e-30):12.1f} "
              f"{wid:12.4e} {dwid / max(bwid.std(), 1e-30):12.1f} "
              f"{dvar / np.var(r0k):11.2e}")
        print(f"{'':9} {'':14} {dmed:12.3e} {'':12} {dwid:12.3e} "
              f"{dvar / max(bvar.std(), 1e-30):11.1f}")
    print()
    print("Two lines per scale: the SIGMA of the paired change, then the change")
    print("itself (and for the variance column, sigma on the second line).")
    print("Negative width/variance change = closer to truth.")


def cmd_dampscan(args):
    """dVar(lambda) for the mode-3 re-centring, and the parabola through it.

    dVar(lambda) = lambda^2 Var(delta_1) + lambda 2Cov_1 IF delta scaled
    linearly, which it does NOT (measured: both terms come in below the naive
    scaling, because the fit re-converges). So the curve is FITTED rather than
    assumed: A lambda^2 + B lambda, with A and B free, minimum at -B/2A.

    Every number is paired and 5-sigma trimmed ON BOTH ARMS. That is not
    cosmetic: untrimmed, two tracks in 35440 flipped the sign of this
    measurement (NOTES_CGFFIT s86)."""
    ch = args.chunks or list(range(75))
    base = _load(args.arms[0], args.sample, None, ch, None, None, args.tag)
    kb = list(zip(*[base[k] for k in _KEY]))

    def resid(a, idx):
        q = np.array([a["refParms"][i][0] for i in idx])
        g = np.array([a["genParms"][i][0] for i in idx])
        ok = np.isfinite(q) & np.isfinite(g) & (g != 0)
        return np.where(ok, (q - g) / np.abs(np.where(g == 0, 1.0, g)), np.nan)

    rng = np.random.default_rng(20260825)
    print()
    print("=" * 92)
    print(f"RE-CENTRING DAMPING vs TRUTH   sample={args.sample}  tag={args.tag or '-'}")
    print("=" * 92)
    print(f"{'lambda':>7} {'n':>7} {'dVar/Var':>12} {'sigma':>7} {'Var(d)/V':>11} "
          f"{'2Cov/V':>11} {'niter':>7}")
    pts = []
    for lam in args.damps:
        try:
            a = _load(args.arms[0], args.sample, None, ch, 3, lam, args.tag)
        except Exception as e:
            logger.warning(f"lambda={lam}: {e}")
            continue
        ka = list(zip(*[a[k] for k in _KEY]))
        ib = {k: i for i, k in enumerate(ka)}
        pr = [(i, ib[k]) for i, k in enumerate(kb) if k in ib]
        r0 = resid(base, [i for i, _ in pr])
        r1 = resid(a, [j for _, j in pr])
        ni = np.asarray(a["niter"], float)[[j for _, j in pr]]
        ok = np.isfinite(r0) & np.isfinite(r1)
        w0, w1 = _robust(r0[ok])[2], _robust(r1[ok])[2]
        keep = ok & (np.abs(np.nan_to_num(r0)) < 5 * w0) & (np.abs(np.nan_to_num(r1)) < 5 * w1)
        x, y = r0[keep], r1[keep]
        d = y - x
        V = np.var(x)
        dv = (np.var(y) - np.var(x)) / V
        bs = [rng.integers(0, len(x), len(x)) for _ in range(400)]
        bdv = np.array([(np.var(y[i]) - np.var(x[i])) / np.var(x[i]) for i in bs])
        pts.append((lam, dv, bdv.std()))
        print(f"{lam:7.2f} {len(x):7d} {dv:12.4e} {dv / max(bdv.std(), 1e-30):7.1f} "
              f"{np.var(d) / V:11.4e} {2 * np.cov(x, d)[0, 1] / V:11.4e} {ni[keep].mean():7.2f}")

    if len(pts) >= 2:
        lam = np.array([p[0] for p in pts])
        dv = np.array([p[1] for p in pts])
        er = np.array([max(p[2], 1e-12) for p in pts])
        # weighted least squares for A lambda^2 + B lambda (no constant: dVar(0) = 0
        # identically, since lambda = 0 IS the baseline arm)
        M = np.vstack([lam ** 2, lam]).T / er[:, None]
        coef, *_ = np.linalg.lstsq(M, dv / er, rcond=None)
        A, B = coef
        print()
        if A > 0:
            lstar = -B / (2 * A)
            print(f"  parabola fit:  A = {A:+.3e}   B = {B:+.3e}")
            print(f"  minimum at lambda* = {lstar:.3f}   dVar/Var = {-B * B / (4 * A):+.3e}")
        else:
            print(f"  parabola fit has A = {A:+.3e} <= 0: no interior minimum, "
                  f"the curve is not convex over the scanned range")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arms", nargs="+", default=["vanilla", "cgf"])
    r.add_argument("--nev", type=int, default=400)
    r.add_argument("--force", action="store_true")
    r.add_argument("--sample", default="pt8", choices=sorted(SAMPLES))
    r.add_argument("--tag", default="", help="campaign tag; suffixes the output dir")
    r.add_argument("--damp", type=float, default=None,
                   help="CgfRecentreDamping (shrinkage on the mode-3 re-centring)")
    r.add_argument("--mode", type=int, default=None,
                   help="CgfQoPMode (1 = weight, 3 = weight + re-centring)")
    r.add_argument("--chunks", nargs="*", type=int, default=None,
                   help="input-file indices; each becomes its own job")
    r.add_argument("--ms", nargs="*", type=float, default=None,
                   help="CVH_MS_SCALE values; each becomes its own arm")
    r.set_defaults(func=cmd_run)
    c = sub.add_parser("cmp")
    c.add_argument("--arms", nargs="+", default=["vanilla", "cgf"])
    c.add_argument("--sample", default="pt8", choices=sorted(SAMPLES))
    c.set_defaults(func=cmd_cmp)

    m = sub.add_parser("mscan")
    m.add_argument("--arms", nargs="+", default=["cgf"])
    m.add_argument("--sample", default="pt0", choices=sorted(SAMPLES))
    m.add_argument("--ms", nargs="+", type=float, default=[0.87, 1.14, 1.30])
    m.add_argument("--chunks", nargs="*", type=int, default=None)
    m.set_defaults(func=cmd_mscan)

    dsc = sub.add_parser("dampscan")
    dsc.add_argument("--arms", nargs="+", default=["cgf"])
    dsc.add_argument("--sample", default="pt0", choices=sorted(SAMPLES))
    dsc.add_argument("--damps", nargs="+", type=float, default=[0.2, 0.4, 0.7])
    dsc.add_argument("--chunks", nargs="*", type=int, default=None)
    dsc.add_argument("--tag", default="conf")
    dsc.set_defaults(func=cmd_dampscan)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
