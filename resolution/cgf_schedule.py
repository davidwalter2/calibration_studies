#!/usr/bin/env python3
"""Is the in-fit CGF/IRLS fixed point SCHEDULE-INDEPENDENT?

NOTES_CGFFIT section 34 left exactly one measurement standing between the
scheme and a physics number: the converged fit depends on WHEN the block
CGF is recomputed.  Freezing the block object after the first Gauss-Newton
sweep and recomputing it every sweep converge to points that differ by
rms 1.0e-4 on q/p -- ten times the target precision -- and tightening the EDM
target by five orders of magnitude shrinks that by only 1.6x, so it is not a
stopping-tolerance artefact.

The diagnosis on the page (s34) is that freezing does not freeze only the
Fisher weight `I` (which eq. 14 says drops out of the fixed point) but the
whole block object, INCLUDING the score table `psi_b`, which eq. 14 depends on
directly.  The cached run scores on the iteration-0 reference trajectory, the
refreshing run on the converged one.  If that is right, the disagreement must
fall monotonically to zero as the refresh period N -> 1, and "refresh every N"
with N small enough is the fix.  If it does not, the scheme has more than one
fixed point and needs rethinking.

This module runs that scan and nothing else.  Every arm is the same 25 events
/ 48 tracks of 15_0-native J/psi ALCARECO that the stage-3/5/6 controls used,
with an explicit iteration cap, and every arm pins the five energy-loss
corrections EXPLICITLY (they are C++-default-ON, and a harness that inherits a
default cannot say what it measured).

  python cgf_schedule.py run --tag s34 --cap 60
  python cgf_schedule.py cmp --tag s34 --cap 60

The CGF switches are still environment variables (their C++ readers are in the
maker and in Geant4ePropagator's own lazy statics, not on a ParameterSet), so
they are exported; the physics switches are ParameterSet parameters and are
passed as cmsRun options.  The two must not be confused -- exporting a PSet
name sets nothing and looks like it did.
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cf_track_resolution as ctr                                # noqa: E402
from wums import logging as _wums_logging                        # noqa: E402

logger = _wums_logging.child_logger(__name__)

CMSSW = "/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev"
SRCTEST = os.path.join(CMSSW, "src/Analysis/HitAnalyzer/test")
# runs/ is gitignored; SCRATCH in fisher_norm points at a session scratchpad
# whose contents do not survive (NOTES_CGFFIT s0.4), so outputs land here.
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "cgfsched")

INPUT = ("/ceph/submit/data/group/cms/store/data/Run2016F/Charmonium/ALCARECO/"
         "TkAlJpsiMuMu-21Feb2020_UL2016-v1/20000/"
         "199B282B-2285-CC45-B5E5-5F82FE7ABC22.root")
FIELD = ("/work/submit/david_w/ZMass/mfs/data/fitresults/"
         "polyfit3d_full_coeffs_lmax18_cmsswnorm.txt")

# mode: CVH_CGF_QOP.  0/unset = legacy alpha-truncated variance; 1 = Fisher
# weight only; 3 = weight + IRLS re-centring.
# refresh: CVH_CGF_QOP_REFRESH.  0 = freeze the block object after iteration 0;
# N > 0 = recompute it every N iterations.
CONFIGS = {
    # Mode 0 is the LEGACY Gaussian weight, which with no truncation in the
    # code means the UNTRUNCATED variance -- a diagnostic limit, not a fit
    # configuration. It has to be named explicitly because 1 is the default.
    "off":    dict(mode=0, refresh=0),
    # The DEFAULT: no CGF options passed at all. It must reproduce `m1_frz`,
    # which names mode 1 and freeze explicitly -- that is the test of the flip.
    "dflt":   dict(mode=None, refresh=None),
    # MS-WEIGHT RESPONSE. `CVH_MS_SCALE` multiplies Q's MS variance, so these
    # arms ask what a Fisher treatment of the MS block would be worth WITHOUT
    # building one: the measured 1/I of the angular directions is 0.63-0.98 of
    # the fit's Gaussian width (median 0.87), and the Fisher weight is simply
    # that block scaled by it.
    "ms087":  dict(mode=None, refresh=None, env={"CVH_MS_SCALE": "0.87"}),
    "ms063":  dict(mode=None, refresh=None, env={"CVH_MS_SCALE": "0.63"}),
    "ms099":  dict(mode=None, refresh=None, env={"CVH_MS_SCALE": "0.99"}),
    "ms0999": dict(mode=None, refresh=None, env={"CVH_MS_SCALE": "0.999"}),
    "ms0995": dict(mode=None, refresh=None, env={"CVH_MS_SCALE": "0.995"}),
    "ms101":  dict(mode=None, refresh=None, env={"CVH_MS_SCALE": "1.01"}),
    "m1_frz": dict(mode=1, refresh=0),
    "m1_r10": dict(mode=1, refresh=10),
    "m1_r5":  dict(mode=1, refresh=5),
    "m1_r2":  dict(mode=1, refresh=2),
    "m1_r1":  dict(mode=1, refresh=1),
    # The radiative channel of the block CGF (a ParameterSet switch, not an
    # env one), against the same arm without it.
    "m1_rad": dict(mode=1, refresh=0, opts="CgfRadiativeChannel=1"),
    # Kokoulin in the block CF. It lives entirely inside the exact-delta
    # channel, so every arm here needs IoniExactDelta=1; the pair differs only
    # in IoniKokoulinCgfNbin, which is the bucket count of the CF term.
    "m1_ed":   dict(mode=1, refresh=0,
                    opts="IoniExactDelta=1 IoniKokoulin=1 IoniKokoulinCgfNbin=0"),
    "m1_edk2": dict(mode=1, refresh=0,
                    opts="IoniExactDelta=1 IoniKokoulin=1 IoniKokoulinCgfNbin=2"),
    "m1_edk4": dict(mode=1, refresh=0,
                    opts="IoniExactDelta=1 IoniKokoulin=1 IoniKokoulinCgfNbin=4"),
    "m1_edk8": dict(mode=1, refresh=0,
                    opts="IoniExactDelta=1 IoniKokoulin=1 IoniKokoulinCgfNbin=8"),
    "m1_edk":  dict(mode=1, refresh=0,
                    opts="IoniExactDelta=1 IoniKokoulin=1 IoniKokoulinCgfNbin=16"),
    "s0_frz": dict(mode=1, refresh=0, scalaronly=True),
    "s0_r1":  dict(mode=1, refresh=1, scalaronly=True),
    "m3_frz": dict(mode=3, refresh=0),
    "m3_r10": dict(mode=3, refresh=10),
    "m3_r5":  dict(mode=3, refresh=5),
    "m3_r2":  dict(mode=3, refresh=2),
    "m3_r1":  dict(mode=3, refresh=1),
}

# THE PHYSICS PIN, IDENTICAL IN EVERY ARM.
#
# `IoniExactDelta` is pinned OFF and it is NOT a free choice: with the exact
# knock-on cross section on, the Urban record is in regime 2/3, where the `a3`
# slot holds xi (an energy) rather than a delta-ray collision count, and
# `cvhcgf::blockExponent` has no branch for it -- Geant4ePropagator THROWS
# rather than answer wrongly.  So the in-fit CGF and the exact-delta
# correction are MUTUALLY EXCLUSIVE, even though the C++ default has both on.
# Lifting that is a port of `cf_track_resolution.exact_delta_exponent` into
# `cvhcgf`, plus `beta2`/`etot` on `IoniStep`; until then a scan that pinned it
# on would measure nothing but the throw.
#
# `IoniKokoulin` follows it: the correction lives entirely inside the
# exact-delta channel, so with that off it is a measured no-op, and pinning it
# to 0 keeps the configuration unambiguous rather than merely inert.
#
# The three REFERENCE corrections stay on -- they move the mean trajectory the
# block is built around, which is what "the model should model the simulation"
# buys, and they do not touch the record's regime encoding.
PHYSICS_ON = {
    "CVH_IONI_EXACTDELTA": "0",
    "CVH_IONI_KOKOULIN": "0",
    "CVH_REF_CHARGEAWARE": "1",
    "CVH_REF_SPECIESDEDX": "1",
    "CVH_REF_HADRAD": "1",
}

_KEEP_ENV = ("HOME", "USER", "LOGNAME", "SHELL", "TERM", "HOSTNAME", "TMPDIR",
             "X509_USER_PROXY", "KRB5CCNAME")


def _env(extra):
    """cmsRun must not inherit the calibration_studies venv (it embeds its own
    interpreter and dies mutely on a foreign PYTHONPATH) -- allowlist."""
    e = {k: os.environ[k] for k in _KEEP_ENV if k in os.environ}
    e["PATH"] = "/usr/local/bin:/usr/bin:/bin"
    e.update({k: str(v) for k, v in extra.items() if v is not None})
    return e


def workdir(tag, lab):
    return os.path.join(OUT, f"{tag}_{lab}")


def outfile(tag, lab):
    return os.path.join(workdir(tag, lab), "globalcor_single_0.root")


def run_one(tag, lab, cap, nev, edm, force=False, opts_extra=""):
    cfg = CONFIGS[lab]
    wd = workdir(tag, lab)
    out = outfile(tag, lab)
    # `wall.txt` is the COMPLETION MARKER, written only on rc == 0, and the
    # skip needs both it and the ROOT file. A killed job leaves a truncated
    # ROOT file behind -- 2.6 MB against 3.3 MB, readable, and short by the
    # events that had not been flushed -- so keying the skip on the ROOT file
    # alone silently compares a partial sample against a complete one.
    if os.path.exists(out) and os.path.exists(os.path.join(wd, "wall.txt")) and not force:
        logger.info(f"[{lab}] exists, skipping")
        return 0
    os.makedirs(wd, exist_ok=True)
    # PSet parameters -> cmsRun options; env-only switches -> exported.
    opts, env_extra = ctr.split_switches(dict(PHYSICS_ON))
    assert not env_extra, f"unexpected env residue {env_extra}"
    # An explicit --opts wins over the pin, and must REPLACE it rather than be
    # appended: VarParsing rejects a repeated option ("Multiple assignment")
    # instead of taking the last one, so appending would fail the job rather
    # than override it.
    if opts_extra:
        named = {o.split("=", 1)[0] for o in opts_extra.split() if "=" in o}
        opts = " ".join(o for o in opts.split()
                        if o.split("=", 1)[0] not in named)
    # The MODE and the REFRESH period are ParameterSet parameters -- the CGF
    # weight is the production default, so the estimator has to be in the
    # file's provenance, not in a shell. They go through the same translation
    # as the physics switches. SCALARONLY is a diagnostic and is an
    # environment variable.
    swopts, env = ctr.split_switches({
        **({"CVH_CGF_QOP": cfg["mode"]} if cfg["mode"] is not None else {}),
        **({"CVH_CGF_QOP_REFRESH": cfg["refresh"]} if cfg["refresh"] is not None else {}),
    })
    opts = f"{opts} {swopts}" if swopts else opts
    if cfg.get("scalaronly"):
        env["CVH_CGF_QOP_SCALARONLY"] = 1
    env.update(cfg.get("env", {}))
    # A config's own cmsRun options win over the pin in the same way --opts
    # does, and for the same reason (VarParsing refuses a repeat).
    cfg_opts = cfg.get("opts", "")
    if cfg_opts:
        named = {o.split("=", 1)[0] for o in cfg_opts.split() if "=" in o}
        opts = " ".join(o for o in opts.split() if o.split("=", 1)[0] not in named)
    extra = (f"input={INPUT} nEvents={nev} scalarPot3DInitFile={FIELD} "
             f"nIters={cap} edmConvergence={edm} {opts} {cfg_opts} {opts_extra}")
    cmd = ("source /cvmfs/cms.cern.ch/cmsset_default.sh >/dev/null 2>&1 && "
           f"cd {CMSSW}/src && eval $(scramv1 runtime -sh) && cd {wd} && "
           f"exec {SRCTEST}/cmsswlock.sh run cmsRun {SRCTEST}/runCvhSingleTrack.py {extra}")
    log = wd + ".log"
    logger.info(f"[{lab}] env={env} cap={cap} -> {out}")
    t0 = time.time()
    with open(log, "w") as fh:
        fh.write(f"### cmd: {cmd}\n### env: {env}\n\n")
        fh.flush()
        p = subprocess.run(["bash", "-c", cmd], stdout=fh,
                           stderr=subprocess.STDOUT, env=_env(env))
    dt = time.time() - t0
    # Wall time is a MEASUREMENT here (the CGF's cost is one of the open
    # questions), so it is written next to the output rather than left in a
    # terminal that will be gone. It is only comparable across arms run with
    # the same number of concurrent jobs. It doubles as the completion marker,
    # hence rc == 0 only.
    if p.returncode == 0:
        with open(os.path.join(wd, "wall.txt"), "w") as fh:
            fh.write(f"{dt:.1f}\n")
    logger.info(f"[{lab}] rc={p.returncode} wall={dt:.1f}s")
    return p.returncode


def cmd_run(args):
    os.makedirs(OUT, exist_ok=True)
    labs = args.labels or list(CONFIGS)
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        rcs = list(ex.map(lambda l: run_one(args.tag, l, args.cap, args.nev,
                                            args.edm, args.force, args.opts), labs))
    bad = [l for l, rc in zip(labs, rcs) if rc]
    if bad:
        raise SystemExit(f"failed: {bad} (see {OUT}/<tag>_<label>.log)")


# ----------------------------------------------------------------------
# comparison
# ----------------------------------------------------------------------
_TRKKEYS = ["run", "lumi", "event", "trackPt", "trackEta", "trackPhi",
            "trackCharge"]


def _load(path):
    import uproot
    t = uproot.open(path)["tree"]
    have = set(t.keys())
    keys = [k for k in _TRKKEYS + ["refParms", "niter", "edmval", "edmvalref",
                                   "chisqval", "ndof"] if k in have]
    return t.arrays(keys, library="np")


def _match(A, B):
    def key(a):
        return list(zip(*[a[k] for k in _TRKKEYS]))
    ka, kb = key(A), key(B)
    ib = {k: i for i, k in enumerate(kb)}
    pairs = [(i, ib[k]) for i, k in enumerate(ka) if k in ib]
    return (np.array([p[0] for p in pairs], int),
            np.array([p[1] for p in pairs], int))


def _qop(a):
    return np.array([p[0] for p in a["refParms"]], float)


def _arm(spec, deftag):
    """'label' -> (deftag, label); 'tag:label' -> (tag, label).

    The tag prefix is what makes a CROSS-TAG comparison expressible -- the
    alpha scan runs the same two labels under several tags, and the question
    is how each label moves BETWEEN tags, not how the labels differ within
    one."""
    if ":" in spec:
        t, l = spec.split(":", 1)
        return t, l
    return deftag, spec


def cmd_cmp(args):
    specs = args.labels or [l for l in CONFIGS if os.path.exists(outfile(args.tag, l))]
    D = {}
    for spec in specs:
        t, l = _arm(spec, args.tag)
        f = outfile(t, l)
        if not os.path.exists(f):
            logger.warning(f"missing {f}")
            continue
        # Same completion marker as the skip logic: a job still writing has a
        # ROOT file with no tree in it yet, and reading one raises deep inside
        # uproot rather than saying which arm is not finished.
        if not os.path.exists(os.path.join(workdir(t, l), "wall.txt")):
            logger.warning(f"{t}/{l} has no wall.txt -- still running or failed; skipped")
            continue
        D[spec if ":" in spec else l] = _load(f)
    if not D:
        raise SystemExit("nothing to compare")

    print("=" * 96)
    print(f"CONVERGENCE   tag={args.tag}  cap={args.cap}  edm target={args.edm}")
    print("=" * 96)
    print(f"{'label':10s} {'ntrk':>5s} {'mean':>7s} {'med':>5s} {'p90':>5s} "
          f"{'max':>5s} {'at cap':>7s} {'edm>tgt':>8s} {'mean|edm|':>11s}")
    conv = {}
    for l, a in D.items():
        ni, ed = np.asarray(a["niter"], float), np.abs(np.asarray(a["edmvalref"], float))
        ok = (ni < args.cap) & (ed < args.edm)
        conv[l] = ok
        print(f"{l:10s} {len(ni):5d} {ni.mean():7.2f} {np.median(ni):5.0f} "
              f"{np.percentile(ni, 90):5.0f} {ni.max():5.0f} "
              f"{int((ni >= args.cap).sum()):3d}/{len(ni):<3d} "
              f"{int((ed >= args.edm).sum()):4d}/{len(ni):<3d} {ed.mean():11.3e}")

    ref = args.ref if args.ref in D else list(D)[0]
    print()
    print("=" * 96)
    print(f"FITTED q/p, RELATIVE, against `{ref}`   (converged tracks in BOTH arms)")
    print("=" * 96)
    print(f"{'label':10s} {'n':>4s} {'mean':>12s} {'rms':>12s} {'max|d|':>12s}")
    A = D[ref]
    for l, B in D.items():
        i, j = _match(A, B)
        keep = conv[ref][i] & conv[l][j]
        i, j = i[keep], j[keep]
        d = (_qop(B)[j] - _qop(A)[i]) / np.abs(_qop(A)[i])
        if not len(d):
            print(f"{l:10s} {0:4d}  (no common converged tracks)")
            continue
        print(f"{l:10s} {len(d):4d} {d.mean():12.4e} {np.sqrt((d**2).mean()):12.4e} "
              f"{np.abs(d).max():12.4e}")

    # the schedule ladder: everything against the self-consistent arm
    if args.ladder and "m3_r1" in D:
        print()
        print("=" * 96)
        print("SCHEDULE LADDER: mode-3 arms against `m3_r1` (recompute every sweep)")
        print("=" * 96)
        A = D["m3_r1"]
        print(f"{'label':10s} {'n':>4s} {'rms':>12s} {'max|d|':>12s}")
        for l in ["m3_frz", "m3_r10", "m3_r5", "m3_r2"]:
            if l not in D:
                continue
            B = D[l]
            i, j = _match(A, B)
            keep = conv["m3_r1"][i] & conv[l][j]
            i, j = i[keep], j[keep]
            d = (_qop(B)[j] - _qop(A)[i]) / np.abs(_qop(A)[i])
            print(f"{l:10s} {len(d):4d} {np.sqrt((d**2).mean()):12.4e} "
                  f"{np.abs(d).max():12.4e}")


# ----------------------------------------------------------------------
# drift: how much does the BLOCK ITSELF move between fit iterations?
# ----------------------------------------------------------------------
# The schedule question of NOTES_CGFFIT s34 is usually posed as "does freezing
# the weight change the fixed point", but the quantity that decides it is more
# basic: how much does the block CGF move when the SAME leg is recomputed on
# the next iteration's reference trajectory? The natural guess is O(1e-4) --
# the relative motion of the leg's momentum -- and it is larger than that,
# because the Geant4 step subdivision is not stable under that motion, and
# `1/I` is a property of the pooled STEP RECORD, not of a smooth function of
# the momentum.
#
# Input: a cmsRun log from a CVH_CGF_QOP=2 (diagnostic) job with
# CVH_CGF_QOP_REFRESH=1 (so the block is recomputed every iteration). The
# propagator emits one `### CVHCGF` line per leg carrying the end-of-leg
# position, and the maker emits `### CVHITER` at the top of each Gauss-Newton
# sweep; the two together identify (track, iteration, leg).


def _parse_kv(line):
    out = {}
    for tok in line.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def cmd_drift(args):
    import collections
    blocks = collections.defaultdict(dict)   # (itrack, legkey) -> {iiter: rec}
    itrack, iiter = -1, -1
    legs_this_iter = []
    for line in open(args.log):
        if line.startswith("### CVHITER"):
            d = _parse_kv(line)
            itrack, iiter = int(d["itrack"]), int(d["iiter"])
            legs_this_iter = []
            continue
        if not line.startswith("### CVHCGF"):
            continue
        d = _parse_kv(line)
        if "endz" not in d:
            raise SystemExit("log predates the endz/endr diagnostic; re-run with "
                             "the current build")
        # The leg's end point is on a module surface and moves by microns
        # between iterations, while neighbouring legs are centimetres apart.
        key = (round(d["endz"] / args.legtol), round(d["endr"] / args.legtol))
        if key in legs_this_iter:      # same surface twice in one sweep: keep both
            key = key + (legs_this_iter.count(key),)
        legs_this_iter.append(key)
        blocks[(itrack, key)][iiter] = d

    # THE QUANTITY IS THE PHYSICAL WEIGHT, `Qcgf`, NOT `invI_z`.
    #
    # `invI_z` is 1/I in units of the internal standardization sigma^2, so its
    # drift is a statement about that sigma convention as much as about the
    # block. Reading it instead makes the block look like it moves by up to
    # 20 % between iterations, which invites blaming Geant4 re-stepping, while
    # the physical weight tracks the LEGACY VARIANCE to 0.996 -- i.e. the leg
    # itself is changing, exactly as it does for the weight the fit already
    # uses. `Qnom` is carried alongside for precisely that comparison.
    rows = []
    for (trk, key), byiter in blocks.items():
        if len(byiter) < 2:
            continue
        ii = sorted(byiter)

        def spr(k):
            v = np.array([byiter[i][k] for i in ii])
            return (v.max() - v.min()) / v.mean() if v.mean() else float("nan")

        q = np.array([byiter[i]["Qcgf"] for i in ii])
        nst = np.array([byiter[i]["nstep"] for i in ii])
        rows.append(dict(trk=trk, n=len(ii),
                         spread=spr("Qcgf"),
                         qnom_spread=spr("Qnom"),
                         invI_spread=spr("invI_z"),
                         first_last=abs(q[-1] - q[0]) / q.mean(),
                         nstep_spread=(nst.max() - nst.min()) / max(nst.mean(), 1),
                         zmode_spread=spr("zmode"),
                         invI0=q[0]))
    if not rows:
        raise SystemExit("no leg seen in two or more iterations -- was "
                         "CVH_CGF_QOP_REFRESH=1 set?")
    sp = np.array([r["spread"] for r in rows])
    qn = np.array([r["qnom_spread"] for r in rows])
    iz = np.array([r["invI_spread"] for r in rows])
    fl = np.array([r["first_last"] for r in rows])
    ns = np.array([r["nstep_spread"] for r in rows])
    zm = np.array([r["zmode_spread"] for r in rows])
    print("=" * 84)
    print(f"BLOCK DRIFT ACROSS FIT ITERATIONS   {len(rows)} legs, "
          f"{len(set(r['trk'] for r in rows))} tracks, {args.log}")
    print("=" * 84)
    print(f"{'quantity':28s} {'median':>10s} {'p90':>10s} {'max':>10s}")
    for name, v in (("(max-min)/mean  Qcgf", sp),
                    ("(max-min)/mean  Qnom  [legacy]", qn),
                    ("(max-min)/mean  invI_z [convention]", iz),
                    ("|last-first|/mean  Qcgf", fl),
                    ("(max-min)/mean  nstep", ns), ("(max-min)/mean  zmode", zm)):
        print(f"{name:28s} {np.median(v):10.3e} {np.percentile(v, 90):10.3e} "
              f"{v.max():10.3e}")
    ratio = sp / np.maximum(qn, 1e-18)
    print()
    print(f"drift(Qcgf)/drift(Qnom): median {np.median(ratio):.3f}  "
          f"p90 {np.percentile(ratio, 90):.3f}   <- 1.0 means the Fisher weight is")
    print("   exactly as stable under the fit's own iteration as the weight the fit")
    print("   already uses, i.e. the motion is the LEG changing and not the CGF.")

    # Is the tail the RE-STEPPING? Split on whether the leg's Geant4 step count
    # changed at all. If the two groups have the same 1/I spread, the step
    # count is a coincidence and the drift is something else.
    same = np.array([r["spread"] for r in rows if r["nstep_spread"] == 0.])
    moved = np.array([r["spread"] for r in rows if r["nstep_spread"] > 0.])
    print()
    print(f"{'group':28s} {'legs':>5s} {'median':>10s} {'p90':>10s} {'max':>10s}")
    for name, v in (("step count CONSTANT", same), ("step count CHANGED", moved)):
        if not len(v):
            print(f"{name:28s} {0:5d}")
            continue
        print(f"{name:28s} {len(v):5d} {np.median(v):10.3e} "
              f"{np.percentile(v, 90):10.3e} {v.max():10.3e}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--tag", default="s34")
    r.add_argument("--labels", nargs="*")
    r.add_argument("--cap", type=int, default=60)
    r.add_argument("--nev", type=int, default=25)
    r.add_argument("--edm", type=float, default=1e-5)
    r.add_argument("--jobs", type=int, default=4)
    r.add_argument("--force", action="store_true")
    r.add_argument("--opts", default="",
                   help="extra cmsRun options, appended AFTER the physics pin so "
                        "they win (e.g. 'IoniExactDelta=1 IoniKokoulin=1')")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("cmp")
    c.add_argument("--tag", default="s34")
    c.add_argument("--labels", nargs="*")
    c.add_argument("--cap", type=int, default=60)
    c.add_argument("--edm", type=float, default=1e-5)
    c.add_argument("--ref", default="m3_frz")
    c.add_argument("--ladder", action="store_true", default=True)
    c.set_defaults(func=cmd_cmp)

    d = sub.add_parser("drift")
    d.add_argument("--log", required=True,
                   help="cmsRun log from a CVH_CGF_QOP=2, CVH_CGF_QOP_REFRESH=1 job")
    d.add_argument("--legtol", type=float, default=1.0,
                   help="mm; end-of-leg positions are binned at this scale to "
                        "identify the same leg across iterations")
    d.set_defaults(func=cmd_drift)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
