#!/usr/bin/env python3
"""Failure taxonomy of the CVH Gauss-Newton fit, from the job logs.

Failed fits never reach the tree, so the only record of them is the maker's
`Abort: Propagation Failed!` line and the propagator's `Geant4e fail[...]`
line that immediately precedes it. Those two together carry everything the
classification needs:

  * `fail[pdrain]` prints the momentum of the leg AT ITS ENTRY (`pT0`,`eta0`),
    i.e. where the CURRENT FIT STATE sits, plus where along the leg it drained
    (`pathLen`, `r`, `z`).
  * the maker's abort prints the SEED (q, pt, eta) of each track, i.e. where the
    standard KF fit -- which is a faithful proxy for the true momentum on a
    gun sample -- says the track actually is.

Classification of a failing leg:
  STOP   the fit state agrees with the seed (p0 >= frac * p_seed) and the seed
         momentum is itself too low to cross the tracker material
         (p_seed < pstop): a physically unfittable track, no step control can
         save it.
  RUN    the fit state has been driven far below the seed (p0 < frac*p_seed):
         a runaway Gauss-Newton step, which is exactly what step damping is
         supposed to prevent.
  HARD   neither: the seed is soft AND the state has moved -- reported apart.

usage:
  python3 stepdamp_failtax_260905.py --tag jpsigun_ul16_260905d_m0 [--ntasks N]
  python3 stepdamp_failtax_260905.py --logs 'path/glob/*.log'
"""
import argparse
import glob
import os
import re
import sys
from collections import Counter

import numpy as np

CEPH = "/ceph/submit/data/user/d/david_w/ZMass/cvh"

RE_PDRAIN = re.compile(
    r"Geant4e fail\[pdrain\]\s+p=(\S+)\s+plimit=(\S+)\s+pT0=(\S+)\s+eta0=(\S+)\s+"
    r"charge=(\S+)\s+r=(\S+)\s+z=(\S+)\s+surf_r=(\S+)\s+surf_z=(\S+)\s+iter=(\S+)\s+"
    r"pathLen=(\S+)\s+particle=(\S+)")
RE_PLIMIT = re.compile(r"Geant4e fail\[plimit\]\s+p=(\S+)\s+plimit=(\S+)\s+charge=(\S+)")
RE_ABORT_TT = re.compile(
    r"Abort: Propagation Failed!\s+icons = (\S+)\s+iiter = (\S+)\s+id = (\S+)\s+ihit = (\S+)\s+"
    r"seed0: q=(\S+) pt=(\S+) eta=(\S+)\s+seed1: q=(\S+) pt=(\S+) eta=(\S+)")
RE_ABORT_ST = re.compile(
    r"Abort: Propagation Failed!\s+iiter = (\S+)\s+ihit = (\S+)\s+seed: q=(\S+) pt=(\S+) eta=(\S+)")
RE_CLAMP = re.compile(r"GN step clamped[^:]*: .*?scale = (\S+)")
RE_BT = re.compile(r"GN step backtracked")
RE_SUM = re.compile(r"(\w+(?:\[\w+\])?)=([-\d.e+]+)")


def parse_log(fn):
    """Return (records, counters). One record per maker-level propagation abort."""
    recs = []
    cnt = Counter()
    clampscales = []
    last_fail = None
    try:
        fh = open(fn, "r", errors="replace")
    except OSError:
        return recs, cnt, clampscales
    with fh:
        for line in fh:
            if "Geant4e fail[" in line:
                m = RE_PDRAIN.search(line)
                if m:
                    last_fail = ("pdrain", float(m.group(3)), float(m.group(4)),
                                 float(m.group(1)), float(m.group(2)),
                                 float(m.group(11)), float(m.group(6)), float(m.group(7)))
                    cnt["pdrain"] += 1
                    continue
                m = RE_PLIMIT.search(line)
                if m:
                    last_fail = ("plimit", np.nan, np.nan, float(m.group(1)),
                                 float(m.group(2)), np.nan, np.nan, np.nan)
                    cnt["plimit"] += 1
                    continue
                last_fail = ("other", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
                cnt["otherfail"] += 1
            elif "Abort: Propagation Failed!" in line:
                m = RE_ABORT_TT.search(line)
                if m:
                    did = int(m.group(3))
                    pt = float(m.group(6) if did == 0 else m.group(9))
                    eta = float(m.group(7) if did == 0 else m.group(10))
                    iiter = int(m.group(2))
                else:
                    m = RE_ABORT_ST.search(line)
                    if not m:
                        continue
                    pt, eta, iiter = float(m.group(4)), float(m.group(5)), int(m.group(1))
                pseed = pt * np.cosh(eta)
                kind, pT0, eta0, pnow, plim, plen, rr, zz = (
                    last_fail if last_fail else ("none",) + (np.nan,) * 7)
                p0 = pT0 * np.cosh(eta0) if np.isfinite(pT0) else np.nan
                recs.append((kind, pseed, p0, pnow, plim, plen, rr, zz, iiter))
                last_fail = None
                cnt["abort"] += 1
            elif "GN step clamped" in line:
                m = RE_CLAMP.search(line)
                if m:
                    clampscales.append(float(m.group(1)))
                cnt["clampline"] += 1
            elif "GN step backtracked" in line:
                cnt["btline"] += 1
            elif "fit summary" in line:
                for k, v in RE_SUM.findall(line):
                    cnt["MK:" + k] += int(float(v))
                cnt["MK:lines"] += 1
            elif "propagateGenericWithJacobianAltD summary" in line:
                for k, v in RE_SUM.findall(line):
                    cnt["PR:" + k] += int(float(v))
                cnt["PR:lines"] += 1
    return recs, cnt, clampscales


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", help="production tag under resolution_trackres_<tag>")
    ap.add_argument("--logs", help="explicit glob of local.log files")
    ap.add_argument("--ntasks", type=int, default=0)
    ap.add_argument("--pstop", type=float, default=1.5,
                    help="seed |p| below which a muon cannot cross the tracker [GeV]")
    ap.add_argument("--frac", type=float, default=0.5,
                    help="p0/p_seed below which the fit state counts as run away")
    ap.add_argument("--nproc", type=int, default=32)
    a = ap.parse_args()

    if a.logs:
        files = sorted(glob.glob(a.logs))
    else:
        files = sorted(glob.glob(f"{CEPH}/resolution_trackres_{a.tag}/task_*/local.log"))
        files = [f for f in files
                 if os.path.exists(os.path.join(os.path.dirname(f), ".complete"))]
    if a.ntasks:
        files = files[:a.ntasks]
    if not files:
        sys.exit("no logs matched")
    print(f"# {len(files)} logs")

    from multiprocessing import Pool
    with Pool(min(a.nproc, len(files))) as p:
        out = p.map(parse_log, files)
    recs = [r for o in out for r in o[0]]
    cnt = Counter()
    for o in out:
        cnt.update(o[1])
    scales = np.array([s for o in out for s in o[2]], dtype=float)

    print("\n== maker summary (summed)")
    print("   " + "  ".join(f"{k[3:]}={v}" for k, v in cnt.items() if k.startswith("MK:")))
    print("== propagator summary (summed)")
    print("   " + "  ".join(f"{k[3:]}={v}" for k, v in cnt.items() if k.startswith("PR:")))
    print(f"== step-control print lines: clamp={cnt['clampline']} backtrack={cnt['btline']}")
    if scales.size:
        print("   clamp scale q05/q50/q95/min/max = "
              + " ".join(f"{np.quantile(scales, q):.4g}" for q in (.05, .5, .95))
              + f" {scales.min():.4g} {scales.max():.4g}"
              + f"   frac at exactly 0: {np.mean(scales == 0.):.4f}")

    if not recs:
        print("\n== no maker-level propagation aborts in these logs")
        return
    kind = np.array([r[0] for r in recs])
    pseed = np.array([r[1] for r in recs], dtype=float)
    p0 = np.array([r[2] for r in recs], dtype=float)
    plim = np.array([r[4] for r in recs], dtype=float)
    plen = np.array([r[5] for r in recs], dtype=float)
    iiter = np.array([r[8] for r in recs], dtype=float)
    n = len(recs)
    plim0 = np.nanmedian(plim) if np.isfinite(plim).any() else 0.2

    moved = np.where(np.isfinite(p0), p0 < a.frac * pseed, False)
    softseed = pseed < a.pstop
    cls = np.where(moved, "RUN", np.where(softseed, "STOP", "HARD"))

    print(f"\n== {n} maker-level propagation aborts   (plimit = {plim0:g} GeV)")
    print(f"   exit kind: " + "  ".join(f"{k}={int((kind == k).sum())}"
                                        for k in sorted(set(kind))))
    for k in ("STOP", "RUN", "HARD"):
        s = cls == k
        if not s.sum():
            print(f"   {k:5s}  0")
            continue
        print(f"   {k:5s}  {int(s.sum()):6d} ({100.*s.mean():6.2f} %)   "
              f"p_seed med {np.median(pseed[s]):7.3f}  "
              f"p_state med {np.nanmedian(p0[s]):7.3f}  "
              f"p0/pseed med {np.nanmedian(p0[s]/pseed[s]):7.3f}  "
              f"iiter med {np.median(iiter[s]):.1f}  "
              f"pathLen med {np.nanmedian(plen[s]):.1f}")
    print("\n   p_seed quantiles (all aborts): "
          + " ".join(f"{np.quantile(pseed, q):.3f}" for q in (.05, .25, .5, .75, .95)))
    for lo, hi in ((0, .3), (.3, .45), (.45, 1.), (1., 3.), (3., 1e9)):
        s = (pseed >= lo) & (pseed < hi)
        if s.sum():
            print(f"   p_seed [{lo:g},{hi:g}) : {int(s.sum()):6d}  "
                  f"RUN {int((s & (cls == 'RUN')).sum()):5d}  "
                  f"STOP {int((s & (cls == 'STOP')).sum()):5d}  "
                  f"HARD {int((s & (cls == 'HARD')).sum()):5d}")


if __name__ == "__main__":
    main()
