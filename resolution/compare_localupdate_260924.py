#!/usr/bin/env python3
"""Paired comparison of the two Gauss-Newton linearisation points
(``localUpdate`` False / True) on identical J/psi MC events.

Input: <cmproot>/c<i>_lu{False,True}/ -- one globalcor file + cmsrun.log per
task, same events in both arms.  Reports, per arm, the fit and propagator
failure counters, the Geant4e stale-target resumes, the iteration count and
EDM, and, on the candidates present in both arms, the paired shift of the
refitted mass relative to the generated one.

    python3 compare_localupdate_260924.py [--cmproot DIR]
"""
import argparse
import datetime
import glob
import os
import re

import numpy as np
import uproot

CMPROOT = "/work/submit/david_w/ZMass/scratch_lu_260924/cmp"
OUTDIR = ("/work/submit/david_w/ZMass/calibration_studies/resolution/runs/"
          f"{datetime.date.today().strftime('%y%m%d')}_localupdate_grads")
MJPSI = 3.0969

_LINES = []


def log(msg=""):
    print(msg, flush=True)
    _LINES.append(str(msg))


def counters(logfn):
    out = {}
    with open(logfn, errors="replace") as fh:
        for line in fh:
            if "fit summary" in line or "propagateGenericWithJacobianAltD summary" in line:
                for k, v in re.findall(r"([A-Za-z0-9_\[\]]+)=([0-9.]+)", line):
                    out[k] = out.get(k, 0.) + float(v)
    return out


def load(d):
    f = glob.glob(os.path.join(d, "globalcor_*.root"))[0]
    t = uproot.open(f)["tree"]
    a = t.arrays(["run", "lumi", "event", "niter", "edmval", "edmvalref",
                  "Jpsi_mass", "Jpsigen_mass", "Muplus_pt", "Muminus_pt"], library="np")
    # a candidate is the event plus its two refitted legs
    a["key"] = list(zip(a["run"], a["lumi"], a["event"],
                        np.round(a["Muplus_pt"], 1), np.round(a["Muminus_pt"], 1)))
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmproot", default=CMPROOT)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    arms = {}
    for lu in ("False", "True"):
        dirs = sorted(glob.glob(os.path.join(args.cmproot, f"c*_lu{lu}")))
        c, parts = {}, []
        for d in dirs:
            for k, v in counters(os.path.join(d, "cmsrun.log")).items():
                c[k] = c.get(k, 0.) + v
            parts.append(load(d))
        a = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "key"}
        a["key"] = [k for p in parts for k in p["key"]]
        arms[lu] = (c, a, len(dirs))

    log(f"cmp root: {args.cmproot}")
    log(f"{'':28s}{'localUpdate=False':>20s}{'localUpdate=True':>20s}")
    rows = [("tasks", lambda c, a, n: n),
            ("candidates attempted", lambda c, a, n: c.get("attempted", 0)),
            ("candidates succeeded", lambda c, a, n: c.get("succeeded", 0)),
            ("fail[prop]", lambda c, a, n: c.get("fail[prop]", 0)),
            ("step backtracks", lambda c, a, n: c.get("backtracked[step]", 0)),
            ("propagation calls", lambda c, a, n: c.get("calls", 0)),
            ("prop failures", lambda c, a, n: c.get("failures", 0)),
            ("exit5[offsurface]", lambda c, a, n: c.get("exit5[offsurface]", 0)),
            ("targetResumes", lambda c, a, n: c.get("targetResumes", 0)),
            ("median niter", lambda c, a, n: np.median(a["niter"])),
            ("frac at niter cap (>=10)", lambda c, a, n: np.mean(a["niter"] >= 10)),
            ("median edmvalref", lambda c, a, n: np.median(a["edmvalref"])),
            ("median edmval", lambda c, a, n: np.median(a["edmval"]))]
    for name, fn in rows:
        vals = [fn(*arms[lu]) for lu in ("False", "True")]
        fmt = (lambda v: f"{v:20.3e}") if isinstance(vals[0], float) and abs(vals[0]) < 1e-2 and vals[0] != 0 \
            else (lambda v: f"{v:20.4g}")
        log(f"  {name:26s}" + "".join(fmt(v) for v in vals))

    # paired candidates
    (_, aF, _), (_, aT, _) = arms["False"], arms["True"]
    iT = {k: i for i, k in enumerate(aT["key"])}
    pF, pT = [], []
    for i, k in enumerate(aF["key"]):
        if k in iT:
            pF.append(i)
            pT.append(iT[k])
    pF, pT = np.array(pF), np.array(pT)
    rF = aF["Jpsi_mass"][pF] / aF["Jpsigen_mass"][pF] - 1.
    rT = aT["Jpsi_mass"][pT] / aT["Jpsigen_mass"][pT] - 1.
    d = rT - rF
    n = len(d)
    log()
    log(f"  paired candidates: {n}")
    log(f"  <m/mgen - 1>  False: {1e4 * rF.mean():+.3f} +- {1e4 * rF.std() / np.sqrt(n):.3f} e-4"
        f"   True: {1e4 * rT.mean():+.3f} +- {1e4 * rT.std() / np.sqrt(n):.3f} e-4")
    log(f"  paired shift True-False: mean {1e4 * d.mean():+.4f} +- {1e4 * d.std() / np.sqrt(n):.4f} e-4,"
        f" median {1e4 * np.median(d):+.4f} e-4, rms {1e4 * d.std():.3f} e-4")
    log(f"  resolution rms(m/mgen-1)  False: {1e4 * rF.std():.2f} e-4   True: {1e4 * rT.std():.2f} e-4")
    with open(os.path.join(OUTDIR, "compare_localupdate.txt"), "w") as fh:
        fh.write("\n".join(_LINES) + "\n")
    log(f"\nwritten {OUTDIR}/compare_localupdate.txt")


if __name__ == "__main__":
    main()
