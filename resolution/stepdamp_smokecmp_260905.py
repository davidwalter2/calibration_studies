#!/usr/bin/env python3
"""Compare the step-damping smoke variants: counters, chi2, niter, masses.

usage: python3 stepdamp_smokecmp_260905.py <dir1> <dir2> ...
each dir holds st/ and/or tt/ with the CVH output + local.log.
"""
import os
import re
import sys

import numpy as np
import uproot
import prodfiles

RE_SUM = re.compile(r"(\w+(?:\[\w+\])?)=([-\d.e+]+)")


def counters(log, key):
    out = {}
    if not os.path.exists(log):
        return out
    for line in open(log, errors="replace"):
        if key in line:
            for k, v in RE_SUM.findall(line):
                try:
                    out[k] = int(float(v))
                except ValueError:
                    pass
    return out


def logcounts(log):
    c = dict(clamp=0, bt=0, clamp0=0, abort=0, nan=0)
    if not os.path.exists(log):
        return c
    for line in open(log, errors="replace"):
        if "GN step clamped" in line:
            c["clamp"] += 1
            if "scale = 0 " in line:
                c["clamp0"] += 1
        elif "GN step backtracked" in line:
            c["bt"] += 1
        elif "Abort: Propagation Failed" in line:
            c["abort"] += 1
        elif "Abort: non-finite" in line:
            c["nan"] += 1
    return c


def tt_stats(fn):
    br = ["Jpsi_mass", "Jpsigen_mass", "Jpsi_sigmamass", "Jpsikin_mass",
          "chisqval", "ndof", "niter", "Muplus_pt", "Muminus_pt",
          "Muplus_eta", "Muminus_eta"]
    a = uproot.open(fn)["tree"].arrays(br, library="np")
    d = {k: np.asarray(v, dtype=np.float64) for k, v in a.items()}
    nc2 = d["chisqval"] / np.maximum(d["ndof"], 1)
    ptmin = np.minimum(d["Muplus_pt"], d["Muminus_pt"])
    frozen = np.abs(d["Jpsi_mass"] - d["Jpsikin_mass"]) < 1e-6
    dm = d["Jpsi_mass"] - d["Jpsigen_mass"]
    ok = np.isfinite(dm) & (d["Jpsi_sigmamass"] > 0) & (d["Jpsi_sigmamass"] < 0.15)
    z = np.where(ok, dm / np.maximum(d["Jpsi_sigmamass"], 1e-12), np.nan)
    s = dict(n=len(nc2),
             c2_q50=np.quantile(nc2, .5), c2_q90=np.quantile(nc2, .9),
             c2_q99=np.quantile(nc2, .99), c2_max=nc2.max(),
             f_c2gt10=np.mean(nc2 > 10), f_c2gt3=np.mean(nc2 > 3),
             frozen=frozen.mean(), atcap=np.mean(d["niter"] >= 10),
             niter_mean=d["niter"].mean(),
             dm_all=np.nanmean(dm[ok]) * 1e3,
             dm_pt2=np.nanmean(dm[ok & (ptmin < 2)]) * 1e3 if (ok & (ptmin < 2)).sum() else np.nan,
             odd=np.nanmean(z[ok] * np.exp(-0.05 * z[ok] ** 2)),
             odd_pt2=(np.nanmean((z * np.exp(-0.05 * z ** 2))[ok & (ptmin > 2)])
                      if (ok & (ptmin > 2)).sum() else np.nan))
    return s


def st_stats(fn):
    br = ["trackPt", "trackEta", "trackCharge", "chisqval", "ndof", "niter"]
    t = uproot.open(fn)["tree"]
    have = [b for b in br if b in t.keys()]
    d = {k: np.asarray(v, dtype=np.float64) for k, v in t.arrays(have, library="np").items()}
    nc2 = d["chisqval"] / np.maximum(d["ndof"], 1)
    return dict(n=len(nc2), c2_q50=np.quantile(nc2, .5), c2_q90=np.quantile(nc2, .9),
                c2_q99=np.quantile(nc2, .99), c2_max=nc2.max(),
                f_c2gt10=np.mean(nc2 > 10), atcap=np.mean(d["niter"] >= 10),
                niter_mean=d["niter"].mean())


def main():
    dirs = sys.argv[1:]
    # a STEM per kind, not a file name: these smokes run numberOfThreads=1, so
    # single_file finds the one stream without naming index 0 (and warns if the
    # directory holds several, which would compare 1/N of each smoke).
    for kind, stem, key, stat in (
            ("tt", "globalcor", "ResidualGlobalCorrectionMakerTwoTrackG4e fit summary", tt_stats),
            ("st", "globalcor_resclosure", "ResidualGlobalCorrectionMakerG4e fit summary", st_stats)):
        rows = []
        for d in dirs:
            fn = prodfiles.single_file(os.path.join(d, kind), stem)
            log = os.path.join(d, kind, "local.log")
            if not os.path.exists(fn):
                continue
            mk = counters(log, key)
            pr = counters(log, "propagateGenericWithJacobianAltD summary")
            lc = logcounts(log)
            try:
                st = stat(fn)
            except Exception as e:
                st = {"n": -1, "err": f"{type(e).__name__}: {e}"}
            rows.append((os.path.basename(d.rstrip("/")), mk, pr, lc, st))
        if not rows:
            continue
        print(f"\n############ {kind} ############")
        for nm, mk, pr, lc, st in rows:
            print(f"-- {nm}")
            print("   maker : " + "  ".join(f"{k}={v}" for k, v in mk.items()))
            print("   prop  : " + "  ".join(f"{k}={v}" for k, v in pr.items()
                                            if k in ("calls", "failures", "exit1[plimit]",
                                                     "exit2[ierr]", "exit3[maxlen]",
                                                     "exit4[pdrain]")))
            print(f"   lines : clamp={lc['clamp']} (scale0 {lc['clamp0']})  bt={lc['bt']}  "
                  f"abort={lc['abort']}  nan={lc['nan']}")
            print("   tree  : " + "  ".join(
                f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in st.items()))


if __name__ == "__main__":
    main()
