#!/usr/bin/env python3
"""Inspect one CVH production output: schema, per-branch volume, global map.

Written for the pre-submission smoke of a production, where the questions are
always the same and always worth answering before 1000 tasks go out:

  * are the FACTORED-Hessian branches there and the packed one absent
    (fillGradsFactored=True, fillGrads=False)?
  * are the raw step-record branches absent (exportStepRecords=False)?
  * are the in-maker CF exponents there, and for what fraction of candidates
    is `cfmass_ok` true?
  * what is the real bytes/candidate, and WHICH branches carry it -- the
    single number is useless for sizing a production if one branch is 80 % of
    it and is about to be switched off
  * how many global parameters does the run tree declare, and how do they
    split by parmtype?  Two productions can only be POOLED in one global fit
    if this map is identical, so it is printed in a form that can be diffed
    between two files (`--compare other.root`).

usage:
    python smoke_inspect.py <one globalcor_*.root> [--compare <other.root>] [--top N]
"""
import argparse
import collections
import sys

import numpy as np
import uproot

# The raw per-step resolution export (exportStepRecords=True). Their only
# consumer was the offline exponent extractor that the in-maker cfmass_*
# export replaces; ~290 kB/candidate.
STEP_RECORD = ("ioniurbanidx", "ioniurbanv", "radstepidx", "radstepv",
               "radstepspecv", "msmoliidx", "msmoliv", "reseigv",
               "resinfv", "resinfbv")


def _fmt(n):
    for unit in ("B", "kB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return "%.1f %s" % (n, unit)
        n /= 1024.


def runtree_bytes(f):
    """Compressed bytes of the run tree -- a FIXED per-FILE cost.

    It is the global parameter catalogue (one entry per global parameter, ~126k
    here), written once per output file regardless of how many candidates the
    file holds. Sizing a production from bytes/candidate alone is wrong by this
    amount times the number of tasks.
    """
    if "runtree" not in f:
        return 0
    return sum(int(getattr(f["runtree"][n], "compressed_bytes", 0) or 0)
               for n in f["runtree"].keys())


def parmtype_map(f):
    """{parmtype: count} from the run tree, plus the total."""
    if "runtree" not in f:
        return None, None
    rt = f["runtree"]
    if "parmtype" not in rt:
        return None, int(rt.num_entries)
    pt = rt["parmtype"].array(library="np")
    # One entry per global parameter (flat), or one array per entry.
    if pt.dtype == object:
        pt = np.concatenate([np.asarray(a) for a in pt]) if len(pt) else np.array([])
    return collections.Counter(pt.tolist()), int(len(pt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--compare", default=None,
                    help="second file whose global parameter map must match")
    ap.add_argument("--top", type=int, default=15, help="branches to list")
    a = ap.parse_args()

    f = uproot.open(a.path)
    print("file      : %s" % a.path)
    print("trees     : %s" % ", ".join(sorted(k.split(";")[0] for k in f.keys()
                                              if hasattr(f[k], "num_entries"))))

    t = f["tree"]
    ncand = int(t.num_entries)
    names = set(t.keys())
    print("tree      : %d entries (candidates)" % ncand)

    # --- schema checks
    have = lambda *b: [x for x in b if x in names]
    print("factored H: %s" % (have("nRank", "nFactor", "hessfactorv",
                                   "hessdroppedmass", "hessrankgap") or "MISSING"))
    print("packed H  : %s" % (have("hesspackedv", "nSym") or "absent (fillGrads=False)"))
    print("jacobians : %s" % (have("jacrefv", "jacRef", "jacmassv", "jacMass",
                                   "globalidxv", "gradv", "nParms") or "?"))
    sr = have(*STEP_RECORD)
    print("step recs : %s" % (sr if sr else "absent (exportStepRecords=False)"))
    cf = sorted(n for n in names if n.startswith("cfmass"))
    print("cf export : %d branches %s" % (len(cf), cf))
    if "cfmass_ok" in names:
        ok = t["cfmass_ok"].array(library="np").astype(bool)
        print("cfmass_ok : %d/%d (%.1f%%)" % (ok.sum(), len(ok), 100. * ok.mean()))

    for scalar in ("nParms", "nRank", "nFactor", "niter", "ndof"):
        if scalar in names:
            v = t[scalar].array(library="np")
            if v.dtype != object:
                print("%-10s: mean %.1f  median %.1f  max %d"
                      % (scalar, v.mean(), np.median(v), v.max()))

    # --- volume
    tot = 0
    rows = []
    for n in t.keys():
        b = t[n]
        nb = int(getattr(b, "compressed_bytes", 0) or 0)
        tot += nb
        rows.append((nb, n))
    rows.sort(reverse=True)
    print("\ntree volume: %s compressed, %d candidates -> %.1f kB/candidate"
          % (_fmt(tot), ncand, tot / 1024. / max(ncand, 1)))
    print("  %-24s %10s %10s %6s" % ("branch", "bytes", "kB/cand", "share"))
    for nb, n in rows[:a.top]:
        print("  %-24s %10d %10.2f %5.1f%%"
              % (n, nb, nb / 1024. / max(ncand, 1), 100. * nb / max(tot, 1)))

    # --- global parameter map
    cnt, ntot = parmtype_map(f)
    rtb = runtree_bytes(f)
    print("\nrun tree   : nglobalparms = %s, %s compressed (FIXED per file)"
          % (ntot, _fmt(rtb)))
    if cnt:
        for k in sorted(cnt):
            print("  parmtype %-3d %8d" % (k, cnt[k]))

    if a.compare:
        g = uproot.open(a.compare)
        cnt2, ntot2 = parmtype_map(g)
        print("\ncompare    : %s -> nglobalparms = %s" % (a.compare, ntot2))
        same_n = (ntot == ntot2)
        same_c = (cnt == cnt2)
        print("  totals   : %s" % ("MATCH" if same_n else "DIFFER (%s vs %s)" % (ntot, ntot2)))
        print("  parmtypes: %s" % ("MATCH" if same_c else "DIFFER"))
        if not same_c and cnt and cnt2:
            for k in sorted(set(cnt) | set(cnt2)):
                if cnt.get(k) != cnt2.get(k):
                    print("    parmtype %-3d %8s vs %8s" % (k, cnt.get(k), cnt2.get(k)))
        # The counts matching is necessary but NOT sufficient: the two files
        # must assign the same INDEX to the same physical parameter. Compare
        # the ordered (parmtype, detid) sequence where the run tree carries it.
        rt, rt2 = f["runtree"], g["runtree"]
        keyed = [b for b in ("parmtype", "rawdetid", "iidx", "subdet", "layer",
                             "glued", "stereo", "xi", "bz")
                 if b in rt.keys() and b in rt2.keys()]
        if keyed:
            allsame = True
            for b in keyed:
                x = rt[b].array(library="np")
                y = rt2[b].array(library="np")
                if x.dtype == object:
                    x = np.concatenate([np.asarray(v) for v in x]) if len(x) else np.array([])
                    y = np.concatenate([np.asarray(v) for v in y]) if len(y) else np.array([])
                eq = (len(x) == len(y)) and bool(np.array_equal(x, y))
                allsame &= eq
                print("  order[%s]: %s" % (b, "MATCH" if eq else "DIFFER"))
            print("  POOLABLE : %s" % ("YES -- identical global index map" if allsame
                                       else "NO -- re-map through runtree before pooling"))
        return 0 if (same_n and same_c) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
