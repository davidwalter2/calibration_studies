#!/usr/bin/env python3
"""Per-module B-field (parmtype 6) and material (parmtype 7) Jacobian columns:
15_0 single-track maker with perModuleBfield=True globalMaterialModel=False vs
the pristine 10_6 maker, on the same input tracks.

The two catalogues index differently, so columns are matched by
(parmtype, rawdetid) through each file's runtree. Tracks are matched by
(run, event, rank within the event) and required to have the same number of
valid hits. Reports the relative agreement of refParms and of the q/p, lambda,
phi rows of jacref per parmtype.

usage (inside a 15_0 cmsenv):
  compare_modulecols_10_6.py REF_10_6.root NEW_15_0.root
"""
import sys
from collections import defaultdict
import numpy as np
import ROOT


def load(fname):
    f = ROOT.TFile.Open(fname)
    rt = f.Get("runtree")
    key = {}
    for i, e in enumerate(rt):
        key[i] = (int(e.parmtype), int(e.rawdetid))
    t = f.Get("tree")
    rows = defaultdict(list)
    for ev in t:
        n = int(ev.nParms)
        nj = int(ev.nJacRef)
        nrow = nj // n if n else 0
        idx = [int(ev.globalidxv[i]) for i in range(n)]
        jac = np.array([ev.jacrefv[i] for i in range(nj)], dtype=np.float64).reshape(nrow, n)
        cols = {key[g]: jac[:3, j] for j, g in enumerate(idx)}
        rows[(int(ev.run), int(ev.event))].append(dict(
            ref=np.array([ev.refParms[i] for i in range(5)], dtype=np.float64),
            nvalid=int(ev.nValidHits), cols=cols, edm=float(ev.edmvalref)))
    f.Close()
    return rows, key


ref, keyref = load(sys.argv[1])
new, keynew = load(sys.argv[2])
print("catalogue sizes: 10_6 %d, 15_0 %d" % (len(keyref), len(keynew)))
for pt in (6, 7):
    a = sum(1 for k in keyref.values() if k[0] == pt)
    b = sum(1 for k in keynew.values() if k[0] == pt)
    print("  parmtype %d: 10_6 %d, 15_0 %d entries" % (pt, a, b))

dref, stats = [], defaultdict(list)
nmatch = 0
for ev, rlist in ref.items():
    nlist = new.get(ev, [])
    for r, n in zip(rlist, nlist):
        if r["nvalid"] != n["nvalid"] or r["edm"] > 1e-2 or n["edm"] > 1e-2:
            continue
        nmatch += 1
        dref.append((n["ref"] - r["ref"]) / np.where(r["ref"] != 0, np.abs(r["ref"]), 1.))
        for k, cr in r["cols"].items():
            if k[0] not in (0, 6, 7) or k not in n["cols"]:
                continue
            cn = n["cols"][k]
            scale = np.max(np.abs(cr))
            if scale > 0:
                stats[k[0]].append(np.max(np.abs(cn - cr)) / scale)
        for pt in (6, 7):
            nr = sum(1 for k in r["cols"] if k[0] == pt)
            nn = sum(1 for k in n["cols"] if k[0] == pt)
            stats["n%d" % pt].append((nr, nn))
dref = np.array(dref)
print("matched tracks:", nmatch)
for i, nm in enumerate(["qop", "lambda", "phi"]):
    print("  refParms %-6s |rel diff| median %.1e  p95 %.1e" %
          (nm, np.median(np.abs(dref[:, i])), np.quantile(np.abs(dref[:, i]), 0.95)))
for pt, nm in ((6, "Bz (6)"), (7, "material (7)"), (0, "align x (0)")):
    v = np.array(stats[pt])
    print("  jacref cols %-12s n=%5d  max|new-ref|/max|ref|: median %.1e  p95 %.1e  max %.1e" %
          (nm, len(v), np.median(v), np.quantile(v, 0.95), v.max()))
for pt in (6, 7):
    c = np.array(stats["n%d" % pt])
    print("  columns of parmtype %d per track: 10_6 %.2f, 15_0 %.2f, identical count on %d/%d tracks" %
          (pt, c[:, 0].mean(), c[:, 1].mean(), int(np.sum(c[:, 0] == c[:, 1])), len(c)))
