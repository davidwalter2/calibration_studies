#!/usr/bin/env python3
"""Single-track CVH maker: closure of the pixel class-correction columns.

Compares two single-track refits of the SAME tracks (same input, one thread,
so the entry order is identical), a reference run `base` and a run `shifted`
in which the pixel hit positions were displaced by a known amount -- either
through `injectLorentzTan` (dtanLA mode) or through a corFile carrying known
class-correction values. The linear prediction of the refit change is

    d(refParms) = jacref[:, class columns] . theta

with theta the injected per-parameter values. Reports, per refParms row, the
observed shift, the predicted shift and their ratio, plus the catalogue and
column bookkeeping (parmtypes present in globalidxv).

usage (inside cmsenv):
  check_singletrack_classcols.py BASE.root SHIFTED.root --lorentz 0.05
  check_singletrack_classcols.py BASE.root SHIFTED.root --corfile COR.root
"""
import argparse
import numpy as np
import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("base")
ap.add_argument("shifted")
ap.add_argument("--lorentz", type=float, default=None,
                help="injectLorentzTan of the shifted run (parmtype 22)")
ap.add_argument("--corfile", default=None,
                help="corFile applied in the shifted run (parmtree/x)")
ap.add_argument("--tree", default="tree")
args = ap.parse_args()


def load(fname):
    f = ROOT.TFile.Open(fname)
    t = f.Get(args.tree)
    rows = []
    for ev in t:
        n = int(ev.nParms)
        idx = np.array([ev.globalidxv[i] for i in range(n)], dtype=np.int64)
        jac = np.array([ev.jacrefv[i] for i in range(5 * n)], dtype=np.float64).reshape(5, n)
        ref = np.array([ev.refParms[i] for i in range(5)], dtype=np.float64)
        rows.append(dict(run=int(ev.run), event=int(ev.event), pt=float(ev.trackPt),
                         idx=idx, jac=jac, ref=ref, edm=float(ev.edmvalref),
                         npix=int(ev.nValidPixelHits)))
    rt = f.Get("runtree")
    ptype = {}
    if rt:
        for i, e in enumerate(rt):
            ptype[int(e.iidx) if hasattr(e, "iidx") else i] = int(e.parmtype)
    f.Close()
    return rows, ptype


base, ptype = load(args.base)
shif, ptype2 = load(args.shifted)
assert len(base) == len(shif), (len(base), len(shif))
assert ptype == ptype2, "catalogues differ"
nglob = len(ptype)
pt_arr = np.array([ptype[i] for i in range(nglob)])
print(f"catalogue: {nglob} parameters; class parmtypes present:",
      {k: int(np.sum(pt_arr == k)) for k in range(16, 23) if np.any(pt_arr == k)})

theta = np.zeros(nglob)
if args.lorentz is not None:
    theta[pt_arr == 22] = args.lorentz
if args.corfile is not None:
    f = ROOT.TFile.Open(args.corfile)
    ct = f.Get("parmtree")
    vals = np.array([e.x for e in ct], dtype=np.float64)
    f.Close()
    assert len(vals) == nglob, (len(vals), nglob)
    theta += vals

ncols = {k: 0 for k in range(16, 23)}
obs, pred, pts = [], [], []
for b, s in zip(base, shif):
    assert (b["run"], b["event"]) == (s["run"], s["event"]) and abs(b["pt"] - s["pt"]) < 1e-6
    for k in ncols:
        ncols[k] += int(np.sum(pt_arr[b["idx"]] == k))
    if b["edm"] > 1e-2 or s["edm"] > 1e-2:
        continue
    th = theta[b["idx"]]
    # average the Jacobian of the two linearization points
    p = 0.5 * (b["jac"] + s["jac"]) @ th
    obs.append(s["ref"] - b["ref"])
    pred.append(p)
    pts.append(b["pt"])
obs = np.array(obs)
pred = np.array(pred)
print(f"{len(obs)} converged tracks of {len(base)}; class columns per parmtype:",
      {k: v for k, v in ncols.items() if v})
names = ["qop", "lambda", "phi", "dxy", "dsz"]
for r in range(5):
    o, p = obs[:, r], pred[:, r]
    sel = np.abs(p) > 0
    if not np.any(sel):
        continue
    slope = np.sum(o[sel] * p[sel]) / np.sum(p[sel] ** 2)
    resid = o[sel] - p[sel]
    print(f"{names[r]:>6}: rms(obs)={np.sqrt(np.mean(o**2)):.3e} rms(pred)={np.sqrt(np.mean(p**2)):.3e} "
          f"slope obs/pred={slope:.5f}  rms(obs-pred)/rms(pred)={np.sqrt(np.mean(resid**2))/np.sqrt(np.mean(p[sel]**2)):.2e}")
