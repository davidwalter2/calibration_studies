#!/usr/bin/env python
"""Era-comparison table of the pixel class-correction fits (BPix).
Reads the classcorr_groups[_withalign].pkl files produced by
fit_classcorr.py for MC + 2016F/G/H. Bracketed values = simultaneous fit
with per-module pixel local-x/y alignment (Gaussian prior, see
fit_classcorr.py --with-alignment)."""
import pickle
import numpy as np
CM2UM = 1e4
runs = {
    "MC":    "/work/submit/david_w/ZMass/calibration_studies/pixelhits/runs/260723_classcorr_grads",
    "2016F": "/ceph/submit/data/user/d/david_w/ZMass/cvh/pixelhits_data_grads_2016F_260723",
    "2016G": "/ceph/submit/data/user/d/david_w/ZMass/cvh/pixelhits_data_grads_260723",
    "2016H": "/ceph/submit/data/user/d/david_w/ZMass/cvh/pixelhits_data_grads_2016H_260723",
}
def load(d, tag=""):
    try:
        with open(d + f"/classcorr_groups{tag}.pkl", "rb") as f:
            fit = pickle.load(f)
    except FileNotFoundError:
        return {}
    return {k: (t*CM2UM, e*CM2UM) for k, t, e in zip(fit["keys"], fit["theta"], fit["err"])
            if not isinstance(k[0], str) and np.isfinite(t)}
PN = {16:"edge-x-mean",17:"edge-x-diff",18:"edge-y-mean",19:"edge-y-diff",20:"sizeX1",21:"sizeY1"}
fits  = {era: load(d) for era, d in runs.items()}
fitsA = {era: load(d, "_withalign") for era, d in runs.items()}
print("BPix, fixed-alignment solve ([with-alignment] where available), um:")
print(f"{'parameter':>16s} {'L':>2s}" + "".join(f" {e:>22s}" for e in runs))
for pt in (16,17,18,19,20,21):
    for lay in (1,2,3):
        row = f"{PN[pt]:>16s} {lay:2d}"
        for era in runs:
            v, va = fits[era].get((pt,0,lay)), fitsA[era].get((pt,0,lay))
            s = f"{v[0]:+7.1f}±{v[1]:4.1f}" if v else "     --"
            if va: s += f" [{va[0]:+6.1f}]"
            row += f" {s:>22s}"
        print(row)
