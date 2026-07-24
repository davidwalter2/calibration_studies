#!/usr/bin/env python
"""Write a CVH corFile (parmtree/x, one float per catalog parameter) seeding
the pixel pathology class-correction parameters (parmtypes 16-21) with the
per-(parmtype, subdet, layer) group values fitted by fit_classcorr.py.
All other parameters are 0. The target run must use the SAME catalog
(pixelHitClassCorrections=True) or the loader's size assert fires.
"""

import argparse
import glob
import pickle

import numpy as np
import uproot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rundir", required=True,
                   help="grads run dir (source of catalog + classcorr_groups.pkl)")
    p.add_argument("-o", "--output", required=True)
    args = p.parse_args()

    with open(args.rundir.rstrip("/") + "/classcorr_groups.pkl", "rb") as f:
        fit = pickle.load(f)
    assert not fit["per_module"], "use the group-level fit pkl"
    groupval = {k: t for k, t in zip(fit["keys"], fit["theta"]) if np.isfinite(t)}

    rt = uproot.open(sorted(glob.glob(args.rundir + "/globalcor_*.root"))[0])["runtree"]
    cat = rt.arrays(["parmtype", "subdet", "layer"], library="np")
    n = len(cat["parmtype"])
    x = np.zeros(n, dtype=np.float32)
    nset = 0
    for i, (t, s, l) in enumerate(zip(cat["parmtype"], cat["subdet"], cat["layer"])):
        if 16 <= t <= 21:
            key = (int(t), int(s), int(l))
            if key in groupval:
                x[i] = groupval[key]
                nset += 1
    print(f"catalog {n} params, seeded {nset} class entries")

    with uproot.recreate(args.output) as fout:
        fout["parmtree"] = {"x": x}
    print("wrote", args.output)


if __name__ == "__main__":
    main()
