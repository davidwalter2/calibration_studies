#!/usr/bin/env python3
"""Light per-track extraction for the convergence variants, keyed by
(run, lumi, event) so the variants can be compared PAIRED on identical
tracks -- which is far more precise than comparing two independent means,
because the fluctuation is common to both.
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CEPH = "/ceph/submit/data/user/d/david_w/ZMass/cvh"
COVTOL = 5e-3
BR = ["run", "lumi", "event", "refCov", "refParms", "refParms_iter0",
      "trackParms", "genParms", "resinfcov", "reseigidx", "resinfvarv",
      "niter", "edmval", "edmvalref", "normalizedChi2", "nValidHits",
      "nValidPixelHits", "trackPt", "genPt", "nChargeFlipProtect"]


def one(fn):
    import uproot
    try:
        f = uproot.open(fn)
        pt = f["runtree"]["parmtype"].array(library="np")
        a = f["tree"].arrays(BR, library="np")
    except Exception as e:
        return None, f"{fn}: {type(e).__name__} {e}"
    o = {k: [] for k in ("run", "lumi", "event", "z", "sigma", "eta", "q",
                         "vgf", "pt", "genpt", "niter", "edm", "edmref",
                         "chi2n", "nvalid", "npixhit", "qop_seed", "qop_it0",
                         "qop_ref", "flip")}
    for ic in range(len(a["resinfcov"])):
        qg = a["genParms"][ic][0]
        c00 = float(a["refCov"][ic][0])
        cov = float(a["resinfcov"][ic])
        if qg == 0. or not (c00 > 0.) or abs(cov / c00 - 1.) > COVTOL:
            continue
        gi = np.asarray(a["reseigidx"][ic])
        vb = np.asarray(a["resinfvarv"][ic], float)
        fam = pt[gi]
        sig = np.sqrt(c00)
        o["run"].append(int(a["run"][ic]))
        o["lumi"].append(int(a["lumi"][ic]))
        o["event"].append(int(a["event"][ic]))
        o["z"].append((a["refParms"][ic][0] - qg) / sig)
        o["sigma"].append(sig)
        o["eta"].append(-np.log(np.tan((np.pi / 2. - a["refParms"][ic][1]) / 2.)))
        o["q"].append(np.sign(qg))
        o["vgf"].append(vb[(fam == 8) | (fam == 9)].sum() / c00)
        o["pt"].append(float(a["trackPt"][ic]))
        o["genpt"].append(float(a["genPt"][ic]))
        o["niter"].append(int(a["niter"][ic]))
        o["edm"].append(float(a["edmval"][ic]))
        o["edmref"].append(float(a["edmvalref"][ic]))
        o["chi2n"].append(float(a["normalizedChi2"][ic]))
        o["nvalid"].append(int(a["nValidHits"][ic]))
        o["npixhit"].append(int(a["nValidPixelHits"][ic]))
        o["qop_seed"].append(float(a["trackParms"][ic][0]))
        o["qop_it0"].append(float(a["refParms_iter0"][ic][0]))
        o["qop_ref"].append(float(a["refParms"][ic][0]))
        o["flip"].append(int(a["nChargeFlipProtect"][ic]))
    return {k: np.asarray(v) for k, v in o.items()}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nproc", type=int, default=20)
    a = ap.parse_args()
    import glob as _g

    import prodfiles
    stem = ("globalcor_resclosure"
            if _g.glob(f"{CEPH}/{a.prod}/task_*/globalcor_resclosure_*.root")
            else "globalcor")
    fs = prodfiles.resolve(f"{CEPH}/{a.prod}/task_*/{stem}_0.root", 0)
    print(f"{a.prod}: {len(fs)} files", flush=True)
    with Pool(a.nproc) as p:
        res = p.map(one, fs)
    for _, e in [r for r in res if r[1]][:5]:
        print("SKIP", e, flush=True)
    parts = [d for d, e in res if d is not None]
    d = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    np.savez_compressed(a.out, **d)
    print(f"wrote {a.out}: {len(d['z'])} tracks", flush=True)


if __name__ == "__main__":
    main()
