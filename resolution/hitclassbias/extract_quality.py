#!/usr/bin/env python3
"""Per-TRACK fit-quality and convergence variables, on exactly the sample
`extract_blocks.py` keeps (same files, same order, same three cuts), so the
arrays line up with `blocks_*.npz` element by element.

What the maker actually exports per track (checked against
ResidualGlobalCorrectionMakerBase.h, 2026-09-08):
  niter, edmval, edmvalref, deltachisqval, gradmax, hessmax
  chisqval, ndof, normalizedChi2, nHits, nValidHits, nValidPixelHits
  nChargeFlipProtect  -- GN iterations where the momentum clamp caught a q/p
                         SIGN crossing; > 0 flags a track that "wanted" the
                         opposite charge
  chargeHypFlipped    -- the two-hypothesis refit kept the opposite charge
  trackParms[5]       -- the SEED (generalTracks) state
  refParms_iter0[5]   -- the state after GN iteration 0
  refParms[5]         -- the converged CVH state
The per-STEP clamp/backtrack counters (`fitStepClamped_`,
`stepBacktrackEvents_`) are JOB-level in this release and are NOT per track;
`nChargeFlipProtect`, `niter` and the iter0->final step are the per-track
proxies for step control.
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
BR = ["reseigidx", "reshitidx", "resinfvarv", "resinfcov", "refCov",
      "refParms", "refParms_iter0", "trackParms", "genParms", "hitDetId",
      "niter", "edmval", "edmvalref", "deltachisqval", "gradmax", "hessmax",
      "chisqval", "ndof", "normalizedChi2", "nHits", "nValidHits",
      "nValidPixelHits", "nChargeFlipProtect", "chargeHypFlipped",
      "trackPt", "genPt", "trackQopErr", "trackPtErr"]
SCAL = ["niter", "edmval", "edmvalref", "deltachisqval", "gradmax", "hessmax",
        "chisqval", "ndof", "normalizedChi2", "nHits", "nValidHits",
        "nValidPixelHits", "nChargeFlipProtect", "chargeHypFlipped",
        "trackQopErr", "trackPtErr"]


def one(fn):
    import uproot
    try:
        f = uproot.open(fn)
        pt = f["runtree"]["parmtype"].array(library="np")
        a = f["tree"].arrays(BR, library="np")
    except Exception as e:
        return None, f"{fn}: {type(e).__name__} {e}"
    out = {k: [] for k in SCAL}
    for k in ("qop_seed", "qop_iter0", "qop_ref", "lam_seed", "phi_seed",
              "npixlay", "firstlay", "firstsd", "nlay"):
        out[k] = []
    for ic in range(len(a["resinfcov"])):
        qg = a["genParms"][ic][0]
        c00 = float(a["refCov"][ic][0])
        cov = float(a["resinfcov"][ic])
        if qg == 0. or not (c00 > 0.) or abs(cov / c00 - 1.) > COVTOL:
            continue
        gi = np.asarray(a["reseigidx"][ic])
        fam = pt[gi]
        vb = np.asarray(a["resinfvarv"][ic], float)
        hidx = np.asarray(a["reshitidx"][ic])
        hd = np.asarray(a["hitDetId"][ic]).astype(np.uint32)
        hit = ((fam == 8) | (fam == 9)) & (vb > 0.) & (hidx >= 0)
        sel = np.where(hit)[0]
        sel = sel[hidx[sel] < len(hd)]
        if len(sel) == 0:
            continue
        d = hd[hidx[sel]]
        sd = ((d >> 25) & 0x7).astype(int)
        lay = np.where(sd <= 2, (d >> 16) & 0xF,
                       np.where((sd == 3) | (sd == 5), (d >> 14) & 0x7,
                                np.where(sd == 4, (d >> 11) & 0x3,
                                         (d >> 14) & 0xF))).astype(int)
        pixlay = np.unique((sd[sd <= 2] * 10 + lay[sd <= 2]))
        allid = np.unique(sd * 10 + lay)
        # innermost measured layer, by the canonical radial order
        ORD = {11: 0, 12: 1, 13: 2, 21: 3, 22: 4, 31: 5, 32: 6, 33: 7, 34: 8,
               41: 9, 42: 10, 43: 11, 51: 12, 52: 13, 53: 14, 54: 15, 55: 16,
               56: 17}
        rk = [ORD.get(int(x), 90 + int(x) % 100) for x in allid]
        j = int(np.argmin(rk))
        for k in SCAL:
            out[k].append(float(a[k][ic]))
        out["qop_seed"].append(float(a["trackParms"][ic][0]))
        out["qop_iter0"].append(float(a["refParms_iter0"][ic][0]))
        out["qop_ref"].append(float(a["refParms"][ic][0]))
        out["lam_seed"].append(float(a["trackParms"][ic][1]))
        out["phi_seed"].append(float(a["trackParms"][ic][2]))
        out["npixlay"].append(len(pixlay))
        out["nlay"].append(len(allid))
        out["firstlay"].append(int(allid[j] % 10))
        out["firstsd"].append(int(allid[j] // 10))
    return {k: np.asarray(v, float) for k, v in out.items()}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod", default="resolution_trackres_mugun_ul16_260903x_m0")
    ap.add_argument("--out", default="data/quality_mugun_ul16_260903x.npz")
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
    print(f"wrote {a.out}: {len(d['niter'])} tracks", flush=True)


if __name__ == "__main__":
    main()
