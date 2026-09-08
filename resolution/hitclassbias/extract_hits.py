#!/usr/bin/env python3
"""Per-HIT table from a `hitres*` production: the truth residual with its
class variables, its signed local angles, and the track it belongs to.

One row per valid hit that has a matched PSimHit. Both coordinates are kept
SEPARATELY (`pull` for the precise/bending coordinate -- local x, or local
phi on the radial-strip wedges -- and `pully` for the pixel local y), because
a forward-disk incidence effect must live in one of them and not the other.

usage: extract_hits.py --prod hitres2_mugun_ul16 --out hits_mugun_ul16.npz
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hcb_common as H  # noqa: E402

CEPH = "/ceph/submit/data/user/d/david_w/ZMass/cvh"

HITBR = ["dxrecsim", "dxerr", "dyrecsim", "dyerr", "hitDetId", "hitUProj",
         "clusterSizeX", "clusterSizeY", "clusterChargeBin", "clusterOnEdge",
         "hitPitch", "hitThickness", "localdxdz", "localdydz"]
TRKBR = ["trackEta", "trackCharge", "genPt", "trackPt", "nValidHits",
         "normalizedChi2"]


def one(fn):
    import uproot
    try:
        t = uproot.open(fn)["tree"]
        keys = set(t.keys())
        extra = [b for b in ("stripsToEdge", "simHitNCand") if b in keys]
        a = t.arrays(HITBR + TRKBR + extra, library="np")
    except Exception as e:
        return None, f"{fn}: {type(e).__name__} {e}"
    nh = np.array([len(v) for v in a["dxerr"]], dtype=np.int64)
    cat = lambda k: np.concatenate(a[k]) if len(a[k]) else np.array([])
    rep = lambda k: np.repeat(np.asarray(a[k], dtype=np.float64), nh)
    d = {k: cat(k) for k in HITBR + extra}
    for k in TRKBR:
        d["trk_" + k] = rep(k)
    for k in ("stripsToEdge", "simHitNCand"):
        if k not in d:
            d[k] = np.full(len(d["dxerr"]), -99, dtype=np.float64)
    return d, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--ntasks", type=int, default=0)
    args = ap.parse_args()

    import glob as _g

    import prodfiles
    # the hitres maker writes `globalcor_resclosure_N.root`, the trackres one
    # `globalcor_N.root`; prodfiles matches the stem EXACTLY, so pick it here
    # rather than passing a glob that would silently return nothing.
    stem = "globalcor"
    if _g.glob(f"{CEPH}/{args.prod}/task_*/globalcor_resclosure_*.root"):
        stem = "globalcor_resclosure"
    fs = prodfiles.resolve(f"{CEPH}/{args.prod}/task_*/{stem}_0.root",
                           args.ntasks)
    print(f"{args.prod}: {len(fs)} files", flush=True)
    with Pool(args.nproc) as p:
        res = p.map(one, fs)
    bad = [e for _, e in res if e]
    for e in bad[:5]:
        print("SKIP", e, flush=True)
    parts = [d for d, e in res if d is not None]
    keys = parts[0].keys()
    d = {k: np.concatenate([q[k] for q in parts]) for k in keys}
    print(f"raw hits {len(d['dxerr'])}", flush=True)

    sd, lay, side = H.decode(d["hitDetId"].astype(np.uint32))
    rng = H.ring(d["hitDetId"].astype(np.uint32))
    ster = H.is_stereo(d["hitDetId"].astype(np.uint32))
    okx = (d["dxrecsim"] > -98) & (d["dxerr"] > 0)
    oky = (d["dyrecsim"] > -98) & (d["dyerr"] > 0) & (sd <= 2)
    keep = okx | oky
    out = dict(
        pullx=np.where(okx, d["dxrecsim"] / np.where(d["dxerr"] > 0, d["dxerr"], 1),
                       np.nan)[keep].astype(np.float32),
        pully=np.where(oky, d["dyrecsim"] / np.where(d["dyerr"] > 0, d["dyerr"], 1),
                       np.nan)[keep].astype(np.float32),
        dx=d["dxrecsim"][keep].astype(np.float32),
        dxerr=d["dxerr"][keep].astype(np.float32),
        dy=d["dyrecsim"][keep].astype(np.float32),
        dyerr=d["dyerr"][keep].astype(np.float32),
        sd=sd[keep].astype(np.int8), lay=lay[keep].astype(np.int8),
        side=side[keep].astype(np.int8), ring=rng[keep].astype(np.int8),
        stereo=ster[keep].astype(np.int8),
        og=H.orient_group(d["hitDetId"].astype(np.uint32))[keep].astype(np.int64),
        detid=d["hitDetId"][keep].astype(np.uint32),
        uproj=d["hitUProj"][keep].astype(np.float32),
        N=d["clusterSizeX"][keep].astype(np.int16),
        NY=d["clusterSizeY"][keep].astype(np.int16),
        qbin=d["clusterChargeBin"][keep].astype(np.int16),
        onedge=d["clusterOnEdge"][keep].astype(np.int16),
        stripedge=d["stripsToEdge"][keep].astype(np.int16),
        nsimcand=d["simHitNCand"][keep].astype(np.int16),
        pitch=d["hitPitch"][keep].astype(np.float32),
        thick=d["hitThickness"][keep].astype(np.float32),
        dxdz=d["localdxdz"][keep].astype(np.float32),
        dydz=d["localdydz"][keep].astype(np.float32),
        teta=d["trk_trackEta"][keep].astype(np.float32),
        tq=d["trk_trackCharge"][keep].astype(np.int8),
        genpt=d["trk_genPt"][keep].astype(np.float32),
        trkpt=d["trk_trackPt"][keep].astype(np.float32),
        nvalid=d["trk_nValidHits"][keep].astype(np.int16),
    )
    out["cls18x"] = H.class18(out["sd"], out["N"], out["uproj"], out["qbin"],
                              False)
    out["cls18y"] = H.class18(out["sd"], out["N"], out["uproj"], out["qbin"],
                              True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"wrote {args.out}: {len(out['pullx'])} hits "
          f"({np.isfinite(out['pullx']).sum()} x, "
          f"{np.isfinite(out['pully']).sum()} y)", flush=True)


if __name__ == "__main__":
    main()
