#!/usr/bin/env python3
"""Which productions carry BOTH the per-hit truth residual and the SIGNED
per-block influence weight? The location study needs the two on the same
tracks, because the prediction is delta z = sum_b (w0_b sigma_b/sigma) pull_b
and the sign of w0_b is what turns a LOCAL bias into a BENDING-sense one."""
import glob
import sys

import uproot

BASE = "/ceph/submit/data/user/d/david_w/ZMass/cvh"
WANT = ["dxrecsim", "dxerr", "dyrecsim", "dyerr", "hitDetId", "hitUProj",
        "clusterSizeX", "clusterChargeBin", "reshitidx", "reseigidx",
        "resinfv", "resinfvarv", "resinfcov", "refCov", "refParms",
        "genParms", "trackPt", "genPt"]

for d in sys.argv[1:]:
    fs = sorted(glob.glob(f"{BASE}/{d}/task_*/globalcor_*.root"))
    if not fs:
        print(f"{d}: NO FILES")
        continue
    try:
        t = uproot.open(fs[0])["tree"]
        keys = set(t.keys())
        n = t.num_entries
    except Exception as e:
        print(f"{d}: ERR {type(e).__name__} {e}")
        continue
    miss = [b for b in WANT if b not in keys]
    print(f"{d}: nfile={len(fs)} nev0={n} nbr={len(keys)}  MISS={miss}")
    if len(sys.argv) == 2:
        print("   ALL:", sorted(keys))
