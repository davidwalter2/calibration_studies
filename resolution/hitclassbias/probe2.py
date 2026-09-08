#!/usr/bin/env python3
import glob, sys, uproot
BASE = "/ceph/submit/data/user/d/david_w/ZMass/cvh"
WANT = ["dxrecsim","dxerr","dyrecsim","dyerr","hitDetId","hitUProj","clusterSizeX",
        "clusterChargeBin","clusterOnEdge","clusterSizeY","stripsToEdge","hitPitch",
        "localdxdz","localdydz","simlocaldxdz","trackCharge","trackEta","trackPt",
        "genPt","genCharge","hitlocalx","hitlocaly","clusterSN","hitThickness",
        "simHitNCand","hitStripRec","hitStripSim","hitFirstStrip"]
for d in sys.argv[1:]:
    fs = sorted(glob.glob(f"{BASE}/{d}/task_*/globalcor_*.root"))
    if not fs:
        print(f"{d}: NO FILES"); continue
    try:
        t = uproot.open(fs[0])["tree"]; keys=set(t.keys())
    except Exception as e:
        print(f"{d}: ERR {e}"); continue
    print(f"{d}: nfile={len(fs)} nev={t.num_entries} MISS={[b for b in WANT if b not in keys]}")
