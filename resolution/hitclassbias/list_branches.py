#!/usr/bin/env python3
import glob, sys, uproot
BASE = "/ceph/submit/data/user/d/david_w/ZMass/cvh"
for d in sys.argv[1:]:
    fs = sorted(glob.glob(f"{BASE}/{d}/task_*/globalcor_*.root"))
    t = uproot.open(fs[0])["tree"]
    print(f"### {d}  ({len(fs)} files, {t.num_entries} ev/file)")
    for k in sorted(t.keys()):
        print("   ", k, t[k].typename)
