#!/usr/bin/env python3
"""Tree-entry count per file (metadata only, no branch reads) -> the
maker-level loss fraction thrown -> ntuple."""
import glob, sys
import numpy as np
import uproot
from multiprocessing import Pool
import prodfiles

def one(fn):
    return uproot.open(fn)["tree"].num_entries

if __name__ == "__main__":
    tag, nev, npart = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    pat = f"/ceph/submit/data/user/d/david_w/ZMass/cvh/resolution_trackres_{tag}/task_*/globalcor*_*.root"
    files = prodfiles.resolve(pat)
    with Pool(16) as p:
        n = p.map(one, files)
    n = np.array(n)
    thrown = len(files) * nev * npart
    print(f"{tag:<36} files {len(files):4d}  thrown {thrown:8d}  tree {n.sum():8d}"
          f"  LOSS {thrown-n.sum():7d} = {100.*(thrown-n.sum())/thrown:7.3f} %"
          f"   per-file tree min {n.min()} max {n.max()}")
