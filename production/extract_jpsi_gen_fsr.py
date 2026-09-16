#!/usr/bin/env python3
"""Extract the gen-level FSR variables of a CVH J/psi MC production.

Reads only the six branches needed to reconstruct the per-candidate FSR
variable ``u = -ln(m_post / m_pre)``:

    Jpsigen_mass          post-FSR bare dimuon mass
    Jpsigenpre_masslep    pre-FSR resonance mass from the two status-746 muons
                          (PHOTOS++ history entries); -99 when the candidate
                          did not radiate, or when the event has other 746
                          muons and the pair could not be identified
    Jpsigen_pt, Jpsigen_eta, chisqval, ndof   -- selection

and writes one .npz cache.  Branch-level reads only; the 220 MB files are never
materialised.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import uproot

sys.path.insert(0, "/work/submit/david_w/ZMass/calibration_studies/resolution")
import prodfiles

BRANCHES = ["Jpsigen_mass", "Jpsigenpre_masslep", "Jpsigen_pt", "Jpsigen_eta",
            "chisqval", "ndof"]


def read_one(path):
    try:
        t = uproot.open(path, array_cache=None)["tree"]
        d = t.arrays(BRANCHES, library="np")
    except Exception as exc:                                   # noqa: BLE001
        return path, str(exc), None
    out = np.empty((len(d["Jpsigen_mass"]), 6), dtype=np.float32)
    for i, b in enumerate(BRANCHES):
        out[:, i] = d[b]
    return path, None, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--files",
                    default="/ceph/submit/data/user/d/david_w/ZMass/cvh/"
                            "jpsimc_20M_260906_v2/task_*/globalcor_0.root")
    ap.add_argument("--ntasks", type=int, default=0)
    ap.add_argument("-j", "--jobs", type=int, default=48)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    files = prodfiles.resolve(args.files, args.ntasks or None,
                              logger=lambda s: print(s, flush=True))
    print(f"{len(files)} files", flush=True)

    chunks, bad = [], []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for n, (path, err, arr) in enumerate(ex.map(read_one, files,
                                                    chunksize=4), 1):
            if err is not None:
                bad.append((path, err))
            else:
                chunks.append(arr)
            if n % 500 == 0:
                tot = sum(len(c) for c in chunks)
                print(f"  {n}/{len(files)} files  {tot} cands  "
                      f"{time.time() - t0:.0f} s", flush=True)

    a = np.concatenate(chunks, axis=0)
    print(f"{len(a)} candidates from {len(files) - len(bad)} files, "
          f"{len(bad)} failed, {time.time() - t0:.0f} s", flush=True)
    for p, e in bad[:10]:
        print("  FAILED", p, e[:120], flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(args.output,
             m_post=a[:, 0], masslep=a[:, 1], pt=a[:, 2], eta=a[:, 3],
             chisqval=a[:, 4], ndof=a[:, 5],
             nfiles=np.int64(len(files) - len(bad)), nbad=np.int64(len(bad)))
    print("wrote", args.output, flush=True)


if __name__ == "__main__":
    main()
