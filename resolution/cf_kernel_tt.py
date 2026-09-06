#!/usr/bin/env python3
"""FSR/gen-mass kernel from a TwoTrack production (Jpsigen_mass per candidate).

`cf_mass_likelihood.py --kernel` reads the SINGLE-track tree (genParms/genCharge
gen pairing) and fails on a two-track tree. The August J/psi-gun kernel
(runs/cf_masskernel_jpsigun.npz, keys dm/mgen, one entry per candidate) was
built from Jpsigen_mass instead; this script is that step, made explicit.
The demo scan reads only k["dm"] (cf_mass_likelihood.py ~line 452).

usage: cf_kernel_tt.py --files '<glob of task_*/globalcor_*.root>' --out runs/x.npz
"""
import argparse, os, sys
import numpy as np, uproot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prodfiles  # noqa: E402

MJPSI = 3.0969
ap = argparse.ArgumentParser()
ap.add_argument("--files", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--ntasks", type=int, default=100000)
a = ap.parse_args()
# --ntasks caps TASKS, not files (a multi-stream task is N files)
files = prodfiles.resolve(a.files, a.ntasks, logger=print)
mg = []
for f in files:
    t = uproot.open(f)["tree"]
    m = t["Jpsigen_mass"].array(library="np").astype(np.float64)
    s = t["Jpsi_sigmamass"].array(library="np")
    keep = np.isfinite(m) & (np.abs(m - MJPSI) < 0.35)
    mg.append(m[keep])
mg = np.concatenate(mg)
dm = mg - MJPSI
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
np.savez_compressed(a.out, dm=dm, mgen=mg)
print(f"{len(files)} files, {dm.size} candidates in window, unique gen masses {np.unique(mg).size}, "
      f"FSR tail (dm < -5 MeV) {100*(dm < -0.005).mean():.2f}%, mean dm {1e3*dm.mean():+.3f} MeV, "
      f"median gen mass {np.median(mg):.6f} -> {a.out}")
