#!/usr/bin/env python3
"""First look: block structure of resinfv, sign of localdxdz, size of the
per-hit residual, and whether v_b == w0^2 sigma_CPE^2 for hit blocks."""
import glob
import sys

import numpy as np
import uproot

BASE = "/ceph/submit/data/user/d/david_w/ZMass/cvh"

# ---- (1) the TRACKRES side: block structure + signed influence ------------
d = "resolution_trackres_mugun_ul16_260903x_m0"
fs = sorted(glob.glob(f"{BASE}/{d}/task_*/globalcor_*.root"))
f = uproot.open(fs[0])
pt = f["runtree"]["parmtype"].array(library="np")
t = f["tree"]
br = ["reseigidx", "reshitidx", "resinfv", "resinfvarv", "resinfcov", "refCov",
      "genParms", "refParms", "dxerr", "dyerr", "hitDetId", "hitUProj",
      "clusterSizeX", "clusterChargeBin", "localdxdz", "localdydz",
      "trackEta", "trackCharge", "nValidHits"]
a = t.arrays(br, library="np", entry_stop=200)
print("parmtype values:", np.unique(pt, return_counts=True))
nb_hist = {}
for ic in range(20):
    gi = np.asarray(a["reseigidx"][ic])
    w = np.asarray(a["resinfv"][ic], float).reshape(-1, 5)
    vb = np.asarray(a["resinfvarv"][ic], float)
    hidx = np.asarray(a["reshitidx"][ic])
    fam = pt[gi]
    nnz = (w != 0).sum(1)
    for fm, n in zip(fam, nnz):
        nb_hist.setdefault(int(fm), []).append(int(n))
    if ic == 0:
        m = (fam == 8)
        dxerr = np.asarray(a["dxerr"][ic], float)
        print("\nfam=8 blocks of track 0:")
        for j in np.where(m)[0][:8]:
            hh = hidx[j]
            se = dxerr[hh] if 0 <= hh < len(dxerr) else np.nan
            print(f"  hit {hh:3d} w0={w[j,0]: .6g} nnz={int(nnz[j])} vb={vb[j]:.6g} "
                  f"dxerr={se:.6g}  w0^2*dxerr^2={w[j,0]**2*se**2:.6g}  "
                  f"ratio={(w[j,0]**2*se**2)/vb[j] if vb[j]>0 else np.nan:.4f}")
        m9 = (fam == 9)
        dyerr = np.asarray(a["dyerr"][ic], float)
        print("fam=9 blocks of track 0:")
        for j in np.where(m9)[0][:6]:
            hh = hidx[j]
            se = dyerr[hh] if 0 <= hh < len(dyerr) else np.nan
            print(f"  hit {hh:3d} w0={w[j,0]: .6g} nnz={int(nnz[j])} vb={vb[j]:.6g} "
                  f"dyerr={se:.6g}  ratio={(w[j,0]**2*se**2)/vb[j] if vb[j]>0 else np.nan:.4f}")
print("\nblock dimension (nonzero dofs in resinfv) per family:")
for fm, v in sorted(nb_hist.items()):
    v = np.array(v)
    print(f"  fam {fm}: n={len(v)} nnz {np.bincount(v, minlength=6)}")

# ---- (2) the HITRES side --------------------------------------------------
d2 = "hitres2_mugun_ul16"
fs2 = sorted(glob.glob(f"{BASE}/{d2}/task_*/globalcor_*.root"))
t2 = uproot.open(fs2[0])["tree"]
b2 = ["dxrecsim", "dxerr", "dyrecsim", "dyerr", "hitDetId", "hitUProj",
      "clusterSizeX", "clusterChargeBin", "localdxdz", "localdydz",
      "genPt", "trackEta", "trackCharge", "clusterOnEdge", "stripsToEdge"]
a2 = t2.arrays(b2, library="np", entry_stop=2000)
cat = lambda k: np.concatenate(a2[k])
dx, dxe = cat("dxrecsim"), cat("dxerr")
sd = (cat("hitDetId").astype(np.uint32) >> 25) & 0x7
dxdz = cat("localdxdz")
m = (dx > -98) & (dxe > 0)
print(f"\n{d2}: {t2.num_entries} ev/file, {len(dx)} hits, {m.sum()} with sim")
print("genPt range:", np.percentile(a2['genPt'], [0, 5, 50, 95, 100]))
print("subdet counts:", np.bincount(sd, minlength=7))
print("localdxdz: median %.4f  frac>0 %.4f  |.| median %.4f"
      % (np.median(dxdz[m]), (dxdz[m] > 0).mean(), np.median(np.abs(dxdz[m]))))
print("localdydz: median %.4f  frac>0 %.4f"
      % (np.median(cat("localdydz")[m]), (cat("localdydz")[m] > 0).mean()))
pull = dx[m] / dxe[m]
print("pull_x: median %.4f mean %.4f  n=%d" % (np.median(pull), pull.mean(), m.sum()))
for s in (1, 2, 3, 4, 5, 6):
    k = m & (sd == s)
    if k.sum() > 100:
        p = dx[k] / dxe[k]
        print(f"  subdet {s}: n={k.sum():7d} median pull {np.median(p):+.4f} "
              f"mean {p.mean():+.4f} rms {p.std():.3f}")
print("uProj:", np.percentile(cat("hitUProj")[m & (sd > 2)], [1, 25, 50, 75, 99]))
