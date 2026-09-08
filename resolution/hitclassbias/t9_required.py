#!/usr/bin/env python3
"""How far would the class locations have to move to make the observed
per-eta charge-even shift? A constrained least squares:

    minimise  sum_k ((mu_k - muhat_k)/sigma_k)^2   subject to   L mu = t

with `muhat` the MEASURED locations, `L` the fit's own lever arms and `t` the
three measured band shifts. The Lagrange solution gives both the required
locations and the chi^2 of the displacement -- i.e. by how many sigma the
measured CPE locations must be wrong for the hit-location mechanism to be
the explanation.
"""
import numpy as np

SDN = {1: "BPix", 2: "FPix", 3: "TIB", 4: "TID", 5: "TOB", 6: "TEC"}
CM = (1. / 1.1) ** 1.5
h = np.load("data/hits_mugun_ul16_h2.npz")
b = np.load("data/blocks_mugun_ul16_260903x.npz")
p = np.load("data/pred_track.npz")
aw = b["b_s"].astype(np.float64) * np.sqrt(b["b_a2"].astype(np.float64))
trk = np.repeat(np.arange(len(b["nblk"])), b["nblk"])
band = np.digitize(np.abs(b["t_eta"]), [0.9, 1.6])
bb = band[trk]
nb = np.bincount(band, minlength=3)

px, py = h["pullx"].astype(np.float64), h["pully"].astype(np.float64)
okx, oky = np.isfinite(px) & (np.abs(px) < 10), np.isfinite(py) & (np.abs(py) < 10)

groups = []
for sd in range(1, 7):
    groups.append((f"{SDN[sd]} x", okx & (h["sd"] == sd), px,
                   (b["b_sd"] == sd) & (b["b_isy"] == 0)))
for sd in (1, 2):
    groups.append((f"{SDN[sd]} y", oky & (h["sd"] == sd), py,
                   (b["b_sd"] == sd) & (b["b_isy"] == 1)))

L = np.zeros((3, len(groups)))
mh = np.zeros(len(groups))
se = np.zeros(len(groups))
for j, (nm, hm, pp, bm) in enumerate(groups):
    v = pp[hm]
    mh[j] = v.mean()
    se[j] = v.std() / np.sqrt(len(v))
    for i in range(3):
        L[i, j] = aw[bm & (bb == i)].sum() / nb[i]

t = np.array([-4.89, -3.57, +0.12]) * 1e-3 / CM     # required MEAN shift
r = t - L @ mh
W = np.diag(se ** 2)
A = L @ W @ L.T
lam = np.linalg.solve(A, r)
dmu = W @ L.T @ lam
chi2 = float(dmu @ np.linalg.solve(W, dmu))

print("required per-group LOCATION vs the measured one (pull units)")
print(f"{'group':10s} {'measured':>10s} {'+-':>8s} {'required':>10s} "
       f"{'shift/sigma':>12s} | " + "".join(f"{f'L {n}':>10s}"
                                            for n in ("barrel", "mid", "endc")))
for j, (nm, _, _, _) in enumerate(groups):
    print(f"{nm:10s} {mh[j]:+10.4f} {se[j]:8.4f} {mh[j]+dmu[j]:+10.4f} "
          f"{dmu[j]/se[j]:+12.1f} | " + "".join(f"{L[i,j]:+10.5f}" for i in range(3)))
print(f"\nchi2 of the required displacement = {chi2:.1f} for 3 constraints "
      f"-> {np.sqrt(chi2):.1f} sigma")
print(f"predicted from the MEASURED locations (odd moment, 1e-3): "
      + " ".join(f"{v:+.2f}" for v in CM * (L @ mh) * 1e3))
print(f"measured                                (odd moment, 1e-3): "
      f"-4.89 -3.57 +0.12")

# the single-subdetector test
print("\nif the whole effect came from ONE group, the location it would need:")
for j, (nm, _, _, _) in enumerate(groups):
    if abs(L[0, j]) < 1e-4:
        continue
    need = t[0] / L[0, j]
    print(f"  {nm:10s} mu = {need:+8.3f} ({need/se[j]:+7.1f} sigma from the "
          f"measured {mh[j]:+.4f}); it would then give "
          + " ".join(f"{CM*L[i,j]*need*1e3:+.2f}" for i in range(3))
          + "  (target -4.89 -3.57 +0.12)")
