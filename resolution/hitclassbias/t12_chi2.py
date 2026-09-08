#!/usr/bin/env python3
"""chi2 of the location displacement required to reproduce the observed
per-band charge-even shift, at each key granularity. 3 constraints; the
locations are free to move within their measurement errors."""
import numpy as np

h = np.load("data/hits_mugun_ul16_h2.npz")
b = np.load("data/blocks_mugun_ul16_260903x.npz")
CM = (1 / 1.1) ** 1.5
aw = b["b_s"].astype(np.float64) * np.sqrt(b["b_a2"].astype(np.float64))
trk = np.repeat(np.arange(len(b["nblk"])), b["nblk"])
band = np.digitize(np.abs(b["t_eta"]), [0.9, 1.6])
bb = band[trk]
nb = np.bincount(band, minlength=3)
px, py = h["pullx"].astype(np.float64), h["pully"].astype(np.float64)
okx = np.isfinite(px) & (np.abs(px) < 10)
oky = np.isfinite(py) & (np.abs(py) < 10)
SPECS = {
    "subdet x coord": [("sd", "b_sd", 8)],
    "subdet x layer": [("sd", "b_sd", 8), ("lay", "b_lay", 10)],
    "subdet x layer x zside": [("sd", "b_sd", 8), ("lay", "b_lay", 10),
                               ("side", "b_side", 3)],
    "+ the 18 classes": [("sd", "b_sd", 8), ("lay", "b_lay", 10),
                         ("side", "b_side", 3), ("cls18x", "b_cls18", 18)],
    "orientation group": [("og", "b_og", 0)],
    "orientation group x class": [("og", "b_og", 0), ("cls18x", "b_cls18", 18)],
}
t = np.array([-4.89, -3.57, +0.12]) * 1e-3 / CM
print(f"{'key':30s} {'ncell':>6s} {'central pred (1e-3)':>28s} "
      f"{'chi2/3':>9s} {'sigma':>7s}")
for nm, spec in SPECS.items():
    Lall, mall, vall = [[], [], []], [], []
    ncell = 0
    for isy, (p, okm) in enumerate(((px, okx), (py, oky))):
        kh = np.zeros(len(h["sd"]), np.int64)
        kb = np.zeros(len(b["b_sd"]), np.int64)
        for a_, b_, n in spec:
            if n == 0:
                u = np.unique(np.concatenate([h[a_], b[b_]]))
                kh = kh * len(u) + np.searchsorted(u, h[a_])
                kb = kb * len(u) + np.searchsorted(u, b[b_])
            else:
                off = 1 if a_ == "side" else 0
                kh = kh * n + np.asarray(h[a_], np.int64) + off
                kb = kb * n + np.asarray(b[b_], np.int64) + off
        kh = kh * 2 + isy
        kb = kb * 2 + isy
        sel = b["b_isy"] == isy
        u, inv = np.unique(kh[okm], return_inverse=True)
        n_ = np.bincount(inv)
        m1 = np.bincount(inv, weights=p[okm]) / np.maximum(n_, 1)
        c = p[okm] - m1[inv]
        var = np.bincount(inv, weights=c ** 2) / np.maximum(n_, 1) / np.maximum(n_, 1)
        keep = n_ >= 100
        cells = u[keep]
        ncell += len(cells)
        idx = {int(k): j for j, k in enumerate(cells)}
        Lb = np.zeros((3, len(cells)))
        for i in range(3):
            s2 = sel & (bb == i)
            uu, ii = np.unique(kb[s2], return_inverse=True)
            Lk = np.bincount(ii, weights=aw[s2]) / nb[i]
            for k, L in zip(uu, Lk):
                j = idx.get(int(k))
                if j is not None:
                    Lb[i, j] = L
        for i in range(3):
            Lall[i].append(Lb[i])
        mall.append(m1[keep])
        vall.append(var[keep])
    L = np.vstack([np.concatenate(Lall[i]) for i in range(3)])
    mu = np.concatenate(mall)
    V = np.concatenate(vall)
    r = t - L @ mu
    A = (L * V) @ L.T
    chi2 = float(r @ np.linalg.solve(A, r))
    print(f"{nm:30s} {ncell:6d} "
          + "".join(f"{CM*(L@mu)[i]*1e3:+9.2f}" for i in range(3))
          + f" {chi2/3:9.1f} {np.sqrt(chi2):7.1f}")
print("\nMEASURED: -4.89 +- 2.46 / -3.57 +- 2.71 / +0.12 +- 2.75 (1e-3)")
