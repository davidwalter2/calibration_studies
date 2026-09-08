#!/usr/bin/env python3
"""Where does the propagated shift come from, and how big is the lever arm?"""
import numpy as np

SDN = {1: "BPix", 2: "FPix", 3: "TIB", 4: "TID", 5: "TOB", 6: "TEC"}
BANDS = [(0.0, 0.9, "barrel"), (0.9, 1.6, "middle"), (1.6, 2.4, "endcap")]
h = np.load("data/hits_mugun_ul16_h2.npz")
b = np.load("data/blocks_mugun_ul16_260903x.npz")

aw = b["b_s"].astype(np.float64) * np.sqrt(b["b_a2"].astype(np.float64))
trk = np.repeat(np.arange(len(b["nblk"])), b["nblk"])
eta = np.abs(b["t_eta"].astype(np.float64))
band = np.digitize(eta, [0.9, 1.6])
bband = band[trk]

print("=== the LEVER ARM: <sum_b s_b a_b> per band, split by subdet/coord ===")
print("(a common per-hit bias c gives <dz> = c * this)")
print(f"{'':14s} {'barrel':>10s} {'middle':>10s} {'endcap':>10s}")
for isy, tag in ((0, "x"), (1, "y")):
    for sd in range(1, 7):
        m = (b["b_sd"] == sd) & (b["b_isy"] == isy)
        if m.sum() < 1000:
            continue
        row = []
        for ib in range(3):
            mm = m & (bband == ib)
            row.append(np.bincount(trk[mm], weights=aw[mm],
                                   minlength=len(b["nblk"]))[band == ib].mean())
        print(f"{SDN[sd]+'_'+tag:14s} " + "".join(f"{v:+10.4f}" for v in row))
row = []
for ib in range(3):
    mm = bband == ib
    row.append(np.bincount(trk[mm], weights=aw[mm],
                           minlength=len(b["nblk"]))[band == ib].mean())
print(f"{'TOTAL':14s} " + "".join(f"{v:+10.4f}" for v in row))

print("\n=== and |sum| of the amplitudes, for scale ===")
for ib in range(3):
    mm = bband == ib
    s1 = np.bincount(trk[mm], weights=np.abs(aw[mm]),
                     minlength=len(b["nblk"]))[band == ib].mean()
    s2 = np.bincount(trk[mm], weights=aw[mm] ** 2,
                     minlength=len(b["nblk"]))[band == ib].mean()
    print(f"  band {ib}: <sum |s a|> {s1:.4f}   <sum a^2> = vgf {s2:.4f}")

print("\n=== per-subdet MEAN pull (the local-frame location) ===")
px = h["pullx"].astype(np.float64)
py = h["pully"].astype(np.float64)
for isy, (p, tag) in enumerate(((px, "x"), (py, "y"))):
    for sd in range(1, 7):
        m = np.isfinite(p) & (np.abs(p) < 10) & (h["sd"] == sd)
        if m.sum() < 1000:
            continue
        v = p[m]
        print(f"{SDN[sd]+'_'+tag:14s} n={m.sum():8d} mean {v.mean():+.4f} "
              f"+- {v.std()/np.sqrt(len(v)):.4f}  median {np.median(v):+.4f} "
              f"skew {((v-v.mean())**3).mean()/v.std()**3:+.3f}")

print("\n=== the og-level prediction, decomposed by subdet (units 1e-3) ===")
# rebuild mu per og
for isy, key in ((0, "pullx"), (1, "pully")):
    pass
mu = np.zeros(len(aw))
for isy, pk, ck in ((0, "pullx", "cls18x"), (1, "pully", "cls18y")):
    p = h[pk].astype(np.float64)
    ok = np.isfinite(p) & (np.abs(p) < 10)
    u, inv = np.unique(h["og"][ok], return_inverse=True)
    n = np.bincount(inv)
    m_ = np.bincount(inv, weights=p[ok]) / np.maximum(n, 1)
    bank = dict(zip(u, np.where(n >= 200, m_, 0.)))
    sel = (b["b_isy"] == isy)
    mu[sel] = np.array([bank.get(int(k), 0.) for k in b["b_og"][sel]])
print(f"{'':14s} {'barrel':>10s} {'middle':>10s} {'endcap':>10s}")
for isy, tag in ((0, "x"), (1, "y")):
    for sd in range(1, 7):
        m = (b["b_sd"] == sd) & (b["b_isy"] == isy)
        if m.sum() < 1000:
            continue
        row = []
        for ib in range(3):
            mm = m & (bband == ib)
            row.append(1e3 * np.bincount(trk[mm], weights=aw[mm] * mu[mm],
                                         minlength=len(b["nblk"]))[band == ib].mean())
        print(f"{SDN[sd]+'_'+tag:14s} " + "".join(f"{v:+10.3f}" for v in row))
