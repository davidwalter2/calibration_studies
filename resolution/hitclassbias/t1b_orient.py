#!/usr/bin/env python3
"""Does the orientation group make the bending sense coherent, and does the
local-frame location bias flip with it?"""
import numpy as np

SDN = {1: "BPix", 2: "FPix", 3: "TIB", 4: "TID", 5: "TOB", 6: "TEC"}
h = np.load("data/hits_mugun_ul16_h2.npz")
b = np.load("data/blocks_mugun_ul16_260903x.npz")
a = np.sqrt(b["b_a2"].astype(np.float64))
s_ = b["b_s"].astype(np.float64)

print("=== coherence of the bending sense, per level of keying ===")
print(f"{'level':34s} {'ngrp':>6s} {'<|<s>_g|> (a-weighted)':>24s}")


def coherence(key, mask):
    kk = key[mask]
    aa = a[mask]
    ss = s_[mask]
    u, inv = np.unique(kk, return_inverse=True)
    num = np.bincount(inv, weights=ss * aa)
    den = np.bincount(inv, weights=aa)
    good = den > 0
    return len(u), float(np.average(np.abs(num[good] / den[good]),
                                    weights=den[good]))


mx = b["b_isy"] == 0
lev = {
    "subdet": b["b_sd"].astype(np.int64),
    "subdet+layer": b["b_sd"] * 10 + b["b_lay"],
    "subdet+layer+side": (b["b_sd"] * 10 + b["b_lay"]) * 4 + b["b_side"] + 1,
    "orientation group": b["b_og"],
    "orientation group x sign(dxdz)": b["b_og"] * 2 + (b["b_dxdz"] > 0),
    "MODULE (full DetId)": b["b_detid"],
}
for k, v in lev.items():
    n, c = coherence(np.asarray(v, np.int64), mx)
    print(f"{k:34s} {n:6d} {c:24.3f}")

print("\n=== per subdet: coherence at orientation-group level ===")
for sd in range(1, 7):
    m = mx & (b["b_sd"] == sd)
    n1, c1 = coherence(np.asarray(b["b_sd"] * 10 + b["b_lay"], np.int64), m)
    n2, c2 = coherence(b["b_og"], m)
    n3, c3 = coherence(b["b_detid"], m)
    print(f"{SDN[sd]:6s} layer {c1:.3f} ({n1:3d} grp) | og {c2:.3f} "
          f"({n2:4d} grp) | module {c3:.3f} ({n3:6d} grp)")

print("\n=== does the LOCAL location bias flip with the orientation group? ===")
px = h["pullx"].astype(np.float64)
ok = np.isfinite(px) & (np.abs(px) < 10)
for sd in range(1, 7):
    m = ok & (h["sd"] == sd)
    u, inv = np.unique(h["og"][m], return_inverse=True)
    n = np.bincount(inv)
    su = np.bincount(inv, weights=px[m])
    mu = su / np.maximum(n, 1)
    keep = n > 500
    print(f"{SDN[sd]:6s} ngrp {keep.sum():4d}  mean {px[m].mean():+.4f}  "
          f"rms over groups {mu[keep].std():.4f}  "
          f"mean|mu| {np.abs(mu[keep]).mean():.4f}  "
          f"min/max {mu[keep].min():+.4f}/{mu[keep].max():+.4f}")
