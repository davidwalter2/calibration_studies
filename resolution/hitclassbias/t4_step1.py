#!/usr/bin/env python3
"""STEP 1: per-class residual LOCATION, skew and density, in the BENDING
sense, plus the influence-weighted lever arm each class carries.

Two quantities per class k:
  mu_k    the mean residual in PULL units in the module's own local frame
          (measured on the hitres arm, ideal geometry, gen-anchored);
  L_k(band) = <sum_{b in k} s_b sqrt(a2_b)>_band, the fit's signed response,
          measured on the trackres arm.
The class's contribution to the track-level charge-even shift is L_k mu_k --
that product IS "the class bias in the bending sense", and its sign is what
the barrel/endcap reversal of the measured shift is a statement about.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

SDN = {1: "BPix", 2: "FPix", 3: "TIB", 4: "TID", 5: "TOB", 6: "TEC"}
BANDS = ["barrel", "middle", "endcap"]
h = np.load("data/hits_mugun_ul16_h2.npz")
b = np.load("data/blocks_mugun_ul16_260903x.npz")

aw = b["b_s"].astype(np.float64) * np.sqrt(b["b_a2"].astype(np.float64))
trk = np.repeat(np.arange(len(b["nblk"])), b["nblk"])
eta = np.abs(b["t_eta"].astype(np.float64))
band = np.digitize(eta, [0.9, 1.6])
bband = band[trk]
nb = np.bincount(band, minlength=3)


def loc(p, m):
    v = p[m]
    v = v[np.isfinite(v) & (np.abs(v) < 10)]
    if len(v) < 30:
        return None
    e = v.std() / np.sqrt(len(v))
    sk = ((v - v.mean()) ** 3).mean() / v.std() ** 3
    ske = np.sqrt(6. / len(v))
    return len(v), np.median(v), v.mean(), e, v.std(), sk, ske


def lever(mask):
    return np.array([aw[mask & (bband == i)].sum() / nb[i] for i in range(3)])


def row(tag, hm, bm, px):
    L = loc(px, hm)
    if L is None:
        return
    n, md, mn, er, rms, sk, ske = L
    Lv = lever(bm)
    print(f"{tag:26s} {n:8d} {md:+8.4f} {mn:+8.4f} {er:6.4f} {rms:6.3f} "
          f"{sk:+7.3f} {ske:5.3f} | "
          + "".join(f"{v:+8.4f}" for v in Lv) + " |"
          + "".join(f"{1e3*v*mn:+8.2f}" for v in Lv))


HDR = (f"{'class':26s} {'n':>8s} {'median':>8s} {'mean':>8s} {'err':>6s} "
       f"{'rms':>6s} {'skew':>7s} {'+-':>5s} | "
       f"{'L barrel':>8s}{'L mid':>8s}{'L endc':>8s} | "
       f"{'L*mu barrel':>8s}{'mid':>8s}{'endc':>8s}   [1e-3]")

px, py = h["pullx"].astype(np.float64), h["pully"].astype(np.float64)
okx, oky = np.isfinite(px), np.isfinite(py)
bx, by = b["b_isy"] == 0, b["b_isy"] == 1

print("=" * 128)
print("T1. per SUBDETECTOR and coordinate")
print(HDR)
for sd in range(1, 7):
    row(f"{SDN[sd]}  local x/phi", okx & (h["sd"] == sd), bx & (b["b_sd"] == sd), px)
for sd in (1, 2):
    row(f"{SDN[sd]}  local y", oky & (h["sd"] == sd), by & (b["b_sd"] == sd), py)

print("\n" + "=" * 128)
print("T2. per SUBDETECTOR x LAYER/DISK x z-SIDE (local x/phi)")
print(HDR)
for sd in range(1, 7):
    for l in range(1, 10):
        for sg in (-1, 0, 1):
            hm = okx & (h["sd"] == sd) & (h["lay"] == l) & (h["side"] == sg)
            bm = bx & (b["b_sd"] == sd) & (b["b_lay"] == l) & (b["b_side"] == sg)
            if hm.sum() < 1500:
                continue
            row(f"{SDN[sd]}-{l} side {sg:+d}", hm, bm, px)

print("\n" + "=" * 128)
print("T3. the CANONICAL 18 CLASSES (as `hitres_classes.py` defines them)")
import hitres_classes as HC  # noqa: E402
print(HDR)
for c in range(18):
    isy = 4 <= c < 8
    p = py if isy else px
    hm = (oky if isy else okx) & ((h["cls18y"] if isy else h["cls18x"]) == c)
    bm = (by if isy else bx) & (b["b_cls18"] == c)
    if hm.sum() < 300:
        continue
    row(f"{c:2d} {HC.CLASSES[c]}", hm, bm, p)

print("\n" + "=" * 128)
print("T4. DEGRADED classes vs the rest, per subdetector (local x/phi)")
print(HDR)
DEG = {
    "N=1 strip": lambda d, p: (d[p + "sd"] >= 3) & (d[p + "N"] == 1),
    "N>=4 strip": lambda d, p: (d[p + "sd"] >= 3) & (d[p + "N"] >= 4),
    "pix onEdge": lambda d, p: (d[p + "sd"] <= 2) & (d[p + "onedge"] == 1),
    "pix 1-col (sizeX=1)": lambda d, p: (d[p + "sd"] <= 2) & (d[p + "N"] == 1),
    "pix 1-row (sizeY=1)": lambda d, p: (d[p + "sd"] <= 2) & (d[p + "NY"] == 1),
    "pix qbin 3 (low Q)": lambda d, p: (d[p + "sd"] <= 2) & (d[p + "qbin"] == 3),
    "uProj > 0.5": lambda d, p: (d[p + "sd"] >= 3) & (d[p + "uproj"] > 0.5),
}
for nm, f in DEG.items():
    hm = okx & f(h, "")
    bm = bx & f(b, "b_")
    if hm.sum() < 300:
        continue
    row(nm, hm, bm, px)
    for sd in range(1, 7):
        h2 = hm & (h["sd"] == sd)
        b2 = bm & (b["b_sd"] == sd)
        if h2.sum() < 1000:
            continue
        row(f"   {SDN[sd]}", h2, b2, px)

print("\n" + "=" * 128)
print("T5. the ANGLE-ODD part: mean pull vs SIGNED local dx/dz (local x/phi)")
E = np.array([-1e9, -0.2, -0.1, -0.05, -0.02, 0., 0.02, 0.05, 0.1, 0.2, 1e9])
print(f"{'det':8s} " + "".join(f"{f'[{E[i]:.2f},{E[i+1]:.2f})':>14s}"
                               for i in range(len(E) - 1)))
for sd in range(1, 7):
    r = []
    for i in range(len(E) - 1):
        m = okx & (h["sd"] == sd) & (h["dxdz"] >= E[i]) & (h["dxdz"] < E[i + 1])
        L = loc(px, m)
        r.append(f"{L[2]:+7.4f}+-{L[3]:5.4f}" if L else f"{'--':>14s}")
    print(f"{SDN[sd]:8s} " + "".join(f"{x:>14s}" for x in r))
