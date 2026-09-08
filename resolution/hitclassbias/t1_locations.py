#!/usr/bin/env python3
"""Step 1a: the per-subdetector / per-layer residual LOCATION, and the
bending-sense weights the same detectors carry in the fit."""
import numpy as np

SDN = {1: "BPix", 2: "FPix", 3: "TIB", 4: "TID", 5: "TOB", 6: "TEC"}
h = np.load("data/hits_mugun_ul16_h2.npz")
b = np.load("data/blocks_mugun_ul16_260903x.npz")

px = h["pullx"].astype(np.float64)
ok = np.isfinite(px)
print("=== LOCAL-FRAME pull_x by subdet/layer (IDEAL geometry gun) ===")
print(f"{'det':10s} {'n':>8s} {'median':>9s} {'mean(t)':>9s} {'err':>7s} "
      f"{'rms':>6s} {'skew':>7s}")
for s in range(1, 7):
    for l in range(0, 10):
        m = ok & (h["sd"] == s) & (h["lay"] == l)
        if m.sum() < 2000:
            continue
        v = px[m]
        vt = v[np.abs(v) < 10]
        print(f"{SDN[s]}-{l:<6d} {m.sum():8d} {np.median(v):+9.4f} "
              f"{vt.mean():+9.4f} {vt.std()/np.sqrt(len(vt)):7.4f} "
              f"{vt.std():6.3f} "
              f"{((vt-vt.mean())**3).mean()/vt.std()**3:+7.3f}")

py = h["pully"].astype(np.float64)
oky = np.isfinite(py)
print("\n=== LOCAL-FRAME pull_y (pixels only) ===")
for s in (1, 2):
    for l in range(0, 4):
        m = oky & (h["sd"] == s) & (h["lay"] == l)
        if m.sum() < 2000:
            continue
        v = py[m]
        vt = v[np.abs(v) < 10]
        print(f"{SDN[s]}-{l:<6d} {m.sum():8d} {np.median(v):+9.4f} "
              f"{vt.mean():+9.4f} {vt.std()/np.sqrt(len(vt)):7.4f} "
              f"{vt.std():6.3f} "
              f"{((vt-vt.mean())**3).mean()/vt.std()**3:+7.3f}")

a = np.sqrt(b["b_a2"].astype(np.float64))
s_ = b["b_s"].astype(np.float64)
print("\n=== the fit's BENDING SENSE per subdet/layer (trackres arm) ===")
print(f"{'det':10s} {'nblk':>9s} {'<s>':>7s} {'sum s a/sum a':>14s} "
      f"{'sum s a':>10s} {'mean a':>8s}")
tot = a.sum()
for isy, tag in ((0, "x"), (1, "y")):
    print(f"-- coordinate {tag} --")
    for sd in range(1, 7):
        for l in range(0, 10):
            m = (b["b_sd"] == sd) & (b["b_lay"] == l) & (b["b_isy"] == isy)
            if m.sum() < 5000:
                continue
            print(f"{SDN[sd]}-{l:<6d} {m.sum():9d} {s_[m].mean():+7.3f} "
                  f"{(s_[m]*a[m]).sum()/a[m].sum():+14.3f} "
                  f"{(s_[m]*a[m]).sum()/tot:+10.4f} {a[m].mean():8.4f}")
