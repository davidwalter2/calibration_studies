#!/usr/bin/env python3
"""Are the two arms the SAME detector configuration? If the hit errors or the
angle distributions differ, a mu measured on one cannot be used on the other."""
import numpy as np
SDN = {1: "BPix", 2: "FPix", 3: "TIB", 4: "TID", 5: "TOB", 6: "TEC"}
h = np.load("data/hits_mugun_ul16_h2.npz")
b = np.load("data/blocks_mugun_ul16_260903x.npz")
print(f"{'det':8s} {'hitres <dxerr>':>16s} {'trackres <err>':>16s} {'ratio':>8s} "
      f"{'hitres <dxdz>':>14s} {'trackres':>10s} {'h |dxdz|':>9s} {'t |dxdz|':>9s}")
for sd in range(1, 7):
    mh = (h["sd"] == sd) & np.isfinite(h["pullx"])
    mb = (b["b_sd"] == sd) & (b["b_isy"] == 0) & (b["b_err"] > 0)
    if mh.sum() < 100 or mb.sum() < 100:
        continue
    print(f"{SDN[sd]:8s} {np.median(h['dxerr'][mh]):16.6g} "
          f"{np.median(b['b_err'][mb]):16.6g} "
          f"{np.median(b['b_err'][mb])/np.median(h['dxerr'][mh]):8.4f} "
          f"{np.mean(h['dxdz'][mh]):14.4f} {np.mean(b['b_dxdz'][mb]):10.4f} "
          f"{np.mean(np.abs(h['dxdz'][mh])):9.4f} "
          f"{np.mean(np.abs(b['b_dxdz'][mb])):9.4f}")
print("\nhits/track  hitres %.2f   trackres(x only) %.2f"
      % (len(h["pullx"]) / len(np.unique(h["teta"])), 0.))
print("class-18 share (influence-weighted, trackres) vs count share (hitres):")
aw = np.abs(np.sqrt(b["b_a2"].astype(np.float64)))
for c in range(18):
    mb = b["b_cls18"] == c
    mh = (h["cls18x"] == c) & np.isfinite(h["pullx"])
    if mb.sum() < 1000:
        continue
    print(f"  cls {c:2d}: trackres n {mb.sum():8d} ({aw[mb].sum()/aw.sum()*100:5.2f}% infl) "
          f"| hitres n {mh.sum():8d} ({mh.sum()/np.isfinite(h['pullx']).sum()*100:5.2f}%)")
