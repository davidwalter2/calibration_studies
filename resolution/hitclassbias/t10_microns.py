#!/usr/bin/env python3
"""The locations in MICRONS, and the FPix +-z asymmetry the endcap result
would want."""
import numpy as np
SDN = {1: "BPix", 2: "FPix", 3: "TIB", 4: "TID", 5: "TOB", 6: "TEC"}
h = np.load("data/hits_mugun_ul16_h2.npz")
b = np.load("data/blocks_mugun_ul16_260903x.npz")
px = h["pullx"].astype(np.float64)
ok = np.isfinite(px) & (np.abs(px) < 10)
aw = b["b_s"].astype(np.float64) * np.sqrt(b["b_a2"].astype(np.float64))
trk = np.repeat(np.arange(len(b["nblk"])), b["nblk"])
band = np.digitize(np.abs(b["t_eta"]), [0.9, 1.6])
nb = np.bincount(band, minlength=3)
print(f"{'cell':16s} {'n':>8s} {'mean [sigma]':>13s} {'+-':>7s} "
      f"{'sigma_CPE':>12s} {'mean [um/urad]':>15s} {'3s bound':>10s} "
      f"{'L bar':>8s}{'L mid':>8s}{'L end':>8s}")
for sd in range(1, 7):
    for l in [0] + list(range(1, 10)):
        for sg in (-1, 0, 1):
            m = ok & (h["sd"] == sd) & (h["side"] == sg)
            bm = (b["b_sd"] == sd) & (b["b_isy"] == 0) & (b["b_side"] == sg)
            tag = SDN[sd]
            if l:
                m = m & (h["lay"] == l)
                bm = bm & (b["b_lay"] == l)
                tag = f"{SDN[sd]}-{l}"
            if sg:
                tag += f" z{sg:+d}"
            if m.sum() < 2000:
                continue
            v = px[m]
            e = v.std() / np.sqrt(len(v))
            s = np.median(h["dxerr"][m])
            un = 1e4 if sd != 4 and sd != 6 else 1e6   # cm->um, rad->urad
            L = [aw[bm & (band[trk] == i)].sum() / nb[i] for i in range(3)]
            print(f"{tag:16s} {m.sum():8d} {v.mean():+13.4f} {e:7.4f} "
                  f"{s*un:12.3f} {v.mean()*s*un:+15.3f} {3*e*s*un:10.3f} "
                  + "".join(f"{x:+8.4f}" for x in L))
    print()
