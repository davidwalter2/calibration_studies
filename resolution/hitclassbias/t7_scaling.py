#!/usr/bin/env python3
"""Is the observed charge-even shift a HIT effect at all?

A hit-term effect must scale with the hit share of the curvature variance,
`vgf = sum_{hit b} v_b / sigma^2`. A material-term or estimator effect does
not. This is the cheapest discriminator available and it needs no model.
"""
import numpy as np

U = 0.05
b = np.load("data/blocks_mugun_ul16_260903x.npz")
p = np.load("data/pred_track.npz")
zt = p["zt"].astype(np.float64)
good = p["good"]
q = p["q"].astype(np.float64)
eta = np.abs(b["t_eta"].astype(np.float64))
band = np.digitize(eta, [0.9, 1.6])
vgf = b["t_vgf"].astype(np.float64)
nv = b["t_nvalid"] if "t_nvalid" in b else None
f = zt * np.exp(-U * zt * zt)
pred = 0.8668 * p["dz_mean"].astype(np.float64) + \
    (-0.0394) * p["dz_k3"].astype(np.float64)
rng = np.random.default_rng(7)


def ev(m, arr=f, nb=200):
    idx = np.where(m)[0]
    v = 0.5 * (arr[m & (q > 0)].mean() + arr[m & (q < 0)].mean())
    bs = np.empty(nb)
    for i in range(nb):
        k = idx[rng.integers(0, len(idx), len(idx))]
        bs[i] = 0.5 * (arr[k][q[k] > 0].mean() + arr[k][q[k] < 0].mean())
    return v, bs.std()


print("=== charge-even odd moment (u=0.05) vs the HIT SHARE vgf, 1e-3 ===")
print("a hit-term effect must be proportional to vgf; a material or "
      "estimator one need not be")
print(f"{'band':10s} {'vgf tertile':>26s} {'n':>7s} {'<vgf>':>7s} "
      f"{'measured':>10s} {'+-':>5s} {'pred':>7s} {'meas/vgf':>9s}")
for ib, nm in enumerate(("barrel", "middle", "endcap")):
    m0 = good & (band == ib)
    e = np.percentile(vgf[m0], [0, 33.33, 66.67, 100])
    for i in range(3):
        m = m0 & (vgf >= e[i]) & (vgf <= e[i + 1] if i == 2 else vgf < e[i + 1])
        v, s = ev(m)
        print(f"{nm:10s} {f'[{e[i]:.3f},{e[i+1]:.3f})':>26s} {m.sum():7d} "
              f"{vgf[m].mean():7.3f} {v*1e3:+10.2f} {s*1e3:5.2f} "
              f"{pred[m].mean()*1e3:+7.2f} {v*1e3/vgf[m].mean():+9.1f}")
print()
print("=== and vs the number of valid hits ===")
nvh = b["t_nvalid"].astype(np.float64) if "t_nvalid" in b.files else None
if nvh is None:
    nvh = np.array([0.])
    print("(nvalid not cached)")
else:
    for ib, nm in enumerate(("barrel", "middle", "endcap")):
        m0 = good & (band == ib)
        e = np.percentile(nvh[m0], [0, 33.33, 66.67, 100])
        for i in range(3):
            m = m0 & (nvh >= e[i]) & (nvh <= e[i + 1] if i == 2 else nvh < e[i + 1])
            if m.sum() < 5000:
                continue
            v, s = ev(m)
            print(f"{nm:10s} nhit [{e[i]:.0f},{e[i+1]:.0f}) {m.sum():7d} "
                  f"{v*1e3:+8.2f} +- {s*1e3:4.2f}")
