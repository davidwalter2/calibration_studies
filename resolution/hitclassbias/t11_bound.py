#!/usr/bin/env python3
"""The MODEL-INDEPENDENT bound. Whatever the per-cell locations really are,
they are measured; so the largest charge-even shift the hit-LOCATION channel
can produce is

    max |sum_k L_k mu_k|  subject to  |mu_k - muhat_k| <= 3 sigma_k
      = |sum_k L_k muhat_k| + 3 sum_k |L_k| sigma_k

evaluated at several key granularities (the bound grows like sqrt(ncell), so
the coarse keys give the strong statement and the fine ones the honest one).
"""
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


def keyset(spec):
    kh = np.zeros(len(h["sd"]), np.int64)
    kb = np.zeros(len(b["b_sd"]), np.int64)
    for p, n in spec:
        kh = kh * n + np.asarray(h[p[0]], np.int64) + (1 if p[0] == "side" else 0)
        kb = kb * n + np.asarray(b[p[1]], np.int64) + (1 if p[0] == "side" else 0)
    return kh, kb


SPECS = {
    "subdet": [(("sd", "b_sd"), 8)],
    "subdet x layer": [(("sd", "b_sd"), 8), (("lay", "b_lay"), 10)],
    "subdet x layer x zside": [(("sd", "b_sd"), 8), (("lay", "b_lay"), 10),
                               (("side", "b_side"), 3)],
    "subdet x layer x zside x class": [(("sd", "b_sd"), 8), (("lay", "b_lay"), 10),
                                       (("side", "b_side"), 3),
                                       (("cls18x", "b_cls18"), 18)],
}
print(f"{'key':34s} {'ncell':>6s} | "
      + "".join(f"{n:>28s}" for n in ("barrel", "middle", "endcap")))
print(f"{'':34s} {'':>6s} | "
      + "".join(f"{'central':>12s}{'+- 3s bound':>16s}" for _ in range(3)))
for nm, spec in SPECS.items():
    kh0, kb0 = keyset(spec)
    tot = np.zeros(3)
    bnd = np.zeros(3)
    ncell = 0
    Ls, Vs = [[], [], []], []
    for isy, (p, okm) in enumerate(((px, okx), (py, oky))):
        kh = kh0 * 2 + isy
        kb = kb0 * 2 + isy
        sel = (b["b_isy"] == isy)
        u, inv = np.unique(kh[okm], return_inverse=True)
        n = np.bincount(inv)
        m1 = np.bincount(inv, weights=p[okm]) / np.maximum(n, 1)
        c = p[okm] - m1[inv]
        sd_ = np.sqrt(np.bincount(inv, weights=c ** 2) / np.maximum(n, 1)
                      / np.maximum(n, 1))
        ok = n >= 100
        mm = dict(zip(u[ok], m1[ok]))
        ss = dict(zip(u[ok], sd_[ok]))
        ncell += int(ok.sum())
        for i in range(3):
            s2 = sel & (bb == i)
            uu, ii = np.unique(kb[s2], return_inverse=True)
            L = np.bincount(ii, weights=aw[s2]) / nb[i]
            mv = np.array([mm.get(int(k), 0.) for k in uu])
            sv = np.array([ss.get(int(k), 0.) for k in uu])
            tot[i] += float(L @ mv)
            bnd[i] += 3 * float(np.abs(L) @ sv)
            Ls[i].append(L)
            if i == 0:
                Vs.append(sv ** 2)
    # chi2 of the displacement needed to hit the target (3 constraints)
    try:
        Lm = np.vstack([np.concatenate(Ls[i]) for i in range(3)])
        V = np.concatenate(Vs)
        t = np.array([-4.89, -3.57, +0.12]) * 1e-3 / CM
        r = t - Lm @ np.concatenate([np.zeros(0)] + [np.zeros(0)]) if False else None
    except Exception:
        pass
    print(f"{nm:34s} {ncell:6d} | "
          + "".join(f"{CM*tot[i]*1e3:+12.2f}{CM*bnd[i]*1e3:16.2f}"
                    for i in range(3)))
print("\nMEASURED charge-even odd moment (1e-3):  -4.89 +- 2.46   "
      "-3.57 +- 2.71   +0.12 +- 2.75")
