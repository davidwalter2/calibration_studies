#!/usr/bin/env python3
"""Step 2 (general): the no-free-parameter prediction under a SET of join
keys, from the coarsest (subdetector) to the finest the statistics allow.

The prediction is linear in mu, so its statistical error from the finite
`hitres` sample is propagated exactly:

    <dz>_band = sum_k L_k(band) mu_k ,  L_k = <sum_{b in k} s_b a_b>_band
    var       = sum_k L_k^2 var(mu_k)

which also says exactly which key carries the prediction (the L_k mu_k table).
"""
import argparse

import numpy as np

SDN = {1: "BPix", 2: "FPix", 3: "TIB", 4: "TID", 5: "TOB", 6: "TEC"}
BANDS = [(0.0, 0.9, "barrel |eta|<0.9"), (0.9, 1.6, "middle 0.9-1.6"),
         (1.6, 2.4, "endcap 1.6-2.4")]
DXB = np.array([-1e9, -0.20, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 0.20, 1e9])
DYB = np.array([-1e9, -0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6, 1e9])
UPB = np.array([-1e9, 0.10, 0.25, 0.50, 1e9])


def dig(v, e):
    return np.clip(np.digitize(v, e[1:-1]), 0, len(e) - 2).astype(np.int64)


def make_key(d, spec, pfx=""):
    g = lambda n: d[pfx + n]
    k = np.zeros(len(g("sd")), dtype=np.int64)
    for part in spec.split("+"):
        if part == "sd":
            v, n = g("sd").astype(np.int64), 8
        elif part == "lay":
            v, n = g("lay").astype(np.int64), 10
        elif part == "side":
            v, n = g("side").astype(np.int64) + 1, 3
        elif part == "og":
            u, v = np.unique(g("og"), return_inverse=True)
            n = len(u) + 1
            v = v.astype(np.int64)
        elif part == "mod":
            u, v = np.unique(d[pfx + "detid"], return_inverse=True)
            n = len(u) + 1
            v = v.astype(np.int64)
        elif part == "cls":
            v = (g("cls18x") if pfx == "" else d["b_cls18"]).astype(np.int64)
            n = 18
        elif part == "N":
            v, n = np.clip(g("N").astype(np.int64), 0, 6), 7
        elif part == "q":
            v, n = np.clip(g("qbin").astype(np.int64), -1, 3) + 1, 5
        elif part == "edge":
            v, n = (g("onedge").astype(np.int64) > 0).astype(np.int64), 2
        elif part == "dxdz":
            v, n = dig(g("dxdz"), DXB), len(DXB) - 1
        elif part == "sdxdz":
            v, n = (g("dxdz") > 0).astype(np.int64), 2
        elif part == "dydz":
            v, n = dig(g("dydz"), DYB), len(DYB) - 1
        elif part == "sdydz":
            v, n = (g("dydz") > 0).astype(np.int64), 2
        elif part == "up":
            v, n = dig(g("uproj"), UPB), len(UPB) - 1
        else:
            raise SystemExit(part)
        k = k * n + v
    return k


def run(h, b, spec, nmin, coord_split=True):
    kh_x = make_key(h, spec)
    kb = make_key(b, spec, "b_")
    isy_b = b["b_isy"] == 1
    if coord_split:
        kh_x = kh_x * 2
        kb = kb * 2 + isy_b.astype(np.int64)
    aw = b["b_s"].astype(np.float64) * np.sqrt(b["b_a2"].astype(np.float64))
    trk = np.repeat(np.arange(len(b["nblk"])), b["nblk"])
    eta = np.abs(b["t_eta"].astype(np.float64))
    band = np.digitize(eta, [0.9, 1.6])
    ntr = len(b["nblk"])

    # mu bank
    mus, errs, ns, keys = {}, {}, {}, {}
    for isy, pk in ((0, "pullx"), (1, "pully")):
        p = h[pk].astype(np.float64)
        ok = np.isfinite(p) & (np.abs(p) < 10)
        kk = (kh_x + isy) if coord_split else kh_x
        kk = kk[ok]
        u, inv = np.unique(kk, return_inverse=True)
        n = np.bincount(inv)
        s1 = np.bincount(inv, weights=p[ok])
        s2 = np.bincount(inv, weights=p[ok] ** 2)
        m_ = s1 / np.maximum(n, 1)
        var = np.maximum(s2 / np.maximum(n, 1) - m_ * m_, 0.) / np.maximum(n, 1)
        ok2 = n >= nmin
        for kv, mv, ev, nv in zip(u[ok2], m_[ok2], var[ok2], n[ok2]):
            mus[int(kv)] = mv
            errs[int(kv)] = ev
            ns[int(kv)] = nv
        if not coord_split:
            break

    mu = np.array([mus.get(int(k), 0.) for k in kb])
    va = np.array([errs.get(int(k), 0.) for k in kb])
    miss = np.array([int(k) not in mus for k in kb])

    out = []
    for ib in range(3):
        sel = band[trk] == ib
        w = np.bincount(trk[sel], weights=(aw * mu)[sel], minlength=ntr)
        pred = w[band == ib].mean()
        # exact error: group the per-band lever arm by key
        u, inv = np.unique(kb[sel], return_inverse=True)
        L = np.bincount(inv, weights=aw[sel]) / (band == ib).sum()
        vk = np.array([errs.get(int(k), 0.) for k in u])
        err = np.sqrt(np.sum(L ** 2 * vk))
        out.append((pred, err))
    return out, len(mus), miss.mean(), np.abs(aw[miss]).sum() / np.abs(aw).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hits", default="data/hits_mugun_ul16_h2.npz")
    ap.add_argument("--blocks", default="data/blocks_mugun_ul16_260903x.npz")
    ap.add_argument("--nmin", type=int, default=150)
    ap.add_argument("--specs", default=(
        "sd,sd+lay,og,og+sdxdz,sd+dxdz,sd+lay+dxdz,og+dxdz,sd+cls,og+cls,"
        "sd+lay+cls,sd+N+up,sd+dxdz+dydz,sd+lay+dxdz+dydz,mod,og+N,sd+cls+dxdz"))
    a = ap.parse_args()
    h = np.load(a.hits)
    b = np.load(a.blocks)

    # the measurement, for reference
    z = b["t_z"].astype(np.float64)
    eta = np.abs(b["t_eta"].astype(np.float64))
    q = b["t_q"].astype(np.float64)
    sig = b["t_sigma"].astype(np.float64)
    vgf = b["t_vgf"].astype(np.float64)
    pf = b["t_pt"].astype(np.float64) * np.cosh(b["t_eta"].astype(np.float64))
    ai = sig * pf * (1 - vgf)
    den = 1 - ai * q * z
    zt = np.where(np.abs(den) > 1e-3, z / den, np.nan)
    good = np.isfinite(zt) & (np.abs(zt) < 30)
    band = np.digitize(eta, [0.9, 1.6])
    rng = np.random.default_rng(20260909)
    print("MEASURED charge-even, units 1e-3:")
    meas = []
    for ib, (_, _, nm) in enumerate(BANDS):
        m = good & (band == ib)
        f = zt * np.exp(-0.05 * zt * zt)
        de = 0.5 * (f[m & (q > 0)].mean() + f[m & (q < 0)].mean())
        idx = np.where(m)[0]
        bs = np.empty(200)
        for i in range(200):
            k = idx[rng.integers(0, len(idx), len(idx))]
            bs[i] = 0.5 * (f[k][q[k] > 0].mean() + f[k][q[k] < 0].mean())
        dm = 0.5 * (zt[m & (q > 0)].mean() + zt[m & (q < 0)].mean())
        meas.append((de, bs.std(), dm))
        print(f"  {nm:20s} odd(u=.05) {de*1e3:+7.2f} +- {bs.std()*1e3:4.2f}   "
              f"<z> {dm*1e3:+7.2f}")

    print(f"\nPREDICTED <dz> per band (units 1e-3), nmin={a.nmin}. "
          f"The odd moment it implies is 0.867x this at u=0.05.")
    print(f"{'key':22s} {'nkey':>6s} {'miss%':>6s} "
          f"{'barrel':>16s} {'middle':>16s} {'endcap':>16s}")
    for spec in a.specs.split(","):
        try:
            o, nk, fm, fw = run(h, b, spec, a.nmin)
        except Exception as e:
            print(f"{spec:22s} ERROR {type(e).__name__} {e}")
            continue
        s = f"{spec:22s} {nk:6d} {fw*100:6.2f} "
        for p, e in o:
            s += f"{p*1e3:+10.2f}+-{e*1e3:4.2f} "
        print(s)
    print(f"\nTARGET (measured <z>): "
          + " ".join(f"{m[2]*1e3:+.2f}" for m in meas))


if __name__ == "__main__":
    main()
