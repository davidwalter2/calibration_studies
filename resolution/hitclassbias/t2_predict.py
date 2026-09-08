#!/usr/bin/env python3
"""Step 2: propagate the measured per-class LOCATION through the fit's own
per-hit influence weights, with NO free parameter.

    delta z_i = sum_{b in i} s_b sqrt(a2_b) mu(key_b)

`mu` is the mean pull of the hit's key measured on the `hitres` arm (ideal
geometry, gen-anchored, so it IS the CPE location bias); `s_b sqrt(a2_b)` is
the fit's own signed response, from `resinfbv` and `resinfvarv`.

The FIRST-order consequence of a location bias is a MEAN SHIFT of z, and the
odd moment `<z e^{-u z^2}>` of a shifted unit Gaussian is
`m (1+2u)^{-3/2}` to first order -- so the prediction is tested on both.
"""
import argparse

import numpy as np

BANDS = [(0.0, 0.9, "|eta| 0.0-0.9"), (0.9, 1.6, "0.9-1.6"),
         (1.6, 2.4, "1.6-2.4")]
PROBES = (0.05, 0.2)


def keys_of(d, level, pfx=""):
    """The join key, from variables both trees carry."""
    sd = d[pfx + "sd"].astype(np.int64)
    if level == "subdet":
        base = sd
    elif level == "layer":
        base = sd * 100 + d[pfx + "lay"]
    elif level == "og":
        base = d[pfx + "og"].astype(np.int64)
    elif level == "module":
        base = d[pfx + "detid"].astype(np.int64)
    else:
        raise SystemExit(level)
    return base


def build_mu(h, level, withcls, nmin, coord):
    """mean pull per key on the hitres arm. coord 0 = x/phi, 1 = pixel y."""
    p = h["pullx" if coord == 0 else "pully"].astype(np.float64)
    ok = np.isfinite(p) & (np.abs(p) < 10.)
    k = keys_of(h, level)
    if withcls:
        k = k * 32 + (h["cls18x"] if coord == 0 else h["cls18y"])
    k = k[ok]
    p = p[ok]
    u, inv = np.unique(k, return_inverse=True)
    n = np.bincount(inv)
    s1 = np.bincount(inv, weights=p)
    s2 = np.bincount(inv, weights=p * p)
    mu = s1 / np.maximum(n, 1)
    var = np.maximum(s2 / np.maximum(n, 1) - mu * mu, 0.)
    err = np.sqrt(var / np.maximum(n, 1))
    bad = n < nmin
    mu[bad] = 0.
    err[bad] = 0.
    return dict(zip(u, mu)), dict(zip(u, err)), dict(zip(u, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hits", default="data/hits_mugun_ul16_h2.npz")
    ap.add_argument("--blocks", default="data/blocks_mugun_ul16_260903x.npz")
    ap.add_argument("--level", default="og",
                    choices=["subdet", "layer", "og", "module"])
    ap.add_argument("--withcls", action="store_true")
    ap.add_argument("--nmin", type=int, default=200)
    ap.add_argument("--nboot", type=int, default=200)
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    rng = np.random.default_rng(20260909)

    h = np.load(a.hits)
    b = np.load(a.blocks)
    mu_x, ex, nx = build_mu(h, a.level, a.withcls, a.nmin, 0)
    mu_y, ey, ny = build_mu(h, a.level, a.withcls, a.nmin, 1)
    print(f"mu bank: level={a.level} withcls={a.withcls} "
          f"nkeys x={len(mu_x)} y={len(mu_y)} (nmin={a.nmin})")

    kb = keys_of(b, a.level, "b_")
    if a.withcls:
        kb = kb * 32 + b["b_cls18"].astype(np.int64)
    isy = b["b_isy"] == 1
    mu = np.zeros(len(kb))
    miss = np.zeros(len(kb), bool)
    for sel, bank in ((~isy, mu_x), (isy, mu_y)):
        idx = np.where(sel)[0]
        vals = np.array([bank.get(int(k), np.nan) for k in kb[idx]])
        miss[idx] = ~np.isfinite(vals)
        mu[idx] = np.nan_to_num(vals)
    aw = b["b_s"].astype(np.float64) * np.sqrt(b["b_a2"].astype(np.float64))
    print(f"blocks {len(kb)}, key not in bank: {miss.mean()*100:.2f}% "
          f"({np.abs(aw[miss]).sum()/np.abs(aw).sum()*100:.2f}% of |s a|)")

    ptr = np.concatenate([[0], np.cumsum(b["nblk"])])
    trk = np.repeat(np.arange(len(b["nblk"])), b["nblk"])
    dz = np.bincount(trk, weights=aw * mu, minlength=len(b["nblk"]))

    z = b["t_z"].astype(np.float64)
    eta = np.abs(b["t_eta"].astype(np.float64))
    q = b["t_q"].astype(np.float64)
    sig = b["t_sigma"].astype(np.float64)
    vgf = b["t_vgf"].astype(np.float64)
    pfit = b["t_pt"].astype(np.float64) * np.cosh(b["t_eta"].astype(np.float64))
    a_i = sig * pfit * (1.0 - vgf)
    den = 1.0 - a_i * q * z
    ztf = np.where(np.abs(den) > 1e-3, z / den, np.nan)
    good = np.isfinite(ztf) & (np.abs(ztf) < 30.)
    print(f"tracks {len(z)}, used {good.sum()}; truth-referenced transform "
          f"a_i median {np.median(a_i):.5f}")

    def stat(f, m):
        v = f(m)
        bs = np.empty(a.nboot)
        idx = np.where(m)[0]
        for i in range(a.nboot):
            k = rng.integers(0, len(idx), len(idx))
            mm = np.zeros(len(m), bool)
            bs[i] = f(idx[k], asidx=True)
        return v, bs.std()

    def even(arr, m):
        """charge-even mean of arr: (mean over q+ plus mean over q-)/2"""
        p = arr[m & (q > 0)].mean()
        n = arr[m & (q < 0)].mean()
        return 0.5 * (p + n), 0.5 * (p - n)

    def boot_even(arr, m, nb):
        idx = np.where(m)[0]
        out = np.empty(nb)
        for i in range(nb):
            k = idx[rng.integers(0, len(idx), len(idx))]
            qq = q[k]
            out[i] = 0.5 * (arr[k][qq > 0].mean() + arr[k][qq < 0].mean())
        return out.std()

    print("\n=== A. MEAN SHIFT <z> (charge-even), units 1e-3 ===")
    print(f"{'band':16s} {'n':>7s} {'data even':>12s} {'+-':>6s} "
          f"{'PREDICT':>10s} {'+-':>6s} {'data odd':>10s}")
    for lo, hi, nm in BANDS:
        m = good & (eta >= lo) & (eta < hi)
        de, do = even(ztf, m)
        pe, po = even(dz, m)
        ee = boot_even(ztf, m, a.nboot)
        pp = boot_even(dz, m, a.nboot)
        print(f"{nm:16s} {m.sum():7d} {de*1e3:+12.2f} {ee*1e3:6.2f} "
              f"{pe*1e3:+10.2f} {pp*1e3:6.2f} {do*1e3:+10.2f}")

    print("\n=== B. ODD MOMENT <z e^{-u z^2}> (charge-even), units 1e-3 ===")
    for u in PROBES:
        print(f"-- u = {u}")
        print(f"{'band':16s} {'data even':>12s} {'+-':>6s} "
              f"{'PREDICT (m/(1+2u)^1.5)':>24s}")
        for lo, hi, nm in BANDS:
            m = good & (eta >= lo) & (eta < hi)
            f = ztf * np.exp(-u * ztf * ztf)
            de, _ = even(f, m)
            ee = boot_even(f, m, a.nboot)
            pe, _ = even(dz, m)
            print(f"{nm:16s} {de*1e3:+12.2f} {ee*1e3:6.2f} "
                  f"{pe*1e3/(1+2*u)**1.5:+24.2f}")

    if a.save:
        np.savez_compressed(a.save, dz=dz.astype(np.float32),
                            z=z.astype(np.float32), eta=b["t_eta"],
                            q=q.astype(np.int8), sigma=sig.astype(np.float32),
                            vgf=vgf.astype(np.float32),
                            pt=b["t_pt"].astype(np.float32))
        print(f"\nwrote {a.save}")


if __name__ == "__main__":
    main()
