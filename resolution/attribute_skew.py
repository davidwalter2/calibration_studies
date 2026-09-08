#!/usr/bin/env python3
"""Attribute the CHARGE-EVEN track skew: is it hit-position (CPE class)?

`cf_skew_closure.py --bin-eta` measured `<z e^{-u z^2}>` data - model at
-0.0050 (barrel) -> +0.0001 (endcap) on the tight-stepper gun, with the
MODEL's own odd moment at ~0 in every band. So `data - model` IS the data's
odd moment to 1e-5, and every split below can be done on the data alone.

Three questions, and the hit-position hypothesis makes a definite prediction
for each:

1. **charge.** A hit displacement in a fixed LOCAL direction bends both charges
   the same way, so the skew must be charge-EVEN. Field, alignment and the
   momentum scale are charge-ODD. Split by charge: same sign and size = even.
2. **phi.** The Lorentz drift is a fixed direction in the LOCAL frame, which
   rotates with the module, so the induced curvature bias has a definite
   GLOBAL sense and must NOT flip with phi. A skew that flips sign with phi
   would be a global-geometry effect instead.
3. **hit composition.** The drift is large in the barrel pixels and strips
   (E perpendicular B) and negligible in the forward pixel disks (drift
   parallel B). So the skew must scale with the INFLUENCE-WEIGHTED share of
   barrel-like hits, and vanish where the endcap disks carry the track.

The influence weight is `hitamp2`, the per-hit weight the track fit gives that
hit in `q/p` -- the same quantity the mass functional's per-hit-class blocks
carry -- so "share of the curvature carried by class c" is
`sum_{hits in c} amp2 / sum amp2`, not a hit count.

    python3 attribute_skew.py [--cache runs/cf_trackres_mugun_ul16_260903x_m0_k0.npz]
"""
import argparse

import numpy as np

PROBES = (0.05, 0.2)


def odd(z, u, w=None):
    return np.average(z * np.exp(-u * z * z), weights=w)


def odd_err(z, u, nboot, rng, w=None):
    n = len(z)
    if n < 50:
        return np.nan
    b = np.empty(nboot)
    for i in range(nboot):
        k = rng.integers(0, n, n)
        b[i] = odd(z[k], u, None if w is None else w[k])
    return b.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache",
                    default="runs/cf_trackres_mugun_ul16_260903x_m0_k0.npz")
    ap.add_argument("--nboot", type=int, default=200)
    ap.add_argument("--zmax", type=float, default=30.0)
    a = ap.parse_args()
    rng = np.random.default_rng(20260908)

    d = np.load(a.cache, allow_pickle=True)
    z = d["z"].astype(np.float64)
    eta = np.abs(d["eta"].astype(np.float64))
    phi = d["phi"].astype(np.float64)
    q = d["charge"].astype(np.float64)
    cnt = d["hitcnt"].astype(np.int64)
    cls = d["hitcls"].astype(np.int64)
    amp2 = d["hitamp2"].astype(np.float64)
    names = [str(s) for s in d["hitclsnames"]]

    ok = np.isfinite(z) & (np.abs(z) < a.zmax)
    # per-track influence-weighted class shares
    ptr = np.concatenate([[0], np.cumsum(cnt)])
    ntr, ncl = len(cnt), len(names)
    trk = np.repeat(np.arange(ntr), cnt)
    tot = np.bincount(trk, weights=amp2, minlength=ntr)
    share = np.zeros((ntr, ncl))
    for c in range(ncl):
        m = cls == c
        share[:, c] = np.bincount(trk[m], weights=amp2[m], minlength=ntr)
    share /= np.maximum(tot, 1e-300)[:, None]
    pix = share[:, 0:8].sum(1)          # all pixel classes
    strp = share[:, 8:18].sum(1)        # all strip classes
    n1 = share[:, 8:10].sum(1)          # single-strip clusters
    print(f"{a.cache}: {ntr} tracks, {len(cls)} hits, {ok.sum()} used "
          f"(|z| < {a.zmax})")
    print(f"influence-weighted shares: pixel {np.median(pix):.3f}, "
          f"strip {np.median(strp):.3f}, single-strip {np.median(n1):.3f}\n")

    BANDS = [(0.0, 0.9, "|eta| 0.0-0.9"), (0.9, 1.6, "0.9-1.6"),
             (1.6, 2.4, "1.6-2.4")]

    def table(title, groups):
        print(title)
        hdr = (f"  {'cut':32s} {'n':>8s}" +
               "".join(f"{'u='+str(u):>18s}" for u in PROBES))
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for lab, m in groups:
            m = m & ok
            if m.sum() < 200:
                print(f"  {lab:32s} {int(m.sum()):8d}   (too few)")
                continue
            row = f"  {lab:32s} {int(m.sum()):8d}"
            for u in PROBES:
                v = odd(z[m], u)
                e = odd_err(z[m], u, a.nboot, rng)
                row += f"  {1e3*v:+8.2f} +-{1e3*e:5.2f}"
            print(row)
        print()

    # 1. charge
    for lo, hi, lab in BANDS:
        b = (eta >= lo) & (eta < hi)
        table(f"1. CHARGE, {lab}   (charge-even <=> same sign and size)",
              [(f"{lab}, q = +1", b & (q > 0)),
               (f"{lab}, q = -1", b & (q < 0)),
               (f"{lab}, both",   b)])

    # 2. phi
    for lo, hi, lab in BANDS[:1] + BANDS[2:]:
        b = (eta >= lo) & (eta < hi)
        groups = []
        for k in range(4):
            p0, p1 = -np.pi + k * np.pi / 2, -np.pi + (k + 1) * np.pi / 2
            groups.append((f"{lab}, phi [{p0:+.2f},{p1:+.2f}]",
                           b & (phi >= p0) & (phi < p1)))
        table(f"2. PHI, {lab}   (a LOCAL drift must NOT flip sign with phi)",
              groups)

    # 3. hit composition
    for lo, hi, lab in BANDS:
        b = (eta >= lo) & (eta < hi)
        if b.sum() < 1000:
            continue
        qp = np.percentile(pix[b & ok], [33, 67])
        qn = np.percentile(n1[b & ok], [50, 90])
        table(f"3. HIT COMPOSITION, {lab}   (influence-weighted shares)",
              [(f"{lab}, pixel share < {qp[0]:.3f}", b & (pix < qp[0])),
               (f"{lab}, pixel share mid",           b & (pix >= qp[0]) & (pix < qp[1])),
               (f"{lab}, pixel share > {qp[1]:.3f}", b & (pix >= qp[1])),
               (f"{lab}, single-strip < {qn[0]:.3f}", b & (n1 < qn[0])),
               (f"{lab}, single-strip > {qn[1]:.3f}", b & (n1 >= qn[1]))])

    print("The model's own odd moment is ~0 in every band (measured: +5e-5 / "
          "-0e-5 / +0e-5),\nso these data numbers ARE `data - model` to 1e-5. "
          "Units: 1e-3.")


if __name__ == "__main__":
    main()
