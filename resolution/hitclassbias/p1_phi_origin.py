#!/usr/bin/env python3
"""WHERE do the n = 8 and n = 10 phi harmonics come from?

The azimuthal segmentation of the 2016 Phase-0 tracker, counted directly from
the DetIds in this sample:

    TEC        8 petals per wheel            -> n = 8
    BPix L1    20 ladders, inner/outer       -> 10 ladder PAIRS -> n = 10
    BPix L2    32 ladders                    -> n = 16
    FPix       24 blades, 2 panels           -> n = 24 / 12
    TIB        30 / 38 / 45 / 56 strings
    TOB        42 / 48 / 54 / 60 / 66 / 74 rods

n = 8 and n = 10 are the ONLY two that match a real structure, and they are
exactly the two harmonics the scan finds. The test: a modulation produced by
TEC petals must scale with the INFLUENCE-WEIGHTED TEC share of the curvature,
and one produced by the BPix-L1 ladder alternation with the BPix-L1 share.
A coincidence of periods will not.

    A_n^sin = 2 <x sin(n phi)>,  sigma = sqrt(2/N)   (matches c6_phi.py)
"""
import numpy as np

C = np.load("data/conv_ref903x_full.npz")
B = np.load("data/blocks_mugun_ul16_260903x.npz")
assert len(C["z"]) == len(B["t_z"]) and np.abs(C["z"] - B["t_z"]).max() == 0.

z = C["z"].astype(np.float64)
q = C["q"].astype(np.float64)
phi = C["genphi"].astype(np.float64)
eta = np.abs(C["eta"].astype(np.float64))
sig = C["sigma"].astype(np.float64)
vgf = C["vgf"].astype(np.float64)
pf = C["pt"].astype(np.float64) * np.cosh(C["eta"].astype(np.float64))
den = 1 - sig * pf * (1 - vgf) * q * z
x = np.where(np.abs(den) > 1e-3, z / den, np.nan)
good = np.isfinite(x) & (np.abs(x) < 30) & np.isfinite(phi)

trk = np.repeat(np.arange(len(B["nblk"])), B["nblk"])
a2 = B["b_a2"].astype(np.float64)
tot = np.bincount(trk, weights=a2, minlength=len(z))
def share(m):
    return np.bincount(trk[m], weights=a2[m], minlength=len(z)) / np.maximum(tot, 1e-300)
s_tec = share(B["b_sd"] == 6)
s_bp1 = share((B["b_sd"] == 1) & (B["b_lay"] == 1))
s_bpix = share(B["b_sd"] == 1)
s_tob = share(B["b_sd"] == 5)


def amp(m, n):
    xx, pp = x[m], phi[m]
    N = m.sum()
    return 2 * np.mean(xx * np.sin(n * pp)), np.sqrt(2. / N)


print("inclusive (charge-even is automatic: x sin(n phi) has no charge in it)")
for n in (8, 10, 16, 12, 24):
    a, e = amp(good, n)
    print(f"  n={n:3d}  A_sin = {a*1e3:+7.2f} +- {e*1e3:4.2f}  ({a/e:+5.1f} sigma)")

print("\n=== A_sin(n=8) vs the TEC influence share, and A_sin(n=10) vs BPix-L1 ===")
for nm, s, ns in (("TEC share", s_tec, (8, 10)),
                  ("BPix-L1 share", s_bp1, (10, 8)),
                  ("BPix (all) share", s_bpix, (10, 8)),
                  ("TOB share (control)", s_tob, (8, 10))):
    e4 = np.percentile(s[good], [0, 25, 50, 75, 100])
    e4 = np.unique(e4)
    print(f"-- {nm}")
    for n in ns:
        row = ""
        for i in range(len(e4) - 1):
            m = good & (s >= e4[i]) & ((s <= e4[i + 1]) if i == len(e4) - 2
                                       else (s < e4[i + 1]))
            if m.sum() < 5000:
                continue
            a, er = amp(m, n)
            row += f"  [{e4[i]:.3f},{e4[i+1]:.3f}) {a*1e3:+7.2f}+-{er*1e3:4.2f}"
        print(f"   n={n:2d}{row}")

print("\n=== the sharpest cut: tracks with NO TEC hit vs TEC-dominated ===")
for nm, m in (("no TEC hit", good & (s_tec == 0)),
              ("TEC share > 0.3", good & (s_tec > 0.3)),
              ("no BPix-L1 hit", good & (s_bp1 == 0)),
              ("BPix-L1 share > 0.1", good & (s_bp1 > 0.1))):
    a8, e8 = amp(m, 8)
    a10, e10 = amp(m, 10)
    print(f"  {nm:22s} n={m.sum():7d}  A8 {a8*1e3:+7.2f}+-{e8*1e3:4.2f}   "
          f"A10 {a10*1e3:+7.2f}+-{e10*1e3:4.2f}")

# ---------------------------------------------------------------------------
# THE CONTROL. Every share above is correlated with |eta|, and the modulation
# is known to grow with |eta|, so an uncontrolled split is largely re-measuring
# that. Repeat inside 12 fine |eta| bins.
print("\n=== corr of each share with |eta| ===")
for nm, s in (("TEC", s_tec), ("BPix-L1", s_bp1), ("BPix all", s_bpix),
              ("TOB", s_tob)):
    print(f"  {nm:10s} corr(share, |eta|) = "
          f"{np.corrcoef(s[good], eta[good])[0,1]:+.3f}")

print("\n=== the BPix-L1 contrast AT FIXED |eta| (12 quantile bins) ===")
idx = np.where(good)[0]
edges = np.percentile(eta[idx], np.linspace(0, 100, 13))
ib = np.clip(np.digitize(eta[idx], edges[1:-1]), 0, 11)
for n in (8, 10):
    num_hi = num_lo = 0.
    n_hi = n_lo = 0
    for j in range(12):
        k = idx[ib == j]
        hi = k[s_bp1[k] > 0]
        lo = k[s_bp1[k] == 0]
        if len(hi) < 500 or len(lo) < 500:
            continue
        num_hi += (x[hi] * np.sin(n * phi[hi])).sum()
        n_hi += len(hi)
        num_lo += (x[lo] * np.sin(n * phi[lo])).sum()
        n_lo += len(lo)
    ah, eh = 2 * num_hi / n_hi, np.sqrt(2. / n_hi)
    al, el = 2 * num_lo / n_lo, np.sqrt(2. / n_lo)
    print(f"  n={n:2d}  has BPix-L1 {ah*1e3:+7.2f}+-{eh*1e3:4.2f} (n={n_hi})   "
          f"no BPix-L1 {al*1e3:+7.2f}+-{el*1e3:4.2f} (n={n_lo})   "
          f"diff {(ah-al)*1e3:+7.2f}+-{np.hypot(eh,el)*1e3:4.2f}")
