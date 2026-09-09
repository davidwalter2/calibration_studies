#!/usr/bin/env python3
"""Is the n = 10 harmonic the BPix-L1 inner/outer LADDER alternation?

`p1_phi_origin.py` showed both harmonics are carried by tracks WITH a BPix-L1
hit (n=10: +30.46 +- 2.97 against +0.82 +- 4.63 without, 5.4 sigma with |eta|
controlled) and proposed that n = 10 is the 20-ladder inner/outer alternation
of BPix layer 1 -- 20 ladders, alternating radius, so 10 pairs.

That was inferred through an influence SHARE. This tests it HEAD-ON, on the
ladder the track actually hit. For PXB the DetId packs
`layer = (id>>16)&0xF`, `ladder = (id>>8)&0xFF`, `module = (id>>2)&0x3F`.

THE PREDICTION, and it is falsifiable. If the mechanism is the inner/outer
alternation then <x> must depend on the ladder's PARITY and the alternation
amplitude must account for A_sin(n=10). If <x> instead varies smoothly with the
ladder's azimuth with no parity structure, the n = 10 is something else that
merely lives at that radius.

    python3 p2_ladder.py
"""
import numpy as np

C = np.load("data/conv_ref903x_full.npz")
B = np.load("data/blocks_mugun_ul16_260903x.npz")
assert len(C["z"]) == len(B["t_z"]) and np.abs(C["z"] - B["t_z"]).max() == 0.

z = C["z"].astype(np.float64); q = C["q"].astype(np.float64)
phi = C["genphi"].astype(np.float64); eta = np.abs(C["eta"].astype(np.float64))
sig = C["sigma"].astype(np.float64); vgf = C["vgf"].astype(np.float64)
pf = C["pt"].astype(np.float64) * np.cosh(C["eta"].astype(np.float64))
den = 1 - sig * pf * (1 - vgf) * q * z
x = np.where(np.abs(den) > 1e-3, z / den, np.nan)
good = np.isfinite(x) & (np.abs(x) < 30) & np.isfinite(phi)

ntr = len(z)
trk = np.repeat(np.arange(len(B["nblk"])), B["nblk"])
# ONE BLOCK PER HIT: every pixel hit contributes TWO blocks, local x and
# local y (`b_isy`), so counting blocks double-counts hits -- 460 579 blocks on
# 226 484 tracks is 2.03 per track, i.e. one hit each, not two.
isL1 = (B["b_sd"] == 1) & (B["b_lay"] == 1) & (B["b_isy"] == 0)
detid = B["b_detid"].astype(np.uint32)
ladder = ((detid >> np.uint32(8)) & np.uint32(0xFF)).astype(np.int64)

lad = np.full(ntr, -1, dtype=np.int64)      # the track's BPix-L1 ladder
nL1 = np.bincount(trk[isL1], minlength=ntr)
sel = np.where(isL1)[0]
lad[trk[sel]] = ladder[sel]                 # last wins; nL1 says if >1
u, c = np.unique(ladder[isL1], return_counts=True)
print(f"BPix-L1 blocks: {int(isL1.sum())} on {int((nL1 > 0).sum())} tracks "
      f"({100*np.mean(nL1 > 0):.1f} %); {int((nL1 > 1).sum())} tracks have >1")
print(f"ladder index: {len(u)} distinct, range {u.min()}-{u.max()}")
print("  counts: " + " ".join(f"{a}:{b}" for a, b in zip(u.tolist(), c.tolist())))

has = good & (nL1 == 1)
print(f"\n{int(has.sum())} tracks with EXACTLY one BPix-L1 hit\n")

def mean_x(m):
    return np.mean(x[m]), np.sqrt(np.mean((x[m] - np.mean(x[m]))**2) / m.sum())

print("=== <x> per BPix-L1 ladder (1e-3).  Parity alternation would show here.")
print(f"  {'ladder':>7s} {'n':>7s} {'<x>':>16s} {'<phi>':>7s}")
rows = []
for L in u.tolist():
    m = has & (lad == L)
    if m.sum() < 200:
        continue
    v, e = mean_x(m)
    rows.append((L, int(m.sum()), v, e))
    print(f"  {L:7d} {int(m.sum()):7d} {1e3*v:+9.2f} +-{1e3*e:5.2f} "
          f"{np.mean(phi[m]):+7.2f}")

L_arr = np.array([r[0] for r in rows]); v_arr = np.array([r[2] for r in rows])
e_arr = np.array([r[3] for r in rows])
ev = L_arr % 2 == 0
for nm, mm in (("even ladders", ev), ("odd ladders", ~ev)):
    w = 1 / e_arr[mm]**2
    print(f"  {nm:14s} weighted <x> = {1e3*np.sum(v_arr[mm]*w)/np.sum(w):+7.2f} "
          f"+- {1e3/np.sqrt(np.sum(w)):4.2f}")
d = (np.sum(v_arr[ev]/e_arr[ev]**2)/np.sum(1/e_arr[ev]**2)
     - np.sum(v_arr[~ev]/e_arr[~ev]**2)/np.sum(1/e_arr[~ev]**2))
sd = np.hypot(1/np.sqrt(np.sum(1/e_arr[ev]**2)), 1/np.sqrt(np.sum(1/e_arr[~ev]**2)))
print(f"\n  PARITY DIFFERENCE (even - odd) = {1e3*d:+7.2f} +- {1e3*sd:4.2f} "
      f"({d/sd:+.1f} sigma)")

print("\n=== the same AT FIXED |eta| (12 quantile bins)")
idx = np.where(has)[0]
edges = np.percentile(eta[idx], np.linspace(0, 100, 13))
ib = np.clip(np.digitize(eta[idx], edges[1:-1]), 0, 11)
se = so = 0.; ne = no = 0
for j in range(12):
    k = idx[ib == j]
    ke, ko = k[lad[k] % 2 == 0], k[lad[k] % 2 == 1]
    if len(ke) < 200 or len(ko) < 200:
        continue
    se += x[ke].sum(); ne += len(ke)
    so += x[ko].sum(); no += len(ko)
ae, ao = se/ne, so/no
sde = np.sqrt(1./ne); sdo = np.sqrt(1./no)
print(f"  even {1e3*ae:+7.2f}+-{1e3*sde:4.2f} (n={ne})   "
      f"odd {1e3*ao:+7.2f}+-{1e3*sdo:4.2f} (n={no})   "
      f"diff {1e3*(ae-ao):+7.2f}+-{1e3*np.hypot(sde,sdo):4.2f}")

print("\n=== A_sin(n) for tracks with exactly one BPix-L1 hit, and the same")
print("    computed against the LADDER-LOCAL azimuth phi - phi_ladder")
phi_lad = 2*np.pi*(lad - 1)/len(u)
for n in (8, 10, 20):
    m = has
    a = 2*np.mean(x[m]*np.sin(n*phi[m])); e = np.sqrt(2./m.sum())
    al = 2*np.mean(x[m]*np.sin(n*(phi[m]-phi_lad[m])))
    print(f"  n={n:3d}  global phi {1e3*a:+7.2f}+-{1e3*e:4.2f}   "
          f"ladder-local {1e3*al:+7.2f}+-{1e3*e:4.2f}")
