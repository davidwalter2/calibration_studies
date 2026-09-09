#!/usr/bin/env python3
# ############################ RETRACTED ############################
# THIS SCRIPT'S RESULTS ARE WRONG. DO NOT USE IT.
#
# It pairs the convergence variants on (run, lumi, event) alone. The muon
# gun puts TWO muons in every event -- 79 965 tracks in 40 000 unique keys,
# measured 2026-09-08 -- so the searchsorted pairing below matches muon A of
# one variant to muon B of the other as soon as a track is added or dropped
# upstream. That manufactures an O(1 sigma) per-track spread and a fat tail,
# and it is what produced the retracted claims of a +3.4e-3 paired shift,
# "15 % of fits unconverged" and "the shift lives in the unconverged 15 %".
#
# The correct tooling is `extract_conv.py`'s `slot` (index within the event)
# plus `conv_common.py` / `c1_conv.py` / `c2_pair.py`, which pair on
# (run, lumi, event, slot). Their answer is that the three variants land on
# the SAME minimum: paired charge-even shift < 0.004e-3, chi2/ndof equal to
# six digits. See STATE.md, PART 3.
# ###################################################################
"""The convergence variants: the bit-check, the paired shifts, and the
before/after table for the bulk / IN decomposition.

Paired on (run, lumi, event): the fluctuation is common to the variants, so
the DIFFERENCE of the charge-even means is measured far better than either
mean on its own.
"""
import numpy as np

BANDS = ["barrel", "middle", "endcap"]
V = {k: np.load(f"data/conv_{k}.npz")
     for k in ("prod40", "base", "tight", "damp")}
rng = np.random.default_rng(83)


def key(d):
    return (d["run"].astype(np.int64) * 10**6 + d["lumi"].astype(np.int64)) \
        * 10**9 + d["event"].astype(np.int64)


def prep(d):
    z = d["z"].astype(np.float64)
    q = d["q"].astype(np.float64)
    sig = d["sigma"].astype(np.float64)
    vgf = d["vgf"].astype(np.float64)
    pf = d["pt"].astype(np.float64) * np.cosh(d["eta"].astype(np.float64))
    den = 1 - sig * pf * (1 - vgf) * q * z
    x = np.where(np.abs(den) > 1e-3, z / den, np.nan)
    return x, q, np.abs(d["eta"].astype(np.float64)), sig * pf


K = {k: key(d) for k, d in V.items()}
common = K["prod40"]
for k in ("base", "tight", "damp"):
    common = np.intersect1d(common, K[k])
print(f"common tracks across all four: {len(common)}")
IDX = {k: np.searchsorted(np.sort(K[k]), common) for k in V}
ORD = {k: np.argsort(K[k]) for k in V}
SEL = {k: ORD[k][IDX[k]] for k in V}
for k in V:
    assert (K[k][SEL[k]] == common).all(), k

X, Q, ETA, SR = {}, None, None, {}
for k, d in V.items():
    x, q, e, sr = prep(d)
    X[k] = x[SEL[k]]
    SR[k] = sr[SEL[k]]
    if Q is None:
        Q, ETA = q[SEL[k]], e[SEL[k]]
band = np.digitize(ETA, [0.9, 1.6])
NIT = {k: V[k]["niter"][SEL[k]] for k in V}
DQ = {k: np.abs((V[k]["qop_ref"] - V[k]["qop_seed"])[SEL[k]]
                / np.abs(V[k]["qop_seed"][SEL[k]])) for k in V}

print("\n=== (2) BIT CHECK: base vs the 260903x_m0 production, same tracks ===")
dz = V["base"]["z"][SEL["base"]] - V["prod40"]["z"][SEL["prod40"]]
print(f"  max |dz| {np.abs(dz).max():.3e}   nonzero {int((dz != 0).sum())} / "
      f"{len(dz)}   median |dz| {np.median(np.abs(dz)):.3e}")
print("  -> " + ("BIT-IDENTICAL" if np.abs(dz).max() == 0 else
                 "NOT bit-identical; interpret the variants against `base`, "
                 "not against the 260903x cache"))

print("\n=== niter census per variant ===")
for k in ("base", "tight", "damp"):
    u, c = np.unique(NIT[k], return_counts=True)
    print(f"  {k:6s} <niter> {NIT[k].mean():5.2f}  " +
          "  ".join(f"{int(a)}:{100*b/len(NIT[k]):.1f}%" for a, b in
                    zip(u[:8], c[:8])))


def ev(x, m, nboot=300):
    idx = np.where(m)[0]
    v = 0.5 * (x[m & (Q > 0)].mean() + x[m & (Q < 0)].mean())
    o = np.empty(nboot)
    for i in range(nboot):
        kk = idx[rng.integers(0, len(idx), len(idx))]
        o[i] = 0.5 * (x[kk][Q[kk] > 0].mean() + x[kk][Q[kk] < 0].mean())
    return v * 1e3, o.std() * 1e3


ok = {k: np.isfinite(X[k]) & (np.abs(X[k]) < 30) for k in X}
allok = ok["base"] & ok["tight"] & ok["damp"] & ok["prod40"]

print("\n=== (3) CHARGE-EVEN <x> per eta band, per variant (1e-3) ===")
print(f"{'variant':8s} " + "".join(f"{n:>18s}" for n in BANDS) + f"{'all':>18s}")
for k in ("base", "tight", "damp"):
    row = ""
    for ib in list(range(3)) + [None]:
        m = allok if ib is None else (allok & (band == ib))
        v, e = ev(X[k], m)
        row += f"{v:+11.2f}+-{e:4.2f} "
    print(f"{k:8s} {row}")
print("\n  PAIRED differences vs base (the common fluctuation cancels):")
for k in ("tight", "damp"):
    row = ""
    for ib in list(range(3)) + [None]:
        m = allok if ib is None else (allok & (band == ib))
        v, e = ev(X[k] - X["base"], m)
        row += f"{v:+11.3f}+-{e:5.3f} "
    print(f"  {k:6s}-base {row}")

print("\n=== (4) the BULK / IN decomposition, per variant ===")
print(f"{'variant':8s} {'cut':5s} " + "".join(f"{n:>18s}" for n in BANDS)
      + f"{'combined':>18s}")
for k in ("base", "tight", "damp"):
    t = np.percentile(DQ[k][allok], 90)
    for lab, cut in (("IN ", DQ[k] > t), ("OUT", DQ[k] <= t)):
        vs, es, row = [], [], ""
        for ib in range(3):
            m = allok & (band == ib) & cut
            v, e = ev(X[k], m)
            vs.append(v)
            es.append(e)
            row += f"{v:+11.2f}+-{e:4.2f} "
        w = 1 / np.array(es) ** 2
        mu = (np.array(vs) * w).sum() / w.sum()
        print(f"{k:8s} {lab:5s} {row}{mu:+11.2f}+-{1/np.sqrt(w.sum()):4.2f}")

print("\n=== (5) TRIM SCAN of the charge-even mean, per variant (1e-3) ===")
TS = [1., 2., 3., 5., 30.]
print(f"{'variant':8s} {'band':8s} " + "".join(f"{f'T={t:g}':>13s}" for t in TS))
for k in ("base", "tight", "damp"):
    for ib in range(3):
        m0 = allok & (band == ib)
        row = ""
        for T in TS:
            v, e = ev(X[k], m0 & (np.abs(X[k]) < T), nboot=150)
            row += f"{v:+8.2f}+-{e:4.2f}"
        print(f"{k:8s} {BANDS[ib]:8s} {row}")
