#!/usr/bin/env python3
"""(2c) The controls and the combined significance.

The pixel-content variables order the charge-even shift, but within a band
they also correlate with |eta| (pixel coverage runs out at the band edges), so
the split is repeated at FIXED |eta| in 12 fine bins. The slope is then
combined across the three bands with its consistency chi2.
"""
import numpy as np

BANDS = ["barrel", "middle", "endcap"]
b = np.load("data/blocks_mugun_ul16_260903x.npz")
Q = np.load("data/quality_mugun_ul16_260903x.npz")
z = b["t_z"].astype(np.float64)
eta = np.abs(b["t_eta"].astype(np.float64))
q = b["t_q"].astype(np.float64)
sig = b["t_sigma"].astype(np.float64)
vgf = b["t_vgf"].astype(np.float64)
pf = b["t_pt"].astype(np.float64) * np.cosh(b["t_eta"].astype(np.float64))
den = 1 - sig * pf * (1 - vgf) * q * z
x = np.where(np.abs(den) > 1e-3, z / den, np.nan)
good = np.isfinite(x) & (np.abs(x) < 30)
band = np.digitize(eta, [0.9, 1.6])
rng = np.random.default_rng(41)
dqop_seed = (Q["qop_ref"] - Q["qop_seed"]) / np.abs(Q["qop_seed"])

CAND = {
    "nValidPixelHits": Q["nValidPixelHits"].astype(float),
    "n pixel layers": Q["npixlay"],
    "n layers": Q["nlay"],
    "nValidHits": Q["nValidHits"].astype(float),
    "chi2/ndof": Q["normalizedChi2"],
    "niter": Q["niter"],
    "vgf": vgf,
    "sigma_rel": sig * pf,
    "|seed->final dq/p|": np.abs(dqop_seed),
    "innermost=BPix1": ((Q["firstsd"] == 1) & (Q["firstlay"] == 1)).astype(float),
}


def slope(m, v, nboot=200):
    """weighted LS slope of the charge-even <x> on v, standardised so the
    coefficient is 'per unit of v'; bootstrap error."""
    idx = np.where(m)[0]
    vv = v[idx]
    if vv.std() == 0:
        return np.nan, np.nan
    xx = x[idx]
    qq = q[idx]

    def s1(sub):
        vs, xs, qs = vv[sub], xx[sub], qq[sub]
        # charge-even slope = mean of the per-charge slopes
        out = 0.
        for sgn in (1, -1):
            k = qs * sgn > 0
            if k.sum() < 20:
                return np.nan
            c = vs[k] - vs[k].mean()
            out += 0.5 * (c @ xs[k]) / max((c * c).sum(), 1e-12)
        return out
    val = s1(np.arange(len(idx)))
    bs = np.array([s1(rng.integers(0, len(idx), len(idx))) for _ in range(nboot)])
    return val, np.nanstd(bs)


def slope_fixed_eta(m, v, nbin=12, nboot=200):
    """the same slope with |eta| removed: v is centred WITHIN fine eta bins."""
    idx = np.where(m)[0]
    e = eta[idx]
    edges = np.percentile(e, np.linspace(0, 100, nbin + 1))
    ib = np.clip(np.digitize(e, edges[1:-1]), 0, nbin - 1)
    vv = v[idx].copy()
    for j in range(nbin):
        k = ib == j
        if k.sum():
            vv[k] -= vv[k].mean()
    xx, qq = x[idx], q[idx]

    def s1(sub):
        vs, xs, qs = vv[sub], xx[sub], qq[sub]
        out = 0.
        for sgn in (1, -1):
            k = qs * sgn > 0
            if k.sum() < 20:
                return np.nan
            c = vs[k] - vs[k].mean()
            out += 0.5 * (c @ xs[k]) / max((c * c).sum(), 1e-12)
        return out
    val = s1(np.arange(len(idx)))
    bs = np.array([s1(rng.integers(0, len(idx), len(idx))) for _ in range(nboot)])
    return val, np.nanstd(bs)


print("=== charge-even SLOPE d<x>/dv per band (1e-3 per unit of v) ===")
print("raw / at fixed |eta| (v centred in 12 fine eta bins)")
print(f"{'variable':22s} " + "".join(f"{n:>30s}" for n in BANDS)
      + f"{'combined (fixed eta)':>26s}")
for nm, v in CAND.items():
    row, vs, es = "", [], []
    for ib in range(3):
        m = good & (band == ib)
        a1, e1 = slope(m, v)
        a2, e2 = slope_fixed_eta(m, v)
        vs.append(a2)
        es.append(e2)
        row += f"{a1*1e3:+8.2f}/{a2*1e3:+8.2f}+-{e2*1e3:5.2f} "
    vs, es = np.array(vs), np.array(es)
    ok = np.isfinite(vs) & np.isfinite(es) & (es > 0)
    if ok.sum() >= 2:
        w = 1 / es[ok] ** 2
        mu = (vs[ok] * w).sum() / w.sum()
        sm = 1 / np.sqrt(w.sum())
        chi2 = float((w * (vs[ok] - mu) ** 2).sum())
        comb = (f"{mu*1e3:+9.2f}+-{sm*1e3:5.2f} "
                f"({mu/sm:+4.1f}s, chi2 {chi2:.1f}/{ok.sum()-1})")
    else:
        comb = "--"
    print(f"{nm:22s} {row}{comb:>26s}")

print("\n=== corr of each with |eta| WITHIN a band (why the control matters) ===")
for nm, v in CAND.items():
    r = []
    for ib in range(3):
        m = good & (band == ib)
        r.append(np.corrcoef(v[m], eta[m])[0, 1] if v[m].std() else np.nan)
    print(f"{nm:22s} " + "".join(f"{y:+9.3f}" for y in r))
