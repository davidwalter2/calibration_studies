#!/usr/bin/env python3
"""(2) WHICH TRACKS carry the charge-even shift? Split per eta by fit quality,
convergence and step control.

BINNING RULE (secs. 0f.33/0f.42): every conditioning variable here is
RECONSTRUCTED, so `corr(v, x)` is printed for each -- a variable correlated
with the SIGNED residual is a Jacobian-edge-style trap; one correlated only
with |x| is a legitimate resolution handle for a MEAN split.
"""
import numpy as np

BANDS = ["barrel", "middle", "endcap"]
b = np.load("data/blocks_mugun_ul16_260903x.npz")
Q = np.load("data/quality_mugun_ul16_260903x.npz")
assert len(Q["niter"]) == len(b["t_z"])
z = b["t_z"].astype(np.float64)
eta = np.abs(b["t_eta"].astype(np.float64))
q = b["t_q"].astype(np.float64)
sig = b["t_sigma"].astype(np.float64)
vgf = b["t_vgf"].astype(np.float64)
pf = b["t_pt"].astype(np.float64) * np.cosh(b["t_eta"].astype(np.float64))
ai = sig * pf * (1 - vgf)
den = 1 - ai * q * z
x = np.where(np.abs(den) > 1e-3, z / den, np.nan)
good = np.isfinite(x) & (np.abs(x) < 30)
band = np.digitize(eta, [0.9, 1.6])
rng = np.random.default_rng(23)

# the seed-to-final and iter0-to-final momentum steps, RELATIVE and signed in
# curvature (the clamp is a cap in |p|, which is asymmetric in q/p)
dqop_seed = (Q["qop_ref"] - Q["qop_seed"]) / np.abs(Q["qop_seed"])
dqop_it0 = (Q["qop_ref"] - Q["qop_iter0"]) / np.abs(Q["qop_ref"])

VARS = {
    "niter": Q["niter"],
    "chi2/ndof": Q["normalizedChi2"],
    "edmval": np.log10(np.maximum(Q["edmval"], 1e-30)),
    "gradmax": np.log10(np.maximum(np.abs(Q["gradmax"]), 1e-30)),
    "nValidHits": Q["nValidHits"],
    "nValidPixelHits": Q["nValidPixelHits"],
    "n pixel layers": Q["npixlay"],
    "n layers": Q["nlay"],
    "first layer (radial rank)": Q["firstsd"] * 10 + Q["firstlay"],
    "seed->final dq/p (signed)": dqop_seed,
    "|seed->final dq/p|": np.abs(dqop_seed),
    "iter0->final dq/p (signed)": dqop_it0,
    "|iter0->final dq/p|": np.abs(dqop_it0),
    "nChargeFlipProtect": Q["nChargeFlipProtect"].astype(float),
    "chargeHypFlipped": Q["chargeHypFlipped"].astype(float),
    "vgf (hit share)": vgf,
    "sigma_rel": sig * pf,
}


def ev(m, nboot=200):
    idx = np.where(m)[0]
    v = 0.5 * (x[m & (q > 0)].mean() + x[m & (q < 0)].mean())
    bs = np.empty(nboot)
    for i in range(nboot):
        k = idx[rng.integers(0, len(idx), len(idx))]
        bs[i] = 0.5 * (x[k][q[k] > 0].mean() + x[k][q[k] < 0].mean())
    return v, bs.std()


print("=== conditioning-variable correlations (binning rule) ===")
print(f"{'variable':30s} {'corr(v,x)':>11s} {'corr(v,|x|)':>13s}")
g = good
for nm, v in VARS.items():
    vv = v[g]
    if vv.std() == 0:
        print(f"{nm:30s} {'constant':>11s}")
        continue
    print(f"{nm:30s} {np.corrcoef(vv, x[g])[0,1]:+11.4f} "
          f"{np.corrcoef(vv, np.abs(x[g]))[0,1]:+13.4f}")

print("\n=== charge-even <x> per eta band, split into TERTILES (units 1e-3) ===")
print(f"{'variable':30s} {'band':8s} {'lo':>16s} {'mid':>16s} {'hi':>16s} "
      f"{'hi-lo':>14s}")
for nm, v in VARS.items():
    for ib in range(3):
        m0 = good & (band == ib)
        e = np.unique(np.percentile(v[m0], [0, 33.333, 66.667, 100]))
        if len(e) < 3:
            # discrete with few values: split at zero / nonzero
            u = np.unique(v[m0])
            if len(u) < 2:
                continue
            groups = [(m0 & (v == u[0]), f"={u[0]:g}"),
                      (m0 & (v != u[0]), f"!={u[0]:g}")]
        else:
            groups = []
            for i in range(len(e) - 1):
                mm = m0 & (v >= e[i]) & ((v <= e[i + 1]) if i == len(e) - 2
                                         else (v < e[i + 1]))
                groups.append((mm, f"[{e[i]:.3g},{e[i+1]:.3g})"))
        out, vals, errs = "", [], []
        for mm, lab in groups:
            if mm.sum() < 300:
                out += f"{'--':>16s}"
                vals.append(np.nan)
                errs.append(np.nan)
                continue
            val, er = ev(mm)
            vals.append(val)
            errs.append(er)
            out += f"{val*1e3:+9.2f}+-{er*1e3:4.2f}"
        d = vals[-1] - vals[0]
        de = np.hypot(errs[-1], errs[0])
        print(f"{nm if ib==0 else '':30s} {BANDS[ib]:8s} {out} "
              f"{d*1e3:+9.2f}+-{de*1e3:4.2f}")
    print()
