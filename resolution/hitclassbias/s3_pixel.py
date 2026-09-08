#!/usr/bin/env python3
"""(2b) The discrete splits done properly, and the step-control census."""
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
rng = np.random.default_rng(31)


def ev(m, nboot=300):
    idx = np.where(m)[0]
    v = 0.5 * (x[m & (q > 0)].mean() + x[m & (q < 0)].mean())
    bs = np.empty(nboot)
    for i in range(nboot):
        k = idx[rng.integers(0, len(idx), len(idx))]
        bs[i] = 0.5 * (x[k][q[k] > 0].mean() + x[k][q[k] < 0].mean())
    return v, bs.std()


print("=== STEP-CONTROL CENSUS on this sample ===")
for nm in ("nChargeFlipProtect", "chargeHypFlipped", "niter"):
    v = Q[nm][good]
    u, c = np.unique(v, return_counts=True)
    print(f"{nm:22s} " + "  ".join(f"{int(a)}:{b_} ({100*b_/len(v):.3f}%)"
                                   for a, b_ in zip(u[:8], c[:8])))
print("  -> the asymmetric momentum clamp NEVER fires on this gun, and the "
      "two-hypothesis refit never flips: step control cannot be the mechanism "
      "HERE (it can still be on low-pT tracks, where the 2 GeV floor is near).")

print("\n=== PIXEL CONTENT, discrete (units 1e-3) ===")
for nm, v in (("n pixel layers", Q["npixlay"]),
              ("nValidPixelHits", Q["nValidPixelHits"]),
              ("innermost layer is BPix-1", ((Q["firstsd"] == 1)
                                             & (Q["firstlay"] == 1)).astype(float)),
              ("n layers", Q["nlay"])):
    print(f"-- {nm}")
    for ib in range(3):
        m0 = good & (band == ib)
        u = np.unique(v[m0])
        row = ""
        vals, errs, ns = [], [], []
        for uu in u:
            m = m0 & (v == uu)
            if m.sum() < 400:
                continue
            val, er = ev(m)
            vals.append(val)
            errs.append(er)
            ns.append(m.sum())
            row += f" {int(uu)}: {val*1e3:+7.2f}+-{er*1e3:4.2f} ({100*m.sum()/m0.sum():4.1f}%)"
        # linear trend
        if len(vals) >= 2:
            uu = np.array([int(w) for w in u if (m0 & (v == w)).sum() >= 400],
                          float)
            w = 1. / np.array(errs) ** 2
            A = np.vstack([np.ones_like(uu), uu - (uu * w).sum() / w.sum()]).T
            cov = np.linalg.inv(A.T @ (A * w[:, None]))
            sol = cov @ (A.T @ (w * np.array(vals)))
            row += f"   slope {sol[1]*1e3:+6.2f}+-{np.sqrt(cov[1,1])*1e3:4.2f}"
        print(f"   {BANDS[ib]:8s}{row}")

print("\n=== DOES REMOVING IT FLATTEN THE CLOSURE? (diagnostic only) ===")
cuts = {
    "all tracks": np.ones(len(x), bool),
    "n pixel layers <= 2": Q["npixlay"] <= 2,
    "n pixel layers >= 3": Q["npixlay"] >= 3,
    "innermost = BPix-1": (Q["firstsd"] == 1) & (Q["firstlay"] == 1),
    "innermost != BPix-1": ~((Q["firstsd"] == 1) & (Q["firstlay"] == 1)),
    "chi2/ndof < 1.2": Q["normalizedChi2"] < 1.2,
    "chi2/ndof > 1.2": Q["normalizedChi2"] >= 1.2,
}
print(f"{'cut':24s} {'kept':>7s} " + "".join(f"{n:>16s}" for n in BANDS)
      + f"{'spread bar-end':>16s}")
for nm, c in cuts.items():
    row, vs, es = "", [], []
    for ib in range(3):
        m = good & (band == ib) & c
        if m.sum() < 500:
            row += f"{'--':>16s}"
            vs.append(np.nan)
            es.append(np.nan)
            continue
        v, e = ev(m)
        vs.append(v)
        es.append(e)
        row += f"{v*1e3:+10.2f}+-{e*1e3:4.2f}"
    d = vs[0] - vs[2]
    print(f"{nm:24s} {100*(good&c).sum()/good.sum():6.1f}% {row}"
          f"{d*1e3:+10.2f}+-{np.hypot(es[0],es[2])*1e3:4.2f}")
