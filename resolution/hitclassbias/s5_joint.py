#!/usr/bin/env python3
"""(3) Joint partial slopes and the flattening test.

One GLOBAL charge-even linear model in the candidate variables, fitted on the
pooled sample; the per-band residual mean then says whether the eta pattern
-4.4 / -4.7 / -1.7 is anything the track-quality variables can account for.
The slopes are global, so this is a genuine test rather than three free
offsets.
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
rng = np.random.default_rng(53)
dq = np.abs((Q["qop_ref"] - Q["qop_seed"]) / np.abs(Q["qop_seed"]))

V = {
    "nValidPixelHits": Q["nValidPixelHits"].astype(float),
    "niter": Q["niter"],
    "|seed->final dq/p|": dq,
    "chi2/ndof": Q["normalizedChi2"],
    "nValidHits": Q["nValidHits"].astype(float),
    "vgf": vgf,
}
g = np.where(good)[0]
X = np.column_stack([(V[k][g] - V[k][g].mean()) / V[k][g].std() for k in V])
Y = x[g]
QQ = q[g]

print("=== JOINT partial slopes, charge-even, per STANDARDISED unit (1e-3) ===")
sols = []
for sgn in (1, -1):
    k = QQ * sgn > 0
    A = np.column_stack([np.ones(k.sum()), X[k]])
    sol, *_ = np.linalg.lstsq(A, Y[k], rcond=None)
    sols.append(sol)
ev = 0.5 * (sols[0] + sols[1])
# bootstrap
bs = []
for i in range(200):
    idx = rng.integers(0, len(g), len(g))
    s2 = []
    for sgn in (1, -1):
        k = QQ[idx] * sgn > 0
        A = np.column_stack([np.ones(k.sum()), X[idx][k]])
        s, *_ = np.linalg.lstsq(A, Y[idx][k], rcond=None)
        s2.append(s)
    bs.append(0.5 * (s2[0] + s2[1]))
bs = np.array(bs)
names = ["intercept"] + list(V)
for j, nm in enumerate(names):
    print(f"  {nm:22s} {ev[j]*1e3:+8.2f} +- {bs[:,j].std()*1e3:5.2f} "
          f"({ev[j]/bs[:,j].std():+5.1f} sigma)")

pred = np.zeros(len(x))
pred[g] = X @ ev[1:]
print("\n=== FLATTENING TEST: charge-even <x> before and after removing the "
      "global track-quality model (units 1e-3) ===")
print(f"{'band':10s} {'raw':>16s} {'model':>9s} {'residual':>16s}")


def evm(m, arr, nboot=200):
    idx = np.where(m)[0]
    v = 0.5 * (arr[m & (q > 0)].mean() + arr[m & (q < 0)].mean())
    o = np.empty(nboot)
    for i in range(nboot):
        k = idx[rng.integers(0, len(idx), len(idx))]
        o[i] = 0.5 * (arr[k][q[k] > 0].mean() + arr[k][q[k] < 0].mean())
    return v, o.std()


for ib, nm in enumerate(BANDS):
    m = good & (band == ib)
    r0, e0 = evm(m, x)
    r1, e1 = evm(m, x - pred)
    print(f"{nm:10s} {r0*1e3:+10.2f}+-{e0*1e3:4.2f} {pred[m].mean()*1e3:+9.2f} "
          f"{r1*1e3:+10.2f}+-{e1*1e3:4.2f}")
print("  -> the global model's per-band prediction is what a track-quality "
      "explanation of the eta pattern would have to supply.")

print("\n=== how much of the sample the candidates isolate ===")
for nm, cut in (("niter >= 3", Q["niter"] >= 3),
                ("nValidPixelHits >= 3", Q["nValidPixelHits"] >= 3),
                ("|seed step| top 10%", dq > np.percentile(dq[good], 90)),
                ("chi2/ndof > 1.5", Q["normalizedChi2"] > 1.5)):
    row = ""
    for ib in range(3):
        m = good & (band == ib)
        a, ea = evm(m & cut, x)
        bq, eb = evm(m & ~cut, x)
        row += (f"  {BANDS[ib]}: in {a*1e3:+7.2f}+-{ea*1e3:4.2f} "
                f"({100*(m&cut).sum()/m.sum():4.1f}%) out {bq*1e3:+7.2f}+-{eb*1e3:4.2f}")
    print(f"{nm:22s}{row}")
