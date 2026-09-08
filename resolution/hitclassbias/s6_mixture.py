#!/usr/bin/env python3
"""(3) THE MIXTURE PICTURE and its robustness.

The apparent eta dependence -4.4 / -4.7 / -1.7 is reproduced EXACTLY by two
eta-INDEPENDENT components whose mixing fraction runs with |eta|: the tracks
with the largest |seed -> final q/p| correction (2.4 % of the barrel, 21.2 %
of the endcap) sit at about +21e-3, and everything else at about -6.4e-3.
This scans the threshold and repeats the split on the CVH-internal
|iter0 -> final| step, which has an even smaller exposure to the residual.
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
rng = np.random.default_rng(67)
dq = np.abs((Q["qop_ref"] - Q["qop_seed"]) / np.abs(Q["qop_seed"]))
di = np.abs((Q["qop_ref"] - Q["qop_iter0"]) / np.abs(Q["qop_ref"]))


def evm(m, arr=None, nboot=300):
    arr = x if arr is None else arr
    idx = np.where(m)[0]
    v = 0.5 * (arr[m & (q > 0)].mean() + arr[m & (q < 0)].mean())
    o = np.empty(nboot)
    for i in range(nboot):
        k = idx[rng.integers(0, len(idx), len(idx))]
        o[i] = 0.5 * (arr[k][q[k] > 0].mean() + arr[k][q[k] < 0].mean())
    return v, o.std()


def comb(vs, es):
    vs, es = np.array(vs), np.array(es)
    w = 1 / es ** 2
    mu = (vs * w).sum() / w.sum()
    return mu, 1 / np.sqrt(w.sum()), float((w * (vs - mu) ** 2).sum())


for tag, v, arr in (("|seed->final dq/p|", dq, None),
                    ("|iter0->final dq/p|", di, None),
                    ("|seed->final| on RAW z", dq, z)):
    print(f"\n=== {tag} ===")
    print(f"{'thr':>6s} {'':6s}" + "".join(f"{n:>19s}" for n in BANDS)
          + f"{'combined':>26s}")
    for pctl in (80., 90., 95.):
        t = np.percentile(v[good], pctl)
        for lab, cut in (("IN ", v > t), ("OUT", v <= t)):
            vs, es, row = [], [], ""
            for ib in range(3):
                m = good & (band == ib) & cut
                val, er = evm(m, arr)
                vs.append(val)
                es.append(er)
                row += (f"{val*1e3:+8.2f}+-{er*1e3:4.2f}"
                        f"({100*m.sum()/(good&(band==ib)).sum():4.1f})")
            mu, sm, c2 = comb(vs, es)
            print(f"{pctl:5.0f}% {lab:6s}{row}  {mu*1e3:+8.2f}+-{sm*1e3:5.2f} "
                  f"(chi2 {c2:4.1f}/2)")

print("\n=== the mixture identity (fractions x components = the band value) ===")
t = np.percentile(dq[good], 90)
for ib, nm in enumerate(BANDS):
    m = good & (band == ib)
    f = (m & (dq > t)).sum() / m.sum()
    a, _ = evm(m & (dq > t))
    o, _ = evm(m & (dq <= t))
    tot, e = evm(m)
    print(f"{nm:10s} f={100*f:5.2f}%  in {a*1e3:+7.2f}  out {o*1e3:+7.2f}  "
          f"-> {(f*a+(1-f)*o)*1e3:+7.2f}   measured {tot*1e3:+7.2f}+-{e*1e3:4.2f}")

print("\n=== what the large-step population LOOKS like ===")
print(f"{'':22s} " + "".join(f"{n:>12s}" for n in
                             ("frac", "<niter>", "chi2/ndof", "<sigma_rel>",
                              "<nvalid>", "<npixhit>", "rms(x)")))
for lab, cut in (("IN  (top 10%)", dq > t), ("OUT (rest)", dq <= t)):
    m = good & cut
    print(f"{lab:22s} " + "".join(f"{v:12.4f}" for v in (
        m.sum() / good.sum(), Q["niter"][m].mean(), Q["normalizedChi2"][m].mean(),
        (sig * pf)[m].mean(), Q["nValidHits"][m].mean(),
        Q["nValidPixelHits"][m].mean(), x[m].std())))
