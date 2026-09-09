#!/usr/bin/env python3
"""How big are the per-track changes, and does the mean shift come from a
tail of badly-converged tracks or from the bulk?"""
import numpy as np

V = {k: np.load(f"data/conv_{k}.npz") for k in ("base", "tight", "damp")}


def key(d):
    return (d["run"].astype(np.int64) * 10**6 + d["lumi"].astype(np.int64)) \
        * 10**9 + d["event"].astype(np.int64)


K = {k: key(d) for k, d in V.items()}
common = np.intersect1d(np.intersect1d(K["base"], K["tight"]), K["damp"])
SEL = {k: np.argsort(K[k])[np.searchsorted(np.sort(K[k]), common)] for k in V}
Z = {k: V[k]["z"][SEL[k]].astype(np.float64) for k in V}
q = V["base"]["q"][SEL["base"]].astype(np.float64)
eta = np.abs(V["base"]["eta"][SEL["base"]].astype(np.float64))
band = np.digitize(eta, [0.9, 1.6])

print(f"{'':16s} " + "".join(f"{f'>{t}':>9s}" for t in
                             (1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 3.0)))
for k in ("tight", "damp"):
    d = Z[k] - Z["base"]
    print(f"{k+'-base frac':16s} " + "".join(
        f"{100*(np.abs(d)>t).mean():8.3f}%" for t in
        (1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 3.0)))
print()
print(f"{'':16s} {'pct of |dz|:':>12s} " +
      "".join(f"{f'{p}%':>10s}" for p in (50, 90, 99, 99.9, 100)))
for k in ("tight", "damp"):
    d = np.abs(Z[k] - Z["base"])
    print(f"{k+'-base':16s} {'':12s} " +
          "".join(f"{np.percentile(d,p):10.4f}" for p in (50, 90, 99, 99.9))
          + f"{d.max():10.2f}")

print("\n=== is the mean shift the TAIL or the BULK? "
      "charge-even <dz> with |dz| trimmed (1e-3) ===")
print(f"{'variant':10s} " + "".join(f"{f'|dz|<{t}':>12s}" for t in
                                    (0.01, 0.1, 0.5, 1., 3., 1e9)))
for k in ("tight", "damp"):
    d = Z[k] - Z["base"]
    row = ""
    for T in (0.01, 0.1, 0.5, 1., 3., 1e9):
        m = np.abs(d) < T
        row += f"{0.5*(d[m&(q>0)].mean()+d[m&(q<0)].mean())*1e3:+9.2f}"
        row += f"({100*m.mean():.0f})"
    print(f"{k:10s} {row}")

print("\n=== per band, charge-even <dz>, full and trimmed at |dz|<0.5 ===")
for k in ("tight", "damp"):
    d = Z[k] - Z["base"]
    for ib, nm in enumerate(("barrel", "middle", "endcap")):
        m0 = band == ib
        f = 0.5 * (d[m0 & (q > 0)].mean() + d[m0 & (q < 0)].mean())
        m = m0 & (np.abs(d) < 0.5)
        t = 0.5 * (d[m & (q > 0)].mean() + d[m & (q < 0)].mean())
        print(f"  {k:6s} {nm:8s} full {f*1e3:+8.2f}   trimmed {t*1e3:+8.2f} "
              f"({100*m.sum()/m0.sum():5.1f}% kept)")
