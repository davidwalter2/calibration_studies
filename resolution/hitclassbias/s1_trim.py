#!/usr/bin/env python3
"""(1) THE TRIM SCAN. Is the charge-even shift a CORE shift or a one-sided
TAIL from a subpopulation?

For a density q(x - m) with q symmetric, the truncated mean responds as
    d<x>_{|x|<T} / dm  =  1 - 2 T q(T) / Q(T)   ==  f(T),
so `m_implied(T) = <x>_even,|x|<T / f(T)` is FLAT in T for a pure core shift
and RISES with T for a one-sided tail. f(T) is taken from the CHARGE-SYMMETRISED
observed density itself -- the model's own odd content is +5e-5 here, so the
symmetrised data IS q to that accuracy, and no model is needed.
"""
import numpy as np

TS = [1., 2., 3., 5., 10., np.inf]
BANDS = ["barrel |eta|<0.9", "middle 0.9-1.6", "endcap 1.6-2.4"]
b = np.load("data/blocks_mugun_ul16_260903x.npz")
z = b["t_z"].astype(np.float64)
eta = np.abs(b["t_eta"].astype(np.float64))
q = b["t_q"].astype(np.float64)
sig = b["t_sigma"].astype(np.float64)
vgf = b["t_vgf"].astype(np.float64)
pf = b["t_pt"].astype(np.float64) * np.cosh(b["t_eta"].astype(np.float64))
ai = sig * pf * (1 - vgf)
den = 1 - ai * q * z
x = np.where(np.abs(den) > 1e-3, z / den, np.nan)
good = np.isfinite(x)
band = np.digitize(eta, [0.9, 1.6])
rng = np.random.default_rng(11)


def even_trim(m, T, nboot=200):
    s = m & (np.abs(x) < T)
    v = 0.5 * (x[s & (q > 0)].mean() + x[s & (q < 0)].mean())
    idx = np.where(s)[0]
    bs = np.empty(nboot)
    for i in range(nboot):
        k = idx[rng.integers(0, len(idx), len(idx))]
        bs[i] = 0.5 * (x[k][q[k] > 0].mean() + x[k][q[k] < 0].mean())
    return v, bs.std(), int(s.sum())


def fT(m, T):
    """1 - 2 T q(T)/Q(T) from the charge-symmetrised sample."""
    if not np.isfinite(T):
        return 1.
    v = np.concatenate([x[m & (q > 0)], -x[m & (q < 0)]])   # symmetrised
    v = np.concatenate([v, -v])
    Q = (np.abs(v) < T).mean()
    h = 0.1 * max(T / 5., 1.)
    dens = ((np.abs(np.abs(v) - T) < h / 2).mean() / h) / 2.   # q(T), one side
    return 1. - 2 * T * dens / max(Q, 1e-9)


print("=== TRIM SCAN of the CHARGE-EVEN mean, units 1e-3 ===")
print("m_implied = <x>_even,|x|<T / f(T); FLAT = core shift, RISING = tail")
print(f"{'band':18s} " + "".join(f"{f'T={t:g}':>26s}" for t in TS))
for ib, nm in enumerate(BANDS):
    m0 = good & (band == ib)
    row = ""
    for T in TS:
        v, e, n = even_trim(m0, T)
        f = fT(m0, T)
        row += f" {v*1e3:+7.2f}+-{e*1e3:4.2f}/{f:4.2f}={v*1e3/f:+6.2f}"
    print(f"{nm:18s}{row}")
print()
print(f"{'band':18s} " + "".join(f"{f'T={t:g}':>12s}" for t in TS))
for ib, nm in enumerate(BANDS):
    m0 = good & (band == ib)
    print(f"{nm:18s} " + "".join(
        f"{100*(np.abs(x[m0])<T).mean():11.3f}%" for T in TS)
        + "   <- fraction kept")

print("\n=== the same, split by CHARGE, to show the tail is one-sided ===")
for ib, nm in enumerate(BANDS):
    m0 = good & (band == ib)
    for T in (1., 3., np.inf):
        s = m0 & (np.abs(x) < T)
        print(f"{nm:18s} T={T:<5g} q+ {x[s&(q>0)].mean()*1e3:+7.2f}  "
              f"q- {x[s&(q<0)].mean()*1e3:+7.2f}  "
              f"even {0.5*(x[s&(q>0)].mean()+x[s&(q<0)].mean())*1e3:+7.2f}  "
              f"odd {0.5*(x[s&(q>0)].mean()-x[s&(q<0)].mean())*1e3:+7.2f}")
    print()

print("=== where the even mean accumulates: contribution of each |x| shell ===")
SH = [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 5), (5, 10), (10, 1e9)]
print(f"{'band':18s} " + "".join(f"{f'{a}-{b}':>11s}" for a, b in SH))
for ib, nm in enumerate(BANDS):
    m0 = good & (band == ib)
    n0 = m0.sum()
    row = ""
    for lo, hi in SH:
        s = m0 & (np.abs(x) >= lo) & (np.abs(x) < hi)
        c = 0.5 * ((x[s & (q > 0)].sum() / (m0 & (q > 0)).sum())
                   + (x[s & (q < 0)].sum() / (m0 & (q < 0)).sum()))
        row += f"{c*1e3:+11.2f}"
    print(f"{nm:18s}{row}")
