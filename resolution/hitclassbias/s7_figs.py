#!/usr/bin/env python3
"""Figures for the WHICH-TRACKS decomposition. One file per panel."""
import datetime
import os
import sys

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pubhtml                                                    # noqa: E402
from wums import logging                                          # noqa: E402

hep.style.use(hep.style.ROOT)
logger = logging.child_logger(__name__)
BANDS = [r"$|\eta|<0.9$", r"$0.9-1.6$", r"$1.6-2.4$"]


def save(fig, outdir, name):
    pubhtml.savefig(fig, os.path.join(outdir, f"{name}.pdf"))
    plt.close(fig)
    logger.info(f"  {name}")


day = datetime.date.today().strftime("%y%m%d")
outdir = os.path.expanduser(f"~/public_html/cvh/{day}_hitclassbias")
os.makedirs(outdir, exist_ok=True)
pubhtml.ensure_index(outdir, logger=logger)

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
dq = np.abs((Q["qop_ref"] - Q["qop_seed"]) / np.abs(Q["qop_seed"]))
rng = np.random.default_rng(71)


def ev(m, nboot=200):
    idx = np.where(m)[0]
    v = 0.5 * (x[m & (q > 0)].mean() + x[m & (q < 0)].mean())
    o = np.empty(nboot)
    for i in range(nboot):
        k = idx[rng.integers(0, len(idx), len(idx))]
        o[i] = 0.5 * (x[k][q[k] > 0].mean() + x[k][q[k] < 0].mean())
    return v * 1e3, o.std() * 1e3


# --- 1. trim scan ----------------------------------------------------------
TS = [1., 2., 3., 5., 10., 30.]
fig, ax = plt.subplots(figsize=(9, 6))
for ib, (nm, col) in enumerate(zip(BANDS, ("tab:red", "tab:blue", "k"))):
    m0 = good & (band == ib)
    vs, es = [], []
    v = np.concatenate([x[m0 & (q > 0)], -x[m0 & (q < 0)]])
    v = np.concatenate([v, -v])
    for T in TS:
        s = m0 & (np.abs(x) < T)
        val, er = ev(s)
        Qf = (np.abs(v) < T).mean()
        h = 0.1 * max(T / 5., 1.)
        d = ((np.abs(np.abs(v) - T) < h / 2).mean() / h) / 2.
        f = 1. - 2 * T * d / max(Qf, 1e-9)
        vs.append(val / f)
        es.append(er / f)
    ax.errorbar(TS, vs, yerr=es, fmt="o-", ms=6, color=col, label=nm,
                capsize=3)
ax.axhline(0, color="0.6", lw=1)
ax.set_xscale("log")
ax.set_xticks(TS)
ax.set_xticklabels([f"{t:g}" for t in TS])
ax.set_xlabel(r"trim $T$  ($|x| < T$)")
ax.set_ylabel(r"implied core shift $m(T) \times 10^{3}$")
ax.legend(fontsize=12)
ax.set_title("flat = a core shift; rising = a one-sided tail", fontsize=15)
save(fig, outdir, "skew_trimscan")

# --- 2. the mixture --------------------------------------------------------
t = np.percentile(dq[good], 90)
fig, ax = plt.subplots(figsize=(9, 6))
xs = np.arange(3)
for lab, cut, col, mk in ((r"large $|\delta(q/p)|$ seed$\to$final (top 10%)",
                           dq > t, "tab:red", "s"),
                          ("the other 90%", dq <= t, "tab:blue", "o"),
                          ("all tracks", np.ones(len(x), bool), "k", "D")):
    vs, es = [], []
    for ib in range(3):
        v, e = ev(good & (band == ib) & cut)
        vs.append(v)
        es.append(e)
    ax.errorbar(xs + 0.06 * (0 if col == "k" else (1 if col == "tab:red" else -1)),
                vs, yerr=es, fmt=mk + "-", ms=9, color=col, label=lab, capsize=3)
for ib in range(3):
    m = good & (band == ib)
    f = 100 * (m & (dq > t)).sum() / m.sum()
    ax.annotate(f"{f:.1f}%", (ib + 0.06, 34), color="tab:red", ha="center",
                fontsize=12)
ax.axhline(0, color="0.6", lw=1)
ax.set_xticks(xs)
ax.set_xticklabels(BANDS)
ax.set_ylim(-20, 42)
ax.set_ylabel(r"charge-even $\langle x\rangle \times 10^{3}$")
ax.legend(fontsize=11, loc="lower left")
ax.set_title(r"both components are flat in $\eta$; only their MIXTURE is not",
             fontsize=15)
save(fig, outdir, "skew_mixture")

# --- 3. partial slopes -----------------------------------------------------
V = {"nValidPixelHits": Q["nValidPixelHits"].astype(float),
     "niter": Q["niter"],
     r"$|\delta(q/p)|$ seed$\to$final": dq,
     r"$\chi^2$/ndof": Q["normalizedChi2"],
     "nValidHits": Q["nValidHits"].astype(float),
     "vgf (hit share)": vgf}
g = np.where(good)[0]
X = np.column_stack([(V[k][g] - V[k][g].mean()) / V[k][g].std() for k in V])
Y, QQ = x[g], q[g]
sol = []
for sgn in (1, -1):
    k = QQ * sgn > 0
    A = np.column_stack([np.ones(k.sum()), X[k]])
    s, *_ = np.linalg.lstsq(A, Y[k], rcond=None)
    sol.append(s)
evv = 0.5 * (sol[0] + sol[1])
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
fig, ax = plt.subplots(figsize=(9, 6))
names = list(V)
ax.errorbar(np.arange(len(names)), evv[1:] * 1e3, yerr=bs[:, 1:].std(0) * 1e3,
            fmt="o", ms=9, color="k", capsize=4)
ax.axhline(0, color="0.6", lw=1)
ax.set_xticks(np.arange(len(names)))
ax.set_xticklabels(names, rotation=25, ha="right")
ax.set_ylabel(r"partial $\partial\langle x\rangle_{\rm even}/\partial v \times 10^{3}$"
              "\n(per standardised unit)")
ax.set_title("joint fit: nothing reaches 3 sigma", fontsize=15)
save(fig, outdir, "skew_partial_slopes")
logger.info(f"done -> {outdir}")
