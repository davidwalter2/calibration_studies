#!/usr/bin/env python3
"""The complete no-free-parameter prediction: LOCATION + SKEW, and the
degraded-hit-content splits the per-class sigma/m result points at.

The hit term contributes to the standardized momentum error
    z_hit = sum_b s_b a_b eps_b ,   a_b = sqrt(v_b)/sigma,
so with mu_b, theta_b, kappa3_b the mean, variance and third cumulant of the
class's measured pull density,
    mean   = sum_b s_b a_b mu_b
    kappa3 = sum_b s_b a_b^3 kappa3_b
and, for a probe e^{-u z^2} on an otherwise unit-variance z (Edgeworth),
    <z e^{-u z^2}> = m (1+2u)^{-3/2} + (kappa3/2) s^3 (s^2 - 1),  s^2=1/(1+2u)
so at u = 0.05 the skew channel enters with coefficient -0.0394 and the mean
channel with +0.8668.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BANDS = [(0.0, 0.9, "barrel |eta|<0.9"), (0.9, 1.6, "middle 0.9-1.6"),
         (1.6, 2.4, "endcap 1.6-2.4")]
U = 0.05
S2 = 1. / (1. + 2 * U)
CM = S2 ** 1.5                       # mean -> odd moment
CK = 0.5 * S2 ** 1.5 * (S2 - 1.)     # kappa3 -> odd moment

h = np.load("data/hits_mugun_ul16_h2.npz")
b = np.load("data/blocks_mugun_ul16_260903x.npz")
rng = np.random.default_rng(20260909)


def bank(key_h, key_b, nmin=150):
    """mu, kappa3 per key, from the hitres arm; mapped onto the blocks."""
    mu = np.zeros(len(key_b))
    k3 = np.zeros(len(key_b))
    ve = np.zeros(len(key_b))
    for isy, pk in ((0, "pullx"), (1, "pully")):
        p = h[pk].astype(np.float64)
        ok = np.isfinite(p) & (np.abs(p) < 10)
        kk = key_h[ok] * 2 + isy
        u, inv = np.unique(kk, return_inverse=True)
        n = np.bincount(inv)
        m1 = np.bincount(inv, weights=p[ok]) / np.maximum(n, 1)
        c = p[ok] - m1[inv]
        m2 = np.bincount(inv, weights=c ** 2) / np.maximum(n, 1)
        m3 = np.bincount(inv, weights=c ** 3) / np.maximum(n, 1)
        m4 = np.bincount(inv, weights=c ** 4) / np.maximum(n, 1)
        good = n >= nmin
        bm = dict(zip(u[good], m1[good]))
        bk = dict(zip(u[good], m3[good]))
        bv = dict(zip(u[good], (m2 / np.maximum(n, 1))[good]))
        sel = (b["b_isy"] == isy)
        kb = key_b[sel] * 2 + isy
        mu[sel] = [bm.get(int(x), 0.) for x in kb]
        k3[sel] = [bk.get(int(x), 0.) for x in kb]
        ve[sel] = [bv.get(int(x), 0.) for x in kb]
    return mu, k3, ve


sd = b["b_sd"].astype(np.int64)
key_h = h["sd"].astype(np.int64) * 100 + h["lay"].astype(np.int64)
key_b = sd * 100 + b["b_lay"].astype(np.int64)
mu, k3, ve = bank(key_h, key_b)

s = b["b_s"].astype(np.float64)
a = np.sqrt(b["b_a2"].astype(np.float64))
trk = np.repeat(np.arange(len(b["nblk"])), b["nblk"])
ntr = len(b["nblk"])
dz_mean = np.bincount(trk, weights=s * a * mu, minlength=ntr)
dz_k3 = np.bincount(trk, weights=s * a ** 3 * k3, minlength=ntr)
pred_odd = CM * dz_mean + CK * dz_k3

# influence-weighted class shares per track (as attribute_skew.py defines them)
tot = np.bincount(trk, weights=a ** 2, minlength=ntr)
pixm = sd <= 2
n1m = (sd >= 3) & (b["b_N"] == 1)
share_pix = np.bincount(trk[pixm], weights=(a ** 2)[pixm], minlength=ntr) / tot
share_n1 = np.bincount(trk[n1m], weights=(a ** 2)[n1m], minlength=ntr) / tot

z = b["t_z"].astype(np.float64)
eta = np.abs(b["t_eta"].astype(np.float64))
q = b["t_q"].astype(np.float64)
sig = b["t_sigma"].astype(np.float64)
vgf = b["t_vgf"].astype(np.float64)
pf = b["t_pt"].astype(np.float64) * np.cosh(b["t_eta"].astype(np.float64))
ai = sig * pf * (1 - vgf)
den = 1 - ai * q * z
zt = np.where(np.abs(den) > 1e-3, z / den, np.nan)
good = np.isfinite(zt) & (np.abs(zt) < 30)
band = np.digitize(eta, [0.9, 1.6])
sigrel = sig * pf


def ev(arr, m, nboot=200):
    p_ = arr[m & (q > 0)].mean()
    n_ = arr[m & (q < 0)].mean()
    idx = np.where(m)[0]
    bs = np.empty(nboot)
    for i in range(nboot):
        k = idx[rng.integers(0, len(idx), len(idx))]
        bs[i] = 0.5 * (arr[k][q[k] > 0].mean() + arr[k][q[k] < 0].mean())
    return 0.5 * (p_ + n_), bs.std()


f = zt * np.exp(-U * zt * zt)
print("=== THE PREDICTION vs THE MEASUREMENT, charge-even, units 1e-3 ===")
print(f"{'sample':34s} {'n':>7s} {'measured':>10s} {'+-':>5s} "
      f"{'PRED loc':>9s} {'PRED skew':>10s} {'PRED tot':>9s} {'pull':>6s}")


def line(tag, m):
    d_, e_ = ev(f, m, 200)
    pl = CM * dz_mean[m].mean()
    pk = CK * dz_k3[m].mean()
    print(f"{tag:34s} {m.sum():7d} {d_*1e3:+10.2f} {e_*1e3:5.2f} "
          f"{pl*1e3:+9.2f} {pk*1e3:+10.3f} {(pl+pk)*1e3:+9.2f} "
          f"{(d_-pl-pk)/max(e_,1e-12):+6.1f}")


for ib, (lo, hi, nm) in enumerate(BANDS):
    line(nm, good & (band == ib))
print()
for ib, (lo, hi, nm) in enumerate(BANDS):
    m0 = good & (band == ib)
    for var, vn in ((share_pix, "pixel share"), (share_n1, "single-strip share"),
                    (sigrel, "sigma_rel")):
        med = np.median(var[m0])
        line(f"{nm[:6]}: {vn} LOW", m0 & (var <= med))
        line(f"{nm[:6]}: {vn} HIGH", m0 & (var > med))
    print()

print("=== the DEGRADED-CONTENT contrast (HIGH minus LOW), 1e-3 ===")
print(f"{'band / variable':34s} {'measured':>10s} {'+-':>5s} {'PRED':>9s}")
for ib, (lo, hi, nm) in enumerate(BANDS):
    m0 = good & (band == ib)
    for var, vn, hi_is_bad in ((share_pix, "pixel share", False),
                               (share_n1, "single-strip share", True),
                               (sigrel, "sigma_rel", True)):
        med = np.median(var[m0])
        mh, ml = m0 & (var > med), m0 & (var <= med)
        dh, eh = ev(f, mh, 200)
        dl, el = ev(f, ml, 200)
        ph = CM * dz_mean[mh].mean() + CK * dz_k3[mh].mean()
        pl_ = CM * dz_mean[ml].mean() + CK * dz_k3[ml].mean()
        print(f"{nm[:6]+' '+vn:34s} {(dh-dl)*1e3:+10.2f} "
              f"{np.hypot(eh,el)*1e3:5.2f} {(ph-pl_)*1e3:+9.2f}")
np.savez_compressed("data/pred_track.npz", dz_mean=dz_mean.astype(np.float32),
                    dz_k3=dz_k3.astype(np.float32), z=z.astype(np.float32),
                    zt=zt.astype(np.float32), eta=b["t_eta"], q=q.astype(np.int8),
                    share_pix=share_pix.astype(np.float32),
                    share_n1=share_n1.astype(np.float32),
                    sigrel=sigrel.astype(np.float32), good=good)
