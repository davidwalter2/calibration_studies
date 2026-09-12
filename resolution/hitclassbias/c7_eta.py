#!/usr/bin/env python3
"""THE ETA STRUCTURE, at 24 bins in SIGNED gen eta -- the companion to the phi
scan, and the sharper version of the three-band table.

The three-band numbers (-4.9 / -3.6 / +0.1 e-3) invite the reading "the
charge-even skew depends on eta". The mixture analysis says it does not -- the
band pattern is two eta-independent components whose fraction runs with
|eta|. This is the direct test at 24 bins, with an exogenous
(generated) eta and analytic errors, and it also splits the profile into its
eta-EVEN and eta-ODD parts: an eta-ODD charge-even curvature bias would be a
z-antisymmetric sagitta twist, which is a different (and more alarming) object
than a uniform offset.
"""
import argparse

import numpy as np

import conv_common as cc


def profile(V, g, v, edges):
    vs, es = [], []
    for i in range(len(edges) - 1):
        m = g & (v >= edges[i]) & (v < edges[i + 1])
        xp, xm = V.x[m & (V.q > 0)], V.x[m & (V.q < 0)]
        if len(xp) < 20 or len(xm) < 20:
            vs.append(np.nan)
            es.append(np.nan)
            continue
        vs.append(0.5 * (xp.mean() + xm.mean()))
        es.append(0.5 * np.hypot(xp.std() / np.sqrt(len(xp)),
                                 xm.std() / np.sqrt(len(xm))))
    return np.asarray(vs), np.asarray(es)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--nbin", type=int, default=24)
    ap.add_argument("--clip", type=float, default=10.)
    a = ap.parse_args()
    for path in a.npz:
        V = cc.Var(path)
        g = V.good & (np.abs(V.x) < a.clip)
        ge = V.d["geneta"].astype(float)
        eb = np.linspace(-2.4, 2.4, a.nbin + 1)
        vs, es = profile(V, g, ge, eb)
        w = 1. / es ** 2
        mu = (vs * w).sum() / w.sum()
        print(f"\n{'='*78}\n=== ETA STRUCTURE -- {V.name}\n{'='*78}")
        print(f"  charge-even <x> in {a.nbin} bins of SIGNED gen eta [1e-3]")
        step = 6
        for i in range(0, a.nbin, step):
            print(f"  eta {eb[i]:+.1f}..{eb[i+step]:+.1f}  " + " ".join(
                f"{vs[j]*1e3:+6.1f}+-{es[j]*1e3:4.1f}"
                for j in range(i, min(i + step, a.nbin))))
        print(f"  mean {mu*1e3:+.2f}+-{1e3/np.sqrt(w.sum()):.2f} e-3   "
              f"chi2(no effect) {float((w*vs**2).sum()):.1f}/{a.nbin}   "
              f"chi2(FLAT) {float((w*(vs-mu)**2).sum()):.1f}/{a.nbin-1}")
        h = a.nbin // 2
        p, m_ = vs[h:], vs[:h][::-1]
        ep, em = es[h:], es[:h][::-1]
        ee = 0.5 * np.hypot(ep, em)
        we = 1. / ee ** 2
        print("  eta-EVEN 0.5(N+S): " + " ".join(f"{0.5*(x+y)*1e3:+6.1f}"
                                                 for x, y in zip(p, m_)))
        print("  eta-ODD  0.5(N-S): " + " ".join(f"{0.5*(x-y)*1e3:+6.1f}"
                                                 for x, y in zip(p, m_)))
        print(f"  chi2(eta-ODD == 0) = "
              f"{float((we*(0.5*(p-m_))**2).sum()):.1f}/{h}  "
              f"-- a z-antisymmetric sagitta twist would live here")


if __name__ == "__main__":
    main()
