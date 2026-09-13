#!/usr/bin/env python3
"""TAIL study, THE CLOSURE: the residual tail is the chi2 distribution.

Every residual of a candidate is a linear functional of the SAME noise vector
whose squared length is the fit's chi2.  If the fit's covariance is right up
to one per-candidate SCALE s -- the true noise being `s^2 V_model` -- then

    chi2/ndof ~= s^2 ,   Var(z | s) = s^2 ,   z = s * N(0,1)

so the pull's marginal density is a SCALE MIXTURE of normals with the mixing
density read off `chi2/ndof` itself.  Its tail is then

    P(|z| > t) = E_s[ 2 Phi(-t/s) ]

with NO free parameter.  This script measures the ladder's slope (which must
be 1), builds the mixture prediction from the measured `chi2/ndof` and
compares it with the observed tail; and it compares the measured `chi2/ndof`
density with the chi2_ndof/ndof one the model implies, which is where the
excess actually is.

usage: python3 tail_mixture.py --npz <dy npz> [--gun <gun npz>] [--label ...]
"""
import argparse, os, sys
import numpy as np
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import genbkg as GB          # noqa: E402
import tail_common as TC     # noqa: E402


def load(fn, need_bs=True, chi2=0.0):
    d = dict(np.load(fn, allow_pickle=False))
    base, chi2n, nl = TC.baseline(d, need_bs=need_bs, max_abs_vtxz=0.0,
                                  chi2=chi2, verbose=False)
    if "genidx_plus" in d:
        cls, names, _ = GB.classify(d)
        base = base & (cls == names.index("signal"))
    zv = np.asarray(d["vtxz"], float)
    nd = np.asarray(d["ndof"], float)
    chi = np.asarray(d["chisq"], float)
    sub = np.asarray(d.get("vtxdchi2", zv ** 2), float)
    nsub = 1.0
    if need_bs and "bschi2fit" in d:
        sub = sub + np.asarray(d["bschi2fit"], float)
        nsub += 3.0
    ndr = np.maximum(nd - nsub, 1.0)
    return d, base, chi2n, (chi - sub) / ndr, ndr, zv


def report(tag, sel, chired, ndr, zs, trunc=None):
    s2 = chired[sel]
    nd = ndr[sel]
    print(f"\n=== {tag}: {int(sel.sum())} candidates")
    print(f"  ndof (reduced) median {np.median(nd):.0f}   "
          f"chi2/ndof median {np.median(s2):.4f}  mean {s2.mean():.4f}  "
          f"p90 {np.percentile(s2,90):.4f}  p99 {np.percentile(s2,99):.4f}")
    # what a correct model implies, candidate by candidate
    u = stats.chi2.cdf(s2 * nd, nd)
    print(f"  the chi2 PROBABILITY, which must be FLAT on [0,1]:")
    for lo, hi in ((0., .1), (.4, .6), (.9, .99), (.99, .999), (.999, 1.0)):
        f = ((u >= lo) & (u < hi)).mean()
        print(f"    P in [{lo:.3f},{hi:.3f}) = {f:.5f}   expected {hi-lo:.5f}"
              f"   ratio {f/max(hi-lo,1e-12):7.2f}")
    print(f"  P(chi2 prob < 1e-3) = "
          f"{TC.pm(int((u > 1-1e-3).sum()), int(sel.sum()))}  (expected 0.001)")
    # the ladder's slope: Var(z|c) should be c
    print(f"\n  the ladder Var(z | chi2/ndof) -- the model says SLOPE 1, "
          f"INTERCEPT 0")
    for nm, z in zs:
        zz = z[sel]
        edges = np.percentile(s2, np.linspace(0, 97, 12))
        cs, vs, es = [], [], []
        for i in range(len(edges) - 1):
            b = (s2 >= edges[i]) & (s2 < edges[i + 1])
            if b.sum() < 40:
                continue
            cs.append(s2[b].mean()); vs.append(np.var(zz[b]))
            es.append(np.var(zz[b]) * np.sqrt(2 / b.sum()))
        cs, vs, es = np.array(cs), np.array(vs), np.array(es)
        A = np.vstack([np.ones(len(cs)), cs]).T
        W = np.diag(1 / es ** 2)
        cov = np.linalg.inv(A.T @ W @ A)
        p = cov @ (A.T @ W @ vs)
        print(f"    {nm:<5s} slope {p[1]:+7.4f} +- {np.sqrt(cov[1,1]):.4f}"
              f"   intercept {p[0]:+7.4f} +- {np.sqrt(cov[0,0]):.4f}")
    # the mixture prediction
    print(f"\n  the SCALE-MIXTURE prediction from the measured chi2/ndof, "
          f"against the measured tail (no free parameter)")
    s = np.sqrt(np.maximum(s2, 1e-12))
    hdr = (f"    {'t':>4s}{'predicted':>13s}" +
           "".join(f"{'obs '+nm:>14s}{'ratio':>8s}" for nm, _ in zs))
    print(hdr)
    for t in (2., 3., 4., 5.):
        pred = np.mean(2 * stats.norm.sf(t / s))
        line = f"    {t:>4.0f}{pred:>13.5f}"
        for nm, z in zs:
            o = (np.abs(z[sel]) > t).mean()
            line += f"{o:>14.5f}{o/max(pred,1e-12):>8.2f}"
        print(line)
    gau = [2 * stats.norm.sf(t) for t in (2., 3., 4., 5.)]
    print(f"    (a single Gaussian would give "
          f"{gau[0]:.5f} / {gau[1]:.5f} / {gau[2]:.6f} / {gau[3]:.7f})")
    return s2, u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--gun", default="")
    ap.add_argument("--label", default="DY, gen signal")
    a = ap.parse_args()
    d, sig, chi2n, chired, ndr, zv = load(a.npz)
    zs = [("z_1", d["bsz"][:, 0]), ("z_2", d["bsz"][:, 1]), ("z_v", zv)]
    report(a.label + " (NO chi2 cut)", sig, chired, ndr, zs)

    d2, sig2, _, chired2, ndr2, zv2 = load(a.npz, chi2=3.0)
    zs2 = [("z_1", d2["bsz"][:, 0]), ("z_2", d2["bsz"][:, 1]), ("z_v", zv2)]
    report(a.label + " (chi2/ndof < 3, the published baseline)",
           sig2, chired2, ndr2, zs2)

    if a.gun and os.path.exists(a.gun):
        g, gs, _, gchired, gndr, gzv = load(a.gun, need_bs=False)
        report("J/psi gun (NO chi2 cut)", gs, gchired, gndr, [("z_v", gzv)])


if __name__ == "__main__":
    main()
