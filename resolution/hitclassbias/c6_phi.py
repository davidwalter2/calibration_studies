#!/usr/bin/env python3
"""IS THE CHARGE-EVEN SHIFT PHI-UNIFORM? (No. It carries n = 8 and n = 10.)

The charge-even pull shift is a SAGITTA-like, charge-dependent momentum bias,
and the mirror argument says it needs a CHIRAL source. The two families of
chiral source differ in exactly one observable:

* a source that is fixed in the MODULE's local frame -- a Lorentz-drift CPE
  bias, a charge-sharing bias on a tilted ladder -- averages to a PHI-UNIFORM
  global curvature bias, because every azimuth sees the same module geometry
  (up to the 12-fold / 20-fold ladder and rod symmetry);
* a residual azimuthal WEAK MODE of the geometry (a twist, a global sagitta
  deformation) is by construction phi-DEPENDENT, with cos(phi) / sin(phi) and
  cos(2 phi) content.

The geometry here is IDEAL (`useIdealGeometry=True`), so the second family
should be absent and this is a null test -- which is exactly why it is worth
doing: a nonzero low-order harmonic would mean the effect is not what any of
the hypotheses on the table say it is.

Reported: the charge-even <x> in phi bins with the first three harmonics per
eta band; a SPLIT-HALF test on the lumi parity (the two halves are disjoint
event sets, so a fluctuation cannot reproduce between them); a PERMUTATION
null that validates the errors; and an UNBINNED harmonic scan, which is the
right tool because binning aliases everything above the Nyquist frequency
(n > nphi/2) down onto the low harmonics -- reading a 24-bin scan naively
turns n = 14, 16, 17 into n = 10, 8, 7.
"""
import argparse

import numpy as np

import conv_common as cc


def harm(V, g, ph, n, fn):
    """Unbinned charge-even harmonic amplitude 2<x f(n phi)> and its error.

    Unbinned because the amplitude of a harmonic is a MEAN, and a mean needs
    no bins: this has no aliasing, no binning bias, and the smallest possible
    error (2 rms(x)/sqrt(N) = 3.6e-3 on this sample).
    """
    x, q, p = V.x[g], V.q[g], ph[g]
    out = []
    for sgn in (+1, -1):
        s = q == sgn
        v = 2. * np.mean(x[s] * fn(n * p[s]))
        e = 2. * np.std(x[s] * fn(n * p[s])) / np.sqrt(s.sum())
        out.append((v, e))
    return (0.5 * (out[0][0] + out[1][0]),
            0.5 * np.hypot(out[0][1], out[1][1]))


def scan(V, g, ph, nmax=40, thresh=9.):
    print(f"\n  --- UNBINNED harmonic scan, n = 1..{nmax} "
          f"(no aliasing, no binning) ---")
    print(f"  {'n':>4s}{'cos [1e-3]':>18s}{'sin [1e-3]':>18s}{'power':>10s}")
    tot, rows = 0., []
    for n in range(1, nmax + 1):
        c, ec = harm(V, g, ph, n, np.cos)
        s, es = harm(V, g, ph, n, np.sin)
        pw = (c / ec) ** 2 + (s / es) ** 2
        tot += pw
        rows.append((pw, n))
        if pw > thresh:
            print(f"  {n:4d}{c*1e3:+13.2f}+-{ec*1e3:4.2f}"
                  f"{s*1e3:+13.2f}+-{es*1e3:4.2f}{pw:10.1f}")
    rows.sort(reverse=True)
    print(f"  total power n=1..{nmax}: {tot:.1f} for {2*nmax} dof "
          f"(expectation {2*nmax})")
    print(f"  strongest: " + ", ".join(f"n={n} ({p:.0f})" for p, n in rows[:6]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--nphi", type=int, default=12)
    a = ap.parse_args()
    rng = np.random.default_rng(66)
    for path in a.npz:
        V = cc.Var(path)
        if "genphi" not in V.d:
            print(f"{V.name}: no phi in this cache -- re-extract")
            continue
        ph = V.d["genphi"].astype(float)
        edges = np.linspace(-np.pi, np.pi, a.nphi + 1)
        print(f"\n{'='*78}\n=== PHI STRUCTURE -- {V.name}\n{'='*78}")
        for band, nm in [(None, "all eta")] + list(enumerate(cc.BANDS_SHORT)):
            base = V.good if band is None else V.good & (V.band == band)
            vs, es, cs = [], [], []
            for i in range(a.nphi):
                m = base & (ph >= edges[i]) & (ph < edges[i + 1])
                v, e, n = cc.even_mean(V.x, V.q, m, rng, 150)
                vs.append(v)
                es.append(e)
                cs.append(0.5 * (edges[i] + edges[i + 1]))
            vs, es, cs = map(np.asarray, (vs, es, cs))
            ok = np.isfinite(vs) & (es > 0)
            w = 1. / es[ok] ** 2
            mu = (vs[ok] * w).sum() / w.sum()
            chi2 = float((w * (vs[ok] - mu) ** 2).sum())
            out = [f"  {nm:10s} mean {mu*1e3:+7.2f}+-{1e3/np.sqrt(w.sum()):4.2f}"
                   f"  chi2(flat) {chi2:6.2f}/{ok.sum()-1}"]
            for k in (1, 2, 3):
                for fn, lb in ((np.cos, "cos"), (np.sin, "sin")):
                    b = fn(k * cs[ok])
                    amp = (w * b * (vs[ok] - mu)).sum() / (w * b * b).sum()
                    eamp = 1. / np.sqrt((w * b * b).sum())
                    out.append(f"{lb}{k}{amp*1e3:+7.2f}+-{eamp*1e3:4.2f}")
            print("  ".join(out))
        # the SPLIT-HALF test: if the phi pattern is real it repeats in two
        # statistically independent halves of the sample; if it is a
        # fluctuation the two halves are uncorrelated. Split on the LUMI
        # parity, which indexes the production task and is therefore
        # independent of everything physical.
        half = (V.d["lumi"] % 2).astype(bool)
        rows = {}
        for lab, hm in (("all", np.ones(len(V), bool)), ("even lumi", ~half),
                        ("odd lumi", half)):
            vs, es = [], []
            for i in range(a.nphi):
                m = V.good & hm & (ph >= edges[i]) & (ph < edges[i + 1])
                v, e, n = cc.even_mean(V.x, V.q, m, rng, 150)
                vs.append(v * 1e3)
                es.append(e * 1e3)
            rows[lab] = (np.asarray(vs), np.asarray(es))
            w = 1. / np.asarray(es) ** 2
            null = float((w * np.asarray(vs) ** 2).sum())
            mu = (np.asarray(vs) * w).sum() / w.sum()
            flat = float((w * (np.asarray(vs) - mu) ** 2).sum())
            print(f"  per-bin {lab:10s} chi2(no effect) {null:6.2f}/{a.nphi}"
                  f"  chi2(flat) {flat:6.2f}/{a.nphi-1}  mean {mu:+6.2f}e-3")
            print("    " + "  ".join(f"{v:+6.1f}+-{e:4.1f}"
                                     for v, e in zip(vs, es)))
        (va, ea) = rows["even lumi"]
        (vb, eb) = rows["odd lumi"]
        w = 1. / (ea ** 2 + eb ** 2)
        num = (w * (va - va.mean()) * (vb - vb.mean())).sum()
        den = np.sqrt((w * (va - va.mean()) ** 2).sum()
                      * (w * (vb - vb.mean()) ** 2).sum())
        chi2ab = float((w * (va - vb) ** 2).sum())
        print(f"  SPLIT-HALF: weighted corr(even, odd) = {num/den:+.3f}, "
              f"chi2(even == odd) = {chi2ab:.2f}/{a.nphi}")
        print("    a REAL phi pattern gives a positive correlation and a small "
              "chi2; a fluctuation gives neither.")

        # PERMUTATION NULL: reassign phi at random. Anything the machinery
        # invents shows up here.
        gg = V.good & (np.abs(V.x) < 10)
        for t in range(3):
            perm = rng.permutation(len(ph))
            vs, es = [], []
            for i in range(a.nphi):
                m = gg & (ph[perm] >= edges[i]) & (ph[perm] < edges[i + 1])
                v, e, _ = cc.even_mean(V.x, V.q, m, rng, 120)
                vs.append(v)
                es.append(e)
            w = 1. / np.asarray(es) ** 2
            print(f"  permutation null {t}: chi2(no effect) = "
                  f"{float((w*np.asarray(vs)**2).sum()):6.2f}/{a.nphi}")

        scan(V, gg, ph)
        print("\n  --- where the strongest harmonics live ---")
        print(f"  {'selection':22s}" + "".join(f"{f'n={n} sin':>18s}"
                                               for n in (8, 10)))
        t90 = np.percentile(V.dq_seed[V.good], 90.)
        for lab, sel in ([(nm, V.band == i)
                          for i, nm in enumerate(cc.BANDS_SHORT)]
                         + [(f"nValidPixelHits=={k}", V.d["npixhit"] == k)
                            for k in range(5)]
                         + [("mixture OUT (bulk)", V.dq_seed <= t90),
                            ("mixture IN (top 10%)", V.dq_seed > t90)]):
            row = ""
            for n in (8, 10):
                s_, es_ = harm(V, gg & sel, ph, n, np.sin)
                row += f"{s_*1e3:+13.1f}+-{es_*1e3:4.1f}"
            print(f"  {lab:22s}{row}")


if __name__ == "__main__":
    main()
