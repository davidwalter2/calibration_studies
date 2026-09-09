#!/usr/bin/env python3
"""PER-VARIANT analysis of the convergence refits: is the charge-even odd
moment a property of the CONVERGED estimator, or of where the iteration
stopped?

Five blocks, all on the truth-referenced pull `x`:
  (A) the charge-even <x> per eta band            -- the target
  (B) the trim scan                               -- core vs tail
  (C) the bulk / IN mixture at p90 of |seed->final dq/p|
  (D) the iteration census (niter, edm, chi2/ndof)
  (E) the sigma_rel scaling of the charge-even mean -- the SECOND-ORDER test:
      a second-order (Box) bias of a nonlinear least-squares estimator is
      O(sigma^2) in the parameter, hence O(sigma) in the PULL, so <x> must run
      LINEARLY THROUGH THE ORIGIN with sigma_rel. A hit-level location bias,
      by contrast, is O(sigma^0) in the parameter and gives <x> ~ 1/sigma_rel.
"""
import argparse

import numpy as np

import conv_common as cc

TS = [1., 2., 3., 5., 10., np.inf]


def fT(x, q, m, T):
    if not np.isfinite(T):
        return 1.
    v = np.concatenate([x[m & (q > 0)], -x[m & (q < 0)]])
    v = np.concatenate([v, -v])
    Q = (np.abs(v) < T).mean()
    h = 0.1 * max(T / 5., 1.)
    dens = ((np.abs(np.abs(v) - T) < h / 2).mean() / h) / 2.
    return 1. - 2 * T * dens / max(Q, 1e-9)


def run(V, rng):
    print(f"\n{'='*78}\n=== VARIANT {V.name}   {len(V)} tracks, "
          f"{int(V.good.sum())} with a finite x\n{'='*78}")

    print("\n(A) charge-even <x> per eta band, units 1e-3 "
          "(charge-ODD in brackets = control)")
    vs, es = [], []
    for ib, nm in enumerate(cc.BANDS):
        m = V.good & (V.band == ib)
        v, e, n = cc.even_mean(V.x, V.q, m, rng)
        o, oe, _ = cc.odd_mean(V.x, V.q, m, rng)
        vs.append(v)
        es.append(e)
        print(f"    {nm:18s} n={n:7d}   even {v*1e3:+7.2f}+-{e*1e3:4.2f}"
              f"   [odd {o*1e3:+8.2f}+-{oe*1e3:5.2f}]")
    mu, sm, c2 = cc.combine(vs, es)
    print(f"    {'ALL BANDS':18s}           even {mu*1e3:+7.2f}+-{sm*1e3:4.2f}"
          f"   (chi2 {c2:4.1f}/2 for eta-independence)")

    print("\n(B) trim scan  m_implied(T) = <x>_even,|x|<T / f(T), units 1e-3 "
          "(FLAT = core shift)")
    print(f"    {'band':18s}" + "".join(f"{f'T={t:g}':>10s}" for t in TS))
    for ib, nm in enumerate(cc.BANDS_SHORT):
        m = V.good & (V.band == ib)
        row = ""
        for T in TS:
            v, e, n = cc.even_mean(V.x, V.q, m & (np.abs(V.x) < T), rng, 100)
            row += f"{v*1e3/fT(V.x, V.q, m, T):+10.2f}"
        print(f"    {nm:18s}{row}")

    print("\n(C) mixture at the 90th percentile of |seed->final dq/p|")
    t = np.percentile(V.dq_seed[V.good], 90.)
    out = {}
    for lab, cut in (("IN  (top 10%)", V.dq_seed > t), ("OUT (rest)", V.dq_seed <= t)):
        vs, es, row = [], [], ""
        for ib in range(3):
            m = V.good & (V.band == ib) & cut
            v, e, n = cc.even_mean(V.x, V.q, m, rng)
            vs.append(v)
            es.append(e)
            f = 100. * m.sum() / max((V.good & (V.band == ib)).sum(), 1)
            row += f"{v*1e3:+8.2f}+-{e*1e3:5.2f}({f:4.1f}%)"
        mu, sm, c2 = cc.combine(vs, es)
        out[lab.split()[0]] = (mu, sm)
        print(f"    {lab:15s}{row}   comb {mu*1e3:+7.2f}+-{sm*1e3:4.2f} "
              f"(chi2 {c2:4.1f}/2)")
    print("    what the IN population is:")
    print(f"      {'':18s}" + "".join(f"{n:>11s}" for n in
          ("frac", "<niter>", "chi2/ndof", "sigma_rel", "<nvalid>", "<npix>",
           "rms(x)", "<edm>")))
    for lab, cut in (("IN  (top 10%)", V.dq_seed > t), ("OUT (rest)", V.dq_seed <= t)):
        m = V.good & cut
        print(f"      {lab:18s}" + "".join(f"{v:11.4f}" for v in (
            m.sum() / V.good.sum(), V.d["niter"][m].mean(),
            V.d["chi2n"][m].mean(), V.sigrel[m].mean(),
            V.d["nvalid"][m].mean(), V.d["npixhit"][m].mean(),
            V.x[m].std(), np.median(V.d["edm"][m]))))

    print("\n(D) iteration census")
    ni = V.d["niter"]
    tot = len(ni)
    cens = ", ".join(f"{k}:{100.*(ni==k).sum()/tot:5.2f}%"
                     for k in sorted(set(ni.tolist()))[:12])
    print(f"    niter  mean {ni.mean():6.3f}  max {ni.max()}   {cens}")
    # AT THE CAP means the tolerance was NOT reached: a variant that only
    # raises nIters and then saturates it has not tested convergence, it has
    # tested the cap.
    cap = int(ni.max())
    print(f"    at the iteration cap ({cap}): {int((ni == cap).sum())} tracks "
          f"({100.*(ni == cap).mean():.4f} %)   "
          f"edmref >= 1e-7 for {100.*(V.d['edmref'] >= 1e-7).mean():.2f} %, "
          f">= 1e-5 for {100.*(V.d['edmref'] >= 1e-5).mean():.4f} %")
    ed = V.d["edm"]
    print(f"    edmval median {np.median(ed):.3e}  "
          f"p90 {np.percentile(ed, 90):.3e}  p99 {np.percentile(ed, 99):.3e}  "
          f"max {ed.max():.3e}")
    print(f"    chi2/ndof mean {V.d['chi2n'].mean():.5f}  "
          f"median {np.median(V.d['chi2n']):.5f}")
    print(f"    nChargeFlipProtect>0: {int((V.d['flip']>0).sum())} / {tot}")
    print(f"    |seed->final dq/p| mean {V.dq_seed.mean():.5f}  "
          f"p90 {np.percentile(V.dq_seed,90):.5f}")
    print(f"    |iter0->final dq/p| mean {V.dq_it0.mean():.3e}  "
          f"p90 {np.percentile(V.dq_it0,90):.3e}")

    print("\n(E) SECOND-ORDER TEST: how does the charge-even mean scale with "
          "the fit error?")
    print("    A second-order (Box) bias of a nonlinear least-squares estimator "
          "is O(sigma^2)")
    print("    in the PARAMETER, hence O(sigma) in the PULL: <x> = c * sigma_rel, "
          "LINEAR THROUGH")
    print("    THE ORIGIN, with c = the relative curvature bias per unit "
          "sigma_rel^2. A fixed")
    print("    hit-LOCATION bias is O(sigma^0) in the parameter and gives "
          "<x> ~ 1/sigma_rel.")
    print("    Bins are in GEN pT, which is exogenous -- binning on the FITTED "
          "sigma or the")
    print("    reco pT is the conditioning trap this campaign has now hit four "
          "times.")
    res = {}
    for tag, binv in (("GEN pT", V.d["genpt"].astype(float)),
                      ("reco sigma_rel (cross-check)", V.sigrel)):
        qs = np.percentile(binv[V.good], [0, 20, 40, 60, 80, 100])
        print(f"\n    binned in {tag}")
        print(f"    {'bin':26s}{'<sigma_rel>':>13s}{'n':>9s}{'<x>_even':>18s}"
              f"{'<x>_odd':>16s}")
        sr, ev, ee = [], [], []
        for i in range(5):
            m = V.good & (binv >= qs[i]) & ((binv < qs[i + 1]) | (i == 4))
            v, e, n = cc.even_mean(V.x, V.q, m, rng)
            o, oe, _ = cc.odd_mean(V.x, V.q, m, rng)
            s_ = V.sigrel[m].mean()
            sr.append(s_)
            ev.append(v)
            ee.append(e)
            print(f"    {qs[i]:10.4f}-{qs[i+1]:<10.4f}{'':4s}{s_:11.5f}"
                  f"{n:9d}{v*1e3:+13.2f}+-{e*1e3:4.2f}"
                  f"{o*1e3:+11.2f}+-{oe*1e3:4.2f}")
        sr, ev, ee = map(np.asarray, (sr, ev, ee))
        w = 1. / ee ** 2
        k0 = (w * sr * ev).sum() / (w * sr * sr).sum()
        ek0 = 1. / np.sqrt((w * sr * sr).sum())
        r0 = float((w * (ev - k0 * sr) ** 2).sum())
        # the LOCATION alternative: <x> = A / sigma_rel
        iv = 1. / sr
        kA = (w * iv * ev).sum() / (w * iv * iv).sum()
        rA = float((w * (ev - kA * iv) ** 2).sum())
        # a constant, i.e. no scaling at all
        kC = (w * ev).sum() / w.sum()
        rC = float((w * (ev - kC) ** 2).sum())
        print(f"    second-order   <x> = c sigma_rel : c = {k0:+.4f} +- {ek0:.4f}"
              f"   chi2 {r0:6.2f}/4")
        print(f"    location       <x> = A / sigma_rel: A = {kA*1e6:+.3f}e-6"
              f"          chi2 {rA:6.2f}/4")
        print(f"    no scaling     <x> = const        : {kC*1e3:+.2f}e-3"
              f"             chi2 {rC:6.2f}/4")
        res[tag] = (k0, ek0, r0, rA, rC)
    return {"band": vs, "IN": out.get("IN"), "OUT": out.get("OUT"),
            "niter": ni.mean(), "scaling": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    a = ap.parse_args()
    rng = np.random.default_rng(2026)
    for p in a.npz:
        run(cc.Var(p), rng)


if __name__ == "__main__":
    main()
