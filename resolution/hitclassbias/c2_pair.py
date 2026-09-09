#!/usr/bin/env python3
"""PAIRED comparison between two convergence variants.

The whole point. Two independent 320 k samples measure the charge-even <x> to
+-1.8e-3 each, so a difference is only visible above ~2.6e-3. The SAME tracks
refitted with different Gauss-Newton knobs share every stochastic ingredient
(the same hits, the same scatters, the same seed), so the per-track difference
`Delta x` has a variance set only by what the estimator did differently -- in
practice two to three orders of magnitude smaller. `<Delta x>_even` per eta
band is therefore the DIRECT measure of what the extra iterations change, and
it is the number the decision rule needs, not the two absolute means.
"""
import argparse

import numpy as np

import conv_common as cc


def boot_even(v, q, mask, rng, nboot=400):
    idx = np.where(mask)[0]
    if len(idx) < 20:
        return np.nan, np.nan, len(idx)
    val = 0.5 * (v[idx][q[idx] > 0].mean() + v[idx][q[idx] < 0].mean())
    bs = np.empty(nboot)
    for i in range(nboot):
        k = idx[rng.integers(0, len(idx), len(idx))]
        bs[i] = 0.5 * (v[k][q[k] > 0].mean() + v[k][q[k] < 0].mean())
    return val, bs.std(), len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--na", default=None)
    ap.add_argument("--nb", default=None)
    args = ap.parse_args()
    A = cc.Var(args.a, args.na)
    B = cc.Var(args.b, args.nb)
    rng = np.random.default_rng(77)
    ia, ib = cc.pair(A, B)
    print(f"\n{'='*78}\n=== PAIRED {A.name} -> {B.name}\n{'='*78}")
    print(f"{A.name}: {len(A)}   {B.name}: {len(B)}   paired {len(ia)}")

    q = A.q[ia]
    band = A.band[ia]
    good = A.good[ia] & B.good[ib]
    dqop = (B.d["qop_ref"][ib] - A.d["qop_ref"][ia]) / np.abs(A.d["qop_ref"][ia])
    dz = B.z[ib] - A.z[ia]
    dx = B.x[ib] - A.x[ia]
    dsig = (B.sigma[ib] - A.sigma[ia]) / A.sigma[ia]
    dni = B.d["niter"][ib].astype(float) - A.d["niter"][ia].astype(float)
    dchi = B.d["chi2n"][ib] - A.d["chi2n"][ia]

    nz = int((dqop != 0).sum())
    print(f"\ntracks whose q/p MOVED at all: {nz} / {len(ia)} "
          f"({100.*nz/len(ia):.3f} %)")
    if nz:
        m = dqop != 0
        print(f"   among those: median |dq/p| = {np.median(np.abs(dqop[m])):.3e}"
              f"   p90 {np.percentile(np.abs(dqop[m]),90):.3e}"
              f"   max {np.abs(dqop[m]).max():.3e}")
    print(f"niter: <A> {A.d['niter'][ia].mean():.4f} -> <B> "
          f"{B.d['niter'][ib].mean():.4f}   <dniter> {dni.mean():+.4f}")
    print(f"chi2/ndof: <A> {A.d['chi2n'][ia].mean():.6f} -> <B> "
          f"{B.d['chi2n'][ib].mean():.6f}   <dchi2n> {dchi.mean():+.3e}"
          f"   (improved for {100.*(dchi<0).mean():.2f} % of tracks)")
    print(f"sigma: <d sigma/sigma> {dsig.mean():+.3e}")

    print("\n=== per-track CHANGE, charge-even and charge-odd, per eta band ===")
    print(f"{'band':18s}{'<d(q/p)/(q/p)>_even':>24s}{'_odd':>22s}"
          f"{'<dx>_even (1e-3)':>22s}")
    for ib_, nm in enumerate(cc.BANDS_SHORT):
        m = good & (band == ib_)
        ve, ee, n = boot_even(dqop, q, m, rng)
        vo = 0.5 * (dqop[m & (q > 0)].mean() - dqop[m & (q < 0)].mean())
        xe, xee, _ = boot_even(dx, q, m, rng)
        print(f"{nm:18s}{ve:+15.3e}+-{ee:8.1e}{vo:+15.3e}"
              f"{xe*1e3:+15.3f}+-{xee*1e3:5.3f}")
    m = good
    ve, ee, n = boot_even(dqop, q, m, rng)
    xe, xee, _ = boot_even(dx, q, m, rng)
    ze, zee, _ = boot_even(dz, q, m, rng)
    print(f"{'ALL':18s}{ve:+15.3e}+-{ee:8.1e}{'':15s}"
          f"{xe*1e3:+15.3f}+-{xee*1e3:5.3f}   (<dz>_even "
          f"{ze*1e3:+.3f}+-{zee*1e3:.3f} e-3)")

    print("\n=== the STATISTIC in each variant on the SAME paired tracks ===")
    print(f"{'band':18s}{A.name:>22s}{B.name:>22s}{'B - A (paired)':>22s}")
    for ib_, nm in enumerate(cc.BANDS_SHORT):
        m = good & (band == ib_)
        va, ea, _ = boot_even(A.x[ia], q, m, rng)
        vb, eb, _ = boot_even(B.x[ib], q, m, rng)
        vd, ed, _ = boot_even(dx, q, m, rng)
        print(f"{nm:18s}{va*1e3:+15.2f}+-{ea*1e3:4.2f}"
              f"{vb*1e3:+15.2f}+-{eb*1e3:4.2f}"
              f"{vd*1e3:+15.3f}+-{ed*1e3:5.3f}")

    print("\n=== the MIXTURE, recomputed in each variant on the paired set ===")
    for nm, V, idx in ((A.name, A, ia), (B.name, B, ib)):
        t = np.percentile(V.dq_seed[idx][good], 90.)
        for lab, cut in (("IN ", V.dq_seed[idx] > t), ("OUT", V.dq_seed[idx] <= t)):
            vs, es, row = [], [], ""
            for ib_ in range(3):
                m = good & (band == ib_) & cut
                v, e, _ = boot_even(V.x[idx], q, m, rng)
                vs.append(v)
                es.append(e)
                row += f"{v*1e3:+8.2f}+-{e*1e3:5.2f}"
            mu, sm, c2 = cc.combine(vs, es)
            print(f"  {nm:12s} {lab}{row}   comb {mu*1e3:+7.2f}+-{sm*1e3:4.2f}")
    # and the PAIRED difference inside the two populations of A
    t = np.percentile(A.dq_seed[ia][good], 90.)
    print("  paired B-A inside A's populations:")
    for lab, cut in (("IN ", A.dq_seed[ia] > t), ("OUT", A.dq_seed[ia] <= t)):
        v, e, n = boot_even(dx, q, good & cut, rng)
        print(f"    {lab} <dx>_even = {v*1e3:+8.3f}+-{e*1e3:5.3f} e-3  (n={n})")


if __name__ == "__main__":
    main()
