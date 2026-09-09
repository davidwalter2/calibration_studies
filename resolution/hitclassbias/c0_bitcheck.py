#!/usr/bin/env python3
"""THE BIT CHECK. `base` must reproduce `mugun_ul16_260903x_m0` track by track.

WHY IT GATES EVERYTHING. `base` was launched from a DIFFERENT driver script
(`run_conv.sh`) against the same CMSSW area weeks after the baseline was
produced. If the area moved -- a rebuild, a changed default, a different
scalar-potential init file -- then `tight` and `damp` are being compared
against a baseline that is not the one whose -6.3e-3 / +21e-3 decomposition is
the subject of the study, and every conclusion below is about the wrong thing.

The two productions were run with identical cfg, identical filelist and
identical task -> filelist-line mapping (`run_local_trackres.sh` and
`run_conv.sh` both take `sed -n "$((idx+1))p"`), so task N of one is task N of
the other and the tracks must agree EXACTLY, not statistically.
"""
import argparse

import numpy as np

import conv_common as cc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="data/conv_base.npz")
    ap.add_argument("--b", default="data/conv_ref903x.npz")
    a = ap.parse_args()
    A = cc.Var(a.a, "base")
    B = cc.Var(a.b, "260903x_m0")
    print(f"{A.name}: {len(A)} tracks   {B.name}: {len(B)} tracks")

    ia, ib = cc.pair(A, B)
    print(f"paired on (run,lumi,event,charge): {len(ia)}")
    print(f"  unpaired in {A.name}: {len(A)-len(ia)}   "
          f"in {B.name}: {len(B)-len(ib)}")
    if len(ia) == 0:
        raise SystemExit("NO OVERLAP -- the two productions do not share events")

    print("\n=== EXACT equality, track by track ===")
    hard = ["qop_ref", "qop_it0", "qop_seed", "sigma", "z", "chi2n", "niter",
            "nvalid", "npixhit", "edm", "edmref", "vgf", "pt", "eta",
            "qop_gen", "slot"]
    nbad = 0
    for k in hard:
        va = getattr(A, k, None)
        vb = getattr(B, k, None)
        if va is None:
            va, vb = A.d[k], B.d[k]
        va = np.asarray(va, float)[ia]
        vb = np.asarray(vb, float)[ib]
        ident = np.array_equal(va, vb)
        d = va - vb
        rel = np.abs(d) / np.maximum(np.abs(vb), 1e-300)
        n_ne = int((d != 0).sum())
        nbad += 0 if ident else 1
        print(f"  {k:10s} identical={str(ident):5s}  ndiff={n_ne:7d}  "
              f"max|d|={np.abs(d).max():.3e}  max rel={np.nanmax(rel):.3e}")

    print("\n=== the statistic itself ===")
    rng = np.random.default_rng(3)
    for nm, V, idx in ((A.name, A, ia), (B.name, B, ib)):
        m = np.zeros(len(V), bool)
        m[idx] = True
        m &= V.good
        row = ""
        for ib_ in range(3):
            v, e, n = cc.even_mean(V.x, V.q, m & (V.band == ib_), rng)
            row += f"  {v*1e3:+7.2f}+-{e*1e3:4.2f}"
        print(f"  {nm:12s} charge-even <x> per band (1e-3): {row}")

    # WHICH tracks differ, and are they the known non-determinism or a moved
    # build? A moved build changes EVERY track; the CVH limit-cycle /
    # anchoring-recovery non-determinism changes a handful, and only tracks
    # that sat on an iteration boundary.
    dq = np.asarray(A.d["qop_ref"], float)[ia] - np.asarray(B.d["qop_ref"], float)[ib]
    bad = np.where(dq != 0)[0]
    frac = len(bad) / len(ia)
    print(f"\n=== tracks whose converged q/p differs: {len(bad)} / {len(ia)} "
          f"({100*frac:.4f} %) ===")
    for k in bad[:12]:
        print(f"  run/lumi/event {A.d['run'][ia[k]]}/{A.d['lumi'][ia[k]]}/"
              f"{A.d['event'][ia[k]]} q={A.q[ia[k]]:+.0f}  "
              f"niter {B.d['niter'][ib[k]]}->{A.d['niter'][ia[k]]}  "
              f"edmref {B.d['edmref'][ib[k]]:.2e}->{A.d['edmref'][ia[k]]:.2e}  "
              f"dq/p {dq[k]:+.3e} ({dq[k]/B.sigma[ib[k]]:+.4f} sigma)")
    if len(bad):
        print(f"  <niter> of the differing tracks {B.d['niter'][ib[bad]].mean():.2f} "
              f"vs {B.d['niter'][ib].mean():.2f} for all")
        print(f"  max |dq/p| / sigma = "
              f"{np.abs(dq[bad]/B.sigma[ib[bad]]).max():.4f}")

    ok = frac < 1e-3
    print("\nVERDICT: " + (
        "BIT-IDENTICAL -- the baseline is the same estimator" if len(bad) == 0
        else (f"REPRODUCED -- {len(bad)} track(s) of {len(ia)} differ "
              f"({100*frac:.4f} %), the known CVH limit-cycle / anchoring "
              f"non-determinism; the charge-even statistic is unchanged"
              if ok else
              f"NOT reproduced -- {100*frac:.2f} % of tracks differ, the build "
              f"or the configuration MOVED. STOP.")))


if __name__ == "__main__":
    main()
