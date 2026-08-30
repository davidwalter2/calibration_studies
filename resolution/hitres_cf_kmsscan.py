#!/usr/bin/env python3
"""Is the hit term wrong, or is it exposing MS? Float the MS scale in both arms.

The CF closure is a statement about the TOTAL residual: the model is
phi_hit x phi_MS x phi_ioni with MS carrying ~82 % of Var(q/p) and the hits
~11 %. So "the closure got worse when the hit term was replaced" (section 14)
does not by itself say the hit term is wrong: if MS is slightly too narrow and
the Gaussian hit term was slightly too wide, the two errors partly cancel and
correcting only the hits EXPOSES the MS one.

The two hypotheses separate once MS is allowed its own scale:

  hit term WRONG   -> with kms free, the gauss arm still reaches the better
                      optimum;
  hit term RIGHT   -> with kms free, the class arm reaches an EQUAL or BETTER
                      optimum, and at a LARGER kms, because the MS error it
                      was masking is now carried by MS.

The hit exponent is computed once per arm and only kms is scanned, so this is
cheap. Statistic: sum over probes of ((data-model)/err)^2 -- the probes are
correlated, so it is a figure of merit, not a chi2 with a calibrated p-value.

usage: python hitres_cf_kmsscan.py [--cache ...] [--ntrk 40000]
"""
import argparse
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cf_track_resolution as ctr
import hitres_classes as hc


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default="runs/cf_trackres_lowpt_cls.npz")
    p.add_argument("--ntrk", type=int, default=40000)
    p.add_argument("--probes", type=float, nargs="+",
                   default=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
    p.add_argument("--kms", type=float, nargs=3, default=[-0.02, 0.06, 17],
                   metavar=("LO", "HI", "N"))
    p.add_argument("--bank-nfiles", type=int, default=25)
    args = p.parse_args()

    d = dict(np.load(args.cache, allow_pickle=False))
    n = len(d["z"])
    rng = np.random.default_rng(11)
    sel = np.sort(rng.choice(n, min(args.ntrk, n), replace=False))
    cnt = d["hitcnt"].astype(np.int64)
    off = np.concatenate(([0], np.cumsum(cnt)))
    keep_blocks = np.concatenate([np.arange(off[i], off[i + 1]) for i in sel])
    sub = {k: (v[sel] if (isinstance(v, np.ndarray) and v.ndim >= 1
                          and len(v) == n and k not in ("hitcls", "hitamp2"))
               else v) for k, v in d.items()}
    sub["hitcls"] = d["hitcls"][keep_blocks]
    sub["hitamp2"] = d["hitamp2"][keep_blocks]
    sub["hitcnt"] = cnt[sel]
    z = sub["z"]
    print(f"{len(z)} of {n} tracks")

    bank, _ = hc.build_cf_bank(nfiles=args.bank_nfiles)
    a0 = types.SimpleNamespace(khit=0.0, kms=0.0, kioni=0.0, hitmode="class")
    Shit = {"class": ctr.hit_exponent(sub, a0, bank),
            "gauss": -0.5 * sub["vgf"][:, None] * ctr.TG[None, :] ** 2}
    Sio = sub["Sio_re"] + 1j * sub["Sio_im"]
    Ed = np.array([np.exp(-u * z ** 2).mean() for u in args.probes])
    err = np.array([np.exp(-u * z ** 2).std() / np.sqrt(len(z)) for u in args.probes])

    def fom(arm, kms):
        phi = np.exp(Shit[arm] + np.exp(kms) * sub["Sms"] + Sio)
        Em = np.array([ctr.weier(phi, u).mean() for u in args.probes])
        return float((((Ed - Em) / err) ** 2).sum()), Ed - Em

    grid = np.linspace(args.kms[0], args.kms[1], int(args.kms[2]))
    print(f"\n  {'kms':>7} {'gauss FOM':>11} {'class FOM':>11}")
    best = {}
    for arm in ("gauss", "class"):
        vals = [fom(arm, k)[0] for k in grid]
        best[arm] = (grid[int(np.argmin(vals))], min(vals), vals)
    for i, k in enumerate(grid):
        star = "".join(" <-" + a[0] if abs(best[a][0] - k) < 1e-9 else ""
                       for a in ("gauss", "class"))
        print(f"  {k:+7.4f} {best['gauss'][2][i]:11.2f} {best['class'][2][i]:11.2f}{star}")
    print()
    for arm in ("gauss", "class"):
        k, f, _ = best[arm]
        _, resid = fom(arm, k)
        print(f"  {arm:<6} best kms {k:+.4f}   FOM {f:8.2f}   "
              f"residuals " + " ".join(f"{r:+.5f}" for r in resid))
    dk = best["class"][0] - best["gauss"][0]
    df = best["class"][1] - best["gauss"][1]
    print(f"\n  class - gauss:  d(kms) = {dk:+.4f}   d(FOM) = {df:+.2f}")
    print("  d(FOM) < 0 with d(kms) > 0  =>  the hit term was MASKING an MS deficit")
    print("  d(FOM) > 0                  =>  the hit term itself is the problem")


if __name__ == "__main__":
    main()
