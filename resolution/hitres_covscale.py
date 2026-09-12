#!/usr/bin/env python3
"""Does correcting the hit covariance move the fit TOWARD gen truth?

The pull study leaves two candidate corrections that disagree in sign for both
detectors:

    core-matching      pixel x0.874   strip x1.166    (makes the 68 % width 1)
    variance-matching  pixel x1.086   strip x0.957    (makes the outlier-
                                                       protected 2nd moment 1)

Only truth can choose. Each arm is the SAME 160-file nominal-fit production
with one number changed, so everything is paired on (task, run, lumi, event,
genPt, charge) and the common part cancels. Quoting a per-track shift without
that pairing is the failure mode catalogued in NOTES_CGFFIT sections 86-87.

Reported per arm, all paired and all bootstrapped over TRACKS (each replica
resamples tracks and recomputes both arms):

  d(bias)     change in the trim-mean of (q/p_fit - q/p_gen)/|q/p_gen|.
              A weight cannot move a mean in the Gaussian limit, so a
              significant shift here is itself informative.
  dVar/Var    = [2 Cov(r_ref, d) + Var(d)] / Var(r_ref), the exact
              decomposition of the variance change into the part correlated
              with the reference residual and the noise the change adds.
              NEGATIVE = the fit got closer to truth.

usage:
  python hitres_covscale.py --base mugun_lowpt --arms _cscore _csvar
"""
import argparse
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hitres_nulltest import load_keyed                            # noqa: E402


def paired(ref, alt, nboot, rng, trim=5.0):
    """Paired statistics on a common track set, 5-sigma trimmed on the REFERENCE."""
    common = sorted(set(ref) & set(alt))
    a = np.array([ref[k][0] for k in common])
    b = np.array([alt[k][0] for k in common])
    s = 0.5 * np.diff(np.percentile(a, [15.865, 84.135]))[0]
    keep = np.abs(a - np.median(a)) < trim * s
    a, b = a[keep], b[keep]
    d = b - a
    n = len(a)

    def tm(v):
        v = np.sort(v)
        lo, hi = int(0.1 * len(v)), max(int(0.9 * len(v)), int(0.1 * len(v)) + 1)
        return float(v[lo:hi].mean())

    def stats(idx):
        aa, dd = a[idx], d[idx]
        va = aa.var()
        dvar = 2 * np.cov(aa, dd)[0, 1] + dd.var()
        return tm(aa + dd) - tm(aa), dvar / va, np.median(aa + dd) - np.median(aa)

    c = stats(np.arange(n))
    bs = np.array([stats(rng.integers(0, n, n)) for _ in range(nboot)])
    return dict(n=n, ncommon=len(common),
                dbias=c[0], dbias_err=bs[:, 0].std(),
                dvar=c[1], dvar_err=bs[:, 1].std(),
                dmed=c[2], dmed_err=bs[:, 2].std(),
                rms_delta=float(np.sqrt((d ** 2).mean())),
                sigma_ref=float(0.5 * np.diff(np.percentile(a, [15.865, 84.135]))[0]))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="mugun_lowpt")
    p.add_argument("--arms", nargs="+", default=["_cscore", "_csvar"])
    p.add_argument("--subdir", default="resolution_trackres")
    p.add_argument("--nfiles", type=int, default=0)
    p.add_argument("--nboot", type=int, default=300)
    args = p.parse_args()
    rng = np.random.default_rng(20260829)

    ref, nref = load_keyed(args.base, args.nfiles, args.subdir)
    print(f"reference {args.subdir}_{args.base}: {nref} files, {len(ref)} tracks")
    print(f"\n  {'arm':<10} {'n':>8} {'d(trim-mean) x1e-4':>22} "
          f"{'d(median) x1e-4':>20} {'dVar/Var':>22} {'rms(delta) x1e-4':>18}")
    for suf in args.arms:
        alt, nalt = load_keyed(f"{args.base}{suf}", args.nfiles, args.subdir)
        if not alt:
            print(f"  {suf:<10} (missing)")
            continue
        r = paired(ref, alt, args.nboot, rng)
        sig_b = abs(r["dbias"]) / max(r["dbias_err"], 1e-30)
        sig_v = abs(r["dvar"]) / max(r["dvar_err"], 1e-30)
        print(f"  {suf:<10} {r['n']:>8} "
              f"{r['dbias']*1e4:+9.3f} +-{r['dbias_err']*1e4:6.3f} ({sig_b:4.1f}s) "
              f"{r['dmed']*1e4:+8.3f} +-{r['dmed_err']*1e4:5.3f} "
              f"{r['dvar']*100:+9.4f} +-{r['dvar_err']*100:7.4f} % ({sig_v:4.1f}s) "
              f"{r['rms_delta']*1e4:12.3f}")
    print("\n  dVar/Var NEGATIVE = the arm is CLOSER to truth.")
    print("  A shift below the fit's own response floor is numerical, not")
    print("  physical: a 1e-16 perturbation of the noise rows already produces")
    print("  rms 2.4e-7 on q/p (NOTES_CGFFIT).")


if __name__ == "__main__":
    main()
