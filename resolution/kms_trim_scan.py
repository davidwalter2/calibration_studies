#!/usr/bin/env python3
"""Is the residual k_ms model shape, or is it the SELECTION?

The historical trimming-dependence -- MS +0.007 at chi2/hit < 10 against -0.073
at < 3 -- was never separable from a model error, because the cached closure
inputs carried no track-quality variable. They do now (chisqval, ndof,
nValidHits, trackPt, genPt added 2026-08-08), so k_ms can be scanned against
the cut instead of quoted at one arbitrary value.

The samples split the question by construction, which is what makes this
decisive rather than another scan:

    sample        chi2/hit p99   p99.9    max      frac > 3
    mugun_ul16        2.06        3.35      4.9      0.24 %
    mugun_lowpt       2.35        4.57     12.8      0.39 %
    kaon_ul16         3.53       69.2     697        1.16 %
    pion_ul16         7.40       83.1        1.8e8   2.04 %
    proton_ul16      16.9        90.6      679       3.01 %

The MUONS have no catastrophic population at all -- a cut at 3 keeps 99.6-99.8%
-- so for them the scan MUST be flat, and a flat scan proves their residual
+0.010/+0.013 is model shape and not selection. The HADRONS do have one, and
its size orders exactly as k_ms does (p > pi > K > mu), so if their deficit is
carried by those tracks the scan must collapse.

A flat muon scan plus a collapsing hadron scan is the clean result. Anything
else -- notably a muon scan that moves -- would mean selection is entangled with
the model residual and the +0.013 cannot be quoted as a model number.

usage:
  python kms_trim_scan.py [--tag mugun_ul16 ...] [--probes 0.05 0.2 1.0 2.0]
                          [--cuts 1.5 2 3 5 10 1e9] [--max-tracks N]
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

TAGS = ("mugun_ul16", "mugun_lowpt", "kaon_ul16", "pion_ul16", "proton_ul16")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", nargs="+", default=list(TAGS))
    p.add_argument("--suffix", default="_sel",
                   help="cache suffix; _sel are the caches that carry the "
                        "selection variables")
    p.add_argument("--probes", nargs="+", type=float, default=[0.05, 0.2, 1.0, 2.0])
    p.add_argument("--cuts", nargs="+", type=float,
                   default=[1.5, 2.0, 3.0, 5.0, 10.0, 1e9],
                   help="chi2/nValidHits upper cuts; 1e9 = no cut")
    p.add_argument("--max-tracks", type=int, default=0,
                   help="0 = all. If subsampling, rows are SHUFFLED: taking "
                        "the first N rows takes the first N/2000 files of a "
                        "sorted list, which is not a random draw and gave "
                        "+0.0012 where the full sample gives +0.0129.")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def load_model_module():
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "cft", os.path.join(here, "cf_track_resolution.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["cft"] = m
    spec.loader.exec_module(m)
    return m


class _Scale:
    khit = 0.
    kms = 0.
    kioni = 0.


def solve_kms(m, d, u, lo=-1.5, hi=1.5, niter=24):
    """Bisect on the scale that closes <exp(-u z^2)>.

    Bracket is wider than the +-0.6 used elsewhere: the hadron samples reach
    +0.6 at small u and a bracket that clips would silently report its own
    endpoint as the answer.
    """
    z = d["z"]
    ed = np.exp(-u * z ** 2).mean()
    for _ in range(niter):
        mid = 0.5 * (lo + hi)
        s = _Scale()
        s.kms = mid
        if ed - m.weier(m.model_phi(d, s), u).mean() > 0:
            hi = mid
        else:
            lo = mid
    k = 0.5 * (lo + hi)
    # flag a solution pinned at the bracket -- it is not a root
    return k, (abs(k - 1.5) < 1e-3 or abs(k + 1.5) < 1e-3)


def main():
    args = parse_args()
    m = load_model_module()
    rng = np.random.default_rng(args.seed)
    here = os.path.dirname(os.path.abspath(__file__))

    for tag in args.tag:
        path = os.path.join(here, "runs", f"cf_trackres_{tag}{args.suffix}.npz")
        if not os.path.exists(path):
            print(f"\n=== {tag}: cache missing ({path})")
            continue
        src = np.load(path)
        # Check the key list BEFORE touching any array. np.load on an npz is
        # lazy but every __getitem__ decompresses, and Sms alone is ~540 MB per
        # sample -- reaching this guard after `len(src["z"])` made the
        # "missing variables" message take minutes instead of milliseconds.
        if "chisqval" not in src.files:
            print(f"\n=== {tag}: cache has no selection variables "
                  f"(keys: {list(src.files)}) -- re-extract with _sel")
            continue
        n = len(src["z"])
        idx = np.arange(n)
        if args.max_tracks and args.max_tracks < n:
            idx = rng.choice(n, args.max_tracks, replace=False)
            idx.sort()
        d_all = {k: (src[k][idx] if src[k].ndim and src[k].shape[0] == n else src[k])
                 for k in src.files}
        r = d_all["chisqval"] / np.maximum(d_all["nvalidhits"], 1.)

        print(f"\n=== {tag}   ({len(idx)} tracks, "
              f"chi2/hit median {np.median(r):.2f}, max {r.max():.4g})")
        print(f"  {'cut':>8} {'kept':>8} {'%':>7}   "
              + "  ".join(f"u={u:<6g}" for u in args.probes) + "     mean")
        for c in args.cuts:
            sel = r < c
            if sel.sum() < 1000:
                print(f"  {c:8.4g} {int(sel.sum()):8d} (too few)")
                continue
            d = {k: (v[sel] if getattr(v, "ndim", 0) and v.shape[0] == len(sel) else v)
                 for k, v in d_all.items()}
            ks, pinned = [], False
            for u in args.probes:
                k, p = solve_kms(m, d, u)
                ks.append(k)
                pinned |= p
            flag = "  <-- PINNED AT BRACKET" if pinned else ""
            print(f"  {c:8.4g} {int(sel.sum()):8d} {100. * sel.mean():6.2f}%   "
                  + "  ".join(f"{k:+.4f}" for k in ks)
                  + f"   {np.mean(ks):+.4f}{flag}")


if __name__ == "__main__":
    main()
