#!/usr/bin/env python3
"""Candidate-by-candidate old-code vs fixed-code comparison of the K_S refit.

Reproduces the defect-(d) impact table of `resolution/DEFECTS_STATE.md` on a
production chunk: the shift of the fitted mass, of the per-candidate mass
resolution, and of the fitted curvatures.  Candidates are matched on
(run, lumi, event) plus the ordinal within the event, exactly as
`resolution/msksign/compare_arms.py` does; the two arms come from the same
input list so the ordinals are the same candidates.

  usage: smoke_compare.py <old task dir|file> <new task dir|file>
"""
import argparse
import glob
import os
import sys

import numpy as np
import uproot

WANT = ["run", "lumi", "event", "Jpsi_mass", "Jpsi_sigmamass", "chisqval",
        "ndof", "niter", "edmval", "Jpsi_qoprefplus", "Jpsi_qoprefminus",
        "Jpsi_fang"]


def files(pat):
    out = []
    for part in pat.split(','):
        part = part.strip()
        fs = sorted(glob.glob(os.path.join(part, '*.root'))) if os.path.isdir(part) else [part]
        out += [f for f in fs if os.path.getsize(f) > 0]
    if not out:
        sys.exit(f'no ROOT file under {pat}')
    return out


def load(paths):
    parts = []
    for p in paths:
        t = uproot.open(p)['tree']
        have = [x for x in WANT if x in t.keys()]
        parts.append(t.arrays(have, library='np'))
    keys = set(parts[0])
    for p in parts[1:]:
        keys &= set(p)
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def ordinals(d):
    key = np.stack([d['run'], d['lumi'], d['event']], axis=1)
    seen, out = {}, np.zeros(len(key), dtype=np.int64)
    for i, k in enumerate(map(tuple, key)):
        out[i] = seen.get(k, 0)
        seen[k] = out[i] + 1
    return [(*k, o) for k, o in zip(map(tuple, key), out)]


def summarise(name, rel):
    rel = rel[np.isfinite(rel)]
    if rel.size == 0:
        print(f'{name:<28} (empty)')
        return
    a = np.abs(rel)
    print(f'{name:<28} n {rel.size:6d}  mean {rel.mean():+.3e} +- '
          f'{rel.std(ddof=1)/np.sqrt(rel.size):.3e}  median {np.median(rel):+.3e}  '
          f'med|.| {np.median(a):.3e}  p95|.| {np.percentile(a, 95):.3e}  '
          f'>1e-4 {100.0*(a > 1e-4).mean():5.2f} %')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('old')
    ap.add_argument('new')
    args = ap.parse_args()

    A, B = load(files(args.old)), load(files(args.new))
    oa, ob = ordinals(A), ordinals(B)
    ia = {k: i for i, k in enumerate(oa)}
    pairs = [(ia[k], j) for j, k in enumerate(ob) if k in ia]
    si = np.array([p[0] for p in pairs])
    sj = np.array([p[1] for p in pairs])
    print(f'old {len(oa)} candidates, new {len(ob)}, matched {len(si)}')
    print()

    ma, mb = A['Jpsi_mass'][si].astype(float), B['Jpsi_mass'][sj].astype(float)
    summarise('Delta m / m', (mb - ma) / ma)
    sa, sb = A['Jpsi_sigmamass'][si].astype(float), B['Jpsi_sigmamass'][sj].astype(float)
    summarise('Delta sigma_m / sigma_m', (sb - sa) / sa)
    ok = sa > 0
    z = (mb - ma)[ok] / sa[ok]
    print(f'{"Delta m / sigma_m":<28} n {z.size:6d}  mean {z.mean():+.3e} +- '
          f'{z.std(ddof=1)/np.sqrt(z.size):.3e}  med|.| {np.median(np.abs(z)):.3e}')
    for leg in ('plus', 'minus'):
        k = f'Jpsi_qopref{leg}'
        if k in A and k in B:
            x, y = A[k][si].astype(float), B[k][sj].astype(float)
            summarise(f'Delta q/p ({leg})', (y - x) / x)
    for k in ('chisqval', 'Jpsi_fang'):
        if k in A and k in B:
            x, y = A[k][si].astype(float), B[k][sj].astype(float)
            summarise(f'Delta {k}', (y - x) / np.where(x != 0, x, np.nan))
    print()
    for k in ('niter', 'ndof'):
        if k in A and k in B:
            x, y = A[k][si].astype(float), B[k][sj].astype(float)
            print(f'{k:<28} old mean {x.mean():7.3f} median {np.median(x):6.1f} '
                  f'max {x.max():6.0f}   new mean {y.mean():7.3f} '
                  f'median {np.median(y):6.1f} max {y.max():6.0f}')


if __name__ == '__main__':
    main()
