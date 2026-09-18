#!/usr/bin/env python3
"""The K_S closure table, OLD code beside FIXED code.

Both sides are read exactly as `ks_table.py` reads one of them -- the
`rabbit_to_json.py` output of each fit plus the lever arm <f> of the cache the
fit ran on -- so the two columns are the same quantity and the difference is
the CVH fix and nothing else.  A fit whose EDM is above the campaign tolerance
is flagged and must not be quoted.

  usage: ks_compare_closure.py --old runs --new runs/fixed [--order ...]
"""
import argparse
import json
import os

import numpy as np

EDM_TOL = 1e-3
M_PI = 0.13957039


def lever(cache):
    if not os.path.exists(cache):
        return float('nan'), 0
    with np.load(cache, allow_pickle=True) as d:
        if 'ks_fmom' in d.files:
            f = np.asarray(d['ks_fmom'], dtype=np.float64)
        else:
            ep = np.sqrt(np.asarray(d['ks_pgenp']) ** 2 + M_PI ** 2)
            em = np.sqrt(np.asarray(d['ks_pgenm']) ** 2 + M_PI ** 2)
            f = 1.0 - M_PI ** 2 * (2.0 + ep / em + em / ep) / np.asarray(d['eta']) ** 2
        w = 1.0 / np.asarray(d['sigma'], dtype=np.float64) ** 2
        return float((w * f).sum() / w.sum()), len(f)


def cache_tag(tag):
    if tag.startswith('all') or tag.startswith('af_'):
        return 'all' if tag.startswith('all') else 'ares'
    if tag.startswith('s_'):
        return {'s_mtight': 'matchtight', 's_mloose': 'matchloose'}.get(tag, 'all')
    return tag


def side(runs, tag):
    fn = os.path.join(runs, 'results', f'{tag}.json')
    if not os.path.exists(fn):
        return None
    r = json.load(open(fn))
    if 'params' not in r or 'alpha' not in r['params']:
        return None
    i = r['params'].index('alpha')
    fw, n = lever(os.path.join(runs, f'kspairs_{cache_tag(tag)}.npz'))
    return dict(alpha=r['fitted'][i], err=r['err'][i],
                edm=r.get('edmval', float('nan')), fw=fw, n=n)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--old', default=os.path.join(here, 'runs'))
    ap.add_argument('--new', default=os.path.join(here, 'runs', 'fixed'))
    ap.add_argument('--order', nargs='*', default=[
        'all_naive', 'all_ares', 'all', 'af_truth',
        'fromb', 'fromb0', 'prompt',
        'r0_2', 'r2_4', 'r4_10', 'r10_60', 'plo', 'pmid', 'phi',
        's_resid3', 's_resid5', 's_bkg', 's_chi2', 's_mtight', 's_mloose'])
    args = ap.parse_args()

    hdr = (f'{"tag":<11} {"n old":>7} {"n new":>7} | {"eps OLD [1e-3]":>21} | '
           f'{"eps FIXED [1e-3]":>21} | {"fixed-old":>17}  EDM')
    print(hdr)
    print('-' * len(hdr))
    bad = []
    for tag in args.order:
        a, b = side(args.old, tag), side(args.new, tag)
        if a is None and b is None:
            continue
        def fmt(s):
            if s is None:
                return f'{"--":>21}'
            return f'{s["alpha"]/s["fw"]:+9.4f} +- {s["err"]/s["fw"]:8.4f}'
        d = ''
        if a is not None and b is not None:
            # the two fits share candidates, so the difference is correlated;
            # the quoted error is the single-fit one, an upper bound on sigma(diff)
            de = b['alpha'] / b['fw'] - a['alpha'] / a['fw']
            d = f'{de:+9.4f}'
        e = ' '.join(f'{s["edm"]:.0e}' if s else '-' for s in (a, b))
        for nm, s in (('old', a), ('new', b)):
            if s is not None and not (s['edm'] < EDM_TOL):
                bad.append(f'{tag}/{nm}')
        print(f'{tag:<11} {a["n"] if a else 0:>7} {b["n"] if b else 0:>7} | '
              f'{fmt(a)} | {fmt(b)} | {d:>17}  {e}')
    if bad:
        print(f'\nNOT CONVERGED (must not be quoted): {bad}')


if __name__ == '__main__':
    main()
