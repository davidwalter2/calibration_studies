#!/usr/bin/env python3
"""Collect the K_S closure fits into one table.

Reads the `rabbit_to_json.py` outputs of `build_ks.sh` and prints alpha +- sigma
with the EDM, so a result that did not converge cannot be quoted by accident
(rabbit's EDM = 1/2 g^T H^-1 g; the campaign tolerance is 1e-3).
"""
import argparse
import glob
import json
import os

EDM_TOL = 1e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=None,
                    help='directory of <tag>.json files (default runs/results)')
    ap.add_argument('--order', nargs='*', default=None)
    args = ap.parse_args()
    d = args.results or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'runs', 'results')
    rows = []
    for fn in sorted(glob.glob(os.path.join(d, '*.json'))):
        tag = os.path.splitext(os.path.basename(fn))[0]
        try:
            r = json.load(open(fn))
        except Exception:
            continue
        if 'params' not in r or 'alpha' not in r['params']:
            continue
        i = r['params'].index('alpha')
        rows.append((tag, r['fitted'][i], r['err'][i], r.get('edmval', float('nan')),
                     r.get('nllvalreduced', float('nan'))))
    if args.order:
        key = {t: i for i, t in enumerate(args.order)}
        rows.sort(key=lambda x: key.get(x[0], 999))
    print(f'{"tag":28s} {"alpha [1e-3]":>18s} {"EDM":>10s}  conv  {"NLL":>14s}')
    for tag, a, e, edm, nll in rows:
        conv = 'OK ' if edm < EDM_TOL else 'NO!'
        print(f'{tag:28s} {a:+9.4f} +- {e:6.4f} {edm:10.2e}  {conv}  {nll:14.3f}')
    bad = [t for t, _, _, edm, _ in rows if not (edm < EDM_TOL)]
    if bad:
        print(f'\nNOT CONVERGED (must not be quoted): {bad}')


if __name__ == '__main__':
    main()
