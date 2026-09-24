#!/usr/bin/env python3
"""Step-record re-production vs its parent, candidate by candidate.

Every candidate of the re-production must exist in the parent with the SAME
fit: the in-maker CF exponents, the mass, sigma and the fit chi2 bit-identical
(the only change is the raw-record export and the event selection), and the
records must be non-empty with the strides the file declares.

usage: python3 smoke_check.py --new <task dir or file glob> --parent <parent prod> [--tasks 0000 0001]
"""
import argparse
import glob
import os
import sys

import numpy as np
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import prodfiles  # noqa: E402

CMP = ['cfmass_ms', 'cfmass_del', 'cfmass_ioni_re', 'cfmass_ioni_im',
       'cfmass_rad_re', 'cfmass_rad_im', 'cfmass_vgf', 'cfmass_ok',
       'Jpsi_mass', 'Jpsi_sigmamass', 'chisqval', 'ndof', 'Muplus_pt', 'Muminus_pt',
       'Muplus_eta', 'Muminus_eta', 'Jpsi_x', 'Jpsi_y', 'Jpsi_z', 'resinfcov',
       'resinfvarv', 'reseigidx', 'Jpsi_covrefmom']
REC = ['msmoliv', 'msmoliidx', 'ioniurbanv', 'ioniurbanidx', 'radstepv', 'radstepidx',
       'resinfv']


def load(fn, br):
    t = uproot.open(fn)['tree']
    have = set(t.keys())
    br = [b for b in br if b in have]
    return t.arrays(['run', 'lumi', 'event'] + br, library='np'), have


def key(a, i):
    # (run, lumi, event, mass bits) -- several candidates share an event;
    # identify by the event plus the two pre-fit leg momenta is not available,
    # so match within the event by ORDER of appearance (the maker loops the
    # candidate collection in its stored order)
    return (int(a['run'][i]), int(a['lumi'][i]), int(a['event'][i]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--new', required=True)
    ap.add_argument('--parent', required=True)
    ap.add_argument('--tasks', nargs='+', default=['0000'])
    args = ap.parse_args()
    nbad = ncmp = 0
    for tk in args.tasks:
        for fn in sorted(glob.glob(os.path.join(args.new, f'task_{tk}', 'globalcor_ks_*.root'))):
            pf = os.path.join(args.parent, f'task_{tk}', os.path.basename(fn))
            a, have = load(fn, CMP + REC + list(prodfiles.stride_keys(
                uproot.open(fn)['tree'].keys(), ('msmoliv', 'ioniurbanv', 'radstepv'))))
            b, _ = load(pf, CMP)
            # group parent entries by event, in order
            from collections import defaultdict
            pe = defaultdict(list)
            for i in range(len(b['run'])):
                pe[key(b, i)].append(i)
            seen = defaultdict(int)
            print(f'{fn}: {len(a["run"])} candidates')
            for i in range(len(a['run'])):
                k = key(a, i)
                j = pe[k][seen[k]] if seen[k] < len(pe[k]) else None
                seen[k] += 1
                if j is None:
                    print('  no parent candidate for', k)
                    nbad += 1
                    continue
                ncmp += 1
                for c in CMP:
                    if c not in a or c not in b:
                        continue
                    x, y = np.atleast_1d(np.asarray(a[c][i])), np.atleast_1d(np.asarray(b[c][j]))
                    if x.shape != y.shape or not np.array_equal(x.view(np.uint8) if x.dtype.kind == 'f' else x,
                                                                y.view(np.uint8) if y.dtype.kind == 'f' else y):
                        d = np.max(np.abs(x.astype(float) - y.astype(float))) if x.shape == y.shape else 'shape'
                        print(f'  cand {i} {k}: {c} differs ({d})')
                        nbad += 1
            # the records
            for br in ('msmoliv', 'ioniurbanv', 'radstepv'):
                idxb = br.replace('v', 'idx', 1) if br != 'msmoliv' else 'msmoliidx'
                idxb = {'msmoliv': 'msmoliidx', 'ioniurbanv': 'ioniurbanidx',
                        'radstepv': 'radstepidx'}[br]
                empty = sum(1 for i in range(len(a['run'])) if len(a[br][i]) == 0)
                st = [prodfiles.entry_stride(a, br, i) for i in range(min(5, len(a['run'])))]
                nrec = [len(a[idxb][i]) for i in range(len(a['run']))]
                ok = all(len(a[br][i]) == nrec[i] * prodfiles.entry_stride(a, br, i)
                         for i in range(len(a['run'])))
                print(f'  {br}: stride {sorted(set(st))}, empty in {empty}/{len(a["run"])}, '
                      f'records/cand median {int(np.median(nrec))}, len==nrec*stride: {ok}')
    print(f'\ncompared {ncmp} candidates, {nbad} differences')
    return 1 if nbad else 0


if __name__ == '__main__':
    sys.exit(main())
