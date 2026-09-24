#!/usr/bin/env python3
"""Per-half event lists of the truth-matched K_S candidates of a finished
production, so a re-production with `exportStepRecords=True` processes only the
events that carry a truth-matched candidate (the records are ~430 kB per
candidate; the full 350 k candidates would be ~150 GB, the matched events a
fraction of that).

The selection is on EVENTS, never on candidates: every candidate of a selected
event is refit exactly as before, so the truth join downstream is unchanged and
the refit of a selected candidate is the same computation as in the parent
production (checked bit-for-bit on the smoke chunks).

usage: python3 make_evlists.py --prod <parent prod> --pairs <kspairs_all.npz> --out <dir>
writes <out>/ev_NNNN_{a,b}.txt, one `run:lumi:event` per line.
"""
import argparse
import glob
import os

import numpy as np
import uproot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prod', required=True)
    ap.add_argument('--pairs', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    d = np.load(args.pairs)
    sel = d['matched'] > 0
    key = set(zip(d['run'][sel].tolist(), d['lumi'][sel].tolist(), d['event'][sel].tolist()))
    print(f'{len(key)} distinct events carry a truth-matched candidate')
    os.makedirs(args.out, exist_ok=True)
    nfound, ncand = set(), 0
    for fn in sorted(glob.glob(os.path.join(args.prod, 'task_[0-9]*', 'globalcor_ks_[ab]_*.root'))):
        idx = os.path.basename(os.path.dirname(fn)).split('_')[-1]
        half = os.path.basename(fn).split('_')[2]
        with uproot.open(fn) as f:
            if 'tree' not in f:
                continue
            a = f['tree'].arrays(['run', 'lumi', 'event'], library='np')
        ev = sorted({k for k in zip(a['run'].tolist(), a['lumi'].tolist(), a['event'].tolist())
                     if k in key})
        ncand += sum(1 for k in zip(a['run'].tolist(), a['lumi'].tolist(), a['event'].tolist())
                     if k in key)
        nfound.update(ev)
        with open(os.path.join(args.out, f'ev_{idx}_{half}.txt'), 'w') as g:
            for r, l, e in ev:
                g.write(f'{r}:{l}:{e}\n')
    print(f'{len(nfound)} events located; {ncand} candidates (all, matched or not) '
          f'live in them')
    if len(nfound) != len(key):
        raise SystemExit(f'{len(key) - len(nfound)} matched events not located')


if __name__ == '__main__':
    main()
