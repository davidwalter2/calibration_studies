#!/usr/bin/env python3
"""The three views of one nuclear-elastic pairs cache that the fits compare.

  base   the cache WITHOUT any Snuc key -- must be the parent cache, row for row
  nuc    Snuc_re = S_ang + S_rec (re), Snuc_im = S_rec (im)   recoil ON (default)
  nucnr  Snuc_re = S_ang                                       recoil OFF

plus the measured sigma-artefact slope `ares_truth` joined from the parent's
`kspairs_ares.npz` (rows asserted identical on run/lumi/event and sigma).

usage: python3 make_variants.py --cache kspairs_nucel.npz --ares kspairs_ares.npz --outdir <dir>
"""
import argparse
import os

import numpy as np

NUCKEYS = ('Snuc_ang', 'Snuc_rec_re', 'Snuc_rec_im')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--ares', required=True)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()
    d = dict(np.load(args.cache, allow_pickle=True))
    r = np.load(args.ares, allow_pickle=True)
    for k in ('run', 'lumi', 'event', 'sigma', 'z'):
        if not np.array_equal(d[k], r[k]):
            raise SystemExit(f'{k}: the a_res cache is not row-aligned')
    d['ares_truth'] = r['ares_truth']
    d['ares_mom'] = r['ares_mom']
    base = {k: v for k, v in d.items() if k not in NUCKEYS}
    ang = d['Snuc_ang'].astype(np.float64)
    rre = d['Snuc_rec_re'].astype(np.float64)
    rim = d['Snuc_rec_im'].astype(np.float64)
    nuc = dict(base, Snuc_re=(ang + rre).astype(np.float32), Snuc_im=rim.astype(np.float32))
    nucnr = dict(base, Snuc_re=ang.astype(np.float32))
    os.makedirs(args.outdir, exist_ok=True)
    for tag, v in (('base', base), ('nuc', nuc), ('nucnr', nucnr)):
        fn = os.path.join(args.outdir, f'kspairs_{tag}_all.npz')
        np.savez_compressed(fn, **v)
        print('wrote', fn, len(v['z']))


if __name__ == '__main__':
    main()
