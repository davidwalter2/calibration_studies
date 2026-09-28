#!/usr/bin/env python3
"""The three views of one nuclear-elastic pairs cache that the fits compare.

  base   the cache WITHOUT any Snuc key -- must be the parent cache, row for row
  nuc    Snuc_re = S_ang + S_rec + S_jnt (re), Snuc_im = S_rec + S_jnt (im)
                                                recoil and joint ON (default)
  nucnr  Snuc_re = S_ang                        recoil (and joint) OFF
  (below, "recoil" is the recoil and joint parts, which go with each collision)
  nucfirst  the first-block-only family (Snuc_first), recoil x N_first/N
  nucsurv   the survival-weighted family (Snuc_surv), recoil x N_surv/N
  nucfirstmig  the first-block family times its migration factor, recoil x N_firstmig/N
  nucan1/2  the anisotropic survival family, response along / independent of the
            block's resolved direction (Snuc_an1/2), recoil x N_an/N

plus the measured sigma-artefact slope `ares_truth` joined from the parent's
`kspairs_ares.npz` (rows asserted identical on run/lumi/event and sigma).

usage: python3 make_variants.py --cache kspairs_nucel.npz --ares kspairs_ares.npz --outdir <dir>
"""
import argparse
import os

import numpy as np

NUCKEYS = ('Snuc_ang', 'Snuc_rec_re', 'Snuc_rec_im', 'Snuc_jnt_re', 'Snuc_jnt_im')


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
    # the parts that go with each collision: the recoil and the joint term
    rre = (d['Snuc_rec_re'] + d['Snuc_jnt_re']).astype(np.float64)
    rim = (d['Snuc_rec_im'] + d['Snuc_jnt_im']).astype(np.float64)
    nuc = dict(base, Snuc_re=(ang + rre).astype(np.float32), Snuc_im=rim.astype(np.float32))
    nucnr = dict(base, Snuc_re=ang.astype(np.float32))
    views = [('base', base), ('nuc', nuc), ('nucnr', nucnr)]
    # the first-block-only and the survival-weighted families, the recoil and
    # joint parts scaled by the collision fraction each keeps
    N = np.maximum(d['nuc_N'].astype(np.float64), 1e-300)[:, None]
    for tag, sk, nk in (('nucfirst', 'Snuc_first', 'nuc_Nfirst'), ('nucsurv', 'Snuc_surv', 'nuc_Nsurv'),
                        ('nucfirstmig', 'Snuc_firstmig', 'nuc_Nfirstmig'),
                        ('nucan1', 'Snuc_an1', 'nuc_Nan'), ('nucan2', 'Snuc_an2', 'nuc_Nan')):
        if sk in d and nk in d and np.any(d[nk] > 0):
            fr = d[nk].astype(np.float64)[:, None] / N
            views.append((tag, dict(base, Snuc_re=(d[sk].astype(np.float64) + fr * rre).astype(np.float32),
                                    Snuc_im=(fr * rim).astype(np.float32))))
    os.makedirs(args.outdir, exist_ok=True)
    for tag, v in views:
        fn = os.path.join(args.outdir, f'kspairs_{tag}_all.npz')
        np.savez_compressed(fn, **v)
        print('wrote', fn, len(v['z']))


if __name__ == '__main__':
    main()
