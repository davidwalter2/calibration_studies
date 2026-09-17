#!/usr/bin/env python3
"""Slice a K_S pairs cache by a boolean expression over its own columns.

The closure is measured inclusively and then in bins of the K_S DECAY RADIUS
(the point of a displaced test: the material model is exercised over a
different flight path in each bin) and of the pion momentum.  Rather than
re-reading the production for each bin, this writes a sliced copy of the cache
that `make_card.py` reads unchanged.

usage:
  python3 ks_subset.py --cache kspairs_all.npz --out kspairs_r0_2.npz \
      --cut 'ks_rdec < 2'
  python3 ks_subset.py --cache kspairs_all.npz --out kspairs_fromb.npz \
      --cut 'ks_fromb > 0'
"""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--cut', required=True,
                    help="python expression over the cache columns, e.g. "
                         "'(ks_rdec >= 2) & (ks_rdec < 6)'")
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    n = len(d['z'])
    env = {k: d[k] for k in d.files if getattr(d[k], 'shape', (0,))[:1] == (n,)}
    env['np'] = np
    mask = np.asarray(eval(args.cut, env), dtype=bool)  # noqa: S307 (own input)
    print(f'{args.cut}: keeps {int(mask.sum())} of {n} ({100*mask.mean():.2f} %)')
    out = {}
    for k in d.files:
        v = d[k]
        out[k] = v[mask] if (getattr(v, 'shape', (0,))[:1] == (n,)) else v
    np.savez_compressed(args.out, **out)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
