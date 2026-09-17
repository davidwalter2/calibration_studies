#!/usr/bin/env python3
"""Collect the K_S closure fits into one table.

Reads the `rabbit_to_json.py` outputs of `build_ks.sh` and prints the fitted
scale with its EDM, so a result that did not converge cannot be quoted by
accident (rabbit's EDM = 1/2 g^T H^-1 g; the campaign tolerance is 1e-3).

TWO SCALES, and they are not the same number.  `MassCFTerm`'s `alpha` is the
relative shift of the MASS: delta_i = mobs_i - m_ref * alpha * 1e-3.  A
MOMENTUM scale eps shifts the mass of candidate i by m_i f_i eps with

    f_i = d ln m / d ln p = [m^2 - m_pi^2 (2 + E1/E2 + E2/E1)] / m^2 ,

which is 1 only in the ultra-relativistic limit.  For J/psi -> mu mu it is
0.9953 -- which is why the J/psi closure never had to distinguish the two --
but for K_S -> pi pi it is 0.685 in a symmetric decay and lower in an
asymmetric one, because 2 m_pi / m_KS = 0.56.  The maximum-likelihood estimate
of a location shift weights candidates by their Fisher information, ~1/sigma^2,
so the momentum scale is

    eps = alpha / <f>_{1/sigma^2} .

Both are printed; `eps` is what compares with the J/psi closure.
"""
import argparse
import glob
import json
import os

import numpy as np

EDM_TOL = 1e-3
M_PI = 0.13957039


def lever(cache):
    """(<f> weighted by 1/sigma^2, <f> unweighted, n) of a pairs cache."""
    with np.load(cache, allow_pickle=True) as d:
        if 'ks_fmom' in d.files:
            f = np.asarray(d['ks_fmom'], dtype=np.float64)
        else:
            ep = np.sqrt(np.asarray(d['ks_pgenp']) ** 2 + M_PI ** 2)
            em = np.sqrt(np.asarray(d['ks_pgenm']) ** 2 + M_PI ** 2)
            f = 1.0 - M_PI ** 2 * (2.0 + ep / em + em / ep) / np.asarray(d['eta']) ** 2
        w = 1.0 / np.asarray(d['sigma'], dtype=np.float64) ** 2
        return float((w * f).sum() / w.sum()), float(f.mean()), len(f)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument('--results', default=os.path.join(here, 'runs', 'results'))
    ap.add_argument('--runs', default=os.path.join(here, 'runs'))
    ap.add_argument('--order', nargs='*', default=None)
    args = ap.parse_args()

    rows = []
    for fn in sorted(glob.glob(os.path.join(args.results, '*.json'))):
        tag = os.path.splitext(os.path.basename(fn))[0]
        try:
            r = json.load(open(fn))
        except Exception:
            continue
        if 'params' not in r or 'alpha' not in r['params']:
            continue
        i = r['params'].index('alpha')
        # the ladder tags all_naive / all_ares run on the inclusive cache
        cache_tag = 'all' if tag.startswith('all') else tag
        cache = os.path.join(args.runs, f'kspairs_{cache_tag}.npz')
        fw = fu = float('nan')
        n = 0
        if os.path.exists(cache):
            fw, fu, n = lever(cache)
        rows.append((tag, r['fitted'][i], r['err'][i], r.get('edmval', float('nan')),
                     r.get('nllvalreduced', float('nan')), fw, fu, n))
    if args.order:
        key = {t: i for i, t in enumerate(args.order)}
        rows.sort(key=lambda x: key.get(x[0], 999))
    print(f'{"tag":12s} {"n":>8s} {"alpha_mass [1e-3]":>22s} {"<f>":>6s} '
          f'{"eps_mom [1e-3]":>20s} {"EDM":>9s} conv')
    for tag, a, e, edm, nll, fw, fu, n in rows:
        conv = 'OK ' if edm < EDM_TOL else 'NO!'
        em_, ee = (a / fw, e / fw) if fw == fw else (float('nan'),) * 2
        print(f'{tag:12s} {n:8d} {a:+10.4f} +- {e:7.4f} {fw:6.3f} '
              f'{em_:+10.4f} +- {ee:7.4f} {edm:9.2e}  {conv}')
    bad = [t for t, _, _, edm, _, _, _, _ in rows if not (edm < EDM_TOL)]
    if bad:
        print(f'\nNOT CONVERGED (must not be quoted): {bad}')
    print('\nalpha_mass = the relative shift of the K_S MASS the term fits;\n'
          'eps_mom    = the momentum scale = alpha_mass / <f>, '
          '<f> = d ln m / d ln p weighted by 1/sigma^2\n'
          '             (J/psi -> mu mu has f = 0.9953, so its alpha IS its '
          'momentum scale).')


if __name__ == '__main__':
    main()
