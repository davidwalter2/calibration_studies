#!/usr/bin/env python3
"""The three gates of the nuclear-elastic family on the K_S mass CF.

(i)   OFF == the in-maker composition.  Every column the standard pairs cache
      carries (the maker's cfmass_* exponents, vgf, sigma, z, the truth join)
      is BIT-IDENTICAL, row by row, to the parent production's cache -- so with
      the family switched off the composed CF is the in-maker one, exactly.
      With NUCEL_CHANNEL=0 `ks_nucel_cf.py` adds identically zero exponents.
(ii)  variance bookkeeping.  The Gaussian + influence closure of the MASS
      functional, vgf + sum_b v_b(MS, ioni)/sigma_m^2 = 1, is re-evaluated from
      the records (the family does not enter it: it is a non-Gaussian EXPONENT,
      not a share of the fit's sigma_m); the family's own second cumulant
      v_nuc = sum_s N_s w^2 <theta^2>/2 must be > 0 on every candidate with
      material, and the CF's total second cumulant is then 1 + v_nuc in
      units of sigma_m^2 (sigma_m stays the fit's Gaussian width).
(iii) the per-candidate density with the family integrates to 1 and is
      positive: phi(0) = 1 exactly (S_nuc(0) = N (g(0) - 1) = 0), and the
      inverted density over +-R is checked numerically.

usage: python3 gates.py --parent kspairs_all.npz --cache kspairs_nucel.npz \
          --prod <steprec prod> [--ntask 40] [--ncand 3000]
"""
import argparse
import glob
import os
import sys

import numpy as np
import uproot
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))


def gate_i(parent, cache):
    a = np.load(cache, allow_pickle=True)
    b = np.load(parent, allow_pickle=True)
    n = len(b['z'])
    if len(a['z']) != n:
        return False, f'row count {len(a["z"])} vs {n}'
    bad = []
    for k in b.files:
        if k in ('cf_source',):
            continue
        x, y = a[k], b[k]
        if x.dtype.kind == 'f':
            same = x.shape == y.shape and np.array_equal(x.view(np.uint8), y.view(np.uint8))
        else:
            same = np.array_equal(x, y)
        if not same:
            bad.append(k)
    return not bad, f'{len(b.files) - 1} columns x {n} rows compared; differing: {bad}'


def gate_ii_records(prod, ntask):
    tasks = sorted(glob.glob(os.path.join(prod, 'task_[0-9]*')))[:ntask]
    worst, nc = 0.0, 0
    for td in tasks:
        for fn in sorted(glob.glob(os.path.join(td, 'globalcor_ks_*.root'))):
            f = uproot.open(fn)
            if 'tree' not in f:
                continue
            pt = f['runtree']['parmtype'].array(library='np')
            a = f['tree'].arrays(['reseigidx', 'resinfvarv', 'resinfcov', 'cfmass_vgf',
                                  'Jpsi_sigmamass', 'Jpsi_massvms', 'Jpsi_massvioni'],
                                 library='np')
            for ic in range(len(a['resinfcov'])):
                s2 = float(a['Jpsi_sigmamass'][ic]) ** 2
                if not (s2 > 0):
                    continue
                fam = pt[a['reseigidx'][ic]]
                vb = np.asarray(a['resinfvarv'][ic], dtype=np.float64)
                vms = vb[fam == 10].sum() / s2
                vio = vb[fam == 11].sum() / s2
                c1 = abs(float(a['cfmass_vgf'][ic]) + vms + vio - 1.0)
                c2 = abs(float(a['cfmass_vgf'][ic]) + float(a['Jpsi_massvms'][ic])
                         + float(a['Jpsi_massvioni'][ic]) - 1.0)
                worst = max(worst, c1, c2)
                nc += 1
    return worst, nc


def gate_iii(cache, ncand, R=40.0):
    d = np.load(cache, allow_pickle=True)
    n = len(d['z'])
    rng = np.random.default_rng(1)
    idx = np.sort(rng.choice(n, size=min(ncand, n), replace=False))
    tg = d['tgrid'].astype(np.float64)
    tf = np.linspace(0, tg[-1], (len(tg) - 1) * 4 + 1)
    S = (-0.5 * d['vgf'][idx][:, None] * tg[None, :] ** 2 + d['Sms'][idx]
         + d['Sio_re'][idx] + 1j * d['Sio_im'][idx]
         + d['Srad_re'][idx] + 1j * d['Srad_im'][idx]
         + d['Snuc_ang'][idx] + d['Snuc_rec_re'][idx] + 1j * d['Snuc_rec_im'][idx]).astype(np.complex128)
    s0 = np.abs(S[:, 0]).max()
    Sf = CubicSpline(tg, S.real, axis=1)(tf) + 1j * CubicSpline(tg, S.imag, axis=1)(tf)
    phi = np.exp(Sf)
    w = np.gradient(tf)
    w[0] *= .5
    w[-1] *= .5
    zg = np.linspace(-R, R, 4001)
    integ, minrel = [], []
    for i0 in range(0, len(idx), 256):
        ph = phi[i0:i0 + 256]
        arg = tf[None, :] * zg[:, None]
        p = (np.cos(arg) @ (ph.real * w).T + np.sin(arg) @ (ph.imag * w).T) / np.pi
        integ.append(np.trapezoid(p, zg, axis=0))
        minrel.append(p.min(0) / p.max(0))
    integ = np.concatenate(integ)
    minrel = np.concatenate(minrel)
    return s0, integ, minrel, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parent', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--prod', required=True)
    ap.add_argument('--ntask', type=int, default=40)
    ap.add_argument('--ncand', type=int, default=3000)
    args = ap.parse_args()
    ok, msg = gate_i(args.parent, args.cache)
    print(f'GATE (i)  OFF == in-maker composition: {"PASS" if ok else "FAIL"} -- {msg}')
    worst, nc = gate_ii_records(args.prod, args.ntask)
    print(f'GATE (ii) vgf + sum v_(MS,ioni)/sigma^2 = 1 on {nc} candidates from the records: '
          f'max |closure - 1| = {worst:.2e}')
    d = np.load(args.cache, allow_pickle=True)
    v = d['nuc_v']
    N = d['nuc_N']
    print(f'          v_nuc > 0 on {int((v > 0).sum())}/{len(v)} candidates (N > 0 on '
          f'{int((N > 0).sum())}); median {np.median(v):.3f}, q10/q90 '
          f'{np.quantile(v, .1):.3f}/{np.quantile(v, .9):.3f} sigma_m^2; the CF second '
          f'cumulant is 1 + v_nuc (+ v_rec, median {np.median(d["nuc_vrec"]):.2e})')
    # the variance a Gaussian of the SAME N would carry is dominated by the rare
    # large angles: quote the share of candidates whose v_nuc exceeds 1 as well
    print(f'          v_nuc > 1 on {100*np.mean(v > 1):.1f} % of candidates -- the '
          f'channel is a tail, its second cumulant is not a width')
    s0, integ, minrel, _ = gate_iii(args.cache, args.ncand)
    print(f'GATE (iii) max|S_total(0)| = {s0:.1e}; integral over |z| < 40 on {len(integ)} '
          f'candidates: min {integ.min():.6f} median {np.median(integ):.6f} max '
          f'{integ.max():.6f}; min density / max density: min {minrel.min():.2e} '
          f'(q01 {np.quantile(minrel, .01):.2e})')


if __name__ == '__main__':
    main()
