#!/usr/bin/env python3
"""Nuclear-elastic kernel tables for the K_S daughters on the REAL geometry.

For every (species, Z, A, momentum) bucket the driver `nucel_g4driver` (the
species-correct Geant4 elastic model and cross-section dataset, exactly as
`cf_nucel_exact` uses it) is run and reduced to

  mu      [cm^2/g]   the mass attenuation Sigma/rho -- the Poisson rate per g/cm^2
  g(u)               E_theta[J0(u theta)], the projected single-collision CF
  h(v)               E_dE[exp(i v dE)], the recoil single-collision CF (dE in MeV)
  <theta>, <theta^2>, <dE>, <dE^2>  the moments (for the cumulant bookkeeping)

on a log-u / log-v grid.  Buckets are cached one npz per bucket under
`--cache`, so the table can be EXTENDED when a production shows a (Z, A) the
first scan did not (the exported effZ/effA are mixtures, rounded to integers
exactly as `cf_nucel_exact.leg_rates` rounds them).

The momentum grid is geometric; `ks_nucel_cf.py` takes the NEAREST bucket's
kernel with its argument rescaled by p_bucket/p_step (the elastic deflection
at fixed momentum transfer scales as 1/p) and interpolates the rate log-log
between the two bracketing buckets.

usage:
  python3 nucel_tables.py scan  --files '<glob>' --out za.json
  python3 nucel_tables.py build --za za.json --cache <dir> --out table.npz [--jobs 48]
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
from scipy.special import j0

DRIVER = os.environ.get(
    'NUCEL_DRIVER',
    '/home/submit/david_w/.claude-work/jobs/28e0dfa8/tmp/nucel/nucel_g4driver.sh')
M_PI = 0.13957039
REF_RHO = 0.1073        # g/cm^3 -- only self-consistency matters (mu is rho-free)
REF_LEN = 100.0         # mm
NSAMP = int(os.environ.get('NUCEL_NSAMP', '500000'))
SEED = 20260924
PGRID = np.geomspace(0.18, 48.0, 74)          # GeV/c, ratio 1.08
UGRID = np.concatenate(([0.0], np.geomspace(1e-1, 1e7, 700)))   # 1/rad
VGRID = np.concatenate(([0.0], np.geomspace(1e-4, 1e5, 500)))   # 1/MeV
NBIN = 20000
SPECIES = (211, -211)


def ekin_mev(p):
    return (np.sqrt(p * p + M_PI * M_PI) - M_PI) * 1e3


def _cf_from_samples(x, grid, kind):
    pos = x[x > 0.0]
    nz = x.size - pos.size
    if pos.size == 0:
        return np.ones(len(grid), dtype=np.complex128 if kind == 'exp' else np.float64)
    edges = np.geomspace(pos.min(), pos.max() * (1 + 1e-12), NBIN + 1)
    cnt, _ = np.histogram(pos, bins=edges)
    ctr = np.sqrt(edges[:-1] * edges[1:])
    wgt = cnt / float(x.size)
    keep = wgt > 0
    ctr, wgt = ctr[keep], wgt[keep]
    out = np.empty(len(grid), dtype=np.complex128 if kind == 'exp' else np.float64)
    for i0 in range(0, len(grid), 64):
        gg = grid[i0:i0 + 64]
        if kind == 'j0':
            out[i0:i0 + 64] = (j0(gg[:, None] * ctr[None, :]) * wgt[None, :]).sum(1)
        else:
            out[i0:i0 + 64] = (np.exp(1j * gg[:, None] * ctr[None, :]) * wgt[None, :]).sum(1)
    out += nz / float(x.size)
    return out


def build_one(task):
    pdg, Z, A, ip, cache = task
    fn = os.path.join(cache, f'b_{pdg}_{Z}_{A}_{ip:03d}.npz')
    if os.path.exists(fn):
        return fn
    p = float(PGRID[ip])
    ek = float(ekin_mev(p))
    prefix = os.path.join(cache, f'tmp_{pdg}_{Z}_{A}_{ip:03d}_{os.getpid()}')
    cmd = [DRIVER, '--pdg', str(pdg), '--Z', '%d' % Z, '--A', '%d' % A,
           '--rho', '%.17g' % REF_RHO, '--ekin', '%.17g' % ek, '--len', '%.17g' % REF_LEN,
           '--n', str(NSAMP), '--seed', str(SEED + ip), '--out', prefix]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'driver failed {task}: {r.stderr[-1500:]}')
    sig = None
    model = xs = ''
    for line in open(prefix + '.rec'):
        w = line.split()
        if w and w[0] == 'rate':
            sig = float(w[1])
        elif w and w[0] == 'model':
            model = w[1]
        elif w and w[0] == 'xs':
            xs = w[1]
    th = np.fromfile(prefix + '.theta.bin', dtype=np.float64)
    de = np.fromfile(prefix + '.eloss.bin', dtype=np.float64)
    for ext in ('.rec', '.theta.bin', '.eloss.bin'):
        try:
            os.remove(prefix + ext)
        except OSError:
            pass
    if th.size == 0 or not np.any(th > 0):
        raise RuntimeError(f'empty kernel for {task}')
    mu = sig * 10.0 / REF_RHO
    g = _cf_from_samples(th, UGRID, 'j0')
    h = _cf_from_samples(de, VGRID, 'exp')
    tmp = fn + f'.tmp{os.getpid()}.npz'
    np.savez(tmp, mu=mu, g=g, h=h, p=p, ekin=ek,
             mth=th.mean(), mth2=(th ** 2).mean(), mde=de.mean(), mde2=(de ** 2).mean(),
             q50=np.quantile(th, 0.5), q99=np.quantile(th, 0.99),
             model=model, xs=xs)
    os.replace(tmp, fn)
    return fn


def scan(args):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    import prodfiles
    import uproot
    W = {}
    # only halves the production has SEALED (a file being written is not read)
    fs = [f for f in sorted(glob.glob(args.files))
          if os.path.exists(os.path.join(os.path.dirname(f),
                                         '.complete_' + os.path.basename(f).split('_')[2]))]
    for fn in fs:
        try:
            a = uproot.open(fn)['tree'].arrays(['msmoliv', 'msmoliidx', 'msmolistride'], library='np')
        except Exception as e:
            print('skip', fn, type(e).__name__)
            continue
        for ic in range(len(a['msmoliidx'])):
            m = prodfiles.reshape_records(a['msmoliv'][ic], nrec=len(a['msmoliidx'][ic]),
                                         stride=int(a['msmolistride'][ic]))
            z = np.rint(m[:, 0]).astype(int)
            aa = np.rint(m[:, 1]).astype(int)
            for k in set(zip(z.tolist(), aa.tolist())):
                sel = (z == k[0]) & (aa == k[1])
                W[k] = W.get(k, 0.0) + float(m[sel, 2].sum())
    tot = sum(W.values())
    za = sorted(([int(k[0]), int(k[1]), v / tot] for k, v in W.items() if k[0] >= 1 and k[1] >= 1),
                key=lambda r: -r[2])
    print(f'{len(fs)} files, {len(za)} (Z, A) buckets, total xg {tot:.1f} g/cm^2')
    old = json.load(open(args.out)) if os.path.exists(args.out) else []
    have = {(r[0], r[1]) for r in old}
    new = [r for r in za if (r[0], r[1]) not in have]
    json.dump(old + new, open(args.out, 'w'))
    print(f'{len(new)} new buckets added -> {args.out}')


def build(args):
    za = json.load(open(args.za))
    os.makedirs(args.cache, exist_ok=True)
    tasks = [(pdg, int(z), int(a), ip, args.cache)
             for (z, a, _) in za for pdg in SPECIES for ip in range(len(PGRID))]
    print(f'{len(tasks)} buckets ({len(za)} (Z, A) x {len(SPECIES)} species x {len(PGRID)} momenta)')
    with Pool(args.jobs) as pool:
        for k, _ in enumerate(pool.imap_unordered(build_one, tasks, chunksize=4)):
            if (k + 1) % 500 == 0:
                print(f'  {k+1}/{len(tasks)}', flush=True)
    nza = len(za)
    MU = np.zeros((len(SPECIES), nza, len(PGRID)))
    G = np.zeros((len(SPECIES), nza, len(PGRID), len(UGRID)))
    H = np.zeros((len(SPECIES), nza, len(PGRID), len(VGRID)), dtype=np.complex128)
    MOM = np.zeros((len(SPECIES), nza, len(PGRID), 6))
    for s, pdg in enumerate(SPECIES):
        for iz, (z, a, _) in enumerate(za):
            for ip in range(len(PGRID)):
                d = np.load(os.path.join(args.cache, f'b_{pdg}_{int(z)}_{int(a)}_{ip:03d}.npz'))
                MU[s, iz, ip] = d['mu']
                G[s, iz, ip] = d['g']
                H[s, iz, ip] = d['h']
                MOM[s, iz, ip] = [d['mth'], d['mth2'], d['mde'], d['mde2'], d['q50'], d['q99']]
    np.savez_compressed(args.out, species=np.array(SPECIES), za=np.array([[int(r[0]), int(r[1])] for r in za]),
                        zafrac=np.array([r[2] for r in za]), pgrid=PGRID, ugrid=UGRID, vgrid=VGRID,
                        mu=MU, g=G, h=H, mom=MOM, nsamp=NSAMP)
    print('wrote', args.out)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    s1 = sp.add_parser('scan')
    s1.add_argument('--files', required=True)
    s1.add_argument('--out', required=True)
    s2 = sp.add_parser('build')
    s2.add_argument('--za', required=True)
    s2.add_argument('--cache', required=True)
    s2.add_argument('--out', required=True)
    s2.add_argument('--jobs', type=int, default=48)
    args = ap.parse_args()
    {'scan': scan, 'build': build}[args.cmd](args)


if __name__ == '__main__':
    main()
