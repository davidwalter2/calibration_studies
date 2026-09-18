#!/usr/bin/env python3
"""Build the mass-CF pairs cache for the K_S -> pi+ pi- closure.

`resolution/cf_inmaker.py pairs` cannot be used directly: it takes the gen mass
from the maker's `Jpsigen_mass`, and the maker's gen matching is muon-specific
(|pdgId| == 13, ResidualGlobalCorrectionMakerTwoTrackG4e.cc:6116), so a pion
channel has no gen branch.  This module reads the same in-maker CF branches and
writes the same cache keys, but takes the gen mass -- and the truth selection --
from the offline Geant4 K_S truth produced by `ks_truth_dump.py`, joined on
(run, lumi, event) plus the decay vertex and both daughter momenta.

The truth mass is the SIMULATED pi+pi- invariant mass at the decay vertex, not
a PDG number: the K_S is decayed by Geant4 (Pythia has ParticleDecays:limitTau0
with tau0Max = 10 mm against c*tau = 26.8 mm) as a pure two-body phase-space
decay with NO radiation, so every decay is exactly at the Geant4 K_S mass and a
delta kernel there is exact.

The join runs CHUNK BY CHUNK: `task_NNNN` of the refit and
`truth/truth_NNNN.npz` come from the same input list, so only one chunk of
truth is ever resident (the sample's whole truth is ~18 M rows).

usage:
  source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
  python3 ks_pairs.py --prod <prod dir> --cache runs/kspairs_all.npz
"""
import argparse
import glob
import math
import os
import sys

import numpy as np
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
sys.path.insert(0, RES)

import cf_inmaker  # noqa: E402

M_PI = 0.13957039

# cf_inmaker's cache layout, reused verbatim so make_card.py and every other
# consumer read this cache unchanged.
FAM_KEYS = ("Sms", "Sdel", "Sio_re", "Sio_im", "Srad_re", "Srad_im")
FAM_BRANCHES = [f"cfmass_{s}" for s in
                ("ms", "del", "ioni_re", "ioni_im", "rad_re", "rad_im")]
# columns the truth join needs, on top of cf_inmaker's own aux set
_GEOM = ("Jpsi_x", "Jpsi_y", "Jpsi_z", "Muplus_phi", "Muminus_phi")
# truth column -> cache column
_TRUTH_OUT = (('rdec', 'ks_rdec'), ('vz', 'ks_zdec'), ('fromb', 'ks_fromb'),
              ('ksp', 'ks_p'), ('kspt', 'ks_pt'), ('prodr', 'ks_prodr'),
              ('pabsp', 'ks_pgenp'), ('pabsm', 'ks_pgenm'),
              ('motherpdg', 'ks_motherpdg'))


def _dphi(a, b):
    d = np.abs(a - b) % (2.0 * np.pi)
    return np.minimum(d, 2.0 * np.pi - d)


def load_truth(files, quiet=False):
    if isinstance(files, str):
        files = sorted(glob.glob(files)) if any(c in files for c in '*?[') else [files]
    if not files:
        raise SystemExit('no truth files')
    cols = None
    for fn in files:
        with np.load(fn) as d:
            n = len(d['run'])
            if cols is None:
                # per-row columns only: the file also carries scalar job
                # counters (nevents, nbadfiles, nskipped), and a length check
                # is what keeps a newly added one from being sorted as a row
                cols = {k: [] for k in d.files if len(d[k]) == n or k == 'run'}
            for k in cols:
                cols[k].append(d[k])
    t = {k: np.concatenate(v) for k, v in cols.items()}
    key = ((t['run'].astype(np.int64) << np.int64(40))
           ^ (t['lumi'].astype(np.int64) << np.int64(20))
           ^ t['event'].astype(np.int64))
    order = np.argsort(key, kind='stable')
    for k in list(t):
        t[k] = t[k][order]
    t['_key'] = key[order]
    pp = np.stack([t['pxp'], t['pyp'], t['pzp']], axis=1)
    pm = np.stack([t['pxm'], t['pym'], t['pzm']], axis=1)
    ep = np.sqrt((pp ** 2).sum(1) + M_PI ** 2)
    em = np.sqrt((pm ** 2).sum(1) + M_PI ** 2)
    ps = pp + pm
    t['mgen'] = np.sqrt(np.maximum((ep + em) ** 2 - (ps ** 2).sum(1), 0.0))
    t['lamp'] = np.arctan2(pp[:, 2], np.hypot(pp[:, 0], pp[:, 1]))
    t['phip'] = np.arctan2(pp[:, 1], pp[:, 0])
    t['pabsp'] = np.sqrt((pp ** 2).sum(1))
    t['lamm'] = np.arctan2(pm[:, 2], np.hypot(pm[:, 0], pm[:, 1]))
    t['phim'] = np.arctan2(pm[:, 1], pm[:, 0])
    t['pabsm'] = np.sqrt((pm ** 2).sum(1))
    t['rdec'] = np.hypot(t['vx'], t['vy'])
    t['ksp'] = np.sqrt(t['kspx'] ** 2 + t['kspy'] ** 2 + t['kspz'] ** 2)
    t['kspt'] = np.hypot(t['kspx'], t['kspy'])
    # CONSISTENCY: the two daughters must carry the parent's momentum. 0.014 %
    # of the dumped rows do not (median 43 % off) -- a SimVertex whose children
    # are not the K_S decay products -- and their `mgen` is meaningless. They
    # would be rejected by the daughter-momentum match anyway; dropping them
    # here keeps the gen mass a clean delta.
    dp = np.abs(np.sqrt(((pp + pm) ** 2).sum(1)) - t['ksp']) / np.maximum(t['ksp'], 1e-9)
    keep = dp < 1e-3
    if not keep.all():
        nb = int((~keep).sum())
        for k in list(t):
            t[k] = t[k][keep]
        if not quiet:
            print(f'  dropped {nb} rows whose daughters do not carry the K_S momentum')
    if not quiet:
        print(f'truth: {len(files)} files, {len(t["mgen"])} K_S -> pipi rows; '
              f'sim mass mean {t["mgen"].mean():.7f} std {t["mgen"].std():.2e}')
    return t


def match(tab, t, args):
    """Row-wise truth match; returns (truth row index or -1, vertex distance)."""
    n = len(tab['run'])
    key = ((tab['run'] << np.int64(40)) ^ (tab['lumi'] << np.int64(20)) ^ tab['event'])
    lo = np.searchsorted(t['_key'], key, 'left')
    hi = np.searchsorted(t['_key'], key, 'right')
    out = -np.ones(n, dtype=np.int64)
    dv = np.full(n, np.nan)
    namb = np.zeros(n, dtype=np.int64)   # how many truth rows passed
    lamp_r = np.arctan(np.sinh(tab['Muplus_eta']))
    lamm_r = np.arctan(np.sinh(tab['Muminus_eta']))
    pp_r = tab['Muplus_pt'] * np.cosh(tab['Muplus_eta'])
    pm_r = tab['Muminus_pt'] * np.cosh(tab['Muminus_eta'])
    for i in range(n):
        best, bestd = -1, 1e9
        for j in range(lo[i], hi[i]):
            if abs(t['lamp'][j] - lamp_r[i]) > args.max_dlam:
                continue
            if abs(t['lamm'][j] - lamm_r[i]) > args.max_dlam:
                continue
            if _dphi(t['phip'][j], tab['Muplus_phi'][i]) > args.max_dphi:
                continue
            if _dphi(t['phim'][j], tab['Muminus_phi'][i]) > args.max_dphi:
                continue
            if abs(pp_r[i] - t['pabsp'][j]) / t['pabsp'][j] > args.max_dp:
                continue
            if abs(pm_r[i] - t['pabsm'][j]) / t['pabsm'][j] > args.max_dp:
                continue
            d = math.dist((tab['Jpsi_x'][i], tab['Jpsi_y'][i], tab['Jpsi_z'][i]),
                          (t['vx'][j], t['vy'][j], t['vz'][j]))
            if d > args.max_dvtx:
                continue
            namb[i] += 1
            if d < bestd:
                best, bestd = j, d
        out[i] = best
        dv[i] = bestd if best >= 0 else np.nan
    return out, dv, namb


def read_one(fn, t, args, cols, state):
    try:
        f = uproot.open(fn)
        if 'tree' not in f:
            return 0, 0
        tr = f['tree']
    except Exception as e:
        print(f'skipping {fn}: {type(e).__name__}')
        return 0, 0
    tg, tag = cf_inmaker._runtree_grid(f)
    if tg is None:
        raise SystemExit(f'{fn} has no cftau in its runtree')
    if state['tgrid'] is None:
        state['tgrid'], state['tag'] = tg, tag
        print(f'tau grid {len(tg)} points [0, {tg[-1]:.4f}]; model {tag}')
    elif not np.array_equal(state['tgrid'], tg):
        raise SystemExit(f'{fn} uses a different tau grid')
    have = set(tr.keys())
    a_names = {k: b for k, b in state['aux'].items() if b in have}
    ai_names = {k: b for k, b in state['auxi'].items() if b in have}
    need = sorted(set(FAM_BRANCHES + ['cfmass_vgf', 'cfmass_ok', 'Jpsi_mass',
                                      'Jpsi_sigmamass', 'Muplus_eta', 'Muminus_eta',
                                      'Muplus_pt', 'Muminus_pt']
                      + list(_GEOM) + list(a_names.values()) + list(ai_names.values())))
    miss = [b for b in need if b not in have]
    if miss:
        raise SystemExit(f'{fn} is missing {miss}')
    a = tr.arrays(need, library='np')
    nt = len(state['tgrid'])
    if not len(a['Jpsi_mass']):
        return 0, 0
    S = {k: cf_inmaker._stack(a, b, nt) for k, b in zip(FAM_KEYS, FAM_BRANCHES)}
    ok = np.asarray(a['cfmass_ok']).astype(bool)
    sig = np.asarray(a['Jpsi_sigmamass'], dtype=np.float64)
    mrec = np.asarray(a['Jpsi_mass'], dtype=np.float64)
    tab = {k: np.asarray(a[k], dtype=np.float64) for k in
           ('Muplus_eta', 'Muminus_eta', 'Muplus_pt', 'Muminus_pt',
            'Muplus_phi', 'Muminus_phi', 'Jpsi_x', 'Jpsi_y', 'Jpsi_z')}
    for k in ('run', 'lumi', 'event'):
        tab[k] = np.asarray(a[k], dtype=np.int64)
    jrow, dvtx, namb = match(tab, t, args)
    base = ok & np.isfinite(sig) & (sig > 0.)
    good = base if args.keep_unmatched else (base & (jrow >= 0))
    idx = np.where(good)[0]
    if not len(idx):
        return len(mrec), int((jrow >= 0).sum())
    j = jrow[idx]
    mg = np.where(j >= 0, t['mgen'][np.maximum(j, 0)], -99.0)

    def push(k, v):
        cols.setdefault(k, []).append(np.asarray(v))
    push('z', (mrec[idx] - mg) / sig[idx])
    push('sigma', sig[idx])
    push('eta', mg)
    push('vgf', np.asarray(a['cfmass_vgf'], dtype=np.float64)[idx])
    for k in FAM_KEYS:
        push(k, S[k][idx])
    for k, b in a_names.items():
        push(k, np.asarray(a[b], dtype=np.float64)[idx])
    for k, b in ai_names.items():
        push(k, np.asarray(a[b], dtype=np.int64)[idx])
    # THE MOMENTUM-SCALE LEVER ARM of this decay.
    #
    #   m^2 = 2 m_pi^2 + 2 (E1 E2 - p1.p2)
    # under p -> (1+eps) p,
    #   d ln m / d eps = [m^2 - m_pi^2 (2 + E1/E2 + E2/E1)] / m^2 .
    #
    # It is 1 only in the ultra-relativistic limit. For J/psi -> mu mu it is
    # 0.9953 and nobody ever had to think about it; for K_S -> pi pi it is
    # 0.685 in a symmetric decay and smaller in an asymmetric one, because
    # 2 m_pi / m_KS = 0.56. The unbinned term measures the relative shift of
    # the MASS (delta_i = mobs_i - m_ref * alpha * 1e-3), so the MOMENTUM
    # scale is alpha / <f>, and ignoring it would understate the momentum
    # scale by ~1.5x.
    ep = np.sqrt(t['pabsp'][np.maximum(j, 0)] ** 2 + M_PI ** 2)
    em = np.sqrt(t['pabsm'][np.maximum(j, 0)] ** 2 + M_PI ** 2)
    fmom = 1.0 - M_PI ** 2 * (2.0 + ep / em + em / ep) / np.maximum(mg, 1e-9) ** 2
    push('ks_fmom', np.where(j >= 0, fmom, -99.0))
    push('dvtx', dvtx[idx])
    push('nambig', namb[idx])
    push('matched', (j >= 0).astype(np.int64))
    push('ks_mreco', mrec[idx])
    for tk, ck in _TRUTH_OUT:
        v = t[tk][np.maximum(j, 0)].astype(np.float64)
        push(ck, np.where(j >= 0, v, -99.0))
    return len(mrec), int((jrow >= 0).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prod', required=True,
                    help='production directory holding task_NNNN/ and truth/')
    ap.add_argument('--truth-dir', default=None)
    ap.add_argument('--maxtasks', type=int, default=0)
    ap.add_argument('--cache', required=True)
    # THE TRUTH MATCH MUST NOT SELECT ON THE OBSERVABLE. The mass resolution of
    # a K_S is angular-dominated (f_ang ~ 0.74), so a tight cut on the daughter
    # angles would truncate the very residual the closure measures. Measured on
    # a 240-file pass: the core is on a plateau -- (0.05, 0.05, 0.20, 1.0) and
    # (0.20, 0.20, 0.40, 1.0) give the IDENTICAL 92 candidates and the identical
    # pull -- and only the far tail moves (0.50/0.50/0.80/1.0: +3 candidates,
    # std 0.894 -> 1.122, robust width 0.712 -> 0.722). The defaults below sit
    # past the plateau; `run_ks_closure.sh` re-runs a tight and a loose variant
    # as a systematic.
    ap.add_argument('--max-dvtx', type=float, default=2.0, help='cm')
    ap.add_argument('--max-dlam', type=float, default=0.25)
    ap.add_argument('--max-dphi', type=float, default=0.25)
    ap.add_argument('--max-dp', type=float, default=0.50)
    ap.add_argument('--keep-unmatched', action='store_true')
    args = ap.parse_args()

    tdir = args.truth_dir or os.path.join(args.prod, 'truth')
    pairs = []
    for td in sorted(glob.glob(os.path.join(args.prod, 'task_[0-9]*'))):
        idx = os.path.basename(td).split('_')[-1]
        if not os.path.exists(os.path.join(td, '.complete')):
            continue
        tn = os.path.join(tdir, f'truth_{idx}.npz')
        if not os.path.exists(tn):
            continue
        fs = sorted(glob.glob(os.path.join(td, 'globalcor_ks_*.root')))
        if fs:
            pairs.append((idx, fs, tn))
    if args.maxtasks:
        pairs = pairs[:args.maxtasks]
    if not pairs:
        raise SystemExit(f'no complete (task, truth) chunk pairs under {args.prod}')
    print(f'{len(pairs)} complete (task, truth) chunk pairs')

    aux = dict(cf_inmaker._MASS_AUX)
    aux.pop('mpre', None)
    aux.pop('w', None)          # doGen=False -> no genweight; every weight is 1
    # extra diagnostics of the vertex constraint, which is ON here: the
    # UNCONSTRAINED mass and the vertex-residual pull let the pull width be
    # attributed (constrained mass against unconstrained sigma would read as a
    # too-narrow pull).
    aux['munc'] = 'Jpsi_mass_unc'
    aux['vtxres'] = 'Jpsi_vtxres'
    aux['covmassvtx'] = 'Jpsi_covmassvtx'
    state = {'tgrid': None, 'tag': '', 'aux': aux,
             'auxi': dict(cf_inmaker._MASS_AUX_INT)}
    cols = {}
    ncand = nmatch = 0
    for k, (cidx, fs, tn) in enumerate(pairs):
        t = load_truth([tn], quiet=True)
        for fn in fs:
            nc, nm = read_one(fn, t, args, cols, state)
            ncand += nc
            nmatch += nm
        if (k + 1) % 25 == 0:
            print(f'  {k+1}/{len(pairs)} chunks, {ncand} candidates, {nmatch} matched')

    if not cols:
        raise SystemExit('nothing matched')
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out['tgrid'] = np.asarray(state['tgrid'], dtype=np.float64)
    out['cf_source'] = np.array(['inmaker-ks'])
    out['cf_model'] = np.array([state['tag']])
    out['ioni_sign_fixed'] = np.array([1])
    n = len(out['z'])
    print(f'{ncand} candidates read, {nmatch} truth-matched '
          f'({100.0*nmatch/max(ncand,1):.2f} %), {n} written')
    sel = out['matched'] > 0
    amb = out['nambig'][sel]
    print(f'truth ambiguity: {100.0*(amb > 1).mean():.3f} % of matched '
          f'candidates had more than one passing truth row '
          f'(max {int(amb.max())}); vertex distance median '
          f'{1e4*np.median(out["dvtx"][sel]):.0f} um, q99 '
          f'{1e4*np.quantile(out["dvtx"][sel], 0.99):.0f} um')
    r = out['z'][sel]
    print(f'pull: mean {r.mean():+.4f} median {np.median(r):+.4f} std {r.std():.4f} '
          f'(robust {0.7413*(np.quantile(r,.75)-np.quantile(r,.25)):.4f})')
    res = (out['z'] * out['sigma'])[sel]
    print(f'm_reco - m_gen: median {1e3*np.median(res):+.3f} MeV, '
          f'RMS {1e3*res.std():.2f} MeV')
    os.makedirs(os.path.dirname(os.path.abspath(args.cache)) or '.', exist_ok=True)
    np.savez_compressed(args.cache, **out)
    print('wrote', args.cache)


if __name__ == '__main__':
    main()
