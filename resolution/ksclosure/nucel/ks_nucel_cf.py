#!/usr/bin/env python3
"""The nuclear-elastic CF family on the K_S -> pi+ pi- MASS functional.

WHAT IS ADDED
-------------
The in-maker mass CF (`cfmass_*`) composes hit / MS / ionisation / radiative
(+ delta) exponents on the 64-point tau grid.  A pion also takes `hadElastic`
collisions -- rare (N ~ 0.05 per track), large (25-35 mrad at 3 GeV, 0.2 rad at
0.2 GeV) isotropic kicks -- which no family carries.  This module builds that
family OFFLINE from the raw step records of the step-record re-production and
writes it into a pairs cache beside the maker's own exponents:

  angular part (the channel of `cf_nucel_exact`, on the mass functional):
      S_ang(tau) = sum_b sum_{s in b} N_s * ( g_s(w_b tau) - 1 )
  recoil sub-channel (`NUCEL_RECOIL`, as in `cf_nucel_exact`):
      S_rec(tau) = sum_s N_s * ( h_s(wq_s tau) - 1 )

* the BLOCKS b and their weights are the maker's own MS blocks: parmtype 10,
  pooled by global index, `w_b = sqrt(v_b / sum_s thp2_s) / sigma_m` with v_b
  the fit's influence variance (`resinfvarv`) -- exactly the weight the maker
  gives the Moliere exponent of that block (`cvhcf::trackExponents`,
  reproduced by `cvhcf_validate.py`).  A collision is an isotropic 2D kick at
  a point inside the step, like a Moliere kick, so it rides on the same
  weight.  That weight is checked here, not assumed: `--gate-ms N` rebuilds
  `cfmass_ms` from the same records and weights and compares it with what the
  maker wrote.
* N_s = mu(species, Z_s, A_s, p_s) * xg_s, PER STEP: the (Z, A) of every
  record (rounded to integers exactly as `cf_nucel_exact.leg_rates` does --
  never the first step of a leg), the step's own momentum, and the species
  from the leg: pi+ on the positive leg, pi- on the negative one.  The records
  are drained leg by leg, so a leg boundary is the one place the momentum
  jumps; each segment is given to the leg whose fitted momentum is nearest.
* g, h, mu: the Geant4 tables of `nucel_tables.py` (the species-correct
  elastic model and cross-section dataset through `nucel_g4driver`), nearest
  momentum bucket with the angular argument rescaled by p_bucket/p_step
  (fixed momentum transfer), rate interpolated log-log between buckets.
* wq_s: the recoil moves q/p like an ionisation loss, so its weight is the
  maker's ionisation weight of the SAME leg at the SAME momentum,
  `w_io(block) * cs * 1e-3 * IONI_SGN`, interpolated along the leg in p.  This
  replaces `cf_nucel_exact`'s leg-mean weight by a per-step one; the
  theta-dE independence approximation is kept.
* NOT centred: the Geant4e reference carries no nuclear-elastic mean (neither
  the kick, which is isotropic, nor the recoil loss, which is one-signed).

The channel is GATED exactly as the offline one: nothing is added unless
`cf_nucel_exact.NUCEL_CHANNEL` (env NUCEL_CHANNEL=1); the recoil only under
`cf_nucel_exact.NUCEL_RECOIL`.  Both parts are written separately so the
recoil-on and recoil-off variants are two views of one cache.

COLUMNS ADDED to the ks_pairs cache (rows identical to `ks_pairs.py`'s):
  Snuc_ang (n, 64)       the angular exponent
  Snuc_rec_re/_im        the recoil exponent
  nuc_N, nuc_Np, nuc_Nm  expected collisions (both legs / + / -)
  nuc_v                  second cumulant of the angular part, in z^2 units
  nuc_vrec, nuc_mrec     second cumulant and MEAN of the recoil part (z units)
  nuc_xgunk              fraction of the candidate's g/cm^2 with no table bucket
  Snuc_first, nuc_Nfirst the angular exponent / collisions of each leg's FIRST MS
                         block (decay vertex to the first measured module)
  nuc_Nbig(first)        collisions whose median kick alone is > 3 sigma_m

usage:
  NUCEL_CHANNEL=1 python3 ks_nucel_cf.py --prod <steprec prod> --table nucel_table.npz \
      --cache out.npz [--jobs 32] [--gate-ms 200]
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
KS = os.path.dirname(HERE)
RES = os.path.dirname(KS)
sys.path.insert(0, RES)
sys.path.insert(0, KS)
os.environ.setdefault("CVH_IONI_KOKOULIN", "0")

import cf_nucel_exact                                   # noqa: E402
import cf_track_resolution as ctr                        # noqa: E402
import ks_pairs                                          # noqa: E402
import prodfiles                                         # noqa: E402
from cf_mass_likelihood import IONI_SGN                  # noqa: E402

M_PI = 0.13957039
_T = {}


def load_table(fn):
    d = np.load(fn)
    t = {k: d[k] for k in d.files}
    t['zaidx'] = {(int(z), int(a)): i for i, (z, a) in enumerate(t['za'])}
    t['lpg'] = np.log(t['pgrid'])
    t['lu'] = np.log(t['ugrid'][1:])
    t['lv'] = np.log(t['vgrid'][1:])
    t['lmu'] = np.log(np.maximum(t['mu'], 1e-300))
    return t


def _p_from_cs(cs):
    """|p| [GeV] from the record's cs = E/p^3 [GeV^-2] (pion mass), by Newton
    in ln p on f = ln(E) - 3 ln p - ln cs, which is monotonic."""
    lc = np.log(np.maximum(cs, 1e-300))
    lp = -lc / 3.0                       # the massless start
    for _ in range(40):
        p = np.exp(lp)
        e2 = p * p + M_PI * M_PI
        f = 0.5 * np.log(e2) - 3.0 * lp - lc
        df = p * p / e2 - 3.0
        step = f / df
        lp = lp - step
        if np.max(np.abs(step)) < 1e-13:
            break
    return np.exp(lp)


def _segments(p):
    """Leg segments of a drain-ordered record list: a new segment wherever the
    momentum INCREASES (a leg only loses momentum) or jumps by > 5 %."""
    if len(p) < 2:
        return [np.arange(len(p))]
    dl = np.diff(np.log(p))
    cut = np.where((dl > 1e-9) | (np.abs(dl) > 0.05))[0] + 1
    return np.split(np.arange(len(p)), cut)


def _leg_sign(p_first, pp, pm):
    return +1 if abs(np.log(p_first / pp)) <= abs(np.log(p_first / pm)) else -1


def candidate(rec, tau, T, want_ms=False):
    """(Snuc_ang, Srec, diag[, Sms_rebuilt]) for one candidate.

    rec: dict of this candidate's arrays (gi, fam, vb, M, midx, I, iidx, qsi,
    qsv, sigma, pp, pm)."""
    nt = len(tau)
    sig = rec['sigma']
    gi, fam, vb = rec['gi'], rec['fam'], rec['vb']
    M, midx = rec['M'], rec['midx']
    Sang = np.zeros(nt)
    Srec = np.zeros(nt, dtype=np.complex128)
    Sms = np.zeros(nt) if want_ms else None
    diag = dict(N=0.0, Np=0.0, Nm=0.0, v=0.0, vrec=0.0, mrec=0.0, xgunk=0.0,
                Nfirst=0.0, Nbig=0.0, Nbigfirst=0.0, Sfirst=np.zeros(nt))
    if not len(M):
        return Sang, Srec, diag, Sms

    # ---- leg / species of every MS record --------------------------------
    p = M[:, 3].astype(np.float64)
    sgn = np.zeros(len(p), dtype=int)
    # the leg's FIRST MS block: the material from the decay vertex up to and
    # including the first measured module -- a kick there is invisible to the
    # pattern recognition (no hit before it to see a kink against)
    first = np.zeros(len(p), dtype=bool)
    for seg in _segments(p):
        sgn[seg] = _leg_sign(p[seg[0]], rec['pp'], rec['pm'])
        first[seg[midx[seg] == midx[seg[0]]]] = True
    # ---- MS block weights: exactly the maker's -------------------------------
    w = np.zeros(len(p))
    sel10 = fam == 10
    for g in np.unique(gi[sel10]):
        vpool = vb[sel10 & (gi == g)].astype(np.float64).sum()
        if vpool <= 0.:
            continue
        rows = np.where(midx == g)[0]
        if not len(rows):
            continue
        sq2 = M[rows, 5].astype(np.float64).sum()
        if sq2 <= 0.:
            continue
        wb = np.sqrt(vpool / sq2) / sig
        w[rows] = wb
        if want_ms:
            Sms += ctr.ms_step_exponent(M[rows].astype(np.float64), wb, tau)

    # ---- per-step rate ------------------------------------------------------
    z = np.rint(M[:, 0].astype(np.float64)).astype(int)
    a = np.rint(M[:, 1].astype(np.float64)).astype(int)
    xg = M[:, 2].astype(np.float64)
    iz = np.array([T['zaidx'].get((int(zz), int(aa)), -1) for zz, aa in zip(z, a)])
    unk = (iz < 0) & (z >= 1) & (a >= 1)
    diag['xgunk'] = float(xg[unk].sum() / max(xg.sum(), 1e-300))
    isp = np.where(sgn > 0, 0, 1)            # table species order (211, -211)
    lp = np.log(p)
    lpg = T['lpg']
    j = np.clip(np.searchsorted(lpg, lp) - 1, 0, len(lpg) - 2)
    f = np.clip((lp - lpg[j]) / (lpg[j + 1] - lpg[j]), 0.0, 1.0)
    ok = (iz >= 0) & (xg > 0.)
    izs = np.maximum(iz, 0)
    lmu = (1 - f) * T['lmu'][isp, izs, j] + f * T['lmu'][isp, izs, j + 1]
    N = np.where(ok, np.exp(lmu) * xg, 0.0)
    ipn = np.where(f < 0.5, j, j + 1)          # the NEAREST kernel bucket
    scale = T['pgrid'][ipn] / p                # theta_s = theta_bucket * p_b/p_s
    diag['N'] = float(N.sum())
    diag['Np'] = float(N[sgn > 0].sum())
    diag['Nm'] = float(N[sgn < 0].sum())
    mth2 = T['mom'][isp, izs, ipn, 1]
    diag['v'] = float(np.sum(N * (w * scale) ** 2 * mth2 / 2.0))
    diag['Nfirst'] = float(N[first].sum())
    # collisions whose MEDIAN kick alone is a > 3 sigma_m mass shift
    big = w * scale * T['mom'][isp, izs, ipn, 4] > 3.0
    diag['Nbig'] = float(N[big].sum())
    diag['Nbigfirst'] = float(N[big & first].sum())

    # ---- the angular exponent, grouped by kernel bucket -------------------
    act = ok & (w > 0.) & (N > 0.)
    keyv = (isp * 10000 + izs) * 1000 + ipn
    for kk in np.unique(keyv[act]):
        r = np.where(act & (keyv == kk))[0]
        s_ = r[0]
        gtab = T['g'][isp[s_], izs[s_], ipn[s_]]
        x = (w[r] * scale[r])[:, None] * tau[None, :]
        g = np.ones_like(x)
        m = x > T['ugrid'][1]
        if m.any():
            g[m] = np.interp(np.log(np.minimum(x[m], T['ugrid'][-1])), T['lu'], gtab[1:])
        Sang += (N[r][:, None] * (g - 1.0)).sum(axis=0)
        diag['Sfirst'] += ((N[r] * first[r])[:, None] * (g - 1.0)).sum(axis=0)

    # ---- the recoil: the ionisation weight of the same leg at the same p ----
    if cf_nucel_exact.NUCEL_RECOIL and len(rec['I']):
        I, iidx = rec['I'], rec['iidx']
        wio = np.zeros(len(I))
        sel11 = fam == 11
        for g in np.unique(gi[sel11]):
            vpool = vb[sel11 & (gi == g)].astype(np.float64).sum()
            if vpool <= 0.:
                continue
            rows = np.where(iidx == g)[0]
            if not len(rows):
                continue
            qs = rec['qsv'][rec['qsi'] == g] if rec['qsv'] is not None else None
            sq2 = ctr.ioni_sq2(I[rows].astype(np.float64),
                               None if qs is None else qs.astype(np.float64))
            if sq2 <= 0.:
                continue
            wio[rows] = IONI_SGN * np.sqrt(vpool / sq2) / sig
        cs = I[:, 10].astype(np.float64)
        wq_i = wio * cs * 1e-3                    # z per MeV of recoil loss
        pio = _p_from_cs(cs)
        sgi = np.zeros(len(pio), dtype=int)
        for seg in _segments(pio):
            sgi[seg] = _leg_sign(pio[seg[0]], rec['pp'], rec['pm'])
        wq = np.zeros(len(p))
        for s in (+1, -1):
            mi = (sgi == s) & (wq_i != 0.)
            mm = sgn == s
            if mi.any() and mm.any():
                o = np.argsort(pio[mi])
                wq[mm] = np.interp(p[mm], pio[mi][o], wq_i[mi][o])
        diag['vrec'] = float(np.sum(N * wq ** 2 * T['mom'][isp, izs, ipn, 3]))
        diag['mrec'] = float(np.sum(N * wq * T['mom'][isp, izs, ipn, 2]))
        actr = ok & (wq != 0.) & (N > 0.)
        for kk in np.unique(keyv[actr]):
            r = np.where(actr & (keyv == kk))[0]
            s_ = r[0]
            htab = T['h'][isp[s_], izs[s_], ipn[s_]]
            x = np.abs(wq[r])[:, None] * tau[None, :]
            sg = np.sign(wq[r])[:, None]
            hv = np.ones(x.shape, dtype=np.complex128)
            m = x > T['vgrid'][1]
            if m.any():
                xl = np.log(np.minimum(x[m], T['vgrid'][-1]))
                hv[m] = (np.interp(xl, T['lv'], htab[1:].real)
                         + 1j * np.broadcast_to(sg, x.shape)[m]
                         * np.interp(xl, T['lv'], htab[1:].imag))
            Srec += (N[r][:, None] * (hv - 1.0)).sum(axis=0)
    return Sang, Srec, diag, Sms


_BR = ['reseigidx', 'resinfvarv', 'msmoliv', 'msmoliidx', 'ioniurbanv', 'ioniurbanidx',
       'Jpsi_sigmamass', 'Muplus_pt', 'Muminus_pt', 'Muplus_eta', 'Muminus_eta',
       'cfmass_ms', 'run', 'lumi', 'event']


def process_task(arg):
    """One (task, truth) chunk: the standard ks_pairs columns + the family."""
    cidx, fs, tn, tabfn, mopts, gate_ms = arg
    if 'T' not in _T:
        _T['T'] = load_table(tabfn)
    T = _T['T']
    t = ks_pairs.load_truth([tn], quiet=True)
    cols = {}
    state = {'tgrid': None, 'tag': '', 'aux': mopts['aux'], 'auxi': mopts['auxi']}
    margs = argparse.Namespace(**mopts['match'])
    extra = {k: [] for k in ('Snuc_ang', 'Snuc_rec_re', 'Snuc_rec_im', 'nuc_N', 'nuc_Np',
                             'nuc_Nm', 'nuc_v', 'nuc_vrec', 'nuc_mrec', 'nuc_xgunk',
                             'nuc_Nfirst', 'nuc_Nbig', 'nuc_Nbigfirst', 'Snuc_first',
                             'zchk')}
    gate = []
    for fn in fs:
        ks_pairs.read_one(fn, t, margs, cols, state)
        f = uproot.open(fn)
        if 'tree' not in f:
            continue
        tr = f['tree']
        pt = f['runtree']['parmtype'].array(library='np')
        have = set(tr.keys())
        br = _BR + [b for b in ('ioniqscaleidx', 'ioniqscalev', 'cfmass_ok', 'Jpsi_mass',
                                'Jpsi_x', 'Jpsi_y', 'Jpsi_z', 'Muplus_phi', 'Muminus_phi')
                    if b in have]
        br += prodfiles.stride_keys(have, ('msmoliv', 'ioniurbanv'))
        a = tr.arrays(br, library='np')
        if not len(a['Jpsi_mass']):
            continue
        tau = np.asarray(state['tgrid'], dtype=np.float64)
        # the SAME selection read_one applied (same inputs, same function)
        tab = {k: np.asarray(a[k], dtype=np.float64) for k in
               ('Muplus_eta', 'Muminus_eta', 'Muplus_pt', 'Muminus_pt',
                'Muplus_phi', 'Muminus_phi', 'Jpsi_x', 'Jpsi_y', 'Jpsi_z')}
        for k in ('run', 'lumi', 'event'):
            tab[k] = np.asarray(a[k], dtype=np.int64)
        jrow, _, _ = ks_pairs.match(tab, t, margs)
        sig = np.asarray(a['Jpsi_sigmamass'], dtype=np.float64)
        okf = np.asarray(a['cfmass_ok']).astype(bool)
        good = okf & np.isfinite(sig) & (sig > 0.) & (jrow >= 0)
        idx = np.where(good)[0]
        for ic in idx:
            gi = np.asarray(a['reseigidx'][ic])
            M = prodfiles.reshape_records(a['msmoliv'][ic], nrec=len(a['msmoliidx'][ic]),
                                          stride=prodfiles.entry_stride(a, 'msmoliv', ic),
                                          branch='msmoliv')
            I = prodfiles.reshape_records(a['ioniurbanv'][ic], nrec=len(a['ioniurbanidx'][ic]),
                                          stride=prodfiles.entry_stride(a, 'ioniurbanv', ic),
                                          branch='ioniurbanv')
            rec = dict(gi=gi, fam=pt[gi], vb=np.asarray(a['resinfvarv'][ic], dtype=np.float32),
                       M=M, midx=np.asarray(a['msmoliidx'][ic]), I=I,
                       iidx=np.asarray(a['ioniurbanidx'][ic]),
                       qsi=(np.asarray(a['ioniqscaleidx'][ic]) if 'ioniqscaleidx' in a else None),
                       qsv=(np.asarray(a['ioniqscalev'][ic], dtype=np.float32).reshape(-1, 2)
                            if 'ioniqscalev' in a else None),
                       sigma=float(sig[ic]),
                       pp=float(a['Muplus_pt'][ic] * np.cosh(a['Muplus_eta'][ic])),
                       pm=float(a['Muminus_pt'][ic] * np.cosh(a['Muminus_eta'][ic])))
            want = gate_ms > 0 and len(gate) < gate_ms
            if cf_nucel_exact.NUCEL_CHANNEL:
                Sa, Sr, dg, Sms = candidate(rec, tau, T, want_ms=want)
            else:
                Sa, Sr = np.zeros(len(tau)), np.zeros(len(tau), dtype=np.complex128)
                dg = dict(N=0., Np=0., Nm=0., v=0., vrec=0., mrec=0., xgunk=0.,
                          Nfirst=0., Nbig=0., Nbigfirst=0., Sfirst=np.zeros(len(tau)))
                Sms = None
            if Sms is not None:
                ref = np.asarray(a['cfmass_ms'][ic], dtype=np.float64)
                gate.append((float(np.max(np.abs(Sms - ref))), float(np.max(np.abs(ref)))))
            extra['Snuc_ang'].append(Sa.astype(np.float32))
            extra['Snuc_rec_re'].append(Sr.real.astype(np.float32))
            extra['Snuc_rec_im'].append(Sr.imag.astype(np.float32))
            for k in ('N', 'Np', 'Nm', 'v', 'vrec', 'mrec', 'xgunk', 'Nfirst', 'Nbig',
                      'Nbigfirst'):
                extra[f'nuc_{k}'].append(dg[k])
            extra['Snuc_first'].append(dg['Sfirst'].astype(np.float32))
            extra['zchk'].append(float(a['Jpsi_mass'][ic]))
    return cidx, cols, extra, gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prod', required=True)
    ap.add_argument('--table', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--jobs', type=int, default=32)
    ap.add_argument('--maxtasks', type=int, default=0)
    ap.add_argument('--gate-ms', type=int, default=0,
                    help='rebuild cfmass_ms from the records for the first N '
                         'candidates of each chunk and compare with the maker')
    args = ap.parse_args()
    print(f'NUCEL_CHANNEL={int(cf_nucel_exact.NUCEL_CHANNEL)} '
          f'NUCEL_RECOIL={int(cf_nucel_exact.NUCEL_RECOIL)}')
    tdir = os.path.join(args.prod, 'truth')
    pairs = []
    for td in sorted(glob.glob(os.path.join(args.prod, 'task_[0-9]*'))):
        idx = os.path.basename(td).split('_')[-1]
        if not os.path.exists(os.path.join(td, '.complete')):
            continue
        tn = os.path.join(tdir, f'truth_{idx}.npz')
        fs = sorted(glob.glob(os.path.join(td, 'globalcor_ks_*.root')))
        if fs and os.path.exists(tn):
            pairs.append((idx, fs, tn))
    if args.maxtasks:
        pairs = pairs[:args.maxtasks]
    print(f'{len(pairs)} complete chunks')
    aux = dict(ks_pairs.cf_inmaker._MASS_AUX)
    aux.pop('mpre', None)
    aux.pop('w', None)
    aux['munc'] = 'Jpsi_mass_unc'
    aux['vtxres'] = 'Jpsi_vtxres'
    aux['covmassvtx'] = 'Jpsi_covmassvtx'
    mopts = dict(aux=aux, auxi=dict(ks_pairs.cf_inmaker._MASS_AUX_INT),
                 match=dict(max_dvtx=2.0, max_dlam=0.25, max_dphi=0.25, max_dp=0.50,
                            keep_unmatched=False))
    tasks = [(c, fs, tn, os.path.abspath(args.table), mopts, args.gate_ms)
             for c, fs, tn in pairs]
    res = {}
    gate = []
    with Pool(args.jobs) as pool:
        for k, (c, cols, extra, g) in enumerate(pool.imap_unordered(process_task, tasks)):
            res[c] = (cols, extra)
            gate += g
            if (k + 1) % 50 == 0:
                print(f'  {k+1}/{len(tasks)}', flush=True)
    out = {}
    tgrid = None
    for c in sorted(res):
        cols, extra = res[c]
        if not cols:
            continue
        nstd = sum(len(x) for x in cols['z'])
        nex = len(extra['zchk'])
        if nstd != nex:
            raise SystemExit(f'chunk {c}: {nstd} standard rows vs {nex} nucel rows')
        mr = np.concatenate(cols['ks_mreco'])
        if not np.array_equal(mr, np.asarray(extra['zchk'])):
            raise SystemExit(f'chunk {c}: row alignment broken')
        for k, v in cols.items():
            out.setdefault(k, []).append(np.concatenate(v))
        for k, v in extra.items():
            if k == 'zchk':
                continue
            out.setdefault(k, []).append(np.asarray(v))
    out = {k: np.concatenate(v) for k, v in out.items()}
    # tau grid and model tag from one file
    f0 = glob.glob(os.path.join(args.prod, 'task_*', 'globalcor_ks_*.root'))[0]
    tg, tag = ks_pairs.cf_inmaker._runtree_grid(uproot.open(f0))
    out['tgrid'] = np.asarray(tg, dtype=np.float64)
    out['cf_source'] = np.array(['inmaker-ks+nucel-offline'])
    out['cf_model'] = np.array([tag])
    out['ioni_sign_fixed'] = np.array([1])
    out['nucel_channel'] = np.array([int(cf_nucel_exact.NUCEL_CHANNEL)])
    out['nucel_recoil'] = np.array([int(cf_nucel_exact.NUCEL_RECOIL)])
    out['nucel_table'] = np.array([os.path.abspath(args.table)])
    np.savez_compressed(args.cache, **out)
    print(f'wrote {args.cache}: {len(out["z"])} rows')
    if gate:
        g = np.array(gate)
        print(f'MS WEIGHT GATE ({len(g)} candidates): max|Sms(records) - cfmass_ms| = '
              f'{g[:, 0].max():.3e} (max|S| {g[:, 1].max():.3g}; float32 floor '
              f'{6e-8 * g[:, 1].max():.1e})')
    if cf_nucel_exact.NUCEL_CHANNEL:
        print(f'N per candidate: mean {out["nuc_N"].mean():.4f} median {np.median(out["nuc_N"]):.4f}; '
              f'v_nuc median {np.median(out["nuc_v"]):.4e} mean {out["nuc_v"].mean():.4e}; '
              f'xg without a table bucket: max {out["nuc_xgunk"].max():.2e}')


if __name__ == '__main__':
    main()
