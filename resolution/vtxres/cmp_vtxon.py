#!/usr/bin/env python3
"""Same-candidate comparison of the two-track fit with the VERTEX CONSTRAINT
ON and OFF.

The two productions read the SAME SIM files with the SAME fit settings and
differ only in `doVtxConstraint`, so every candidate can be matched on
(run, lumi, event) and the constraint's effect read off candidate by candidate.

What is compared
  * `r_v`, `sigma_v`  -- the one-Newton-step identity.  With the constraint ON
    the fit does not report a DCA; `Jpsi_vtxres` is what the UNCONSTRAINED fit
    would have reported, rebuilt from the half-gradient at the constrained
    optimum.  Agreement with the OFF fit's own `Jpsi_vtxres` is gate (b).
  * `Jpsi_mass_unc`(ON) vs `Jpsi_mass`(OFF) -- the same identity on the mass.
  * `Jpsi_sigmamass`(ON) vs (OFF) -- THE RESOLUTION GAIN.  Freezing theta_6
    conditions the mass on it, so

        sigma_m(ON) / sigma_m(OFF) = sqrt(1 - rho^2),
        rho = cov(m, theta_6) / (sigma_m(OFF) sigma_v),

    and `rho` is measurable from the ON file ALONE through the exported
    `Jpsi_covmassvtx`.  Both the prediction and the measurement are reported.
  * corr(w_mass, w_vtx) in the V metric, per candidate, from the exported
    influence vectors -- the free-regime rms 0.19 must go to zero.

usage:
  python3 cmp_vtxon.py --on '<prod_vtxon>/task_*/globalcor_*.root' \
                       --off '<prod>/task_*/globalcor_*.root' [--max N]
"""
import argparse, glob, sys
import numpy as np
import uproot

SEL_CUTS = ('Jpsi_vtxok', 'cfmass_ok', 'sigma_v > 0', 'sigma_m > 0',
            'chi2/ndof < 3', '|vtxvchk| < 1e-4')

SCAL = ['Jpsi_vtxres', 'Jpsi_vtxsig', 'Jpsi_vtxz', 'Jpsi_vtxok', 'Jpsi_vtxvchk',
        'Jpsi_mass', 'Jpsi_sigmamass', 'Jpsi_mass_unc', 'Jpsi_covmassvtx',
        'Jpsi_d', 'Jpsi_pt', 'Jpsi_eta', 'Jpsigen_mass', 'chisqval', 'ndof',
        'Muplusgen_pt', 'Muminusgen_pt', 'Muplusgen_eta', 'Muminusgen_eta',
        'cfmass_ok', 'run', 'lumi', 'event']
VEC = ['resinfv', 'resinfvtxv']


def load(pattern, nmax=None, vec=False):
    files = sorted(glob.glob(pattern))
    out, got = {}, 0
    want = SCAL + (VEC if vec else [])
    for f in files:
        try:
            fh = uproot.open(f)
        except Exception as e:
            print(f'# skip {f}: {e}', file=sys.stderr)
            continue
        with fh:
            t = fh['tree']
            keys = set(k.split(';')[0] for k in t.keys())
            a = t.arrays([b for b in want if b in keys], library='np')
        for k, v in a.items():
            out.setdefault(k, []).append(v)
        got += len(a['run'])
        if nmax and got >= nmax:
            break
    return {k: np.concatenate(v)[:nmax] for k, v in out.items()}, len(files)


def selection(d):
    """The extraction's own selection, cut by cut, so the two regimes'
    candidate sets can be compared rather than assumed equal."""
    n = len(d['run'])
    ndof = np.asarray(d['ndof'], dtype=float)
    c2 = np.where(ndof > 0, np.asarray(d['chisqval'], dtype=float)
                  / np.maximum(ndof, 1), 1e9)
    masks = [np.asarray(d['Jpsi_vtxok'], dtype=bool),
             (np.asarray(d['cfmass_ok'], dtype=bool) if 'cfmass_ok' in d
              else np.ones(n, bool)),
             np.asarray(d['Jpsi_vtxsig'], dtype=float) > 0,
             np.asarray(d['Jpsi_sigmamass'], dtype=float) > 0,
             c2 < 3.0,
             np.abs(np.asarray(d['Jpsi_vtxvchk'], dtype=float)) < 1e-4]
    keep = np.ones(n, bool)
    flow = []
    for nm, m in zip(SEL_CUTS, masks):
        keep &= m
        flow.append((nm, int(keep.sum())))
    return keep, flow


def q(name, x, scale=1.0, fmt='.4g'):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)] * scale
    if not x.size:
        print(f'  {name:46s} (empty)'); return
    print(f'  {name:46s} median {np.median(x):{fmt}}  p90 {np.percentile(x,90):{fmt}}  '
          f'p99 {np.percentile(x,99):{fmt}}  max {x.max():{fmt}}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--on', required=True)
    ap.add_argument('--off', required=True)
    ap.add_argument('--max', type=int, default=None)
    ap.add_argument('--max-vchk', type=float, default=1e-4)
    ap.add_argument('--dump', default=None,
                    help='npz of the per-candidate matched arrays, for plotting')
    ap.add_argument('--nvec', type=int, default=20000,
                    help='candidates for the per-candidate influence correlation')
    a = ap.parse_args()

    on, nfon = load(a.on, a.max, vec=True)
    off, nfoff = load(a.off, a.max, vec=True)
    print(f'# ON  {nfon} files, {len(on["run"])} candidates')
    print(f'# OFF {nfoff} files, {len(off["run"])} candidates')

    key = lambda d: [tuple(int(x) for x in t) for t in
                     np.stack([d[k] for k in ('run', 'lumi', 'event')], 1)]
    kon, koff = key(on), key(off)
    moff = {}
    for j, k in enumerate(koff):
        moff.setdefault(k, j)
    dup = len(koff) - len(moff)
    pair = [(i, moff[k]) for i, k in enumerate(kon) if k in moff]
    i0 = np.array([p[0] for p in pair]); j0 = np.array([p[1] for p in pair])
    print(f'# matched {len(pair)} candidates ({dup} duplicate OFF keys)')

    gd = (on['Jpsi_vtxok'][i0].astype(bool) & off['Jpsi_vtxok'][j0].astype(bool)
          & (on['Jpsi_vtxsig'][i0] > 0) & (off['Jpsi_vtxsig'][j0] > 0)
          & (np.abs(on['Jpsi_vtxvchk'][i0]) < a.max_vchk)
          & (np.abs(off['Jpsi_vtxvchk'][j0]) < a.max_vchk))
    i0, j0 = i0[gd], j0[gd]
    print(f'# {i0.size} good matched pairs (|vtxvchk| < {a.max_vchk:g} on both)')

    print('\n=== Jpsi_d under the constraint')
    if 'Jpsi_d' in on:
        jd = on['Jpsi_d'][i0]
        print(f'  ON : max|Jpsi_d| = {np.abs(jd).max():.6g}  nonzero {np.count_nonzero(jd)}')

    print('\n=== (b) the DCA:  r_v(ON, one-step) vs r_v(OFF, reported)')
    rc, rf = on['Jpsi_vtxres'][i0], off['Jpsi_vtxres'][j0]
    sc, sf = on['Jpsi_vtxsig'][i0], off['Jpsi_vtxsig'][j0]
    q('|dr_v| / sigma_v', np.abs(rc - rf) / sf)
    q('|r_v(ON)/r_v(OFF) - 1|', np.abs(rc / rf - 1.))
    q("(b') |sigma_v(ON)/sigma_v(OFF) - 1|", np.abs(sc / sf - 1.))
    d = np.abs(rc - rf) / sf
    for t in (0.05, 0.1, 0.5):
        print(f'  P(|dr_v|/sigma_v > {t}) = {(d>t).mean()*100:.4f} %  ({int((d>t).sum())})')

    print('\n=== (a) the MASS:  m_unc(ON) vs m(OFF)')
    if 'Jpsi_mass_unc' in on:
        mu, mo, mc = on['Jpsi_mass_unc'][i0], off['Jpsi_mass'][j0], on['Jpsi_mass'][i0]
        q('|m_unc(ON)/m(OFF) - 1|', np.abs(mu / mo - 1.))
        q('|m_c(ON)/m(OFF) - 1|   (no identity)', np.abs(mc / mo - 1.))
        r = mu / mo - 1.
        print(f'  signed m_unc/m(OFF)-1: mean {r.mean():+.4g}  rms {r.std():.4g}')
        print(f'  m_c - m_unc: mean {np.mean(mc-mu)*1e3:+.4f} MeV  '
              f'rms {np.std(mc-mu)*1e3:.4f} MeV')

    print('\n=== THE RESOLUTION GAIN:  sigma_m(ON) / sigma_m(OFF)')
    smc, smo = on['Jpsi_sigmamass'][i0], off['Jpsi_sigmamass'][j0]
    ratio = smc / smo
    print(f'  measured ratio: mean {ratio.mean():.6f}  median {np.median(ratio):.6f}  '
          f'p16 {np.percentile(ratio,16):.6f}  p84 {np.percentile(ratio,84):.6f}  '
          f'min {ratio.min():.6f}')
    print(f'  -> mean GAIN in sigma {100*(1-ratio.mean()):.3f} %  '
          f'(in variance {100*(1-ratio.mean()**2):.3f} %)')
    if 'Jpsi_covmassvtx' in on:
        cov = on['Jpsi_covmassvtx'][i0]
        smu = np.sqrt(smc ** 2 + (cov / sc) ** 2)     # the unconstrained sigma_m
        rho = cov / (smu * sc)
        pred = np.sqrt(1. - rho ** 2)
        print(f'  rho = cov(m,theta_6)/(sigma_m(unc) sigma_v): mean {rho.mean():+.5f}  '
              f'rms {np.sqrt((rho**2).mean()):.5f}  p1 {np.percentile(rho,1):+.4f}  '
              f'p99 {np.percentile(rho,99):+.4f}')
        print(f'  predicted sqrt(1-rho^2): mean {pred.mean():.6f}  '
              f'median {np.median(pred):.6f}')
        q('|predicted/measured - 1|', np.abs(pred / ratio - 1.))
        q('|sigma_m(unc from ON)/sigma_m(OFF) - 1|', np.abs(smu / smo - 1.))
        # vs the GEN pT of the softer muon
        gpt = np.minimum(on['Muplusgen_pt'][i0], on['Muminusgen_pt'][i0])
        ed = np.percentile(gpt, np.linspace(0, 100, 9))
        print('\n  gain vs GEN pT of the softer muon (8 quantile bins):')
        print('   ' + ''.join(f'{v:>9.2f}' for v in ed[1:]))
        gain, rr = [], []
        for b in range(8):
            m = (gpt >= ed[b]) & (gpt < ed[b + 1] if b < 7 else gpt <= ed[8])
            gain.append(100 * (1 - ratio[m].mean())); rr.append(np.sqrt((rho[m]**2).mean()))
        print('   gain%' + ''.join(f'{v:>9.3f}' for v in gain))
        print('   rms rho' + ''.join(f'{v:>7.4f}' for v in rr))

    print('\n=== corr(w_mass, w_vtx) in the V metric, per candidate')
    corrs = {}
    for tag, dd, idx in (('ON', on, i0), ('OFF', off, j0)):
        if 'resinfv' not in dd or 'resinfvtxv' not in dd:
            print(f'  {tag}: influence vectors not exported'); continue
        cc = []
        for i in idx[:a.nvec]:
            am = np.asarray(dd['resinfv'][i], float)
            av = np.asarray(dd['resinfvtxv'][i], float)
            if am.size != av.size or am.size == 0:
                continue
            nm, nv = np.linalg.norm(am), np.linalg.norm(av)
            if nm <= 0 or nv <= 0:
                continue
            cc.append(float(am @ av) / (nm * nv))
        cc = np.array(cc)
        corrs[tag] = cc
        print(f'  {tag}: n {cc.size}  mean {cc.mean():+.6f}  rms {np.sqrt((cc**2).mean()):.6f}  '
              f'max|.| {np.abs(cc).max():.3e}')
    if a.dump and corrs:
        np.savez_compressed(a.dump.replace('.npz', '') + '_corr.npz',
                            **{f'corr_{k}': v for k, v in corrs.items()})

    if a.dump:
        out = dict(sigma_v_on=sc, sigma_v_off=sf, r_v_on=rc, r_v_off=rf,
                   sigma_m_on=smc, sigma_m_off=smo,
                   z_v_on=on['Jpsi_vtxz'][i0], z_v_off=off['Jpsi_vtxz'][j0],
                   genpt_soft=np.minimum(on['Muplusgen_pt'][i0],
                                         on['Muminusgen_pt'][i0]),
                   geneta_plus=on['Muplusgen_eta'][i0],
                   mass_on=on['Jpsi_mass'][i0], mass_off=off['Jpsi_mass'][j0],
                   jpsipt=on['Jpsi_pt'][i0])
        if 'Jpsi_mass_unc' in on:
            out['mass_unc_on'] = on['Jpsi_mass_unc'][i0]
            out['covmassvtx_on'] = on['Jpsi_covmassvtx'][i0]
        np.savez_compressed(a.dump, **out)
        print(f'\n  dumped {a.dump}')

    print('\n=== the extraction selection, cut by cut')
    kon, flon = selection(on)
    koff, floff = selection(off)
    print(f'  {"cut":24s} {"ON":>8s} {"OFF":>8s}')
    print(f'  {"(entries)":24s} {len(on["run"]):8d} {len(off["run"]):8d}')
    for (nm, a1), (_, a2) in zip(flon, floff):
        print(f'  {nm:24s} {a1:8d} {a2:8d}')
    # candidates one regime selects and the other does not -- and what their
    # pull looks like, since a tail comparison over different candidate sets
    # is not a comparison of the two regimes.
    kk = lambda d, m: set(tuple(int(x) for x in t) for t in
                          np.stack([d[k] for k in ('run', 'lumi', 'event')],
                                   1)[m])
    son, soff = kk(on, kon), kk(off, koff)
    print(f'  selected both {len(son & soff)}  ON only {len(son - soff)}  '
          f'OFF only {len(soff - son)}')
    for tag, dd, m, other in (('ON', on, kon, soff), ('OFF', off, koff, son)):
        key = np.stack([dd[k] for k in ('run', 'lumi', 'event')], 1)
        only = np.array([m[i] and tuple(int(x) for x in key[i]) not in other
                         for i in range(len(key))])
        if only.sum():
            z = np.asarray(dd['Jpsi_vtxz'], dtype=float)[only]
            print(f'  {tag}-only ({int(only.sum())}): P(|z_v|>5) = '
                  f'{np.mean(np.abs(z) > 5)*100:.2f} %, '
                  f'P(|z_v|>3) = {np.mean(np.abs(z) > 3)*100:.2f} %')

    print('\n=== the pull, both regimes (matched candidates)')
    for tag, z in (('ON ', on['Jpsi_vtxz'][i0]), ('OFF', off['Jpsi_vtxz'][j0])):
        zt = z[np.abs(z) < 5]
        print(f'  {tag}: mean {z.mean():+.5f}  Var {z.var():.4f}  '
              f'trimmed(|z|<5) Var {zt.var():.4f} skew '
              f'{((zt-zt.mean())**3).mean()/zt.std()**3:+.4f} '
              f'kurt {((zt-zt.mean())**4).mean()/zt.var()**2:.4f}  '
              f'trim {100*(1-zt.size/z.size):.3f} %')


if __name__ == '__main__':
    main()
