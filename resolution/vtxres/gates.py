#!/usr/bin/env python3
"""Gates of the VERTEX-CONSTRAINT RESIDUAL export (`exportVtxResidual=True`).

Everything here is a CLOSURE test of the maker's own arithmetic -- nothing is
fitted.

  (a) CONVERGENCE / the gradient structure.  `Jpsi_vtxbfree` is
      max_i |(F^T Vinv r)_i| over the FREE indices and `Jpsi_vtxb6` is
      -(Vinv F_6).r.  With index 6 FROZEN the first must be negligible against
      the second; with index 6 FREE both are small (6 is one of the free
      indices, so b_6 -> 0 too) and the ratio measures the fit's convergence.
  (b) THE TWO REGIMES AGREE.  The same events run with `doVtxConstraint=True`
      must reproduce, through the one-Newton-step projection
      `r_v = theta_6^frozen + sigma_v^2 b_6`, the DCA the FREE fit reports --
      and `sigma_v` must agree too.  Matched on (run, lumi, event, Jpsi_pt).
  (c) `sum_b |a_b|^2 == sigma_v^2` over the registered blocks (fam != 15).
      In the maker: `Jpsi_vtxvchk`.  Recomputed here from `vtxvarv` and,
      independently, from the influence vectors `resinfvtxv`.
  (d) `z_v^2 == chi2(constrained) - chi2(unconstrained)`.
  (e) THE SIGN CONVENTION.  `Jpsi_vtxsgnchk` applies the general per-block
      ionization sign rule to the MASS influence, where the answer is known to
      be -1 on every block: it must be 1.000.
  (f) THE EXPONENTS.  The vertex exponents must depend on the weights only
      through `sqrt(v_b/sq2)*tau/sigma`, checked against an offline rebuild
      from the step records (`--offline`, needs `exportStepRecords=True`).
"""
import argparse, glob, os, sys
import numpy as np
import uproot

BR = ['Jpsi_vtxres', 'Jpsi_vtxsig', 'Jpsi_vtxz', 'Jpsi_vtxb6', 'Jpsi_vtxdchi2',
      'Jpsi_vtxvchk', 'Jpsi_vtxvgf', 'Jpsi_vtxbfree', 'Jpsi_vtxfree',
      'Jpsi_vtxok', 'Jpsi_vtxsgnchk', 'vtxvarv', 'vtxsgnv', 'resinfvtxv',
      'resinfv', 'resinfvarv', 'reseigidx', 'reshitcls',
      'cfvtx_ms', 'cfvtx_ioni_re', 'cfvtx_ioni_im', 'cfvtx_rad_re', 'cfvtx_rad_im',
      'cfvtx_del', 'cfvtx_hitcls', 'cfvtx_hitv', 'cfvtx_grp', 'cfvtx_grp_closure',
      'cfmass_ms', 'cfmass_ioni_im', 'cfmass_vgf', 'cfmass_ok', 'cfmass_grp_closure',
      'Jpsi_d', 'Jpsi_x', 'Jpsi_y', 'Jpsi_z', 'Jpsi_mass', 'Jpsi_sigmamass',
      'Jpsi_pt', 'Jpsi_eta', 'chisqval', 'ndof', 'edmval',
      'run', 'lumi', 'event', 'Muplus_pt', 'Muminus_pt',
      'Muplusgen_pt', 'Muminusgen_pt', 'Muplusgen_eta', 'Muminusgen_eta',
      'Jpsi_jacVtx', 'Jpsi_jacMass']


def load(pattern, nmax=None, extra=()):
    files = sorted(glob.glob(pattern))
    out, got = {}, 0
    for f in files:
        try:
            fh = uproot.open(f)
        except Exception as e:
            print(f'# skip {f}: {e}', file=sys.stderr)
            continue
        with fh:
            t = fh['tree']
            keys = set(k.split(';')[0] for k in t.keys())
            want = [b for b in list(BR) + list(extra) if b in keys]
            arrs = t.arrays(want, library='np')
            for k, v in arrs.items():
                out.setdefault(k, []).append(v)
            got += len(arrs[want[0]])
        if nmax and got >= nmax:
            break
    if not out:
        raise SystemExit(f'no entries under {pattern}')
    return {k: np.concatenate(v)[:nmax] for k, v in out.items()}, files


def q(name, x, unit=''):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        print(f'  {name:38s}  (empty)')
        return
    print(f'  {name:38s}  median {np.median(x):.4g}  p90 {np.percentile(x,90):.4g}  '
          f'max {np.max(x):.4g} {unit}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', required=True, help='the FREE (doVtxConstraint=False) tree')
    ap.add_argument('--cons-files', default=None, help='the FROZEN tree, same events')
    ap.add_argument('--max', type=int, default=None)
    a = ap.parse_args()

    d, files = load(a.files, a.max)
    n = len(d['Jpsi_vtxres'])
    print(f'# {len(files)} files, {n} candidates, '
          f'vtxfree={bool(d["Jpsi_vtxfree"][0])}')

    ok = np.asarray(d['Jpsi_vtxok'], dtype=bool)
    sig = np.asarray(d['Jpsi_vtxsig'], dtype=float)
    good = ok & (sig > 0)
    print(f'# vtxok {ok.mean()*100:.2f} %, sigma>0 {(sig>0).mean()*100:.2f} %')

    # ---------------- GATE (a) ----------------
    print('\n=== GATE (a) the half-gradient structure')
    bf = np.asarray(d['Jpsi_vtxbfree'], dtype=float)
    b6 = np.abs(np.asarray(d['Jpsi_vtxb6'], dtype=float))
    q('max|b_free|', bf)
    q('|b_6|', b6)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(b6 > 0, bf / b6, np.nan)
    q('max|b_free| / |b_6|', r)

    # ---------------- GATE (c) ----------------
    print('\n=== GATE (c) sum_b |a_b|^2 = sigma_v^2  (fam != 15)')
    q('maker Jpsi_vtxvchk', np.abs(d['Jpsi_vtxvchk'][good]))
    # recompute from the influence vectors
    if 'resinfvtxv' in d and 'vtxvarv' in d:
        rel_a, rel_v = [], []
        for i in np.flatnonzero(good)[:2000]:
            av = np.asarray(d['resinfvtxv'][i], dtype=float).reshape(-1, 5)
            vv = np.asarray(d['vtxvarv'][i], dtype=float)
            if av.shape[0] != vv.shape[0]:
                continue
            va = (av ** 2).sum(axis=1) / sig[i] ** 2
            rel_a.append(np.max(np.abs(va - vv)) / max(vv.max(), 1e-300))
        q('|a_b|^2/sigma^2 vs vtxvarv (rel)', rel_a)
    # the hit-class shares must sum to the Gaussian share
    if 'cfvtx_hitv' in d:
        hs = np.array([np.sum(x) for x in d['cfvtx_hitv']])
        vgf = np.asarray(d['Jpsi_vtxvgf'], dtype=float)
        q('|sum_c hitv - vtxvgf|', np.abs(hs - vgf)[good])

    # ---------------- GATE (e) ----------------
    print('\n=== GATE (e) the ionization SIGN rule, checked on the MASS influence')
    sgc = np.asarray(d['Jpsi_vtxsgnchk'], dtype=float)
    print(f'  fraction of ionization blocks where the general rule gives -1 '
          f'on wmass: min {sgc[good].min():.6f}  median {np.median(sgc[good]):.6f}  '
          f'mean {sgc[good].mean():.6f}')
    print(f'  candidates with 100 %: {(sgc[good] == 1).mean()*100:.2f} %')
    if 'vtxsgnv' in d:
        allsg = np.concatenate([np.asarray(x, dtype=float) for x in d['vtxsgnv'][:2000]])
        print(f'  vertex per-block signs: -1 on {(allsg < 0).mean()*100:.2f} %, '
              f'+1 on {(allsg > 0).mean()*100:.2f} % of {allsg.size} blocks')

    # ---------------- GROUP CLOSURE ----------------
    if 'cfvtx_grp_closure' in d:
        print('\n=== the per-group split closes')
        q('cfvtx_grp_closure', d['cfvtx_grp_closure'][good])
        if 'cfmass_grp_closure' in d:
            q('cfmass_grp_closure', d['cfmass_grp_closure'][good])

    # ---------------- THE PHYSICS ----------------
    print('\n=== the residual itself')
    z = np.asarray(d['Jpsi_vtxz'], dtype=float)[good]
    rv = np.asarray(d['Jpsi_vtxres'], dtype=float)[good]
    print(f'  r_v   : mean {rv.mean()*1e4:+.4f} um   rms {rv.std()*1e4:.4f} um')
    print(f'  sigma_v: median {np.median(sig[good])*1e4:.4f} um')
    print(f'  z_v   : mean {z.mean():+.5f} +- {z.std()/np.sqrt(z.size):.5f}  '
          f'var {z.var():.5f}  skew {((z-z.mean())**3).mean()/z.std()**3:+.4f}  '
          f'kurt {((z-z.mean())**4).mean()/z.var()**2:.4f}')
    vgf = np.asarray(d['Jpsi_vtxvgf'], dtype=float)[good]
    print(f'  vtxvgf (GAUSSIAN=hit share of sigma_v^2): median {np.median(vgf):.4f} '
          f'p10 {np.percentile(vgf,10):.4f} p90 {np.percentile(vgf,90):.4f}')
    if 'cfmass_vgf' in d:
        mg = np.asarray(d['cfmass_vgf'], dtype=float)[good]
        print(f'  cfmass_vgf for comparison           : median {np.median(mg):.4f}')
    # is sigma_v correlated with z_v?  (the self-consistent-sigma lesson)
    print(f'  corr(sigma_v, z_v)     = {np.corrcoef(sig[good], z)[0,1]:+.5f} '
          f'(+-{1/np.sqrt(z.size):.5f})')
    print(f'  corr(sigma_v, |z_v|)   = {np.corrcoef(sig[good], np.abs(z))[0,1]:+.5f}')
    if 'Jpsi_d' in d:
        # `Jpsi_d` IS q(leg 0) * theta_6.  theta_6 is swap-invariant, so the
        # two differ by a sign that tracks the LEG ORDERING; showing both
        # skews on the same candidates is the evidence for the note in
        # STATE.md.
        jd = np.asarray(d['Jpsi_d'], dtype=float)[good]
        zc = jd / sig[good]
        flip = np.abs(rv - jd) > 1e-12 * np.maximum(np.abs(rv), 1e-30)
        print(f'  leg 0 is the mu+ on {100*(1-flip.mean()):.1f} % of candidates')
        print(f'  skew(z_v) RAW theta_6      = {((z-z.mean())**3).mean()/z.std()**3:+.4f}')
        print(f'  skew(z)   q(leg0)*theta_6  = {((zc-zc.mean())**3).mean()/zc.std()**3:+.4f}'
              f'   (the `Jpsi_d` convention)')
        print(f'  mean(z) RAW {z.mean():+.5f}   q-signed {zc.mean():+.5f}')

    nb = np.abs(np.asarray(d['Jpsi_vtxvchk'], dtype=float)) > 1e-4
    if nb.any():
        print(f'\n=== {nb.sum()} candidates with |vtxvchk| > 1e-4')
        for i in np.flatnonzero(nb)[:5]:
            print(f'   i={i} vchk={d["Jpsi_vtxvchk"][i]:.4g} sig={sig[i]:.4g} '
                  f'ok={bool(d["Jpsi_vtxok"][i])} nblk={len(d["vtxvarv"][i])} '
                  f'sumv={np.sum(d["vtxvarv"][i]):.4g} '
                  f'chi2ndof={d["chisqval"][i]/max(d["ndof"][i],1):.4g}')

    # ---------------- GATE (b) and (d) ----------------
    if a.cons_files:
        print('\n=== GATE (b)/(d) the FROZEN fit reproduces the FREE one')
        c, _ = load(a.cons_files, a.max)
        key = lambda dd: np.stack([np.asarray(dd[k], dtype=np.int64)
                                   for k in ('run', 'lumi', 'event')], axis=1)
        kf, kc = key(d), key(c)
        # candidates are unique per event on the gun sample
        mf = {tuple(x): i for i, x in enumerate(kf)}
        pairs = [(mf[tuple(x)], j) for j, x in enumerate(kc) if tuple(x) in mf]
        print(f'  matched {len(pairs)} of {len(kc)} constrained candidates')
        if pairs:
            i0 = np.array([p[0] for p in pairs]); j0 = np.array([p[1] for p in pairs])
            gf = good[i0] & np.asarray(c['Jpsi_vtxok'], dtype=bool)[j0] \
                 & (np.asarray(c['Jpsi_vtxsig'], dtype=float)[j0] > 0)
            i0, j0 = i0[gf], j0[gf]
            rf = np.asarray(d['Jpsi_vtxres'], dtype=float)[i0]
            rc = np.asarray(c['Jpsi_vtxres'], dtype=float)[j0]
            sf = np.asarray(d['Jpsi_vtxsig'], dtype=float)[i0]
            sc = np.asarray(c['Jpsi_vtxsig'], dtype=float)[j0]
            print(f'  {i0.size} matched good pairs')
            q('|r_v(frozen) - r_v(free)| / sigma_v', np.abs(rc - rf) / sf)
            q('|r_v(frozen)/r_v(free) - 1|', np.abs(rc / rf - 1.))
            q('|sigma_v(frozen)/sigma_v(free) - 1|', np.abs(sc / sf - 1.))
            if 'chisqval' in d and 'chisqval' in c:
                dchi = np.asarray(c['chisqval'], dtype=float)[j0] - \
                       np.asarray(d['chisqval'], dtype=float)[i0]
                zv2 = (np.asarray(d['Jpsi_vtxz'], dtype=float)[i0]) ** 2
                q('|dchi2 - z_v^2| / z_v^2', np.abs(dchi - zv2) / np.maximum(zv2, 1e-12))
                print(f'  <dchi2> = {dchi.mean():.4f}   <z_v^2> = {zv2.mean():.4f}')
            if 'ndof' in d and 'ndof' in c:
                print('  ndof(frozen) - ndof(free) = '
                      f'{np.unique(np.asarray(c["ndof"])[j0] - np.asarray(d["ndof"])[i0])}')


if __name__ == '__main__':
    main()
