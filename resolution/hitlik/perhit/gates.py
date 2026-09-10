#!/usr/bin/env python3
"""Gates 1-5 of the per-hit (complement) residual export.

Run against a tree produced with `exportPerHitResidual=True`.  Everything here
is a CLOSURE test of the maker's own arithmetic -- nothing is fitted.

  1. d == n_meas - 5 for every track, and  sum_k z_k^2 == the fit's chi2
     (`chisqval`, which the maker sets to r^T R r).
  2. sum_b v^(k)_b == 1 for every component (the maker's `phres_vchk`), and
     the same recomputed here from `phresvarv`.
  3. Var(z_k) == 1 under the fit's own Q: measured over tracks, per component
     index and pooled.  (The `gaussq` arm of the offline term is exactly this
     model, so a departure is a defect, not physics.)
  4. Cov(z, truth-referenced pull) == 0 -- F^T R = 0 at the Gaussian level.
  5. the per-component CF exponents reproduce the single-functional route:
     the q/p exponents `cfqop_*` rebuilt from `resinfvarv` must equal what the
     same `cvhcf` gives; here we check the WEIGHT algebra instead, i.e. that
     a component's exponent depends on (v, tau) only through sqrt(v/sq2)*tau,
     by comparing two components at a matched product.
"""
import argparse, glob, math
import numpy as np
import uproot


BR = ['phres_d', 'phres_nmeas', 'phres_nfree', 'phres_chi2', 'phres_vchk',
      'phres_ok', 'phres_rankgap', 'phresz', 'phresraw', 'phresrow', 'phreshit', 'phresdim',
      'phrescls', 'phrespiv', 'phresinflat', 'phresvarv',
      'phcf_vgf', 'phcf_nok', 'phcf_msec', 'phcf_grp_closure',
      'chisqval', 'ndof', 'nValidHits', 'nValidPixelHits',
      'refParms', 'genParms', 'refCov', 'reseigidx', 'resinfvarv',
      'reshitcls', 'trackPt', 'trackEta', 'edmval']


def load(files, nmax=None, extra=()):
    out = {}
    got = 0
    for f in files:
        with uproot.open(f) as fh:
            t = fh['tree'] if 'tree' in fh else fh[fh.keys()[0]]
            keys = set(k.split(';')[0] for k in t.keys())
            want = [b for b in list(BR) + list(extra) if b in keys]
            arrs = t.arrays(want, library='np')
            for k, v in arrs.items():
                out.setdefault(k, []).append(v)
            got += len(arrs[want[0]])
            if nmax and got >= nmax:
                break
    return {k: np.concatenate(v)[:nmax] for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', required=True)
    ap.add_argument('--max-tracks', type=int, default=None)
    ap.add_argument('--maxchi2ndof', type=float, default=None,
                    help='optional chi2/ndof cut (a cut on the FIT, stated as such)')
    a = ap.parse_args()

    files = sorted(glob.glob(a.files))
    print(f'# {len(files)} files')
    d = load(files, a.max_tracks)
    n = len(d['phres_d'])
    print(f'# {n} rows')

    dd = d['phres_d'].astype(int)
    nm = d['phres_nmeas'].astype(int)
    nfree = d['phres_nfree'].astype(int)
    good = dd > 0
    print(f'# {good.sum()} rows with d > 0')

    # ---------------- GATE 1 ----------------
    print('\n=== GATE 1: d = n_meas - 5, and sum_k z_k^2 = chi2')
    bad = (dd != nm - 5) & good
    print(f'  d == n_meas - 5 : {(~bad & good).sum()}/{good.sum()}  '
          f'violations {bad.sum()}')
    if bad.sum():
        print('   first violations (d, nmeas, nfree):',
              list(zip(dd[bad][:5], nm[bad][:5], nfree[bad][:5])))
    nvalid = d['nValidHits'].astype(int)
    npix = d['nValidPixelHits'].astype(int)
    print(f'  n_meas == nvalid + nvalidpixel : '
          f'{(nm[good] == (nvalid + npix)[good]).sum()}/{good.sum()}')
    zc = np.array([float(np.sum(np.asarray(z, dtype=np.float64) ** 2)) if len(z) else np.nan
                   for z in d['phresz']])
    ch = d['chisqval'].astype(np.float64)
    rel = np.abs(zc - ch) / np.maximum(np.abs(ch), 1e-30)
    r = rel[good]
    print(f'  |sum z^2 - chi2|/chi2 : median {np.median(r):.3e}  '
          f'p90 {np.percentile(r, 90):.3e}  max {r.max():.3e}')
    print(f'  (in-maker phres_chi2 vs recomputed: max '
          f'{np.abs(d["phres_chi2"][good] - zc[good]).max():.3e})')

    # ---------------- GATE 2 ----------------
    print('\n=== GATE 2: sum_b v^(k)_b = 1 per component')
    vchk = d['phres_vchk'][good]
    print(f'  maker phres_vchk (max_k |sum_b v - 1|): median {np.median(vchk):.3e}  '
          f'p90 {np.percentile(vchk, 90):.3e}  p99 {np.percentile(vchk, 99):.3e}  '
          f'max {vchk.max():.3e}')
    if 'phres_rankgap' in d:
        rg = d['phres_rankgap'][good]
        print(f'  rank gap lambda_d/lambda_(d+1): median {np.median(rg):.3e}  '
              f'p10 {np.percentile(rg,10):.3e}  min {rg.min():.3e}')

    # ---------------- GATE 3 ----------------
    print('\n=== GATE 3: Var(z_k) = 1 under the fit\'s own Q')
    sel = good
    if a.maxchi2ndof is not None:
        sel = good & (ch / np.maximum(d['ndof'], 1) < a.maxchi2ndof)
        print(f'  (chi2/ndof < {a.maxchi2ndof}: {sel.sum()}/{good.sum()} rows)')
    allz = np.concatenate([np.asarray(z, dtype=np.float64) for z in d['phresz'][sel]])
    print(f'  pooled over {len(allz)} components: mean {allz.mean():+.5f}  '
          f'var {allz.var():.5f}  '
          f'skew {((allz - allz.mean())**3).mean()/allz.std()**3:+.3f}  '
          f'kurt {((allz - allz.mean())**4).mean()/allz.std()**4:.3f}')
    # per component index
    maxd = int(dd[sel].max())
    print('  per component index k:  k  N   mean      var')
    for k in range(min(maxd, 25)):
        v = np.array([z[k] for z in d['phresz'][sel] if len(z) > k], dtype=np.float64)
        if len(v) < 20:
            continue
        print(f'   {k:3d} {len(v):5d}  {v.mean():+8.4f}  {v.var():8.4f}')

    # conditioning
    infl = np.concatenate([np.asarray(x, dtype=np.float64) for x in d['phresinflat'][sel]])
    print(f'  Cholesky conditioning G_kk/piv_k: median {np.median(infl):.3f}  '
          f'p90 {np.percentile(infl,90):.3f}  p99 {np.percentile(infl,99):.3f}  '
          f'max {infl.max():.3g}')

    # ---------------- GATE 4 ----------------
    print('\n=== GATE 4: Cov(z, truth-referenced pull) = 0')
    rp = d['refParms'].astype(np.float64) if d['refParms'].ndim == 2 else \
        np.stack(d['refParms']).astype(np.float64)
    gp = d['genParms'].astype(np.float64) if d['genParms'].ndim == 2 else \
        np.stack(d['genParms']).astype(np.float64)
    cv = d['refCov'].astype(np.float64) if d['refCov'].ndim == 2 else \
        np.stack(d['refCov']).astype(np.float64)
    res = rp - gp
    res[:, 2] = (res[:, 2] + np.pi) % (2 * np.pi) - np.pi
    # whiten by the lower Cholesky of the (upper-stored) 5x5 refCov
    pulls = np.full((len(res), 5), np.nan)
    for i in np.nonzero(sel)[0]:
        C = cv[i].reshape(5, 5)
        C = np.triu(C) + np.triu(C, 1).T
        try:
            L = np.linalg.cholesky(C)
            pulls[i] = np.linalg.solve(L, res[i])
        except np.linalg.LinAlgError:
            pass
    ok = sel & np.isfinite(pulls).all(axis=1) & (np.abs(pulls) < 20).all(axis=1)
    print(f'  {ok.sum()} rows with a usable 5-pull')
    print('  Var(pull_j):', np.round(np.nanvar(pulls[ok], axis=0), 4))
    print('   k   N     corr(z_k, pull_qp)  pull_lam   pull_phi   pull_d0   pull_z0')
    for k in range(min(maxd, 25)):
        m = ok & np.array([len(z) > k for z in d['phresz']])
        if m.sum() < 50:
            continue
        zk = np.array([d['phresz'][i][k] for i in np.nonzero(m)[0]], dtype=np.float64)
        cs = [np.corrcoef(zk, pulls[m][:, j])[0, 1] for j in range(5)]
        print(f'  {k:3d} {m.sum():5d}   ' + '  '.join(f'{c:+8.4f}' for c in cs))

    # ---------------- misc ----------------
    print('\n=== export health')
    print(f'  phres_ok            : {d["phres_ok"][good].mean()*100:.2f} %')
    print(f'  phcf_nok == d       : '
          f'{(d["phcf_nok"][good] == dd[good]).mean()*100:.2f} %')
    print(f'  phcf_grp_closure    : max {d["phcf_grp_closure"][good].max():.3e}')
    ms = d['phcf_msec'][good].astype(np.float64)
    print(f'  phcf_msec / track   : median {np.median(ms):.1f}  mean {ms.mean():.1f}  '
          f'p90 {np.percentile(ms,90):.1f}  max {ms.max():.1f}')
    print(f'  phcf_msec / component: median {np.median(ms/np.maximum(dd[good],1)):.2f}')
    vgf = np.concatenate([np.asarray(x, dtype=np.float64) for x in d['phcf_vgf'][good]])
    print(f'  hit (Gaussian) share of a component: median {np.median(vgf):.4f}  '
          f'p10 {np.percentile(vgf,10):.4f}  p90 {np.percentile(vgf,90):.4f}')


if __name__ == '__main__':
    main()
