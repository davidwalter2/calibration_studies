#!/usr/bin/env python3
"""Gates of the BEAM-LINE (luminous-region) constraint and its two transverse
residuals (`bsConstraint=True exportBsResidual=True`).

Everything here is a CLOSURE test of the maker's own arithmetic -- nothing is
fitted.

  G1  ndof.  The three beam rows are ONE Gaussian block shared by the two
      legs, so on matched candidates `ndof(ON) - ndof(OFF) == 3` exactly.
      (They used to be EMITTED twice while `ndof` counted +3 once -- the
      defect this branch fixes.)
  G2  WEIGHTLESS ROWS == ROWS OFF.  With `beamWidthScale=1e6` the rows carry
      no weight and the fit must reproduce the rows-OFF fit on every export,
      candidate by candidate.
  G3  THE BEAM chi2 ENTERS ONCE.  `Jpsi_bschi2fit` must equal
      `dbs^T covBS^-1 dbs` recomputed offline from the exported vertex,
      centroid, widths and slopes -- ONCE, not twice.  `Jpsi_bschi20` (the
      value actually added to `chisq0val`, at the linearisation point) must
      agree with it at convergence.
  G4  THE DOUBLE EMISSION, MEASURED.  Two identical rows with covariance S are
      one row with S/2, so the OLD build at nominal widths is the NEW build at
      `beamWidthScale = 1/sqrt(2)`.  `--half` against `--old` must be
      identical; `--files` (nominal) against `--old` is the SIZE of the defect.
  G5  sum_b |a_b|^2 == Cov_ii for the two beam functionals, and the same
      closure for the MASS and VERTEX functionals WITH the beam block
      registered (`Jpsi_vtxvchk`; `resinfcov + resinfcovhit + massvbs`).
  G6  THE MEAN-TERM IDENTITY.  `w_i` restricted to the beam rows is exactly
      `P_i^T`, so `Jpsi_bsmeanbs == -P` to machine precision.  It is the
      analytic check that the influence rows are the ones the beam-spot
      parameters would move.
  G7  THE LEAVE-ONE-OUT IDENTITY.  The rows-OFF fit's vertex minus the beam
      line, whitened with `C_vtx(OFF) + covBS` projected to the transverse
      plane, must equal the ON export to linearisation precision (the vertex
      residual's own gate is 1.4e-3 median / 1.3e-2 p90; match that).
"""
import argparse, glob, sys
import numpy as np
import uproot

BR = ['Jpsi_bsres', 'Jpsi_bscov', 'Jpsi_bsz', 'Jpsi_bschi2', 'Jpsi_bschi2fit',
      'Jpsi_bschi20', 'Jpsi_bsvchk', 'Jpsi_bsok', 'Jpsi_bsvtx', 'Jpsi_bsspot',
      'Jpsi_bsslope', 'Jpsi_bswidth', 'Jpsi_bsmeanmass', 'Jpsi_bsmeanvtx',
      'Jpsi_bsmeanbs', 'Jpsi_bsvbs', 'Jpsi_bsvhit', 'Jpsi_bsvms', 'Jpsi_bsvioni',
      'Jpsi_massvbs', 'Jpsi_vtxvbs', 'bsvarv', 'resinfbsv',
      'Jpsi_vtxres', 'Jpsi_vtxsig', 'Jpsi_vtxz', 'Jpsi_vtxvchk', 'Jpsi_vtxvgf',
      'Jpsi_vtxok', 'Jpsi_vtxdchi2', 'Jpsi_covvtx', 'vtxvarv', 'resinfvtxv',
      'resinfv', 'resinfvarv', 'resinfcov', 'resinfcovhit',
      'Jpsi_x', 'Jpsi_y', 'Jpsi_z', 'Jpsi_mass', 'Jpsi_sigmamass', 'Jpsi_mass_unc',
      'Jpsi_pt', 'Jpsi_eta', 'chisqval', 'ndof', 'edmval', 'niter',
      'run', 'lumi', 'event', 'Muplus_pt', 'Muminus_pt',
      'Muplusgen_pt', 'Muminusgen_pt', 'Jpsigen_x', 'Jpsigen_y', 'Jpsigen_z',
      'cfmass_vgf', 'cfvtx_grp_closure', 'cfmass_grp_closure', 'cfbs_grp_closure']

COMPARE = ['Jpsi_vtxres', 'Jpsi_vtxsig', 'Jpsi_vtxz', 'Jpsi_mass',
           'Jpsi_sigmamass', 'Jpsi_mass_unc', 'Jpsi_x', 'Jpsi_y', 'Jpsi_z',
           'Jpsi_pt', 'Jpsi_eta', 'Muplus_pt', 'Muminus_pt', 'chisqval',
           'Jpsi_vtxvgf', 'resinfcov', 'resinfcovhit', 'cfmass_vgf']


def load(pattern, nmax=None):
    files = sorted(glob.glob(pattern, recursive=True))
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
            want = [b for b in BR if b in keys]
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
        print(f'  {name:44s}  (empty)')
        return
    print(f'  {name:44s}  median {np.median(x):.4g}  p90 {np.percentile(x,90):.4g}  '
          f'max {np.max(x):.4g} {unit}')


def key(d):
    return np.array([f'{r}:{l}:{e}:{p:.6f}'
                     for r, l, e, p in zip(d['run'], d['lumi'], d['event'], d['Jpsi_pt'])])


def match(da, db):
    ka, kb = key(da), key(db)
    ua, ia = np.unique(ka, return_index=True)
    ub, ib = np.unique(kb, return_index=True)
    common = np.intersect1d(ua, ub)
    sa = ia[np.searchsorted(ua, common)]
    sb = ib[np.searchsorted(ub, common)]
    return sa, sb


def covbs(width, slope):
    """The 3x3 luminous-region covariance the maker builds (corrb12 = 0)."""
    s1, s2, s3 = width
    dxdz, dydz = slope
    v1, v2, v3 = s1*s1, s2*s2, s3*s3
    C = np.zeros((3, 3))
    C[0, 0] = v1
    C[1, 1] = v2
    C[2, 2] = v3
    C[2, 0] = C[0, 2] = dxdz*(v3 - v1)
    C[2, 1] = C[1, 2] = dydz*(v3 - v2)
    return C


def projector(C):
    P = np.zeros((2, 3))
    P[0, 0] = 1.
    P[1, 1] = 1.
    P[0, 2] = -C[0, 2]/C[2, 2]
    P[1, 2] = -C[1, 2]/C[2, 2]
    return P


def unpack6(v):
    a = np.asarray(v, dtype=float)
    return np.array([[a[0], a[1], a[2]], [a[1], a[3], a[4]], [a[2], a[4], a[5]]])


def cmp_trees(da, db, tag, nshow=None):
    sa, sb = match(da, db)
    print(f'  {len(sa)} matched candidates ({tag})')
    worst = 0.
    for b in COMPARE:
        if b not in da or b not in db:
            continue
        x = np.asarray(da[b], dtype=float)[sa]
        y = np.asarray(db[b], dtype=float)[sb]
        ok = np.isfinite(x) & np.isfinite(y)
        sc = np.maximum(np.abs(x[ok]), np.abs(y[ok]))
        sc = np.where(sc > 0, sc, 1.)
        r = np.abs(x[ok] - y[ok])/sc
        worst = max(worst, np.max(r) if r.size else 0.)
        print(f'    {b:22s} max rel {np.max(r) if r.size else 0.:.3e}  '
              f'median {np.median(r) if r.size else 0.:.3e}  n {ok.sum()}')
    ndofa = np.asarray(da['ndof'], dtype=float)[sa]
    ndofb = np.asarray(db['ndof'], dtype=float)[sb]
    print(f'    ndof(A)-ndof(B): unique {np.unique(ndofa-ndofb)}')
    print(f'    WORST relative difference over all compared branches: {worst:.3e}')
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', required=True, help='the rows-ON tree (nominal widths)')
    ap.add_argument('--off', default=None, help='the rows-OFF tree, same events')
    ap.add_argument('--wide', default=None, help='the beamWidthScale=1e6 tree')
    ap.add_argument('--half', default=None, help='the beamWidthScale=1/sqrt(2) tree')
    ap.add_argument('--old', default=None, help='the OLD (double-emitting) build, nominal widths')
    ap.add_argument('--max', type=int, default=None)
    a = ap.parse_args()

    d, files = load(a.files, a.max)
    n = len(d['Jpsi_bsres'])
    print(f'# ON: {len(files)} files, {n} candidates')
    bsok = np.asarray(d['Jpsi_bsok'], dtype=bool)
    print(f'# Jpsi_bsok {bsok.mean()*100:.3f} %')

    res = np.asarray(d['Jpsi_bsres'], dtype=float).reshape(-1, 2)
    cov = np.asarray(d['Jpsi_bscov'], dtype=float).reshape(-1, 3)
    z = np.asarray(d['Jpsi_bsz'], dtype=float).reshape(-1, 2)
    vtx = np.asarray(d['Jpsi_bsvtx'], dtype=float).reshape(-1, 3)
    spot = np.asarray(d['Jpsi_bsspot'], dtype=float).reshape(-1, 3)
    wid = np.asarray(d['Jpsi_bswidth'], dtype=float).reshape(-1, 3)
    slo = np.asarray(d['Jpsi_bsslope'], dtype=float).reshape(-1, 2)

    # ---------------- G1 ----------------
    if a.off:
        doff, _ = load(a.off, a.max)
        print('\n=== G1 ndof(ON) - ndof(OFF) == 3')
        sa, sb = match(d, doff)
        dn = np.asarray(d['ndof'], dtype=float)[sa] - np.asarray(doff['ndof'], dtype=float)[sb]
        u, c = np.unique(dn, return_counts=True)
        print(f'  matched {len(sa)}; ndof difference: ' +
              ', '.join(f'{int(x)} x{int(y)}' for x, y in zip(u, c)))
        print(f'  PASS' if (u.size == 1 and u[0] == 3) else '  FAIL')
    else:
        doff = None

    # ---------------- G2 ----------------
    if a.wide and a.off:
        print('\n=== G2 beamWidthScale=1e6 reproduces the rows-OFF fit')
        dw, _ = load(a.wide, a.max)
        w = cmp_trees(dw, doff, 'wide vs off')
        print(f'  {"PASS" if w < 1e-5 else "CHECK"} (threshold 1e-5)')

    # ---------------- G3 ----------------
    print('\n=== G3 the beam chi2 enters ONCE')
    chi2fit = np.zeros(n)
    for i in range(n):
        C = covbs(wid[i], slo[i])
        dbs = vtx[i] - spot[i]
        chi2fit[i] = dbs @ np.linalg.solve(C, dbs)
    ex = np.asarray(d['Jpsi_bschi2fit'], dtype=float)
    good = np.isfinite(ex) & (ex > 0)
    q('|recomputed/exported - 1| (chi2fit)', np.abs(chi2fit[good]/ex[good] - 1.))
    ex0 = np.asarray(d['Jpsi_bschi20'], dtype=float)
    g0 = good & np.isfinite(ex0) & (ex0 > 0)
    q('|chi2(linpoint)/chi2(optimum) - 1|', np.abs(ex0[g0]/ex[g0] - 1.))
    print(f'  <chi2fit> = {ex[good].mean():.4f} on 3 rows '
          f'(2x would give {2*ex[good].mean():.4f})')

    # ---------------- G5 ----------------
    print('\n=== G5 sum_b |a_b|^2 == Cov_ii  (the registered blocks, fam != 15)')
    q('maker Jpsi_bsvchk', np.abs(d['Jpsi_bsvchk'][bsok]))
    if 'resinfbsv' in d and 'bsvarv' in d:
        rel = []
        for i in np.flatnonzero(bsok)[:2000]:
            av = np.asarray(d['resinfbsv'][i], dtype=float)
            bv = np.asarray(d['bsvarv'][i], dtype=float)
            nb = bv.size//2
            if nb == 0:
                continue
            av = av.reshape(2, nb, 5)
            for k, cii in enumerate((cov[i, 0], cov[i, 2])):
                if cii <= 0:
                    continue
                s = (av[k]**2).sum()
                rel.append(abs(s/cii - 1.))
        q('from resinfbsv: |sum|a_b|^2/Cov_ii - 1|', rel)
    q('Jpsi_vtxvchk (vertex, WITH the beam block)', np.abs(d['Jpsi_vtxvchk'][bsok]))
    if 'resinfcov' in d and 'resinfcovhit' in d and 'Jpsi_massvbs' in d:
        sm2 = np.asarray(d['Jpsi_sigmamass'], dtype=float)**2
        tot = (np.asarray(d['resinfcov'], dtype=float)
               + np.asarray(d['resinfcovhit'], dtype=float)
               + np.asarray(d['Jpsi_massvbs'], dtype=float)*sm2)
        gm = np.isfinite(tot) & (sm2 > 0)
        q('mass: |(mat+hit+bs)/sigma_m^2 - 1|', np.abs(tot[gm]/sm2[gm] - 1.))
    for b in ('cfmass_grp_closure', 'cfvtx_grp_closure'):
        if b in d:
            q(f'{b}', np.abs(np.asarray(d[b], dtype=float)))
    if 'cfbs_grp_closure' in d:
        q('cfbs_grp_closure', np.abs(np.asarray(d['cfbs_grp_closure'], dtype=float).ravel()))

    # ---------------- G6 ----------------
    print('\n=== G6 the mean-term identity: Jpsi_bsmeanbs == -P')
    mb = np.asarray(d['Jpsi_bsmeanbs'], dtype=float).reshape(-1, 2, 3)
    dev = []
    for i in np.flatnonzero(bsok):
        P = projector(covbs(wid[i], slo[i]))
        dev.append(np.max(np.abs(mb[i] + P)))
    q('max |Jpsi_bsmeanbs + P|', dev)

    # ---------------- G7 ----------------
    if doff is not None and 'Jpsi_covvtx' in doff:
        print('\n=== G7 the leave-one-out identity against the rows-OFF fit')
        sa, sb = match(d, doff)
        offx = np.column_stack([np.asarray(doff[k], dtype=float)
                                for k in ('Jpsi_x', 'Jpsi_y', 'Jpsi_z')])[sb]
        offc = np.asarray(doff['Jpsi_covvtx'], dtype=float).reshape(-1, 6)[sb]
        dz, dr = [], []
        for j, (i, _) in enumerate(zip(sa, sb)):
            if not bsok[i]:
                continue
            C = covbs(wid[i], slo[i])
            P = projector(C)
            Cv = unpack6(offc[j])
            r_off = P @ (offx[j] - spot[i])
            Cov_off = P @ (Cv + C) @ P.T
            L = np.linalg.cholesky(Cov_off)
            z_off = np.linalg.solve(L, r_off)
            sg = np.sqrt(np.diag(Cov_off))
            dr.append(np.max(np.abs(res[i] - r_off)/sg))
            dz.append(np.max(np.abs(z[i] - z_off)))
        q('|r_bs(ON) - r_bs(OFF)| / sigma', dr)
        q('|z_bs(ON) - z_bs(OFF)|', dz)
        # the covariance itself
        dc = []
        for j, i in enumerate(sa):
            if not bsok[i]:
                continue
            C = covbs(wid[i], slo[i])
            P = projector(C)
            Cv = unpack6(offc[j])
            Cov_off = P @ (Cv + C) @ P.T
            Cov_on = np.array([[cov[i, 0], cov[i, 1]], [cov[i, 1], cov[i, 2]]])
            dc.append(np.max(np.abs(Cov_on - Cov_off))/np.max(np.abs(Cov_off)))
        q('|Cov(ON) - Cov(OFF)| / |Cov|', dc)

    # ---------------- G4 ----------------
    if a.old:
        dold, _ = load(a.old, a.max)
        if a.half:
            print('\n=== G4a the OLD build == the NEW build at beamWidthScale=1/sqrt(2)')
            dh, _ = load(a.half, a.max)
            w = cmp_trees(dh, dold, 'half vs old')
            print(f'  {"PASS -- the rows DID enter twice" if w < 1e-4 else "CHECK"}')
        print('\n=== G4b the SIZE of the double-emission defect (nominal vs old)')
        cmp_trees(d, dold, 'on vs old')

    # ---------------- the distribution, for orientation ----------------
    print('\n=== the two pulls (orientation only; the study does the real work)')
    for k, nm in ((0, 'z_bs,x'), (1, 'z_bs,y')):
        v = z[bsok, k]
        v = v[np.isfinite(v)]
        lo, hi = np.percentile(v, [0.1, 99.9])
        t = v[(v > lo) & (v < hi)]
        print(f'  {nm}: n {v.size}  mean {v.mean():+.4f}  Var {v.var():.4f}  '
              f'trimVar {t.var():.4f}  '
              f'P(|z|>3) {np.mean(np.abs(v) > 3):.5f}  '
              f'P(|z|>5) {np.mean(np.abs(v) > 5):.6f}')
    print('\n=== the family composition of the two beam functionals')
    for nm in ('Jpsi_bsvbs', 'Jpsi_bsvhit', 'Jpsi_bsvms', 'Jpsi_bsvioni'):
        v = np.asarray(d[nm], dtype=float).reshape(-1, 2)[bsok]
        print(f'  {nm:16s} x {np.nanmean(v[:,0]):.4f}   y {np.nanmean(v[:,1]):.4f}')
    print(f'  Jpsi_massvbs     {np.nanmean(np.asarray(d["Jpsi_massvbs"],dtype=float)[bsok]):.6f}')
    print(f'  Jpsi_vtxvbs      {np.nanmean(np.asarray(d["Jpsi_vtxvbs"],dtype=float)[bsok]):.6f}')


if __name__ == '__main__':
    main()
