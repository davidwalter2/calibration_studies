#!/usr/bin/env python3
"""Same-candidate comparison of the two-track fit with the BEAM-LINE
(luminous-region) rows ON and OFF.

The two productions read the SAME MiniAOD files with the SAME fit settings
and differ only in `bsConstraint`, so every candidate can be matched on
(run, lumi, event, Jpsi_pt) and the constraint's effect read off candidate by
candidate.

What is compared
  * `sigma_m(ON)/sigma_m(OFF)` -- what the beam rows buy the MASS, and its
    dependence on the softer muon's GEN pT.
  * `Jpsi_vtxres` / `Jpsi_vtxsig` / `Jpsi_vtxz` -- the VERTEX residual, which
    must be UNCHANGED: the beam line constrains the vertex POSITION, the DCA
    is a different direction of the same 10-dim vertex state.
  * `ndof(ON) - ndof(OFF)`, which must be exactly 3.
  * the LEAVE-ONE-OUT identity: the beam residual the ON export reports
    against the one built directly from the OFF fit's vertex and covariance.
  * THE MEAN TERM: the size of the response of the mass, the vertex residual
    and the two beam residuals to a 5 um centroid shift and a 1e-4 slope
    change, read off the exported influence weights.

usage:
  python3 cmp_bson.py --on '<dy_bs>/task_*/globalcor_*.root' \
                      --off '<dy_vtxon>/task_*/globalcor_*.root' [--max N]
"""
import argparse, glob, sys
import numpy as np
import uproot

SCAL = ['Jpsi_vtxres', 'Jpsi_vtxsig', 'Jpsi_vtxz', 'Jpsi_vtxok', 'Jpsi_vtxvchk',
        'Jpsi_vtxdchi2', 'Jpsi_vtxvhit', 'Jpsi_vtxvms', 'Jpsi_vtxvioni',
        'Jpsi_mass', 'Jpsi_sigmamass', 'Jpsi_mass_unc', 'Jpsi_covmassvtx',
        'Jpsi_x', 'Jpsi_y', 'Jpsi_z', 'Jpsi_covvtx',
        'Jpsi_pt', 'Jpsi_eta', 'Jpsigen_mass', 'chisqval', 'ndof',
        'Muplusgen_pt', 'Muminusgen_pt', 'Muplusgen_eta', 'Muminusgen_eta',
        'cfmass_ok', 'run', 'lumi', 'event',
        'Jpsi_bsres', 'Jpsi_bscov', 'Jpsi_bsz', 'Jpsi_bschi2', 'Jpsi_bschi2fit',
        'Jpsi_bsok', 'Jpsi_bsvchk', 'Jpsi_bsvtx', 'Jpsi_bsspot', 'Jpsi_bswidth',
        'Jpsi_bsslope', 'Jpsi_bsmeanmass', 'Jpsi_bsmeanvtx', 'Jpsi_bsmeanbs',
        'Jpsi_massvbs', 'Jpsi_vtxvbs']


def load(pattern, nmax=None):
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
            want = [b for b in SCAL if b in keys]
            a = t.arrays(want, library='np')
            for k, v in a.items():
                out.setdefault(k, []).append(v)
            got += len(a['run'])
        if nmax and got >= nmax:
            break
    if not out:
        raise SystemExit(f'no entries under {pattern}')
    d = {k: np.concatenate(v)[:nmax] for k, v in out.items()}
    for k, nc in (('Jpsi_bsres', 2), ('Jpsi_bscov', 3), ('Jpsi_bsz', 2),
                  ('Jpsi_bsvtx', 3), ('Jpsi_bsspot', 3), ('Jpsi_bswidth', 3),
                  ('Jpsi_bsslope', 2), ('Jpsi_bsmeanmass', 3),
                  ('Jpsi_bsmeanvtx', 3), ('Jpsi_bsmeanbs', 6),
                  ('Jpsi_covvtx', 6)):
        if k in d:
            d[k] = np.asarray(d[k], float).reshape(-1, nc)
    return d, files


def key(d):
    return np.array([f'{r}:{l}:{e}:{p:.6f}' for r, l, e, p
                     in zip(d['run'], d['lumi'], d['event'], d['Jpsi_pt'])])


def match(da, db):
    ka, kb = key(da), key(db)
    ua, ia = np.unique(ka, return_index=True)
    ub, ib = np.unique(kb, return_index=True)
    c = np.intersect1d(ua, ub)
    return ia[np.searchsorted(ua, c)], ib[np.searchsorted(ub, c)]


def covbs(width, slope):
    s1, s2, s3 = width
    v1, v2, v3 = s1*s1, s2*s2, s3*s3
    C = np.zeros((3, 3))
    C[0, 0], C[1, 1], C[2, 2] = v1, v2, v3
    C[2, 0] = C[0, 2] = slope[0]*(v3 - v1)
    C[2, 1] = C[1, 2] = slope[1]*(v3 - v2)
    return C


def projector(C):
    P = np.zeros((2, 3))
    P[0, 0] = P[1, 1] = 1.
    P[0, 2] = -C[0, 2]/C[2, 2]
    P[1, 2] = -C[1, 2]/C[2, 2]
    return P


def unpack6(v):
    return np.array([[v[0], v[1], v[2]], [v[1], v[3], v[4]], [v[2], v[4], v[5]]])


def sel(d):
    ok = np.asarray(d['Jpsi_vtxok'], bool)
    ok &= np.asarray(d.get('cfmass_ok', np.ones(len(ok), bool)), bool)
    ok &= np.asarray(d['Jpsi_vtxsig'], float) > 0
    ok &= np.asarray(d['Jpsi_sigmamass'], float) > 0
    nd = np.asarray(d['ndof'], float)
    ok &= nd > 0
    ok &= np.asarray(d['chisqval'], float)/np.maximum(nd, 1) < 3.
    ok &= np.abs(np.asarray(d['Jpsi_vtxvchk'], float)) < 1e-4
    return ok


def q(name, x, unit=''):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        print(f'  {name:46s}  (empty)')
        return
    print(f'  {name:46s}  median {np.median(x):.4g}  p90 {np.percentile(x,90):.4g}  '
          f'max {np.max(x):.4g} {unit}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--on', required=True)
    ap.add_argument('--off', required=True)
    ap.add_argument('--max', type=int, default=None)
    a = ap.parse_args()

    don, fon = load(a.on, a.max)
    dof, fof = load(a.off, a.max)
    ia, ib = match(don, dof)
    print(f'# ON {len(don["run"])} candidates ({len(fon)} files), '
          f'OFF {len(dof["run"])} ({len(fof)} files), matched {len(ia)}')
    son, sof = sel(don), sel(dof)
    keep = son[ia] & sof[ib]
    ia, ib = ia[keep], ib[keep]
    bsok = np.asarray(don['Jpsi_bsok'], bool)[ia]
    print(f'# selected on BOTH: {len(ia)}; of those Jpsi_bsok {bsok.mean()*100:.3f} %')

    # ---------------- ndof ----------------
    dn = np.asarray(don['ndof'], float)[ia] - np.asarray(dof['ndof'], float)[ib]
    u, c = np.unique(dn, return_counts=True)
    print('\n=== ndof(ON) - ndof(OFF): ' + ', '.join(f'{int(x)} x{int(y)}'
                                                     for x, y in zip(u, c)))

    # ---------------- the MASS ----------------
    smon = np.asarray(don['Jpsi_sigmamass'], float)[ia]
    smof = np.asarray(dof['Jpsi_sigmamass'], float)[ib]
    r = smon/smof
    g = np.isfinite(r) & (r > 0) & (r < 2)
    print(f'\n=== sigma_m(ON)/sigma_m(OFF)  n {g.sum()}')
    print(f'  mean {r[g].mean():.6f}  median {np.median(r[g]):.6f}  '
          f'-> {100*(1-r[g].mean()):.3f} % in sigma, '
          f'{100*(1-r[g].mean()**2):.3f} % in variance')
    ptg = np.minimum(np.asarray(don['Muplusgen_pt'], float)[ia],
                     np.asarray(don['Muminusgen_pt'], float)[ia])
    ok = g & np.isfinite(ptg) & (ptg > 0)
    ed = np.unique(np.percentile(ptg[ok], np.linspace(0, 100, 9)))
    print('  vs softer muon GEN pT:')
    for i in range(len(ed)-1):
        m = ok & (ptg >= ed[i]) & (ptg < ed[i+1])
        if m.sum() < 20:
            continue
        print(f'    [{ed[i]:6.1f}, {ed[i+1]:6.1f}) GeV  n {m.sum():6d}  '
              f'ratio {r[m].mean():.6f}  ({100*(1-r[m].mean()):.3f} %)')
    mon = np.asarray(don['Jpsi_mass'], float)[ia]
    mof = np.asarray(dof['Jpsi_mass'], float)[ib]
    q('|m(ON) - m(OFF)| [MeV]', np.abs(mon-mof)*1e3, 'MeV')

    # ---------------- the VERTEX residual ----------------
    print('\n=== the VERTEX residual must be UNCHANGED')
    for b, lab in (('Jpsi_vtxres', 'r_v'), ('Jpsi_vtxsig', 'sigma_v'),
                   ('Jpsi_vtxz', 'z_v'), ('Jpsi_vtxvhit', 'vhit'),
                   ('Jpsi_vtxvms', 'vms')):
        x = np.asarray(don[b], float)[ia]
        y = np.asarray(dof[b], float)[ib]
        m = np.isfinite(x) & np.isfinite(y)
        sc = np.maximum(np.abs(x[m]), np.abs(y[m]))
        sc = np.where(sc > 0, sc, 1.)
        q(f'|{lab}(ON)-{lab}(OFF)| / max|.|', np.abs(x[m]-y[m])/sc)
    zv_on = np.asarray(don['Jpsi_vtxz'], float)[ia]
    zv_of = np.asarray(dof['Jpsi_vtxz'], float)[ib]
    m = np.isfinite(zv_on) & np.isfinite(zv_of)
    print(f'  max |z_v(ON) - z_v(OFF)| = {np.max(np.abs(zv_on[m]-zv_of[m])):.4g} '
          f'(median {np.median(np.abs(zv_on[m]-zv_of[m])):.4g})')
    print(f'  Var(z_v) ON {np.var(zv_on[m]):.5f}  OFF {np.var(zv_of[m]):.5f}')

    # ---------------- the LEAVE-ONE-OUT identity ----------------
    print('\n=== the leave-one-out identity against the rows-OFF fit')
    if 'Jpsi_covvtx' in dof:
        res = don['Jpsi_bsres'][ia]
        zz = don['Jpsi_bsz'][ia]
        cv = don['Jpsi_bscov'][ia]
        wid = don['Jpsi_bswidth'][ia]
        slo = don['Jpsi_bsslope'][ia]
        spot = don['Jpsi_bsspot'][ia]
        offx = np.column_stack([np.asarray(dof[k], float)[ib]
                                for k in ('Jpsi_x', 'Jpsi_y', 'Jpsi_z')])
        offc = dof['Jpsi_covvtx'][ib]
        dr, dz, dc = [], [], []
        for j in np.flatnonzero(bsok):
            C = covbs(wid[j], slo[j])
            P = projector(C)
            Cv = unpack6(offc[j])
            r_off = P @ (offx[j] - spot[j])
            Cov_off = P @ (Cv + C) @ P.T
            try:
                L = np.linalg.cholesky(Cov_off)
            except np.linalg.LinAlgError:
                continue
            z_off = np.linalg.solve(L, r_off)
            sg = np.sqrt(np.diag(Cov_off))
            dr.append(np.max(np.abs(res[j] - r_off)/sg))
            dz.append(np.max(np.abs(zz[j] - z_off)))
            Con = np.array([[cv[j, 0], cv[j, 1]], [cv[j, 1], cv[j, 2]]])
            dc.append(np.max(np.abs(Con - Cov_off))/np.max(np.abs(Cov_off)))
        q('|r_bs(ON) - r_bs(OFF)| / sigma', dr)
        q('|z_bs(ON) - z_bs(OFF)|', dz)
        q('|Cov(ON) - Cov(OFF)| / |Cov|', dc)
    else:
        print('  (the OFF tree has no Jpsi_covvtx -- rerun it with the new build)')

    # ---------------- THE MEAN TERM ----------------
    print('\n=== the beam-spot MEAN TERM (the numbers that decide whether the '
          'beam-spot parameters must float)')
    d5 = 5e-4          # 5 um, in cm
    ds = 1e-4          # a slope change
    zv = don['Jpsi_bsvtx'][ia][:, 2] - don['Jpsi_bsspot'][ia][:, 2]
    mm = don['Jpsi_bsmeanmass'][ia]      # d m / d (x0, y0, z0), GeV/cm
    mv = don['Jpsi_bsmeanvtx'][ia]       # d r_v / d (x0, y0, z0), cm/cm
    mb = don['Jpsi_bsmeanbs'][ia].reshape(-1, 2, 3)
    sv = np.asarray(don['Jpsi_vtxsig'], float)[ia]
    bcov = don['Jpsi_bscov'][ia]
    sbx = np.sqrt(np.maximum(bcov[:, 0], 0.))
    sby = np.sqrt(np.maximum(bcov[:, 2], 0.))
    f = bsok
    print(f'  a 5 um shift of the beam-spot CENTROID (x0 then y0):')
    print(f'    mass  |dm| = {np.abs(mm[f,0]).mean()*d5*1e3:.5f} / '
          f'{np.abs(mm[f,1]).mean()*d5*1e3:.5f} MeV (mean), '
          f'p99 {np.percentile(np.abs(mm[f,0])*d5*1e3,99):.5f} MeV; '
          f'in units of sigma_m: {np.abs(mm[f,0]*d5/smon[f]).mean():.5f}')
    print(f'    vertex |dr_v|/sigma_v = {np.abs(mv[f,0]*d5/sv[f]).mean():.5f} / '
          f'{np.abs(mv[f,1]*d5/sv[f]).mean():.5f}')
    print(f'    beam   |dz_bs,x| = {np.abs(mb[f,0,0]*d5/sbx[f]).mean():.5f}, '
          f'|dz_bs,y| = {np.abs(mb[f,1,1]*d5/sby[f]).mean():.5f}')
    print(f'  a 1e-4 change of the SLOPE (dxdz then dydz), response = weight x (z_v - z0):')
    print(f'    mass  |dm| = {np.abs(mm[f,0]*zv[f]).mean()*ds*1e3:.5f} / '
          f'{np.abs(mm[f,1]*zv[f]).mean()*ds*1e3:.5f} MeV; '
          f'in sigma_m {np.abs(mm[f,0]*zv[f]*ds/smon[f]).mean():.5f}')
    print(f'    vertex |dr_v|/sigma_v = {np.abs(mv[f,0]*zv[f]*ds/sv[f]).mean():.5f}')
    print(f'    beam   |dz_bs,x| = {np.abs(mb[f,0,0]*zv[f]*ds/sbx[f]).mean():.5f}, '
          f'|dz_bs,y| = {np.abs(mb[f,1,1]*zv[f]*ds/sby[f]).mean():.5f}')
    print(f'  <|z_v - z0|> = {np.abs(zv[f]).mean():.4f} cm')
    print(f'  <sigma_m> = {smon[f].mean()*1e3:.2f} MeV, '
          f'<sigma_bs,x> = {sbx[f].mean()*1e4:.2f} um, '
          f'<sigma_bs,y> = {sby[f].mean()*1e4:.2f} um')

    # ---------------- the beam block's share of the other functionals ----
    print('\n=== the luminous region as a share of the other two functionals')
    print(f'  <Jpsi_massvbs> = {np.nanmean(np.asarray(don["Jpsi_massvbs"],float)[ia][f]):.6f}')
    print(f'  <Jpsi_vtxvbs>  = {np.nanmean(np.asarray(don["Jpsi_vtxvbs"],float)[ia][f]):.6f}')


if __name__ == '__main__':
    main()
