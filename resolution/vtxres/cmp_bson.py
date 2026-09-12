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
        'Jpsi_massvbs', 'Jpsi_vtxvbs', 'Jpsi_fang',
        'Jpsi_sigmarelplus', 'Jpsi_sigmarelminus', 'Jpsi_rhomom',
        'Jpsigen_mass', 'Muplusgen_pt', 'Muminusgen_pt',
        'Muplus_nvalid', 'Muminus_nvalid']


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
    """(run, lumi, event, rank by Jpsi_pt within the event).

    `Jpsi_pt` itself cannot be part of the key: the two regimes are DIFFERENT
    FITS, so their fitted pT differ in the last digits and a value-based key
    matches nothing.  The ORDER within an event is stable.
    """
    run = np.asarray(d['run'], np.int64)
    lumi = np.asarray(d['lumi'], np.int64)
    evt = np.asarray(d['event'], np.int64)
    pt = np.asarray(d['Jpsi_pt'], float)
    if 'Muplus_nvalid' in d:
        nvp = np.asarray(d['Muplus_nvalid'], np.int64)
        nvm = np.asarray(d['Muminus_nvalid'], np.int64)
        base = np.array([f'{r}:{l}:{e}:{a}:{b}'
                         for r, l, e, a, b in zip(run, lumi, evt, nvp, nvm)])
    else:
        base = np.array([f'{r}:{l}:{e}' for r, l, e in zip(run, lumi, evt)])
    rank = np.zeros(len(base), np.int64)
    order = np.lexsort((-pt, base))
    prev, k = None, 0
    for i in order:
        if base[i] != prev:
            prev, k = base[i], 0
        rank[i] = k
        k += 1
    return np.char.add(np.char.add(base, ':'), rank.astype(str))


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


MINLEG = 8   # the gen-background study's baseline (STATE_bkg.md)


def sel(d):
    """The SAME selection on both regimes -- that is the point of it.

    On top of the vtxres extraction's cuts it adds a FINITENESS check and a
    minimum of 8 valid hits on the weaker leg: a leg of <= 3-4 hits can give
    an infinite `Jpsi_sigmamass` and an absurd sigma_v with every flag true.
    Both are cuts on the FIT'S OWN covariance, never on a residual.
    """
    ok = np.asarray(d['Jpsi_vtxok'], bool)
    ok &= np.isfinite(np.asarray(d['Jpsi_sigmamass'], float))
    ok &= np.isfinite(np.asarray(d['Jpsi_vtxsig'], float))
    if 'Muplus_nvalid' in d:
        ok &= np.minimum(np.asarray(d['Muplus_nvalid'], np.int64),
                         np.asarray(d['Muminus_nvalid'], np.int64)) >= MINLEG
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
    # A per-candidate shift is EXPECTED: conditioning on a new measurement
    # moves the estimate, and its spread must be sqrt(sigma_OFF^2 -
    # sigma_ON^2).  What matters for a mass measurement is the ENSEMBLE MEAN.
    dm = (mon - mof)*1e3
    pred = np.sqrt(np.maximum(smof**2 - smon**2, 0.))*1e3
    okd = np.isfinite(dm) & np.isfinite(pred) & (pred > 0)
    print(f'  rms(m_ON - m_OFF) = {dm[okd].std():.2f} MeV against the predicted '
          f'{np.sqrt(np.mean(pred[okd]**2)):.2f} MeV '
          f'(= sqrt(sigma_OFF^2 - sigma_ON^2))')
    print(f'  MEAN m(ON) - m(OFF) = {dm[okd].mean():+.3f} +- '
          f'{dm[okd].std()/np.sqrt(okd.sum()):.3f} MeV   '
          f'(relative {dm[okd].mean()/np.mean(mon[okd])/1e3:+.3e})')

    # ---------------- THE MECHANISM ----------------
    if 'Jpsi_sigmarelplus' in don and 'Jpsi_sigmarelplus' in dof:
        print('\n=== where the gain comes from')
        for b, lab in (('Jpsi_sigmarelplus', 'sigma(p)/p, mu+'),
                       ('Jpsi_sigmarelminus', 'sigma(p)/p, mu-')):
            x = np.asarray(don[b], float)[ia]
            y = np.asarray(dof[b], float)[ib]
            m2 = np.isfinite(x) & np.isfinite(y) & (y > 0)
            r2 = x[m2]/y[m2]
            print(f'  {lab}: ON/OFF median {np.median(r2):.5f}  mean {r2.mean():.5f}'
                  f'   (OFF median {np.median(y[m2]):.5f})')
        if 'Jpsi_fang' in dof:
            fa = np.asarray(dof['Jpsi_fang'], float)[ib]
            fa = fa[np.isfinite(fa)]
            print(f'  the ANGULAR share of sigma_m^2 (Jpsi_fang, rows OFF): '
                  f'median {np.median(fa):.5f}, mean {fa.mean():.5f} '
                  f'-- so the gain is on the CURVATURES, not the opening angle')
        if all(k in don and k in dof for k in
               ('Jpsi_sigmarelplus', 'Jpsi_sigmarelminus', 'Jpsi_rhomom')):
            def vmass(d, idx):
                sp = np.asarray(d['Jpsi_sigmarelplus'], float)[idx]
                sm = np.asarray(d['Jpsi_sigmarelminus'], float)[idx]
                rr = np.asarray(d['Jpsi_rhomom'], float)[idx]
                return 0.25*(sp**2 + sm**2 + 2*rr*sp*sm)
            vo, vn = vmass(dof, ib), vmass(don, ia)
            m3 = np.isfinite(vo) & np.isfinite(vn) & (vo > 0)
            pr = np.sqrt(vn[m3]/vo[m3])
            me = (smon/smof)[m3]
            print(f'  predicted sigma_m ratio from the two curvatures alone: '
                  f'median {np.median(pr):.5f}, mean {pr.mean():.5f}')
            print(f'  measured                                             : '
                  f'median {np.median(me):.5f}, mean {me.mean():.5f}')

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
    nf = int(f.sum())

    def report(lab, resp, unit, scale=1.0, ref=None):
        """resp = d(functional)/d(parameter) per candidate, already multiplied
        by the parameter shift.  THE TWO NUMBERS ARE DIFFERENT QUESTIONS:

          <resp>   the ENSEMBLE MEAN -- a BIAS on the functional, which is
                   what a mass measurement feels.  The per-candidate response
                   depends on the pair's phi, so it largely cancels and this
                   is what has to be compared with the target.
          rms      the per-candidate SPREAD -- it adds in quadrature to the
                   resolution, it does not move the scale.
        """
        v = resp[f]*scale
        mu, sd = float(np.mean(v)), float(np.std(v))
        err = sd/np.sqrt(max(nf, 1))
        line = (f'    {lab:<26} mean {mu:+11.5g} +- {err:.3g} {unit}   '
                f'rms {sd:10.5g}   mean|.| {float(np.mean(np.abs(v))):10.5g}')
        if ref is not None:
            r = ref[f]
            line += (f'   [mean/sigma {float(np.mean(v/r)):+.6f}, '
                     f'rms/sigma {float(np.std(v/r)):.6f}]')
        print(line)

    print(f'  a 5 um shift of the beam-spot CENTROID ({nf} candidates)')
    report('mass, d x0 [MeV]', mm[:, 0]*d5*1e3, 'MeV', ref=smon*1e3)
    report('mass, d y0 [MeV]', mm[:, 1]*d5*1e3, 'MeV', ref=smon*1e3)
    report('mass, d x0 [rel]', mm[:, 0]*d5/np.maximum(mon, 1e-9), '')
    report('mass, d y0 [rel]', mm[:, 1]*d5/np.maximum(mon, 1e-9), '')
    report('vertex r_v / sigma_v, d x0', mv[:, 0]*d5/np.maximum(sv, 1e-30), '')
    report('vertex r_v / sigma_v, d y0', mv[:, 1]*d5/np.maximum(sv, 1e-30), '')
    report('beam z_bs,x, d x0', mb[:, 0, 0]*d5/np.maximum(sbx, 1e-30), '')
    report('beam z_bs,y, d y0', mb[:, 1, 1]*d5/np.maximum(sby, 1e-30), '')
    print(f'  a 1e-4 change of the SLOPE (response = the same weight x (z_v - z0))')
    report('mass, d dxdz [MeV]', mm[:, 0]*zv*ds*1e3, 'MeV', ref=smon*1e3)
    report('mass, d dydz [MeV]', mm[:, 1]*zv*ds*1e3, 'MeV', ref=smon*1e3)
    report('mass, d dxdz [rel]', mm[:, 0]*zv*ds/np.maximum(mon, 1e-9), '')
    report('vertex r_v / sigma_v, d dxdz', mv[:, 0]*zv*ds/np.maximum(sv, 1e-30), '')
    report('beam z_bs,x, d dxdz', mb[:, 0, 0]*zv*ds/np.maximum(sbx, 1e-30), '')
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
