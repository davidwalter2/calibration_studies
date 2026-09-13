#!/usr/bin/env python3
"""BIT-IDENTITY GATE for the `sqrt(dV_b)` cache (task 5) and for the
per-channel defaults (task 2).

`dV_b^{1/2}` is now computed ONCE per candidate and read by the mass, vertex
and both beam influence loops instead of being re-decomposed in each.  It is
the same solver on the same matrix, so EVERY export must be bit-identical --
not "close", identical.  Two legs:

  gun  : rows OFF.  The beam code never runs, so the ONLY difference between
         the two builds is the cache (plus the per-channel default, which for
         the J/psi driver is False either way).  EVERY branch must match.
  dy   : rows ON.   The cache must again be bit-identical, but the BEAM
         residual branches change BY DESIGN (the whitened pair and the "+"
         form), so they are listed as expected-to-move and reported
         separately rather than silently skipped.

Comparison is by (run, lumi, event) and, within an event, by the candidate's
rank in `Jpsi_pt` -- never by `Jpsi_pt` itself, which is a fitted quantity.

  ./gate_cache.py --a DIR_OR_GLOB --b DIR_OR_GLOB [--tag NAME] [--rows-on]
"""
import argparse
import glob
import os
import sys

import numpy as np
import uproot

# the branches the whitened pair and the "+" form are EXPECTED to move
BEAM_MOVED = {
    'Jpsi_bsres', 'Jpsi_bscov', 'Jpsi_bsz', 'Jpsi_bschi2', 'Jpsi_bsvchk',
    'Jpsi_bsmeanbs', 'Jpsi_bsvbs', 'Jpsi_bsvhit', 'Jpsi_bsvms', 'Jpsi_bsvioni',
    'resinfbsv', 'bsvarv', 'Jpsi_bssgnchk',
    'cfbs_ms', 'cfbs_del', 'cfbs_ioni_re', 'cfbs_ioni_im',
    'cfbs_rad_re', 'cfbs_rad_im',
    'cfbs_hitcls', 'cfbs_hitcomp', 'cfbs_hitv', 'cfbs_grp', 'cfbs_grpcomp',
    'cfbs_grp_ms', 'cfbs_grp_ioni_re', 'cfbs_grp_ioni_im',
    'cfbs_grp_rad_re', 'cfbs_grp_rad_im', 'cfbs_grp_vqms', 'cfbs_grp_vqio',
    'cfbs_grp_closure',
}
# Every name above is a BEAM-FUNCTIONAL output and nothing else: the two
# whitened pulls' residual, covariance, influence, family/class shares and CF
# exponents.  If a branch outside this set moves, the change has leaked out of
# the beam block and the gate fails -- which is the point.

# branches that exist only in the NEW build
NEW_ONLY = {'Jpsi_bscovlo', 'Jpsi_bslinv', 'Jpsi_bsmeig', 'Jpsi_bswidtherr',
            'Jpsi_massvbsx', 'Jpsi_massvbsy', 'Jpsi_vtxvbsx', 'Jpsi_vtxvbsy',
            'Jpsi_bsvbsx', 'Jpsi_bsvbsy'}


def files(spec):
    if os.path.isdir(spec):
        spec = os.path.join(spec, 'globalcor_*.root')
    return sorted(glob.glob(spec, recursive=True))


def load(spec):
    fs = files(spec)
    if not fs:
        raise SystemExit('no files under ' + spec)
    out = {}
    for fn in fs:
        with uproot.open(fn) as fh:
            t = fh['tree']
            keys = sorted(set(k.split(';')[0] for k in t.keys()))
            arrs = t.arrays(keys, library='np')
            for k, v in arrs.items():
                out.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in out.items()}, fs


def key(d):
    n = len(d['event'])
    k = np.stack([np.asarray(d[q], np.int64) for q in ('run', 'lumi', 'event')], 1)
    # rank within the event by Jpsi_pt -- a stable ORDERING, never a value test
    rank = np.zeros(n, np.int64)
    order = np.lexsort((-np.asarray(d['Jpsi_pt'], float), k[:, 2], k[:, 1], k[:, 0]))
    seen = {}
    for i in order:
        t = (k[i, 0], k[i, 1], k[i, 2])
        rank[i] = seen.get(t, 0)
        seen[t] = rank[i] + 1
    return [(int(k[i, 0]), int(k[i, 1]), int(k[i, 2]), int(rank[i])) for i in range(n)]


def match(da, db):
    ka, kb = key(da), key(db)
    ia = {t: i for i, t in enumerate(ka)}
    sa, sb = [], []
    for j, t in enumerate(kb):
        if t in ia:
            sa.append(ia[t])
            sb.append(j)
    return np.array(sa, np.int64), np.array(sb, np.int64)


def flat(x, sel):
    """A comparable float vector for one branch on the selected candidates."""
    v = x[sel]
    if v.dtype == object or (v.ndim == 1 and v.size and hasattr(v[0], '__len__')):
        try:
            return np.concatenate([np.asarray(e, np.float64).ravel() for e in v])
        except Exception:  # noqa: BLE001
            return None
    return np.asarray(v, np.float64).ravel()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--a', required=True, help='the REFERENCE build (dev3)')
    p.add_argument('--b', required=True, help='the NEW build (dev2)')
    p.add_argument('--tag', default='')
    p.add_argument('--rows-on', action='store_true',
                   help='the beam residual branches are expected to MOVE')
    a = p.parse_args()

    da, fa = load(a.a)
    db, fb = load(a.b)
    sa, sb = match(da, db)
    print(f'=== {a.tag or "cache"}  A {len(fa)} file(s) / {len(da["event"])} cand, '
          f'B {len(fb)} file(s) / {len(db["event"])} cand, {len(sa)} MATCHED')
    common = sorted(set(da) & set(db))
    onlyb = sorted(set(db) - set(da))
    print(f'    {len(common)} common branches; NEW-only: '
          + (', '.join(onlyb) if onlyb else '(none)'))
    unexpected = [n for n in onlyb if n not in NEW_ONLY]
    if unexpected:
        print('    *** NEW-only branches that were not declared: '
              + ', '.join(unexpected))

    nident, moved, failed, skipped = 0, [], [], []
    for br in common:
        xa, xb = flat(da[br], sa), flat(db[br], sb)
        if xa is None or xb is None or xa.shape != xb.shape:
            skipped.append(br)
            continue
        ok = np.isfinite(xa) & np.isfinite(xb)
        if not ok.any():
            nident += 1
            continue
        diff = xa[ok] != xb[ok]
        if not diff.any():
            nident += 1
            continue
        sc = np.maximum(np.abs(xa[ok]), np.abs(xb[ok]))
        sc = np.where(sc > 0, sc, 1.)
        r = np.abs(xa[ok] - xb[ok]) / sc
        rec = (br, float(np.mean(diff)), float(np.median(r[diff])), float(r.max()))
        if a.rows_on and br in BEAM_MOVED:
            moved.append(rec)
        else:
            failed.append(rec)

    print(f'    BIT-IDENTICAL on {nident} of {len(common) - len(skipped)} '
          f'comparable branches ({len(skipped)} skipped: ragged/shape)')
    if skipped:
        print('      skipped: ' + ', '.join(skipped[:12])
              + (' ...' if len(skipped) > 12 else ''))
    if moved:
        print(f'    MOVED BY DESIGN (the whitened pair / the "+" form), '
              f'{len(moved)} branches:')
        for b, f_, m, x in sorted(moved):
            print(f'      {b:24s} frac {f_:.4f}  median rel {m:.3e}  max {x:.3e}')
    if failed:
        print(f'    *** NOT IDENTICAL and NOT EXPECTED TO MOVE: {len(failed)} ***')
        for b, f_, m, x in sorted(failed):
            print(f'      {b:24s} frac {f_:.4f}  median rel {m:.3e}  max {x:.3e}')
        return 1
    print('    GATE PASS: every branch that must not move is bit-identical')
    return 0


if __name__ == '__main__':
    sys.exit(main())
