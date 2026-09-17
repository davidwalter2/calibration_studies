#!/usr/bin/env python3
"""THE CONCATENATED-TAU-GRID GATE.

The four resolution-CF functionals of a two-track candidate (mass, vertex DCA,
the two whitened beam-line pulls) now share ONE `cvhcf::trackExponents` pass on
the concatenated argument list { w_{b,k} tau_j } instead of costing one pass
each.  The trick is EXACT -- `phi_{aU}(tau) = phi_U(a tau)`, and the products
`w tau` are formed by the same expression the single-functional path forms them
with -- so this compares the two builds branch by branch and demands:

  (a) every `cfmass_*` / `cfvtx_*` / `cfbs_*` array agrees (reported per
      family: max |dx| and max relative |dx|/|x|);
  (b) EVERY OTHER BRANCH is bit-identical -- the change must not touch the fit,
      the influence vectors, the variance shares or any closure;
  (c) the candidates line up one to one on (run, lumi, event).

Usage:
  ./cmp_taugrid.py --old '<dir>/globalcor_*.root' --new '<dir>/globalcor_*.root'
"""
import argparse, glob, sys
import numpy as np
import uproot

FAM = [('cfmass_', 'mass'), ('cfvtx_', 'vertex'), ('cfbs_', 'beam'),
       ('cfqop_', 'qop')]  # cfqop_ is the SINGLE-track maker's, i.e. the n = 1 path


def famof(name):
    for pre, tag in FAM:
        if name.startswith(pre):
            return tag
    return 'other'


def load(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f'no files matching {pattern}')
    trees = []
    for f in files:
        fh = uproot.open(f)
        trees.append((f, fh))
    return trees


def arrays(trees, keys):
    out = {}
    for f, fh in trees:
        t = fh['tree']
        got = t.arrays([k for k in keys if k in set(x.split(';')[0] for x in t.keys())],
                       library='np')
        for k, v in got.items():
            out.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


def flat(a):
    """A jagged object array of per-entry vectors -> (values, counts)."""
    if a.dtype == object:
        cnt = np.array([len(x) for x in a], dtype=np.int64)
        if cnt.sum() == 0:
            return np.zeros(0), cnt
        return np.concatenate([np.asarray(x).ravel() for x in a]), cnt
    a = np.asarray(a)
    if a.ndim == 1:
        return a, np.ones(len(a), dtype=np.int64)
    return a.ravel(), np.full(len(a), a.shape[1], dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old', required=True)
    ap.add_argument('--new', required=True)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()

    told, tnew = load(a.old), load(a.new)
    kold = set()
    for f, fh in told:
        kold |= set(k.split(';')[0] for k in fh['tree'].keys())
    knew = set()
    for f, fh in tnew:
        knew |= set(k.split(';')[0] for k in fh['tree'].keys())
    only_old, only_new = sorted(kold - knew), sorted(knew - kold)
    keys = sorted(kold & knew)

    print(f'=== {a.tag or "taugrid"} gate ===')
    print(f'branches: {len(keys)} common, {len(only_old)} old-only, {len(only_new)} new-only')
    if only_old:
        print('  OLD ONLY:', ' '.join(only_old))
    if only_new:
        print('  NEW ONLY:', ' '.join(only_new))

    ao, an = arrays(told, keys), arrays(tnew, keys)
    n = len(ao.get('run', []))
    if n != len(an.get('run', [])):
        print(f'FAIL: entry count {n} vs {len(an.get("run", []))}')
        return 1
    for k in ('run', 'lumi', 'event'):
        if k in ao and not np.array_equal(ao[k], an[k]):
            print(f'FAIL: {k} does not line up')
            return 1
    print(f'candidates: {n}')

    stats = {}
    bad_other = []
    nz_counts = {}
    skipped, nint = [], 0
    for k in keys:
        if k not in ao or k not in an:
            continue
        vo, co = flat(ao[k])
        vn, cn = flat(an[k])
        if not np.array_equal(co, cn):
            print(f'FAIL {k}: jagged counts differ')
            bad_other.append((k, 'counts'))
            continue
        if vo.dtype.kind not in 'fiub' or len(vo) == 0:
            skipped.append(k)
            continue
        if vo.dtype.kind in 'iub':
            nint += 1
            if not np.array_equal(vo, vn):
                bad_other.append((k, 'int mismatch'))
            continue
        vo = vo.astype(np.float64)
        vn = vn.astype(np.float64)
        # NaN is a VALUE here (`edmvalref` is NaN on every candidate), so
        # bit-identity has to say NaN == NaN.
        bad = np.isnan(vo) & np.isnan(vn)
        d = np.where(bad, 0., np.abs(vn - vo))
        scale = np.maximum(np.abs(vo), np.abs(vn))
        rel = np.where(scale > 0, d / np.where(scale > 0, scale, 1.), 0.)
        rel = np.where(np.isfinite(rel), rel, 0.)
        d = np.where(np.isfinite(d), d, 0.)
        nbit = int(np.count_nonzero((vn != vo) & ~bad))
        fam = famof(k)
        s = stats.setdefault(fam, dict(maxabs=0., maxrel=0., nbit=0, nval=0,
                                       worst='', nbr=0, brbad=[]))
        s['maxabs'] = max(s['maxabs'], float(d.max()))
        if float(rel.max()) > s['maxrel']:
            s['maxrel'] = float(rel.max())
            s['worst'] = k
        s['nbit'] += nbit
        s['nval'] += len(vo)
        s['nbr'] += 1
        if nbit:
            s['brbad'].append((k, nbit, float(d.max()), float(rel.max())))
        if fam == 'other' and nbit:
            bad_other.append((k, f'{nbit}/{len(vo)} values differ, max |d| {d.max():.3e}'))
        nz_counts[k] = int(np.count_nonzero(vo))

    print(f'{"family":8s} {"branches":>9s} {"values":>12s} {"non-bit-id":>11s} '
          f'{"max |d|":>11s} {"max rel":>10s}  worst branch')
    for fam in ('mass', 'vertex', 'beam', 'qop', 'other'):
        if fam not in stats:
            continue
        s = stats[fam]
        print(f'{fam:8s} {s["nbr"]:9d} {s["nval"]:12d} {s["nbit"]:11d} '
              f'{s["maxabs"]:11.3e} {s["maxrel"]:10.3e}  {s["worst"]}')
        for k, nb, da, dr in sorted(s['brbad'], key=lambda x: -x[3])[:8]:
            print(f'    {k:26s} {nb:8d} differ  max|d| {da:.3e}  max rel {dr:.3e}')

    print(f'integer branches compared exactly: {nint}; '
          f'skipped (empty or non-numeric): {len(skipped)}'
          + (' -> ' + ' '.join(skipped) if skipped else ''))
    print()
    if bad_other:
        print('FAIL: non-CF branches changed:')
        for k, why in bad_other:
            print(f'  {k}: {why}')
        return 1
    print('PASS (b): every non-CF branch is bit-identical')
    worst = max((stats[f]['maxrel'] for f in ('mass', 'vertex', 'beam', 'qop') if f in stats),
                default=0.)
    nbit = sum(stats[f]['nbit'] for f in ('mass', 'vertex', 'beam', 'qop') if f in stats)
    if nbit == 0:
        print('PASS (a): every CF array is BIT-IDENTICAL '
              '(the products w*tau are formed by the same expression)')
    elif worst <= 1e-6:
        print(f'PASS (a): CF arrays agree to {worst:.3e} relative '
              f'({nbit} values differ in their last bits)')
    else:
        print(f'FAIL (a): CF arrays differ by {worst:.3e} relative')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
