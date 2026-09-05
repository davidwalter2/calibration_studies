#!/usr/bin/env python3
"""Summarise one or more runCvhProfile.py profiling points.

Reads, per run directory:
  time.log  -- /usr/bin/time -v (wall, user, sys, maxrss)
  run.log   -- CMSSW: TimeReport / Timing service TimeEvent> lines, the maker's
               own end-of-job summaries (fit / propagator / pixel hit quality)
  *.root    -- the per-candidate tree (niter, nParms, nhits, ...)

Per-event wall time (TimeEvent>) is joined to the tree on (run, lumi, event)
so the cost can be attributed per candidate rather than per event.
"""
import argparse, glob, json, os, re, sys
import numpy as np

def _f(m, i=1, cast=float):
    return cast(m.group(i)) if m else None

def parse_time(path):
    out = {}
    if not os.path.exists(path): return out
    txt = open(path, errors='replace').read()
    m = re.search(r"Elapsed \(wall clock\) time.*?:\s*([\d:.]+)", txt)
    if m:
        p = [float(x) for x in m.group(1).split(':')]
        out['wall_s'] = p[0]*3600+p[1]*60+p[2] if len(p) == 3 else p[0]*60+p[1]
    out['user_s'] = _f(re.search(r"User time \(seconds\):\s*([\d.]+)", txt))
    out['sys_s']  = _f(re.search(r"System time \(seconds\):\s*([\d.]+)", txt))
    out['maxrss_mb'] = (_f(re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", txt)) or 0)/1024.
    return out

FIT_RE = re.compile(r"fit summary\s+attempted=(\d+)\s+succeeded=(\d+)\s+failed=(\d+)")
PROP_RE = re.compile(r"propagateGenericWithJacobianAltD summary\s+calls=(\d+)\s+failures=(\d+)")
PIX_RE = re.compile(r"pixel hit-quality summary\s+seen=(\d+)")

def parse_run(path):
    out = {'timeevent': []}
    if not os.path.exists(path): return out
    for line in open(path, errors='replace'):
        if line.startswith('TimeEvent>'):
            # TimeEvent> <eventnum> <runnum> <timetaken>
            p = line.split()
            try: out['timeevent'].append((int(p[2]), int(p[1]), float(p[3])))
            except Exception: pass
            continue
        if line.startswith('TimeModule>'):
            # TimeModule> <eventnum> <runnum> <label> <type> <timetaken>
            p = line.split()
            try: out.setdefault('timemodule', []).append((int(p[2]), int(p[1]), p[3], float(p[5])))
            except Exception: pass
            continue
        for key, rx, names in (
            ('fit', FIT_RE, ('attempted', 'succeeded', 'failed')),
            ('prop', PROP_RE, ('calls', 'failures')),
            ('pix', PIX_RE, ('seen',))):
            m = rx.search(line)
            if m:
                for i, n in enumerate(names): out[f'{key}_{n}'] = int(m.group(i+1))
        for tag, rx in (('nglobalparms', r"nglobalparms = (\d+)"),
                        ('nmodes', r"scalar-potential field correction: (\d+) modes"),
                        ('nmatgroups', r"global material model: (\d+) groups"),
                        ('backwardLegs', r"backwardLegs=(\d+)"),
                        ('clamped', r"clamped\[step\]=(\d+)"),
                        ('backtracked', r"backtracked\[step\]=(\d+)"),
                        ('btchi2', r"btchi2\[step\]=(\d+)")):
            m = re.search(rx, line)
            if m: out[tag] = int(m.group(1))
        m = re.search(r"Total loop:\s*([\d.]+)", line)
        if m: out['total_loop_s'] = float(m.group(1))
        m = re.search(r"Total init\s*:\s*([\d.]+)", line)
        if m: out['total_init_s'] = float(m.group(1))
        m = re.search(r"Avg event:\s*([\d.]+)", line)
        if m: out['avg_event_s'] = float(m.group(1))
    # per-module TimeReport
    mods = {}
    txt = open(path, errors='replace').read()
    for m in re.finditer(r"^TimeReport\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)\s+(\S+)\s*$",
                         txt, re.M):
        mods[m.group(6)] = float(m.group(1))   # per-event-run CPU
    if mods: out['modules'] = mods
    return out

def parse_tree(rundir):
    import uproot
    files = sorted(glob.glob(os.path.join(rundir, '*.root')))
    if not files: return {}
    f = files[0]
    out = {'root_bytes': os.path.getsize(f), 'root_file': os.path.basename(f)}
    try:
        t = uproot.open(f)['tree']
    except Exception as e:
        out['tree_err'] = str(e); return out
    out['ncand'] = t.num_entries
    if not t.num_entries: return out
    want = ['niter', 'nParms', 'ndof', 'chisqval', 'run', 'lumi', 'event',
            'Muplus_nvalidFinal', 'Muminus_nvalidFinal', 'Muplus_pt', 'Muminus_pt',
            'Muplus_eta', 'Muminus_eta', 'Jpsi_mass', 'Jpsi_pt']
    have = [w for w in want if w in t.keys()]
    a = t.arrays(have, library='np')
    for k in ('niter', 'nParms', 'ndof'):
        if k in a: out[f'{k}_mean'] = float(np.mean(a[k])); out[f'{k}_med'] = float(np.median(a[k]))
    if 'niter' in a:
        ni = np.asarray(a['niter'], dtype=int)
        out['niter_atcap_frac'] = float(np.mean(ni >= 10))
        out['niter_hist'] = np.bincount(ni, minlength=11)[:12].tolist()
    for k in ('Muplus_nvalidFinal', 'Muminus_nvalidFinal'):
        if k in a: out[f'{k}_mean'] = float(np.mean(a[k]))
    if 'Muplus_nvalidFinal' in a:
        out['nhits_per_track'] = float(np.mean(np.concatenate(
            [a['Muplus_nvalidFinal'], a['Muminus_nvalidFinal']])))
    out['_arrays'] = a
    return out

def summarise(rundir):
    r = {'tag': os.path.basename(rundir.rstrip('/'))}
    r.update(parse_time(os.path.join(rundir, 'time.log')))
    r.update(parse_run(os.path.join(rundir, 'run.log')))
    tr = parse_tree(rundir)
    arrays = tr.pop('_arrays', None)
    r.update(tr)
    args = os.path.join(rundir, 'args.txt')
    if os.path.exists(args): r['args'] = open(args).read().strip().replace('\n', ' | ')
    te = r.pop('timeevent', [])
    tm = r.pop('timemodule', [])
    if tm:
        from collections import defaultdict
        bymod = defaultdict(float)
        for _run, _ev, lab, t in tm: bymod[lab] += t
        r['module_tot'] = dict(bymod)
        # per-(run,event) maker time
        mk = {}
        for _run, _ev, lab, t in tm:
            if lab == 'trackrefitdimuon': mk[(_run, _ev)] = mk.get((_run, _ev), 0.) + t
        r['_makertime'] = mk
        if mk:
            v = np.array(list(mk.values()))
            r['maker_tot_s'] = float(v.sum())
    if te:
        te = np.array(te, dtype=float)              # run, event, time
        r['nevents'] = len(te)
        r['loop_wall_s'] = float(te[:, 2].sum())
        # drop the first event: it carries the one-off Geant4/geometry init
        srt = te[np.argsort(-te[:, 2])]
        r['first_event_s'] = float(te[0, 2])
        rest = te[1:, 2] if len(te) > 1 else te[:, 2]
        r['s_per_event'] = float(rest.mean())
        r['s_per_event_med'] = float(np.median(rest))
        # attribute per-event wall to candidates
        if arrays is not None and 'event' in arrays:
            key = dict(r.get('_makertime') or {})
            if not key:
                key = {(int(te[i, 0]), int(te[i, 1])): te[i, 2] for i in range(len(te))}
            ev = list(zip(np.asarray(arrays['run'], dtype=np.int64),
                          np.asarray(arrays['event'], dtype=np.int64)))
            from collections import Counter
            cnt = Counter(ev)
            per_cand, matched = [], 0
            for k, n in cnt.items():
                if k in key:
                    matched += n
                    per_cand.extend([key[k] / n] * n)
            if per_cand:
                pc = np.array(per_cand)
                # Geant4 builds its energy-loss / MS tables lazily on the FIRST
                # real propagation, so the first candidate-bearing event of every
                # job carries a fixed ~10 s that belongs to init, not to the fit.
                # Drop it (identified as the first entry above 10x the median).
                if len(pc) > 4:
                    med0 = np.median(pc)
                    bad = np.where(pc > 10 * med0)[0]
                    if len(bad):
                        r['init_lazy_s'] = float(pc[bad[0]])
                        pc = np.delete(pc, bad[0])
                r['s_per_cand'] = float(pc.mean())
                r['s_per_cand_med'] = float(np.median(pc))
                r['s_per_cand_p90'] = float(np.percentile(pc, 90))
                r['s_per_cand_max'] = float(pc.max())
                r['cand_matched'] = matched
                # what fraction of the total is carried by the worst 5%?
                s = np.sort(pc)[::-1]
                k5 = max(1, int(0.05 * len(s)))
                r['top5pct_share'] = float(s[:k5].sum() / s.sum())
                r['_percand'] = pc
        if 'ncand' in r and r.get('nevents'):
            r['cand_per_event'] = r['ncand'] / r['nevents']
    if r.get('prop_calls') and r.get('ncand'):
        r['propcalls_per_cand'] = r['prop_calls'] / r['ncand']
    if r.get('root_bytes') and r.get('ncand'):
        r['bytes_per_cand'] = r['root_bytes'] / r['ncand']
    return r

COLS = [('tag', '{:<16s}'), ('nevents', '{:>7.0f}'), ('ncand', '{:>6.0f}'),
        ('cand_per_event', '{:>6.2f}'), ('s_per_event', '{:>8.2f}'),
        ('s_per_cand', '{:>8.2f}'), ('s_per_cand_med', '{:>8.3f}'),
        ('s_per_cand_p90', '{:>8.2f}'), ('s_per_cand_max', '{:>8.1f}'),
        ('top5pct_share', '{:>6.2f}'), ('niter_mean', '{:>6.2f}'),
        ('niter_atcap_frac', '{:>7.3f}'), ('propcalls_per_cand', '{:>7.1f}'),
        ('nhits_per_track', '{:>6.2f}'), ('nParms_mean', '{:>8.1f}'),
        ('nmodes', '{:>6.0f}'), ('bytes_per_cand', '{:>10.0f}'),
        ('wall_s', '{:>8.0f}'), ('maxrss_mb', '{:>8.0f}')]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--json', default='')
    ap.add_argument('--modules', action='store_true', help='print per-module TimeReport')
    a = ap.parse_args()
    rows = []
    for d in a.dirs:
        if not os.path.isdir(d): continue
        rows.append(summarise(d))
    hdr = ''.join(f.replace('.2f', 's').replace('.3f', 's').replace('.1f', 's')
                  .replace('.0f', 's').format(c[:len(c)]) for c, f in COLS)
    print(hdr)
    for r in rows:
        line = ''
        for c, f in COLS:
            v = r.get(c)
            line += (f.format(v) if v is not None else
                     f.replace('.2f', 's').replace('.3f', 's').replace('.1f', 's')
                      .replace('.0f', 's').format('-'))
        print(line)
    if a.modules:
        for r in rows:
            if 'modules' in r:
                print(f"\n-- {r['tag']} per-module CPU/event (TimeReport):")
                for k, v in sorted(r['modules'].items(), key=lambda x: -x[1])[:8]:
                    print(f"   {v:10.4f}  {k}")
    if a.json:
        clean = [{k: v for k, v in r.items() if not k.startswith('_')} for r in rows]
        json.dump(clean, open(a.json, 'w'), indent=1, default=str)

if __name__ == '__main__':
    main()
