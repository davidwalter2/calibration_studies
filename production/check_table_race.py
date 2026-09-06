#!/usr/bin/env python3
"""Per-task outlier scan for the shared-Geant4-table race (NOTES 2026-09-06 (V) sec.6).

The race that could corrupt QUIETLY -- two threads building the two process-wide
G4TablesForExtrapolatorForCVH objects under two DIFFERENT mutexes -- happens at
JOB STARTUP and would leave a wrong dE/dx table in place for the WHOLE task. Its
signature is therefore a task-level offset, not a per-candidate one, and the
cheapest observable is the mean reconstructed mass of the task relative to the
generated one: a wrong mean energy loss moves the momentum scale.

    python3 check_table_race.py <outbase> [--stream-glob 'globalcor_*.root']

Prints, per task, <m_reco> and <m_reco - m_gen> with their errors, then flags
tasks more than `--nsig` from the ensemble median in units of their own error.
A task that lost the race stands out as a whole; scattered candidate-level
outliers are NOT this mechanism.
"""
import argparse, glob, os, sys
import numpy as np
import uproot

ap = argparse.ArgumentParser()
ap.add_argument("outbase")
ap.add_argument("--stream-glob", default="globalcor_*.root")
ap.add_argument("--nsig", type=float, default=5.0)
ap.add_argument("--ntasks", type=int, default=0, help="0 = all")
ap.add_argument("--mass-branch", default="Jpsikin_mass")
ap.add_argument("--gen-branch", default="Jpsigen_mass")
a = ap.parse_args()

tasks = sorted(glob.glob(os.path.join(a.outbase, "task_*")))
tasks = [t for t in tasks if os.path.exists(os.path.join(t, ".complete"))]
if a.ntasks:
    tasks = tasks[: a.ntasks]
print(f"{len(tasks)} complete tasks under {a.outbase}")

rows = []
for t in tasks:
    vals, gvals = [], []
    for f in sorted(glob.glob(os.path.join(t, a.stream_glob))):
        try:
            tr = uproot.open(f)["tree"]
        except Exception as e:
            print(f"  [skip] {f}: {e}")
            continue
        keys = set(tr.keys())
        if a.mass_branch not in keys:
            continue
        arrs = tr.arrays([a.mass_branch] + ([a.gen_branch] if a.gen_branch in keys else []),
                         library="np")
        vals.append(arrs[a.mass_branch])
        if a.gen_branch in arrs:
            gvals.append(arrs[a.gen_branch])
    if not vals:
        continue
    v = np.concatenate(vals)
    v = v[np.isfinite(v)]
    if v.size < 50:
        continue
    d = None
    if gvals:
        g = np.concatenate(gvals)
        m = np.isfinite(v) & np.isfinite(g)
        d = (v - g)[m]
    rows.append((os.path.basename(t), v.size, v.mean(), v.std(ddof=1) / np.sqrt(v.size),
                 None if d is None else d.mean(),
                 None if d is None else d.std(ddof=1) / np.sqrt(d.size)))

if not rows:
    sys.exit("no usable tasks")
mu = np.array([r[2] for r in rows])
er = np.array([r[3] for r in rows])
med = np.median(mu)
pull = (mu - med) / er
print(f"\nensemble median <{a.mass_branch}> = {med:.6f}   "
      f"task-to-task RMS = {mu.std(ddof=1):.6f}   median err = {np.median(er):.6f}")
print(f"pull (task - median)/err :  RMS = {pull.std(ddof=1):.2f}   max|pull| = {np.abs(pull).max():.2f}")
bad = [(rows[i][0], mu[i], pull[i]) for i in np.argsort(-np.abs(pull)) if abs(pull[i]) > a.nsig]
if bad:
    print(f"\n{len(bad)} task(s) beyond {a.nsig} sigma -- inspect these:")
    for n, m, p in bad[:40]:
        print(f"  {n}  <m> = {m:.6f}  pull = {p:+.1f}")
else:
    print(f"\nno task beyond {a.nsig} sigma: no evidence of a task-level table corruption")
