# `jpsimc_20M_260906_v2` — the J/psi re-production, on HTCondor

Companion to `production/STATE.md` (the slurm `jpsimc_20M_260905` it replaces),
`condor_dymc_v2/STATE_dy_v2.md` (the thread scan and the grid mechanics, which
are not repeated here) and `production/PRODUCTION_NEXT.md` (what the new
exports are for). Same 1642-chunk list, same task index -> same event range.

| | |
|---|---|
| tag | `jpsimc_20M_260906_v2` |
| output | `/ceph/submit/data/user/d/david_w/ZMass/cvh/jpsimc_20M_260906_v2/task_XXXX/globalcor_<stream>.root` |
| chunks | `condor_jpsimc_v2/chunks_jpsimc_20M_260906_v2.txt` (1642, 410 files, ~20 M events) |
| CMSSW | `CMSSW_15_0_19_patch2_dev2` @ `fab515e` (`cvh-exports-260906`) |
| driver | `Analysis/HitAnalyzer/test/runCvhJpsiGenMC.py` |
| new switches | `exportCfGroupExponents=True exportMaterialNoise=True exportVarianceGrads=True varianceGradFamilies=15` |
| threads / memory | **4 / 5000 MB** |
| batch | HTCondor -> CMS global pool |

---

## 1. THE INPUT IS ADDRESSED BY DOOR, NEVER BY LFN

This is the one thing that makes the J/psi leg different from DY, and getting
it wrong would not have produced an error — it would have produced a wrong
sample that looked right.

The J/psi inputs are **our own repacked split-1 copies**, written into the
submit group store under the CMS `/store` namespace. **That same LFN exists
centrally, with different content** — the un-repacked split-99 original.
Measured 2026-09-06 on the first file of the list, from a grid slot at DESY and
from submit82:

| door | size served |
|---|---:|
| `root://submit50.mit.edu/` (and 51–55) | **1 783 846 343 B — ours, split-1** |
| `root://cms-xrd-global.cern.ch/` | 1 840 069 145 B — **the central original** |
| `root://xrootd.cmsaf.mit.edu/` (T2 redirector) | `[3011] Too many attempts to gain dfs read access` |
| `root://t2dsk0011.cmsaf.mit.edu/` | `[3011] Too many DFS read attempts` |

So the DY leg's trick — map the ceph path to an LFN and let the global
redirector find a replica — **is not safe here**. It would feed the production
files whose split level makes the CVH refit fail on ~99 % of candidates
(`project_jpsi_mc_repack_split1`), silently, with every output still valid on
inspection. The MIT **T2** door does not serve this tree at all, because it is
the T3/submit CephFS and not the T2 store.

**Readability from the grid: YES.** Three probe jobs at DESY
(`scratch_condorprobe_260906/probe_jpsi.sh`) all report `OK (OURS)` for
`submit50` and `submit51`, for both the `/store/...` form and the
`/data/group/cms/store/...` form, and for the re-staged copies in the user
store; and all three read 10 MB of real bytes back through `xrdcp`. No export
request, no staging, nothing to ask anyone for.

The mapping the wrapper uses is therefore a **pure prefix strip against a named
door** — submit's xrootd namespace root *is* `/ceph/submit`:

```
/ceph/submit/X   ->   root://submit5N.mit.edu//X
```

exact, no redirector in the path, and uniform across the 407 group-store files
and the 12 chunk lines repointed at the 3 re-staged copies in the user store.

**Two guards on top of that:**

1. the chunk list carries each file's **size as a fourth field**, taken on the
   ceph copy at submit time, and the wrapper **refuses any door that serves a
   different size** (exit 8, `WRONG REPLICA`). If the door layout ever changes
   under us, the job dies before it processes an event instead of producing a
   plausible wrong answer;
2. the door is chosen as `(IDX + k) mod 6` over submit50–55, falling through to
   the next door if one does not answer — so 1642 jobs do not pull ~740 GB
   through a single login node (~14 MB/s per door over the production), and one
   sick door does not stall the leg.

---

## 2. Sizing

From the thread scan (`condor_dymc_v2/STATE_dy_v2.md` §1), J/psi arm:

| threads | wall (s) | CPU (s) | speedup | peak RSS (GB) |
|---:|---:|---:|---:|---:|
| 1 | 2039.9 | 2028.0 | 1.00 | 1.36 |
| 2 | 1024.7 | 2003.4 | 1.99 | 1.45 |
| 4 | 535.5 | 2020.6 | **3.81** | 1.61 |
| 8 | 286.4 | 2024.2 | 7.12 | 1.89 |

`wall(N) = 36.0 + 2003.9/N` to 1.3 %; total CPU **falls 0.2 %** from 1 to 8
threads. On a real 12 185-event chunk with the new exports that projects to
3.40 h -> **0.86 h at 4 threads (3.97x)**.

`request_memory = 5000` MB: memory is set by the single-task TAIL, not the core
count — ~1.28 GB shared plus ~76 MB per stream, against a 1-thread MaxRSS of
median 1.9 / p99 3.1 / **max 3.56 GB** over the 1616 tasks of
`jpsimc_20M_260905`. `3.56 + 3 x 0.076 = 3.79`, x1.3 = 4.9 GB. That is **less
memory per task than the slurm leg's 6 G, with four times the cores**.

`request_disk = 6 000 000` KB: 64 MB payload + 228 MB unpacked area + ~856 MB
of output (`4 x 13.09 MB` runtree + `12 148 x 66.2 kB`). Total volume ~1.4 TB
over 1642 tasks.

---

## 3. Validation, 2026-09-06

One chunk, 400 events (398 J/psi candidates), the production switch set:

| run | where | input | wall |
|---|---|---|---:|
| reference | local dev2, submit82, 1 thread | POSIX `/ceph` | — |
| grid | condor, **Wisconsin** (`g32n02.hep.wisc.edu`), 4 threads | `root://submit50.mit.edu/`, size guard passed | 431 s |

```
local(1 thr) vs grid(4 thr)  PASS  all 228 branches bit-identical over 398 candidates
runtree                      126 452 entries, identical in every stream file
```

Bit-identity across the batch system, the site, xrootd-through-a-door versus a
POSIX read, and 1 versus 4 streams — with the door reporting the repacked
size, i.e. demonstrably the right file.

---

## 4. Submission and first-half-hour throughput

**Cluster `3803264`, 1642 jobs, submitted 2026-09-06 15:51:16, 4 threads,
`request_memory = 5000` MB.**

| t + | running | idle | held | complete |
|---:|---:|---:|---:|---:|
| 5 min | 91 | 1551 | 0 | 0 |
| 10 min | 149 | 1493 | 0 | 0 |
| 15 min | 222 | 1420 | 0 | 0 |
| 20 min | 271 | 1371 | 0 | 0 |
| 25 min | 342 | 1300 | 0 | 0 |
| **31 min** | **498** | 1144 | **0** | 0 |
| 35 min | 558 | 1084 | 0 | 0 |

over 14+ sites — DESY 165, Caltech 70, INFN Legnaro 58, MIT T2 71, KISTI 31,
CIEMAT 27, INFN Roma 23, Budapest 10, RAL 8, Pisa 6, IHEP 6, Wisconsin 5,
NCHC 5. The ramp is slower than the DY leg's (which reached 379/380 in 31 min)
for two reasons that are both expected: it is 4.3x more jobs, and the DY leg
was concurrently holding ~200-300 slots of the same share. Nothing was held.

No task had completed at t+31 min, which is right: a full J/psi chunk is 12 185
events, i.e. ~1.5 h at 4 threads on a grid CPU, against DY's 22 375 events of a
cheaper channel.

### Failures in the first half hour

| what | n | where | verdict |
|---|---:|---|---|
| **SIGILL (rc 132)** | 36 | **34 on five `ultralight.org` machines + 2 on `t2bat0310`** — `compute-6-34` alone ate 16 | black-hole NODES, fenced |
| `no door served <path>` | 1 | task_0106 | **transient** |

The SIGILL here crashes in `edm::StreamSchedule::fillWorkers` (the DY leg's
were inside `XrdCl` on the `stagein` Prepare) — different call site, same
meaning: **the node cannot run the release's binaries at all**. It concentrates
on six machines while the rest of Caltech and MIT T2 ran hundreds of jobs
cleanly, so the fence is by NODE, not by site — applied with `condor_qedit` to
both live clusters and written into both `config_*.sh`.

The one door failure is genuinely transient: all six doors returned "no answer"
for `1239FEDB-…root` inside that job's 180 s window, and both `submit50` and
`submit53` serve it correctly now at the expected 1 696 464 519 B. Condor
retries it. **The size guard did not fire once** — no door has ever served a
wrong replica.

### DY v2 at the same moment (t+87 min for that leg)

`dymc_8p5M_260906_v2`: **164 of 380 complete (43.2 %)**, 211 running, 0 idle,
0 held, at **177.8 tasks/h**, projected to finish 17:22 — i.e. **1.2 h left
against the slurm leg's 17.5 h at 11.4 tasks/h. A factor 15.6 on rate.**

Five DY tasks (0000, 0002, 0003, 0260, 0293) exhausted their four starts, all
with **SIGSEGV (rc 139)**, at four different sites. See §6.

---

## 5. Operating it

```bash
cd calibration_studies/production/condor_jpsimc_v2
./submit_jpsimc_v2.sh --dry-run
./submit_jpsimc_v2.sh                    # all 1642
./status_jpsimc_v2.sh [--slow]           # --slow also checks streams/task
./resume_jpsimc_v2.sh --dry-run
```

Identical contract to `condor_dymc_v2`: completion is the `.complete` sentinel
alone, `resume` never re-queues an index already in the queue, and the payload
is pinned inside the production's own output tree so a later resume runs the
same binary.

## 6. The SIGSEGV in `fillRadiativeSpectrum` — open

Five of 380 DY v2 tasks (1.3 %) died with `rc=139`, each after exhausting all
four starts, at DESY, Caltech, Wisconsin and INFN-LNL. The stack is the same
every time:

```
G4MuPairProductionModel::ComputeDMicroscopicCrossSection
  <- Geant4ePropagator::fillRadiativeSpectrum
  <- Geant4ePropagator::propagateGenericWithJacobianAltD
```

**It is NOT a deterministic per-event crash, and it is NOT the new exports.**
Task 0000's chunk — the same file, `skipEvents=0 nEvents=22120`, the same
`EXTRA` including `exportCfGroupExponents=True` — was re-run locally from dev2
and completed **22 120 / 22 120 events, rc=0**
(`/work/submit/david_w/ZMass/scratch_segv_260906/grpON/`). So a bad candidate
that always crashes is excluded.

What is left is a rare, intermittent fault in that call path. The leading
hypothesis is a **thread-safety problem**: `fillRadiativeSpectrum` reaches into
a G4 model whose state may not be properly per-thread, which would be
intermittent, possibly stream-count dependent, and invisible to the
bit-identity gate (when it does not crash, the numbers are exactly right — that
is what §1 of `STATE_dy_v2.md` measured). A 4-thread local re-run of the same
chunk, matching the grid configuration, was in flight when this was written;
its result is the next thing to look at.

**Consequences for the sample.** Retries do not recover these tasks — 4 starts
each were spent. `resume_dymc_v2.sh` picks exactly them (it reported
`to resubmit=5`) and a resubmission has whatever per-attempt probability of
surviving; that is the pragmatic mitigation until the cause is found. The loss
is 1.3 % of tasks and it is not obviously random in kinematics, so **it should
not be waved through for a calibration sample** — but it is also identifiable
task by task and re-runnable once fixed.

The J/psi leg had **zero** rc=139 in its first half hour.

---

## 7. If the grid route had NOT worked

It did, so this is not the plan — but `production/array_jpsimc_v2.sbatch`
exists as the slurm fallback and as the recommended *slurm* configuration
going forward: `--cpus-per-task=4 --mem=5G --time=4:00:00` from dev2. **It must
not be submitted while the `jpsimc20M_*` arrays are in `squeue`** — it would
only queue behind them for the same fair share.

---

## 8. 2026-09-07 — the four bad inputs, repaired (cluster `3803425`)

Sixteen chunks were re-run from proper split-1 repacks. Four separate defects
came out of it, three of them previously invisible.

### 8.1 The "recovery" of 2026-09-06 made things worse, not better

The 12 chunk lines that §1 of `production/STATE.md` describes as "recovered by
re-fetching to a user area" were repointed at files pulled from
`root://cms-xrd-global.cern.ch/`. **That is the central, UN-REPACKED original.**
`splitlevels.py` on all three:

```
split=99  nsub=2   SiStripClusteredmNewDetSetVector_ALCARECOTkAlJpsiMuMu__RECO.
    sub split=98   ...present / ...obj
```

against `split=1 / sub-0` on a correctly repacked file. Their sizes are
byte-identical to what the redirector serves, which is the proof: they *are*
the originals. So the fix for a corrupt provenance blob re-introduced
ROOT #19773, which is the very thing the whole repack campaign exists to avoid.
This is exactly the trap `config_jpsimc_v2.sh` warns about at length — and it
was walked into anyway, because a redirector was used to *repair* an input
rather than to *read* one.

### 8.2 What the split-99 chunks produced: 0.82 cand/event, and 100 % of the survivors are garbage

| | control (1223, 1224, 1338, 1413) | split-99 (the 12) |
|---|---:|---:|
| candidates / event | **0.9968** | **0.8242** |
| fit failures | 0.00–0.01 % | **17.4 %**, all `fail[prop]` |
| `clampevents[step]` | 8 | **116 113** |
| propagator calls | 1.39 M | **5.75 M** |
| chi2/ndof median | 0.955 | **3.5e6** |
| chi2/ndof > 10 | 0.011 % | **100.000 %** |
| `niter == 10` (the limit) | 1.42 % | **99.81 %** |
| m(mumu) in [2.9, 3.3] | 98.47 % | **2.90 %** |
| gen match lost | 0.017 % | **88.63 %** |
| valid strip hits (mu+) | 14.898 | 15.005 |
| `nvalidFinal == nvalid` | 100 % | 100 % |
| **input** `Jpsitrk_mass` | 3.08768 ± 0.00024 | 3.08834 ± 0.00016 |

Read the last three rows together: **every pre-refit quantity agrees and every
post-CVH quantity is destroyed.** The fit did not reject the corrupted hits —
it kept all of them (`nvalidFinal == nvalid` for 100 % of tracks) and paid for
them in the residuals, at an effective per-hit pull of **1860 sigma** against
0.98 in the control. The 17 % that fail are only the ones that also hit a hard
propagation abort; the 83 % that survive are non-converged fits written out at
the iteration limit. The 250 candidates with `niter < 10` are no better
(median chi2/ndof 4.6e6). **There is no salvage cut.** All ~128.7 k candidates
were discarded.

The 12 outputs are kept as the evidence, one level down in
`task_XXXX/bad_split99_260906/`, where the single-level `task_*/globalcor_*.root`
globs every downstream script uses cannot reach them.

### 8.3 The rc=91 of tasks 1552-1555: our repack corrupted its own output in memory

`70000/03249796-...root` is **correctly split-1** and its `Events` tree is
intact (49 516 entries) — so this was never the same fault as §8.1. cmsRun dies
constructing `PoolSource`:

```
FormatIncompatibility ... EntryError can not convert representation of
   Recommend2D: <34 NUL bytes> to value of type vector<string_hex>
```

Characterised byte by byte:

* 28 of the 8147 `ParameterSets` entries carry **52 runs of NUL, 2153 bytes**;
* each run replaces a pset value's *content* **with its length exactly
  preserved** (`Recommend2D`'s encoded default is exactly 34 bytes);
* the zeroing is confined to payload lengths **32–46 bytes** — 47 of 17781 in
  that band, **0 of 30228 outside it** — i.e. `std::string` lengths just above
  libstdc++'s 15-char SSO threshold, the signature of a freed-and-reused small
  allocation;
* every basket **decompresses cleanly**, so the NULs are in the *uncompressed*
  payload: they were written that way.

So this is **write-side memory corruption inside our own July-2026 repack
job**, not disk rot and not transfer damage. `edmProvDump` is clean on the
central original and fails on our copy. The same signature is present in the
other three group-store repacks (32/42/33 entries), which is why *they* threw
exit 91 too — the 2026-09-06 note's "the MIT copies rotted" is the right
observation with the wrong mechanism.

### 8.4 The repair

`production/repack_fix_260907/` — `repack_worker.sh` (the 10_6 `splitLevel=0`
repack in the el7 container, tmp-then-promote), `run_repack_fix.sh`,
`fetch_and_repack_03249796.sh`, `verify_repacks.sh`, `repoint_and_quarantine.sh`,
`validate_refit.sh`, `scan_pset_nulls.py`, `scan_all_inputs.sh`, `scan_fitfail.sh`.

The three staged files needed **no second download**: they already were the
central originals, so they were repacked in place. Only `03249796` was re-fetched
(1 986 290 360 B, size-verified, not exit-code-verified).

Output: `/ceph/submit/data/user/d/david_w/ZMass/restaged/jpsimc_20M_260906_repack/`
The old directory is now `restaged_split99_DO_NOT_USE/` with a README; nothing
was deleted and the group store was not touched.

| file | new size | entries | split | provdump | NUL scan |
|---|---:|---:|:-:|:-:|:-:|
| `BDA060EF-...` | 1 872 652 106 | 51 231 | 1 | OK | clean |
| `0909778B-...` | 2 055 629 896 | 56 267 | 1 | OK | clean |
| `4B9D2D77-...` | 1 770 114 676 | 48 621 | 1 | OK | clean |
| `03249796-...` | 1 801 876 175 | 49 516 | 1 | OK | clean |

Entry counts equal exactly what the chunk list tiles. All six doors serve them
at the recorded sizes, so the 4th-field size guard passes.

**Acceptance gate** — 2000 events, dev2 @ `ca6058d`, the production switch set,
4 threads, against the same run on a known-good group-store file:

| | repack `BDA060EF` | control `BDB523D0` |
|---|---:|---:|
| candidates / event | **0.9995** | 0.9955 |
| fit failures | **0.000 %** | 0.050 % |
| `clampevents[step]` | 0 | 7 |
| chi2/ndof median | 0.9638 | 0.9489 |
| chi2/ndof > 10 | 0.0000 % | 0.0000 % |
| `niter == 10` | 1.35 % | 1.86 % |
| m(mumu) median | 3.0948 | 3.0958 |

### 8.5 Two bugs found on the way, both fixed here

**`recover_xrdfix.sh` could never have submitted.** It appended `REDIR`, which
`config_jpsimc_v2.sh` does not define (only the DY config does), so under
`set -u` it died *after* writing `.recover_idx.txt` and *before* `condor_submit`
— which is why the 1552-1555 recovery attempted on 2026-09-06 left an index
file and no jobs. It also never passed `INDOORS`, so a recovered job would have
fallen back to the single default door. Both fixed. (The comment explaining it
is *above* `condor_submit`: a `#` inside a backslash-continued argument list
silently truncates the command.)

**The `attempted=0` guard has a hole, and one task fell through it.**
`task_1313` staged four empty stream files and a `.complete` for a whole
**19 797-event chunk**. Its input, `2830000/FDB8C946-...root`, has **no
StreamerInfo**: ROOT opens it and reports 19 797 entries, so the 2026-09-05
zombie scan passed it, but CMSSW cannot deserialise it and `PoolSource` drops
it —

```
Input file: ... was not found or could not be opened, and will be skipped.
```

The maker then never runs, so **no `fit summary` line is written at all** and
`grep -q "fit summary  attempted=0 "` matches nothing. Both wrappers now also
fail on the skip message (exit 6) and on the *absence* of a summary (exit 5).
**Task 1313 has NOT been re-run** — that is a fifth file and a 17th chunk,
outside this pass; it needs the same repack treatment.

### 8.6 Sample-wide integrity: the rest of the production is clean

Two cheap scans were run over everything, and both say the damage is contained.

`scan_pset_nulls.py` over all **410 production inputs** (ParameterSets is a few
MB even for a 2 GB ALCARECO, so this is minutes, not hours):

| verdict | n | which |
|---|---:|---|
| OK | 403 | |
| NULS, and cmsRun died (exit 91) | 4 | the four repaired here |
| NULS, but cmsRun ran fine | 2 | `0B395A0D-...` (8 entries, 428 B), `290E1F42-...` (8 entries, 413 B) |
| unopenable | 1 | `FDB8C946-...` (no StreamerInfo) — task_1313 |

`scan_fitfail.sh` over all **1642 task logs** — the per-task CVH failure rate,
which is what a silently corrupted *event* payload would show up in:

```
baseline (1625 tasks):  mean 0.0067 %   max 0.0414 %
the 12 split-99 tasks:  16.72 - 17.96 %
```

Nothing else is an outlier — in particular the seven tasks reading the two
NUL-carrying-but-runnable files sit at 0.000-0.020 %, dead in the baseline. So
the memory corruption of §8.3 hit only the provenance blob in every case where
it did not stop the job, and the physics payload of the sample is sound.
