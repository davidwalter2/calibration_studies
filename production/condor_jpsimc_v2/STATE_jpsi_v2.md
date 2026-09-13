# `jpsimc_20M_260906_v2` — the J/psi leg on HTCondor

The production record for the scripts in this directory. The configuration
rationale, the parameter map, the export list, the cost model and the full
defect list are in `production/PRODUCTIONS.md`; the grid mechanics (why a
condor leg is a grid job, the payload/routing/stage-out contract) are in
`condor_dymc_v2/STATE_dy_v2.md` §1 and are not repeated here.

| | |
|---|---|
| tag | `jpsimc_20M_260906_v2` |
| output | `/ceph/submit/data/user/d/david_w/ZMass/cvh/jpsimc_20M_260906_v2/task_XXXX/globalcor_<0..3>.root` |
| chunks | `chunks_jpsimc_20M_260906_v2.txt` — **1645** tasks, 410 files, 21 750 740 events |
| result | **1645/1645**, 4 stream files + `.complete` each, **21 678 062 candidates** (0.9967/event, 0.0067 % fit failures), **1.5 TB** |
| CMSSW | `CMSSW_15_0_19_patch2_dev2` @ `fab515e` (then `cvh-exports-260906`, merged into `cvh-exports-clean-260911` and deleted 2026-09-13; the SHA still resolves); the 20 recovered tasks @ `ca6058d` |
| driver | `Analysis/HitAnalyzer/test/runCvhJpsiGenMC.py` |
| exports | the 260905 set + `exportCfGroupExponents=True exportMaterialNoise=True exportVarianceGrads=True varianceGradFamilies=15` |
| resources | **4 threads**, `request_memory = 5000` MB, `request_disk = 6 000 000` KB, `max_retries = 3` |
| clusters | `3803264` (the 1642 bulk), `3803425` (16 repaired chunks), plus the round-2 recovery of 1313 / 1642–1644 |

---

## 1. THE INPUT IS ADDRESSED BY DOOR, NEVER BY LFN

This is the one thing that makes the J/psi leg different from DY, and getting
it wrong does not produce an error — it produces a wrong sample that looks
right.

The J/psi inputs are **our own repacked split-1 copies**, written into the
submit group store under the CMS `/store` namespace. **That same LFN exists
centrally, with different content** — the un-repacked split-99 original, which
mis-deserialises its `SiStripCluster`s in 15_X (ROOT #19773). Measured on the
first file of the list:

| door | size served |
|---|---:|
| `root://submit50.mit.edu/` (and 51–55) | **1 783 846 343 B — ours, split-1** |
| `root://cms-xrd-global.cern.ch/` | 1 840 069 145 B — **the central original** |
| `root://xrootd.cmsaf.mit.edu/` (T2 redirector) | `[3011] Too many attempts to gain dfs read access` |

So the DY leg's trick — map the ceph path to an LFN and let the global
redirector find a replica — **is not safe here**. The MIT T2 door does not serve
this tree at all, because it is the T3/submit CephFS and not the T2 store.

The mapping the wrapper uses is a **pure prefix strip against a named door** —
submit's xrootd namespace root *is* `/ceph/submit`:

```
/ceph/submit/X   ->   root://submit5N.mit.edu//X
```

exact, no redirector in the path, uniform across the group-store files and the
re-staged copies in the user store. Readability from the grid was probed at
DESY: `OK (OURS)` for submit50/51, for both the `/store/...` and the
`/data/group/cms/store/...` form, and for the re-staged copies. Two guards on
top:

1. the chunk list carries each file's **size as a fourth field**, taken on the
   ceph copy at submit time, and the wrapper **refuses any door that serves a
   different size** (exit 8, `WRONG REPLICA`). It never fired in the production —
   no door has ever served a wrong replica;
2. the door is chosen as `(IDX + k) mod 6` over submit50–55, falling through to
   the next if one does not answer, so 1645 jobs do not pull ~740 GB through one
   login node (~14 MB/s per door) and one sick door does not stall the leg.

## 2. Operating it

```bash
cd calibration_studies/production/condor_jpsimc_v2
./submit_jpsimc_v2.sh --dry-run
./submit_jpsimc_v2.sh                    # all 1645; --only "0 1 2" / --idxfile FILE / --threads / --mem
./status_jpsimc_v2.sh [--slow]           # --slow also checks streams/task
./resume_jpsimc_v2.sh --dry-run
./recover_xrdfix.sh --only "<indices>"   # re-run named tasks from the SECOND pinned payload
```

Completion is the `.complete` sentinel alone; `resume` never re-queues an index
already in the queue; the payload is pinned inside the production's own output
tree (`$OUTBASE/payload/`) so a later resume runs the same binary.
`recover_xrdfix.sh` exists because the jobs still queued re-fetch the pinned
tarball on every restart — overwriting it would mix two binaries into one
sample — and it also drives the diagnostic wrapper.

Sizing came from the thread scan (`condor_dymc_v2/STATE_dy_v2.md` §1): the
J/psi arm measured 3.81× at 4 threads on 2000 events, `wall(N) = 36.0 +
2003.9/N`, total CPU flat to −0.2 %, projecting 3.40 h → **0.86 h (3.97×)** on a
real 12 185-event chunk. `request_memory = 5000` is set by the single-task tail
(~1.28 GB shared + ~76 MB/stream against a 1-thread MaxRSS max of 3.56 GB over
the 1616 slurm tasks), i.e. **less memory per task than the slurm leg's 6 G with
four times the cores**.

Placement: 91 running at 5 min, 498 at 31, 558 at 35, over 14+ sites (DESY 165,
MIT T2 71, Caltech 70, INFN Legnaro 58, KISTI 31, CIEMAT 27, …), nothing held.
No task completes inside the first hour — a full chunk is ~1.5 h at 4 threads on
a grid CPU.

## 3. Validation

400 events (398 candidates), production switch set: a grid run at Wisconsin on
4 threads is **bit-identical to a local dev2 1-thread run on submit82**, all 228
branches, `runtree` 126 452 entries identical in every stream file — across the
batch system, the site, xrootd-through-a-door versus POSIX, and 1 versus 4
streams, with the door reporting the repacked size.

## 4. Defects seen on this leg

* **SIGILL (rc 132), 36 in the first half hour** — 34 on five `ultralight.org`
  machines (`compute-6-34` alone ate 16) plus 2 on `t2bat0310`, crashing in
  `edm::StreamSchedule::fillWorkers`: the node cannot run the release's binaries
  at all. Fenced **by node** (the rest of Caltech and MIT T2 ran hundreds of
  jobs cleanly) in `config_jpsimc_v2.sh` and with `condor_qedit`.
* **`no door served <path>`, once** — all six doors returned "no answer" inside
  one job's 180 s window for a file both submit50 and submit53 serve correctly.
  Condor retries it.
* **Four inputs with a corrupted provenance blob** (exit 91) and **three more
  defective ones** (one truncated + StreamerInfo-less, two carrying inert NUL
  runs). All seven repacked to split-1 into
  `restaged/jpsimc_20M_260906_repack/`; the 16 + 4 affected chunks re-run. Full
  mechanism, acceptance gates and the sample-wide scans:
  `PRODUCTIONS.md` §8.1 and §9.
* **XrdAdaptor null dereference** (rc 139) — fixed in `c2d74e3e647`; the J/psi
  leg saw none in its first half hour but the recovery payload carries the fix.
  `PRODUCTIONS.md` §8.2.
* **`recover_xrdfix.sh` could never have submitted** as first written (undefined
  `REDIR` under `set -u`, and `INDOORS` never passed). Fixed.
* **`status_jpsimc_v2.sh` should flag per-task runtime outliers.** The 12
  split-99 chunks ran 3.8× the median on the slurm twin, which was a free
  corruption detector nobody was reading. The sentinel already carries
  `seconds=`.

## 5. If the grid route is ever unavailable

`production/array_jpsimc_v2.sbatch` is the slurm fallback and the recommended
*slurm* configuration going forward: `--cpus-per-task=4 --mem=5G
--time=4:00:00` from dev2. It must not be submitted alongside a running
`jpsimc20M_*` array — it would only queue behind it for the same fair share.
