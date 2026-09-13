# `dymc_8p5M_260906_v2` — the Z leg on HTCondor

The production record for the scripts in this directory, plus the grid
mechanics that both condor legs rely on. Configuration rationale, parameter
map, export list and the full defect list are in `production/PRODUCTIONS.md`.

| | |
|---|---|
| tag | `dymc_8p5M_260906_v2` |
| output | `/ceph/submit/data/user/d/david_w/ZMass/cvh/dymc_8p5M_260906_v2/task_XXXX/globalcor_<0..3>.root` |
| chunks | `production/chunks_dymc_8p5M_260905.txt` — **380** tasks, 104 files, 8 502 597 events |
| result | **380/380**, 4 stream files + `.complete` each, **3 799 624 candidates** (0.447/event; 6 104 fit failures = 0.16 %: `prop` 3 594, `kinfit` 2 301, `ndof` 208, `nan` 1), **267 GB** |
| CMSSW | `CMSSW_15_0_19_patch2_dev2` @ `fab515e` (then `cvh-exports-260906`, merged into `cvh-exports-clean-260911` and deleted 2026-09-13; the SHA still resolves) |
| driver | `Analysis/HitAnalyzer/test/runCvhDimuonMiniAOD.py` |
| exports | the 260905 set + `exportCfGroupExponents=True exportMaterialNoise=True exportVarianceGrads=True varianceGradFamilies=15` |
| resources | **4 threads**, `request_memory = 5000` MB, `request_disk = 4 000 000` KB, `max_retries = 3` |
| cluster | `3803254` |

Same chunk list as the superseded slurm leg `dymc_8p5M_260905`, so `task_NNNN`
is the same event range in both and they compare task by task.

---

## 1. Why a condor leg is a grid job

The submit HTCondor pool is a **submission portal, not a cluster**:
`COLLECTOR_HOST = submit06.mit.edu:9615`, `FLOCK_TO =
t3serv009.mit.edu:11000`, and the local pool has **exactly one execute slot** —
`slot1@submit06`, 1 CPU, `Start` gated on
`Name == "submit06.mit.edu" && !isUndefined(Submit_LocalTest)`. Everything else
is glideins: of 16 471 jobs through this schedd the execute hosts were DESY,
IHEP, IIHE, RAL, LNL, Caltech, MIT T2 and NERSC — the **CMS global pool**.

Probe jobs on real slots:

| route | OS | /ceph | /work | /home | cvmfs | xrootd read of the input |
|---|---|---|---|---|---|---|
| `mit_tier3` (t3btch001) | **el7** | ABSENT | ABSENT | ABSENT | present | denied |
| global pool (DESY, IIHE, Caltech) under `cms:rhel9-x86_64` | el9 | ABSENT | ABSENT | ABSENT | present | **OK** |

So there is **no ceph-mounted condor capacity at all**, and `mit_tier3` is
excluded twice over (native el7, no singularity, against an `el9_amd64_gcc12`
release). The job carries all three things a slurm task takes for granted:

1. **the CMSSW area** — cvmfs base release plus a 64 MB overlay
   (`lib biglib cfipython python src`). The area is **not relocatable whole**:
   `.SCRAM/RuntimeCache.json` and `.SCRAM/*/MakeData/*.mk` bake in the absolute
   build path, and an untar-in-place run reports the ORIGINAL `CMSSW_BASE`
   while cvmfs supplies a complete release underneath it — i.e. it silently
   runs the wrong libraries. The worker runs `scram project` (7 s) for a clean
   skeleton, untars the overlay on top, and **asserts
   `CMSSW_BASE == $_CONDOR_SCRATCH_DIR/$RELEASE`**;
2. **the input** — `/ceph/submit/data/group/cms/store/...` mirrors the CMS
   `/store` namespace, so the chunk path maps to an LFN and streams through
   `root://cms-xrd-global.cern.ch/` (verified from DESY and IIHE). **The J/psi
   leg must NOT do this** — see `condor_jpsimc_v2/STATE_jpsi_v2.md` §1;
3. **the output** — `xrdcp` to `root://submit50.mit.edu/`, whose namespace root
   **is** `/ceph/submit`. `copy_verified()` reads the remote size back and
   retries three times, and the `.complete` sentinel is copied LAST, only after
   every payload file has been checked. `xrdcp` has truncated outputs on this
   cluster before while returning 0.

**Proxy:** `use_x509userproxy = true` with
`x509userproxy = /home/submit/david_w/x509up_u125124` — **not** `/tmp`, which is
per-node while the schedd's shadow has to read it. `submit_dymc_v2.sh` refreshes
it from `$X509_USER_PROXY` whenever that has more time left and refuses to
submit under 2 h. Renew with `voms-proxy-init -voms cms -valid 192:00`.

**Routing — four load-bearing lines:**

```
use_x509userproxy = true
+AccountingGroup  = "analysis.david_w"              # required by submit06's submit policy
+ProjectName      = "CpuTestingProject"
+SingularityImage = ".../cmssw/cms:rhel9-x86_64"    # CMSSW_15_0 is el9
+DESIRED_Sites    = "<explicit list>"               # WITHOUT this the job never leaves Idle
```

The `mit_tier3` route needs the opposite (`+ProjectName = "MIT_submit"`, with
`+AccountingGroup` and `+SingularityImage` REMOVED); irrelevant here, but it is
in `BtoJpsiX_MCprod/.../t3test/test_tier3.sub` if ever needed.

**Why 4 threads on the grid.** A 4-core, 5 GB, 1-day request is easy to place:
the whole 380-job submission was running within half an hour (76 at 2 min, 204
at 5, 318 at 16, **379 of 380 at 31 min with 0 idle and 0 held**) across 12
sites — MIT T2 82, DESY 70, NCHC 57, RWTH 38, INFN Legnaro 33, INFN Roma 22,
KISTI 20, IHEP 17, CIEMAT 11, Pisa 7, JINR 6, Caltech/Wisconsin — against
137 000 idle jobs already in the pool. (The BtoJpsiX production's 4 → 1 change
is sometimes read as a matchability result; its own note says the reason was a
cfg that ran single-threaded, and that what had blocked matching was a missing
`+DESIRED_Sites`.) A 4-thread task is also 34 min rather than 2.25 h, which
matters on a preemptible glidein.

For comparison: at the same moment the slurm DY leg had **1 task running and
198 pending** at 11.4 tasks/h, starved by the J/psi arrays' fair share, while
this leg ran at 177.8 tasks/h — **a factor 15.6 on rate**, as ~380 concurrent
tasks of *additional* capacity.

## 2. The thread scan (both legs)

The CVH refit in CMSSW_15 is multithreading-capable: the maker is an
`edm::stream::EDProducer<>` and the Geant4 master (`CvhMasterThread`, an
EventSetup product on `CvhMasterRecord`) builds `DDDWorld` and the master field
ONCE before any TBB worker starts, so each stream only attaches per-thread G4
state. `numberOfStreams` follows `numberOfThreads` in both drivers.

One chunk, first 2000 events, one arm at a time on an otherwise quiet 64-core
node, the full re-production switch set, `/usr/bin/time -v`
(`production/threadscan/run_scan.sh`):

| threads | DY wall (s) | DY CPU (s) | speedup | RSS (GB) | J/psi wall (s) | J/psi CPU (s) | speedup | RSS (GB) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 770.7 | 761.3 | 1.00 | 1.76 | 2039.9 | 2028.0 | 1.00 | 1.36 |
| 2 | 404.2 | 765.5 | 1.91 | 1.85 | 1024.7 | 2003.4 | 1.99 | 1.45 |
| 4 | 224.9 | 770.5 | 3.43 | 2.00 | 535.5 | 2020.6 | 3.81 | 1.61 |
| 8 | 131.7 | 772.8 | 5.85 | 2.12 | 286.4 | 2024.2 | 7.12 | 1.89 |

(897 Z candidates and 1991 J/psi candidates in every arm.)

Total CPU is flat — +1.5 % on DY, −0.2 % on J/psi, over the whole range — so
there is no parallel overhead, only a fixed serial head of geometry + field +
Geant4 physics-list construction: `wall(N) = 40.4 + 730.3/N` (DY) and
`36.0 + 2003.9/N` (J/psi) fit all four points to ~1 %. That head is paid once
per JOB, so **a 2000-event scan understates production scaling**: on a real
22 120-event DY chunk 4 threads gives 3.94× (0.57 h) and 8 gives 7.73×; on a
12 185-event J/psi chunk, 3.97× (0.86 h) and 7.84×.

**Memory is set by the single-task tail, not the core count.** Peak RSS splits
into ~1.71 GB shared + ~51 MB per extra stream (DY) and ~1.28 GB + ~76 MB
(J/psi); the tail a 2000-event scan never sees is the 1-thread MaxRSS over the
260905 productions — median 1.9 / p99 3.1 / **max 3.56 GB** (J/psi, 1616 tasks)
and median 2.6 / max 3.56 GB (DY, 164 tasks). Sizing on
`max + (N−1) × per-stream` with 1.3× headroom gives 4.5 / 5 / **5** / 5.5 GB at
1 / 2 / 4 / 8 threads, i.e. **1.25 GB per core at 4 instead of 6**. Eight
threads is better still per core but makes a task harder to backfill and pays a
larger fraction of the serial head. (Two tasks reporting MaxRSS within 0.1 MiB
of each other *and* of the limit is a cgroup CEILING, not a peak; these numbers
are VmHWM on an unlimited node.)

**Correctness gate: bit-identical, not merely consistent.** An N-thread run
writes N files and TBB hands events to whichever stream is free, so the
comparison is per candidate after a permutation, keyed on `(run, lumi, event)`
with a stable sort — exact, because an event is never split across streams
(`production/threadscan/compare_threads.py`):

```
DY,    896 candidates, all 213 branches:  t1 vs t2 / t4 / t8   ALL BIT-IDENTICAL
J/psi, 1991 candidates, all 228 branches: t1 vs t2 / t4 / t8   ALL BIT-IDENTICAL
runtree 126 452 entries: identical in every stream file, both legs
```

The mechanism is worth recording because it is not automatic: one thread-local
Geant4 state per stream and no shared mutable state; the geometry/field master
built before any worker starts, so no first-touch race; and **the drivers do
give each stream its own CLHEP engine** (`RandomNumberGeneratorService.
trackrefitdimuon`, seeds derived per stream, bound into Geant4's thread-local
RNG at the top of every `produce()`). The t8 streams carry different seeds from
the t1 stream and still reproduce it bit for bit — so this is an invariant of a
fit that does not sample, **not a structural guarantee**. Any change that makes
the fit consume randomness breaks it silently; `compare_threads.py` is the
guard.

**The one real cost of threading** is that the `runtree` is written into every
stream file. Output for the same 896 candidates goes 75.3 / 88.4 / 114.6 /
166.8 MB at 1 / 2 / 4 / 8 threads, and a two-term fit
`total = N × F + n_cand × b` reproduces all four points to 0.02 % with
`F = 13.07 MB` (the runtree, measured at 13.02 MB / 126 452 entries) and
`b = 67.8 kB/candidate`; the J/psi leg gives `F = 13.09 MB`, `b = 66.2 kB`
independently. On a real chunk that is **+5.7 % output at 4 threads, +13.4 % at
8**. Consequence downstream: readers must list inputs through
`resolution/prodfiles.py`, which widens the stream index, decides completeness
per task, and returns exactly one `runtree`.

## 3. Operating it

```bash
cd calibration_studies/production/condor_dymc_v2
./submit_dymc_v2.sh --dry-run          # render without submitting
./submit_dymc_v2.sh                    # all 380
./status_dymc_v2.sh                    # sentinels + condor queue + rate
./resume_dymc_v2.sh --dry-run          # indices with no sentinel and no queue entry
./resume_dymc_v2.sh
./recover_xrdfix.sh --only "0 2 3"     # named tasks, from the SECOND pinned payload
./submit_diag.sh "<idx…>" [REQMEM_MB] [SUFFIX]   # the instrumented wrapper
```

Completion is the `.complete` sentinel alone (cmsRun creates its output at
START, so a killed job leaves a non-empty but truncated file an `-s` test would
accept forever); `resume` never re-queues an index already in the queue. The
payload is built once and **pinned inside the production's own output tree**
(`$OUTBASE/payload/`), so a resume months later runs the same binary;
`--rebuild-payload` is the deliberate override.

## 4. Validation

One chunk, 400 events (186 Z candidates), production switch set, compared
candidate by candidate:

| run | where | wall | output |
|---|---|---:|---|
| reference | local dev2, submit82 (AMD EPYC 9965), 1 thread, POSIX ceph input | — | `globalcor_0.root` 25 842 502 B |
| grid, 1 thread | condor, **INFN Pisa**, xrootd input | 903 s | `globalcor_0.root` **25 842 502 B** |
| grid, 4 threads | condor, **Caltech**, xrootd input | 751 s | 4 stream files, 65.1 MB |

Both **PASS, all 213 branches bit-identical over 186 candidates**, `runtree`
126 452 entries identical in every file — across the batch system, two sites,
two CPU vendors, xrootd versus POSIX, and 1 versus 4 streams. The 1-thread grid
file is byte-for-byte the same size as the local reference.

The first grid attempt died 130 s in with
`ScalarPot3DEval: cannot open /work/submit/.../polyfit3d_full_coeffs_lmax18_custom50.txt`:
`scalarPot3DInitFile` is the one option in `EXTRA` whose value is an absolute
path outside the release. The wrapper now ships that file, rewrites the option
to the transferred copy, **asserts it exists**, and refuses any remaining option
whose value starts `/work/`, `/ceph/` or `/home/`. (`materialGroupsFile` is
safe — it defaults inside the payload.)

## 5. Defects seen on this leg

30 of the first 380 attempts failed and were retried, 0 held (restart histogram
at t+31 min: 347 first starts, 27 second, 1 third, 1 fourth).

| rc | n | what | where |
|---:|---:|---|---|
| **132** | **16** | **SIGILL** in `XrdCl::PostMaster::Start()` (CVMFS `libXrdCl.so.3`) on the `stagein` Prepare, while PoolSource is constructed | **8 at RWTH Aachen, 8 at JINR — nowhere else** |
| 139 | 5 (6 tasks) | **SIGSEGV**: the XrdAdaptor null dereference | 4 sites, all different nodes |
| — | 8 | `xrdfs stat` of the LFN timed out in the wrapper's pre-check | scattered, self-healing |
| 65 | 1 | `Input/output error` reading a plugin `.so` from cvmfs | NCHC |

The SIGILL is inside the **release's own external**, the release is loading its
`scram_x86-64-v2` variants, and all sixteen are at two sites — not our payload,
not the chunk. Fenced in `config_dymc_v2.sh` and on the live cluster with
`condor_qedit`:

```
(regexp("physik.rwth-aachen.de", Machine, "i") =!= true) && (regexp("jinr.ru", Machine, "i") =!= true)
```

Use `regexp(..., "i")`, never `=!=` on the machine name — `=!=` is the ClassAd
IDENTITY operator and is case sensitive, and MIT T2 advertises `Machine` in
uppercase, so a lowercase `=!=` fence silently never excludes anything. The
black-hole `ultralight.org` / `t2bat0310` nodes found by the J/psi leg are
fenced here too.

**The SIGSEGVs were XrdAdaptor, not Geant4 and not threading.** The `.err`
stacks end inside
`fillRadiativeSpectrum → G4MuPairProductionModel::ComputeDMicroscopicCrossSection`,
which is not where the crash is: `tail -60 local.log` keeps only the LAST
threads of a gdb dump printed in descending order, and every non-faulting thread
carries a `<signal handler called>` frame too because CMSSW pauses them all
before forking gdb. Dump 400 lines and grep for `sig_dostack_then_abort`. The
real crash is a null dereference in the release's
`Utilities/XrdAdaptor/src/XrdRequestManager.cc`, reached from
`tracerouteRedirections` on the XrdCl JobManager thread — the framework says so
itself (`Module: non-CMSSW (crashed)`). It is an **xrootd-path** bug, which is
why the slurm leg on POSIX `/ceph` never saw it: task_0000's chunk runs
22120/22120 events clean at 4 AND 8 threads from `/ceph` and SIGSEGVs at event
4401 when the only change is streaming through `cms-xrd-global`. Fixed in
`c2d74e3e647`; the 5.5–6.1 GB `MemoryUsage` logged just before each crash is the
crash handler forking gdb (instrumented re-runs peak at 2.39–2.55 GB), so
`request_memory = 5000` stands.
