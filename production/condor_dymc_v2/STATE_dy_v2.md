# `dymc_8p5M_260906_v2` — the DY re-production, on HTCondor

Companion to `production/STATE_dy.md` (the slurm `dymc_8p5M_260905` it
replaces) and `production/PRODUCTION_NEXT.md` (what the new exports are for).
Same 380-chunk list, same task index -> same event range, so the two
productions are comparable task by task.

| | |
|---|---|
| tag | `dymc_8p5M_260906_v2` |
| output | `/ceph/submit/data/user/d/david_w/ZMass/cvh/dymc_8p5M_260906_v2/task_XXXX/` |
| chunks | `production/chunks_dymc_8p5M_260905.txt` (380, 8.5 M events) |
| CMSSW | `CMSSW_15_0_19_patch2_dev2` @ `fab515e` (`cvh-exports-260906`) |
| driver | `Analysis/HitAnalyzer/test/runCvhDimuonMiniAOD.py` |
| new switches | `exportCfGroupExponents=True exportMaterialNoise=True exportVarianceGrads=True varianceGradFamilies=15` |
| batch | HTCondor -> **CMS global pool** (see "Why this is a grid job") |

---

## 1. Thread scan (2026-09-06)

The CVH refit in CMSSW_15 is multithreading-capable: the maker is an
`edm::stream::EDProducer<>` and the Geant4 master (`CvhMasterThread`, an
EventSetup product on `CvhMasterRecord`) builds `DDDWorld` and the master field
ONCE before any TBB worker starts, so each stream only attaches per-thread G4
state. `numberOfStreams` follows `numberOfThreads` in both drivers.

**One chunk of `chunks_dymc_8p5M_260905.txt`, first 2000 events, one arm at a
time on an otherwise quiet 64-core node (submit50, load 7.5), the full
re-production switch set, `/usr/bin/time -v`:**

| threads | wall (s) | CPU (s) | CPU/wall | speedup | events/s/core | peak RSS (GB) | RSS/core (GB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 770.7 | 761.3 | 0.99 | 1.00 | 2.60 | 1.76 | 1.76 |
| 2 | 404.2 | 765.5 | 1.89 | **1.91** | 2.47 | 1.85 | 0.93 |
| 4 | 224.9 | 770.5 | 3.43 | **3.43** | 2.22 | 2.00 | 0.50 |
| 8 | 131.7 | 772.8 | 5.87 | **5.85** | 1.90 | 2.12 | 0.27 |

(2000 events, 897 Z candidates in every arm.)

**The same scan on the J/psi leg** (`config_jpsimc20M.sh` chunk,
`runCvhJpsiGenMC.py`, 2000 ALCARECO events / 1991 candidates, submit51):

| threads | wall (s) | CPU (s) | CPU/wall | speedup | peak RSS (GB) | RSS/core (GB) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2039.9 | 2028.0 | 0.99 | 1.00 | 1.36 | 1.36 |
| 2 | 1024.7 | 2003.4 | 1.96 | **1.99** | 1.45 | 0.73 |
| 4 | 535.5 | 2020.6 | 3.77 | **3.81** | 1.61 | 0.40 |
| 8 | 286.4 | 2024.2 | 7.07 | **7.12** | 1.89 | 0.24 |

with `wall(N) = 36.0 + 2003.9/N` (all four points to 1.3 %) and total CPU
**flat to −0.2 %**. The J/psi leg scales BETTER than DY at the same event
count simply because its arm is 2.6x longer, so the same ~36 s serial head is
a smaller fraction — which is the whole point about the production projection.
Projected on a real 12 185-event J/psi chunk with the new exports: 3.40 h at
1 thread, **0.86 h at 4 (3.97x)**, 0.43 h at 8 (7.84x).

**Total CPU is flat**: 761.3 -> 772.8 s from 1 to 8 threads, **+1.5 %**. There
is no parallel overhead to speak of; what limits the speedup is a fixed serial
head. Amdahl with a single serial term fits all four points to better than 1 %:

```
wall(N) = 40.4 s  +  730.3 s / N        (measured: 770.7, 404.2, 224.9, 131.7)
```

The 40.4 s is geometry + field + Geant4 physics-list construction, and it is
paid once per JOB, not once per event. **The 2000-event scan therefore
UNDERSTATES production scaling**, because a real chunk is 22 120 events:

| threads | projected wall on a 22 120-event chunk | speedup |
|---:|---:|---:|
| 1 | 2.25 h | 1.00 |
| 2 | 1.14 h | 1.97 |
| 4 | 0.57 h | **3.94** |
| 8 | 0.29 h | **7.73** |

### Memory is essentially thread-independent

Peak RSS goes 1.76 -> 2.12 GB for 8x the cores. Splitting that into a shared
part and a per-stream part:

```
DY    : shared ~ 1.71 GB   per additional stream ~ 51 MB
J/psi : shared ~ 1.28 GB   per additional stream ~ 76 MB
```

So the memory request is set by the **single-task tail**, not by the core
count. The tail is not in a 2000-event scan: over the 260905 productions the
1-thread MaxRSS was median 1.9 GB / p99 3.1 GB / max 3.56 GB (J/psi, 1616
tasks) and median 2.6 GB / max 3.56 GB (DY, 164 tasks), against 1.76 GB here.
Sizing on `max + (N-1) x 0.051 GB`, with 1.3x headroom:

| threads | worst plausible peak | recommended `--mem` / `request_memory` | per core |
|---:|---:|---:|---:|
| 1 | 3.56 GB | **4.5 GB** | 4.5 GB |
| 2 | 3.61 GB | **5 GB** | 2.5 GB |
| 4 | 3.71 GB | **5 GB** | 1.25 GB |
| 8 | 3.92 GB | **5.5 GB** | 0.7 GB |

**The recommendation for the slurm legs is `--cpus-per-task=4 --mem=5G`**: 3.9x
the throughput per task for 0.83x the memory the current `--mem=6G
--cpus-per-task=1` asks for, i.e. **1.25 GB/core instead of 6**. 8 threads is
better still on memory per core but makes a task harder to backfill and pays a
larger fraction of the serial head; 4 is the point where both are comfortable.

> A caution carried over from `merge/merge_slurm_v3.sh`: two tasks reporting
> MaxRSS within 0.1 MiB of each other and of the limit is the signature of a
> cgroup CEILING, not a peak. The numbers above are `/usr/bin/time -v` VmHWM on
> an unlimited node, so they are peaks.

### Correctness gate: bit-identical, not merely consistent

An N-thread run writes **N output files**, `globalcor_<stream>.root`
(`ResidualGlobalCorrectionMakerBase.cc:451`), and TBB hands events to whichever
stream is free, so the tree order changes. The comparison is therefore per
candidate after a permutation, keyed on `(run, lumi, event)` with a **stable**
sort — exact, because an event is never split across streams, so all candidates
of one event are written consecutively by one stream in the maker's own order.
`production/threadscan/compare_threads.py` does this.

```
DY,    896 candidates, all 213 branches:  t1 vs t2 / t4 / t8   ALL BIT-IDENTICAL
J/psi, 1991 candidates, all 228 branches: t1 vs t2 / t4 / t8   ALL BIT-IDENTICAL
runtree 126 452 entries: identical in every stream file, both legs
```

**Nothing differs — not one value of one branch.** That is the answer to the
1e-5-closure question, and the mechanism is worth recording because it is not
automatic:

* the maker is `edm::stream::EDProducer<>`: one instance and one thread-local
  Geant4 state per stream, no shared mutable state;
* the geometry/field master is built before any worker starts, so there is no
  first-touch race and no per-thread field cache to diverge;
* **the drivers DO give each stream its own CLHEP engine**
  (`RandomNumberGeneratorService.trackrefitdimuon`, seeds derived per stream,
  bound into Geant4's thread-local RNG at the top of every `produce()`). If the
  CVH propagation ever *sampled*, the result would be stream-dependent by
  construction. It does not: the t8 run's streams 1-7 carry different seeds
  from the t1 run's single stream and still reproduce it bit for bit. **This is
  an invariant, not a guarantee** — any future change that makes the fit
  consume randomness (stochastic energy loss sampling, a randomised
  initialisation) breaks it silently. `compare_threads.py` is the guard; run it
  after any such change.

### The one real cost of threading: the runtree is written N times

Output bytes for the same 896 candidates:

| threads | total output | kB/candidate |
|---:|---:|---:|
| 1 | 75.3 MB | 82.1 |
| 2 | 88.4 MB | 96.3 |
| 4 | 114.6 MB | 124.9 |
| 8 | 166.8 MB | 181.8 |

That looks alarming and is entirely one thing. A two-term fit
`total = N x F + n_cand x b` reproduces all four points to 0.02 %:

```
F = 13.07 MB per stream FILE        b = 67.8 kB per candidate
```

and `F` is exactly the **`runtree`**, measured at **13.02 MB / 126 452
entries** — byte-identical in every stream file. The candidate tree itself is
69.4 kB/candidate at 1 thread and 71.6 at 8, i.e. flat. The J/psi leg gives the
same two constants independently: `F = 13.09 MB`, `b = 66.2 kB/candidate`. So the penalty is a
duplicated global-parameter map, and it is amortised by candidates per file.
On a real 22 120-event chunk (9 862 candidates):

| threads | projected output per task | penalty | over 380 tasks |
|---:|---:|---:|---:|
| 1 | 682 MB | — | 259 GB |
| 4 | 721 MB | **+5.7 %** | 274 GB |
| 8 | 773 MB | **+13.4 %** | 294 GB |

Tolerable, and a second (mild) reason to prefer 4 threads over 8. It would be
removed entirely by writing the `runtree` from stream 0 only — not done here,
because every existing reader opens the file it happens to have.

### One consequence downstream

N threads means N files per task. Readers that hard-code `globalcor_0.root`
would silently take **1/N of the statistics**. **Fixed 2026-09-06**: every
reader in `calibration_studies/` lists its inputs through
`resolution/prodfiles.py` (shell twin `prodfiles.sh`), which widens the stream
index, decides completeness per TASK, and makes `--ntasks` a cap on tasks
rather than on files. The `runtree` is written once PER STREAM, identical each
time, and `prodfiles.runtree_file()` returns the first EXISTING stream of a
task so exactly one copy is read and never the concatenation. See
`PRODUCTION_NEXT.md` sec. 10 for the API and the validation table.

---

## 2. Why this is a grid job

The submit HTCondor pool is a **submission portal, not a cluster**. Measured
2026-09-06:

* `condor_config_val COLLECTOR_HOST` = `submit06.mit.edu:9615`;
  `FLOCK_TO` = `t3serv009.mit.edu:11000`
* the local pool has **exactly one execute slot** — `slot1@submit06`, 1 CPU,
  whose `Start` expression is
  `Name == "submit06.mit.edu" && !isUndefined(Submit_LocalTest)`. It is a
  deliberate single-slot local-test resource.
* everything else is glideins. Of 16 471 jobs running through this schedd, the
  execute hosts are DESY, IHEP, IIHE, RAL, LNL, Caltech/ultralight, MIT T2 and
  NERSC — i.e. the **CMS global pool**.

Probe jobs (`scratch_condorprobe_260906/probe.sh`) on real slots:

| route | OS | /ceph | /work | /home | cvmfs | xrootd read of the input |
|---|---|---|---|---|---|---|
| `mit_tier3` (t3btch001) | **el7** | ABSENT | ABSENT | ABSENT | present | denied |
| global pool (DESY, IIHE, Caltech) under `cms:rhel9-x86_64` | el9 | ABSENT | ABSENT | ABSENT | present | **OK** |

So there is **no ceph-mounted condor capacity at all**, and `mit_tier3` is
excluded twice over — it is native el7 with no singularity, and CMSSW_15_0 is
`el9_amd64_gcc12`. The job therefore has to carry all three things a slurm task
takes for granted:

1. **the CMSSW area** — base release from cvmfs plus a 64 MB overlay
   (`lib biglib cfipython python src`). The area is **not relocatable as a
   whole**: `.SCRAM/RuntimeCache.json` and `.SCRAM/*/MakeData/*.mk` bake in the
   absolute build path, and an untar-in-place run reports the ORIGINAL
   `CMSSW_BASE` while cvmfs happily supplies a complete release underneath it —
   i.e. it silently runs the wrong libraries. The worker runs `scram project`
   (7 s) for a clean skeleton and untars the overlay on top, then **asserts
   `CMSSW_BASE == $_CONDOR_SCRATCH_DIR/$RELEASE`**.
2. **the input** — `/ceph/submit/data/group/cms/store/...` is a mirror of the
   CMS `/store` namespace, so the chunk path maps to an LFN and streams through
   `root://cms-xrd-global.cern.ch/`. Verified from DESY and IIHE.
3. **the output** — `xrdcp` to `root://submit50.mit.edu/`, whose namespace root
   **is** `/ceph/submit`. Verified both directions.

### Proxy

`use_x509userproxy = true` with `x509userproxy = /home/submit/david_w/x509up_u125124`
(NOT `/tmp`: that is per-node, and the schedd's shadow has to read it). The
copy in `$HOME` had **expired**; `submit_dymc_v2.sh` now refreshes it from
`$X509_USER_PROXY` whenever that one has more time left, and refuses to submit
under 2 h. Current proxy: 164 h. To renew:
`voms-proxy-init -voms cms -valid 192:00`.

### Routing — four things, all of which are load-bearing

```
use_x509userproxy = true
+AccountingGroup  = "analysis.david_w"      # required by submit06's submit policy
+ProjectName      = "CpuTestingProject"
+SingularityImage = ".../cmssw/cms:rhel9-x86_64"   # CMSSW_15_0 is el9
+DESIRED_Sites    = "<explicit list>"       # WITHOUT this the job never leaves Idle
```

The `mit_tier3` route needs the OPPOSITE (`+ProjectName = "MIT_submit"`, and
`+AccountingGroup` and `+SingularityImage` REMOVED) — irrelevant here, since
those pilots cannot run el9, but it is in
`BtoJpsiX_MCprod/.../t3test/test_tier3.sub` if it is ever needed.

### Why `NTHREADS=1` on the grid when the scan says 4

A glidein slot is not a node you own. The pool is overwhelmingly 1-core, a
4-core request queues behind every 1-core one, and the BtoJpsiX production
measured exactly this and dropped 4 -> 1 for matchability. The grid leg runs
1 thread at `request_memory = 2600` MB; **the slurm legs are where the 4-thread
recommendation belongs.**

### Stage-out is verified by size, never by exit code

`xrdcp` has truncated outputs on this cluster before (two condor attempts of
the BtoJpsiX production were destroyed that way) and returned 0 while doing it.
`copy_verified()` reads the remote size back and retries three times, and the
`.complete` sentinel is copied LAST, only after every payload file has been
checked — so a partial stage-out can never look finished.

---

## 3. Validation, 2026-09-06

One chunk, 400 events (186 Z candidates), the production switch set, compared
candidate by candidate on the sorted key with `compare_threads.py`:

| run | where | wall | output |
|---|---|---:|---|
| reference | local dev2, submit82 (AMD EPYC 9965), 1 thread, POSIX ceph input | — | `globalcor_0.root` 25 842 502 B |
| grid 1-thread | condor, **INFN Pisa** (`s2wn2.pi.infn.it`), xrootd input | 903 s | `globalcor_0.root` **25 842 502 B** |
| grid 4-thread | condor, **Caltech** (`compute-13-253.ultralight.org`), xrootd input | 751 s | 4 stream files, 65.1 MB |

```
local(1 thr) vs grid 1-thread  PASS  all 213 branches bit-identical over 186 candidates
local(1 thr) vs grid 4-thread  PASS  all 213 branches bit-identical over 186 candidates
runtree                        126 452 entries, identical in every file
```

That is bit-identity across the batch system, **two different sites, two
different CPU vendors**, xrootd-streamed input versus a POSIX read, and 1
versus 4 streams. The 1-thread grid file is byte-for-byte the same SIZE as the
local reference, not merely equivalent in content.

The 4-core job matched at Caltech about a minute after submission, which is the
answer to the only real objection to running 4 threads on the grid.

### What the first attempt caught, and why the wrapper looks like it does

The first grid job died 130 s in with
`ScalarPot3DEval: cannot open /work/submit/.../polyfit3d_full_coeffs_lmax18_custom50.txt`.
`scalarPot3DInitFile` is the one option in `EXTRA` whose value is an absolute
path outside the release. The wrapper now ships that file, rewrites the option
to the transferred copy, **asserts it exists**, and refuses any remaining
option whose value starts `/work/`, `/ceph/` or `/home/` — so the next such
option fails in the first second instead of after the conditions load.
(`materialGroupsFile` is safe: it defaults to
`$CMSSW_BASE/src/Analysis/HitAnalyzer/data/materialGroups50.txt`, which is in
the payload.)

---

## 4. Submission and first-half-hour throughput

**Cluster `3803254`, 380 jobs, submitted 2026-09-06 14:56:50, 4 threads,
`request_memory = 5000` MB.**

| t + | running | idle | held | complete |
|---:|---:|---:|---:|---:|
| 2 min | 76 | 304 | 0 | 0 |
| 5 min | 204 | 176 | 0 | 0 |
| 10 min | 255 | 125 | 0 | 0 |
| 16 min | 318 | 62 | 0 | 0 |
| 20 min | 361 | 18 | 0 | 1 |
| **31 min** | **379** | **0** | **0** | **1** |

Essentially the whole submission was running inside half an hour, across 12
sites: MIT T2 (82), DESY (70), NCHC Taiwan (57), RWTH Aachen (38), INFN Legnaro
(33), INFN Roma (22), KISTI (20), IHEP (17), CIEMAT (11), INFN Pisa (7), JINR
(6), Caltech/Wisconsin. This is against **137 000 idle jobs** already in the
pool from other users — a 4-core, 5 GB, 1-day request is simply easy to place.

**The comparison that matters.** At the same moment the slurm DY leg
(`dymc_8p5M_260905`) had **1 task running and 198 pending**, at 11.4 tasks/h
with 17.5 h to go, because the 90 running J/psi array tasks were consuming the
whole fair share. The condor leg is not competing for that share at all: it is
**379 concurrent tasks of additional capacity**, obtained while the J/psi
arrays keep their 90. That is the entire point of the exercise, and the factor
is not 2 or 3, it is ~380.

First completed task, 31 min in: `task_0262` at DESY, 2 349 events in **504 s**
on 4 streams, 1 057 candidates, 4 output files totalling 126 MB — which is the
`4 x 13.07 MB + 1057 x 67.8 kB = 123 MB` model to 2 %.

### First-attempt failures, and one site fence

30 of the first 380 attempts failed and condor retried them (`max_retries = 3`;
at t+31 min the restart histogram was 347 jobs on their first start, 27 on
their second, 1 on a third, 1 on a fourth, and **zero held**):

| rc | n | what | where |
|---:|---:|---|---|
| **132** | **16** | **SIGILL** in `XrdCl::PostMaster::Start()` (CVMFS `libXrdCl.so.3`) on the `stagein` Prepare, while PoolSource is constructed | **8 at RWTH Aachen, 8 at JINR — nowhere else** |
| 139 | 5 | SIGSEGV mid-processing (e.g. event 601 of a chunk the local scan runs clean) | 4 at INFN-LNL-2, 1 at DESY, all different nodes |
| — | 8 | `xrdfs stat` of the LFN timed out in the wrapper's pre-check | scattered |
| 65 | 1 | `Input/output error` reading a plugin .so from cvmfs | NCHC |

The SIGILL is the only one with structure, and it is unambiguous: it is inside
the **release's own external**, the release is loading its `scram_x86-64-v2`
variants, and all sixteen are at two sites. Not our payload, not the chunk.
Fenced on the live cluster with `condor_qedit` and in `config_dymc_v2.sh`:

```
(regexp("physik.rwth-aachen.de", Machine, "i") =!= true) && (regexp("jinr.ru", Machine, "i") =!= true)
```

`regexp(..., "i")` and not `=!=` on the name — `=!=` is the ClassAd IDENTITY
operator and is case sensitive, and MIT T2 advertises `Machine` in uppercase,
so a lowercase `=!=` fence silently never excludes anything.

The 5 SIGSEGVs are on 5 different nodes, 4 of them at LNL, which is also
running 33 jobs successfully; not enough structure to fence yet, but if it
concentrates further LNL is the next candidate. The 8 pre-check timeouts are
the global redirector being momentarily unhappy about a file that certainly
exists — the wrapper's `xrdfs stat` is stricter than cmsRun's own open, which
retries across the redirector. It costs a retry and self-heals.

### The SIGSEGVs, resolved 2026-09-06 — XrdAdaptor, not Geant4, not threading

Six tasks (0, 1, 2, 3, 260, 293, 294 — 1 succeeded on retry) exhausted their
attempts with rc=139. The `.err` stacks end inside
`fillRadiativeSpectrum -> G4MuPairProductionModel::ComputeDMicroscopicCrossSection`,
which is **not** where the crash is: `tail -60 local.log` keeps only the LAST
threads of a gdb dump printed in descending order, and every non-faulting thread
carries a `<signal handler called>` frame too, because CMSSW pauses them all
before forking gdb. The faulting thread is above the cut.

The real crash is a null dereference in the RELEASE's
`Utilities/XrdAdaptor/src/XrdRequestManager.cc` (`getQueryTransport` leaves its
`std::string*` uninitialised; `AnyObject::Get` sets it to 0 when the transport
query fails, which is exactly what happens for a redirect hop with no live
connection), reached from `tracerouteRedirections` on the **XrdCl JobManager
thread**. The framework says so itself: `Module: non-CMSSW (crashed)`.

It is an XROOTD-path bug, which is why the slurm 260905 production (POSIX
`/ceph`) never saw it: the chunk of task_0000 runs 22120/22120 events clean at
4 AND 8 threads from `/ceph` and SIGSEGVs at event 4401 when the only change is
streaming the input through `cms-xrd-global`.

The 5.5-6.1 GB MemoryUsage before each crash is the crash handler forking gdb,
not a leak — the instrumented re-runs peak at 2.39/2.43 GB. `request_memory =
5000` stands.

Fixed on `cvh-exports-260906` (`c2d74e3e647`); recover with
`condor_dymc_v2/recover_xrdfix.sh`, which uses a SECOND pinned payload so the
jobs still in the queue keep the binary they were submitted with. Full account:
`Documents/Resolution/NOTES.md`, 2026-09-06 (V).

---

## 5. Operating it

```bash
cd calibration_studies/production/condor_dymc_v2
./submit_dymc_v2.sh --dry-run          # render without submitting
./submit_dymc_v2.sh                    # all 380
./status_dymc_v2.sh                    # sentinels + condor queue + rate
./resume_dymc_v2.sh --dry-run          # which indices have no sentinel and no queue entry
./resume_dymc_v2.sh
```

`resume` never re-queues an index that is already in the queue (two jobs writing
the same task dir would interleave their outputs), and completion is judged on
the `.complete` sentinel alone — cmsRun creates its output at START, so a
killed job leaves a non-empty but truncated file that an `-s` test would accept
forever.

The payload is built once and **pinned inside the production's own output
tree** (`$OUTBASE/payload/`), so a resume months later runs the same binary as
the original submission. `--rebuild-payload` is the deliberate override.

