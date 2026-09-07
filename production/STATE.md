# jpsimc_20M_260905 — CVH full-scale feasibility production

**What this is.** The first large-scale run of the two-track J/psi CVH fit with
*both* products switched on at once: the global-correction gradients (factored
Hessian) and the in-maker resolution-CF exponents. Sample: 20M events of the
UL16 `JPsiToMuMu_Pt8toInf` MC ALCARECO. The question it answers is not a
physics one — it is whether the machinery runs, at what cost per candidate, and
at what output volume.

| | |
|---|---|
| tag | `jpsimc_20M_260905` |
| output | `/ceph/submit/data/user/d/david_w/ZMass/cvh/jpsimc_20M_260905/task_XXXX/globalcor_0.root` |
| CMSSW | `/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev` @ `6fde319` (+ the `skipEvents` driver option) |
| driver | `Analysis/HitAnalyzer/test/runCvhJpsiGenMC.py` |
| input | `.../RunIISummer20UL16RECO/JPsiToMuMu_Pt8toInf-pythia8/ALCARECO/TkAlJpsiMuMu-106X_mcRun2_asymptotic_v13{-v2,_ext1-v3}` |

## Configuration and why

| option | value | why |
|---|---|---|
| `trackSrc` | `ALCARECOTkAlJpsiMuMu` | the collection this ALCARECO actually carries |
| `useLegacyPairLoop` | `True` | **required**: TkAlJpsiMuMu has no persisted `…JpsiOnlyResonances` candidate collection (only TkAlJpsiX does), so the maker must use the all-pairs loop |
| `fitFromGenParms` | `False` | **required**: the default `True` freezes the 10-dim vertex/kinematic reference block and seeds it from gen. We want a reconstruction-level fit. Note the freeze is *not* conditioned on the gen match succeeding, so leaving the default would be silently wrong, not a fallback |
| `doVtxConstraint` | `False` | plain two-track fit; PCA rather than a vertex constraint |
| `doMassConstraint` | `False` | ditto — no J/psi mass constraint (a mass constraint would add a second `nicons` pass) |
| `doTrigger` | `True` | store the per-path HLT decisions (input has `TriggerResults::HLT`) |
| `applyHltFilter` | `False` | do not pre-filter; the ALCARECO selection already ran |
| `doSimHits` | `False` | **required**: the input has no `SimTrack`s, and `doSimHits=True` with `fitSimHitPositions=False` disables the genParticle ΔR matching *before* the hard-coded `requireGen=True` cut, which would reject every candidate |
| `useIdealGeometry` | `False` | aligned MC geometry from the GT — the "rung-B" real-geometry configuration |
| `useDefaultField` | `True` | **the field the SIM propagated through** — the unlabelled `VolumeBasedMagneticField` 160812 with `useParametrizedTrackerField=True`, i.e. `OAE_1103l_071212` in the tracker. See "Field model" below |
| `globalTag` | `106X_mcRun2_asymptotic_v17` | UL16 MC conditions |
| `scalarPot3DInitFile` | `mfs/…/polyfit3d_full_coeffs_lmax18_custom50.txt` | always required — it defines the 50 parmtype-14 modes and their Jacobian columns even when another model supplies the field |
| `materialGroupsFile` | driver default `materialGroups50.txt` | 42 groups; what the reference calibration production and every recent run used. `materialGroupsV2.txt` exists but **no driver or production references it** |
| `fillJac` | `True` | per-track ∂x_ref/∂a |
| `fillGrads` | `False` | the packed Hessian `hesspackedv` is the thing being replaced |
| `fillGradsFactored` | `True` | low-rank `H = BᵀB` as `nRank`/`nFactor`/`hessfactorv`. Not mutually exclusive with `fillGrads` — both can be on and both would be written; we want only the factored one |
| `doRes` | `True` | registers the resolution families (parmtypes 8/9/10/11) and enables the CF export block |
| `exportCfExponents` | `True` | the `cfmass_*` exponents on the 64-point tau grid (default True; named explicitly for the record) |
| `exportStepRecords` | **`False`** | the slimming. 430 kB/candidate of raw per-step records whose only consumer was the offline exponent extractor that the in-maker export replaces |
| `CgfQoPMode` | `0` | mode ≥1 substitutes the CGF estimator through the uncached branch at ~10× cost |
| `propagationPtotLimit` | `0.2` | G4e momentum floor (cfi default is 1.0) |
| `clampMomentumFloor` | derived → **0.25 GeV** | `<0` ⇒ `1.25 × plimit`. The historical hard-coded 2.0 GeV against a 0.2 GeV limit pinned 12 % of soft candidates and carried the whole +0.21e-3 mass-scale offset |
| `maxMomentumStepFactor` | `2.0` | relative GN trust region: p may at most halve or double per iteration |
| `stepBacktracking` | `True` | Armijo chi2 backtracking from iteration 2, ≤4 halvings, slack 1.0 |
| `numberOfThreads` | `1` | Geant4e; also makes the output exactly `globalcor_0.root` |

### Global parameter map (`runtree`, 126 452 entries)

| parmtype | count | what |
|---|---:|---|
| 0,5 | 16 588 each | alignment local-x, θ_z (per detUnit) |
| 1 | 8 656 | alignment local-y (pixel + endcap) |
| 2,3,4 | 13 300 each | alignment local-z, θ_x, θ_y (per align-detid) |
| 8 | 16 588 | resolution: hit local-x (doRes) |
| 9 | 1 440 | resolution: hit local-y (doRes) |
| 10 | 13 300 | resolution: multiple scattering (doRes) |
| 11 | 13 300 | resolution: ionization (doRes) |
| 14 | 50 | scalar-potential B-field modes |
| 15 | 42 | global material-model groups |
| **total** | **126 452** | 81 732 alignment + 44 628 resolution + 92 global |

**The alignment parameters are included** — there is no switch to leave them
out, and they dominate the per-candidate payload (`nParms` ≈ 248 of which ~240
are alignment). The offline consumer `global_corrections/fit_global_grads.py`
freezes them at selection time (`--parmtypes` defaults to `[14, 15]`).

The reference DATA production `jpsi_calib2016_grads50_globalmat_260717` had
`nglobalparms = 81824` — the same map **without** the four doRes families. Its
arguments were:
`nEvents=-1 numberOfThreads=8 doMassConstraint=True fillJac=True fillGrads=False fillGradsFactored=True scalarPot3DInitFile=…custom50.txt`
via `runCvhJpsi.py` (the DATA driver, which exposes none of the step-damping or
CF options and therefore ran at the old 2.0 GeV clamp against a 1.0 GeV limit).

> ⚠️ **Global indices shift with `doRes`.** A `doRes=True` file (126 452 params)
> and a `doRes=False` file (81 824) index the same physical parameter
> differently. **These outputs must not be pooled with the 260715/260717 data
> productions in one global fit** without re-mapping through `runtree`.

## Field model — why `useDefaultField`, not `useOpera3D`

**This is a feasibility/closure test, so the fit's field must be the
simulation's field.** The UL16 SIM propagated through the default
OAE-parametrised tracker field (`useParametrizedTrackerField=True` on the 160812
volume-based map). Refitting the same events with the full 3D TOSCA grid
instead does not cancel: it leaves an **η/φ-coherent ~8e-4 dp/p pattern** in the
pull width (measured 2026-08-07 — 0.155 % of unit variance, and φ is where OAE
is structurally blind). On a closure test that pattern is indistinguishable
from the thing being measured.

The first submission (arrays 6406885/6406886) went out with `useOpera3D=True`,
following the `btojpsix_v3` precedent, and was cancelled ~7 min in with **0
tasks completed**. Its output directory is kept, minus the 8 truncated
mid-write `.root` files, as
`/ceph/…/cvh/jpsimc_20M_260905_opera3d` — see its `PROVENANCE.txt`.

**The Opera3D variant is reserved as a later injected-field test on a subset**:
refitting a slice of this sample with the 3D grid and comparing to the
default-field result measures that η/φ pattern directly, which is a useful
systematic — but as a deliberate injection, not as the baseline.

## Input selection

`filelist_jpsimc_20M_260905.txt` — the exact 410 files, one absolute path per
line. `filelist_jpsimc_20M_260905.counts.txt` adds the event count per file.

**The repacked ALCARECO has a ~5 % tail of TRUNCATED files** (ROOT zombie, "no
keys recovered"). cmsRun runs with `skipBadFiles=True`, so a truncated input is
*silently skipped* and the task writes a valid but empty output. Every
candidate file was therefore opened and its `Events` entry count read before
selection (`mkchunks.py` drops anything that fails), and the array task
additionally fails on `attempted=0` in the fit summary.

**File size does not predict corruption** — zombies were found up to 2.76 GB,
and valid files go down to 959 events / 38 MB. Only the ROOT open is decisive.

**The 43 truncated files are listed in
`production/truncated_inputs_260905.txt`** (path + size, with the diagnosis in
its header). That is 43 of the 620 files scanned = **6.9 %**; if the rest of the
2031-file repack is similar, ~140 of them are bad, so **re-scan with
`scan_events.py` before using any other slice of this sample**.

## Scale, as submitted

| | |
|---|---|
| files | 410 good (of 620 scanned; **43 unreadable, dropped**) |
| events | **21 719 059** (mean 52 973/file) |
| tasks | **1642** (mean 13 227 events; range 959…20 168) |
| chunks/file | 4 for 328 files; 1:8, 2:3, 3:22, 5:45, 6:3, 7:1 |
| slurm | arrays **`6406906`** (idx 0-999) and **`6406907`** (idx 0-641, offset 1000), `%200` each |
| candidates | ~1.0/event ⇒ **~21.7M** |
| output | ~21.7M × 80.7 kB ≈ **1.75 TB** (quota headroom was 35 TB of 50) |
| CPU | ~21.7M × 0.75 s ≈ **4 500 CPU-h** |
| wall/task | ~2.8 h typical, ~5 h for the largest chunk (12 h limit) |
| wall total | 1642 × 2.8 h / N_concurrent — **~12 h at 400 slots, ~23 h at 200** |

The chunk list was validated to tile every file exactly: no gaps, no overlaps,
Σ chunk events == Σ file events, 410 distinct basenames.

## Chunking

One 53k-event file is 10+ h of Geant4e, too long for one task. Each file is
split into event ranges via `skipEvents`/`nEvents`, sized by a **target of
13 500 events per task** rather than a fixed 4-way split — the sample's files
range 959…62k events, and a fixed split would make 240-event tasks that still
pay the ~3 min startup.

`skipEvents` did not exist; it was added to `runCvhJpsiGenMC.py` (python only,
no C++ change). It asserts a single input file per job, because PoolSource
counts the skip across the whole concatenated `fileNames` list.

> The pre-existing `file:<path>,<skip>,<max>` filelist format in
> `resolution/simprod/` **has no parser** — feeding it raw to `input=` splits on
> the commas as path separators and, with `skipBadFiles=True`, silently
> processes the *entire* file. It is not used here.

## Smoke results (300 events of the first input file)

The production configuration (`useDefaultField=True`), re-run after the field
switch:

| | value |
|---|---|
| candidates | 298 attempted, **298 succeeded, 0 failed (0 %)** |
| propagator | 31 531 calls, **0 failures**, `exit1[plimit]=0`, all other exits 0 |
| `[cvh] effective:` | `clampMomentumFloor=0.25 GeV` (derived), `maxMomentumStepFactor=2`, `stepBacktracking=1 (fromIter=2, maxChi2Backtrack=4, armijoC=0.0001, armijoSlack=1)`, `CgfQoPMode=0`, `IoniTruncationAlpha=0.999` |
| `cfmass_ok` | **298/298** |
| output | 24 062 184 B = **80.7 kB/candidate** |
| peak RSS | 1.37 GB |
| `nglobalparms` | 126 452 (unchanged — the field switch changes the baseline field, not the parameter map) |

Cross-check against the cancelled Opera3D smoke, same 300 events: identical
branch set (196 in `tree`), identical `nParms` (247.6) and `nRank` (28.5) means,
298/298 in both, 80.7 kB/candidate in both — and `gradmax` **differs**, which is
how we know the field switch actually took effect rather than being ignored.

The `exportStepRecords` comparison below was made on the Opera3D pair; it is a
statement about a storage switch and does not depend on the field model.

| | slim (`exportStepRecords=False`) | full (`=True`) |
|---|---|---|
| candidates | 298 attempted, **298 succeeded, 0 failed** | identical |
| propagator | 31 928 calls, **0 failures**, `exit1[plimit]=0` | identical |
| output | 24 061 626 B = **80.7 kB/candidate** | 123 492 882 B = 414.4 kB/candidate |
| peak RSS | 1.45 GB | 1.62 GB |

* `skipped[samesign]=1`, `clamped[step]=1`, `backtracked[step]=0`.
* `[cvh] effective:` echoed `clampMomentumFloor=0.25 GeV`, `maxMomentumStepFactor=2`,
  `stepBacktracking=1 (fromIter=2, maxChi2Backtrack=4, armijoC=0.0001, armijoSlack=1)`,
  `CgfQoPMode=0`, `IoniTruncationAlpha=0.999`.
* `cfmass_*` present (10 branches), `cfmass_ok` true for 298/298.
* Factored-Hessian branches present: `nRank` (mean 28.5), `nFactor` (mean 7206
  → 28.1 kB/candidate for `hessfactorv` alone), `hessdroppedmass`,
  `hessrankgap`, `globalidxv`, `gradv`, `nParms` (mean 247.6).
  No `hesspackedv`/`nSym` — `fillGrads=False`.
* **Slimming proven inert**: `root_bitcompare.py` reports *all 232 common
  branches bit-identical*; the only schema difference is exactly the 10
  documented step-record branches (`ioniurban{idx,v}`, `radstep{idx,v,specv}`,
  `msmoli{idx,v}`, `reseigv`, `resinf{v,bv}`).
  (`--exclude` was added to `root_bitcompare.py` for this: without it a storage
  switch can never be tested, since a missing branch alone reports DIFFERENT.)
* **CF chain runs on the slim file**: `cf_inmaker.py pairs` read 298/298
  candidates, 0 dropped; `cf_masskernel_tt.py` found 298 in the J/psi window;
  `cf_masslik_fit.py --model r` converged in 1.6 s (`|grad|inf` 9.3e-7,
  positive-definite, 0/298 negative L_i). Values are meaningless at 298
  candidates — this is a plumbing check.

Timing: ~**0.79–0.91 s/candidate** plus ~3 min startup when two smokes
contend on one node; a clean single task on a compute node did 40 candidates
plus startup in 66 s, i.e. ~0.65 s/candidate and ~40 s startup. Budget
0.65–0.9 s/candidate.

### Slurm path validated before the real submission

A 2-task array was run first: one 40-event chunk (40/40 candidates succeeded,
`.complete` written, `[done]` in the .out) and one deliberately non-existent
input, which failed as intended with `[FATAL] input not readable on submit05`
and left no sentinel and no task directory.

`skipEvents` was checked against a known event sequence rather than assumed:
`skipEvents=100` starts on `Event 3242205836`, the 101st record of the same
file read from 0.

`resume.sh`'s queue-exclusion mapping was checked to expand the live queue to
exactly the index set {0…1641}.

## Commands

```bash
cd /work/submit/david_w/ZMass/calibration_studies/production
./status.sh                 # complete/running/failed, events, projected finish
./status.sh --slow          # + output bytes and candidate counts (walks ceph)
./resume.sh --list-only     # which chunks are missing
./resume.sh                 # resubmit only those (skips anything still queued)
```

`resume.sh` keys off the `.complete` sentinel, never the `.root`: cmsRun
creates its output at *start*, so a killed task leaves a non-empty but
truncated file. The task index is the chunk-list line number, so a resumed task
reads exactly the same events as the one it replaces.

## 2026-09-06 — the 13 failed tasks: three corrupt inputs and one full GPU

Neither is a code failure. `jpsimc_20M_260905` has **zero** exit-134
(SIGABRT) failures — the `ndof == 0` abort that cost the DY leg 28 % of its
chunks (`STATE_dy.md`, NOTES 2026-09-06 (III)) cannot reach it, because
ALCARECO carries the full RECO hit list and a J/psi pair never lands on
`nvalid + nvalidpixel == 10`.

**12 tasks, exit 91 — three corrupt input files.** Tasks 1219-1222, 1334-1337,
1409-1412: four consecutive chunks each of

```
.../TkAlJpsiMuMu-106X_mcRun2_asymptotic_v13-v2/2830000/BDA060EF-B8F8-7349-9277-363C3AB7EA76.root
.../TkAlJpsiMuMu-106X_mcRun2_asymptotic_v13-v2/60000/0909778B-8728-764F-B41B-1C9DCE5C849E.root
.../TkAlJpsiMuMu-106X_mcRun2_asymptotic_v13-v2/60000/4B9D2D77-92ED-8440-8697-DD4AC560E61C.root
```

cmsRun dies while *constructing the input source*:

```
An exception of category 'FormatIncompatibility' occurred while
   [2] Creating ParameterSets from file
EntryError can not convert representation of SelectEvents: <blanks> to value of type vector<string_hex>
```

**This is a new corruption class the 2026-09-05 zombie scan cannot see.** That
scan reads `Events->GetEntries()`; all three files have a perfectly good Events
tree (51 231 / 56 267 / 48 621 entries) and are *not* in
`truncated_inputs_260905.txt`. What is damaged is the **provenance blob**, and
`skipBadFiles=True` does not help because the failure happens before any
per-file skip logic.

It is **local**: `edmProvDump` throws on all three /ceph copies and **succeeds
on all three over `root://cms-xrd-global.cern.ch`**. So the MIT copies rotted,
the dataset did not. Recovered by re-fetching to a user area (the group store
is managed data and is not ours to overwrite):

```
/ceph/submit/data/user/d/david_w/ZMass/restaged/jpsimc_20M_260905/
```

`scratch_bugfix_260906/restage.sh` does the xrdcp into a `.part` and promotes it
only if `edmProvDump` reads it — an xrdcp that reports rc=0 on a truncated file
is a documented failure mode here. The twelve chunk lines of
`chunks_jpsimc_20M_260905.txt` were then repointed at the staged copies (path
field only, line numbers untouched, so the task index is still the chunk-list
line) and resubmitted with `resume.sh`.

**1 task, exit 66 — a full GPU.** Task 0254 died in 15 s constructing services:
`CUDAService ... cudaErrorMemoryAllocation: out of memory` on a node whose GPU
was already full. Purely environmental; `resume.sh` re-drives it onto another
node. (The job needs no GPU at all — `CUDAService` is only being constructed
because the standard service set includes it.)

**A latent bug in `resume.sh` itself, found while doing this.** The script built
its `--array` spec in a variable called `GROUPS`, which is a **bash built-in
array** holding the caller's group ids. `declare -A GROUPS` fails with "cannot
convert indexed to associative array" and the writes land in the builtin, so
the spec came out as `--array=100999%200`, `--array=169571%200`,
`--array=1000000%200` — the user's gids. Renamed to `CHUNKGRP` in `resume.sh`,
`resume_dy.sh` and `resume_dy_dev2.sh`. The resume path had never been
exercised before today.

## Notes

* **submit82's ceph client is evicted** (`cephx … failed: -13`); `/ceph/submit`
  is EPERM there. All ceph work for this setup was done from submit81. The
  array task asserts input readability on its node for the same reason.
* `fillGrads=False` does **not** save the packing CPU — the packed Hessian is
  computed unconditionally and the flag only controls whether it is written.

## 2026-09-07 — the 2026-09-06 "recovery" was wrong, and what the repair found

The section above says the three exit-91 files were "recovered by re-fetching
to a user area". **They were not.** `cms-xrd-global` serves the CENTRAL copy of
this dataset, which is the un-repacked **split-99** original — precisely what
the whole repack campaign exists to avoid (ROOT #19773). The staged files are
byte-size-identical to what the redirector serves today, and `splitlevels.py`
reports `split=99 / sub-98` on all three cluster and rechit branches.

The full account, with every number, is in
**`condor_jpsimc_v2/STATE_jpsi_v2.md` §8**. In brief:

* the 12 chunks made from those files yield **0.824 candidates/event** against
  0.9968, with 17.4 % `fail[prop]` — and the surviving 83 % are **also
  garbage**: median chi2/ndof 3.5e6, 99.8 % at the iteration limit, 2.9 % inside
  the J/psi mass window, effective per-hit pull 1860 sigma. Every *pre-refit*
  quantity agrees with the control, so it is the refit reading corrupted strip
  clusters, not a different sample. **No salvage cut exists.**
* the exit-91 provenance damage is **write-side memory corruption in our own
  July-2026 repack job**: NUL runs inside the *uncompressed* ParameterSets
  payload, each replacing a value's content with its length preserved, confined
  to the 32-46 byte band (`std::string` lengths just above the SSO threshold).
  Not disk rot, not transfer damage — `edmProvDump` is clean on the central
  original and fails on our copy.
* all four files were re-repacked to split-1 and validated; the 16 chunks were
  re-run on condor as cluster **`3803425`** and all 16 succeeded on the first
  start — **204 957 candidates from 205 635 events = 0.9967/event at 0.0073 %
  failures**, chi2/ndof median 0.9531 against the controls' 0.9540, 4 streams
  and a sentinel each. **`jpsimc_20M_260906_v2` is now 1642/1642.** New inputs:
  `/ceph/submit/data/user/d/david_w/ZMass/restaged/jpsimc_20M_260906_repack/`.
  The bad staged copies are kept as `restaged_split99_DO_NOT_USE/`.

### The slurm v1 leg `jpsimc_20M_260905` — 12 affected chunks, deliberately NOT repaired

`chunks_jpsimc_20M_260905.txt` carries the same 12 repointed lines, so the same
task indices are affected. The array has now drained: **1633 of 1642 tasks have
a `.complete`**, and the 9 that do not are exactly these:

| task | input | v1 outcome |
|---|---|---|
| 1219 1220 1221 | `BDA060EF-...` | COMPLETED 10:47-10:52 — **garbage, discard** |
| **1222** | `BDA060EF-...` | **TIMEOUT at 12:00:02, no `.complete`** |
| 1337 | `0909778B-...` | COMPLETED 11:56 — **garbage, discard** |
| **1334 1335 1336** | `0909778B-...` | **TIMEOUT at 12:00, no `.complete`** |
| 1409 1410 1412 | `4B9D2D77-...` | COMPLETED 10:15-10:19 — **garbage, discard** |
| **1411** | `4B9D2D77-...` | **TIMEOUT at 12:00:24, no `.complete`** |
| **1552 1553 1554 1555** | `03249796-...` | **exit 91, never produced output** |

So of the 12 split-99 chunks, 7 finished (with garbage) and 5 hit the 12 h wall
limit. **DISCARD all 12.** v1 is a cross-check sample only, so it was left alone
rather than repaired; a marker listing them sits in the output tree as
`BAD_TASKS_split99_260907.txt`. If v1 is ever used quantitatively, either
exclude the 12 or re-run them against `jpsimc_20M_260906_repack/`. **No re-run
is needed** — this is a record, not an action item.

**And the wall clock was a free corruption detector all along.** Every one of
those 12 tasks ran **10:14-12:00** against a normal J/psi chunk's ~2.8 h — a
factor ~3.8, which is exactly the 4.1x propagator-call excess the diverged fits
generate (5.75 M calls vs 1.39 M). Nothing else in the array is remotely near
the limit. **Adding a per-task runtime outlier check to the status script costs
nothing and would have caught this on day one**, without knowing anything about
split levels or ROOT #19773.

### Round 2 — the other three files, and why the task count is 1645

* **`task_1313` was silently empty, and its input was also TRUNCATED.**
  `2830000/FDB8C946-...root` has **no StreamerInfo** (ROOT opens it and counts
  19 797 entries, so the 2026-09-05 zombie scan passed it; CMSSW cannot
  deserialise it and `skipBadFiles=True` drops it). The maker then never runs,
  so **no `fit summary` line is written** and the `attempted=0` guard matched
  nothing — 4 empty stream files and a `.complete`. Both wrappers now also fail
  on the skip message and on a missing summary.
  Worse: the local repack is **724 MB against a 1942 MB original and holds
  19 797 events against 51 478**. `mkchunks.py` reads `Events->GetEntries()` on
  the local file, so it tiled 38 % of it — **31 681 events were never chunked at
  all**, on top of the 19 797 task_1313 skipped.
  Repaired: line 1314 keeps `0 19797` repointed at the repack, and the tail is
  **appended as tasks 1642-1644** so no existing index moves. All four ran:
  **51 290 candidates at 0.9958-0.9965/event**, chi2/ndof median 0.9518 against
  the controls' 0.9538. **`task_1313` is now usable.**
* **the two inert-NUL inputs are proven inert.** `0B395A0D-...` and
  `290E1F42-...` carry the same NUL corruption in psets cmsRun never parses.
  Rather than assume that, both were repacked (the new files are **264 B and
  299 B smaller** — only the provenance blob changed) and their 7 chunks re-run
  into a side tree with the **original** payload, so the input was the only
  difference. Result: **BIT-IDENTICAL over 102 507 candidates**, all 228 tree
  branches and 36 `runtree` branches, zero differing values. **The existing
  outputs stay**; the side tree `cvh/jpsimc_20M_260906_v2_nulcheck/` is kept as
  evidence with a README saying it is deletable.

### Final state

**`jpsimc_20M_260906_v2` is 1645/1645**, four stream files and a sentinel in
every task, **21 750 740 events** (was 1642 / 21 719 059 — the difference is
exactly the FDB8C946 tail). All **410 distinct inputs are clean**: 403 untouched
group-store repacks plus **7 repaired files** in
`restaged/jpsimc_20M_260906_repack/`, every one split-1, with an entry count
matching the *central* original, `edmProvDump` clean and NUL-scan clean.

### Sample-wide integrity, measured rather than assumed

Two scans in `production/repack_fix_260907/`:

* `scan_pset_nulls.py` over all 410 inputs: **403 OK, 6 with NUL runs, 1
  unopenable**;
* `scan_fitfail.sh` over all 1642 task logs: baseline **mean 0.0067 %, max
  0.0414 %**; only the 12 split-99 tasks are outliers, at 16.7-18.0 %.

So the corruption is contained: where it did not stop the job it hit only the
provenance blob, and the physics payload of the sample is sound. **Run both
scans on any future repack before a production reads it** — together they cost
minutes and they catch all three classes (NUL provenance, missing StreamerInfo,
and any silent event-level damage, via the failure rate).

## 2026-09-07 — the production area is consolidated onto `cvh-exports-260906`

This production is finished, so the area it ran from is no longer pinned. The
two CVH areas have been merged back into one tree.

**Preconditions checked before touching anything** (all three, in this order):

* `squeue -u david_w` — **no `jpsimc*` / `dymc*` job of any state**. The slurm
  `jpsimc20M_*` arrays completed on the morning of 2026-09-07; the slurm DY leg
  was cancelled on 2026-09-06 in favour of the condor v2 leg;
* `condor_q -constraint 'Owner=="david_w"'` — **only cluster `3803425`**, the
  16 repair chunks of §"the four bad inputs, repaired". Their `TransferInput`
  is the payload tarball
  `/ceph/.../cvh/jpsimc_20M_260906_v2/payload/overlay_jpsimc_20M_260906_v2_xrdfix.tgz`,
  **pinned inside the production's own output tree**, so they carry their own
  binaries and depend on neither area's `lib/`. Nothing was done to any condor
  cluster;
* `git status` clean in both areas.

**What was done.**

| step | result |
|---|---|
| sparse checkout of `..._dev` aligned to `..._dev2` | `/Utilities/XrdAdaptor/` added — it was added to the dev2 worktree for the null-pointer patch `c2d74e3e647`, and sparse-checkout files are **per-worktree** (`.git/info/` vs `.git/worktrees/src/info/`), so it did not propagate on its own. Both lists now identical, 18 patterns |
| `git merge --ff-only cvh-exports-260906` | **clean fast-forward**, `157775e2f3a` -> `ca6058d96fc`, 17 commits, 20 files, +2755/-122. No merge commit |
| tree equality | `git diff cvh-exports-260906 --stat` **empty**; `diff -rq` of the two `src/` trees (excluding `.git`, `__pycache__`) reports **no tracked difference** — only six untracked leftovers in `..._dev` (`globalcor_0.root`, `globalcor_resclosure_0.root`, `run.log`, `toyPlanes_pt3.py`, two `debugG4e_*.log`) |
| `cmsswlock.sh build scram b -j32` | **rc=0 in 152 s**, 0 compiler errors; six libraries relinked, including the newly materialised `libUtilitiesXrdAdaptor.so` / `pluginUtilitiesXrdAdaptorPlugin.so` |
| `resolution/smoke_exports_260906.sh` on `..._dev` | rc=0, entries **59 / 120 / 48** as asserted (an empty tree is a FAILED smoke, see the script's header) |
| bit comparison | `v8_devmerged` vs the dev2 references `v7_xrdfix` **and** `v6_default`: **IDENTICAL on all three smokes**, 246 / 157 / 264 branches, `runtree` 126 452 entries, output files the same size to the byte |

The two dev2 references were also compared to each other and are identical, so
`c2d74e3e647` (XrdAdaptor) and `ca6058d96fc` (the Geant4e shared-table mutex)
are confirmed inert on POSIX input — they only matter on the xrootd path and
under multiple streams.

**Which area to use.** `PRODUCTION_NEXT.md` §0 is the authority. Short version:
run from **`/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2`, branch
`cvh-exports-260906`** — every live script already defaults to it. `..._dev`
(branch `WmassNanoProd_15_0_19_patch2_dev`) is now the identical, built spare,
which is what makes this file's own v1 configs (`config_jpsimc20M.sh`,
`array_jpsimc.sbatch`) safe again rather than a trap. Keep the two in step with
`git merge --ff-only`; if that ever refuses, they have diverged and the reason
must be understood before either runs a production. **Nothing was pushed to any
remote.**
