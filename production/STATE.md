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

## Notes

* **submit82's ceph client is evicted** (`cephx … failed: -13`); `/ceph/submit`
  is EPERM there. All ceph work for this setup was done from submit81. The
  array task asserts input readability on its node for the same reason.
* `fillGrads=False` does **not** save the packing CPU — the packed Hessian is
  computed unconditionally and the flag only controls whether it is written.
