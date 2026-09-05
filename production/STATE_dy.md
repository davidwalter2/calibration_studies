# dymc_8p5M_260905 — CVH full-scale feasibility production, Z → μμ leg

**What this is.** The Z arm of the same test as `jpsimc_20M_260905`: the
two-track CVH fit with *both* products switched on at once — the
global-correction gradients (factored Hessian) and the in-maker resolution-CF
exponents — this time on Z → μμ, driven straight off UL16 DY MiniAODv2. It
answers the same question (does it run, at what cost per candidate, at what
output volume) on the channel whose kinematics are ten times harder, and it is
the leg that has to *pool* with the J/psi one in a single global fit.

| | |
|---|---|
| tag | `dymc_8p5M_260905` |
| output | `/ceph/submit/data/user/d/david_w/ZMass/cvh/dymc_8p5M_260905/task_XXXX/globalcor_0.root` |
| CMSSW | `/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev` @ `157775e` |
| driver | `Analysis/HitAnalyzer/test/runCvhDimuonMiniAOD.py` (added in `157775e`) |
| input | `/DYJetsToMuMu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos/RunIISummer20UL16MiniAODv2-106X_mcRun2_asymptotic_v17-v2/MINIAODSIM`, the 104-file / 8.5M-event priority head on ceph |

## Configuration and why

Chain: `slimmedMuons` → `TrackProducerFromPatMuons` →
`diMuonTrackVertexCandidates` (60–120 GeV, opposite sign) →
`ResidualGlobalCorrectionMakerTwoTrackG4e`. This is the same producer chain the
WMass custom NanoAOD uses, and the same *maker* as the J/psi leg — only the
track source and the mass window differ.

| option | value | why |
|---|---|---|
| `muonSrc` | `slimmedMuons` | the MiniAOD muon collection; `innerTrackOnly=False` |
| `massMin/massMax` | 60 / 120 | the Z window on the candidate producer |
| `CgfQoPMode` | **`0`** | passed explicitly even though the cfi default came back to 0 at `1cd4453`. Mode ≥1 runs `cvhcgf::inverseFisher` (two 262144-point FFTs) on every propagate call, ~88 calls/candidate ⇒ **8.5 s/candidate against 0.63** — for a block the two-track maker has no `setCgfOverride` hook to use. This was the one integer that decided whether the Z leg is 2k or 20k core-hours (`profiling/NOTES.md`) |
| `doRes` | `True` | registers the resolution families (parmtypes 8/9/10/11) and enables the CF export |
| `fillGradsFactored` | `True` | low-rank `H = BᵀB` as `nRank`/`nFactor`/`hessfactorv` |
| `fillGrads` | `False` | the packed Hessian is the thing being replaced. It does **not** save CPU — the packing runs unconditionally, the flag only controls whether it is written |
| — | | **`doRes` and `fillGradsFactored` are only expensive TOGETHER** (0.062 and 0.064 s/cand alone, 0.63 for the pair): the in-maker CF export is gated on `doRes_ && (fillGrads_ \|\| fillGradsFactored_)` |
| `exportCfExponents` | `True` | `cfmass_*` on the 64-point τ grid |
| `exportStepRecords` | **`False`** | 317 vs 28 kB/candidate for records whose only consumer the in-maker export replaces |
| `fillJac` | `True` | per-track `∂x_ref/∂a`. Costs **+6.2 kB/candidate** (`Mu{plus,minus}_jacRef` + `Jpsi_jacMass`) over the profiling runs, which had it off — that is the whole difference between the profiled 28.1 kB/cand and the 34.7 measured here |
| `fillRunTree` | `True` | the global parameter catalogue. Without it `globalidxv` cannot be interpreted at all |
| `doGen` | `True` | with the **MiniAOD retags** `genParticles=prunedGenParticles`, `pileupInfo=slimmedAddPileupInfo`. Both are dereferenced *unguarded* by the maker when `doGen=True`. `genParticles:xyz0` (the gen PV) does not exist in MiniAOD and *is* guarded — `genl3d` gets the −99 sentinel |
| `requireGen` | **`False`** | **unlike the J/psi leg.** There the all-pairs loop needs the gen match to reject combinatorics; here an opposite-sign pair in a 60–120 GeV window on DY is the Z by construction, so the cut would only couple the yield to the MiniAOD gen-pruning thresholds. `Mu{plus,minus}gen_*` are filled either way, so the cut can be made offline — and it has to be, for the CF chain: 5 of 145 smoke candidates have no gen match |
| `fitFromGenParms` | `False` | a reconstruction-level fit. The freeze is *not* conditioned on the gen match succeeding, so `True` would be silently wrong here, not a fallback |
| `doSimHits` | `False` | MiniAOD has no `PSimHit`s |
| `doVtxConstraint` / `doMassConstraint` | `False` / `False` | plain two-track fit, PCA rather than a vertex constraint |
| `doTrigger` | `False` | matches what was profiled. The option and a 2016 single-muon path list are in the driver if the decisions are wanted |
| `useIdealGeometry` | `False` | aligned MC geometry from the GT |
| `useDefaultField` | `True` | **the field the SIM propagated through** — the unlabelled `VolumeBasedMagneticField` 160812 with `useParametrizedTrackerField=True`, i.e. `OAE_1103l_071212` in the tracker. Same argument as the J/psi leg: refitting UL16 MC with the full 3D grid leaves an η/φ-coherent ~8e-4 dp/p pattern in the pull width, which on a closure test is indistinguishable from the thing being measured |
| `globalTag` | `106X_mcRun2_asymptotic_v17` | UL16 MC conditions; 2016 DDD geometry `XMLFILE_Geometry_2016_81YV1_Extended2016_mc` |
| `scalarPot3DInitFile` | `mfs/…/polyfit3d_full_coeffs_lmax18_custom50.txt` | the **50-mode** dump. Always required (it defines the parmtype-14 modes and their Jacobian columns even when another model supplies the field), and it must be the same dump as the J/psi leg or the two global catalogues differ. The 360-mode dump costs 0.21 s/cand in the per-step basis against 0.013 at 50 |
| `materialGroupsFile` | driver default `materialGroups50.txt` | 42 groups; same as the J/psi leg |
| `propagationPtotLimit` | `0.2` | G4e momentum floor (cfi default 1.0) |
| `clampMomentumFloor` | derived → **0.25 GeV** | `<0` ⇒ `1.25 × plimit` |
| `maxMomentumStepFactor` | `2.0` | relative GN trust region |
| `stepBacktracking` | `True` | Armijo χ² backtracking from iteration 2, ≤4 halvings, slack 1.0 |
| `propagationDirection` | `anyDirection` | 11 backward legs recovered in the 1000-event smoke, 0 propagation failures |
| `numberOfThreads` | `1` | Geant4e; also makes the output exactly `globalcor_0.root` |
| **no** `BeamSpotProducer` | | MiniAOD already carries `offlineBeamSpot`, which is the *hard-coded* tag the maker consumes (`ResidualGlobalCorrectionMakerBase.cc:319`). Adding a second producer with that label would shadow it |

### `tightG4eStepper` — the option was inert, and is still bit-inert here

Two separate things, both worth writing down.

**1. The knob was wired to the wrong PSet.** `runCvhJpsiGenMC.py` and
`profiling/runCvhProfile.py` implement `tightG4eStepper` as a patch of
`process.geopro.MagneticField.ConfGlobalMFM.OCMS.StepperParam`. In this release
`geopro` is neither scheduled nor consumed — `CvhMaster`/`CvhWorker` own the G4
world and build the field manager from **`cvhMasterESProducer.MagneticField`**
(`CvhMaster.cc`: `m_pField(p.getParameter<edm::ParameterSet>("MagneticField"))`
→ `sim::FieldBuilder`; `CvhWorker.cc:87` does the same per thread), and
`geantRefit_cff` hands `geopro` an *independent* clone of the same g4SimHits
PSet. So the geopro patch cannot reach the propagation. **The currently running
J/psi production is therefore running at the loose CMSSW defaults**, whatever
its `[cvh]` echo says. `runCvhDimuonMiniAOD.py` patches
`cvhMasterESProducer.MagneticField` (and keeps `geopro` in step).

**2. With the knob correctly wired it makes no difference — measured.** Two
300-event smokes on the same events, `tightG4eStepper=True` vs `False`, are
**bit-identical**: `root_bitcompare.py` reports all 217 branches identical,
0 branches unique to either. That is the expected answer and not a
contradiction of the SIM result: the CVH propagation caps the G4e step at
10 mm (`/geant4e/limits/stepLength 10.0 mm`, applied in `CvhMaster::initG4`),
and over 10 mm a 45 GeV track in 3.8 T has a sagitta of ~3e-4 mm, so the RK
integration error never approaches even the tight 1e-5 mm tolerance. The
tolerance binds in the *simulation*, where steps are metres long. It does not
bind in the refit.

Consequence: the production runs `tightG4eStepper=True` (it is the defensible
setting and it is free), and **the Z and J/psi legs are not made inconsistent
by the J/psi leg's inert knob**, because the setting has no effect on either.
If the 10 mm step cap is ever raised, this stops being true and both drivers
have to be checked.

### Global parameter map (`runtree`, 126 452 entries) — POOLABLE with the J/psi leg

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
| **total** | **126 452** | identical to `jpsimc_20M_260905` |

The counts matching is necessary but not sufficient — the two files must give
the same *index* to the same physical parameter. Checked directly against a
20-event J/psi reference run made with the exact production arguments of
`config_jpsimc20M.sh`: **the two `runtree`s are bit-identical**, all 36 branches
and all 126 452 entries (`root_bitcompare.py --trees runtree` → IDENTICAL).
That covers `parmtype`, `rawdetid`, `iidx`, `subdet`/`layer`/`glued`/`stereo`,
and also `xi` (the material payload) and `bz`/`b0`/`baxial`/`bradial` (the
field at each module). **The two productions can be pooled in one global fit
without re-mapping.**

> ⚠️ Neither can be pooled with the 260715/260717 **data** productions, which
> had `nglobalparms = 81 824` (the same map *without* the four doRes families)
> — a `doRes=True` file and a `doRes=False` file index the same physical
> parameter differently.

## Input

`filelist_dymc_8p5M_260905.txt` — the 104 files; `.counts.txt` adds the event
count per file; `scan_dymc_260905.txt` is the `path nevents` scan
`mkchunks.py` consumes.

**No ROOT-open zombie scan was needed here**, unlike the J/psi repack. These
files came off the grid through `transfer/transfer_one.sh`, which publishes a
file at its final name only after an independent size + adler32 check against
DAS, so *present at its final name* means *verified*. All 104 were re-checked
present and size-correct against `transfer/dy_miniaod_subset.tsv` when the scan
file was written. The array task still asserts `attempted>0` in the fit
summary: the silent-empty-output failure mode also covers a ceph EPERM on the
node, which no pre-scan can rule out.

More files of this dataset are still arriving (1731 files / 4.386 TB total);
this production is the 8.5M-event head only. A later extension re-runs
`transfer/make_filelists.py`, then `mkchunks.py` on the new scan, into a **new
tag** — resuming into this one would renumber the tasks.

## Chunking

104 files × ~82k events (min 2 349, max 88 479). One whole file is ~7 h of
Geant4e, so each is split into event ranges via `skipEvents`/`nEvents`, sized
by a target of **25 000 events per task** rather than a fixed N-way split.

```
mkchunks.py scan_dymc_260905.txt --nfiles 104 --nchunk 4 --target-events 25000 \
    --filelist filelist_dymc_8p5M_260905.txt \
    --chunks   chunks_dymc_8p5M_260905.txt \
    --counts   filelist_dymc_8p5M_260905.counts.txt
```

→ **380 tasks**, mean 22 375 events, min 2 349, median 22 120, max 30 929;
4 chunks for 83 files, 3 for 8, 2 for 11, 1 for 2. Validated to tile every file
exactly: no gaps, no overlaps, Σ chunk events == Σ file events == 8 502 597,
104 distinct basenames.

## Smoke results

Three runs, all on `04F1216D-…-46B4CA49E67A.root`, all with the production
configuration (`config_dymc8p5M.sh`), under
`/ceph/…/cvh/dymc_8p5M_260905_smoke/`.

| | 300 events (`tight`) | 1000 events (`timing`) |
|---|---|---|
| candidates | 146 attempted, **145 succeeded**, 1 failed (0.68 %) | 460 attempted, **459 succeeded**, 1 failed (0.22 %) |
| the failure | `fail[kinfit]=1` — the seed `KinematicParticleVertexFitter`, not the CVH fit | same |
| propagator | 12 603 calls, **0 failures**, every exit counter 0 | 40 232 calls, **0 failures**, every exit 0, 11 `backwardLegs` |
| candidates/event | 0.487 attempted | **0.460 attempted, 0.459 succeeded** |
| `cfmass_ok` | **145/145** | **459/459** |
| `nParms` | mean 240.6, median 236, max 313 | mean 242.5, median 238, max 327 |
| `nRank` / `nFactor` | 27.0 / 6 625 | 27.4 / 6 763 |
| `niter` | mean 2.7, median 3, max 9 | mean 2.7 |
| clamped / backtracked | 2 / 0 | 11 / 0 |
| tree volume | 4.8 MB → **34.1 kB/candidate** | 15.6 MB → **34.7 kB/candidate** |
| run tree | 12.4 MB compressed — **fixed per FILE**, not per candidate | same |
| peak RSS | 1.83 GB | **1.86 GB** (request 6 G) |
| CPU | 169 s user (contended) | **355 s user** ⇒ 0.323 s/event, **0.704 s/candidate** |

* `[cvh] effective:` echoed `CgfQoPMode=0`, `IoniTruncationAlpha=0.999`,
  `clampMomentumFloor=0.25 GeV` (derived), `maxMomentumStepFactor=2`,
  `stepBacktracking=1 (fromIter=2, maxChi2Backtrack=4, armijoC=0.0001,
  armijoSlack=1)`, `PropagationPtotLimit=0.2 GeV`, and the tight stepper values
  on the PSet that acts.
* Factored-Hessian branches present: `nRank`, `nFactor`, `hessfactorv`,
  `hessdroppedmass`, `hessrankgap`, plus `globalidxv`/`gradv`/`nParms`.
  **No `hesspackedv`/`nSym`** — `fillGrads=False`.
* **No step-record branches** (`ioniurban*`, `radstep*`, `msmoli*`, `reseigv`,
  `resinfv`, `resinfbv`) — `exportStepRecords=False`.
* `cfmass_*`: 10 branches, `cfmass_ok` true for every candidate.
* Where the bytes go: `hessfactorv` **71.5 %** (24.8 kB/cand), the two
  `*_jacRef` 15.4 % (5.4 kB), `Jpsi_jacMass` 2.6 %, `gradv` 2.5 %,
  `globalidxv` 1.1 %. The six `cfmass_*` arrays together are **1.4 %**
  (0.24 kB each) — the CF export is a CPU cost, not a storage cost.

### CF chain read test

Plumbing only — the Z is not a δ resonance and the "kernel" here is the Z gen
mass distribution, not an FSR kernel, so **every number below is meaningless as
physics**. What is being tested is that the in-maker export can be read and
carried through the existing likelihood machinery.

```
cf_inmaker.py pairs --files .../tight/globalcor_0.root \
    --mass-window 91.1876 30 --cache runs/cf_masspairs_dysmoke_260905.npz
        -> 140 selected, 5 dropped (the 5 with no gen match)
cf_masskernel_tt.py --window 120 --kernel-cache runs/cf_masskernel_dysmoke_260905.npz
cf_masslik_fit.py --model r --window-lo 60 --window-hi 120
        -> trust-exact success, 11 iterations, 1.4 s
           |grad|inf = 3.0e-6, Hessian positive definite, cond 6.2e4
           L_i < 0 for 0/140 candidates
```

**`--mass-window` had to be added to `cf_inmaker.py`**: `build_pairs_tt` had the
J/psi acceptance `|m_gen − 3.0969| ≤ 0.35` hard-coded, so a Z production came
back "0 selected out of 145" with the misleading built-in diagnosis that the
maker had not filled the branches. The default is unchanged (3.0969, 0.35), so
every existing cache is bit-identical; a Z needs `--mass-window 91.1876 30`.
`cf_masslik_fit.py` still measures `mobs` relative to `MJPSI` and puts the
resonance δ there, which is why `--window-lo/-hi` have to be given in absolute
mass and why `alpha` is not interpretable — **a real Z fit needs a Z resonance
model in that module**, and that is the next piece of work on this leg, not a
production issue.

## Scale, as submitted

| | |
|---|---|
| files | 104 (all verified, none dropped) |
| events | **8 502 597** (mean 81 756/file) |
| tasks | **380** (mean 22 375 events; range 2 349…30 929) |
| slurm | array **`dymc8p5M_a0`**, indices 0–379, `%200` |
| candidates | 0.460/event ⇒ **~3.91 M attempted, ~3.90 M succeeded** |
| output | 3.90 M × 34.7 kB + 380 × 12.4 MB ≈ **140 GB** |
| CPU | 8.50 M × 0.323 s ≈ **765 core-hours** |
| wall/task | ~2.0 h typical, ~2.8 h for the largest chunk (12 h limit) |
| wall total | 380 × 2.0 h / N_concurrent — **~3.8 h at 200 slots**, if 200 slots were actually available |

Against `profiling/NOTES.md`'s projection (0.311 s/event, ~700 core-h for 8M):
+4 % on CPU/event, and the volume is 34.7 rather than 28.1 kB/cand because the
profiling runs had `fillJac=False`. Both are as expected.

**Concurrency.** Slurm here has `MaxArraySize=1001` (380 fits in one array),
`MaxJobCount=50000`, and QOS `normal` `MaxJobsPU=5000`; the queue already holds
the 1 642 J/psi tasks, so 2 022 total is well inside every limit. Nothing forces
the concurrency down, so `%200` matches the J/psi arrays. The *binding*
constraint is not a limit but fair share (0.025 at submission, 8 of 1 643 J/psi
tasks running against a full cluster): the two productions share it, and the Z
leg will interleave with the J/psi one rather than adding throughput.

## Commands

```bash
cd /work/submit/david_w/ZMass/calibration_studies/production
./submit_dymc8p5M.sh --dry-run   # print the sbatch line without submitting
./submit_dymc8p5M.sh             # submit
./status_dy.sh                   # complete/running/failed, events, projected finish
./status_dy.sh --slow            # + output bytes and candidate counts (walks ceph)
./resume_dy.sh --list-only       # which chunks are missing
./resume_dy.sh                   # resubmit only those (skips anything still queued)
python smoke_inspect.py <out.root> [--compare <jpsi out.root>]   # schema + poolability
```

`resume_dy.sh` keys off the `.complete` sentinel, never the `.root`: cmsRun
creates its output at *start*, so a killed task leaves a non-empty but
truncated file. The task index is the chunk-list line number, so a resumed task
reads exactly the same events as the one it replaces.

## Notes

* **submit82's ceph client is evicted** (`stat /ceph/submit/data` → EPERM).
  All ceph work for this setup was done from submit81. The array task asserts
  input readability on its own node for the same reason.
* The `file:<path>,<skip>,<max>` filelist in
  `resolution/simprod/filelist_dy_miniaod_260905_evt3000.txt` **has no parser** —
  feeding it raw to `input=` splits on the commas as path separators and, with
  `skipBadFiles=True`, silently processes the entire file. It is not used here;
  `chunks_dymc_8p5M_260905.txt` (`path skip nev`, whitespace separated) is.
* `produceValueMaps=False`: this driver has no output module, so the EDM
  ValueMaps would be produced and dropped. The profiling runs had them on; the
  difference is not measurable.
