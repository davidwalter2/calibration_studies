# Maker cleanup + vertex-constraint-on-by-default + re-analysis productions

Owner: cleanup agent (job 28e0dfa8).  Area
`/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2` (git repo at `src/`).

## Scope
Three jobs, in order, with gates:
1. consolidate `cvh-exports-260906` + `perhit-residual-cf-260910` +
   `vtxres-cf-260911` onto `cvh-exports-clean-260911` and clean the FULL diff
   against the merge base;
2. `doVtxConstraint = True` by default + export the unconstrained mass;
3. launch the two re-analysis productions, then STOP.

## Facts pinned at start
| item | value |
|---|---|
| merge base (upstream `my-cmssw/WmassNanoProd_15_0_19_patch2_dev`) | `4feac6f93d0e5b866d6d6424c0f0ac9c1a5ec2ee` |
| HEAD at start (`vtxres-cf-260911`) | `4b3984312f67f94bdb49cca0806bce58ff170b62` |
| commits in the PR | 45 |
| PR diffstat | 47 files, +14240 / -487 |
| running jobs at start | NONE (`squeue`, `condor_q`, `ps` on submit50/51/52/82 all empty) -> dev2 is safe to build |
| ceph | NOT readable from submit82; use submit50/51/52 |
| helpers | `/home/submit/david_w/.claude-work/jobs/28e0dfa8/s5{0,1,2}.sh '<cmd>'`, `build_dev2.sh` |
| work dir | `/home/submit/david_w/.claude-work/jobs/28e0dfa8/tmp/makercleanup` |

`cvh-exports-260906` is AT the base commit `ca6058d96fc`; `perhit-...-260910`
and `vtxres-cf-260911` are LINEAR descendants of it, so `vtxres-cf-260911`
already contains all three branches' work and the clean branch is cut from it.

## Status
- [x] step 0: orient, confirm no running production, confirm build is current
- [ ] GATE 1 baseline runs (in flight)
- [ ] job 1 cleaning
- [ ] GATE 1 comparison
- [ ] job 2
- [ ] GATE 2
- [ ] job 3 launch

## GATE 1 — the baseline runs
Scripts `$W/gate_tt.sh`, `$W/gate_st.sh` (`W = /home/submit/david_w/.claude-work/jobs/28e0dfa8/tmp/makercleanup`),
run through `s50.sh`/`s51.sh` so ceph is readable.  Every export switch ON,
`doVtxConstraint` OFF:
* two-track: `runCvhJpsiGenMC.py`, 200 events of
  `resolution_simprod_jpsigun_ul16/task_0000/step2.root`, the vtxres study's fit
  settings + `exportVtxResidual exportCfExponents exportCfGroupExponents
  exportHitResBlocks exportStepRecords exportMaterialNoise exportVarianceGrads
  exportObjective fillGrads fillGradsFactored fillJac` -> `$W/base_tt`
  (192 candidates, 270 + 36 branches over `tree` + `runtree`).
* single-track: `runCvhResClosure.py`, 200 events of
  `resolution_simprod_mugun_ul16/task_0000/step2.root`, `exportPerHitResidual
  perHitCfGroups perHitRefComponents perHitInfluenceBlocks` + the same export
  switches -> `$W/base_st`.
Comparator: `$W/cmp_trees.py` (uproot, all trees, every branch, exact equality,
`--allow` for branches whose redefinition is intended).  Container python:
`$W/py.sh` (the wmassdev image; the `mfs` venv has no numpy on submit82).

## Job 1 — what was removed (branch work, uncommitted at this point)
Cleaning was done by five workers on DISJOINT files (four subagents + this
agent), then verified to be comment-only by diffing with comments stripped.

| area | files | date stamps | history sentences | notebook/study cross-refs | dead code |
|---|---|---|---|---|---|
| `ResidualGlobalCorrectionMakerBase.{cc,h}` | 2 | 12 | 14 | 6 | 0 |
| `ResidualGlobalCorrectionMakerG4e.cc` | 1 | 11 | 18 | 8 | 0 |
| `...NTrackG4e.cc` + Geant4e (`CvhCfExponents`, `CGFQoPBlock`, `Geant4ePropagator`, the extrapolator tables, `XrdRequestManager`, `ProcessActivationWatcher`) | 17 | 16 + 3 commit hashes | ~40 | ~18 | 1 member + 2 orphaned comment blocks |
| python drivers + `doc/resolution-cf-export.md` | 14 | ~23 | ~28 | ~19 | 1 unused tuple in `cvhSwitches.py` |
| `...TwoTrackG4e.cc` (this agent) | 1 | 8 | 9 | 3 | the `CVH_VTX_DEBUG` getenv + its 16-line dump, `vtxConstraintZeroSeed` |

Behaviour-affecting items resolved (the only intended output changes):
* **`vtxConstraintZeroSeed` folded into `doVtxConstraint`.** A vertex
  constraint means DCA = 0, always; the flag, its cfi/VarParsing plumbing and
  the seed-DCA behaviour are gone. Freezing index 6 at the SEED's DCA was never
  a common-vertex constraint.
* **`Jpsi_d` / `Jpsicons_d`** no longer multiply the swap-invariant theta_6 by
  q(leg 0). `Jpsi_d = statepcaupd[6]`, the raw signed DCA
  `n_hat . (x_b - x_a)` with `n_hat = (p_a x p_b)^`. They are NOT the same
  quantity as `Jpsi_vtxres`: with the constraint ON, `Jpsi_d` is identically
  zero and `Jpsi_vtxres` is the DCA the unconstrained fit would have reported;
  with it OFF the two coincide. Both branches are kept.
* A pre-existing change-log block in `...G4e.cc` about the removed
  `nValidHitsFinal`/`nValidPixelHitsFinal` branches (a superseded export and
  its branches) was deleted.

## GATE 1 — PASSED (2026-09-11)
Build: `scram b -j32` twice (the second pass because one cleaning worker was
still editing when the first started), exit 0, zero errors.
Re-ran both makers on the SAME inputs with every export switch ON and
`doVtxConstraint=False`, and compared every branch of every tree.

| maker | branches compared | identical | different |
|---|---|---|---|
| two track (`tree` 192 cand + `runtree`) | 306 | **305** | 1 — `Jpsi_d`, the documented redefinition |
| single track (`tree` 400 trk + `runtree`) | 217 | **216** | 1 — `phcf_msec` |

* `Jpsi_d`: 100/192 candidates flip SIGN and nothing else — exactly the
  candidates whose leg 0 is the mu-, i.e. the `q(leg 0)` factor that was
  removed. `Jpsicons_d` is identical because `doMassConstraint` is off in this
  configuration, so the icons == 1 pass never runs.
* `phcf_msec` is the WALL CLOCK of the per-hit CF loop
  (`ResidualGlobalCorrectionMakerG4e.cc`, "the wall clock of this loop, to be
  read against the ~2 s the fit itself costs"), so it cannot be bit-identical
  between two runs. 1986.1 ms vs 1975.2 ms, 0.6 %.
* Two branches exist only in the new file: `Jpsi_mass_unc`, `Jpsi_covmassvtx`
  (job 2).

Comparator `$W/cmp_trees.py`; outputs `$W/{base,after}_{tt,st}`.

## Job 2 — the vertex constraint is ON by default
`doVtxConstraint` flipped False -> True in the 8 channel cfis
(`ResidualGlobalCorrectionMaker{DiMuon,TwoTrackJpsiMuMu,TwoTrackJpsiKMuMu,
TwoTrackUpsilonMuMu,TwoTrackZMuMu,TwoTrackPiPi,TwoTrackKPi,TwoTrackProtonPi}
G4e_cfi.py`) and in the drivers that set it inline
(`test/{RunGlobalCorRecJpsiData,RunGlobalCorRecJpsiAlca,runCvhJpsi}.py`) and
as a VarParsing default (`test/{runCvhJpsiGenMC,runCvhDimuonMiniAOD}.py`).
`runReferenceJacobianGate.py` already defaulted its own `refVtxConstraint` to
True. The N-track maker accepts the parameter but has nothing to do with it
(its common vertex is imposed by the parameterisation), so the J/psi-K
configs that clone `TwoTrackJpsiKMuMuG4e_cfi` are unaffected.

NOT CHANGED (owned by another agent this session):
`calibration_studies/production/{config_dymc8p5M.sh,config_jpsimc20M.sh,
threadscan/run_scan.sh,condor_dymc_v2/config_dymc_v2.sh,
condor_jpsimc_v2/config_jpsimc_v2.sh}` and
`resolution/vtxres/run_prod_dy.sh` each pass `doVtxConstraint=False`
EXPLICITLY on the cmsRun line, so they override the new default and must be
edited (or the argument dropped) by whoever owns those files.
`resolution/vtxres/run_prod.sh` does NOT pass it and picks the new default up.

### The new export
`Jpsi_mass_unc` (GeV) and `Jpsi_covmassvtx` (GeV cm), both under
`exportVtxResidual`, computed where the vertex residual is formed. Derivation
in the code and in `doc/resolution-cf-export.md`; the slope is
`-(C a_f).h_f6`, one back-substitution, and sigma_v^2 cancels out of the mass.
The MASS functional's CF exponents and influence weights are left exactly as
the fit computes them in the constrained regime.

## GATE 2 — PASSED (192 J/psi-gun candidates, the PRODUCTION export set)
`$W/gate_tt_prod.sh` run twice on the same 200 events, `doVtxConstraint`
False and True, matched on (run, lumi, event); 192/192 matched.
(The first gate pair, with `exportMaterialNoise=True`, gives IDENTICAL (a) and
(b) numbers -- the parmtype-15 blocks do not touch the fit -- but its
fam-15 blocks are a RE-PARTITION of the 10/11 noise, so a sum over all
exported blocks double-counts and gate (d) reads rms 0.023 instead of 0. The
production sets `exportMaterialNoise=False`, and that is the configuration
quoted below.)

| gate | number |
|---|---|
| **(a) `Jpsi_mass_unc`(ON) vs `Jpsi_mass`(OFF)** | rel. diff median **5.30e-5**, p90 5.58e-4, p99 1.45e-3, max 3.37e-3; signed mean −3.96e-5, rms 4.30e-4 |
| for reference, `Jpsi_mass`(ON) vs `Jpsi_mass`(OFF) | median **7.29e-4**, p90 3.73e-3, max 4.94e-2 — the one-step identity removes a factor 14 in the median; what is left is the second-order term |
| **(b) r_v(frozen) vs r_v(free)** | \|dr\|/sigma_v median **1.89e-3**, p90 1.91e-2, max 0.26 (the study measured 1.5e-3 / 1.5e-2 / 0.26) |
| **(b') sigma_v** | \|ratio−1\| median **8.45e-4**, p90 6.55e-3 (study 8.7e-4 / 6.0e-3) |
| **(c) closure, vertex** | max \|sum_b\|a_b\|^2/sigma_v^2 − 1\| = **2.90e-6** frozen, 4.46e-6 free (the maker's own `Jpsi_vtxvchk` agrees to the last digit) |
| **(c) closure, mass** | max \|sum_b\|a_b\|^2 − (resinfcov+resinfcovhit)\|/(…) = **3.69e-7** (float32); against sigma_m^2, **3.82e-6** |
| **(d) corr(w_mass, w_vtx) in the V metric** | frozen: mean +0.0000, **rms 0.0000, max \|rho\| 3.3e-6**; free: mean −0.005, **rms 0.194**, max 0.877 |
| the exported covariance element | `cov(m,theta_6) sigma_v^-2 r_v` reproduces `m_u − m_c` to **2.19e-7 GeV** |
| ndof | `ndof(frozen) − ndof(free) = 1` on EVERY candidate |
| `Jpsi_d` under the constraint | identically **0** on every candidate (the seed zeroing works) |

One candidate in 192 is the documented degenerate population (`sigma_v` = 76 m,
`Jpsi_vtxvchk` = 1); with the constraint on it comes out as `sigma_v = 0`,
`Jpsi_vtxok = False`. It is excluded from (c) and (d) by the study's standard
`--max-vchk 1e-4` cut. Not a new defect.

## Branch and commits
`cvh-exports-clean-260911`, cut from `vtxres-cf-260911` (which already
contained `cvh-exports-260906` and `perhit-residual-cf-260910` linearly):
* `b0fb25a154e` CVH export development: the comments describe the final state
* `3b4931af605` The two-track fit constrains the vertex by default, and exports both masses

## Job 3 — the re-analysis productions (LAUNCHED 2026-09-11 23:19)
Slurm arrays on partition `submit`, driven by
`calibration_studies/slurm/array.sbatch` (READ ONLY -- the two submitters live
in `$W/submit_{gun,dy}_vtxon.sh` so nothing under `calibration_studies/` is
modified). Both confirmed running with `doVtxConstraint=True` on the cmsRun
line and the right inputs.

| job | id | array | mem | time | output |
|---|---|---|---|---|---|
| J/psi gun, two track | **6432438** | 0-159%60 | 2200M (1.3x the 1.61 GB measured) | 6 h | `/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/prod_vtxon/task_NNNN/` |
| DY MiniAOD (Z-like) | **6432439** | 0-5 | 5G | 12 h | `.../runs_vtxres_260911/dy_vtxon/task_NNNN/` |

Expected wall time: the gun is ~31 min/task at 60 concurrent, so ~1.5 h for
all 160; the DY leg runs 6 tasks x 4000 events in parallel. Slurm logs in
`<outdir>/logs/`.

Settings are the vtxres study's verbatim, with ONLY `doVtxConstraint=True`
added (gun) or substituted (DY):
* gun: `nEvents=-1 numberOfThreads=1 doRes=True fillGrads=True
  fitFromGenParms=False scalarPot3DInitFile=<lmax18_custom50>
  trackSrc=generalTracks useLegacyPairLoop=True doTrigger=False
  applyHltFilter=False useIdealGeometry=True useDefaultField=True
  globalTag=150X_mcRun2_asymptotic_v1 CgfQoPMode=0 armijoSlack=1.0
  doVtxConstraint=True exportVtxResidual=True exportCfGroupExponents=True
  exportStepRecords=False`
  filelist `resolution/simprod/filelist_jpsigun_ul16.txt` (160 files x 2000 ev)
* DY: the `run_prod_dy.sh` COMMON verbatim with `doVtxConstraint=True`,
  filelist = the SAME first 6 files of
  `production/filelist_dymc_8p5M_260905.txt`, copied to `$W/filelist_dy6.txt`.

NOTE for the resumed session: `array.sbatch` does NOT write the `.complete`
sentinel that `run_prod.sh` uses, so a resume must check the output file size /
the slurm exit code, not a sentinel.

## STOP POINT
Jobs 1-3 done. Waiting for the coordinator to resume for the re-analysis once
the `calibration_studies/` cleanup finishes.
