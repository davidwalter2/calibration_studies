# CVH calibration productions — reference

Everything needed to *use* or *re-run* the large CVH refit productions: which
samples exist, what is in them, where they live on ceph, the exact
configuration, the cost and resource numbers, and the defects that bite.

(The two condor campaigns keep a short record of their own next to their
submit scripts: `condor_jpsimc_v2/STATE_jpsi_v2.md`,
`condor_dymc_v2/STATE_dy_v2.md`.)

---

## 1. Inventory

| tag | channel / input | events | tasks × streams | candidates (succ.) | volume | batch | use |
|---|---|---:|---|---:|---:|---|---|
| **`jpsimc_20M_260906_v2`** | UL16 `JPsiToMuMu_Pt8toInf` MC ALCARECO (`TkAlJpsiMuMu`), two-track | 21 750 740 | 1645 × 4 | **21 678 062** (0.9967/ev, 0.0067 % fail) | 1.5 TB | HTCondor / CMS global pool | **the J/psi sample** |
| **`dymc_8p5M_260906_v2`** | UL16 DY MiniAODv2 (`DYJetsToMuMu_H2ErratumFix…powhegMiNNLO`), two-track Z | 8 502 597 | 380 × 4 | **3 799 624** (0.447/ev, 0.16 % fail) | 267 GB | HTCondor / CMS global pool | **the Z sample** |
| `jpsimc_20M_260905` | same J/psi chunks, older exports | 21 719 059 | 1642 × 1 | ~21.6 M | 778 GB | slurm | cross-check only; 1633/1642 complete, **12 tasks are garbage** |
| `dymc_8p5M_260905` | same DY chunks, older exports | 182 of the 380 chunks | 216 dirs × 1 | ~1.8 M | 71 GB | slurm | cross-check only; **182/380 complete**, leg cancelled |
| `jpsimc_20M_260906_v2_nulcheck` | 7 chunks re-run with repacked inputs | — | 7 × 4 | 102 507 | 6.9 GB | HTCondor | evidence only, **deletable** (proved bit-identical) |
| `jpsimc_20M_260905_opera3d` | first, cancelled J/psi submission (`useOpera3D=True`) | — | 0 complete | — | — | slurm | kept for its `PROVENANCE.txt` only |

Output root: `/ceph/submit/data/user/d/david_w/ZMass/cvh/<tag>/task_XXXX/globalcor_<stream>.root`
(plus `local.log`, the `.complete` sentinel, `logs/`, and for the v2 productions
`payload/` — the pinned CMSSW overlay — and `PROVENANCE.txt`).

The v1 (slurm, 260905) and v2 (condor, 260906) legs run the **same chunk lists**,
so `task_NNNN` means the same event range in both and they can be compared task
by task. v2 adds the 2026-09-06 exports (§3) and the `ndof == 0` fix; **the
analysis uses v2.**

Related samples produced by the same machinery but outside this reference: the
private gun samples (`resolution/simprod/SAMPLES.md`), the per-species
resolution closures (`cvh/hitres*`), and the 2016 DATA calibration productions
`jpsi_calib2016_grads50_globalmat_260717` / `_260715` (which have
`nglobalparms = 81 824`, i.e. **no** doRes families — see the pooling warning in
§4).

---

## 2. Where to run from

| | |
|---|---|
| CMSSW area | `/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2`, branch **`cvh-exports-260906`** |
| spare | `/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev`, branch `WmassNanoProd_15_0_19_patch2_dev`, fast-forwarded to the same commit and rebuilt — identical tree, built |
| release / arch | `CMSSW_15_0_19_patch2`, `el9_amd64_gcc12` (runs natively on submit; no container) |
| drivers | J/psi: `Analysis/HitAnalyzer/test/runCvhJpsiGenMC.py`  ·  Z: `runCvhDimuonMiniAOD.py` |
| both v2 productions ran | `ca6058d96fc` (the recovered tasks) / `fab515e4cdf` (the bulk) — physics bit-identical, the two commits are I/O and locking only |

Every live script (`condor_{jpsimc,dymc}_v2/config_*.sh`, `array_jpsimc_v2.sbatch`,
`array_dymc_dev2.sbatch`, `threadscan/run_scan.sh`, `repack_fix_260907/*`)
defaults to **dev2**.

**Keeping the two areas in step:** commit on `cvh-exports-260906` in dev2; when
validated, `git merge --ff-only cvh-exports-260906` in dev and rebuild. If that
merge is ever *not* a fast-forward the two have diverged and the divergence must
be understood before either runs a production. Sparse-checkout files are
**per worktree** (`.git/info/` vs `.git/worktrees/src/info/`), so a pattern added
in one area does not propagate — `Utilities/XrdAdaptor` had to be added by hand.

**Never `scram b` in an area a production is running from.** A relink swaps the
`.so` under the running jobs and they segfault in the same second. The v2 condor
jobs are immune: they run from a payload tarball pinned in the production's own
output tree (`<OUTBASE>/payload/overlay_<tag>[_xrdfix].tgz`).

Gate for a build: `resolution/smoke_exports_260906.sh <CMSSW_AREA> <OUTROOT>
[gun_tt|gun_st|data_tt …]` — three cmsRun jobs between them touching every code
path the export work changes (J/psi-gun two-track 59 candidates, low-pT mu-gun
single-track 120 tracks, 48 events of the 15_0-native 2016F Charmonium
ALCARECO). An empty tree is a FAILED smoke; compare two areas branch by branch
with `root_bitcompare.py` (its `--exclude` exists so that a storage switch can
be tested at all — without it a missing branch alone reports DIFFERENT).
Reference outputs: `/work/submit/david_w/ZMass/scratch_smoke_260906/`
(`v6_default`, `v7_xrdfix`, `v8_devmerged` — all bit-identical: 246 / 157 / 264
branches, `runtree` 126 452 entries).

---

## 3. What the output contains

Common to both legs:

| option | value | what it does |
|---|---|---|
| `doRes` | `True` | registers the resolution families (parmtypes 8/9/10/11) and enables the CF export block |
| `exportCfExponents` | `True` | the `cfmass_*` exponents on the 64-point τ grid (the default; named explicitly for the record) |
| `exportStepRecords` | **`False`** | the slimming: 430 kB/candidate of raw per-step records whose only consumer was the offline exponent extractor that the in-maker export replaces. Proven inert — all 232 common branches bit-identical, the only schema difference being exactly the 10 step-record branches |
| `fillJac` | `True` | per-track ∂x_ref/∂a. Costs +6.2 kB/candidate (`Mu{plus,minus}_jacRef` + `Jpsi_jacMass`) |
| `fillGradsFactored` | `True` | the low-rank `H = BᵀB` as `nRank`/`nFactor`/`hessfactorv` |
| `fillGrads` | `False` | the packed Hessian is the thing being replaced. Not mutually exclusive with the factored one — both can be written; we want only the factored. It does **not** save CPU: the packing runs unconditionally |
| `fillRunTree` | `True` | the global parameter catalogue. Without it `globalidxv` cannot be interpreted at all |
| — | | **`doRes` and `fillGradsFactored` are only expensive TOGETHER** (0.062 and 0.064 s/cand alone, 0.63 for the pair): the in-maker CF export is gated on `doRes_ && (fillGrads_ \|\| fillGradsFactored_)` |

The v2 productions add four switches:

```
exportCfGroupExponents=True exportMaterialNoise=True \
exportVarianceGrads=True varianceGradFamilies=15
```

| export | branches | what it is for |
|---|---|---|
| per-material-group CF exponents (`exportCfGroupExponents`) | `cf{qop,mass}_grp`, `_grp_ms`, `_grp_ioni_re/_im`, `_grp_rad_re/_im`, `_grp_del` (single-track), `_grp_closure` | lets the mass term float the **parmtype-15 material amounts** the hit chi2 already measures, instead of the four unphysical `k_hit/k_ms/k_ioni/k_rad` knobs. **+26 kB/candidate and +0.28 s/candidate** — the one expensive switch |
| hit-class blocks in the two-track maker (`exportHitResBlocks`, default True) | `reshitcls`, `resinfcovhit`, `cf{qop,mass}_hitcls`, `_hitv`, plus parmtype-8/9 entries in `reseigidx`/`resinfvarv`/`resinfv`/`reshitidx` | the per-hit-class resolution parameters had never been fitted: the two-track maker registered no hit blocks at all. `False` reproduces a pre-2026-09-06 tree exactly |
| the two legs' reference covariance (always on, no switch) | `Jpsi_covrefmom` (21 floats, upper triangle of the 6×6), `Jpsi_jacrefmom`, `Jpsi_qoprefplus/minus`, `Jpsi_sigmarelplus/minus`, `Jpsi_rhomom`, `Jpsi_fang` | makes the Jensen and self-consistent-σ corrections **truth-free on DATA**. 148 B/candidate |
| pre-FSR gen mass (always on, needs `doGen=True`) | `Jpsigenpre_mass`, `Jpsigenpre_status`, `Jpsigenpre_masslep`, `Jpsigen_massdressed` | the Z channel's FSR kernel comes off the production's own pairs cache instead of a separate FWLite pass. 8.5 B/candidate |
| material-group process noise (`exportMaterialNoise`) | `resinfcovgrp` + parmtype-15 entries in `reseigidx`/`resinfvarv` | the quadratic hit-chi2 term now differentiates a group's WIDTH as well as its mean loss. Single-track: +17 ms and +3 kB per track. On the two-track maker it only *registers* the blocks — `exportVarianceGrads` is what puts them in the gradient |
| two-track variance (log-det) gradient (`exportVarianceGrads`) | `gradchisqv`, `gradllv`, `nHessVar`, `hessvaridxv`, `hessvarpackedv`, and CHANGED values on the parmtype-15 columns of `gradv`/`hesspackedv`/`Jpsi_jacMass`/`*_jacRef` | the mean-loss channel is the weak one: on the J/psi gun the parmtype-15 Fisher information goes from 4.07 to **166.3 (41×)** once the width is differentiated |
| per-leg reference energy loss (always on) | `Mu{plus,minus}_maxfracloss`, `dEref`/`maxfracloss` (single-track) | a `dE_ref/p < 0.01` quality requirement on the quadratic term's material information, without the step records |
| material group on every step record | last column of `ioniurbanv` / `radstepv`; new `ioniurbanstride` | deletes the offline `pair_ioni_rows` heuristic. Only visible with `exportStepRecords=True` |

**`varianceGradFamilies=15`, not the default.** The default (empty) is
`{8,9,10,11,15}`. Family 15 is layout-preserving — the material-group globals
are already columns of `globalidxv`, so `nParms`, `gradv`, `jacrefv`,
`Jpsi_jacMass` and `hessfactorv` keep their shapes and the output still pools
with `jpsimc_20M_260905`. Families 8–11 APPEND per-module columns (median
`nParms` 238 → 324, `nHessVar` 21.8 → 109.9) and cost 8× more, for parameters
this maker cannot apply back anyway (it does not scale the hit covariance by
`exp(corparms)` the way the single-track maker does).

`exportObjective` and `varianceFDGlobalIdx` are validation-only (an
`ncons × ncons` eigendecomposition, resp. a full re-profile, per candidate) —
leave them off. `genResonancePdgIds` defaults to
`{23, 443, 100443, 553, 100553, 200553}` and only needs setting for an unusual
resonance. The branch-by-branch contract is
`Analysis/HitAnalyzer/doc/resolution-cf-export.md` in the release area.

### Volume of the new exports

Compressed **kB per tree entry**, step records off, measured on the three smokes:

| sample | baseline | new defaults | + per-group exponents | + material noise |
|---|---:|---:|---:|---:|
| J/psi gun, two-track (59 cand) | 110.26 | 110.79 | **136.40** | — |
| 2016F data ALCARECO, two-track (48 cand) | 31.18 | 31.71 | **55.78** | — |
| low-pT mu gun, single-track (120 trk) | 61.51 | 61.60 | **87.98** | 64.67 |

i.e. the per-group exponents are +23 % / +76 % / +43 %, everything else together
is **0.3 kB/candidate**. Per branch group (bytes/candidate): `cf*_grp*`
24 650–27 013, `Jpsi_covrefmom`+jac+derived ~150, `cf*_hitcls`+`_hitv` ~67,
`reshitcls` ~47, `maxfracloss`+`dEref` ~25, `resinfcovhit`+`resinfcovgrp` ~5,
`Jpsigenpre_*` ~9.

The variance block costs **+7.0 %** at family 15 (24 gun ditracks: 37 902 →
40 568 B/cand) and **+105 %** at all families; on 48 real data candidates
32 473 → 34 847 B/cand (+7.3 %). Carrying it as extra ROWS of `hessfactorv`
instead (an equally exact alternative) costs +57 % / +429 % — the separate
packed block is 8× cheaper at family 15. `globalfit/extract.py` knows how to add
it, and REFUSES a factored file whose `gradllv` is filled but which has no
`hessvaridxv` (an inconsistent `(G, K)` pair).

**If the per-group exponents ever have to be smaller**, the measured options, in
order of return: (1) a rank-16 PCA of the τ axis against a fixed basis shipped
in the runtree — 6.8 kB/candidate for a relative error of 4.2e-7 (ms), 1.3e-5
(ioni), 1.2e-3 (rad), not implemented because the raw rows are exported
precisely so the basis can be fitted rather than guessed; (2) a coarsened group
tier — 24 of the 42 groups are active-silicon layers carrying under 1 % of the
resolution each, merging them per subdetector takes 42 → ~20 with no measurable
loss to the mass term while the hit chi2 keeps its own 42; (3) pruning rows
below 1e-3 of the candidate's own max |S| removes 5.15 % of them. (1)+(2) would
be ~3.2 kB/candidate.

---

## 4. The global parameter map, and pooling

`runtree`, **126 452 entries**, bit-identical between the two legs and between
the 260905 and 260906 builds (verified on all three smokes and against a
20-event J/psi reference run: all 36 branches, all entries, including `xi` and
`bz`/`b0`/`baxial`/`bradial`).

| parmtype | count | what |
|---|---:|---|
| 0, 5 | 16 588 each | alignment local-x, θ_z (per detUnit) |
| 1 | 8 656 | alignment local-y (pixel + endcap) |
| 2, 3, 4 | 13 300 each | alignment local-z, θ_x, θ_y (per align-detid) |
| 8 | 16 588 | resolution: hit local-x (`doRes`) |
| 9 | 1 440 | resolution: hit local-y (`doRes`) |
| 10 | 13 300 | resolution: multiple scattering (`doRes`) |
| 11 | 13 300 | resolution: ionization (`doRes`) |
| 14 | 50 | scalar-potential B-field modes |
| 15 | 42 | global material-model groups |
| **total** | **126 452** | 81 732 alignment + 44 628 resolution + 92 global |

Alignment parameters cannot be switched off and dominate the per-candidate
payload (`nParms` ≈ 240–248, of which ~240 are alignment). The offline consumer
`global_corrections/fit_global_grads.py` freezes them at selection time
(`--parmtypes` defaults to `[14, 15]`).

> **Global indices shift with `doRes`.** These files (126 452 params) and the
> 2016 DATA productions `jpsi_calib2016_grads50_260715` /
> `…_globalmat_260717` (81 824 params, the same map *without* the four doRes
> families) index the same physical parameter differently. **Do not pool them
> in one global fit** without re-mapping through `runtree`.

What a reader of a 260906 file must handle relative to a 260905 one:

| branch | change | consequence |
|---|---|---|
| `ioniurbanv` | stride 11/13 → **12/14** (group column appended last) | a reader that hard-codes 11 or 13 **mis-parses**; read `ioniurbanstride`. Step records only |
| `radstepv` | `radstepstride` 11 → **12** | same; `radstepstride` was already exported |
| `reseigidx`, `resinfvarv`, `resinfv`, `reshitidx` (two-track) | **gain** parmtype-8/9 entries, interleaved with the material ones in propagation order | code that assumed "every entry is material" must filter — on `reshitcls >= 0`, or on the runtree parmtype of `reseigidx` |
| `reshitidx` (two-track) | was empty, now filled (−1 = material, else leg index 0/1) | additive |
| same four arrays (single-track, `exportMaterialNoise=True`) | **gain** ~53 parmtype-15 entries per track | filter on the runtree parmtype; `resinfcov` deliberately EXCLUDES them (they re-partition the parmtype-10/11 noise; `resinfcovgrp` is their sum) |
| `gradv`, `gradllv`, `gradchisqv`, `hesspackedv`/`hessfactorv` (single-track, `exportMaterialNoise=True`) | the parmtype-15 columns gain the width term | intended; with the switch off they are bit-identical to the 260905 build |

**`resinfcov` and `cfmass_vgf` do NOT change** (verified bit-identical with the
hit blocks on and off). The hit-block variances go into the new `resinfcovhit`:
`cfmass_vgf = (σ_m² − resinfcov)/σ_m²` has to keep meaning the TOTAL Gaussian
share (hits + beamspot + pointing), because that is what the
self-consistent-σ correction's `a_i = (1 + f_hit) σ_i/m_i` uses and what every
cache built before the hit blocks assumes.

### Reading a multi-stream production

Both v2 productions run `numberOfThreads=4`, so **a task is four files**,
`task_XXXX/globalcor_0..3.root`. An event never splits across streams and the
content is bit-identical to a single-thread run after sorting on
`(run, lumi, event)`, so the four files simply concatenate — but a reader that
names stream 0 takes **1/4 of the candidates and says nothing**.

Every reader in `calibration_studies/` lists its inputs through
`resolution/prodfiles.py` (shell twin `prodfiles.sh`); see the top-level
`README.md` for the API. Two rules that are easy to get wrong:

* the `runtree` (13.02 MB, 126 452 entries) is written into **every** stream
  file, byte-identical. Read exactly one copy (`runtree_file()` returns the
  first existing stream), never the concatenation;
* completeness is decided **per task**: `.complete` present, no empty stream
  file, and as many streams as the sentinel's `streams=N` line declares.
  Otherwise the task is skipped whole and counted.

---

## 5. Per-leg configuration

### 5.1 J/psi leg — `runCvhJpsiGenMC.py` on `TkAlJpsiMuMu` ALCARECO

| option | value | why |
|---|---|---|
| `trackSrc` | `ALCARECOTkAlJpsiMuMu` | the collection this ALCARECO carries |
| `useLegacyPairLoop` | `True` | **required**: TkAlJpsiMuMu has no persisted `…JpsiOnlyResonances` candidate collection (only TkAlJpsiX does), so the maker must use the all-pairs loop |
| `fitFromGenParms` | `False` | **required**: the default `True` freezes the 10-dim vertex/kinematic reference block and seeds it from gen, and the freeze is *not* conditioned on the gen match succeeding — leaving the default would be silently wrong, not a fallback |
| `doVtxConstraint` / `doMassConstraint` | maker default (ON) / `False` | the v2 productions below were made with the vertex constraint OFF -- the scripts passed `doVtxConstraint=False` explicitly.  That override is gone, so any RE-production picks up the maker default, which is now ON: `Jpsi_d` is then identically zero and `Jpsi_vtxres` carries the DCA the unconstrained fit would have reported.  A mass constraint would add a second `nicons` pass, so it stays off |
| `doTrigger` / `applyHltFilter` | `True` / `False` | store the per-path HLT decisions; the ALCARECO selection already ran |
| `doSimHits` | `False` | **required**: the input has no `SimTrack`s, and `doSimHits=True` with `fitSimHitPositions=False` disables the genParticle ΔR matching *before* the hard-coded `requireGen=True` cut, which would reject every candidate |
| `useIdealGeometry` | `False` | aligned MC geometry from the GT |
| `useDefaultField` | `True` | **the field the SIM propagated through** — see below |
| `globalTag` | `106X_mcRun2_asymptotic_v17` | UL16 MC conditions |
| `scalarPot3DInitFile` | `mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt` | always required: it defines the 50 parmtype-14 modes and their Jacobian columns even when another model supplies the field. Must be the **same dump** on both legs or the two global catalogues differ |
| `materialGroupsFile` | driver default `materialGroups50.txt` | 42 groups. `materialGroupsV2.txt` exists but no driver or production references it |
| `CgfQoPMode` | `0` | mode ≥ 1 substitutes the CGF estimator through the uncached branch at ~10× cost (`profiling/NOTES.md`) |
| `propagationPtotLimit` | `0.2` | G4e momentum floor (cfi default is 1.0) |
| `clampMomentumFloor` | derived → **0.25 GeV** | `< 0` ⇒ `1.25 × plimit`. The historical hard-coded 2.0 GeV against a 0.2 GeV limit pinned 12 % of soft candidates and carried the whole +0.21e-3 mass-scale offset |
| `maxMomentumStepFactor` | `2.0` | relative GN trust region: p may at most halve or double per iteration |
| `stepBacktracking` | `True` | Armijo χ² backtracking from iteration 2, ≤ 4 halvings, slack 1.0 |

### 5.2 Z leg — `runCvhDimuonMiniAOD.py` on DY MiniAODv2

Chain: `slimmedMuons` → `TrackProducerFromPatMuons` →
`diMuonTrackVertexCandidates` (60–120 GeV, opposite sign) →
`ResidualGlobalCorrectionMakerTwoTrackG4e` — the same producer chain the WMass
custom NanoAOD uses and the same *maker* as the J/psi leg; only the track source
and the mass window differ. Everything in 5.1 carries over except:

| option | value | why |
|---|---|---|
| `muonSrc` | `slimmedMuons` | the MiniAOD muon collection; `innerTrackOnly=False` |
| `massMin` / `massMax` | 60 / 120 | the Z window on the candidate producer |
| `doGen` | `True`, with `genParticles=prunedGenParticles`, `pileupInfo=slimmedAddPileupInfo` | both are dereferenced *unguarded* by the maker when `doGen=True`. `genParticles:xyz0` (the gen PV) does not exist in MiniAOD and *is* guarded — `genl3d` gets the −99 sentinel |
| `requireGen` | **`False`** | unlike the J/psi leg: an opposite-sign pair in a 60–120 GeV window on DY is the Z by construction, so the cut would only couple the yield to the MiniAOD gen-pruning thresholds. `Mu{plus,minus}gen_*` are filled either way, so the cut can be made offline — and it has to be, for the CF chain |
| `doTrigger` | `False` | the option and a 2016 single-muon path list are in the driver if the decisions are wanted |
| `propagationDirection` | `anyDirection` | recovers backward legs (11 in a 1000-event smoke), 0 propagation failures |
| `tightG4eStepper` | `True` | free, and defensible — but see the note below |
| no `BeamSpotProducer` | | MiniAOD already carries `offlineBeamSpot`, which is the *hard-coded* tag the maker consumes (`ResidualGlobalCorrectionMakerBase.cc:319`). Adding a second producer with that label would shadow it |
| `produceValueMaps` | `False` | this driver has no output module, so the ValueMaps would be produced and dropped |

**`tightG4eStepper` — the knob is inert in the refit, and mis-wired in one
driver.** `runCvhJpsiGenMC.py` and `profiling/runCvhProfile.py` implement it as a
patch of `process.geopro.MagneticField.ConfGlobalMFM.OCMS.StepperParam`, but in
this release `geopro` is neither scheduled nor consumed: `CvhMaster`/`CvhWorker`
build the G4 world and the field manager from
`cvhMasterESProducer.MagneticField`. So the patch cannot reach the propagation
in those drivers. It does not matter, because with the knob *correctly* wired
(as in `runCvhDimuonMiniAOD.py`) two 300-event smokes are **bit-identical**, all
217 branches: the CVH propagation caps the G4e step at 10 mm
(`/geant4e/limits/stepLength 10.0 mm`), and over 10 mm a 45 GeV track in 3.8 T
has a sagitta of ~3e-4 mm, so RK integration error never approaches even the
tight 1e-5 mm tolerance. The tolerance binds in the *simulation*, where steps
are metres long — it does not bind in the refit. **If the 10 mm step cap is ever
raised this stops being true** and both drivers have to be fixed.

### 5.3 Field model — why `useDefaultField`, not `useOpera3D`

These are closure/feasibility samples, so **the fit's field must be the
simulation's field**: the UL16 SIM propagated through the default
OAE-parametrised tracker field (`useParametrizedTrackerField=True` on the 160812
volume-based map). Refitting the same events with the full 3D TOSCA grid does
not cancel — it leaves an η/φ-coherent **~8e-4 dp/p** pattern in the pull width
(0.155 % of unit variance, and φ is where OAE is structurally blind), which on a
closure test is indistinguishable from the thing being measured.

The Opera3D variant is reserved as a later **injected-field** test on a subset:
refitting a slice with the 3D grid and comparing to the default-field result
measures that η/φ pattern directly. The first J/psi submission went out with
`useOpera3D=True` and was cancelled with 0 tasks complete; its directory
survives as `cvh/jpsimc_20M_260905_opera3d` for the `PROVENANCE.txt`.

---

## 6. Cost, volume and threading

### Per candidate

| | J/psi ALCARECO | Z MiniAOD |
|---|---:|---:|
| CPU | 0.65–0.9 s/candidate (~40 s job startup) | 0.704 s/candidate, 0.323 s/event |
| candidates/event | 0.9967 | 0.447 |
| stored | ~66 kB/candidate + 13.09 MB runtree per stream file (realised: 1.5 TB / 21.68 M = 69 kB) | ~68 kB/candidate + 13.07 MB per stream file (realised: 267 GB / 3.80 M = 70 kB) |
| where the bytes go | — | on the pre-v2 export set (34.7 kB/cand): `hessfactorv` **71.5 %**, the two `*_jacRef` 15.4 %, `Jpsi_jacMass` 2.6 %, `gradv` 2.5 %, `globalidxv` 1.1 %, all six `cfmass_*` **1.4 %**. The v2 per-group exponents add ~26 kB on top of that |

The CF export is a **CPU** cost, not a storage cost. `fillGrads=False` does not
save the packing CPU — the packed Hessian is computed unconditionally and the
flag only controls whether it is written. `doRes` and `fillGradsFactored` are
nearly free *alone* (0.062 and 0.064 s/cand) and cost 0.63 s **together**,
because the in-maker CF export is gated on
`doRes_ && (fillGrads_ || fillGradsFactored_)`.

The per-group exponent export adds **+0.28 s/candidate** (+30–45 %), measured in
a controlled A/B on user CPU. That is irreducible without changing the model: a
"block" is pooled by parmtype-10/11 global index over a whole
surface-to-surface propagation and typically STRADDLES several material groups,
so the primitives genuinely run once per (block, group); single-group blocks
already run once and feed both destinations bitwise. Offline per-group
extraction costs 1.9 s/candidate against ~0.5 s for the flat one, so doing it
in the maker is still a 7× saving on top of not writing 430 kB/candidate of
step records. `exportMaterialNoise` and `exportVarianceGrads` are **not
resolvable in CPU** (bounded well below 2 %, within-rep paired differences
median −1.3 s / −1.1 s on 60 gun events).

### Threading

The CVH refit in CMSSW_15 is multithreading-capable — the maker is an
`edm::stream::EDProducer<>` and the Geant4 master (`CvhMasterThread`, an
EventSetup product on `CvhMasterRecord`) builds `DDDWorld` and the master field
ONCE before any TBB worker starts. `numberOfStreams` follows `numberOfThreads`
in both drivers. Scanned on a quiet 64-core node with the production switch set
(`threadscan/run_scan.sh`):

| threads | DY wall (s) | DY CPU (s) | speedup | J/psi wall (s) | J/psi CPU (s) | speedup | DY RSS (GB) | J/psi RSS (GB) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 770.7 | 761.3 | 1.00 | 2039.9 | 2028.0 | 1.00 | 1.76 | 1.36 |
| 2 | 404.2 | 765.5 | 1.91 | 1024.7 | 2003.4 | 1.99 | 1.85 | 1.45 |
| 4 | 224.9 | 770.5 | 3.43 | 535.5 | 2020.6 | 3.81 | 2.00 | 1.61 |
| 8 | 131.7 | 772.8 | 5.85 | 286.4 | 2024.2 | 7.12 | 2.12 | 1.89 |

(2000 events each: 897 Z candidates, 1991 J/psi candidates.)

Total CPU rises 1.5 % on DY and falls 0.2 % on J/psi over the whole range: there
is **no parallel overhead**, only a fixed serial head (geometry + field + G4
physics list) — `wall(N) = 40.4 + 730.3/N` (DY) and `36.0 + 2003.9/N` (J/psi)
fit all four points to ~1 %. That head is paid once per JOB, so a 2000-event
scan UNDERSTATES production scaling. On real chunks:

| | 22 120-event DY chunk | 12 185-event J/psi chunk |
|---|---|---|
| 1 thread | 2.25 h | 3.40 h |
| **4 threads** | **0.57 h (3.94×)** | **0.86 h (3.97×)** |
| 8 threads | 0.29 h (7.73×) | 0.43 h (7.84×) |

**The output is bit-identical.** Sorted candidate by candidate on
`(run, lumi, event)` with a stable key (`threadscan/compare_threads.py`), all
213 branches of all 896 DY candidates and all 228 branches of all 1991 J/psi
candidates agree exactly at 2, 4 and 8 threads, and the `runtree` is identical
in every stream file. Note the drivers give each stream its OWN CLHEP engine
wired into Geant4's thread-local RNG, so this is an invariant of a fit that does
not sample, **not a structural guarantee** — re-run `compare_threads.py` after
any change that could make the fit consume randomness.

**Memory is set by the task tail, not the core count.** Peak RSS splits into
~1.71 GB shared (DY) / ~1.28 GB (J/psi) plus ~51 MB (DY) / ~76 MB (J/psi) per
extra stream. The 1-thread MaxRSS over the 260905 productions was median 1.9 /
p99 3.1 / **max 3.56 GB** (J/psi, 1616 tasks) and median 2.6 / max 3.56 GB (DY,
164 tasks) — a tail a 2000-event scan never sees. Sizing on
`max + (N−1) × per-stream`, ×1.3:

| threads | recommended `--mem` / `request_memory` | per core |
|---:|---:|---:|
| 1 | 4.5 GB | 4.5 GB |
| 2 | 5 GB | 2.5 GB |
| **4** | **5 GB** | **1.25 GB** |
| 8 | 5.5 GB | 0.7 GB |

(Two tasks reporting MaxRSS within 0.1 MiB of each other and of the limit is the
signature of a cgroup CEILING, not a peak; the numbers above are
`/usr/bin/time -v` VmHWM on an unlimited node.)

The one real cost of threading is that the 13 MB `runtree` is written into every
stream file: **+5.7 % output at 4 threads, +13.4 % at 8**. (It would be removed
by writing it from stream 0 only — not done, because every existing reader opens
the file it happens to have.)

---

## 7. Re-running

### 7.1 HTCondor (the grid) — what the v2 productions used

The submit condor pool is a **submission portal, not a cluster**:
`COLLECTOR_HOST = submit06.mit.edu:9615`, `FLOCK_TO = t3serv009.mit.edu:11000`,
and the local pool has **exactly one execute slot** (`slot1@submit06`, 1 CPU,
gated on `Submit_LocalTest`). Everything else is glideins flocked to the CMS
global pool. **No condor slot reachable from submit mounts `/ceph`, `/work` or
`/home`** — verified on `mit_tier3` (t3btch001) and on global-pool slots at
DESY, IIHE and Caltech. `mit_tier3` is excluded twice over: native el7 with no
singularity, against an `el9_amd64_gcc12` release.

So a condor leg carries all three things a slurm task takes for granted:

1. **the CMSSW area** — cvmfs base release + a 64 MB overlay
   (`lib biglib cfipython python src`). The area is **not relocatable whole**:
   `.SCRAM/RuntimeCache.json` and `.SCRAM/*/MakeData/*.mk` bake in the absolute
   build path, and an untar-in-place run reports the ORIGINAL `CMSSW_BASE`
   while cvmfs supplies a complete release underneath — i.e. it silently runs
   the wrong libraries. The worker runs `scram project` (7 s) for a clean
   skeleton, untars the overlay on top, and **asserts
   `CMSSW_BASE == $_CONDOR_SCRATCH_DIR/$RELEASE`**;
2. **the input** — DY: `/ceph/submit/data/group/cms/store/...` is a mirror of
   the CMS `/store` namespace, so the chunk path maps to an LFN and streams
   through `root://cms-xrd-global.cern.ch/`. J/psi: **NEVER** (§8.1) — the
   repacked files are addressed by door, `/ceph/submit/X →
   root://submit5N.mit.edu//X`, with a size guard;
3. **the output** — `xrdcp` to `root://submit50.mit.edu/`, whose namespace root
   **is** `/ceph/submit`, with the transferred size read back (`xrdcp` has
   truncated outputs here before while returning 0). The `.complete` sentinel is
   copied LAST.

Routing — four load-bearing lines:

```
use_x509userproxy = true       # x509userproxy=/home/submit/david_w/x509up_u125124, NOT /tmp
+AccountingGroup  = "analysis.david_w"                  # required by submit06's submit policy
+ProjectName      = "CpuTestingProject"
+SingularityImage = ".../cmssw/cms:rhel9-x86_64"        # CMSSW_15_0 is el9
+DESIRED_Sites    = "<explicit list>"                   # WITHOUT this the job never leaves Idle
```

(The `mit_tier3` route needs the opposite — `+ProjectName = "MIT_submit"` and no
`+AccountingGroup` / `+SingularityImage`; irrelevant here since those pilots
cannot run el9.) The proxy must live where the schedd's shadow can read it, not
in `/tmp`; `submit_*_v2.sh` refreshes it from `$X509_USER_PROXY` and refuses to
submit under 2 h. Renew with `voms-proxy-init -voms cms -valid 192:00`.

```bash
cd calibration_studies/production/condor_jpsimc_v2     # or condor_dymc_v2
./submit_jpsimc_v2.sh --dry-run     # render without submitting
./submit_jpsimc_v2.sh               # all 1645 (--only "0 1 2", --idxfile, --threads, --mem, --maxidle)
./status_jpsimc_v2.sh [--slow]      # sentinels + queue + rate (--slow also checks streams/task)
./resume_jpsimc_v2.sh --dry-run     # which indices have no sentinel and no queue entry
./resume_jpsimc_v2.sh
./recover_xrdfix.sh --only "<idx…>" # re-run named tasks from the SECOND pinned payload
```

Resources as submitted: **4 threads, `request_memory = 5000` MB,
`request_disk = 6 000 000` KB (J/psi) / 4 000 000 KB (DY)**, `max_retries = 3`.
Placement is easy at that size: the DY leg went from 0 to **379 of 380 running
in 31 min** across 12 sites against 137 000 idle jobs already in the pool; the
J/psi leg (4.3× more jobs) reached 558 running at 35 min. For comparison the
slurm DY leg at that same moment had 1 running and 198 pending at 11.4 tasks/h,
starved by the J/psi arrays' fair share — the condor route is ~380 concurrent
tasks of *additional* capacity, a factor 15.6 on rate.

`resume` never re-queues an index already in the queue (two jobs writing the
same task dir would interleave their outputs) and judges completion on the
`.complete` sentinel alone. The payload is built once and **pinned inside the
production's own output tree**, so a resume months later runs the same binary;
`--rebuild-payload` is the deliberate override, and `recover_xrdfix.sh` exists
because overwriting the pinned tarball while ~1000 jobs can still re-fetch it
would mix two binaries into one sample.

### 7.2 slurm

`array_jpsimc_v2.sbatch` is the slurm form of the v2 configuration and the
recommended slurm setup going forward: **`--cpus-per-task=4 --mem=5G
--time=4:00:00`** from dev2 (3.9× the throughput per task for 0.83× the memory
of the old `--cpus-per-task=1 --mem=6G`). The v1 machinery is still in place:

```bash
cd calibration_studies/production
./submit_jpsimc20M.sh [--dry-run] [--max-running N] [--only "0 1"]
./submit_dymc8p5M.sh  [--dry-run] [--max-running N] [--time T]
./status.sh [--slow]     ./status_dy.sh [--slow]       # complete/running/failed, projected finish
./resume.sh --list-only  ./resume_dy.sh --list-only    # which chunks are missing
./resume.sh              ./resume_dy_dev2.sh           # resubmit only those (skips anything queued)
python smoke_inspect.py <out.root> [--compare <other.root>]   # schema, per-branch volume, poolability
```

`resume*` keys off the `.complete` sentinel and a live `squeue` check, never the
`.root` (cmsRun creates its output at *start*, so a killed task leaves a
non-empty but truncated file) and never `sacct` (crashed, cancelled and
never-started are the same case). The task index is the chunk-list line number,
so a resumed task reads exactly the same events as the one it replaces —
**never renumber a chunk list**; append instead (that is how tasks 1642–1644
were added).

Slurm limits here: `MaxArraySize = 1001` (hence two arrays for the 1642 J/psi
tasks), `MaxJobCount = 50000`, QOS `normal` `MaxJobsPU = 5000`. The binding
constraint is fair share, not a limit: two productions in the queue interleave
rather than adding throughput.

### 7.3 Making a new production

```bash
# 1. scan every candidate input (ROOT open + Events entry count; -1 = unopenable)
python scan_events.py filelist.txt scan.txt
# 2. tile the good files into tasks of ~target events (skipEvents/nEvents ranges)
python mkchunks.py scan.txt --nfiles 410 --nchunk 4 --target-events 13500 \
    --filelist filelist_<tag>.txt --chunks chunks_<tag>.txt --counts filelist_<tag>.counts.txt
```

Sizing by a **target event count** rather than a fixed N-way split matters: the
J/psi files run 959…62 k events and the DY ones 2 349…88 479, so a fixed split
makes 240-event tasks that still pay the ~3 min startup. Targets used: 13 500
(J/psi → 1642 tasks, mean 13 227) and 25 000 (DY → 380 tasks, mean 22 375, median
22 120). Validate that the chunk list tiles every file exactly — no gaps, no
overlaps, Σ chunk events == Σ file events, N distinct basenames.

`skipEvents` asserts a single input file per job, because PoolSource counts the
skip across the whole concatenated `fileNames` list. Check it against a known
event sequence rather than assuming (`skipEvents=100` must start on the 101st
record of the same file read from 0).

---

## 8. Known defects and their fixes

### 8.1 Input integrity (J/psi MC ALCARECO)

* **Split level: the ALCARECO must be ROOT split-1.** The centrally produced,
  10_6-written files are split-99 and mis-deserialise their `SiStripCluster`s in
  15_X (ROOT #19773). Symptoms of reading one: 0.824 candidates/event (vs
  0.9968), 17.4 % `fail[prop]`, and **the 83 % that survive are also garbage** —
  median chi2/ndof 3.5e6, 99.81 % at the iteration limit, 2.9 % in the J/psi mass
  window, effective per-hit pull 1860 σ against 0.98, gen match lost in 88.6 %.
  Every *pre-refit* quantity (valid hits, input `Jpsitrk_mass`, `attempted`)
  agrees with a good control, so nothing upstream flags it and **there is no
  salvage cut.**
* **NEVER read the repacked J/psi via `cms-xrd-global`.** The repacked split-1
  copies live under the CMS `/store` namespace on the submit group store, and
  **that same LFN exists centrally with different content** (the split-99
  original; 1 840 069 145 B vs our 1 783 846 343 B on the first file). The MIT
  T2 door does not serve this tree at all. The only doors that serve our bytes
  are **submit50–55**, whose xrootd namespace root *is* `/ceph/submit`, so the
  mapping is a pure prefix strip with no redirector. The chunk list carries each
  file's **size as a fourth field** and the wrapper refuses any door serving a
  different size (exit 8, `WRONG REPLICA`). This trap has been walked into once —
  by using a redirector to *repair* an input rather than to read one.
* **Truncated (zombie) repacks.** 43 of 620 scanned files (**6.9 %**) are ROOT
  zombies ("no keys recovered"); listed in `truncated_inputs_260905.txt`. File
  size does not predict it (zombies up to 2.76 GB; valid files down to 959
  events / 38 MB) — only the ROOT open is decisive. `cmsRun` runs
  `skipBadFiles=True`, so a truncated input is *silently skipped* and the task
  writes a valid but empty output. **Re-scan with `scan_events.py` before using
  any other slice of this sample.**
* **A file can open, count entries, and still be undeserialisable.**
  `FDB8C946-…` has **no StreamerInfo**: ROOT reported 19 797 entries (so the
  zombie scan passed it) but `PoolSource` dropped it. Worse, the local repack was
  724 MB against a 1 942 MB original and held 19 797 events against 51 478 — and
  `mkchunks.py` sizes chunks from the *local* `Events->GetEntries()`, so it
  tiled 38 % of the file and **31 681 events were never chunked at all**.
* **Write-side memory corruption in our own July-2026 repack.** Four inputs
  threw `FormatIncompatibility … EntryError can not convert representation of
  <name>: <NULs> to value of type vector<string_hex>` while *constructing* the
  input source (exit 91, before any per-file skip logic, so `skipBadFiles` does
  not help). Characterised: NUL runs inside the **uncompressed** ParameterSets
  payload, each replacing a value's content with its length preserved, confined
  to the 32–46 byte band (`std::string` lengths just above libstdc++'s SSO
  threshold) — the signature of a freed-and-reused small allocation. Not disk
  rot, not transfer damage: `edmProvDump` is clean on the central original and
  fails on our copy.
* **The two cheap scans to run on any future repack before a production reads
  it** (`repack_fix_260907/`): `scan_pset_nulls.py` (minutes over 410 inputs:
  403 OK, 6 with NUL runs, 1 unopenable) and `scan_fitfail.sh` over the task
  logs (baseline mean 0.0067 %, max 0.0414 %; the 12 split-99 tasks sat at
  16.7–18.0 %). Together they catch all three classes — NUL provenance, missing
  StreamerInfo, and silent event-level damage.
* **Wall clock is a free corruption detector.** The 12 split-99 slurm tasks all
  ran 10:14–12:00 h against a normal J/psi chunk's ~2.8 h — a factor 3.8, which
  is exactly the 4.1× propagator-call excess a diverged fit generates (5.75 M
  calls vs 1.39 M). A per-task runtime outlier check costs nothing and would
  have caught this on day one. The `.complete` sentinel already carries
  `seconds=`.
* Repaired inputs (7 files, all split-1, entry counts matching the *central*
  originals, `edmProvDump` clean, NUL-scan clean):
  `/ceph/submit/data/user/d/david_w/ZMass/restaged/jpsimc_20M_260906_repack/`.
  The bad staged copies are kept as `restaged_split99_DO_NOT_USE/`.
* The DY MiniAOD needed no zombie scan: those files came through
  `transfer/transfer_one.sh`, which publishes at the final name only after an
  independent size + adler32 check against DAS, so *present at its final name*
  means *verified*.

### 8.2 Code defects, fixed

| defect | symptom | fix |
|---|---|---|
| **`ndof == 0` abort** (two-track maker) | exit 134 (SIGABRT), one Eigen bounds assert; **the whole task is lost** — a crashed cmsRun output has no keys at all. Cost **37 of the first 133 finished DY tasks (28 %)**, rate 0.033 per 1000 Z candidates | `fab515e`: `ndof` computed **signed** and clamped (below ten the unsigned subtraction UNDERFLOWED to ~4e9, which `min` read as full rank), `ndof <= 0` counted as `fail[ndof]` and dropped through the existing `valid` path, rank report guarded. Bit-identical on all three smokes |
| **XrdAdaptor null dereference** | SIGSEGV (rc 139) on the grid only, ~1.3 % of tasks; stack *appears* to end in `G4MuPairProductionModel` | `c2d74e3e647`: `getQueryTransport` left its `std::string*` uninitialised and `AnyObject::Get` sets it to 0 when the transport query fails (a redirect hop with no live connection); `tracerouteRedirections` then formatted `*hostname_method` unconditionally. **Worth pushing upstream to CMSSW** |
| **Geant4e shared-table race** | a job-startup race whose failure mode is a silently wrong dE/dx table for a whole job | `ca6058d96fc`: the reference and ionization-only `G4TablesForExtrapolatorForCVH` builds were guarded by two DIFFERENT mutexes although both `Initialise()` the same Geant4 EM models (which write process-wide non-const statics) — now one shared mutex + atomic pointers |
| **bash `GROUPS` builtin** in `resume.sh` | `--array=100999%200`, `--array=169571%200` — the user's **gids**. `declare -A GROUPS` fails ("cannot convert indexed to associative array") and the writes land in the builtin | renamed to `CHUNKGRP` in `resume.sh`, `resume_dy.sh`, `resume_dy_dev2.sh` |
| **`attempted=0` guard had a hole** | a skipped input means the maker never runs, so **no `fit summary` line is written at all** and `grep -q "attempted=0"` matched nothing — `task_1313` staged 4 empty stream files and a `.complete` for a 19 797-event chunk | both wrappers now also fail on the PoolSource skip message (exit 6) and on the *absence* of a summary (exit 5) |
| **`recover_xrdfix.sh` could never submit** | wrote `.recover_idx.txt` then died under `set -u` before `condor_submit` | it appended `REDIR`, which only the DY config defines, and never passed `INDOORS`. Both fixed. (Note: a `#` comment inside a backslash-continued argument list silently truncates the command) |
| **readers naming `globalcor_0.root`** | silently 1/N of the statistics under 4 threads; cleanups that deleted stream 0 only left streams 1–3 of a truncated task for a widened glob to swallow | everything goes through `resolution/prodfiles.py` / `prodfiles.sh`; `--ntasks` caps TASKS; `pf_clean_incomplete` removes **every** stream of an incomplete task. There is no `globalcor_0.root` left in any `.py` under `calibration_studies/` |

### 8.3 Environment and operations

* **submit82's ceph client is evicted** (`libceph: auth protocol 'cephx' …
  failed: -13`); every path under `/ceph/submit` returns EPERM there, including
  files the process just wrote. It is node-local — use submit50/51/52/81. The
  array task asserts input readability on its own node for the same reason.
  **submit60** has ceph but **no AVX2**, so CMSSW 15_0 dies there with an
  illegal instruction.
* **Site / node fences** (in both `config_*_v2.sh` and applied live with
  `condor_qedit`): SIGILL (rc 132) inside the CVMFS release's own `libXrdCl` on
  the `stagein` Prepare — 16 of 16 at RWTH Aachen and JINR, 8 each; and SIGILL
  in `edm::StreamSchedule::fillWorkers` — 34 of 36 on five `ultralight.org`
  machines (`compute-6-34` alone ate 16) plus `t2bat0310`, i.e. **black-hole
  nodes**, fenced by node not by site. Use `regexp(..., "i")`, never `=!=` on the
  machine name: `=!=` is the ClassAd IDENTITY operator and is case sensitive,
  and MIT T2 advertises `Machine` in uppercase, so a lowercase `=!=` fence
  silently never excludes anything.
* **A 5–6 GB `MemoryUsage` logged just before a crash is the crash handler
  forking gdb**, not a leak: exactly the procs that returned 139 exceeded
  5120 MB, no other proc exceeded 2.73 GB, and instrumented re-runs peak at
  2.39–2.55 GB. `request_memory = 5000` stands.
* **Reading a CMSSW gdb dump:** CMSSW prints thread stacks in DESCENDING order
  and pauses every other thread first (`InitRootHandlers.cc`), so every thread
  carries a `<signal handler called>` frame and a `tail -60` keeps the LAST
  threads, never the faulting one. Dump 400 lines and grep for
  **`sig_dostack_then_abort`** (the paused ones are `sig_pause_for_stacktrace`).
* `CUDAService … cudaErrorMemoryAllocation: out of memory` at service
  construction is a full GPU on the node and is purely environmental — the job
  needs no GPU, `CUDAService` is only constructed because the standard service
  set includes it. `resume` re-drives it elsewhere.
* The `file:<path>,<skip>,<max>` filelist form in `resolution/simprod/`
  **has no parser**: fed raw to `input=` it splits on the commas as path
  separators and, with `skipBadFiles=True`, silently processes the *entire*
  file. The chunk lists (`path skip nev`, whitespace separated) are what the
  production uses.
* `globalfit/extract.py` **without** `--no-mass` dies on both productions with
  `ValueError: … partial radiative export ['radvgrid']`: `extract.py` requires
  the radiative step branches to be all-four or none, and every production ships
  `radvgrid` alone (`exportStepRecords=False`). Not a threading matter — it is
  the offline mass builder not having caught up with the `radvgrid` export. The
  quadratic path (`--no-mass`) is unaffected.
* `cf_inmaker.py pairs` cuts on the **gen** mass and its default window is the
  J/psi one (3.0969 ± 0.35), which selects **zero** Z candidates. Pass
  `--mass-window 91.1876 60` for a Z production.

---

## 9. Per-production status detail

### `jpsimc_20M_260906_v2` — complete, 1645/1645

Four stream files and a `.complete` in every task, 21 750 740 events, 21 678 062
succeeded candidates. All **410 distinct inputs are clean**: 403 untouched
group-store repacks plus 7 repaired files in `restaged/jpsimc_20M_260906_repack/`.

The task count is 1645 rather than 1642 because the truncated `FDB8C946-…` input
(§8.1) cost 51 478 events: line 1314 keeps its `0 19797` range repointed at the
repack, and the missing tail was **appended as tasks 1642–1644** so that no
existing index moved. Those four tasks produced 51 290 candidates where there
were zero, at chi2/ndof median 0.9518 against controls' 0.9538.

The 16 chunks of the four provenance-damaged inputs were re-run as cluster
`3803425` and all succeeded on the first start: 204 957 candidates from 205 635
events = 0.9967/event at 0.0073 % failures, chi2/ndof median 0.9531 vs the
controls' 0.9540. `attempted` is identical task by task to the split-99 runs —
the same muon pairs were found, only the fit outcome changed, which is the
cleanest possible demonstration that this was a hit-content problem and not a
selection difference.

Two further inputs carry inert NUL runs in psets cmsRun never parses. Rather
than assume they were harmless, both were repacked and their 7 chunks re-run
into a side tree with the *original* payload so the input was the only
difference: **bit-identical over 102 507 candidates**, all 228 tree branches and
36 `runtree` branches. The existing outputs stay; the side tree
`cvh/jpsimc_20M_260906_v2_nulcheck/` (6.9 GB) is evidence and is deletable.

### `dymc_8p5M_260906_v2` — complete, 380/380

Four stream files and a `.complete` in every task, 8 502 597 events, 3 799 624
succeeded candidates. The 6 104 failures (0.16 %) split as `fail[prop]` 3 594,
`fail[kinfit]` 2 301 (the seed `KinematicParticleVertexFitter`, not the CVH
fit), `fail[ndof]` 208 (now counted instead of aborting the task), `fail[nan]`
1, everything else zero. Validated
before the bulk submission: 400 events of one chunk run at INFN Pisa (1 thread)
and Caltech (4 threads) are **bit-identical, all 213 branches, to a local dev2
run on submit82** — across the batch system, two sites, two CPU vendors,
xrootd-streamed versus POSIX input, and 1 versus 4 streams.

More of this dataset is on ceph (1731 files / 4.386 TB; this production is the
8.5 M-event head only). An extension re-runs `transfer/make_filelists.py`, then
`mkchunks.py` on the new scan, **into a new tag** — resuming into this one would
renumber the tasks.

### `jpsimc_20M_260905` (slurm v1) — 1633/1642, 12 tasks unusable

Kept as a cross-check sample only. Sixteen task indices are affected, and the
output tree carries the list as `BAD_TASKS_split99_260907.txt`:

* **tasks 1219–1222, 1334–1337, 1409–1412** — the 12 chunks of the three
  split-99 staged inputs. Seven finished (with garbage output) and five hit the
  12 h wall limit. **Discard all twelve**; the finished ones look valid on
  inspection and are not;
* **tasks 1552–1555** — the fourth damaged input, exit 91, never produced
  output.

That is why the array is 1633/1642: the nine tasks without a sentinel are those
five timeouts plus these four. If v1 is ever used quantitatively, either exclude
the twelve or re-run them against `restaged/jpsimc_20M_260906_repack/`. No
re-run is planned.

### `dymc_8p5M_260905` (slurm v1) — 182 of 380, cancelled

Superseded by the condor v2 leg (same 380 chunks, physics bit-identical, plus
the 2026-09-06 exports and the `ndof == 0` fix). The 182 completed tasks stay on
disk as a cross-check sample — single stream, old exports; note that the tasks
recovered onto dev2 before the cancellation carry three extra always-on branches
(`Jpsi_covrefmom`, `Jpsigenpre_*`, `Mu*_maxfracloss`) and ran with
`exportHitResBlocks=False`, so the set is not homogeneous.
