# The next production: what the 2026-09-06 exports change

Companion to `STATE.md` (`jpsimc_20M_260905`) and `STATE_dy.md`
(`dymc_8p5M_260905`). Those two ran from
`/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev` @ `157775e`. The new
exports are on the branch **`cvh-exports-260906`**, built in a SECOND area,
`/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2` (a git worktree of the
same repo, so the production area was never touched while its jobs ran).

Nothing here needs to be re-derived: the physics is bit-identical to
`157775e` on all three smoke samples, so this is a configuration and a volume
decision, not a revalidation.

---

## 1. What is new, and what it is for

| export | branches | why the next production wants it |
|---|---|---|
| **per-material-group CF exponents** | `cf{qop,mass}_grp`, `_grp_ms`, `_grp_ioni_re/_im`, `_grp_rad_re/_im`, `_grp_del` (single-track), `_grp_closure` | lets the mass term float the **parmtype-15 material amounts** the hit chi2 already measures, instead of four unphysical `k_hit/k_ms/k_ioni/k_rad` knobs. It is the input the joint fit of NOTES 2026-09-05 (II) §6 needs at full statistics |
| **hit-class blocks in the two-track maker** | `reshitcls`, `resinfcovhit`, `cf{qop,mass}_hitcls`, `_hitv`, plus the parmtype-8/9 entries in `reseigidx`/`resinfvarv`/`resinfv`/`reshitidx` | the per-hit-class resolution parameters have **never been fitted** — the two-track maker registered no hit blocks at all, so the mass functional had nothing to weigh them with |
| **the two legs' reference covariance** | `Jpsi_covrefmom` (21 floats, upper triangle of the symmetric 6x6), `Jpsi_jacrefmom`, `Jpsi_qoprefplus/minus`, `Jpsi_sigmarelplus/minus`, `Jpsi_rhomom`, `Jpsi_fang` | makes the Jensen and self-consistent-sigma corrections **truth-free on DATA**. Today `rho` and `f_ang` come from an MC measurement and carry +-5 %; each correction alone is 15-43 MeV on `m_Z` |
| **pre-FSR gen mass** | `Jpsigenpre_mass`, `Jpsigenpre_status`, `Jpsigenpre_masslep`, `Jpsigen_massdressed` | the Z channel's FSR kernel comes off the production's own pairs cache instead of a separate FWLite pass whose selection has to be kept in step by hand |
| **material-group process noise** (`exportMaterialNoise`) | `resinfcovgrp` + parmtype-15 entries in `reseigidx`/`resinfvarv` | the quadratic hit-chi2 term now differentiates a group's WIDTH as well as its mean loss — the two functionals of `k_g` were measuring different halves of it. **Single-track maker only**; the two-track one has no log-det machinery to extend |
| **per-leg reference energy loss** | `Mu{plus,minus}_maxfracloss`, `dEref`/`maxfracloss` (single-track) | a `dE_ref/p < 0.01` quality requirement on the quadratic term's material information, without the step records |
| **the material group on every step record** | `ioniurbanv` / `radstepv` gain a last column; new `ioniurbanstride` | deletes the offline `pair_ioni_rows` heuristic. Only visible with `exportStepRecords=True` |

## 2. The configuration

Take `config_jpsimc20M.sh` verbatim and add **one option**:

```diff
 EXTRA="numberOfThreads=1 \
  doRes=True exportCfExponents=True exportStepRecords=False \
+ exportCfGroupExponents=True \
  fillJac=True fillGrads=False fillGradsFactored=True \
  ...
```

Everything else is unchanged, and every other new export is **on by default**:

| option | value | note |
|---|---|---|
| `exportCfGroupExponents` | **`True`** | default False. This is the +26 kB/candidate decision, see §3 |
| `exportHitResBlocks` | `True` (default) | the parmtype-8/9 blocks. `False` reproduces a pre-2026-09-06 two-track tree exactly |
| `exportMaterialNoise` | **`True` on the single-track legs**, irrelevant on the two-track ones | the parmtype-15 width term. +17 ms/track and +3 kB/track; it CHANGES the exported G and H of the parmtype-15 columns, which is the point |
| `exportStepRecords` | `False` (**now the default**) | it was `True`; every production since 2026-09-05 set it False explicitly anyway |
| `Jpsi_covrefmom` etc. | always on | 148 B/candidate, no switch — without them the corrections have no truth-free input |
| `Jpsigenpre_*` | always on, needs `doGen=True` | 8.5 B/candidate |
| `genResonancePdgIds` | default `{23, 443, 100443, 553, 100553, 200553}` | only needs setting for an unusual resonance |

The DY leg (`config_dymc8p5M.sh`) takes the same one-line addition. It is the
leg that most wants `Jpsigenpre_mass`, because that is where the FSR kernel is
built.

---

## 3. Volume

Measured on the smokes, **compressed bytes per tree entry**, with the raw step
records dropped (`exportStepRecords=False`) — i.e. the production format:

| sample | baseline | new defaults | + per-group exponents | + material noise |
|---|---:|---:|---:|---:|
| J/psi gun, two-track (59 cand) | 110.26 | 110.79 | **136.40** | — |
| 2016F data ALCARECO, two-track (48 cand) | 31.18 | 31.71 | **55.78** | — |
| low-pT mu gun, single-track (120 trk) | 61.51 | 61.60 | **87.98** | 64.67 |

as deltas:

| | J/psi gun (2-trk) | data (2-trk) | mu gun (1-trk) |
|---|---:|---:|---:|
| new defaults (hit blocks + covariance + gen + dEref) | +0.53 kB (+0.5 %) | +0.53 kB (+1.7 %) | +0.09 kB (+0.1 %) |
| **per-group exponents** | **+25.6 kB (+23 %)** | **+24.1 kB (+76 %)** | **+26.4 kB (+43 %)** |
| material noise (`exportMaterialNoise`) | — | — | +3.07 kB (+5.0 %) |

Per new branch group (bytes/candidate, the `grp` files):

| | J/psi gun | mu gun | data |
|---|---:|---:|---:|
| `cf*_grp*` | 26 221 | 27 013 | 24 650 |
| `Jpsi_covrefmom` + jac + derived | 148 | — | 150 |
| `cf*_hitcls` + `_hitv` | 67 | 51 | 69 |
| `reshitcls` | 47 | 27 | 46 |
| `maxfracloss` + `dEref` | 25 | 9 | 27 |
| `resinfcovhit` + `resinfcovgrp` | 5 | 5 | 6 |
| `Jpsigenpre_*` + `_massdressed` | 8.5 | — | 9.0 |

so everything except the per-group exponents is **0.3 kB/candidate together**.

The `exportMaterialNoise` +3.07 kB is NOT new numbers: it is ~53 more entries
per track in the existing `reseigidx`/`resinfvarv`/`reshitidx`/`reshitcls`
arrays (~0.7 kB) plus `hesspackedv` compressing worse once the parmtype-15
columns are no longer mostly zero.

**At production scale.** `jpsimc_20M_260905` measured **80.7 kB/candidate** and
~21.7 M candidates = **1.75 TB**. The smokes are heavier per candidate than the
production (they carry `fillGrads=True`'s dense Hessian, and a gun candidate has
more blocks), so scale the DELTA, not the ratio:

| | candidates | today | + per-group | note |
|---|---:|---:|---:|---|
| `jpsimc_20M` | 21.7 M | 1.75 TB | **2.32 TB** | +0.57 TB |
| `dymc_8p5M` | 3.9 M | 0.14 TB | **0.24 TB** | +0.10 TB |
| the full 38 M dimuon calibration | 38 M | 3.07 TB | **4.06 TB** | +0.99 TB |

Quota headroom at the last submission was 35 TB of 50, so this fits. It is
still the largest single line item after the factored Hessian, and it is why
the switch is opt-in.

**If it has to be smaller**, the measured options (NOTES 2026-09-05 (II) §8d)
are, in order of return:

1. a **rank-16 PCA of the tau axis** against a fixed basis shipped in the
   runtree: 6.8 kB/candidate for a relative error of 4.2e-7 (ms), 1.3e-5
   (ioni), 1.2e-3 (rad). Not implemented — the raw rows are exported first
   precisely so the basis can be fitted to data rather than guessed;
2. a **coarsened group tier**: 24 of the 42 groups are active-silicon layers
   carrying under 1 % of the resolution each; merging them per subdetector
   takes 42 groups to ~20 with no measurable loss to the mass term, and the hit
   chi2 keeps its own 42;
3. **pruning** rows below 1e-3 of the candidate's own max |S| removes 5.15 % of
   them.

1 + 2 together would be ~3.2 kB/candidate, i.e. 0.12 TB over the full 38 M.

---

## 4. CPU

Controlled A/B, the same 60 events of the J/psi gun, run **sequentially** on the
same node, **user CPU** (this login node's load swings by 4x within minutes, so
wall time is not a measurement — the same job took 26 s and 307 s wall on the
same day):

| config | rep 1 | rep 2 | mean | delta | per candidate |
|---|---:|---:|---:|---:|---:|
| both switches off | 65.30 | 63.56 | 64.43 s | — | — |
| `exportCfGroupExponents=True` | 80.45 | 81.69 | 81.07 s | **+16.6 s** | **+0.28 s** |
| `exportMaterialNoise=True` | 63.96 | 64.59 | 64.28 s | −0.15 s | **0** |

and on the single-track maker (120 mu-gun tracks, 2 reps each):

| config | mean user | delta | per track |
|---|---:|---:|---:|
| `exportMaterialNoise=False` | 44.08 s | — | — |
| `exportMaterialNoise=True` | 46.12 s | +2.04 s | **+17 ms (+4.6 %)** |

**The per-group export costs +0.28 s/candidate** against a production budget of
0.65–0.9 s/candidate (`STATE.md`), i.e. **+30 to +45 %**. That is the real price,
and it is not a bug: a "block" is pooled by parmtype-10/11 global index over a
whole surface-to-surface propagation and typically STRADDLES several material
groups, so the primitives genuinely run once per (block, group). Single-group
blocks were already made to run once and feed both destinations (bitwise, which
is why `sum_g` equals the flat exponent to the last bit there); the remaining
cost is irreducible without changing the model.

For the same reason the per-group extraction offline costs 1.9 s/candidate
against ~0.5 s for the flat one — so this is still a 7x saving over doing it
outside the maker, on top of not writing 430 kB/candidate of step records.

---

## 5. What a reader must know before pooling with the 260905 productions

**The global parameter map does NOT move.** Verified: the `runtree` is
bit-identical (126 452 entries, all 36 branches) between the 260905 build and
this one, on all three smokes. Parmtypes 8/9 were already registered in
`detidparms` by the base class whenever `doRes`, for BOTH makers, so adding the
hit BLOCKS adds no parameter. `globalidxv`, `gradv`, `hessfactorv` and the
runtree can be pooled across the two productions without remapping.

What DOES change, and what a reader has to handle:

| branch | change | consequence |
|---|---|---|
| `ioniurbanv` | stride 11/13 -> **12/14** (group column appended last) | a reader that hard-codes 11 or 13 **mis-parses**. Read the new `ioniurbanstride`. Only present with `exportStepRecords=True` |
| `radstepv` | `radstepstride` 11 -> **12** | same; `radstepstride` was already exported |
| `reseigidx`, `resinfvarv`, `resinfv`, `reshitidx` (two-track only) | **gain** the parmtype-8/9 entries, interleaved with the material ones in propagation order | any code that assumed "every entry is material" must filter — on `reshitcls >= 0`, or on the runtree parmtype of `reseigidx` |
| `reshitidx` (two-track only) | was empty, now filled (-1 = material, else the leg index 0/1) | additive |
| `reseigidx`, `resinfvarv`, `resinfv`, `reshitidx` (single-track, `exportMaterialNoise=True` only) | **gain** ~53 parmtype-15 entries per track | filter on the runtree parmtype; `resinfcov` deliberately EXCLUDES them (they re-partition the parmtype-10/11 noise, `resinfcovgrp` is their sum) |
| `gradv`, `gradllv`, `gradchisqv`, `hesspackedv`/`hessfactorv` (single-track, `exportMaterialNoise=True` only) | the parmtype-15 columns gain the width term | **this is the intended change**; with the switch off they are bit-identical to the 260905 build |

**`resinfcov` and `cfmass_vgf` do NOT change** — verified bit-identical with the
hit blocks on and off. The hit-block variances go into the new `resinfcovhit`.
That is deliberate: `cfmass_vgf = (sigma_m^2 - resinfcov)/sigma_m^2` has to keep
meaning the TOTAL Gaussian share (hits + beamspot + pointing), because that is
what the self-consistent-sigma correction's `a_i = (1 + f_hit) sigma_i/m_i` uses
and what every cache built before the hit blocks existed assumes.

---

## 6. Order of operations

1. **Do not build in `CMSSW_15_0_19_patch2_dev` while `jpsimc20M_*` /
   `dymc8p5M_*` are in `squeue`.** A `scram b` relinks the .so under the running
   jobs and they segfault in the same second. The work is on a branch in a
   second area precisely so that this decision can be taken later.
2. When the productions finish: merge `cvh-exports-260906` into
   `WmassNanoProd_15_0_19_patch2_dev`, build the production area, and re-run the
   smoke (`resolution/smoke_exports_260906.sh <area> <outdir>`) there — it must
   reproduce `scratch_smoke_260906/v2_*` bit for bit.
3. Add `exportCfGroupExponents=True` to `config_jpsimc20M.sh` and
   `config_dymc8p5M.sh`, bump the tags, and resubmit with the existing
   `submit_*.sh` / `resume.sh` machinery unchanged.
4. Point `matres/extract_groups.py` at the new branches instead of the raw step
   records. That is where the 1.9 s/candidate offline extraction disappears:
   the per-group arrays are already in the file.

---

## 7. Provenance

* branch `cvh-exports-260906` off `157775e`, area
  `/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2`
* smoke driver `calibration_studies/resolution/smoke_exports_260906.sh`
  (three cmsRun jobs: J/psi-gun two-track, mu-gun single-track, 48 events of
  2016F Charmonium ALCARECO)
* outputs and logs `/work/submit/david_w/ZMass/scratch_smoke_260906/`
* validation report `calibration_studies/resolution/runs/exports260906/`
* the export contract itself:
  `Analysis/HitAnalyzer/doc/resolution-cf-export.md`
