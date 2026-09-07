# The next production: what the 2026-09-06 exports change

Companion to `STATE.md` (`jpsimc_20M_260905`) and `STATE_dy.md`
(`dymc_8p5M_260905`). Those two ran from
`/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev` @ `157775e`. The new
exports are on the branch **`cvh-exports-260906`**, built in a SECOND area,
`/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2` (a git worktree of the
same repo, so the production area was never touched while its jobs ran).
**As of 2026-09-07 the split is over: both areas hold the same tree at
`ca6058d`. Read §0 first — it says which area to run from.**

Nothing here needs to be re-derived: the physics is bit-identical to
`157775e` on all three smoke samples, so this is a configuration and a volume
decision, not a revalidation.

---

## 0. THE TWO AREAS ARE NOW ONE (2026-09-07) — and which one to run from

**Consolidated.** With the slurm `jpsimc20M_*` arrays finished and the slurm DY
leg cancelled, nothing was running out of the production area any more, so
`WmassNanoProd_15_0_19_patch2_dev` was **fast-forwarded onto
`cvh-exports-260906`** (17 commits, `157775e` -> `ca6058d`, a clean
fast-forward, no merge commit, nothing pushed to any remote). The
`Utilities/XrdAdaptor` pattern — added to the dev2 worktree's sparse checkout
for the null-pointer patch — was added to the production area's sparse checkout
first, so the two checkouts materialise the same set of packages.

**The two areas now hold the same tree, byte for byte:**

* `git diff cvh-exports-260906 --stat` in the production area is empty;
* `diff -rq` of the two `src/` trees (excluding `.git` and `__pycache__`)
  reports only six untracked leftovers in the production area (old
  `globalcor_*.root`, `run.log`, two `debugG4e_*.log`, `toyPlanes_pt3.py`) —
  no tracked file differs;
* the production area was rebuilt (`cmsswlock.sh build scram b -j32`, 152 s,
  0 errors) and re-gated on the three smokes: **every branch bit-identical to
  the dev2 reference** (§7).

| | |
|---|---|
| **run the next production from** | **`/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2`, branch `cvh-exports-260906`** |
| why that one | it is the default of every *live* script — `condor_{jpsimc,dymc}_v2/config_*.sh`, `array_jpsimc_v2.sbatch`, `array_dymc_dev2.sbatch`, `resume_dy_dev2.sh`, `threadscan/run_scan.sh`, `repack_fix_260907/*` — and it is where new export work is committed |
| the other area | `/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev` @ the same commit, branch `WmassNanoProd_15_0_19_patch2_dev`. Identical and built; keep it as the spare / as the non-worktree checkout that owns the repo |
| what this fixes | the v1 configs that still default to `..._dev` (`config_jpsimc20M.sh`, `config_dymc8p5M.sh`, `array_{jpsimc,dymc}.sbatch`, `profiling/run_profile.sh`) are **no longer wrong** — they now pick up the same binaries |

**Keep them in step the same way.** Do the work on `cvh-exports-260906` in
dev2; when it is validated, `git merge --ff-only cvh-exports-260906` in dev and
rebuild. If that merge is ever *not* a fast-forward, the two have diverged and
the divergence must be understood before either is used for a production.

### What made the consolidation non-optional: the `ndof == 0` abort

**A re-production from `157775e` would have lost ~13 % of its DY chunks
outright.** The two-track maker's factored-Hessian rank report reads
`eigvals(nparsfinal - nrank)` with `nrank = min(ndof, nParms)`, and `ndof` is
an **unsigned** member equal to `nvalid + nvalidpixel - 10` summed over both
legs. A Z pair whose two MiniAOD legs carry exactly ten valid-hit-equivalents
lands on `ndof == 0`, the index goes one past the end of the eigenvalue vector,
and Eigen's bounds assert **aborts the process**. A crashed `cmsRun` output has
NO KEYS, so the whole task is lost, not the one candidate.

* measured rate **0.033 aborts per 1000 Z candidates**; `ndof == 0` is EXACTLY
  EMPTY across 197 417 candidates of 20 *completed* `dymc_8p5M_260905` tasks
  while `ndof == 1` and `2` hold 4 and 5 (0.020 and 0.025 per 1000) — the bin
  is empty because landing in it kills the job
* it cost **37 of the first 133 finished tasks (28 %)** of `dymc_8p5M_260905`
* the J/psi leg is untouched: ALCARECO carries the full RECO hit list, so
  `ndof == 0` does not occur there (0 exit-134 failures in `jpsimc_20M_260905`)

Fixed on `cvh-exports-260906` at **`fab515e`** ("A two-track fit with
ndof == 0 must fail, not index past the eigenvalues"): `ndof` is computed
signed and clamped, `ndof <= 0` is a counted fit failure (`fail[ndof]` in the
per-job summary) dropped through the existing `valid` path, and the rank report
is guarded. Bit-identical on all three smokes. It is now in **both** areas, so
the cherry-pick this section used to demand is done.

---

## 1. What is new, and what it is for

| export | branches | why the next production wants it |
|---|---|---|
| **per-material-group CF exponents** | `cf{qop,mass}_grp`, `_grp_ms`, `_grp_ioni_re/_im`, `_grp_rad_re/_im`, `_grp_del` (single-track), `_grp_closure` | lets the mass term float the **parmtype-15 material amounts** the hit chi2 already measures, instead of four unphysical `k_hit/k_ms/k_ioni/k_rad` knobs. It is the input the joint fit of NOTES 2026-09-05 (II) §6 needs at full statistics |
| **hit-class blocks in the two-track maker** | `reshitcls`, `resinfcovhit`, `cf{qop,mass}_hitcls`, `_hitv`, plus the parmtype-8/9 entries in `reseigidx`/`resinfvarv`/`resinfv`/`reshitidx` | the per-hit-class resolution parameters have **never been fitted** — the two-track maker registered no hit blocks at all, so the mass functional had nothing to weigh them with |
| **the two legs' reference covariance** | `Jpsi_covrefmom` (21 floats, upper triangle of the symmetric 6x6), `Jpsi_jacrefmom`, `Jpsi_qoprefplus/minus`, `Jpsi_sigmarelplus/minus`, `Jpsi_rhomom`, `Jpsi_fang` | makes the Jensen and self-consistent-sigma corrections **truth-free on DATA**. Today `rho` and `f_ang` come from an MC measurement and carry +-5 %; each correction alone is 15-43 MeV on `m_Z` |
| **pre-FSR gen mass** | `Jpsigenpre_mass`, `Jpsigenpre_status`, `Jpsigenpre_masslep`, `Jpsigen_massdressed` | the Z channel's FSR kernel comes off the production's own pairs cache instead of a separate FWLite pass whose selection has to be kept in step by hand |
| **material-group process noise** (`exportMaterialNoise`) | `resinfcovgrp` + parmtype-15 entries in `reseigidx`/`resinfvarv` | the quadratic hit-chi2 term now differentiates a group's WIDTH as well as its mean loss — the two functionals of `k_g` were measuring different halves of it. As of the `exportVarianceGrads` row below this applies to the TWO-TRACK maker too; on that maker `exportMaterialNoise` alone only registers the blocks (for `resinfcovgrp` and the influence export), and it takes `exportVarianceGrads` as well to put them in the gradient |
| **the two-track variance (log-det) gradient** (`exportVarianceGrads`) | `gradchisqv`, `gradllv`, `nHessVar`, `hessvaridxv`, `hessvarpackedv`, and CHANGED values in `gradv`/`hesspackedv`/`Jpsi_jacMass`/`*_jacRef` on the parmtype-15 columns | the OTHER half of the line above. The single-track maker got `k_g`'s width term on 2026-09-06; the two-track one had no log-det machinery at all, so its material information was still mean-loss only -- and the mean channel is the WEAK one: on the J/psi gun the parmtype-15 Fisher information goes from 4.07 to 166.3 (41x) when the width is differentiated |
| **per-leg reference energy loss** | `Mu{plus,minus}_maxfracloss`, `dEref`/`maxfracloss` (single-track) | a `dE_ref/p < 0.01` quality requirement on the quadratic term's material information, without the step records |
| **the material group on every step record** | `ioniurbanv` / `radstepv` gain a last column; new `ioniurbanstride` | deletes the offline `pair_ioni_rows` heuristic. Only visible with `exportStepRecords=True` |

## 2. The configuration

Take `config_jpsimc20M.sh` verbatim and add **one option**:

```diff
 EXTRA="numberOfThreads=1 \
  doRes=True exportCfExponents=True exportStepRecords=False \
+ exportCfGroupExponents=True \
+ exportMaterialNoise=True exportVarianceGrads=True varianceGradFamilies=15 \
  fillJac=True fillGrads=False fillGradsFactored=True \
  ...
```

Everything else is unchanged, and every other new export is **on by default**:

| option | value | note |
|---|---|---|
| `exportCfGroupExponents` | **`True`** | default False. This is the +26 kB/candidate decision, see §3 |
| `exportHitResBlocks` | `True` (default) | the parmtype-8/9 blocks. `False` reproduces a pre-2026-09-06 two-track tree exactly |
| `exportMaterialNoise` | **`True`** on BOTH legs now | the parmtype-15 width term. Single track: +17 ms/track and +3 kB/track. Two track: it registers the blocks, and `exportVarianceGrads` below is what puts them in the gradient |
| `exportVarianceGrads` | **`True`** | default False. The two-track log-det gradient/Hessian |
| `varianceGradFamilies` | **`15`** | default (empty) = `{8,9,10,11,15}`. **Use `15`.** It is the layout-preserving subset -- the material-group globals are already columns of `globalidxv` -- so `nParms`, `gradv`, `jacrefv`, `Jpsi_jacMass` and `hessfactorv` keep their shapes and the output still pools with `jpsimc_20M_260905`. Families 8-11 APPEND per-module columns (238 -> 324 median `nParms`) and cost 8x more (see §3) for parameters this maker cannot even apply back: it does not scale the hit covariance by `exp(corparms)` the way the single-track maker does |
| `exportObjective`, `varianceFDGlobalIdx` | leave off | validation-only; an ncons x ncons eigendecomposition (and, for the second, a full re-profile) per candidate |
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

### The variance (log-det) term

Sum of compressed branch bytes EXCLUDING `hesspackedv` -- i.e. the layout a
production runs (`fillGradsFactored`, `exportStepRecords=False`) -- on 24 gun
ditrack candidates. The absolute number is not the production's (24 entries do
not amortize ROOT's per-basket overhead); the DELTA is:

| config | B/candidate | delta |
|---|---:|---:|
| `exportVarianceGrads=False` | 37 902 | — |
| `varianceGradFamilies=15` | 40 568 | **+7.0 %** |
| `varianceGradFamilies=8,9,10,11,15` | 77 732 | **+105 %** |

and the same on 48 REAL DATA candidates (2016F Charmonium ALCARECO, step
records excluded as well): **32 473 -> 34 847 B/candidate, +7.3 %**, of which
1 888 B is the three new branches themselves (`gradchisqv`, `gradllv`, and the
packed variance block at `nHessVar` = 17-29 columns).

Family 15 costs 2.7 kB/candidate: `gradchisqv` + `gradllv` (two more
`nParms`-long float arrays, ~1.8 kB) and the packed variance Hessian block
(`nHessVar` = 21.8 columns, so 21.8*22.8/2 = 249 floats, ~1 kB). It does not
change the SHAPE of `nParms`, `globalidxv`, `gradv`, `jacrefv`, `Jpsi_jacMass`
or `hessfactorv` -- verified on all 59 gun candidates: layout identical, and
every non-parmtype-15 entry of `gradv` and of the `hesspackedv` block
bit-identical. What it does change, and must, are the VALUES on the
parmtype-15 columns of `gradv`, `hess` and `Jpsi_jacMass`.

Families 8-11 cost 40 kB because they APPEND per-module columns: `nParms` goes
from 245.6 to 333.7, every `nParms`-long array grows with it, and `nHessVar`
goes to 109.9, whose triangle is 6 100 floats. **This is the reason to run with
`varianceGradFamilies=15`.**

An earlier version carried the variance block as extra ROWS of `hessfactorv`
instead (`nRank = ndof + nvarcols`). That is also exact, and it needs no
offline change at all, but it costs `nvar x nParms` floats rather than
`nvar (nvar+1)/2`: **+57 %** and **+429 %** for the same two configurations.
The separate block is 8x cheaper at family 15 and 4x at all families, and
`globalfit/extract.py` was taught to add it (and to REFUSE a factored file
whose `gradllv` is filled but which has no `hessvaridxv` -- that combination is
an inconsistent `(G, K)` pair).

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

and the two-track VARIANCE (log-det) term, same protocol, `fillGradsFactored`,
same 60 gun events:

user seconds per rep, SIX interleaved reps (`off`, `var15`, `varall`, repeat):

| config | reps | median |
|---|---|---:|
| `exportVarianceGrads=False` | 65.02, 58.82, 92.49, 72.80, 71.94, 64.00 | 68.5 s |
| `varianceGradFamilies=15` | 64.10, 99.11, 82.29, 71.88, 69.77, 62.39 | 70.8 s |
| `varianceGradFamilies=8,9,10,11,15` | 63.60, 93.54, 82.93, 75.78, 69.18, 63.25 | 72.5 s |

**Not resolvable, and bounded well below 2 %.** The spread of a single arm is
59-92 s, so the medians say nothing. What does is the WITHIN-REP paired
difference, `arm - off` in the same rep:

* family 15: -0.92, +40.29, -10.20, -0.92, -2.17, -1.61 -> median **-1.3 s**
* all families: -1.42, +34.72, -9.56, +2.98, -2.76, -0.75 -> median **-1.1 s**

both NEGATIVE, because the node cools through a rep and `off` runs first. The
one +40 s is the rep that overlapped the 2 GB data smoke.

That is what the structure predicts: the assembly runs ONCE per candidate, at
the converged iteration, on the blocks' own ~5x5 row ranges
(`tr(dV_i R) = tr(D_i R_ii)`, `tr(dV_i R dV_j R) = tr(D_i R_ij D_j R_ji)`)
rather than as ncons x ncons sparse products -- a few 1e6 flops, i.e. single-
digit milliseconds, against a 0.65-0.9 s fit. Unlike the per-group CF exponents
above, there is no reason to trade this one off.

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

1. ~~Do not build in `CMSSW_15_0_19_patch2_dev` while `jpsimc20M_*` /
   `dymc8p5M_*` are in `squeue`.~~ **Done 2026-09-07.** The rule still holds for
   any future build in an area a production is running from — a `scram b`
   relinks the .so under the running jobs and they segfault in the same second
   — but the condition that forced the two-area split is gone: the slurm J/psi
   leg finished, the slurm DY leg was cancelled, and the condor v2 recovery
   jobs run from a payload tarball **pinned in the production's own output
   tree** (`.../jpsimc_20M_260906_v2/payload/overlay_*_xrdfix.tgz`), so they
   depend on neither area's live libraries.
2. ~~Merge `cvh-exports-260906` into `WmassNanoProd_15_0_19_patch2_dev`.~~
   **Done 2026-09-07** — clean fast-forward `157775e` -> `ca6058d` (17 commits),
   sparse checkout aligned, area rebuilt in 152 s, and the smoke re-run there
   reproduces the dev2 reference **bit for bit on all three samples** (§0, §7).
   Nothing was pushed to any remote.
3. Add `exportCfGroupExponents=True` to `config_jpsimc20M.sh` and
   `config_dymc8p5M.sh`, bump the tags, and resubmit with the existing
   `submit_*.sh` / `resume.sh` machinery unchanged. **This is the next step.**
4. Point `matres/extract_groups.py` at the new branches instead of the raw step
   records. That is where the 1.9 s/candidate offline extraction disappears:
   the per-group arrays are already in the file.

---

## 7. Provenance

* branch `cvh-exports-260906` off `157775e`, area
  `/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2`; **as of 2026-09-07 the
  production area `..._dev` (branch `WmassNanoProd_15_0_19_patch2_dev`) is
  fast-forwarded to the same commit `ca6058d` and holds an identical tree**
* smoke driver `calibration_studies/resolution/smoke_exports_260906.sh`
  (three cmsRun jobs: J/psi-gun two-track, mu-gun single-track, 48 events of
  2016F Charmonium ALCARECO)
* outputs and logs `/work/submit/david_w/ZMass/scratch_smoke_260906/`. The
  dev2 references are `v6_default` (build of `fab515e`) and `v7_xrdfix` (build
  of `ca6058d`) — **bit-identical to each other**, i.e. `c2d74e3e647`
  (XrdAdaptor) and `ca6058d96fc` (the Geant4e table mutex) are inert on a POSIX
  input, as intended. The 2026-09-07 re-gate of the merged production area is
  `v8_devmerged`: **all IDENTICAL** against both references —
  gun_tt 246 branches (210 `tree` + 36 `runtree`) over 59 candidates,
  gun_st 157 (121 + 36) over 120 tracks,
  data_tt 264 (228 + 36) over 48 candidates, `runtree` 126 452 entries in every
  file, and the three output files byte-for-byte the same size as the reference
* validation report `calibration_studies/resolution/runs/exports260906/`
* the export contract itself:
  `Analysis/HitAnalyzer/doc/resolution-cf-export.md`

---

## 8. Multithreading (measured 2026-09-06)

The CVH refit in CMSSW_15 is multithreading-capable and the productions were
not using it: every 260905 task ran `numberOfThreads=1` on `--cpus-per-task=1
--mem=6G`. Scanned on one 2000-event DY chunk with the §2 switch set, one arm
at a time on a quiet 64-core node (`production/threadscan/run_scan.sh`):

| threads | wall (s) | CPU (s) | speedup | peak RSS (GB) | RSS/core (GB) |
|---:|---:|---:|---:|---:|---:|
| 1 | 770.7 | 761.3 | 1.00 | 1.76 | 1.76 |
| 2 | 404.2 | 765.5 | 1.91 | 1.85 | 0.93 |
| 4 | 224.9 | 770.5 | 3.43 | 2.00 | 0.50 |
| 8 | 131.7 | 772.8 | 5.85 | 2.12 | 0.27 |

and the J/psi leg (`runCvhJpsiGenMC.py`, 2000 ALCARECO events / 1991 candidates):

| threads | wall (s) | CPU (s) | speedup | peak RSS (GB) | RSS/core (GB) |
|---:|---:|---:|---:|---:|---:|
| 1 | 2039.9 | 2028.0 | 1.00 | 1.36 | 1.36 |
| 2 | 1024.7 | 2003.4 | 1.99 | 1.45 | 0.73 |
| 4 | 535.5 | 2020.6 | 3.81 | 1.61 | 0.40 |
| 8 | 286.4 | 2024.2 | 7.12 | 1.89 | 0.24 |

Total CPU rises **1.5 %** on DY and **falls 0.2 %** on J/psi over the whole range: there is no parallel overhead,
only a fixed 40.4 s serial head (geometry + field + Geant4 physics list), and
`wall(N) = 40.4 + 730.3/N` fits all four points to under 1 %. That head is paid
once per JOB, so on a real 22 120-event chunk the speedup is **3.94x at 4
threads and 7.73x at 8**, not 3.43 and 5.85.

J/psi fits `wall(N) = 36.0 + 2003.9/N` and scales better than DY at the same
event count only because its arm is 2.6x longer, so the serial head is a
smaller fraction. Projected on a real 12 185-event J/psi chunk with the new
exports: 3.40 h at 1 thread, **0.86 h at 4 (3.97x)**, 0.43 h at 8 (7.84x).

**The output is bit-identical.** N threads means N streams and N output files;
sorted candidate by candidate on `(run, lumi, event)` with a stable key
(`production/threadscan/compare_threads.py`), all 213 branches of all 896 DY
candidates and all 228 branches of all 1991 J/psi candidates agree exactly at
2, 4 and 8 threads, and the `runtree` is identical in every stream file. Note that the drivers give each stream its OWN CLHEP
engine wired into Geant4's thread-local RNG, so this is an invariant of the fit
not sampling, not a structural guarantee — re-run the comparison after any
change that could make it consume randomness.

**Memory is set by the task tail, not by the core count.** Peak RSS splits into
~1.71 GB shared (geometry, field, G4 tables, conditions) plus ~51 MB per extra
stream. The 260905 productions' 1-thread MaxRSS was median 1.9 / p99 3.1 / max
3.56 GB (J/psi, 1616 tasks) and median 2.6 / max 3.56 GB (DY, 164 tasks) — the
tail a 2000-event scan does not see. Sizing on `max + (N-1) x 51 MB` with 1.3x
headroom gives **4 threads at `--mem=5G`**: 3.9x the throughput for 0.83x the
memory of today's request, i.e. **1.25 GB/core instead of 6**.

The one real cost is that the `runtree` (13.02 MB, 126 452 entries) is written
into EVERY stream file. On a production chunk that is +5.7 % output at 4
threads and +13.4 % at 8; the candidate tree itself is flat at ~70 kB/candidate.

**Before any multithreaded production is consumed**, the readers that hard-code
`task_*/globalcor_0.root` must move to the `globalcor_*.root` glob or they take
1/N of the statistics, and a reader must take ONE stream's `runtree`, never the
concatenation. **Done on 2026-09-06** -- see sec. 10, which is now a record of
the change rather than a to-do list.

### 8.1 The 4-thread SIGSEGVs were NOT threading (resolved 2026-09-06)

`dymc_8p5M_260906_v2` lost 6 of 380 tasks and `jpsimc_20M_260906_v2` 5 more to
`ExitCode 139`. The `.err` files ended inside
`fillRadiativeSpectrum -> G4MuPairProductionModel::ComputeDMicroscopicCrossSection`
with a second thread in `G4ErrorFreeTrajState::PropagateError`, which reads as a
Geant4 model shared between streams. **It is not.**

* the wrapper dumps `tail -60 local.log`; CMSSW prints the gdb dump in
  DESCENDING thread order, so those 58 lines are the LAST threads, never the
  faulting one. Every other thread also carries a `<signal handler called>`
  frame, because CMSSW's handler sends PAUSE_SIGNAL to all of them
  (`InitRootHandlers.cc:445-505`) -- the paused handler is
  `sig_pause_for_stacktrace`, the crashing one `sig_dostack_then_abort`, and the
  name is what the truncation cut off. **Dump 400 lines and grep for
  `sig_dostack_then_abort`.**
* the crash is a null dereference in the RELEASE's
  `Utilities/XrdAdaptor/src/XrdRequestManager.cc:124-131`: `getQueryTransport`
  leaves its `std::string*` uninitialised and `AnyObject::Get` sets it to 0 when
  the transport query fails, which is exactly what happens for a redirect hop
  with no live connection (`Unable to initiate the connection ... network is
  unreachable`, `Redirect limit has been reached`). `tracerouteRedirections`
  then formats `*hostname_method` and friends unconditionally. It runs on the
  XrdCl JobManager thread -- the framework's own report says
  `Module: non-CMSSW (crashed)`.
* **it is an XROOTD-path bug, so only the grid legs can see it.** The same chunk
  that failed 4/4 on the grid runs 22120/22120 events clean at 4 AND 8 threads
  from POSIX `/ceph`, and SIGSEGVs at event 4401 when the only change is
  streaming the input through `cms-xrd-global`. The 260905 slurm productions
  read `/ceph` directly and never saw it.
* the 5.5-6.1 GB `MemoryUsage` condor logged just before each crash is **the
  crash handler forking gdb**, not a leak: exactly the 7 procs that ever
  exceeded 5120 MB are the 7 that ever returned 139, no other proc exceeded
  2.73 GB, and the instrumented re-runs peak at 2.39 / 2.55 GB.
  `request_memory = 5000` stands; §8's sizing was never the problem.

Fixed in `cvh-exports-260906`: `c2d74e3e647` (null guards in XrdAdaptor -- this
should also go upstream to CMSSW) and `ca6058d96fc` (the audit's one real race:
the reference and ionization-only `G4TablesForExtrapolatorForCVH` builds were
guarded by two DIFFERENT mutexes although both `Initialise()` the same Geant4 EM
models, which write process-wide non-const statics -- a job-startup race whose
failure mode is a silently wrong dE/dx table for a whole job, now one shared
mutex + atomic pointers). Both are I/O-only or locking-only and the three
smokes are bit-identical to `scratch_smoke_260906/v6_default`.

Recovery: `condor_dymc_v2/recover_xrdfix.sh` -- a SECOND pinned payload
(`overlay_<tag>_xrdfix.tgz`) for the named tasks only, because the ~1000 jobs
still queued re-fetch the original pinned tarball on every restart and
overwriting it would mix two binaries into one sample. Same recipe applies to
the J/psi leg once its queue drains.

## 9. HTCondor is a grid submission, not a second cluster

Measured 2026-09-06 (details and probe evidence in
`condor_dymc_v2/STATE_dy_v2.md`): the submit condor pool has **one** local
execute slot (1 CPU on submit06, gated on `Submit_LocalTest`); everything else
is glideins flocked to the CMS global pool. **No condor slot reachable from
submit mounts /ceph, /work or /home** — verified on `mit_tier3` (t3btch001) and
on global-pool slots at DESY, IIHE and Caltech. `mit_tier3` is excluded twice
over: native el7 with no singularity, against an `el9_amd64_gcc12` release.

So a condor leg has to carry the CMSSW area (cvmfs base release + a 64 MB
overlay; the area is NOT relocatable whole — `.SCRAM/RuntimeCache.json` bakes
in the build path and an untar-in-place run silently uses the wrong libraries),
stream its input through `root://cms-xrd-global.cern.ch/` (the ceph chunk paths
are a mirror of the CMS `/store` namespace) and `xrdcp` its output back to
`root://submit50.mit.edu/`. `condor_dymc_v2/` does all three, and verifies the
stage-out by reading the remote size back — `xrdcp` has truncated outputs here
before while returning 0.

### The DY re-production is running there now

`dymc_8p5M_260906_v2`, cluster **3803254**, all 380 chunks, **4 threads,
`request_memory = 5000` MB**, from `CMSSW_15_0_19_patch2_dev2 @ fab515e`.
Validated first: 400 events of chunk 0 run on the grid at INFN Pisa (1 thread)
and Caltech (4 threads) are **bit-identical, all 213 branches, to a local dev2
run on submit82** — across the batch system, two sites, two CPU vendors,
xrootd-streamed versus POSIX input, and 1 versus 4 streams.

Concurrency after submission: 76 running at 2 min, 204 at 5, 318 at 16,
**379 of 380 at 31 min with 0 idle and 0 held**, spread over 12 sites. At that
same moment the slurm DY leg had **1 task running and 198 pending** at 11.4
tasks/h, starved by the 90 running J/psi array tasks. The condor leg does not
compete for that fair share: it is ~380 concurrent tasks of *additional*
capacity while the J/psi arrays keep theirs.

30 first attempts failed and were retried (0 held). The only structured
failure is **SIGILL inside the CVMFS release's own `libXrdCl` at `stagein`** —
16 of 16 at RWTH Aachen and JINR, 8 each — now fenced by `Requirements` in
`condor_dymc_v2/config_dymc_v2.sh` and on the live cluster.

**Note for the operator:** `dymc_8p5M_260905` (slurm) and
`dymc_8p5M_260906_v2` (condor) are the SAME 380 chunks. The v2 has the new
exports and the `ndof == 0` fix; the 260905 one does not. Both were left
running deliberately — the slurm arrays are not to be cancelled from here —
but only one of them is the sample the next fit should use.

---

## 10. Reading a multi-stream production (done 2026-09-06)

Both v2 productions run `numberOfThreads=4`, so a task is **four** files,
`task_XXXX/globalcor_0..3.root`. An event never splits across streams and the
candidate content is bit-identical to a single-thread run after sorting on
(run, lumi, event), so the four files simply concatenate -- but a reader that
names stream 0 takes **1/4 of the candidates** and says nothing.

The audit found ~134 such lines in 78 files, against 33 lines in 29 files that
name stream 0 for the `runtree` only (correct: every stream carries a
byte-identical copy of the 13 MB parameter map, so it must be read once per
run and never concatenated). All of it is fixed. **There is no
`globalcor_0.root` left in any `.py` in `calibration_studies/`**, and the
handful of remaining ones in `.sh` / `.md` are historical statements about
single-threaded productions, not file listings.

### One helper: `resolution/prodfiles.py` (+ `prodfiles.sh`)

```python
task_dirs(base)                        # sorted task directories
stream_files(task_dir)                 # that task's streams, in STREAM order
task_complete(task_dir) / task_reason  # usable?  or why not
runtree_file(task_dir)                 # ONE file: the FIRST EXISTING stream
single_file(task_dir)                  # the one stream of a 1-thread output,
                                       #   warning loudly if there are several
iter_files(base, max_tasks)            # every stream of the first N usable tasks
resolve(spec, max_tasks)               # DROP-IN for sorted(glob(spec))[:ntasks]
last_stats()                           # ntasks_used / skipped / why
```

`resolve` is what the readers call. `spec` may be a glob naming stream 0 (the
index is **widened**, so existing command lines keep working and now see the
whole production), a glob already naming every stream, a production or task
**directory**, or an **`@list.txt`** of explicit inputs -- which is how the
sharded wrappers hand a worker a subset without a symlink farm. A directory
spec auto-detects the output stem (`globalcor` vs `globalcor_resclosure`) and
refuses a directory holding both.

The shell twin `prodfiles.sh` delegates to the same module, so the drivers
cannot drift from the readers: `pf_files`, `pf_task_dirs`, `pf_runtrees`,
`pf_stream_files`, `pf_task_complete`, `pf_clean_incomplete`, `pf_stats`.

```bash
python3 resolution/prodfiles.py "$PROD" --stats      # what a spec resolves to
python3 resolution/prodfiles.py "$PROD" --runtrees   # one file per task
```

### The two ordering hazards, and what they became

1. **`--ntasks` was a FILE cap.** It now caps **TASKS**: `resolve` returns
   every stream of each kept task, so `--ntasks 160` is 160 tasks at any thread
   count. Same for `--nfiles`, `--max-files`, `NTASKS=` in
   `globalfit/run_validation.sh`. Defaults were left where they were, so a
   single-stream production reads exactly what it read before.
2. **Cleanups deleted stream 0 only**, which under 4 threads left streams 1-3
   of a truncated task for a widened glob to swallow. Fixed FIRST, in its own
   commit: the six analysis drivers call `pf_clean_incomplete`, which removes
   **every** stream of a task that has no `.complete`, has an empty stream
   file, or has fewer streams than its sentinel's `streams=N` line declares.
   The pre-run wipes (`cvhcf_e2e_260905.sh`, `kinkfinder/run_v3_species.sh`,
   `runs/stepdamp260905/slurm/smoke.sbatch`) take out every stream too.

**Completeness is decided per TASK, everywhere.** A task is usable iff its
`.complete` exists, none of its stream files is empty, and it has as many
streams as the sentinel declares; otherwise it is skipped whole and counted
(`prodfiles.last_stats()['reasons']`). A directory tree with **no** sentinels
at all -- a hand-made smoke, a staged shard directory -- keeps working: the
requirement turns itself off, with a warning, the moment no task anywhere has
one.

### The symlink farms are gone

`masspairs_parallel.sh` and `extract_parallel.sh` staged each shard as
`task_NNNN/globalcor[_resclosure]_0.root` symlinks -- which cannot represent N
streams of one task and carry no `.complete`. Both now write the shard's file
list to a `.txt` and pass it to `--files`.

`run_pairs_tt_shards.sh`, `run_pairs_tt_parallel.sh` and the two `fixsign`
variants drove **one worker per file** and named the part after the file's task
directory, so four streams of one task collapsed onto one
`part_task_NNNN.npz`: first worker wins, other three hit the resume guard, and
the merge sees a quarter of the candidates -- with a race for the part file on
top. They now iterate over **task directories** and pass the directory as
`--files`. Same for the per-task slurm shards
(`runs/*/slurm/{pairs,extract}_shard.sbatch`) and `engaging/extract_cpu.sbatch`.

### Validation (2026-09-06)

20 tasks of `dymc_8p5M_260906_v2` (4 streams) and 20 of `jpsimc_20M_260905`
(1 stream):

| check | dymc_8p5M_260906_v2 | jpsimc_20M_260905 |
|---|---|---|
| files listed | 20 tasks -> 80 files | 20 tasks -> 20 files |
| `tree` entries read | 197 699 | 260 351 |
| what stream 0 alone would give | 49 169 (**ratio 4.02**) | 260 351 (ratio 1.00) |
| `cf_inmaker.py pairs`: helper vs concat of per-stream-file caches | 194 206 = 194 206, all 9 per-candidate arrays and every provenance scalar **identical** | — |
| `globalfit/extract.py` G, factored H (+ the `hessvar*` block) vs the per-file sum | `max abs dG/abs G = 0`, `max abs dH/abs H = 0`, 197 699 candidates in the quadratic term both ways | — |
| pre-change reader vs post-change reader, same inputs | — | `cf_inmaker` cache **md5-identical**; `globalfit/extract.py` all 18 keys identical |
| an existing cache in `resolution/runs/` | `cf_masspairs_dysmoke_260905.npz` rebuilt from the flat, sentinel-less smoke directory: **md5-identical** | — |

### Known, unrelated, and NOT fixed here

`globalfit/extract.py` **without** `--no-mass` dies on both the 260905 and the
260906 productions:

```
ValueError: .../globalcor_0.root: partial radiative export ['radvgrid']
```

`extract.py:315-321` requires the radiative step branches to be all-four or
none, and every one of these productions ships `radvgrid` alone (they run
`exportStepRecords=False`). Reproduced with the **pre-change** reader on the
same single file, so it is not a threading matter -- it is the offline mass
builder not having caught up with the `radvgrid` export. The quadratic path
(`--no-mass`) is unaffected, which is what the table above measures.

`resolution/runs/` is gitignored, so the fixes to
`runs/{matres,stepdamp260905,clampfix260904}/...` and `runs/solve*_scan.sh`
are on disk but not in the history.
