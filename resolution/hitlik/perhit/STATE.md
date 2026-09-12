# perhit — the DATA version of the hit-residual CF likelihood

## Purpose

The `hitlik` prototype (see `../STATE.md`) builds an exact per-track CF density
for the TRUTH-REFERENCED residual `r = refParms - genParms`.  That object does
not exist on data.  The data-side object is the residual of the HITS about the
FITTED track — the part of the measurement vector the fit has not absorbed,
living in the `n_meas - 5` dimensional complement.  It is exactly what the
quadratic hit chi2 is the Gaussian approximation of, so **this term is the CF
generalisation of the hit chi2**, and it needs no gen information.

What this study answers:

1. Can that complement be whitened stably, per track, inside the maker?  (Yes:
   gates below, `sum_k z_k^2 = chi2` to 2.4e-8.)
2. What does the full non-Gaussian PDF buy over the Gaussian chi2 on those
   components?  **Nothing on the hit classes (0.999), ~10 % in variance on the
   material** — because a per-hit innovation is ~97 % Gaussian hit noise.
3. What do the per-hit components buy over the 5 truth-referenced ones?  On the
   hit classes, **5.8x tighter in sigma (33x in variance)** on the same tracks.
   That is the division of labour: the per-hit term measures the hit classes,
   the truth-referenced term measures the material.
4. The WG question — are the mass term and the hit residuals correlated, and
   how is that accounted for?  Uncorrelated EXACTLY at second order
   (`F^T R = 0`, measured at 2.14e-9); they share fourth cumulants (0.46
   median), and the joint sandwich prices it.
5. What does it cost to export, and what should a data fit actually do?
   **Run the per-hit CF term on a SMALL subsample for the HIT CLASSES** — they
   saturate their priors at tens to hundreds of tracks — and leave the MATERIAL
   to the mass and vertex-constraint terms: measured through the per-hit
   residuals the material width needs ~6e5 tracks for the median group and
   1e8 for the worst.

## The object / model

CVH single-track fit: `ncons` constraints, `nstatefree` free local params,
`V = sum_b dV_b` block diagonal (the `dVs` the maker already builds),
`F = Ffull(:, freestateidxs)`, `C = (F^T V^-1 F)^-1`,
`R = V^-1 - V^-1 F C F^T V^-1`.

* post-fit residual in constraint space `rho = V R r = r + F dxfree`;
* `V R V = V - F C F^T`, and `M = V^{1/2} R V^{1/2}` is an ORTHOGONAL PROJECTOR
  of rank `d = ncons - nstatefree`, with `chi2 = r^T R r = |M s|^2`,
  `s = V^{-1/2} r`;
* row counting: `ncons = 5*nhits + nvalid + nvalidpixel`,
  `nstatefree = 5*(nhits+1)`  =>  **d = n_meas - 5** exactly, with
  `n_meas = nvalid + nvalidpixel`;
* the KINK rows of `rho` are a deterministic function of the MEASUREMENT rows,
  so restricting to the measurement rows loses nothing: the Mahalanobis form is
  invariant under a bijective linear map of the support, hence
  `rho_m^T G^+ rho_m = chi2` with `G = (V R V)[m,m] = V_mm - F_m C F_m^T`, rank
  `d`;
* whitening: any `C_w` with `C_w C_w^T = G` gives `z = C_w^+ rho_m` with
  `Cov(z) = I_d` and `sum_k z_k^2 = chi2`.

**The basis: LDL^T of `G` in MEASUREMENT-ROW ORDER** (hit order, inner to
outer, second pixel coordinate after the first), null pivots skipped.
Component `k` is then hit-row `k`'s post-fit residual CONDITIONED ON the inner
rows' post-fit residuals — local to a hit in the same sense the prototype's q/p
component was local to q/p, and the natural sequence for a term whose
parameters are per-hit-class.  It is NOT the Kalman innovation sequence
(Brown-Durbin-Evans recursive residuals), which is a different orthonormal
basis of the same `d`-space: BDE conditions on the RAW inner measurements and
drops the FIRST 5 components, the LDL of the post-fit covariance drops the LAST
5.  Both give `sum z^2 = chi2`; the composite likelihood differs between them,
and the difference is measured, not assumed.

**It is computed as an orthogonal projection in the STANDARDIZED space**, never
through an inverse (see Defects for why the direct assembly of `G` fails).
With `Fw = V^-1/2 F` and `Q1` from a column-pivoted QR of `Fw`,
`R = V^-1/2 (I - Q1 Q1^T) V^-1/2` is an identity, and

* `s = (I - Q1 Q1^T) V^-1/2 r`, with `chi2 = |s|^2 = r^T R r`;
* `Gs = I - Q1_m Q1_m^T` is its covariance on the measurement rows — a
  submatrix of an orthogonal projector, so its null eigenvalues sit at 1e-16 of
  the unit diagonal instead of 1e-9;
* rank IMPOSED at `dexp = ncons - rank(Fw)`, `S = U sqrt(Lambda)` truncated
  there, then modified Gram-Schmidt (one re-orthogonalization pass) on the ROWS
  of `S` in measurement-row order.  What comes out is the LDL's own triangular
  whitener, `z_k = (s_k - sum_{j<k} c_kj z_j)/|e_k|`, computed stably.
* influence: with `Psi` the whitener (`z = Psi^T rho_m`),
  `Wstd = E_m Psi - Q1 Q1_m^T Psi`, `W = V^-1/2 Wstd`, so
  `sum_b W_b^T dV_b W_b = |Wstd_k|^2 = 1` by construction, and
  `v^(k)_b = W[r0:r0+nb, k]^T dV_b W[r0:r0+nb, k]` are the per-block weights
  the CF evaluator needs.
* the ionization/radiative CF weight is SIGN-SENSITIVE: the sign is
  `-sign(W_b . u)` with `u` the rank-1 direction of `dQI` oriented by its qop
  component.  The q/p functional's single `ioniSign = charge` is the special
  case (see Defects).

`V` is block diagonal over exactly the blocks `resblockrng` enumerates, so
`V^-1/2` is a per-block symmetric square root of `Vinvfull`.

**The five TRUTH-REFERENCED components** (the prototype's term) are appended to
the same arrays by `perHitRefComponents`, so one production carries both sets
and they can be fitted alone (`--comps hit` / `ref`) or together (`all`) on the
SAME tracks.  With `genParms[4]` fixed in the maker, all five are usable.

### Two approximations to state with every number

1. **The composite likelihood.**  The product of whitened marginals drops the
   joint cumulants.  The POINT ESTIMATE stays consistent; the quoted error does
   not, and the sandwich is the accounting — `fisher_cmp.py` estimates `J` by
   BATCH MEANS over batches of whole tracks, so every within-track correlation
   is inside `J`.  Two caveats: the batching is by ROW COUNT while rows per
   track vary, so a fraction `nbatch/ntrk` of tracks straddles a boundary and
   loses its cross terms (2 % at `nbatch = 200`, `ntrk = 10 000`); and the
   efficiency loss relative to the true joint density is measured by none of
   this — only the disjoint-subsample spread is assumption-free.
   At SECOND order the components are exactly uncorrelated, and the hit-class
   information is a second-order (variance) quantity, so the composite form
   mis-counts only the SHAPE information — mild for the hit classes, worse for
   the material.  Both are measured below.
2. **The conditioning cut.**  `max_inflat` cuts ROWS on `Gs_kk/pivot_k`, the
   conditioning of the sequential whitening.  It is a cut on the FIT'S
   COVARIANCE and cannot bias the residual distribution, but it does select hit
   positions: at the default 1e4 it removes ~7 % of rows, concentrated in the
   LAST one or two components of a track, i.e. the outermost hits.

## How to run

### The maker

Area `/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2`, branch
`perhit-residual-cf-260910`, commits **`d1d10985cc9`** (the export) and
**`86380e52d35`** (the whitener and the sign).  Config switches on
`Analysis/HitAnalyzer/test/runCvhResClosure.py`, all off/neutral by default:
`exportPerHitResidual` (False), `perHitCfGroups` (True), `perHitRefComponents`
(True), `perHitShareMin` (0.0).

### The production

```bash
cd /work/submit/david_w/ZMass/calibration_studies/resolution/hitlik/perhit
./run_prod.sh [nparallel] [task_from] [task_to]      # default 60 0 159
```

`resolution_trackres_mugun_ul16_260910_perhit`, output
`/ceph/submit/data/user/d/david_w/ZMass/cvh/resolution_trackres_mugun_ul16_260910_perhit/task_*/globalcor_resclosure_*.root`.
Same input sample and selection as the prototype's
`resolution_trackres_mugun_ul16_260903x_m0`
(`resolution/simprod/filelist_mugun_ul16.txt`, UL16 20-60 GeV mu gun, both
charges, |eta| < 2.4, generalTracks, ideal geometry, default field,
`CgfQoPMode=0`, `fitFromGenParms=False`, tight G4e stepper), so the two are
directly comparable.  What differs is only the export:
`exportPerHitResidual=True perHitCfGroups=True perHitRefComponents=True
exportStepRecords=False exportCfExponents=True exportCfGroupExponents=False`
— the 430 kB/track raw step records are no longer needed because the maker
evaluates the exponents itself.

160 tasks x ~2000 tracks = **320 000 tracks, 190 GiB, 634 kB/track**.  One
60-way block is ~2.4 h, so the run was split over three nodes (submit50 tasks
0-59 at 60-way, submit51 60-109 and submit52 110-159 at 32-way).  Resume is on
the per-task `.complete` sentinel, so re-running any range is safe; do NOT run
two overlapping ranges.

### The offline chain

`run_all.sh <step>` is the single-step driver, `run_stage2.sh [stage ...]` runs
everything after the production in dependency order and skips a stage whose
output exists.  Every `hitlik` script is reused, not forked: they take `R=`,
`NPZ=`, `QNPZ=`, `COMPS=` from the environment, and `comps` is a SPEC resolved
by the loader (`"hit"`, `"ref"`, `"ref0123"`, `"all"`, `"hit:N"`, or a digit
string for the fixed 5 truth-referenced components).

```bash
cd /work/submit/david_w/ZMass/calibration_studies/resolution/hitlik/perhit
./run_stage2.sh                       # extract gates xcum quad cards fits
                                      # fisher efficiency saturation cost
                                      # plots recovery finaltable
./run_stage2.sh certify effall joint tails subfits      # the rest

# or one step at a time
./run_all.sh extract                  # tree -> $R/perhit.npz
./run_all.sh gates --max-tracks 20000 # gates 1-5 straight off the trees
./run_all.sh xcum -o $R/xcum.npz      # the cross-cumulant tables
./run_all.sh cards                    # the 15-card ladder
./run_all.sh fits
./run_all.sh fisher --max-tracks 10000 --nbatch 200 --chunk 8192 -o $R/fisherHJ.npz
./run_all.sh cost ; ./run_all.sh plots ; ./run_all.sh subfits
```

Defaults: `NTRKF=10000` tracks for the Fisher/sandwich step, `NTRKC=8000` for
the card ladder.  Individual cards and fits:
`./run_card.sh <name> --arm cf --comps hit ...` and `./run_fit.sh <name>`.
Figures for the cross-cumulants come from
`plot_xcum.py --npz $R/perhit20k.npz --outpath ~/public_html/ZMass/cvh/260910_perhit`.

### Outputs

`$R` = `/work/submit/david_w/ZMass/calibration_studies/resolution/runs/perhit/`.
`$R/perhit.npz` and `$R/cards` are SYMLINKS into
`/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_perhit_260910/` (32 G) —
`/work/submit` is a 500 G quota, so anything above ~0.5 G goes to ceph.

| file | what |
|---|---|
| `perhit.npz` | the extraction, 8750 tracks (the card/fit/sandwich sample) |
| `perhit20k.npz` | 20 000 tracks (gates, tails, cross-cumulants) |
| `fisherHJ.npz` | H, J, per-batch gradients: `hit` + `ref`, 3 arms |
| `fisherHJ_all.npz` | the same-track joint (`all`), 2 arms |
| `fisherHJ_joint.npz` | the disjoint joint (per-hit + mass), via `fisher_joint.py` |
| `mass_HJ.npz` | the J/psi-gun mass term's own H and J, cached |
| `eff_{hit,ref,all,hitmass,mass,refmass}.npz` | the sandwich tables |
| `cards/*.hdf5`, `fits/*/fitresults.hdf5` | the 15-card ladder + 16 subsample fits |

Figures: **`~/public_html/ZMass/cvh/260910_perhit/`** — 65 panels as PDF+PNG
plus `index.php`: `density_*` (per component set, per hit class, per relative
position), `tailclosure_*`, `efficiency_{material,hitres}`, `xcum_*`.

### Operational standing rules

* **Do not edit a bash script that is running.**  bash reads a script
  incrementally by byte offset, so a running instance resumes at a shifted
  offset and dies with a syntax error.  Copy it, or wait.
* **Do not `scram b`** in `CMSSW_15_0_19_patch2_dev2` while a production from
  that area is running.
* ceph is not readable from every submit node (submit82 has the mount but no
  permission); `run_tf.sh` binds `/ceph/submit` only when the host can read it.

## Results

### The gates (all 160 tasks, 20 000 tracks)

| gate | number |
|---|---|
| 1a `d == n_meas - 5` and `rank(Fw) == nstatefree` | **20000/20000, 0 violations** |
| 1b `n_meas == nValidHits + nValidPixelHits` | 20000/20000 |
| **1c `\|sum_k z_k^2 - chi2\|/chi2` (THE BINDING GATE)** | median **2.43e-8**, p90 5.76e-8, max 2.08e-5 |
| 1c' in-maker `phres_chi2` vs recomputed | max 4.66e-6 |
| 2 `phres_vchk` = `max_k \|sum_b v^(k)_b - 1\|` | median 3.7e-7, p90 4.7e-6, p99 5.6e-5, max 4.5e-2 |
| 2b rank gap `lambda_d/lambda_(d+1)` | median **1.14e15**, p10 5.3e14, min 7.9e13 |
| 3 model `Var(z_k)` under the fit's own Q | **1.00000 exactly** |
| 3 DATA `Var(z)`, per-hit pooled (N = 288 991) | **0.9584**, mean +0.0108, skew +0.009, **kurt 3.43** |
| 3' DATA `Var(z)`, truth-referenced pooled (N = 100 000) | 1.2559, skew +0.43, kurt 22.2 |
| 4 `corr(z_k, truth-referenced pull_j)` | all `\|corr\| < 0.02` for k <= 17 (N > 4000) |
| 4b algebraic `\|sum_b A_b[k].A_b[qp]\|` per track | median **2.14e-9**, p99 1.5e-8, max 8.3e-6 |
| 5 reference component 0 vs the validated `cfqop_*` | **1.2e-7** worst track, all six families |
| 5b `phcf_grp_closure` (`sum_g S_g` vs `S`) | max **9.24e-15** |
| export health | `phres_ok` **100.00 %**, `phcf_nok == d + nref` **100.00 %** |

**Which gate binds.**  Gate **1c** is the binding one: the chi2 side comes from
`|sfull|^2 - |Q1^T sfull|^2`, which never touches the whitener, so the identity
tests the projector, the eigen-truncation, the Gram-Schmidt and the residual
together.  Gate **5** is the other: reference component 0 IS the q/p functional
(column 0 of `L^-T` is `e_0/sigma_qp`), so comparing it to the independently
validated `cfqop_*` tests the influence, the per-block weights, the pooling and
the sign.  Gate **2** is largely BY CONSTRUCTION (`|Wstd_k|^2 = 1` is what the
Gram-Schmidt sets) — it still catches a wrong `dV_b`, a mis-indexed block or a
missing family, but it is not evidence about the truncation.
`phres_gchk` (median 0.02, max 1.8e4) is NOT a defect: it is
`|T (Gs - Gs_trunc) T^T|`, the discarded eigenvalues (~1e-16) times
`|T_k|^2 ~ 1/pivot_k`, i.e. it re-measures `phresinflat`.

### The physics already visible in the gates

| quantity | value |
|---|---|
| DATA `Var(z)`, per-hit components | **0.958** — the fit's assumed hit variances are ~4 % too large |
| per-component DATA variance | 1.012 at k=0 falling to ~0.82-0.90 by k ~ 17 |
| DATA `Var(pull_j)`, the 5 truth-referenced components (8750 tracks) | 1.015 / 1.033 / 1.070 / 1.115 / **1.818** |
| `phcf_vgf`, the hit (Gaussian) share of a per-hit component | median **0.968**, p10 0.382 |
| `phresinflat` = `Gs_kk/pivot_k` | median 3.9, p90 456, p95 8.3e6; `max_inflat = 1e4` keeps **92.7 %** of rows |
| `phcf_msec` per track under 60-way contention | median 2092 ms (993 ms uncontended), 163 ms/component |

Two readings.  (i) The per-hit term should be strong on hit classes and weak on
material: a per-hit innovation is ~97 % Gaussian hit noise, and the outermost
KEPT components sit nearest the rank boundary where the conditional variance is
a small difference and the fit's assumed hit error is least well matched.
(ii) It is not as material-blind as a small smoke suggested — `vgf` p10 is
0.382, so a tail of rows carries substantial material.

### Gate 3 in full (`valdens.py`, 120 tracks, all three arms)

Row-averaged predicted density by the same inverse Fourier transform the term
uses, integrated on |z| < 40:

| component set | N | arm | norm | mean | var(density) | var(model) | var(DATA) |
|---|---|---|---|---|---|---|---|
| per-hit | 1608 | cf | 1.000000 | +0.000000 | 1.00320 | 1.00318 | **0.89864** |
| per-hit | 1608 | gauss | 1.000000 | +0.000000 | 1.00318 | 1.00318 | 0.89864 |
| per-hit | 1608 | gaussq | 1.000000 | +0.000000 | **1.00000** | **1.00000** | 0.89864 |
| reference | 600 | cf | 0.999996 | +0.000010 | 1.03281 | 1.02836 | **1.00720** |
| reference | 600 | gauss | 1.000000 | +0.000000 | 1.02836 | 1.02836 | 1.00720 |
| reference | 600 | gaussq | 1.000000 | +0.000000 | **1.00000** | **1.00000** | 1.00720 |

1. `gaussq`'s variance is **1.00000 exactly** for BOTH component sets — it
   really is the fit's own Q-matrix decomposition of a unit-variance component,
   i.e. it IS the Gaussian hit chi2 this study is measured against.
2. The CF's model variance is **1.0032** for the per-hit components against
   **1.0328** for the reference ones: the same Rossi-vs-Moliere gap, diluted by
   the hit share.  **That is quantitatively why the non-Gaussianity cannot buy
   anything on the per-hit components and does on the reference ones.**
3. `cf`'s norm 0.999996 / var 1.0328 against var(model) 1.0284 on the reference
   set is the tau-grid truncation at 7.89 — 0.4 % on the second moment, the
   same truncation the prototype ran with.

### NLL at MC truth (card build, 8000 tracks)

| card | rows | NLL(0) | vs CF |
|---|---|---|---|
| `ph_cf` (per-hit, full CF densities) | 106 664 | **149 289.577** | -- |
| `ph_gaussq` (per-hit, the fit's own Q = the chi2) | 106 664 | 149 316.290 | **+26.71** |
| `ph_gauss` (per-hit, variance-matched Gaussian) | 106 664 | 149 321.023 | +31.45 |
| `ph_cf_ref` (the 5 truth-referenced components) | 40 000 | 59 785.719 | -- |
| `ph_cf_all` (both sets, same tracks) | 146 664 | 209 075.296 | -- |
| the J/psi-gun mass term (24 000 candidates) | | -47 431.190 | |

The CF is the best description on the per-hit rows too, but only by
**2.50e-4 per row** against the prototype's 0.0155 per row on the
truth-referenced ones — a factor 62, the same factor the hit share predicts.
And `ph_cf_all` = `ph_cf` + `ph_cf_ref` to **6 decimal places**
(209 075.296 = 149 289.577 + 59 785.719), i.e. the joint card really is the
product of the two composite likelihoods on the same tracks — which is exactly
the object whose error has to be checked with a sandwich.

### The fits, certified (`certify.py`, 8000 tracks)

`NLLred` is `nllvalreduced` at the minimum, EDM is rabbit's own
`1/2 g^T H^-1 g` with the full Hessian.  Values are PHYSICAL: `k` = ln material
amount (the card value x 20, since the card is whitened by the 0.05 tier prior
/ 0.0025 card sigma) and `eps` = the linear hit-variance scale (the card value
itself, `--hit-prior 1.0`).

| fit | npar | NLLred(min) | EDM | `k(material_tib_support)` | `eps(hitres_str_N3_lo)` |
|---|---|---|---|---|---|
| `ph_cf` | 60 | 148 811.8385 | 5.05e-12 | -0.01214 +- 0.04574 | +0.07197 +- 0.01588 |
| `ph_gauss` | 60 | 148 870.1369 | 7.65e-10 | -0.04245 +- 0.04495 | +0.07038 +- 0.01583 |
| `ph_gaussq` | 60 | 148 867.7788 | 1.64e-09 | -0.01967 +- 0.04543 | +0.07232 +- 0.01585 |
| `ph_cf_ref` | 60 | 59 270.3048 | 4.59e-10 | +0.00963 +- 0.03861 | **-0.13748 +- 0.09170** |
| `ph_cf_all` | 60 | 208 289.9535 | 5.38e-11 | -0.03106 +- 0.03680 | +0.06377 +- 0.01555 |
| `ph_inj_cf` | 60 | 148 812.1568 | 1.48e-11 | -0.02042 +- 0.04538 | +0.07164 +- 0.01587 |
| `ph_inj_gaussq` | 60 | 148 868.2487 | 1.66e-12 | -0.02849 +- 0.04508 | +0.07197 +- 0.01584 |
| `ph_inj_hit` | 60 | 148 811.8362 | 1.02e-09 | -0.01214 +- 0.04574 | **-0.02546 +- 0.01443** |
| `ph_inj_hit_gaussq` | 60 | 148 867.7765 | 3.05e-14 | -0.01967 +- 0.04543 | -0.02514 +- 0.01441 |
| `ph_inj_cf_ref` | 60 | 59 270.3112 | 1.86e-12 | -0.01043 +- 0.03812 | -0.14123 +- 0.09147 |
| `ph_inj_cf_all` | 60 | 208 290.7813 | 2.17e-12 | -0.05376 +- 0.03632 | +0.06353 +- 0.01554 |
| `ph_mass` | 42 | -47 432.0275 | 2.04e-14 | -0.02377 +- 0.03787 | -- |
| `ph_inj_mass` | 42 | -47 431.3575 | 9.33e-11 | -0.04506 +- 0.03722 | -- |
| `ph_resmass` | 60 | 101 379.7506 | 1.05e-11 | -0.03099 +- 0.03664 | +0.07212 +- 0.01587 |
| `ph_inj_resmass` | 60 | 101 380.5804 | 1.65e-10 | -0.05404 +- 0.03601 | +0.07198 +- 0.01587 |

**15/15 certified at EDM < 1e-3.**  The injection tables further down use the
same physical units.

**THE NUMBER THAT SHOWS WHAT THE PER-HIT COMPONENTS BUY.**  On
`hitres_str_N3_lo` the five truth-referenced components give **+-0.0917** and
the per-hit ones **+-0.0159** — **5.8x tighter in sigma, 33x in variance** on
the same 8000 tracks.  (`ph_cf_ref`'s -0.1375 +- 0.0917 also reproduces the
prototype's -0.13711 +- 0.11688 within its error, from a different production.)
That is the whole point of the per-hit term, and it is the extra COMPONENTS,
not the non-Gaussianity.  On the MATERIAL the two sets are comparable
(`k` +-0.0457 per-hit vs +-0.0386 truth-referenced) and the same-track joint
`ph_cf_all` is the tightest, +-0.0368.

### THE HEADLINE — the sandwich

`fisher_cmp.py` (H by 60 HVPs, J by 200 batch means over whole tracks) +
`efficiency.py`, at MC truth, with the parmtype-15 tier priors.  Every number
is the ACTUAL (sandwich) variance, never the nominal Fisher one.  Build times:
`fisherHJ.npz` (`hit` + `ref`, 3 arms) H in 880-957 s (`hit`) / 334-352 s
(`ref`) per arm; `fisherHJ_all.npz` (`all`, 2 arms) 1057-1085 s.
`|sum_m g_m - g| = 5.6e-11` in every arm; bootstrap/sandwich 0.997-0.999 (as it
must be algebraically — it validates the batching, not the sandwich).

**EFFICIENCY** `sigma^2(chi2, ACTUAL) / sigma^2(CF, ACTUAL)`:

| component set | rows | MATERIAL marginal | MATERIAL prior-free | HIT CLASSES marginal | HIT CLASSES prior-free |
|---|---|---|---|---|---|
| **`hit`** (per-hit complement, 13.33 comps/track) | 106 664 | **1.097** (1.043-1.174) | **1.171** (1.083-1.240) | **0.999** (0.969-1.010) | **1.001** (0.969-1.018) |
| **`ref`** (the 5 truth-referenced components) | 40 000 | **1.486** (1.405-1.863) | **1.393** (1.129-1.976) | **1.108** (0.975-1.338) | **1.117** (0.991-1.313) |
| **`all`** (BOTH, same tracks) | 146 664 | **1.510** (1.400-1.860) | **1.657** (1.267-2.166) | **1.011** | **1.018** (0.98-1.33 per class) |
| what the chi2 CLAIMS (quoted/quoted) | | 1.003 / 0.998 / 1.003 | | 0.981 / 0.943 | |

**This is the central result and it is exactly what `phcf_vgf` predicted.**  A
per-hit innovation is ~97 % Gaussian hit noise, so replacing the chi2 by the
full non-Gaussian PDF buys **nothing** on the hit classes (0.999) and only
~10 % in variance on the material.  On the truth-referenced components, which
are material dominated, it buys **1.49x** in variance on the material and
1.11x on the hit classes.

### SANDWICH / QUOTED — and the composite likelihood, MEASURED

| component set | CF marginal | chi2 marginal | **CF prior-free, material** | **chi2 prior-free, material** | CF prior-free, hit classes | chi2 prior-free, hit classes |
|---|---|---|---|---|---|---|
| `hit` | 0.964 | 0.982 | **0.529** | 0.563 | **1.067** | 1.090 |
| `ref` | 0.934 | 1.028 | **0.999** | **1.237** | 1.082 | 1.191 |
| `all` | 0.905 | 0.957 | **0.939** | 1.204 | | |

Read the PRIOR-FREE column: `S/Q = sqrt(J_pp/H_pp)`, and `H` is a sum over ROWS
while `J` is estimated by batch means over WHOLE TRACKS, so the ratio measures
exactly the within-track correlation the product-of-marginals likelihood drops.

* **`ref`, CF: 0.999.**  The five truth-referenced components are effectively
  independent for the material amounts — the composite form is right and the CF
  model describes the data (the same statement the prototype made).
* **`ref`, chi2: 1.237.**  The Gaussian's own error is optimistic by 24 % in
  sigma, 1.53x in variance — the Moliere tail it does not model.
* **`hit`: 0.529 for BOTH arms** (`J_pp/H_pp = 0.28`).  The per-hit
  innovations' material scores are strongly ANTI-correlated within a track,
  because `sum_k z_k^2 = chi2` with only `d = n_meas - 5` degrees of freedom.
  **The per-hit composite likelihood over-counts the MATERIAL information by
  1/0.28 = 3.6x in variance, and the sandwich is what corrects it.**  It is the
  same factor in both arms, so the EFFICIENCY ratio is unaffected.
* **`hit`, hit classes: 1.067.**  No over-counting: the hit-class information
  is a per-row VARIANCE quantity and the rows really are nearly independent for
  it.  This is the measured version of approximation 1 above — the composite
  form mis-counts the shape information and not the second-order information.

**The same-track joint (`all`).**  `sandwich/quoted` is **0.905 (CF) / 0.957
(chi2)** marginal and **0.939 / 1.204** prior-free — NOT above 1 for the CF, so
the joint does not over-count relative to what its arms already do.  What it
does show is the absolute errors: for `material_tib_support` the CF ACTUAL
sigma is 0.0028 (`hit`), 0.0037 (`ref`) and 0.0038 (`all`), while the QUOTED
sigma falls monotonically 0.0072 -> 0.0062 -> 0.0058.  **The joint's quoted
error improves and its actual error does not** — that IS the over-counting,
seen in the only place it can be seen, and the sandwich prices it correctly.

### The tails

Per-hit components, 20 000 tracks / 288 641 rows; all three arms normalise to
**1.000000**, DATA `Var(z) = 0.9527`, mean +0.0058:

| P(\|z\| > t) | data | CF | data/CF | chi2 (fit's Q) | data/chi2 |
|---|---|---|---|---|---|
| 1 | 0.30621 | 0.31709 | 0.97 | 0.31734 | 0.96 |
| 2 | 0.03866 | 0.04578 | 0.84 | 0.04551 | 0.85 |
| 3 | 0.00280 | 0.00293 | **0.96** | 0.00270 | 1.04 |
| 4 | 0.00051 | 0.00015 | **3.43** | 0.00006 | **7.98** |
| 5 | 0.00010 | 0.00003 | **2.90** | 6e-7 | **169** |

The CF is **2.3x (4 sigma) to 58x (5 sigma)** closer to the data than the chi2,
but both arms MISS a real tail that neither models.  **That tail is the hit
noise, not the material**: both arms treat the hit (Gaussian) share — ~97 % of
a per-hit innovation — as exactly Gaussian, so a 5e-4 data probability beyond
4 sigma against a CF 1.5e-4 is a statement about hit-resolution tails (bad
clusters, unmodelled pixel/strip response), which the 18 hit-class parameters
absorb in WIDTH and cannot absorb in SHAPE.

Truth-referenced components, same 20 000 tracks (`--comps ref --group-by
refparm`):

| P(\|z\|>4) | data | CF | data/CF | chi2 | data/chi2 |
|---|---|---|---|---|---|
| q/p | 0.00235 | 0.00248 | **0.95** | 0.00006 | **37** |
| lambda | 0.00215 | 0.00069 | 3.10 | 0.00006 | 34 |
| phi | 0.00365 | 0.00287 | **1.27** | 0.00006 | **58** |
| d0 | 0.00205 | 0.00068 | 3.03 | 0.00006 | 32 |
| **z0** | 0.00955 | 0.00037 | **25.7** | 0.00006 | **151** |

The first four rows reproduce the prototype's table **digit for digit** from a
completely different production and export path.  At 5 sigma the chi2 is wrong
by 680-3400x and the CF by 0.8-4.0.  **z0 is the exception and it is new**:
`Var(z_4) = 1.928` (1.39x in sigma against 1.04-1.10 for the other four) and
its 4-sigma tail is 26x what the CF predicts — see Open items.

### The fitted hit-class scales at MC truth (physical `eps`, 8000 tracks)

What the per-hit term MEASURES.  They are large and structured — the fit's
assumed hit variances are wrong class by class, not by an overall factor:

| class | CF `eps` | class | CF `eps` |
|---|---|---|---|
| `str_N1_lo` | **-0.141 +- 0.016** | `pix_x_q3` | **+0.574 +- 0.067** |
| `str_N2_lo` | -0.031 +- 0.015 | `pix_x_q2` | +0.261 +- 0.060 |
| `str_N3_lo` | +0.072 +- 0.016 | `pix_x_q1` | -0.111 +- 0.034 |
| `str_N4_lo` | **-0.419 +- 0.026** | `pix_y_q1` | -0.138 +- 0.032 |
| `str_N2_hi` | -0.091 +- 0.031 | `pix_y_q3` | +0.144 +- 0.050 |
| `str_N3_hi` | +0.167 +- 0.028 | `pix_y_q2` | +0.066 +- 0.049 |
| | | `pix_y_q0` | -0.106 +- 0.046 |

The Gaussian arm gives the same values to well within one sigma, as the
efficiency 1.00 says it must.  The per-class spread (-0.42 to +0.57 in
variance) is a measurement the truth-referenced term could not make: its errors
on the same parameters are 0.09-0.12.

### Injection recovery (8000 tracks)

**Material**, `material_tib_support` x1.05 (`k` = +0.0487902, i.e. +5.000 %
material).  Values are PHYSICAL `k`.  Prior-corrected by
`f = 1 - sigma_post^2/sigma_pri^2` with the tier prior **0.05 in `k`**:

| channel | baseline | injected | shift | /truth | f_pri | **corrected/truth** | pull | leak rms |
|---|---|---|---|---|---|---|---|---|
| per-hit, CF | -0.01214 +- 0.04574 | -0.02042 +- 0.04538 | -0.00828 | -0.170 | 0.176 | **0.964** | -0.02 | 0.016 |
| per-hit, fit's Q (the chi2) | -0.01967 +- 0.04543 | -0.02849 +- 0.04508 | -0.00882 | -0.181 | 0.187 | **0.965** | -0.02 | 0.015 |
| truth-referenced (5 comps), CF | +0.00963 +- 0.03861 | -0.01043 +- 0.03812 | -0.02006 | -0.411 | 0.419 | **0.982** | -0.01 | 0.028 |
| **per-hit + truth-referenced, SAME tracks** | -0.03106 +- 0.03680 | **-0.05376 +- 0.03632** | -0.02271 | -0.465 | 0.472 | **0.985** | -0.01 | 0.028 |
| J/psi-gun MASS term alone | -0.02377 +- 0.03787 | -0.04506 +- 0.03722 | -0.02128 | -0.436 | 0.446 | **0.978** | -0.02 | 0.052 |
| **per-hit + MASS, DISJOINT samples** | -0.03099 +- 0.03664 | **-0.05404 +- 0.03601** | -0.02304 | -0.472 | 0.481 | **0.981** | -0.02 | 0.043 |

Every channel recovers the 5 % injection to **96-99 %** with a pull below
0.02 sigma.  The two joints are the tightest and the most accurate and they
agree with each other (0.0368 / 0.0366) — one consistent amount across three
different objectives.  Largest leakage: `material_tob_support` -0.15 to
-0.27 sigma in the material-sensitive channels, `hitres_str_N3_hi` -0.06 sigma
in the per-hit ones.  (Prototype, same injection: residual CF 0.0404 / 0.979,
mass 0.0372 / 0.978, joint 0.0342 / 0.985.)

**Hit class**, `hitres_str_N3_lo` variance x1.10 on the data side.  `hit_mode`
is LINEAR, so the expected shift is
`-eps_inj/(1+eps_inj) x (1+eps_base) = -0.09745`:

| arm | baseline | injected | shift | **/truth** | f_pri | pull | leak rms |
|---|---|---|---|---|---|---|---|
| per-hit, CF | +0.07197 +- 0.01588 | -0.02546 +- 0.01443 | -0.09743 | **1.000** | 1.000 | -0.00 | 0.000 |
| per-hit, fit's Q | +0.07232 +- 0.01585 | -0.02514 +- 0.01441 | -0.09746 | **1.000** | 1.000 | -0.00 | 0.000 |

Exact recovery, zero pull, largest leakage onto any other parameter below
0.005 sigma.  The hit classes carry no prior shrinkage here
(`f_prior = 1.000`), so this is the raw, uncorrected number.

### The assumption-free check: 8 disjoint subsample fits per arm

`subfits.sh` K=8 x NSUB=1000 = the SAME 8000 tracks the sandwich used, both
arms, each subsample FITTED for real; `sigma_full = spread/sqrt(K)`.  Assumes
nothing — in particular not `H = J`.

Convergence, itself a result: **8/8 in BOTH arms** (CF EDM 1.3e-17 .. 2.6e-12,
chi2 4.8e-16 .. 1.6e-11).  The prototype's truth-referenced version had CF 7/8
and the Gaussian only 5/8; on the per-hit components, where both densities are
nearly the same, both are equally well conditioned.

| | EMPIRICAL (spread), median | 16-84 % | SANDWICH, median | agree? |
|---|---|---|---|---|
| **MATERIAL groups** | **1.130** | 0.928 - 1.219 | **1.097** | **yes** |
| **HIT CLASSES** | **1.025** | 0.953 - 1.086 | **1.002** | **yes** |

With K = 8 the spread carries a 27 % statistical error per parameter, so the
MEDIAN is the number.  Both agree with the sandwich within that — the per-hit
headline (the full PDF buys ~10 % in variance on the material and nothing on
the hit classes) is confirmed with no model assumption.

### Saturation, and the recommendation it decides

`saturation.py`, `N_sat = n0 (sigma_free(n0)/sigma_prior)^2` at
`n0 = 8000` tracks, standalone form `sqrt(J_pp)/H_pp`:

| component set / arm | groups with information | median `N_sat`, MATERIAL | 16-84 % | max | median `N_sat`, HIT CLASSES |
|---|---|---|---|---|---|
| `hit`, CF | 26/42 | **6.37e5** | 2.9e4 - 4.7e6 | **1.01e8** | **14.5** (max 307) |
| `hit`, chi2 | 27/42 | 9.53e5 | 3.6e4 - 7.3e6 | 5.56e9 | 14.2 (max 302) |
| `ref`, CF | 27/42 | 3.74e6 | 2.6e4 - 5.5e7 | **4.1e9** | 7.3 - 39 (the 7 printed) |

**The two parameter families are four to five orders of magnitude apart.**  The
HIT CLASSES saturate almost immediately — median 14.5 tracks, worst 307 — so a
few thousand tracks already know them better than their prior does.  The
MATERIAL does not: through the per-hit residuals the median group needs
**6.4e5** tracks and the worst **1.0e8**, i.e. a large fraction of, or more
than, the whole 41 M-track sample.

**THE RECOMMENDATION.**  Run the per-hit CF term on a SMALL subsample and read
the HIT CLASSES off it: 10^5 tracks is 300x the worst hit-class saturation
point and costs 63 GB as produced (7.3 GB in the `cost.py` layout), so that
term is essentially free, and the ordinary quadratic term (which handles the
alignment/field MEANS exactly, through `gradv`/`hesspackedv`) runs on the
DISJOINT remainder, with `H` and `J` adding exactly and nothing double counted.
**The MATERIAL amounts do not come from this term.**  Their information is in
the track-parameter and constraint directions — the mass and vertex-constraint
terms, whose material sandwich efficiency is 1.49-2.7x — or from the per-hit
term over the FULL sample, and the export bill of a per-hit term large enough
to matter for the material is the full 23.2 TB, not a subsample.  The
alternative — exporting the per-track mean Jacobian `D = W^T J` (13 kB/track,
0.6 TB at 41 M) so the CF term can carry the means itself — buys hit-class
parameters that are already saturated, so it is NOT justified by these numbers;
it becomes interesting only if the CF term is wanted for the alignment/field
means themselves.

### Cost (one pinned CPU, `taskset -c 40`, `OMP_NUM_THREADS=1`, 2000 tracks, arm `cf`)

| | `hit` (13.40 comps/track) | `ref` (5) | `all` (18.40) |
|---|---|---|---|
| rows | 26 801 | 10 000 | 36 801 |
| group rows | 282 977 | 129 939 | 412 916 |
| NLL | 29.1 us/row, **389 us/track** | 34.7 / 174 | 30.4 / 559 |
| NLL + gradient | 106.4 us/row, **1426 us/track** | 129.8 / 649 | 110.4 / 2031 |
| one HVP | 398.5 us/row | 490.8 | 414.3 |
| full Hessian (60 HVPs) | 641 s | 295 s | 915 s |
| card in memory | 0.483 GB (**241.6 kB/track**) | 0.218 GB (109.0) | 0.701 GB (350.5) |
| one NLL+grad over 320 k tracks | **456 s** | 208 s | 650 s |
| **one NLL+grad at 41 M tracks** | **16.24 h CPU** | 7.39 h | 23.13 h |
| one HVP at 41 M tracks | 60.8 h CPU | 27.9 h | 86.8 h |

The prototype's 4-component truth-referenced term was 1.3 h CPU for one
NLL+grad at 41 M; the per-hit version is **12x** that (3.4x the rows, 10.6
material groups on each).  It is a dense `(chunk, nt)` matmul per family, so a
GPU is 20-60x this hardware: **~20-50 min of GPU** for one NLL+grad over 41 M
tracks, i.e. still the same order as the mass term.

The maker's per-track CF evaluation is `phcf_msec` median 2092 ms under 60-way
contention (993 ms uncontended), 163 ms/component; the naive `d` separate
`cvhcf` passes therefore cost about as much as the fit itself (~1 s/track).

### The export bill (read off `task_0000`, 2000 tracks, compressed)

| branch | kB/track compressed | raw |
|---|---|---|
| `phcf_grp_rad_im` | 87.25 | 94.23 |
| `phcf_grp_ioni_im` | 87.24 | 94.23 |
| `phcf_grp_del` | 85.57 | 94.23 |
| `phcf_grp_ioni_re` | 84.75 | 94.23 |
| `phcf_grp_ms` | 84.65 | 94.23 |
| `phcf_grp_rad_re` | 84.63 | 94.23 |
| **the six per-group exponent tables** | **514.1** | 565.4 |
| `phcf_{ms,del,ioni_re,ioni_im,rad_re,rad_im}` (flat, per component) | 28.2 | 29.9 |
| `phresbv` (the influence vectors) | 16.33 | 21.70 |
| `phresvarv` | 4.2 | 4.4 |
| `phcf_grp_vq{ms,io}` | 2.9 | 3.0 |
| `phcf_hit*`, `phresz/raw/row/hit/dim/cls/piv/inflat` | < 1 | |
| **all `ph*`** | **567.0** | |
| (`hesspackedv`, the quadratic term, for scale) | 57.88 | 99.24 |
| whole file | **630.2** | 736.1 |

The per-group tables are 91 % of the bill and compress by only 9-10 % (float32
noise to zlib), so the compression has to be ALGORITHMIC:

| variant | kB/track | TB at 41 M |
|---|---|---|
| everything as produced here | 567 | **23.2** |
| per-group split OFF (one exponent set, all groups summed) | 53 | **2.2** |
| ... + the card's own `--prune-frac 0.001` (discards 83 % of group rows) | ~140 | 5.7 |
| ... + rank-16 tau PCA on top of the pruning | ~35 | 1.4 |
| `cost.py` at the measured multiplicities, rank-16 tau PCA, `hit` | **73.3** | **3.00** |
| ... `ref` | 32.0 | 1.31 |
| ... `all` | 106 | 4.32 |
| ... `q/p` only (the status quo) | 5.5 | 0.22 |

So the algorithmic compression is worth a factor **7.7** on what this
production actually wrote, and the per-hit term then costs **3.0 TB** against
the truth-referenced version's 1.3 TB and the status quo's 0.22 TB.

### The WG question: mass term vs hit residuals (`xcum_perhit.py`, 20 000 tracks)

**Gaussian level: uncorrelated, EXACTLY.**  Per track the algebraic
`|sum_b A_b[k].A_b[qp]|` (this IS `F^T R = 0`) is median **2.14e-9**, p99
1.5e-8, max 8.3e-6.  Ensemble `corr(z_k, z_qp)` by hit position: -0.0005 /
+0.0015 / -0.0076 / -0.0086 / -0.0063 / +0.0030 / +0.0132 against a 1/sqrt(N)
of 0.0043-0.0094 — consistent with zero in every bin.  **On MC their Fisher
information ADDS at the Gaussian level.**

The reference-parameter cross-cumulants reproduce the prototype's table to
three digits from a completely independent route (the maker's `phresbv` instead
of the offline step records): 0.138 / 0.088 / 0.095 / 0.329 / 0.279 / 0.712.

**Fourth order: they DO share non-Gaussianity, and it depends on where the hit
is.**  `kappa(z_qp,z_qp,z_k,z_k)/sqrt(kappa4 kappa4)`, per track:

| relpos | 0.00-0.12 | 0.12-0.25 | 0.25-0.38 | 0.38-0.50 | 0.50-0.62 | 0.62-0.75 | 0.75-0.88 | ALL |
|---|---|---|---|---|---|---|---|---|
| median | 0.395 | 0.594 | **0.654** | 0.590 | 0.331 | 0.118 | **0.052** | **0.464** |
| p90 | 0.690 | 0.802 | 0.817 | 0.790 | 0.683 | 0.337 | 0.160 | 0.765 |

The inner-to-middle hits share most of their fourth cumulant with the q/p pull
and the outermost almost none — the material distribution seen from the q/p
functional's own influence.  By hit class it splits by STRIP
MULTIPLICITY/charge rather than by layer: the `_hi` classes 0.56-0.61, the
`_lo` ones 0.25-0.42, pixels 0.34 (y) to 0.56 (x).

**Adjacent per-hit innovations**: median **0.505**, p90 0.902 (247 732 pairs),
rising from 0.34 at the innermost pair to 0.63 in the middle of the track —
the composite-likelihood sizing between neighbours.

READ IT RIGHT: a per-hit innovation is only ~3 % material, so a 0.46
correlation of the fourth cumulants is 0.46 of a small number.  The consequence
is measured, not assumed, by the same-track joint sandwich (`--comps all`,
above).  Figures `xcum_corr_zqp_vs_relpos.pdf`, `xcum_ref0_vs_relpos.pdf`,
`xcum_adjacent_vs_relpos.pdf`, `xcum_ref0_by_class.pdf`.

**The two joints are different objects.**  SAME tracks (`ph_cf_all`,
`fisherHJ_all.npz`): uncorrelated exactly but not independent, so the product
form can over-count and the signature is `sandwich/quoted > 1` on the joint
that neither arm shows alone.  DISJOINT samples (per-hit + the J/psi-gun mass
term, `fisher_joint.py` summing by parameter NAME into
`fisherHJ_joint.npz`): `H` and `J` are additive exactly and there is nothing to
over-count.

The mass term's own `H` and `J` are MEASURED and cached in `$R/mass_HJ.npz`:
24 000 J/psi-gun candidates, 498 559 group rows (27 045 pruned at
`--prune-frac 0.001`), 42 material parameters, **NLL(0) = -47 431.190018** —
identical to the `ph_mass` card's, so it is the same object the fits use.  `H`
= 42 HVPs in **304 s**; `J` = 200 batch means in 6 s with
`|sum_m g_m - g| = 6.9e-9`.  `--mass-cache` reuses it.

## Defects found and fixed

1. **Do not assemble `G = V_mm - F_m C F_m^T` and factorize it.**  `G` is a
   Schur complement: its five null directions are a DIFFERENCE of nearly equal
   numbers, and `C = (F^T V^-1 F)^-1` is badly conditioned because a thin layer
   gives its kink block very small process noise.  MEASURED on 200 tracks: the
   null eigenvalues came out at ~1e-9 of the diagonal instead of 1e-16, the LDL
   kept a spurious SIXTH component on 34/60 tracks (`d = n_meas - 4`), and on
   exactly those tracks `sum_b v^(k)_b` missed 1 by up to a factor 19 — always
   on the LAST component.  One step of iterative refinement made it worse
   (`vchk` p99 0.69 -> 215) and was removed.  **FIX: the orthogonal projection
   in the standardized space** (the model section); the rank gap improves from
   4.5e7 to 1.14e15.
2. **The ionization/radiative sign.**  Reference component 0 IS the q/p
   functional, so comparing it to the validated `cfqop_*` distinguishes the
   candidate conventions by their worst-track relative difference on the
   IMAGINARY families (the real ones are even and agree at 1.2e-7 under all
   three):

   | candidate | worst rel |
   |---|---|
   | `sign(W[r0,k])` | **2.0** — i.e. the opposite sign, on every track |
   | `-sign(W[r0,k])` | 5.3e-2 |
   | **`-sign(W_b . u)`**, `u` = the rank-1 direction of `dQI` oriented by its qop component | **1.2e-7** |

   Two separate effects: the process-noise constraint row is built as
   (propagated - state), which is one global minus; and `dQI` is
   `e_0 e_0^T sigma^2` in the CURVILINEAR frame while `Hm` spreads it over all
   five local rows, so the signed coefficient is `W_b . u`, not `W[r0,k]`.
3. **Two indexing traps in `gates.py`** (same family, both fixed).  (a) Gates 1c
   and 3 summed the WHOLE `phresz`, which carries the five truth-referenced
   components AFTER the `d` per-hit ones; that added ~5 to a chi2 of ~13 and
   made gate 1c read a spurious 0.36 median, and polluted the per-k variance
   table above `k = d`.  Both now slice `z[:d]`.  (b) Gate 4 indexed
   `phresz[k]` without checking `k < phres_d`, so on a short track slot 24 is a
   TRUTH-REFERENCED component — correlated with the pull it was built from by
   construction — and the table read up to +0.38.  The per-track algebraic
   check (`|sum_b A_b[k].A_b[qp]|`) is what caught it.
4. **`saturation.py` returned `N_sat = 1e-27`.**  It used the MARGINAL
   prior-free sandwich `psd_inv(H) J psd_inv(H)`, and with 15-16 of the 42
   groups carrying no information of their own the pseudo-inverse dropped their
   directions and returned `sigma = 0`.  It now uses the STANDALONE form
   `sqrt(J_pp)/H_pp` (what `efficiency.py` calls `aa`, well defined for every
   parameter and exactly `1/sqrt(N)`-scaling) and reports only parameters with
   `H_pp > 1e-8 max(H_pp)`.
5. **The prior a recovery is corrected by is the group's own.**  A card is
   whitened, so the injection 0.00243951 of card value IS `k = 0.0487902` and
   the tier prior is 0.05 in `k`, not 1.0.  With a prior of 1.0 the shrinkage
   factor comes out `f = 1.000` and the RAW, prior-shrunk shift (-0.17) is
   quoted as the recovery.  `recovery.py` now takes the prior from the groups
   file and reports in physical `k`, with no flag to get wrong.
6. **`tails.py` grouping on a per-hit file.**  `mean_density` takes ROW
   INDICES; passing a component index straight through broke it.  And a
   truth-referenced component sits at slot `d + j`, so grouping by the raw
   `comp` value mixes q/p with z0 — it must be ranked within the track
   (`--group-by refparm`).
7. **`genParms[4]` and the `phi` branch cut** (the prototype's two maker
   defects) are FIXED in this maker.  The five truth-referenced pull variances
   are now 0.98 / 1.01 / 0.94 / 1.21 / 1.26 on the smoke and z0 is a usable
   observable for the first time (the prototype had `Var(z_4) = 4.0e6`).
8. **One unit convention, no unit flags** (standing rule, shared with the
   prototype).  Every `H`, `J`, `G`, fitted value and error that LEAVES a term
   is in physical units — `k`, the log material amount, and `eps`, the linear
   hit-variance scale — with the factor read off the object that carries it
   (`matres/groups.py`).  So the prior `efficiency.py` applies is the group's
   parmtype-15 tier prior itself, and every number in this file uses it.  A
   prior 20x too tight collapses `(H+P)^-1` onto the prior and the
   informativeness test `sq < 0.98 pv` then rejects every material group — an
   empty material table is the signature of a mismatched prior.

## Open items

None blocking; each is a new study.

1. **`z0` is not described.**  `Var(z_4) = 1.928` (1.39x in sigma against
   1.04-1.10 for the other four) and its 4-sigma tail is 26x the CF.  It is the
   most weakly conditioned Cholesky component so part of it is amplification,
   but a factor 1.9 in variance is too large for that alone.  The observable
   did not exist before the `genParms[4]` fix, so nothing was known about it:
   it says the fit's `z0` error is underestimated and that whatever does it is
   NOT in the resolution model.
2. **A non-Gaussian HIT model.**  Both arms treat the hit noise as exactly
   Gaussian; the data have 5.1e-4 beyond 4 sigma against a CF 1.5e-4.  It would
   enter the CF machinery as one more family.
3. **The residual MEAN, `D = W^T J`** — the one piece a DATA fit still needs
   for alignment and field.  On MC with ideal geometry and the simulation's own
   field the mean of `z` is zero and the term is pure width/shape; on data a
   shift `da` in a global parameter moves the constraint residual by `J da`
   (`Jfull`, already built in the maker), so `dz_k = W[:,k]^T J da`.  Its
   columns are not `npars` wide in practice (field modes and material groups
   collapse to 50 + 42 global columns, and the alignment block touches only the
   track's own ~17 modules at 5 columns each), so ~19 x 177 float32 =
   **13 kB/track**, 0.6 TB at 41 M.  NOT justified by the saturation numbers
   unless the CF term is wanted for the means themselves.
4. **The concatenated-tau trick is not ported to the maker.**  `phcf_msec` is
   163 ms/component, ~2.1 s/track under 60-way contention; the prototype's
   `extract_res5.py` gets all components in one evaluator pass because every
   exponent primitive depends on `(w, tau)` through the product alone.  Worth
   porting for a 40 M-track production; it was not needed here.
