# globalfit — the CVH quadratic hit-chi2 term as a rabbit external-likelihood card

## Purpose

Turn the CVH global-correction quadratic term (the gradient `G` and Hessian `K`
the two-track fit exports over the global calibration parameters) into a rabbit
external-likelihood card, join it with the unbinned J/psi mass term through the
per-candidate Jacobian rows `D = dm/dtheta` so that BOTH functionals fit ONE set
of parameters, and certify the whole chain on MC, where the truth is "no
correction".

The premise, now quantified: the hit chi2 measures the material well and is
nearly blind to the overall momentum scale; the masses measure the scale and are
worse on the material. Neither alone is enough, and the joint fit beats both on
every parameter.

## The object

### Conventions (the load-bearing ones)

| what | value | source |
|---|---|---|
| `gradv` | `d(chi2)/dtheta` = **2 x** `d(NLL)/dtheta` (`grad = 2 J^T R r`) | `ResidualGlobalCorrectionMakerBase/G4e.cc` |
| `hesspackedv` | `d2(chi2)/dtheta2 = 2 J^T R J`, **row-major packed UPPER triangle** (`np.triu_indices(n)` order) | `G4e.cc:4758` |
| `hessfactorv` | `B` (`nRank x nParms`, row-major), `H = B^T B` = the **chi2** Hessian, factor 2 inside; `nRank = min(ndof, nParms)` | `TwoTrackG4e.cc:4776` |
| `hessvaridxv` / `hessvarpackedv` | the variance (log-det) block, shipped SEPARATELY on an `exportVarianceGrads` production; `hessfactorv` does NOT contain it, `hesspackedv` does | `extract.py` docstring |
| card storage | `g = G/2`, `H = K/2` (rabbit's external term is in NLL units) -> same minimum `-K^-1 G`, same covariance `2 K^-1` | `make_global_term.py` |
| `Jpsi_jacMass` | `dm/dtheta`, 1 x nParms, on the same `globalidxv` columns; **this is `D`**. Two-track output has NO `jacrefv` (single-track only) | `TwoTrackG4e.cc:4550-4564` |
| **`D` sign in the card** | `D_card = -dm/dtheta`. `MassCFTerm` evaluates at `delta_i = mobs_i - shift - (D_card theta)_i`, i.e. treats its rows as `d(predicted mass)/dtheta`; here it is the *reconstructed* mass that moves with theta | `unbinned.py:_chunk_li` |
| **injection sign** | a "true correction `dtheta`" means quadratic `G -> G - K dtheta` AND mass `mobs -> mobs + D_card dtheta` (= `mobs - dm/dtheta dtheta`). Both then prefer `theta_base + dtheta` | derived, tested in `tests/test_global_term.py::test_injection` |
| parmtype 14 units | coefficient of the scalar potential, **T cm**. Mode 0 = (l=1, m=0), uniform Bz: 1 unit = `dBz = 1/319.9556 = 3.125e-3 T`, i.e. `dB/B = 8.2e-4`, so `dB/B = 1e-4` <-> `bfield_mode0 = 0.12195`. **There is no l=0 mode** | `ScalarPot3DEval`, `polyfit3d_full_coeffs_lmax18_custom50.txt` |
| parmtype 15 units | dimensionless log energy-loss scale, `dE = exp(k_g) dE_0`, `k_g > 0` = more loss | `G4ErrorEnergyLossForCVH.cc:115` |
| naming | `bfield_mode<modeIdx>`, `material_<groupName>` (a repeated group name gets its index appended — `bpix_support` occurs twice in `materialGroups50.txt`), else `glob_t<parmtype>_<subIdx>` | `make_global_term.name_params` |
| catalog | global index order = ascending `(parmtype, rawId)`; for 14/15 `rawdetid` **is** the mode / group index, so the blocks are contiguous and ordered | `Base.cc:725-796, 903` |
| priors | none on the field modes unless `--field-prior`; material from column 10 of the groups file x `--material-prior-scale`; declared through the ParamModel prior mechanism (`0.5 ((p-mu)/sigma)^2`), identical to `+2/sigma^2` on the chi2 Hessian | mirrors `fit_global_grads.py` |
| alpha | **removed by default**. The momentum scale is `bfield_mode0` + the material groups; `--with-alpha` restores it for the standalone comparison against the step-1/step-2 mass fits, where it is nearly degenerate with mode 0 | `make_global_term.py` |
| no double counting | the productions run `doMassConstraint=False`, so `gradv`/`hesspackedv` come from the *unconstrained* ditrack fit and adding a mass likelihood on top does not re-use the mass constraint | `run_local_trackres.sh` |

### Whitening (`--whiten`, mandatory in practice)

The raw parmtype-14 coefficients are NOT comparable to each other: the basis is
`(R/r_scale)^(l-1)/r_scale`, so a high-`l` mode needs a huge coefficient to move
the field at all — one unit is 0.29e-3 to 26e-3 of relative field depending on
the mode (mode 0: 8.2e-4). Always quote them whitened. With `--whiten` a card
unit is **1 T of RMS `|dB|` in the tracker** (parmtype 14) or **1 group prior
sigma** (parmtype 15); the scale vector is stored in the `global_index_map`
bundle, so `theta_raw = theta_card / scale`.

This is not a nicety: the raw field block is **6.6e9**-conditioned against
**1.9e7** whitened, and `rabbit_fit.py` with trust-krylov did NOT converge on
the raw 50-mode card in > 9 min.

### Multi-stream inputs

A `numberOfThreads=N` production writes `task_XXXX/globalcor_0..N-1.root`.
`extract.py` lists them through `resolution/prodfiles.py`, so `--files` may name
stream 0, every stream, the production directory or an `@list.txt`, and
**`--ntasks` caps TASKS, not files**. The `runtree` catalog is built from ONE
file (the first existing stream of the first usable task) — every stream carries
a byte-identical copy of the 13 MB parameter map.

## How to run

```bash
RES=/work/submit/david_w/ZMass/calibration_studies/resolution
RUNS=$RES/runs/globalfit
GRP=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/data/materialGroups50.txt
cd $RES/globalfit
```

Quadratic-only, physical basis, outliers trimmed (the recipe that is used at
production scale):

```bash
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
python3 extract.py --files '<production>/task_*/globalcor_*.root' --parmtypes 14 15 \
        --no-mass --max-chi2-ndof 3 --max-hess 1e8 --max-grad 1e6 --max-dEref-p 0.01 \
        -j 10 -o $RUNS/extract_quadonly.npz
python3 diagnose_quadratic.py -i $RUNS/extract_quadonly.npz --groups $GRP
python3 solve_reference.py -i $RUNS/extract_quadonly.npz --parmtypes 14 \
        --groups $GRP --whiten -o $RUNS/reference.npz
deactivate

run_tf.sh python3 make_global_term.py -i $RUNS/extract_quadonly.npz \
        --parmtypes 14 --groups $GRP --no-mass --whiten -o $RUNS/cards/quad.hdf5
run_tf.sh bash -c 'PATH=/work/submit/david_w/ZMass/rabbit-vmass/bin:$PATH \
        rabbit_fit.py '$RUNS'/cards/quad.hdf5 -o '$RUNS'/fits/quad -t 0 --unblind \
        --minimizerMethod trust-exact --paramModel ExternalParams bundle:global_params'
run_tf.sh python3 compare_fit.py --fit $RUNS/fits/quad/fitresults.hdf5 --ref $RUNS/reference.npz
```

`run_validation.sh` runs the whole ladder (extract / reference / quad / mass /
joint / three injections), resumable, one step per argument.

**Recommended default for ANY production accumulation:**
`--max-chi2-ndof 3 --max-hess 1e8 --max-grad 1e6 --max-dEref-p 0.01`
(see Defects). **For a quadratic-only card use `--minimizerMethod
trust-exact`** — it is a 50x50 exact-Hessian problem and trust-krylov's CG
needs ~sqrt(cond) iterations.

### Scripts

| script | what |
|---|---|
| `extract.py` | one parallel streaming pass over a two-track production producing, from the SAME candidates: `G`, `K` (both storage formats + the variance block), the per-candidate mass-likelihood inputs (identical construction to `cf_mass_likelihood.build_pairs_tt`), the `D` rows from `Jpsi_jacMass`, and the FSR kernel samples `dm = Jpsigen_mass - m_ref`. CF primitives are imported lazily (~6 min cold) so `--no-mass` is fast |
| `make_global_term.py` | turns that npz into a rabbit datacard: external term + `MassCFTerm` with the sparse `D` rows, the `global_params` bundle for `ExternalParams`, and a `global_index_map` bundle (rabbit name <-> global index <-> parmtype <-> sub-index, prior sigmas, injected vector, JSON provenance). `--check-sign` prints the `dlnm/dtheta` diagnostics and the alpha <-> mode-0 mapping |
| `solve_reference.py` | the offline reference solve `theta = -(K + 2P)^-1 G`, `cov = 2 (K + 2P)^-1`, and an element-wise comparison against `fit_global_grads.py --save-info` |
| `compare_fit.py` | rabbit fit results against the reference or each other: NLL breakdown (external / unbinned / constraints), correlations, injection pull table |
| `diagnose_quadratic.py` | whitened eigen-analysis of the quadratic block: physical scales per parameter, eigenspectrum and near-null directions of `K`, the gradient in the eigenbasis, and the solve with/without priors and restricted to the field modes or the material groups alone |
| `run_validation.sh` | the ladder |

Outputs: `$RUNS/` (`extract_*.npz`, `reference_*.npz`, `diag_*.txt`,
`cards/`, `fits/`). No figures — this study produces tables; the diagnostics are
`diag_*.txt` and the `compare_fit.py` stdout.

At production scale the same entry point builds the full-scale cards, e.g.
`runs/quad_dyv2.npz` (DY quadratic term, 380 tasks, 3 751 687 candidates).

## Results

### rabbit — all six unit checks PASS

`/work/submit/david_w/ZMass/rabbit-vmass`, branch `vmass-conditioning`. The
five commits this section was written against — `371954b` `ExternalParams`,
`a3e1065` the XLA empty-block guard, `6067d74` the joint test + CLAUDE.md
section, `44c9142` the freeze/injection fixes, `4bb47e6` the quadratic-only
recipe — were carried on the since-deleted `global-term-card` and are now
ancestors of the working branch.
`OMP_NUM_THREADS=8`, ~8 min:

| check | result |
|---|---|
| 1 quadratic only (`ExternalParams`) vs `-H^-1 g` | PASS, values and errors to **2e-17 abs / 5e-16 rel** |
| 2 unbinned mass term alone vs the analytic Gaussian NLL | PASS, **3.7e-16** relative |
| 3 joint fit vs the closed form (linear solve inside a 1D minimization over `k`) | PASS, values **< 7e-9**, errors **< 5e-9** relative |
| 4 Gaussian priors on shared + external-only parameters | PASS, **< 5e-8** |
| 5 frozen shared parameter | PASS (frozen value exact, profiled rest to **6e-8**) |
| 6 injection into both terms | PASS, recovery pulls **1e-10** |

### The accumulation is exact

* `extract.py` vs `fit_global_grads.py` on 4 files / 10 381 candidates:
  `max|dG|/scale = 8.5e-16`, `max|dK|/scale = 4.6e-15` (float64 summation
  order), solved `theta` agreeing to ~4e-8 relative.
* Vectorised packed-Hessian unpack (`np.triu_indices`) vs the reference row
  loop: **bit-identical (0.0)**.
* Multi-stream reader, validated on 20 tasks of `dymc_8p5M_260906_v2`: `G` and
  the factored Hessian (with the `hessvar*` block) agree with the per-file sum
  to **0**, and on the single-stream `jpsimc_20M_260905` the output is identical
  to the pre-change reader.

### Outlier trimming — the sanity gate, and it passes

`btojpsix v3 260904f`, 128 707 candidates, 50 field modes + 42 material groups,
in the **whitened** basis of `diagnose_quadratic.py`:

| quantity | no cut | chi2/ndof < 10 | chi2/ndof < 3 |
|---|---|---|---|
| candidates removed | 0 | 29 (0.023 %) | 1084 (0.84 %) |
| lambda_max of whitened `K` | 4.16e18 | 3.44e9 | 3.43e9 |
| field-block condition number | 1.40e16 | 1.88e7 | 1.89e7 |
| field-block rank (of 50) | 42 | **50** | **50** |
| max abs pull (92 params, material priors) | 1365 | **2.6** | **2.5** |
| rms pull | 369 | **1.2** | **1.2** |
| chi2 improvement of the 92-param fit | 1.10e7 | 9.30e2 | 9.39e2 |
| implied \|dB\|rms, no field prior | 44 700 mT | 475 mT | 432 mT |
| implied \|dB\|rms, prior \|dB\|/B < 1e-3 per mode | 3185 mT | 4.8 mT | 4.1 mT |

**With the trimming, the hit-chi2 block is consistent with "no correction":
rms pull 1.2, max 2.5 over 92 parameters.**

Ideal-geometry control, `resolution_trackres_jpsigun_ul16_260904f_m0`
(298 749 candidates):

| gun | chi2/ndof < 3 | + hessmax < 1e8, gradmax < 1e6 |
|---|---|---|
| extra candidates removed | — | 3 |
| lambda_max of whitened `K` | 3.14e20 | **9.77e9** |
| field-block rank (of 50) | 26 | **50** |
| field-block condition number | 3.77e16 | **8.4e6** |
| max abs pull / rms pull | 1.5 / 0.3 | 8.4 / 1.5 |
| Delta chi2 (92 params) | 12 | 450 |

The pulls *rise* because with rank 26 the pseudo-inverse was zeroing most
directions; the full 50-dimensional field block is only actually measured after
the guard. That residual is now understood — see Defects, item 4.

### Validation (a): the quadratic term through rabbit

`btojpsix v3 260904f`, `chi2/ndof < 3`, 127 623 candidates, 50 field modes,
whitened card, no priors (the field block alone is full rank):

| | rabbit vs `solve_reference.py` |
|---|---|
| max abs error difference / error | **1.7e-10** |
| max abs value difference / error (trust-exact) | **5.8e-10** -> PASS at 1e-6 |
| max abs value difference / error (trust-krylov) | 1.4e-3, i.e. the minimizer's own tolerance |
| fit time, trust-krylov | 56 s |
| fit time, trust-exact | **7.3 s** |
| fit time, RAW (unwhitened) card | did not converge in > 9 min |

So the card, the chi2 -> NLL factor of 2 and the `ExternalParams` declaration
are right: the covariance is reproduced exactly and the minimum to the
minimizer's own tolerance.

### Validation (c): injection into the quadratic term

`--inject bfield_mode0:3.8114e-4 --inject-quad-only` on the whitened card
(3.8114e-4 T = `dB/B` of 1e-4 on the uniform-Bz mode), refit with trust-exact
against the baseline fit:

| quantity | value |
|---|---|
| recovered shift on `bfield_mode0` | 3.8114000003e-4 T |
| (recovered - injected)/injected | **8.6e-11** |
| pull | **3.2e-13** |
| max abs shift on the 49 non-injected modes | **2.3e-14** T |

Exactly what `theta' = -K^-1 (G - K d) = theta_base + d` requires, with no
leakage, so the injection convention `G -> G - K dtheta` is right.

### Validation (b): the mass term alone reproduces the standalone fit

Card built with `rabbit/tests/make_unbinned_mass_tensor.py --model r` from the
matched caches `cf_masspairs_btojpsix_v3_260903x_m0.npz` +
`cf_masskernel_btojpsix_v3_260903x_m0.npz` (128 687 candidates, families
ms / ioni / rad, 760 MB card, 15.5 s to write), fitted with
`--paramModel UnbinnedParams` (330 s incl. covariance), against
`cf_masslik_fit.py`'s `runs/masslikfit_btojpsix_v3_260903x_r.npz`:

| parameter | rabbit | standalone | (d)/err | err rabbit | err standalone | (derr)/err |
|---|---|---|---|---|---|---|
| `alpha` | -0.046255694 | -0.046255713 | **7.4e-7** | 0.025546150 | 0.025546150 | -7.3e-9 |
| `r` | 0.981822128 | 0.981822146 | **-3.9e-6** | 0.004487830 | 0.004487830 | -3.0e-8 |

`nllvalreduced` = -255177.48034457705 against -255177.48034457708, i.e.
**3e-11 absolute** (the float64 last digit).

### The division of labour, measured

The forecast from the numbers above (`bfield_mode0` is the l=1, m=0 uniform-Bz
mode, i.e. the momentum scale — `dm/m = -dB/B` exactly, both legs scale
together; one whitened unit is 1 T of uniform dBz = `dB/B` of 0.2624):

| constraint on `bfield_mode0` | sigma |
|---|---|
| hit chi2 (btojpsix, 127 623 cand, whitened, no priors) | **0.102 T** = 27e-3 relative |
| J/psi masses (128 687 cand, `sigma_m` median 31.3 MeV, Gaussian limit) | **9.2e-5 T** = 2.4e-5 relative |
| ratio | **~1100x** |

and the delivered joint fit (the `matres` `MaterialCFTerm` card: the quadratic
term over 299 069 gun candidates + a mass term over 24 000 of them on the SAME
92 parameters, rabbit trust-krylov, **EDM 9.4e-11**, 2467 s):

| sigma of ... | quadratic only | mass only | **joint** |
|---|---|---|---|
| `bfield_mode0` | 0.0339 T = `dB/B` 8.9e-3 | 3.90e-4 T = `dB/B` 1.02e-4 | **2.50e-4 T = `dB/B` 0.66e-4** |
| `material_tib_support` sigma(k) | 0.01543 | 0.03293 | **0.01203** |
| `material_tec_structure` | 0.02929 | 0.03501 | **0.02421** |
| `material_tob_support` | 0.01982 | 0.04089 | **0.01769** |
| `material_bpix_support` (6) | 0.04477 | 0.04259 | **0.03806** |
| `material_tibtid_services` | 0.04359 | 0.07883 | **0.03999** |
| every ACTIVE-silicon group | 0.0199 (its prior) | 0.0200 | 0.0199 |

The masses measure the scale **87x better than the hit chi2 does on 12x the
candidates**, the hit chi2 measures the material 1.0-2.1x better than the
masses, and the joint fit beats BOTH on every parameter — 1.55x better than the
masses alone on the scale, because the quadratic term tightens the material
nuisances the scale is correlated with. `bfield_mode0` moves from +1.36 sigma
(quadratic only) to -0.30 sigma (joint), i.e. onto zero, as it must on MC.

### Speed and scale

* Quadratic-only extraction: 48 files / 128 707 candidates in **137 s**
  (8 workers); 160 files / 300 k gun candidates in **167 s** (6 workers).
  With the CF (the radiative exponent dominates at ~0.6 s/candidate) it is
  ~75 min for 48 files with 24 workers.
* rabbit on a 50-parameter dense quadratic card: **7.3 s** (trust-exact), 56 s
  (trust-krylov), no convergence unwhitened.
* The mass term's per-candidate arrays dominate the card: 448 t-points x 5
  families x 4 B = **9.0 kB/candidate** (1.2 GB at 128 k). The
  `cfcompress` study reduces this to 64 B/candidate at 140x.

## Defects found and fixed

1. **0.02 % of the candidates carried the entire "1e2-1e3 unit corrections with
   300-1000 sigma pulls".** The two-track fits are healthy in the bulk (median
   `chisqval/ndof` = 0.95 on btojpsix v3 260904f, 0.91 on the gun) but the tail
   runs to 4.6e8, and **~0.02 % of candidates carry ~99.997 % of the summed
   chi2**. `G` and `K` are sums over all candidates, so without trimming the
   "global fit" is the fit of a handful of runaway candidates. Fixed by
   `--max-chi2-ndof`; `chi2ndof` is always stored per candidate so the cut can
   also be applied at card-writing time (`make_global_term.py --max-chi2-ndof`).
   Without the cut the summed Hessian is **not even positive semi-definite** on
   the gun (a whitened eigenvalue of -2.5e6).
2. **A pathological Jacobian with an acceptable chi2.** After the chi2 cut the
   gun's summed Hessian still had a spurious eigenvalue of 3.1e20 and rank
   26/50 on the field block: **three** candidates out of 298 749 with `hessmax`
   up to 3.4e14 against a p99.999 of 7.0e7. Fixed by `--max-hess` / `--max-grad`.
3. **The raw parmtype-14 basis does not converge.** See Whitening above; fixed
   by `--whiten` in `make_global_term.py` and `solve_reference.py`.
4. **The residual few-sigma structure is a mean-energy-loss bias, not a field
   or material one.** The parmtype-15 derivative of the quadratic term is
   mean-loss only, and it is biased above `dE_ref/p ~ 0.03`; at
   `dE_ref/p < 0.01` the material block closes on MC (max pull 0.96, rms 0.19)
   and agrees with the mass term on `tec_services` to 0.0008 in `k`. Fixed by
   `--max-dEref-p 0.01`, now used at production scale. Full derivation in
   `resolution/qmsmodel/README.md`; the error to quote on a material group is
   the **sandwich**, not `2 K^-1`.
5. **Three material groups carry (essentially) zero information** from J/psi
   tracks and make the 92-parameter solve singular: `material_pp1_cables`
   (exactly zero — no track reaches it), `material_beampipe`
   (`lambda/lambda_max ~ 3e-18`) and `material_thermal_screen` (~6e-15,
   degenerate with `material_support_tube`). `solve_reference.py` /
   `diagnose_quadratic.py` fall back to a pseudo-inverse and name them; they
   must be frozen or priored in a production fit. The mass functional
   independently finds the same three unmeasurable, which cross-checks the group
   definition.
6. **rabbit: a frozen parameter was not frozen inside an external term.**
   `_compute_external_nll` read `self.x` instead of `self.get_x()`, which
   silently breaks any scan over a parameter shared with an unbinned term
   (`371954b`).
7. **rabbit: a card with `npoi == 0` and/or `nsyst == 0` failed silently.**
   `get_x()` slices a length-0 piece out of `x` and `tf.where`s over it; XLA has
   no gradient kernel for either ("Scatter dimension 0 is of size zero" /
   `StridedSliceGrad`), every jit-compiled loss+gradient call fails, and
   `Fitter.fit()` swallows the exception, logs one warning and returns the
   STARTING POINT — so the fit "succeeds" with every parameter still at its
   default. `jit_compile` is now disabled for that case (`a3e1065`).
8. **`extract.py` refused a SINGLE-track production even under `--no-mass`**,
   because the `Jpsi_jacMass` guard sat in front of the mass branch instead of
   inside it. Fixed; the quadratic term over 318 390 mu-gun tracks now builds in
   45 s.

## Open items

* **`cf_global_masslik.py` injects `dm0 += D dtheta`, the OPPOSITE sign** to the
  convention derived and tested here. Its injection closure must be re-checked.
* **The three zero-information material groups** must be frozen or priored in
  any production fit rather than left to the pseudo-inverse.
* **The sparse `D` layout is a pessimisation.** `jac_indices` is
  `int64 (nnz, 2)` + `float64` values = 24 B/nnz and every candidate touches
  *all* 92 field+material parameters, so it costs 2.2 kB/candidate against
  368 B for a dense `(n, 92) float32`. A dense-`D` option in `MassCFTerm` is
  worth having before 1e7 candidates (22 GB -> 3.7 GB). (Storing `D` dense was
  separately found to be required on GPU: `tf.sparse.sparse_dense_matmul` has
  no deterministic kernel.)
* **Alignment scale (108 k parameters) has never been exercised.** The summed
  Hessian is sparse (module pairs that share a track); rabbit's own docstring
  quotes a 329M-nnz external Hessian as a case its CSR `sm.matmul` path already
  handles, so ~5 GB of nnz plus the CSR build is the cost. `extract.py` has a
  scipy COO->CSR accumulation path and `make_global_term.py` a `wums.SparseHist`
  storage path above `--dense-max`; **neither has been run at that size**.
* **The exported `D` rows are built with the *expected* (Fisher) curvature.**
  The per-candidate value scatters by ~6 % about an exact refit
  (`Documents/Resolution/NOTES_EXPORTS.md` sec. 8 item 4), which is zero-mean
  and averages away for aggregate quantities but is a genuine per-candidate
  error here, because the `D` rows are used *per candidate*. For parmtype 15
  (and 7) the same note reports a *coherent* +35..53 % Jacobian bias: treat the
  material columns of `D` as indicative until an exact export exists.
* `gradchisqv` is empty in every two-track file, so the censored-likelihood
  trimming of `fit_global_grads.py --censor-cut` is not available here.
