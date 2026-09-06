# globalfit — state (2026-09-04, evening)

Goal: the CVH global-correction quadratic term (gradient + Hessian over the
global parameters) as a rabbit external-likelihood card, joined with the
unbinned J/psi mass term through the per-candidate Jacobian rows, and a first
combined fit on MC.

Two trees are touched:

* `/work/submit/david_w/ZMass/rabbit`, branch **`global-term-card`** (5 commits
  on `unbinned-mass-term`, +830/-2, nothing pushed)
* this directory (new files only; `calibration_studies` is NOT to be committed)

---

## Conventions pinned (the load-bearing ones)

| what | value | source |
|---|---|---|
| `gradv` | `d(chi2)/dtheta` = **2 x** `d(NLL)/dtheta` (`grad = 2 J^T R r`) | `ResidualGlobalCorrectionMakerBase/G4e.cc` |
| `hesspackedv` | `d2(chi2)/dtheta2 = 2 J^T R J`, **row-major packed UPPER triangle** (`np.triu_indices(n)` order) | G4e.cc:4758 |
| `hessfactorv` | `B` (nRank x nParms, row-major), `H = B^T B` = the **chi2** Hessian, factor 2 inside; `nRank = min(ndof, nParms)` | TwoTrackG4e.cc:4776 |
| card storage | `g = G/2`, `H = K/2` (rabbit's external term is in NLL units) -> same minimum `-K^-1 G`, same covariance `2 K^-1` | `make_global_term.py` |
| `Jpsi_jacMass` | `dm/dtheta`, 1 x nParms, on the same `globalidxv` columns; **this is `D`**. Two-track output has NO `jacrefv` (that is single-track only). | TwoTrackG4e.cc:4550-4564 |
| **D sign in the card** | `D_card = -dm/dtheta`. `MassCFTerm` evaluates at `delta_i = mobs_i - shift - (D_card theta)_i` i.e. treats its rows as `d(predicted mass)/dtheta`; here it is the *reconstructed* mass that moves with theta. | `unbinned.py:_chunk_li` |
| **injection sign** | "true correction dtheta" => quadratic `G -> G - K dtheta` AND mass `mobs -> mobs + D_card dtheta` (= `mobs - dm/dtheta dtheta`). Both then prefer `theta_base + dtheta`. NB `cf_global_masslik.py` injects `dm0 += D dtheta`, which is the OPPOSITE sign; its injection closure should be re-checked. | derived, tested in `tests/test_global_term.py::test_injection` |
| parmtype 14 units | coefficient of the scalar potential, **T cm**. Mode 0 = (l=1,m=0), uniform Bz: 1 unit = `dBz = 1/319.9556 = 3.125e-3 T`, i.e. `dB/B = 8.2e-4`. So `dB/B = 1e-4` <-> `bfield_mode0 = 0.12195`. There is no l=0 mode. | `ScalarPot3DEval`, `polyfit3d_full_coeffs_lmax18_custom50.txt` |
| parmtype 15 units | dimensionless log energy-loss scale, `dE = exp(k_g) dE_0`, `k_g > 0` = more loss | `G4ErrorEnergyLossForCVH.cc:115` |
| naming | `bfield_mode<modeIdx>`, `material_<groupName>` (a repeated group name gets its index appended — `bpix_support` occurs twice in `materialGroups50.txt`), else `glob_t<parmtype>_<subIdx>`. `fit_global_grads.py` uses the bare group name, so only the 50 field modes match by name between the two. | `make_global_term.name_params` |
| catalog | global index order = ascending `(parmtype, rawId)`; for 14/15 `rawdetid` **is** the mode / group index, so the blocks are contiguous and ordered | Base.cc:725-796, 903 |
| priors | none on the field modes unless `--field-prior`; material from column 10 of the groups file x `--material-prior-scale`; declared through the ParamModel prior mechanism (`0.5 ((p-mu)/sigma)^2`), identical to `+2/sigma^2` on the chi2 Hessian | mirrors `fit_global_grads.py` |
| alpha | **removed by default**. The momentum scale is `bfield_mode0` + the material groups; `--with-alpha` restores it for the standalone comparison against the step-1/step-2 mass fits, where it is nearly degenerate with mode 0. | `make_global_term.py` docstring |
| no double counting | the productions run `doMassConstraint=False`, so `gradv`/`hesspackedv` come from the *unconstrained* ditrack fit and adding a mass likelihood on top does not re-use the mass constraint. The exact factorisation write-up is still TODO. | `run_local_trackres.sh` |

---

## Done

### rabbit branch `global-term-card`

1. `371954b` **`rabbit/param_models/external_params.py`** — `ExternalParams`, a
   param model that declares fit parameters (name, default, Gaussian prior,
   POI flag) from an `auxiliary` bundle, so a card whose only free parameters
   come from an external term can exist. Registered in
   `param_models/helpers.py`. Same commit moves `_compute_external_nll` from
   `self.x` to `self.get_x()` so a frozen parameter is actually frozen inside
   an external term (it was not, which silently breaks any scan over a
   parameter shared with an unbinned term).
2. `a3e1065` **XLA guard** — a card with `npoi == 0` (nothing is a signal
   strength) and/or `nsyst == 0` makes `get_x()` slice a length-0 piece out of
   `x` and `tf.where` over it; XLA has no gradient kernel for either
   ("Scatter dimension 0 is of size zero" / `StridedSliceGrad`) and every
   jit-compiled loss+gradient call fails. `Fitter.fit()` swallows the
   exception, logs one warning and returns the starting point, so the fit
   "succeeds" with every parameter still at its default — this cost about an
   hour to find. Now `jit_compile` is disabled for that case, like the sparse
   and unbinned cases.
3. `6067d74` **`tests/test_global_term.py`** — the joint test. A quadratic
   external term and a *Gaussian-equivalent* unbinned mass term (single gauss
   family, no kernel CF, no background, no floor => `L_i = N(delta_i; 0,
   k sigma_i^2)`) over the same parameters, joined only by name through the
   sparse `m_i(theta) = m_i^0 + D_i theta` rows. The whole joint problem then
   has a closed form (linear solve inside a 1D minimization over `k`), so the
   fit is checked against it exactly. Plus a CLAUDE.md section on external
   terms / `ExternalParams`.
4. `44c9142` fixes to the freeze (trust-krylov) and injection checks; the
   suite is green end to end (~8 min at `OMP_NUM_THREADS=8`).
5. `4bb47e6` documents how to fit a quadratic-only card (trust-exact, and
   scale the parameters before writing H).

**Test status — ALL SIX PASS** (`OMP_NUM_THREADS=8`, ~8 min):

| check | result |
|---|---|
| 1 quadratic only (ExternalParams) vs `-H^-1 g` | PASS, values and errors to **2e-17 abs / 5e-16 rel** |
| 2 unbinned mass term alone vs the analytic Gaussian NLL | PASS, **3.7e-16** relative |
| 3 joint fit vs the closed form | PASS, values **<7e-9**, errors **<5e-9** relative |
| 4 Gaussian priors on shared + external-only parameters | PASS, **<5e-8** |
| 5 frozen shared parameter | PASS (frozen value exact, profiled rest to **6e-8**) |
| 6 injection into both terms | PASS, recovery pulls **1e-10** |

### this directory

* **`extract.py`** — one parallel pass over a two-track production producing
  everything from the *same* candidates: `G`, `K` (both storage formats), the
  per-candidate mass-likelihood inputs (identical construction to
  `cf_mass_likelihood.build_pairs_tt`: same `IONI_SGN`/`RAD_SGN`, `ioni_sq2`,
  `TG`), the `D` rows from `Jpsi_jacMass`, and the FSR kernel samples
  `dm = Jpsigen_mass - m_ref` of the selected candidates. CF primitives are
  imported lazily (they cost ~9 min cold / ~2 min warm) so `--no-mass` is fast.
* **`make_global_term.py`** — turns that npz into a rabbit datacard: external
  term + `MassCFTerm` with the sparse `D` rows, the `global_params` bundle for
  `ExternalParams`, and a `global_index_map` auxiliary bundle (rabbit name <->
  global index <-> parmtype <-> sub-index, prior sigmas, injected vector, JSON
  provenance). `--check-sign` prints the `dlnm/dtheta` diagnostics and the
  alpha <-> mode-0 mapping.
* **`solve_reference.py`** — the offline reference solve
  (`theta = -(K+2P)^-1 G`, `cov = 2 (K+2P)^-1`) and an element-wise comparison
  against `fit_global_grads.py --save-info`.
* **`compare_fit.py`** — compares rabbit fit results against the reference or
  each other, prints the NLL breakdown (external / unbinned / constraints) and
  correlations, and does the injection pull table.
* **`diagnose_quadratic.py`** — the whitened eigen-analysis of the quadratic
  block: physical scales per parameter, eigenspectrum and near-null directions
  of `K`, the gradient in the eigenbasis, and the solve with/without priors and
  restricted to the field modes or the material groups alone.
* **`run_validation.sh`** — the whole ladder (extract / reference / quad /
  mass / joint / three injections), resumable, one step per argument.

### Numbers measured so far

* **Accumulation is exact**: on 4 files / 10381 candidates, `extract.py` vs
  `fit_global_grads.py`: `max|dG|/scale = 8.5e-16`, `max|dK|/scale = 4.6e-15`
  (float64 summation order), solved `theta` agreeing to ~4e-8 relative.
* Vectorised packed-Hessian unpack (`np.triu_indices`) vs the reference
  row loop: **bit-identical** (0.0).
* **Speed**: quadratic-only extraction of the full 48-file production
  (128,707 candidates, 92 parameters) = **137 s** with 8 workers.
  With the CF (the radiative exponent dominates, ~0.4 s/candidate) it is
  ~20 min/file; 48 files with 24 workers ~ 45 min. Card/fit not yet timed.
* Production used: `resolution_trackres_btojpsix_v3_260904f_m0`
  (48 tasks x `globalcor_0.root`, two-track, `hesspackedv`, nglobal 126452,
  parmtype 14 x50 + 15 x42, both dense in every candidate's `globalidxv`).
* **Multi-stream inputs (from 2026-09-06).** A `numberOfThreads=N` production
  writes `task_XXXX/globalcor_0..N-1.root`; `extract.py` lists them through
  `resolution/prodfiles.py`, so `--files` may name stream 0, every stream, the
  production directory or an `@list.txt`, and **`--ntasks` caps TASKS, not
  files**. The `runtree` catalog is still built from ONE file (`files[0]`, the
  first existing stream of the first usable task) -- every stream carries a
  byte-identical copy of the 13 MB parameter map and concatenating them would
  duplicate it N times. Validated on 20 tasks of `dymc_8p5M_260906_v2`: G and
  the factored Hessian (with the `hessvar*` block) agree with the per-file sum
  to 0, and on the single-stream `jpsimc_20M_260905` the output is identical to
  the pre-change reader.

### The "1e2-1e3 unit corrections with 300-1000 sigma pulls" — RESOLVED

**0.02 % of the candidates carried the entire effect.** The two-track fits are
healthy in the bulk (median `chisqval/ndof` = 0.95 on btojpsix v3 260904f, 0.91
on the gun) but the tail runs to 4.6e8, and **~0.02 % of candidates carry
~99.997 % of the summed chi2**. `gradv` and the Hessian are summed over all
candidates, so without a trimming the "global fit" is the fit of a handful of
runaway candidates. `fit_global_grads.py` has `--max-chi2-per-hit` /
`--censor-cut` for exactly this; `extract.py` had no cut. It does now
(`--max-chi2-ndof`, and it always stores `chi2ndof` per candidate so the cut can
also be applied at card-writing time via `make_global_term.py --max-chi2-ndof`).

btojpsix v3 260904f, 128,707 candidates, 50 field modes + 42 material groups,
all numbers in the **whitened** basis of `diagnose_quadratic.py` (each
parmtype-14 mode scaled by the RMS |dB| it produces in the tracker per unit
coefficient, so its value is Tesla; each group by its own prior sigma):

| quantity | no cut | chi2/ndof < 10 | chi2/ndof < 3 |
|---|---|---|---|
| candidates removed | 0 | 29 (0.023 %) | 1084 (0.84 %) |
| lambda_max of whitened K | 4.16e18 | 3.44e9 | 3.43e9 |
| field-block condition number | 1.40e16 | 1.88e7 | 1.89e7 |
| field-block rank (of 50) | 42 | **50** | **50** |
| max abs pull (92 params, material priors) | 1365 | **2.6** | **2.5** |
| rms pull | 369 | **1.2** | **1.2** |
| chi2 improvement of the 92-param fit | 1.10e7 | 9.30e2 | 9.39e2 |
| implied \|dB\|rms, no field prior | 44,700 mT | 475 mT | 432 mT |
| implied \|dB\|rms, prior \|dB\|/B < 1e-3 per mode | 3185 mT | 4.8 mT | 4.1 mT |

So on MC, with the trimming, **the hit-chi2 block is consistent with "no
correction": rms pull 1.2, max 2.5 over 92 parameters**. That is the sanity
gate, and it is passed.

Secondary findings from the same diagnostic:

* **Three material groups carry (essentially) zero information** from J/psi
  tracks and make the 92-parameter solve singular:
  `material_pp1_cables` (exactly zero -- no track reaches it),
  `material_beampipe` (lambda/lambda_max ~ 3e-18) and
  `material_thermal_screen` (~6e-15, degenerate with `material_support_tube`).
  They must be frozen or priored; `solve_reference.py` / `diagnose_quadratic.py`
  fall back to a pseudo-inverse and name them.
* **The softest field direction is `bfield_mode0`** (the l=1, m=0 uniform Bz,
  i.e. the momentum scale): lambda = 1.8e2 against lambda_max = 3.4e9, error
  0.18 T with no prior. The hit chi2 essentially does not constrain the overall
  scale — which is precisely the direction the mass term has to supply, so the
  architecture's premise is confirmed quantitatively.
* Without the cut the summed Hessian is **not even positive semi-definite** on
  the gun sample (a whitened eigenvalue of -2.5e6), another symptom of the same
  runaway candidates.
* The raw parmtype-14 coefficients are *not* comparable to each other: the
  basis is `(R/r_scale)^(l-1)/r_scale`, so a high-l mode needs a huge
  coefficient to move the field at all. Always quote them whitened. One unit
  is 0.29e-3 to 26e-3 of relative field depending on the mode (mode0 8.2e-4).

### Ideal-geometry control (the gun sample) and the whitening

`resolution_trackres_jpsigun_ul16_260904f_m0` (ideal geometry, default field),
298,749 candidates after `chi2/ndof < 3`: **max abs pull 1.5, rms 0.3, Delta
chi2 = 12 for 92 parameters, implied |dB|rms = 0.0006 mT**. Perfect closure --
with an ideal geometry the hit-chi2 block finds nothing, as it must on MC.
btojpsix v3 (the *real* geometry + Opera3D refit) sits at rms pull 1.2 /
Delta chi2 940, i.e. a small but real residual that the field+material block
absorbs because the alignment is frozen. That is expected and is the reason
the production calibration floats the 108k alignment parameters too.

**The `hessmax` guard (now implemented: `--max-hess`, `--max-grad`).** After
the chi2 cut the gun's summed Hessian still had a spurious eigenvalue of 3.1e20
and only rank 26/50 on the field block -- **three** candidates out of 298,749
with `hessmax` up to 3.4e14 against a p99.999 of 7.0e7, i.e. a pathological
Jacobian with a perfectly acceptable chi2. Adding `--max-hess 1e8
--max-grad 1e6` removes those 3 and:

| gun (298,749 cand) | chi2/ndof < 3 | + hessmax < 1e8, gradmax < 1e6 |
|---|---|---|
| extra candidates removed | - | 3 |
| lambda_max of whitened K | 3.14e20 | **9.77e9** |
| field-block rank (of 50) | 26 | **50** |
| field-block condition number | 3.77e16 | **8.4e6** |
| max abs pull / rms pull | 1.5 / 0.3 | 8.4 / 1.5 |
| Delta chi2 (92 params) | 12 | 450 |

The pulls *rise* because with rank 26 the pseudo-inverse was zeroing most
directions; the full 50-dimensional field block is only actually measured after
the guard. rms pull 1.5 with a 8.4 tail on an ideal-geometry sample is a real
but small residual (the hit-resolution parameters are frozen here), not the
1000-sigma pathology. **Recommended default for any production accumulation:
`--max-chi2-ndof 3 --max-hess 1e8 --max-grad 1e6`.**

**Whitening (`--whiten`, now in `make_global_term.py` and
`solve_reference.py`).** The raw parmtype-14 coefficients differ by orders of
magnitude in what they do to the field, and the raw field block is
**6.6e9**-conditioned against **1.9e7** whitened -- a factor 350. `rabbit_fit.py`
with trust-krylov did **not** converge on the raw 50-mode card in >9 min, so
the whitened basis is not a nicety. With `--whiten` a card unit is 1 T of RMS
|dB| in the tracker (parmtype 14) or 1 group prior sigma (parmtype 15); the
scale vector is stored in the `global_index_map` bundle, so `theta_raw =
theta_card / scale`.

### Validation (a): the quadratic term through rabbit

btojpsix v3 260904f, `chi2/ndof < 3`, 127,623 candidates, 50 field modes,
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
minimizer's own tolerance. **For a quadratic-only card use `--minimizerMethod
trust-exact`** (it is a 50 x 50 exact-Hessian problem; trust-krylov's CG needs
~sqrt(cond) iterations).

### Validation 5c: injection into the quadratic term

`--inject bfield_mode0:3.8114e-4 --inject-quad-only` on the whitened card
(3.8114e-4 T = dB/B of 1e-4 on the uniform-Bz mode), refit with trust-exact
and compared to the baseline fit:

| quantity | value |
|---|---|
| recovered shift on `bfield_mode0` | 3.8114000003e-4 T |
| (recovered - injected)/injected | **8.6e-11** |
| pull | **3.2e-13** |
| max abs shift on the 49 non-injected modes | **2.3e-14** T |

Exactly what the algebra requires (`theta' = -K^-1 (G - K d) = theta_base + d`,
no leakage), so the injection convention `G -> G - K dtheta` in
`make_global_term.py` / `solve_reference.py` is right.

### Validation (b): the mass term alone reproduces step 1/step 2

Card built with `rabbit/tests/make_unbinned_mass_tensor.py --model r` from the
matched caches `cf_masspairs_btojpsix_v3_260903x_m0.npz` +
`cf_masskernel_btojpsix_v3_260903x_m0.npz` (128,687 candidates, families
ms / ioni / rad, 760 MB card, 15.5 s to write), fitted with
`--paramModel UnbinnedParams` (330 s incl. covariance), against the standalone
`cf_masslik_fit.py` result `runs/masslikfit_btojpsix_v3_260903x_r.npz`:

| parameter | rabbit | step 1 | (d)/err | err rabbit | err step 1 | (derr)/err |
|---|---|---|---|---|---|---|
| `alpha` | -0.046255694 | -0.046255713 | **7.4e-7** | 0.025546150 | 0.025546150 | -7.3e-9 |
| `r` | 0.981822128 | 0.981822146 | **-3.9e-6** | 0.004487830 | 0.004487830 | -3.0e-8 |

`nllvalreduced` = -255177.48034457705 against -255177.48034457708, i.e. **3e-11
absolute** (the float64 last digit). So the unbinned path is exact on this
production too, independently of the quadratic-term work.

---

## Open / next commands

```bash
RUNS=/work/submit/david_w/ZMass/calibration_studies/resolution/runs/globalfit
GRP=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/data/materialGroups50.txt
SP=/tmp/.../scratchpad/globalfit        # the patched run_tf.sh + wums 0.2.0 live here
cd /work/submit/david_w/ZMass/calibration_studies/resolution/globalfit
```

**Recipe that works** (everything below was run and passed):

```bash
# quadratic-only, physical basis, outliers trimmed
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
python3 extract.py --files '<prod>/task_*/globalcor_*.root' --parmtypes 14 15     --no-mass --max-chi2-ndof 3 -j 10 -o $RUNS/extract_quadonly.npz     # ~140 s / 48 files
python3 diagnose_quadratic.py -i $RUNS/extract_quadonly.npz --groups $GRP
python3 solve_reference.py -i $RUNS/extract_quadonly.npz --parmtypes 14     --groups $GRP --whiten -o $RUNS/reference.npz
deactivate
$SP/run_tf.sh python3 make_global_term.py -i $RUNS/extract_quadonly.npz     --parmtypes 14 --groups $GRP --no-mass --whiten -o $RUNS/cards/quad.hdf5
$SP/run_tf.sh bash -c 'PATH=/work/submit/david_w/ZMass/rabbit/bin:$PATH     rabbit_fit.py '$RUNS'/cards/quad.hdf5 -o '$RUNS'/fits/quad -t 0 --unblind     --minimizerMethod trust-exact --paramModel ExternalParams bundle:global_params'
$SP/run_tf.sh python3 compare_fit.py --fit $RUNS/fits/quad/fitresults.hdf5 --ref $RUNS/reference.npz
```

### Still to do

1. **The CF extraction** (mass-likelihood inputs + `D` rows) was relaunched and
   takes **~75 min** for 48 files with 24 workers (the radiative exponent is
   ~0.6 s/candidate and dominates). It was started **without**
   `--max-chi2-ndof`, but `extract.py` now always stores `chi2ndof` per
   candidate, so the trimming can be applied at card-writing time with
   `make_global_term.py --max-chi2-ndof 3`. Check
   `$RUNS/extract_btojpsix_v3_260904f.npz`.
2. Repeat validation (b) through `make_global_term.py --no-quadratic --no-jac
   --with-alpha` once the `260904f` CF extraction is in (it was done above via
   the existing `260903x` caches and the stock converter, which is the
   independent check; doing it through the new writer additionally exercises
   that code path). Expect a different alpha: `260904f` has the 0.25 GeV
   momentum-floor clamp, `260903x` the 2.0 GeV one.
3. **Validation (c)**: the joint fit. The premise is already quantified from
   the numbers in hand. `bfield_mode0` is the uniform-Bz mode, i.e. the
   momentum scale (`dm/m = -dB/B` exactly, both legs scale together), and one
   whitened unit is 1 T of uniform dBz = `dB/B` of 0.2624.

   | constraint on `bfield_mode0` | sigma |
   |---|---|
   | hit chi2 (btojpsix, 127,623 cand, whitened, no priors) | **0.102 T** = 27e-3 relative |
   | J/psi masses (128,687 cand, sigma_m median 31.3 MeV, Gaussian limit) | **9.2e-5 T** = 2.4e-5 relative |
   | ratio | **~1100x** |

   So the mass term is expected to pin, by three orders of magnitude, exactly
   the direction the hit term leaves softest -- the premise of the whole
   architecture. The joint fit has to show it, and show that it does *not*
   drag the other 49 modes.
4. **Validations 5 / 5b** (injection into both terms / into the mass term only).
   5c is done and exact.
5. Understand the residual few-sigma structure that survives the guards
   (rms pull 1.2 on btojpsix, 1.5 with an 8.4 tail on the ideal-geometry gun).
   Frozen alignment on btojpsix explains part of it; the gun has ideal
   geometry, so there it is the resolution model / deweighted strip second
   coordinates, or a further outlier class.
6. Freeze or prior the three zero-information material groups
   (`material_pp1_cables`, `material_beampipe`, `material_thermal_screen`).
7. Feed all of this back into `NOTES.md`, and re-check `cf_global_masslik.py`'s
   injection sign (it is opposite to the one derived and tested here).

### Scale

* Measured: quadratic-only extraction of 48 files / 128,707 candidates in
  **137 s** (8 workers); 160 files / 300k gun candidates in **167 s** (6
  workers). CF extraction ~75 min / 48 files (24 workers).
* rabbit on a 50-parameter dense quadratic card: **7.3 s** (trust-exact),
  56 s (trust-krylov), no convergence unwhitened.
* The mass term's per-candidate arrays dominate the card: 448 t-points x 5
  families x 4 B = **9.0 kB/candidate** (1.2 GB at 128k, ~90 GB at 1e7).
* **The sparse `D` layout is a pessimisation here**: `jac_indices` is
  `int64 (nnz, 2)` + `float64` values = 24 B/nnz and every candidate touches
  *all* 92 field+material parameters, so it costs 2.2 kB/candidate against
  368 B for a dense `(n, 92) float32`. Worth a dense-`D` option in
  `MassCFTerm` before 1e7 candidates (22 GB -> 3.7 GB).
* Alignment (108k parameters): the summed Hessian is sparse (module pairs that
  share a track); rabbit's own docstring quotes a **329M-nnz** external Hessian
  as a case its CSR `sm.matmul` path already handles, so ~5 GB of nnz plus the
  CSR build is the cost. `extract.py` has a scipy COO->CSR accumulation path
  above `--dense-max` and `make_global_term.py` a `wums.SparseHist` storage path
  above the same threshold; **neither has been exercised at that size**.
