# qmsmodel — why the hit-chi2 and the mass CF term disagreed on `material_tec_services`

## Purpose

The first joint fit (quadratic hit-chi2 term + `MaterialCFTerm`, both on the
same 42 parmtype-15 material groups) produced exactly one tension: on the
ideal-geometry J/psi gun, where the truth is `k_g = 0` for every group, the
**hit-chi2 term wanted `material_tec_services` at `k = -0.4646 +- 0.0682`**
(37 % less material) while the **mass term saw nothing, `k = -0.0087 +-
0.0950`**. This study establishes what the two functionals actually measure,
what the -6.8 sigma is, and what to do about it.

The answer in one line: **the hit-chi2 term's `k_g` derivative is mean-energy-
loss only**, so it cannot see a wrong `Q` at all; what it does see is a
mean-loss bias that grows with `dE_ref/p`, and `tec_services` is simply the
group with 58 % of its Fisher information above `dE_ref/p = 0.1`.

## The object

`k_g` scales three things in the propagator — the mean energy loss
(`G4ErrorEnergyLossForCVH.cc:115`, `xifact = exp(dxieff)`), the step MS
covariance and the step ionization variance (`Geant4ePropagator.cc:1217-1219`,
`:1263-1265`, `matStepFact = exp(k_g)`) — but **only one of the three is
differentiated**. The parmtype-15 column of the design matrix is
`transportJacobianBxByBzD`'s `dxi` column, and

```
Geant4ePropagator.cc:2967  const double dqopdxi = -x111*x3;   // x3 = dEdx*s
           :2976,2985,2994,3003  dlamdxi = dphidxi = dxtdxi = dytdxi = 0;
```

i.e. one non-zero row, `q/p`, from the mean loss. No derivative of `Q` w.r.t.
`k_g` is ever formed: the `dVs` / `residxs` / `resglobidx` machinery (with its
log-det term `gradll += tr(dV_i R)`, `G4e.cc:4674`) is registered **only** to
parmtype 10 (MS) and 11 (ionization), never to `matGroupGlobalIdx_`, and the
two-track maker has no `gradll` at all.

So the two functionals are not two measurements of one number: the hit-chi2
term measures the group's **mean energy loss**, the mass CF term measures its
**width contributions** (MS-dominated) plus a weak mean through the `D` rows.
They agree only if the sole error is the amount of material.

The comparison statistic for `Q` vs Moliere is built from the same `msmoliv`
step records the CF uses, with the production kernel
(`MS_ELEC_TMAX=1, MS_ELEC_EDGE=1, MS_FINE_G=1, MS_SNAP_YMAX=0`), in two
conventions: the **core-matched** Gaussian `sigma^2 = -2 S_g(u*)/u*^2` at the
`u*` where the candidate's own total exponent is `-1/2`, and the **second
moment** (`u -> 0`).

## How to run

All offline, ~25 min on 24 cores.

```bash
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CVH_IONI_KOKOULIN=0
cd /work/submit/david_w/ZMass/calibration_studies/resolution/qmsmodel
F='/ceph/submit/data/user/d/david_w/ZMass/cvh/resolution_trackres_jpsigun_ul16_260905d_m0/task_*/globalcor_*.root'

# 1. Q(Rossi) vs Moliere, per material group, from the msmoliv step records
#    --max-files caps TASKS (a numberOfThreads=N task is N stream files)
python3 qms_steps.py    --files "$F" --max-files 24 -j 24 -o tmp/gun_qms_24f.npz

# 2. the quadratic term's per-candidate gradient / Hessian diagonal + sandwich
python3 qms_quadgrad.py --files "$F" -j 6                      -o tmp/gun_quadgrad.npz
python3 qms_quadgrad.py --files "$F" -j 6 --max-chi2-ndof 0 --max-grad 0 --max-hess 0 \
                                                               -o tmp/gun_quadgrad_nocut.npz
python3 qms_quadgrad.py --files "$F" -j 6 --fr-cuts 0.03 0.01  -o tmp/gun_quadgrad_frcut.npz

# 3. the per-candidate step census of one group (36 = tec_services, 37 = reference)
python3 qms_census.py   --files "$F" -j 6 --group 36 --refgroup 37 -o tmp/gun_census36.npz

# 4. the three mechanisms, in a 40-line toy of the fit's structure
python3 qms_toy.py -n 20000 --sigh 0.05

# 5. tables + figures
python3 qms_report.py
```

Inputs: `resolution_trackres_jpsigun_ul16_260905d_m0` (300 k ideal-geometry
J/psi-gun ditrack candidates, 160 tasks). Caches: `qmsmodel/tmp/` (gitignored).
Figures: **`~/public_html/ZMass/cvh/260906_qmsmodel/`** — `ratio_per_group`,
`ratio_vs_moliere_log`, `sandwich_factor`, `leverage_concentration`,
`khat_vs_fractional_eloss`, `bias_vs_stepthickness`, `bias_vs_msratio`,
`toy_wrong_variance`, `toy_wrong_mean`, `frcut_material_fit`.

## Results

The quadratic block is reproduced bit-for-bit from the joint-fit card:
`material_tec_services` = **-0.4646 +- 0.0682** over 299 069 candidates.

### `tec_services` is not an outlier in the Q/Moliere MS ratio

44 909 candidates, 42.8 M steps:

| group | MS share | steps/cand | x_g/step [g/cm2] | Lm | R_core | R_2nd |
|---|---|---|---|---|---|---|
| `tib_support` | 29.00 % | 155.7 | 0.0646 | 18.44 | **0.906** | 1.032 |
| `tob_support` | 12.73 % | 145.1 | 0.0415 | 18.60 | **0.921** | 1.040 |
| `bpix_support` | 12.18 % | 112.7 | 0.0479 | 18.45 | **0.895** | 1.032 |
| `tec_structure` | 7.60 % | 275.7 | 0.0322 | 18.67 | **0.954** | 1.044 |
| `tibtid_services` | 7.14 % | 13.6 | 0.3447 | 18.29 | **0.944** | 1.028 |
| `tid_support` | 3.93 % | 68.9 | 0.0489 | 18.68 | **0.948** | 1.041 |
| `bpix_services` | 3.39 % | 4.1 | 0.9028 | 18.30 | **0.943** | 1.032 |
| `tob_services` | 3.24 % | 2.3 | 0.5300 | 18.10 | **0.916** | 1.024 |
| `pixel_patch` | 0.98 % | 0.4 | 1.2940 | 17.88 | **0.940** | 1.021 |
| `tec_services` | 0.91 % | 0.4 | 2.4007 | 18.31 | **0.942** | 1.026 |
| **all groups** | 100 % | 954 | — | — | **0.9145** | **1.0314** |

`tec_services` sits at 0.942, *between* `tec_structure` and `tibtid_services`
and ABOVE the three groups that carry 54 % of the MS. The full spread over all
live groups is `R_core` 0.872-0.970 and `R_2nd` 1.012-1.131. **To make
`k = -0.465` out of a variance mismatch needs `exp(0.465) = 1.59`** — 16x the
entire observed spread, and with the wrong sign (a variance fit reading
`R_core = 0.94` would want `k = -ln 0.94 = +0.06`).

Why the ratio is so universal: with `x_g` the areal density,

```
R = sigma^2_Moliere / theta_p^2(Q) = (0.157e-6/2.25e-4) * ZZA * Xs_g * F(y),
ZZA = sum_i w_i Z_i(Z_i+1)/A_i,  Xs_g = rho Xs,  F(y) = -2G(y)/y^2,  F(0) = Lm/2
```

— INDEPENDENT of the step's thickness, and the prefactor is **0.1264 (p1) /
0.1282 (median) / 0.1287 (p99)** over every tracker material (0.195 only in
beam vacuum): Rossi's `Xs` prescription does its job to 1 %. All of the
material dependence sits in `Lm = ln(theta_FF^2/chi_a^2) - 1`, 17.4-18.9-21.7,
i.e. +-5 %.

### The toy: a wrong `Q` shifts no parameter, a wrong mean is recovered exactly

`qms_toy.py`, the fit's structure in 40 lines (hits + process-noise rows, a
mean-only `k` column, FIXED weights), 20 000 trials each:

| test | result |
|---|---|
| true straggling variance = `r` x model `Q`, `r` = 0.25 … 4 | **`<k> = 0` for every `r`** (max 1.5 MC sigma); only `rms/Fisher` moves, 0.70 -> 1.75 |
| true mean = `rho` x model mean | **`<k> = rho - 1` exactly** (-0.402, -0.200, -0.002, +0.201 for `rho` = 0.6, 0.8, 1.0, 1.2) |
| correct mean, Moyal (skew 1.54) straggling, no cut | `<k> = 0` — a linear fixed-weight estimator does not see skew |
| the same + `chi2/ndof < 1.5` (15 % removed) | `<k>` = -0.001 / -0.010 / -0.051 / -0.175 as the straggling sd/mean goes 0.5 -> 4 |

### What the -6.8 sigma actually is

**(a) Its error is understated by 1.74.** `c_g = sum_i G_{i,g}^2 / (2 K_gg)`
measures the scatter of the fit's own gradients against the scatter its own
weighting predicts (`= 1` if `R` is right):

| group | `c_g` | occupancy | top-30 candidates' share of `sum G` |
|---|---|---|---|
| **`tec_services`** | **3.03** | 20.1 % | **54.9 %** |
| `tec_active_R7` | 1.24 | 42.2 % | — |
| `tec_structure` | 1.19 | 62.9 % | (sum ~ 0) |
| `pixel_patch` | 1.13 | 21.3 % | 48.7 % |
| `tob_support` | 1.02 | 70.7 % | 7.7 % |
| `tib_support` | 0.95 | 89.1 % | 10.0 % |

`tec_services` is the unique outlier. Honest sandwich errors: standalone
**-0.798 +- 0.153** (not +- 0.088), marginal **~ -0.465 +- 0.119**, i.e.
**-3.9 sigma, not -6.8**. Every other group's weighting is right to +-25 %,
which is the empirical bound on how wrong `Q` can be.

**(b) It is not the chi2 cut.** Removing `chi2/ndof < 3` entirely (300 025
candidates, `hessmax`/`gradmax` guards kept) moves `tec_services` from -0.7982
to -0.7945; tightening to 1.2 moves it to -0.7709.

**(c) It is a function of the fractional reference energy loss, and it is
UNIVERSAL across groups.** `k_hat` standalone, no chi2 cut, binned in
`max(dE_ref/p)` over the two legs (`Mu{plus,minus}_dEref` is the reference
propagation's own `sum (eIn - eOut)`):

| group | <0.003 | 0.003-0.01 | 0.01-0.03 | 0.03-0.1 | **>0.1** | % info >0.1 |
|---|---|---|---|---|---|---|
| `tec_services` | -4.50+-3.28 | -0.11+-0.54 | -0.30+-0.25 | -0.70+-0.17 | **-0.969+-0.115** | **58.2 %** |
| `tob_services` | -0.98+-2.00 | -0.29+-0.28 | -0.34+-0.13 | +0.04+-0.10 | **-0.617+-0.146** | 20.9 % |
| `pixel_patch` | +2.4+-10.3 | -0.87+-0.71 | +0.07+-0.32 | -0.03+-0.20 | **-0.581+-0.220** | 35.9 % |
| `tibtid_services` | +1.59+-1.24 | -0.15+-0.15 | -0.00+-0.07 | +0.12+-0.04 | **-0.205+-0.049** | 32.6 % |
| `tob_support` | +0.08+-0.29 | -0.03+-0.04 | -0.06+-0.02 | -0.02+-0.02 | **-0.211+-0.042** | 10.2 % |
| `tid_support` | +1.48+-1.67 | -0.19+-0.19 | -0.05+-0.09 | +0.09+-0.06 | **-0.170+-0.070** | 30.7 % |
| `tec_structure` | +0.43+-0.69 | -0.06+-0.12 | -0.02+-0.06 | +0.13+-0.03 | **-0.166+-0.038** | 34.1 % |
| `tib_support` | +0.12+-0.20 | -0.01+-0.03 | -0.02+-0.02 | -0.00+-0.01 | **-0.137+-0.030** | 10.9 % |

**Every group turns negative above `dE_ref/p = 0.1` and none is significantly
non-zero below 0.03.** The arithmetic closes: the information-weighted average
of this row reproduces each group's standalone value to the third digit
(`tec_services` -0.795 predicted vs -0.798 measured; `tob_support` -0.0585 vs
-0.0582). **So the ordering of the pulls across the 42 groups is not a
statement about the groups — it is where each group's Fisher information sits
in `dE_ref/p`.** `tec_services` is worst because it is crossed in ~2 THICK
steps (2.40 g/cm2/step against 0.03-0.06 for the support groups), 20 %
occupancy, median 4.7 g/cm2 and 0.185 X0 when touched, and 83 % of its
information comes from the 10 % of candidates whose softer muon has
`pT < 1 GeV` (curlers — every one of the top-15 by `|G_i|` has a leg at
0.5-0.7 GeV, with up to 1.99 X0 and 53 g/cm2 of `tec_services` on one
candidate).

Correlations of the `>0.1` bias across the 10 measurable groups:
**Pearson `r = -0.61` against the group's areal density per Geant4 step**, and
**`r = +0.03` against `R_core`**, the Moliere/Q MS ratio. That is the whole
result in two numbers.

### The tension disappears where the mean-loss model closes

Exact 92-parameter quadratic fit (50 field modes + 42 groups, material priors),
`G` and `K` re-accumulated over the candidates below a `dE_ref/p` threshold
(not a rescaling — `qms_quadgrad.py --fr-cuts`):

| | no cut (299 069) | `dE_ref/p < 0.03` (274 996) | `dE_ref/p < 0.01` (188 288) |
|---|---|---|---|
| `material_tec_services` | **-0.4646 +- 0.0682** | -0.0431 +- 0.0923 | **-0.0079 +- 0.0985** |
| `material_tob_support` | -0.0380 +- 0.0198 | -0.0519 +- 0.0261 | -0.0397 +- 0.0415 |
| `material_tec_structure` | +0.0535 +- 0.0293 | +0.0144 +- 0.0445 | -0.0079 +- 0.0491 |
| max abs pull over the 42 | **6.81** | 1.99 | **0.96** |
| rms pull over the 42 | 1.17 | 0.41 | **0.19** |

and the mass CF term says **-0.0087 +- 0.0950**. **At `dE_ref/p < 0.01` the two
functionals agree on `tec_services` to 0.0008 in `k`, and the whole material
block closes on MC (max pull 0.96, rms 0.19).** The few-sigma residual that
`globalfit/STATE.md` used to list as open on the ideal-geometry gun is this and
only this.

## Defects found and fixed

* **`qms_report.py`'s default figure directory is wrong.** It writes to
  `~/public_html/cvh/<stamp>_qmsmodel`; the canonical location is
  `~/public_html/ZMass/cvh/<stamp>_qmsmodel` (the non-`ZMass` path holds only
  symlinks). One line, owned by whoever next touches the script.
* **The hit-chi2 term's material parameter is a MEAN-LOSS parameter, not a
  material amount** — established structurally (section "The object") and
  confirmed by the toy. Any statement of the form "the two functionals measure
  the same `k_g`" is false above `dE_ref/p ~ 0.03`.
* **The quoted error on a material group is not `2 K^-1`.** `c_g` is the
  sandwich correction and it is 3.03 on `tec_services`; it is the only
  diagnostic that would have flagged the group before the joint fit did.
* **Working point, now implemented**: accumulate the quadratic term's material
  information with `dE_ref/p < 0.01`. This landed as
  `globalfit/extract.py --max-dEref-p 0.01` and is used in the full-scale
  productions. It costs 37 % of the J/psi candidates (~0.1 % of Z candidates)
  and 3-6 % on the material errors of the groups that matter, against removing
  a 6.8-sigma false pull. `Mu{plus,minus}_dEref` exists in every two-track
  production, so the cut is portable across v1 and v2 exports.
* **The 2026-08-16 "Q is 14 % too small" statement needs its convention.**
  Against the pure-Wentzel second moment the ratio is **1.173** (the 14 %
  stands), but the production model has the G4 electron kinematic ceiling,
  which removes most of it: `R_2nd` = **1.031**. Against the **core** — the
  statistic Rossi's 15 MeV actually fits and the one a Gaussian process noise
  is supposed to carry — `Q` is **8.6 % too LARGE** in variance
  (`R_core = 0.9145`). The sign depends entirely on the statistic, and that
  must be said whenever it is quoted. Scale sensitivity: `R_core` = 0.995 at
  `u*/3` and 0.796 at `3 u*`.

Step pathologies, for the record (96 768 sampled steps): 21.5 % have
`Omega_0 = chi_c^2/chi_a^2 < 1` (fewer than one scatter — the Gaussian is a bad
SHAPE there but the variance is fine), 13.3 % have `x_g < 1e-4 g/cm2`, and
`ymax = theta_FF/chi_a` is 5.4e3-8.5e4 for **every** step, so the `gshape`
table clip at [1e1, 1e7] never fires.

## Open items

1. **What the large-`dE/p` mean-loss bias is** is hypothesised, not
   established, and is not establishable offline. Excluded: a wrong `Q`
   (structurally), the chi2 selection, skewness alone with a linear estimator,
   and the Moliere/Q ratio (`r = 0.03`). What is left is that the reference
   mean loss the fit propagates
   (`G4EnergyLossForExtrapolator::EnergyAfterStep`, half-step-rescaled at
   `G4ErrorEnergyLossForCVH.cc:150-153`) is too large relative to what the
   tracks realize, growing with the fractional loss and with the thickness of
   the individual Geant4 step. In order of prior: (1) restricted vs
   unrestricted `dE/dx` — the extrapolator's tables and the SIM's
   `G4MuIonisation` need not share a production cut, and the difference is the
   delta-ray term, which scales with `x_g` and with density, both of which
   order the groups the way the measurement does; (2) the half-step
   linearisation at 10-20 % fractional loss; (3) the fit's own low-momentum
   guards.
   **The discriminating test**, when a production slot is free: a single-track
   gun at `p = 0.4-1.5 GeV` with `simPabsFirst`/`simPabsLast` filled (they are
   `-99` in the two-track production), comparing `simPabsFirst - simPabsLast`
   against `Mu*_dEref` per material group — model-free, 10 k events.
2. **The real fix, so that the two functionals are the same parameter**: give
   the quadratic term the variance derivative it is missing. The machinery
   exists — `dVs` / `residxs` / `resglobidx` with `gradll(i) += tr(dV_i R)`
   (`G4e.cc:4674`) and the Fisher curvature `tr(dV_i R dV_j R)` (`:4703`),
   today registered only to parmtypes 10 and 11. Registering parmtype 15 makes
   `k_g` measure amount-of-material through the mean AND the width, which is
   what the mass term already does; `Documents/Resolution/NOTES_EXPORTS.md`
   sec. 3.2 sizes the
   missing shape channel at 1.88x of the marginal information.
3. **`Q`'s MS convention** does not touch the tension and should be decided on
   its own merits. The inconsistency is real: `Q` carries Rossi's core width at
   a mass-averaged `effZ`, the CF carries the Moliere shape at the per-element
   sums. The exact change in `Geant4ePropagator::PropagateErrorMSC`
   (`:1957-2100`) is to replace the `Xs`/`DD` pair with the two lines the CF
   uses, from quantities `CalculateMoliereSums` (`:2411-2447`) already computes
   at the same step, with the SAME tabulated shape the maker already loads
   (`CvhCfExponents.cc:86`, `data/cvhcf_gshape_elec_v1.bin`). Three notes:
   `CalculateMoliereSums` is currently called only inside the
   `if (ioniStepLogging_)` guard (`:1233`) and would have to move out; the CORE
   convention needs one scalar `u` per leg; `res(1,1) = S2` (`:2074`) is the
   only element that has to change. Sizing: a coherent **-8.6 %** on the MS
   variance plus a **+-3-5 %** material-dependent part that a single global
   `CVH_MS_SCALE` cannot absorb. The effect on the fitted momentum is second
   order and must be MEASURED with `CVH_MS_SCALE` (scan 0.91 to 1.17, the span
   between the core and the pure-Wentzel conventions), not asserted — and only
   after the running productions finish.
