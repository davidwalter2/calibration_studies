# fullscale — the unbinned CVH mass likelihood at Run-2 scale

Reference for the full-scale feasibility study of `m_Z` and `Gamma_Z` from the
unbinned per-candidate CVH mass likelihood: the statistical reach, the MC
closure, and the joint fit with the J/psi channel that transfers the momentum
scale.

* WG-facing summary: `SUMMARY.md` next to this file, and
  `/work/submit/david_w/Documents/Resolution/FULLSCALE_SUMMARY_260908.md`.
* Programme-wide reference (resolution model, likelihood terms, certification
  rules): `/work/submit/david_w/Documents/Resolution/RESOLUTION.md`.
* Figures: `~/public_html/ZMass/cvh/260909_fullscale/` (final certified table),
  `~/public_html/ZMass/cvh/260906_fullscale/` (phase-1 inputs, correction
  forms, clip scan), `~/public_html/ZMass/cvh/260907_fullscale/` (full-scale
  post-fit spectrum). All carry `index.php`.

---

## 1. Purpose and phases

| phase | what | status |
|---|---|---|
| 1 | Z alone, resolution and alignment fixed at MC truth, `K(m)` floated | DONE, certified |
| 2 | joint J/psi + Z + hit-chi2 quadratic; the scale transferred through the field modes | subsample fit certified; the full-card fit descends but does not converge |
| 3 | the same with the parmtype-15 material amounts floated in both mass terms | card built and verified; fit not attempted |

---

## 2. The headline numbers

Z -> mumu MC, **3 682 662** CVH-refit candidates (all 380 tasks of
`dymc_8p5M_260906_v2`), unbinned per-candidate mass likelihood, resolution and
alignment fixed at MC truth, `K(m)` floated with 5 Legendre terms, both
mass-likelihood corrections in the fluctuation form.

| | value | note |
|---|---:|---|
| `sigma(m_Z)` | **2.27 MeV** | measured sandwich; inverse Hessian 2.07 |
| `sigma(Gamma_Z)` | **4.16 MeV** | measured sandwich; inverse Hessian 3.78 |
| the same with `K(m)` FIXED | 1.51 / 2.91 MeV | Asimov; floating the LO -> MiNNLO shape costs x1.5 / x1.4 |
| **`m_Z` closure, v form** | **-1.54 +- 2.31 MeV** | 0.7 sigma — CLOSES |
| `m_Z` closure, m form | -11.06 +- 2.27 MeV | 4.8 sigma — the parameterisation, not the data (sec. 5) |
| `K(m)` truncation systematic | **2.4 MeV** over `K` = 5..7, v form | the one open item on the closure |
| `Gamma_Z` closure, v form `K` 5 | +6.81 +- 4.21 MeV | but `K`-truncation sensitivity is 7.5 MeV, so NOT quotable at its statistical precision |

**Errors.** All errors carry the MiNNLO weight factor: `genweight` clipped at
100x the modal and rescaled to mean 1, 4.80 % negative, `N_eff/N = 0.8130`,
hence **x1.109**. For the m-form reference the sandwich was evaluated directly
(2.27); for every other row the error is the inverse Hessian x 1.109 — for the
m-form reference those two agree to 1 % (2.27 measured against 2.296
approximated).

**Scaling.** On the full Run-2 muon sample (~10x this MC) the statistical
`sigma(m_Z)` is well below 1 MeV; systematics dominate there.

---

## 3. The certified closure table

Acceptance test: **value AND NLL AND EDM** (sec. 9). `certtable.py` applies it
mechanically to every stored fit and marks each row QUOTE or not. MeV from the
generator (`m_Z = 91.153509740726733`, `Gamma_Z = 2.4932018986110700`).

### 3.1 Inclusive, and the `K(m)` ladder

| `K` terms | m form `m_Z` | v form `m_Z` | m form `Gamma_Z` | v form `Gamma_Z` |
|---:|---:|---:|---:|---:|
| **5** | **-11.06 +- 2.29** | **-1.54 +- 2.31** | -5.26 +- 4.19 | **+6.81 +- 4.21** |
| 6 | -13.98 +- 2.27 | -3.92 +- 2.36 | +27.16 +- 4.38 | +14.28 +- 4.16 |
| 7 | -17.23 +- 2.32 | -3.87 +- 2.53 | +8.96 +- 4.55 | +12.86 +- 4.83 |
| 9 | +9.39 +- 2.56 | NOT ATTAINABLE (sec. 7.3) | -1.79 +- 5.09 | — |
| 12 | NOT ATTAINABLE | NOT ATTAINABLE | — | — |

Two independent converged runs of the v card agree: -1.533 and -1.538.

m form: **26.6 MeV** of `m_Z` spread over 5 -> 9 and not monotone; the 9-term
fit is preferred at 24 sigma with all coefficients O(1)
(`shape9 = +0.003683 +- 0.000154`), so this is structure in the LO -> MiNNLO
K-factor that the 5-term basis cannot carry, not a runaway.
v form: **2.4 MeV** over 5 -> 7, inside one sigma. That is the truncation
systematic to quote.

`Gamma_Z`'s `K`-truncation sensitivity is the real limit in BOTH forms:
7.5 MeV (v, 5 -> 6) and 32.4 MeV (m) against a 4.2 MeV statistical error.

NLL ladder, m form: 11075392.4657 / 11075277.1487 / 11075192.7817 /
11074904.2322 for `K` = 5/6/7/9, i.e. `2 dNLL` = 230.6 / 168.7 / 577.1.

### 3.2 The `eta` bands — the lead-band pattern was the SELECTOR

| band | lead band, `\|eta\|` of the RECO-leading leg | **safe band, `max(\|eta_p\|,\|eta_m\|)`** | prediction, recorded in advance |
|---|---:|---:|---:|
| `\|eta\|` < 0.9 | -21.09 +- 3.19 | **+2.68 +- 4.38** | +4.3 |
| 0.9 - 1.6 | +12.39 +- 4.21 | **+2.92 +- 3.77** | -4.6 |
| 1.6 - 3.0 | +34.22 +- 5.30 | **-9.83 +- 3.95** | -3.9 |
| **endcap - barrel** | **+55.31 +- 6.19 (8.9 sigma)** | **-12.51 +- 5.90 (2.1 sigma)** | +67.8 -> **-8.2** |
| chi2 against a common value | **93.8 / 2** | **6.7 / 2** | |
| weighted mean | -0.79 +- 2.29 | **-1.54 +- 2.32** | |

All v form, `K` = 5, errors inverse Hessian x 1.109. NLL / EDM of the safe
rows: `VXetaB` -1926526.7947 / 1.52e-10; `VXetaT` -3371566.6113 / 6.46e-11;
`VXetaE` -4526804.5804 / 3.33e-15; `VXetaBslo` -952284.2646 / 1.12e-20;
`VXetaBshi` -978837.0716 / 1.23e-11. `certtable.py` marks all five QUOTE.

**The safe bands' weighted mean equals the inclusive v-form closure to three
digits** (-1.538 +- 2.311), which is what "the inclusive closure is not a
cancellation" means operationally. The two definitions select different
candidates (safe barrel 699 414 against lead barrel 1 572 534), so the
comparison is spread-to-spread, not cell-by-cell.

The corresponding m-form lead bands are -26.60 +- 2.87 / +3.87 +- 3.88 /
+16.55 +- 4.71 (inverse Hessian).

**What survives:** -12.51 +- 5.90 MeV (2.1 sigma) of endcap - barrel spread,
*opposite in sign* to the lead-band pattern. It is an **upper bound on a true
detector-side `eta` effect, not a demonstrated effect**:
`max(|eta_p|,|eta_m|)` still carries `corr(., z) = +0.0025` against `+0.0004`
for the gen definition, so part of it may itself be selection. A clean number
needs a gen predictor, which a card cannot cut on.

### 3.3 The `sigma/m` split at fixed `eta`

| cell | lead band | **safe band** |
|---|---:|---:|
| barrel `sigma/m` LOW | -4.56 +- 4.10 | **-2.53 +- 6.04** |
| barrel `sigma/m` HIGH | -40.90 +- 5.14 | **+9.99 +- 6.62** |
| barrel HIGH - LOW | **-36.35 +- 6.57 (5.5 sigma)** | **+12.52 +- 8.96 (1.4 sigma)**, predicted +18.6 |
| endcap `sigma/m` LOW | +28.38 +- 5.75 | — |
| endcap `sigma/m` HIGH | +65.87 +- 9.64 | — |
| endcap HIGH - LOW | **+37.5 +- 11.2 (3.3 sigma)** | — |

The two lead-band slopes are equal in size and **opposite in sign**
(difference +73.8 +- 13.0, 5.7 sigma), which no pure scale error on `sigma`
can produce. On the safe band the barrel slope changes sign and falls to
1.4 sigma, 0.7 sigma from its pre-registered prediction. A `sigma/m` cut is a
cut on the residual **whatever the band variable is** (standing rule 3), so
changing the band removes the band's own selection effect and not the
`sigma/m` one — that was the non-obvious half of the prediction and it holds.

### 3.4 Phase 2

`P2smoke` (`joint_ok_n500k`: 500 k J/psi + 500 k Z + the hit-chi2 external
quadratic over 20 706 999 candidates, 95 free of 103), scipy `trust-exact`
through `rabbit_fit.py`, 3 h 39, **EDM 6.2e-19**, NLL 615177.106228:

| | |
|---|---:|
| `m_Z` | +31.86 +- 5.78 MeV |
| `Gamma_Z` | +8.75 +- 10.32 MeV |
| `bfield` pulls | RMS 5.45, max 32.7 |
| `material` pulls | RMS 61.0, max 325 |

**This is not a closure.** It is a 500 k subsample, and `theta = 0` is not the
hit-chi2 minimum on this MC, so the large pulls are the quadratic term pulling
the calibration parameters off their nominal zero — a property of the input.
What it demonstrates is that the loop closes: card -> `rabbit_fit.py` ->
converged joint minimum with an EDM. `sigma(m_Z) = 5.78` at that subsample
means transferring the absolute scale from the J/psi through the field modes
costs ~1 % in precision.

The full-card fit (`P2X`, `joint_ok_full`, exact delta-kernel J/psi term)
descends — EDM 1.4e5 -> 150 over 19 Hessians in 6 h 15, condition number
3.1e19 -> 1.0e15 — and is **not converged**. A flat EDM on this card family is
a shelf, not a stall (the converged m-form 9-term rung sat on one for twenty
iterations), but leaving the shelf is not sufficient either (sec. 7.3). The
named remedy is trust-region preconditioning or the 2-GPU candidate sharding,
not a resubmit.

**Before the first quotable phase-2 number**: the J/psi term's physics kernel
is a delta at the PDG mass, but the J/psi MC has FSR, so for a radiating
candidate the post-FSR gen mass is not the PDG mass. Two things must be
measured — what fraction of candidates radiate enough to matter and the
resulting shift of `<m_gen>`, and what the difference between the delta-at-PDG
and own-gen-mass treatments costs on the **extracted momentum scale**. On the
gun the comparison is made against the candidate's own gen mass, which is the
correct reference. This is a check that has not been done, not a demonstrated
defect.

---

## 4. Inputs, caches and cards

### 4.1 Productions

| | `dymc_8p5M_260906_v2` | `jpsimc_20M_260906_v2` |
|---|---|---|
| channel | Z -> mumu, POWHEG MiNNLO, official UL16 SIM/RECO (106X) | J/psi -> mumu |
| path | `/ceph/submit/data/user/d/david_w/ZMass/cvh/dymc_8p5M_260906_v2` | `/ceph/submit/data/user/d/david_w/ZMass/cvh/jpsimc_20M_260906_v2` |
| tasks | 380 / 380 | **1645 / 1645** |
| events | 8 502 597 | 21 750 740 |
| candidates produced | 3 799 624 | 21 678 062 |
| candidates in the cache | 3 733 323 (mass window) | 7 923 460 (600-task cache) / 16 955 312 (quadratic term, all 1645, 4 722 750 cut) |
| after the card selection | **3 682 662** (98.6 % of the cache) | 3 000 000 used in the joint card |
| volume on ceph | 267 GB | 1.5 TB |

A third input, the **hit-chi2 external quadratic** over **20 706 999**
candidates, carries the alignment/field/material curvature into the joint fit
(`globalfit/extract.py --no-mass`).

Details of the productions themselves, including the defects found and
repaired, are in `production/` (see `production/PRODUCTIONS.md`).

**Which count every input used — quote this with any number:**

| input | tasks | content |
|---|---|---|
| `runs/quad_jpsiv2_ok.npz` | 1645 / 1645 | 16 955 312 in the quadratic term, 4 722 750 cut |
| `runs/jpairs_v2_n600.npz` | 600 by choice (tasks 0-599) | 7 923 460 candidates |
| `runs/gpairs_v2_n50.npz` | 0-49 | 651 672 candidates, 20.5 GB — the phase-3 per-group cache |
| `runs/gzpairs_dyv2_n50.npz` | 0-49 of DY | 487 742 candidates, 17.0 GB |
| `runs/quad_dyv2.npz` | 380 / 380 | 3 751 687 candidates |
| `runs/zpairs_dyv2_jac_full.npz` | 380 / 380 | 3 733 323 with the (n,92) mass Jacobian |
| `runs/quad_jpsiv1.npz` | — | 15 965 797 candidates, CROSS-CHECK ONLY |

**Phases 2 and 3 run on J/psi v2, not v1.** Measured reason: on the same 120
tasks v2's parmtype-15 information is 120x the trace and 666x the median
diagonal of v1's, while parmtype 14 is bit-equal.

The 50-task group caches are sized by a ~20 GB budget (30.6 / 34.1 kB per
candidate), not by statistics.

### 4.2 Selection and model choices

1. `m_obs` in [60, 120] GeV. The production selected on the PRE-refit
   `Jpsitrk_mass`; the mismatch is 0.191 % and one-sided.
2. `chi2/ndof < 3`, `sigma_m/m < 0.10`.
3. Weights: `genweight` clipped at 100x the modal, rescaled to mean 1.
4. `K(m)`: 5 Legendre terms, floated, over the fit window.
5. Resolution fixed at MC truth in phases 1-2 (`--freezeParameters k_hit k_ms
   k_ioni k_rad`); phase 3 replaces the knobs with the parmtype-15 amounts.
6. Alignment fixed at MC truth throughout.
7. `f_ang` is folded into `jensen_s2` per candidate (median 8e-5 at the Z; on
   J/psi v2 it comes from `Jpsi_covrefmom`, truth-free, median 6.2e-2).
8. Corrections in the **fluctuation** form; `corr_coeff_max = 0.08`;
   `floor_scale = 1e-7`.

### 4.3 Cards

| card | what |
|---|---|
| `cards/z_full380_fl.hdf5` | the m-form reference, 3 682 662 candidates |
| `cards/z_V_full.hdf5` | the v form, same candidates, `vpow = 1.264` (3.65 GB) |
| `cards/z_VX_eta{B,T,E}.hdf5`, `z_VX_etaB_{slo,shi}.hdf5` | the safe bands |
| `cards/z_V_eta{B,T,E}.hdf5`, `z_V_eta{B,E}_{slo,shi}.hdf5` | the lead bands |
| `cards/z_V_s{6,7,9,12}.hdf5`, `z_full380_fl_s{6,7,9,12}.hdf5` | the `K(m)` ladder |
| `cards/z_F_{toy,toydc,dc8,w70110}.hdf5` | the mechanism diagnostics (sec. 5) |
| `cards/joint_ok_full.hdf5`, `joint_ok_n500k.hdf5` | phase 2 (10.7 GB full) |
| `cards/joint_mat_v3.hdf5` | phase 3, **28.369 GB** |
| `cards/z_n300k.hdf5`, `smoke_zls.hdf5`, `joint_smoke.hdf5` | gates and smoke tests |

`runs/` and `cards/` are gitignored.

`z_V_full.hdf5` round-trips by hand: `vpow = 1.264`,
`norm_window = [0.169341, 0.384254]`, `m_ref = 0.3037924`, and
`v(60) - v(91.1876) + m_ref^{1-p} = 0.16909`,
`v(120) - v(91.1876) + m_ref^{1-p} = 0.38420`, `91.1876^{-0.264} = 0.3037924`.
**Not one candidate hits `corr_coeff_max` in the v form**, where the m form
bounded 364 of 300 000 (0.12 %).

---

## 5. The physics result: the m-form -11 MeV is the resolution-mass pairing

### 5.1 What it is NOT — seven things excluded by direct measurement

| excluded | measurement |
|---|---|
| the two corrections | they move `m_Z` by **+8.00 MeV** toward zero, inside the spec's own -5...-14 MeV prediction |
| the Born lineshape and `K(m)` | at generator level the same provider with 5 terms closes to -0.45 +- 0.50 MeV pre-FSR and +0.15 +- 0.56 post-FSR-folded |
| the coefficient bound | 0.049 MeV on `m_Z`, 0.022 on `Gamma_Z`, measured |
| a loose SIM stepper | the official UL16 SIM already sets `DeltaOneStep = 1e-5` / `DeltaIntersection = 1e-6`; and DY's `eta` spread is 0.41e-4 against 16.70e-4 for a private tight-stepper gun, and DY is 4 % NARROWER inclusively |
| the reconstruction's momentum scale | DY inclusive charge-even `A = -0.682 +- 0.090e-4`, i.e. reconstructed masses **+6.2 MeV HIGH** — the opposite sign — and flat in `eta` to 3.7 MeV |
| the resolution model | the mass pull width is **0.9959** inclusively and flat in `eta` to +-0.7 %; with the kernel removed entirely the detector half closes at **+0.88 +- 2.12 MeV** |
| the kernel | with the detector removed entirely (the selected candidates' own gen masses through the same lineshape (x) A (x) FSR (x) K chain) the kernel half closes at **+0.76 +- 1.36 MeV** inclusively, +0.6 / +4.2 / -2.7 per `eta` band |

**Both halves close in isolation and the defect lives only in the
combination.**

### 5.2 The mechanism — Punzi's variable-resolution problem

The likelihood is `prod_i p(m_i | sigma_i)` and the model computes
`int p(m') K_{sigma_i}(m_i - m') dm' / Z_i` — the SAME Born spectrum `p(m')`
for every candidate whatever its `sigma_i`. On this sample that is badly
false. Octiles of the absolute `sigma_m`:

| `sigma_m` [GeV] | `<sigma>` | `<m_gen>` [GeV] | `<sigma/m>` |
|---|---:|---:|---:|
| 0.332 - 0.783 | 0.699 | **84.94** | 0.0083 |
| 0.914 - 1.012 | 0.965 | 89.21 | 0.0109 |
| 1.103 - 1.227 | 1.161 | 90.50 | 0.0129 |
| 1.403 - 1.758 | 1.553 | 91.12 | 0.0171 |
| 1.758 - 11.9 | 2.597 | **91.28** | 0.0285 |

`rho(sigma, m_gen) = 0.168` and the conditional mean of the TRUE mass runs
over **6.3 GeV**. `K(m)` cannot absorb it: the class-conditional spectra
average to the marginal, but the observed spectrum is
`sum_c P(c) [p(m'|c) (x) K_c]` while the model can only produce
`sum_c P(c) [p(m') (x) K_c]`; `K(m)` multiplies the Born spectrum BEFORE the
convolution and cannot repair a pairing of kernel width with mass.

It is invisible to both closure tests of sec. 5.1 by construction: the
generator-level fit has no `sigma`, and the kernel-free residual fit has a
delta lineshape, so there is no true-mass distribution left to correlate with
`sigma`.

**Fit-free confirmation.** Build the two observed spectra the model can and
cannot produce and ask what mass shift reconciles them with a floated 5-term
`K(m)` — numpy quadrature and a Nelder-Mead, no minimiser:

| conditioning label | barrel | transition | endcap | **inclusive** |
|---|---:|---:|---:|---:|
| absolute `sigma` | -12.04 | -16.50 | -25.61 | **-15.33** |
| **`k = sigma/m^1.264`** | **-0.23** | **+0.44** | **+0.20** | **+0.30** |

Stable at 8 and 48 classes (-16.1, -14.6). Letting the width additionally
scale along the integration changes +0.30 to +0.46, i.e. nothing. **The bias
is not the width gradient — it is that `sigma_i` as a conditioning label
carries mass information and `k_i` does not.**

**Reweighting proof.** `F_dc8` — the real data reweighted by
`p(m_gen)/p(m_gen|class)` so that the true mass is independent of the `sigma`
class, nothing else changed — gives **-2.035** against the reference -11.064,
on the same candidates with the same model. The assembly toy `F_toy`
(`m_gen_i + sigma_i z_j`, residuals shuffled inside 20 `sigma/m` classes)
gives -3.752: ~70 % of the bias lives in the residual-level part, which is
what the v form fixes.

### 5.3 The fix: change the convolution variable

Write the smearing as `m_i = m' + k_i m'^{1+f} x` and substitute

```
v(m) = Int dm / m^{1+f} = m^{-f} / (-f)          (v = ln m when f = 0)
```

which gives `v_i = v(m') + k_i x` to first order — a **fixed-width convolution
in `v`**, with a width `k_i` that is independent of the true mass. The FFT
still applies on a grid uniform in `v`; the Born density carries the Jacobian
`p_v(v) = p(m(v)) m(v)^{1+f}`; the conditioning is on `k_i`, for which
`p(m'|k_i) = p(m')`.

The exponent is a **single common** `p = 1.264 = 1 + <vgf>`, the `a`
correction's own. That is a measurement, not a convenience: the per-candidate
`1 + vgf_i` version is WORSE (`rho(k, m_gen) = -0.129`, `<m_gen>` spread
3.27 GeV across `k` octiles, against `-0.011` and 0.514 GeV for the common
`p`). `vgf` varies for reasons unrelated to the resolution's mass exponent.
The fit-free optimum is `p = 1.235`; the predicted bias is within +-0.35 MeV
over `p` in [1.20, 1.264], so it is not a knob to tune per fit.
**No `f`-class axis is needed.**

`sigma_obs/m_obs^{1+f} = k_bar` EXACTLY when `a = (1+f) sigma_bar/m_bar`,
which is the `a` the term already uses — so conditioning on `k_i` is both
legitimate and consistent with the correction that is already there. The
consequence is that **there is NO `(1 - a_i x)` measure term in `v`**: the
substitution absorbs it, and carrying it over would double-count.

Coefficients in `v`:

```
a^v_i = a_i - p sigma_i/m_i                       the residual self-consistency
g^v_i = -a^v_i + (1 - p/2) sigma_i/m_i            Jensen (+1) plus the
                                                   substitution's curvature (-p/2)
d^v_i = m_i^{1-p} s_i^2 / 2                        the Jensen mean shift / m^p
```

**Numerical validation** (`proto_vmass.py`, pure numpy quadrature against the
smearing that actually happens, width taken at the TRUE mass, 5-term `K(m)`
floated over 60-120):

| `sigma/m` at 91 GeV | the m model | **the v model** |
|---:|---:|---:|
| 0.006 | -6.50 | **-0.00** |
| 0.010 | -15.57 | **-0.00** |
| 0.016 | -29.07 | **-0.01** |
| 0.020 | -31.38 | **-0.01** |

The sample's median `sigma/m` is 0.0123. Scanning the coefficient of `x^2` in
`v`, the optimum is `-0.625 k sigma/m` against the derived `-(1+f)/2 = -0.632`
and there the residual is 0.03 MeV — the curvature is confirmed by scan, not
assumed.

**Both Jacobians are required and their ratio is the effect.**
`L_v(v_i) = E_x[p_v(v_i - u^v(x))]` and the density in `m` is
`L_v / m_i^{1+f}`; their ratio `(m'/m_i)^{1+f}` is 7 % over the kernel's own
support, exactly the size of the thing being corrected. Dropping it leaves a
residual of -2 to -37 MeV that looks like a partial fix.

**Cost.** At full statistics the v form's inverse-Hessian `sigma(m_Z)` is 2.08
against the m form's 2.07 — i.e. essentially free. (A profiled 50 k
reference-point estimate had projected a 9 % cost, 2.27 -> ~2.47 MeV; that did
not materialise at full statistics. `sigma(Gamma_Z)` was projected flat and
is.)

### 5.4 The `eta` pattern: the fifth conditioning trap

`make_card.py:365-375` defines the band by the `|eta|` of the leg with the
larger **RECO** `pT`. When the legs have similar `pT`, which one leads is
decided by which one fluctuated up, so the band edge is a cut on the residual.

| variable | `corr(., z)` |
|---|---:|
| `\|eta\|` lead, **RECO** (what every band card cuts on) | **+0.0203** |
| `\|eta\|` lead, **GEN** | **+0.0004** |
| `max(\|eta_p\|, \|eta_m\|)` (the safe band) | +0.0025 |
| GEN-predicted `sigma/m` | +0.0024 |
| gen `pT` of the softer leg | -0.0005 |
| `sigma/m` (reco) | -0.0055 |

A factor of 50 between the reco and gen definitions of the same variable. 2 %
of candidates change band between the two definitions and they move the
charge-even per-leg bias `A` by 2.7e-4.

The per-leg charge-even momentum bias `A` (`legscale.py`; `A > 0` means reco
`p` too LOW, hence `dm/m = -A`), in 1e-4:

| cell | `A`, RECO `eta_lead` | `A`, GEN `eta_lead` (safe) |
|---|---:|---:|
| barrel `\|eta\|` < 0.9 | **+2.572 +- 0.102** | **-0.171 +- 0.103** |
| 0.9 - 1.6 | -0.066 +- 0.128 | +0.632 +- 0.156 |
| endcap 1.6 - 3.0 | **-4.867 +- 0.217** | **+0.550 +- 0.223** |
| barrel `sigma/m` LOW -> HIGH | `dA` **+4.94** | `dA` **-0.64 +- 0.16** |
| barrel, GEN-PREDICTED `sigma/m` | `dA` +3.56 | `dA` **-0.03 +- 0.18** |

**Consequence 1.** With the safe definition the legs are flat in `eta` to
**0.8e-4** and the barrel `sigma/m` split collapses to zero on the gen
predictor. **The `sigma/m` pattern is NOT a per-leg momentum bias.**

**Consequence 2.** The selector alone shifts the selected candidates' TRUE
masses by almost exactly what the fits report — `-A` predicted against the
certified fitted `m_Z`: barrel **-23.4** against -21.08; middle +0.6 against
+12.39; endcap **+44.4** against +34.22.

**The confirming refit ran and both pre-registered predictions hold**
(sec. 3.2, 3.3). Per-band agreement with `-A` is 0.4 / 2.0 / 1.5 sigma, the
middle band having the wrong sign — which is why the established claim is the
SPREAD claim, the one nominated in advance.

The "GEN-PREDICTED `sigma/m`" split uses a predictor built ONLY from gen
quantities: the mean `sigma/m` in a 40x40 quantile grid of (gen `|eta|` lead,
gen `pT` of the softer leg), ~2300 candidates per cell.
`sigma_bar = sigma(1 - a z)` is **not** a repair: built from `z`, it is
anti-correlated with the residual by construction,
`corr(sigma_bar/m, z) = -0.090` against `corr(sigma/m, z) = -0.006`.

### 5.5 Smaller effects, measured

| item | size | status |
|---|---|---|
| the two mass-likelihood corrections | +8.00 MeV on `m_Z`, additive to 0.2 MeV (`a_res` alone +15.10, Jensen alone -7.26, sum +7.84); neither touches `Gamma_Z` | implemented, gated |
| `k_ms` in the kernel-free fit | **1.0298 +- 0.0042** (7 sigma) — the multiple-scattering TAIL is ~3 % short | real material; 0.5 MeV on `m_Z`; phase 3 is its test |
| `k_hit` in the kernel-free fit | 1.0647 +- 0.0291 (2.2 sigma) | — |
| the `_norm_z` class-sigma approximation | 0.016 MeV on `m_Z` | closed |
| the coefficient bound `corr_coeff_max = 0.08` | 0.049 MeV on `m_Z`, 0.022 on `Gamma_Z` | closed |
| the 0.19 % one-sided selection-variable mismatch | — | recorded |
| post-fit spectrum | `chi2/ndof = 4.75` over 240 bins, unchanged when the model subsample is grown 6.7x | genuine few-% shape mismodelling |
| GN second-order (Box) charge-odd bias | -4.7e-5 on the momentum scale at 20-60 GeV, ~4 MeV on `m_Z` uncalibrated, ~0.4-0.7 MeV after the J/psi anchors the scale | analytic per-track correction, no new production |
| the `a`-coefficient deficit | measured 1.2110 +- 0.0004 against the closed form 1.2625 on the Z legs, 0.8928 +- 0.0130 against 1.0996 on the J/psi legs | open, small; two explanations excluded (sec. 8) |

**The Z alone cannot separate the resolution model from `m_Z`.** Floating one
knob at a time on a 300 k card: `k_ms` comes out at the MC truth
(+1.001 +- 0.034) and costs nothing on `m_Z`; `k_rad` is not measurable at all
(-0.76 +- 9.61); `k_ioni` is degenerate with `m_Z` (+7.32 +- 4.10, moving
`m_Z` by -18 MeV and inflating its error 8.1 -> 14.3). **None of the three
closes `m_Z`** — freeing a resolution knob inflates the error instead. That is
the physics case for phases 2 and 3 in one table.

**The "2.8 % width deficit" is retracted as a statement about the
resolution.** `resid_decompose.py` projects the post-fit residual onto a shift
and a width template and finds SHIFT +18.79 +- 1.70 MeV and WIDTH
+2.838 +- 0.222 %, but with both templates removed `chi2/ndof` is still 3.48:
they do not describe the residual, and the direct pull says the resolution is
right to 0.4 %. The width template is simply the closest available basis
vector to a kernel-side mismodelling.

---

## 6. How to run

```bash
FS=/work/submit/david_w/ZMass/calibration_studies/fullscale
Z=/work/submit/david_w/ZMass/calibration_studies/zchannel
RES=/work/submit/david_w/ZMass/calibration_studies/resolution
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate     # numpy/uproot side

# 1. pairs cache  (~25 min; ceph, so run on submit50/51)
python3 $RES/cf_inmaker.py pairs \
  --files /ceph/submit/data/user/d/david_w/ZMass/cvh/dymc_8p5M_260906_v2 \
  --ntasks 0 --cache $FS/runs/zpairs_dyv2_full.npz --mass-window 91.1876 60
#   add --jac-parmtypes 14 15 for the phase-2 version (the (n,92) mass Jacobian)
#   add --groups for the phase-3 per-group CF exponents

# 2. quadratic term  (~2.5 min DY, ~16 min a J/psi production)
python3 $RES/globalfit/extract.py --files <production> --ntasks 0 \
  --parmtypes 14 15 --no-mass --max-chi2-ndof 3 --max-hess 1e8 --max-grad 1e6 \
  --max-dEref-p 0.01 -j 16 -o $FS/runs/quad_<tag>.npz
#   --no-mass is MANDATORY on these productions (sec. 10, pitfall 1)

# 3. card  (~13 min, mostly npz decompression). Defaults: fluctuation form,
#    corr_coeff_max 0.08, floor_scale 1e-7. Add --vpow 1.264 for the v form.
RABBIT=/work/submit/david_w/ZMass/rabbit-vmass THREADS=32 $FS/run_tf.sh \
  python3 -u $FS/make_card.py \
    --pairs $FS/runs/zpairs_dyv2_jac_full.npz --shape 5 --chunk 32768 \
    --fsr $Z/data/kern_loose_band3.3e-4.npz --acc $Z/data/acc_loose_d8.json \
    --vpow 1.264 -o $FS/cards/z_V_full.hdf5

# 4. gates 2 + 2c, ALWAYS, before any fit -- seconds, no minimiser
RABBIT=/work/submit/david_w/ZMass/rabbit-vmass THREADS=8 $FS/run_tf.sh \
  python3 -u $FS/gate_fd.py --card $FS/cards/z_V_full.hdf5 \
    --jacobian-compare $FS/cards/z_full380_fl.hdf5 -o $FS/results/gate2_v.json

# 5. the fit, through rabbit_fit.py (scipy trust-exact is the campaign minimiser)
RABBIT=/work/submit/david_w/ZMass/rabbit-vmass $FS/run_tf.sh \
  python3 -u $RABBIT/bin/rabbit_fit.py $FS/cards/z_V_full.hdf5 \
    --paramModel UnbinnedParams --minimizerMethod trust-exact \
    --freezeParameters k_hit k_ms k_ioni k_rad \
    -t 0 --unblind --diagnostics \
    --snapshotFile $FS/results/vfull.snapshot.hdf5 --snapshotInterval 0.25 \
    --outpath $FS/results --outname rabbit_Vfull.hdf5

# on Engaging (H200): stage, then submit
$FS/stage_eng.sh card $FS/cards/z_V_full.hdf5
eng "cd ~/orcd/pool/zmass/engaging && sbatch -A mit_general -G h200:1 \
  -p mit_preemptable -t 04:00:00 \
  --export=ALL,CARD=\$HOME/orcd/pool/zmass/cards/z_V_full.hdf5,\
ARGS='--chunk 32768 --minimizerMethod trust-exact' rabbit_vmass.sbatch Vfull"

# 6. the sandwich, at rabbit's minimum
python3 $FS/rabbit_to_json.py $FS/results/rabbit_Vfull.hdf5 -o $FS/results/vfull_start.json
$FS/run_tf.sh python3 -u $FS/fit.py --card $FS/cards/z_V_full.hdf5 \
    --no-fit --start-from $FS/results/vfull_start.json

# 7. collect, certify, tabulate, plot
$FS/collect.sh --summary            # rsyncs Engaging and re-makes the certified table
python3 $FS/certtable.py            # value AND NLL AND EDM, per row
python3 $FS/plot_closure.py
THREADS=8 $FS/run_tf.sh python3 -u $FS/plot_postfit.py \
  --card $FS/cards/z_V_full.hdf5 --fit $FS/results/fit_Vfull.json --nsub 4000

# 8. the J/psi gun gate on the correction forms (in process, no card, no provider)
THREADS=24 $FS/run_tf.sh python3 -u $FS/gate_fluct_gun.py \
  --pairs $RES/runs/cf_masspairs_jpsigun_ul16_260905d_m0.npz \
  --kernel $RES/runs/cf_masskernel_jpsigun_ul16_260905d_m0.npz \
  [--vpow 1.264] -o $FS/results/gate1.json
```

`./collect.sh --summary` is the one command to run first on any fresh session.

**The fit code** is `/work/submit/david_w/ZMass/rabbit-vmass`, branch
`vmass-conditioning` — the material CF term + the Z lineshape kernel +
the native-minimiser PRs (#153 native TF trust-region minimizers, #154
multi-device, #155 snapshots) + the `vpow` change of variable, all merged
into this ONE branch (2026-09-13); it is the only rabbit branch and the only
rabbit worktree. It is not on any remote.

**Tools.** `make_card.py`, `make_joint_card.py` (`--material` for phase 3),
`gate_fd.py`, `gate_fluct_gun.py`, `gate_nanstep.py`, `certtable.py`,
`checkconv.py`, `cardkey.py`, `collect.sh`, `rabbit_to_json.py`,
`json_to_snapshot.py`, `seed_rows.sh`, `legscale.py`, `measure_a.py`,
`mixture_legs.py`, `qopbias.py`, `resid_decompose.py`, `proto_vmass.py`,
`vtable.py`, `plot_{inputs,postfit,closure,corrforms,mixture}.py`. The
standalone drivers `fit.py` / `fit_joint.py` / `chunkfit.py` / `devobj.py` /
`shardobj.py` survive as the **reference implementation** the rabbit path is
checked against, as the source of the sandwich covariance (rabbit does not
compute it), and for the `--ares` / `--jensen` / `--corr-clip` model switches
that turn one card into a scan. New fits go through `rabbit_fit.py`.

---

## 7. Certification, minimisers and conditioning

### 7.1 The acceptance test for every fit in this campaign

Three parts, each failure mode separately diagnosable:

1. **the value** — does it return the number the card is known to give?
2. **the NLL** — is it the SAME minimum? Two converged fits of one card whose
   NLL differs by more than float noise are at different stationary points
   (measured: 81.6 units apart on `z_n300k`).
3. **the EDM** — did it stop early? `0.5 g^T H^-1 g`, small. A displacement of
   `d` sigma in the worst direction costs `d^2/2`, so EDM < 1e-3 means
   `d` < 0.045 sigma.

**EDM certifies stationarity, not optimality**, so EDM alone is not enough;
and the value alone is not enough, because a fit that never moves reproduces
its start perfectly. NLL is compared only across fits of the SAME model.

**`|grad|_inf` is NOT a convergence test for this objective.** The reference
Hessian's eigenvalues span 0.05 to 1e5 (the soft end is `1/sigma^2` for `m_Z`
and `Gamma_Z`, the stiff end is the `K(m)` shape block), so a stopping rule on
the unscaled gradient infinity norm converges the stiff directions and
abandons the soft ones. Four numbers were retracted to this before the rule
was adopted; one fit reported `m_Z = -0.00019 +- 2.32` having never taken a
POI step at all, while `|grad|_inf = 1.42` was `shape5` and genuinely
converged.

### 7.2 The minimiser

**scipy `trust-exact` through `rabbit_fit.py` is the campaign minimiser.**

Both TF ports fail the trust-region subproblem at full statistics, in their
own ways: `tf-trust-krylov` (GLTR) loses Lanczos orthogonality at a 3.4e12
condition number and stops 14.7 NLL units up the hill while reporting
convergence; `tf-trust-exact` hits its lambda-search iteration cap
(`maxiter=50`) because `tf.linalg.cholesky` cannot report the index of the
first non-positive-definite leading minor the way LAPACK's `potrf` does, so
the safeguarded bisection has no accelerated bound. On `z_full380_fl` from the
same start:

| run | method | `m_Z` | NLL | EDM |
|---|---|---:|---:|---:|
| reference (scipy `trust-exact`) | — | **-11.064** | **11075392.4657** | 1.8e-18 |
| `tf-trust-krylov` | — | -0.127 | 11075407.1841 | 14.72 |
| `tf-trust-exact` + frozen fix | — | +0.233 | 11075420.3476 | 16.79 |

The native path is **16x faster on the wall** (1206 s against 19 600 s on the
full card) and takes GPU utilisation from 9 % to ~88 %, but it is not usable
here. On a well-conditioned 300 k card it is exact: device vs host agree to 0
on the value and 1e-15 on the gradient and HVP, and every converged row lands
on the same minimum (NLL to 2.4e-8 of 1.07e7, parameters to 9.8e-7).
`tf-trust-ncg` must never be used on this objective (Steihaug-CG truncates at
the boundary and stops 2.5e3-3.2e4 NLL units high).

**Frozen parameters.** `get_x` stop-gradients a frozen parameter, so its
Hessian row and column are exactly zero and the matrix handed to the
subproblem is singular; `trust-exact` then meets the hard case on every
iteration, never returns an interior Newton step, and converges LINEARLY.
`Fitter.hess_for_minimizer` puts 1 on the frozen diagonal before the subproblem
sees it — exact, since the frozen gradient components are zero, so
`p_frozen = -0/1 = 0`. `scipy_hess` already applies it. **This is NOT the same
as an unconstrained-but-floating parameter**, which also has a zero row and is
PHYSICS: it must be named, not regularised. `Fitter.warn_unconstrained` tests
the ROW (not the diagonal) and names any such parameter once.

`--precondition` is a pure reparameterisation and cannot give a wrong answer,
but it buys nothing against a SINGULAR block; it is untested against the
ill-CONDITIONED one, which is sec. 7.3.

**Warm starts cannot flatter a result.** The acceptance test is entirely about
where a fit ARRIVES. Every row except the controls starts from the lowest-NLL
point any earlier fit of that card reached (`json_to_snapshot.py`,
`seed_rows.sh`); the controls stay cold. Seeding from a stored minimum rather
than from the MC truth also removes the coincidence that made stalled fits
read as perfect closures.

### 7.3 The conditioning blocker

The v-form `K(m)` ladder **ends at 7 terms**, and after three attempts the
reason is localised:

| attempt | outcome |
|---|---|
| 1, un-floored card | NaN in the DENSITY — a trust step drove a shape term to ~-1 and `log` of a non-positive `L_i` is `inf` |
| 2 | never loaded (the card/fitter version skew, sec. 8) |
| **3, floor at 1e-7** | **65 Hessians, NO density NaN — the floor works — then scipy raised `array must not contain infs or NaNs` in the subproblem and the postfit died on `Cholesky decomposition failed, Hessian is not positive-definite`** |

The run's own opening line gives the cause: the **Hessian diagonal spans 0.447
to 6.24e13, fourteen orders of magnitude**. (That line arrives labelled "2
floating parameters have an essentially zero Hessian diagonal ... `m_Z`,
`Gamma_Z`", which is the pre-fix `warn_unconstrained` testing the diagonal
instead of the row — a known false positive. The diagnosis is wrong; the
number is not.)

So the degeneracy above 7 terms is neither the density, nor the floor, nor the
formulation: it is the **conditioning of the 9-term Hessian**.
**Trust-region preconditioning is the single named blocker**, for this and for
`P2X`, and it is UNTESTED. Its validation, when run, is to require an already
converged cell to reproduce to 0.01 MeV.

Related: the initial trust radius of 1.0 is in RAW parameter units on a card
whose natural scales span 1e3 (`sigma(m_Z)` 2-9 MeV, `k ~ 1`, Legendre
coefficients 0.002-0.08), so it is 0.48 sigma for `m_Z` and **227 sigma for
`shape5`** — and at an active trust region the step follows the gradient,
which is largest along the STIFFEST coordinate. On one cell that drove
`shape5` to -0.983 and 323 candidate densities non-positive. `shape5` alone
does it; `m_Z`, `Gamma_Z` and `shape1-4` drive zero. The cell was recovered by
warm-starting it from the inclusive fit's point.

**The positivity floor.** `make_card.py --floor-scale` defaults to **1e-7**
(the reference `FLOOR_SCALE = 1e-9` in `unbinned.py` underflows in float64).
A floor does not paper over a bad model: it makes the objective *evaluable* at
a bad trial point so the trust region can REJECT it instead of the fit dying
on a NaN. 1e-7 against a peak density of ~0.4 is a 2.5e-7 relative bias; 1e-4
is too aggressive (it moved `alpha` to +71 +- 25 MeV). `make_joint_card.py`
now passes `floor_scale` to the hand-built J/psi term, which it previously did
not, so every joint card's J/psi leg had run at 1e-9. Cards whose fit already
converged were NOT rebuilt: a floor change makes NLLs incomparable, and the
ladder's rungs are different models whose `m_Z` values are compared, never
their NLLs.

### 7.4 Standing rules — each was bought with a retracted number

1. **Every fit through `rabbit_fit.py`, `--minimizerMethod trust-exact`.**
2. **Never bin on a reconstructed variable correlated with the residual.**
   Measured: reco leading `pT` `+0.042` (its top tertile sits **+225 MeV above
   its own gen mass**), signed seed->final `dq/p` `+0.113`, reco `|eta|` lead
   `+0.0203`, `maxfraclossp` `-0.034`. Safe: `chi2/ndof` `+0.0004`, `vgf`
   `-0.0009`, `eta_pair` `-0.0001`, gen `|eta|` lead `+0.0004`, `m_gen`
   `-0.014`. Every table states `corr(., z)`.
3. **Never bin on `sigma/m` or `sigma`** — `sigma = sigma_bar(1 + a x)`, so
   the bin is a cut on the residual.
4. **Truth-referenced pull for anything charge-split**: `x = z/(1 - a q z)`,
   else `<q z> = -a` is all you measure.
5. **At mass level subtract the model** — its own odd moment is
   +14.05 / +18.20 / +25.41 across the `eta` bands (`model_odd_mass.py`). At
   track level it is ~0 and raw data is fine.
6. **ceph via `ssh submit50` / `submit51`** — submit82's cephx client is
   evicted, which is why sandbox shells there see Permission denied.

---

## 8. Defects found and fixed

### Fitting machinery
* **`--freezeParameters` did not freeze the STEP.** Freezing was
  `tf.stop_gradient` only, while scipy minimised the full vector, so frozen
  directions were an exactly-null Hessian subspace the trust region walked in:
  measured displacements of the frozen `k` up to 0.126. Fixed by minimising
  over `floating_indices` only (7-test suite). One certified row is affected
  at 1.6e-3.
* **The frozen-diagonal singularity** (sec. 7.2), fixed by
  `Fitter.hess_for_minimizer`. With it, `tf-trust-exact` on `z_n300k` goes
  from 66 iterations at EDM 83 and still crawling to **18 iterations at EDM
  1.0e-15** — textbook quadratic convergence.
* **`tfhelpers.tf_edmval` returned the function, not the value** (GPU branch
  only), so `--diagnostics` on a GPU printed a function object.
* **`tfhelpers.cond_number`'s GPU branch called `tf.linalg.cond`**, which is
  not a TF symbol, so the first `--diagnostics` iteration on a GPU raised
  `AttributeError` — reported as "Minimizer raised" and turned into a fit that
  stopped where it stood. Replaced by the ratio of the extreme singular values.
* **The `--diagnostics` line did not mask frozen parameters**, so
  `--diagnostics --freezeParameters` was a failed fit on GPU and CPU alike.
  `Fitter.log_diagnostics` now masks and swallows anything left with a
  warning: a diagnostic must never be able to fail a fit.
* **`warn_unconstrained` tested the DIAGONAL** and fired on the POIs of any
  card mixing a hit-chi2 term (curvature 7.1e13) with mass POIs (0.06). Fixed
  to test the ROW.
* **The sparse `D` had no deterministic GPU kernel** on the phase-2 path.
* **`plot_closure.py` drew one series twice** (the v form overpainted the m
  form in the m form's colour); every panel it had produced was affected.

### Model
* **The corrections must act on the resolution FLUCTUATION, not on the
  residual.** At the J/psi those coincide; at the Z the window is +-27 sigma
  and the deviation out there is FSR and the Breit-Wigner tail. Fed the full
  `delta`, the exact Jensen map moved the residual by a median 57.7 MeV and up
  to 10.7 GeV against the 20.6 MeV mean shift it exists to apply, and the
  300 k fit ran away to `Gamma_Z = -421 MeV`. `corr_clip` bounded that by
  SATURATING both corrections outside a few sigma, and the clipped answer is
  **not stable in the clip**: `m_Z` swings over 130 MeV and `Gamma_Z` over
  425 MeV across clip 3 / 5 / 10 against an 8 MeV statistical error. The
  fluctuation form is the treatment and `corr_clip` survives only as a
  diagnostic.
* **`ZGammaLineshape.config()` did not carry `vpow`.** The datacard stores the
  provider as its `config()` dict, so every v card silently rebuilt its
  provider in the MASS variable while the term treated its CF as the one in
  `v`. Symptoms: `L_v/(L_m m^p)` ran from 0.026 at the peak to 1.6 in the
  tails instead of 1; the truncation normalisation came out 21x too small
  (0.0461 against 0.9775); the fit ran away to NLL -10.3 M with the trust
  radius collapsing to 4.8e-7. **It was caught by asking the density a
  question it had to answer** — `L_v` must equal `L_m m^p` and `Z_v` must
  equal `Z_m` — not by the fit failing. With the fix: `L_v/(L_m m^p)` =
  0.99997 median, `Z_v` = 0.977515 against `Z_m` = 0.977518. **Anything that
  changes what `_cf_tab` transforms MUST go in `config()`.**
  And the warning that travels with it: the off-switch gate (`--vpow 1e-9`
  against no `--vpow`) **PASSED WITH THE BUG IN PLACE**, at 1.8e-7, because at
  `p -> 0` the two providers are the same object. **An off-switch test is
  necessary and not sufficient**; only a test evaluated AT the working
  exponent with an independent right answer catches it.
* **First-order-corrected density negative for 2 of 3 000 000 J/psi
  candidates** at `sigma/m` ~ 2.3-2.5 %, whose `a_res/(sigma/m) = 1.95`
  against the 1.1-1.3 expected — enough to make the joint fit non-finite at
  its start. Ruled: no coefficient bounds and no density clip. Delta-kernel
  terms (J/psi) move to the **exact residual form**, which is positive by
  construction and coincides with the fluctuation form at a delta kernel;
  wide-kernel terms (Z) keep the fluctuation form with a per-candidate
  positivity check and an exact x-space fallback. Measured on the card that
  failed: `gate_nanstep.py --amax-scan 0` finds **zero** non-positive
  densities on either leg at the default point and four displaced points, with
  `min L_i` between 7.9e-05 and 1.1e-04, so the wide-kernel fallback is not
  needed at this working point. Validated on the J/psi gun, where the two
  forms are genuinely different functionals (741 NLL units apart) and agree on
  `alpha` to **0.00224e-3** against a 0.01e-3 gate. **Its cost**: the J/psi
  leg's NLL moved by 17 108 units, so no phase-2 NLL from before the switch is
  comparable with one after it.

### Infrastructure
* **A card can out-run its fitter, silently.** `read_unbinned_terms_from_h5`
  splats the card's stored `config` JSON into the term constructor as
  `**cfg`, so a card is only loadable by a rabbit **at least as new as the one
  that wrote it**. The submit-side builder ran four commits ahead of the
  Engaging checkout for a day; one of those commits added a single inert key
  (`corr_a_max`, guarded by `if self.corr_a_max > 0.0`) to
  `MassCFTerm.config()`, and **eight cards became unloadable**. Remedied by
  stripping the inert key (`cardkey.py --strip`), which keeps every row of the
  certified table on ONE code version. Certified by a three-row triple on the
  same 300 k card, same freeze, `trust-exact`: the new code with and without
  the key is **bit-identical** in all eleven parameters, the NLL and the EDM;
  old against new differ by one ulp in NLL and 8e-13 MeV in `m_Z` (the newer
  checkout solves a 7-dimensional problem instead of an 11-dimensional one
  with a null subspace) and the minimum is the same. Because the bit-identity
  is exact, this certificate does not have to be repeated if the key
  reappears.
* **Every crashed batch stage reported `rc=0`.** `rc=$?` sat on the same line
  as a `date` command substitution and read the *echo's* status. Fixed; and
  `engaging/rabbit_vmass_batch.sbatch` now echoes the checkout, the freeze
  list and the extra flags per row, so a log records which code and which
  functional a fit ran. Anything that trusted an exit code must be re-checked
  against the HDF5 content, which is what `certtable.py` does.
* **Two jobs running the same stage list concurrently** on one node collide
  with `BlockingIOError ... unable to lock file` on the shared output path.

### Excluded hypotheses, recorded so they are not re-derived
* **The two-component mixture hypothesis is REFUTED on both channels.** The
  gun's `eta` dependence decomposes into two `eta`-independent components split
  on `|seed -> final dq/p|` at its 90th percentile, mixing 2.4 -> 21.2 %. The
  mixing FRACTION reproduces strikingly on the real legs (Z legs 2.68 / 9.69 /
  20.04 %), but **neither component is `eta`-flat**: the OUT component is
  85-90 % of the sample and carries the whole dependence (`chi2` vs
  `eta`-flat 578/2 on the Z legs, 77/2 on the J/psi legs, 72/2 and 102/2 at
  mass level). The two-component picture is a property of the gun.
* **The `a`-coefficient deficit**: two explanations excluded on properly
  conditioned variables. The ionisation share — the Q-matrix `f_ioni` is
  **4.5e-06** on the Z legs and 3.2e-04 on the J/psi legs, so
  `1 + f_hit - f_ioni` is indistinguishable from `1 + vgf` and closes 0.0 % of
  the deficit. The per-leg asymmetry term — on the gen `pT` ratio the deficit
  is a U (-0.080 / -0.001 / -0.082 from most to least asymmetric) where the
  per-leg form predicts monotone growth. En route the Q-matrix shares were
  shown to close exactly (`f_hit + f_ms + f_ioni = 1.000000031`, max deviation
  3.9e-07), unlike the CF-exponent "shares", which are cut-dependent because
  Moliere and Landau have no finite second moment.

---

## 9. Gates

Every one of these is re-runnable and none needs a minimiser except where said.

| gate | requirement | measured |
|---|---|---|
| unit 1 | no correction -> bit-identical in either form | exact |
| unit 2 | density normalises to 1, mean = closed form | 1.0000000000, exact to 1e-10 in all three arms |
| unit 3 | at a DELTA kernel the two correction forms agree | 0.00073e-3 apart |
| unit 4 | bounded at the Z | fluctuation `d_i` median 7.3 MeV, max 15.7 MeV (residual form: median 1979 MeV, max 9.1 GeV) |
| unit 5 | gradient vs finite differences | 3e-10 |
| unit 6 | `c_i = -vgf sigma^2/m` | 7e-18 |
| unit 7 | `set_corrections` == purpose-built terms | bit-identical, all four ladder points |
| **gun (gate 1)** | the real J/psi gun, 299 422 candidates, delta kernel | fluctuation vs residual **0.00087e-3** (`a_res` only) and **0.00224e-3** (both), against 0.01e-3; both within 0.003e-3 of the offline numpy implementation and of the spec's +0.1457 / +0.0559e-3 |
| **gate 1, v form** | the same with `--vpow 1.264` | `\|alpha_v - alpha_m\|` = 0.00000e-3 (`a_res` only), 0.00087e-3 (both); `\|alpha_v - spec\|` = 0.00176e-3 |
| **gate 2 (FD)** | analytic vs central FD on `m_Z`, `Gamma_Z`, `shape1`, `k_ms` | 1.2e-7 against 1e-5 |
| **gate 2b (off switch)** | `--vpow 1e-9` reproduces no `--vpow` | 7.9e-10 (1.8e-7 before the `config()` fix — see sec. 8) |
| **gate 2c (Jacobian identity)** | `L_v = L_m m^p`, `Z_v = Z_m` at the WORKING exponent | 0.99997 median; 0.977515 against 0.977518 |
| **gate 3 (assembly toy)** | `V_toy` consistent with zero | **+1.01 +- 2.30**, 0.44 sigma, EDM-converged (from -1.4 sigma in the m form) |
| gate `hessp` | `hessp(x,p)` vs the assembled Hessian | 3.7e-16 over four random tangents |
| rabbit path | graph vs eager chunk loop; term vs `ChunkedObjective`; Fitter HVP Hessian vs standalone `pfor`; Fitter gradient vs central FD | **0** on the value; <=5e-14 gradient/HVP; 2.4e-15; 1.1e-7 |
| end to end | `rabbit_fit.py` vs standalone `fit.py`, `z_n300k`, 7 free | worst parameter 1.9e-8, errors identical to six decimals; rabbit 135 s at EDM 4.98e-13 against 776.7 s at `\|grad\|inf` 1.7e-3 |

**The gun gate is the ABSOLUTE `alpha`, not the shift.** In `v` the
"uncorrected" term is not the same object as in `m` — the substitution absorbs
the self-consistent width, so a v term with no corrections already carries
most of it. The v form's uncorrected reference sits **+0.13571e-3** above the
m form's, against the +0.1457e-3 that the `a` correction applies in `m`: the
substitution absorbs 93 % of the self-consistent width by construction and the
`a^v = a - p sigma/m` residual applies the remaining +0.01053e-3. Nothing is
double-counted and nothing is lost.

Tests are run as SCRIPTS, not pytest (two use positional args that pytest
reads as fixtures — pre-existing):

```bash
cd calibration_studies/fullscale
for T in test_unbinned_mass test_material_cf test_global_term test_unbinned_norm \
         test_zgamma_kernel test_jensen test_zgamma_shape test_fluctuation; do
  THREADS=16 ./run_tf.sh python3 -u ../../rabbit-vmass/tests/$T.py; done
```

---

## 10. Phase 3 — the card

`make_joint_card.py --material` writes **`cards/joint_mat_v3.hdf5`, 28.369 GB**
in 54.3 s and then re-reads both terms to verify that each parameter list is
the one its configuration implies (118.3 s); 10 min 12 s end to end on
`mit_normal` (16 cores, 250 GB).

| | |
|---|---|
| hit-chi2 quadratic | 20 706 999 candidates, 92 parameters (dense Hessian) |
| J/psi mass term | 645 517 candidates, 110 parameters (42 shared material amounts), 14 832 301 CSR group rows (22.98 groups/candidate), 18.99 GB of exponents |
| Z mass term | 481 020 candidates, 117 parameters (42 shared), 12 365 835 CSR rows (25.71 groups/candidate), 15.83 GB of exponents |
| hit classes | 18; 12.2 (J/psi) and 12.8 (Z) rows per candidate |
| fit vector | 18 hit-resolution parameters + `m_Z`, `Gamma_Z`, `shape1-5` + the 92 calibration parameters; 50 flagged POI |

**Three parameters are unconstrained and the card says so rather than freezing
them** (what a fit floats is a physics decision): `material_pp1_cables` is
touched by **no candidate on any leg**, `material_thermal_screen` and
`material_support_tube` by **< 1 %**, and the hit-chi2 term is blind to those
three plus `material_beampipe`. The whitened quadratic Hessian has
**cond 1.8e32** with 4 directions below 1e-12 of the maximum and **cond 4.2e9
over the rest** — the ill-conditioning is exactly those four blind directions
and nothing else. `material_beampipe` is touched by 99.99 % of candidates and
is fine from the mass side.

Where the material information is, by share of `sum_i max_tau |S|`:
J/psi `material_tec_structure` 26.6 %, `material_tib_support` 16.3 %,
`material_tob_support` 11.3 %, `material_tibtid_services` 8.3 %;
Z `material_tib_support` 28.0 %, `material_tec_structure` 15.1 %,
`material_tob_support` 11.7 %, `material_tibtid_services` 10.9 %.

The command the builder prints:

```
rabbit_fit.py .../cards/joint_mat_v3.hdf5 -o out/ -t 0 --unblind \
    --paramModel ExternalParams bundle:global_params \
    --freezeParameters material_pp1_cables material_support_tube material_thermal_screen
```

**NOT RUN.** And the memory question is open rather than settled: the
"141.4 GB of an H200's 143.8 GB" figure was measured on the phase-2 FULL card
(6.68 M candidates); this card is a subsample (1.13 M candidates, 34.8 GB of
exponents in total), so whether one GPU suffices is one cheap job to
establish, not a known requirement.

**How the group caches are made.** `matres/extract_groups.py` cannot be used
on either production: it rebuilds the per-group CF offline from Geant4 step
records and dies on the missing `ioniurbanidx` (both ran
`exportStepRecords=False`). Both ran `exportCfGroupExponents=True`, so
`cf_inmaker.py pairs --groups` reads the maker's own split into the CSR layout
`MaterialCFTerm` / `make_material_card.py` consume.
`matres/validate_inmaker_groups.py` audits it on 200 000 candidates of each
leg: summing the group rows reproduces the flat exponent to **7.5e-8 relative
on both legs** (the float32 storage floor), and the maker's own float64
`cfmass_grp_closure` reads **1.4e-14**.

**Occupancy** (which decides constrainability): 23.0 material groups per
candidate on the J/psi leg (median 23, p1 16, p99 31, max 35 of 42) and 25.7
on DY.

**`vg_other` is noise, not a remainder.** Measured, `sum_c cfmass_hitv` equals
`cfmass_vgf`: `|vg_other|/vgf` is 1e-6 median and 6e-6 at q99 on both legs, of
BOTH signs. So on two-track candidates the hit classes account for all of
`vgf`. **`MaterialCFTerm` should clip `vg_other` at 0 rather than trust its
sign.** The candidates with `vg_other/vgf ~ 1` are the `vgf = 1` pathologies
(no influence decomposition at all, `sigma_m/m` up to 3828); 11 in J/psi and
62 in DY, all removed by the `sigma_m/m < 0.10` cut.

---

## 11. Pitfalls

1. **`extract.py`'s mass path is blocked on these productions** —
   `exportStepRecords=False` ships `radvgrid` alone and the guard rejects it.
   Use `--no-mass`; the mass side comes from `cf_inmaker.py pairs`.
2. **`chi2ndof` is not stored on the `--no-mass` path**, so the chi2 cut must
   be given to `extract.py` and cannot be deferred.
3. **`tf.gather` / `tensor_scatter_nd_update` gradients are `IndexedSlices`**
   and `tf.autodiff.ForwardAccumulator` cannot differentiate through one. The
   parameter map in `chunkfit.py` is a dense affine map for that reason.
4. **HDF5 datasets are stored FLAT** with an `original_shape` attribute; a raw
   `np.asarray(g[k])` is 1-D. Take arrays off the TERM.
5. **`nm=32768` with `fsr=` is a 16 GB constant.** `make_card.py` defaults to
   `nm=8192` (1.0 GB, `dm = 9.8 MeV`). `nm` 32768 -> 8192 moves `m_Z` by
   0.01 MeV.
6. **`np.savez_compressed` of a multi-GB cache takes minutes**, during which
   the file exists and is a truncated zip. Wait for the reader's `wrote` line.
7. **git refuses to fetch into a checked-out branch** — `stage_eng.sh`
   detaches.
8. The `_norm_z` truncation is evaluated at the STORED class sigma and without
   the fluctuation-form correction: 0.016 MeV, and the second shifts the
   density by `d_i` (~7 MeV) against a window edge 30 GeV away as a candidate
   CONSTANT, so it cannot bias `m_Z`.
9. `rho(shape4, shape5) = -0.964` at detector level — the Legendre basis is
   orthogonal over the Born window, not over the observed spectrum.
10. **Sampling a single candidate's density on a mass grid** by making the
    grid the `mobs` column gives every grid point its OWN `c_i`, `d_i` (they
    depend on `m_i = mobs_i + m_ref`). Over a +-13 GeV grid that is a +-15 %
    spread and shows up as a 3e-5 normalisation deficit. Pin them for that
    test.
11. **`corr_form` is consumed at CONSTRUCTION** — the per-candidate `c_i`,
    `d_i` are baked in. Flipping `self_consistent_sigma` / `jensen_mode` by
    hand no longer switches the correction; call `term.set_corrections(...)`.
12. **`pfor`'s Hessian memory is set by `chunk x nparams`, NOT by the number
    of candidates.** 99 free parameters at `chunk 16384` peaks at 270 GB; the
    same card at `chunk 8192` peaks at 144 GB. A joint card must therefore
    choose its chunk at WRITE time with the Hessian in mind, and it cannot be
    re-chunked afterwards because its per-candidate sparse `D` is sliced per
    chunk when it is written. (`--unbinnedChunk` retunes a card whose `D` was
    not sliced at write time; on one that was, it raises instead of silently
    mis-aligning.)
13. **`trust-exact` needs the full Hessian at EVERY iteration**, which is why
    a 6.68 M joint fit with 99 free parameters is a GPU job.
14. **The graph chunk loop bounds memory only inside a `tf.function`.**
    `tf.while_loop` executes as a plain python loop in eager mode, so calling
    `term.nll` at the prompt holds the whole sample. Every path the Fitter
    uses is a `tf.function`, so this never bites a fit; it bit the gate.
15. **`mit_preemptable` requeues a job by re-running the script from the top.**
    `rabbit_vmass.sbatch` resumes from the snapshot when one is there;
    `FRESH=1` overrides, and a `FRESH=1` job that is requeued restarts from
    the prefit point rather than its snapshot.
16. **`MaterialCFTerm` with CSR group/hit blocks is not graph-chunkable** (it
    indexes a numpy pointer array with `int(...)`); such a term falls back to
    the host loop with a clear message.
17. **The sandwich is the standalone implementation** and rabbit does not
    compute it. It is not optional — the MiNNLO weights alone are a flat
    x1.109 on every error.
18. **The mass and hit-chi2 scores of the SAME candidate are correlated** and
    no extraction stored the cross block, so `fit_joint.py` adds the two
    sandwich meats as if independent. It says so in its own output rather than
    quoting a robust error that assumes it away.
19. **The Engaging SSH master expires after 8 h, silently.** `eng` then prints
    instructions instead of output and every `rsync`/`sbatch` fails with
    `connection unexpectedly closed`. Only a human can fix it: `!eng-master`.

---

## 12. Open items, ranked by their size on `m_Z`

| # | item | size | next step |
|---|---|---|---|
| 1 | **`K(m)` truncation** | **2.4 MeV** over 5 -> 7 in the v form (the number to quote); +26.6 MeV over 7 -> 9 in the m form | the v-form ladder ends at 7 on the Hessian conditioning (sec. 7.3), so **preconditioning (item 6) bounds this**. The structural cure is a theory-predicted `K(m)` — an NNLO parton luminosity (SCETLib+DYTurbo in POWHEG's constant-width scheme) with its uncertainties as nuisances — which also returns the x1.5 / x1.4 statistical penalty |
| 2 | the momentum scale from J/psi (phase 2) | not yet measured | `P2X` descends but does not converge; preconditioning or 2-GPU sharding, not a resubmit. And the J/psi FSR question (sec. 3.4) belongs in front of the first quotable number |
| 3 | the material amounts (phase 3) | not yet measured | card built and verified; establish whether one GPU suffices, then fit with the three unconstrained groups frozen |
| 4 | residual `eta` spread | -12.51 +- 5.90 MeV (2.1 sigma), an upper bound | a band variable cleaner than `max(\|eta_p\|,\|eta_m\|)` needs a gen predictor a card cannot cut on |
| 5 | GN second-order (Box) charge-odd bias | ~4 MeV uncalibrated, ~0.4-0.7 MeV after the J/psi anchors the scale | analytic per-track correction from the exported steps; no new production |
| 6 | trust-region **preconditioning** | bounds items 1 and 2 | UNTESTED. Validation: require an already converged cell to reproduce to 0.01 MeV |
| 7 | the `a`-coefficient deficit | bounded small, unquantified on `m_Z` | none proposed; both leading explanations excluded |
| 8 | post-fit shape | `chi2/ndof = 4.75` over 240 bins | genuine few-% shape mismodelling |
| 9 | `k_ms = 1.0298 +- 0.0042` | 0.5 MeV | the MS **tail** is ~3 % short; phase 3 is its test |

Item 1 is larger than everything else in the list combined, and it is a
**model/analysis effect, not a detector or reconstruction one.**

---

## 13. Compute, measured

| | |
|---|---|
| full Z fit (3.68 M candidates, 11 free) | 38 iterations, **16 940 s** on a preemptable H200 (~4.7 GPU-h), scipy `trust-exact` |
| the same with 9 `K(m)` terms | **4 h 23** |
| joint 500 k + 500 k + hit-chi2, 95 free | **3 h 39**, EDM 6.2e-19 |
| a warm band/cell refit | 10-35 min |
| native TF minimiser | **16x** on the wall (1206 s against 19 600 s) but fails the subproblem at full statistics |
| reference-point Hessian, full card | 420 s host (`pfor`) against 62 s device (`nfree` HVP columns) |
| card sizes | phase 2 full **10.7 GB**; phase 3 material **28.37 GB**, built in 10 min |
| phase-2 Hessian | **141.4 GB** of an H200's 143.8 GB **on the FULL card** |
| exports | ~81 kB/candidate slim; +26 kB with per-group material exponents |
| production | condor **15.6x** slurm on the same payload |
| offline auxiliary extractions | 3.7 M (DY) and 7.9 M (J/psi) candidates in 3 and 17 min on 32 cores |
