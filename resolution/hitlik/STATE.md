# hitlik — a per-track CF likelihood of the TRACK-PARAMETER residual vector

## Purpose

The resolution/material calibration uses a Gaussian hit chi2 to constrain the
material amounts and the hit-class resolutions.  The CF (characteristic
function) machinery of the CVH fit can instead give the EXACT per-track density
of a residual, including the Moliere/Landau tails.  This study builds that
density for the whole reference-state residual VECTOR (not just the scalar
q/p), uses it as a likelihood term, and measures what the full non-Gaussian
densities buy over the Gaussian chi2.

It is the PROTOTYPE, and it is truth-referenced (`r = refParms - genParms`), so
it works on MC only.  The DATA version — the same construction on the post-fit
hit residuals, which needs no gen information — is the sibling study in
`perhit/` (see `perhit/STATE.md`).

Two headline answers:

* the full PDF constrains the material amounts **1.8-2.1x better in variance**
  than the chi2 does, while the chi2 CLAIMS parity (its own quoted error is
  optimistic by 34 % in variance);
* almost all of the gain on the HIT CLASSES comes from the extra COMPONENTS,
  not from the non-Gaussianity (information ratio 18.9 for 4 components over
  q/p alone).

## The object / model

`ResidualGlobalCorrectionMakerG4e` exports, per resolution block `b` (aligned
with `reseigidx`),

    resinfbv[b] = B_b = W5[b]^T dV_b^{1/2}        (5 x 5, row major, padded)
    W5          = Vinv F C E5

i.e. the influence of that block's standardized noise dofs on all five
reference-state parameters.  Hence, exactly,

    r = refParms - genParms = sum_b B_b u_b ,    u_b standardized, independent
    V = refCov              = sum_b B_b B_b^T

VERIFIED: `sum_b B_b B_b^T` reproduces `refCov` element by element to a median
**1.1e-7**, p90 3.4e-6, max 2.3e-4 relative (that is the float32 storage of
`B_b`), and `(B_b B_b^T)_00 = resinfvarv[b]` to 1.4e-7.  `refCov` is stored
UPPER-TRIANGULAR (the lower triangle is literally 0).

**Whitening** is by the LOWER Cholesky factor of `V`, `z = L r` with
`L = inv(chol_lower(V))`, so that `z_0 = r_0/sqrt(V_00)` IS the established q/p
pull and `z_1..z_4` are the successively orthogonalized lambda / phi / d0 / z0
residuals.  The components are NESTED, so "what the vector buys over q/p alone"
is read off directly.  Each component is

    z_k = sum_b a_{b,k} . u_b ,   a_{b,k} = (L B_b)[k, :] ,
    v^(k)_b = |a_{b,k}|^2        (sums to 1 over blocks, exactly)

and its CF is the product over blocks of the SAME block CFs the q/p model uses,
at the block's own effective scalar weight `wstd_k = sqrt(v_pool^(k)/sq2)`.

**Components used: 4** (q/p, lambda, phi, d0).  `z0` is unusable on this
production — see Defects.  Cholesky variance inflation `V_kk/d_k` (median over
2000 tracks) is 1.00 / 1.002 / 1.39 / 3.67 / 5.41, so the nesting is well
conditioned; `cond(refCov)` median 7.5e3.

**The extended-grid trick.**  All four exponent primitives depend on
`(wstd, tau)` through the PRODUCT only, so ONE evaluator call per (block,
group) on `tau_ext = concat_k(TG * wstd_k/wstd_ref)` gives all five components.
Measured 0.63 s/track/worker for 5 components against ~1.9 s for ONE with
`matres`'s ditrack functional: **the cost of the vector is the cost of the
scalar**.

### The three arms (`hitlik_term.py`)

| arm | what replaces the per-(row, group) log-CF exponent | model Var(z) |
|---|---|---|
| `cf` | nothing -- the extracted exponents, the full non-Gaussian densities | 1.013-1.077 |
| `gauss` | `-1/2 kappa2_g tau^2`, `kappa2 = -(16 S(t1) - S(2 t1))/(6 t1^2)` off the SAME arrays (tau^4 term eliminated); imaginary parts dropped | 1.013-1.068 |
| `gaussq` | the variance the FIT used: `thp2` (Rossi) for MS, `ioni_sq2` for ionization, ZERO radiative and delta | **1.00000 exactly** |

`gauss` has the same variance as `cf`, so CF-vs-`gauss` is purely about SHAPE;
`gaussq` IS the fit's own pull model, i.e. the Gaussian hit chi2.

### Parameters

Exactly the parameters `matres` uses, so a joint fit floats ONE set:

* `material_<group>` (42), `k_g` = ln of the group's material amount, entering
  as `S_f = S^fix + sum_g e^{k_g} S_{f,g}`;
* `hitres_<class>` (18), `eps_c` the LINEAR scale of that class's Gaussian
  variance share, `v_i = v_other + sum_c (1 + eps_c) v_{c,i}`.

Field and alignment are NOT parameters of this term: on a truth-referenced
residual they move only the MEAN, and the mean of a whitened residual carries
no material or hit-resolution information.  The term is a pure width/shape
term; field and alignment stay in the quadratic term.

### Two approximations to state with every number

1. **The composite likelihood.**  The whitened components are uncorrelated by
   construction but NOT independent, so the product of marginals drops the
   joint cumulants.  The point estimate stays consistent; the quoted error does
   not, and the SANDWICH is the accounting (`fisher_cmp.py` estimates `J` by
   batch means over batches of whole tracks, so every within-track correlation
   is inside `J`).  Sized below (`xcum`).
2. **The conditioning cut.**  `max_inflat` cuts rows on `V_kk/d_k`.  That is a
   cut on the FIT'S COVARIANCE, not on the residual being measured, so it
   cannot bias the residual distribution.

## How to run

Environment: the offline extraction runs in the `mfs` venv
(`/work/submit/david_w/ZMass/mfs/.venv`); everything that touches rabbit runs
in the TF container through `./run_tf.sh` (binds `/ceph/submit` automatically
when the host can read it).

Inputs:

| what | path |
|---|---|
| production (20-60 GeV mu gun, UL16, both charges, \|eta\| < 2.4) | `/ceph/submit/data/user/d/david_w/ZMass/cvh/resolution_trackres_mugun_ul16_260903x_m0/task_*/globalcor_resclosure_*.root` |
| material groups | `/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/data/materialGroups50.txt` |
| J/psi-gun MASS term (for the joint) | `/work/submit/david_w/ZMass/calibration_studies/resolution/runs/matres/gun_groups_probe.npz` |

Scripts (all in `/work/submit/david_w/ZMass/calibration_studies/resolution/hitlik/`):

```bash
cd /work/submit/david_w/ZMass/calibration_studies/resolution/hitlik
R=/work/submit/david_w/ZMass/calibration_studies/resolution/runs/hitlik

# 1. extraction: tree -> npz (20 000 tracks in 705 s on 26 workers)
./run_extract.sh $R/mugun20kv2.npz -j 26 --max-tracks 20000
#    (adds --validate to re-run the q/p gate; PROD=... selects the production)

# 2. the quadratic (hit-chi2) term over the SAME production, same cuts
./run_quad.sh                      # -> $R/mugun_quad.npz  (318 390 tracks)

# 3. cards and fits (NTRK=6000 tracks, COMPS=0123 by default)
./build_all.sh                     # every card of the ladder
./fit_all.sh cf gauss gaussq joint inj_cf inj_gaussq inj_hit inj_hit_gaussq \
             mass resmass inj_mass inj_resmass inj_joint
#    single steps: ./run_ladder.sh card_<name> ; ./run_fit.sh <name>

# 4. Fisher H and J per arm (the sandwich inputs)
./run_fisher.sh --max-tracks 20000 --nbatch 500 -o $R/fisherHJ20k.npz

# 5. the tables
./run_tf.sh python3 report.py     --fisher $R/fisherHJ20k.npz --quad $R/mugun_quad.npz
./run_tf.sh python3 efficiency.py --fisher $R/fisherHJ20k.npz --cset 0123 \
      --arms cf gauss gaussq --ref cf -o $R/efficiency.npz
./run_tf.sh python3 scaleinfo.py  --npz $R/mugun20kv2.npz
./run_tf.sh python3 tails.py      --npz $R/mugun20kv2.npz --max-tracks 20000
./run_tf.sh python3 cost.py       --npz $R/mugun20kv2.npz
./run_tf.sh python3 recovery.py   --pairs cf=$R/fits/cf:$R/fits/inj_cf ... \
      --param material_tib_support --truth 0.00243951
./run_tf.sh python3 perhit/certify.py --fits $R/fits
./run_tf.sh python3 final_table.py --efficiency $R/efficiency.npz --fits $R/fits

# 6. the assumption-free check: 8 disjoint subsample fits per arm
K=8 NSUB=2500 ./subfits.sh
./run_tf.sh python3 subspread.py --fits $R/fits --k 8 --arms cf gaussq \
      --compare $R/efficiency.npz

# 7. figures
./run_tf.sh python3 plot_hitlik.py --npz $R/mugun20kv2.npz --densities \
      --efficiency $R/efficiency.npz \
      --outpath ~/public_html/ZMass/cvh/260909_hitlik
```

Outputs, `runs/hitlik/` = `/work/submit/david_w/ZMass/calibration_studies/resolution/runs/hitlik/`:

| file | what |
|---|---|
| `mugun20k.npz`, `mugun20kv2.npz` | the extraction (v2 = with the phi wrap; 2.8 GB each) |
| `mugun_quad.npz` | the quadratic term over 318 390 tracks |
| `fisherHJ20k.npz` | H, J and the per-batch gradients, 3 arms x 2 component sets |
| `efficiency.npz`, `report20k.npz`, `subspread.npz` | the sandwich / ratio / spread tables |
| `cards/*.hdf5`, `fits/*/fitresults.hdf5` | the rabbit ladder |

Figures: **`~/public_html/ZMass/cvh/260909_hitlik/`** (18 panels as PDF+PNG plus
`index.php`): `density_c{0..3}[_log]`, `tailclosure_c{0..3}`,
`efficiency_{material,hitres}`, `inforatio_CF_over_gaussq_0123_{material,hitres}`,
`inforatio_CF0123_over_CF0_{material,hitres}`.

## Results

### Densities: the three arms are exact, and `gaussq` IS the chi2

`valdens.py`, 60 tracks, 4 components: all three densities integrate to
**1.000000** with mean **0.00000**; `gaussq`'s variance is **1.00000** to five
decimals for every component — the fit's own Q-matrix decomposition of `refCov`
sums to unity, so `gaussq` is the pull model the quadratic hit-chi2 term
assumes, not an approximation to it.  `cf` is **1.3-7.7 % wider**: the
Rossi-vs-Moliere gap of NOTES 2026-08-16, seen directly as a model variance.

DATA pull variance (2000 tracks): `Var(z_0..3)` = 1.070 / 1.113 / 0.985 / 1.056.

### NLL at MC truth (card build, lower = describes the data better)

| rows | CF | Gaussian (variance-matched) | Gaussian (fit's Q = the chi2) | dNLL/row |
|---|---|---|---|---|
| 20 000 (q/p only) | **28 376.25** | 28 660.35 | 28 643.17 | 0.0133 |
| 80 000 (4 components) | **114 064.76** | 115 332.10 | 115 308.25 | 0.0155 |

### The fits, certified (`certify.py`, 6000 tracks x 4 components)

`NLLred` is `nllvalreduced` at the minimum; EDM is rabbit's own
`1/2 g^T H^-1 g` with the full Hessian.  Values are PHYSICAL:
`k` = ln material amount (the card value x 20, since the card is whitened by
the 0.05 tier prior / 0.0025 card sigma) and `eps` = the linear hit-variance
scale (the card value itself, `--hit-prior 1.0`).

| fit | npar | NLLred(min) | EDM | `k(material_tib_support)` | `eps(hitres_str_N3_lo)` |
|---|---|---|---|---|---|
| `cf` | 60 | 34 159.2629 | 2.08e-12 | -0.02439 +- 0.04083 | -0.13711 +- 0.11688 |
| `gauss` | 60 | 34 513.7254 | 3.23e-10 | -0.07966 +- 0.03949 | -0.20599 +- 0.11464 |
| `gaussq` | 60 | 34 509.5279 | 2.24e-09 | -0.02277 +- 0.04049 | -0.15988 +- 0.11883 |
| `cf_c0` (q/p only) | 60 | 8 469.9422 | **5.63e-01 FAIL** | -0.02043 +- 0.04447 | -0.22190 +- 0.26134 |
| `joint` (residual + quadratic) | 110 | 34 157.2163 | 9.78e-14 | -0.02692 +- 0.04075 | -0.13690 +- 0.11689 |
| `mass` (J/psi-gun mass term) | 42 | -47 432.0275 | 2.04e-14 | -0.02377 +- 0.03787 | -- |
| `resmass` (residual + mass) | 60 | -13 272.0953 | 7.04e-18 | -0.02853 +- 0.03481 | -0.12780 +- 0.11665 |
| `inj_cf` | 60 | 34 159.8998 | 4.89e-13 | -0.04100 +- 0.04038 | -0.14337 +- 0.11648 |
| `inj_gaussq` | 60 | 34 509.9191 | 3.61e-09 | -0.03943 +- 0.03994 | -0.16335 +- 0.11841 |
| `inj_hit` | 60 | 34 159.2767 | 8.70e-18 | -0.02440 +- 0.04083 | -0.21482 +- 0.10643 |
| `inj_hit_gaussq` | 60 | 34 509.5430 | 4.19e-18 | -0.02278 +- 0.04049 | -0.23555 +- 0.10821 |
| `inj_joint` | 110 | 34 157.8838 | 1.81e-12 | -0.04314 +- 0.04031 | -0.14323 +- 0.11649 |
| `inj_mass` | 42 | -47 431.3575 | 9.33e-11 | -0.04506 +- 0.03722 | -- |
| `inj_resmass` | 60 | -13 271.2906 | 2.16e-12 | -0.05407 +- 0.03422 | -0.13107 +- 0.11646 |

13/14 certified at EDM < 1e-3; `cf_c0` is the known failure (below), and
`quad` was never fitted.  The injection tables further down use the same
physical units.

### THE HEADLINE — the sandwich: what the chi2's error ACTUALLY is

The Gaussian-likelihood estimator of a width is a weighted sum of `z^2`, whose
variance is set by the FOURTH moment of the real residual.  Its actual error is
therefore `(H+P)^-1 J (H+P)^-1` with `H` and `J` both evaluated on the real
data (`efficiency.py` on `fisherHJ20k.npz`, 20 000 tracks, 4 components, MC
truth, the parmtype-15 tier priors both terms carry in every fit).

| | material groups | hit classes |
|---|---|---|
| sandwich/quoted, CF (1.000 = the model is right) | **0.974** | **1.052** |
| sandwich/quoted, Gaussian chi2 | **1.338** | **1.222** |
| EFFICIENCY `sigma^2(chi2, ACTUAL)/sigma^2(CF, actual)`, marginal | **1.825** (1.616-2.124) | **1.204** (1.069-1.607) |
| ... prior-free (standalone) | **2.093** (1.521-3.707) | **1.221** (1.055-1.565) |
| what the chi2 CLAIMS (quoted/quoted) | 1.028 | 0.969 |

Per material group (marginal, 20 000 tracks): `fpix_support` 2.48,
`bpix_services` 2.17, `tibtid_services` 2.12, `pixel_patch` 2.07,
`tec_services` 2.04, `other` 1.83, `tid_support` 1.80, `tob_services` 1.77,
`tib_support` 1.63, `tec_structure` 1.62, `tob_support` 1.54,
`bpix_support6` 1.48.

**So the full PDF DOES constrain better — by 1.8-2.1x in variance (1.35-1.45x
in sigma) on the material amounts and 1.20-1.22x on the hit classes — while the
chi2 claims parity.  The CF's own quoted error is right to 3-5 %; the chi2's is
optimistic by 34 % (material) / 22 % (hit classes) in variance.**
Figures `efficiency_{material,hitres}.pdf`.

The analytic version (`scaleinfo.py`, pure scale limit): the measured DATA
kurtosis is 4.87-11.93, so a Gaussian scale estimator's variance exceeds its
quoted one by `(k4+2)/2`, and against the CF's own `I(ln s)` the efficiency is
`[(k4+2)/4] x I_CF`:

| component | kurtosis | k4 | Gauss actual/quoted (var) | EFFICIENCY |
|---|---|---|---|---|
| q/p | 5.504 | 2.504 | 2.252 | **1.852** |
| lambda | 6.569 | 3.569 | 2.785 | **2.595** |
| phi | 11.931 | 8.931 | 5.466 | **4.375** |
| d0 | 4.866 | 1.866 | 1.933 | **1.813** |

and it is cut-dependent exactly as it should be: trimming at |z| < 5 / 4 / 3
(removing 0.1 / 0.2 / 0.8 % of the rows) drops `(k4+2)/2` to 1.3 / 1.2 / 1.0.
The Gaussian estimator's excess variance lives entirely in the tail the chi2
does not model.

### The NOMINAL information ratio — what each model CLAIMS (not the answer)

`I_CF/I_chi2` from the nominal Fisher matrices alone, marginal, 4 components:

| parameter | ratio | | parameter | ratio |
|---|---|---|---|---|
| `material_bpix_support6` | 0.647 | | `hitres_pix_x_q0` | 0.358 |
| `material_bpix_services` | 0.724 | | `hitres_pix_x_q2` | 0.689 |
| `material_tib_support` | 0.751 | | `hitres_pix_y_q2` | 0.697 |
| `material_fpix_support` | 0.851 | | `hitres_str_N2_lo` | 0.732 |
| `material_tec_structure` | 0.902 | | `hitres_str_N3_hi` | 0.742 |
| `material_tob_support` | 0.922 | | `hitres_str_N3_lo` | 0.761 |
| median over the 10 informative groups | **0.85** | | median over 18 classes | **0.737** |

The CF's nominal information is LOWER, and `scaleinfo.py` says why from first
principles: `I(ln s) = E[(1 + z dlnp/dz)^2]` is 2 for a Gaussian and
`2 nu/(nu+3) < 2` for a Student-t — a heavier tail carries LESS information
about its own scale.  GATES: exact `N(0,1)` -> **2.0000**; `t_5` -> **1.2500**
against the analytic 1.25.  On the actual row-averaged densities:

| component | I(ln s) CF | Gaussian | ratio |
|---|---|---|---|
| q/p | 1.6445 | 2.0000 | **0.822** |
| lambda | 1.8658 | 2.0000 | 0.933 |
| phi | 1.6012 | 2.0000 | **0.801** |
| d0 | 1.8764 | 2.0000 | 0.938 |

This is not the decision-relevant number — the sandwich above is — but it is
the whole explanation of the ratio: the chi2 is NOT conservative, it is
optimistic, because it treats the Moliere tail as Gaussian.

### What the residual VECTOR buys over the q/p scalar (`I_4comp/I_qp`)

| | material | hit classes |
|---|---|---|
| median | 1.00 (`bpix_support6` **4.3**, `bpix_services` 1.7, `tib_support` 1.5) | **18.9** |
| range | 1.0 - 4.3 | 3.0 - 538 |

The hit-class resolutions are essentially UNMEASURABLE from `q/p` alone
(sigma(eps) 0.17-0.95) and well measured from the four components (0.024-0.10).
That is the dominant gain of the whole exercise, and it comes from the extra
COMPONENTS, not from the non-Gaussianity.

### Tails (20 000 rows per component)

| P(\|z\| > 4) | data | CF | data/CF | Gaussian chi2 | data/chi2 |
|---|---|---|---|---|---|
| q/p | 0.00235 | 0.00248 | **0.95** | 0.00006 | **37** |
| lambda | 0.00215 | 0.00069 | 3.10 | 0.00006 | 34 |
| phi | 0.00365 | 0.00287 | **1.27** | 0.00006 | **58** |
| d0 | 0.00205 | 0.00068 | 3.03 | 0.00006 | 32 |

At 5 sigma the chi2 is wrong by **1650-3400x** while the CF is within 0.8-4.
In the core (|z| > 1, > 2) all three arms agree with the data to 3-20 %.

### The composite-likelihood approximation, SIZED (`xcum`)

Fourth-cumulant correlation
`kappa(z_j,z_j,z_k,z_k)/sqrt(kappa_4(j) kappa_4(k))` of the MS blocks, per
track (median):

| pair | q/p-lam | q/p-phi | q/p-d0 | lam-phi | lam-d0 | phi-d0 |
|---|---|---|---|---|---|---|
| median | 0.138 | 0.088 | 0.095 | 0.329 | 0.279 | **0.712** |

`q/p` is nearly independent of the other three (0.09-0.14); `phi` and `d0`
share 71 % of theirs (the same bending-plane measurement at two lever arms).
So the product form is a good approximation for what `q/p` adds and a
noticeable one for the `phi`/`d0` pair: their joint information is over-counted
and the honest reading is that `phi + d0` together contribute less than the
product form says.  The sandwich is what prices it.

### Injection recovery (`material_tib_support` x1.05 material)

Injected `k` = +0.0487902 = +5.000 % material; values are PHYSICAL `k`.  SIGN:
scaling the card's exponents by `exp(+k)` declares the model at `k = 0` to
already have that much material, so the MLE moves by `-k`; the test is
`|shift|` against `|truth|`.  PRIOR SHRINKAGE: the group carries its 0.05 tier
prior and the injection is about one prior sigma, so the posterior moves by
`f = 1 - sigma_post^2/sigma_pri^2` of it.

| channel | baseline | injected | shift | /truth | f_pri | **corrected/truth** | pull | leak rms |
|---|---|---|---|---|---|---|---|---|
| residual vector, CF | -0.02439 +- 0.04083 | -0.04100 +- 0.04038 | -0.01662 | -0.341 | 0.348 | **0.979** | -0.01 | 0.026 |
| residual vector, fit's Q | -0.02277 +- 0.04049 | -0.03943 +- 0.03994 | -0.01666 | -0.341 | 0.362 | **0.943** | -0.04 | 0.027 |
| residual + quadratic (110 params) | -0.02692 +- 0.04075 | -0.04314 +- 0.04031 | -0.01622 | -0.332 | 0.350 | **0.949** | -0.04 | 0.024 |
| J/psi-gun MASS term alone | -0.02377 +- 0.03787 | -0.04506 +- 0.03722 | -0.02128 | -0.436 | 0.446 | **0.978** | -0.02 | 0.052 |
| **residual + mass** | -0.02853 +- 0.03481 | **-0.05407 +- 0.03422** | -0.02554 | -0.524 | 0.532 | **0.985** | -0.02 | 0.044 |

Every channel recovers the 5 % injection to 2-6 % with a sub-0.05-sigma pull;
the joint is both the tightest and the most accurate, and the two objectives
agree on ONE material amount.  Largest leakage is `material_tob_support` at
-0.11 sigma (residual) to -0.27 sigma (mass).

**Hit class**, `hitres_str_N3_lo` variance x1.10 on the data side.  `hit_mode`
is LINEAR, so the expected shift is
`-eps_inj/(1+eps_inj) x (1+eps_base)`:

| arm | baseline | injected | shift | expected | raw | f_pri | corrected/truth |
|---|---|---|---|---|---|---|---|
| CF | -0.13711 +- 0.11688 | -0.21482 +- 0.10643 | -0.07771 | -0.07844 | 0.991 | 0.989 | **1.002** |
| fit's Q | -0.15988 +- 0.11883 | -0.23555 +- 0.10821 | -0.07567 | -0.07637 | 0.991 | 0.988 | **1.003** |

Pull below 0.01 sigma, leakage below 0.005 sigma on every other parameter.

### The assumption-free check: 8 DISJOINT subsample fits per arm

`subfits.sh` + `subspread.py`, K = 8 x 2500 tracks x 4 components, each
subsample FITTED for real in both arms; `sigma_full = spread/sqrt(K)`.
Assumes nothing — in particular not `H = J`.

Convergence is itself a result: **CF 7/8 converged, the Gaussian only 5/8**
(dropped at EDM > 1e-6: `sub_cf_3` 5.0e-1; `sub_gaussq_1` 1.1e-5,
`sub_gaussq_3` 7.2e-2, `sub_gaussq_6` 1.4e-2).  A misspecified likelihood is
also a worse-conditioned one.

| | EMPIRICAL (spread) | SANDWICH | agree? |
|---|---|---|---|
| **hit classes** (prior does nothing here, S/Q ~ 1) | **1.362** (16-84 % 1.05-3.36) | **1.301** | **yes** |
| material groups | 3.159 (2.12-4.23) | 1.811 | same direction |

Read the hit classes as the validation: there the subsample errors are far
below the 1.0 prior and the empirical 1.36 matches the sandwich 1.30 within the
29 %/sqrt(10) precision of the median.  For the MATERIAL the 2500-track
subsamples are strongly prior-dominated (`S/Q` 0.09-0.77), so the empirical
3.16 is measured in the shrunk regime and exaggerates; **the sandwich at full
statistics, 1.81, is the number to quote.**

A 2000-resample BOOTSTRAP over the stored per-batch gradients reproduces the
sandwich to 1.0007 / 0.9992 / 0.9987 for `cf` / `gauss` / `gaussq` — as it must
algebraically at one Newton step, so it validates the linearisation and the
batch independence, not the sandwich itself.

### Why the quadratic term adds nothing on THIS production

The `joint` fit (residual vector + the quadratic hit-chi2 over all 318 390
tracks, 110 parameters, EDM 9.8e-14) gives material values and errors
indistinguishable from the residual-only fit (`tib_support` `k` = -0.0270 +-
0.0408 against -0.0244 +- 0.0408).  Measured directly on `mugun_quad.npz`: over
318 390 tracks the quadratic constrains `material_tib_support` to sigma
**0.289 standalone** and **0.0493 marginal against its 0.050 prior** — ~3 % of
the prior's information, nothing.

The reason is structural: this production has `gradllv` and `hesspackedv` but
**no `hessvaridxv`/`hessvarv`**, so its quadratic term differentiates a
material group's MEAN LOSS and never its WIDTH.  A single-track hit chi2 has
almost no mean-loss lever on the material amount; the `matres` number
(sigma 0.0154 on 299 069) came from a TWO-TRACK gun whose objective carries the
mass constraint.  So on this sample "residual term vs quadratic term" is really
"width information vs no width information".  Re-running on a production with
`exportVarianceGrads` is what makes the comparison meaningful.

### Cost (4000 tracks x 4 components, 1 CPU, nt = 64)

| | | per row | per track |
|---|---|---|---|
| NLL | 120 ms | 7.5 us | 29.9 us |
| NLL + gradient | 458 ms | 28.6 us | 114 us |
| one HVP | 2.27 s | 142 us | -- |
| full Hessian (60 HVPs) | 136 s | | |
| card in memory | 0.367 GB | | 91.6 kB |

One NLL+gradient over this production (320 k tracks) is **37 s**; over
7 M Z legs + 34 M J/psi legs = 41 M tracks, **1.3 h CPU** (one HVP 6.5 h CPU).
The term is a dense `(chunk, nt)` matmul per family, so a GPU is 20-60x that —
the same order as the mass term.

### Export bill

What this prototype consumed came from a production with `exportStepRecords_`
ON, i.e. the 430 kB/candidate raw mode:

| branch | what it is | bytes/track |
|---|---|---|
| `resinfbv` | `B_b` = `W5^T dV_b^{1/2}`, 5x5 per block | 68 x 25 x 4 = **6.8 kB** |
| `msmoliv`/`ioniurbanv`/`radstepv`/`radstepspecv` | the raw step records | ~430 kB |
| `refParms`, `refCov`, `genParms` | the residual and its covariance | 140 B |
| `reshitidx` + `hitDetId`/`hitUProj`/`clusterSizeX`/`clusterChargeBin` | the hit classes | ~0.5 kB |

The raw step records are what makes this offline-only.  The maker already
solved that once for the q/p functional (`exportCfExponents_`,
`cvhcf::trackExponents`, 1.4 kB/candidate); the vector version is the SAME
evaluator run at the `n_res` weights `w^(k)_b = sqrt(v^(k)_b/sq2)` — both `B_b`
and `refCov` are already in scope at the `W5` block of
`ResidualGlobalCorrectionMakerG4e.cc`, and by the extended-grid trick those
weights are ONE evaluator pass on a concatenated tau grid.

`cost.py`'s bill at the measured multiplicities (float32, 13.7 groups and 9.0
hit classes per (track, component) after pruning at 1e-3, rank-16 tau PCA):

| variant | per track | at 41 M tracks |
|---|---|---|
| `q/p` only (status quo) | 6.7 kB | 0.27 TB |
| 4 reference-state components | **26.7 kB** | **1.10 TB** |
| 5 components | 33.4 kB | 1.37 TB |
| 18 per-HIT residuals | 120.3 kB | 4.93 TB |

## Defects found and fixed

1. **`genParms[4]` (the gen `dsz`) is identically wrong.**
   `ResidualGlobalCorrectionMakerG4e.cc:1147` computes it as
   `(vtx.z() - myBeamSpot.z()) * pt/p - (transverse)` with
   `myBeamSpot = bsH->position(vtx.z())`, and `BeamSpot::position(z)` returns
   `Point(x(z), y(z), z)` — so `vtx.z() - myBeamSpot.z()` is **identically 0**
   and `genParms[4]` is the transverse term alone (~1e-4 cm) while
   `refParms[4]` is the fitted z0 (up to several cm).  Measured on 2000 tracks:
   `Var(z_4) = 4.0e6`, median `z_4 = +375`.  **This prototype therefore uses 4
   components.**  FIX: `vtx.z() - bs.z0()` (or store the gen z0); done in the
   maker for the `perhit` production, where all five components are usable.
2. **`phi` is an angle and the residual was not wrapped.**  `genParms[2] =
   g->phi()` is in `(-pi, pi]` and the fitted `refParms[2]` is not, so a track
   near the branch cut has `r_2 = +-2 pi = 3.7e4 sigma(phi)`.  8 tracks in
   20 000 (0.04 %), `max|z_2| = 24 011`, which alone made `Var(z_2) = 2.9e4`
   and, through the Cholesky nesting, `Var(z_3) = 9.7e4`.  FIXED in
   `extract_res5.py` (`r[2] = (r[2] + pi) % 2pi - pi`); better in the maker,
   where a `genParms` in the fit's own convention costs nothing.
3. **The Cholesky conditioning needs a guard.**  `V_kk/d_k` has a tail; it is
   exported as `inflat` and `hitlik_term.load(max_inflat=...)` cuts on it.  It
   is a cut on the FIT'S COVARIANCE, so it cannot bias the residual.
4. **`globalfit/extract.py` rejected single-track productions.**  The
   unconditional `Jpsi_jacMass` guard fired even under `--no-mass`; moved
   inside the mass branch.  The quadratic term over the 318 390 mu-gun tracks
   then builds in 45 s.
5. **`tails.py` integrated one trapezoid over the union of the two tails**,
   adding a spurious slab across the core.  The two tails are now integrated
   SEPARATELY.
6. **A q/p-only card must freeze the hit classes.**  `cf_c0` does not converge
   (EDM 0.56): with `q/p` alone the 18 hit classes are nearly unconstrained.
   It is an information deficit, not a conditioning one, and rabbit's
   trust-region preconditioner settles it: `--precondition --preconditionParams
   '.*' --preconditionBlocks none --preconditionTransform spectral` whitens the
   60-parameter block from condition number **7.78e7 to 1** and buys EDM
   0.563 -> 0.157 (`tf-trust-krylov`) or **0.101** (scipy `trust-exact`, at a
   lower NLL, 8469.9285 against the unpreconditioned 8469.9422) -- still two
   orders above the 1e-3 threshold. Adding `--stallRelTol 1e-4` makes it
   restart four times, each rebuilding the transform where the fit has got to,
   and it stops at 8469.9585 / EDM 0.579: no better. `--freezeParameters
   'hitres_.*'` certifies the card immediately at **EDM 4.2e-20**
   (NLLred 8477.3032, `k(material_tib_support)` -0.02287 +- 0.04089), which is
   the fix.

### Standing rule — ONE unit convention, and no unit flags

A card may float a rescaled variable for minimiser conditioning, but every
object that LEAVES a term — Fisher/Hessian/score matrices, fitted values and
errors, priors, injected amounts, tables — is in PHYSICAL units: `k`, the log
material amount of a group, and `eps`, the linear variance scale of a hit
class.  The conversion factor is read off the object that carries it
(`matres/groups.py`: `card_group_units`, `term_units`, `card_units`), so no
tool takes a unit flag and the prior every tool applies is the parmtype-15
tier prior itself (`gprior` = 0.05 for a material group, 1.0 for a hit class).

A prior 20x too tight makes `(H+P)^-1` collapse onto the prior and the
informativeness test `sq < 0.98 pv` then rejects every material group — an
empty material table is the signature of a mismatched prior.  Unit check:
`material_tib_support`'s CF quoted sigma from the cached `H` is **0.0324**,
the same scale as the fitted `k = -0.0244 +- 0.0408`.

## Open items

1. **The `z0` component** is unusable here (defect 1).  It is fixed in the
   maker and measured in `perhit/`, where it is over-dispersed
   (`Var(z_4) = 1.928`) — an open physics item there, not here.
2. **The residual-vs-quadratic comparison needs a production with
   `exportVarianceGrads`**, so the quadratic term carries WIDTH information and
   the comparison stops being trivially won.
3. **The DATA version** needs the residual about the FITTED track instead of
   the gen one: per track `r_h` (the `n_res` components), `B_b`
   (`n_res` x 5 per block) and `cfres_*` (`n_res` x nfam x NTAU).  That is the
   `perhit/` study, which is done; this prototype remains the MC-truth
   reference it is validated against.
