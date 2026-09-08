# hit-class LOCATION bias — STATE

Task: measure each hit class's residual LOCATION (not just its width/shape),
propagate the odd part through the per-hit influence weights with NO free
parameter, and compare against the measured track-level and mass-level odd
moments.

## Step 0 — provenance and the exact mechanism (DONE 2026-09-08)

### The two productions and why the study needs BOTH

No single production carries the truth residual AND the influence weight:

| production | `dxrecsim` (truth residual) | `resinfv/resinfbv` (influence) | class vars |
|---|---|---|---|
| `hitres*_<gun>` (`fitFromGenParms=True`, `doRes=False`) | YES | no | YES |
| `resolution_trackres_*` (`fitFromGenParms=False`, `doRes=True`) | no | YES | YES |

`fitFromGenParms=True` is what BOOKS the per-hit validation branches and
`doRes=False` is what removes the parmtype 8-11 blocks, so the two are
mutually exclusive by construction of the maker. The join is therefore on the
hit's own OBSERVABLES, which both trees carry: `hitDetId`, `hitUProj`,
`clusterSizeX`, `clusterChargeBin`, `clusterOnEdge`, `clusterSizeY`,
`hitPitch`, `hitThickness`, `localdxdz`, `localdydz`.

**The productions named in the brief** (`resolution_trackres_mugun260807*`,
`*_csvar` of 260829) have NO per-hit branches at all (`MISS=[dxrecsim, dxerr,
hitDetId, hitUProj, clusterSizeX, clusterChargeBin, reshitidx]`), so they
cannot be used. The usable pair, matched in gun kinematics (mu, pT 20-60 GeV,
the Z-momentum arm) is

* residuals: `hitres2_mugun_ul16` (40 tasks) [+ `hitres_mugun_ul16` as a
  statistics/duplication check], IDEAL geometry (`useIdealGeometry=True`,
  `run_local_hitres.sh`), default OAE field, GT `150X_mcRun2_asymptotic_v1`
  -> any nonzero residual LOCATION is a CPE bias, not misalignment;
* influence: `resolution_trackres_mugun_ul16_260903x_m0`, the tight-stepper
  cache whose per-eta charge-even odd moments (-4.88 / -3.58 / +0.14)e-3 are
  the target.

### The propagation, exactly

`delta(q/p) = sum_b w_b . n_b` with `v_b = w_b^T dV_b w_b` (maker header
`ResidualGlobalCorrectionMakerBase.h` ~line 897). Verified numerically on
track 0 of the trackres production: for rank-1 hit blocks
`v_b = w_b[0]^2 dxerr^2` to 5e-5 relative. Hence a location bias `mu_b` in
PULL units gives

        delta z_b = s_b sqrt(v_b)/sigma * mu_b = s_b sqrt(hitamp2_b) mu_b

with `s_b` the BENDING SENSE = the sign of the fit's q/p response to a
positive residual in that module's local frame. It is charge-INDEPENDENT
(signed curvature is a geometric property of the trajectory), which is why a
fixed local bias produces a charge-EVEN curvature bias -- the observed
symmetry of the target.

`s_b` must come from `resinfbv` (row 0 = `B_b[0,j] = s_b sqrt(v_b)` for a
rank-1 block), NOT from `resinfv[b][0]`: for a 2-D PIXEL block the parmtype-8
and parmtype-9 entries SHARE the block range, so `resinfv[b][0]` is the
local-x weight for both and using it for the y entry is wrong by a factor
200 (measured).

### Sizes already visible (2000 events of one `hitres2_mugun_ul16` file)

MEAN pull `(rec-sim)/sigma` per subdetector, ideal geometry:

| subdet | n | median pull | mean pull | rms |
|---|---:|---:|---:|---:|
| BPix | 3745 | +0.0786 | **+0.1125** | 1.022 |
| FPix | 741 | +0.0322 | +0.0613 | 1.038 |
| TIB | 7154 | +0.0253 | +0.0166 | 0.961 |
| TID | 3290 | +0.0594 | +0.0201 | 0.954 |
| TOB | 7809 | +0.0245 | +0.0120 | 1.003 |
| TEC | 11615 | -0.0020 | +0.0077 | 0.949 |

So the locations are NOT zero, they are subdetector-dependent, and BPix is
0.11 sigma_CPE. `hitres_classes.py:103` throws exactly this away.

## PRE-REGISTERED PREDICTIONS (written BEFORE the per-class measurement)

The coordinator's extra target: at fixed eta, the poorer-resolution half
gives m_Z **-40.9 +- 4.6** (barrel) against -4.6 +- 3.7 for the better half,
and **+65.9 +- 8.7** (endcap) against +34 for the band -- degraded-hit-content
tracks carry the bias in BOTH regions with OPPOSITE signs.

Stated in advance, from the mechanism alone:

* **P1** The bending-sense class location `<s mu>` is nonzero and COHERENT
  (does not average away over modules) in the barrel subdetectors, because
  the Lorentz drift there is `E x B` with `E` along the module normal
  (radial) and `B` along `+z`, i.e. a displacement of FIXED GLOBAL azimuthal
  sense, which is exactly the bending coordinate.
* **P2** Its sign for the DEGRADED classes (single-strip `N=1`, pixel
  edge / single-column, large `|uProj|`) is OPPOSITE between the barrel
  (BPix / TIB / TOB, local x = r-phi) and the forward subdetectors
  (FPix tilted blades, TID / TEC radial strips measuring local phi with the
  `yAxisOrientation` handedness flip).
* **P3** Therefore the propagated track-level `delta z` has opposite sign in
  the two regions for the degraded-hit-content tracks, reproducing
  -40.9 (barrel) against +65.9 (endcap).

P2 is the falsifiable one: if every subdetector's degraded classes have the
SAME bending-sense sign, the mechanism cannot produce the observed sign
reversal and the hypothesis fails on this point.

## Next
Step 1: extract per-hit residual locations + densities per class (18 classes
and the finer subdet / layer / side / local-coordinate / signed-angle key).

---

## Step 1 — the per-class residual LOCATIONS and densities (DONE 2026-09-08)

`t4_step1.py` -> `out_step1.txt`. `hitres2_mugun_ul16`, 1 375 697 hits with a
matched PSimHit, IDEAL geometry, gen-anchored. Figures in
`~/public_html/cvh/260908_hitclassbias/` (date-derived per the standing rule;
the brief said 260909).

**T1. per subdetector, local frame, pull units:**

| det | n | median | mean | +- | rms | skew |
|---|---:|---:|---:|---:|---:|---:|
| BPix x | 148 985 | +0.0930 | **+0.1193** | 0.0027 | 1.028 | +0.127 |
| FPix x | 30 149 | +0.0285 | **+0.0400** | 0.0061 | 1.063 | +0.096 |
| TIB x | 283 258 | +0.0039 | +0.0021 | 0.0018 | 0.978 | -0.008 |
| TID phi | 135 812 | +0.0002 | -0.0028 | 0.0026 | 0.962 | -0.010 |
| TOB x | 308 215 | +0.0017 | +0.0017 | 0.0018 | 1.007 | -0.010 |
| TEC phi | 469 276 | +0.0003 | -0.0001 | 0.0014 | 0.944 | +0.005 |
| BPix y | 148 934 | +0.0048 | +0.0021 | 0.0027 | 1.050 | -0.003 |
| FPix y | 30 150 | +0.0044 | +0.0038 | 0.0055 | 0.960 | +0.030 |

So the LOCATION is a PIXEL effect and nothing else: BPix +0.119 sigma_CPE
(BPix-1 alone **+0.227**, skew **+0.302**), FPix +0.040, and every strip
subdetector consistent with zero at +-0.002. It is EVEN in the incidence
angle (BPix runs +0.34 / +0.23 / +0.13 / +0.11 / +0.09 | +0.09 / +0.10 /
+0.09 / +0.16 / +0.17 across signed `dx/dz`), so it is not an uncorrected
angle-linear drift term.

Largest class locations: `pix_x_q2` +0.121, `pix_x_q3` +0.119 (BPix alone
+0.134), `pix_x_q1` +0.102; pixel single-ROW clusters +0.034 with skew +0.18.
Degraded STRIP classes are all null: `N=1` -0.0032 +- 0.0020, `N>=4`
+0.0012 +- 0.0022, `uProj>0.5` +0.0021 +- 0.0032.

## Step 2 — the no-free-parameter prediction (DONE): IT DOES NOT REPRODUCE

`t2_predict.py`, `t3_keys.py`, `t6_final.py`. The extraction reproduces the
target sample exactly (319 854 tracks, 6 219 371 hit blocks, charge-even odd
moments -4.89 / -3.57 / +0.12 against sec. 0f.37's -4.88 / -3.58 / +0.14).

| band | measured (charge-even, u=0.05) | PRED location | PRED skew | PRED total |
|---|---:|---:|---:|---:|
| barrel | **-4.89 +- 2.46** | +0.86 | -0.001 | **+0.86** |
| middle | **-3.57 +- 2.71** | +1.59 | -0.002 | **+1.58** |
| endcap | **+0.12 +- 2.75** | +1.39 | -0.001 | **+1.39** |

(units 1e-3). Stable at +1.0 to +2.8 under EVERY join key tried -- subdet,
subdet x layer, orientation group, orientation group x signed angle, per
module, and each of those crossed with the 18 classes. The skew channel is
1e-6, because at u = 0.05 the probe is a mean probe (coefficient -0.0394 on
kappa3 against +0.867 on the mean).

**How far the locations would have to be wrong** (`t9_required.py`,
constrained least squares on the 8 subdet x coordinate locations):

| group | measured | +- | REQUIRED | shift |
|---|---:|---:|---:|---:|
| BPix x | +0.1193 | 0.0027 | -0.261 | -143 sigma |
| TIB x | +0.0021 | 0.0018 | +0.096 | +51 sigma |
| TOB x | +0.0017 | 0.0018 | +0.088 | +48 sigma |
| TEC phi | -0.0001 | 0.0014 | -0.098 | -71 sigma |

chi2 = 33 497 for 3 constraints (183 sigma). No single group can do it
either: BPix alone would need -0.549 and would then give -4.89 / -6.32 /
-4.42 against the target's +0.12 in the endcap.

**Verdict: the CPE class LOCATION bias is measured, is a pixel-only effect of
+0.12 sigma_CPE, contributes +1.0e-3 to the charge-even odd moment with
almost no eta dependence, and cannot be the -4.9 / -3.6 / +0.1 pattern.**

### The PRE-REGISTERED predictions, scored

* **P1 CONFIRMED in form, refuted in size.** The bending sense IS coherent
  once keyed on the module orientation (`<|<s>|>` 0.072 per layer -> 0.761
  per orientation group -> 0.905 per module), and the pixel location IS
  coherent in the local frame (positive in all 12 BPix orientation groups,
  +0.022 to +0.272). But the lever arm is only `<sum_b s_b a_b>` =
  -0.020 / +0.023 / +0.018, so a 0.12 sigma bias buys 1e-3, not 5e-3.
* **P2 REFUTED.** The degraded classes do NOT have opposite bending-sense
  sign between barrel and forward. `L mu` for `pix qbin3` is
  +0.25 / +0.30 / +0.12 (all positive), for pixel single-row
  +0.16 / +0.00 / -0.01, for strip `N=1` +0.03 / -0.00 / +0.00. The
  strip classes have no location to reverse.
* **P3 REFUTED.** The propagated degraded-content contrasts are
  +1.19 / +2.95 / +1.39 (pixel share HIGH-LOW) and -0.19 / -0.81 / +0.35
  (single-strip HIGH-LOW) against measured -3.63 +- 5.16 / -0.48 +- 5.66 /
  -0.11 +- 5.28 and -6.64 +- 5.06 / -1.83 +- 5.76 / -7.00 +- 5.15.

## Step 3 — NOT DONE, by the brief's own condition
The prediction does not reproduce the pattern, so the class densities are NOT
put into the CF hit term and no recipe is handed to the analysis agent.

### The exclusion at every granularity (`t12_chi2.py`)

| key | ncell | central prediction (1e-3) | chi2 (3 dof) | sigma |
|---|---:|---:|---:|---:|
| subdet x coordinate | 8 | +1.00 / +1.38 / +0.98 | 33 497 | 183 |
| subdet x layer | 32 | +0.86 / +1.59 / +1.39 | 1 333 | 36.5 |
| subdet x layer x z side | 48 | +0.86 / +2.10 / +0.58 | 239 | 15.4 |
| + the 18 classes | 275 | +1.16 / +2.40 / +0.96 | 215 | 14.7 |
| orientation group | 440 | +0.48 / +0.07 / +1.15 | 28.5 | 5.3 |
| orientation group x class | 1728 | +1.11 / +0.90 / +1.68 | 34 | 5.8 |

`t11_bound.py` gives the worst-case 3-sigma envelope instead: +-0.31 / 0.22 /
0.19 at subdetector level, growing to +-8.8 / 16.1 / 8.8 at the finest key (it
scales as sqrt(ncell/N)), so the fine keys are statistics-limited for an
ADVERSARIAL arrangement while the chi2 -- how far the locations must actually
move -- is >= 5.3 sigma everywhere.

### The mass level is BLOCKED on a cache field, not on a production
`fullscale/runs/gzpairs_dyv2_n50.npz` carries `hit_cls`, `hit_v` (the variance
share) and `hit_ptr`, but **no sign and no `hitDetId`**. A location needs the
sign, and the orientation-group key needs the DetId. The two-track maker's
`resinfv` already holds the SIGNED mass-projected weights
(`cf_mass_likelihood.build_pairs_tt` docstring), so this is a pairs-cache
RE-EXTRACTION -- add `hit_s` and `hit_detid` -- not a production.

---

# PART 2 — WHICH TRACKS carry the charge-even shift? (2026-09-08)

Scripts `s1_trim.py` .. `s7_figs.py`, output `out_quality.txt`, figures
`skew_trimscan`, `skew_mixture`, `skew_partial_slopes` in the same directory.
Sample: the same 319 850 tight-gun tracks; statistic: charge-even
`<x>` with the truth-referenced pull `x = z/(1 - a q z)`.

## (1) TRIM SCAN — it is a CORE SHIFT, not a one-sided tail

`m_implied(T) = <x>_even,|x|<T / f(T)` with `f(T) = 1 - 2 T q(T)/Q(T)` taken
from the charge-SYMMETRISED data itself (the model's own odd content is
+5e-5 here, so no model is needed):

| band | T=1 | T=2 | T=3 | T=5 | T=10 | T=inf |
|---|---:|---:|---:|---:|---:|---:|
| barrel | -7.74 | -4.44 | -5.68 | -5.73 | -4.84 | **-5.21** |
| middle | -2.34 | -6.52 | -3.53 | -4.43 | -5.06 | **-4.36** |
| endcap | +3.05 | +3.32 | +0.85 | +0.07 | -1.02 | **-1.97** |

Flat from T = 2 within +-2.8e-3. **A one-sided tail is not what this is.**
The `|x|` shells confirm it: in the barrel the -5.2e-3 is built as
-0.20 / -1.55 / -1.79 / +0.21 / -2.12 / -0.26 / +0.87 / -0.37 over
|x| = 0-0.5 / 0.5-1 / 1-1.5 / 1.5-2 / 2-3 / 3-5 / 5-10 / >10, i.e. two thirds
of it inside |x| < 2.

## (2) STEP CONTROL IS NOT THE MECHANISM — it never fires

| variable | census on this sample |
|---|---|
| `nChargeFlipProtect` | **0 for 319 849 / 319 850 tracks** (one track has 1) |
| `chargeHypFlipped` | **0 for every track** |
| `niter` | 2: 81.4 %, 3: 18.5 %, >=4: 0.10 % |

The asymmetric momentum clamp (`clampMomentumFloor` = 2 GeV) and the
two-hypothesis refit never engage on a 20-60 GeV gun -- the floor is 10-30x
away. The per-STEP clamp/backtrack counters (`fitStepClamped_`,
`stepBacktrackEvents_`) are JOB-level in this release, NOT per track, so
`nChargeFlipProtect`, `niter` and the iteration-0-to-final step are the
per-track proxies; all three say the same thing. **An estimator pathology
driven by the step control cannot be the explanation here.** (It is not
excluded at LOW pT, where the 2 GeV floor is close.)

## (3) THE ONE VARIABLE THAT ORDERS IT, AND THE CONDITIONING TRAP

`corr(v, x)` for every conditioning variable (binning rule):

| variable | corr(v,x) | corr(v,\|x\|) |
|---|---:|---:|
| **seed->final dq/p, SIGNED** | **+0.1132** | +0.0018 |
| niter | +0.0052 | +0.0558 |
| \|seed->final dq/p\| | +0.0061 | +0.0537 |
| chi2/ndof | +0.0022 | +0.0935 |
| nValidPixelHits | -0.0041 | -0.0005 |
| nValidHits / n layers / vgf / sigma_rel | <0.001 | <0.007 |

**The SIGNED seed-to-final step is a trap** (`corr = +0.113`; its tertiles run
-169 / -7 / +163 in the barrel, a 332e-3 artefact). The ABSOLUTE step is not.

Charge-even slopes AT FIXED |eta| (v centred in 12 fine |eta| bins),
combined over the three bands:

| variable | barrel | middle | endcap | combined | chi2/2 |
|---|---:|---:|---:|---:|---:|
| nValidPixelHits | -8.28+-3.44 | -1.83+-4.13 | -5.30+-3.05 | **-5.50+-2.00 (2.7s)** | 1.4 |
| n pixel layers | -8.96+-3.61 | -0.55+-4.22 | -3.53+-3.33 | -4.64+-2.12 (2.2s) | 2.5 |
| niter | +13.53+-11.60 | +12.14+-10.21 | +14.59+-7.35 | +13.71+-5.30 (2.6s) | 0.0 |
| \|seed->final dq/p\| | -158+-1694 | +1354+-845 | +442+-215 | +488+-207 (2.4s) | 1.2 |
| n layers / nValidHits / chi2ndof / vgf / sigma_rel | | | | all < 1.2s | |

Joint fit (all six together, per standardised unit, 1e-3): nValidPixelHits
**-4.98+-2.37**, |seed->final| **+6.38+-2.95**, niter +3.06+-2.11, chi2/ndof
+1.40+-3.28, nValidHits +1.91+-2.19, vgf -3.48+-2.16. **Nothing reaches 3
sigma.**

## (4) THE RESULT: the eta dependence is a MIXTURE, not a region effect

Split on `|seed->final dq/p|` at its 90th percentile:

| | barrel | middle | endcap | combined | chi2/2 |
|---|---:|---:|---:|---:|---:|
| **IN** (top 10 %) | +21.28+-29.41 (2.4 %) | +20.69+-15.98 (7.0 %) | +20.53+-7.26 (21.2 %) | **+20.59+-6.45** | 0.0 |
| **OUT** (rest) | -5.03+-2.95 | -6.64+-3.29 | -7.68+-3.37 | **-6.33+-1.84** | 0.4 |

**Both components are flat in |eta|; only their MIXING FRACTION runs
(2.4 % -> 7.0 % -> 21.2 %)**, and the mixture reproduces the band values
exactly: 0.024(+21.28) + 0.976(-5.03) = **-4.40** against -4.42+-3.09;
-4.73 against -4.72; -1.70 against -1.65.

ROBUST: the same at the 80th and 95th percentile (OUT -6.79+-2.03 and
-5.17+-1.83, always chi2 <= 0.6/2), on the RAW `z` instead of the
truth-referenced `x` (OUT -6.16+-1.88, IN +20.90+-6.69), and with the same
sign on the CVH-INTERNAL `|iter0->final|` step, whose exposure to the residual
is smaller still (`corr(v,x) = -0.0016`): OUT -4.64+-1.85 flat, IN +7.03+-7.40.

**What the IN population is**: 2.1x the `sigma_rel` (0.0385 vs 0.0182), 2.72
GN iterations against 2.13, chi2/ndof 1.098 against 0.988, **fewer pixel hits
(1.67 vs 2.30)** at the same total hit count (17.0 vs 17.2), and rms(x) 1.22
against 1.02. It is the high-curvature-error, pixel-poor, hard-to-fit tail --
the same kind of track the mass-level result names, though the two
observables' signs are not directly comparable.

**So the answer to "which tracks":** a ~10 % pixel-poor / high-sigma
subpopulation at **+21e-3** and the remaining ~90 % at **-6.3+-1.8e-3**
(3.4 sigma from zero), BOTH eta-independent. The "eta-dependent charge-even
skew" is better described as an eta-INDEPENDENT bulk shift of -6.3e-3 plus a
subpopulation whose fraction grows with |eta|.

**A single relative curvature bias cannot describe both components**: at
`sigma_rel` 0.0182 the OUT shift is `beta = -1.15e-4` and at 0.0385 the IN
shift is `beta = +7.9e-4`.

**Removing the track-quality variables does NOT flatten the eta pattern**: the
global joint model predicts only -0.84 / -0.38 / +1.29 per band and the
residual is -3.58 / -4.34 / -2.93 -- i.e. flatter, but by construction of the
mixture, not by explaining the bulk. The bulk -6.3e-3 remains.

## CAVEATS
* `|seed->final dq/p|` is RECONSTRUCTED, `corr(.,|x|) = +0.054`. The
  charge-even average is protected to first order against the
  `sigma = sigma_bar(1 + a q x)` selection (the induced term is charge-ODD),
  and the raw-`z` and `|iter0->final|` cross-checks agree, but this is a
  2.4 sigma ordering on a correlated variable: HYPOTHESIS, not established.
* Everything here is 2-3 sigma. The gun is 320 k tracks; the same
  decomposition on the Z and J/psi legs is what would settle it.

## FOR THE OTHER AGENT'S `hit_s` / `hit_detid` WORK — the branch names
Checked directly on `dymc_8p5M_260906_v2` and `jpsimc_20M_260906_v2`:
**`resinfv`, `resinfbv` and `hitDetId` are NOT in those trees at all** (213 and
228 branches; the hit-related ones are `cfmass_hitcls`, `cfmass_hitv`,
`reshitcls`, `reshitidx`, `resinfcovhit`, `resinfvarv`). `resinfv`/`resinfbv`
are booked under `if (exportStepRecords_)` in
`ResidualGlobalCorrectionMakerBase.cc:547-549`, which was OFF for the slim
v2 productions. So a guessed `{prefix}_hits` / `{prefix}_hitdetid` will
silently skip, and the SIGN is not recoverable from these files: it needs a
re-production with `exportStepRecords=True` (the 430 kB/candidate path) or a
small maker change exporting just an int8 sign per block plus `hitDetId`.
Given PART 1's result (the location channel is >= 5.3 sigma from what is
needed), that re-production is probably not worth doing.
