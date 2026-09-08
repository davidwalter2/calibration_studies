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
