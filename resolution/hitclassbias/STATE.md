# hitclassbias — what sources the charge-even and charge-odd momentum bias of the CVH single-track fit

## Purpose

On an IDEAL-geometry muon gun the CVH fit leaves a nonzero mean in the
truth-referenced momentum pull. This study asks **what makes it**, and how much
of it reaches `m_Z`. Four mechanisms were put on trial:

1. the per-hit CPE **location** bias (each hit class's residual MEAN, which
   `hitres_classes.py` throws away by centring every class on its own median);
2. **incomplete convergence** of the Gauss-Newton iteration;
3. **seed / path dependence** of the estimator;
4. the **second-order (Box) bias** of the converged non-linear least-squares
   estimator.

It is the bias-side companion to `hitlik/` and `hitlik/perhit/`, which measure
hit-resolution WIDTHS; here the LOCATIONS are kept. The deliverable is the size
of each effect on the momentum scale and on the dimuon mass, and whether a
hit-class or a per-track correction is needed.

**The verdict.** (2), (3) and the charge-even projection of (4) are excluded at
1000x below the effect. (1) is real, is pixel-only, and is far too small in the
phi-AVERAGE — but its phi-MODULATED part is a 16-sigma BPix layer-1 ladder-parity
defect of +-36e-3 per track that averages to nothing. (4) is real and large in the
charge-ODD channel: a genuine momentum-SCALE bias of the estimator worth
+1.5 to +3.9 MeV on `m_Z` before calibration. The phi-averaged charge-even bulk
(-3.6 to -3.9 +- 1.8e-3, 2 sigma) is still not established as nonzero.

---

## The object / model

### The sample

| arm | production | tracks |
|---|---|---|
| influence / track level | `resolution_trackres_mugun_ul16_260903x_m0` (20-60 GeV tight-stepper mu gun, 160 tasks) | 319 854 |
| out-of-sample arm | `resolution_trackres_mugun_lowpt_260903x_m0` (2-20 GeV) | 313 245 |
| hit residuals | `hitres2_mugun_ul16` (40 tasks, `fitFromGenParms=True`) | 1 375 697 hits with a matched PSimHit |
| convergence variants | `resolution_trackres_mugun_ul16_260909_conv_{base,tight,damp}` | 319 85x each |

All IDEAL geometry (`useIdealGeometry=True`), default OAE field, GT
`150X_mcRun2_asymptotic_v1`, `CgfQoPMode=0`. On ideal geometry any nonzero hit
residual LOCATION is a CPE bias, not misalignment.

No single production carries BOTH the truth residual and the influence weight:
`fitFromGenParms=True` is what BOOKS the per-hit validation branches
(`dxrecsim`, `dxerr`, `hitDetId`, `hitUProj`, cluster variables) and
`doRes=False` is what removes the parmtype 8-11 blocks. The two arms are
therefore joined on the hit's own OBSERVABLES, which both trees carry.

### The statistic

    z = (refParms[0] - genParms[0]) / sqrt(refCov[0,0])        the q/p pull
    x = z / (1 - a q z),   a = sigma * p * (1 - vgf)           truth-referenced

`sigma` is the FITTED error, so it fluctuates with the fit and a pull normalised
by its own fluctuating error has a spurious odd moment; `x` undoes the leading
part (`sigma = sigma_bar (1 + a q z)` to first order, from the 1/p^2 dependence
of the multiple-scattering term, `vgf` being the HIT share of the variance,
which does not scale that way). `<x>` is always split into

* **charge-EVEN** `0.5(<x>_+ + <x>_-)` = a charge-DEPENDENT momentum shift = a
  **SAGITTA** bias. It largely CANCELS in a dimuon mass. Under the mirror map
  (reflection in a plane containing the beam) a `+` track maps to a `-` track and
  `q/p -> -(q/p)`, so this channel is FORBIDDEN unless the source is CHIRAL:
  the module layout (tilted BPix ladders, the FPix turbine, stereo angles), a
  Lorentz-drift CPE bias with a fixed azimuthal sense, or an azimuthal-twist
  weak mode.
* **charge-ODD** `0.5(<x>_+ - <x>_-)` = a charge-INDEPENDENT momentum shift = a
  **SCALE** bias. It ADDS on both legs of a dimuon mass (`dm/m = dp/p`). This is
  where a second-order estimator bias, an energy-loss mismodelling or a
  field-scale error lives.

### The hit-location propagation (no free parameter)

`delta(q/p) = sum_b w_b . n_b` with `v_b = w_b^T dV_b w_b`, so a location bias
`mu_b` in PULL units contributes

    delta z_b = -s_b sqrt(v_b)/sigma * mu_b = -s_b sqrt(hitamp2_b) mu_b

with `s_b` the **BENDING SENSE**, the sign of the fit's q/p response to a
positive residual in that module's local frame. `s_b` is charge-INDEPENDENT
(signed curvature is a geometric property of the trajectory), which is why a
fixed local bias makes a charge-EVEN curvature bias.

* `s_b` comes from `resinfbv` row 0 (`B_b = M_b dV_b^{1/2}`, whose single
  nonzero entry for a rank-1 hit block is exactly `s_b sqrt(v_b)`), **not** from
  `resinfv[b][0]`: for a 2-D PIXEL block the parmtype-8 and parmtype-9 entries
  SHARE the block range, so `resinfv[b][0]` is the local-x weight for both and
  using it for the y entry is wrong by a factor 200 (measured).
* **The MINUS is the corrected sign.** `W5 = V^-1 F C^-1 E5`
  (`...G4e.cc:4619`) while the step the fit takes is `dx = -C^-1 F^T V^-1 r`
  (:4098), so `W5^T r = -dx`: the exported influence functional has the OPPOSITE
  sign to the fit's response. VARIANCE uses of `resinfv`/`resinfbv` are
  unaffected (`v_b = w^T dV w` is quadratic and sign-blind).

### The second-order (Box) bias

For `chi2(theta) = r^T W r` with `r = m - f(theta)`, a Gauss-Newton step from a
point displaced by `d1` from the minimum overshoots by a second-order remainder,

    d2 = b0 + a d1 + (K/2) d1^2 + O(d1^3)
    d1 = refParms_iter0[0] - trackParms[0],   d2 = refParms[0] - refParms_iter0[0]

and with `B = sum f'^2`, `C = sum f' f''` the stationarity expansion gives
`E[delta_2] = -(sigma^2/2)(C/B)` while the NOISELESS two-step gives
`d2 = +(C/2B) d1^2`. So `K = C/B` and

    <z>_Box = -(K/2) sigma          (a MINUS; this is Box 1971)

`K` is measured from the data's own `(d1, d2)` regression, so the prediction has
no free parameter. All four exported ingredients (`trackParms`,
`refParms_iter0`, `refParms`, `refCov`) are already in the production.

---

## How to run

`/ceph` is not readable from submit82 (cephx eviction) and the mfs venv does not
run there. Every command below goes through the SSH wrapper, which shells to
submit50 over a persistent ControlMaster and activates the venv:

    cd /work/submit/david_w/ZMass/calibration_studies/resolution/hitclassbias
    ./rrun.sh <command>

### Inputs (ceph)

    /ceph/submit/data/user/d/david_w/ZMass/cvh/hitres2_mugun_ul16/task_*/
    /ceph/submit/data/user/d/david_w/ZMass/cvh/resolution_trackres_mugun_ul16_260903x_m0/task_*/
    /ceph/submit/data/user/d/david_w/ZMass/cvh/resolution_trackres_mugun_lowpt_260903x_m0/task_*/
    /ceph/submit/data/user/d/david_w/ZMass/cvh/resolution_trackres_mugun_ul16_260909_conv_{base,tight,damp}/task_*/

Gun SIM filelist: `resolution/simprod/filelist_mugun_ul16.txt`.
Scalar-potential init: `mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt`.

### Extraction (caches land in `hitclassbias/data/`)

    ./rrun.sh python3 extract_hits.py   --prod hitres2_mugun_ul16 \
        --out data/hits_mugun_ul16_h2.npz
    ./rrun.sh python3 extract_blocks.py --prod resolution_trackres_mugun_ul16_260903x_m0 \
        --out data/blocks_mugun_ul16_260903x.npz
    ./rrun.sh python3 extract_quality.py     # defaults to the same production ->
                                             # data/quality_mugun_ul16_260903x.npz
    ./rrun.sh python3 extract_conv.py --prod resolution_trackres_mugun_ul16_260903x_m0 \
        --ntasks 160 --out data/conv_ref903x_full.npz

`blocks_*.npz` and `conv_ref903x_full.npz` are row-aligned (verified,
max |dz| = 0.0), which is what lets the per-hit and per-track arms be joined.

### The convergence variants

    NT=160 NP=10 ./run_conv_all.sh                   # produces all three variants
    NTASKS=160 ./rrun.sh ./run_conv_analysis.sh      # extraction + c0..c8 + figures

`run_conv.sh <variant> <ntasks> <nparallel>` reproduces the
`mugun_ul16_260903x_m0` arm EXACTLY and changes only the Gauss-Newton knobs:

| variant | knobs |
|---|---|
| `base` | nothing changed (the like-for-like bit check) |
| `tight` | `edmConvergence=1e-7 nIters=20` |
| `damp` | `+ gnDampAfter=1 gnDampFactor=0.5 nIters=30` — every step halved from iteration 1, so the fit cannot land at the seed-proximal point in two iterations |

`NTASKS` pins the sample so that a re-extraction cannot grow it under the reader.
`out_conv.txt` is the 40-task record; `out_conv_t160.txt` / `out_phi_t160.txt`
the 160-task one, from the frozen caches `data/conv_{base,tight,damp}_t160.npz`.

### The analyses

| script | what it answers | output |
|---|---|---|
| `t4_step1.py` | per-class residual LOCATION, skew, density, lever arm | `out_step1.txt` |
| `t2_predict.py`, `t3_keys.py`, `t6_final.py` | the no-free-parameter propagation (location + skew) | `out_step1.txt` |
| `t9_required.py`, `t11_bound.py`, `t12_chi2.py` | how far the locations must be wrong; the 3-sigma envelope; the chi2 per join key | |
| `s1_trim.py` .. `s6_mixture.py` | trim scan, quality census, the IN/OUT mixture | `out_quality.txt` |
| `c0_bitcheck.py` | `base` reproduces the baseline track by track | `out_conv.txt` |
| `c1_conv.py`, `c2_pair.py` | per-variant and PAIRED comparison | `out_c1.txt`, `out_c2.txt` |
| `c4_secondorder.py` | the Box `K` from `(d1, d2)` | `out_c4.txt` |
| `c5_scaling.py` | which SHAPE the charge-even shift has | |
| `c6_phi.py`, `c7_eta.py`, `c8_phiquad.py` | the phi harmonics, the eta structure, the rotated-charge-odd loophole | `out_c6.txt` |
| `c9_phipred.py` | the harmonic content of PART 1's propagation (`--sign +1` = corrected) | |
| `c10_boxodd.py` | the Box bias in the charge-ODD channel and what it does at Z momenta | `out_c10.txt` |
| `p1_phi_origin.py` | which subdetector carries the harmonics | |
| `p2_ladder.py` | the BPix-L1 ladder-parity test, head-on | |
| `t8_figs.py`, `s7_figs.py`, `c3_figs.py` | figures, one file per panel | |

**Retracted:** the first convergence-variant probes paired on
`(run, lumi, event)` alone (see "Defects"); they are deleted and `c1_conv.py`,
`c2_pair.py` and `conv_common.py` replace them.

### Figures

`~/public_html/ZMass/cvh/260908_hitclassbias/` — 13 location/quality panels
(`loc_subdet`, `loc_cls18`, `loc_vs_dxdz`, `dens_*`, `lever_subdet`,
`pred_vs_meas`, `skew_trimscan`, `skew_mixture`, `skew_partial_slopes`) plus the
8 convergence panels (`conv_band`, `conv_mixture`, `conv_niter`, `conv_scaling`,
`conv_dqop`, `conv_boxbias`, `conv_phi`, `conv_phiharm`), `index.php` present.
Every `conv_*` panel shows measurements (or the Box prediction, which comes from
`(d1, d2)`), so none is affected by the influence-sign correction.

---

## Results

### 1. The per-hit CPE LOCATION bias is real, and PIXEL-ONLY

`t4_step1.py`, `hitres2_mugun_ul16`, 1 375 697 hits, pull units, module-local
frame (local x for pixels and barrel strips, local phi for the radial-strip
wedges):

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

* **BPix layer 1 alone: +0.227 sigma_CPE = +2.71 um, skew +0.302**; L2 +0.058,
  L3 +0.044. Every strip subdetector is null at +-0.002 (3-sigma bounds
  0.04-0.23 um).
* It is EVEN in the incidence angle — BPix runs +0.34 / +0.23 / +0.13 / +0.11 /
  +0.09 | +0.09 / +0.10 / +0.09 / +0.16 / +0.17 across signed `dx/dz` — so it is
  not an uncorrected angle-linear drift term.
* Largest class locations: `pix_x_q2` +0.121, `pix_x_q3` +0.119 (BPix alone
  +0.134), `pix_x_q1` +0.102; pixel single-ROW clusters +0.034 with skew +0.18.
  Degraded STRIP classes are all null: `N=1` -0.0032 +- 0.0020, `N>=4`
  +0.0012 +- 0.0022, `uProj>0.5` +0.0021 +- 0.0032.

### 2. ...and its PHI-AVERAGE cannot make the observed charge-even shift

`t2_predict.py` / `t6_final.py`, 319 854 tracks, 6 219 371 hit blocks, probe
`<z e^{-u z^2}>` at u = 0.05 (a MEAN probe: coefficient -0.0394 on kappa3
against +0.867 on the mean), units 1e-3:

| band | measured (charge-even) | PRED location | PRED skew | PRED total |
|---|---:|---:|---:|---:|
| barrel `\|eta\|<0.9` | **-4.89 +- 2.46** | +0.86 | -0.001 | **+0.86** |
| middle 0.9-1.6 | **-3.57 +- 2.71** | +1.59 | -0.002 | **+1.58** |
| endcap 1.6-2.4 | **+0.12 +- 2.75** | +1.39 | -0.001 | **+1.39** |

Stable at +1.0 to +2.8 under EVERY join key tried. How far the locations would
have to be wrong (`t9_required.py`, constrained least squares on the 8
subdet x coordinate locations):

| group | measured | +- | REQUIRED | shift |
|---|---:|---:|---:|---:|
| BPix x | +0.1193 | 0.0027 | -0.261 | -143 sigma |
| TIB x | +0.0021 | 0.0018 | +0.096 | +51 sigma |
| TOB x | +0.0017 | 0.0018 | +0.088 | +48 sigma |
| TEC phi | -0.0001 | 0.0014 | -0.098 | -71 sigma |

chi2 = 33 497 for 3 constraints (183 sigma). No single group can do it either:
BPix alone would need -0.549 and would then give -4.89 / -6.32 / -4.42 against
the measured +0.12 in the endcap. Per join key (`t12_chi2.py`, 3 dof):

| key | ncell | central prediction (1e-3) | chi2 | sigma |
|---|---:|---:|---:|---:|
| subdet x coordinate | 8 | +1.00 / +1.38 / +0.98 | 33 497 | 183 |
| subdet x layer | 32 | +0.86 / +1.59 / +1.39 | 1 333 | 36.5 |
| subdet x layer x z side | 48 | +0.86 / +2.10 / +0.58 | 239 | 15.4 |
| + the 18 classes | 275 | +1.16 / +2.40 / +0.96 | 215 | 14.7 |
| orientation group | 440 | +0.48 / +0.07 / +1.15 | 28.5 | 5.3 |
| orientation group x class | 1728 | +1.11 / +0.90 / +1.68 | 34 | 5.8 |

`t11_bound.py` gives the model-independent worst case instead: +-0.31 / 0.22 /
0.19 at subdetector level, growing as sqrt(ncell/N) to +-8.8 / 16.1 / 8.8 at the
finest key. So the fine keys are statistics-limited for an ADVERSARIAL
arrangement, while the chi2 — how far the locations must ACTUALLY move — is
>= 5.3 sigma everywhere.

**Pre-registered predictions, scored** (written before the measurement):

* **P1 (the bending sense is coherent) CONFIRMED IN FORM, REFUTED IN SIZE.**
  `<|<s>|>` runs 0.016 (subdetector) -> 0.072 (+layer) -> 0.159 (+z side) ->
  0.761 (orientation group) -> 0.905 (module), and the pixel location is
  positive in all 12 BPix orientation groups (+0.022 to +0.272). But the lever
  arm `<sum_b s_b a_b>` is only -0.020 / +0.023 / +0.018, so a 0.12 sigma bias
  buys 1e-3, not 5e-3.
* **P2 (degraded classes reverse sign between barrel and forward) REFUTED.**
  `L mu` for `pix qbin3` is +0.25 / +0.30 / +0.12 (all positive), for pixel
  single-row +0.16 / +0.00 / -0.01, for strip `N=1` +0.03 / -0.00 / +0.00 — the
  strip classes have no location to reverse.
* **P3 (the propagated degraded-content contrast reproduces the coordinator's
  barrel/endcap reversal) REFUTED.** Propagated +1.19 / +2.95 / +1.39 (pixel
  share HIGH-LOW) and -0.19 / -0.81 / +0.35 (single-strip HIGH-LOW) against
  measured -3.63 +- 5.16 / -0.48 +- 5.66 / -0.11 +- 5.28 and
  -6.64 +- 5.06 / -1.83 +- 5.76 / -7.00 +- 5.15. The mass-level target it was
  meant to explain was `m_Z` -40.9 +- 4.6 (barrel, poorer-resolution half at
  fixed eta) against -4.6 +- 3.7, and +65.9 +- 8.7 (endcap) against +34.

Step 3 of the brief (putting the class densities into the CF hit term) was
therefore NOT done, by the brief's own condition.

### 3. Convergence and seed/path dependence: EXCLUDED at >1000x

`c0`-`c2`, all three variants at 160/160 tasks, PAIRED on
(run, lumi, event, charge):

| | base -> tight | base -> damp |
|---|---:|---:|
| paired tracks | 319 851 | 319 833 |
| q/p moved at all | 47.56 % | **98.73 %** |
| median moved \|dq/p\|/(q/p) | 2.54e-07 | 2.47e-06 |
| `<niter>` | 2.188 -> 2.868 | 2.188 -> 3.493 |
| chi2/ndof improved for | **0.00 %** of tracks | **0.00 %** |
| **paired `<dx>_even`, ALL** | **-0.001 +- 0.001 e-3** | **-0.000 +- 0.002 e-3** |
| statistic, barrel | -5.21 -> -5.21 | -5.21 -> -5.20 |
| middle | -4.36 -> -4.36 | -4.36 -> -4.36 |
| endcap | -1.95 -> -1.95 | -2.01 -> -2.01 |
| mixture OUT | -6.49+-1.88 -> -6.51+-1.93 | -6.57+-1.92 -> -6.56+-1.93 |

Per-task cost and convergence census (40-task pass, unchanged by sample size):

| | base | tight | damp |
|---|---:|---:|---:|
| `edmvalref >= 1e-7` | **65.66 %** | 0.11 % | 0.02 % |
| at the iteration cap | 15 (0.019 %) | 91 (0.114 %) | 17 (0.021 %) |
| `<chi2/ndof>` | 0.998874 | 0.998874 | 0.998894 |
| wall per task (2000 ev) | 587 s | 690 s (**+18 %**) | 842 s (**+43 %**) |

`damp` moves 98.7 % of tracks by a median 2.5e-6 in relative curvature — a
genuinely different path to the minimum — and the charge-even statistic moves by
less than 0.005e-3, i.e. **more than 1000x below the -6.3e-3 bulk**. `tight`
takes the fraction of tracks above a 1e-7 EDM from 66 % to 0.1 % and changes the
statistic by less than 0.001e-3. The chi2/ndof is identical to six digits in all
three: they are the same minimum.

**Criterion:** `edmConvergence=1e-7` costs +18 % wall and +0.68 GN iterations per
track and buys nothing measurable for this observable. Not worth adopting; the
only thing it changes is the 0.019 -> 0.114 % of tracks at the iteration cap,
which is a diagnostic, not a bias.

**Bit check** (`c0_bitcheck.py`, `base` vs the baseline production, 79 965 paired
tracks): 3 tracks (0.0038 %) have a different converged q/p, largest difference
1e-4 sigma; every other exported variable is EXACTLY equal on every track; the
per-band charge-even `<x>` agrees to the printed precision. The three are the
known CVH limit-cycle / anchoring non-determinism (`<niter>` 5.67 against 2.19).
**CVH is reproducible run to run everywhere except on tracks sitting at an
iteration boundary.**

### 4. The second-order (Box) bias: charge-ODD, measured, and it reaches m_Z

`c4_secondorder.py`, full 160-task baseline (314 151 tracks after trimming the
extreme 1 % in |d1| and |d2|):

| charge | a | K [GeV] | predicted `<z> = -(K/2) sigma` |
|---|---:|---:|---:|
| q = +1 | +0.00010 +- 0.00003 | **+20.85 +- 0.30** | **-2.593 +- 0.038 e-3** |
| q = -1 | +0.00025 +- 0.00003 | **-21.29 +- 0.32** | **+2.632 +- 0.039 e-3** |

* `K_+ + K_- = -0.44 +- 0.44`: **K_- = -K_+ to 2 %**, the mirror map measured.
* **PREDICTED charge-EVEN `<z>` = +0.019 +- 0.027 e-3** against a MEASURED
  **-4.26 +- 1.75 e-3** — 200x too small, and small BY SYMMETRY, not by accident.
* **PREDICTED charge-ODD `<z>` = -2.61 +- 0.03 e-3**, measured -1.46 +- 1.90 e-3.
* `d2` is essentially PURE quadratic in `d1` (the linear coefficient absorbing
  seed noise is 1e-4 to 2.5e-4); `rms(d1)/<sigma> = 0.28` and the fit range
  reaches 1.45 sigma, so applying `K` at the sigma scale is interpolation.

**The charge-ODD channel, per track** (`c10_boxodd.py`, `b_i = -(K/2) sigma_i`
with K refit on a band x sigma-quintile grid), units 1e-3:

| gun | PRED `<z>` | MEAS `<x>` | MEAS probe u=0.05 | per-track regression slope | cell-wise scale k on an exogenous gen (pT,\|eta\|) grid |
|---|---:|---:|---:|---:|---:|
| 20-60 GeV (313 454 trk) | **-3.172** | **-3.228 +- 1.803** | -4.098 +- 1.523 | +0.586 +- 0.447 | **+0.67 +- 0.47** (chi2(k=1) 8.6/9, chi2(k=0) 10.1/9) |
| low-pT (313 245 trk) | **-2.277** | **-5.354 +- 1.841** | -9.480 +- 1.525 | +1.924 +- 0.663 | **+1.86 +- 0.79** (chi2(k=1) 19.0/9, chi2(k=0) 23.4/9) |
| **combined** | | | | | **k = +0.98 +- 0.40** (2.4 sigma from 0) |

In probe space, model(-2.13) + Box(-2.75) = -4.88 against the measured
-4.10 +- 1.52 (pull +0.51); the CF ionisation+radiation model alone pulls -1.30.
**So the Box bias IS the charge-odd residual the CF model leaves.**

What it does to the momentum scale (`dp/p = -<z>_odd sigma_rel`,
charge-INDEPENDENT, and `dm/m = dp/p` in a dimuon mass):

| selection (20-60 GeV gun) | `<sigma_rel>` | `<p>` [GeV] | dp/p | `dm` at m_Z |
|---|---:|---:|---:|---:|
| barrel | 0.01172 | 45.9 | **+21.50e-6** | +1960 keV |
| middle | 0.01888 | 77.4 | +129.10e-6 | +11 773 keV |
| endcap | 0.02934 | 154.0 | **+43.27e-6** | +3946 keV |
| Z-like: pT 40-50, \|eta\|<0.4 | 0.01090 | 46.2 | **+16.07e-6** | **+1465 keV** |
| Z-like: pT 40-50, all eta | 0.02023 | 102.0 | +65.33e-6 | +5957 keV |
| low-pT gun, barrel | 0.00961 | 12.8 | +25.31e-6 | +2308 keV |
| low-pT gun, middle | 0.01580 | 21.6 | +29.31e-6 | +2672 keV |
| low-pT gun, endcap | 0.02228 | 42.7 | +51.20e-6 | +4669 keV |

**The calibration caveat is the key number.** The low-pT gun gives +25.3e-6 in
the barrel against +21.5e-6 at 20-60 GeV, so a J/psi-anchored scale absorbs most
of it and the RESIDUAL is the difference: **~4e-6 -> ~0.35 MeV (barrel),
~8e-6 -> ~0.7 MeV (endcap)**.

### 5. What SHAPE the charge-even shift has

`c5_scaling.py`, per-track regression, every regressor entered as `f` (even) and
`q f` (odd), 318 239 tracks with |x| < 10 and sigma_rel < 0.1:

| shape | meaning | even coefficient | dchi2 (even, odd already in) |
|---|---|---:|---:|
| `c sigma_rel` | second order | -0.118 +- 0.083 (1.4s) | 2.09 |
| `D/sigma` | fixed additive `Delta(q/p)` | -0.0094 +- 0.0039 (2.4s) | 6.10 |
| `D_T/(sigma cosh eta)` | fixed `Delta(q/pT)`, a sagitta | -0.0187 +- 0.0068 (2.8s) | **7.91** |
| constant | the per-band null shape | -0.0037 +- 0.0018 (2.0s) | 4.37 |

i.e. `Delta(q/pT) = -1.87e-6 GeV^-1`, 50x inside the AN's misalignment bound
|M| < 1e-4 GeV^-1. The exogenous gen pT x gen |eta| grid (24 cells, sigma_rel
lever arm 4.96) gives the same ranking. **The data mildly prefer a fixed sagitta
offset over an O(sigma) second-order bias, but dchi2 ~ 6 is a preference, not a
discrimination.**

### 6. The PHI structure: a BPix layer-1 LADDER-PARITY CPE defect, 16 sigma

**Unbinned harmonic scan** (`c6_phi.py`, amplitudes `2<x cos/sin(n phi)>`,
charge-even, no binning hence no aliasing), 160-task baseline, units 1e-3:

| n | cos | sin | power |
|---|---:|---:|---:|
| 7 | +6.52 +- 2.55 | +5.54 +- 2.56 | 11.2 |
| **8** | +4.07 +- 2.56 | **+18.96 +- 2.55** | **57.7** |
| **10** | +0.51 +- 2.55 | **+21.62 +- 2.55** | **71.7** |
| 14 | +5.50 +- 2.55 | +6.58 +- 2.55 | 11.3 |
| 17 | -10.41 +- 2.56 | +3.59 +- 2.55 | 18.6 |

Total power n = 1..40 is **281.5 for 80 dof**; n = 12, 16, 24 sit at -6.3, -5.2,
-2.7. The mean over phi is -3 to -4e-3; the MODULATION is five times larger.

It is real, not machinery:
* PERMUTATION null (phi reassigned at random): chi2(no effect) = 11.1 / 7.0 /
  16.4 for 12 bins — the estimator and its errors are correct;
* SPLIT-HALF on the lumi parity (disjoint event sets): 24-bin chi2(A==B) =
  20.9/24 with **corr(A,B) = +0.75** (e.g. bin 5: -27.9 vs -24.9);
* it survives every tail cut (chi2 rises from |x|<30 to |x|<5); the within-pair
  correlation of the gun's two muons is -0.0007; the gun is phi-uniform and
  charge-balanced in every bin (frac q+ 0.4968 to 0.5020) and its two muons are
  NOT back-to-back (median |dphi| 1.57).

**It is genuinely charge-EVEN AT THE MODULE** (`c8_phiquad.py`). `genphi` is the
azimuth at the PCA while modules are crossed at `phi_PCA + q delta` with
`delta ~ 0.3 B r / pT` = 0.057 rad at pT 20 GeV, so a charge-ODD modulation at
the module would leak into the charge-EVEN channel at the PCA in the other
quadrature with weight `sin(n delta)` = 54 % at n = 10. Two tests kill that:

| n | EVEN cos | EVEN sin | ODD cos | ODD sin | err |
|---|---:|---:|---:|---:|---:|
| 8 | +4.1 | **+19.0** | +1.1 | +0.3 | 2.6 |
| 10 | +0.5 | **+21.6** | -2.7 | +0.4 | 2.6 |
| 17 | **-10.4** | +3.6 | +0.6 | -0.6 | 2.6 |

the charge-ODD amplitudes are null at every harmonic, and the charge-EVEN
amplitude does not fall as 1/pT (n=10: +18.1 / +16.1 / +18.8 / +33.5 across
pT 20-30/30-40/40-50/50-60; n=8: +16.4 / +19.6 / +28.8 / +11.0), whereas a
rotated charge-odd source must fall by a factor 3 over that range.

The azimuthal segmentation of the 2016 Phase-0 tracker, counted directly from
the DetIds in this sample: **TEC 8 petals per wheel** (-> n = 8), **BPix L1 20
ladders** in an inner/outer alternation (-> 10 pairs -> n = 10), BPix L2 / L3
32 / 44 ladders (n = 16 / 22), FPix 24 blades x 2 panels (n = 24 / 12), TIB
30 / 38 / 45 / 56 strings, TOB 42 / 48 / 54 / 60 / 66 / 74 rods. Only n = 8 and
n = 10 match a real structure, and they are exactly the two the scan finds
(`A_sin` +18.04 and +21.81 +- 2.50 on the full sample). The petal count makes
n = 8 look like a TEC effect; the ladder-local test below shows it is not.

**Both harmonics are carried by tracks with a BPix layer-1 hit**
(`p1_phi_origin.py`, at FIXED |eta| in 12 quantile bins):

| | has BPix-L1 hit (226 481) | no BPix-L1 hit (93 369) | difference |
|---|---:|---:|---:|
| `A_sin(n=8)` | +24.00 +- 2.97 | **+3.59 +- 4.63** | +20.40 +- 5.50 (3.7 s) |
| `A_sin(n=10)` | +30.46 +- 2.97 | **+0.82 +- 4.63** | **+29.64 +- 5.50 (5.4 s)** |

and the split by INFLUENCE SHARE is not the discriminator it looks like: the TEC
and TOB shares are `corr = +0.886` and `-0.861` with `|eta|`, so those splits are
`|eta|` in disguise; the BPix-L1 share is not (`corr = -0.137`).

**The head-on test settles it** (`p2_ladder.py`). For PXB the DetId packs
`layer = (id>>16)&0xF`, `ladder = (id>>8)&0xFF`. Selecting hits, not blocks
(`(b_sd==1) & (b_lay==1) & (b_isy==0)`), 222 674 tracks have exactly one BPix-L1
hit across 20 ladders. `<x>` alternates with LADDER PARITY with no exception in
20 ladders (1e-3): odd ladders +40.6 +30.2 +34.3 +26.6 +57.2 +34.9 +31.9 +42.4
+23.8 +15.4; even ladders -43.7 -41.3 -51.4 -28.4 -35.0 -30.6 -35.2 -44.4 -36.2
-48.9.

| quantity | value (1e-3) |
|---|---|
| odd ladders, n = 107 902 | **+33.01 +- 3.15** |
| even ladders, n = 114 772 | **-38.77 +- 3.07** |
| parity difference (even - odd) | **-71.58 +- 4.40 (16.3 sigma)** |
| same, at fixed \|eta\| (12 quantile bins) | **-71.78 +- 4.24** |

Transforming to ladder-local azimuth `phi - phi_lad`, `phi_lad = 2 pi (lad-1)/20`,
kills BOTH harmonics:

| harmonic | global azimuth | ladder-local azimuth |
|---|---|---|
| n = 8 | +24.63 +- 3.00 | **-0.62 +- 3.00** |
| n = 10 | +30.77 +- 3.00 | **+0.42 +- 3.00** |

**n = 8 is not a separate mode and not the TEC petals: it is leakage from the
same 20-fold ladder pattern, which is not a pure sinusoid.**

It is charge-EVEN, so it is a CPE/incidence-angle effect and not misalignment
(which would be charge-odd):

| | q = +1 | q = -1 |
|---|---|---|
| odd ladders | +29.76 +- 4.45 | +36.25 +- 4.48 |
| even ladders | -38.83 +- 4.33 | -38.71 +- 4.34 |
| parity difference | -68.59 +- 6.21 | -74.96 +- 6.23 |

charge-EVEN part **-71.77 +- 4.40**, charge-ODD part +3.18 +- 4.40 (consistent
with zero). `|eta|` dependence of the parity difference (1e-3): -71.5 +- 10.4
(0-0.5), -26.9 +- 11.2 (0.5-0.9), -52.5 +- 10.9 (0.9-1.3), -69.5 +- 9.2
(1.3-1.8), **-112.7 +- 8.3 (1.8-2.4)** — largest at shallow incidence, as a
cluster-shape effect should be.

**But it carries NO net mean, which closes the phi lead as a lead for the bulk:**

| sample | `<x>` (1e-3) |
|---|---|
| all good, n = 319 850 | -3.59 +- 1.84 |
| exactly one BPix-L1 hit, n = 222 674 | **-3.99 +- 2.20** |
| no BPix-L1 hit, n = 93 369 | **-1.35 +- 3.40** |

-2.6 +- 4.1 between them. Per band, has vs no BPix-L1: -6.79 +- 3.80 vs
-0.36 +- 4.97 (|eta|<0.9), -6.08 +- 3.97 vs -0.84 +- 6.67 (0.9-1.6),
-1.04 +- 3.59 vs -3.57 +- 6.51 (1.6-2.4) — every band consistent. The residual
-4e-3 of the BPix-L1 sample is just the occupancy/magnitude imbalance of the
alternation itself (107 902 x +33.01 + 114 772 x -38.77 over 222 674 = -3.99),
at 1.8 sigma.

In size the modulation is `<x> ~ 25e-3` at `sigma_rel ~ 0.03`, `p ~ 100 GeV`,
i.e. **`Delta(q/pT) ~ 4e-5 GeV^-1`** — the same order as the AN's misalignment
bound |M| < 1e-4 GeV^-1, on IDEAL geometry, where there is no misalignment at
all. What survives is a per-track bias/resolution term that a ladder-parity (or
incidence-angle) hit class would absorb.

### 7. The ETA structure: there is none

`c7_eta.py`, 24 bins in SIGNED gen eta, analytic errors: **mean
-3.64 +- 1.81 e-3, chi2(no effect) 19.9/24, chi2(FLAT) 15.8/23**, and the
eta-ODD part (a z-antisymmetric sagitta twist) is 7.9/12. The charge-even bias is
a FLAT offset in eta with a large modulation in PHI. The phrase "eta-dependent
charge-even skew" is retired.

### 8. WHICH tracks: the IN/OUT mixture, and why it is a property of the gun

`s1_trim.py` .. `s6_mixture.py`, same 319 850 tracks.

**(a) It is a CORE shift, not a one-sided tail.** `m_implied(T) =
<x>_even,|x|<T / f(T)` with `f(T) = 1 - 2 T q(T)/Q(T)` from the
charge-SYMMETRISED data itself (units 1e-3):

| band | T=1 | T=2 | T=3 | T=5 | T=10 | T=inf |
|---|---:|---:|---:|---:|---:|---:|
| barrel | -7.74 | -4.44 | -5.68 | -5.73 | -4.84 | **-5.21** |
| middle | -2.34 | -6.52 | -3.53 | -4.43 | -5.06 | **-4.36** |
| endcap | +3.05 | +3.32 | +0.85 | +0.07 | -1.02 | **-1.97** |

Flat from T = 2 within +-2.8e-3. In the barrel the -5.2e-3 is built as
-0.20 / -1.55 / -1.79 / +0.21 / -2.12 / -0.26 / +0.87 / -0.37 over the shells
`|x|` = 0-0.5 / 0.5-1 / 1-1.5 / 1.5-2 / 2-3 / 3-5 / 5-10 / >10, i.e. two thirds
of it inside `|x| < 2`.

**(b) The step control NEVER fires.** `nChargeFlipProtect` = 0 for
319 849 / 319 850 tracks (one has 1); `chargeHypFlipped` = 0 for every track;
`niter` = 2 for 81.4 %, 3 for 18.5 %, >=4 for 0.10 %. The asymmetric momentum
clamp (`clampMomentumFloor` = 2 GeV) and the two-hypothesis refit never engage on
a 20-60 GeV gun — the floor is 10-30x away. (Not excluded at LOW pT.)

**(c) The ordering variables.** Charge-even slopes AT FIXED |eta| (v centred in
12 fine |eta| bins), 1e-3 per standardised unit:

| variable | barrel | middle | endcap | combined | chi2/2 |
|---|---:|---:|---:|---:|---:|
| nValidPixelHits | -8.28+-3.44 | -1.83+-4.13 | -5.30+-3.05 | **-5.50+-2.00 (2.7s)** | 1.4 |
| n pixel layers | -8.96+-3.61 | -0.55+-4.22 | -3.53+-3.33 | -4.64+-2.12 (2.2s) | 2.5 |
| niter | +13.53+-11.60 | +12.14+-10.21 | +14.59+-7.35 | +13.71+-5.30 (2.6s) | 0.0 |
| \|seed->final dq/p\| | -158+-1694 | +1354+-845 | +442+-215 | +488+-207 (2.4s) | 1.2 |
| n layers / nValidHits / chi2ndof / vgf / sigma_rel | | | | all < 1.2s | |

Joint fit of all six: nValidPixelHits -4.98 +- 2.37, |seed->final| +6.38 +- 2.95,
niter +3.06 +- 2.11, chi2/ndof +1.40 +- 3.28, nValidHits +1.91 +- 2.19,
vgf -3.48 +- 2.16. **Nothing reaches 3 sigma.**

**(d) The mixture.** Split on `|seed->final dq/p|` at its 90th percentile:

| | barrel | middle | endcap | combined | chi2/2 |
|---|---:|---:|---:|---:|---:|
| **IN** (top 10 %) | +21.28+-29.41 (2.4 %) | +20.69+-15.98 (7.0 %) | +20.53+-7.26 (21.2 %) | **+20.59+-6.45** | 0.0 |
| **OUT** (rest) | -5.03+-2.95 | -6.64+-3.29 | -7.68+-3.37 | **-6.33+-1.84** | 0.4 |

Both components flat in |eta| on the gun; only the mixing fraction runs
(2.4 -> 7.0 -> 21.2 %), and the mixture reproduces the band values exactly
(0.024(+21.28) + 0.976(-5.03) = -4.40 against -4.42 +- 3.09; -4.73 against
-4.72; -1.70 against -1.65). Robust at the 80th and 95th percentile (OUT
-6.79 +- 2.03 and -5.17 +- 1.83, chi2 <= 0.6/2), on the RAW `z` (OUT
-6.16 +- 1.88, IN +20.90 +- 6.69), and on the CVH-INTERNAL `|iter0->final|` step
(OUT -4.64 +- 1.85, IN +7.03 +- 7.40). The IN population is 2.1x the `sigma_rel`
(0.0385 vs 0.0182), 2.72 GN iterations against 2.13, chi2/ndof 1.098 against
0.988, **fewer pixel hits (1.67 vs 2.30)** at the same total hit count (17.0 vs
17.2), rms(x) 1.22 against 1.02. A single relative curvature bias cannot describe
both: `beta = -1.15e-4` for OUT at sigma_rel 0.0182 and `+7.9e-4` for IN at
0.0385.

**(e) The mixture reading is REFUTED on the real legs** (`fullscale/STATE.md`
sec. 0f.52, `resolution/oddmoment/aux_seed.py`, at 8-16x the gun's statistics).
`|seed -> final dq/p|` is recoverable from the slim v2 two-track trees
(`Mu{plus,minus}trk_pt/eta` = the generalTracks KF seed,
`Jpsi_qopref{plus,minus}` = the CVH reference q/p, `== Mu*_refParms[0]` to 1e-7);
caches `fullscale/runs/auxseed_dyv2.npz` (3 733 323 rows) and
`auxseed_jpsiv2.npz` (7 923 460 rows), aligned row by row to the pairs caches.

* The MIXING FRACTION reproduces strikingly: **2.68 / 9.69 / 20.04 %** on the Z
  legs against the gun's 2.4 / 7.0 / 21.2 % — the step selects the same
  population.
* **But neither component is eta-flat there** (leg level, truth-referenced
  charge-even, 1e-3):

| | IN (top 10 %) | OUT (the bulk) | chi2 vs eta-flat |
|---|---|---|---|
| Z legs, 7.46 M | +7.65 / +5.77 / -9.49 | +3.97 / -3.46 / **-18.22** | 35.8 / **577.7** per 2 |
| J/psi legs, 15.8 M | +3.82 / -0.17 / -1.18 | +1.47 / -0.54 / **-3.26** | 1.0 / **77.4** |

and the same at mass level (OUT chi2 72.2/2 on Z, 102.0/2 on J/psi; OUT
+2.08 / +1.09 / -4.73). The bulk is 85-90 % of every sample and it alone carries
the whole eta dependence. **The two-component picture is a property of the gun,
not of the reconstruction.** On the gun both components looked flat only because
the per-band errors were +-3e-3 against the legs' +-0.5-0.8e-3.

Two caveats on reading that comparison: `corr(|dq|, |x|)` is +0.082 (Z legs),
+0.100 (J/psi legs) and +0.156 (mass level), so the IN/OUT *values* carry a
selection effect (the fractions and the flatness chi2 do not); and the Z legs'
band pattern (+4.1 / -2.6 / -16.4) has the OPPOSITE eta trend to the gun's
(-4.9 / -3.6 / +0.1) — the gun is single-track in 150X with a different alignment
payload, the legs are two-track (mass- and vertex-constrained) in 106X. Different
estimators; do not difference them.

What survives from this section: the trim scan, the step-control census, and the
conditioning trap below. The MIXTURE reading does not.

### 9. The phi-averaged bulk is NOT established

On the FULL 160-task baseline with no subsample selection the per-track
regression gives `<x> = -3.7 +- 1.8e-3` (**2.0 sigma**); the convergence
extraction gives all-bands **-3.85 +- 1.81e-3** (chi2 0.6/2 for
eta-independence, bands -5.21 +- 3.00 / -4.36 +- 3.37 / -1.97 +- 3.09) and the
ladder analysis -3.59 +- 1.84e-3 — the same number under slightly different cuts.
The 3.4 sigma quoted for the OUT component is a subpopulation selected on a
variable with `corr(v,|x|) = +0.054`. **The most economical reading is that the
phi-averaged bulk is not yet established at all.**

(The per-band target used in section 2, -4.89 / -3.57 / +0.12, comes from the
`extract_blocks.py` arm with the `cf_track_resolution.py` cut set; the
convergence arm's -5.21 / -4.36 / -1.97 is the same quantity under the
`extract_conv.py` cuts. They agree within the +-3e-3 per-band errors.)

### 10. Sizes, and what to do about them

| effect | size on the momentum scale | on m_Z | recommended next step |
|---|---|---|---|
| **Box second-order bias** (charge-ODD, scale-like) | +21.5e-6 barrel, +43.3e-6 endcap, +16.1e-6 for a Z-like pT 40-50 \|eta\|<0.4 muon | **+1.5 MeV central, up to +3.9 MeV endcap, UNCALIBRATED**; **~0.35 MeV barrel / ~0.7 MeV endcap RESIDUAL after a J/psi-anchored scale** | Apply the analytic per-track correction `Delta(q/p) = +(K/2) sigma^2` with K from the `(d1,d2)` regression. Needs NO new production. Validate on the Z and J/psi legs, where the statistics are 10x. |
| **BPix-L1 ladder-parity CPE bias** (charge-EVEN, sagitta-like) | +-36e-3 per track, `Delta(q/pT) ~ 4e-5 GeV^-1` at n = 8 / n = 10; **zero net mean** | first-order CANCELLING in the pair mass; matters through resolution and through its degeneracy with alignment | **Fix it in the CPE** (BPix-1 local x is +2.7 um): a per-class local-x offset removes it for every consumer. At analysis level the correction is `Delta(q/p) = -sum_b s_b sqrt(v_b) mu_b` per track, which needs per hit block the DetId (for the ladder/orientation group) and the class — see the export gap below. |
| **Convergence / seed path** | < 0.004e-3 on the pull | < 0.1 keV | nothing |
| **CPE location bias, phi-AVERAGED** | +1.0e-3, wrong sign | negligible | nothing; excluded at >= 5.3 sigma at every granularity |

---

## Defects found and fixed

1. **The exported influence functional has the WRONG SIGN for odd moments.**
   `W5 = V^-1 F C^-1 E5` (`ResidualGlobalCorrectionMakerG4e.cc:4619`) while the
   fit's step is `dx = -C^-1 F^T V^-1 r` (:4098), so `W5^T r = -dx`. Anyone
   reusing `resinfv` / `resinfbv` for an ODD-moment (LOCATION) calculation must
   apply `dx = -W5^T r`. VARIANCE uses are unaffected. `c9_phipred.py --sign +1`
   (the default) applies the corrected sign; `--sign -1` reproduces the earlier,
   wrong-signed numbers. With the correction the no-free-parameter propagation of
   the measured CPE locations PREDICTS the harmonics:

   | | predicted | measured |
   |---|---|---|
   | n = 8 sin | **+18.68 +- 0.35** | +18.96 +- 2.55 |
   | n = 10 sin | **+19.88 +- 0.39** | +21.62 +- 2.55 |

   over n = 1..20, `chi2(measured == PREDICTED) = 61.5/40` against
   `chi2(measured == 0) = 227.6`; the pixel dose response is +0.2 / +11.3 /
   +19.6 / +25.8 / +23.2 against measured -6.9 +- 14.2 / +10.3 +- 6.1 /
   +24.0 +- 4.1 / +26.1 +- 4.4 / +30.9 +- 10.9; and the phi-AVERAGED bands pull
   -1.09 / -1.07 / +0.23 instead of the ">= 5.3 sigma" of the wrong-signed
   version. **The figures `pred_vs_meas` and `lever_subdet` in the figure
   directory still carry the WRONG sign** and must be regenerated from
   `c9_phipred.py` before they are shown.

2. **Pairing the convergence variants on (run, lumi, event) alone is wrong.**
   The muon gun puts TWO muons in every event (79 965 tracks in 40 000 unique
   keys), so a searchsorted pairing matches muon A of one variant to muon B of
   the other as soon as one track is dropped upstream by the covariance-consistency
   cut. That manufactures an O(1 sigma) per-track spread and a fat tail, and it
   produced retracted claims of a +3.4e-3 paired shift, "15 % of fits
   unconverged" and "the shift lives in the unconverged 15 %".
   **Fix:** the pairing key is `(run, lumi, event, charge)` — unique on this gun
   and stable under a dropped candidate — with the within-event `slot` kept as a
   cross-check (`conv_common.Var.key`, `conv_common.pair`).
   The probes built on the wrong key, and every claim above that came out of
   them, are RETRACTED; they are deleted and `c1_conv.py` / `c2_pair.py` /
   `conv_common.py` are what to use.

3. **Blocks are not hits in the ladder test.** A first pass counted 460 579
   BPix-L1 "hits" on 226 484 tracks (2.03/track): every pixel hit books a local-x
   AND a local-y block. The correct hit selection is
   `(b_sd==1) & (b_lay==1) & (b_isy==0)`, which gives 222 674 tracks with exactly
   one BPix-L1 hit over 20 distinct ladders, as Phase-0 BPix geometry requires.

4. **`resinfv[b][0]` is the wrong weight for pixel local-y.** The parmtype-8 and
   parmtype-9 entries of a 2-D pixel block SHARE the block range, so
   `resinfv[b][0]` is the local-x weight for both; using it for the y entry is
   wrong by a factor 200 (measured). Use `resinfbv` row 0.

5. **`fitFromGenParms=True` deletes the observable.** Measured on
   `hitres2_mugun_ul16`: that switch freezes the reference block exactly at gen
   (`max|refParms-genParms| = 0`, `refCov(0,0) = 0`, `niter = 1`), so
   `z = (refParms[0]-genParms[0])/sigma` is identically zero. It removes the
   measurement, not the seed dependence — which is why the convergence variants
   run with `fitFromGenParms=False`.

6. **A 24-bin phi scan ALIASES** n = 14/16/17 onto 10/8/7, and is what almost
   sent the harmonic identification the wrong way. The scan is UNBINNED
   (`2<x cos/sin(n phi)>`) for that reason.

### Export gap (not a defect, but blocking)

`resinfv`, `resinfbv` and `hitDetId` are **absent from the slim v2 two-track
productions** `dymc_8p5M_260906_v2` and `jpsimc_20M_260906_v2` (213 and 228
branches; the hit-related ones there are `cfmass_hitcls`, `cfmass_hitv`,
`reshitcls`, `reshitidx`, `resinfcovhit`, `resinfvarv`). They are booked under
`if (exportStepRecords_)`, `ResidualGlobalCorrectionMakerBase.cc:547-549`, which
was OFF. So a SIGNED per-hit mass-projected weight cannot be extracted from them,
a guessed `{prefix}_hits` / `{prefix}_hitdetid` will silently skip, and applying
the ladder-parity correction at mass level needs either
`exportStepRecords=True` (the 430 kB/candidate path) or a small maker change
exporting an int8 sign per block plus `hitDetId`. Likewise
`fullscale/runs/gzpairs_dyv2_n50.npz` carries `hit_cls`, `hit_v` and `hit_ptr`
but **no sign and no `hitDetId`** — adding `hit_s` and `hit_detid` is a pairs-cache
RE-EXTRACTION, not a production.

---

## Conditioning traps and standing rules

1. **`seed -> final dq/p`, SIGNED, is a trap.** `corr(v, x) = +0.1132` — bigger
   than the reco-pT trap's +0.042 — and its tertiles run -169 / -7 / +163 e-3 in
   the barrel, a 332e-3 artefact. The seed is the generalTracks KF, which shares
   its hits with CVH, so the difference is partly the residual itself. The
   ABSOLUTE step is safe (`corr = +0.0061`).
2. **Never bin the charge-ODD channel on the FITTED `sigma`**: it gives
   -68 -> +44e-3 across quintiles, pure artefact.
3. **Every conditioning variable must be printed with `corr(v, x)` and
   `corr(v, |x|)`.** A variable correlated with the SIGNED residual is a
   Jacobian-edge-style trap; one correlated only with `|x|` is a legitimate
   resolution handle for a MEAN split.
4. **The charge-even average is protected to first order** against the
   `sigma = sigma_bar(1 + a q x)` selection (the induced term is charge-ODD),
   which is why `x`, not `z`, is the statistic — but a 2-4 sigma ordering on a
   reconstructed variable is a HYPOTHESIS, not a result.
5. **CVH is not bit-reproducible run to run** at the 1e-4 level of tracks (the
   limit-cycle / anchoring tracks). A moved build would change every track, not
   three — that is how to tell the two apart.
6. **The per-STEP clamp/backtrack counters** (`fitStepClamped_`,
   `stepBacktrackEvents_`) are JOB-level in this release, NOT per track;
   `nChargeFlipProtect`, `niter` and the iteration-0-to-final step are the
   per-track proxies.
7. `/ceph` is not readable from submit82 and the mfs venv does not run there —
   use `rrun.sh`.

---

## Open items

1. **Regenerate `pred_vs_meas.pdf/png` and `lever_subdet.pdf/png`** in
   `~/public_html/ZMass/cvh/260908_hitclassbias/` from `c9_phipred.py` (default
   `--sign +1`). They currently show the wrong-signed prediction. The figure
   scripts still write to `~/public_html/cvh/<YYMMDD>_hitclassbias`, which is not
   the canonical directory — they need pointing at `~/public_html/ZMass/cvh/`.
2. **Apply and validate the per-track Box correction**
   `Delta(q/p) = +(K/2) sigma^2` on the Z and J/psi legs. It needs no new
   production, and its post-J/psi-calibration residual (~0.35 MeV barrel,
   ~0.7 MeV endcap) is the number that matters for `m_Z`.
3. **Repeat the ladder-parity measurement on the legs.** The correction needs a
   signed per-hit weight and the DetId at mass level; see the export gap above.
   A ladder-parity (or incidence-angle) HIT CLASS in the CF hit term is the
   alternative that needs no new per-track machinery.
4. **Match the ladder-parity amplitude to a CPE mechanism.** The parity
   alternation is +-36e-3 and grows with |eta| (largest at shallow incidence);
   the +2.71 um BPix-1 local-x location and the turbine tilt of alternate
   ladders are the proposed ingredients, but the phases have not been matched to
   the surveyed ladder azimuths.
5. **The phi-averaged charge-even bulk** (-3.6 to -3.9 +- 1.8e-3) remains at
   2 sigma with every estimator mechanism excluded. Settling it needs the legs,
   not more gun statistics.
