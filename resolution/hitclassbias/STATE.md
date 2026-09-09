# SUMMARY FOR THE FEASIBILITY REPORT — hitclassbias, 2026-09-08

Sample throughout: the 20-60 GeV tight-stepper muon gun, IDEAL geometry,
default field, GT `150X_mcRun2_asymptotic_v1`, 160 tasks = **319 854 tracks**
(`resolution_trackres_mugun_ul16_260903x_m0`); the low-pT gun
(`mugun_lowpt_260903x_m0`, 313 245 tracks) is the out-of-sample arm.
Statistic: the truth-referenced pull `x = z/(1-a q z)`,
`z = (refParms[0]-genParms[0])/sigma`. **charge-EVEN `<x>` = a SAGITTA
(charge-dependent) momentum bias, which largely CANCELS in a dimuon mass;
charge-ODD `<x>` = a momentum SCALE bias, which ADDS.**

## ESTABLISHED

| # | result | numbers | effect on p / on m_Z |
|---|---|---|---|
| 1 | **Bit check.** A re-launched `base` reproduces the baseline production track by track | 3 tracks of 79 965 differ (0.0038 %), max 1e-4 sigma; all are high-`niter` (5.67 vs 2.19) limit-cycle tracks | -- (CVH is reproducible except on tracks at an iteration boundary) |
| 2 | **Incomplete convergence EXCLUDED.** `edmConvergence` 1e-5 -> 1e-7, `nIters` 10 -> 20 | `<niter>` 2.190 -> 2.870, tracks above the 1e-7 EDM 65.7 % -> 0.11 %, 47.5 % of q/p values move (median 2.6e-7); **paired `<dx>_even` = -0.000 +- 0.000e-3**; +18 % wall | none |
| 3 | **Seed / path dependence EXCLUDED.** `gnDampAfter=1 gnDampFactor=0.5 nIters=30` halves every step from iteration 1 | `<niter>` 3.500, **98.8 % of tracks move** (median 2.5e-6 relative), chi2/ndof identical to 6 digits; **paired `<dx>_even` = +0.000 +- 0.002e-3**, i.e. >1500x below the effect; +43 % wall | none |
| 4 | **The second-order (Box) bias is CHARGE-ODD, by symmetry.** `d2 = (K/2) d1^2` from the exported `(seed->iter0, iter0->final)` steps; `<z> = -(K/2) sigma` | `K_+ = +20.85+-0.30`, `K_- = -21.29+-0.32 GeV`, `K_+ + K_- = -0.44+-0.44` -- the mirror map, measured. Predicted charge-EVEN `<z> = +0.019+-0.027e-3` (measured -4.26+-1.75): **200x too small** | -- |
| 5 | **...and it IS the charge-odd residual the CF ionisation+radiation model leaves.** Per-track `b_i = -(K/2)sigma_i` with K refit on a (band x sigma-quintile) grid | 20-60 gun: pred **-3.17e-3** vs measured **-3.23+-1.80e-3**; low-pT gun: pred -2.28 vs -5.35+-1.84. Cell-wise scale on an exogenous gen (pT,\|eta\|) grid **k = 0.98 +- 0.40 combined** (2.4 sigma from 0). In probe space model(-2.13) + Box(-2.75) = -4.88 vs measured -4.10+-1.52 (pull +0.51); model alone pulls -1.30 | **dp/p = +16e-6 (Z-like central) to +43e-6 (endcap)** -> **+1.5 to +3.9 MeV on m_Z** uncalibrated; see the calibration caveat below |
| 6 | **The CPE location bias is PIXEL-ONLY.** IDEAL geometry, 1 375 697 hits | BPix local x **+0.1193+-0.0027 sigma_CPE** (BPix-1 alone **+0.227**, +2.7 um, skew +0.302), FPix +0.040; every strip subdetector null at +-0.002 | -- |
| 7 | **The charge-even shift is strongly PHI-MODULATED**, with no eta structure | unbinned amplitudes `2<x sin(n phi)>`: **n=8 +18.96+-2.55e-3, n=10 +21.62+-2.55e-3**, total power 281.5/80 dof; permutation null 11/7/16 for 12 bins; lumi-parity split-half chi2(A==B) 20.9/24 with corr **+0.75**; charge-ODD amplitudes null and no 1/pT fall, so it is not a rotated field effect. In eta: chi2(FLAT) **15.8/23** | modulated **Delta(q/pT) ~ 4e-5 GeV^-1**, the same order as the AN's misalignment bound \|M\| < 1e-4 -- on ideal geometry |
| 8 | **The modulation IS the CPE location bias, and PART 1's exclusion was a SIGN ERROR.** `W5 = V^-1 F C^-1 E5` (`...G4e.cc:4619`) while the step taken is `dx = -C^-1 F^T V^-1 r` (:4098), so **`W5^T r = -dx`**: the exported influence functional has the OPPOSITE sign to the fit's response. (Variance uses are unaffected -- `v_b` is quadratic.) With the corrected sign the no-free-parameter propagation gives | **n=8 sin +18.68+-0.35 (meas +18.96+-2.55), n=10 sin +19.88+-0.39 (meas +21.62+-2.55)**; over n=1..20, **chi2(measured==PREDICTED) = 61.5/40 against chi2(measured==0) = 227.6**; the pixel dose response +0.2/+11.3/+19.6/+25.8/+23.2 against measured -6.9+-14.2/+10.3+-6.1/+24.0+-4.1/+26.1+-4.4/+30.9+-10.9; and the phi-AVERAGED bands now pull **-1.09 / -1.07 / +0.23** instead of PART 1's ">= 5.3 sigma" | as row 7 |
| 9 | **The traps.** Four conditioning variables that manufacture the signal | `corr(seed->final dq/p SIGNED, x) = +0.1132`, tertiles -169/-7/+163e-3; binning the charge-ODD channel on the FITTED `sigma` gives -68 -> +44e-3 across quintiles, pure artefact; a 24-bin phi scan ALIASES n=14/16/17 onto 10/8/7; `fitFromGenParms=True` deletes the observable (`refCov(0,0)=0`, `niter=1`) | -- |

## HYPOTHESISED

* **The identification of the harmonics.** n = 10 <- BPix layer 1 (Phase-0: 20
  ladders alternating inner/outer, so a bias fixed in local x enters the
  bending coordinate with period 2 ladders = 36 deg = n = 10); n = 8 <- the
  TEC petals, 8 per disk face. SUPPORTED by: the frequencies, the clean pixel
  dose response of n = 10 and its absence for n = 8, and the fact that the
  propagation keyed on the DetId ORIENTATION GROUP reproduces both. NOT
  established: the phases have not been matched to the surveyed ladder and
  petal azimuths (that needs the 160-task pass, ~4x the precision).
* **The phi-AVERAGED charge-even offset.** On the full baseline with no
  subsample selection it is `-3.7 +- 1.8e-3` (**2.0 sigma**) and the corrected
  prediction accounts for -1.3e-3 of it; the 3.4 sigma quoted earlier is for a
  subpopulation selected on a variable with `corr(v,\|x\|) = +0.054`. Treat it
  as not yet established.

## SIZES AND WHAT TO DO ABOUT THEM

| effect | size on the momentum scale | on m_Z | recommended next step |
|---|---|---|---|
| **Box second-order bias** (charge-ODD, scale-like) | `dp/p = -<z>_odd sigma_rel`: +21.5e-6 (barrel, `<p>` 46), +43.3e-6 (endcap, `<p>` 154), +16.1e-6 for a Z-like pT 40-50 \|eta\|<0.4 muon | **+1.5 MeV central, up to +3.9 MeV in the endcap, UNCALIBRATED** | Apply the analytic per-track correction `Delta(q/p) = +(K/2) sigma^2` with K from the `(d1, d2)` regression -- it needs NO new production (`refParms_iter0`, `trackParms`, `refParms`, `refCov` are already exported). **The calibration caveat is the key number**: the low-pT gun gives +25.3e-6 in the barrel against +21.5e-6 at 20-60 GeV, so a J/psi-anchored scale absorbs most of it and the RESIDUAL is the difference, **~4e-6 -> ~0.35 MeV (barrel), ~8e-6 -> ~0.7 MeV (endcap)**. Validate on Z and J/psi legs, where the statistics are 10x. |
| **CPE location bias**, phi-MODULATED (charge-EVEN, sagitta-like) | `Delta(q/pT) ~ 4e-5 GeV^-1` at n = 8 and n = 10 | first-order CANCELLING in the pair mass; matters through resolution and through its degeneracy with alignment | **Fix it in the CPE** (BPix-1 local x is +2.7 um): a per-class local-x offset removes it for every consumer. At analysis level the correction is `Delta(q/p) = +sum_b s_b sqrt(v_b) mu_b` per track (note the SIGN of row 8), which needs, per hit block: the DetId (for the orientation group) and the class -- `hitDetId` is exported by the single-track maker but **`resinfv`/`resinfbv`/`hitDetId` are NOT in the slim two-track productions** (booked under `exportStepRecords_`, `...Base.cc:547-549`), so the two-track path needs either `exportStepRecords=True` or a small maker change exporting an int8 sign plus the orientation-group index per block. |
| **Convergence / seed path** | < 0.004e-3 on the pull | < 0.1 keV | nothing. `edmConvergence=1e-7` costs +18 % wall for no measurable change. |

## FIGURES: ONE PANEL IS NOW WRONG
Every `conv_*` panel in `~/public_html/cvh/260908_hitclassbias/` shows
MEASUREMENTS (or the Box prediction, which comes from `(d1, d2)` and not from
`resinfbv`), so none of them is affected by the sign of row 8. The PART-1
panels `pred_vs_meas` and `lever_subdet` DO use it and therefore have the
prediction with the WRONG SIGN; regenerate them from `c9_phipred.py` (default
`--sign +1`) before any of them is shown.

## THE ONE THING TO CARRY FORWARD
The sign of the exported influence functional (row 8) inverts a previously
recorded conclusion. Anyone reusing `resinfv` / `resinfbv` for an ODD-moment
(location) calculation must apply `dx = -W5^T r`; VARIANCE uses are unaffected.

---


## ADDENDUM (hit-class agent, 2026-09-08 late) — the 160-task pass, and where
## the phi harmonics live

### (a) PART 3 confirmed at FULL statistics (160/160 tasks, 319 8xx tracks each)
The three variants finished; caches `data/conv_{base,tight,damp}_t160.npz`,
output `out_conv_t160.txt` (the 40-task record in `out_conv.txt` and
`data/conv_*_t40.npz` is untouched). Paired on `slot`:

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

At 4x the 40-task statistics the paired shift is **< 0.005e-3, more than
1000x below the -6.3e-3 bulk**, with errors of 0.001-0.002e-3. The 40-task
conclusion is confirmed, not merely unrefuted.

### (b) The azimuthal segmentation, counted from the DetIds in this sample
| structure | count | harmonic it can make |
|---|---:|---|
| **TEC petals per wheel** | **8** | **n = 8** |
| **BPix L1 ladders** | **20** | inner/outer alternation -> **n = 10** |
| BPix L2 / L3 ladders | 32 / 44 | n = 16 / 22 |
| FPix blades, panels | 24, 2 | n = 24 / 12 |
| TIB strings | 30 / 38 / 45 / 56 | -- |
| TOB rods | 42 / 48 / 54 / 60 / 66 / 74 | -- |

n = 8 and n = 10 are the only two that match a real structure, and they are
exactly the two the scan finds (`A_sin` +18.04 and +21.81 +- 2.50 on the full
sample; n = 12, 16, 24 are at -6.3, -5.2, -2.7).

### (c) BUT the two harmonics are NOT separable by subdetector, and BOTH are
### carried by tracks with a BPix LAYER-1 hit (5.4 sigma, eta-controlled)
Splitting on the influence-weighted share does NOT give one harmonic to TEC
and the other to BPix-L1 -- both rise with both shares -- and the TEC and TOB
shares are `corr = +0.886` and `-0.861` with `|eta|`, so those splits are
`|eta|` in disguise. The BPix-L1 share is not (`corr = -0.137`), and it is the
one that separates:

| at FIXED `\|eta\|` (12 quantile bins) | has BPix-L1 hit | no BPix-L1 hit | difference |
|---|---:|---:|---:|
| `A_sin(n=8)` | +24.00 +- 2.97 | **+3.59 +- 4.63** | +20.40 +- 5.50 (3.7 s) |
| `A_sin(n=10)` | +30.46 +- 2.97 | **+0.82 +- 4.63** | **+29.64 +- 5.50 (5.4 s)** |
| n | 226 481 | 93 369 | |

**Tracks without a BPix layer-1 hit show NO phi modulation at either
harmonic.** That localises the effect to the innermost pixel layer, which is
also where PART 1 measured by far the largest CPE location bias
(BPix-1 +0.227 sigma_CPE = +2.71 um, skew +0.302, against +0.058 and +0.044 in
L2 and L3 and null in every strip subdetector).

HYPOTHESIS, not established: `n = 10` is the BPix-L1 inner/outer ladder
alternation (20 ladders -> 10 pairs). `n = 8` is NOT explained by BPix-L1
geometry, so either the two harmonics have different origins and only the
lever arm to the innermost radius is common, or the split-by-share is too blunt
to separate them. CAVEAT: "no BPix-L1 hit" is 29 % of the sample and may be a
peculiar population in ways the `|eta|` control does not cover.

Script: `p1_phi_origin.py`. Data: `data/conv_ref903x_full.npz` joined to
`data/blocks_mugun_ul16_260903x.npz` (verified aligned, max |dz| = 0.0).

# RESUME HERE — 2026-09-08 (night), PART 3 = the CONVERGENCE test, DONE

## THE ANSWER
**The charge-even shift is NOT incomplete convergence, NOT seed/path
dependence, and NOT the second-order bias of the converged estimator.** All
three refits landed on the same minimum and the same number, and the Box bias
computed from the data's own curvature is ~200x too small in that channel and
zero by mirror symmetry. What the campaign DID find is a large phi-modulated
sagitta bias at n = 8 and n = 10 that the phi-averaged analysis had been
integrating away.

## WHAT IS RUNNING NOW (started by someone else, ~20:58)
`NT=160 NP=10 ./run_conv_all.sh` -> `logs/conv_master_160.log`, extending all
THREE variants from 40 to 160 tasks (30 cmsRun, ~3-4 h for `damp`). The
numbers below are pinned to the **first 40 tasks** (79 965 / 79 966 / 79 961
tracks), whose caches are frozen as `data/conv_{base,tight,damp}_t40.npz`.
`run_conv_analysis.sh` takes `NTASKS` (default 40) precisely so an unpinned
re-extraction cannot grow the sample under the reader; rerun it with
`NTASKS=160` when that pass finishes for 4x the precision on every number.

## THE TABLE (40 tasks, ~80 k tracks each, PAIRED where it matters)

| | base | tight | damp |
|---|---:|---:|---:|
| knobs | -- | `edmConvergence=1e-7 nIters=20` | `+ gnDampAfter=1 gnDampFactor=0.5 nIters=30` |
| `<niter>` | 2.190 | 2.870 | **3.500** |
| at the iteration cap | 15 (0.019 %) | 91 (0.114 %) | 17 (0.021 %) |
| `edmvalref >= 1e-7` | **65.66 %** | 0.11 % | 0.02 % |
| wall per task (2000 ev) | 587 s | 690 s (**+18 %**) | 842 s (**+43 %**) |
| tracks whose q/p moved vs base | -- | 47.5 % | **98.8 %** |
| median moved \|dq/p\|/(q/p) | -- | 2.6e-7 | 2.5e-6 |
| `<chi2/ndof>` | 0.998874 | 0.998874 | 0.998894 |
| charge-even `<x>` barrel | **-4.29** | **-4.29** | **-4.29** |
| middle | **-10.04** | **-10.04** | **-10.04** |
| endcap | **+5.17** | +5.15 | +5.00 |
| mixture OUT (bulk 90 %) | -5.53+-3.74 | -5.10+-3.75 | -5.44+-3.61 |
| mixture IN (top 10 %) | +19.71+-12.95 | +19.49+-13.06 | +18.87+-13.08 |
| **PAIRED `<dx>_even` vs base** | -- | **-0.000+-0.000 e-3** | **+0.000+-0.002 e-3** |

The paired column is the measurement. `damp` moves **98.8 %** of tracks by a
median 2.5e-6 in relative curvature -- a genuinely different path to the
minimum, not a longer walk down the same one -- and the charge-even statistic
moves by less than **0.004e-3**, i.e. **more than 1500x smaller than the
-6.3e-3 bulk**. `tight` reduces the fraction of tracks with `edmvalref` above
1e-7 from 66 % to 0.1 % and changes it by less than 0.001e-3. The chi2/ndof is
identical to six digits in all three: they are the same minimum.

**Criterion, if one were wanted anyway**: `edmConvergence=1e-7` costs +18 %
wall and +0.68 GN iterations per track and buys nothing measurable here. It is
not worth adopting for THIS observable; the only thing it changes is the 0.019
-> 0.114 % of tracks that hit the iteration cap, which is a diagnostic, not a
bias.

## STEP 2 — THE BIT CHECK: PASSED (2026-09-08, all 40 `base` tasks)

`c0_bitcheck.py --a data/conv_base.npz --b data/conv_ref903x.npz`,
79 965 paired tracks (0 unpaired on the `base` side):

* **3 tracks of 79 965 (0.0038 %) have a different converged q/p**, and the
  largest difference is **1e-4 sigma**. Every other exported variable
  (`qop_seed`, `chi2n`, `nvalid`, `npixhit`, `pt`, `qop_gen`, `slot`) is
  EXACTLY equal on every track.
* The charge-even `<x>` per band agrees to the printed precision
  (-4.29 / -10.04 / +5.17 e-3 in both).
* The three tracks are the known CVH limit-cycle / anchoring non-determinism:
  their `<niter>` is **5.67 against 2.19** for the sample, one goes 3 -> 5
  iterations and one sits at the `nIters` cap of 10 with `edmref` ~ 1.
  A moved build would change every track, not three.

**The baseline is the same estimator. The variants can be compared to it.**
Note for the record that CVH is NOT bit-reproducible run to run at the 1e-4
level of tracks -- it is reproducible everywhere except on tracks that sit on
an iteration boundary.

## RESULT ALREADY IN HAND (does not depend on the three refits) — 2026-09-08

### The second-order (Box) bias of the CONVERGED estimator: computed, and it is
### charge-ODD, not charge-even
`c4_secondorder.py` on the full 160-task baseline (319 854 tracks, 314 151
after trimming the extreme 1 % in |d1| and |d2|).

Regressing `d2 = b0 + a d1 + (K/2) d1^2` with
`d1 = refParms_iter0[0] - trackParms[0]`, `d2 = refParms[0] - refParms_iter0[0]`:

| charge | a | K [GeV] | predicted `<z> = -(K/2) sigma` |
|---|---:|---:|---:|
| q = +1 | +0.00010+-0.00003 | **+20.85 +- 0.30** | **-2.593 +- 0.038 e-3** |
| q = -1 | +0.00025+-0.00003 | **-21.29 +- 0.32** | **+2.632 +- 0.039 e-3** |

* `K_+ + K_- = -0.44 +- 0.44`, i.e. **K_- = -K_+ to 2 %**, exactly as the
  mirror map (q/p -> -(q/p)) requires of any ACHIRAL estimator effect.
* **PREDICTED charge-EVEN `<z>` = +0.019 +- 0.027 e-3** against a MEASURED
  **-4.26 +- 1.75 e-3**: the second-order GN bias is ~200x too small in the
  channel of interest, and it is small BY SYMMETRY, not by accident.
* **PREDICTED charge-ODD `<z>` = -2.61 +- 0.03 e-3**, measured (on `x`)
  -1.46 +- 1.90 e-3 -- consistent. In physical terms that is a MOMENTUM SCALE
  bias of `-2.6e-3 * sigma_rel` = **-4.7e-5 at sigma_rel = 0.018**, i.e. about
  5x the 1e-5 Z-mass target, and it is a property of the estimator, not of the
  detector.
* The linear coefficient `a` is +1e-4 to +2.5e-4, i.e. the seed-noise
  contamination the linear column is there to absorb is negligible: `d2` is
  essentially PURE quadratic in `d1`. `rms(d1)/<sigma> = 0.28` and the fit
  range reaches 1.45 sigma, so applying K at the sigma scale is interpolation,
  not extrapolation.

**Derivation of the sign** (it flips the answer, so it is written out in the
script docstring): with `B = sum f'^2`, `C = sum f' f''`, the stationarity
expansion gives `E[delta_2] = -(sigma_theta^2/2)(C/B)` while the NOISELESS
two-step gives `d2 = +(C/2B) d1^2`, so `K = C/B` and the bias carries a MINUS.
This is Box (1971).

### The CHARGE STRUCTURE, which is the sharp statement
Under the mirror map (reflection in a plane containing the beam) a `+` track
maps to a `-` track and `q/p -> -(q/p)`, so `Delta(q/p) -> -Delta(q/p)`:

* charge-**ODD** `<z>` == a charge-INDEPENDENT momentum shift == a SCALE bias.
  Allowed by the mirror. This is where a second-order estimator bias, an
  energy-loss mismodelling or a field-scale error lives.
* charge-**EVEN** `<z>` == a charge-DEPENDENT momentum shift == a SAGITTA-like
  bias. FORBIDDEN by the mirror. It can only be sourced by something CHIRAL:
  the module layout (tilted BPix ladders, the FPix turbine, stereo angles), a
  Lorentz-drift CPE bias with a fixed azimuthal sense, or a residual
  azimuthal-twist misalignment.

So the target `-6.3e-3` is a sagitta-type bias and it CANNOT be a generic
second-order estimator bias -- which is what the numbers above then confirm
numerically, with the data's own K.

### What SHAPE the charge-even shift has (`c5_scaling.py`, full baseline)
Per-track regression, every regressor entered as `f` (even) and `q f` (odd),
318 239 tracks with |x| < 10 and sigma_rel < 0.1:

| shape | meaning | even coefficient | dchi2 (even, odd already in) |
|---|---|---:|---:|
| `c sigma_rel` | second order | -0.118 +- 0.083 (1.4s) | 2.09 |
| `D/sigma` | fixed additive `Delta(q/p)` | -0.0094 +- 0.0039 (2.4s) | 6.10 |
| `D_T/(sigma cosh eta)` | fixed `Delta(q/pT)`, a sagitta | -0.0187 +- 0.0068 (2.8s) | **7.91** |
| constant | the per-band null shape | -0.0037 +- 0.0018 (2.0s) | 4.37 |

i.e. `Delta(q/pT) = -1.87e-6 GeV^-1` (the AN's misalignment bound is
|M| < 1e-4 GeV^-1, so this is 50x inside it). The exogenous GEN pT x GEN |eta|
grid (24 cells, sigma_rel lever arm 4.96) says the same with the same ranking.
**The data mildly PREFER a fixed sagitta offset over an O(sigma) second-order
bias, but dchi2 ~ 6 is a preference, not a discrimination.**

---

## NEW RESULT — THE CHARGE-EVEN SHIFT IS STRONGLY PHI-MODULATED (n = 8 and n = 10)

`c6_phi.py` on the full 160-task baseline. This was NOT in the brief; it fell
out of adding `genphi` to the cache for a null test that turned out not to be
null.

**Unbinned harmonic scan** (amplitudes `2<x cos/sin(n phi)>`, charge-even; no
binning, hence no aliasing -- a 24-bin scan folds n = 14/16/17 onto 10/8/7 and
is what almost sent this the wrong way):

| n | cos [1e-3] | sin [1e-3] | power |
|---|---:|---:|---:|
| 7 | +6.52+-2.55 | +5.54+-2.56 | 11.2 |
| **8** | +4.07+-2.56 | **+18.96+-2.55** | **57.7** |
| **10** | +0.51+-2.55 | **+21.62+-2.55** | **71.7** |
| 14 | +5.50+-2.55 | +6.58+-2.55 | 11.3 |
| 17 | -10.41+-2.56 | +3.59+-2.55 | 18.6 |

Total power n = 1..40 is **281.5 for 80 dof**. The mean over phi is the
familiar -3 to -4e-3; the MODULATION is five times larger.

**It is real, not a machinery artefact:**
* PERMUTATION null (phi reassigned at random): chi2(no effect) = 11.1 / 7.0 /
  16.4 for 12 bins -- the estimator and its errors are correct.
* SPLIT-HALF on the LUMI parity (disjoint event sets): 24-bin
  chi2(A == B) = 20.9/24 and **corr(A, B) = +0.75**; per-bin the two halves
  track each other (e.g. bin 5: -27.9 vs -24.9). A fluctuation cannot do that.
* It survives every tail cut (chi2 rises, not falls, from |x| < 30 to |x| < 5)
  and the within-pair correlation of the gun's two muons is -0.0007, so the
  bootstrap is not being fooled by event-level correlation.
* The gun is phi-uniform and charge-balanced in every bin (frac q+ = 0.4968 to
  0.5020) and its two muons are NOT back-to-back (median |dphi| = 1.57), so
  the charge-even average at fixed phi is well defined.

**Where the two harmonics live:**

| selection | n=8 sin | n=10 sin |
|---|---:|---:|
| barrel | +11.5+-4.2 | +10.6+-4.2 |
| middle | +18.3+-4.7 | +22.5+-4.7 |
| endcap | +28.0+-4.4 | +33.3+-4.4 |
| nValidPixelHits >= 3 | +22.4+-4.0 | **+26.5+-4.0** |
| nValidPixelHits <= 1 | +24.7+-5.6 | **+7.7+-5.6** |

Both GROW with |eta|. **n = 10 is carried by the pixel-rich tracks and dies on
the pixel-poor ones; n = 8 is indifferent to the pixel content.** The natural
identifications in the Phase-0 2016 geometry are

* **n = 10 <- BPix layer 1, 20 ladders in an alternating inner/outer
  (turbine) arrangement**: the inner/outer pattern has period 2 ladders = 18
  deg = the n = 10 harmonic, and PART 1 measured exactly the ingredient it
  needs -- a **+0.227 sigma_CPE location bias in BPix-1 local x**, whose
  BENDING-SENSE projection flips with the ladder orientation. Averaged over
  phi that bias buys only +1e-3 (PART 1's null result); MODULATED at the
  ladder frequency it does not average away, and it is +21e-3.
* **n = 8 <- TEC petals** (8 per disk face), which is also why it is
  independent of the pixel content and grows into the endcap.

**The obvious loophole is CLOSED** (`c8_phiquad.py`). `genphi` is the azimuth
at the PCA, but the modules are crossed at `phi_PCA + q delta` with
`delta ~ 0.3 B r / pT` = 0.057 rad at pT 20 GeV, so a modulation that is
charge-ODD at the MODULE leaks into the charge-EVEN channel at the PCA, in the
other quadrature, with weight `sin(n delta)` = 54 % at n = 10. Two tests kill
that explanation:

| n | EVEN cos | EVEN sin | ODD cos | ODD sin | err |
|---|---:|---:|---:|---:|---:|
| 8 | +4.1 | **+19.0** | +1.1 | +0.3 | 2.6 |
| 10 | +0.5 | **+21.6** | -2.7 | +0.4 | 2.6 |
| 17 | **-10.4** | +3.6 | +0.6 | -0.6 | 2.6 |

* the **charge-ODD amplitudes are null at every harmonic** -- there is no
  charge-odd modulation to rotate;
* the charge-EVEN amplitude does **not fall as 1/pT** (n = 10: +18.1 / +16.1 /
  +18.8 / +33.5 across pT 20-30/30-40/40-50/50-60; n = 8: +16.4 / +19.6 /
  +28.8 / +11.0), whereas a rotated charge-odd source has to fall by a factor
  3 over that range.

So the modulation is genuinely charge-even AT THE MODULE -- a sagitta effect,
not a field effect.

### The ETA structure, for contrast: there is none (`c7_eta.py`)
24 bins in SIGNED gen eta, analytic errors: **mean -3.64 +- 1.81 e-3,
chi2(no effect) 19.9/24, chi2(FLAT) 15.8/23**, and the eta-ODD part
(a z-antisymmetric sagitta twist) is **7.9/12**. The charge-even bias is a
FLAT offset in eta with a large modulation in PHI -- which retires the phrase
"eta-dependent charge-even skew" for good and confirms PART 2's mixture
reading from a completely different direction.

**This does not explain the phi-averaged -6.3e-3** -- these harmonics
integrate to zero over phi by construction -- but it is the missing piece of
PART 1: the CPE location bias IS producing a large sagitta bias, just not a
phi-uniform one. In size, `<x> ~ 25e-3` at `sigma_rel ~ 0.03`, `p ~ 100 GeV`
is `Delta(q/pT) ~ 4e-5 GeV^-1`, i.e. the same order as the AN's misalignment
bound |M| < 1e-4 GeV^-1 -- on IDEAL geometry, where there is no misalignment
at all. HYPOTHESIS for the identification; ESTABLISHED for the existence,
the frequencies, and the pixel/eta dependence.

## WHAT REMAINS, and how the campaign now stands

Every estimator mechanism on the table has been excluded:

| candidate | verdict | where |
|---|---|---|
| a one-sided TAIL from a subpopulation | excluded (core shift, flat from T = 2) | PART 2 (1) |
| the step control / momentum clamp | excluded (never fires on a 20-60 GeV gun) | PART 2 (2) |
| the hit-class CPE LOCATION bias, phi-averaged | excluded (+1.0e-3, wrong sign, >= 5.3 sigma at every granularity) | PART 1 |
| an eta-dependent region effect | excluded (chi2(FLAT) 15.8/23 over 24 signed-eta bins) | `c7_eta.py` |
| INCOMPLETE CONVERGENCE | **excluded** (paired `<dx>_even` < 0.001e-3 under a 100x tighter tolerance) | PART 3 |
| SEED / PATH dependence | **excluded** (paired `<dx>_even` = +0.000 +- 0.002e-3 with 98.8 % of tracks moved) | PART 3 |
| the SECOND-ORDER (Box) bias of the converged estimator | **excluded in the charge-even channel** (+0.02 +- 0.03e-3 predicted, and zero by the mirror map) -- but it is a REAL **-2.6e-3** in the charge-ODD channel | PART 3 |

What is LEFT for the phi-averaged -3.6 to -6.3e-3:

1. **A statistical fluctuation.** On the FULL 160-task baseline with no
   subsample selection the per-track regression gives `<x> = -3.7 +- 1.8e-3`,
   i.e. **2.0 sigma**. The 3.4 sigma quoted in PART 2 is for the OUT
   subpopulation, selected on a variable with `corr(v, |x|) = +0.054`. The
   most economical reading is that the phi-AVERAGED bulk is not yet
   established at all.
2. **A chiral source beyond the hit-location channel** -- which is exactly
   what the n = 8 / n = 10 modulation shows exists and is large; its
   phi-average is zero only if the module layout is exactly periodic, and it
   is not (the pixel barrel has 20/32/44 ladders, the TEC 8 petals per face,
   so the harmonics do not commensurate).

**The recommendation**: stop spending statistics on the phi-averaged number
and go after the modulation, which is 5x larger, 8 sigma per harmonic, and has
a dose-response in the pixel content. The two concrete next steps are
(a) rerun with `NTASKS=160` when the running pass finishes -- 4x the precision
on every number above, and enough to map the harmonic phases against the
ladder/petal positions; (b) repeat the harmonic scan on the Z and J/psi legs,
where a phi-modulated sagitta bias of `Delta(q/pT) ~ 4e-5 GeV^-1` is directly
relevant to the calibration and is NOT removed by the per-module alignment
corrections if it is a CPE effect rather than a geometric one.

## Established results (do not re-derive)
* **PART 1**: the per-hit CPE LOCATION bias is real but pixel-only -- BPix
  +0.119 sigma_CPE (+1.26 um; layer 1 +0.227 / +2.71 um, skew +0.302), FPix
  +0.040, every strip subdetector null at +-0.002 (3-sigma bounds 0.04-0.23 um)
  -- and CANNOT make the observed odd moment: it predicts +0.86 / +1.58 /
  +1.39 e-3 against a measured -4.89 / -3.57 / +0.12, wrong sign and flat in
  eta, and the chi2 of the displacement it would need is >= 5.3 sigma at every
  key granularity from per-subdetector to per-orientation-group-x-class. The
  skew channel is 1e-6. Step 3 of that brief was therefore NOT done.
* **PART 2**: the charge-even shift is a CORE shift (trim scan flat from
  T = 2), the step control NEVER fires (`nChargeFlipProtect` = 0 for
  319 849/319 850, `chargeHypFlipped` = 0, `niter` = 2 for 81 %), and the eta
  dependence is a MIXTURE: `+20.59 +- 6.45 e-3` (top 10 % in
  `|seed->final dq/p|`) and `-6.33 +- 1.84 e-3` (the rest), BOTH flat in eta,
  mixing 2.4 % -> 7.0 % -> 21.2 %.
* **NEW CONDITIONING TRAP, the fourth of the campaign**:
  `corr(seed->final dq/p SIGNED, x) = +0.1132` -- bigger than the reco-pT
  trap's +0.042 -- and its tertiles run **-169 / -7 / +163 e-3**. The seed is
  the generalTracks KF, which shares its hits with CVH, so the difference is
  partly the residual itself. The ABSOLUTE step is safe (`+0.0061`).
* **`resinfv` / `resinfbv` / `hitDetId` are absent from the slim v2 two-track
  productions** (booked under `if (exportStepRecords_)`,
  `ResidualGlobalCorrectionMakerBase.cc:547-549`), so a SIGNED per-hit
  mass-projected weight cannot be extracted from them.

## Pointers
* the analysis agent is `a0808263b90f22ec5`; its state is
  `calibration_studies/fullscale/STATE.md` (secs. 0f.35-0f.47).
* its gen-leg caches, when they land:
  `fullscale/runs/auxgen_dyv2.npz`, `fullscale/runs/auxgen_jpsiv2.npz`
  (produced by `fullscale/run_auxgen.sh`, commit `fbcecef`).
* figures: `~/public_html/cvh/260908_hitclassbias/` (13 PART-1/2 panels plus
  the 8 PART-3 panels `conv_band`, `conv_mixture`, `conv_niter`,
  `conv_scaling`, `conv_dqop`, `conv_boxbias`, `conv_phi`, `conv_phiharm`;
  index.php present).
* PART 3 outputs: `out_conv.txt` (the driver's single pass), plus `out_c1.txt`
  / `out_c2.txt` / `out_c4.txt` / `out_c6.txt` from the pinned `_t40` caches.
* NOTES: two dated entries in `/work/submit/david_w/Documents/Resolution/NOTES.md`.
* **`/ceph` is NOT readable from submit82** (cephx eviction) and the mfs venv
  does not run there either. Use `resolution/hitclassbias/rrun.sh`, which
  shells to submit50 over a persistent SSH ControlMaster.

---

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

## >>> SETTLED, AND THE HYPOTHESIS IS REFUTED (see `fullscale/STATE.md` 0f.52)
The Z- and J/psi-leg test was done at 8-16x the gun's statistics, using the
`|seed -> final dq/p|` that the two-track trees DO carry
(`Mu{plus,minus}trk_pt/eta` = the KF seed, `Jpsi_qopref{plus,minus}` = the CVH
reference q/p; `oddmoment/aux_seed.py`). The result:

* **the mixing FRACTION reproduces the gun strikingly** -- 2.68 / 9.69 /
  20.04 % on the Z legs against the gun's 2.4 / 7.0 / 21.2 % -- so the
  population the step selects IS the same one;
* **but NEITHER component is `eta`-flat**, which is what the hypothesis
  required. The OUT component alone (85-90 % of the sample) runs
  +3.97 -> -3.46 -> -18.22 e-3 on the Z legs (`chi2` vs flat **577.7 / 2**)
  and +2.08 -> +1.09 -> -4.73 at mass level (**102.0 / 2**). Removing the top
  10 % in `|dq/p|` removes essentially none of the `eta` dependence.

**So the `eta` dependence lives in the BULK, not in the mixing fraction, and
the two-component mixture is NOT the explanation.** On the gun both components
looked flat only because the per-band errors were +-3e-3 against the legs'
+-0.5-0.8e-3; and the Z legs' band pattern has the OPPOSITE `eta` trend to the
gun's, which is a caveat on comparing the two channels at all. Note also
`corr(|dq|, |x|)` = +0.082 / +0.100 / +0.156 on Z legs / J/psi legs / mass
level -- larger than the reco-pT trap's +0.042 -- so the IN/OUT *values* carry
a selection effect (the fractions and the flatness `chi2` do not).

What survives from this section: the trim scan (core shift, not a tail), the
step-control census (it never fires), and the conditioning trap
`corr(signed seed->final, x) = +0.1132`. The MIXTURE reading does not.

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

---

## NOTE FROM THE ANALYSIS AGENT — 2026-09-08 21:00 (appended, nothing above touched)

**Your discriminator is now available on the REAL legs, and the mixture does
NOT reproduce there.** Full detail in `fullscale/STATE.md` sec. 0f.52 and the
dated NOTES entry; the short version:

`|seed -> final dq/p|` IS recoverable from the slim v2 two-track trees --
`Mu{plus,minus}trk_pt/eta` is the generalTracks KF seed and
`Jpsi_qopref{plus,minus}` the CVH reference `q/p` (`== Mu*_refParms[0]`, checked
to 1e-7). `resolution/oddmoment/aux_seed.py` extracts it; outputs, aligned row
by row to the pairs caches and validated by `sigrel` coming back bit-identical
to the cache's `sigrelp`/`sigrelm`:

| file | rows | aligned to |
|---|---:|---|
| `fullscale/runs/auxseed_dyv2.npz` | 3 733 323 | `runs/zpairs_dyv2_full.npz` |
| `fullscale/runs/auxseed_jpsiv2.npz` | 7 923 460 | `runs/jpairs_v2_n600.npz` |
| `fullscale/runs/auxgen_dyv2.npz` | 3 733 323 | same (gen leg kinematics + Q-matrix shares) |
| `fullscale/runs/auxgen_jpsiv2.npz` | 7 923 460 | same |

**RESULT (`fullscale/mixture_legs.py`).** Your MIXING FRACTION reproduces
almost exactly on the Z legs -- 2.68 / 9.69 / 20.04 % against your
2.4 / 7.0 / 21.2 % -- so the step selects the same population. **But neither
component is `eta`-flat there.** Leg level, truth-referenced charge-even, 1e-3:

| | IN (top 10 %) | OUT (the bulk) | chi2 vs `eta`-flat |
|---|---|---|---|
| Z legs, 7.46 M | +7.65 / +5.77 / -9.49 | +3.97 / -3.46 / **-18.22** | 35.8 / **577.7** per 2 |
| J/psi legs, 15.8 M | +3.82 / -0.17 / -1.18 | +1.47 / -0.54 / **-3.26** | 1.0 / **77.4** |

and the same at mass level (OUT chi2 72.2 / 2 on Z, 102.0 / 2 on J/psi). The
bulk is 85-90 % of every sample and it alone carries the whole `eta`
dependence. **So the two-component picture is a property of the gun, not of the
reconstruction**, at 8x and 16x its statistics.

Two things that matter for how you read this:
1. `corr(|dq|, |x|)` is **+0.082** (Z legs), **+0.100** (J/psi legs) and
   **+0.156** at mass level -- bigger than your gun's +0.0061 and bigger than
   the reco-`pT` trap. The FRACTIONS and the flatness `chi2` are safe; the
   IN/OUT *values* carry a selection effect.
2. The Z legs' band pattern (+4.1 / -2.6 / -16.4) has the OPPOSITE `eta` trend
   to your gun's (-4.9 / -3.6 / +0.1). Your gun is single-track in 150X with a
   different alignment payload; the legs are two-track (mass- and
   vertex-constrained) in 106X. Different estimators -- do not difference them.

**This does not settle your convergence question**, which is still the right
one: if `tight`/`damp` shrink the components on the gun, the gun's mixture was
incomplete convergence and the disagreement with the legs is explained. Your
refits stood at base 39/40, tight 30/40, damp 22/40 at 2026-09-08 21:00.

**One more thing you will want**: `--freezeParameters` in `rabbit-vmass` does
NOT hold frozen parameters fixed during the step (`tf.stop_gradient` only,
minimiser on the full vector -> an exactly-null Hessian subspace ->
`trust-exact`'s hard case walks in it). Measured displacements of the frozen
`k_hit` run up to 0.126. `fullscale/STATE.md` sec. 0f.54. If any of your
numbers came from a rabbit fit with frozen parameters, check them.

---

## 2026-09-09 — the phi harmonics ARE BPix-L1 ladder parity (ladder-index test)

Written by the fullscale agent, taking up the ladder-index test handed over at
the end of the phi section ("the ladder index is in the DetId, `(id>>8)&0xFF`,
which would test the n=10 assignment head-on rather than through a share").
Script: `p2_ladder.py`. Sample: mugun UL16 `conv_ref903x_full` +
`blocks_mugun_ul16_260903x`, truth-referenced pull `x = z/(1 - a q z)`,
`|x| < 30`, 319 850 good tracks.

**Selection fix that matters.** A first pass counted 460 579 BPix-L1 "hits" on
226 484 tracks (2.03/track) — those are *blocks*, and every pixel hit books a
local-x AND a local-y block. The correct hit selection is
`(b_sd==1) & (b_lay==1) & (b_isy==0)`. 222 674 tracks then have exactly one
BPix-L1 hit; 20 distinct ladders, indices 1-20, as Phase-0 BPix geometry
requires.

### The result: a perfect parity alternation

`<x>` per ladder alternates with ladder parity with no exception in 20 ladders
(1e-3): odd +40.6 +30.2 +34.3 +26.6 +57.2 +34.9 +31.9 +42.4 +23.8 +15.4;
even -43.7 -41.3 -51.4 -28.4 -35.0 -30.6 -35.2 -44.4 -36.2 -48.9.

| quantity | value (1e-3) |
|---|---|
| odd ladders, n=107 902 | **+33.01 +- 3.15** |
| even ladders, n=114 772 | **-38.77 +- 3.07** |
| parity difference (even - odd) | **-71.58 +- 4.40  (16.3 sigma)** |
| same, at fixed \|eta\| (12 quantile bins) | **-71.78 +- 4.24** |

**Both harmonics are this and nothing else.** Transforming to ladder-local
azimuth `phi - phi_lad`, `phi_lad = 2 pi (lad-1)/20`, kills both:

| harmonic | global azimuth | ladder-local azimuth |
|---|---|---|
| n = 8 | +24.63 +- 3.00 | **-0.62 +- 3.00** |
| n = 10 | +30.77 +- 3.00 | **+0.42 +- 3.00** |

This answers the open question left in the phi section ("n = 8 is NOT explained
by BPix-L1 geometry"). It is: n=8 is not a separate mode, it is leakage from
the same 20-fold ladder pattern, which is not a pure sinusoid. The n=10
assignment is confirmed head-on rather than through a share, and the "no
BPix-L1 hit = 29 % of the sample" caveat is bypassed entirely — this test never
uses that control.

### It is charge-EVEN, so it is scale-like, not alignment-like

| | q = +1 | q = -1 |
|---|---|---|
| odd ladders | +29.76 +- 4.45 | +36.25 +- 4.48 |
| even ladders | -38.83 +- 4.33 | -38.71 +- 4.34 |
| parity difference | -68.59 +- 6.21 | -74.96 +- 6.23 |

charge-EVEN part **-71.77 +- 4.40**, charge-ODD part **+3.18 +- 4.40**
(consistent with zero). A misalignment would be charge-odd. This is a
CPE/incidence-angle effect that flips sign with the turbine tilt of alternate
(inner/outer) ladders in a BPix-L1 pair.

`|eta|` dependence of the parity difference (1e-3): -71.5 +- 10.4 (0-0.5),
-26.9 +- 11.2 (0.5-0.9), -52.5 +- 10.9 (0.9-1.3), -69.5 +- 9.2 (1.3-1.8),
**-112.7 +- 8.3 (1.8-2.4)**. Largest at shallow incidence, as a cluster-shape
effect should be.

### BUT it does NOT carry the bulk shift — this closes the phi lead as a lead

The alternation is occupancy-balanced and averages away:

| sample | `<x>` (1e-3) |
|---|---|
| all good, n=319 850 | -3.59 +- 1.84 |
| has exactly one BPix-L1 hit, n=222 674 | **-3.99 +- 2.20** |
| no BPix-L1 hit, n=93 369 | **-1.35 +- 3.40** |

-2.6 +- 4.1 between them: nothing. Per `eta` band, has vs no BPix-L1:
-6.79 +- 3.80 vs -0.36 +- 4.97 (`|eta|`<0.9), -6.08 +- 3.97 vs -0.84 +- 6.67
(0.9-1.6), -1.04 +- 3.59 vs -3.57 +- 6.51 (1.6-2.4) — every band consistent.
The residual -4e-3 of the BPix-L1 sample is just the small occupancy/magnitude
imbalance of the alternation itself (107 902 x +33.01 + 114 772 x -38.77 over
222 674 = -3.99), at 1.8 sigma.

**Verdict.** The phi structure is a real, 16-sigma, charge-even BPix-L1 ladder
CPE defect of amplitude +-36e-3 per track — worth reporting on its own — but it
has zero net mean and is NOT the source of the bulk -6.3e-3 charge-even shift,
and not of the mass-level endcap miss either. The phi lead is closed as a lead
for the bulk. What survives is a per-track resolution/bias term that a
ladder-parity (or incidence-angle) hit class would absorb.
