# PART 3 RESULT (2026-09-08, 40/160 tasks) — THE FIT IS NOT CONVERGED FOR ~15 % OF TRACKS

The three variants COMPLETED on 40 tasks (79 965 tracks each) and the 160-task
extension is running detached (same commands, `NT=160`). Numbers below are the
40-task set, paired on (run, lumi, event); 40 000 tracks common to all four.

## The knobs did what they should
| variant | `<niter>` | niter distribution | wall/task | cost |
|---|---:|---|---:|---:|
| base | 2.19 | 2: 81.2 %, 3: 18.7 % | 669 s | -- |
| tight | 2.87 | 2: 16.6 %, 3: 81.8 %, >=4: 1.5 % | 753 s | **+12.6 %** |
| damp | 3.50 | 2: 34.6 %, 3: 21.3 %, 4: 20.1 %, 5: 13.9 %, >=6: 10 % | 947 s | **+41.6 %** |

`base` vs the `260903x_m0` production on the same 40 tasks: **2 tracks of
40 000 differ** (max |dz| 7.8e-4, median 0) -- reproducible, the two areas are
the same build. The variants are interpreted against `base`.

## THE HEADLINE: 15 % of single-track fits are not converged, and they are invisible

Per-track change when the tolerance is tightened 1e-5 -> 1e-7:

| |z_tight - z_base| > | 1e-4 | 1e-3 | 0.01 | 0.1 | 0.5 | 1.0 | 3.0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| fraction | 23.4 % | 21.4 % | 21.3 % | 20.1 % | **15.4 %** | **10.1 %** | 0.8 % |
| damp | 80.9 % | 49.1 % | 48.9 % | 46.3 % | **35.0 %** | 22.9 % | 2.0 % |

The MEDIAN change is 0.0000 and the 90th percentile is 1.01 sigma: about 79 %
of tracks are converged to <1e-4 sigma and the rest move by O(1) sigma. The
unconverged fraction is **FLAT in eta** (15.48 / 14.82 / 15.88 %).

**And they are indistinguishable from the converged ones in every diagnostic
the maker exports:**

| | frac | `<niter>` | chi2/ndof | sigma_rel | nvalid | npixhit | log10 edmval | `<z>`_even |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| unconverged | 0.154 | 2.189 | 1.006 | 0.0203 | 17.27 | 2.249 | 0.600 | **-21.41e-3** |
| converged | 0.846 | 2.191 | 0.998 | 0.0203 | 17.19 | 2.239 | 0.615 | **-0.31e-3** |

**The charge-even shift lives entirely in the unconverged 15 %**:
0.846 x (-0.31) + 0.154 x (-21.41) = -3.56e-3, which is the base sample's
inclusive value. The converged 85 % sit at -0.31e-3, i.e. at ZERO.
(CAVEAT: the partition is defined by the PAIR `z_tight - z_base`, so it is not
a selection on `z_base` alone but it is not independent of it either. The
selection-free version is the paired inclusive shift below.)

## The selection-free numbers, and why 40 tasks is not yet enough
Paired charge-even shift, truth-referenced, all bands:
**tight - base = +3.37 +- 3.45e-3**, **damp - base = +5.21 +- 5.19e-3**
(on raw z: +4.04 and +6.09). Both POSITIVE, both about the size of the
-6.3e-3 bulk, both ~1 sigma. Per band, tight-base = +9.66 +- 5.75 /
+4.76 +- 6.55 / -4.98 +- 5.60.

And the mean shift is ALL tail: trimmed at |dz| < 0.5 it is +0.64 / +1.40 /
-0.05 (tight) and +0.74 / +2.99 / -2.39 (damp), i.e. zero, against
+11.64 / +4.32 / -4.80 and +6.62 / +8.72 / +3.18 untrimmed.

**160 tasks will give +-1.7e-3 on the paired shift, a 2.5-3.7 sigma
discrimination between "the bulk vanishes" (+6.3) and "it persists" (0).**
That run is live; re-extract with `extract_conv.py` and re-run
`s8_variants.py`, `s9_who_moved.py`, `s10_tail.py`, `s11_unconv.py`.

## What is already ESTABLISHED regardless of the 160-task outcome
* The CVH single-track fit at the default `edmConvergence = 1e-5` leaves
  **15 % of tracks moving by more than half their own resolution** and 10 % by
  more than a full sigma when the tolerance is tightened 100x. The maker's own
  comment already says the iteration does not monotonically decrease the chi2
  because every iteration re-propagates and re-linearises; this measures what
  that costs.
* **No exported per-track diagnostic identifies them** -- `niter`, `edmval`,
  `chi2/ndof`, `sigma_rel`, hit counts are all equal to 3 decimal places
  between the two populations. (`edmval` is stored but the criterion uses
  `edmvalref` on the reference block; both were extracted and neither
  separates them.)
* The price of fixing it is **+12.6 % wall time** for `edmConvergence=1e-7`
  (`<niter>` 2.19 -> 2.87), or +41.6 % for the damped path.

---

# RESUME HERE — 2026-09-08 (resumed session), PART 3 = the CONVERGENCE test

## Status of PART 3
* **Step 1 (jobs)**: three refits launched 20:00 EDT are RUNNING on submit50,
  10 tasks in parallel each, ~5 min/task, ~12 min per wave of 10, so ~50 min
  for 40 tasks. Progress: `ls -d <dir>/task_*/.complete | wc -l` (out of 40).
* **Analysis chain, WRITTEN AND SMOKE-TESTED, ready to run**:

  | script | what it does |
  |---|---|
  | `extract_conv.py` | per-track cache; now also writes `slot`, `geneta`, `qop_gen`, and takes `--ntasks` (needed to cut the 160-task baseline down to the same 40) |
  | `conv_common.py` | the truth-referenced `x = z/(1-a q z)`, the bands, the charge-even/odd bootstrap estimators, and `pair()` |
  | `c0_bitcheck.py` | `base` vs `260903x_m0`, EXACT equality track by track — the gate |
  | `c1_conv.py` | per variant: (A) charge-even per band, (B) trim scan, (C) mixture at p90, (D) iteration census, (E) the sigma-scaling second-order test |
  | `c2_pair.py` | PAIRED variant-to-variant `<dx>_even` per band + the mixture recomputed on the paired set |
  | `c4_secondorder.py` | the no-free-parameter Box-bias prediction from `(d1, d2) = (seed->iter0, iter0->final)` |
  | `c3_figs.py` | 5 panels, each with a ratio panel, into `~/public_html/cvh/260908_hitclassbias/` |

* **Two facts established while waiting, both needed to read the results**:
  1. The gun puts **TWO muons of OPPOSITE charge in each event** (999 of 1000
     events in task_0000 have 2 candidates), so the pairing key is
     `(run, lumi, event, charge)` — NOT the within-event slot, which shifts if
     one candidate of a pair fails the covariance cut in one variant only.
  2. The convergence break is on **`edmvalref`**, not `edmval`
     (`ResidualGlobalCorrectionMakerG4e.cc:4351-4354`; `edmval` is O(1-30) at
     the exported iteration while `edmvalref` is O(1e-7)). Its median is
     2.8e-7 and **64 % of tracks have `edmvalref` > 1e-7**, so `tight` really
     does buy extra iterations for most of the sample — the test has lever arm.
  3. `gnDampAfter`/`gnDampFactor`/`nIters`/`edmConvergence` are all real,
     wired cfi parameters (`:418-432`, `:4195-4198`, `:4351-4354`) — no silent
     no-op.
* `refParms_iter0` is filled at `iiter==0`, i.e. AFTER the first GN update
  (`:4309-4311`), so `d1 = refParms_iter0[0] - trackParms[0]` is the first GN
  step from the seed and `d2 = refParms[0] - refParms_iter0[0]` is everything
  after it. That is what `c4_secondorder.py` regresses.

## STEP 2 — THE BIT CHECK: PASSED (2026-09-08, on the first 21 `base` tasks)

`c0_bitcheck.py --a data/conv_base_partial.npz --b data/conv_ref903x.npz`,
41 985 paired tracks (0 unpaired on the `base` side):

* **3 tracks of 41 985 (0.0071 %) have a different converged q/p**, and the
  largest difference is **1e-4 sigma**. Every other exported variable
  (`qop_seed`, `chi2n`, `nvalid`, `npixhit`, `pt`, `qop_gen`, `slot`) is
  EXACTLY equal on every track.
* The charge-even `<x>` per band agrees to the printed precision
  (+5.36 / -5.24 / +3.23 e-3 in both).
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

## THE DECISION RULE (unchanged)
If the bulk `-6.3e-3` and the `+21e-3` IN component SHRINK under `tight` or
`damp`, the mechanism is incomplete convergence / seed dependence: quote the
cost per track and the criterion to adopt. If they PERSIST, they are a
property of the converged estimator, and `c4_secondorder.py` is the
first-principles comparison.

---

# RESUME HERE — 2026-09-08, session ended mid-campaign

## What is RUNNING right now (detached, survives the session)
Three convergence-diagnostic refits of the 20-60 GeV tight muon gun, launched
2026-09-08 ~20:00 EDT on **submit50** with

    setsid nohup ./run_conv_all.sh > logs/conv_master.log 2>&1 < /dev/null &

(`resolution/hitclassbias/run_conv_all.sh` -> `run_conv.sh <variant> 40 10`;
40 tasks x 2000 events, 10 parallel per variant, 30 cmsRun total, under the
32 cap). Rate measured on the smoke: ~100 ev/min/task, so ~20 min/task,
~80 min per variant for `base`/`tight` and longer for `damp`.

| variant | output dir (under `/ceph/submit/data/user/d/david_w/ZMass/cvh/`) | knobs |
|---|---|---|
| base | `resolution_trackres_mugun_ul16_260909_conv_base` | none — like-for-like bit check |
| tight | `..._260909_conv_tight` | `edmConvergence=1e-7 nIters=20` (100x tighter) |
| damp | `..._260909_conv_damp` | `edmConvergence=1e-7 nIters=30 gnDampAfter=1 gnDampFactor=0.5` |

Per-task logs `<dir>/task_XXXX/local.log`, sentinel `<dir>/task_XXXX/.complete`,
driver logs `resolution/hitclassbias/logs/conv_{base,tight,damp}.log` and
`logs/conv_master.log`. `run_conv.sh` RESUMES on the sentinel, so re-running it
with the same arguments picks up where it stopped. **Check progress with**
`ls -d <dir>/task_*/.complete | wc -l` (out of 40).

Everything reproduces `run_all_260903x.sh`'s `mugun_ul16_260903x_m0` arm
exactly (same cfg in `CMSSW_15_0_19_patch2_dev`, commit `ca6058d96fc`, which is
byte-identical to dev2; same filelist, same COMMON, `CgfQoPMode=0`); only the
GN knobs differ. **No build was done and none is needed.**

**`fitFromGenParms=True` was NOT used, and must not be**: measured on
`hitres2_mugun_ul16`, it freezes the reference block EXACTLY at gen
(`max|refParms-genParms| = 0.0`, `refCov(0,0) = 0`, `niter = 1`), so
`z = (refParms[0]-genParms[0])/sigma` is identically zero — it deletes the
observable rather than removing the seed dependence. `damp` is the substitute:
halving the step from iteration 1 on means the fit CANNOT land at the
seed-proximal point in two iterations, so it reaches the same minimum by a
different path. There is no `minIters` cfi parameter; adding one needs a build.

## What to do with them when they finish
1. `python3 extract_conv.py --prod resolution_trackres_mugun_ul16_260909_conv_<v> --out data/conv_<v>.npz`
   (keys `run/lumi/event` are kept so the variants pair track-by-track — a
   PAIRED comparison is far more precise than two independent means, because
   the fluctuation is common).
2. `base` vs `resolution_trackres_mugun_ul16_260903x_m0` on the same 40 tasks:
   `z` must agree bit-for-bit. If it does not, the build moved and every
   comparison below is against a different baseline.
3. On each variant, the same four analyses as in PART 2 below:
   truth-referenced charge-even `<x>` per eta band (`s2`/`s3` recipe), the
   bulk/IN decomposition on `|seed->final dq/p|` at its 90th percentile
   (`s6_mixture.py`), the trim scan (`s1_trim.py`), and the `niter` census.
   Report a before/after table per variant.
4. **DECISION RULE.** If the bulk `-6.3e-3` and the `+21e-3` IN component
   shrink under `tight` or `damp`, the mechanism is incomplete convergence and
   the fix is the convergence criterion — then quote the cost per track
   (measure `<niter>` and the wall time per task against `base`). If they
   PERSIST at a tighter tolerance and along a damped path, they are properties
   of the CONVERGED estimator (a second-order bias) and the coordinator wants
   those numbers for the derivation.

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
* figures: `~/public_html/cvh/260908_hitclassbias/` (13 panels + index.php).
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
