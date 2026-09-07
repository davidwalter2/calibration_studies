# fullscale — STATE  (checkpoint 2026-09-07 07:30)

**Read this file top to bottom before touching anything.** The running log of
how each number was obtained is in `STATE_log.md` next to this file; this file
is what someone with no context needs.

Goal: the statistical precision and the MC closure of `m_Z` and `Gamma_Z` from
the unbinned CVH mass likelihood, alone (phase 1) and jointly with the J/psi
channel that fixes the momentum scale (phase 2), then with the full design
(phase 3).

Figures: `~/public_html/cvh/260906_fullscale/` (the phase-1 inputs, the two
correction forms and the clip scan) and `~/public_html/cvh/260907_fullscale/`
(the full-scale post-fit spectrum). Both have `index.php`.

---

## THE TABLE  (what to take to the group leader)

Z -> mumu MC, 3 682 662 CVH-refit candidates (all 380 tasks of
`dymc_8p5M_260906_v2`), unbinned per-candidate mass likelihood, resolution and
alignment FIXED at the MC truth, K(m) floated, both MASSCFTERM_SPEC corrections
in the fluctuation form. Errors are the x1.109 sandwich.

| | value | limited by |
|---|---:|---|
| **`sigma(m_Z)`** | **2.27 MeV** | statistics + the floated K(m) (x1.5 over K(m) fixed) |
| **`sigma(Gamma_Z)`** | **4.16 MeV** | statistics + the floated K(m) (x1.4) |
| **`m_Z` closure** | **-11.06 +- 2.27 MeV** | **the detector-level resolution model** (sec. 0b) |
| **`Gamma_Z` closure** | **-5.26 +- 4.16 MeV** | closes at 1.3 sigma — **but see the K(m) caveat below: NOT yet a 4 MeV result** |
| the two resolution corrections | +8.00 MeV on `m_Z`, additive to 0.2 MeV | done; inside the spec's own -5...-14 MeV prediction |
| the momentum scale | phase 2 (running) | the J/psi transfers it through the field modes, not a free alpha |
| the material amounts | phase 3 (blocked on one reader) | they are what would float the resolution model |
| the post-fit spectrum | `chi2/ndof = 4.75` over 240 bins | genuine few-% shape mismodelling; unchanged when the model subsample is grown 6.7x |

**What limits it, in order.** (1) The `m_Z` closure: -11 MeV at 4.9 sigma, and
it is NOT the corrections, NOT the Born lineshape, NOT K(m) and NOT the
coefficient bound — all measured in sec. 0b. What is left is the per-candidate
resolution CF, whose four scale knobs are held at the MC truth in phases 1-2 and
which phase 3 replaces with the parmtype-15 material amounts.
(2) The K(m) shape costs a factor 1.5 on `sigma(m_Z)` and cannot be dropped
(without it `Gamma_Z` carries +175 MeV of LO->MiNNLO K-factor). Five terms are
saturated for `m_Z` (5 -> 7 moves it -1.3 MeV) but **NOT for `Gamma_Z`**, which
5 -> 7 moves by +42 MeV at 300 k against 0.4 MeV at generator level. Until that
ladder is repeated at full statistics the `Gamma_Z` closure is a
tens-of-MeV statement, not a 4 MeV one.
(3) The MiNNLO weights cost a flat x1.109 (`N_eff/N = 0.8130`).
(4) Not limiting, but recorded: the `_norm_z` class-sigma approximation
(0.016 MeV), the coefficient bound (0.049 MeV), and the 0.19 % one-sided
selection-variable mismatch.

---

## 0. THE STATISTICAL NUMBER, AND THE PHYSICS PROBLEM — NOW SOLVED

**Statistical precision, MEASURED at the fitted minimum** on all 3 682 662
candidates, resolution and alignment fixed at the MC truth, K(m) floated
(sandwich errors, `results/fit_f380fl_base.json`):

| | sandwich | inverse Hessian |
|---|---:|---:|
| **`sigma(m_Z)`** | **2.27 MeV** | 2.07 MeV |
| **`sigma(Gamma_Z)`** | **4.16 MeV** | 3.78 MeV |

with K(m) FIXED the Asimov projection is 1.51 / 2.91 MeV, so floating the
LO->MiNNLO shape costs x1.5 / x1.4 — the price of not having to trust an LO
parton luminosity. `Gamma_Z` at 4.2 MeV is already at the interesting scale
(the Z-width sensitivity note targets ~2 MeV). **The MC closure is in sec. 0b.**

### The problem (phase-1 finding, 2026-09-06 19:30)

Both corrections of `resolution/oddmoment/MASSCFTERM_SPEC.md` are expansions in
the RESOLUTION fluctuation, and what the implementation fed them was `delta_i`,
the deviation from the reference mass. At the J/psi those coincide; at the Z
they do NOT — the window is +-27 sigma and the deviation out there is FSR and
the Breit-Wigner tail. Fed the full `delta`, the exact Jensen map moved the
residual by a **median 57.7 MeV and up to 10.7 GeV** against the **20.6 MeV**
mean shift it exists to apply, and the 300 k fit ran away to
`Gamma_Z = -421 MeV`. `corr_clip` (rabbit `b64f49f`) bounded that by SATURATING
both corrections outside a few sigma. It is a stopgap, not a treatment.

### The treatment (2026-09-06 20:45) — `corr_form="fluctuation"`

rabbit `cd6c165` + `a3958f9`. Both corrections are ONE deterministic
per-candidate map of the fluctuation, applied **inside the convolution**:

```
m_i = m_true + u_i(x),   u_i(x) = sigma_i x + c_i x^2 + d_i
c_i = -a_i sigma_i + sigma_i^2/m_i   (= -vgf_i sigma_i^2/m_i : they CANCEL)
d_i = m_i s_i^2/2
L_i(theta) = Int K_theta(m_i - u_i(x)) p_i(x) dx ,  p_i(x) = p_x(x)(1 - a_i x)
```

`x` is the standardized fluctuation whose CF is the exported `e^{S_i(tau)}`.
The `(1 - a_i x)` measure is the Jacobian of recovering the UNCONDITIONAL width
`sigma_bar_i` from the exported `sigma_i = sigma_bar_i(1 + a_i x)` — i.e. of
profiling `sigma_bar_i` out against the observed `sigma_i`. **It is not
optional**: without it the score at the truth is `+a_i/sigma_bar_i`, a bias of
order `a_i sigma_i` = 27 MeV at the Z, and the form would not reduce to the
residual form at a delta kernel. (The task sketch omitted it; the J/psi gate is
what settles it.)

To first order in `a_i` and `c_i` this is ONE multiplicative factor on the
resolution CF, on the term's own `tau` grid, plus a shift `d_i` of the residual:

```
Phi_i(tau)/phi_i(tau) = 1 + i a_i (S'(tau) - S'(0))
                          - i (c_i/sigma_i) tau (S''(tau) + S'(tau)^2)
```

from `E[x e^{i tau x}] = -i phi'` and `E[x^2 e^{i tau x}] = -phi''`,
`phi'' = (S'' + S'^2) phi`. `- S'(0)` normalises `Phi_i(0) = 1`.
`c_i/sigma_i = -a_i + sigma_i/m_i`, no division at evaluation time.
`S'`, `S''` are fixed cubic-spline differentiation matrices from the stored
`tau` grid onto the integration grid (the Gaussian family analytically).

**No clip, no log-Jacobian, no dependence on `delta_i`, `sigma_i` stays
constant so the cheap static path returns.**

**The one bound the form needs — `corr_coeff_max`, on the COEFFICIENT.** The
quadratic term enters as `g_i x^2` against the linear `x`, so
`g_i = c_i/sigma_i` IS the expansion parameter, and where `|g_i x| ~ 1` the
first-order truncation stops being a correction: the modelled density can go
negative in a large-`sigma_m/m` candidate's tail, and one negative density takes
the NLL to `-inf`. Scanned on the 300 k Z card at five parameter points
(reference, `m_Z` +-30 MeV, `Gamma_Z` +-60 MeV):

| cap | both | noares | nojensen | noboth |
|---|---|---|---|---|
| none | \|g\|max 0.097, **0** bad | 0.100, **0** | 0.197, **19** | **0** |
| 0.10 | capped 0, **0** | 0, **0** | 1997, **2-3** | **0** |
| **0.08** | capped 364, **0** | 509, **0** | 3242, **0** | **0** |
| 0.06 | capped 1088, **0** | 1426, **0** | 6261, **0** | **0** |

**0.08 is the default.** The 364 of 300 000 (0.12 %) it bounds in the physics
configuration all have `sigma_m/m > 0.066`, i.e. a factor 40 less weight in the
mass than a typical candidate — and **the bound is measured to cost 0.049 MeV
on `m_Z` and 0.022 MeV on `Gamma_Z`** (the same 300 k fit with and without it:
-20.616 +- 8.094 against -20.665 +- 8.093), i.e. 2 % of the statistical error
even at the full 3.68 M. This bounds a per-candidate CONSTANT computed
from observables — theta-independent, so it cannot deform the likelihood's
dependence on the parameters. That is precisely what `corr_clip`, which bounded
the ARGUMENT, could not say.

What is exact / what is approximated: exact for the first moment,
`E[Delta_i] = Var(x)(c_i - a_i sigma_i) + d_i` (measured to 1e-10 in the unit
test); neglected `O(a^2, ac, c^2) ~ 1e-4` of a correction that is itself ~1e-2
of the width; the second moment loses `2(c_i/sigma_i)^2 ~ 2e-4` relative
(<0.1 MeV on `Gamma_Z`). The **O(c^2) term is NOT needed** — it contributes
nothing to the mean, which is the whole content of both corrections.

### Gates

| gate | requirement | measured |
|---|---|---|
| unit 1 | no correction -> bit-identical in either form | exact |
| unit 2 | density normalises to 1, mean = closed form | 1.0000000000, exact to 1e-10 in all three arms |
| unit 3 | at a DELTA kernel the two forms agree | correction +0.11016 (residual) vs +0.11089 e-3 (fluctuation), **0.00073 e-3** apart |
| unit 4 | bounded at the Z | residual form moves the residual by median 1979 MeV / max 9.1 GeV; fluctuation `d_i` median **7.3 MeV**, max 15.7 MeV; `|w| e^{Re S}` < 0.03 |
| unit 5 | gradient vs FD | 3e-10 |
| unit 6 | `c_i = -vgf sigma^2/m` | 7e-18 |
| unit 7 | `set_corrections` == purpose-built terms | bit-identical, all four ladder points |
| **gun** | the REAL J/psi gun, 299 422 candidates | **PASS**, table below |

**GATE 1, the real J/psi gun** (`gate_fluct_gun.py`, 299 422 candidates, five
rabbit terms sharing them, so the shift carries no statistical error):

| | alpha [e-3] | shift [e-3] | MASSCFTERM_SPEC |
|---|---:|---:|---:|
| uncorrected | -0.00825 | — | -0.0047 |
| a_res only, residual | +0.13705 | **+0.14530** | +0.1457 |
| a_res only, fluctuation | +0.13792 | **+0.14617** | +0.1457 |
| both, residual | +0.05082 | **+0.05907** | +0.0559 |
| both, fluctuation | +0.04858 | **+0.05683** | +0.0559 |

`|fluctuation - residual|` = **0.00087 e-3** (a_res alone) and **0.00224 e-3**
(both), against the 0.01 e-3 the gate asks; both forms sit within 0.003 e-3 of
the numbers the offline numpy implementation measured.

`tests/test_fluctuation.py` in the rabbit worktree; the seven pre-existing
suites all still pass (the refactor is bit-identical at `upsample == 1`).

---

## 0b. PHASE 1 — THE FINAL NUMBER

**Z alone, all 380 DY tasks, 3 682 662 candidates after cuts, resolution and
alignment fixed at the MC truth, K(m) floated (5 Legendre terms), both
corrections in the FLUCTUATION form, `corr_coeff_max = 0.08`, x1.109 sandwich
errors.** `fitted - generator` with the generator at
`m_Z = 91.153509740726733`, `Gamma_Z = 2.4932018986110700`:

| | fitted - generator | stat | pull |
|---|---:|---:|---:|
| **`m_Z`** | **-11.06 MeV** | **+- 2.27** | -4.9 |
| **`Gamma_Z`** | **-5.26 MeV** | **+- 4.16** | -1.3 |

(38 iterations, 16 940 s on a preemptable H200; sandwich/inverse-Hessian error
ratio 1.096-1.099 against `sqrt(N/N_eff) = 1.109`. `results/fit_f380fl_base.json`.)

**`Gamma_Z` closes at 1.3 sigma. `m_Z` does not: -11.1 +- 2.3 MeV.**

### The 300 k ladder that gets there — and the clip that does not (GATE 2)

Both ladders are the same model with one thing changed per row, resolution and
alignment at the MC truth, K(m) floated. They are NOT the same 300 000
candidates (`z_n300k.hdf5` subsamples the 373-task cache, `z_n300k_fl.hdf5` the
380-task one), so cross-form comparisons carry ~8 MeV of scatter; within a
ladder every row is the same candidates.

**The residual form and its clip:**

| variant | `m_Z` | `Gamma_Z` |
|---|---:|---:|
| unclipped, both corrections | **-35.51 +- 6.81** | **-421.03 +- 12.33** |
| unclipped, `a_res` off | -53.98 +- 6.88 | -427.47 +- 12.33 |
| unclipped, Jensen off | -14.42 +- 8.55 | -1.85 +- 14.89 |
| unclipped, neither | -26.08 +- 8.00 | -19.70 +- 14.44 |
| Jensen as a mean shift | -32.76 +- 9.60 | -0.01 +- 15.51 |
| **`corr_clip = 3`** | **+40.07 +- 8.77** | +4.11 +- 14.76 |
| **`corr_clip = 5`** | **+94.32 +- 7.92** | -12.37 +- 14.68 |
| **`corr_clip = 10`** | **+74.32 +- 7.30** | -49.00 +- 14.49 |

`m_Z` swings over **130 MeV** and `Gamma_Z` over **425 MeV** across a knob with
no physics in it, against an 8 MeV statistical error on the same candidates.
**The clipped answer is not stable in the clip** — the question STATE said would
decide the matter. The unclipped fit also buys an NLL **27 000 units** lower
than every well-behaved variant with K(m) coefficients of order one: it is
fitting the shape to a deformed resolution model, not measuring a mass.

**The fluctuation form:**

| variant | `m_Z` | `Gamma_Z` |
|---|---:|---:|
| **both corrections (the model)** | **-20.67 +- 8.09** | **+18.66 +- 14.72** |
| the same with `corr_coeff_max = 0.08` | -20.62 +- 8.09 | +18.64 +- 14.72 |
| `a_res` off (Jensen only) | -35.92 +- 8.08 | +17.14 +- 14.61 |
| Jensen off (`a_res` only) | -13.56 +- 8.13 | +22.29 +- 14.77 |
| neither | -28.66 +- 8.10 | +17.66 +- 14.67 |
| K(m) FIXED, both | -27.44 +- 5.86 | +175.25 +- 11.99 |
| K(m) with **7** terms | -21.94 +- 7.90 | +60.44 +- 16.11 |
| window **80-100** instead of 60-120 | -30.12 +- 25.01 | -87.68 +- 48.37 |

Finite, clip-free, every variant converging in 10-38 iterations. The two
corrections do what the census predicts and **are additive to 0.2 MeV**:
`a_res` alone moves `m_Z` by **+15.10 MeV** (its per-candidate mean shift is
-32.5 MeV), the Jensen map alone by **-7.26 MeV** (mean shift +20.6 MeV), and
together **+8.00 MeV** against the sum +7.84. Neither touches `Gamma_Z`, as it
must be: both are location effects.

### What the -11 MeV is NOT

* **Not the corrections.** Without them the fit is -28.7 MeV at 300 k and they
  move it +8.0 toward zero. The spec's own prediction for the NET defect at Z
  momenta is -5...-14 MeV (the two terms partially cancel); +8.0 MeV is inside it.
* **Not the Born lineshape or K(m) — for `m_Z`.** At GENERATOR level the same
  provider with 5 Legendre terms closes to **-0.45 +- 0.50 MeV** pre-FSR and
  **+0.15 +- 0.56** post-FSR-folded (`zchannel/README.md`). Going 5 -> 7 terms at
  detector level moves `m_Z` by **-1.3 MeV**, inside its own error: for `m_Z` the
  basis is saturated.

  **But NOT for `Gamma_Z`.** The same 5 -> 7 moves `Gamma_Z` by **+42 MeV**
  (+18.66 +- 14.72 -> +60.44 +- 16.11 on the SAME 300 k candidates), where at
  generator level 5 -> 6 moved 0.4 MeV. **The `Gamma_Z` closure of -5.3 +- 4.2 MeV
  is therefore shape-basis dependent at the tens-of-MeV level and must not be
  quoted as a 4 MeV result until the 5 -> 6 -> 7 ladder is repeated at full
  statistics.** That is the single most important loose end of phase 1.
* **Not obviously the tails.** Narrowing the window to 80-100 GeV (88.0 % of the
  candidates) gives -30.1 +- 25.0: the error triples because the shoulders carry
  the information, and the central value does not move outside it. Inconclusive
  rather than exculpatory.
* **Not the coefficient bound.** 0.049 MeV, measured.

### What it might be

The post-fit spectrum has **real structure, and it is not a plotting
artifact**. At full scale (`260907_fullscale/12_..._nsub6k`,
`13_..._nsub40k.png`) `chi2/ndof` over 240 bins is **4.53 with a 6 000-candidate
model subsample and 4.75 with 40 000** — 6.7x the model statistics and the chi2
goes UP, not down, so what it measures is the model and not the subsample. The
shape is data/model ~ +5-10 % at 60-66 GeV, flat to 80, a shallow dip on the
82-90 GeV shoulder and ~+3 % on the 95-105 GeV one. That is an S-shape a smooth K(m) cannot
absorb and it is exactly the shape that biases a mass. Since generator level
closes and the shape basis is saturated, what is left between them is the
**detector-level resolution model**: the per-candidate CF (`k_hit`, `k_ms`,
`k_ioni`, `k_rad` are FIXED at 1 here), the truncation `Z` evaluated at the
stored class sigma, and the 0.19 % one-sided selection-variable mismatch. The
first is phase 3's job -- it replaces the four knobs with the parmtype-15
material amounts and floats them. **The natural next diagnostic is a
`sigma_m/m` quantile split**: the resolution-model hypothesis predicts the bias
grows with `sigma_rel`, and the spec's own J/psi differential test (J4) is the
template.

---

## 1. WHAT EXISTS

### The merged rabbit — this is not on any remote
`/work/submit/david_w/ZMass/rabbit-material`, branch **`material-resolution`**,
HEAD **`a3958f9`** (+ `set_corrections`, uncommitted at the time of writing —
check `git log`). It is `material-resolution` (self-consistent resolution,
`MaterialCFTerm`, `ExternalParams`) with **`z-lineshape-kernel` merged in**
(`ZGammaLineshape`, `norm_window`, in-graph `upsample`):

| commit | what |
|---|---|
| `4f762c0` | the merge itself. All five suites green; additivity gate 5.3e-14 |
| `58a3f82` | the floated smooth **K(m)** in the provider (`shape=`, `shape_window=`) |
| `5658252` | the **exact Jensen map** in `MassCFTerm` (`jensen_s2=`, `jensen_mode=`) |
| `b64f49f` | **`corr_clip`** — the residual form's stopgap, now a diagnostic only |
| `cd6c165` | **`corr_form="fluctuation"`** — the treatment (sec. 0) |
| `a3958f9` | `tests/test_fluctuation.py`, 7 checks |

Tests (run as SCRIPTS, not pytest — two of them use positional args that pytest
reads as fixtures, which is pre-existing on BASE and both parents):
```bash
cd calibration_studies/fullscale
for T in test_unbinned_mass test_material_cf test_global_term \
         test_unbinned_norm test_zgamma_kernel test_jensen test_zgamma_shape \
         test_fluctuation; do
  THREADS=16 ./run_tf.sh python3 -u ../../rabbit-material/tests/$T.py; done
```

### Caches and cards (`fullscale/runs`, `fullscale/cards`; both gitignored)

| file | content | how it was made |
|---|---|---|
| `runs/zpairs_dyv2_full.npz` | **3 733 323, all 380 DY tasks** (4.62 GB) | `cf_inmaker.py pairs` + `append_pairs.py` |
| `runs/zpairs_dyv2_jac_full.npz` | **3 733 323, all 380 tasks, WITH the (n,92) mass Jacobian** | same + `--jac-parmtypes 14 15`; the 3 late tasks (`task_0000/0002/0003`, 29 168 candidates) appended 2026-09-06 20:38 |
| `runs/zpairs_dyv2.npz`, `_tail`, `_jac`, `_jac_tail` | the pieces they were built from | |
| `runs/quad_dyv2.npz` | DY quadratic term, **380 tasks**, 3 751 687 candidates | `globalfit/extract.py --no-mass --parmtypes 14 15 --max-chi2-ndof 3 --max-hess 1e8 --max-grad 1e6 --max-dEref-p 0.01` |
| `runs/quad_jpsiv1.npz` | J/psi **v1** quadratic, 15 965 797 candidates (CROSS-CHECK ONLY) | same |
| `runs/xcheck_v1.npz`, `runs/xcheck_v2.npz` | the same 120 J/psi tasks through v1 and v2 | same, `--ntasks 120` |
| `cards/z_full380.hdf5` | 3 682 662 candidates, **`corr_form="residual"`, `corr_clip=5`** — the STOPGAP card | `make_card.py --pairs runs/zpairs_dyv2_full.npz --shape 5 --fsr ... --acc ...` |
| `cards/z_full.hdf5` | the 373-task twin, same (stopgap) model | same |
| `cards/z_n300k.hdf5` | a 300 000 subsample, same (stopgap) model | same, `--maxn 300000` |
| `../zchannel/data/kern_loose_band3.3e-4.npz` | the FSR kernel: 11 bands, sigma_cap 3.3e-4, 14 315 atoms, 13.03 M gen events, pT>5, \|eta\|<2.4 | `fit_gen.py kernel` |
| `../zchannel/data/acc_loose_d8.json` | the acceptance, Bernstein 8, pT>5 | already existed |

**`make_card.py` now defaults to `--corr-form fluctuation --corr-clip 0`.**
The three cards above predate that and are the residual-form ones; the phase-1
FINAL cards are `cards/z_full380_fl.hdf5` and `cards/z_n300k_fl.hdf5`.

### Scripts (all committed)

| file | what |
|---|---|
| `run_tf.sh` | runs a python script in the rabbit TF singularity image with the MERGED worktree first on PYTHONPATH. `RABBIT=`, `THREADS=`, `--ceph` |
| `make_card.py` | the Z card. `--corr-form {fluctuation,residual}`. REFUSES to build if the rabbit it imports lacks a piece of the model |
| `chunkfit.py` | value/grad/Hessian accumulated over candidate chunks + sandwich meat by forward-mode autodiff |
| `fit.py` | the driver. `--fix`, `--ares/--jensen/--corr-clip` overrides (via `term.set_corrections`), `--start-from`, truth comparison, projections |
| `gate_fluct_gun.py` | **GATE 1**: the fluctuation vs the residual form on the REAL J/psi gun candidates |
| `validate_hessian.py`, `check_normz_sigma.py`, `append_pairs.py`, `report.py` | as before |
| `plot_inputs.py`, `plot_postfit.py` | the figures |
| `run_extract.sh`, `run_phase1.sh`, `run_phase2.sh`, `run_full380.sh` | the launchers |
| `make_joint_card.py` | phase 2, SCAFFOLD (`sum_quadratic` finished) |
| `patches/{zgamma_shape,unbinned_jensen}.py` | idempotent, already applied |
| `../engaging/fullscale_{gpu,variants,clipscan}.sbatch` | the GPU jobs |
| `stage_eng.sh` | push the merged rabbit + scripts + a card to Engaging |

---

## 2. WHAT IS RUNNING  (checkpoint 2026-09-07 07:00)

### On Engaging (ORCD) — `eng 'timeout 30 squeue -u david_w'`
The 8 h SSH master expires silently and then MIT wants a Kerberos password plus
a Duo approval; when `eng` starts printing instructions instead of output, a
human has to run **`!eng-master`** in an interactive terminal. It was down
06:45-07:50 today.

| job | what | state at 07:55 |
|---|---|---|
| **22167631** | `zvarfl` on `z_full380_fl.hdf5`, `mit_normal_gpu` | PENDING since 21:00 yesterday, `scontrol` estimate 14:40. Runs the whole phase-1 ladder from the start. |
| **22199038** | the same on `mit_preemptable` | PENDING (submitted 07:53). Whichever starts first wins; cancel the other. |
| **22199037** | **`zjoint`** — `fit_joint.py` on the full `cards/joint_v2.hdf5`, `mit_preemptable`, `--hess-mode pfor` | PENDING (07:53). **This is phase 2 at full size.** |
| ~~22171547~~ | the preemptable ladder | ran 01:28-07:1x, produced `base` (collected) and was cut off during `noares` |

Notes: **H200**; **do not pass `--chunk`** on a card with a sparse `D`; the
per-user GPU limit is one job per partition, so submitting to both
`mit_normal_gpu` and `mit_preemptable` is how anything gets scheduled;
`chunk 262144` OOMs.

### On submit82 — detached, `setsid`, logs in `fullscale/logs/`

| what | output | note |
|---|---|---|
| `fit_f380fl_noboth` | `results/fit_f380fl_noboth.json` | the full-scale "neither correction" row, on CPU with 64 threads, started 22:22. The CPU `base` twin was killed once the GPU produced it. |
| `joint300k.sh` | `cards/joint_v2_n300k.hdf5` -> `results/fit_joint_v2_n300k.json` | **PHASE 2 on CPU**, 300 k + 300 k, 99 free parameters, 38 chunks of 16384. Measured at the reference point: value+grad **29.5 s**, pfor Hessian **1841.9 s**, peak RSS **270 GB**. `trust-exact` needs one Hessian per iteration, so at the ~38 iterations the Z-alone fit took this is 10-19 h. |
| `srelsplit.sh` | `cards/z_srel_{lo,mid,hi}.hdf5` -> `results/fit_srel_*.json` | **the differential test**: `sigma_m/m` tertiles (`< 0.0110`, `0.0110-0.0140`, `> 0.0140`), 400 k each, sequential. Both corrections and any error in the per-candidate CF scale as `sigma_rel^2`, so a bias that IS the resolution model must GROW across the slices and one that is the lineshape must not. The Z analogue of MASSCFTERM_SPEC's gate J4. |
| `shapeladder.sh` | `cards/z_full380_fl_s{6,7}.hdf5`, staged | the K(m) ladder at FULL statistics, which is the one loose end of phase 1 (`Gamma_Z` moved +42 MeV under 5 -> 7 at 300 k). The cards build on submit and the fits run from `engaging/shape_ladder.sbatch`. |
| `joint100k.sh` | `cards/joint_v2_n100k.hdf5` -> `results/fit_joint_v2_n100k.json` | the same at 100 k + 100 k and `--chunk 8192`, i.e. 13 chunks and half the pfor tape. ~3-4 h, so it is the one that gives a phase-2 NUMBER today; the 300 k twin is the better one if it finishes. **pfor's peak is set by chunk x nparams, NOT by the number of candidates**, which is why the smaller card was written at a smaller chunk. |

### PHASE 2 — what exists

* `cards/joint_v2.hdf5` (10.7 GB), REBUILT 06:38 against `quad_jpsiv2_x16.npz`:
  a J/psi `MassCFTerm` (3 000 000 candidates, 96 parameters, delta kernel at the
  PDG mass, `scale_param=None`), a Z `MassCFTerm` (3 682 662, 103), the
  `hitchi2` external quadratic (92 parameters, from **20 506 733** candidates)
  and the `global_params` bundle. Staged to Engaging (the pre-x16 version;
  **re-stage it**).
* `fit_joint.py` — the driver `fit.py` could not be. Three gates, all measured
  on a 60 k + 60 k smoke card:
  * **(a)** with the 92 globals fixed it reproduces `fit.py`'s objective
    **exactly** — 0 difference in NLL and 0 in the gradient;
  * **(b)** analytic gradient vs central finite differences at the FD noise
    floor, on `m_Z`, `Gamma_Z` and two of the 92;
  * **(c)** `pfor` and `hvp` Hessians agree to **3.7e-16** (and 1.7e-21 on the
    two-term card).
  It also flags the one thing nobody has: the mass and hit-chi2 scores of the
  SAME candidate are correlated and neither extraction stored that cross block,
  so the two sandwich meats are added as if independent. It says so in its own
  output rather than quoting a robust error that assumes it away.
* **The `--inject` translation closure, re-measured in the fluctuation form:**
  `bfield_mode0` +0.0023 % of the injection against **+12.14 %** in the residual
  form (5300x better); modes 1 and 2 at 1e-11, four orders better. What did not
  translate was the residual form's Jensen map, whose argument `r = delta/m`
  carries the observed mass. The 2.3e-7 that remains is the second-order
  dependence of `a_i`, `c_i`, `d_i` on the shifted masses — 1.6e-3 of
  `bfield_mode0`'s own statistical error, and not the minimiser (the Newton step
  still implied by the residual gradient is 1.8e-12).
* The `hitchi2` Hessian is **singular in 4 directions** and carries no
  information at all on `material_beampipe`, `material_thermal_screen`,
  `material_support_tube`, `material_pp1_cables`. In the joint fit the mass
  terms constrain them through `D`; alone they would have to be frozen or given
  a prior.

## 3. HOW TO REPRODUCE EACH STEP

```bash
FS=/work/submit/david_w/ZMass/calibration_studies/fullscale
Z=/work/submit/david_w/ZMass/calibration_studies/zchannel
RES=/work/submit/david_w/ZMass/calibration_studies/resolution
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate     # numpy/uproot side

# 1. pairs cache  (~25 min, ceph; submit82/50/51 have it)
python3 $RES/cf_inmaker.py pairs \
  --files /ceph/submit/data/user/d/david_w/ZMass/cvh/dymc_8p5M_260906_v2 \
  --ntasks 0 --cache $FS/runs/zpairs_dyv2_full.npz --mass-window 91.1876 60
# add --jac-parmtypes 14 15 for the phase-2 version (the (n,92) mass Jacobian)

# 2. quadratic term  (~2.5 min DY, ~16 min a J/psi production)
python3 $RES/globalfit/extract.py --files <production> --ntasks 0 \
  --parmtypes 14 15 --no-mass --max-chi2-ndof 3 --max-hess 1e8 --max-grad 1e6 \
  --max-dEref-p 0.01 -j 16 -o $FS/runs/quad_<tag>.npz
#   --no-mass is MANDATORY (see pitfall 1)

# 3. card  (~13 min, mostly npz decompression).  The DEFAULT is now the
#    fluctuation form; --corr-form residual --corr-clip 5 rebuilds the stopgap.
THREADS=32 $FS/run_tf.sh python3 -u $FS/make_card.py \
  --pairs $FS/runs/zpairs_dyv2_full.npz --shape 5 \
  --fsr $Z/data/kern_loose_band3.3e-4.npz --acc $Z/data/acc_loose_d8.json \
  -o $FS/cards/z_full380_fl.hdf5

# 4. fit  (locally ~45 min at 300 k; on an H200 ~20 min at 3.6 M)
THREADS=48 $FS/run_tf.sh python3 -u $FS/fit.py --card $FS/cards/z_full380_fl.hdf5 \
  --fix k_hit k_ms k_ioni k_rad --chunk 32768 \
  --label base -o $FS/results/fit_fl_base.json

# 5. the J/psi gun gate on the reformulation
THREADS=32 $FS/run_tf.sh python3 -u $FS/gate_fluct_gun.py \
  --pairs $RES/runs/cf_masspairs_jpsigun_ul16_260905d_m0.npz \
  --kernel $RES/runs/cf_masskernel_jpsigun_ul16_260905d_m0.npz \
  -o $FS/results/gate_fluct_gun.json

# 6. tables and figures
python3 $FS/report.py --results "$FS/results/fit_*.json"
python3 $FS/plot_inputs.py --pairs $FS/runs/zpairs_dyv2_full.npz
THREADS=8 $FS/run_tf.sh python3 -u $FS/plot_postfit.py \
  --card $FS/cards/z_full380_fl.hdf5 --fit $FS/results/fit_fl_base.json --nsub 4000
```

---

## 4. THE PLAN, AND THE DECISIONS ALREADY TAKEN

### Decisions
1. **The corrections are applied in the FLUCTUATION form** (sec. 0). `corr_clip`
   and `corr_form="residual"` survive only as the J/psi reference and as the
   "for the record" column of phase 1.
2. **Phases 2 and 3 run on J/psi v2**, `/ceph/.../cvh/jpsimc_20M_260906_v2/`
   (complete, 1642 tasks). Measured reason: on the SAME 120 tasks v2's
   parmtype-15 information is **120x the trace and 666x the median diagonal**
   of v1's, while parmtype 14 is bit-equal. v1 is the cross-check only.
3. **The DY leg is all 380 tasks**, both with and without the Jacobian.
4. **Cuts**: `m_obs` in [60,120] (the production selected on the PRE-REFIT
   `Jpsitrk_mass`; the mismatch is 0.191 % and one-sided), `chi2/ndof < 3`,
   `sigma_m/m < 0.10`. 3 682 662 of 3 733 323 survive (98.6 %).
5. **Weights**: `genweight` clipped at 100x the modal, rescaled to mean 1.
   4.80 % negative, `N_eff/N = 0.8130`, so every error x1.109 in the sandwich.
6. **K(m)**: 5 Legendre terms, floated, over the FIT window.
7. **The Hessian is chunk-accumulated.** The monolithic construction is
   115.6 GB at 300 k and TWO parameters.
8. **Resolution fixed at the MC truth** in phases 1-2 (`--fix k_hit k_ms
   k_ioni k_rad`); phase 3 replaces the knobs with the parmtype-15 amounts.
9. Alignment is fixed at the MC truth throughout this test.
10. `f_ang` is folded into `jensen_s2` per candidate (median 8e-5 at the Z, so
    negligible there; on J/psi v2 it comes from `Jpsi_covrefmom`, truth-free).

### Phase 1 — DONE (sec. 0b). What is left of it
Only the full-scale VARIANT ladder: `base` landed (that is the number), and
`noares / nojensen / noboth / noshape` at 3.68 M are what job 22171547 (running
blind) and 22167631 (starting ~14:40) produce. The CPU `noboth` twin has been
in its minimisation for 9 h and needs ~5 more. The 300 k ladder in sec. 0b
already says what each variant is worth; the full-scale ladder only tightens
those differences from ~8 MeV to ~2.3 MeV.

### Phase 2 — the card and the driver EXIST; the fit is the long pole
`cards/joint_v2.hdf5` is built and `fit_joint.py` is gated (sec. 2). What
remains is to run it. On a GPU it is a few hours; on this CPU node it is 10-19 h
at 300 k + 300 k, which is why a 100 k twin is running alongside. When it lands,
report `m_Z`/`Gamma_Z` closure with the transferred scale, the pulls of the 92
globals, and `rho(m_Z, field)` / `rho(m_Z, material)` — `fit_joint.py` prints
all of them, quoting the largest |rho| per block and its RMS rather than 92
numbers.

Two things to say honestly with the result: the sandwich adds the mass-term and
hit-chi2 meats **as if independent** (the same candidate contributes to both and
no extraction stored the cross block), and the `hitchi2` Hessian is singular in
4 directions, so `material_beampipe`, `material_thermal_screen`,
`material_support_tube` and `material_pp1_cables` are constrained by the mass
terms alone.

### Phase 3 (full design) — and the ONE reader that blocks it

**`matres/extract_groups.py` CANNOT be used on these productions.** It rebuilds
the per-group CF offline from Geant4 step records and dies with
`KeyInFileError: not found: 'ioniurbanidx'` — both `jpsimc_20M_260906_v2` and
`dymc_8p5M_260906_v2` ran `exportStepRecords=False` (the same flag that blocks
`globalfit/extract.py`'s mass path, pitfall 1).

**But the per-group CF is there anyway**, written by the maker itself because
those productions ran `exportCfGroupExponents=True`. The tree carries

    cfmass_grp                    the material-group index axis
    cfmass_grp_ms
    cfmass_grp_ioni_re / _im
    cfmass_grp_rad_re  / _im
    cfmass_grp_closure            the maker's own closure check
    cfmass_hitcls, cfmass_hitv    the hit-class axis and its variance shares
    resinfcovgrp

next to the flat `cfmass_*` that `cf_inmaker.py pairs` already reads. **What is
missing is a reader**: `cf_inmaker.py` has no per-group mode (`grp` appears
nowhere in its pairs path), so nothing produces the CSR layout
(`grp_ptr`/`grp_id`/`Sg_*`, `hit_ptr`/`hit_cls`/`hit_v`/`vg_other`) that
`rabbit.unbinned.MaterialCFTerm` and `matres/make_material_card.py` consume.

That reader, its closure validation (per-group sum == the flat exponent;
`vg_other + sum_c hit_v == vgf`), and the two per-group caches are what phase 3
starts from. After them: a `--material` mode in `make_joint_card.py` that makes
BOTH mass terms `MaterialCFTerm`s sharing the parmtype-15 parameters with the
hit-chi2 term, hit-class parameters from the mass terms, corrections truth-free
from `Jpsi_covrefmom` (already the case on the J/psi leg: per-candidate `f_ang`,
median 6.2e-2), Asimov/toy pulls, and the group-leader table.

---

## 5. PITFALLS FOUND (each cost time; none is obvious)

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
   `nm=8192` (1.0 GB, `dm = 9.8 MeV`).
6. **`np.savez_compressed` of a multi-GB cache takes minutes**, during which
   the file exists and is a truncated zip. Wait for the reader's `wrote` line.
7. **git refuses to fetch into a checked-out branch** — `stage_eng.sh` detaches.
8. The `_norm_z` truncation is evaluated at the STORED class sigma and WITHOUT
   the fluctuation-form correction. Measured cost of the first: **0.016 MeV on
   `m_Z`** (`check_normz_sigma.py`). The second shifts the density by `d_i`
   (~7 MeV) against a window edge 30 GeV away and is a candidate CONSTANT, so
   it cannot bias `m_Z`.
9. `rho(shape4, shape5) = -0.964` at detector level — the Legendre basis is
   orthogonal over the Born window, not over the observed spectrum.
10. **Sampling a single candidate's density on a mass grid** by making the grid
    the `mobs` column gives every grid point its OWN `c_i`, `d_i` (they depend
    on `m_i = mobs_i + m_ref`). Over a +-13 GeV grid that is a +-15 % spread and
    it shows up as a 3e-5 normalisation deficit. Pin them for that test.
11. **`corr_form` is consumed at CONSTRUCTION** — the per-candidate `c_i`, `d_i`
    are baked in. Flipping `self_consistent_sigma` / `jensen_mode` by hand no
    longer switches the correction; call `term.set_corrections(...)`.
12. **`pfor`'s Hessian memory is set by `chunk x nparams`, NOT by the number of
    candidates.** 99 free parameters at `chunk 16384` peaks at **270 GB**;
    the same card written at `chunk 8192` peaks at 144 GB. The candidate count
    changes the TIME (one Hessian per chunk) and nothing else. A joint card
    therefore has to choose its chunk at WRITE time with the Hessian in mind —
    and it cannot be re-chunked afterwards, because its per-candidate sparse `D`
    is sliced per chunk when it is written.
13. **`trust-exact` needs the full Hessian at EVERY iteration.** The Z-alone fit
    took 38 of them. That is why a 6.68 M joint fit with 99 free parameters is a
    GPU job and not a CPU one: 1842 s per Hessian at 600 k on submit.
14. **The Engaging SSH master expires after 8 h, silently.** `eng` then prints
    instructions instead of output and every `rsync`/`sbatch` fails with
    `connection unexpectedly closed`. Only a human can fix it: `!eng-master`.
    Budget for it — it cost 65 minutes today.
15. **The K(m) Legendre basis is saturated for `m_Z` and NOT for `Gamma_Z`.**
    Do not carry "5 terms close it" from the generator-level study to the
    detector-level `Gamma_Z`.
