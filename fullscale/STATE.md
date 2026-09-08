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
| **`m_Z` closure** | **-11.06 +- 2.27 MeV**, **IDENTIFIED AND FIXED** (sec. 0c-0g) | **Punzi's variable-resolution problem**: the likelihood evaluates `p(m_i \| sigma_i)` with the SAME Born spectrum for every candidate, and `sigma_i` carries information about the true mass (`<m_gen>` runs 84.94 -> 91.28 GeV across `sigma` octiles, `rho = 0.168`). Both halves of the model close in isolation — the detector half at **+0.9 +- 2.1** (kernel-free) and the kernel half at **+0.76 +- 1.36** (detector-free, the selected candidates' own gen masses) — and the defect lives only in the combination. **Proved**: reweighting the dependence away gives **-0.12 +- 2.42** on the same candidates (`F_dc8`). **Fixed by construction**: convolve in `v(m) = Int dm/m^p` and condition on `k_i = sigma_i/m_i^p`, for which `rho(k, m_gen) = -0.011`. Phase 3 was never going to fix this. |
| **`Gamma_Z` closure** | **-5.26 +- 4.16 MeV** | closes at 1.3 sigma — **but see the K(m) caveat below: NOT yet a 4 MeV result** |
| the two resolution corrections | +8.00 MeV on `m_Z`, additive to 0.2 MeV | done; inside the spec's own -5...-14 MeV prediction |
| the momentum scale | phase 2 (running) | the J/psi transfers it through the field modes, not a free alpha |
| the material amounts | phase 3 (blocked on one reader) | they are what would float the resolution model |
| the post-fit spectrum | `chi2/ndof = 4.75` over 240 bins | genuine few-% shape mismodelling; unchanged when the model subsample is grown 6.7x |

**What limits it, in order.** (1) The `m_Z` closure: -11 MeV at 4.9 sigma.
SEVEN things are now excluded by direct measurement — the two corrections, the
Born lineshape, K(m), the coefficient bound, a loose SIM stepper, the
reconstruction's momentum scale (which gives masses +6.2 MeV HIGH, the opposite
sign, and is flat in `eta` to 3.7 MeV), **the resolution model** (the mass pull
is 0.996 and flat in `eta`; with the kernel removed entirely the detector half
closes at +0.9 +- 2.1 MeV) and, since 2026-09-07, **the KERNEL** (with the
detector removed entirely — the selected candidates' own gen masses against the
same lineshape (x) A (x) FSR (x) K chain — the kernel half closes at
+0.76 +- 1.36 MeV inclusively and at +0.6 / +4.2 / -2.7 per `eta` band, against
a detector-level barrel-endcap difference of 40 +- 12 MeV; sec. 0c).
**What is left is the COMBINATION**, and there is one concrete mechanism that
lives only there: the likelihood evaluates `p(m_i | sigma_i)` with the SAME
Born spectrum for every candidate, and on this sample the true mass and the
per-candidate resolution are strongly dependent (sec. 0c). **Phase 3 does not
address it**; nor does `zchannel`'s generator-level chain, which closes.
(1b) Real but separate: `k_ms` = 1.0298 +- 0.0042 in the kernel-free fit — the
multiple-scattering TAIL is ~3 % short. That IS the material and phase 3 is its
test; it moves `m_Z` by 0.5 MeV.
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

### THE DIRECT MEASUREMENT: the momentum scale is FLAT in eta; the FIT is not

`qopbias.py` measures the per-leg curvature bias against GEN,
`d kappa/kappa = pT_gen/pT_reco - 1` (so a POSITIVE value means the reco `pT` is
too LOW), split into the charge-EVEN part `A` (field/scale-like) and the
charge-ODD part `M` (misalignment-like), trimmed 5 % per tail because the
residual has Landau-like tails. Three samples, `pT` matched where it matters:

**Charge-EVEN `A(eta)` [1e-4]**

| `\|eta\|` | DY v2 Z legs, `pT` 25-60 (OFFICIAL UL16 SIM, 10_6/106X) | J/psi v2 legs (private tight, 10_6/106X) | mu gun 20-60, `pT` 25-60 (private tight, **15_0/150X**) |
|---|---:|---:|---:|
| 0.0-0.4 | -0.45 +- 0.12 | -1.57 +- 0.10 | +1.22 +- 0.39 |
| 0.4-0.8 | -0.53 +- 0.14 | -1.70 +- 0.12 | +2.62 +- 0.46 |
| 0.8-1.2 | -0.80 +- 0.20 | -1.76 +- 0.16 | +4.15 +- 0.62 |
| 1.2-1.6 | -0.82 +- 0.26 | -2.34 +- 0.15 | +6.66 +- 0.76 |
| 1.6-2.0 | -0.41 +- 0.30 | -2.00 +- 0.17 | +9.00 +- 0.82 |
| 2.0-2.4 | -0.72 +- 0.59 | -2.02 +- 0.27 | +17.92 +- 1.34 |
| **inclusive** | **-0.682 +- 0.090** | **-1.838 +- 0.068** | **+6.266 +- 0.284** |
| **`eta` SPREAD** | **0.41** | **0.77** | **16.70** |

**Residual RMS(`d kappa/kappa`) [1e-3]** — the width half

| `\|eta\|` | DY (`<pT>` 40) | J/psi (`<pT>` 5.3) | gun (`<pT>` 42) |
|---|---:|---:|---:|
| 0.0-0.4 | 8.12 | 5.87 | 7.88 |
| 0.4-0.8 | 9.41 | 7.38 | 9.28 |
| 0.8-1.2 | 12.76 | 10.29 | 12.59 |
| 1.2-1.6 | 15.51 | 12.00 | 15.45 |
| 1.6-2.0 | 16.62 | 12.72 | 16.54 |
| 2.0-2.4 | 28.88 | 18.28 | 26.88 |
| **inclusive** | **13.49** | 11.41 | **14.11** |

#### 1. The loose-SIM-stepper hypothesis is REJECTED for this sample

* **Premise**: this repo's own `resolution/simprod/step1_gensim.py` documents
  that **the official UL16 SIM already sets `DeltaOneStep = 1e-5` /
  `DeltaIntersection = 1e-6` globally**, and that CMSSW_10_6 has no
  region-specific variants so those globals ARE what the tracker used. The
  private guns set the same. The sample that genuinely ran 100x loose (1e-4) is
  the **B->J/psi+X MC produced in 10_6_20**, not DY. (That is an assertion in
  the driver, established earlier in this project; I did not re-verify the
  official campaign's `cmsDriver`.)
* **Bias test**: at matched `pT` the official DY's `eta` spread is **0.41e-4** —
  flat — against **16.70e-4** for the private tight-stepper gun. A loose stepper
  would make DY show MORE `eta` structure. It shows 40x LESS.
* **Width test**: DY and the gun agree bin by bin to a few per cent at matched
  `pT` (8.12/7.88, 9.41/9.28, 12.76/12.59, 15.51/15.45, 16.62/16.54,
  28.88/26.88) and DY is **4 % NARROWER** inclusively (13.49 against 14.11), not
  the ~12 % WIDER a loose stepper would give.

**A private tight-stepper Z sample is therefore not needed**, and no cost
estimate is offered for one.

*Caveat*: the gun was simulated AND reconstructed in CMSSW_15_0 with GT
`150X_mcRun2_asymptotic_v1` while DY and J/psi v2 both used 106X, so its
ABSOLUTE bias (+6.3e-4, spread 16.7e-4) is a statement about the 150X alignment
payload, not about the stepper. It is a valid control for the WIDTH, which is
material- and hit-dominated. The comparable pair for the bias is DY vs J/psi v2,
both 106X: spreads 0.41 and 0.77e-4, both small, and the J/psi's is the LARGER
despite 7.5x lower `pT` — the opposite of a `pT`-proportional sagitta.

#### 2. And it overturns the "eta = field/alignment" reading

The TRUE per-leg momentum scale is **flat in `eta` to 0.41e-4**, which is
**3.7 MeV** on the mass. The FITTED `m_Z` varies by **40-57 MeV** across the
same `eta` bands. The `eta` dependence of the fitted `m_Z` is therefore an order
of magnitude larger than the `eta` dependence of the momenta, so it is **NOT a
field or alignment effect in the data** — it is the LIKELIHOOD's `eta`-dependent
response. My earlier "an eta-dependent momentum bias is the field/alignment
signature" is **retracted**; the eta dependence is real but it lives in the fit.

That reconnects to the width story and closes it: the resolution RMS varies by
**3.5x across `eta`** (8.1 -> 28.9e-3), the model is **2.8 +- 0.2 % too narrow**
and `k_hit` floats to **1.133 +- 0.039**, so a width mis-scaling is necessarily
`eta`-dependent and produces exactly an `eta`-dependent pull on `m_Z`. **Phase 3
— floating the parmtype-15 material amounts and the hit-class parameters — is
the direct fix for it**, and the hit-class parameters are the test of the
hit-side half.

#### 3. The absolute scale also does not explain `m_Z`

DY inclusive `A = -0.682 +- 0.090e-4` means the reconstructed leg momenta are
0.68e-4 too HIGH, i.e. the reconstructed masses are **+6.2 MeV** high. The
fitted `m_Z` closure is **-11.06 +- 2.27 MeV** — the OPPOSITE sign. The two
differ by ~17 MeV, and that difference is the likelihood's own modelling, not
the reconstruction's momentum scale.

### THE PURE-DETECTOR TEST: the detector half CLOSES
   (the conclusion drawn here — "so it is the KERNEL" — is RETRACTED in sec. 0c:
   the kernel half closes too)

`make_card.py --residual-mode` models **`m_reco - m_gen` directly against the
per-candidate resolution CF**: the observable is the residual, the physics
kernel is a DELTA, and the FSR fold, the acceptance and K(m) are all dropped.
Nothing of the mass model survives, so what is measured is the detector half
alone. 297 557 candidates (`|m_reco - m_gen| < 10 GeV`, normalised),
both corrections on:

| free | fitted | in MeV |
|---|---:|---:|
| `alpha` only | **+0.00966 +- 0.02324** e-3 | **+0.88 +- 2.12** |
| `alpha` + `k_hit` | `k_hit` = **1.0647 +- 0.0291** | 2.2 sigma |
| `alpha` + `k_ms` | `k_ms` = **1.0298 +- 0.0042** | **7.0 sigma** |

**With no kernel at all the detector half closes at +0.9 +- 2.1 MeV.** The
`m_Z` closure of the full fit is **-11.06 +- 2.27 MeV**. The two are 4 sigma
apart and the residual-mode number is compatible with zero, so **the -11 MeV
lives on the KERNEL side** — the Z lineshape, the FSR fold, the acceptance and
K(m) — and NOT in the resolution model, the corrections, or the reconstruction.

### THE PULL WIDTH: the resolution CORE is right to 0.4 %, flat in `eta`

The mass pull `(m_reco - m_gen)/sigma_m`, robust width IQR/1.349, on the full
3 682 662-candidate selection — and the same with the TRUTH-REFERENCED
`sigma_bar = sigma - a delta` (`x = z/(1 - a z)`), which is what keeps the
pull-normalisation artefact out:

| `\|eta\|` lead | n | `sigma_m/m` | `vgf` | width (`sigma_fit`) | width (`sigma_bar`) |
|---|---:|---:|---:|---:|---:|
| 0.0-0.4 | 712 630 | 0.01002 | 0.224 | 0.9901 | 0.9894 |
| 0.4-0.8 | 692 709 | 0.01049 | 0.203 | 0.9907 | 0.9900 |
| 0.8-1.2 | 654 293 | 0.01202 | 0.199 | 0.9924 | 0.9925 |
| 1.2-1.6 | 597 606 | 0.01317 | 0.186 | 1.0005 | 1.0007 |
| 1.6-2.0 | 533 756 | 0.01388 | 0.228 | 0.9981 | 0.9984 |
| 2.0-2.6 | 491 668 | 0.02041 | 0.367 | 1.0027 | 1.0092 |
| **inclusive** | 3 682 662 | 0.01228 | 0.217 | **0.9959** | **0.9962** |

and in `sigma_m/m` tertiles: **0.9803 / 0.9895 / 1.0168**.

**The per-candidate mass resolution is correct to 0.4 % and flat in `eta` to
+-0.7 %.** The truth-referenced correction moves it by less than 0.1 %
(0.9959 -> 0.9962), so the pull-normalisation artefact is NOT contaminating it.
The one residual structure is a **3.7 % swing across the `sigma_m/m` tertiles**
(over-estimated for well-measured candidates, under-estimated for poorly
measured ones) — real, but +-2 % about 1.00, not a uniform deficit. For
comparison the private tight-stepper gun's own single-track pull
(`cf_trackres_mugun_ul16_260830`, its validated `z`) is **0.94-0.96, also flat
in `eta`**.

### SO WHAT IS THE "2.8 % WIDTH DEFICIT"? NOT THE RESOLUTION — see sec. 0c
   (read here as "kernel-side"; sec. 0c shows the kernel closes, so the width
   template is absorbing the sigma-mass dependence, not an FSR defect)

`resid_decompose.py` projects the post-fit spectrum residual onto a SHIFT
template (`model(m_Z + 20 MeV)/model - 1`, odd about the pole) and a WIDTH
template (`model(sigma x 1.01)/model - 1`, even). On the full-scale fit,
240 bins, positive shift = the data prefers a larger `m_Z`, positive width = the
data is wider than the model:

| | value |
|---|---:|
| raw residual | `chi2/ndf` = **1130.9/240 = 4.71** |
| after shift + width | **825.1/237 = 3.48** |
| SHIFT | +18.79 +- 1.70 MeV |
| WIDTH | +2.838 +- 0.222 % on `sigma` |

**That +2.8 % is NOT a measurement of a resolution error.** The direct pull says
the resolution is right to 0.4 %, and the no-kernel fit says `k_ms` wants only
+3.0 % of the MS FAMILY (which is a sub-percent effect on `sigma` itself). The
width template is simply the closest available basis vector to a kernel-side
mismodelling, and the giveaway is that with BOTH templates removed `chi2/ndf` is
still **3.48** — they do not describe the residual. **Retract "the modelled
resolution is 2.8 % too narrow" as a statement about the resolution.**

What survives of it: the model's TAILS are under-predicted at both window ends,
and `k_ms` = 1.0298 +- 0.0042 at 7 sigma in the kernel-free fit says the
multiple-scattering tail specifically is ~3 % short. That is real, it is the
material, and phase 3 is its test — but it is not what moves `m_Z`.

### The Z ALONE CANNOT SEPARATE THE RESOLUTION MODEL FROM `m_Z`### The Z ALONE CANNOT SEPARATE THE RESOLUTION MODEL FROM `m_Z` — measured

Float ONE resolution knob at a time on the 300 k card, everything else as the
base fit (the knobs' MC truth is 1.0):

| floated | fitted knob | `m_Z` [MeV] | `Gamma_Z` [MeV] |
|---|---:|---:|---:|
| none (base) | — | -20.67 +- 8.09 | +18.66 +- 14.72 |
| `k_ms` | **+1.001 +- 0.034** | -20.62 +- 8.10 | +17.88 +- 26.55 |
| `k_ioni` | +7.32 +- 4.10 | **-38.89 +- 14.26** | -9.57 +- 23.33 |
| `k_rad` | -0.76 +- 9.61 | -18.25 +- 15.19 | +20.78 +- 18.82 |

Read off three things.

1. **`k_ms` IS measurable from the Z mass spectrum and comes out at the MC
   truth**: 1.001 +- 0.034, a 3.4 % measurement of the multiple-scattering CF
   that costs nothing on `m_Z` (-20.62 against -20.67) and doubles the error on
   `Gamma_Z` (both are widths).
2. **`k_rad` is not measurable at all** — 9.6 on a parameter whose truth is 1.
   The radiative tail cannot be tested, or blamed, from the Z spectrum alone,
   which kills the tidy "the low-side residual is a too-small brems tail"
   hypothesis as something this fit could settle.
3. **`k_ioni` is degenerate with `m_Z`**: floating it moves `m_Z` by -18 MeV and
   inflates its error from 8.1 to 14.3.

**None of the three closes `m_Z`.** Freeing a resolution knob does not remove
the -11 MeV, it inflates the error — which is the physics case for phases 2 and
3 in one table: the J/psi and the hit-chi2 term are what constrain the
resolution model INDEPENDENTLY of `m_Z`, and the Z alone never can.

## 0c. THE KERNEL SIDE CLOSES TOO — THE -11 MeV IS IN THE COMBINATION
   (2026-09-07 21:00)

Two retractions first, both mine, both from today.

**RETRACTED: "the FSR fold describes a sample radiating 1.66x more than the
candidates".** The comparison was wrong. `<u> = <-ln(m_post/m_pre)>` is 0.0240
in the kernel and 0.0144 in the selected sample because the kernel is
deliberately built with NO mass cut — the fold has to be able to move a Born
mass anywhere, and the window is what the truncation normalisation implements.
The model's prediction for the selected sample is the same gen sample
restricted to the window: **0.014234, against 0.014447 measured, 1.5 %**.

**RETRACTED: "what is left is the KERNEL".** It closes. Fitting the SELECTED
candidates' own GEN masses with exactly the detector-level chain (no detector
anywhere), `zchannel/fit_gensel.py`, `shape 5`, window 60-120:

| `\|eta_lead\|` | (a) pre-FSR, no fold | (b) post-FSR, FULL fold | (b)-(a) | detector-level `m_Z` |
|---|---:|---:|---:|---:|
| barrel | -3.28 +- 1.90 | **+0.61 +- 2.05** | +3.89 | -35.79 +- 7.13 |
| transition | +1.01 +- 2.31 | **+4.16 +- 2.51** | +3.15 | +0.08 +- 7.89 |
| endcap | -2.99 +- 2.42 | **-2.70 +- 2.63** | +0.29 | +4.07 +- 9.64 |
| inclusive | -1.94 +- 1.25 | **+0.76 +- 1.36** | +2.69 | -11.06 +- 2.27 |

The fold is worth **+2.7 MeV with a 3.6 MeV `eta` spread**, against a
detector-level barrel-endcap difference of 40 +- 12 MeV. Rebuilding BOTH the
kernel and the acceptance from the selected candidates themselves
(`zchannel/kern_from_selected.py`; the acceptance has to be a `grid` because
`P(selected \| m_pre)` is a top-hat that no degree-8 Bernstein fits) moves `m_Z`
by **+1.5 MeV** and leaves the `eta` pattern alone. `nm` 32768 -> 8192 moves it
by 0.01 MeV.

So the detector half closes at +0.9 +- 2.1 and the kernel half at
+0.8 +- 1.4, and the whole is -11.06 +- 2.27.

### What lives only in the combination: the resolution is not independent of the mass

The likelihood is `prod_i p(m_i \| sigma_i)` and the model computes
`int p(m') K_{sigma_i}(m_i - m') dm' / Z_i` — the SAME Born spectrum `p(m')`
for every candidate whatever its `sigma_i`. On this sample that is badly false.
Octiles of the absolute `sigma_m`:

| `sigma_m` [GeV] | `<sigma>` | **`<m_gen>`** | `<sigma/m>` |
|---|---:|---:|---:|
| 0.332 - 0.783 | 0.699 | **84.94** | 0.0083 |
| 0.914 - 1.012 | 0.965 | 89.21 | 0.0109 |
| 1.103 - 1.227 | 1.161 | 90.50 | 0.0129 |
| 1.403 - 1.758 | 1.553 | 91.12 | 0.0171 |
| 1.758 - 11.9 | 2.597 | **91.28** | 0.0285 |

`rho(sigma, m_gen) = 0.168` and the conditional mean of the TRUE mass runs over
**6.3 GeV**; `sigma` grows with `pT` and `pT` with the mass. In `sigma/m`
classes the same quantity moves only 1.1 GeV (`rho = 0.035`), which is why the
`sigma_rel` splits looked mild and the absolute-`sigma` structure did not show
up.

`p(m_gen \| class)/p(m_gen)` is a factor 5000 tilt in the lowest class (empty
above 110 GeV) and a factor 16 the other way in the highest.

**Why it is invisible to both closure tests.** The generator-level fit has no
`sigma`. The kernel-free residual fit has a DELTA lineshape — every candidate's
true mass is identically zero — so there is no true-mass distribution left to
correlate with `sigma`.

**Why `K(m)` cannot absorb it.** The class-conditional spectra average to the
marginal, `sum_c P(c) p(m'\|c) = p(m')`, but the observed spectrum is
`sum_c P(c) [p(m'\|c) (x) K_c]` and the model can only produce
`sum_c P(c) [p(m') (x) K_c]`. Low true masses get NARROW kernels and high ones
WIDE kernels; the model gives every mass the average mixture. `K(m)` multiplies
the Born spectrum BEFORE the convolution and cannot repair a pairing of kernel
width with mass.

**Two tests are running** (`--sigma-range`, `--decorrelate-sigma`, both added
to `make_card.py` today):
1. six narrow absolute-`sigma` slices at 400 k each with `K(m)` floated —
   inside a slice the tilt is smooth (log-residual to five Legendre terms
   0.03-0.07 for classes 2-6), so the slices should close and the inclusive fit
   should not;
2. the same 400 k sample reweighted by `p(m_gen)/p(m_gen\|class)` so that the
   true mass and `sigma` ARE independent — if that closes, the mechanism is
   proved.

**If confirmed, the fix is a per-resolution-class Born reweighting**
`w_c(m) = p(m\|c)/p(m)`, exactly analogous to the acceptance and measurable
from MC. The class machinery already exists in the term for `_norm_z`.

---

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

## 2. WHAT IS RUNNING  (checkpoint 2026-09-08 00:10 — see also sec. 0d)

**Engaging, native minimiser** (`engaging/fullscale_gpu_native.sbatch`,
`PYTHONPATH` at `rabbit-native`, branch `material-resolution-native` tip
53c1d0f which already carries my `corr_mass`), all six at full statistics,
`--chunk 32768 --method tf-trust-krylov` with 15-minute snapshots on
`mit_preemptable -t 04:00:00`:

| job | card | what it decides |
|---|---|---|
| 22254764 | `z_F_toy` | the assembly, against -11.06 +- 2.27 |
| 22254765 | `z_F_toydc` | the toy plus the decorrelation weights |
| 22254766 | `z_F_dc8` | the real data with those weights |
| 22254767 | `z_F_w70110` | the window normalisation, at 2.3 MeV |
| 22254768 | `z_full380_fl_s6` | the K(m) ladder, 6 terms |
| 22254769 | `z_full380_fl_s7` | the K(m) ladder, 7 terms |

The last two are the third attempt: the 5h30 `mit_normal_gpu` limit killed both
on the scipy path (22227231 at 4h36, 22224280 cancelled at 4h14). At the
measured 16x they are ~20 minutes each. `22210973 zjoint` (phase 2, 500 k) is
still running on the scipy path.

Collect with the helper in the scratchpad or simply
`rsync -a engaging:orcd/pool/zmass/fitresults/fit_*.json results/eng/`.

### (superseded, kept for the job ids)


### On Engaging — `eng 'timeout 30 squeue -u david_w'`
The 8 h SSH master expires silently; when `eng` prints instructions instead of
output a human must run **`!eng-master`**. **Preemptable jobs get preempted** —
`22171547` and `22199336` both restarted from scratch once — so a 4.7 h
full-scale fit on `mit_preemptable` is a coin flip.

| job | what | state |
|---|---|---|
| **22199336** | `zshape` — the K(m) ladder at full statistics (`s6`, then `s7`) | RUNNING since 10:42 (restarted after a preemption). **The loose end of phase 1**: at 300 k, 5 -> 7 terms moves `Gamma_Z` by +42 MeV. |
| **22210973** | `zjoint` — `fit_joint.py --method trust-krylov` on `cards/joint_ok_n500k.hdf5` (500 k + 500 k, the 1641-task quadratic term) | PENDING. **Phase 2 at 5x the statistics of the one that landed.** |
| ~~22199038~~ | the phase-1 variant ladder | ran `base` again (38 iterations, 16 764 s — reproducing 06:32 exactly) and was cut off during `noares`. **The full-scale variant ladder has now failed to get past `base` twice**; the 300 k ladder in sec. 0b answers those questions and the full-scale version is not worth a third GPU-day. |
| ~~22204679~~ | `zjoint`, first Krylov attempt | died on `JointObjective` having no `hessp` — fixed (`17721a0`), and its reference Hessian had SUCCEEDED (456 s, 43.4 GB at chunk 8192 on an H200) |
| ~~22199037~~ | `zjoint`, `trust-exact` at chunk 32768 | OOMed the H200 |

### On submit82 — detached, `setsid`, logs in `fullscale/logs/`

| what | state |
|---|---|
| `fit_f380fl_noboth` | the full-scale "neither correction" row, 16 h in on CPU. Low value now — the 300 k ladder has the answer — but it costs nothing to let it finish. |
| `fit_locK_{etaB,etaT,etaE,srlo,srmid}` | **the splits with K(m) FIXED**, so they share one shape model and become comparable. This is the measurement that decides whether the barrel/`sigma_rel` contradiction in sec. 0b is real or is the shape absorbing differently in each subsample. |
| `fit_loc_{chgP,chgM,vgfH}`, `fit_srel_hi` | the rest of the K(m)-floated splits |

### J/psi v2 — FINAL, no exclusions (2026-09-07 15:30)

`jpsimc_20M_260906_v2` is **1645 / 1645 tasks, 21 750 740 events**.
`./run_phase2.sh xlist` writes `runs/jpsiv2_tasks_ok.txt` from whatever carries
a `.complete` sentinel — **1645 tasks / 6580 files** — and `JV2_BAD` is now
empty.

**History, so nobody re-derives it.** Three separate defects were excluded and
then repaired: **1219-1222, 1334-1337, 1409-1412** were the "re-staged" grid
copies (0.82 candidates/event against 0.997, `chi2/ndof` median **3.5e6** —
worthless, so the exclusion was necessary and not conservative); **1552-1555**
exited rc=91 with no output; **1313** was silently EMPTY (four 15.5 kB streams
WITH a `.complete` sentinel, because its input lacked a StreamerInfo, so the
sentinel check could not catch it and it had to be named). All were re-produced
from properly repacked inputs and validated at 0.9967 candidates/event, 0.0073 %
failures, `chi2/ndof` median 0.953 against 0.954 for the control. Repairing 1313
also recovered its input's missing tail as three NEW tasks, **1642-1644**
(~51 k candidates). Two NUL-signature files' existing outputs were verified
bit-identical against re-runs.

**Which count every input used** — quote this with any number:

| input | tasks | content |
|---|---|---|
| `runs/quad_jpsiv2_ok.npz` | **1645 / 1645** | **16 955 312** in the quadratic term, 4 722 750 cut, 642 s. THE FINAL ONE; `cards/joint_ok_*` are being rebuilt against it |
| ~~the same, 1641~~ | 1313 out | 16 915 249 in the quadratic term; what `cards/joint_ok_*` currently hold |
| ~~the same, 1626~~ | the 16 out | 16 755 046; what `results/fit_joint_v2_n100k.json` used |
| `runs/jpairs_v2_n600.npz` | 600 by CHOICE (tasks 0-599) | 7 923 460 candidates. Untouched by every one of these defects — all of them are above 1219 |
| `runs/gpairs_v2_n50.npz` | 50 (tasks 0-49) | the phase-3 per-group cache; likewise untouched |
| `runs/quad_dyv2.npz`, `runs/zpairs_dyv2_jac_full.npz` | 380 / 380 | the DY production is complete and was never affected |

The successive quadratic terms differ by ~1 %, so a number already taken on an
earlier one is not wrong, only slightly less complete — **say which it used**.

### PHASE 2 AT FULL SIZE DOES NOT FIT `trust-exact` — MEASURED

`fit_joint.py` inherits `fit.py`'s `scipy.optimize.minimize(method="trust-exact")`,
which needs the FULL Hessian at **every** iteration; the Z-alone fit took 38 of
them. On the full joint card that is not affordable in either resource:

* **memory**: `pfor` over 99 free parameters at `chunk 32768` **OOMs an H200**
  (job 22199037, `ResourceExhaustedError`). On CPU the same shape peaks at
  270 GB at `chunk 16384` and 155 GB at 8192. Halving the chunk halves the peak
  and doubles the chunk count, so it buys memory and **not** time.
* **time**: the phase-1 ladder's `pfor` Hessian is 390 s for 113 chunks and 7
  parameters on the same H200. Scaling to the joint card's 205 chunks and 99
  parameters is ~10 000 s per Hessian, i.e. ~105 h for 38 iterations. `hvp` is
  4.3x slower still.

**The way out is DONE** (`4d8cc34`): `ChunkedObjective.hessp(x, p)` is one
forward-over-reverse product with an arbitrary tangent — the same arithmetic as
one column of `_hess_piece_hvp`, plus the analytic prior diagonal — so a Krylov
trust-region step costs O(10) of them per iteration instead of 99 columns, each
~2 gradients and INDEPENDENT of the parameter count. `fit.py` and `fit_joint.py`
take `--method {trust-exact,trust-krylov,trust-ncg}`; one full Hessian is still
built AFTER the fit, for the covariance. `test_hessp.py` checks it against the
assembled Hessian on the real 300 k Z card: **max |hessp - H@p| / max|H@p| =
3.7e-16** over four random tangents.

Two further reductions, if it is still not enough: freeze the parameters nothing
constrains (`material_pp1_cables` is touched by NOBODY, `thermal_screen` and
`support_tube` by < 0.1 %, and the hit-chi2 term is blind to all three — see
phase 3), and reduced statistics.

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

### PHASE 3 — the blocker is GONE (2026-09-07)

`matres/extract_groups.py` cannot be used on either production: it rebuilds the
per-group CF offline from Geant4 step records and dies on the missing
`ioniurbanidx` (both ran `exportStepRecords=False`). Both ran
`exportCfGroupExponents=True`, so **`cf_inmaker.py pairs --groups`** (commits
`b7e7f9d`, `6af07ed`) reads the maker's own split into the CSR layout
`MaterialCFTerm` / `make_material_card.py` consume. The per-group families are
named `grp_*` because `Sms`/`Sio_re`/... are already the FLAT exponents in the
same file. `matres/validate_inmaker_groups.py` (`317a33a`) audits it on 200 000
candidates of each leg: summing the group rows reproduces the flat exponent to
**7.5e-8 relative on both legs**, exactly the float32 storage floor, and the
maker's own float64 `cfmass_grp_closure` reads **1.4e-14**. `make_material_card.py`
reads the cache unchanged and writes a card — which is the proof the layout is
right rather than merely plausible (it caught a real bug first: the
`fit_parmtype`/`fit_subidx` aliases were written before the `jac_*` catalog they
copy).

| cache | tasks | candidates | size |
|---|---|---|---|
| `runs/gpairs_v2_n50.npz` | 0-49 of J/psi v2 (none of the 16 excluded) | 651 672 | 20.5 GB |
| `runs/gzpairs_dyv2_n50.npz` | 0-49 of DY v2 | 487 742 | 17.0 GB |

both with `--jac-parmtypes 14 15`, DY with `--mass-window 91.1876 60`.
30.6 / 34.1 kB per candidate, measured on 1-task runs, so 50 tasks is what fits
a ~20 GB budget — **not** a statistics-driven choice.

**Occupancy, which decides constrainability**: 23.0 material groups per
candidate on the J/psi leg (median 23, p1 16, p99 31, max 35 of 42) and 25.7 on
DY. **`material_pp1_cables` is touched by NOBODY on either leg**;
`material_thermal_screen` (2e-4 / 1e-3) and `material_support_tube`
(2e-4 / 8e-4) by under 0.1 %. Those are three of the four parameters the
hit-chi2 term is ALSO blind to, so they are unconstrained by any term in the
design and must be frozen or given a prior. `material_beampipe`, the fourth, is
touched by 99.99 % of candidates and is fine from the mass side.

**A caveat against the export doc**: it says `cfmass_vgf` keeps meaning the
total Gaussian share (hits + beamspot + pointing) on the two-track tree, so
`vg_other` should be a positive remainder. Measured, `sum_c cfmass_hitv` equals
`cfmass_vgf` — `|vg_other|/vgf` is 1e-6 median and 6e-6 at q99 on both legs, of
BOTH signs. So on two-track candidates the hit classes account for all of `vgf`
and `vg_other` is float32 accumulation noise. **`MaterialCFTerm` should clip
`vg_other` at 0 rather than trust its sign.** The candidates with
`vg_other/vgf ~ 1` are the `vgf = 1` pathologies (no influence decomposition at
all, `sigma_m/m` up to 3828); 11 in J/psi and 62 in DY, all removed by the
`sigma_m/m < 0.10` cut.

**What is left of phase 3**: a `--material` mode in `make_joint_card.py` that
makes BOTH mass terms `MaterialCFTerm`s sharing the parmtype-15 parameters with
the hit-chi2 term, hit-class parameters from the mass terms, corrections
truth-free from `Jpsi_covrefmom` (already so on the J/psi leg: per-candidate
`f_ang`, median 6.2e-2), then Asimov/toy pulls and the group-leader table. And
it inherits phase 2's minimiser problem, with 42 + 18 more parameters.

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

---

## 0d. THE ASSEMBLY — WHAT IS RUNNING AND WHAT IS ALREADY SETTLED
   (checkpoint 2026-09-07 23:45)

Both halves close (sec. 0c). Of the five ways the halves could be joined
wrongly, three are settled and two are being measured.

| item | status | number |
|---|---|---|
| the `_norm_z` class-sigma approximation | **CLOSED** | a 5 % sigma error moves `d lnZ/d m_Z` by 2.1e-8 against its own 7.6e-7 over 5 MeV, i.e. **0.02 MeV**. `Z` = 0.9768-0.9776 |
| the corrections' reference mass (`m_i` vs the shifted `d_i`) | **CLOSED by construction** | `m_true` in the map IS the convolution's integration variable, i.e. the post-FSR mass; `c_i`, `d_i` are per-candidate CONSTANTS |
| the acceptance's position in the fold | **CLOSED by the gen-level fit** | `A` multiplies the BORN spectrum, and the gen fit that uses exactly that order closes at +0.76 +- 1.36 MeV |
| the `tau` grid / upsampling | queued | `--fit-upsample 8` vs 4 on the same candidates |
| the window normalisation | inconclusive at 300 k | 70-110 gives -2.75 +- 9.82, 75-105 gives -24.50 +- 11.57; the full-statistics 70-110 fit is submitted |

**The mechanism that is left, quantified without any fit.** Build the two
observed spectra the model can and cannot produce —
`sum_c P(c)[p(m'|c) (x) N(0,sigma_c)]` against
`sum_c P(c)[p(m') (x) N(0,sigma_c)]` — and ask what mass shift makes the second
match the first with a floated 5-term `K(m)`:

**-15.3 MeV** (stable: -16.1 at 8 classes, -14.6 at 48), against the measured
**-11.06 +- 2.27**. The peak of the truth sits 20 MeV below the peak of the
model. `K(m)` alone removes 30 % of the mismatch; with the shift free, 45 %.

It does NOT explain the `eta` pattern — it predicts -12 barrel / -26 endcap
where the fit gives -36 / +4 — and neither does the kernel, whose whole
band-to-band spread at generator level is 3 MeV. **The `eta` pattern is a
second, separate open item.**

### The four full-statistics fits submitted to settle it (Engaging, H200)

`22254360 F_toy`, `22254361 F_toydc`, `22254368 F_dc8`, `22254370 F_w70110`,
all 3 682 662 candidates, `--engine device --method tf-trust-krylov`
(the rabbit-native agent measured that path at 8x `trust-exact` end to end and
flat in memory). Read them against the base **-11.06 +- 2.27**:

| card | what it is | reading |
|---|---|---|
| `F_toy` | `m_gen_i + sigma_i z_j`, `z_j` shuffled inside 20 `sigma/m` classes | ~-11 => the ASSEMBLY is wrong given correct inputs; ~0 => the data differ from the model INSIDE the convolution |
| `F_toydc` | the same toy, plus the `p(m_gen)/p(m_gen\|class)` weights | with `F_toy` ~-11 and this ~0, the sigma-mass pairing is PROVED to be the cause |
| `F_dc8` | the real data with those weights | the same test without the toy |
| `F_w70110` | the real data, window 70-110 | the window normalisation, at last with a 2.3 MeV error |

**If the pairing is confirmed, the fix is a per-resolution-class Born
reweighting** `w_c(m) = p(m|c)/p(m)`, exactly analogous to the acceptance and
measurable from MC. The class machinery already exists in the term for
`_norm_z`. The rabbit-native agent has asked that the parameter-only prologue
be hoisted out of the per-chunk path first (`MassCFTerm._prologue(values)`
returning the (class, tau) tables and `_norm_z`), because otherwise one FFT per
class is recomputed on every chunk: at 64 classes x 113 chunks that turns a
4 ms/chunk redundancy into the dominant cost.

---

## 6. THE NATIVE MINIMISER PATH  (added 2026-09-07; APPEND ONLY, nothing above changes)

**The default path is untouched.** `--engine auto` resolves to `host` for
every scipy method, so `fit.py` / `fit_joint.py` with the arguments used in
sections 0-5 run what they always ran — checked against the pre-change drivers
on `cards/smoke_zls.hdf5` under `rabbit-material`: identical NLL, identical
iteration count, parameters agreeing to 2e-15 (multithreaded round-off).

### 6.1 The problem, measured

`chunkfit.ChunkedObjective` loops the candidate chunks in **python**, in eager
TF, and calls `.numpy()` on the value and the gradient of **every chunk**. Each
of the 113 chunks of the 3.68 M card is a few hundred individually dispatched
ops followed by a forced device sync, so the device is idle between them.
Sampled through the 300 k fit on an H200 (`fit_n300k_host_trust-exact`):

> **GPU utilisation: mean 9.1 %, median 0.0 %, p90 32.9 %** over 732 samples.

That is the whole of it. The arithmetic was never the bottleneck.

### 6.2 What was built

| piece | where | what |
|---|---|---|
| `ChunkTable` / `JacChunkTable` | rabbit `unbinned.py` | `term._chunks[ci]` for a **traced** `ci`: the same `arr[lo:hi]` becomes a dynamic slice. Python int in, python ints out — the eager path is unchanged |
| `MassCFTerm.candidate_slice` | rabbit `unbinned.py` | the unbinned analogue of PR #154's `ShardIndataView`: a term over candidates `[a, b)` with its per-candidate tensors on one device |
| `DeviceChunkedObjective` | `fullscale/devobj.py` | subclasses `ChunkedObjective` and calls its `_chunk_nll`; the loop is a `tf.while_loop` inside a `tf.function`, `parallel_iterations=1` so one chunk is live, gradient taken **inside** the body |
| `ShardedChunkedObjective` | `fullscale/shardobj.py` | the same, split over N GPUs by candidate |
| `minimize_driver` | `fullscale/` | `--method tf-*` (rabbit PR #153), `--engine`, `--devices`, snapshots (PR #155), `--check-device`, `--gpu-monitor` |

Nothing is approximated. Device vs host, relative:

| card | value | gradient | HVP |
|---|---|---|---|
| `z_n300k` (300 k, 7 free), H200, chunks 8192 / 32768 / 131072 / 300000 | **0** | 1.2e-15 … 9.9e-15 | 1.8e-15 … 9.8e-14 |
| `joint_smoke` (2 mass terms, 120 k, **99 free**, sparse D + external quadratic) | **0** | 1.6e-17 | 3.1e-16 |

and 1, 2, 3 candidate shards reproduce the single-device objective to 4e-16 —
sharding splits a sum, so it is an identity, not an approximation.

**Per-call cost, `z_n300k` on an H200** (best of 3 after the trace):

| chunk | nchunk | trace | device v+g | device HVP | host v+g | host HVP | x v+g | x HVP | GPU dev | GPU host |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 37 | 4.9 s | 0.407 s | 0.801 s | 2.169 s | 11.383 s | **5.3** | **14.2** | 25 % (p90 88) | 14 % (p90 25) |
| **32768** (card) | 10 | 1.2 s | 0.301 s | 0.543 s | 1.043 s | 4.001 s | **3.5** | **7.4** | 32 % (p90 85) | 33 % (p90 42) |
| 131072 | 3 | 0.9 s | 0.269 s | 0.478 s | 0.725 s | 1.984 s | 2.7 | 4.1 | 29 % (p90 82) | 45 % (p90 62) |
| 300000 | 1 | 0.9 s | 0.260 s | 0.452 s | 0.630 s | 1.419 s | 2.4 | 3.1 | 28 % (p90 83) | 56 % (p90 73) |

Read the last two columns together with the first: the host path's cost is
**per chunk**, so it improves as the chunks get larger and the python loop
gets shorter — and at full scale it cannot take that route, because the chunk
size is what bounds the memory (host RSS goes 9.7 -> 22.3 GB across this
scan, and the `pfor` Hessian is `nfree` times a chunk's tape). The device
path is nearly flat in the chunk size: the residual 4 ms per chunk is the
parameter-only prologue (the Z/gamma* transform and the truncation
normalisation), which is re-evaluated once per chunk. Hoisting it was
measured to be worth 14 % of a gradient at `chunk 32768` on the full card and
~2 % at `chunk 262144`, so it was left alone.

### 6.3 How to switch

```bash
RABBIT=/work/submit/david_w/ZMass/rabbit-native \
  ./run_tf.sh python3 -u fit.py --card cards/z_full380_fl.hdf5 \
      --fix k_hit k_ms k_ioni k_rad \
      --method tf-trust-krylov \
      --snapshot-file results/f380.snapshot.hdf5 --snapshot-interval 0.25 \
      -o results/fit_f380_native.json
```

- `RABBIT=.../rabbit-native`, branch **`material-resolution-native`** =
  `material-resolution` + rabbit PRs **#153** (native TF trust-region
  minimizers), **#154** (multi-device) and **#155** (snapshots). The device
  engine refuses to start against any other rabbit and says why.
- `--method tf-trust-krylov` implies `--engine device`. `tf-trust-ncg` and
  `tf-trust-exact` are the other two; the scipy `trust-exact` /
  `trust-krylov` / `trust-ncg` are unchanged and remain the reference.
- `--engine device --method trust-krylov` is the halfway point (fast
  objective, scipy steps) and separates the two effects.
- `--snapshot-file` + `--snapshot-interval <hours>` write the parameter vector
  after every accepted iteration, on SIGTERM (a slurm wall clock kill, a
  preemption) and on failure; `--resume <file>` seeds a fit from one. Three
  fits have been lost to interruptions — this is the fix, and it is what makes
  `mit_preemptable` (which starts H200 jobs in minutes instead of hours) the
  right partition.
- `--devices N` shards the candidates over N GPUs.
- `--check-device` asserts device == host before fitting; `--gpu-monitor 1.0`
  puts the utilisation distribution in the json.

### 6.4 The measurement

**`cards/z_n300k.hdf5`** (300 000 candidates, 7 free), one H200, every row the
same card from the same start, `--gtol 1e-6`:

| engine | method | it | wall | vs default | t/it | GPU mean | GPU p90 | host RSS | TF device | dNLL | max rel dx |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| host | **trust-exact** (today's default) | 20 | **776.7 s** | 1.0x | 38.8 s | **9.1 %** | 32.9 % | 19.1 GB | — | ref | ref |
| host | trust-krylov | 42 | 1299.0 s | 0.6x | 30.9 s | 15.6 % | 29.0 % | 18.2 GB | — | -1e-10 | 4.3e-8 |
| device | trust-krylov | 42 | 109.9 s | 7.1x | 2.62 s | 90.8 % | 94.0 % | 10.0 GB | 8.88 GB | -1e-10 | 4.3e-8 |
| device | **tf-trust-krylov** | 41 | 95.8 s | 8.1x | 2.34 s | 80.1 % | 93.0 % | 10.1 GB | 9.85 GB | +1e-10 | 1.9e-8 |
| device | **tf-trust-exact** | 20 | **73.1 s** | **10.6x** | 3.65 s | 88.3 % | 94.0 % | 10.0 GB | 8.93 GB | **0** | 1.2e-8 |
| device (2 GPU) | tf-trust-krylov | 41 | **63.7 s** | **12.2x** | 1.55 s | 58.3 %* | 92.1 % | 15.5 GB | 9.94 GB | **0** | 1.9e-8 |
| device | tf-trust-ncg | 200 | 198.0 s | — | 0.99 s | 84.6 % | 94.0 % | 10.2 GB | 9.98 GB | **+2.5e+3** | **DID NOT CONVERGE** |

\* averaged over both cards.

**`cards/z_full380.hdf5`** (3 682 662 candidates, 7 free), one H200:

| engine | method | it | wall | t/it | GPU mean | host RSS | TF device | dNLL | max rel dx |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| device | trust-krylov | 61 | 1846.8 s | 30.3 s | 77.4 % | 16.6 GB | 13.14 GB | ref | ref |
| device | **tf-trust-krylov** | 46 | **1205.8 s** | 26.2 s | 81.0 % | 16.6 GB | 14.15 GB | +2.4e-8 | **9.8e-7** |
| device | tf-trust-ncg | 200 | 2550.7 s | 12.8 s | 79.9 % | 16.6 GB | 14.22 GB | +3.2e+4 | **DID NOT CONVERGE** |
| host | **trust-exact** (today's default) | ~38 | **~19 600 s** (see note) | **515 s** | — | 20.9 GB | — | — | — |

The host `trust-exact` row is a projection from a MEASURED rate, not a
completed run: 8 of its ~38 iterations were timed on this exact card at
**515 s/iteration** before the job was cancelled to give a GPU back to the
physics campaign. It is corroborated by `results/fit_f380fl_base.json`, which
took **16 764 s over 38 iterations** of the same minimiser on the sibling
`z_full380_fl.hdf5`.

The reference-point Hessian alone, on that card: **420 s** (host, `pfor`) against
**62 s** (device, `nfree` HVP columns) — and the device one is flat in memory
where `pfor` is `nfree` times a chunk's tape. That single number is most of
what a `trust-exact` iteration costs on the host, which is why its 515 s/it
against `tf-trust-krylov`'s 26.2 s/it is a factor **20 per iteration** and
about **16x** on the wall.

**What each factor buys, separated.** `host trust-krylov -> device
trust-krylov` is the objective alone at *identical* iterates (42 both, same
minimum): **11.8x**. `device trust-krylov -> device tf-trust-krylov` is the
native minimiser on top: 1.15x at 300 k (95.8 vs 109.9 s) and 1.53x on the
full card (1205.8 vs 1846.8 s, and 46 iterations against 61 — GLTR solves the
subproblem to optimality inside the Krylov space, so its steps are better,
not just cheaper). Sharding over a second GPU: 1.50x.

**Every converged row lands on the same minimum.** NLL agrees to 1e-10
absolute out of 8.7e5 at 300 k and 2.4e-8 out of 1.07e7 on the full card;
parameters to **4.3e-8** relative at 300 k and **9.8e-7** on the full card,
against the 1e-6 the comparison asked for.

**`tf-trust-ncg` should not be used on this objective.** Steihaug-CG truncates
at the trust-region boundary instead of solving the subproblem in the Krylov
space, and on a Hessian whose spectrum spans 4.9e-3 to ~1e4 that is not enough:
it hits `maxiter = 200` at NLL +2.5e3 (300 k) / +3.2e4 (full) above the
minimum, with `m_Z` and `Gamma_Z` still at their starting values. This is the
algorithm, not the port — scipy's own `trust-ncg` is the same subproblem.
Use **`tf-trust-krylov`** (GLTR) or **`tf-trust-exact`**.

### 6.5 Caveats

- `--chunk` on a card that carries a per-candidate sparse `D` (the joint
  cards) now **raises** instead of silently mis-aligning the write-time
  blocks. That is a pre-existing bug, not a new restriction.
- `MaterialCFTerm` with CSR group / hit blocks is not graph-chunkable (it
  indexes a numpy pointer array with `int(...)`); such a term falls back to
  the host loop with a clear message.
- The sandwich is still the host implementation — it runs once per fit.
- A sharded rabbit `Fitter` (`--nDevices > 1`, PR #154) evaluates unbinned
  terms in the **global, unsharded** term. That is correct but not sharded;
  before this branch they were silently **dropped** from the likelihood.

### 6.6 The snapshots earned their keep on the first try

The 99-parameter joint benchmark (`cards/joint_v2_n500k.hdf5`, 500 k + 500 k
candidates, `tf-trust-krylov`) was **preempted by slurm at iteration 25** —
`JOB 22253478 CANCELLED DUE TO PREEMPTION`, 1507 s in. The periodic snapshot
written 57 s earlier survived intact:
`fitresults/native/joint500k_device_tf-trust-krylov.snapshot.hdf5`, 99
parameters, `reason='periodic'`, `nit=24`, `elapsed=1496 s`. That is exactly
the failure `mit_preemptable` trades against its minutes-instead-of-hours
start time, and with `--resume` it costs the last snapshot interval instead of
the fit. The run itself was not restarted: its remaining value was one timing
row, against a GPU slot the physics campaign wanted.

(That snapshot carries no `trust_radius`: the job started before the commit
that added it. Newer runs do.)

### 0d.1 The per-resolution-class Born reweighting — the design, agreed with the
rabbit-native agent (2026-09-08)

Not built. Build it only if `F_toy` ~ -11 and `F_toydc` ~ 0 (sec. 0d). The
agent declined to add the prologue hook today, correctly: the hoist means the
gradient no longer flows through one tape and needs a two-stage chain, a subtle
error there is a WRONG GRADIENT THAT STILL CONVERGES, and six full-statistics
fits are running off that branch right now. Ask again when the physics is
confirmed, and gate it on finite differences.

**Where `w_c(m)` goes.** Pre-fold, in the same place `K(m)` acts. The seam
already exists: `ZGammaLineshape.pdf` is `fold_fsr(born_pdf(values)) * _edge`,
normalised. So the hook is on `pdf`, NOT on `cf_tab_ext`:

```python
def pdf(self, values=None, weight=None, **kw):
    y = self.born_pdf(values, **kw)
    if weight is not None:          # (nm,) or (K, nm) on m_grid, NO parameters
        y = y * weight              # <- before the fold
    y = self.fold_fsr(y) * self._edge
    return y / (reduce_sum(y, -1, keepdims=True) * dm)
```

Batched over a LEADING class axis, `_cf_tab` becomes one `tf.signal.rfft` on a
`(K, nfft)` batch and the fold stays a matmul, so 16-32 classes cost close to
one — provided the class axis is never looped in python.

**The prologue contract** on the `MassCFTerm` side:

```python
def _prologue(self, values):
    return {"ktab": self.kernel.provider.cf_tab_ext(values),   # (K, ntau)
            "z":    None if self._norm is None else self._norm_z(values)}
```

with `_chunk_li(values, ci, prologue=None)` and `_mix` taking it and falling
back to computing it, and `nll()` calling it once. The class gather is then one
`tf.gather(ktab, self._class[lo:hi])` next to the `_norm_class` gather that is
already there. Without the hoist, `ChunkedObjective` rebuilds the prologue
inside every chunk's tape: 4 ms/chunk today (14 % of a gradient at
`chunk 32768`), but 16-32 FFTs x 112 chunks once the classes exist.

**Two traps.**
1. `_norm_z` is already per resolution class. If the Born class and the
   resolution class are DIFFERENT partitions you need the outer product, and a
   per-class Born that does not also change `Z` is inconsistent — `Z` is what
   enforces the truncated-likelihood normalisation.
2. The class column must survive `candidate_slice` (the multi-GPU shard view).
   It will if stored as a tf tensor or a length-`n` numpy array; it will NOT if
   stored as a python list or a dict.

**And `w_c` is MC input**, the same status as the FSR kernel and the
acceptance: `w_c(m) = p(m | class c) / p(m)` measured on simulation. Say so
when quoting the result.

---

## 0e. THE FIX, AS INSTRUCTED (2026-09-08) — Punzi's variable-resolution problem

The coordinator has accepted the diagnosis and named it: this is **Punzi's
variable-resolution likelihood problem**, `p(m_true | sigma_i) != p(m_true)`.
The instruction has two parts and the second is the design.

### (a) The MC per-class reweighting is the DIAGNOSTIC ONLY

`w_c(m) = p(m|c)/p(m)` measured on simulation closes the loop — do it with the
`F_toy` family, show the -11 MeV goes away, and report what it does to the
`eta` pattern — but it is **not** the model, because it imports the MC's mass
spectrum per class into the physics.

### (b) The design: condition the kernel on the OBSERVED angular variables

`sigma_i` is a deterministic function of the candidate's kinematics (the muon
`eta`s, `pT`s and hit pattern). At fixed **observed** `(y, cos theta*, phi)` —
equivalently `(eta1, eta2, dphi)`, all measured with smearing negligible
against their own structure — the true mass and the resolution are linked ONLY
through `p ~ m`, which the `a` correction already models. So the kernel must be
the CONDITIONAL spectrum `p(m_true | y, cos theta*, ...)`:

* the **LO differential cross section** `dsigma/(dm dy dcos theta*)` = the Born
  helicity structure times the luminosity `L(x1, x2)` with
  `x1,2 = (m/sqrt(s)) e^{+-y}` — **not** the `y`-integrated `L(m)` in use;
* times the smooth `K` (global 5 terms, or per-`y`-band nuisances if the data
  demand it);
* times the acceptance `A(m | cell)`, which at fixed `(y, cos theta*,
  pT_Z ~ 0)` is an **analytic threshold in `m`** from the muon `pT` cuts — to be
  checked against the selected candidates' gen record;
* times the FSR fold banded in `m_pre` as now — with the gen record used to
  verify whether the fold itself needs a `y` / `cos theta*` dependence.

This is the multi-dimensional cross section David asked for in the
`sin^2 theta_W` context. **The 1D mass fit with a per-candidate resolution is
only consistent when conditioned this way.**

### Plumbing, in order

1. The class axis on the kernel CF, ONCE: class = a `(y, cos theta*)` cell,
   ~10 x 10; a per-class `rfft` of the hat-basis spectrum;
   `MassCFTerm._prologue(values)` hoisted out of the chunk loop; the class
   column a length-`n` tensor so `candidate_slice` shards it; the
   normalisation `Z` per class. The hook design agreed with the rabbit-native
   agent is in sec. 0d.1 and is unchanged by this.
2. On a **branch of `material-resolution-native`**, with a **finite-difference
   gradient gate** on `m_Z`, `Gamma_Z`, `K` and one class weight, BEFORE any
   fit uses it.
3. Provider changes belong in the rabbit lineshape module: `ZGammaLineshape`
   gains the `(y, cos theta*)` differential form, and `make_lumi_table.py`
   gains `L(m, y)`.
4. Report (a) and (b) each with `m_Z`, `Gamma_Z` inclusive and per `eta` band,
   plus the `K(m)` 5/6/7 ladder for `Gamma_Z`.

### What can be checked before any code is written

`sigma_i` and `m_gen` must decorrelate inside a `(y, cos theta*)` cell — that
is the claim the whole design rests on. `rho(sigma, m_gen) = 0.168`
inclusively; if it does not fall to ~0 inside a cell, the cell variables are
not the right conditioning and the design has to change. Both variables are
computable from the cache: `y` from `Jpsi_pt`, `Jpsi_eta` and the mass, and
`cos theta*` from the two legs' `(pT, eta)` (`cos dphi = cosh d_eta -
m^2/(2 pT1 pT2)` for massless muons, and at `pT_Z ~ 0`,
`cos theta* ~ tanh(d_eta/2)`).

### 0e.1 THE PREMISE IS CONFIRMED, AND IT NAMES A SIMPLER FIX (2026-09-08 00:30)

The design rests on "at fixed observed angular variables the true mass and the
resolution are linked only through `p ~ m`, which the `a` correction already
models". Tested, and the sharp version of it is TRUE — with the exponent read
off the data.

**The naive version fails informatively.** Conditioning on a `(y, cos theta*)`
cell makes the raw `sigma`-mass dependence WORSE (`rho` 0.168 -> 0.233 at
10 x 10), because at `pT_Z ~ 0` and fixed `(y, cos theta*)` the mass DETERMINES
both muon `pT`s, so `sigma` becomes a near-deterministic function of `m` inside
a cell.

**The correct version holds.** The link is `sigma ~ m^{1+f}` — the curvature
resolution is what is constant, so `sigma_pT/pT ~ pT`. Scanning
`rho(log sigma - p log m_obs, log m_gen)` it crosses zero at **p = 1.25**, and
the `a` correction's own exponent is `1 + <vgf> = 1.2640`. The RESOLUTION
CONSTANT `k_i = sigma_i / m_i^{1+f_i}` is independent of the true mass to
`rho = -0.01`, inside a cell AND inclusively.

**And that alone accounts for the whole bias.** The same fit-free calculation
as sec. 0d, 16 classes, Gaussian kernels, only the CLASS VARIABLE changed:

| conditioning label | barrel | transition | endcap | **inclusive** |
|---|---:|---:|---:|---:|
| absolute `sigma` | -12.04 | -16.50 | -25.61 | **-15.33** |
| **`k = sigma/m^1.264`** | **-0.23** | **+0.44** | **+0.20** | **+0.30** |

and letting the width additionally SCALE as `k m'^{1.264}` along the
integration changes `+0.30` to `+0.46`, i.e. nothing. **The bias is not the
width gradient. It is that `sigma_i` as a conditioning label carries mass
information and `k_i` does not.**

### The fix this names: change the convolution variable

Writing the smearing as `m_i = m' + k_i m'^{1+f} x` and substituting

```
v(m) = Int dm / m^{1+f} = m^{-f} / (-f)          (v = ln m when f = 0)
```

gives `v_i = v(m') + k_i x` to first order — **a fixed-width convolution in
`v`**, with a width `k_i` that is independent of the true mass. So:

* the FFT still applies, on a grid uniform in `v` instead of in `m`;
* the Born density carries the Jacobian, `p_v(v) = p(m(v)) m(v)^{1+f}`;
* the conditioning is on `k_i`, which is legitimate: `p(m'|k_i) = p(m')`;
* **no per-class Born reweighting is needed for this, and no
  `(y, cos theta*)` cells either**;
* the existing `a` and Jensen corrections stay exactly as they are — they are
  the second-order terms of the same map and are unchanged by the substitution.

The `f_i = vgf_i` spread means the ideal `v` is per candidate; a class axis in
`f` (~8 classes, or a single `<f> = 0.264` with the residual treated as the
existing `c_i x^2`) is the practical form, and that is a MUCH smaller class
axis than the 10 x 10 `(y, cos theta*)` cells.

**What the cells are still for**: the Born spectrum `p(m'|y, cos theta*)` from
`dsigma/(dm dy dcos theta*)` and the acceptance `A(m|cell)` — the physics the
coordinator asked for, and what `sin^2 theta_W` needs — not the resolution
conditioning, which `k_i` settles on its own.

---

## 0f. THE v-FORMULATION — APPROVED, AND ITS GATES (2026-09-08)

The coordinator has approved the change of convolution variable and added the
consistency argument: `k_i` is fluctuation-independent to first order by the
SAME relation the `a` correction encodes,

```
sigma_obs / m_obs^{1+f} = (sigma_bar / m_bar^{1+f}) (1 + (a - (1+f) sigma_bar/m_bar) x)
                        = k_bar   EXACTLY when a = (1+f) sigma_bar/m_bar,
```

which is the `a` the term already uses. So conditioning on `k_i` is legitimate
AND consistent with the correction that is already there.

### What to build

* the fixed-width convolution in `v(m) = -m^{-f}/f`, with the Born density
  carrying the Jacobian `m^{1+f}`;
* a class axis in `f` (~8 classes), with the class-width sensitivity measured;
* the conditioning on `k_i = sigma_i / m_i^{1+f_i}`;
* on a **branch of `material-resolution-native`**.

**The trap the coordinator names**: in `v` the first-order width change with
the fluctuation is ALREADY EXACT — that was the `a` term's job — so a piece can
be double-counted. The residual second-order term of the substitution is

```
v_i - v(m') = k x - (1+f) k^2 m'^f x^2 / 2 + ...
```

i.e. a quadratic coefficient `-(1+f) k sigma_i / (2 m_i)` in `v`, against the
current `c_i = -vgf_i sigma_i^2 / m_i` in `m`. They are the same size. **The
gates decide which pieces survive, not the algebra.**

### The gates, in order, before any full fit

1. **The J/psi gun at a delta kernel**: the final `alpha` — `+0.051 +- 0.017e-3`
   on the gun and `+0.006 +- 0.025e-3` on v3 — reproduced within `0.01e-3`
   with the SAME correction set.
2. **Gradient / finite differences** on `m_Z`, `Gamma_Z`, `K`, `k_ms`.
3. **The `F_toy` assembly toy** closes to its statistical error.
4. **Z alone at full statistics**: `m_Z`, `Gamma_Z` vs the generator, inclusive
   and per `eta` band, and the `K(m)` 5/6/7 ladder for `Gamma_Z`. **The `eta`
   pattern is the open question — report it plainly either way.**
5. The fit-free prediction table (absolute-`sigma` against `k` conditioning)
   reported alongside.

### Deferred, explicitly

The `(y, cos theta*)` provider work stays the plan for the DIFFERENTIAL CROSS
SECTION and `sin^2 theta_W`. It is **not** needed for the resolution
conditioning, which `k_i` settles on its own. Do not build the 10 x 10 class
axis for this.

Then phase 2 and phase 3 with the corrected term.

### 0f.1 THE FORMULATION IS VALIDATED NUMERICALLY (2026-09-08 01:00)

`fullscale/proto_vmass.py`, no TensorFlow: build the smearing that actually
happens — `Int p(m') N(m_i - m'; k m'^{1+f}) dm'`, direct quadrature with the
width taken AT THE TRUE MASS — and ask what mass shift each model needs to
match it, with a 5-term Legendre `K(m)` floated over 60-120.

| `sigma/m` at 91 GeV | the m-model (what the term does today) | **the v-model** |
|---:|---:|---:|
| 0.006 | -6.50 | **-0.00** |
| 0.008 | -10.77 | **-0.00** |
| 0.010 | -15.57 | **-0.00** |
| 0.012 | -20.53 | **-0.00** |
| 0.016 | -29.07 | **-0.01** |
| 0.020 | -31.38 | **-0.01** |

**The v-formulation is exact to 0.01 MeV** where the m-formulation is wrong by
7 to 31 MeV, and the sample's median `sigma/m` is 0.0123.

**And the quadratic coefficient is confirmed by scan, not assumed.** The
script's truth is a pure Gaussian smearing with NO Jensen effect, so what it
can test is the substitution's own curvature. Scanning the coefficient of
`x^2` in `v`, the optimum is `-0.625 k sigma/m` against the derived
`-(1+f)/2 = -0.632`, and there the residual is 0.03 MeV. The Jensen `+1` is a
separate physical effect and its gate is the J/psi gun (gate 1).

Two things this settles for the implementation:

1. **Both Jacobians are required and their ratio is the effect.**
   `L_v(v_i) = E_x[p_v(v_i - u^v(x))]` with `p_v = p(m) m^{1+f}`, and the
   density in `m` is `L_v / m_i^{1+f}`. Their ratio `(m'/m_i)^{1+f}` is 7 %
   over the kernel's own support — exactly the size of the thing being
   corrected. Dropping it was the first bug the script found, and it left a
   residual of -2 to -37 MeV that looked like a partial fix.
2. **There is NO measure term in `v`.** `(1 - a_i x)` is the Jacobian of
   profiling the unconditional width out against the observed one, and the
   substitution absorbs it exactly — that is the coordinator's identity
   `k_obs = k_bar` when `a = (1+f) sigma/m`. Carrying it over would be the
   double count.

---

## 0g. THE DIAGNOSTICS ARE IN — THE MECHANISM IS PROVED (2026-09-08)

All at FULL statistics (3.68 M) on the native minimiser, sandwich errors, MeV.

| card | what it is | `m_Z` | `Gamma_Z` | nit |
|---|---|---:|---:|---:|
| `f380fl_base` | **the reference** | **-11.06 +- 2.27** | -5.26 +- 4.16 | 38 |
| **`F_dc8`** | **the real data, reweighted so the true mass is independent of the `sigma` class** | **-0.12 +- 2.42** | -0.22 +- 4.44 | 76 |
| `F_toy` | the assembly toy: `m_gen_i + sigma_i z_j`, `z_j` shuffled inside 20 `sigma/m` classes | -3.08 +- 2.19 | -2.31 +- 4.13 | 91 |
| `F_toydc` | the same toy PLUS the reweighting | +3.80 +- 2.41 | -0.83 +- 4.41 | 61 |
| `F_w70110` | the real data, window 70-110 | -1.87 +- 3.15 | -0.52 +- 5.01 | 90 |

**`F_dc8` settles it. Removing the dependence of the true mass on the
resolution class removes the ENTIRE -11 MeV**: -0.12 +- 2.42 against
-11.06 +- 2.27, on the same candidates with the same model. The weights are
`p(m_gen)/p(m_gen|class)` and nothing else changed.

**And `F_toy` says where inside the mechanism it sits.** The toy keeps each
candidate's `sigma_i` paired with its own `m_gen_i` and only redraws the
residual from the class pool, so it removes the part of the effect that lives
in the RESIDUAL — the width being wrong at the true mass — and keeps the part
that lives in the class composition. It goes -11.06 -> **-3.08**, i.e.
**~70 % of the bias is the residual-level part**, which is exactly what the
v-formulation fixes and is why `F_toy` does not close on its own.

`F_w70110` is consistent: a narrower window spans less mass range, so less of
the effect, and it lands at -1.87 +- 3.15.

### The `K(m)` ladder at FULL statistics — `Gamma_Z` is still NOT closed

| terms | `m_Z` | `Gamma_Z` |
|---|---:|---:|
| 5 | -11.06 +- 2.27 | -5.26 +- 4.16 |
| 6 | -14.01 +- 2.22 | **+27.11 +- 4.35** |
| 7 | -7.81 +- 2.26 | **+1.14 +- 4.40** |

`m_Z` moves over a 6.2 MeV range, comparable to its own error, so the earlier
"the basis is saturated for `m_Z`" survives at full statistics. **`Gamma_Z`
moves by 32 MeV between 6 and 7 terms with a 4.4 MeV statistical error.** The
`Gamma_Z` closure is a TENS-OF-MeV statement and must be quoted as one until
the shape basis is understood. This is unchanged by everything above — the
ladder was run on the uncorrected term.

### 0f.2 IMPLEMENTATION AND GATES — WHERE IT STANDS (2026-09-08)

Branch **`vmass-conditioning`**, cut from `material-resolution-native`,
worktree `/work/submit/david_w/ZMass/rabbit-vmass`, commit `9d10dad`.

**What was changed, and only this.**

| where | change |
|---|---|
| `ZGammaLineshape(vpow=p)` | resamples its density onto a grid uniform in `v = Int dm/m^p`, carrying the Jacobian `dm/dv = m^p`, and transforms THAT. Nothing upstream moves — the Born spectrum, the acceptance, the FSR fold and `K(m)` are properties of the MASS and stay on the mass grid. `tau_max` is rescaled by `window_hi^p`: `dv` shrinks by the same factor so `ntau` is unchanged, i.e. a change of units, not of resolution |
| `MassCFTerm(vpow=p)` | takes `sigma` as the width in `v` and `mobs` as `v(m) - v(m_ref)`, and rebuilds the fluctuation coefficients (below). Requires `corr_form='fluctuation'` and `corr_mass` |
| `make_card.py --vpow p` | does the transformation on the card: `sigma -> k`, `mobs -> v(m) - v(m_ref)`, `corr_mass -> m`, the term's `m_ref -> m_ref^{1-p}` (the `alpha` LEVER, **not** `v(m_ref)`; they differ by `1/(1-p)`), the norm window mapped, the provider given `vpow` |

**The coefficients in `v`**, and what each is:

```
a^v_i = a_i - p sigma_i/m_i      the RESIDUAL self-consistency. The
                                 substitution absorbs p of it EXACTLY --
                                 k_obs = k_bar to first order when
                                 a = p sigma/m -- and carrying the full a_i
                                 over would be the double count.
g^v_i = -a^v_i + (1 - p/2) sigma_i/m_i
                                 = the Jensen u^2 term (+1) plus the
                                 SUBSTITUTION's own curvature (-p/2).
d^v_i = m_i^{1-p} s_i^2 / 2      the Jensen mean shift, divided by m^p.
                                 There is NO (1 - a_i x) measure term in v.
```

**Gate 2 (finite differences): PASS.** On a 50 k v card, worst
`|analytic - FD| / max(|FD|, 1)` = **1.2e-7** against a 1e-5 requirement, for
`m_Z`, `Gamma_Z`, `shape1` and `k_ms`; and it confirms `vpow` round-trips
through the HDF5 card (`fullscale/gate_fd.py`, `results/gate2_vtest.json`).

**Gate 1 (J/psi gun, delta kernel)**: running. `gate_fluct_gun.py --vpow`
builds the v arm with the FSR kernel mapped to the distribution of
`v(M + dm) - v(M)`. At a delta kernel the true mass is a single point, so the
two formulations must COINCIDE up to the few-per-cent spread the FSR kernel
itself puts on `m_true` — the v form must reproduce the spec's shifts, not
improve on them.

**Gate 3/4**: the 400 k A/B (`vref` against `vp1264`, the same candidates)
is running, then the full-statistics fit, per `eta` band, with the `K(m)`
5/6/7 ladder.

**The exponent.** A SINGLE COMMON `p` is used, not a per-candidate `1 + vgf_i`,
and that is a measurement, not a convenience: the per-candidate version is
WORSE (`rho(k, m_gen) = -0.129` and a 3.27 GeV `<m_gen>` spread across `k`
octiles, against `-0.011` and 0.514 GeV for the common `p = 1.264`). `vgf`
varies for reasons unrelated to the resolution's mass exponent, and dividing by
`m^{1+vgf_i}` injects its own correlation with the mass. So **no `f`-class axis
is needed**, which removes the whole per-class prologue problem. The optimum by
the fit-free prediction is `p = 1.235`; the `a` correction's own exponent is
`1 + <vgf> = 1.264`; the predicted bias is within +-0.35 MeV over
`p in [1.20, 1.264]`.

### 0f.3 GATE 1 PASSES — and it measures the substitution absorbing the a term

`gate_fluct_gun.py --vpow 1.264` on the 299 422 real J/psi gun candidates,
delta kernel, the same correction set. `alpha` in `1e-3`:

| term | `alpha` |
|---|---:|
| uncorrected, m form | -0.00825 |
| **uncorrected, v form** | **+0.12746** |
| `a_res` only, m form | +0.13799 |
| `a_res` only, v form | **+0.13799** |
| both, m form | +0.04857 |
| both, v form | **+0.04944** |
| the spec | +0.0512 +- 0.0167 |

**The gate is the ABSOLUTE `alpha`, not the shift**, and my first version of it
got that wrong and reported FAIL. In `v` the "uncorrected" term is not the same
object as in `m`: the substitution absorbs the self-consistent width, so a v
term with NO corrections at all already carries most of it. The shift relative
to each form's own reference is therefore not like-for-like. What has to agree
is the physical answer with the same correction set, which is what the spec
quotes:

```
  a_res only   |alpha_v - alpha_m| = 0.00000 e-3
  both         |alpha_v - alpha_m| = 0.00087 e-3
  both         |alpha_v - spec|    = 0.00176 e-3      requirement < 0.01  PASS
```

**And the failed comparison is itself the measurement.** The v form's
uncorrected reference sits **+0.13571 e-3** above the m form's, against the
+0.1457 e-3 that the `a` correction applies in `m`. That is the identity
`k_obs = k_bar when a = p sigma/m` working in practice: the substitution
absorbs 93 % of the self-consistent width by construction, and what the
`a^v = a - p sigma/m` residual then applies is the remaining
+0.01053 e-3. Nothing is double counted and nothing is lost.

### 0f.4 GATES 3 AND 4 — WHAT IS RUNNING (checkpoint 2026-09-08)

Engaging, `fullscale_gpu_vmass.sbatch` (PYTHONPATH at `rabbit-vmass`, branch
`vmass-conditioning`), `--chunk 32768 --method tf-trust-krylov`, 15-minute
snapshots, `mit_preemptable -t 04:00:00`. All at full statistics.

| job | card | reads against |
|---|---|---|
| 22272248 | `z_V_full` | `f380fl_base` -11.06 +- 2.27 |
| 22272249 | `z_V_s6` | `f380fl_s6` -14.01, `Gamma_Z` +27.11 |
| 22272250 | `z_V_s7` | `f380fl_s7` -7.81, `Gamma_Z` +1.14 |
| 22272251 | `z_V_etaB` | the barrel, and the `M_etaB` twin now building |
| 22272252 | `z_V_etaT` | the transition |
| 22272253 | `z_V_etaE` | the endcap |
| 22272306 | `z_V_toy` | **GATE 3**: `F_toy` -3.08 +- 2.19 must go to ~0 |

The m-form `eta` bands in `results/` are at 300 k (errors 7-10 MeV) while the
v-form ones are at 3.68 M, so `M_eta{B,T,E}` are being built at full statistics
to make that comparison like-for-like. **The `eta` pattern is the open
question and it is to be reported plainly either way**: nothing measured so far
explains the 40 +- 12 MeV barrel-endcap difference — not the momenta (flat to
3.7 MeV), not the resolution width (pull flat to +-0.7 %), not the kernel
(3 MeV of band-to-band spread at generator level), and not the sigma-mass
pairing, which predicts the OPPOSITE ordering (-12 barrel, -26 endcap).

`fullscale/vtable.py` assembles the gate-4 table from `results/` and
`results/eng/` as the fits land, with the fit-free prediction alongside.

### 0f.5 GATE 2b — the off switch reproduces the m formulation

`v(m) = Int dm/m^p -> m` as `p -> 0`, and every coefficient follows:
`k -> sigma`, `a^v -> a`, `g^v -> -a + sigma/m` (the m form's), `d^v -> m s^2/2`
(the m form's), and the provider's v grid becomes the mass grid. So a card
built with `--vpow 1e-9` must reproduce one built without `--vpow` at all.
Measured on 50 000 candidates (`gate_fd.py --compare`,
`results/gate2b_offswitch.json`):

| displacement | `--vpow 1e-9` | no `--vpow` | \|diff\| |
|---|---:|---:|---:|
| `m_Z` + 5 MeV | +0.70821371 | +0.70821370 | 9.6e-9 |
| `Gamma_Z` + 10 MeV | -3.06461914 | -3.06461932 | 1.8e-7 |
| `shape1` + 0.05 | +10.80402919 | +10.80402913 | 5.9e-8 |
| `k_ms` + 0.02 | -1.28164596 | -1.28164613 | 1.7e-7 |

**PASS at 1.8e-7 against a 1e-5 requirement.** The ABSOLUTE NLL differs by
+0.000227 out of 150 665 — the sum of the per-candidate Jacobians, a constant
the fit cannot see.

Gate 2 also passes on this card (worst relative gradient error 1.5e-6).

### 0f.6 A REAL BUG, CAUGHT BEFORE IT COST A RESULT (2026-09-08)

`ZGammaLineshape.config()` did not carry `vpow`. The datacard stores the
provider as its `config()` dict and `make_provider` rebuilds it on the read
side, so **every v card silently rebuilt its provider in the MASS variable
while the term went on treating its CF as the one in `v`**.

It is not an error and nothing in the card looks wrong. The symptoms:

* the modelled density loses its resonance peak entirely —
  `L_v / (L_m m^p)` ran from **0.026 at the Z peak to 1.6 in the tails**
  instead of being 1;
* the truncation normalisation came out **21x too small**, 0.0461 against
  0.9775;
* and the fit ran away. The full-statistics barrel job reached
  **NLL -10.3 M with the trust radius collapsing to 4.8e-7** over 24
  iterations — a likelihood unbounded below, which is what an under-normalised
  density gives you.

**How it was caught**: not by the fit failing (a runaway can look like a hard
problem), but by asking the density a question it had to answer —
`L_v` must equal `L_m m^p` on the same candidates, and `Z_v` must equal `Z_m`.
Both are cheap and neither needs a minimiser.

With `"vpow": self.vpow` in `config()`: `L_v/(L_m m^p)` = **0.99997** median
(p05 0.99982, p95 1.00021), `Z_v` = 0.977515 against `Z_m` = 0.977518, and the
NLL's dependence on `m_Z` and the shape parameters matches the m card's.

**Gate 1 is unaffected** — `gate_fluct_gun.py` builds its terms in process with
a `DeltaKernel` and no provider, so nothing round-tripped. Gates 2 and 2b were
run on cards and are being re-run; all seven full-statistics jobs were
cancelled and the cards rebuilt.

**The lesson, for the next argument added to a provider**: anything that
changes what `_cf_tab` transforms MUST go in `config()`. There is no test that
would have caught this except one that compares the two formulations' densities
directly, which is now `L_v = L_m m^p` in this file.

**And a warning about the gates themselves.** Gate 2b — the off switch,
`--vpow 1e-9` against no `--vpow` — **PASSED WITH THE BUG IN PLACE**, at
1.8e-7. It had to: at `p -> 0` the v provider and the m provider are the same
object, so a bug that only bites at working `p` is invisible to it. (With the
fix it passes at 7.9e-10, which is the difference between "the two code paths
agree" and "the two code paths agree because they are the same path".)

**Only gate 2c catches it**, because it is the only test evaluated AT the
working exponent that has an independent right answer. An off-switch test is
necessary and is not sufficient.

**A compatibility note that cost three jobs**: `make_card.py` on this branch
writes `"vpow"` into every term config, `None` included, and
`rabbit-native`'s `MassCFTerm` does not take that keyword —
`TypeError: MassCFTerm.__init__() got an unexpected keyword argument 'vpow'`.
So **any card built after `26a86bb` must be read by `rabbit-vmass`**, m-form
cards included. Run everything through `fullscale_gpu_vmass.sbatch`; the branch
is a strict superset of `material-resolution-native`, so nothing is lost by
doing so. (The three m-form `eta`-band jobs failed this way in 33-60 s and were
resubmitted on the vmass runtime.)

### 0f.7 RUNBOOK for the v formulation

```bash
# a card, m form and v form, on the SAME candidates
RABBIT=/work/submit/david_w/ZMass/rabbit-vmass THREADS=24 ./run_tf.sh \
  python3 -u make_card.py --pairs runs/zpairs_dyv2_jac_full.npz \
    --fsr ../zchannel/data/kern_loose_band3.3e-4.npz \
    --acc ../zchannel/data/acc_loose_d8.json --shape 5 --chunk 32768 \
    [--vpow 1.264] -o cards/z_<tag>.hdf5

# GATE 2 + 2c, always, before any fit -- seconds, no minimiser
RABBIT=/work/submit/david_w/ZMass/rabbit-vmass THREADS=8 ./run_tf.sh \
  python3 -u gate_fd.py --card cards/z_<v>.hdf5 \
    --jacobian-compare cards/z_<m>.hdf5 -o results/gate2_<tag>.json

# GATE 1, the J/psi gun (in process, no card, no provider)
RABBIT=/work/submit/david_w/ZMass/rabbit-vmass THREADS=24 ./run_tf.sh \
  python3 -u gate_fluct_gun.py \
    --pairs ../resolution/runs/cf_masspairs_jpsigun_ul16_260905d_m0.npz \
    --kernel ../resolution/runs/cf_masskernel_jpsigun_ul16_260905d_m0.npz \
    --vpow 1.264 -o results/gate1_vmass.json

# the fit, on Engaging (16x the scipy path; snapshots make preemptable safe)
./stage_eng.sh card cards/z_<tag>.hdf5
eng "cd ~/orcd/pool/zmass/engaging && sbatch -A mit_general -G h200:1 \
  -p mit_preemptable -t 04:00:00 \
  --export=ALL,CARD=\$HOME/orcd/pool/zmass/cards/z_<tag>.hdf5,\
ARGS='--chunk 32768 --method tf-trust-krylov' fullscale_gpu_vmass.sbatch <tag>"

# collect and tabulate
rsync -a engaging:orcd/pool/zmass/fitresults/fit_*.json results/eng/
python3 vtable.py
```

**The exponent** is `--vpow 1.264` = `1 + <vgf>`, the `a` correction's own. The
predicted bias is within +-0.35 MeV over `p in [1.20, 1.264]` and the fit-free
optimum is 1.235, so it is not a knob to tune per fit; scan it only to quote
the sensitivity.

### 0f.8 WHAT TO DO NEXT, in order (checkpoint 2026-09-08)

1. **Collect the running fits** and run `python3 vtable.py`.
   `22274048 V_full`, `22274049 V_toy`, `22274050 V_s6`, `22274052 V_s7`,
   `22274053/54/55 V_eta{B,T,E}`, `22274132/33/34 M_eta{B,T,E}` (the m-form
   twins at full statistics, so the `eta` comparison is like-for-like).
   Read `V_full` against `f380fl_base` = -11.06 +- 2.27 and `V_toy` against
   `F_toy` = -3.08 +- 2.19 (that is GATE 3).
2. **Quote the cost honestly**: the v form costs 9 % on `sigma(m_Z)` and
   nothing on `Gamma_Z` (STATE_log.md). 2.27 -> ~2.47 MeV at full statistics.
3. **The `eta` pattern.** `scratchpad/vetakern.sh` is written and held: it
   builds `VK_eta{B,T,E}`, the bands fitted with the PER-BAND FSR kernel
   measured on that band's own selected candidates
   (`zchannel/data/kern_sel_eta{B,T,E}.npz`). The `<u>` data-minus-model
   mismatch IS barrel-worst (+2.0e-4 barrel, -0.04e-4 endcap, ~18 MeV of
   spread, the right sign for the 40 +- 12 MeV pattern) even though the
   GENERATOR-level fits showed only 3.6 MeV — because there a floated `K(m)`
   absorbs it and at detector level, in combination with the resolution, it may
   not. That is the same "lives only in the combination" structure as the main
   result and it is the one surviving hypothesis. Run it after (1).
4. **`Gamma_Z` is still not closed** and is independent of all of this: the
   `K(m)` 5/6/7 ladder moves it by 32 MeV on a 4.4 MeV error at full
   statistics (sec. 0g). `V_s6`/`V_s7` repeat the ladder in the v form; if the
   swing survives, the shape basis is the next thing to understand and
   `Gamma_Z` stays a tens-of-MeV statement until it is.
5. Then **phase 2** (the J/psi joint fit, `22210973` was still running on the
   scipy path) and **phase 3** (`MaterialCFTerm`, the 3 % MS tail,
   `k_ms` = 1.030 +- 0.004) with the corrected term.

### 0f.9 DO NOT QUOTE `V_full` — IT DID NOT CONVERGE IN THE POI DIRECTIONS

`V_full` reported `m_Z = -0.00019 +- 2.32` and `Gamma_Z = +0.0036 +- 4.19` on
all 3 682 662 candidates. **That is not a result.** Landing within 1e-4 of the
starting values with 2-4 MeV errors has a chance probability of order 1e-4, and
the shapes moving proves nothing about the POI directions. The gradient at the
reported point, measured (`ChunkedObjective.value_grad`, chunk 32768):

| parameter | gradient | `sigma` | Newton step, in `sigma` |
|---|---:|---:|---:|
| **`m_Z`** | +0.4234 | 2.3207 | **0.98** |
| **`Gamma_Z`** | -0.4907 | 4.1898 | **-2.06** |
| `shape1` | -0.0032 | 0.0294 | -9.5e-5 |
| `shape2` | -0.2484 | 0.0360 | -0.0089 |
| `shape3` | +0.0447 | 0.0245 | +0.0011 |
| `shape4` | +0.4509 | 0.0105 | +0.0047 |
| `shape5` | +1.4235 | 0.0023 | +0.0033 |

**The POIs are ~1 sigma and ~2 sigma from their minimum; every shape is within
0.01 sigma.** The reported `|grad|inf = 1.42` is `shape5`, and it is converged:
the infinity norm is measuring the stiffest direction and saying nothing about
the softest.

**Why.** The reference-point Hessian eigenvalues span **0.05 to 1e5** — the
soft eigenvalues 0.147 and 0.0498 are exactly `1/sigma^2` for `m_Z` (2.32) and
`Gamma_Z` (4.19), the stiff 1e4-1e5 are the shapes. A stopping rule on the
UNSCALED gradient infinity norm cannot converge a problem with a 1e6 condition
number: it stops when the stiff directions are done. This is the scale problem
the coordinator anticipated, and `rabbit/preconditioner.py` (PR #153) is on the
branch but is not wired into `fit.py`.

**It is not the physics and it is not the v formulation's fault**: the m-form
reference reached `|grad|inf = 8.5e-4` with `--method trust-exact`, on the same
card structure and the same conditioning. What changed is the minimiser.

**Three jobs are settling it**, all from the reported point or a displaced one:

| job | what |
|---|---|
| `22275679 V_conv` | `--method trust-exact --engine device --start-from fit_V_full.json` — scipy's own subproblem, which converged the m form |
| `22275680 V_tight` | the same start, `tf-trust-krylov`, `--gtol 1e-9` |
| `22275601 V_disp` | **the displaced restart**: `m_Z +10`, `Gamma_Z +20`, shapes at their fitted values. It must return to the same minimum within the errors |
| `22275605 V_traj` | the same fit with 72-second snapshots, to see whether `m_Z` and `Gamma_Z` ever left the start |

Until those land the honest statement is: **the v formulation's full-statistics
`m_Z` is not yet measured.** What the gradient does say is that the minimum is
near `-2.3 MeV` in `m_Z` (one Newton step from the stop, diagonal
approximation), against the m form's `-11.06 +- 2.27` — but a one-step estimate
with a correlated Hessian is not a number to quote either.

### 0f.10 GATE 3 PASSES — and it is a CONVERGED fit

`V_toy`: the assembly toy (`m_gen_i + sigma_i z_j`, residuals shuffled inside
20 `sigma/m` classes) at full statistics in the v formulation.

| | `m_Z` | `Gamma_Z` | `\|grad\|inf` | nit |
|---|---:|---:|---:|---:|
| m form (`F_toy`) | -3.08 +- 2.19 | -2.31 +- 4.13 | — | 91 |
| **v form (`V_toy`)** | **+1.01 +- 2.30** | +4.50 +- 4.20 | **0.0049** | 53 |

**Consistent with zero at 0.44 sigma**, from -1.4 sigma in the m form. And
`|grad|inf = 0.0049` — with `sigma(m_Z) = 2.30` that is a Newton step of
0.011 sigma, i.e. **this one is converged**, unlike `V_full` (sec. 0f.9).

That matters for two reasons beyond the gate. It shows the minimiser CAN
converge the v form on this card structure, so `V_full`'s stop is a stopping-
rule accident on one trajectory and not a systematic property of the
formulation. And `V_toy`'s `m_Z` moved from 0 to +1.01, so the POI direction
was explored.

### 0f.11 THE CONVERGENCE GATE IS NOW PART OF EVERY RESULT

`fit.py` and `fit_joint.py` now compute, print and store the per-parameter
gradient and the **diagonal Newton step at the stop, in units of each
parameter's own error** (`g_i sigma_i`, since `H_ii ~ 1/sigma_i^2`), and mark a
result `converged` only if every POI — `m_Z`, `Gamma_Z`, `alpha`, `k_ms`,
`k_hit`, `k_ioni`, `k_rad` — is within `--conv-tol` (default **0.05 sigma**).
When it is not, the fit says so on stdout and names the remedy. The JSON gains
`grad`, `newton_step_sigma`, `worst_poi_step_sigma`, `conv_tol`, `converged`.
`vtable.py` prints the verdict on every cell: a trailing `!` means DO NOT QUOTE,
`?` means the result predates the gate.

`checkconv.py` does the same for a result that already exists, from its card.

**Why this was needed.** `|grad|inf` is not a convergence test for this
objective. Sorting every stored result by `|grad|inf` x `sigma(m_Z)` — a loose
upper bound, but enough to rank them:

| fit | `m_Z` | `\|grad\|inf` | bound [sigma] |
|---|---:|---:|---:|
| `f380fl_base` **(the reference)** | -11.06 +- 2.27 | 8.5e-4 | **0.002 — converged** |
| `V_toy` (gate 3) | +1.01 +- 2.30 | 0.0049 | **0.011 — converged** |
| `V_etaT` | +12.39 +- 4.16 | 1.6e-4 | **0.0007 — converged** |
| `V_full` | -0.00 +- 2.32 | 1.42 | 3.3 (measured: `m_Z` 0.98) |
| `F_dc8` | -0.12 +- 2.42 | 3.04 | 7.4 |
| `F_toy` | -3.08 +- 2.19 | 3.81 | 8.4 |
| `F_toydc` | +3.80 +- 2.41 | 3.01 | 7.2 |
| `F_w70110` | -1.87 +- 3.15 | 43.1 | 136 |
| `f380fl_s6` | -14.01 +- 2.22 | 0.68 | 1.5 |
| `f380fl_s7` | -7.81 +- 2.26 | 57.5 | 130 |

**`fit.py` has always written `gradmax` and I never looked at it.** That is the
mistake, and it is mine: I took the reference's convergence as evidence for the
others because they shared a method.

Six re-runs are in flight from each fit's own stored point with
`--method trust-exact --engine device` (the subproblem that converged the
reference): `22276469 F_dc8_cv`, `22276470 F_toy_cv`, `22276471 F_toydc_cv`,
`22276472 F_w70110_cv`, `22276473 f380fl_s6_cv`, `22276474 f380fl_s7_cv`.

### WHAT DOES NOT DEPEND ON ANY MINIMISER

Three legs of the argument are fits with 2 free parameters or no fit at all,
and none of them is touched by the above:

1. **The fit-free prediction** (sec. 0f.1, 0d): building the two observed
   spectra the model can and cannot produce and asking what mass shift
   reconciles them with a floated 5-term `K(m)`. Conditioning on `sigma` needs
   **-15.33 MeV**; conditioning on `k = sigma/m^1.264` needs **+0.30**. Numpy
   and a Nelder-Mead over 6 parameters; no rabbit, no TensorFlow, no
   trust region. The `eta`-band version and the empirical-kernel version
   (-7.7 to -11.6) are the same machinery.
2. **The generator-level closure** (sec. 0c): the selected candidates' own gen
   masses against the full lineshape (x) A (x) FSR (x) K chain, +0.76 +- 1.36
   MeV inclusively. `zchannel/fit_gen.py`'s own `trust-exact` on a 7-parameter
   binned likelihood, converged in 1-13 s per fit, `|g|inf` printed each time.
3. **The kernel-free detector closure**, +0.9 +- 2.1 MeV, and the mass pull
   width 0.996 flat in `eta` — a delta-kernel fit and a histogram.
4. **`proto_vmass.py`**: the v formulation reproduces a mass-dependent-width
   smearing to 0.01 MeV where the fixed-width form is wrong by 7-31 MeV. Pure
   numpy quadrature.

So the DIAGNOSIS — that the bias is the pairing of the per-candidate resolution
with the mass, and that conditioning on `k` removes it — rests on
minimiser-independent evidence. What is pending re-verification is the
size of the effect AS MEASURED BY THE FULL LIKELIHOOD, i.e. `F_dc8` and
`V_full`.

### 0f.12 THE AUDIT — measured, per parameter (`checkconv.py`)

The diagonal Newton step at each stored fit's own stopping point, in units of
that parameter's own error. Requirement for a POI: < 0.05 sigma.

| fit | `m_Z` reported | `m_Z` step | `Gamma_Z` step | verdict |
|---|---:|---:|---:|---|
| **`f380fl_base`** (the reference) | **-11.064 +- 2.267** | **0.0000** | **0.0000** | **CONVERGED** (gradient 4.6e-11) |
| `F_dc8` | -0.117 +- 2.423 | **+0.972** | **+3.323** | NOT CONVERGED |
| `F_toy` | -3.082 +- 2.192 | +0.323 | +0.687 | NOT CONVERGED |
| `f380fl_s7` | -7.812 +- 2.256 | **+4.825** | -1.958 | NOT CONVERGED |
| `V_full` | -0.000 +- 2.321 | +0.983 | -2.056 | NOT CONVERGED |
| `V_toy` | +1.015 +- 2.300 | ~0.011 | — | CONVERGED |
| `V_etaT` | +12.393 +- 4.157 | ~0.0007 | — | CONVERGED |

In every unconverged case **every `shape` is within 0.03 sigma** and the POIs
are not: the stopping rule converged the stiff directions and abandoned the
soft ones, exactly as sec. 0f.11 says.

**What this costs, honestly.**

* **The reference stands.** `-11.06 +- 2.27` is converged to a gradient of
  5e-11. The thing being explained is solid.
* **`F_dc8` -0.12 is RETRACTED as a number.** I called it "the proof". Its
  `m_Z` is 0.97 sigma from its minimum, and a diagonal step puts that minimum
  near **-2.5 MeV** — still most of the way from -11 to 0, but not the clean
  zero I reported. The re-run decides.
* **`F_toy` -3.08 is RETRACTED**; diagonal estimate ~-3.8.
* **The `K(m)` ladder is RETRACTED.** `f380fl_s7` has `m_Z` 4.8 sigma out.
  The "`Gamma_Z` swings 32 MeV between 6 and 7 terms" statement is NOT
  established — it may be entirely a convergence artefact, and it was one of
  the headline caveats. `f380fl_s6`/`s7` are re-running.
* **`V_full` was already retracted** (sec. 0f.9).
* `V_toy` (gate 3) and `V_etaT` are converged and stand.

The diagonal extrapolations above are indicative only: the POIs correlate with
the shapes, so the true displacement is larger, and none of them is a number to
quote. The re-runs are the answer.

### 0f.13 EDM IS THE CRITERION — and the re-runs MOVED

**Why the diagonal proxy was inadequate.** `g_i sigma_i` is a coordinate-wise
Newton step: it assumes `H` is diagonal. On this objective it is not — the
Hessian's condition number is ~1e6 and the POIs are strongly correlated with
the `K(m)` shapes, so the displacement along a POI is driven by the OFF-diagonal
block and the proxy misses it. Proof from the re-runs below: `F_w70110_cv` and
`f380fl_s7_cv` were marked "converged" by the proxy (POI diagonal steps 0.000)
while still carrying `|grad|inf` of 12.3 and 39.6.

**The criterion is now rabbit's own EDM**, `0.5 g^T H^-1 g`
(`rabbit.tfhelpers.edmval`), computed with the FULL Hessian at every stop —
free in `fit.py`, because `C = H^-1` is already inverted for the errors.
rabbit's fitter has **no `edmtol`**: it runs to `gtol = 0`, terminating when the
quadratic model predicts no further improvement, and reports EDM as a
diagnostic. So the threshold is ours and is chosen to mean something: a
displacement of `d` sigma in the worst direction costs `d^2/2`, so
**EDM < 1e-3 is `d` < 0.045 sigma**. The full Newton step `-H^-1 g` per
parameter, in units of its own error, is kept as the interpretable column.

**The re-runs from each stored point with `trust-exact`** — and they moved:

| fit | `m_Z` before | `m_Z` after | `Gamma_Z` before | after | `\|g\|inf` after |
|---|---:|---:|---:|---:|---:|
| `F_dc8` | -0.117 | **-2.035** | -0.220 | -10.997 | 0.010 |
| `F_toy` | -3.082 | -3.752 | -2.308 | -4.730 | 0.006 |
| `F_toydc` | +3.796 | +8.734 | -0.829 | -5.256 | 2.0e-4 |
| `F_w70110` | -1.873 | **-13.618** | -0.518 | -13.441 | **12.3** |
| `f380fl_s6` | -14.006 | -13.981 | +27.113 | +27.161 | 0.029 |
| `f380fl_s7` | -7.812 | **-17.235** | +1.140 | +8.956 | **39.6** |

`f380fl_s6` moved by 0.025 MeV in one iteration — it WAS essentially at its
minimum. The others moved by 0.7 to 12 MeV. Two are still not converged.

**And `V_traj` reproduces `V_full` EXACTLY** (-0.000 +- 2.32, `|g|inf` 1.42, 68
iterations, same NLL): the stop is deterministic, so **check (2) is answered —
`m_Z` and `Gamma_Z` never left their starting values.** That is not a closure,
it is a fit that never took a POI step.

**Nothing in the full-likelihood table is quotable yet.** Six re-runs are in
flight with `--gtol 0` and EDM reporting (`22292576-87`), and the EDM audit of
every stored result is running.

---

## 7. BACK INSIDE `rabbit_fit.py`  (added 2026-09-08; APPEND ONLY)

### 7.1 Why the standalone drivers existed, and why they no longer have to

`fit.py` / `fit_joint.py` / `chunkfit.py` / `devobj.py` were written for ONE
reason: an unbinned term evaluated inside `Fitter` built the whole sample's
forward tape, and the `pfor` Hessian kept `nparams` copies of it — **116 GB at
300 000 candidates and 5 parameters**, i.e. unrunnable at the 3.68 M of the
real sample. Everything else about them was a consequence, and the price was
everything rabbit already had: the **EDM**, the termination convention, the
snapshots, the standard result file, the impacts, the scans, the plotting —
and the convergence trap that cost two fits.

That reason is gone. The candidate loop is now the **term's own
implementation** (`rabbit/unbinned.py`), so an unbinned term is an ordinary
differentiable function of its parameters with a memory footprint of one
chunk, and `Fitter.minimize` / `rabbit_fit.py` need to know nothing about it.

### 7.2 What moved into rabbit

| piece | what it does |
|---|---|
| `UnbinnedTerm._chunk_loop` | `tf.while_loop` over the candidate chunks, `parallel_iterations=1` — which is what bounds the live memory to one chunk, not a tuning knob |
| `UnbinnedTerm._nll_graph` | `tf.custom_gradient` twice over: the forward pass is the value loop, its VJP is the **gradient** loop (gradient taken in the body and accumulated), and the gradient's own VJP is the forward-over-reverse **HVP** loop. So `loss_val`, `loss_val_grad` and the revrev `loss_val_grad_hessp` work unchanged and nothing outside ever differentiates *through* a chunk |
| `MassCFTerm._chunk_contribution` | one chunk's `-sum log L`, self-contained so the same body serves a python and a traced chunk index |
| `UnbinnedTerm.chunk_mode` | `graph` (default) or `eager` — the original python loop, kept as the reference the graph path is checked against and as the fallback for a term whose slicing needs a python index |
| `UnbinnedTerm.rechunk` | the chunk size as a fit-time knob (`--unbinnedChunk`), refusing a card whose sparse `D` blocks were sliced at write time |
| `Fitter` | with unbinned terms present the Hessian is assembled **column by column from HVPs** (`hessian_from_hvps`, PR #154) instead of `t2.jacobian`, one at a time, and `fwdrev` falls back to `revrev` (forward mode does not traverse a registered VJP). Both automatic — an unbinned card must never take the pfor route and the user should not have to know that |

### 7.3 How to run a fit now

```bash
RABBIT=/work/submit/david_w/ZMass/rabbit-native \
./run_tf.sh python3 -u $RABBIT/bin/rabbit_fit.py cards/z_full380_fl.hdf5 \
    --paramModel UnbinnedParams \
    --minimizerMethod tf-trust-exact \
    --freezeParameters k_hit k_ms k_ioni k_rad \
    -t 0 --unblind --diagnostics \
    --snapshotFile results/f380.snapshot.hdf5 --snapshotInterval 0.25 \
    --outpath results --outname rabbit_f380.hdf5
```

- `--freezeParameters` is `fit.py`'s `--fix`.
- `--minimizerMethod tf-trust-exact` or `tf-trust-krylov` (section 6); the
  scipy methods work too.
- `--diagnostics` prints the **EDM** every iteration — the thing the
  standalone drivers never had, and the reason the convergence trap went
  unseen.
- `--unbinnedChunk N` retunes the chunk without rebuilding the card.
- The result is an ordinary rabbit fit file: `io_tools.get_fitresult` gives
  `parms` (values and variances), `cov`, `edmval`, `nllvalreduced`,
  `epoch_loss`, `postfit_profile`.
- `engaging/rabbit_native.sbatch` runs the gate and then the fit on an H200,
  and resumes from its own snapshot if the job is requeued.
- **The sandwich still comes from the standalone driver.** rabbit does not
  compute it, and it is not optional (the MiNNLO weights alone are a flat
  x1.109 on every error). `rabbit_to_json.py` is the link:

  ```bash
  python3 rabbit_to_json.py results/rabbit_f380.hdf5 -o results/f380_start.json
  python3 fit.py --card ... --no-fit --start-from results/f380_start.json
  ```

  so the sandwich is evaluated at rabbit's minimum rather than at wherever
  another minimiser stopped. The EDM travels with the json.

### 7.4 The gates

`fullscale/test_rabbit_path.py`, run by `engaging/rabbit_native.sbatch`
before every fit. Relative differences:

| gate | `smoke_zls` (50 k) | `joint_smoke` (2 terms, 120 k, 99 free, sparse D + external quadratic) | `z_n300k` (300 k, H200) |
|---|---|---|---|
| graph vs eager chunk loop — value | **0** | **0** (both terms) | **0** |
| — gradient | 1.9e-16 | 1.6e-16 / 1.1e-16 | 5.0e-14 |
| — HVP | 2.3e-18 | 5.8e-16 / 1.8e-16 | 5.0e-14 |
| the term vs `ChunkedObjective` — value | **0** | **0** | **0** |
| — gradient / HVP | 3.1e-15 / 1.7e-16 | 3.5e-19 / 1.9e-18 | 5.0e-17 / 1.3e-16 |
| the term vs `DeviceChunkedObjective` | **0** | 3.5e-16 / 6.6e-16 | 4.0e-16 |
| the Fitter's HVP Hessian vs the standalone `pfor` Hessian | 4.6e-17 | — | **2.4e-15** / 3.1e-15 |
| the Fitter's gradient vs central finite differences | 3.6e-7 | 1.0e-12 | 1.1e-7 |

and end to end, `rabbit_fit.py` against the standalone `fit.py` on
`smoke_zls` from the same start, all six parameters free:

| | `rabbit_fit.py` | standalone `fit.py` | rel |
|---|---:|---:|---:|
| `m_Z` | -177.934677801 | -177.934634181 | 2.5e-7 |
| `Gamma_Z` | +153.835324960 | +153.835303478 | 1.4e-7 |
| `k_hit` | +1.160845148 | +1.160845176 | 2.3e-8 |
| `k_ms` | +0.674593691 | +0.674593584 | 1.6e-7 |
| `k_ioni` | -30.611844062 | -30.611850249 | 2.0e-7 |
| `k_rad` | +203.561926297 | +203.561897317 | 1.4e-7 |

with the errors identical to six decimals and the same NLL
(149975.618606). The 2.5e-7 is not a disagreement about the objective — the
gates above say that is exact — it is two minimisers stopping at slightly
different points: the standalone one at `gtol 1e-6`, rabbit at `tol 0.0`,
which took it to **EDM 2.1e-24**.

### 7.4b Three upstream bugs this turned up, all pre-existing on `main`

They matter here because every one of them is on the path this migration
puts the fits back onto, and none is reachable from the standalone drivers.

1. **`tfhelpers.tf_edmval` returned the function `edmval`, not the value.**
   It is the GPU branch and the only one, so `--diagnostics` on a GPU printed
   `<function edmval at 0x...>` where the EDM should be. The CPU branch goes
   through `scipy_edmval` and was always right.
2. **`tfhelpers.cond_number`'s GPU branch called `tf.linalg.cond`, which is
   not a TF symbol.** The first `--diagnostics` iteration on a GPU therefore
   raised `AttributeError`, which the fitter reports as *"Minimizer raised"*
   and turns into a fit that stops where it stands. Replaced by the ratio of
   the extreme singular values.
3. **The `--diagnostics` line did not mask frozen parameters.** A frozen
   parameter contributes an exactly zero row and column to the Hessian, so the
   full matrix is singular as soon as anything is frozen and the EDM solve
   raised *"Input matrix is not invertible"* -- again reported as *"Minimizer
   raised"*. `edmval_cov` had always masked them; the diagnostics line had
   not. So `--diagnostics --freezeParameters` was a failed fit, on GPU and
   CPU alike.

`Fitter.log_diagnostics` now does the masking and swallows anything left with
a warning: a diagnostic must never be able to fail a fit.

### 7.5 Features verified through the card → `rabbit_fit.py` path

| feature | where it is exercised |
|---|---|
| the **external quadratic** term (`hitchi2`, 92 parameters) | `joint_smoke` gate: it dominates the loss (3.4e8) and the Fitter's gradient matches central finite differences to **1.0e-12** |
| **`scale_param=None` J/psi delta-kernel** | `joint_smoke` gate, the `jpsi` term: graph vs eager **0** on the value |
| **`norm_window`** (60-120 GeV) and **`upsample`** (4) | `z_n300k` gate: `nt` 64 -> integration 253 |
| the **per-candidate sparse `D`** (92 jac params) | `joint_smoke` gate, both terms |
| the **fluctuation-form corrections** and **`corr_mass`** | `z_n300k_fl` gate |
| **`vpow`** | as written today it has no rabbit code path: `make_card.py --vpow` is a change of variable applied to the STORED `sigma`, `mobs`, `m_ref` and window, so the term sees ordinary arrays. Verified by reading the writer, not by a run. (The `vmass-conditioning` branch makes `MassCFTerm(vpow=p)` a real path; this row needs redoing when that lands) |
| **frozen parameters** (`--freezeParameters` = `fit.py --fix`) | the 300 k fit; and see bug 3 above, which this is what found |

### 7.5b Two things the migration had to learn

1. **The graph chunk loop bounds memory only inside a `tf.function`.**
   `tf.while_loop` executes as a plain python loop in eager mode, so calling
   `term.nll` at the prompt is correct but holds the whole sample -- an H200
   OOM at 3.68 M candidates. Every path the Fitter uses (`loss_val`,
   `loss_val_grad`, `loss_val_grad_hessp`) is a `tf.function`, so this never
   bites a fit; it bit the gate, which now wraps its comparisons the same way
   and says so.
2. **`mit_preemptable` requeues a job by re-running the script from the top.**
   `rabbit_native.sbatch` therefore resumes from the snapshot when one is
   there (`--externalPostfit`), which is what makes that partition usable for
   a multi-hour fit at all. `FRESH=1` overrides.

### 7.6 What the standalone drivers are still for

Nothing that a fit needs. They stay as the **reference implementation** the
rabbit path is checked against (`test_rabbit_path.py` compares against them
directly) and for the diagnostics built on them (the sandwich covariance,
`report_cov`, the `--ares` / `--jensen` / `--corr-clip` model switches that
turn one card into a scan). New fits should go through `rabbit_fit.py`.

### 7.7 The full-scale runs

`cards/z_n300k.hdf5` and `cards/z_full380.hdf5` through `rabbit_fit.py` on an
H200, gate first, snapshots on. The gates pass on both. **One finding worth
having before you run the full card:**

`--minimizerMethod tf-trust-exact` in RAW parameter coordinates **crawls** on
this problem. Measured on the 300 k card, the EDM goes
16495 -> 1150 -> 421 -> 227 -> 210 and then moves about 1 % per iteration:
208.9, 207.0, 203.0, 201.0, 197.1, 195.2, 191.4, 189.4. That is the trust
region failing to grow where the curvatures span the **3.4e12** condition
number of this Hessian (the soft end is `1/sigma^2` for `m_Z` and `Gamma_Z`,
the stiff end is the K(m) shape block): a radius of 1 is enormous for the
shapes and negligible for the POIs, so steps are rejected or clipped and the
radius never doubles. It is the same disease as an unscaled gradient-norm
stop, one level down.

**Use `--precondition`** (off by default). It reparameterises a block so the
reference Hessian is the identity there -- a pure change of variables, so the
minimum is unchanged -- and the trust region then measures distance in units
of the local curvature instead of GeV-versus-dimensionless.

`--diagnostics` is not free: it builds the full Hessian every iteration for
the EDM, which is `nfree` HVP columns. Worth it while establishing that these
fits converge; afterwards read the final `edmval` out of the result file.

### 0f.14 BACK INSIDE `rabbit_fit.py` — the convergence problem has a proper fix

The rabbit-native agent has moved the unbinned candidate loop into rabbit
itself (`UnbinnedTerm._chunk_loop`, a `tf.while_loop` wrapped in
`tf.custom_gradient` twice), so **an unbinned card is now an ordinary rabbit
fit**: `rabbit_fit.py` works on it and brings rabbit's EDM, its termination
convention, its snapshots, its result file, impacts and scans. Their gates:
the graph loop against the eager loop is EXACTLY 0 on the value and <=5e-14 on
gradient and HVP; rabbit's term against my `ChunkedObjective` is exactly 0;
the Fitter's HVP-assembled Hessian against my `pfor` one is 2.4e-15; and end to
end `rabbit_fit.py` against `fit.py` agrees to 2.5e-7 on all six parameters of
the smoke card — the residual being rabbit running to **EDM 2.1e-24** where
`fit.py` stopped at `gtol 1e-6`. That difference is the whole of my problem.

**`material-resolution-native` is merged into `vmass-conditioning`** (`558637a`).
Clean: their changes are in `UnbinnedTerm`, `MassCFTerm._chunk_contribution`
and `MaterialCFTerm._csr_bounds`; mine are `ZGammaLineshape(vpow)` and
`MassCFTerm(vpow)`. **All three v gates re-pass after the merge**: finite
differences 1.5e-6, the off switch 7.9e-10, the Jacobian identity 2.1e-4.

**`--precondition` is not optional here.** The agent measures `tf-trust-exact`
on `z_n300k` going EDM 16495 -> 1150 -> 421 -> 227 -> 210 and then CRAWLING at
about 1 % per iteration — the trust region cannot grow in raw coordinates whose
curvatures span twelve orders of magnitude, because a radius of 1 is enormous
for the shape block and negligible for `m_Z`. Preconditioning reparameterises
so the reference Hessian is the identity: a pure change of variables, same
minimum, but the trust region then measures distance in units of local
curvature. On a **3.4e12** condition number (measured, sec. 0f.12) it is the
difference between converging and crawling.

**Three fits are running through it** with `--precondition`, `--diagnostics`
and `tf-trust-exact` (`engaging/rabbit_vmass.sbatch`, PYTHONPATH at
`rabbit-vmass`):

| job | card | what it decides |
|---|---|---|
| `22300667 f380ref` | `z_full380_fl` | **the control**: it must come back at -11.06 |
| `22300668 Rdc8` | `z_F_dc8` | the mechanism, at last with EDM |
| `22300669 Rvfull` | `z_V_full` | the v closure, at last with EDM |

`fullscale/rabbit_to_json.py` (the agent's, `151bfcf`) converts a rabbit result
into the json `fit.py --start-from` reads, carrying `edmval`, so the sandwich
covariance and the `x1.109` weight factor can still be evaluated at rabbit's
minimum with `fit.py --no-fit`.
