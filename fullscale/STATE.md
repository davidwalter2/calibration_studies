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
| **`m_Z` closure** | **-11.06 +- 2.27 MeV** | **the COMBINATION of the kernel and the resolution — neither alone** (sec. 0b, 0c). BOTH halves close in isolation: with no kernel at all (`m_reco - m_gen` against the resolution CF) the detector half gives **+0.9 +- 2.1 MeV** and the mass pull width is 0.996 flat in `eta`; with no detector at all (the selected candidates' own GEN masses against the full lineshape (x) A (x) FSR (x) K chain) the kernel half gives **+0.8 +- 1.4 MeV**. The leading candidate for what only exists in the combination is that **the per-candidate resolution is not independent of the mass**: `<m_gen>` runs from 84.94 GeV in the lowest absolute-`sigma` octile to 91.28 in the highest, and the likelihood gives every candidate the same Born spectrum. Phase 3 cannot fix this one either. |
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

## 2. WHAT IS RUNNING  (checkpoint 2026-09-07 15:15)

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
sections 0-8 run what they always ran — checked against the pre-change drivers
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
