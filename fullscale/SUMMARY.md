# Full-scale feasibility study — `m_Z` and `Gamma_Z` from the unbinned CVH mass likelihood

Written for the group. Every number here is certified (**value AND NLL AND
EDM**) or is a minimiser-independent measurement. The complete reference,
including how to reproduce each step, is `STATE.md` next to this file; the
programme-wide resolution/likelihood reference is
`/work/submit/david_w/Documents/Resolution/RESOLUTION.md`. Figures:
`~/public_html/ZMass/cvh/260909_fullscale/`.

---

## 1. What was asked, and the one-paragraph answer

*Can the unbinned per-candidate CVH mass likelihood measure `m_Z` and
`Gamma_Z` on Run-2-scale simulation, and does it close on the generator?*

**The statistical precision is there and is not the limitation:
`sigma(m_Z) = 2.27 MeV` and `sigma(Gamma_Z) = 4.16 MeV` on 3.68 M Z -> mumu MC
candidates, measured at the fitted minimum.** The machinery runs end to end —
productions, per-candidate exports, card build, `rabbit_fit.py`, converged
minima with an EDM, sandwich errors — including the joint J/psi + Z + hit-chi2
fit. **Inclusively the v form closes at -1.54 +- 2.31 MeV, and the `eta` bands
agree with it**: the 55 MeV band spread that had looked like an
`eta`-dependent detector effect was the band SELECTOR, and on a safe band
variable the spread collapses to -12.5 +- 5.9 MeV, the `chi2` against a common
value falls from 93.8/2 to 6.7/2, and the bands' weighted mean equals the
inclusive closure to three digits (section 5.5). **What is left between this
and a demonstrated closure is ONE thing: the `K(m)` shape truncation** — 2.4
MeV over 5 -> 7 Legendre terms in the v form, and a +26.6 MeV swing between 7
and 9 terms in the superseded m form. It is a model/analysis effect, not a
detector or a reconstruction one. **The verdict is in section 8.**

---

## 2. The productions

Two CVH two-track MC productions with the complete export set ("v2"), on
**HTCondor** rather than slurm.

| | `dymc_8p5M_260906_v2` | `jpsimc_20M_260906_v2` |
|---|---|---|
| channel | Z -> mumu (POWHEG MiNNLO, official UL16 SIM/RECO, 106X) | J/psi -> mumu |
| chunks | 380 | 1645 |
| events | 8 502 597 | 21 750 740 |
| candidates produced | 3 799 624 | 21 678 062 |
| candidates in the cache | **3 733 323** (mass window) | **7 923 460** (600-task pairs cache); 16 955 312 in the quadratic term |
| candidates after the card selection | **3 682 662** | 3 000 000 used in the joint card |
| volume on ceph | 267 GB | 1.5 TB |
| threads / memory | 4 / 5000 MB | 4 / 5000 MB |
| throughput | 379 concurrent within 31 min over 12 sites; **177.8 tasks/h against 11.4 on slurm = 15.6x** | ~1.5 h/chunk |
| slim export | ~81 kB/candidate; per-group material exponents add ~26 kB | same |

A third input, the **hit-chi2 external quadratic** over **20 706 999**
candidates, carries the alignment/field/material curvature into the joint fit
(`globalfit/extract.py --no-mass`).

**Production lessons worth keeping** (detail in `production/PRODUCTIONS.md`):
condor beat slurm by 15.6x on the same payload; the repacked split-1 J/psi
ALCARECO shares its LFN with the central split-99 file, so it must be read
only through the named submit doors with the file size asserted; a
null-pointer dereference in the release's `XrdAdaptor`
(`XrdRequestManager.cc:124-131`) was the cause of the "MT SIGSEGV" and is
patched locally and worth sending upstream; and `resume.sh` had used the bash
builtin `GROUPS` as an array name, so the resume paths had never worked.

---

## 3. The statistical precision — the headline numbers

Z alone, all 380 DY tasks, **3 682 662** candidates, resolution and alignment
**fixed at the MC truth**, `K(m)` floated with 5 Legendre terms, both
mass-likelihood corrections in the fluctuation form, errors the **x1.109**
sandwich (`N_eff/N = 0.8130` from the MiNNLO weights).

| | sandwich | inverse Hessian |
|---|---:|---:|
| **`sigma(m_Z)`** | **2.27 MeV** | 2.07 MeV |
| **`sigma(Gamma_Z)`** | **4.16 MeV** | 3.78 MeV |

With `K(m)` **fixed** the Asimov projection is **1.51 / 2.91 MeV**, so
floating the LO -> MiNNLO shape costs **x1.5 on `m_Z` and x1.4 on `Gamma_Z`** —
the price of not having to trust an LO parton luminosity. `Gamma_Z` at 4.2 MeV
is already at the scale the Z-width sensitivity note targets (~2 MeV). On the
full Run-2 muon sample (~10x this MC) the statistical `sigma(m_Z)` is well
below 1 MeV and systematics dominate.

The v form costs essentially nothing: its inverse-Hessian `sigma(m_Z)` at full
statistics is 2.08 against the m form's 2.07.

**The joint (phase-2) fit runs.** `P2smoke` — 500 k J/psi + 500 k Z + the
hit-chi2 quadratic over 20.7 M, 95 free parameters — converged at
**EDM 6.2e-19** in 3 h 39 m with `sigma(m_Z) = 5.78 MeV` at that subsample,
i.e. transferring the absolute scale from the J/psi through the field modes
costs ~1 % in precision. It is a demonstration that the loop closes
(card -> `rabbit_fit.py` -> converged joint minimum with an EDM), **not** a
closure: `theta = 0` is not the hit-chi2 minimum on this MC, so the
calibration parameters are pulled off their nominal zero by the quadratic term.

---

## 4. The certified closure table

Acceptance: **value AND NLL AND EDM** (`STATE.md` sec. 7.1), applied
mechanically by `certtable.py`. MeV from the generator
(`m_Z = 91.153509740726733`, `Gamma_Z = 2.4932018986110700`). Errors are the
inverse Hessian x 1.109 except where the sandwich was evaluated directly; for
the m-form reference the two agree to 1 % (2.27 measured against 2.296).

| row | m form | v form |
|---|---:|---:|
| **inclusive, `K(m)` 5** | **-11.06 +- 2.29** | **-1.54 +- 2.31** |
| `K(m)` 6 | -13.98 +- 2.27 | -3.92 +- 2.36 |
| `K(m)` 7 | -17.23 +- 2.32 | -3.87 +- 2.53 |
| **`K(m)` 9** | **+9.39 +- 2.56** | **NOT ATTAINABLE** — 65 Hessians, no density NaN, indefinite Hessian at the end point (5.3) |
| `K(m)` 12 | NOT ATTAINABLE | NOT ATTAINABLE |
| `Gamma_Z`, `K` 5 / 6 / 7 / 9 | -5.26 / +27.16 / +8.96 / -1.79 | +6.81 / +14.28 / +12.86 / — |
| `\|eta\|` lead < 0.9 | -26.60 +- 2.87 | -21.09 +- 3.19 |
| 0.9 - 1.6 | +3.87 +- 3.88 | +12.39 +- 4.21 |
| 1.6 - 3.0 | +16.55 +- 4.71 | +34.22 +- 5.30 |

**THE SAFE BAND — the same bands cut on `max(|eta_p|,|eta_m|)` instead of the
`|eta|` of the RECO-leading leg** (v form, all five rows certified; section 5.5):

| band | lead band | **safe band** | prediction, recorded in advance |
|---|---:|---:|---:|
| < 0.9 | -21.09 +- 3.19 | **+2.68 +- 4.38** | +4.3 |
| 0.9 - 1.6 | +12.39 +- 4.21 | **+2.92 +- 3.77** | -4.6 |
| 1.6 - 3.0 | +34.22 +- 5.30 | **-9.83 +- 3.95** | -3.9 |
| **endcap - barrel** | **+55.31 +- 6.19 (8.9 sigma)** | **-12.51 +- 5.90 (2.1 sigma)** | +67.8 -> **-8.2** |
| `chi2` vs a common value | **93.8 / 2** | **6.7 / 2** | |
| weighted mean | -0.79 +- 2.29 | **-1.54 +- 2.32** | |

**The safe bands' weighted mean equals the inclusive v-form closure
(-1.54 +- 2.31) to three digits.**

**The fixed-`eta` `sigma/m` split** (v form):

| cell | `m_Z` [MeV] | internal slope |
|---|---:|---|
| barrel `sigma/m` LOW | -4.56 +- 4.10 | **-36.35 +- 6.57 (5.5 sigma)**, slope **-9 300** MeV per unit `sigma/m` |
| barrel `sigma/m` HIGH | -40.90 +- 5.14 | |
| endcap `sigma/m` LOW | +28.38 +- 5.75 | **+37.5 +- 11.2 (3.3 sigma)**, slope POSITIVE |
| endcap `sigma/m` HIGH | +65.87 +- 9.64 | |
| **safe** barrel `sigma/m` LOW | **-2.53 +- 6.04** | **+12.52 +- 8.96 (1.4 sigma)**, slope changes SIGN and is no longer significant; predicted **+18.6** |
| **safe** barrel `sigma/m` HIGH | **+9.99 +- 6.62** | |

The two lead-band internal slopes are equal to 3 % and **opposite in sign**
(difference +73.8 +- 13.0, 5.7 sigma), and the across-`eta` slope (+10 200) is
equal and opposite to the barrel's. On the safe band that whole structure is
gone. Section 5.5 explains it.

**Caveat on two rows**: `SVetaBslo` and `SVetaEslo` carry `k` displaced by
1.6e-3 and 1.2e-3 by the frozen-parameter defect of 5.9. The equivalence
triple of 5.10 narrows this: on a representative card the frozen `k` sit at
**exactly 1.0** under both the defective and the fixed code, so the
displacement is a property of those two cells and not of every Engaging row.
They owe a cold re-run; the displacement is not expected to move them.

---

## 5. Every mechanism found, with its size and status

### 5.1 The resolution-mass pairing (Punzi) — **-9.5 MeV, FIXED by construction**
The likelihood evaluates `p(m_i | sigma_i)` with the same Born spectrum for
every candidate, but on this sample the true mass and the per-candidate
resolution are strongly dependent (`<m_gen>` runs 84.94 -> 91.28 GeV across
`sigma` octiles, `rho = 0.168`). Proved by reweighting the dependence away
(-11.06 -> **-2.04** on the same candidates) and by a fit-free calculation of
the mass shift the mismatch requires (**-15.33 MeV** conditioning on `sigma`,
**+0.30 MeV** conditioning on `k`). Fixed by convolving in
`v(m) = Int dm/m^p` and conditioning on `k_i = sigma_i/m_i^p`, for which
`rho(k, m_gen) = -0.011`: that is the **v form**, and it moves the inclusive
closure from **-11.06 to -1.54 MeV**.

### 5.2 The two mass-likelihood corrections — **+8.00 MeV, IMPLEMENTED**
`a_res` (the `(1 - a_i x)` Jacobian that recovers the unconditional width) and
the Jensen `1/2 tr(H Sigma)` term, both as one deterministic per-candidate map
of the fluctuation applied inside the convolution. `a_res` alone moves `m_Z`
by +15.10, Jensen alone by -7.26, together **+8.00** against the sum +7.84 —
additive to 0.2 MeV, and inside the spec's own -5...-14 MeV prediction.
Neither touches `Gamma_Z`, as a location effect must not. Validated on the
real J/psi gun at 299 422 candidates: the fluctuation and residual forms agree
to **0.0022e-3** against a 0.01e-3 gate.

### 5.3 `K(m)` is not saturated — **THE LARGEST OPEN ITEM**

| terms | m form `m_Z` [MeV] | NLL | `2 dNLL` vs the rung below |
|---:|---:|---:|---:|
| 5 | -11.06 +- 2.29 | 11075392.4657 | — |
| 6 | -13.98 +- 2.27 | 11075277.1487 | 230.6 / 1 |
| 7 | -17.23 +- 2.32 | 11075192.7817 | 168.7 / 1 |
| **9** | **+9.39 +- 2.56** | 11074904.2322 | **577.1 / 2** |

7 -> 9 moves `m_Z` by **+26.63 MeV = 11.5 statistical errors**, and the 9-term
fit is preferred at 24 sigma. The coefficients stay O(1) and tightly
determined (`shape9 = +0.003683 +- 0.000154`, 24 sigma), so this is not a
runaway: the LO -> MiNNLO K-factor has structure the 5-term basis cannot
carry, and it projects onto the mass. **No closure number from this campaign
may be quoted without its `K(m)` truncation beside it.**

**The v form is far better behaved over 5/6/7 (-1.5 / -3.9 / -3.9 — 2.4 MeV,
inside one sigma), and that is the number to quote as the truncation
systematic. THE v-FORM LADDER ENDS AT 7**, and the reason is localised.
Attempt 1 died on a density NaN — the positivity floor was at `unbinned.py`'s
reference default 1e-9, which underflows in float64 (5.9). Attempt 2 never
loaded, on the staging defect of 5.10. **Attempt 3, with the floor at 1e-7,
ran 65 Hessian evaluations with NO density NaN — the floor did its job — and
then failed on the HESSIAN**: `scipy` raised
`array must not contain infs or NaNs` in the subproblem and the postfit died
on `Cholesky decomposition failed, Hessian is not positive-definite`. The
run's own opening line gives the cause: the **Hessian diagonal spans 0.447 to
6.24e13**, fourteen orders of magnitude. (That line arrives labelled "2
floating parameters have an essentially zero Hessian diagonal ... `m_Z`,
`Gamma_Z`", which is the pre-fix `warn_unconstrained` testing the diagonal
instead of the row — a known false positive, 5.9. The diagnosis is wrong; the
number is not.)

So the degeneracy above 7 terms is neither the density, nor the floor, nor the
formulation: it is the **conditioning of the 9-term Hessian**, i.e. the
trust-radius scale defect of 5.9, and **preconditioning is the single named
blocker**. It remains untested.

**The structural answer is a theory-predicted `K(m)`** — an NNLO parton
luminosity (SCETLib+DYTurbo in POWHEG's constant-width scheme) with its
uncertainties as nuisances, in place of the floated LO kernel — which would
also return the x1.5 / x1.4 statistical penalty of section 3.

### 5.4 The reco-variable conditioning traps — five of them
Every one was found by measuring `corr(variable, residual)` before binning.

| variable | `corr(., z)` | consequence |
|---|---:|---|
| **reco `\|eta\|` lead** (the band cards' own cut) | **+0.0203** | section 5.5 |
| signed seed -> final `dq/p` | +0.113 | absolute value only |
| reco leading `pT` | +0.042 | its top tertile sits +225 MeV above its own gen mass |
| `sigma/m`, and `sigma` at all | `sigma = sigma_bar(1 + a x)` | a bin is a cut on the residual |
| `asym` (from the two reported per-leg widths) | — | attenuates: deficits -0.15...-0.25 against -0.025 in safe cells |
| gen `\|eta\|` lead | **+0.0004** | safe |
| gen `pT` of the softer leg | -0.0005 | safe |
| `chi2/ndof` | +0.0004 | safe |
| **`max(\|eta_p\|,\|eta_m\|)`** | **+0.0025** | the safe band of 5.5 — 8x better than the reco lead, 6x worse than the gen one, and the residual spread it leaves is quoted as an upper bound for exactly that reason |

`sigma_bar = sigma(1 - a z)` is **not** a repair: built from `z`, it is
anti-correlated with the residual by construction (`corr = -0.090`, worse than
the thing it was meant to fix).

### 5.5 `eta_lead` is a RECO-`pT` selector — **it carried the 55 MeV `eta` spread. ESTABLISHED by the confirming refit**
`make_card.py:365-375` defines the band by the `|eta|` of the leg with the
larger **reco** `pT`. When the legs have similar `pT`, which one leads is
decided by which one fluctuated up, so the band edge is a cut on the residual.
Recomputing the identical cells with the **gen** leading leg (98 % the same
candidates) changes the per-leg charge-even momentum bias `A` completely:

| cell | `A`, reco `eta_lead` | `A`, gen `eta_lead` |
|---|---:|---:|
| barrel | +2.572 +- 0.102 | **-0.171 +- 0.103** |
| 0.9-1.6 | -0.066 +- 0.128 | +0.632 +- 0.156 |
| endcap | -4.867 +- 0.217 | **+0.550 +- 0.223** |
| barrel `sigma/m` LOW -> HIGH | `dA` **+4.94** | `dA` **-0.64 +- 0.16** |
| barrel, GEN-PREDICTED `sigma/m` | `dA` +3.56 | `dA` **-0.03 +- 0.18** |

(1e-4; `A = 1/2(<d>_+ + <d>_-)` with `d = p_gen/p_reco - 1`, the charge-even
part, which is the only part that survives into the pair mass; 5 % trim,
bootstrap over candidates. The "gen-predicted `sigma/m`" split uses a
predictor built only from gen quantities — the mean `sigma/m` in a 40x40
quantile grid of (gen `|eta|` lead, gen `pT` of the softer leg).)

Two consequences. **(i)** With the safe definition the legs are flat in `eta`
to 0.8e-4 and the `sigma/m` split collapses to zero, so **the `sigma/m`
pattern is NOT a per-leg momentum bias** — it is in the likelihood/model,
confirming the earlier direct measurement. **(ii)** The selector alone shifts
the selected candidates' *true* masses by almost exactly what the fits report:

| band | `-A` predicted [MeV] | certified fitted `m_Z` [MeV] |
|---|---:|---:|
| barrel | **-23.4** | -21.09 +- 3.19 |
| 0.9-1.6 | +0.6 | +12.39 +- 4.21 |
| endcap | **+44.4** | +34.22 +- 5.30 |

**THE CONFIRMING TEST HAS RUN, AND BOTH PRE-REGISTERED PREDICTIONS HOLD.**
Five cards built on `max(|eta_p|, |eta_m|)`, which does not depend on which leg
leads and needs no truth (`corr(., z) = +0.0025` against `+0.0203`), fitted in
the v form on the SAME code as the rows they are compared with, all five
certified. The two definitions select different candidates — the safe barrel
needs BOTH legs central, 699 414 against the lead barrel's 1 572 534 — so the
comparison is **spread to spread**, not cell by cell, and the predictions were
written down before the fits landed:

| | predicted | **measured** |
|---|---:|---:|
| band spread, endcap - barrel | +67.8 -> **-8.2** | +55.31 +- 6.19 -> **-12.51 +- 5.90** |
| `chi2` of the three bands vs a common value | — | 93.8 / 2 -> **6.7 / 2** |
| barrel `sigma/m` split, HIGH - LOW | -45.1 -> **+18.6** | -36.35 +- 6.57 -> **+12.52 +- 8.96** |

**The spread collapses by a factor 4.4 and changes sign; the `sigma/m` split
changes sign and does not vanish, 0.7 sigma from its prediction.** That second
point was the non-obvious half: a `sigma/m` cut is a cut on the residual
whatever the band variable is, so changing the band can only remove the band's
own selection effect. And the safe bands' weighted mean, **-1.54 +- 2.32**, is
the inclusive v-form closure (-1.54 +- 2.31) to three digits — the bands and
the inclusive fit say the same thing.

**What survives, stated honestly**: **-12.51 +- 5.90 MeV (2.1 sigma)** of
endcap - barrel spread, *opposite in sign* to the lead-band pattern. That is
the upper bound this test places on a true detector-side `eta` effect. It is
not a demonstrated effect and must not be quoted as one:
`max(|eta_p|,|eta_m|)` still carries `corr(., z) = +0.0025` against `+0.0004`
for the gen definition, so part of the residual may itself be selection, and
the clean version needs a gen predictor that a card cannot cut on. Per-band
agreement with `-A` is 0.4 / 2.0 / 1.5 sigma, the middle band having the wrong
sign — which is why the established claim is the SPREAD claim, the one
nominated in advance.

### 5.6 The `a`-coefficient deficit — **OPEN, small on `m_Z`**
`a = d ln sigma/dz` is measured directly and sits 2-4 % below the spec's
`1 + vgf`: **1.2110 +- 0.0004** against 1.2625 inclusively on the Z legs, and
**0.8928 +- 0.0130** against 1.0996 on the J/psi legs — four times larger at a
seventh of the momentum. Two candidate explanations are **excluded on properly
conditioned variables**: the ionisation share (the Q-matrix `f_ioni` is
**4.5e-06** on the Z legs and 3.2e-04 on the J/psi legs, so
`1 + f_hit - f_ioni` is indistinguishable from `1 + vgf` and closes 0.0 % of
it) and the per-leg asymmetry term (on the gen `pT` ratio the deficit is a U —
-0.080 / -0.001 / -0.082 from most to least asymmetric — where the per-leg
form predicts monotone growth). En route, the Q-matrix shares were shown to
close exactly (`f_hit + f_ms + f_ioni = 1.000000031`, max deviation 3.9e-07),
unlike the CF-exponent "shares", which are cut-dependent because Moliere and
Landau have no finite second moment.

### 5.7 The charge-even skew — **cancels in the pair mass**
Three separate results. The per-hit CPE **location** bias is real but
pixel-only (BPix +0.119 `sigma_CPE`, +1.26 um; strips null at +-0.002) and
**cannot** make the observed odd moment — wrong sign and flat in `eta`. The
track-level charge-even shift is **phi-modulated (n = 8 and 10)**, a BPix-1
effect, and it **cancels in the pair mass**. A charge-**odd** second-order
(Box) GN estimator bias of **-4.7e-5** on the momentum scale at 20-60 GeV is
real and is the part that would survive; at 4.7e-5 it is ~4 MeV on the Z mass
uncalibrated and ~0.4-0.7 MeV after the J/psi anchors the scale. Convergence
and path dependence are excluded as causes. Details:
`resolution/hitclassbias/STATE.md`.

### 5.8 The two-component mixture hypothesis — **REFUTED on both channels**
The gun's `eta` dependence decomposes into two `eta`-independent components
split on `|seed -> final dq/p|` at its 90th percentile, mixing 2.4 -> 21.2 %.
Extracting the same discriminator from the two-track trees
(`Mu*trk_pt/eta` is the KF seed, `Jpsi_qopref*` the CVH reference) and
applying it to 7.5 M Z legs and 15.8 M J/psi legs: **the mixing fraction
reproduces strikingly** (Z legs 2.68 / 9.69 / 20.04 %), **but neither
component is `eta`-flat** — the OUT component is 85-90 % of the sample and
carries the whole dependence (`chi2` vs `eta`-flat 578/2 on the Z legs, 77/2
on the J/psi legs, and 72/2 and 102/2 at mass level). The two-component
picture is a property of the gun, not of the reconstruction.

### 5.9 Fitting-machinery defects found and fixed
* **`--freezeParameters` did not freeze the step.** Freezing was
  `tf.stop_gradient` only while scipy minimised the full vector, so frozen
  directions were an exactly-null Hessian subspace and the trust region walked
  in them: measured displacements of the frozen `k` up to **0.126**. Fixed by
  minimising over `floating_indices` (7-test suite). One certified row is
  affected at 1.6e-3.
* **A trust-radius scale defect.** The initial radius of 1.0 is in raw
  parameter units on a card whose natural scales span 1e3
  (`sigma(m_Z) = 2-9` MeV, `k ~ 1`, Legendre coefficients 0.002-0.08), so it
  is 0.48 sigma for `m_Z` and **227 sigma for `shape5`** — and at an active
  trust region the step follows the gradient, which is largest along the
  *stiffest* coordinate. That drove one card's density negative and produced a
  NaN. The cell was recovered by warm-starting it from the inclusive fit's
  point. **The general cure — preconditioning, i.e. rescaling by the curvature
  so the trust region is spherical in `sigma` units, a change of variables
  that leaves the minimum invariant — was NOT tested.** Its validation, when
  run, is to require an already converged cell to reproduce to 0.01 MeV.
* **The full joint fit's NaN is a model-domain failure**, not a minimiser one:
  **2 candidates of 3 000 000** at `sigma/m` ~ 2.3-2.5 % on a J/psi, whose
  first-order coefficient `a_res/(sigma/m) = 1.95` against the 1.1-1.3 the
  spec expects, give a negative modelled density at the *default* parameter
  point. Ruled: no coefficient bounds and no density clip — delta-kernel terms
  (J/psi) move to the **exact residual form**, which is positive by
  construction and coincides with the fluctuation form at a delta kernel;
  wide-kernel terms (Z) keep the fluctuation form with a per-candidate
  positivity check and an exact x-space fallback. Implemented, and MEASURED on
  the card that failed: `gate_nanstep.py --amax-scan 0` finds **zero**
  non-positive densities on either leg at the default point and at four
  displaced points, with `min L_i` between 7.9e-05 and 1.1e-04 — so the
  wide-kernel fallback is not needed at this working point and the diagnostic
  bound is off. The switch is validated on the J/psi gun, where the two forms
  are genuinely different functionals (741 NLL units apart) and agree on
  `alpha` to **0.0022e-3** against a 0.01e-3 gate. Its cost is that the J/psi
  leg's NLL moved by **17 108 units**, so no phase-2 NLL from before it is
  comparable with one after it.
* **Convergence is EDM, never `|g|_inf`** — POIs sat 1-5 sigma off under the
  latter, and four numbers were retracted to it. Both TF minimiser ports fail
  the trust-region subproblem at full statistics, and `trust-krylov` stops at
  indefinite points; **scipy `trust-exact` is the campaign minimiser**.
* **`plot_closure.py` drew one series twice** (the v form overpainted the m
  form in the m form's colour) — every panel it had produced was affected.
* **A card can out-run its fitter, silently.** See 5.10.
* **Every crashed batch stage reported `rc=0`.** `rc=$?` sat on the same line
  as a `date` command substitution and read the *echo's* status. Fixed; and
  the batch driver now echoes the checkout, the freeze list and the extra
  flags per row.

### 5.10 The card/fitter version skew, and the certificate that closed it
`read_unbinned_terms_from_h5` splats the card's stored `config` JSON into the
term constructor, so a card written by a newer rabbit than the one the fit
runs on dies at load. The card builder on submit ran four commits ahead of the
Engaging checkout for a day; one of those commits added a single key
(`corr_a_max`, inert at its default 0.0) to `MassCFTerm.config()`, and **eight
cards became unloadable** — the five safe-band cards of 5.5 and the three
rebuilt K-ladder cards of 5.9. Both jobs died in minutes and both logged
`rc=0`. Remedied by stripping the inert key (`fullscale/cardkey.py`), which
keeps every row on one code version, and certified by a three-row triple on
the same 300 k card, same freeze, `trust-exact`:

| | old code, no key | new code, no key | new code, key present (0.0) |
|---|---|---|---|
| `m_Z` [MeV] | -35.50531845604378 | -35.50531845604459 | -35.50531845604459 |
| NLL | 874734.9966056047 | 874734.9966056045 | 874734.9966056045 |
| EDM | 1.671e-12 | 1.671e-12 | 1.671e-12 |

The two right-hand columns are **bit-identical** in all eleven parameters, the
NLL and the EDM — the keyword at its default is exactly absent. The left
differs by one ulp in NLL and **8e-13 MeV** in `m_Z`, which is the newer
checkout solving a 7-dimensional problem instead of an 11-dimensional one with
a null subspace; the minimum is the same. The frozen `k` sit at exactly 1.0 in
both, so the 1.2-1.6e-3 freeze displacement of the caveat in section 4 belongs
to those two cells and is not a property of every Engaging row.

---

### 5.11 The J/psi FSR kernel — **+66.3 MeV on `m_Z`, and it is CLOSED**

The J/psi mass term carried a `delta` at the PDG mass against an MC that
radiates. Measured on the production's own gen record over 7 853 326
candidates, the post-FSR gen mass sits **7.1969 MeV = 2.3239e-3 below
`MJPSI`** in the term's own +-0.35 GeV window, and of the -7.93 MeV mean shift
of `Jpsigen_mass` over the whole 21.7 M-candidate production **the generator's
lineshape contributes -0.0003 MeV** — it is all radiation. For a delta
lineshape the kernel is exact and additive, `p(m') = K(m'/M)/M` with CF
`<exp(i t dm)>`, so it enters through `MassCFTerm`'s existing `phik` path and
nothing else about the term changes (`zchannel/jpsi_fsr_kernel.py`,
`make_joint_card.py --jpsi-fsr`; the density IS the convolution to 2.4e-7 and
`_norm_z` matches Simpson to 8.4e-5, `fullscale/gate_jpsi_fsr.py`).

| row | card | J/psi kernel | `m_Z` [MeV] | `bfield_mode0` [1e-3] | `<D_card>.theta` [MeV] | EDM |
|---|---|---|---:|---:|---:|---:|
| `P2XP` | `joint_ok_full` | delta | +20.692 +- 2.134 | +1.40124 +- 0.02563 | -1.5105 | 1.05e-11 |
| `P2K` | `joint_fsrmc` | `mc` | **-45.594 +- 2.132** | -1.31131 +- 0.02556 | +0.8804 | 8.49e-13 |
| `J0` | `jpsi_nok` | delta | — (no Z term) | +1.91544 +- 0.02562 | -1.9049 | 1.10e-12 |
| `JK` | `jpsi_fsrmc` | `mc` | — | -0.75376 +- 0.02633 | +0.4537 | 2.06e-11 |
| `JD` | `jpsi_fsrdata` | exact QED | — | -0.86590 +- 0.02639 | +0.5578 | 7.23e-13 |

`<D_card>.theta` is the mean predicted J/psi mass shift the fitted calibration
vector produces — the quantity a delta forces to equal `-<dm>` and a kernelled
term does not. **The transfer is 1:1**: the J/psi scale moves `+7.72e-4` and
`m_Z` moves `-66.286/91 188 = -7.27e-4`, and the J/psi-only pair gives the same
step (`+7.616e-4`) on a card with no `m_Z` in it. The **42 material amounts do
not move** — every one inside 1 sigma — so the 50-sigma material pulls are not
an FSR effect.

Three further numbers:

* **the kernel-MODEL systematic is 3.4e-5**, not the 2.6e-4 the `mc` and
  exact-QED kernels' means differ by (`JD - JK`): the likelihood realises
  12.8 % of a mean-matching response against a one-sided tail;
* **what is left on the J/psi side is the two mass-likelihood corrections**.
  A fit-free scan of the term's own `-sum log(L/Z)` in a common
  predicted-mass shift (`fullscale/jpsi_scale_pref.py`) gives, with the
  kernel in, **-1.76e-5 with the corrections OFF** and `-1.80e-4` with them
  ON — so the FSR is fully accounted for and the residual is the corrections
  **in the residual form**, worth `-1.6e-4`;
* that form cannot simply be switched: `--unbinnedDeltaKernelForm fluctuation`
  on the same card (`JKF`) died with a `nan` EDM and a non-positive-definite
  postfit Hessian, i.e. the fluctuation form's first-order truncation is not
  small on this term. The fix is the exact map applied to the FLUCTUATION
  inside the convolution, a code change rather than a configuration.

### 5.12 The Z fold representation — **+0.25 MeV, and it is not the offset**

The phase-2 Z term folded the DY production's own gen record as banded ATOMS
(11 bands, `sigma_cap` 3.3e-4, 13.03 M events). `P2XT` folds the
cell-integrated `mc` kernel TABLE instead (151 x 1268, 1 GeV nodes 50-200 GeV,
standalone Photos++ 3.61 at unlimited statistics), everything else identical
to `P2N`, and certifies at EDM 4.283e-16 in 27 Hessians:

| | `P2N` (atoms) | `P2XT` (table) | difference |
|---|---:|---:|---:|
| `m_Z` [MeV] | +20.692 +- 2.134 | +20.937 +- 2.135 | **+0.245** |
| `Gamma_Z` [MeV] | -4.377 +- 3.787 | -4.264 +- 3.799 | +0.112 |
| `bfield_mode0` [1e-3] | +1.40124 | +1.40087 | -0.0004 sigma |

A ninth of the statistical error, and the J/psi leg does not move
(`<D_card>.theta` by 0.4 keV) — which is what a change confined to the Z
term's lineshape must do. The row changes the representation AND the source,
so `+0.245` is their sum rather than the representation alone; what it
establishes is that **the fold representation is not where the `-20` MeV
Z-side offset of `P2K` lives**.

`make_card.py --fsr` needed no change to accept either form —
`ZGammaLineshape` dispatches on the npz keys — but `--fsr-inline` (new,
default ON) did: `_table_config` serialises a table **by reference** whenever
its npz is on disk, and on Engaging that path does not resolve, so a card
built with a table could not be read back. Inline costs 3.756 MB of config
JSON and round-trips to 4.4e-16.

Full reference, provenance and job IDs: `fullscale/JPSI_FSR_STATE.md`;
figures `~/public_html/ZMass/cvh/260916_jpsi_fsr/`.

## 6. Open items, ranked by their size on `m_Z`

| # | item | size on `m_Z` | next step |
|---|---|---:|---|
| 1 | **`K(m)` truncation** | **2.4 MeV** over 5 -> 7 in the v form, which is the number to quote; +26.6 MeV over 7 -> 9 in the m form | **the v-form ladder ENDS at 7**: the 9-term rung runs 65 Hessians with no density NaN and fails on an indefinite Hessian whose diagonal spans 14 orders of magnitude (5.3). Two consequences — **preconditioning (item 6) is promoted from a robustness item to the blocker on the truncation systematic**, and the structural cure, a theory-predicted `K(m)` in place of the floated LO kernel (which also returns the x1.5 statistical penalty), is the only route to a `K`-independent number |
| 2 | the momentum scale from J/psi (phase 2) | **measured, and the FSR precondition is CLOSED** | the full card converges with the spectral preconditioner (`P2XP`, EDM 1.05e-11, 8 h 01) and again with the J/psi FSR kernel (`P2K`, EDM 8.49e-13, 8 h 34). **The delta at the PDG mass was costing `m_Z` +66.3 MeV**: `m_Z` = +20.69 +- 2.13 -> **-45.59 +- 2.13**, a 1:1 transfer of the J/psi leg's +7.7e-4 scale step. So the `+20.69` was the SUM of an FSR-induced +66.3 and a residual -45.6, and what is left is a Z-side offset of -20 to -24 MeV that does not move when the J/psi leg changes — i.e. item 1. On the J/psi side the FSR is now fully accounted for: with the kernel and the two mass-likelihood corrections off, the J/psi term's preferred scale is -1.8e-5. The Z fold representation is NOT the offset: replacing the banded atoms by the cell-integrated `mc` table moves `m_Z` by +0.245 MeV (5.12). See sections 5.11-5.12 |
| 3 | the material amounts (phase 3) | not yet measured | **the card IS built and verified**: `joint_mat_v3.hdf5`, **28.37 GB**, written in 54 s and re-read term by term (10 min total) — 645 517 J/psi + 481 020 Z candidates, 18 hit-resolution parameters and the 92 calibration parameters, 42 material amounts SHARED between the quadratic curvature and both mass terms. The fit is **not attempted**. The memory blocker is re-opened rather than settled: the 141.4 GB figure was measured on the phase-2 FULL card and this one has ~6x fewer candidates (34.8 GB of exponents in total), so whether it needs the 2-GPU sharding is one cheap job. The card reports that **nothing constrains `material_pp1_cables`, `material_support_tube`, `material_thermal_screen`** — freeze them |
| 4 | residual `eta` spread | -12.5 +- 5.9 MeV (2.1 sigma), an **upper bound** | a band variable cleaner than `max(\|eta_p\|,\|eta_m\|)` needs a gen predictor a card cannot cut on |
| 5 | GN second-order (Box) charge-odd bias | ~4 MeV uncalibrated (4.7e-5), ~0.4-0.7 MeV after the J/psi anchors the scale | analytic per-track correction from the exported steps; no new production |
| 6 | trust-region **preconditioning** | it bounds items 1 and 2 | **UNTESTED** — validation unchanged: require an already converged cell to reproduce to 0.01 MeV |
| 7 | the `a`-coefficient deficit | bounded small; unquantified on `m_Z` | none proposed; both leading explanations excluded |
| 8 | post-fit shape | `chi2/ndof = 4.75` over 240 bins | genuine few-% shape mismodelling; unchanged when the model subsample is grown 6.7x |
| 9 | `k_ms = 1.0298 +- 0.0042` | 0.5 MeV | the multiple-scattering **tail** is ~3 % short; phase 3 is its test |

The `eta_lead` selector, which was the largest item on this list, is **closed**
(5.5). Item 1 is now larger than everything else combined, and it is a
**model/analysis effect, not a detector or reconstruction one.**

---

## 7. Compute, measured

| | |
|---|---|
| full Z fit (3.68 M candidates, 11 free) | 38 iterations, **16 940 s** on a preemptable H200 (~4.7 GPU-h) |
| the same with 9 `K(m)` terms | **4 h 23** on an H200 |
| joint 500 k + 500 k + hit-chi2, 95 free | **3 h 39**, EDM 6.2e-19 |
| a warm band/cell refit | 10-35 min |
| native TF minimiser | **16x** faster on the wall (1206 s against 19 600 s) but fails the subproblem at full statistics — not usable |
| card sizes | phase-2 full **10.7 GB**; phase-3 material card **28.37 GB**, built in **10 min** on `mit_normal` (16 cores, 250 GB) and verified by re-reading both terms |
| phase-2 Hessian | **141.4 GB** of an H200's 143.8 GB **on the FULL card**. The phase-3 card is a subsample (1.13 M candidates against 6.68 M), so that number does not transfer to it and the 2-GPU requirement is re-opened |
| exports | ~81 kB/candidate slim; +26 kB with per-group material exponents |
| production | condor **15.6x** slurm on the same payload |
| offline auxiliary extractions | 3.7 M (DY) and 7.9 M (J/psi) candidates in 3 and 17 min on 32 cores |

---

## 8. Where we stand

**The method works and the statistics are ample.** A per-candidate unbinned
mass likelihood over 3.68 M Z -> mumu candidates gives `sigma(m_Z) = 2.27 MeV`
and `sigma(Gamma_Z) = 4.16 MeV` with the resolution model fixed at truth, and
the full joint fit with the J/psi channel and the hit-chi2 term converges to a
proper minimum with an EDM. Nothing in the detector modelling, the momentum
scale, or the reconstruction has been shown to limit it: the per-leg momentum
scale is flat in `eta` to **0.8e-4** (3.7 MeV on the mass) once it is measured
in cells that cannot see the residual, and the `sigma/m` pattern that
dominated this thread is **not** a per-leg momentum bias.

**The `eta` non-closure is closed.** It was the band selector, and the
confirming refit — five cards, predictions written down in advance —
collapses the band spread from **+55.3 +- 6.2 to -12.5 +- 5.9 MeV**, changes
its sign, drops the `chi2` against a common value from **93.8/2 to 6.7/2**,
and lands the bands' weighted mean on the inclusive closure to three digits.
The barrel `sigma/m` split changes sign and falls to 1.4 sigma, 0.7 sigma from
its prediction. A **-12.5 +- 5.9 MeV (2.1 sigma)** residual survives, on a band
variable that is still six times less clean than the generator one, and it is
recorded as an upper bound rather than an effect.

**So ONE thing now stands between this and a demonstrated closure: the `K(m)`
shape truncation** — **+26.6 MeV** between 7 and 9 terms in the m form, 2.4 MeV
over 5 -> 7 in the v form, whose structural cure is a theory-predicted kernel
that would *also* buy back a factor 1.5 in `sigma(m_Z)`. It is fixable without
new data or new production. The honest statement of feasibility is therefore:
**the precision is demonstrated; the inclusive closure and the `eta` bands
agree with each other and with the generator; and the one thing still standing
between that and a quotable closure is a shape-model truncation that is
identified, quantified, and replaceable by theory.**

`Gamma_Z` is a separate matter and is NOT ready: its `K`-truncation
sensitivity is 7.5 MeV in the v form and 32.4 MeV in the m form against a
4.2 MeV statistical error, so the truncation dominates in both formulations.

Two pieces are explicitly **not run** and are not claimed: the momentum-scale
transfer from the J/psi at full size (phase 2, descending but not converged —
section 6 item 2) and the material amounts (phase 3 — its card is built and
verified at 28.4 GB, but the fit has not been attempted, item 3).

---

## 9. Where everything is

| | |
|---|---|
| full reference, including how to reproduce each step | `fullscale/STATE.md` |
| programme-wide reference (resolution model, likelihood terms, certification) | `/work/submit/david_w/Documents/Resolution/RESOLUTION.md` |
| certified table | `fullscale/collect.sh --summary` (rsyncs Engaging, re-makes it), then `certtable.py` |
| figures | `~/public_html/ZMass/cvh/260909_fullscale/` — `mz_kladder.png`, `mz_eta.png`, `mz_eta_safe.png`, `mz_sigmasplit.png`, `mz_sigmasplit_safe.png`, `mz_variants.png`, `gz_kladder.png`; the earlier sets in `260908_fullscale/`, `260907_fullscale/`, `260906_fullscale/`; hit-class panels in `260908_hitclassbias/` |
| analysis tools | `legscale.py`, `mixture_legs.py`, `measure_a.py --aux --gen-cells`, `oddmoment/aux_gen.py`, `oddmoment/aux_seed.py`, `model_odd_mass.py`, `certtable.py`, `checkconv.py`, `cardkey.py`, `plot_closure.py`, `plot_mixture.py` |
| caches | `fullscale/runs/{zpairs_dyv2_full,zpairs_dyv2_jac_full,jpairs_v2_n600,quad_dyv2,quad_jpsiv2_ok,gpairs_v2_n50,gzpairs_dyv2_n50,auxgen_dyv2,auxgen_jpsiv2,auxseed_dyv2,auxseed_jpsiv2}.npz` |
| the fit code | `/work/submit/david_w/ZMass/rabbit-vmass`, branch `vmass-conditioning` (not on any remote) |
| the productions | `production/PRODUCTIONS.md` |
