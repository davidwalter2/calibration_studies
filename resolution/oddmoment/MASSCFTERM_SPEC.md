# Spec: the mass-CF likelihood term, its two mandatory corrections, and the v form

The unbinned mass likelihood treats the CVH fit's own per-candidate resolution
`Jpsi_sigmamass` as if it were a known constant and propagates the process
noise linearly into the mass. Both assumptions are wrong at a level that
matters, and each produces a deterministic per-candidate location shift:

1. **the self-consistent resolution** — `sigma_i` is a function of the very
   fluctuation the likelihood is measuring, and recovering the unconditional
   width carries a **measure Jacobian** that is not optional;
2. **the Jensen term** — the mass is a non-linear function of the fitted
   parameters and the CF propagates only the linear part, so `1/2 tr(H Sigma)`
   is missing.

They have opposite signs and partially cancel, so neither may be put in alone.

**Status.** Both are implemented in rabbit (`MassCFTerm`, worktree
`/work/submit/david_w/ZMass/rabbit-vmass`, branch `vmass-conditioning`) in the
**fluctuation form** of section 4, which is the form to use; `jensen_mode`
defaults to `exact`. The offline reference implementation is
`calibration_studies/resolution/oddmoment/masslik_np.py`. This file is the
derivation and the acceptance gates, not a history.

Do not re-implement from memory: the numbers in section 7 are the gates.

---

## 1. The term

```
L_i = 1/(pi s_i) Int_0^T e^{Sre_i(t)} [ pK_re(t/s_i) cos psi - pK_im(t/s_i) sin psi ] dt
psi = Sim_i(t) - (t/s_i) delta_i ,      delta_i(theta) = m_i - M(1 + alpha)
```

`Sms`, `Sio_re/im`, `Srad_re/im`, `vgf` are functions of the STANDARDIZED `t`
and describe the SHAPE; `s_i` is the absolute scale. Both corrections act on
the scale and on the location, never on the shape exponents.

---

## 2. Correction 1 — the self-consistent resolution

`Jpsi_sigmamass` is the fit's own error, assembled from the block variances at
the converged state. The process noise that dominates it scales with the
momenta, so `sigma` is a monotone function of the FITTED mass. Writing
`sigma_i = sigma_bar_i (1 + a_i x_i)` with `x_i` the truth-referenced
standardized residual,

```
z_i = x_i/(1 + a_i x_i),   E[z_i] = -a_i,   E[m_i]/m_gen unaffected
```

so a likelihood that treats `sigma_i` as known is fitting a density whose width
is correlated with the residual. The first-order bias on a mass-scale parameter
`alpha` (entering as `m_i - M(1 + alpha)`) is

```
E[alpha_hat] - alpha = -a_m F sigma_bar_eff / M ,
F = (1 + E[x^2 psi'(x)]) / E[psi'(x)] ,        psi = -(ln g)' of the model,
sigma_bar_eff = sum_i (1/sigma_bar_i) / sum_i (1/sigma_bar_i^2) .
```

`F = 2` for a Gaussian model; **`F = 1.624` measured on the J/psi gun CF
model** (`F = 2` is excluded by gate G3).

**The correction, with no truth anywhere.** `sigma_i = sigma_bar_i + a_i (m_i -
mu_i)` is the DEFINITION of `a_i`, so

```
sigma_bar_i(theta) = sigma_i - a_i delta_i(theta)
s_i(theta)         = max( sigma_bar_i(theta), FLOOR * sigma_i ),   FLOOR = 0.2
```

recovers the unconditional resolution from observed quantities alone. At the
true `alpha` it equals `sigma_bar_i` exactly, so the score has zero expectation
and the estimator is unbiased to first order.

**The measure Jacobian `(1 - a_i x)` is mandatory.** It is the Jacobian of
profiling `sigma_bar_i` out against the observed `sigma_i = sigma_bar_i(1 +
a_i x)` with a flat prior; the delta-function calculation gives
`p(Delta, sigma_obs) = p_x(x*)/sigma_bar(Delta)`. Without it the score at the
truth is `+a_i/sigma_bar_i` instead of 0 — a bias of order `a_i sigma_i`, which
is **27 MeV at the Z**, and gate G2 fails by half the correction. (Checked
analytically on a Gaussian: with the measure `E[dlnL/dDelta] = a/sigma - (x +
a x^2)/sigma -> 0`; without it, `+a/sigma`.)

`s_i(theta)` then enters in the three places `sigma_i` already appears — the
`1/(pi s_i)` prefactor, `tgi = TG/s_i`, and the FSR-kernel CF argument
`phi_K(TG/s_i)` — and `-ln s_i(theta)` in `ln L_i` becomes alpha-dependent and
**must not be dropped**. (In the fluctuation form of section 4 all of this
collapses into one multiplicative CF factor and `s_i` is constant again.)

### `a_i`

```
a_i = (1 + f_hit,i - f_ioni,i) * sigma_i / m_gen,i
```

* `f_hit,i` = the cached `vgf` = `cfmass_vgf` = `(sigma_m^2 - resinfcov)/sigma_m^2`
* `f_ioni,i` = `sum_{parmtype==11} resinfvarv / sigma_m^2`; **1.1e-3** on the
  gun, i.e. it moves `a_i` by 0.1 % and may be dropped.

Derived from `sigma_m^2 = A m^4 + B m^2 + C` (the hit contribution to
`sigma_rel` grows as `p`, the MS one is flat, the ionization one falls as
`1/p`), giving `d ln sigma_m/d ln m = 2 f_hit + f_ms = 1 + f_hit - f_ioni`.

**`a` is directly measurable and the closed form is 2-4 % high.** `a =
d ln sigma/dx` is a regression slope: measured **1.2110 +- 0.0004** inclusively
on the Z legs against `1 + vgf = 1.2625`. Two candidate refinements were tested
and both FAIL — the per-leg asymmetry term moves `a` 1.7-4.3x further from the
measurement, and `e = f_hit - f_ioni` closes a quarter of the gap
(`f_ioni` from `-S''(0)` of the stored CF is 0.00525, 2.5 % of `f_hit`, where
the deficit needs 10-20 %). The residual 2-4 % is a known limitation.

**No new C++ export is required**: `reseigidx`, `resinfvarv`, `resinfcov`,
`runtree/parmtype` and `cfmass_vgf` all survive the slim production format
(`exportStepRecords=False`).

---

## 3. Correction 2 — the Jensen (second-order) map

The CF propagates the block fluctuations LINEARLY (`resinfv` holds the
mass-projected dof weights), but `m` is a non-linear function of the fitted
parameters. The missing quadratic term has a non-zero mean: for
`m ~ (kappa1 kappa2)^{-1/2}`,

```
1/2 tr(H Sigma)/m = (3 A + B)/8 ,
A = sigma_rel1^2 + sigma_rel2^2 ,   B = 2 rho sigma_rel1 sigma_rel2 .
```

Measured on the gun with GEN leg kinematics: **`rho` = -0.004 … -0.009** (the
legs are uncorrelated, `B = 0`) and the angular share of the mass variance
**`f_ang` = 0.10-0.14**, so the truth-free cache-only closed form

```
s_i^Jensen = 1.5 (sigma_m,i / m_i)^2          (use 1.5 - f_ang if f_ang is per-candidate)
```

is **4.6 % high** and is the recommended default. It is additive with
correction 1: measured -0.1234e-3 on top of the naive fit and -0.1237e-3 on top
of the corrected one, identical to 3e-7.

### The map must be INVERTED, not applied as a mean shift

Treating `1/2 tr(H Sigma)` as a deterministic location shift
(`delta_i -> delta_i - M s_i^Jensen`) assumes the MLE responds to it with
weight 1. It does not: the missing term is a QUADRATIC form and the measured
response is **`F_J` = 0.73 +- 0.14** at J/psi `sigma_m/m` and **0.56 +- 0.22**
at Z-like `sigma_m/m = 1.85 %`, agreeing with the analytic
`(0.5 x 0.606 + 1.0 x 0.803)/1.5 = 0.729`. The mean-shift form therefore
OVER-corrects by **27 %** at the J/psi and **~44 %** at the Z.

**The exact second-order map is the one to implement.** With `u` the linear
relative fluctuation (what the CF models) and `s^2 = Var(u) = (sigma_m/m)^2`,
the conditional expectation of the quadratic form given `u` is, for
uncorrelated equal legs,

```
m_hat/m - 1 = u + u^2 + 1/2 s^2                (mean 1.5 s^2, as it must be)
```

so, with `r = delta_i(alpha)/m`,

```
u_i        = 1/2 ( sqrt( max(1 + 4(r - s_i^2/2), floor) ) - 1 )
delta_i^eff = m u_i ,      L_i -> L_i * du/dr = L_i / (1 + 2 u_i) .
```

**The Jacobian `1/(1 + 2u)` must be kept.** No response factor is needed and
none is assumed. Measured difference between the two forms on real candidates
(`alpha` in 1e-3):

| | mean-shift form | **exact map** | difference |
|---|---|---|---|
| gun | +0.0174 +- 0.0167 | **+0.0512 +- 0.0167** | +0.034 |
| B -> J/psi X v3 | -0.0326 +- 0.0253 | **+0.0059 +- 0.0254** | +0.039 |
| gun, `sigma_m/m` = 0.0083 | +0.0171 +- 0.0335 | -0.0036 +- 0.0336 | -0.021 |
| gun, `sigma_m/m` = 0.0185 (Z-like) | -0.1499 +- 0.0708 | **+0.0484 +- 0.0709** | **+0.198** |

3-4x above the 0.01e-3 target at the J/psi and ~+0.11e-3 (~10 MeV) at Z-like
`sigma_m/m`. The exact map is also the only one that is sigma-INDEPENDENT
(-0.004 vs +0.048 across a factor 2.2 in `sigma_m/m`, 0.7 sigma apart, against
2.2 sigma for the mean-shift form) and it gives the best NLL (v3: -255186.17
against -255128.37 for the shift and -255177.43 naive).

`delta_i(theta)` is used BOTH in the phase `psi = Sim - tgi delta` AND inside
`sigma_bar_i(theta) = sigma_i - a_i delta_i(theta)`.

---

## 4. The implementation form: ONE map of the FLUCTUATION

Both corrections are expansions in the RESOLUTION fluctuation. Feeding them
`delta_i(theta) = m_i - M(theta)` — the deviation from the reference mass — is
equivalent at the J/psi, where every gate was measured, **but not at the Z**:
there the window is +-30 GeV = +-27 sigma and what sits out there is FSR and
the Breit-Wigner tail, not resolution. Measured on 3 682 662 DY candidates, the
exact map fed the full `delta` moves the residual by a **median 57.7 MeV and up
to 10.7 GeV**, against the **20.6 MeV** mean shift it exists to apply, and the
fit runs away to `Gamma_Z = -421 MeV` with O(1) `K(m)` coefficients
compensating. Clipping bounds the damage but is a device, not a treatment.

**Put both inside the convolution, as one deterministic per-candidate map of
the fluctuation `x` (whose CF is the exported `e^{S_i(tau)}`):**

```
m_i = m_true + u_i(x) ,     u_i(x) = sigma_i x + c_i x^2 + d_i
c_i = -a_i sigma_i + sigma_i^2/m_i ,        d_i = m_i s_i^2/2
L_i(theta) = Int K_theta(m_i - u_i(x)) p_i(x) dx ,   p_i(x) = p_x(x) (1 - a_i x)
```

`-a_i sigma_i` in `c_i` is the self-consistent width; `+sigma_i^2/m_i` and
`d_i` are the exact map's `u^2` and `s^2/2` terms; `(1 - a_i x)` is the measure
Jacobian of section 2. With `a_i = (1 + vgf_i) sigma_i/m_i` the two quadratic
coefficients very nearly cancel, `c_i = -vgf_i sigma_i^2/m_i` — which is why
the naive Z bias is -15 … -27 MeV rather than the full -35.

In Fourier space, to first order in `a_i` and `c_i`, this is ONE multiplicative
factor on the resolution CF on the term's own `tau` grid, plus a shift `d_i` of
the residual:

```
Phi_i(tau)/phi_i(tau) = 1 + i a_i (S'(tau) - S'(0))
                          - i (c_i/sigma_i) tau (S''(tau) + S'(tau)^2)
```

using `E[x e^{i tau x}] = -i phi'`, `E[x^2 e^{i tau x}] = -phi''` and
`phi'' = (S'' + S'^2) phi`. The `-S'(0)` normalises `Phi_i(0) = 1`;
`c_i/sigma_i = -a_i + sigma_i/m_i`, so no division at evaluation time. `S'` and
`S''` are fixed cubic-spline differentiation matrices from the stored `tau`
grid onto the integration grid (the Gaussian family analytically).
**No clip, no log-Jacobian, no dependence on `delta_i`, and `sigma_i` is
constant again, so the cheap static (alpha-independent) cache path returns.**

Exact / approximated: EXACT for the first moment,
`E[Delta_i] = Var(x)(c_i - a_i sigma_i) + d_i`. Neglected
`O(a^2, ac, c^2) ~ 1e-4` of a correction that is itself ~1e-2 of the width; the
second moment loses `2(c_i/sigma_i)^2 ~ 2e-4` relative, under 0.1 MeV on
`Gamma_Z`. The `O(c^2)` (`x^4` <-> `phi''''`) term is NOT needed — it
contributes nothing to the mean, which is the entire content of both
corrections.

`corr_form` is consumed at CONSTRUCTION (`c_i`, `d_i`, `a_i` are baked in), so
a variant ladder must go through `term.set_corrections(...)`, not through
flipping `self_consistent_sigma` / `jensen_mode` in place.

---

## 5. The v form: condition on a mass-independent width

A third, independent defect of the same family shows up only at the Z. The
likelihood evaluates `p(m_i | sigma_i)` with the same Born spectrum for every
candidate, but the true mass and the per-candidate resolution are strongly
dependent: `<m_gen>` runs **84.94 -> 91.28 GeV across `sigma` octiles**,
`rho(sigma, m_gen) = +0.168`. This is the resolution-mass pairing (Punzi)
effect and it is worth **-9.5 MeV** on `m_Z`. Proved by reweighting the
dependence away — the same candidates and the same model, reweighted by
`p(m_gen)/p(m_gen|class)` so that the true mass is independent of the `sigma`
class and nothing else changed: **-11.06 -> -2.04 MeV**. (An earlier -0.12 from
that arm is RETRACTED: it was not converged, its POI sitting 0.97 sigma from
the minimum.) The assembly toy `m_gen_i + sigma_i z_j`, residuals shuffled
inside 20 `sigma/m` classes, gives -3.75, i.e. ~70 % of the bias lives in the
residual-level part, which is what the v form fixes.

**The fix is a change of variable.** Write the smearing as `m_i = m' + k_i m'^{1+f} x` and substitute

```
v(m) = Int dm/m^{1+f} = m^{-f}/(-f)          (v = ln m when f = 0)
```

which gives `v_i = v(m') + k_i x` to first order — a **fixed-width convolution
in `v`**, with a width `k_i` that is independent of the true mass. The FFT
still applies on a grid uniform in `v`; the Born density carries the Jacobian
`p_v(v) = p(m(v)) m(v)^{1+f}`; the conditioning is on `k_i`, for which
`p(m'|k_i) = p(m')`. The exponent is a single common

```
p = 1 + f = 1.264 = 1 + <vgf>        (measured)
```

Then `rho(k, m_gen) = -0.011` against `rho(sigma, m_gen) = +0.168`, a factor
15, and the pairing bias is gone by construction.

Certified fits (value AND NLL AND EDM), MeV from the generator; errors are the
inverse Hessian x **1.109** (the MiNNLO sandwich ratio) except the m-form
`K` = 5 reference, whose sandwich was evaluated directly (+- 2.27; the x1.109
approximation gives +- 2.29, agreeing to 1 %):

| `K(m)` terms | m form `m_Z` | **v form `m_Z`** |
|---:|---:|---:|
| **5** | **-11.06 +- 2.29** | **-1.54 +- 2.31** |
| 6 | -13.98 +- 2.27 | -3.92 +- 2.36 |
| 7 | -17.23 +- 2.32 | -3.87 +- 2.53 |
| 9 | +9.39 +- 2.56 | NOT ATTAINABLE (indefinite Hessian) |
| truncation spread | 26.6 MeV over 5 -> 9, not monotone | **2.4 MeV over 5 -> 7** |
| `Gamma_Z` `K`-truncation sensitivity | 32.4 MeV | 7.5 MeV |

**The -11 MeV is a property of the m parameterisation, not of the data** — it
does not survive the change of variable. Two independent converged runs of the
v card agree to 0.005 MeV. `m_Z` in the v form can be quoted at its statistical
precision (2.3 MeV at 3.68 M candidates); `Gamma_Z` cannot yet, because its
`K(m)` truncation sensitivity (7.5 MeV) still dominates its statistical error
(4.2 MeV).

---

## 6. The inputs, and the C++ exports

Neither correction needs truth.

**Correction 1** needs only `cfmass_vgf` and `Jpsi_sigmamass`, both already in
the slim output — nothing to add.

**The Jensen term** needs `A = sigma_rel1^2 + sigma_rel2^2` and
`B = 2 rho sigma_rel1 sigma_rel2`. The two-track maker now exports the full
symmetric 6x6 reference-momentum covariance as **`Jpsi_covrefmom`** (upper
triangle, row-major, 21 floats, +84 B/candidate = +0.10 % of the 80.7 kB slim
record; `ResidualGlobalCorrectionMakerTwoTrackG4e.cc`), whose state indices 0-2
and 3-5 are the two legs' `(q/p, lambda, phi)` at the reference and whose
leg <-> plus/minus map is `idxplus`/`idxminus`. That gives `A` and `B` exactly,
`f_ang` for free as `1 - (J_kappa C J_kappa^T)/sigma_m^2` using the exported
`Jpsi_jacMass`, the kappa-angle cross terms of `1/2 tr(H Sigma)`, and the
closure test `J Sigma J^T = Jpsi_sigmamass^2`.

Two further options were considered and are NOT required by either correction:
per-leg `resinfvarv` + `sigmaqop2` (~550 B, gives the per-leg family split),
and porting the single-track hit-block registration into the two-track maker
(~1.9 kB, gives per-leg `f_hit` — and it would put hit blocks into `resinfcov`
and so change the meaning of `cfmass_vgf`, breaking every existing cache).

**The single-track maker needs nothing**: it already exports the full 5x5
`refCov`, from which `sigma_rel = sqrt(refCov[0]) * p` and hence the
track-level `a_i` follow directly (`oddmoment/track_truthfree.py`).

---

## 7. Acceptance gates (all MEASURED)

`alpha` in 1e-3 throughout; statistical error 0.0167 on the gun, 0.0254 on
B -> J/psi X v3. Sources: `oddmoment/out/{toyfits,realfits,jensenfits}.txt`.

### Correction 1

| gate | requirement | measured |
|---|---|---|
| G1 | `a_i = 0` reproduces the uncorrected term | the offline numpy objective reproduces the published `cf_masslik_fit.py` single-`r` alpha to 1e-4: gun **-0.00470** vs -0.00459, v3 **-0.04637** vs -0.0463 |
| G2 | on a toy drawn from the REAL per-candidate CF models with a constant `a` injected, the corrected fit returns `alpha(a) = alpha(0)` | `a` = 0.011: **-0.09300** vs -0.09187 (residual -0.0011 = 0.07 sigma, **99.2 %** removed); `a` = 0.050: **-0.09665** vs -0.09187 (**99.3 %** removed) |
| G3 | the toy's NAIVE alpha reproduces `-a F sigma_bar_eff/M` | slope **-13.4e-3 per unit `a`** (linear to 0.4 % between `a` = 0.011 and 0.05) against **-14.3e-3** predicted with `F = 1.624`; `F = 2` would give -17.6e-3 and is EXCLUDED |
| G4 | the corrected alpha agrees with the truth-referenced `sigma_bar` fit | gun: truth-free **+0.14102**, true `sigma_bar` **+0.13160** (0.009 apart = 0.4 sigma), naive -0.00470 |
| G5 | the toy fit with the TRUE `sigma_bar` is bit-identical to the `a = 0` toy | **exactly** (-0.09187, NLL -590499.5705 in all three arms) |

### The Jensen term

| gate | measured |
|---|---|
| J1 | additivity: the term moves alpha by **-0.1234** from naive and **-0.1237** from corrected |
| J2 | gun ladder: naive -0.0047 -> corrected +0.1410 -> mean-shift +0.0174 -> **exact +0.0512 +- 0.0167** |
| J3 | v3 ladder: naive -0.0464 -> corrected +0.0810 -> mean-shift -0.0326 -> **exact +0.0059 +- 0.0254** |
| J4 | differential: in truth-free `sigma_bar` quintiles the ratio alpha/prediction is CONSTANT at **0.94 +- 0.11** over a factor 8 in the predicted size |
| J5 | the response factor: a toy injecting `m = m_gen e^{s x}` on the real per-candidate CFs shifts alpha by +0.0239 against +0.0413 for response 1, i.e. **`F` = 0.58 measured against 0.606 predicted** (`E[x^2 psi']/I`) |

### The fluctuation form

Unit tests (`rabbit-vmass/tests/test_fluctuation.py`, 7/7): no correction is
bit-identical in either form; the density sampled on a mass grid normalises to
1.0000000000 and its mean equals the closed form to 1e-10 in all three arms; at
a DELTA kernel the two forms' corrections agree to 0.0007e-3; over a +-27 sigma
window the residual form moves the residual by a median 1979 MeV / max 9.1 GeV
while the fluctuation form's shift stays at `d_i` (median 7.3 MeV, max
15.7 MeV) and its CF factor weighted by `e^{Re S}` stays under 0.03; gradient
vs finite difference 3e-10; `c_i = -vgf sigma^2/m` to 7e-18; `set_corrections`
reproduces purpose-built terms bit-identically.

**Real J/psi gun, 299 422 candidates, five rabbit terms sharing them** (so the
shift carries no statistical error):

| | alpha [1e-3] | shift [1e-3] | this spec, offline |
|---|---:|---:|---:|
| uncorrected | -0.00825 | — | -0.0047 |
| correction 1 only, residual form | +0.13705 | **+0.14530** | +0.1457 |
| correction 1 only, fluctuation form | +0.13792 | **+0.14617** | +0.1457 |
| both, residual form | +0.05082 | **+0.05907** | +0.0559 |
| both, fluctuation form | +0.04858 | **+0.05683** | +0.0559 |

`|fluctuation - residual|` = **0.00087** (correction 1 alone) and **0.00224**
(both), against a 0.01 gate; both forms sit within 0.003 of the offline numpy
numbers.

---

## 8. Size, and why both terms must be in

Both scale as `sigma_rel^2` — the resolution artefact as
`-(1 + f_hit) F sigma_rel^2` and the Jensen term as `+(1.5 - f_ang)
sigma_rel^2` — so they partially CANCEL, with a ratio fixed by
`(1.5 - f_ang)/[(1 + f_hit) F]`: **0.85 at the J/psi** (measured 0.124/0.146)
and 0.52-0.86 at the Z. The cancellation is an accident of two numbers that
both move with the sample.

**At the J/psi** (`a_m` = 0.0107): predicted bias -0.155e-3, measured on a toy
built from the real per-candidate models **-0.148e-3**; on the real gun the
correction moves alpha by **+0.146e-3** and on B -> J/psi X v3 by
**+0.127e-3**. The naive gun alpha of -0.005 +- 0.017e-3 is therefore NOT an
unbiased scale: it is a -0.14e-3 artefact cancelling a +0.14e-3 genuine
residual.

**At the Z** (`a_m` = 0.016-0.020, `sigma_m/m` = 0.013-0.015) the a-priori
estimate was -0.26 … -0.48e-3 per term, i.e. -24 … -43 MeV on `m_Z` alone
(25-45x the 1e-5 target), with a predicted net of -5 … -14 MeV
(`oddmoment/zproject.py`). **Measured at full scale** on 3 682 662 DY
candidates: correction 1 moves `m_Z` by **+15.10 MeV**, the Jensen term by
**-7.26 MeV**, and the two together by **+8.00 MeV** against a sum of +7.84 —
additive to 0.2 MeV, and inside the predicted net. Neither touches `Gamma_Z`,
as a pure location effect must not.

---

## 9. The same defect at track level

For a single track, `sigma = sigma_bar (1 + a q x)` with

```
a_i = sigma_rel,i * d ln sigma/d ln kappa ~= sigma_i * p_fit,i * (1 - vgf_i)
```

and the truth-free inversion is `x_i = z_i / (1 - a_i q_i z_i)`. Validated in
`oddmoment/track_truthfree.py`: it reproduces the truth-referenced `sigma_bar`
answer to **2e-4 pull units** at every statistic, on both muon-gun momentum
ranges.

---

## 10. Known limitations

* **`a_i` is 2-4 % high** in its closed form against the directly measured
  regression slope (1.2625 vs 1.2110 +- 0.0004 on the Z legs); the two
  candidate refinements tested both fail (section 2).
* **`s_i^Jensen = 1.5 (sigma_m/m)^2` is 4.6 % high** while `f_ang` is taken
  from MC (0.10-0.14) rather than per candidate; `Jpsi_covrefmom` now makes the
  per-candidate value available and it has not yet been used.
* **`_norm_z` does not see the correction.** The density shifts by `d_i`
  (~7 MeV) against a window edge 30 GeV away and `d_i` is a candidate CONSTANT,
  so it cannot bias `m_Z`; it is a documented sub-MeV item alongside the
  stored-class-sigma approximation (measured at 0.016 MeV).
* **The `bfield_mode0` injection closure is a property of the form, and the
  fluctuation form fixes it.** In the RESIDUAL form mode 0 closed only to
  **+12.14 % of the injection** while modes 1 and 2 closed to 3e-4: what did
  not translate was the residual form's Jensen map, whose argument
  `r = delta/m` carries the OBSERVED mass, so the map does not commute with a
  shift of the observed masses (turning that one correction off moves the
  defect from 12.14 % to 0.34 %; `a_res` off leaves 11.78 %, so `a_res` is not
  it). Re-measured in the FLUCTUATION form on the joint smoke card, where the
  residual is linear in theta again and the Jensen map contributes only a
  per-candidate constant `d_i` and a `delta`-free CF factor:
  **mode 0 +0.0023 % of the injection (5300x better), modes 1 and 2 at 1e-11
  (four orders better)**. The residual **2.3e-7** is the second-order
  dependence of `a_i`, `c_i`, `d_i` themselves on the shifted masses — 1.6e-3
  of `bfield_mode0`'s own statistical error, and not the minimiser (the Newton
  step still implied by the residual gradient is 1.8e-12).
