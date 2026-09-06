# Spec: self-consistent-resolution correction for rabbit's `MassCFTerm`

**Status**: the defect is measured and the correction is validated offline
(`calibration_studies/resolution/oddmoment/`, NOTES.md 2026-09-05). This is the
change the rabbit `unbinned-mass-term` branch needs. It is additive: with the
new input set to zero the term must be bit-identical to today's.

Do not implement this from memory — the numbers below are the acceptance gates.

## 1. The defect, in one paragraph

`Jpsi_sigmamass` (and any per-candidate resolution the CVH fit exports) is the
fit's OWN error, assembled from the block variances at the CONVERGED state. The
process noise that dominates it scales with the momenta, so sigma is a monotone
function of the FITTED mass — of the very fluctuation the likelihood is
measuring. Writing sigma_i = sigma_bar_i (1 + a_i x_i) with x_i the
truth-referenced standardized residual,

    z_i = x_i/(1 + a_i x_i),   E[z_i] = -a_i,   E[m_i]/m_gen unaffected,

so a likelihood that treats sigma_i as a known constant is fitting a density
whose width is correlated with the residual. The resulting bias on a mass-scale
parameter alpha (entering as m_i - M(1+alpha)) is, to first order,

    E[alpha_hat] - alpha = -a_m F sigma_bar_eff / M ,
    F = (1 + E[x^2 psi'(x)]) / E[psi'(x)] ,   psi = -(ln g)' of the model,
    sigma_bar_eff = sum_i (1/sigma_bar_i) / sum_i (1/sigma_bar_i^2) .

F = 2 for a Gaussian model; **F = 1.624 measured on the J/psi gun CF model**.

## 2. The correction (no truth anywhere)

sigma_i = sigma_bar_i + a_i (m_i - mu_i) is the DEFINITION of a_i, so

    sigma_bar_i(theta) = sigma_i - a_i * delta_i(theta),
    delta_i(theta) = m_i - M(1 + alpha)          [the existing `delta`]

recovers the unconditional resolution from observed quantities alone. Use
sigma_bar_i(theta) as the model's absolute scale. At the true alpha it equals
sigma_bar_i exactly, so the score has zero expectation and the estimator is
unbiased to first order.

## 3. What changes in `MassCFTerm`

Today the density is

    L_i = 1/(pi s_i) Int_0^T e^{Sre_i(t)} [ pK_re(t/s_i) cos psi - pK_im(t/s_i) sin psi ] dt
    psi = Sim_i(t) - (t/s_i) delta_i ,      s_i = sigma_i   (alpha-INDEPENDENT)

The change is exactly: **`s_i` becomes alpha-dependent.**

```
s_i(theta) = max( sigma_i - a_i * delta_i(theta), FLOOR * sigma_i )   FLOOR = 0.2
```

and `s_i` then enters in the three places it already appears:
1. the `1/(pi s_i)` prefactor,
2. `tgi = TG / s_i`,
3. the FSR-kernel CF argument `phi_K(TG / s_i)`.

The exponents `Sms`, `Sio_re/im`, `Srad_re/im`, `vgf` are functions of the
STANDARDIZED t and are **left untouched** — they describe the shape; only the
absolute scale is corrected.

The `-ln s_i(theta)` in `ln L_i` is now alpha-dependent. With autodiff nothing
extra is needed; it is the "log-Jacobian" term and it must NOT be dropped.

### Cost / caching
The existing alpha-cache (`set_k_cache`: `A = e^{Sre} Re phiK`,
`B = e^{Sre} Im phiK`, `Sim`, `tgi`) is no longer alpha-independent, because
`phiK` and `tgi` move with `s_i(alpha)`. Two options:
* keep `e^{Sre}` and `Sim` cached (they do NOT depend on alpha) and recompute
  `tgi` and the two `phiK` interpolations per alpha. This is what the offline
  numpy implementation does (`oddmoment/masslik_np.py`, `_kcache` + the
  `corrected` branch); it costs ~1.6x per NLL evaluation.
* `phi_K` is tabulated on a FIXED absolute-t grid, so the interpolation is a
  differentiable gather+lerp (`tfp.math.interp_regular_1d_grid`, or a manual
  `tf.gather` on `floor(t/dt)` with a linear weight). No retabulation per alpha.

### The new input
One float64 per candidate:

```
a_i = (1 + f_hit,i - f_ioni,i) * sigma_i / m_gen,i
```

* `f_hit,i` = the cached `vgf` = `cfmass_vgf` = `(sigma_m^2 - resinfcov)/sigma_m^2`
* `f_ioni,i` = `sum_{parmtype==11} resinfvarv / sigma_m^2`; it is **1.1e-3** on
  the gun and may be dropped (it moves a_i by 0.1 %)
* **All of these survive the slim production format** (`exportStepRecords=False`):
  `reseigidx`, `resinfvarv`, `resinfcov`, `runtree/parmtype` and `cfmass_vgf`
  are all in the keep list. **No new C++ export is required.**

The closed form is derived from sigma_m^2 = A m^4 + B m^2 + C (the hit
contribution to sigma_rel grows as p, the MS one is flat, the ionization one
falls as 1/p), giving d ln sigma_m/d ln m = 2 f_hit + f_ms = 1 + f_hit - f_ioni.

## 4. Acceptance gates

All four are MEASURED (2026-09-05, `oddmoment/out/{toyfits,realfits}.txt`;
alpha in 1e-3, statistical error 0.0167 on the gun, 0.0254 on v3):

| gate | requirement | measured |
|---|---|---|
| G1 | `a_i = 0` reproduces today's term BIT-IDENTICALLY | the offline numpy objective reproduces the published `cf_masslik_fit.py` single-r alpha to 1e-4: gun **-0.00470** vs -0.00459, v3 **-0.04637** vs -0.0463 |
| G2 | on the toy (drawn from the REAL per-candidate CF models with a constant a injected) the corrected fit returns alpha(a) = alpha(0) | a = 0.011: **-0.09300** vs -0.09187 (residual -0.0011, 0.07 sigma, **99.2 %** of the bias removed); a = 0.050: **-0.09665** vs -0.09187 (residual -0.0048, **99.3 %** removed) |
| G3 | the toy's NAIVE alpha reproduces `-a F sigma_bar_eff/M` | slope measured **-13.4 e-3 per unit a** (linear to 0.4 % between a = 0.011 and 0.05) against **-14.3 e-3** predicted with F = 1.624; F = 2 would give -17.6 e-3 and is excluded |
| G4 | the corrected alpha agrees with the truth-referenced `sigma_bar` fit | gun: truth-free **+0.14102**, true sigma_bar **+0.13160** (0.009 apart, 0.4 sigma), naive -0.00470 |
| G5 | the toy fit with the TRUE sigma_bar is bit-identical to the a = 0 toy | **exactly** (-0.09187 / NLL -590499.5705 in all three arms) |

## 4b. A SECOND, INDEPENDENT per-candidate deterministic shift: the Jensen term

The CF propagates the block fluctuations LINEARLY (`resinfv` holds the
mass-projected dof weights), but m is a nonlinear function of the fitted
parameters.  The missing quadratic term has a nonzero mean:

    1/2 tr(H Sigma)/m = (3 A + B)/8 ,
    A = sigma_rel1^2 + sigma_rel2^2 ,  B = 2 rho sigma_rel1 sigma_rel2 ,

for m ~ (kappa1 kappa2)^{-1/2}.  Measured on the gun with GEN leg kinematics:
**rho = -0.004...-0.009 (legs uncorrelated, B = 0)** and the angular share of
the mass variance **f_ang = 0.10-0.14**, so the closed form

    **s_i^Jensen = 1.5 (sigma_m,i / m_i)^2**       (truth-free, cache-only)

is **4.6 % high** and is the recommended default (use `1.5 - f_ang` if f_ang is
ever measured per candidate).  It is a SECOND deterministic per-candidate
location shift and it is ADDITIVE with the resolution correction of s2-4
(measured: -0.1234e-3 on top of the naive fit, -0.1237e-3 on top of the
corrected one -- identical to 3e-7).

### THE EXACT FORM IS THE ONE TO IMPLEMENT (measured 2026-09-05 IV)

There are two ways to put the term in, and they are NOT equivalent.

**(a) mean shift (do NOT use as the default).**  Treat 1/2 tr(H Sigma) as a
deterministic location shift:

    delta_i(theta) = m_i - M(1 + alpha) - M s_i^Jensen .

This assumes the MLE responds to it with weight 1.  It does not: the missing
term is a QUADRATIC form, and the measured response is
**F_J = 0.73 +- 0.14 at J/psi sigma_m/m and 0.56 +- 0.22 at Z-like
sigma_m/m = 1.85 %**, in agreement with the analytic
(0.5 x 0.606 + 1.0 x 0.803)/1.5 = 0.729.  Using form (a) therefore
OVER-corrects by 27 % at the J/psi and ~44 % at the Z.

**(b) exact to second order (THE DEFAULT).**  Invert the second-order map per
candidate.  With u the linear relative fluctuation (what the CF models) and
s^2 = Var(u) = (sigma_m/m)^2, the conditional expectation of the quadratic form
given u is, for uncorrelated equal legs and m ~ (k1 k2)^{-1/2},

    m_hat/m - 1 = u + u^2 + 1/2 s^2          (mean 1.5 s^2, as it must be)

so, with r = delta_i(alpha)/m,

    u_i = 1/2 ( sqrt( max(1 + 4(r - s_i^2/2), floor) ) - 1 )
    delta_i^eff = m u_i ,     L_i -> L_i * du/dr = L_i / (1 + 2 u_i) .

**The Jacobian 1/(1+2u) must be kept.**  No response factor is needed and none
is assumed.  `masslik_np.py --jensen-mode exact` is the reference
implementation (`--jensen-mode shift` reproduces form (a)).

Measured difference between the two, on the real candidates:
| | form (a) shift | form (b) EXACT | (b) - (a) |
|---|---|---|---|
| gun | +0.0174 +- 0.0167 | **+0.0512 +- 0.0167** | +0.034e-3 |
| v3 | -0.0326 +- 0.0253 | **+0.0059 +- 0.0254** | +0.039e-3 |
| gun, sigma_m/m = 0.0083 | +0.0171 +- 0.0335 | -0.0036 +- 0.0336 | -0.021e-3 |
| gun, sigma_m/m = 0.0185 (Z-like) | -0.1499 +- 0.0708 | **+0.0484 +- 0.0709** | **+0.198e-3** |
The difference is **3-4x above 0.01e-3 at the J/psi and ~+0.11e-3 (~10 MeV) at
Z-like sigma_m/m** -- far above 1 MeV, so form (a) is not acceptable at the Z.
Form (b) is also the only one that is sigma-INDEPENDENT (-0.004 vs +0.048
across a factor 2.2 in sigma_m/m, 0.7 sigma apart, against 2.2 sigma for form
(a)) and it gives the best NLL (v3: -255186.17 against -255128.37 for (a),
-255177.43 naive).

Implementation hook: whichever form, `delta_i(theta)` is used BOTH in the phase
`psi = Sim - tgi * delta` AND inside
`sigma_bar_i(theta) = sigma_i - a_i delta_i(theta)`.

Gates (measured, `oddmoment/out/jensenfits.txt`):
| gate | measured |
|---|---|
| J1 | additivity: the shift moves alpha by -0.1234e-3 from naive and -0.1237e-3 from corrected |
| J2 | gun alpha: naive -0.0047 -> corrected +0.1410 -> +Jensen(shift) +0.0174 -> **+Jensen(EXACT) +0.0512 +- 0.0167** |
| J3 | v3 alpha: naive -0.0464 -> corrected +0.0810 -> +Jensen(shift) -0.0326 -> **+Jensen(EXACT) +0.0059 +- 0.0254** |
| J5 | the response factor: a toy injecting the pure exponentiation m = m_gen e^{s x} on the real per-candidate CFs shifts alpha by +0.0239e-3 against +0.0413e-3 for response 1, i.e. **F = 0.58 measured against 0.606 predicted (E[x^2 psi']/I)** |
| J4 | differential: in truth-free `sigma_bar` quintiles the ratio alpha/prediction is CONSTANT at **0.94 +- 0.11** over a factor 8 in the predicted size |

## 4c. C++ EXPORT SPEC: what the two-track maker must add

Neither correction needs truth.  The RESOLUTION correction (s2-4) needs only
`cfmass_vgf` and `Jpsi_sigmamass`, both already in the slim output -- **nothing
to add**.  The JENSEN term needs A = sigma_rel1^2 + sigma_rel2^2 and
B = 2 rho sigma_rel1 sigma_rel2, and the two-track maker exports **no covariance
at all** (only `Jpsi_sigmamass`, `resinfcov` and `Jpsi_jacMass`), so today they
have to be taken from the MC measurement (rho = 0, f_ang = 0.10-0.14, closed
form 4.6 % high).  On DATA at production scale they must be exported.

**The quantity is already in scope**: `covrefmom`
(`Matrix<double,6,6>`, `ResidualGlobalCorrectionMakerTwoTrackG4e.cc:4168`, set
from `covstate.topLeftCorner<6,6>()`), whose state indices 0-2 and 3-5 are the
two legs' (q/p, lambda, phi) at the reference; the leg<->plus/minus map is
`idxplus`/`idxminus`.  No new computation, only a `tree->Branch`.

| option | what | floats | bytes/cand | % of the 80.7 kB slim record | what it buys |
|---|---|---|---|---|---|
| **A (minimum)** | `covrefmom(0,0)`, `(3,3)`, `(0,3)` | 3 | **12 B** | **+0.015 %** | A and B EXACTLY; and f_ang for free, as 1 - (J_kappa C J_kappa^T)/sigma_m^2 using the already-exported `Jpsi_jacMass` |
| **B (recommended)** | the full symmetric 6x6 `Jpsi_covrefmom` | 21 | **84 B** | **+0.10 %** | everything in A, plus the kappa-angle CROSS terms of 1/2 tr(H Sigma) (not evaluated anywhere yet) and the exact angular Hessian piece (-sigma_theta^2/8, ~1e-7, negligible); plus the closure test J Sigma J^T = `Jpsi_sigmamass`^2 |
| C | per-leg `Muplus/Muminus_resinfvarv` + `sigmaqop2` | ~68 x 2 + 2 | ~550 B | +0.7 % | the per-leg FAMILY split (f_ms, f_ioni per leg).  **NOT needed**: a_i uses the candidate-level f_hit only |
| D | port the single-track hit-block registration into the two-track maker | ~68 more blocks | ~1.9 kB | +2.4 % | per-leg f_hit.  **NOT needed**, and it would put hit blocks into `resinfcov` and so CHANGE the meaning of `cfmass_vgf`, breaking every existing cache |

**Take option B.**  Option A is the fallback if 84 B is contested; C and D are
not required by either correction.

**The single-track maker needs nothing**: it already exports the full 5x5
`refCov` (`ResidualGlobalCorrectionMakerG4e.cc:579/4181`), from which
sigma_rel = sqrt(refCov[0])*p and hence the track-level a_i follow directly --
which is exactly what `oddmoment/track_truthfree.py` uses.

**Timing.**  The export must NOT be built before `jpsimc_20M_260905`
(slurm arrays 6406906 / 6406907) and `dymc_8p5M_260905` (6406978) finish.  A
`scram b` in an area a production is running from relinks the .so under the
running jobs and they segfault in the same second (NOTES, "dev-area rebuild
kills running jobs").  Until then the closed form with the MC-measured
rho = 0 and f_ang = 0.11 is the working configuration, with the +-5 % it carries.

## 5. The same defect at track level

For a single track, sigma = sigma_bar (1 + a q x) with
`a_i = sigma_rel,i * d ln sigma/d ln kappa ~= sigma_i * p_fit,i * (1 - vgf_i)`,
and the truth-free inversion is

    x_i = z_i / (1 - a_i q_i z_i) .

Validated in `oddmoment/track_truthfree.py`: it reproduces the truth-referenced
`sigma_bar` answer to 2e-4 pull units at every statistic, on both muon-gun
momentum ranges.

## 6. Why it matters

**Both terms scale as sigma_rel^2** -- the resolution artefact as
-(1 + f_hit) F sigma_rel^2 and the Jensen term as +(1.5 - f_ang) sigma_rel^2 --
so they partially CANCEL, with a ratio fixed by (1.5-f_ang)/[(1+f_hit) F]:
0.85 at the J/psi (measured 0.124/0.146) and 0.52-0.86 at the Z.  At Z momenta
each term ALONE is 15-43 MeV and the net is -5...-14 MeV.  The cancellation is
an accident of two numbers that both move with the sample; **both terms must be
put in explicitly.**

At J/psi momenta a_m = 0.0107, the predicted bias is -0.155e-3 and the MEASURED
one (toy, real per-candidate models) is **-0.148e-3**; on the real gun the
correction moves alpha by **+0.146e-3** and on B->J/psi X v3 by **+0.127e-3**.
The naive gun alpha of -0.005 +- 0.017e-3 is therefore NOT an unbiased scale: it
is a -0.14e-3 artefact cancelling a +0.14e-3 genuine residual.
At Z momenta both factors grow (a_m = 0.016-0.020, sigma_m/m = 0.013-0.015) and
the same defect is **-0.26 to -0.48e-3, i.e. -24 to -43 MeV on m_Z** — 25-45x
the 1e-5 target. See `oddmoment/zproject.py`.
