# RESUME HERE — 2026-09-09 evening

**You are resuming from files only. Read this section, then sec. 0f.43-0f.50 for
the open physics. Everything below the horizontal rule predates this.**

## IN FLIGHT — nothing needs babysitting; all of it is detached or slurm

| what | where | result lands | done when |
|---|---|---|---|
| **auxgen J/psi v2** (the coordinator's, detached on **submit50**) | `logs/run_auxgen_260909b.log` | `runs/auxgen_jpsiv2.npz` | the log ends `-> .../auxgen_jpsiv2.npz` |
| **auxgen DY v2** | — | **`runs/auxgen_dyv2.npz` HAS LANDED** (348 MB, 3 733 323 rows) | done |
| `22315719` K(m) ladder | Engaging, `engaging/zrabbitvb_22315719.out` | `fitresults/native/rabbit_{Ss9,SVs9}.hdf5` | `#### <TAG> done ... rc=0` per row -- **`Ss9` LANDED and is certified (sec. 0f.53); `SVs9` running since 20:05** |
| `22333692` K(m) 12 rungs (submitted 2026-09-08 20:45, the ladder job's walltime cannot hold them) | Engaging | `fitresults/native/rabbit_{Ss12,SVs12}.hdf5` | same |
| ~~`22328595` **P2X**~~ | — | — | **FAILED on a NaN at the first step (sec. 0f.54-55). The file it wrote is the UNMOVED start point -- not a phase-2 number.** |
| `22315802 insitu-tnp` | Engaging | NOT MINE -- another workstream, leave it | |

`./collect.sh --summary` rsyncs Engaging and re-makes the certified table. It is
the ONE command to run first.

**The two failed jobs -- DIAGNOSED 2026-09-08, see sec. 0f.54-55:**
* `SVetaEslo` and `P2X` fail the SAME way and the card is **NOT** singular.
  Both are NaN inside scipy `trust-exact`; the `Cholesky ... not
  positive-definite` is the downstream postfit at the UNMOVED start point, not
  the cause. Both cards were scanned dataset by dataset and are NaN/inf-free.
  The leading mechanism is sec. 0f.54: `--freezeParameters` uses
  `tf.stop_gradient` only while the minimiser runs on the FULL vector, so the
  four frozen `k` parameters are an exactly-null subspace of the Hessian --
  scipy's trust-region HARD CASE -- and the step walks in it, through
  `k_hit <= 0` and hence a non-positive density. Handed to the
  fit-infrastructure agent with the evidence; the fix is to minimise over
  `floating_indices`, NOT to regularise.
* `22328636` phase-3 card build -- `resolution/globalfit/` was never staged to
  Engaging. Being staged and resubmitted (`zcard3` 22332184).

## CERTIFIED (value AND NLL AND EDM, sec. 0f.16), MeV from the generator

| row | m form | v form |
|---|---:|---:|
| inclusive, K(m) 5 | **-11.06 +- 2.27** | **-1.54 +- 2.08** |
| K(m) 6 | -13.98 +- 2.22 | -3.92 +- 2.13 |
| K(m) 7 | -17.24 +- 2.27 | -3.87 +- 2.28 |
| `Gamma_Z` over K 5/6/7 | -5.26 / +27.16 / +8.96 | +6.81 / +14.28 / +12.86 |
| `\|eta\|<0.9` | -26.60 +- 2.87 | -21.08 +- 3.24 |
| `0.9-1.6` | +3.87 +- 3.88 | +12.39 +- 4.16 |
| `1.6-3.0` | +16.55 +- 4.71 | +34.22 +- 5.47 |

**The fixed-`eta` `sigma/m` split** (v form): barrel LOW **-4.56 +- 3.69**,
barrel HIGH **-40.90 +- 4.63** (a 36.3 +- 5.9 MeV split, 6.2 sigma, INSIDE one
band); endcap HIGH **+65.87 +- 8.69**. The barrel slope is **-9 300** MeV per
unit `sigma/m` and the across-`eta` slope is **+10 200** -- equal and OPPOSITE.
The bias is a function of neither `sigma/m` alone nor `eta` alone.

**Phase 2, first number**: `P2smoke` (500 k J/psi + 500 k Z + hit-chi2 over
20.7 M, 95 free) converged at **EDM 6.2e-19**, `m_Z = +31.86 +- 5.78`. **NOT a
closure** -- a subsample, and `theta = 0` is not the hit-chi2 minimum on this MC.

**`K(m)` is not saturated**: 5->6 and 6->7 are 15.2 and 13.0 sigma in
`2 deltaNLL`. K9/K12 are in flight.

## THE QUEUE, IN ORDER, WHEN `auxgen_jpsiv2.npz` LANDS

1. **`a_m` closed form.** `a_m = (1 + f_hit - f_ioni) sigma_m/m` against the
   MEASURED `a/(sigma/m)` = 1.2110 +- 0.0004 inclusive, 1.2503 / 1.1667 /
   1.2725 per band (`resolution/measure_a.py`). **BLOCKER, already visible in
   `auxgen_dyv2.npz`**: its `f_ioni` is **0.0000** and `f_other` is **1.0**,
   because `aux_gen.one()` groups `resinfvarv` by `parmtype == 10 / 11` and the
   v2 productions use **parmtype 14 (bfield) / 15 (material)**. Fix the
   grouping to this production's convention before the check means anything.
   Do NOT use the CF exponent's `-S''(0)` instead (sec. 0f.47b: no finite
   second moment, cut-dependent).
2. **`|seed->final dq/p|` per leg**, from `Jpsitrk_*` / the per-leg trk
   branches. **ABSOLUTE VALUE ONLY** -- the signed one has `corr(., x) = +0.113`,
   a worse trap than reco pT -- and state `corr(|delta|, |x|)`.
3. **The two-component decomposition per `eta`**, on the Z and J/psi legs
   (8x the gun's statistics) and at mass level kernel-free per band, IN vs OUT.
   Sec. 0f.50 tried it with `chi2/ndof` as a proxy: the IN component is
   `eta`-flat but OUT (90 % of the sample) is not, so `chi2/ndof` is not the
   discriminator and the test is NOT done.
4. Hand both auxgen files to the hit-class agent.

## WHEN P2X / K9-K12 / the phase-3 fit LAND

`./collect.sh --summary`, then `python3 plot_closure.py`. Certify every row with
**value AND NLL AND EDM** and nothing else; `certtable.py` applies it
mechanically and marks the error kind (`s` measured sandwich, `~` transported
ratio, `H` Hessian only -- an `H` row still owes `submit_sandwich.sh`).

## STANDING RULES — each was bought with a retracted number

1. **Every fit through `rabbit_fit.py`**, `--minimizerMethod trust-exact`
   (scipy). Both TF ports fail the subproblem at full statistics (sec. 0f.20)
   and `trust-krylov` stops at indefinite points (it killed P2K and
   `SVetaEslo`).
2. **Never bin on a reconstructed variable correlated with the residual.**
   Measured: reco leading pT `+0.042` (its top tertile sits **+225 MeV above
   its own gen mass**), signed seed->final `+0.113`, `maxfraclossp` `-0.034`.
   Safe: `chi2/ndof` `+0.0004`, `vgf` `-0.0009`, `eta_pair` `-0.0001`,
   `|eta|` lead `-0.008`, `m_gen` `-0.014`.
3. **Never bin on `sigma/m` or `sigma`** -- `sigma = sigma_bar(1 + a x)`, so
   the bin is a cut on the residual (sec. 0f.33).
4. **Truth-referenced pull for anything charge-split**: `x = z/(1 - a q z)`,
   else `<q z> = -a` is all you measure (sec. 0f.37).
5. **At mass level subtract the model** -- its own odd moment is
   +14.05 / +18.20 / +25.41 across the bands (`resolution/model_odd_mass.py`).
   At track level it is ~0 and raw data is fine.
6. **ceph via `ssh submit50` / `submit51`** -- submit82's cephx client is
   evicted, which is why every sandbox shell sees Permission denied.

## THE OTHER AGENT

`resolution/hitclassbias/STATE.md` -- the hit-class agent. It has REFUTED the
per-class location mechanism at track level (pixel locations are real but
+1e-3 wrong-signed and flat; strips null) and found the track-level `eta`
dependence to be a MIXTURE: an `eta`-independent bulk `-6.3 +- 1.8e-3` plus a
`+21e-3` subpopulation whose fraction grows 2.4 -> 21.2 % with `|eta|`. It is
testing the estimator (tighter GN convergence, gen seeding) on the gun.
Figures: `~/public_html/cvh/260908_hitclassbias/`. Mine:
`~/public_html/cvh/260908_fullscale/`.

---

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

**CORRECTION (same day, measured).** `--precondition` is NOT the cure, and
the 3.4e12 condition number is not the cause. The cause is
**`--freezeParameters`**: `get_x` stop-gradients a frozen parameter, so its
Hessian row and column are exactly zero and the matrix handed to the
trust-region subproblem is **singular**. `trust-exact` then meets the "hard
case" on every iteration, never returns an interior Newton step, so
`hits_boundary` is False, so the radius never doubles — linear convergence.
A singular block cannot be whitened, which is why preconditioning does
nothing (the preconditioned run crawls too, from a different start).

Two independent cures, both measured on `z_n300k` with the same four
parameters frozen:

| | outcome |
|---|---|
| `tf-trust-exact` | **51 iterations, EDM 111.5, still crawling** |
| **`tf-trust-krylov`** | **converged, EDM 4.98e-13, 135 s total** |
| `tf-trust-exact` + `Fitter.hess_for_minimizer` (rabbit `d83342e`) | **18 iterations, EDM 1.0e-15** |

GLTR is immune because Lanczos never explores the null space — the frozen
directions carry no gradient, so they are never in the Krylov subspace.
`hess_for_minimizer` puts 1 on the frozen diagonal before the matrix reaches
the subproblem, which is EXACT rather than a regularisation (the frozen
gradient components are zero, so `p_frozen = -0/1 = 0` for any positive value
there) and simply makes the matrix invertible.

The fixed `trust-exact` EDM ends 222 -> 46.2 -> 0.179 -> 3.0e-6 -> **1.0e-15**
-- textbook quadratic convergence, which is what a Newton-type method does
once its subproblem is not singular. Against the unfixed run's 66 iterations
at EDM 83 and still falling 1 % per iteration.

**But it converged somewhere else, and that is worth more than the fix.** On
this card the two methods reach two DIFFERENT stationary points:

| | NLL | EDM | `Gamma_Z` |
|---|---:|---:|---:|
| standalone `fit.py` | 874734.9966056045 | — | -421.03 |
| `tf-trust-krylov` | **874734.9966056045** | 4.98e-13 | -421.03 |
| `tf-trust-exact` + the fix | 874816.6155744941 | 1.04e-15 | -368.17 |

**81.6 NLL units apart, and the one with the SMALLER EDM is the worse
point.** Both are genuine stationary points -- an EDM of 1e-15 is not a
convergence failure, it is a converged fit at a local minimum. `z_n300k` is
the residual-form stopgap card, the one sec. 0 records as running away to
`Gamma_Z = -421 MeV`, so a multi-modal likelihood there is expected; but the
lesson generalises:

> **EDM certifies stationarity, not optimality.** It is exactly the tool for
> "did this fit stop early", which is the failure that cost four numbers, and
> it says nothing about "is this the right minimum". Compare NLL between runs
> as well, and treat a large `|delta NLL|` between two converged fits of the
> same card as the alarm it is.

**Use `tf-trust-krylov` when anything is frozen** (it is immune with no fix
at all, and it pays ~5 HVPs per step where `trust-exact` pays `nfree`
columns -- at the 99 parameters of a joint card that is the whole cost).
**But read 7.9 before making it the default at full statistics.**

### 7.7b The standalone drivers are unchanged by any of this

`fit.py --card cards/smoke_zls.hdf5 --chunk 8192 --maxiter 20`, the
pre-change drivers on `rabbit-material` against the current drivers on
`rabbit-native`:

| | |
|---|---|
| reference-point NLL | **identical** (150039.08298253382) |
| final NLL | **identical** (149975.61860577622) |
| iterations | **identical** (10) |
| fitted parameters | 1.8e-15 |
| errors | 1.0e-15 |

So the graph chunk loop becoming the default, `ChunkTable`, the
`MaterialCFTerm` CSR change and the Fitter's new Hessian route leave the
standalone path bit-equivalent. The 1e-15 is multithreaded-reduction
round-off, not a code difference.

### 7.8 The 300 k card, end to end

`rabbit_fit.py --minimizerMethod tf-trust-krylov --freezeParameters k_hit
k_ms k_ioni k_rad` against the standalone `fit.py --engine host --method
trust-exact` on `cards/z_n300k.hdf5`, same 7 free parameters:

| | `rabbit_fit.py` | standalone `fit.py` | rel |
|---|---:|---:|---:|
| `m_Z` | -35.505318815 | -35.505318456 | 1.0e-8 |
| `Gamma_Z` | -421.028782317 | -421.028790154 | 1.9e-8 |
| `shape1` | -1.061860625 | -1.061860623 | 1.4e-9 |
| `shape2` | -0.830130441 | -0.830130438 | 2.8e-9 |
| `shape3` | -0.608301723 | -0.608301721 | 2.4e-9 |
| `shape4` | -1.846358060 | -1.846358064 | 2.1e-9 |
| `shape5` | +0.305905435 | +0.305905436 | 2.6e-9 |

errors identical to six decimals, worst parameter difference **1.9e-8** --
well inside the 1e-6 the migration had to hit. rabbit took **135 s total**
including the card load and the postfit Hessian, against **776.7 s** for the
standalone fit alone, and finished at **EDM 4.98e-13** where the standalone
stopped at `|grad|inf` 1.7e-3 with no EDM at all.

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

**RETRACTED, twice, and the second answer is measured.** First I recorded
`--precondition` as the cure for the trust region crawling. It is not: a
preconditioned 300 k run crawls just as badly. **The cause is
`--freezeParameters`** — a frozen parameter is stop-gradiented, so its Hessian
row and column are EXACTLY ZERO, and four of them hand the trust-region
subproblem a SINGULAR matrix on every iteration. `trust-exact`'s nearly-exact
subproblem meets that head-on and treats it as the hard case, so it never
returns an interior Newton step, `hits_boundary` stays False, the radius never
doubles, and the fit converges LINEARLY. A singular block cannot be whitened,
which is why preconditioning did nothing. The 3.4e12 condition number is real
and bad, but it is not what causes the crawl.

**This bites every Z fit in this campaign**, all of which freeze `k_hit`,
`k_ms`, `k_ioni`, `k_rad` at the MC truth.

**Two independent fixes, both available:**

| | frozen | outcome (300 k card) |
|---|---|---|
| `tf-trust-exact` | 4 | 51 iterations, EDM still 111.5, crawling |
| **`tf-trust-krylov`** | 4 | **converged, EDM 4.98e-13, 135 s** |

GLTR is immune because Lanczos never explores the null space: the frozen
directions carry no gradient, so they are never in the Krylov subspace.
`d83342e` (merged here as `80899f9`) additionally makes `trust-exact` work, by
putting 1 on the frozen diagonal before the subproblem sees it — exact, not a
regularisation, since the frozen gradient components are zero so
`p_frozen = -0/1 = 0`. **The method is what is used here**; the fix is carried
as well. `--precondition` is dropped: a pure reparameterisation, so it cannot
give a wrong answer, it just buys nothing.

**And the migration's own number, on the 300 k card**: `rabbit_fit.py`
(`tf-trust-krylov`) against the standalone `fit.py` (host `trust-exact`), same
card, same 7 free parameters — `m_Z` -35.505318815 against -35.505318456
(1.0e-8), `Gamma_Z` -421.028782317 against -421.028790154 (1.9e-8), shapes
<=2.8e-9, errors identical to six decimals. rabbit took **135 s** against
776.7 s and finished at **EDM 4.98e-13** where the standalone stopped at
`|grad|inf` 1.7e-3 with no EDM at all.

**Three fits are running through it** with `--precondition`, `--diagnostics`
and `tf-trust-exact` (`engaging/rabbit_vmass.sbatch`, PYTHONPATH at
`rabbit-vmass`):

| job | card | what it decides |
|---|---|---|
| `22301214 f380ref` | `z_full380_fl` | **the control**: it must come back at -11.06 |
| `22301216 Rdc8` | `z_F_dc8` | the mechanism, at last with EDM |
| `22301217 Rvfull` | `z_V_full` | the v closure, at last with EDM |
| `22301218 Rvtoy` | `z_V_toy` | gate 3, re-verified |
| `22301219 Rs6`, `22301220 Rs7` | the `K(m)` ladder | the `Gamma_Z` shape question |

(`22300667-9` and `22301104-6` were cancelled while still PENDING, before the
two corrections above -- nothing was lost.)

`fullscale/rabbit_to_json.py` (the agent's, `151bfcf`) converts a rabbit result
into the json `fit.py --start-from` reads, carrying `edmval`, so the sandwich
covariance and the `x1.109` weight factor can still be evaluated at rabbit's
minimum with `fit.py --no-fit`.

**Both cures are now measured** (the agent's confirmation of `d83342e`, on
`z_n300k`, same four parameters frozen, `tf-trust-exact`):

| | EDM trajectory | outcome |
|---|---|---|
| without the fix | 16495, 1150, 421, 227, 210, 209, 207, 203 ... | 66 iterations, EDM 83, crawling |
| **with the fix** | ... 551, 538, 513, 466, 377, 222, **46.2, 0.179, 3.0e-6, 1.0e-15** | **18 iterations, EDM 1.0e-15** |

That tail is textbook quadratic convergence — what a Newton method does once
the subproblem is not singular. So the diagnosis is settled: the frozen rows,
not the conditioning, and not preconditioning.

**Which method to use.** `tf-trust-krylov` is the default here: it is immune
without any fix, and it pays ~5 HVPs per step where `trust-exact` pays `nfree`
HVP columns. At 7 free parameters they are close; on the **99-parameter joint
cards of phase 2** krylov should win by a wide margin. Reach for `trust-exact`
only when the Hessian at every step is wanted anyway.

### 0f.15 EDM CERTIFIES STATIONARITY, NOT OPTIMALITY

A caveat that arrived before it could cost anything, and it changes how the
control is read. The agent compared two CONVERGED rabbit fits of the same card
(`z_n300k`, same freezing, same start):

| | NLL | EDM | `Gamma_Z` |
|---|---:|---:|---:|
| `fit.py` | 874734.9966056045 | — | -421.03 |
| `tf-trust-krylov` | **874734.9966056045** | 4.98e-13 | -421.03 |
| `tf-trust-exact` + the frozen fix | **874816.6155744941** | **1.04e-15** | -368.17 |

**81.6 NLL units apart, and the one with the SMALLER EDM is the WORSE point.**
Both are genuine stationary points: an EDM of 1e-15 is not a convergence
failure, it is a converged fit at a DIFFERENT LOCAL MINIMUM. (`z_n300k` is the
residual-form stopgap card that sec. 0 records running away to
`Gamma_Z = -421`, so multi-modality there is expected and nothing physical
should be read into it.)

**So the acceptance test is EDM AND NLL, not EDM alone.** EDM answers "did this
fit stop early" — the failure that cost four numbers today — and says nothing
about "is this the right minimum". Two converged fits of the same card whose
NLL differs by more than float noise is an alarm, not a rounding difference.

**The control's target**: `f380fl_base` sits at `m_Z = -11.064299431899864`,
`Gamma_Z = -5.264591805106016`, **NLL = 11075392.465686228**. `22301214
f380ref` passes only if it returns that `m_Z` at a small EDM AND at that NLL.

### 0f.16 THE ACCEPTANCE TEST FOR EVERY FIT IN THIS CAMPAIGN

Three parts, and each failure mode is separately diagnosable:

1. **the value** — does it return the number the card is known to give?
2. **the NLL** — is it the SAME minimum? Two converged fits of one card whose
   NLL differs by more than float noise are at different stationary points
   (measured: 81.6 units apart on `z_n300k`, sec. 0f.15).
3. **the EDM** — did it stop early? `0.5 g^T H^-1 g`, small.

EDM alone is not enough (it certifies stationarity, not optimality) and the
value alone is not enough (a fit that never moves reproduces its start
perfectly -- that is exactly what `V_full` did). **Both of the failures that
cost numbers today are caught by this and neither is caught by `|grad|inf`.**

### 0f.17 SINGULAR HESSIANS: TWO KINDS, AND THEY MUST BE TREATED DIFFERENTLY

* **A FROZEN parameter** has an exactly zero Hessian row because `get_x`
  stop-gradients it. That is an ARTEFACT of the parameterisation and it made
  every `trust-exact` fit in this campaign converge linearly (sec. 0f.13).
  `Fitter.hess_for_minimizer` puts 1 on the frozen diagonal -- exact, since the
  frozen gradient components are zero.
* **An UNCONSTRAINED BUT FLOATING parameter** also has a zero row, and that one
  is PHYSICS: nothing in the likelihood measures it. It must be caught and
  named, not regularised.

The fix applies its unit diagonal on the frozen mask only, so the second case
stays singular by construction rather than by luck, and rabbit `54e47f6` adds
`Fitter.warn_unconstrained`: any floating parameter whose Hessian diagonal is
below 1e-12 of the largest is named once, with the message that nothing
constrains it and that this is deliberately not regularised away.

**This matters for phase 2 before it is run.** The joint card's Hessian is
singular in four directions BY CONSTRUCTION: the occupancy census
(sec. "PHASE 3") found `material_pp1_cables` touched by **no candidate on
either leg** and `material_thermal_screen` / `material_support_tube` by under
0.1 %, and the hit-chi2 term is blind to the same ones. **Freeze them
explicitly, or give them a prior, and say which in the result** -- do not let
the frozen-diagonal fix absorb them, because then an unmeasured parameter would
look measured.

### 0f.18 THE CONTROL FAILED — `tf-trust-krylov` STOPS EARLY ON THE FULL CARD

`22301214 f380ref`: `z_full380_fl`, `tf-trust-krylov`, four frozen, on the
merged branch. 20 minutes, a "converged" snapshot, and:

| | rabbit `tf-trust-krylov` | the reference (`fit.py`, host `trust-exact`) |
|---|---:|---:|
| `m_Z` | **-0.127 +- 2.064** | **-11.064 +- 2.267** |
| `Gamma_Z` | -0.016 +- 3.786 | -5.265 |
| NLL | **11075407.184058** | **11075392.465686228** |
| EDM | **14.723157930030274** | 1.8e-18 |

**The NLL is 14.718 above the reference and the EDM is 14.723. They are the
same number.** EDM did exactly its job: it said "14.7 NLL units are still to be
gained", and they were, and the reference is at the bottom of them. So this is
NOT a second minimum — it is one minimum, and krylov stopped 14.7 units up the
hill from it while reporting convergence. And it stopped at `m_Z = -0.127`
against a start of 0: **the POIs barely moved, the same signature as the
`V_full` failure**.

**So every fit of mine that stopped early used `tf-trust-krylov`, and the one
that converged used host `trust-exact`.** The scoreboard on this card is the
opposite of `z_n300k`'s, where krylov reached EDM 4.98e-13.

**The cause: GLTR loses its model at this conditioning, AND the outer loop
mistook that for convergence.** At a 3.4e12 condition number the Lanczos
recurrence loses about twelve digits of orthogonality in float64, so the Krylov
model of the subproblem becomes unreliable and its predicted reduction goes
non-positive long before the true one does. `rabbit/minimizer/base.py` then
did

```python
if predicted_reduction <= 0:
    warnflag = 2
    break          # logged at DEBUG as "the standard end state of a converged fit"
```

— which is true only when the subproblem is solved ACCURATELY. For GLTR here it
means "my Krylov model cannot find descent", a SOLVER failure, and the loop
could not tell that from "no descent exists" and took the flattering reading.
That is why a fit 14.7 units up the hill wrote a "converged" snapshot.

**Fixed at `c5f46f0`** (merged here as `a8b2bbc7`): on a non-positive predicted
reduction the loop now shrinks the radius and RE-SOLVES, and separates the two
cases on the Cauchy first-order gain `radius |g|` against float noise on `fun`
— below it, no step of any kind can help and stopping is right (status 2,
unchanged for an accurately solved subproblem, so `trust-exact` is untouched);
above it, the model is wrong rather than the point, and it fails LOUDLY with a
new status 4 naming `|g|` and the radius.

**RETRACTED: "`tf-trust-krylov` cannot be the campaign default at full
statistics."** I drew that against a loop that mistook a solver failure for
convergence, so it is not yet a statement about the objective. `22304185
f380refK2` re-runs exactly this fit on `c5f46f0`. If krylov recovers and walks
down the remaining 14.7 units, it is usable after all and only the loop was at
fault; if it cannot, the new code says so with a warning instead of a number,
which is the outcome to want either way.

`22303682/3/4` (`f380refX`, `Rdc8X`, `RvfullX`) test `tf-trust-exact` WITH the
frozen-diagonal fix, a combination not yet tried at full statistics. If
`f380refX` returns -11.0643 at NLL 11075392.4657 with a small EDM, the answer
for this campaign is **`tf-trust-exact` + `d83342e`** and the FIX rather than
the method is what makes the migration usable here.


### 7.9 A subproblem that cannot predict descent is not a converged fit

`tf-trust-krylov` on the FULL-statistics card wrote a "converged" snapshot at
a point **14.718 NLL units above the minimum**, with its own EDM reading
**14.723** -- the two agreeing to three digits -- and the POIs barely moved
from their starting values (`m_Z` -0.127 against a reference -11.064). That
is the worst failure a minimizer can have, because the answer looks clean.

The mechanism is in the outer loop, not only in GLTR. `base.py` had

```python
if predicted_reduction <= 0:
    warnflag = 2
    break
```

and logged `warnflag == 2` at DEBUG as "the standard end state of a converged
fit". **That is a statement about the minimum only when the subproblem is
solved accurately.** For a Krylov subproblem it is equally the signature of a
model that has gone bad -- GLTR's Lanczos recurrence loses roughly twelve
digits of orthogonality at this Hessian's 3.4e12 condition number, so its
truncated model predicts no descent while the true objective still has a long
way to fall. The loop could not distinguish "no descent exists" from "I
cannot find it", and took the flattering reading.

**Fixed in rabbit `c5f46f0`.** On a non-positive predicted reduction the loop
now shrinks the radius and RE-SOLVES rather than concluding. At a small enough
radius the model is the local quadratic and must predict descent whenever the
gradient is non-zero, so a retry either recovers the fit or proves the point
stationary. The Cauchy first-order gain `radius * |g|` against float noise on
`fun` separates the two: below it no step can help and stopping is right
(status 2, quiet, and unchanged for an accurately solved subproblem, so
`trust-exact` behaves exactly as before); above it the model is wrong rather
than the point, and the loop fails **loudly** with a new status 4 naming
`|g|` and the radius it reached. GLTR reuses its radius-independent Krylov
data, so a retry usually costs no new HVPs.

**Consequence for the recommendation.** "Keep `tf-trust-krylov` as the
default" was measured on `z_n300k` and does not transfer to full statistics
on the OLD code. Whether it transfers on `c5f46f0` is an open question and
needs one re-run before "krylov cannot be the campaign default" is recorded
as a property of the objective -- that conclusion was drawn against a loop
that mistook a solver failure for convergence.

**And the general rule this is the fifth instance of:** every failure in this
migration -- the three `--diagnostics` bugs, the frozen-row singularity, and
this -- had the same shape. A numerical component could not do its job, and
the surrounding code reported success instead of saying so. None would be
caught by a test that asserts a fit RUNS; every one was caught by asserting
something about WHERE IT ARRIVED. The EDM is the instrument that caught them
all, and here it did more than detect the failure: it MEASURED it, to three
digits, before anyone knew there was one.

**The second pattern, which is physics and not software** (the analysis
agent's, recorded here because it is the reason the acceptance test has to be
permanent): *every one of the five produced a number that flattered the
hypothesis.*

| reported | what it was |
|---|---|
| `V_full` `m_Z` = -0.0002 +- 2.32 | a fit that never took a POI step |
| `F_dc8` `m_Z` = -0.12 +- 2.42 | 2.44 sigma from its minimum |
| `f380ref` `m_Z` = -0.127 | 14.7 NLL units above the true -11.064 |

All three read as *"the closure is perfect"*. That is not coincidence: an
under-converged fit sits near its starting point, and the starting point of
every closure test here is **the MC truth** -- so a fit that fails to move
reads as a beautiful closure. The bias runs towards the answer one is hoping
for, which is exactly when a check is least likely to be demanded and most
needed.

**This gate suite was vulnerable to the same thing** and has been hardened:
`test_devobj.py` compared two minimisers from a shared start, which passes
when BOTH stall. It now anchors absolutely -- the reference fit must descend
from the starting NLL by far more than the tolerance on their agreement, or
the comparison is refused with "two minimisers agreeing about a point neither
of them reached".
### 0f.19 THE FAILURE MODE, AND WHY THE ACCEPTANCE TEST IS PERMANENT

Five separate defects were found in one day (three `--diagnostics` bugs, the
singular frozen Hessian, the outer loop reading a solver failure as
convergence). **Every one of them produced a number rather than an error**, and
— this is the part that matters for a physics analysis — **every one produced a
number that FLATTERED the hypothesis.**

That is not a coincidence. A fit that stops early sits near its starting point,
and the starting point of every closure test here is the MC TRUTH. So an
under-converged closure test reads as "closes beautifully". The three numbers
retracted today:

| reported | what it actually was |
|---|---|
| `V_full` `m_Z = -0.0002 +- 2.32` | a fit that never took a POI step |
| `F_dc8` `m_Z = -0.12 +- 2.42` | 2.44 sigma from its minimum |
| `f380ref` `m_Z = -0.127` | 14.7 NLL units above the true -11.064 |

All three said "the closure is perfect". None would have been caught by a test
that asserts a fit RUNS; every one is caught by asserting something about WHERE
IT ARRIVED. **That is why sec. 0f.16's three-part test (value AND NLL AND EDM)
is a permanent fixture and not this week's remedy** — the bias is towards the
answer one is hoping for, which is exactly when a check is least likely to be
demanded and most needed.

### 0f.20 THE MINIMISER DECISION — BOTH TF PORTS FAIL THE SUBPROBLEM AT FULL
### STATISTICS; SCIPY `trust-exact` IS THE CAMPAIGN MINIMISER (2026-09-08 14:00)

All the controls of sec. 0f.18 are in. **`22304185 f380refK2` never ran** (it
was CANCELLED while PENDING); the native agent's `22304228` is the krylov +
loop-fix control and it is the one that answers the question.

Every one of these is `z_full380_fl`, four frozen, from the prefit point. The
reference is the standalone `fit.py` (host `trust-exact`), `f380fl_base`.

| run | method | branch | `m_Z` | NLL | EDM |
|---|---|---|---:|---:|---:|
| `f380fl_base` (reference) | scipy `trust-exact`, standalone | — | **-11.064** | **11075392.4657** | 1.8e-18 |
| `22301214 f380ref` | `tf-trust-krylov` | pre-`c5f46f0` | -0.127 | 11075407.1841 | 14.72 |
| **`22304228 f380krylovfix`** | `tf-trust-krylov` | **`c5f46f0` (loop fix)** | **-0.127** | **11075407.1841** | **14.72** |
| `22303682 f380refX` | `tf-trust-exact` + frozen fix | `a8b2bbc` | +0.233 | 11075420.3476 | 16.79 |

**The loop fix changes nothing here: `f380krylovfix` is BIT-IDENTICAL to
`f380ref`** (`m_Z`, NLL and EDM agree to every digit stored). So krylov's stop
on this card is NOT through `predicted_reduction <= 0` — the fix is right and
necessary, but this failure is a different one, and the retraction of 7.9 is
therefore itself retracted: **`tf-trust-krylov` cannot be the campaign
default at full statistics**, now measured against the fixed loop.

**And `tf-trust-exact` + the frozen-diagonal fix is WORSE**, not better: NLL
27.88 above the reference against krylov's 14.72, and again with the POIs
sitting at their start (+0.233 against -11.064). Its EDM trajectory
7210 -> 46.3 -> 224 -> 94 -> 33 -> 19 -> 16.93 -> 16.9439 -> ... -> 16.7943880899
is a plateau, not a descent: the last eight iterations move the EDM in the
sixth decimal. And the log names the cause 20 times:

```
WARNING:exact.py: trust-region subproblem lambda search hit maxiter=50;
                  returning safeguarded step
```

**Both TF ports fail the SUBPROBLEM at this conditioning, in their own way.**
GLTR loses Lanczos orthogonality (sec. 7.9); the TF More-Sorensen port hits its
lambda-search iteration cap. The second is documented in `exact.py`'s own
docstring as a deliberate omission: LAPACK's `potrf` returns the index `k` of
the first non-positive-definite leading minor, scipy feeds it to
`singular_leading_submatrix` to tighten `lambda_lb` below the critical damping,
and `tf.linalg.cholesky` cannot report `k`. Without that accelerated bound the
safeguarded bisection needs more than 50 iterations here. **This is exactly the
difference between the TF port and the scipy original, and it is exactly the
difference between the runs that stall and the run that converges.**

**The decision: `--minimizerMethod trust-exact` (scipy, explicit Hessian)
through `rabbit_fit.py`.** It is the reference algorithm, it is inside rabbit
(so the standing rule holds and the EDM, the snapshots and the result file come
with it), `scipy_hess` already applies `hess_for_minimizer` so the frozen rows
are handled, and its per-iteration cost is the same `nfree` HVP columns
`tf-trust-exact` already pays.

**It is NOT free of the sec. 0f.15 caveat.** On two other cards
`tf-trust-exact` + the fix DID converge, and to two different verdicts:

| card | krylov | `tf-trust-exact` + fix | which is lower |
|---|---:|---:|---|
| `z_F_dc8` | 11014854.5548 (EDM 4.44) | **11014819.0793** (EDM 3.8e-12) | trust-exact, by 35.5 |
| `z_V_full` | **-9827099.7568** (EDM 1.98) | -9827069.1806 (EDM 1.6e-12) | krylov, by 30.6 |

so on `z_V_full` a fit that converged to EDM 1.6e-12 sits 30.6 NLL units ABOVE
a fit that did not converge at all. Every row of the table needs BOTH numbers.

**The suite that decides it** (all `rabbit_fit.py`, `--minimizerMethod
trust-exact`, `GATE=0 FRESH=1`, four frozen; `engaging/submit_scipy_suite.sh`):
`22311742 f380refS` is the control and passes only at `m_Z = -11.0643`,
NLL `11075392.4657`, small EDM. `22311743-63` are the other 18 rows of the
certified table (m form: `Sdc8 Stoy Stoydc Sw70110 Ss6 Ss7 SMetaB/T/E`;
v form: `SVfull SVtoy SVs6 SVs7 SVetaB/T/E SVKetaB/T`).

### 0f.21 THROUGHPUT: WARM STARTS, AND WHY THEY CANNOT FLATTER A RESULT

scipy `trust-exact` is the minimiser that works here and it is also the slow
one: ~38 iterations and **~5.5 h** on the full-statistics card (19 600 s,
measured in sec. 6 against `tf-trust-krylov`'s 1206 s), against ~114 s per
iteration for the TF port that does not converge. At 19 rows and a
`QOSMaxGRESPerUser` of 4 GPUs on `mit_preemptable` (plus 2 on `mit_normal_gpu`)
a cold suite is ~20 h of wall clock.

**So every row except the controls starts from the LOWEST-NLL point any earlier
fit of that card reached** (`json_to_snapshot.py`, `seed_rows.sh`, 18
snapshots in `results/seed/`; `rabbit_vmass.sbatch` resumes from
`$OUT/rabbit_$TAG.snapshot.hdf5` when `FRESH != 1`).

**This cannot flatter a result and it is worth saying why.** The acceptance
test of sec. 0f.16 is entirely about where a fit ARRIVES -- the value, the NLL
and the EDM at the stopping point -- and says nothing about where it started.
A warm start changes only the number of iterations. The failure mode of
sec. 0f.19 was the opposite one: fits that stopped near a start that happened
to be the MC TRUTH, and so read as perfect closures. Seeding from a stored
minimum rather than from the truth removes that particular coincidence as well.

**The controls stay COLD**, because they are testing the minimiser and not the
card: `22311742 f380refS` (`z_full380_fl`, must find -11.0643 by itself),
`22311743 SVfull` (`z_V_full`) and `22311744 Sdc8` (`z_F_dc8`). Those last two
are also the answer to the sec. 0f.15 caveat on their cards -- `z_V_full` is
the one where a converged `tf-trust-exact` sat 30.6 NLL units ABOVE an
unconverged krylov point -- so a cold and a warm run of the same card that meet
at the same NLL is the evidence that there is one minimum and not two.

**The subproblem cap is NOT reproducible synthetically.** A Hessian with the
measured spectrum (7 free directions spanning 3.4e12 in curvature plus the four
frozen unit rows) never hits `maxiter=50` at any radius from 1e-3 to 100, over
5 random orthogonal bases. So the 20 cap hits in `f380refX` are a property of
the ACTUAL Hessian at those points, not of the conditioning alone, and raising
`MAXITER_DEFAULT` is not demonstrably the cure. That line of enquiry is
dropped: the empirical fact -- scipy's More-Sorensen converges on this card and
the TF port does not -- is what the campaign runs on.

### 0f.22 CHECKPOINT 2026-09-08 14:30 — what is certified so far, and what is running

`certtable.py` is the mechanical form of the sec. 0f.16 acceptance test: it
enumerates EVERY stored fit of every card (`fit.py` json and `rabbit_fit.py`
result alike) and reports value / NLL / EDM / full POI Newton step, with the
NLL compared only across fits of the SAME model — `fit.py` turns one card into
a scan and `f380fl_noboth` sits 2512 NLL units BELOW `f380fl_base` on identical
candidates. `collect.sh` pulls Engaging and re-makes it.

**CERTIFIED (all four parts pass), m formulation:**

| row | `m_Z` | `Gamma_Z` | NLL | EDM |
|---|---:|---:|---:|---:|
| inclusive, K 5 | **-11.06 +- 2.27** | -5.26 +- 4.16 | 11075392.4657 | 1.8e-18 |
| K 6 | -13.98 +- 2.22 | **+27.16 +- 4.35** | 11075277.1487 | 1.7e-12 |
| K 7 | **-17.24 +- 2.27** | +8.96 +- 4.42 | 11075192.7817 | 8.1e-10 |
| the assembly toy `F_toy` | -3.75 +- 2.19 | -4.73 +- 4.12 | 10994223.7216 | 5.7e-15 |
| `F_dc8`, sigma-reweighted | **-2.03 +- 2.06 (H)** | -5.19 +- 3.76 | 11014819.0793 | 3.8e-12 |
| `F_toydc` | +8.73 +- 2.40 | -5.26 +- 4.40 | 10933537.0862 | 4.1e-19 |
| `F_w70110` | -13.62 +- 2.88 | -13.44 +- 4.94 | 10063943.1582 | 5.1e-11 |

**(H) = inverse-Hessian error, sandwich pass still owed.** Everything else is
the measured sandwich.

**THREE NUMBERS OF sec. 0g CHANGE ONCE THE FITS ARE CONVERGED**, and all three
move AWAY from closure — the sec. 0f.19 pattern again:

| | sec. 0g (as run) | converged | |
|---|---:|---:|---|
| `F_dc8` `m_Z` | -0.12 +- 2.42 | **-2.03** | at NLL 35.5 LOWER |
| `F_toy` `m_Z` | -3.08 +- 2.19 | -3.75 | |
| `F_toydc` `m_Z` | +3.80 +- 2.41 | +8.73 | |
| `F_w70110` `m_Z` | -1.87 +- 3.15 | **-13.62** | |
| K 7 `Gamma_Z` | +1.14 +- 4.40 | **+8.96** | |

**`F_dc8` still carries the sec. 0g conclusion but weakened**: the reweighting
that removes the dependence of the true mass on the resolution class takes
`m_Z` from -11.06 +- 2.27 to **-2.03 +- 2.06** — 80 % of the effect, not 100 %,
and 1.0 sigma from zero rather than 0.05 sigma. `F_w70110` no longer supports
it at all: the narrow window was read as "less mass range, less effect" at
-1.87, and converged it is **-13.62 +- 2.88**, i.e. as large as the baseline.

**`Gamma_Z` is NOT `K(m)`-saturated at full statistics, and convergence does
not rescue it.** Certified: -5.26 (K5), **+27.16 (K6)**, +8.96 (K7) against a
4.2-4.4 MeV statistical error. The 32 MeV swing of sec. 0g SURVIVES — it has
moved from the 6->7 step to the 5->6 step, but the span over the ladder is
unchanged at 32.4 MeV. **`Gamma_Z` remains a tens-of-MeV statement.**
And `m_Z` over the same ladder is now MONOTONE — -11.06, -13.98, -17.24, a
6.2 MeV drift over its own 2.3 MeV error, so "the basis is saturated for `m_Z`"
is weaker than sec. 0g had it: the drift is 2.7 sigma, not 1 sigma.

**Certified, v formulation** (only one row so far): `|eta_lead| < 0.9`
**-21.08 +- 3.24**, `Gamma_Z` +1.37 +- 5.82, NLL -4266842.7487, EDM 7.8e-12.
The warm start reproduced its seed's NLL to the 4th decimal, so
`fit_V_etaB.json` WAS at its minimum; its "no EDM" verdict was ignorance, not
failure.

**Running**: the three cold controls (`22311742 f380refS`, `22311743 SVfull`,
`22311744 Sdc8`, ~1.7 min/iteration against `fit.py`'s 7.4, so ~80 min), the
warm rows in two batched jobs (`22312979`, `22312980`), the warm control twins
(`22312981`), the phase-2 smoke (`22312856`) and the phase-2 krylov stage
(`22312984 P2K`, `joint_ok_full`).

### 0f.23 THE CONTROL PASSES — `rabbit_fit.py --minimizerMethod trust-exact`
### REPRODUCES THE REFERENCE EXACTLY (2026-09-08 14:40)

`22312981 f380refW`: `z_full380_fl`, scipy `trust-exact` through
`rabbit_fit.py`, four frozen, warm-started at `f380fl_base`'s point.

| | `m_Z` | `Gamma_Z` | NLL | EDM |
|---|---:|---:|---:|---:|
| `f380fl_base` (the standalone reference) | -11.064 +- 2.27 | -5.265 +- 4.16 | 11075392.465686 | 1.777e-18 |
| **`f380refW` (rabbit)** | **-11.06 +- 2.27** | **-5.26 +- 4.16** | **11075392.4657** | **1.779e-18** |

All three parts of the acceptance test agree, and it took **two iterations**
(EDM 1.779e-18 at the first, 4.53e-23 at the second) — i.e. rabbit's scipy
`trust-exact` immediately certifies the reference point as a minimum rather
than walking away from it, which is what the two TF ports did (they stopped
14.7 and 27.9 NLL units above it, from a cold start).

**So the campaign minimiser is settled**: `rabbit_fit.py --minimizerMethod
trust-exact`, four resolution knobs frozen. The standing rule (every fit
through `rabbit_fit.py`) and the reference algorithm are the same thing.

**The `sandwich.sbatch` stage may often be skipped.** `certtable.py`
transports the measured `sandwich_ratio` onto a rabbit row that sits at the
same point as a `fit.py` row of the same card (same `m_Z` to 0.01 MeV, same
NLL to 0.01), and marks the row `~`. `s` is a directly measured sandwich, `H`
an inverse-Hessian error that still owes one. The ratio is 1.088 to 1.177
across these cards — it is NOT the flat x1.109 and cannot be applied by hand.

### 0f.24 A REAL BUG IN THE PHASE-2 PATH: THE SPARSE `D` HAS NO DETERMINISTIC
### GPU KERNEL (2026-09-08)

`22312856 P2smoke` — `joint_ok_n500k` through `rabbit_fit.py` — loaded both
mass terms and the external quadratic correctly and then died at the FIRST
loss evaluation:

```
UNIMPLEMENTED: A deterministic GPU implementation of
               SparseTensorDenseMatmulOp is not currently available
   [[{{node while/SparseTensorDenseMatMul/SparseTensorDenseMatMul}}]]
```

`bin/rabbit_fit.py` line 21 calls `tf.config.experimental.enable_op_determinism()`
at import, and `MassCFTerm._chunk_residual` multiplied the per-candidate sparse
`D` by `theta` with `tf.sparse.sparse_dense_matmul`. **Every joint card in
phases 2 and 3 hits this**, and it is not reachable from the standalone
drivers — `chunkfit.py` uses a DENSE affine map (pitfall 3), which is why the
gates passed and the fit did not.

**Fixed by storing `D` dense** (`JacChunkTable(dense=)`, default: dense above
1/3 fill). That is not a workaround, it is the better representation on these
cards: a COO nonzero costs two int64 indices plus a float64 value, 24 bytes
against 8 for a dense entry, so sparse only pays below ~1/3 density — and the
joint cards' `D` blocks are **79.3 % (J/psi) and 82.3 % (Z) dense**, where the
sparse form is larger, slower AND undifferentiable on a GPU under determinism.
The J/psi block goes from 2627 MB of card to 1.1 GB in memory.

Gate (`fullscale/gate_jacdense.py`): the same card loaded twice, `dense=False`
and `dense=True`, compared at the terms' own defaults on both chunk loops.
J/psi term, eager loop: **value exactly 0**, gradient 2.6e-15, HVP 1.6e-15.

---

## 10. PHASE 3 — THE CARD BUILDER LANDED (2026-09-08)

`make_joint_card.py --material` (commit `4c592f6`, with
`rabbit-vmass` `216c548`) is what sec. 4's "what is left of phase 3" asked for:
BOTH mass terms are `MaterialCFTerm`s over the 42 parmtype-15
`material_<group>` parameters the quadratic hit-chi2 term and the sparse `D`
already carry, plus 18 new `hitres_<class>` from the mass terms alone. Each
material amount now appears in THREE places in one likelihood — the quadratic
curvature, the mass-term width, the mass-term mean.

### THE COMMAND THAT BUILDS THE FULL PHASE-3 CARD

```bash
FS=/work/submit/david_w/ZMass/calibration_studies/fullscale
Z=/work/submit/david_w/ZMass/calibration_studies/zchannel
GRP=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/data/materialGroups50.txt
RABBIT=/work/submit/david_w/ZMass/rabbit-vmass THREADS=8 $FS/run_tf.sh python3 -u \
  $FS/make_joint_card.py --material \
  --jpsi-pairs $FS/runs/gpairs_v2_n50.npz \
  --z-pairs    $FS/runs/gzpairs_dyv2_n50.npz \
  --quad $FS/runs/quad_jpsiv2_ok.npz $FS/runs/quad_dyv2.npz \
  --groups $GRP --whiten --shape 5 \
  --fsr $Z/data/kern_loose_band3.3e-4.npz --acc $Z/data/acc_loose_d8.json \
  --chunk 16384 -o $FS/cards/joint_mat_v3.hdf5
```

`RABBIT=` is NOT optional: `run_tf.sh` defaults to `rabbit-material`, which has
no fixed norm families and would refuse the Z term. Budget ~100 GB of RAM: the
group blocks are ~36 GB as numpy, the term holds a tf copy of the leg being
built, and the default `--verify` read-back holds a second copy of both terms
(`--no-verify` drops that last one, at the cost of the round-trip assertion).
`--chunk 16384`, not phase 2's 32768: the per-chunk intermediate is now
`chunk x 25.7 groups x 64 tau x 8 B` per family component, i.e. 215 MB x 5 x
(forward + backward) ~ 2 GB, against `chunk x 64` before. The chunk is frozen
at WRITE time (sec. 5, the `_jac_chunks` caveat), so the fit must use the same
16384. Then, as an ordinary rabbit fit:

```bash
rabbit_fit.py $FS/cards/joint_mat_v3.hdf5 -o out/ -t 0 --unblind \
  --paramModel ExternalParams bundle:global_params \
  --minimizerMethod trust-exact \
  --freezeParameters material_pp1_cables material_support_tube material_thermal_screen
```

`--minimizerMethod trust-exact` is NOT optional either -- it is sec. 0f.20's
decision, and the default is `trust-krylov`, which stopped 14.7 NLL units above
the reference at full statistics on the Z-only card. **But it wants a dense
Hessian, and this card is where that hurts**: the Fitter assembled the smoke
card's 117 x 117 Hessian from HVP columns in **862 s** at 40 k candidates and
chunk 8192. The cost is ~linear in the candidate count at fixed chunk, so the
full card (1.127 M candidates, 28x) is **~6.7 h per Hessian on this CPU** --
sec. 2's problem, inherited, with 14 more parameters than phase 2. That is the
number the GPU queue has to be planned against, not the card build.

### SIZE, MEASURED AND EXTRAPOLATED

| | J/psi | Z |
|---|---|---|
| candidates after the cuts | 645 517 / 651 672 (99.06 %) | 481 020 / 487 742 (98.62 %) |
| group CSR rows | 14 832 301 (22.98/cand) | 12 365 835 (25.71/cand) |
| group exponents (5 families x 64 tau, float32) | 19.0 GB | 15.8 GB |
| hit CSR + sparse D | 0.67 GB | 0.49 GB |

**~36 GB card.** That is what the model costs: 30 kB per candidate is the
per-group CF block, and it is the same number sec. 4 quotes for the caches.
A 20 k + 20 k smoke card is 0.99 GB and builds in 15 s.

**`--group-prune` is NOT a safe size lever, measured.** Folding small rows into
the fixed baseline is exact at k = 0 (they go into `S^fix`), but it removes the
group's FREEDOM. At `--group-prune 0.001` — 5 % of the J/psi rows, 17 % of the
Z rows — `material_beampipe` keeps **0.02 %** of its leverage on the Z leg, and
beampipe is one of the four the hit-chi2 term is blind to, i.e. the mass terms
are its only constraint. Prune only with a per-group leverage table in hand.

### GATES

**(a) the rabbit path — `test_rabbit_path.py --card cards/joint_mat_smoke.hdf5
--model "ExternalParams bundle:global_params"`** (that script now splits the
model spec, so the bundle name reaches `load_model`; it could not before):

```
1. graph chunk loop vs the eager python loop, per term, at 2 points
     jpsi   value 0.000e+00   grad 3.20e-15   hvp 2.88e-16   OK
     zmass  value 0.000e+00   grad 2.66e-15   hvp 6.58e-16   OK
2. the term vs the standalone objectives, every parameter free
     jpsi  vs ChunkedObjective        value 1.1e-16  grad 5.9e-16  hvp 2.3e-16  OK
     jpsi  vs DeviceChunkedObjective  worst 6.2e-19                             OK
     zmass vs ChunkedObjective        value 1.8e-16  grad 1.7e-14  hvp 4.7e-14  OK
     zmass vs DeviceChunkedObjective  worst 3.0e-14                             OK
3. the Fitter (117 fit parameters, hvp revrev, Hessian route = hvps)
     loss_val vs loss_val_grad                                     OK
     gradient . p: analytic -3.0814345e+11, central FD the same, rel 1.98e-12  OK
```

4. the Hessian route: check 4 reported **MISMATCH 7.167e5**, and the test is
   at fault, not the Hessian. It compares the Fitter's Hessian BLOCK on one
   term's parameters against THAT TERM'S standalone `ChunkedObjective.hess`,
   which is right for the single-term Z card it was written for and wrong for
   any joint card: the block also carries the other mass term (they share 110
   parameters) and the `hitchi2` external quadratic. The evidence is in its
   own output -- the Fitter Hessian's largest eigenvalue, **7.243e13**, is to
   four digits the largest eigenvalue of the external quadratic alone
   (**7.2433e13**, recomputed from the two `quad_*.npz`, whitened, x 1/2),
   while the mass terms live at ~1e8. The negative eigenvalue it also prints,
   -7.7e-4, is -1e-17 of the largest, on one of the four known `hitchi2` null
   directions, and its EDM 3.2e9 is a random point, not a minimum. Nobody had
   seen this because check 4 had never RUN on a joint card -- the model spec
   could not be parsed until `4e81bed`.

`gate_joint_hessian.py` (new) makes the statement that is well posed on a
multi-term card, without assembling any Hessian:

```
=== 1. H p vs the central difference of the Fitter's gradient ===
  eps 1e-05  1.711e-13     eps 1e-06  1.123e-12     eps 1e-07  1.065e-11
=== 2. H p vs sum(term HVPs) + K p + p/sigma^2 ===
  jpsi 110 params, zmass 117, hitchi2 92 (max|K/2| 7.1162e+13), 42 constrained
  max |H p - (sum of the pieces)| / max|H p| = 1.507e-16
```

i.e. the whole route -- both `MaterialCFTerm`s, the external quadratic, the
Gaussian constraints -- reproduces the derivative of its own gradient at the FD
floor (**1.7e-13**) and IS exactly the sum of its pieces (**1.5e-16**).
`test_rabbit_path.py` now SKIPS the block comparison when other terms share the
parameters, with the reason, instead of reporting a mismatch.

`rabbit_fit.py` itself was run on the smoke card and gets past loading and into
the minimiser -- it declares 117 fit parameters through
`ExternalParams bundle:global_params`, loads both terms as MaterialCF with the
graph chunk loop, honours `--freezeParameters material_pp1_cables
material_support_tube material_thermal_screen`, and reports the same
`hitchi2` singularity note sec. 2 already documents (4 null directions, the
pseudo-inverse is used and absolute NLL values are offset).

**(b) at k = 0 the material card IS the phase-2 card** —
`gate_material_k0.py`, on two cards built from the same caches with the same
`--*-maxn 20000 --seed 1234`, so the candidates are identical:

| | J/psi | Z |
|---|---|---|
| resolution exponent Re S, rel | **6.94e-08** | **7.35e-08** |
| resolution exponent Im S, rel | 6.98e-08 | 4.49e-08 |
| Gaussian share `v`, abs | **0.0 (exact)** | **0.0 (exact)** |
| NLL, rel | 3.34e-09 | 1.68e-11 |
| gradient of the 50 field modes + Z POIs, rel | 1.86e-06 | 4.65e-08 |
| per-candidate density, median / q99 / max rel | 2.1e-09 / 2.3e-06 / 1.9e-04 | 1.0e-09 / 1.0e-07 / 9.0e-07 |

It is NOT exactly equal, and the reason is one number: the group rows and the
flat rows are both stored as float32, and their sum differs by the float32
floor — **6.9-7.4e-8 relative on the exponent**, which is gate (c)'s number
arriving through a completely different path. Everything else agrees: the
Gaussian share to 0.0 exactly (see below), and the truncation normalisation by
construction (it is the same class exponents with coefficient 1). The gradient
of a field mode is a near-cancelling sum over candidates, so the same relative
density perturbation shows up ~30x amplified there on the J/psi leg; that is a
property of the sample, not of the parameterisation.

Gate (b) was re-run after `rabbit-vmass` `35b9394` (the dense `D`, sec. 9) and
returns the SAME numbers to every digit: the phase-3 card stores `D` in the
same `jac_indices`/`jac_values` datasets, so it inherits that fix -- and the
GPU-determinism failure it fixes -- automatically, and `jac_dense` is a
constructor option that is not written to the card.

**(c) sum-of-groups closure** — `matres/validate_inmaker_groups.py --maxn
200000` on the two phase-3 caches, re-run today:

| family | J/psi rel | Z rel |
|---|---|---|
| `grp_ms` | 7.28e-08 | 7.54e-08 |
| `grp_io_re` | 5.19e-08 | 5.84e-08 |
| `grp_io_im` | 6.80e-08 | 4.88e-08 |
| `grp_rad_re` | 6.01e-08 | 6.70e-08 |
| `grp_rad_im` | 6.50e-08 | 7.48e-08 |

both PASS; the maker's own float64 `cfmass_grp_closure` is 1.38e-14 (J/psi) and
1.04e-14 (Z). Sec. 4's 7.5e-8 is confirmed, on the full caches rather than a
sample of them.

### THREE THINGS THAT CONTRADICT OR REFINE WHAT SEC. 4 SAYS

1. **`--jpsi-fang` is not needed, and never was for these caches.**
   `runs/gpairs_v2_n50.npz` carries a per-candidate `fang` (median **6.19e-02**
   over the cache, 6.08e-02 over the selected 20 k), so BOTH legs are
   truth-free from `Jpsi_covrefmom` with no scalar to scan. The card now prints
   which source each leg used and records it in `provenance.fang_source`.
   The Z leg's `fang` median is 9.2e-05, i.e. negligible there as expected.

2. **`vg_other` must NOT be clipped at 0.** Sec. 4 says "`MaterialCFTerm`
   should clip `vg_other` at 0 rather than trust its sign". Measured on the
   full caches, `vg_other + sum_c v_c - vgf` is **exactly 0.0** — the maker
   DEFINES `vg_other` as the remainder, so it is algebra, not accumulation
   noise. It is negative for **40.2 %** of J/psi and **36.6 %** of DY
   candidates (median |vg_other|/vgf = 3.2e-07, max 2.3e-04). Clipping would
   ADD up to 2.3e-04 of `vgf` to those 40 % and break the k = 0 identity that
   gate (b) rests on. `--clip-vg-other` exists and is OFF by default. The real
   risk clipping was aimed at — a negative TOTAL Gaussian variance — cannot
   happen at eps = 0 (the total is `vgf > 0`) and would need a large negative
   `hitres`; the guard belongs on the total, in rabbit, and is not implemented.

3. **`material_thermal_screen` is a boundary case, not a clear "under 0.1 %".**
   Full-cache occupancy is 2.18e-04 (J/psi) and **9.86e-04** (DY) — just under
   0.1 % on the DY leg, and 1.05e-03 on the 20 k smoke subsample, i.e. it
   crosses the threshold with the sample. The census therefore reports the
   numbers and uses a 1 % threshold for "thin", intersected with the
   hit-chi2 null space, to decide what is unconstrainable.

### THE CENSUS THE CARD PRINTS (requirement 5)

On the 20 k + 20 k smoke card:

```
  === what constrains the 42 material amounts ===
    group                             jpsi occ         z occ  hit-chi2
    material_beampipe                 0.999950      0.999950     BLIND
    material_thermal_screen           0.000200      0.001050     BLIND
    material_support_tube             0.000200      0.000600     BLIND
    material_pp1_cables               0.000000      0.000000     BLIND
    touched by NO candidate on any leg: ['material_pp1_cables']
    the hit-chi2 Hessian is blind to:   ['material_beampipe', ...]
    NOTHING in this card constrains: material_pp1_cables material_support_tube
    -> add to the fit:  --freezeParameters ...
```

`material_beampipe` is fine — 99.995 % of candidates on both legs touch it, so
the mass terms carry it even though the quadratic term does not. The card does
NOT freeze anything itself; which parameters a fit floats is a physics decision
and belongs in the fit command. At full statistics the freeze list is
`material_pp1_cables material_support_tube material_thermal_screen`.

The card also reports the per-group LEVERAGE (share of `sum_i max_tau |S|`),
which is what says whether a constrained group matters: on the J/psi leg
`tec_structure` 26.3 %, `tib_support` 16.5 %, `tob_support` 11.5 %; on the Z leg
`tib_support` 28.1 %, `tec_structure` 15.1 %, `tob_support` 11.8 %.

### WHAT IS NOT DONE

* **The truncation normalisation does not respond to the material amounts.**
  `_norm_z` sums over resolution CLASSES, which have no per-group
  decomposition. `rabbit-vmass` `216c548` adds FIXED norm families
  (coefficient 1) so `Z_c` is evaluated at the production's own resolution —
  which is EXACTLY the constant phase 2 evaluates it at, since phase 2 runs
  with `--fix k_hit k_ms k_ioni k_rad`, so nothing is lost RELATIVE to phase 2.
  What is missing is the second-order response `dZ_c/dk`. Sec. 5 pitfall 8
  already measured the cost of the neighbouring approximation (the norm at the
  stored class sigma) at 0.016 MeV on `m_Z`.
* **`--inject` on a material amount is refused in `--material` mode.** A true
  extra amount of material moves the mean (injected through `D`) AND scales the
  group's resolution exponents by `A(dtheta)` (not injected).
  `matres/make_material_card.py` has that scaling; porting it is what an
  injection closure on a material parameter needs. Injecting a FIELD mode is
  unaffected and still works.
* **The minimiser problem of sec. 2 is inherited, with 60 more parameters**
  (117 in the fit vector against phase 2's 103). Nothing here changes it.
* No Asimov/toy pulls, no group-leader table, no full-scale fit — deliberately
  left to the GPU queue.

### 0f.25 PHASE 2 COST, AND THE TWO-STAGE PLAN

`joint_ok_full.hdf5` is 6 682 662 candidates over 95 free parameters (50 field
modes + 38 material groups + `m_Z`, `Gamma_Z`, 5 `K(m)`; the 4 resolution
knobs and the 4 unconstrained material groups frozen). Measured scaling from
the Z-alone card (113 chunks of 32768, 11 Hessian columns, ~1.7 min per
`trust-exact` iteration on an H200 => ~9 s per HVP column), the joint card at
816 chunks of 8192 and 95 columns is **~50 min per `trust-exact` iteration**,
i.e. ~32 h for a 38-iteration fit. Not affordable.

**So phase 2 is two-stage, both stages through `rabbit_fit.py`:**

1. **`P2K`** — `--minimizerMethod trust-krylov` (scipy, `hessp` only): O(10)
   HVPs per iteration instead of 95 columns, ~3-10 min per iteration. It gets
   to the neighbourhood.
2. **`P2X`** — `--minimizerMethod trust-exact --externalPostfit <P2K snapshot>`:
   a few iterations to certify, and the one full Hessian the covariance needs
   anyway (~50 min).

Stage 1 is not trusted on its own — sec. 0f.18 is exactly the record of krylov
stopping early and reporting convergence — which is why stage 2 exists and why
the quoted EDM is stage 2's.

**Freezing, named**: `k_hit k_ms k_ioni k_rad` (the MC truth) and
`material_beampipe material_thermal_screen material_support_tube
material_pp1_cables`. The last four are the directions the hit-chi2 Hessian is
exactly blind to. Note that `material_beampipe` is touched by 99.99 % of the
mass candidates, so freezing it is CONSERVATIVE — the mass terms could
constrain it and this fit does not let them. `material_pp1_cables` is touched
by no candidate on either leg and is the only one that is unconstrained by
everything. `Fitter.warn_unconstrained` (rabbit `54e47f6`) names anything else
whose Hessian diagonal is below 1e-12 of the largest, rather than letting the
frozen-diagonal fix make an unmeasured parameter look measured.

### 0f.26 RESUME POINT (2026-09-08 14:50) — what is running and what to do next

**One command re-makes everything**: `./collect.sh --summary` (rsyncs Engaging,
re-dumps the rabbit results, prints the certified table). Add
`--needs-sandwich` for the rows still owing a sandwich pass.

| what | where | state |
|---|---|---|
| the certified table | `certtable.py`, `collect.sh` | 10 of 19 rows certified; the rest are the jobs below |
| the v-form rows | `22312979` (SVs6 SVs7 SVetaT SVetaE SVtoy SVKetaB SVKetaT) | running, SVs6 converged at EDM 2.0e-22 |
| the m-form rows | `22312980` (Ss6 Ss7 SMetaB SMetaT SMetaE Stoy Stoydc Sw70110) | queued |
| the warm controls | `22312981` (f380refW SVfullW Sdc8W) | f380refW **DONE and PASSED**; SVfullW at EDM 5.5e-6 |
| the cold controls | `22311742 f380refS`, `22311743 SVfull` | running, ~20 iterations, EDM ~8e3; `22311744 Sdc8` was cancelled to free a slot |
| the preconditioning control | `22313540 f380refP` | running — if it converges quickly it is the answer for phase 2's 95 parameters as well |
| the extended `K(m)` ladder | `22314340` (Ss9 SVs9 Ss12 SVs12) | queued; cards built and staged |
| **phase 2** | `22312984 P2K` (`joint_ok_full`, `trust-krylov`), then `P2X` from its snapshot | queued; `22314264 P2smoke` re-queued after the dense-`D` fix |
| **phase 3** | the card build, pid on submit, `logs/card_joint_mat_v3.log` | running (~36 GB, hours) |

**Do next, in order:**
1. `./collect.sh --summary` and re-run `plot_closure.py`.
2. `./submit_sandwich.sh $(python3 certtable.py --summary --needs-sandwich | tail -1 | cut -d: -f2)`.
3. When `P2K` lands, submit `P2X` warm from `rabbit_P2K.snapshot.hdf5` with
   `METHOD=trust-exact` — that is the number to quote and the covariance.
4. Stage `cards/joint_mat_v3.hdf5` and fit it the same two-stage way.

**A refinement to the phase-2 freeze list, measured by the phase-3 card's own
census (sec. 10) and NOT yet acted on**: `material_beampipe` has **99.995 %
occupancy on both legs**, so the mass terms do constrain it and freezing it is
conservative — it costs information. Only `material_pp1_cables`,
`material_support_tube` and `material_thermal_screen` are unconstrained by
everything. The running phase-2 job freezes all four (the instruction's
"four"); a 3-frozen variant is the cross-check worth having.

### 0f.27 THE v-FORM INCLUSIVE CLOSURE, CERTIFIED — AND IT IS A CANCELLATION

`22312981 SVfullW`: `z_V_full`, scipy `trust-exact` through `rabbit_fit.py`,
warm from the krylov point.

| | `m_Z` | `Gamma_Z` | NLL | EDM |
|---|---:|---:|---:|---:|
| `V_full` / `Rvfull` (krylov, STALLED) | -0.000 +- 2.32 | +0.004 +- 4.19 | -9827099.7568 | 1.98 |
| `RvfullX` (`tf-trust-exact`, converged) | -1.53 +- 2.08 | -0.80 +- 3.80 | -9827069.1806 | 1.6e-12 |
| **`SVfullW` (scipy, converged)** | **-1.54 +- 2.08 (H)** | **+6.81 +- 3.80 (H)** | **-9827101.7477** | **6.4e-12** |

**This settles sec. 0f.15 on `z_V_full`.** There ARE two stationary points and
`RvfullX`'s is the WORSE one, 32.6 NLL units above; `SVfullW`'s is 2.0 units
BELOW the krylov stall. So the campaign minimiser found the better minimum on
the one card where `tf-trust-exact` had found a different one, and the "EDM
certifies stationarity, not optimality" caveat is real but resolvable by
comparing NLL, exactly as sec. 0f.16 requires.

**The v formulation takes the inclusive `m_Z` closure from -11.06 +- 2.27 to
-1.54 +- ~2.3** — 0.7 sigma from zero, against 4.9 sigma. The mechanism of
sec. 0e/0f is confirmed at full statistics and with a converged fit. The
earlier `-0.0002 +- 2.32` was a fit that never took a POI step; the real number
is -1.5, not 0.

**BUT the inclusive closure is a CANCELLATION between `eta` bands**, and that
is the finding that matters. Certified so far, v form:

| band | `m_Z` | n |
|---|---:|---:|
| `\|eta_lead\| < 0.9` | **-21.08 +- 3.24** | 1 572 534 |
| `0.9 - 1.6` | **+12.39 +- 4.16** | 1 084 704 |
| `1.6 - 3.0` | (+34.22 +- 5.47, NOT yet converged) | 1 025 424 |
| inverse-variance mean of the three | **-0.80** | |
| the inclusive fit | **-1.54 +- 2.08** | 3 682 662 |

i.e. the inclusive number is the average of a spread of ~55 MeV, and it is
small because the barrel and the endcap pull in opposite directions, not
because the bias is gone. **The v formulation fixes the INCLUSIVE closure and
leaves an `eta`-dependent residual as large as the effect it removed.** A Z
mass measurement that fits `eta` bands separately, or that weights them
differently from this MC, does not inherit the inclusive closure.

### 0f.28 QUEUE AUDIT (2026-09-08 15:05, at David's request)

Every job in the queue was submitted by THIS agent after 14:00. **Nothing was
left over from the account swap**: the previous agents' jobs are `22292576-87`
(the `--gtol 0` re-runs), `22301214-20`, `22303682-4` and `22304228`, and all
of them COMPLETED before 13:45. The ~25 cancellations between 14:20 and 14:31
were mine: I submitted the 19-row suite one job per row, then re-submitted it
in priority order, then re-submitted it again warm-started and BATCHED (a warm
row converges in 3-5 iterations but pays ~5 min of card load and TF start-up,
so one process per row spent the 4-GPU allowance on start-up).

| job | card / variant | purpose | keep |
|---|---|---|---|
| `22311742 f380refS` | `z_full380_fl`, COLD | the cold minimiser control | **CANCELLED** — `f380refW` (warm) already returned the reference EXACTLY: `m_Z` -11.06, NLL 11075392.4657, EDM 1.78e-18, in 2 iterations. The cold run was at EDM 8e3 after 57 min and tests only "no other minimum from a cold start", which no local method proves anyway |
| `22311743 SVfull` | `z_V_full`, COLD | cold twin + the sec. 0f.15 two-minima question | **CANCELLED** — `SVfullW` settled it: NLL -9827101.7477 at EDM 6.4e-12, BELOW both the krylov stall (-9827099.7568) and `RvfullX` (-9827069.1806). The two points are real and the better one is found |
| `22313540 f380refP` | `z_full380_fl`, COLD, `--precondition` | is the trust-radius scaling the cure? | **CANCELLED** — answered: EDM 7210 -> 8351 -> 476464, i.e. it wanders exactly as the unpreconditioned cold run does. Preconditioning buys nothing here, confirming sec. 0f.14 for the scipy subproblem too |
| `22312979 zrabbitvb` | v-form rows: `SVs6 SVs7 SVetaT SVetaE SVtoy SVKetaB SVKetaT` | step 2's deliverable | **KEEP** |
| `22312980 zrabbitvb` | m-form rows: `Ss6 Ss7 SMetaB SMetaT SMetaE Stoy Stoydc Sw70110` | step 1's table | **KEEP** |
| `22312981 zrabbitvb` | `f380refW SVfullW Sdc8W` | the warm controls | **KEEP** — on its last row |
| `22312984 P2K` | `joint_ok_full`, `trust-krylov` | **phase 2**, stage 1 | **KEEP** |
| `22314264 P2smoke` | `joint_ok_n500k`, `trust-exact` | phase 2's path check after the dense-`D` fix | **KEEP** — cheap, and it must pass before the 1.5-day job burns time |
| `22314340 zrabbitvb` | the extended `K(m)` ladder `Ss9 SVs9 Ss12 SVs12` | the `Gamma_Z` truncation question | **CANCELLED and queued in this plan instead** — it is the lowest-priority row set and it was holding a pending slot against the GPU cap. Resubmit with `FRESH=1 ./submit_scipy_suite.sh` (or the batch sbatch) when `22312979`/`22312980` finish; the cards are built and staged |
| local `rabbit_fit.py cards/joint_mat_smoke.hdf5` | phase-3 smoke | requirement-6 demonstration | **KILLED** — it had already shown the card loads, declares 117 parameters through the bundle and enters `trust-krylov`; it was 65 min of CPU on a node at load 180 |

Cancelling the three frees three of the four `mit_preemptable` GPUs, which is
what `22312984 P2K` and `22314264 P2smoke` were blocked on.

**Rule adopted**: at most 4 `mit_preemptable` + 2 `mit_normal_gpu` jobs live at
a time, and rows are BATCHED (`rabbit_vmass_batch.sbatch ROWS="a b c"`) rather
than one job per row. Anything beyond the cap is written into this plan
instead of held pending.

### 0f.29 THE COMMON-p HYPOTHESIS FOR THE eta PATTERN — MEASURED AND REFUTED
### (2026-09-08, at the coordinator's request)

**The hypothesis.** The card uses ONE `p = 1.264` for two different jobs: the
conditioning label `k_i = sigma_i/m_i^p` (a labelling choice — any
mass-independent `p` will do) and the convolution variable (physics, whose
answer is `1 + f_i` per candidate with `f_i = vgf_i`). With a common `p` the
model mis-scales each candidate's width by `(m/m_i)^{p-(1+f_i)}` — a
MASS-DEPENDENT width error whose sign is the sign of `p-(1+f_i)`, and a
mass-dependent width error biases `m_Z`. If `vgf` ran 0.22 (barrel) to 0.37
(endcap) the sign would flip between them, which is the observed pattern.

**Measured, on the m-form band cards themselves** (median `vgf`,
inverse-variance-weighted `sigma/m`):

| band | `vgf` med | q25 | q75 | `1+f` | `p-(1+f)` | `sigma/m` |
|---|---:|---:|---:|---:|---:|---:|
| `\|eta\|<0.9` | 0.2104 | 0.1703 | 0.2760 | 1.2104 | **+0.0536** | 0.00968 |
| `0.9-1.6` | 0.1925 | 0.1544 | 0.2478 | 1.1925 | **+0.0715** | 0.01247 |
| `1.6-3.0` | 0.2696 | 0.2021 | 0.4176 | 1.2696 | **-0.0056** | 0.01510 |
| inclusive | 0.2166 | 0.1702 | 0.2985 | 1.2166 | +0.0474 | 0.01121 |

**`vgf` does NOT run 0.22 -> 0.37 across the bands.** It runs 0.21 -> 0.19 ->
0.27, and the MIDDLE band is the lowest. So `p-(1+f)` has the SAME sign in the
barrel and the middle band (both positive, the middle one larger) and is
essentially ZERO in the endcap — the ordering is not the observed one and there
is no sign flip between barrel and endcap to be had.

**And the size is an order of magnitude short.** `proto_vmass.py --bands` and
`gate_pmismatch.py` run the same quadrature that produced the sigma-vs-k
prediction — truth `Int p(m') N(m_i - m'; k m'^{1+f}) dm'` with the candidate's
own `f`, model in the common-`p` variable, 5-term Legendre `K(m)` floated:

| band | predicted from the mismatch | **observed (certified)** |
|---|---:|---:|
| `\|eta\|<0.9` | **+0.67** | **-21.08 +- 3.24** |
| `0.9-1.6` | **+1.28** | **+12.39 +- 4.16** |
| `1.6-3.0` | **-0.13** | ~+34 (converging) |
| inclusive | +0.74 | -1.54 +- 2.08 |

and the scan over the whole range the sample spans says why it cannot be
rescued by the spread within a band: the bias is **linear in `p-(1+f)` at
about 22 MeV per unit**, so even at `vgf = 0.10` and `vgf = 0.50` — well
outside the q25-q75 of any band — it only reaches **+3.6 and -5.2 MeV**. The
`v matched` control in the same table is 0.00 MeV, so the machinery is
sensitive; it is the effect that is small.

**Conclusion: the common-p mismatch is a <= 1.3 MeV effect at the band medians
and <= 5 MeV at the extreme tails. It cannot make the 55 MeV `eta` spread.**
The separation into an `f`-class axis (~8 classes each convolved in their own
`v_f`) is therefore NOT worth building for this: it would be a large piece of
machinery, with its own gates, for at most a couple of MeV. It stays on the
list as a sub-MeV refinement, not as the explanation.

**What the residual DOES track, and the test that separates it.** The observed
per-band `m_Z` is monotone in `sigma/m` (-21.1 at 0.0097, +12.4 at 0.0125,
~+34 at 0.0151 — about +10 MeV per 0.001 of `sigma/m`), and in this sample
`sigma/m` and `|eta|` are nearly collinear (barrel q25-q75 0.0087-0.0119
against endcap 0.0136-0.0215, almost no overlap). So "the `eta` pattern" and
"a residual `sigma/m` pattern" are the same measurement so far, and `sigma/m`
is the variable the v substitution was designed to decorrelate — which points
at the substitution's COEFFICIENT rather than its exponent.

The discriminating test is a `sigma/m` split AT FIXED `eta`:
`build_srsplit.sh` builds `z_V_etaB_slo/shi` (the barrel, cut at its own median
`sigma/m` = 0.01032) and `z_V_etaE_slo/shi` (the endcap at 0.01658). If `m_Z`
moves strongly between the two halves of one band, the residual is a resolution
effect and `eta` is only its proxy; if it does not, it is genuinely `eta`.

### 0f.30 THE v-FORM K(m) LADDER IS FAR BETTER BEHAVED THAN THE m-FORM ONE

Certified (scipy `trust-exact` through `rabbit_fit.py`, EDM < 1e-3, NLL the
best known for each card):

| terms | m form `m_Z` | m form `Gamma_Z` | **v form `m_Z`** | **v form `Gamma_Z`** |
|---|---:|---:|---:|---:|
| 5 | -11.06 +- 2.27 | -5.26 +- 4.16 | **-1.54 +- 2.08** | **+6.81 +- 3.80** |
| 6 | -13.98 +- 2.22 | +27.16 +- 4.35 | **-3.92 +- 2.13** | **+14.28 +- 3.75** |
| 7 | -17.24 +- 2.27 | +8.96 +- 4.42 | **-3.87 +- 2.28** | **+12.86 +- 4.35** |
| span | 6.2 | **32.4** | **2.3** | **7.5** |

**The substitution stabilises the `K(m)` truncation as well as the closure.**
In the m form `Gamma_Z` swings 32 MeV over the ladder and `m_Z` drifts 6.2 MeV
monotonically; in the v form the spans are 7.5 and 2.3 MeV, both comparable to
the statistical error. So the m-form ladder's instability was not a property of
the Legendre basis alone — a large part of it was the shape absorbing the
mass-dependent width error the m form leaves behind, which is exactly what
`K(m)` is the wrong tool for. The 15.2/13.0 sigma significance of terms 6 and 7
is still there and still says the 5-term shape is insufficient; what changes is
that in the v form the POIs no longer move with it.

### 0f.31 THE eta PATTERN IS IN BOTH FORMULATIONS — IT IS NOT THE SUBSTITUTION

With every band now converged (scipy `trust-exact` through `rabbit_fit.py`,
EDM < 1e-3, NLL the best known for each card):

| band | n | **m form** | **v form** | `sigma/m` |
|---|---:|---:|---:|---:|
| `\|eta_lead\| < 0.9` | 1 572 534 | **-26.60 +- 2.87** | **-21.08 +- 3.24** | 0.0097 |
| `0.9 - 1.6` | 1 084 704 | **+3.87 +- 3.88** | **+12.39 +- 4.16** | 0.0125 |
| `1.6 - 3.0` | 1 025 424 | **+16.55 +- 4.71** | **+34.22 +- 5.47** | 0.0151 |
| span | | **43.2** | **55.3** | |
| inverse-variance mean | | -9.5 | -0.80 | |
| the inclusive fit | 3 682 662 | **-11.06 +- 2.27** | **-1.54 +- 2.08** | |

**The m-form band numbers of sec. 0b were all unconverged** (-15.96 / +0.01 /
+7.46, POI steps of 4.0 / 1.1 / 2.2 sigma) and every one of them moves away
from zero once converged, the sec. 0f.19 pattern for the sixth time.

**Read together, this is the cleanest statement of what the v substitution does
and does not do.** It removes ~9.5 MeV of the INCLUSIVE offset
(-11.06 -> -1.54) and leaves the `eta` spread essentially untouched
(43.2 -> 55.3 MeV, if anything slightly larger). In both formulations the
inclusive number is just the inverse-variance mean of the bands, so:

* the `eta`/`sigma-m` structure is **NOT** a property of the v formulation —
  it is there in the m form at the same size and with the same sign structure,
  and it was hidden before only because those band fits had not converged;
* the v substitution's achievement is real but narrower than
  "the -11 MeV closes": it re-centres the band average, it does not flatten
  the bands.

**And the per-band FSR kernel is not the cause either.** `VK_etaB` = -17.54
+- 2.87 against `V_etaB` = -21.08 (+3.5 MeV) and `VK_etaT` = +17.24 +- 3.93
against `V_etaT` = +12.39 (+4.9 MeV): giving each band the FSR kernel measured
on its own selected candidates moves it by ~4 MeV in the SAME direction in both
bands, against a 33 MeV difference BETWEEN the bands. The kernel is worth its
4 MeV and is not the pattern.

### 0f.32 THE KERNEL-FREE DETECTOR TEST IS **NOT** FLAT PER BAND — the pattern
### is on the DETECTOR side (2026-09-08, at the coordinator's request)

`--residual-mode`: `m_reco - m_gen` against the per-candidate resolution CF,
delta kernel, no FSR fold, no acceptance, no `K(m)`. `--maxn 700000` per band,
`--floor-scale 1e-7` (see below), fitted through `rabbit_fit.py` with
`trust-exact` and then re-evaluated with `fit.py --start-from` for the sandwich
— which took **nit = 0** on every row, i.e. rabbit's point was already the
minimum.

| split | `sigma/m` | n | **kernel-free `alpha`** | EDM | the full fit, same band (v form) |
|---|---:|---:|---:|---:|---:|
| `0.9 < \|eta\| < 1.6` | 0.0125 | 693 553 | **+11.99 +- 1.51** | 1.6e-15 | **+12.39 +- 4.16** |
| `sigma/m` tertile 1 | 0.0094 | 699 359 | **-5.06 +- 1.05** | 2.4e-24 | — |
| `sigma/m` tertile 2 | 0.0123 | ~696 000 | **+4.80 +- 1.31** | 2.2e-08 | — |
| inclusive (297 k, sec. 0b) | 0.0123 | 297 557 | +0.88 +- 2.12 | — | -1.54 +- 2.08 |

**With NO kernel at all the detector half shows the pattern**: `+11.99 +- 1.51`
on the middle `eta` band is 7.9 sigma from zero, and it reproduces that band's
FULL v-form closure (`+12.39 +- 4.16`) to within its own error. The
`sigma/m` tertiles are monotone, `-5.06 -> +4.80`, i.e. **+3.4 MeV per 1e-3 of
`sigma/m`** on the detector side alone.

**So sec. 0b's `+0.88 +- 2.12` "the detector half CLOSES" was a CANCELLATION**,
exactly like the v-form inclusive closure: bands of opposite sign averaging to
zero. It was never evidence that the detector model is right — only that its
errors average out on this MC's `eta` composition.

**This is the coordinator's first branch**: a coefficient error proportional to
`sigma/m` gives a bias linear in `sigma/m`, which is what the tertiles show, so
the `a_res` / Jensen / substitution coefficients are wrong as a function of
`sigma/m` and **the fix is on the detector side**, not in the kernel, the
acceptance or `K(m)`. The `sigma/m` split at fixed `eta`
(`z_V_eta{B,E}_s{lo,hi}`, `22316192`) and the per-band `K` ladder remain
useful, but they are no longer where the cause is expected to be.

**Still running**: `residB`, `residE`, `residShi` (the machine is at load ~180;
they are past their minimiser and in the postfit). Their numbers complete the
picture and the barrel is the one to watch — the full fit gives -21.1 there.

**The floor-scale trap, recorded because it produced no error message.** At
`make_card.py`'s default `--floor-scale 1e-9` the residual cards give
`NLL = inf` before any minimiser runs: three candidates in 695 757 have a
NEGATIVE density from CF ringing (`li` = -5.7e-5, -2.4e-5, -1.9e-5), all
large-`sigma` (2.3-2.4 GeV), near-pure-Gaussian (`vgf` 0.91-0.94) candidates at
~4 sigma, and `s*softplus(li/s)` with `s = 1e-9` is exactly 0 there. rabbit
reports it as *"diagnostics unavailable: SVD did not converge"* followed by
*"Minimizer raised: array must not contain infs or NaNs"* and then writes a
result file at the unmoved starting point — the sec. 0f.19 failure shape again.
The scale is SET, not chosen: `s*softplus(li/s)` underflows once `li/s < -745`
and distorts the density by >1 % once `li < 4.6 s`, so `5.7e-5/745 = 7.7e-8 < s`
and `4.6 s` must stay below the densities that matter. **`s = 1e-7`.**
`s = 1e-4` was tried first and is wrong — it inflates everything past ~4 sigma
and blew `alpha` up to +71 +- 25 MeV against an expected +-1.4.
`--floor clip` is no help either: `max(li, 0) = 0` gives `log 0` as well.

### 0f.33 `a_i` MEASURED IN TRUTH CELLS — the per-leg term is REFUTED, and the
### real coefficient error is the MISSING IONISATION TERM (2026-09-08)

**The derivation** (mine, to be read against the coordinator's). Two legs with
independent relative momentum fluctuations `eps_l`, `Var(eps_l) = r_l^2`;
`d ln m = (eps_1 + eps_2)/2 + (angular)`, so `(sigma_m/m)^2 = R^2/4 + A` with
`R^2 = r_1^2 + r_2^2` and `f_ang = A/(sigma_m/m)^2`. `sigma_fit` is evaluated at
the FITTED momenta, and `sigma_m = m * sqrt(R^2/4 + A)` with `A` inert, so

    d ln sigma_fit = d ln m + (1/2) (1 - f_ang) d ln R^2
                   = d ln m + (1 - f_ang) sum_l s_l e_l eps_l ,
    s_l = r_l^2/R^2 ,  e_l = d ln r_l / d ln p_l = f_hit,l - f_ioni,l .

Conditioning on `x = (m_reco - m_gen)/sigma`,
`E[eps_l | x] = 2 s_l (1 - f_ang) (sigma_m/m) x` (the `(1-f_ang)` is there
because only `R^2/4` of the mass variance comes from the momenta), and
`E[d ln m | x] = (sigma_m/m) x`. Hence, with `a = d ln sigma_fit/dx`,

    **a = (sigma_m/m) [ 1 + 2 (1 - f_ang)^2 sum_l s_l^2 e_l ]**

Symmetric legs (`s_l = 1/2`, `f_ang = 0`) give `1 + e`, the spec's `1 + vgf`
(and `e = vgf` exactly when ionisation is dropped: `r^2 = (h p)^2 + m_0^2` has
`d ln r/d ln p = (h p)^2/r^2` = the hit share). Fully asymmetric legs give
`1 + 2 e_1`. **This differs from the coordinator's placement of `f_ang`**: the
leading `1` comes from `sigma_m ~ m_fit` and carries NO `(1-f_ang)`, while the
`e` term carries it TWICE. At the Z it does not matter -- `f_ang` has median
**8.4e-5** -- so the whole `f_ang` question is a J/psi-only issue (median 6.2e-2
there).

**`a` is measurable with no fit**, and `measure_a.py` measures it: bin, regress
`ln sigma_fit` on `z = (m_reco - m_gen)/sigma` inside the cell, the slope IS
`a`. 3 481 415 candidates, `|z| < 2` (the linear regime), MiNNLO weights.
`s_l` comes from `sigrelp`/`sigrelm`, which ARE in the pairs cache -- the
per-leg split did not need the hit blocks after all.

| cell | `sigma/m` | `vgf` | `asym` | **MEASURED `a/(sigma/m)`** | spec `1+vgf` | per-leg |
|---|---:|---:|---:|---:|---:|---:|
| inclusive | 0.0141 | 0.263 | 0.218 | **1.2110 +- 0.0004** | 1.2625 | 1.3196 |
| `\|eta\|<0.9` | 0.0088 | 0.274 | 0.062 | **1.2503 +- 0.0006** | 1.2738 | 1.2909 |
| `0.9-1.6` | 0.0119 | 0.205 | 0.177 | **1.1667 +- 0.0003** | 1.2046 | 1.2407 |
| `1.6-3.0` | 0.0180 | 0.301 | 0.314 | **1.2725 +- 0.0004** | 1.3007 | 1.3950 |

(`asym = 2(s_1^2 + s_2^2) - 1`, 0 symmetric to 1 fully asymmetric; with
`e_l ~ e ~ vgf` the per-leg column is `1 + vgf (1 + asym)(1 - f_ang)^2`.)

**1. THE PER-LEG TERM IS REFUTED.** The measured `a` is BELOW `1 + vgf` in
every cell, and the leg-asymmetry term moves `a` UP -- the wrong direction. It
is **1.7 to 4.3 times further from the measurement than the spec's own form**,
worst exactly where it is largest (the endcap: measured 1.273, spec 1.301,
per-leg 1.395). Binned on `asym` alone the discrepancy `meas - spec` runs
-0.148, -0.206, -0.247, -0.150, **-0.044** over the quintiles -- it SHRINKS
where `asym` is largest. There is no leg-asymmetry signal in `a`.

**2. THERE IS A REAL COEFFICIENT ERROR, AND IT IS THE IONISATION TERM.** The
measured exponent is `e_meas = a/(sigma/m) - 1` = 0.211 inclusively against
`vgf = 0.263`, i.e. **`e_meas = 0.80 vgf`** (0.91 / 0.81 / 0.91 in the three
bands). That is exactly `e = f_hit - f_ioni` with `f_ioni ~ 0.1-0.2 f_hit` --
the OTHER ingredient of the coordinator's formula, and the one the spec
dropped when it set `f = vgf`. It is a 100-sigma effect on the inclusive row.

**It also explains the J/psi.** There `a_fluct` was measured at 0.0107 against
`sigma/m = 0.0112`, a ratio of **0.955 < 1** -- below even the leading term --
which `1 + vgf = 1.086` cannot produce and which the spec's own sec. 1 recorded
as "the closed form runs 16 % HIGH". With `e = f_hit - f_ioni` and the J/psi's
much larger `f_ioni` (soft muons, `sigma_pT/pT` ionisation term ~ 1/pT), a
NEGATIVE `e` is exactly what is expected.

**3. BUT IT IS NOT THE eta PATTERN.** `meas - spec` is -0.024 / -0.038 / -0.028
across the three bands: roughly CONSTANT, not monotone, and nothing like the
-21 / +12 / +34 MeV it would have to generate. So correcting `a` is worth doing
-- it is a real, measured, first-principles error with no free parameter -- but
it is not what makes the bands differ.

**A caveat on the finer cells.** Binning in `sigma/m` (or in `asym`, which
correlates with it) CONDITIONS on `sigma`, and `sigma = sigma_bar(1 + a x)`, so
such a bin is a cut on `x` and attenuates the very slope being measured. The
`eta` bands and the inclusive row are clean (`eta` does not respond to the
residual); the `sigma/m` x `asym` grid in `measure_a.py`'s output is attenuated
and must not be read as a measurement of `a`.

### 0f.34 (1) THE MEAN DOES NOT CLOSE BAND BY BAND — and its pattern is NOT the
### fit's pattern, which is the skew signature (2026-09-08)

`<m_reco - m_gen>` against the ONLY mean shift the model predicts (the exact
Jensen term `1.5 s^2 m`; the CF's own mean is zero by construction). MiNNLO
weights, `sigma_m/m < 0.10`.

| band | n | `<m-m_gen>` raw | truncated `\|z\|<4` | model Jensen | **gap** | `sigma/m` |
|---|---:|---:|---:|---:|---:|---:|
| `\|eta\|<0.9` | 703 365 | -9.30 | +16.22 +- 1.04 | 11.01 | **+5.21** | 0.0088 |
| `0.9-1.6` | 1 271 998 | -124.44 | +27.11 +- 1.09 | 20.38 | **+6.73** | 0.0121 |
| `1.6-3.0` | 1 737 771 | -26.20 | +39.23 +- 1.44 | 54.79 | **-15.56** | 0.0180 |
| inclusive | 3 713 134 | -56.87 | +30.60 +- 0.79 | 34.48 | **-3.88** | 0.0142 |

(MeV. The RAW means are useless — the mid band's -124 MeV shows how completely
the untruncated mean is owned by whichever few candidates radiated — which is
exactly why the odd-moment statistic `<z e^{-uz^2}>` exists. The truncated
column is the meaningful one and it IS truncation-dependent; the gap should be
read as "does the location close", not as a calibrated number.)

**The location does NOT close band by band**: +5.2 / +6.7 / -15.6, a 22 MeV
spread, non-monotone in `eta`.

**And its pattern is NOT the fit's pattern.** The kernel-free `alpha` on the
middle band is +11.99 +- 1.51 and the full v-form band closures are
-21.1 / +12.4 / +34.2; the endcap's mean gap is **-15.6 where its fitted bias
is +34.2** — the opposite sign. A location bias that does not follow the mean
is precisely the coordinator's point: **the MLE location is pulled by the ODD
part of the density, not by its mean**, so a mean that misses and a fit bias
that misses differently is the signature of a SKEW mismatch rather than a
mis-centred density.

### WHAT (2)-(5) NEED, AND THE ONE PIECE OF PLUMBING THAT IS MISSING

The family attribution the coordinator asks for -- the odd closure and the
kernel-free `alpha` with the radiative family OFF and with the ionisation
family OFF -- cannot be run today:

* `make_card.py --del-family` is NOT a family remover: it ADDS the delta-ray
  family (`discover_families(keys, want_del)` appends `"del"`);
* neither `fit.py` nor `rabbit_fit.py` can SET a parameter to a non-default
  value and freeze it there -- `--fix` / `--freezeParameters` fix at the
  default, and `--start-from` only seeds FREE parameters. `k_rad = 0` removes
  the radiative block, but there is no way to ask for it.

**The cheapest route needs no code at all**: `discover_families` only appends a
family whose `re_k` is present in the cache, so a card built from a cache with
`Srad_re`/`Srad_im` deleted simply HAS no radiative family. So the recipe is
`np.savez_compressed` of the band subsets with those keys dropped, then the
ordinary `--residual-mode` build. Doing it on the band subsets rather than the
full 4.6 GB cache keeps it inside the /work quota, which is the binding
constraint (two 36 GB builds already died there today).

The alternative, and the better one if these switches are wanted repeatedly, is
a `--set NAME=VALUE` in `fit.py` and `make_card.py` that writes the value into
`param_defaults` -- three lines, and it makes `k_rad = 0`, `k_ioni` scans and
the single-`r` location-only fit all one flag.

**Also still owed from this thread**: `e = f_hit - f_ioni` in the card builder
(the measured 2-4 % correction of sec. 0f.33, `f_ioni` per candidate from
`Sio_re`/`Sio_im`/`ioni_sign_fixed` in the cache), and item (4), the per-`eta`
odd-moment closure on the tight-stepper 20-60 GeV gun at TRACK level
(`cf_skew_closure.py --cache runs/cf_trackres_mugun_ul16_*`), whose inclusive
1.05 ratio was never checked per `eta` and which would say whether the CF
family model itself is wrong per region.

### 0f.35 ITEM (4): THE TRACK-LEVEL SKEW CLOSURE **IS** eta-DEPENDENT, AND THE
### RADIATIVE FAMILY IS NOT THE CAUSE (2026-09-08)

`cf_skew_closure.py --bin-eta --nbins 3` (new: `make_bins(d, "abseta", 3)` uses
the Z analysis's own leading-muon edges 0 / 0.9 / 1.6 / 2.4, so a track-level
number sits next to a mass-level one with no re-binning). Tight-stepper 20-60
GeV muon gun, `cf_trackres_mugun_ul16_260903x_m0_k0.npz`, truth-referenced
`sigma_bar`, 319 854 tracks, bootstrap 200.

**`<z e^{-u z^2}>` DATA - MODEL, per band:**

| band | n | u = 0.05 | u = 0.2 |
|---|---:|---:|---:|
| `\|eta\| 0.0-0.9` | 120 063 | **-0.0050 +- 0.0025** | **-0.0036 +- 0.0018** |
| `0.9 - 1.6` | 93 615 | -0.0036 +- 0.0029 | -0.0024 +- 0.0022 |
| `1.6 - 2.4` | 106 176 | **+0.0001 +- 0.0027** | **+0.0007 +- 0.0020** |
| TOTAL | 319 854 | -0.0029 +- 0.0011 | -0.0018 +- 0.0011 |

**Monotone in `\|eta\|`: -0.0050 -> -0.0036 -> +0.0001.** The barrel misses
closure by 2.0 sigma and the ENDCAP CLOSES EXACTLY. The barrel-to-endcap trend
is +0.0051 +- 0.0037 (u=0.05) and +0.0043 +- 0.0027 (u=0.2) -- 1.4-1.6 sigma
each, but the same sign and the same ordering at every probe, on 320 k tracks.
**The inclusive 1.05-style number hid this**: the total, -0.0029, is the
average of a band that misses and a band that does not.

**The radiative family is NOT the cause.** `--krad 0` changes the model column
by <= 1e-5 and leaves `data - model` identical to four decimals in every band.
On a 20-60 GeV gun the radiative block contributes essentially nothing to the
odd moment, so it cannot carry an `eta` dependence.

**And the model has almost NO skew here at all**: its own `<z e^{-uz^2}>` is
+0.00005 / -0.00000 / +0.00000 across the bands, against a DATA value of
-0.0049 in the barrel. So the gap is not "the model's skew is mis-sized", it is
**"the data is skewed and the model is not"** -- in the barrel. By elimination
the only remaining odd channel in the model is `Im S_ioni`, which the model
puts at ~0 at these momenta.

**The single ionisation scale cannot close it, and it wants OPPOSITE signs.**
The `k_hat` that would close each band runs **-2.00 / +6.00 / +6.00** -- the
barrel pinned at the scan's lower limit and the other two at its upper limit.
One number cannot do it, which is what "the family model is wrong per region"
looks like from inside the fit.

**This is the first-principles finding the coordinator was after**: the CF
family model's odd content is wrong per detector region at TRACK level, before
any mass likelihood, any kernel and any pair. The fix belongs in the model
tables (Moliere / Urban / radiative content against material and
`E = pT cosh eta`), not in the likelihood. And the sign matches the mass-level
pattern -- the barrel is the most negative in both.

**Statistics caveat**: 2.0 sigma per band. The gun cache is what it is; a
larger one, or the same test on the J/psi and Z legs, would settle the
significance. The MONOTONICITY across three bands and two probes is what makes
it worth acting on, not any single band's pull.

### 0f.36 THE ATTRIBUTION TABLE — charge, phi, hit composition (2026-09-08)

`resolution/attribute_skew.py`, tight-stepper 20-60 GeV gun, 319 854 tracks,
6 219 371 hits with their 18-class labels and their per-hit influence weights
`hitamp2`. The model's own odd moment is ~0 in every band (sec. 0f.35), so
these DATA numbers are `data - model` to 1e-5. Units 1e-3, bootstrap 200.

**1. CHARGE — and this is the headline: the odd moment is dominated by a
charge-ODD part that the charge-average removes BY CONSTRUCTION.**

| band | `q = +1` | `q = -1` | **even = (+ plus -)/2** | **odd = (+ minus -)/2** |
|---|---:|---:|---:|---:|
| `\|eta\| 0.0-0.9` | -13.96 +- 3.72 | +4.12 +- 3.81 | **-4.92** | **-9.04** |
| `0.9-1.6` | -16.11 +- 3.67 | +8.98 +- 3.77 | **-3.57** | **-12.55** |
| `1.6-2.4` | -13.96 +- 3.78 | +14.21 +- 3.79 | **+0.13** | **-14.09** |

The coordinator's reading is CONFIRMED in its structure: the charge-averaged
number IS the charge-even part, it IS the `eta`-dependent piece
(-4.9 -> -3.6 -> +0.1), and the endcap's "exact closure" is a genuine ABSENCE
of the even part rather than a cancellation. What was not visible before is
that it rides on a charge-ODD component **two to three times larger**
(-9 -> -14, nearly flat in `eta`) which is the ionisation energy loss --
`delta(q/p)` carries the sign of `q`, so a one-sided energy loss is charge-odd
in `z` -- and which `cf_skew_closure --charge 0` cancels deliberately.

**2. PHI — the Lorentz-drift prediction FAILS.** A hit displacement in a fixed
LOCAL direction rotates with the module and so has a definite GLOBAL sense; the
coordinator's test is that the sign must not flip with `phi`. It flips:

| band | `phi` [-pi,-pi/2] | [-pi/2,0] | [0,pi/2] | [pi/2,pi] |
|---|---:|---:|---:|---:|
| `\|eta\| 0.0-0.9` | -3.93 +- 5.06 | -13.53 +- 4.66 | -12.83 +- 4.78 | **+10.58 +- 4.52** |
| `1.6-2.4` | +5.61 +- 4.93 | +1.85 +- 5.13 | +2.45 +- 5.03 | **-9.28 +- 5.34** |

The barrel charge-even skew runs -3.9, -13.5, -12.8, **+10.6** -- a sign flip
at 2.3 sigma against the neighbouring octants, and the same octant is the
outlier in the endcap with the OPPOSITE sign. That is not what a uniform local
drift does. It is a phi-modulated, global-sense effect.

**3. HIT COMPOSITION — one positive indication, and it is the single-strip
class.** Shares are INFLUENCE-weighted (`sum_{hits in c} hitamp2 / sum hitamp2`),
which is the quantity the mass functional's per-hit-class blocks carry.

| band | single-strip share LOW | HIGH | difference |
|---|---:|---:|---:|
| `\|eta\| 0.0-0.9` | -1.72 +- 3.30 | -8.03 +- 7.58 | **-6.3 +- 8.3** |
| `0.9-1.6` | -2.81 +- 3.75 | -11.20 +- 8.68 | **-8.4 +- 9.5** |
| `1.6-2.4` | +3.58 +- 3.56 | -9.12 +- 8.28 | **-12.7 +- 9.0** |

Tracks whose curvature is carried by **single-strip (N1) clusters** are more
negatively skewed in ALL THREE bands, same sign, ~1 sigma each (~1.6 sigma
combined). That is the one measurement pointing at a hit class. **The PIXEL
share does not**: it is non-monotonic (-1.8 / -9.8 / -2.9 across its own
tertiles in the barrel), so "scales with the barrel-pixel share" is not
supported.

**Where this leaves the hypothesis.** Partially supported and partially
refuted, and the two halves are separable:
* the charge-EVEN structure is real, `eta`-dependent, and absent in the endcap
  -- as predicted;
* the single-strip class carries extra negative skew in every band -- as
  predicted, at ~1.6 sigma;
* but the `phi` behaviour is NOT a fixed local drift, and the pixel-share
  scaling is absent. So "Lorentz drift in the barrel pixels and strips" as the
  specific mechanism is not what the data shows.
* and the DOMINANT odd feature is charge-ODD ionisation, 2-3x larger, which
  every charge-averaged closure in this campaign has been cancelling away
  without saying so.

**Before the fix of item (3) is built** -- replacing the Gaussian per hit class
in the CF product by the class's measured residual density -- the thing to
settle is the single-strip indication at more than 1.6 sigma, and the `phi`
structure, which no hit-class mechanism as stated predicts.

### 0f.37 (A) RECONCILED: the charge-ODD part WAS the pull-normalisation
### artefact; the charge-EVEN part survives the transform UNCHANGED (2026-09-08)

The coordinator was right and the check is exact. `attribute_skew.py` now
applies `oddmoment/track_truthfree.py`'s transform,

    x_i = z_i / (1 - a_i q_i z_i),
    a_i = sigma_rel,i (1 - vgf_i),  sigma_rel,i = sigma_i p_fit,i ,

all from observed quantities. On this cache **`a_i` has median 0.01432**, so
the artefact it removes is `<q z> = -a = -14.32e-3`.

**That is precisely what I reported as "charge-odd physics": -9.04 / -12.55 /
-14.09.** It was the pull-normalisation artefact -- `sigma_fit` is larger for
the fluctuation that made it larger -- and it is RETRACTED.

**With the transform applied:**

| band | `q = +1` | `q = -1` | **even** | **odd** |
|---|---:|---:|---:|---:|
| `\|eta\| 0.0-0.9` | -8.80 +- 3.71 | -0.97 +- 3.81 | **-4.88** | -3.92 |
| `0.9-1.6` | -7.35 +- 4.02 | +0.20 +- 3.73 | **-3.58** | -3.78 |
| `1.6-2.4` | -3.34 +- 3.77 | +3.62 +- 3.86 | **+0.14** | -3.48 |

* the charge-ODD part collapses from -9...-14 to **-3.5...-3.9, flat in
  `eta`** -- the modelled ionisation skew, and consistent with the oddmoment
  study's `<qz> = -0.0022 +- 0.0018` closing against a CF model of -0.0021
  (the damping by `e^{-uz^2}` and the band split account for the rest);
* the charge-EVEN part is **-4.88 / -3.58 / +0.14 against -4.92 / -3.57 /
  +0.13 before the transform** -- identical to 0.04e-3, under 1 % of itself.

**So the coordinator's second worry -- that the artefact's second-order pieces
leak into the charge-even part because `a` varies with `eta` -- is measured and
is negligible.** The `eta`-dependent charge-even signal is real, it is not a
normalisation artefact, and every conclusion of sec. 0f.35 and 0f.36 about the
charge-EVEN part stands unchanged.

**One consequence for the tooling**: `cf_skew_closure.py` does NOT apply the
transform -- there is no `1/(1 - a q z)` anywhere in it -- so its numbers are
on the RAW `z`. For a charge-AVERAGED closure that is harmless, because the
charge-even part is invariant under the transform to 1 %, which is now
measured rather than assumed. It is NOT harmless for anything charge-split,
and `--charge +1/-1` in that script is therefore reporting the artefact. That
should be fixed before the script is used per charge.

### 0f.38 (1) `--charge` FIXED, (2) THE DY GROUPS CACHE HAS THE HIT BLOCKS, AND
### (3) THE CLASS BANK IS CENTRED — it cannot predict a location bias

**(1) `cf_skew_closure.py --charge` is fixed.** `truth_ref_z(d)` implements
`x = z/(1 - a q z)` with `a_i = sigma_rel,i (1 - vgf_i)` and is applied
AUTOMATICALLY whenever `--charge` separates the charges; `--truth-ref` forces
it for a charge-averaged run and `--raw-pull` reproduces the pre-fix behaviour
with a warning saying what it is measuring. Verified: the run now logs
*"truth-referenced pull applied: a_i median 0.01432, so the pull-normalisation
artefact removed is `<q z> = -a = -14.32e-3`"*. Charge-averaged numbers are
unchanged by construction (the default is off there, and the even part is
invariant to 1 %).

**(2) The DY groups cache carries the hit blocks.**
`runs/gzpairs_dyv2_n50.npz` (17.0 GB, 487 742 candidates) has
`hit_cls`, `hit_v`, `hit_ptr`, `hit_classes` -- the same builder as the J/psi
one -- plus `sigrelp/sigrelm`, `etap/etam`, `ptp/ptm`, `mgen`, `w`. So (B)'s
class-share half and (C) can both be done on the ACTUAL Z legs at ~3x the gun's
statistics, with no new production. **Per-CHARGE is not available there**: it
is a two-track mass cache and the pair carries both charges, so the charge
split stays a track-level (gun / single-track) measurement.

**(3) THE CLASS BANK CANNOT MAKE THE NO-FREE-PARAMETER PREDICTION AS IT
STANDS, and the reason is a deliberate design choice.** `hitres_classes.py`
stores `log phi_c(s)`, the COMPLEX empirical characteristic function of the raw
per-hit pull -- so the odd part is there in `Im log phi_c`. But line 103:

```python
    x = v[keep] - med       # centred: the CF's mean is a bias,
    # and a per-hit bias belongs to alignment, not to the resolution model
```

**every class is centred on its own median, and the median is not stored**
(`meta` keeps `n`, `var`, `core`, `trimmed` and nothing else).

The coordinator's mechanism -- a hit displacement in a fixed local direction --
is precisely a per-class LOCATION bias. **The bank is blind to exactly the
effect proposed**, by construction. What it can predict is a per-class SKEW
about the median, which is a different thing.

So item (C) as posed needs the per-class MEDIANS re-extracted from the
260807 / 260829 / 260831 hit-class productions. And the physics is worth
stating: a per-hit bias belongs to alignment only if alignment can absorb it,
and alignment fits MODULE POSITIONS, not per-cluster-class offsets. A CPE bias
that depends on cluster class (edge, single-column, N1 strip) is NOT absorbed
by a rigid module shift, so it survives into the track fit as a charge-even
curvature bias -- which is the hypothesis. The comment in the bank is an
assumption, and it is the assumption under test.

### 0f.39 (C) ON THE ACTUAL Z LEGS: the odd moment tracks the PIXEL influence
### share, and in the endcap at 4.7 sigma (2026-09-08)

`resolution/attribute_skew_mass.py` on `gzpairs_dyv2_n50.npz` -- **487 742 Z
candidates, 6 234 742 hits** with their 18-class labels and per-hit influence
weights `hit_v`, MiNNLO weights, truth-referenced `x = z/(1 - a z)` with the
MEASURED `a = 1.211 sigma/m` of sec. 0f.33 (not the spec's `1 + vgf`). Units
1e-3.

**A. per `\|eta\|` band** (the DATA odd moment; the model side is not
subtracted here, so read the TREND, not the offset):

| band | n | u = 0.05 | u = 0.2 |
|---|---:|---:|---:|
| `\|eta\| 0.0-0.9` | 91 718 | +15.58 +- 3.03 | +10.28 +- 2.49 |
| `0.9-1.6` | 164 185 | +15.00 +- 2.56 | +8.18 +- 1.71 |
| `1.6-3.0` | 225 875 | +10.86 +- 2.03 | +4.72 +- 1.61 |
| inclusive | 481 778 | +13.20 +- 1.45 | +6.98 +- 0.98 |

**B. hit composition — the pixel influence share, and it is the biggest effect
in this study:**

| band | pixel share LOW | pixel share HIGH | difference |
|---|---:|---:|---:|
| `\|eta\| 0.0-0.9` | +9.87 +- 6.21 | +11.31 +- 6.11 | +1.4 +- 8.7 |
| `0.9-1.6` | +5.93 +- 3.87 | +13.00 +- 3.91 | +7.1 +- 5.5 |
| **`1.6-3.0`** | **-1.27 +- 3.87** | **+23.94 +- 3.73** | **+25.2 +- 5.4 (4.7 sigma)** |

and the single-strip share:

| band | N1 LOW | N1 HIGH | difference |
|---|---:|---:|---:|
| `\|eta\| 0.0-0.9` | +10.04 +- 5.08 | +15.09 +- 9.83 | +5.1 +- 11.0 |
| `0.9-1.6` | +6.51 +- 3.36 | +14.70 +- 8.63 | +8.2 +- 9.3 |
| `1.6-3.0` | +8.60 +- 2.89 | -1.68 +- 7.08 | -10.3 +- 7.6 |

**THE RESULT.** In the endcap, Z legs whose curvature is carried by PIXEL hits
have an odd moment of **+23.9 +- 3.7** and those that are pixel-poor have
**-1.3 +- 3.9** -- a **4.7 sigma** difference, and the dependence is monotone
in `eta` (+1.4 barrel, +7.1 middle, +25.2 endcap). This is the first thing in
the whole attribution above 2 sigma, and it is a HIT-COMPOSITION variable, not
a material or field one.

**It supports the coordinator's class of mechanism and contradicts their
specific one.** The prediction was that the forward pixel disks are CLEAN
(drift parallel to B) and the barrel carries the effect. Measured, it is the
other way round: the barrel is flat in pixel share and **the endcap pixel hits
carry the skew**. So the effect is in the pixel CPE, and it is the FORWARD
disks -- where the incidence angle is large and shallow, the clusters are long
in the local `y`, and the template/generic CPE has the least support -- rather
than the barrel Lorentz drift.

**The single-strip lead reverses sign in the endcap** (+5.1, +8.2, **-10.3**),
so the two composition variables are not the same effect. At track level on the
gun the N1 effect was negative in all three bands; here it is positive in the
barrel and middle and negative in the endcap. It stays a ~1 sigma indication.

**CAVEAT, and it must be checked before this is quoted as a class effect**: the
pixel influence share correlates with the hit count, with `pT` and with `eta`
WITHIN a band. A 4.7 sigma split on a correlated variable can be a proxy for
any of them. The next step is the same split at fixed `pT` and fixed hit count,
and the per-class breakdown (`pix_x_q*` against `pix_y_q*` -- the local `x` and
`y` classes, which is exactly where a forward-disk incidence-angle effect must
show up as `y` and not `x`).

**And (C) as posed still cannot be closed**: the no-free-parameter prediction
needs the per-class residual DENSITIES including their location, and
`hitres_classes.py` centres every class on its median and does not store it
(sec. 0f.38). The medians must be re-extracted from the 260807/260829/260831
productions before the propagation through `hit_v` can be done.

### 0f.40 THE CONTROLS: the 4.7 sigma is SUBSTANTIALLY A pT PROXY, and the odd
### moment is dominated by a pT dependence nothing had accounted for (2026-09-08)

**C1. the pixel-share split AT FIXED leading-muon pT, in `1.6 < \|eta\| < 3.0`:**

| pT bin | pixel LOW | pixel HIGH | difference |
|---|---:|---:|---:|
| [0, 36] GeV | -46.54 +- 5.82 | -8.97 +- 6.37 | +37.6 +- 8.6 |
| [36, 46] | -51.08 +- 6.66 | -54.41 +- 5.75 | **-3.3 +- 8.8** |
| [46, inf] | +103.93 +- 6.22 | +119.71 +- 5.55 | +15.8 +- 8.3 |

The three differences are **inconsistent with each other** (chi2 = 11.1 / 2 dof,
p = 0.004) and the middle bin is consistent with ZERO. The inclusive +25.2 +-
5.4 becomes **+17.0 +- 4.9** once pT is held, and it is not a stable
coefficient.

**And look at the first two columns**: the odd moment itself runs
**-47, -51, +104** across the three pT bins -- a 150e-3 swing, **twenty times
any composition effect in this study**. That is a kinematic property of the Z
mass residual (which leg is hard sets the sign of the mass skew), it was in
none of the previous tables, and it means **any composition split that does not
control pT is largely measuring pT.** The earlier per-band and per-class
numbers (secs. 0f.36, 0f.39) all inherit that caveat.

**C2. the same split AT FIXED hit count** -- here it DOES survive:

| hit-count bin | pixel LOW | pixel HIGH | difference |
|---|---:|---:|---:|
| [0, 11] | -15.46 +- 7.36 | +28.12 +- 8.24 | +43.6 +- 11.0 |
| [11, 13] | +4.42 +- 6.02 | +15.99 +- 6.48 | +11.6 +- 8.8 |
| [13, inf] | +3.03 +- 5.76 | +27.76 +- 5.24 | +24.7 +- 7.8 |

weighted **+23.6 +- 5.4**, consistent across bins (p = 0.06). So it is not a
hit-count proxy; it is a pT proxy.

**D. the local coordinate -- the discriminating test, and it does NOT
discriminate.** A forward-disk incidence/drift effect on the tilted Phase-0
turbine blades must sit in ONE local coordinate (the drift direction):

| band | `pix_x` LOW -> HIGH | `pix_y` LOW -> HIGH |
|---|---:|---:|
| `0.9-1.6` | +7.53 -> +10.27 (**+2.7 +- 5.8**) | +5.92 -> +20.65 (**+14.7 +- 5.7**) |
| `1.6-3.0` | -0.04 -> +22.61 (**+22.7 +- 4.9**) | +0.02 -> +17.30 (**+17.3 +- 5.1**) |

In the endcap BOTH local coordinates carry it, `x` if anything more strongly;
only in the middle band is it `y`-dominated with `x` flat. That is not the
signature of a single drift direction.

**VERDICT.** The pixel-share lead is weakened, not dead: **+17 +- 5 survives at
fixed pT and +23.6 +- 5.4 at fixed hit count**, but it is not a stable
coefficient across pT and it is not localised in one local coordinate, which
were the two things that would have made it a CPE class effect rather than a
correlate. **The dominant unexplained feature is now the pT dependence of the
odd moment itself** (-47 / -51 / +104), which is 20x larger than anything
attributed so far and which every previous split in this thread was partly
measuring.

**So items (2)-(4) are NOT the next step.** Re-extracting the per-class medians
and building the no-free-parameter propagation is a large piece of work
motivated by a lead that the pT control has just cut in half and whose
local-coordinate signature has failed. The next measurement is the pT
dependence itself: it is kinematic in origin (the mass residual's skew depends
on the leg asymmetry, which is what sec. 0f.36's `s_1` measures) and it has to
be understood and removed before ANY composition variable can be read.

### 0f.41 THE MASS-LEVEL TABLES WERE RAW — corrected, and it changes the `eta`
### table but NOT the pT swing or the pixel split (2026-09-08)

**(1) Yes, they were raw.** Secs. 0f.39 and 0f.40 report the DATA odd moment
with no model subtracted. At TRACK level that is legitimate -- the model's odd
moment there is measured at +5e-5 (sec. 0f.35) -- but at MASS level the model
carries the exact Jensen mean shift `d_i = 1.5 s_i^2 m_i`, a displacement of
`1.5 (sigma_i/m_i)(1 + f_ang,i)` in the standardized variable, contributing

    model_odd_i(u) = 1.5 (sigma_i/m_i)(1 + f_ang,i) / (1 + 2u)^{3/2}

**The coordinator's estimate is exact**: at the endcap's `sigma/m = 0.018` and
`u = 0.05` that is **+23.4e-3**, and the measured model column is **+23.42**.

**(2) The corrected tables.** Model column = the Jensen term; `d-m` = data
minus it. NOT yet subtracted: the ionisation and radiative skews of the CF
itself, so `d-m` is `data - Jensen`, not the full `data - model`.

**Per `\|eta\|` band -- this DOES change, and the sense reverses:**

| band | data | model (Jensen) | **data - Jensen** |
|---|---:|---:|---:|
| `\|eta\| 0.0-0.9` | +15.58 +- 3.03 | +11.43 | **+4.16** |
| `0.9-1.6` | +15.00 +- 2.56 | +15.65 | **-0.66** |
| `1.6-3.0` | +10.86 +- 2.03 | +23.42 | **-12.56** |
| inclusive | +13.20 +- 1.45 | +18.42 | **-5.23** |

The raw reading was "+15.6 -> +10.9, mildly decreasing"; corrected it is
**+4.2 -> -12.6**, a much steeper decrease that crosses zero. So the `eta`
structure is REAL but the raw table understated it and got its zero point
wrong. Sec. 0f.39's per-band row is superseded.

**The pT swing -- this does NOT change:**

| pT bin (endcap) | data LOW / HIGH | model | data - Jensen LOW / HIGH |
|---|---:|---:|---:|
| [0, 36] | -46.54 / -8.97 | +22.2 / +20.8 | **-68.71 / -29.72** |
| [36, 46] | -51.08 / -54.41 | +25.4 / +23.9 | **-76.52 / -78.35** |
| [46, inf] | +103.93 / +119.71 | +29.5 / +26.9 | **+74.42 / +92.78** |

**The model is nearly FLAT in pT (+22, +25, +30)** because `sigma/m` barely
moves across the leading-muon pT range at the Z -- it is set by `eta`. So the
150e-3 data swing survives the subtraction essentially intact
(-69 / -77 / +74). **The Jensen term is not the pT swing.** The remaining model
piece is the ionisation and radiative skew, which is negative and largest at
low pT -- it can plausibly account for the -69 / -77 but NOT for the +74 at
high pT, which is a change of SIGN.

**And the pixel-share split does not change either** -- if anything it grows,
because the model is slightly SMALLER for pixel-rich candidates: the
differences go from +37.6 / -3.3 / +15.8 (raw) to **+39.0 / -1.8 / +18.4**
(data - Jensen). Still inconsistent across pT bins, still with the middle bin
at zero.

**(3) The judgement on the pixel lead is unchanged**: not a model artefact, not
a hit-count proxy, but not a stable coefficient across pT and not localised in
one local coordinate. And the dominant unexplained structure remains the pT
swing, now shown to be neither the Jensen term nor a composition effect.

**What is still owed for a complete `data - model`**: the ionisation and
radiative odd content of the per-candidate CF, integrated with the same
`e^{-u z^2}` weight and the same truncation. The arrays are in the cache
(`Sio_re/Sio_im/Srad_re/Srad_im/Sms/tgrid`) and the integral is
`weier_odd`'s; what it needs is the mass-level CF assembly, which lives in
`unbinned.MassCFTerm` rather than in `cf_skew_closure`'s track-level builder.
That is the next piece, and until it is done no mass-level odd moment should be
quoted as an attribution.

### 0f.42 THE pT SWING IS THE JACOBIAN-EDGE TRAP — and removing it RESTORES the
### pixel lead at 5.4 sigma (2026-09-08)

**The coordinator's diagnosis is confirmed, and it is not close.** My pT bins
were on the RECONSTRUCTED leading-muon pT, `max(ptp, ptm)`. Measured on the
same candidates:

| conditioning variable | `corr(., z)` |
|---|---:|
| **reco leading pT** `max(ptp,ptm)` | **+0.0421** |
| pixel influence share | +0.0141 |
| `m_gen` | -0.0143 |
| `\|eta\|` lead | -0.0081 |
| `eta_pair` | **-0.0001** |

and the smoking gun, `<z>` and `<m_reco - m_gen>` per reco-pT tertile (edges
40.1 and 48.0 GeV; the Jacobian peak is `m_Z/2 = 45.6`):

| tertile | n | `<z>` | `<m - m_gen>` |
|---|---:|---:|---:|
| [0, 40.1] | 158 929 | -0.0568 | -68.97 MeV |
| [40.1, 48.0] | 163 745 | -0.0429 | -58.69 MeV |
| **[48.0, inf]** | 158 929 | **+0.1020** | **+225.4 MeV** |

The high bin's candidates sit **225 MeV above their own gen mass on average**.
That is selection on the fluctuation, exactly as diagnosed: above the Jacobian
peak the spectrum falls steeply, so a `pT_reco` bin there is populated by legs
that fluctuated UP. **The -47 / -51 / +104 odd-moment swing of sec. 0f.40 is
the regression-to-the-mean / Jacobian-edge effect and is RETRACTED as physics.**
It is the same trap as binning on `sigma/m` (sec. 0f.33) and it is now the
third time this campaign has walked into a conditioning artefact.

**And removing it restores the pixel lead.** The same endcap pixel-share split
inside **gen-mass** tertiles (`m_gen`, `corr = -0.014`) instead of reco-pT
tertiles, `data - Jensen`:

| gen-mass tertile | pixel LOW | pixel HIGH | difference |
|---|---:|---:|---:|
| [0, 89.7] | -16.27 | -2.54 | **+13.7** |
| [89.7, 91.6] | -34.67 | -3.49 | **+31.2** |
| [91.6, inf] | -29.54 | +5.70 | **+35.2** |

all three positive, **consistent** (chi2 = 3.6 / 2 dof, p = 0.16) -- against
the reco-pT binning's chi2 = 11.1 / 2 dof, p = 0.004 -- and the weighted mean
is **+26.7 +- 4.9, i.e. 5.4 sigma**.

**So sec. 0f.40's verdict is superseded.** The pixel-share effect was not
"substantially a pT proxy"; the pT CONTROL was itself the confounded
measurement, and it was destroying a real effect rather than exposing a false
one. With a gen-safe conditioning variable the effect is larger, consistent
across bins, and 5.4 sigma.

**What remains true from sec. 0f.40**: the local-coordinate test still does not
discriminate (endcap `pix_x` +22.7 +- 4.9 against `pix_y` +17.3 +- 5.1, both
present), so a single drift direction is still not the signature; and the
pixel share carries a residual exposure of its own (`corr = +0.014`, a third of
the reco-pT's but not zero), so the consistency across gen-mass bins is the
check that carries the result, not the inclusive number.

**Still owed before this is an attribution rather than a correlation**: the
ionisation and radiative odd content of the mass CF, for a complete
`data - model` (only the Jensen term is subtracted above); and the same split
against a GEN leading-muon pT, which **is not in either pairs cache** -- `ptp`,
`ptm` are reconstructed (`corr = +0.04`) and there is no gen leg momentum. A
gen-pT split needs a re-extraction, and until then `m_gen` and `eta_pair` are
the safe kinematic handles.

### 0f.43 (1) COMPLETE `data - model` AT MASS LEVEL — the endcap misses at
### 7 sigma, and it is the PIXEL-POOR tracks that miss (2026-09-08)

`resolution/model_odd_mass.py` assembles the per-candidate mass CF exactly as
`unbinned.MassCFTerm._family_parts` does -- `tgrid` is the STANDARDIZED
argument (the Gaussian family enters as `-0.5 vgf t^2`), so

    log phi_z(t) = Sms + (Sio_re + i Sio_im) + (Srad_re + i Srad_im)
                        - 0.5 vgf t^2 + i t (1.5 (sigma/m)(1 + f_ang))

with every `k` at 1, and `<z e^{-uz^2}> = 1/sqrt(pi u) Int (t/2u) e^{-t^2/4u}
Im phi dt` on the term's own grid. **The model's own odd moment, decomposed:**

| band | full | Jensen alone | ioni+rad skew alone |
|---|---:|---:|---:|
| `\|eta\|<0.9` | +14.15 | +11.55 | +2.62 |
| `0.9-1.6` | +18.63 | +16.07 | +2.58 |
| `1.6-3.0` | +27.86 | +25.85 | +2.04 |

The Jensen term dominates and the CF skew adds a nearly flat +2.0...+2.6, so
sec. 0f.41's Jensen-only subtraction was ~90 % of the correction.

**A. per `\|eta\|` band, COMPLETE `data - model`:**

| band | data | model (full) | **data - model** |
|---|---:|---:|---:|
| `\|eta\| 0.0-0.9` | +15.58 +- 3.03 | +14.05 | **+1.53** (0.5 sigma) |
| `0.9-1.6` | +15.00 +- 2.56 | +18.20 | **-3.21** |
| `1.6-3.0` | +10.86 +- 2.03 | +25.41 | **-14.56 (7.2 sigma)** |
| inclusive | +13.20 +- 1.45 | +20.73 | **-7.54** |

**The barrel CLOSES and the endcap misses by 7 sigma.** This is the first
properly model-subtracted mass-level statement in the thread, and the
`eta` dependence is now unambiguous: +1.5 -> -3.2 -> -14.6.

**B. and the endcap split INVERTS once the model is subtracted:**

| endcap cut | data | model | **data - model** |
|---|---:|---:|---:|
| pixel share LOW | -1.27 +- 3.87 | +27.60 | **-28.88 (7.5 sigma)** |
| pixel share HIGH | +23.94 +- 3.73 | +25.78 | **-1.85 (0.5 sigma)** |
| single-strip LOW | +8.60 +- 2.89 | +22.32 | -13.73 |
| single-strip HIGH | -1.68 +- 7.08 | +32.43 | **-34.11** |

**It is not that pixel-RICH endcap tracks are anomalous -- they CLOSE
(-1.85 +- 3.73). It is that pixel-POOR ones fail, by -28.9 +- 3.9.** The raw
table read the opposite way because the model is large and nearly equal in the
two samples (+27.6 against +25.8): subtracting it moves the pixel-HIGH sample
onto zero and leaves the pixel-LOW sample stranded.

The single-strip variable says the same thing from the other side: tracks whose
curvature is carried by single-strip clusters miss by **-34.1** against -13.7
for the rest.

**So the coherent statement is: in the endcap, tracks with DEGRADED HIT
CONTENT -- few pixel hits, many single-strip clusters -- carry an odd moment
the model does not describe, and tracks with a normal hit complement close.**
That is a hit-content effect, it is 7 sigma, and it is now measured against the
model rather than against zero.

**What this does NOT yet say** is whether the mechanism is the CPE residual
density per class (the standing hypothesis) or the CF's hit term being wrong
for a degraded complement in some other way. The no-free-parameter test of item
(3) is exactly what separates them, and it needs the per-class residual
LOCATIONS, which `hitres_classes.py` discards (sec. 0f.38).

---

## HANDOFF: PER-CLASS RESIDUAL LOCATIONS  (written 2026-09-08, for a fresh agent)

**Everything needed is in this section.** You do not need the rest of this file
except sections 0f.35-0f.43, which are the measurements you are reproducing.

### THE QUESTION

The mass likelihood's odd content **closes in the barrel and misses in the
endcap at 7 sigma**, and the tracks it misses on are the ones with DEGRADED HIT
CONTENT. Complete `data - model` (sec. 0f.43), on the DY v2 Z legs, units 1e-3:

| endcap cut | data - model |
|---|---:|
| pixel influence share LOW | **-28.88 +- 3.87  (7.5 sigma)** |
| pixel influence share HIGH | **-1.85 +- 3.73  (closes)** |
| single-strip share HIGH | **-34.11** |
| single-strip share LOW | -13.73 |

The standing hypothesis is that the CPE has a **cluster-class-dependent
LOCATION bias** which a rigid module alignment cannot absorb (alignment fits
module positions, not per-cluster-class offsets), and which therefore survives
into the track fit. **Your job is the no-free-parameter test of it.**

### WHY IT CANNOT BE DONE WITH WHAT EXISTS

`resolution/hitres_classes.py: build_cf_bank` measures, per class, the log
characteristic function of the raw per-hit pull -- so the SHAPE and the SKEW are
there -- but line 103 is

```python
    x = v[keep] - med          # centred: the CF's mean is a bias,
    # and a per-hit bias belongs to alignment, not to the resolution model
```

**every class is centred on its own median and the median is discarded**
(`meta[c]` keeps only `n`, `var`, `core`, `trimmed`). That comment is the
assumption under test. You must re-extract the medians.

### THE INPUT

`build_cf_bank(subdir="hitres3", tag="mugun_lowpt")` reads, with `uproot`:

```
/ceph/submit/data/user/d/david_w/ZMass/cvh/{subdir}_{tag}/task_*/globalcor_resclosure_*.root
   tree "tree", branches:
     dxrecsim, dxerr      local-x residual and its CPE error
     dyrecsim, dyerr      local-y residual and its CPE error (PIXEL ONLY)
     hitDetId             subdet = (hitDetId >> 25) & 0x7 ; <= 2 is pixel
     hitUProj             the CPE's own independent variable (strip split at 0.25)
     clusterSizeX         strip cluster width N (clipped to 1..5)
     clusterChargeBin     pixel template charge bin (clipped to 0..3)
```

**Sign convention: `dxrecsim` is REC minus SIM in the LOCAL frame.** A positive
value is a hit reconstructed at larger local x than the simulated crossing. The
pull is `dxrecsim/dxerr`; the selection is `> -98` (the sentinel) and
`dxerr > 0`. `dyrecsim` is used only for `subdet <= 2`.

The 18 classes are `hitres_classes.CLASSES` --
`pix_{x,y}_q{0..3}` and `str_N{1..5}_{lo,hi}` -- and `class_of(subdet, N,
uproj, qbin, isy)` assigns them. **Use that function; do not re-derive the
boundaries**, they are shared with `cf_track_resolution.py` and `cf_inmaker.py`
and must not drift.

`prodfiles.resolve(pattern, nfiles)` returns only COMPLETE task outputs.
`/ceph` is **not readable from a Claude Code sandbox shell** (permission denied
at the top level); use `run_tf.sh --ceph` or run on a node with the bind.

### WHAT TO EXTRACT

Per class, and **additionally split by**:
* **local coordinate** -- `x` and `y` separately (they already are, in the class
  names, but the strip classes are x-only);
* **subdetector / layer / disk / +-z** -- decode `hitDetId`. Pixel barrel layer
  and forward disk+side are in the DetId; the endcap result above makes
  **FPix disk and side the first thing to look at**;
* the residual **in the track's bending sense**, not only in local coordinates
  -- a curvature bias is what propagates to `q/p`, and the local-to-global
  rotation is what turns a local-x bias into one.

and report, per cell: `n`, the **median** (the location bias, in units of
`sigma_CPE` and in microns), the core width, and the log CF as
`build_cf_bank` already does.

### THE PROPAGATION (no free parameter)

A hit's influence on the track's `q/p` is exported per hit:
* single-track caches: `hitamp2` (per hit, aligned by `hitcnt` / `cumsum`),
  e.g. `runs/cf_trackres_mugun_ul16_260903x_m0_k0.npz`;
* mass caches: `hit_v` with `hit_ptr` and `hit_cls`,
  e.g. `fullscale/runs/gzpairs_dyv2_n50.npz` (487 742 Z candidates) and
  `gpairs_v2_n50.npz` (651 672 J/psi).

The predicted odd moment of a track is then the class-weighted sum of the
per-class odd content propagated through those weights, integrated with the
SAME statistic the data uses:

    <z e^{-u z^2}> = 1/sqrt(pi u) Int_0^inf (t/2u) e^{-t^2/4u} Im phi(t) dt

(`cf_skew_closure.weier_odd`, and `resolution/model_odd_mass.py` shows the
assembly for the mass CF). **No fitted scale anywhere** -- that is the point of
the test.

### THE TARGET NUMBERS

**Track level, tight-stepper 20-60 GeV gun** (`cf_trackres_mugun_ul16_260903x_
m0_k0.npz`), charge-EVEN odd moment, truth-referenced pull, `u = 0.05`, 1e-3:

| band | charge-even |
|---|---:|
| `\|eta\| 0.0-0.9` | **-4.88** |
| `0.9-1.6` | **-3.58** |
| `1.6-2.4` | **+0.14** |

The model's own odd moment is ~0 there (+5e-5), so this IS `data - model`.

**Mass level, DY v2 Z legs**: the table at the top of this section.

### THE BINNING RULES — three artefacts have already been walked into

1. **Never bin on a RECONSTRUCTED kinematic variable.** `corr(reco leading pT,
   z) = +0.0421`, and the top reco-pT tertile sits **+225 MeV above its own gen
   mass** (the Jacobian peak is `m_Z/2 = 45.6` GeV). Sec. 0f.42. Safe handles
   measured on the DY cache: `m_gen` (`corr = -0.014`), `eta_pair`
   (`-0.0001`), `\|eta\|` lead (`-0.008`). `ptp`/`ptm` are RECO and there is no
   gen leg momentum in either pairs cache.
2. **Never bin on `sigma/m` or on `sigma`**: `sigma = sigma_bar(1 + a x)`, so
   such a bin is a cut on the residual and attenuates the slope. Sec. 0f.33.
3. **Always use the truth-referenced pull for anything charge-split**:
   `x = z/(1 - a q z)` with `a = sigma_rel (1 - vgf)`. Without it
   `<q z> = -a = -14.3e-3` on this gun and that is ALL you measure.
   `oddmoment/track_truthfree.py`; `cf_skew_closure.py` now applies it
   automatically under `--charge` (sec. 0f.38).
4. **At mass level the model's odd moment is NOT ~0** -- it is +14 / +18 / +25
   across the bands, dominated by the Jensen shift. Subtract it
   (`model_odd_mass.py`). Sec. 0f.43.

### IF IT REPRODUCES: THE IMPLEMENTATION PATH

1. **offline CF hit term** -- `cf_track_resolution.py`, the hit family. Today it
   consumes `hitres_classes.build_cf_bank`'s centred `log phi_c`; it must take
   location + shape. Re-run the TRACK-level closure per `eta`
   (`cf_skew_closure.py --bin-eta --nbins 3`): **it must go flat**.
2. **`cf_inmaker.py`** -- the per-hit-class blocks it writes into the pairs
   caches (`hit_cls`, `hit_v`) are what the mass functional consumes; the
   location has to travel with them.
3. **the maker export path** -- the per-class blocks in the CVH producer, for
   the mass functional at production time.
4. then the **kernel-free per-band mass closure**
   (`fullscale/build_resid_bands.sh`, `run_resid_bands.sh` -- note
   `--floor-scale 1e-7`, sec. 0f.34) and the **full certified Z band fits**
   (`fullscale/collect.sh`, and the acceptance test of sec. 0f.16: value AND
   NLL AND EDM).

### IF IT DOES NOT REPRODUCE

Report which subdetector / class / coordinate carries the residual, and against
what it does scale. The alternative already on the table is that the CF's hit
term is wrong for a DEGRADED complement in some way that is not a per-class
location -- e.g. the influence weights themselves being mis-estimated when hits
are missing, which the same caches can test by comparing `sum_c v_c` with
`vgf` per candidate (`vg_other` is the exported remainder; sec. "PHASE 3" and
the note that it is NOT float32 noise but an exact remainder, negative for
~40 % of candidates).

### TOOLS WRITTEN FOR THIS, ALL COMMITTED

`resolution/attribute_skew.py` (track level, `--raw` for the pre-transform
behaviour), `resolution/attribute_skew_mass.py` (mass level, full
`data - model`), `resolution/model_odd_mass.py` (the mass CF's own odd moment),
`resolution/cf_skew_closure.py --bin-eta / --truth-ref / --raw-pull`.

### 0f.44 THE sigma/m SPLIT AT FIXED eta — and the slope REVERSES between the
### barrel and the endcap (2026-09-08)

The discriminating test of sec. 0f.29: `sigma/m` and `\|eta\|` are nearly
collinear inclusively, so the only clean separation is to split `sigma/m` INSIDE
one `eta` band. v form, certified (scipy `trust-exact` through
`rabbit_fit.py`, EDM < 1e-3):

| cell | n | `m_Z` | EDM |
|---|---:|---:|---:|
| barrel, `sigma/m` LOW (< 0.01032) | 785 836 | **-4.56 +- 3.69** | 8.3e-13 |
| barrel, `sigma/m` HIGH | 786 698 | **-40.90 +- 4.63** | 2.4e-15 |
| barrel, both | 1 572 534 | -21.08 +- 3.24 | 7.8e-12 |
| endcap, `sigma/m` HIGH (> 0.01658) | 512 595 | **+65.87 +- 8.69** | 5.1e-16 |
| endcap, both | 1 025 424 | +34.22 +- 5.47 | 9.1e-18 |

**Within the barrel alone, `sigma/m` splits `m_Z` by -36.3 +- 5.9 MeV
(6.2 sigma)** -- comparable to the ENTIRE `eta` span of 55 MeV. So the residual
is a resolution effect and `eta` is largely its proxy, which is what sec. 0f.29
set out to decide.

**But the two slopes have OPPOSITE SIGNS, and nearly equal magnitude:**

| | `sigma/m` range | `m_Z` range | slope, MeV per unit `sigma/m` |
|---|---|---|---:|
| within the barrel | 0.0080 -> 0.0119 | -4.6 -> -40.9 | **-9 300** |
| across the `eta` bands | 0.0097 -> 0.0151 | -21.1 -> +34.2 | **+10 200** |

At fixed `eta` the bias FALLS with `sigma/m`; across `eta` it RISES with
`sigma/m`, at the same rate to 10 %. **So the bias is a function of neither
`sigma/m` alone nor `eta` alone**: a pure `sigma/m` dependence would give the
same slope both ways, and a pure `eta` dependence would give none within a
band. There are two dependences of opposite sign, and the inclusive numbers are
where they partly cancel.

The endcap half-sample points the same way: `sigma/m` HIGH gives +65.9 against
+34.2 for the whole endcap, i.e. the endcap's internal `sigma/m` slope is
POSITIVE where the barrel's is negative. **`z_V_etaE_slo` is needed to close
this** and it FAILED -- `ValueError: Cholesky decomposition failed, Hessian is
not positive-definite`, rabbit's `is_linear` branch -- and has been resubmitted.

### 0f.45 `e = f_hit - f_ioni` DOES NOT CLOSE THE MEASURED `a`

`f_ioni` is extractable from the stored CF with no new production: a CF
exponent `S(t)` contributes variance `-S''(0)`, so a three-point second
difference of `Sio_re` on `tgrid` gives the ionisation variance share per
candidate. Measured on `zpairs_dyv2_full.npz`:

| share | median |
|---|---:|
| hit (`vgf`) | 0.2173 |
| ms | 0.8148 |
| **ioni** | **0.00525** |
| rad | -0.00007 |
| sum | **1.0384** (should be 1) |

| band | MEASURED `a/(sigma/m)` | spec `1+vgf` | `1+vgf-f_ioni` | spec error | new error |
|---|---:|---:|---:|---:|---:|
| `\|eta\|<0.9` | 1.2503 | 1.2738 | 1.2683 | +0.0235 | **+0.0180** |
| `0.9-1.6` | 1.1667 | 1.2046 | 1.1980 | +0.0379 | **+0.0313** |
| `1.6-3.0` | 1.2725 | 1.3007 | 1.2953 | +0.0282 | **+0.0228** |
| inclusive | 1.2110 | 1.2625 | 1.2567 | +0.0515 | **+0.0457** |

**It improves the closure by only ~25 %, not to zero.** `f_ioni` is **0.005**,
i.e. 2.5 % of `f_hit`, where sec. 0f.33 inferred 10-20 % from the size of the
deficit. So the ionisation term is real, has the right sign, and is **five to
eight times too small** to be the explanation. Sec. 0f.33's attribution of the
deficit to `f_ioni` is **retracted**; what remains true is the MEASUREMENT that
`a` is 2-4 % smaller than `1 + vgf`.

**Caveat on the extraction**: the four shares sum to 1.0384 rather than 1, a
3.8 % closure failure, so the individual shares carry a systematic of that
order. It does not change the conclusion -- 0.005 against a needed 0.03 is not
a 4 % problem -- but the second-difference estimator should be checked against
the term's own variance before `f_ioni` is put in the card builder.

**So `e = f_hit - f_ioni` is NOT implemented in the card builder.** It would
buy a quarter of a 2-4 % coefficient error and it is not what the measurement
asks for. What the measurement asks for is whatever accounts for the other 75 %.

### 0f.46 PHASE 2 RUNS, CONVERGES AND PRODUCES A NUMBER (2026-09-08)

**`22314264 P2smoke`** -- `joint_ok_n500k` (500 000 J/psi + 500 000 Z + the
hit-chi2 external quadratic over 20 706 999 candidates), scipy `trust-exact`
through `rabbit_fit.py`, 3 h 39 m, **EDM 6.2e-19**. The first phase-2 fit that
has ever run end to end. It needed the dense-`D` fix (sec. 0f.24) to start at
all.

| | |
|---|---:|
| `m_Z` | **+31.86 +- 5.78 MeV** |
| `Gamma_Z` | +8.75 +- 10.32 |
| EDM | 6.2e-19 |
| NLL (unbinned terms) | 615177.106228 |
| free | 95 (50 field + 42 material - 4 frozen + 2 POI + 5 shapes) |
| `bfield` pulls | RMS 5.45, max 32.7 |
| `material` pulls | RMS 61.0, max 325 |

**Do NOT read `m_Z = +31.9` as a closure.** Two reasons, both already in this
file: it is a 500 k subsample, and **`theta = 0` is not the hit-chi2 minimum on
this MC**, so the enormous pulls are the quadratic term pulling the calibration
parameters off their MC-truth zero -- which is a property of the input, not of
the fit. The number is the demonstration that the machinery closes the loop:
card -> `rabbit_fit.py` -> converged joint minimum with an EDM.

**`22312984 P2K`** -- the SAME card family at FULL size (`joint_ok_full`) with
`trust-krylov` -- **FAILED**, at the postfit:

```
File "rabbit/tfhelpers.py", line 54, in tf_edmval_cov
ValueError: Cholesky decomposition failed, Hessian is not positive-definite
```

i.e. krylov stopped somewhere the Hessian is INDEFINITE, which is not a
minimum. That is sec. 0f.18's failure again and it is now a hard error rather
than a flattering number, which is the better outcome. **`P2X` (the same card,
scipy `trust-exact`) is submitted**; `P2smoke` shows that combination converges.
The same error killed `z_V_etaE_slo`, also resubmitted.

**A `warn_unconstrained` FALSE POSITIVE worth fixing.** On the joint card the
warning fires on the POIs:

```
WARNING:fitter.py: 2 floating parameter(s) have an essentially zero Hessian
diagonal (0.0608 against a scale of 7.12e+13): [m_Z, Gamma_Z]. Nothing in the
likelihood constrains them ...
```

`m_Z`'s error is 5.78 MeV -- finite and sensible. The check is RELATIVE to the
largest diagonal, and the hit-chi2 term's curvature is **7.12e13**, so on a
card mixing that with mass POIs at 0.06 every normal parameter looks
unconstrained at a 1e-12 relative threshold. The warning is right that the
trust-region subproblem is badly conditioned in the POI directions -- which is
exactly why `trust-exact` and not krylov -- but its message is wrong. It should
compare against a per-BLOCK scale, or against the parameter's own prior width,
not against the global maximum.

### 0f.47 THE VARIANCE-SHARE NORMALISATION IS NOT GOOD ENOUGH TO EXTRACT
### `f_ioni` — the excess is in `Sms`, and it is 2.5x `f_ioni` itself

Asked which family is over-counted. Answer, with the estimator fixed first.

**The estimator.** `Re S(t)` is EVEN in `t` (because `phi(-t) = phi(t)*`) and
`tgrid` starts at 0 and is UNIFORM (`h = 0.125280`, 64 points to 7.8926 --
checked, not assumed). So the second derivative at 0 must use a MIRRORED
central stencil, `S''(0) = (2 S(h) - 2 S(0))/h^2`, not the forward difference
`(S(2h) - 2S(h) + S(0))/h^2` I used in sec. 0f.45 -- that one estimates
`S''(h)`. `S(0) = 0` exactly for every family, as it must be.

| family | 3-point (mirrored) | 5-point (mirrored, O(h^4)) |
|---|---:|---:|
| `Sms` | **0.83288** | **0.83588** |
| `Sio_re` | 0.01676 | 0.01864 |
| `Srad_re` | 0.00816 | 0.00948 |
| hit (`vgf`) | 0.21734 | 0.21734 |
| **SUM** | **1.07443** | **1.08071** |

The 3- and 5-point stencils agree to 0.6 %, so **this is not truncation
error**: the shares genuinely do not sum to 1, and the excess is **7.4 %**
(sec. 0f.45's 3.8 % was the forward-stencil artefact and is superseded).

**Which family**: if the other three are right, `Sms` must be 0.7577 for the
sum to close and it measures 0.8329 -- **9.9 % high**. `vgf` would have to be
34 % wrong to carry it instead, which the pull width (0.9959, sec. 0b)
excludes. So the excess sits in the multiple-scattering block, and the likely
cause is a normalisation convention on `Sms` -- whether it is tabulated against
the standardized `t` or against an absolute `tau` -- which is a question for
`cf_inmaker`'s export, not for the fit.

**THE CONSEQUENCE, and it is the point.** The quantity being extracted,
`f_ioni = 0.0168`, is **four times smaller than the 0.074 the normalisation is
off by**. So these shares cannot measure `f_ioni` at the level `e = f_hit -
f_ioni` needs, and sec. 0f.45's conclusion must rest on the DIRECT measurement
of `a` (1.2110 +- 0.0004 against `1 + vgf = 1.2625`) and not on them. That
conclusion is unchanged -- `e = f_hit - f_ioni` closes at most a quarter of the
deficit -- and it now has a second, independent reason not to be implemented:
**the input it would need is not measurable to the required precision from what
is exported.**

### 0f.48 `warn_unconstrained` FIXED — it tests the ROW (rabbit `e006795`)

"Nothing in the likelihood depends on it" means the parameter's entire Hessian
ROW vanishes, and that is scale-free; testing the DIAGONAL against the global
maximum is not. On the phase-2 joint card the hit-chi2 block sits at 7.12e13
and `m_Z` at 0.0608, so every mass parameter fell below `rtol * max(diag)` and
was named as unconstrained while its error was a sensible 5.78 MeV
(sec. 0f.46). With the row test `m_Z`'s row maximum is ~2e4 against a threshold
of 71.2 -- five orders clear -- because the mass term couples it to the field
modes, while a material group no candidate touches still has an exactly zero
row and is still caught.

**Not a complete fix, and the docstring says so**: a whole BLOCK whose
curvature sits below `rtol * max|H|` is still flagged (the `K(m)` shapes are
the candidate on this card). The complete answer is a per-block scale from the
terms' own `param_names`, left for the branch.

### 0f.47b RESOLVED — the "shares" are NOT variance fractions, and that is
### physics, not an export bug (2026-09-08, coordinator's diagnosis, confirmed)

**Do not chase an export normalisation.** The Moliere exponent is not analytic
at `t = 0`: the single-scattering tail gives `S(t) ~ -t^2 (A - B ln t)`, so a
finite-difference `S''(0)` returns `2(A - B ln(step))` -- a CUT-DEPENDENT
number, not a variance. The Moliere second moment is formally tail-dominated,
and `sigma_m` (the standardisation) comes from the fit's **Q matrix**, not from
that moment. This is the recorded Q-vs-Moliere difference
(`resolution/qmsmodel/`, NOTES 2026-09-06: `R_2nd = 1.01-1.13`,
`R_core = 0.87-0.97`).

**Confirmed operationally.** The mirrored central estimator at step `j h`:

| family | step 1h | 2h | 4h | 8h | slope per `ln(step)` |
|---|---:|---:|---:|---:|---:|
| `Sms` | 0.83288 | 0.82388 | 0.80038 | **0.75785** | **-0.0359** |
| `Sio_re` | 0.01676 | 0.01110 | 0.00653 | 0.00359 | -0.0064 |
| `Srad_re` | 0.00816 | 0.00426 | 0.00198 | 0.00094 | -0.0035 |

`Sms` falls by 0.0359 per doubling-in-log, i.e. `B ~ 0.018` -- the predicted
`2 B ln 2` behaviour, measured. **There is no step at which the shares are
"right"**: their SUM runs 1.0751 / 1.0566 / 1.0262 / 0.9797 across the four
steps and crosses 1 between 4h and 8h at no privileged scale.

**And it is not only the MS family.** `Sio_re` and `Srad_re` show the same
signature (slopes -0.0064 and -0.0035): the ionisation Landau tail and the
radiative tail are heavy-tailed too, with formally divergent second moments.
So `f_ioni` is **also** not a variance fraction -- it reads 0.0168 at one step
and 0.0036 at eight, a factor 4.7.

**Consequence.** Sec. 0f.45's decision not to implement `e = f_hit - f_ioni`
now rests on three independent grounds: it closes at most a quarter of the
deficit; the input is not measurable from the exported shares; and **the input
is not a well-defined quantity in the first place** -- `f_ioni` as "the
ionisation variance share" does not exist for a Landau tail. If the correction
is wanted it must come from the CF's behaviour at the scale the fit actually
uses, not from a second moment.

**The MS normalisation is separately fine**: `k_ms = 1.001 +- 0.034` from the
fit says the exponent matches the data at the 3 % level.

**The `a`-coefficient deficit is untouched by all of this** -- it is a DIRECT
measurement (`a = d ln sigma/dx` regressed on the data) and it uses only
`vgf`, which is an exact Gaussian variance and has no tail problem. It stays
open at **2-4 %**: measured 1.2110 +- 0.0004 against `1 + vgf = 1.2625`, with
the per-leg term and the ionisation term both excluded as the explanation.

### 0f.49 GEN LEG KINEMATICS + `hit_s`/`hit_detid` — WRITTEN, NOT RUN, and why

**`fullscale/run_auxgen.sh`** drives the existing
`resolution/oddmoment/aux_gen.py` on **both** v2 productions against the
in-order caches:

| production | cache | output |
|---|---|---|
| `dymc_8p5M_260906_v2` | `runs/zpairs_dyv2_full.npz` | `runs/auxgen_dyv2.npz` |
| `jpsimc_20M_260906_v2` | `runs/jpairs_v2_n600.npz` | `runs/auxgen_jpsiv2.npz` |

Per cache row: `Mu{plus,minus}gen_pt/eta/phi` (charge is the +/- label),
`Jpsigen_pt/eta/phi/mass`, the reco leg `pt`/`eta`/`nvalid`, and
`fhit`/`fms`/`fioni`/`fother`.

**The alignment is validated, not assumed.** `aux_gen.py` matches row by row on
`z` and asserts that `z` AND `sigma` come back BIT-IDENTICAL -- a stronger
contract than a run/lumi/event join, which cannot detect a reordering within an
event. It requires the cache to be an in-order SUBSEQUENCE of the tree, which
the two above are; a cache built with a random `--maxn` is not, and it fails
loudly rather than mis-joining.

**And it reopens `e = f_hit - f_ioni` on a proper footing.** `aux_gen`'s
`fioni` is `sum_{parmtype==11} resinfvarv / sigma_m^2` -- from the fit's own **Q
matrix**, not from the CF exponent. That is exactly the quantity sec. 0f.47b
showed does NOT exist as a second derivative of a Moliere/Landau exponent. So
the question sec. 0f.45 closed on the wrong input can be asked again on the
right one, and `aux_gen` even carries the closed form already:
`a_m = (1 + f_hit - f_ioni) sigma_m/m`, which is precisely the coefficient
measured at 1.2110 +- 0.0004 against `1 + vgf = 1.2625`. **That is the first
thing to do when the file lands.**

**`cf_inmaker.py pairs --groups` now also exports** `hit_s` (the SIGNED
mass-projected influence weight, from `resinfv`) and `hit_detid`, both optional
so an older production still builds. `hit_v` is a variance and is unsigned, so
it cannot say which way a hit's displacement pushes the curvature -- a location
bias is a shift and a shift needs a direction; `hit_detid` is what turns a
class label into subdetector / layer / disk / +-z. **The branch names
`{prefix}_hits` and `{prefix}_hitdetid` are GUESSED** and must be confirmed
against a production tree; absent, the fields are skipped silently, and present
with a mismatched layout the builder fails loudly.

**NEITHER HAS BEEN RUN.** `/ceph` is permission-denied from a Claude Code
sandbox shell (top level, not just the leaf) and the local `slurm.conf` is
missing, so `sbatch` cannot be used either. Both need an ordinary submit login
shell:

```bash
cd /work/submit/david_w/ZMass/calibration_studies/fullscale && ./run_auxgen.sh
```

The hit-class agent has been told the paths, the alignment contract, the
`fioni` caveat and the guessed branch names.

### 0f.50 THE MIXTURE HYPOTHESIS DOES **NOT** REPRODUCE ON THE Z LEGS WITH
### `chi2/ndof` AS THE DISCRIMINATOR (2026-09-09)

The hit-class agent found the track-level `eta` dependence to be a MIXTURE:
splitting on `|seed->final dq/p|` at its 90th percentile gives two
`eta`-INDEPENDENT components (+20.59 +- 6.45 and -6.33 +- 1.84 e-3) whose
mixing fraction runs 2.4 -> 7.0 -> 21.2 % with `|eta|` and reproduces the band
values exactly.

The seed->final step is not in the mass caches, but the IN population's
signatures are, and **`chi2/ndof` is BOTH their discriminator (1.098 against
0.988) and the safest conditioning variable this cache has**:

| variable | `corr(., z)` |
|---|---:|
| **`chi2/ndof`** | **+0.0004** |
| `vgf` | -0.0009 |
| `\|eta\|` lead | -0.0081 |
| `nhit` | +0.0133 |
| pixel share | +0.0141 |
| `sigma/m` | -0.0158 |
| `maxfraclossp` | -0.0335 |
| (their signed seed->final step) | +0.113 |

`mixture_test.py`, `data - model`, 1e-3, u = 0.05:

| split | band | `f_IN` | IN | OUT |
|---|---|---:|---:|---:|
| p90 | `\|eta\|<0.9` | 9.6 % | +0.83 +- 9.92 | +1.61 +- 3.54 |
| | `0.9-1.6` | 10.6 % | +5.26 +- 8.94 | -4.21 +- 2.30 |
| | `1.6-3.0` | 9.7 % | -16.33 +- 8.03 | **-14.37 +- 2.23** |
| | **chi2 vs `eta`-flat** | | **3.65 / 2** | **18.12 / 2** |

**The IN component is `eta`-flat (chi2 3.65/2, p = 0.16) but the OUT component
is NOT (18.12/2, p = 1e-4)** -- and OUT is 90 % of the sample, so the bulk
still carries the whole `eta` dependence. Same at p80 (OUT 16.7/2) and p95
(OUT 22.8/2), and the IN fraction is **flat in `eta`** here (9.6 / 10.6 / 9.7 %)
where theirs grows 2.4 -> 21.2 %.

**So `chi2/ndof` is not their discriminator at mass level.** It selects a
population that is `eta`-flat, but it does not select THE population whose
fraction grows with `eta`. The test is not refuted -- it has not been done:
it needs the actual `|seed->final dq/p|` per leg, which the coordinator says is
in the trees (`Jpsitrk_*` / per-leg trk branches) and which the auxgen pass
can pick up.

**THE PLAN when `auxgen_*.npz` land** (in order):
1. the `a_m = (1 + f_hit - f_ioni) sigma_m/m` closed-form check with the
   Q-matrix `f_ioni`, against the measured 1.2110 +- 0.0004 / 1.2503 / 1.1667 /
   1.2725;
2. `|seed->final dq/p|` per leg, **absolute value only** -- the signed one has
   `corr(., x) = +0.113`, a worse trap than reco pT -- with
   `corr(|delta|, |x|)` stated;
3. the same two-component decomposition per `eta` on the Z and J/psi legs at
   8x the gun's statistics, and at mass level kernel-free per band, IN vs OUT.

**`hit_s` is dead as a column** and the guessed names are REMOVED from
`cf_inmaker` rather than left to skip silently: the v2 slim trees carry exactly
`cfmass_hitcls`, `cfmass_hitv`, `reshitcls`, `reshitidx`, `resinfcovhit` per
hit. `resinfv`/`resinfbv` are booked under `if (exportStepRecords_)` (off for
the 81 kB/candidate path) and `hitDetId` only in the fitFromGenParms block, so
a signed per-hit weight needs a re-production or a small maker change.

---

### 0f.51 THE Q-MATRIX SHARES DO CLOSE, `f_ioni` IS 4.5e-06, AND THE `a_m`
### CLOSED FORM CLOSES **NONE** OF THE DEFICIT (2026-09-08)

**The grouping bug was not where the resume note said it was.** The v2 runtree
carries parmtypes 0-5 (alignment), **8** (hit resolution, local x), **9**
(local y, pixel only), **10** (multiple scattering), **11** (ionisation),
**14** (50 B-field modes) and **15** (42 material groups). Per candidate
`reseigidx` points at 8/9/10/11 **and** 15, and

> the parmtype-15 blocks are a RE-PARTITION of the parmtype-10/11 noise
> (`sum_g dQ_g == dQMS + dQI`), not an addition
> -- `ResidualGlobalCorrectionMakerTwoTrackG4e.cc:4758`

so `f_ms` and `f_ioni` in `aux_gen.py` were **always right**. What was wrong is
`f_other`, which swept 8, 9 and 15 into one bucket and therefore printed a
closure of 2.0 and an `f_other` of 1.0 -- the symptom that was read as
"f_ioni = 0 because the parmtypes changed". Fixed (`cd9b58a`): each family is
named, parmtype 15 is carried as an independent CHECK, `f_other` is now the
unrecognised-parmtype bucket and is 0. On one DY v2 task, 2405 candidates:

| check | value |
|---|---:|
| `f_hit + f_ms + f_ioni` | **1.000000031**, max\|.-1\| **3.9e-07** |
| `f_mat - (f_ms + f_ioni)` | -2.9e-10 mean, 3.5e-08 max |
| `f_other` (unrecognised) | **0 exactly** |

**So the Q-matrix shares ARE variance fractions** -- which is exactly what the
CF-exponent "shares" of sec. 0f.47b are NOT (they sum to 1.075 at one
finite-difference step and 0.980 at another, because Moliere and Landau have no
finite second moment). That was the whole point of asking the Q matrix instead.

**THE `a_m` CLOSED FORM** (`measure_a.py --aux`, which asserts bit-identical
`z` against the pairs cache first). `a_m = (1 + f_hit - f_ioni) sigma_m/m`:

| cell | MEASURED `a/(sigma/m)` | spec `1+vgf` | closed `1+f_hit-f_ioni` | `f_ioni` | meas - closed |
|---|---:|---:|---:|---:|---:|
| **Z legs**, inclusive | **1.2110 +- 0.0004** | 1.2625 | **1.2625** | 4.5e-06 | **-0.0516** |
| `\|eta\|<0.9` | 1.2503 +- 0.0006 | 1.2738 | 1.2738 | 5.0e-06 | -0.0236 |
| `0.9-1.6` | 1.1667 +- 0.0003 | 1.2046 | 1.2046 | 4.9e-06 | -0.0379 |
| `1.6-3.0` | 1.2725 +- 0.0004 | 1.3007 | 1.3007 | 3.9e-06 | -0.0282 |
| **J/psi legs**, inclusive | 0.8928 +- 0.0130 | 1.0996 | 1.0993 | 3.2e-04 | **-0.2064** |
| `\|eta\|<0.9` | 0.9163 +- 0.0223 | 1.1186 | 1.1181 | 5.7e-04 | -0.2017 |
| `0.9-1.6` | 1.0070 +- 0.0113 | 1.0920 | 1.0917 | 3.4e-04 | -0.0847 |
| `1.6-3.0` | 0.9899 +- 0.0090 | 1.0951 | 1.0949 | 2.1e-04 | -0.1051 |

**`f_ioni` from the Q matrix is 4.5e-06 on the Z legs** -- five orders of
magnitude below `f_hit` -- **so `1 + f_hit - f_ioni` is identical to `1 + vgf`
to four decimals and closes 0.00 % of the deficit.** Sec. 0f.45's "it improves
the closure by ~25 %" is **RETRACTED**: it rested on the CF-exponent
`f_ioni = 0.005`, which sec. 0f.47b then showed is not a variance at all
(it reads 0.0168 at one step and 0.0036 at eight).

**The `e = f_hit - f_ioni` line is now CLOSED, on the right input.** The
deficit is `-0.052` on the Z legs and **`-0.206` on the J/psi legs** -- four
times larger at a seventh of the momentum -- and neither the per-leg term nor
the ionisation term explains it. Figure: `am_closed.png`.

### 0f.52 THE MIXTURE HYPOTHESIS: THE **FRACTION** REPRODUCES, THE
### **COMPONENTS** DO NOT — REFUTED ON BOTH CHANNELS (2026-09-08)

Sec. 0f.50 could not do this test because the mass caches had no
`|seed -> final dq/p|`. **They do have what it takes**: the two-track trees
carry `Mu{plus,minus}trk_pt/eta` (the generalTracks KF seed) and
`Jpsi_qopref{plus,minus}` (the CVH reference `q/p`, `== Mu*_refParms[0]`), so

    dq/p = |q/p final - q/p seed| / |q/p seed|

`oddmoment/aux_seed.py` (new) extracts it plus `Jpsi_sigmarel*` and the gen leg
momenta, and aligns to the pairs caches with `aux_gen.join_to_cache` -- the
join is independently validated by `sigrel` coming back **bit-identical**
(max diff 0.000e+00) to the cache's `sigrelp`/`sigrelm`. ABSOLUTE step only.
`fullscale/run_auxseed.sh`; `runs/auxseed_{dyv2,jpsiv2}.npz` (3 733 323 and
7 923 460 rows).

**A. THE MIXING FRACTION REPRODUCES THE GUN, STRIKINGLY**, split at the 90th
percentile of `|dq/p|`:

| `\|eta\|` | gun (hit-class agent) | **Z legs** | **J/psi legs** |
|---|---:|---:|---:|
| 0.0-0.9 | 2.4 % | **2.68 %** | 0.64 % |
| 0.9-1.6 | 7.0 % | **9.69 %** | 11.84 % |
| 1.6-2.4 | 21.2 % | **20.04 %** | 15.19 % |

**B. BUT NEITHER COMPONENT IS `eta`-FLAT** -- and that is what the hypothesis
requires. Leg level, truth-referenced pull `x = z/(1 - a q z)` with
`a = sigrel(1 - vgf)`, charge-even `<x e^{-0.05 x^2}>`, 1e-3:

| sample | band | IN (top 10 %) | OUT (the bulk) | all |
|---|---|---:|---:|---:|
| **Z legs** | 0.0-0.9 | +7.65 +- 3.75 | +3.97 +- 0.54 | +4.06 |
| 7 459 917 legs | 0.9-1.6 | +5.77 +- 2.20 | -3.46 +- 0.58 | -2.55 |
| | 1.6-2.4 | -9.49 +- 1.83 | **-18.22 +- 0.75** | -16.43 |
| | **chi2 vs `eta`-flat** | **35.8 / 2** | **577.7 / 2** | |
| **J/psi legs** | 0.0-0.9 | +3.82 +- 6.02 | +1.47 +- 0.41 | +1.49 |
| 15 846 860 legs | 0.9-1.6 | -0.17 +- 1.16 | -0.54 +- 0.38 | -0.46 |
| | 1.6-2.4 | -1.18 +- 1.00 | -3.26 +- 0.36 | -2.93 |
| | **chi2 vs `eta`-flat** | 1.0 / 2 | **77.4 / 2** | |

and at MASS level, `data - model` with the FULL per-candidate CF, split on
`max(dq_p, dq_m)`:

| sample | band | `f_IN` | IN | OUT |
|---|---|---:|---:|---:|
| **Z**, 3 687 738 | 0.0-0.9 | 2.70 % | +9.42 +- 9.21 | -1.46 +- 1.28 |
| | 0.9-1.6 | 5.96 % | -13.66 +- 3.89 | -4.32 +- 0.87 |
| | 1.6-2.4 | 14.68 % | -6.69 +- 1.99 | **-12.09 +- 0.76** |
| | **chi2** | | 6.0 / 2 | **72.2 / 2** |
| **J/psi**, 7 920 956 | 0.0-0.9 | 0.34 % | -13.09 +-16.62 | +2.08 +- 0.61 |
| | 0.9-1.6 | 8.14 % | +9.64 +- 2.41 | +1.09 +- 0.60 |
| | 1.6-2.4 | 15.09 % | +14.17 +- 1.14 | **-4.73 +- 0.46** |
| | **chi2** | | 5.4 / 2 | **102.0 / 2** |

**THE VERDICT. The `eta` dependence lives in the BULK, not in the mixing
fraction.** The OUT component is 85-90 % of every sample and it alone runs
+4.0 -> -3.5 -> -18.2 on the Z legs (`chi2` 578/2) and +2.1 -> +1.1 -> -4.7 at
mass level (102/2). Removing the top 10 % in `|dq/p|` removes essentially none
of the `eta` dependence. **The two-component mixture is therefore not the
explanation on either channel, at 8x and 16x the gun's statistics.** The
mixture IDENTITY `f IN + (1-f) OUT = all` holds to < 0.06e-3, as it must --
it is algebra, not evidence; the FLATNESS is the test and it fails.

The `f_IN` agreement with the gun is real and says the *population* the step
selects is the same one; what does not carry over is the claim that the two
components are `eta`-independent. Robust across p80 / p90 / p95 (OUT `chi2`
452 / 565 / 497 on the Z legs) and across the `vgf` systematic on `a`
(`a = sigrel`, `sigrel(1-vgf)`, `sigrel(1-2vgf)` give the inclusive
charge-even as -3.52 / -3.51 / -3.51 e-3).

**CAVEATS, both to be stated wherever this is quoted.** (i) `corr(|dq|, |x|)`
is **+0.082** on the Z legs, **+0.100** on the J/psi legs and **+0.156** at
mass level -- larger than the reco-`pT` trap's +0.042, so the split is NOT a
clean conditioning variable and the IN/OUT *values* carry a selection effect
(the FRACTIONS and the flatness `chi2` do not). (ii) The Z legs' band pattern
(+4.1 / -2.6 / -16.4) has the OPPOSITE `eta` trend to the gun's
(-4.9 / -3.6 / +0.1); the gun is single-track, 150X reco with a different
alignment payload, the legs are two-track (mass- and vertex-constrained) 106X.
The two are not the same estimator and should not be differenced.

Figure: `mixture_legs.png`. Tools: `oddmoment/aux_seed.py`,
`fullscale/mixture_legs.py`, `fullscale/run_auxseed.sh`.

### 0f.53 `K(m)` IS NOT SATURATED FOR `m_Z` EITHER — 7 -> 9 MOVES IT
### **+26.6 MeV** (2026-09-08)

`Ss9` landed (4 h 23 on an H200, cold start). Certified, m form, full 3 682 662
candidates:

| terms | `m_Z` [MeV] | NLL | `2 dNLL` vs the previous rung | EDM |
|---:|---:|---:|---:|---:|
| 5 | -11.06 +- 2.27 | 11075392.4657 | — | 1.8e-18 |
| 6 | -13.98 +- 2.22 | 11075277.1487 | 230.6 / 1 dof | 1.7e-12 |
| 7 | -17.24 +- 2.27 | 11075192.7817 | 168.7 / 1 dof | 8.1e-10 |
| **9** | **+9.39 +- 2.31** | **11074904.2322** | **577.1 / 2 dof** | 1.3e-11 |

**`m_Z` moves +26.63 MeV between 7 and 9 terms -- 11.5 statistical errors** --
and the 9-term fit is preferred at `2 dNLL = 577` for 2 dof. **Sec. 0b's
"for `m_Z` the basis is saturated" is RETRACTED**: it was inferred from a
-1.3 MeV shift over 5 -> 7 at 300 k, where the full-statistics shift is
-6.2 MeV and the next rung is +26.6.

So the caveat that sec. 0b attached only to `Gamma_Z` applies to `m_Z` as well:
**the -11.06 MeV closure is a statement AT `K(m)` = 5, and the `K(m)`
truncation moves it by tens of MeV.** The shape coefficients are still O(1) and
tightly determined (`shape7 = +0.1507 +- 0.0061`, `shape8 = -0.0457 +- 0.0018`,
`shape9 = +0.00368 +- 0.00015`), so this is not a runaway -- the LO->MiNNLO
K-factor genuinely has structure the 5-term basis cannot carry, and it projects
onto the mass. `SVs9` (the v form, which is the better-behaved one: -1.5 /
-3.9 / -3.9 over 5/6/7) is running; `Ss12`/`SVs12` will not fit in the job's
remaining walltime and need a resubmission. **Until the v-form 9- and 12-term
rungs are in, no closure number from this campaign should be quoted without the
`K(m)` truncation stated next to it.** Figure: `mz_kladder.png`.

### 0f.54 `--freezeParameters` DOES NOT FREEZE THE **STEP** — a null subspace
### the trust region walks in, and the likely cause of the NaNs (2026-09-08)

**Found while certifying `Ss9`**: its frozen `k_hit/k_ms/k_ioni/k_rad` come out
at **0.99981823**, not 1, with `err = 0`. It is not unique:

| tag | `k_hit` | `k_ms` | `k_ioni` | `k_rad` |
|---|---:|---:|---:|---:|
| `n300kfix` | **1.12578622** | 0.87421378 | 1.12578622 | 0.87421378 |
| `Rdc8X` / `Sdc8W` | 1.01460855 | 0.98539145 | 1.01460855 | 0.98539145 |
| `RvfullX` | 0.98083680 | 1.01916320 | 0.98083680 | 1.01916320 |
| `SVetaBslo` | 1.00162416 | 1.00162416 | 1.00162416 | 1.00162416 |
| `P2smoke` | 0.99115554 | 0.99115554 | 0.99115554 | 0.99115554 |
| `Ss9` | 0.99981823 | 0.99981823 | 0.99981823 | 0.99981823 |

Note `k_hit + k_ms = 2.000000000` **exactly** in the alternating rows: the
displacement is a pure `1 +- delta` vector inside the 4-dimensional subspace.

**THE MECHANISM.** `rabbit/fitter.py` freezes with `tf.stop_gradient` only
(`frozen_params_mask`, ~l.649-820). `edmval_cov` (l.1054) and the Hessian
sub-block (l.1087) correctly `tf.gather(self.floating_indices)`, **but the
minimiser at l.2972 calls `scipy.optimize.minimize(scipy_loss, xval, ...)` on
the FULL parameter vector.** A frozen direction therefore has an exactly zero
gradient component AND an exactly zero Hessian row and column -- i.e. the
Hessian handed to `trust-exact` has a null subspace of dimension = the number
of frozen parameters. That is scipy's trust-region **HARD CASE**, in which
`IterativeSubproblem` deliberately adds a multiple of the null eigenvector to
reach the trust boundary. So the frozen parameters take an arbitrary walk in
their own subspace, of size set by the trust radius, and `delta` runs from
1.8e-4 to **0.126**.

**I PROPOSED THIS AS THE CAUSE OF THE TWO NaN FAILURES. IT IS NOT** -- see
sec. 0f.56, which measures the actual step. The drift is real, is fixed, and
would become the cause at a trust radius of ~1.5-2, but at the radius these
fits use it is three orders of magnitude too small. Keep the two findings
separate.

**IT IS THE MINIMISER, NOT THE OUTPUT WRITER -- verified.** The displacement is
present in `cb.xval` itself, i.e. in the vector the minimiser carried at the end
of its last iteration, which is what the SNAPSHOT stores (`n300kfix` snapshot
`k_hit = 1.12578622`, `Ss9` 0.99981823, `SVetaBslo` 1.00162416, `Sdc8W`
1.01460855 -- identical to the numbers in the result file). So the NLL and the
POI values those rows report were evaluated at the DISPLACED `k`, not at 1.
Conversely `SVetaEslo`'s failure snapshot has `k` at exactly 1, because it
failed on the first step and rabbit restored iteration 0.

**THE FIX IS NOT A REGULARISER**: minimise over the FLOATING SUBSPACE only.
`self.floating_indices` already exists and is already used for the EDM and the
covariance; the minimiser has to see the reduced vector and scatter back. Handed
to the fit-infrastructure agent with the evidence.

**CONSEQUENCE FOR THE TABLE.** "Resolution and alignment FIXED at the MC truth"
is not exactly what the affected rows measured. `F_dc8` (`Rdc8X`/`Sdc8W`,
`m_Z = -2.03 +- 2.06`) ran with `k_hit` 1.5 % high and `k_ms` 1.5 % low; the
`sigma/m` split's barrel-low cell ran 0.16 % off. Every row with
`delta > 1e-3` has to be re-run once the fix is in. The headline
`-11.06 +- 2.27` (`fit_f380fl_base.json`, through `fit.py`, not rabbit) and the
inclusive v-form row (`SVfullW`, `k` exactly 1) are NOT affected.

### 0f.55 HOUSEKEEPING (2026-09-08)

* **`P2X` FAILED**, same NaN as `SVetaEslo` (first iteration, `Condition
  number: nan` / `edmval: nan`); it still wrote `rabbit_P2X.hdf5` from the
  unmoved start point. **It is not a phase-2 number.** `certtable.py` rejects
  it on EDM, but do not read the file.
* Both cards were scanned dataset by dataset and are **NaN/inf-FREE**
  (`joint_ok_full.hdf5`: both unbinned terms' `a_res, jensen_s2, mobs, sigma,
  vgf, weights, jac_values`, the five `S_*` tables and their `*_norm`, plus
  `hitchi2`'s `grad_values`/`hess_dense`). `hitchi2`'s eigenvalues run
  6.5e-19 to 7.24e13 with 32 of 92 below 1e-6 x max, but the identical warning
  appears in `P2smoke`, which converged.
* **`plot_closure.py` was drawing ONE series twice.** `band()` was called for
  the m form and for the v form but never told which cards to draw, so both
  drew every card and the v form painted over the m form, in the m form's
  colour. Every panel it has ever produced is affected. Fixed (`d363f92`).
* `runs/auxgen_jpsiv2.npz` landed at 20:03 (7 923 460 rows, `z` and `sigma`
  bit-identical, `<f_hit>` 0.0996 `<f_ms>` 0.9001 `<f_ioni>` 3.2e-04).

### 0f.56 THE NaN IS THE `K(m)` BLOCK: A TRUST RADIUS OF 1.0 IS **500 SIGMA**
### ON `shape5` (2026-09-08, fit-infrastructure agent, measured)

`gate_nanstep.py` reconstructs scipy `trust-exact` around the real Fitter on
`z_V_etaE_slo` and reproduces the failed job's start point exactly
(cond 2.06683e+08, `edmval` 3353.85). Then:

* the FIRST trial point is `|dx| = 1` -- the initial trust radius -- and it is
  **98.3 % along `shape5`** (`shape5` 0 -> -0.9835, `shape4` -0.178,
  `shape3` -0.033, everything else < 0.005). There, 105 Hessian entries are
  non-finite and the loss is `inf`;
* bisecting `x0 -> x_bad`, the first non-positive density appears at
  `t = 0.9815`: **one** candidate of 512 595 at `L_i = -2.5e-7`, with the
  truncation normalisation `min Z_c` collapsed from 0.9775 to 0.0029;
* coordinate-wise at the full step, **`shape5` alone gives 323 non-positive
  densities** (`Z_c = 7.8e-4`) while `shape4`, `shape3`, `shape2`, `shape1`,
  `m_Z`, `Gamma_Z` and all four `k` give **zero**.

**So the NaN is the `K(m)` Legendre block, and specifically its highest-order
term.** `sigma(shape5)` is 0.0020 in the inclusive fit, so a step of 1.0 is
**500 sigma** in it.

**AND THE UNDERLYING DEFECT IS SCALING, NOT THE ENDCAP.** The trust radius is
in RAW parameter units and this card's parameters differ by three orders of
magnitude in natural scale: `m_Z`/`Gamma_Z` have `sigma` of order 2-4 (MeV),
the `k` order 1, the Legendre coefficients 0.002-0.04. A radius of 1.0 is
~0.3 sigma for `m_Z` and ~500 sigma for `shape5`. That predicts the failure is
one unlucky draw away on EVERY card in the ladder, which is consistent with it
hitting `z_V_etaE_slo` and sparing `z_V_etaB_slo`; the card is **not**
singular and sec. 0f.44's "genuinely singular" is retracted.

The candidate fix that is neither a regulariser nor a change of physics is
therefore **preconditioning** -- rescaling the parameters by their own
curvature so the trust region is spherical in `sigma` units. It is a change of
variables and leaves the minimum invariant; `f380refP` exists to test exactly
that, and sec. 0f.14 dropped `--precondition` for a cause (the singular
subproblem) that scipy's subproblem has since removed. It must be validated by
re-running an ALREADY CONVERGED cell with and without it and requiring `m_Z` to
agree to 0.01 MeV. David's two physics options (fewer `K(m)` terms for the
sub-cells, or the inclusive `K(m)` held fixed there) remain available but both
change what the sub-cell fits MEASURE, so they break the comparison with the
three cells already certified and should be the fallback, not the first move.

**`P2X` IS A THIRD THING**: its NaN is at the START point, `x` = all defaults,
before the minimiser proposes anything -- scipy's `norm(hess, inf)` in
`IterativeSubproblem.__init__` firing on the first construction. Under
diagnosis (`22332382`).

**The frozen-parameter drift of sec. 0f.54 is confirmed and fixed** (rabbit
`vmass-conditioning`, `tests/test_frozen_subspace.py`, 7 tests). Measured on
the same step it is +0.00117 on each `k` -- visible, and three orders too small
to matter here; the density stays positive to `k_hit = -0.5` and `k_ms = -0.2`.
It would become the cause at a radius of ~1.5-2. `delta = max|k-1|` over all 36
result files: `n300kfix` 1.26e-01, `RvfullX` 1.92e-02, `Sdc8W`/`Rdc8X`
1.46e-02, `P2smoke` 8.8e-03, `f380refX` 8.1e-03, `SVetaBslo` 1.6e-03,
`Ss9` 1.8e-04, the other 28 exactly 0. **The only affected row in the certified
table is `SVetaBslo`** (barrel `sigma/m` LOW, -4.56 +- 3.69); the rest fail on
EDM anyway. The staging of the fix to Engaging is HELD until `22315719` and
`22333692` finish, so that every rung of the `K(m)` ladder is the same code.

### 0f.57 BINNING ON `|eta|` IS SAFE (gen == reco to 0.002); BINNING ON `asym`
### IS **NOT**, AND THE LEG-ASYMMETRY TERM IS EXCLUDED ON A SAFE VARIABLE
### (2026-09-08)

`auxgen` carries the gen leg momenta, so the `a` measurement can be repeated in
cells that CANNOT see the residual (`measure_a.py --gen-cells`). Two things
come out, and the second retracts a claim in `measure_a.py`'s own docstring.

**1. `|eta|` binning is validated.** With the SAME definition
(`max(|eta_p|, |eta_m|)`), gen and reco cells agree to 0.002 in
`a/(sigma/m)` -- two orders below the effect:

| band | reco `\|eta\|` | GEN `\|eta\|` | spec | deficit (gen) |
|---|---:|---:|---:|---:|
| < 0.9 | 1.2503 +- 0.0006 | **1.2485 +- 0.0006** | 1.2738 | -0.0254 |
| 0.9-1.6 | 1.1667 +- 0.0003 | **1.1683 +- 0.0003** | 1.2046 | -0.0364 |
| 1.6-3.0 | 1.2725 +- 0.0004 | **1.2724 +- 0.0004** | 1.3007 | -0.0283 |

So the 2-4 % deficit is real and is not an artefact of conditioning on a
reconstructed variable.

**2. `asym` is NOT a safe binning variable** -- `measure_a.py`'s docstring
says it is "to first order a property of the kinematics, not of the residual",
and the data say otherwise. In `asym` quintiles the measured coefficient is
1.078 / 1.030 / 0.986 / 1.093 / 1.332, i.e. deficits of **-0.148 / -0.206 /
-0.247 / -0.150 / -0.044** -- every one far larger than the -0.025...-0.036 the
safe `|eta|` cells give, and the `sigma/m` x `asym` grid is worse still
(-0.15...-0.33). `asym` is built from the two REPORTED per-leg widths, each of
which is `sigma_bar(1 + a q x)`, so binning on it is a (weaker) version of the
forbidden `sigma/m` bin and it attenuates the slope. **Standing rule 3 extends
to `asym`.**

**3. And the leg-asymmetry term is excluded on a variable that IS safe.** The
gen `pT` ratio `min/max` is the truth-level proxy for `asym` (q1 = most
asymmetric, q5 = most symmetric):

| gen `pT` ratio | q1 | q2 | q3 | q4 | q5 |
|---|---:|---:|---:|---:|---:|
| measured `a/(sigma/m)` | 1.2235 | 1.2161 | 1.2474 | 1.1890 | 1.1692 |
| deficit | **-0.0801** | -0.0377 | -0.0014 | -0.0612 | **-0.0820** |

The deficit is a **U**, equal at the two extremes and vanishing in the middle
-- it does not track leg asymmetry at all, where the per-leg form predicts a
monotone growth (`per-leg` runs 1.379 -> 1.301 across the same quintiles). Gen
`|d eta|` between the legs gives +0.012 / -0.078 / -0.034 / -0.017 / -0.130,
also non-monotone. **The per-leg asymmetry term is therefore excluded as the
explanation of the deficit on a variable that cannot attenuate the slope**,
which is a stronger statement than sec. 0f.45's, which used `asym` itself.

What is left: the deficit is real, is 2-4 % in `|eta|` cells and 0-13 % in
truth cells, is FOUR TIMES larger on the J/psi legs (-0.206) than on the Z legs
(-0.052) at a seventh of the momentum, and is explained by neither the
ionisation share (sec. 0f.51) nor the leg-asymmetry term (here).
