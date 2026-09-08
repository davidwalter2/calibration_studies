# fullscale — STATE

Full-scale feasibility of the unbinned CVH mass likelihood: the statistical
precision and the MC closure of `m_Z` and `Gamma_Z`, alone and jointly with the
J/psi channel that fixes the momentum scale.

Figures: `~/public_html/cvh/260906_fullscale/`.
Notes:   `/work/submit/david_w/Documents/Resolution/NOTES.md` (dated entries).

## Phases

| phase | what | status |
|---|---|---|
| 1 | Z alone, DY v2, resolution FIXED at MC truth, both corrections ON | IN PROGRESS |
| 2 | joint J/psi v1 + Z, quadratic term (field 50 + material 42) | pending phase 1 |
| 3 | full design on J/psi v2 (per-group exponents both legs) | waits for the user |

## Inputs

| leg | path | tasks | note |
|---|---|---|---|
| Z | `/ceph/submit/data/user/d/david_w/ZMass/cvh/dymc_8p5M_260906_v2` | 380, 370 complete at 2026-09-06 17:20 | 4 streams/task; 5 re-running after a threading fix |
| J/psi v1 | `/ceph/submit/data/user/d/david_w/ZMass/cvh/jpsimc_20M_260905` | 1642, 1539 complete at 17:20 | single stream, slurm |
| J/psi v2 | `/ceph/submit/data/user/d/david_w/ZMass/cvh/jpsimc_20M_260906_v2` | 1642, 105 complete at 17:20 | condor, running |

## Log

### 2026-09-06 — setup
* `resolution/cf_inmaker.py pairs` extended with the OPTIONAL aux columns the
  Z channel and the two corrections need and that cannot be recovered after the
  pass: `w` (`genweight`), `mpre` (`Jpsigenpre_mass`), `sigrelp/sigrelm`,
  `rhomom`, `fang` (the Jensen inputs, from `Jpsi_covrefmom`), `chisqval`,
  `ndof`, `maxfraclossp/m`, the pair and leg kinematics, and `run/lumi/event`.
  A branch that is absent is not read, so a v1 production still produces the
  reference cache. The mass path is also VECTORIZED (it appended one row per
  candidate; ~0.1 ms/candidate is hours at 3.9 M). **Verified bit-identical**
  on the reference columns over 2 DY v2 tasks / 19422 candidates.
* rabbit: `material-resolution` CONTAINS `global-term-card`; the Z channel is
  on `z-lineshape-kernel`. Both fork from `unbinned-mass-term` and both rewrite
  `MassCFTerm`, so phase 1 needs a MERGE of the two.

### 2026-09-06 — phase 1 inputs
* **pairs cache** `runs/zpairs_dyv2.npz` from **373 usable tasks / 1492 files**
  of `dymc_8p5M_260906_v2` (7 skipped: 1 no-`.complete`, 6 no stream files),
  `--mass-window 91.1876 60`. Aux columns present: 20.
* **quadratic extraction** `runs/quad_{dyv2,jpsiv1}.npz`, parmtypes 14 (50
  modes) + 15 (42 groups), `--no-mass` (both productions ship `radvgrid` alone
  and the mass path rejects that as a partial radiative export), cuts
  `chi2/ndof < 3`, `gradmax < 1e6`, `hessmax < 1e8`, **`dE_ref/p < 0.01`**.
  DY: 3 690 769 candidates, 47 155 cut, 105 s.
* `globalfit/extract.py` gained two things it needed and did not have:
  `--max-dEref-p` (the NOTES sec. 7(a) mean-loss quality requirement, from
  `Mu*_dEref`, which is in BOTH v1 and v2 -- the quantity itself rather than
  the daughter-pT proxy) and the **sandwich meat** `jsand = sum_i G_i G_i^T`,
  accumulated unconditionally. On 2 DY tasks the sandwich inflation
  `sqrt(J/2K)` is 1.02 (median, field modes) and 1.21 (median, material) with
  one group at 8.5 -- exactly the diagnostic NOTES asked for.
* **FSR kernel** `zchannel/data/kern_loose_band3.3e-4.npz`: banded (11 bands in
  `m_pre`), `sigma_cap 3.3e-4`, `--acc-pt 5 --acc-eta 2.4`, 14 315 atoms from
  13.03 M selected gen events. No prebuilt kernel was both banded and
  `<= 3.3e-4`. Acceptance `zchannel/data/acc_loose_d8.json` already matched.
* **The production's selection**: `60 < Jpsitrk_mass < 120` (the PRE-REFIT
  track mass), opposite sign, no muon ID, no pT/eta cut, no trigger. The
  effective muon floor is MiniAOD's `slimmedMuonTrackExtras` `pt > 4.5`, hence
  the `--acc-pt 5` kernel. `Jpsitrk_mass` is now cached as `mtrk` so the
  mismatch between the selection variable and the modelled observable is
  measurable.

### New code
| file | what |
|---|---|
| `chunkfit.py` | value/grad/Hessian accumulated over candidate chunks (pfor or forward-over-reverse HVP) + the sandwich meat by forward-mode autodiff |
| `make_card.py` | the full-scale Z card: weights, both corrections, provider-side multiplicative FSR + acceptance + floated K(m), documented cuts |
| `fit.py` | the driver: reference-point information, fit, sandwich, projections, truth comparison |
| `run_extract.sh` | the quadratic extraction for both legs |
| `patches/zgamma_shape.py` | adds the floated smooth K(m) (5 Legendre) to `ZGammaLineshape` |
| `patches/unbinned_jensen.py` | adds the EXACT second-order (Jensen) map to `MassCFTerm` |

### Blocking
* the rabbit **merge** of `material-resolution` (corrections, MaterialCFTerm,
  ExternalParams) and `z-lineshape-kernel` (provider, norm_window, upsample):
  both rewrote `MassCFTerm`, 8 conflict regions, two of them large. In
  progress in a scratch sandbox; the two patches above apply on top.

### 2026-09-06 18:00 — checkpoint: phase-1 inputs COMPLETE
| input | file | content |
|---|---|---|
| Z pairs | `runs/zpairs_dyv2.npz` (4.54 GB) | **3 663 056** candidates, 373/380 tasks, 65 045 dropped (no gen match / `cfmass_ok`), 34 columns |
| Z quadratic | `runs/quad_dyv2.npz` | 3 690 769 candidates, 47 155 cut, 105 s |
| J/psi v1 quadratic | `runs/quad_jpsiv1.npz` | **15 965 797** candidates, 4 448 019 cut (21.8 %, dominated by `dE_ref/p < 0.01` as predicted), 1550/1615 tasks, 945 s |
| FSR kernel | `zchannel/data/kern_loose_band3.3e-4.npz` | 11 bands, sigma_cap 3.3e-4, 14 315 atoms, 13.03 M gen events |
| acceptance | `zchannel/data/acc_loose_d8.json` | Bernstein 8, pT>5, \|eta\|<2.4 |

Measured on the full Z cache:
* weights 4.80 % negative, **N_eff/N = 0.8129** -> every error x1.109 in the
  sandwich; max \|w\| 6.0e4 = 25x the modal 2374 (the 1e19 MiNNLO outliers do
  not survive the gen-mass window, so `--wclip 100` is inactive here);
* `sigma_m` median 1.106 GeV, `sigma_m/m` median 0.01232;
* **the selection-variable mismatch is 0.191 % and ONE-SIDED**: 7009 of
  3 663 056 candidates were selected on `60 < Jpsitrk_mass < 120` but have the
  CVH-refit mass outside, and none the other way (the sample cannot contain
  what the production cut). The window cut on the modelled observable
  therefore DISCARDS 0.19 %, it does not admit anything unmodelled.

### 2026-09-06 18:40 — pipeline validated end to end (uncorrected model)
A 50 k-candidate card built on the `z-lineshape-kernel` branch alone (no
corrections, `--shape 0`) runs the whole chain: pairs -> card -> HDF5 ->
`read_unbinned_terms_from_h5` -> chunked fit -> sandwich. Result, resolution
fixed at 1:

| | fitted [MeV] | sandwich err | truth | pull |
|---|---:|---:|---:|---:|
| `m_Z` | -31.31 | 14.16 | 0 | -2.2 |
| `Gamma_Z` | +129.96 | 29.12 | +0.0019 | +4.5 |

which is the expected failure mode of the uncorrected model: `Gamma_Z` carries
the LO->MiNNLO K-factor (+76 MeV at generator level, more here) and `m_Z`
carries the two missing corrections. Sandwich/Hessian error ratio 1.089/1.120
against `sqrt(N/N_eff) = 1.107`.

**The memory claim is now measured, not quoted**: the MONOLITHIC Hessian at
300 000 candidates and only TWO free parameters peaks at **115.7 GB**. At the
full 3.66 M and ~8 free parameters the same construction asks for ~5 TB. The
chunked accumulation is not an optimisation, it is the only way this fit runs.

### 2026-09-06 19:00 — the rabbit merge landed, plus two required additions
`rabbit-material` @ `4f762c0` is `material-resolution` with
`z-lineshape-kernel` merged in. All five suites green as scripts; the two that
error under pytest do so for a pre-existing reason (`test_*` helpers take
positional args, so pytest reads them as fixture requests) that is
byte-identical on BASE and on both parents. The additivity gate: with the
material-only features off the merged term reproduces the z-lineshape `nll` and
gradient, and vice versa, worst relative deviation **5.3e-14**, most cases
bit-identical.

Then two things the phase-1 model needs and neither branch had:

* `58a3f82` **the floated smooth K(m)** in `ZGammaLineshape` (`shape=5`,
  `shape_window=`). It existed only in `zchannel/fit_gen.py`; without it the
  LO->MiNNLO ratio is +76 MeV on `Gamma_Z`. `tests/test_zgamma_shape.py`, 7/7.
* `5658252` **the EXACT Jensen map** in `MassCFTerm` (`jensen_s2=`,
  `jensen_mode="exact"|"shift"|"off"`). The branch shipped the three hooks and
  no implementation. `tests/test_jensen.py`, 5/5.

### The chunked Hessian — measured, at 300 000 candidates and 2 free parameters
| construction | dNLL | dgrad | dHess | t(grad) | t(Hess) | peak RSS |
|---|---|---|---|---|---|---|
| monolithic (`fit_z.py`, `rabbit_fit.py`) | — | — | — | (in 55.3 s total) | | **115.6 GB** |
| chunked, 65536 (5 chunks) | 1.3e-16 | 3.0e-14 | 1.1e-15 | 5.7 s | 30.6 s | **61.1 GB** |
| chunked, 262144 (2 chunks) | — | — | — | 5.8 s | 30.2 s | 105.2 GB |

and at a DISPLACED point (0.7 sigma away, where the gradient does not vanish):
dNLL 0, dgrad 4.5e-15, dHess 9.5e-16. The chunked construction is exact.

`hvp` mode needed a fix: `tf.gather`/`tensor_scatter_nd_update` give an
`IndexedSlices` gradient and `tf.autodiff.ForwardAccumulator` cannot
differentiate through one. The parameter map is now an affine dense one
(a 0/1 selection matrix), which also makes the sandwich work.

### 2026-09-06 18:55 — phase 1 running
* full card `cards/z_full.hdf5`, **3 613 320 candidates, 3.56 GB**, built in
  7.5 s after the pairs cache is in memory. Parameters
  `k_hit k_ms k_ioni k_rad m_Z Gamma_Z shape1..shape5`; POIs `m_Z`, `Gamma_Z`.
* a 300 000-candidate twin `cards/z_n300k.hdf5` (random subsample, seed 1234 --
  a head slice would be one contiguous run range) carries the same model. The
  base fit plus five variants (`--ares off`, `--jensen off`, both off,
  `--jensen shift`, K(m) fixed) run on it in parallel on submit82.
* the full fit is queued on Engaging as **job 22161390** (`-G h200:1`,
  `mit_normal_gpu`); the card and the merged rabbit are staged. The CPU path on
  submit is the fallback and runs in parallel.

### Not done in phase 1, and why
* **`resolution/cfcompress` (rank-16 PCA) is not applied.** It is a
  card-SIZE measure and the card is 3.56 GB, which loads in 7 s. Its own study
  puts the joint-basis rank needed for `max_t W|dS| < 1e-4` at 48-64 on the
  mass caches, i.e. rank 16 is a 1e-3 approximation -- acceptable for storage
  at 38 M candidates, not something to introduce when it buys nothing here.
* **The truncation normalisation `Z` is evaluated at the STORED class sigma,
  not at the self-consistent `s_i(theta)`.** `_norm_z` is a class-level
  Gil-Pelaez edge integral and does not see the per-candidate dynamic sigma.
  The 64 quantile classes already coarse-grain sigma by more than the 2.5 %
  that `a_i delta_i` moves it, and the window edges sit ~28 sigma from the
  peak, so this is expected to be far below the class discretisation itself
  (measured at 1e-6 on a +-30 GeV Z window). To be quantified.

### The `_norm_z` sigma approximation — MEASURED, and it is 0.016 MeV
`check_normz_sigma.py` scales every class resolution by `1 + eps` and asks
what moves. On the 300 k card (64 classes):

| eps | max \|dZ/Z\| | max \|d(dlnZ/dm_Z)\| over 5 MeV |
|---|---|---|
| 0.005 | 1.12e-05 | 2.14e-09 |
| **0.025** (the relevant one: `a_i \|delta_i\|/sigma_i`) | 5.70e-05 | **1.08e-08** |
| 0.050 | 1.17e-04 | 2.17e-08 |

against `dlnZ/dm_Z = 7.66e-07` over the same 5 MeV, and `Z` itself in
[0.9767, 0.9776] -- so 2.3 % of the model density is outside the window and the
truncation is NOT inert. What the approximation costs is the 1.4 % it moves the
`m_Z`-DEPENDENT part of `lnZ`: at 3.66 M candidates that is a gradient error of
0.008 per MeV against a curvature of 1/1.45^2 = 0.48 per MeV^2, i.e.
**0.016 MeV on `m_Z`** -- 1 % of the statistical error. An error in `Z` that
does not move with `m_Z` is absorbed by the normalisation and biases nothing.

### 2026-09-06 19:05 — phase 2 inputs launched while phase 1 fits run
* `cf_inmaker.py pairs --jac-parmtypes 14 15` now also exports the dense
  `(n, 92)` mass Jacobian `dm_i/dtheta_k` plus its parameter map. That is the
  one thing a joint fit needs and neither the CF exponents nor the quadratic
  term carry, and `globalfit/extract.py`'s mass path (which would otherwise
  supply it) is blocked on these productions by the `radvgrid` guard AND would
  need a per-candidate join afterwards. Vectorized with awkward: bit-identical
  to the per-candidate loop, which would have cost five hours on the J/psi.
* running: `runs/jpairs_v1.npz` (J/psi v1, default window, with D) and
  `runs/zpairs_dyv2_jac.npz` (DY v2, with D). J/psi v1 reports **13 aux columns,
  absent `fang`, `sigrelp/m`, `rhomom`, `mpre`, `maxfracloss*`** -- as expected,
  it predates `Jpsi_covrefmom`, so its Jensen `s^2` needs the MC-measured
  `f_ang` (gun 0.086 / data 0.106) and carries that +-5 %.
* `make_joint_card.py` scaffolded; `sum_quadratic` (which refuses to add two
  extractions whose parameter maps differ) is the part that is finished.

### Phase-1 fits in flight
| where | what | state |
|---|---|---|
| submit82 | 300 k card, 6 variants in parallel, chunk 32768, 7 free params | running; ~150 s per Hessian, 48-56 GB each |
| Engaging `22161787` | **full 3.61 M card**, H200, chunk 32768, 111 chunks | running (an earlier attempt at chunk 262144 OOM'd on 141 GB: the lineshape CF gather is a `(chunk, nt_int)` int64 tiled per pfor parameter) |
| Engaging `22161788` | same card, `--hess-mode hvp`, chunk 65536 | pending (per-user GPU cap) |

### 2026-09-06 19:10 — DECISION: phases 2 and 3 run on J/psi **v2**
`jpsimc_20M_260906_v2` finishes at the same time as the v1 slurm arrays, so
there is no reason to calibrate on the weaker leg. **Phase 2 and phase 3 both
take `/ceph/submit/data/user/d/david_w/ZMass/cvh/jpsimc_20M_260906_v2/`**
(multi-stream; complete at 1642 tasks). What v2 adds that phase 2 wants
immediately:

* `Jpsi_covrefmom` and its reductions, so the Jensen `s^2` is **truth-free per
  candidate** on the J/psi leg instead of carrying the +-5 % of an MC-measured
  `f_ang` (gun 0.086 / data 0.106);
* the **two-track variance (log-det) gradient** (`varianceGradFamilies=15`), so
  the quadratic term's parmtype-15 information is the WIDTH as well as the mean
  loss -- on the J/psi gun that is a factor 41 (Fisher 4.07 -> 166.3);
* `Mu*_maxfracloss`, and the per-group CF exponents phase 3 needs.

The v1 extraction already made (`runs/quad_jpsiv1.npz`, 15 965 797 candidates;
`runs/jpairs_v1.npz`, running) is **kept only as a cross-check**: the same
candidates through a mean-loss-only quadratic term versus v2's variance-block
version, quoted as the parmtype-15 material-information ratio. It is not the
phase-2 input.

### First phase-1 number, 2026-09-06 19:10 — K(m) FIXED (300 k)
Both corrections on, resolution fixed at the MC truth, the 5 Legendre
coefficients held at 0 (i.e. the provider's LO shape):

| | fitted [MeV] | sandwich err | truth | pull |
|---|---:|---:|---:|---:|
| `m_Z` | +15.90 | 5.79 | 0 | +2.7 |
| `Gamma_Z` | **+158.18** | 12.16 | +0.0019 | **+13.0** |

`+158 MeV on Gamma_Z` is the LO->MiNNLO K-factor, the thing a floated `K(m)`
exists to absorb (the generator-level fit to the pre-FSR spectrum measured
+75.8 MeV; the detector-level number is larger). 9 iterations, 773 s, sandwich
23 s; error ratio sandwich/inverse-Hessian 1.084 / 1.127 against
`sqrt(N/N_eff) = 1.107`.

Asimov errors at 300 k are `m_Z` 5.244, `Gamma_Z` 10.107 MeV, i.e. **1.51 and
2.91 MeV at 3 613 320** -- within a few per cent of the 1.45 / 2.76 MeV the
459-candidate smoke projected.

### The v1 -> v2 cross-check: the material information, on the SAME candidates
`extract.py --no-mass --ntasks 120` on both J/psi productions -- the same 120
tasks, i.e. the same events reconstructed twice (1 242 026 vs 1 240 851
candidates, 0.1 % apart, from one task of v2 not yet complete). Hessians
normalised per candidate:

| block | tr(K) ratio v2/v1 | per-parameter diag ratio (median, q10, q90, max) | d(log det) | stiffest eigenvalue |
|---|---:|---|---:|---|
| **14 field** (50) | **1.0000** | 1.000, 1.000, 1.000, 1.000 | -0.0019 | 73.21 -> 73.20 |
| **15 material** (42) | **120.2** | **666**, 124, 33 089, 1.3e6 | **+277** | 0.01424 -> **0.8512** |

The field block is BIT-EQUAL, which is the control: `varianceGradFamilies=15`
touches parmtype 15 and nothing else, exactly as advertised. The material block
gains a factor **120 on the trace and 666 in the median parameter** -- larger
even than the factor 41 the J/psi gun predicted. **v1's mean-loss-only
quadratic term is very nearly blind to the material amounts**, and that alone
settles the choice of v2 for phases 2 and 3.

One thing to carry forward: the material sandwich inflation `sqrt(J/2K)` goes
from 0.993 median (v1) to **1.326 median / 5.43 max** (v2). With real
information the per-candidate score distribution is heavier-tailed, so the
robust error is the one to quote, not `2K^-1`.

### 2026-09-06 19:20 — the DY production completed; the cache is appended, not rebuilt
`dymc_8p5M_260906_v2` reached 380/380 after the phase-1 cache was written. The
7 tasks it missed are identified EXACTLY, from the reader's own per-file log
rather than from mtimes: `task_0000, 0002, 0003, 0148, 0168, 0260, 0293`.
Their pairs were built on their own (28 stream files, 70 267 candidates) and
appended with `append_pairs.py`, which refuses the merge unless the two caches
carry the same columns, the same tau grid, the same CF model, AND are disjoint
on `(run, lumi, event)` -- the reason those columns are cached at all.

**Which fits used which**: the 300 k variant ladder and the first full-scale
H200 fit were built from the **373-task** cache (3 663 056 candidates before
cuts). The final full-scale numbers are rebuilt on the **380-task** cache. The
appended tasks are 1.9 % of the sample, so the two differ by ~1 % on every
error and by nothing systematic.

### THE HEADLINE STATISTICAL NUMBER (H200, 3 613 320 candidates, 373-task cache)
Resolution fixed at the MC truth, K(m) floated (5 Legendre terms), both
corrections on, expected (Asimov) errors from the reference-point information:

| | sigma [MeV], inverse Hessian | x1.109 sandwich |
|---|---:|---:|
| **`m_Z`** | **1.979** | **2.19** |
| **`Gamma_Z`** | **4.035** | **4.48** |
| `shape1..5` | 0.0063, 0.0055, 0.0053, 0.0075, 0.0014 | |

against **1.51 / 2.91 MeV with K(m) FIXED** (the 300 k `noshape` Asimov scaled
to 3.61 M), i.e. floating the LO->MiNNLO shape costs **x1.31 on `m_Z` and
x1.39 on `Gamma_Z`** -- close to the 1.25x / 1.21x the generator-level study
measured, and the price of not having to trust an LO parton luminosity.

Correlations worth carrying forward: `rho(shape4, shape5) = -0.964` (the
Legendre basis is orthogonal over the Born window, not over the *observed*
spectrum after resolution and acceptance), `rho(m_Z, shape3) = +0.517` and
`rho(Gamma_Z, shape4) = +0.593` -- above the < 0.40 the generator-level fit
saw, so the shape is less orthogonal to the POIs at detector level than it is
at generator level, but nowhere near degenerate.

## PHASE 1 FINDING — the two corrections do not transfer from the J/psi to the Z

Both corrections of MASSCFTERM_SPEC are expansions in the RESOLUTION
fluctuation. What they are fed is `delta_i`, the deviation from the reference
mass. At the J/psi those are the same object; at the Z they are not, because
the +-30 GeV window is +-27 sigma and the deviation out there is FSR and the
Breit-Wigner tail, not resolution.

Measured on the 3 682 662 selected candidates:

| | J/psi (+-0.35 GeV on 3.0969) | **Z (+-30 GeV on 91.19)** |
|---|---|---|
| `\|r\| = \|delta\|/m` max | 0.113 | **0.520** (q99 0.437) |
| `\|a_i delta_i\|/sigma_i` | < 0.1 | median 0.028, **q99 0.53, max 1.02** |
| what the exact Jensen map MOVES the residual by | ~ its 20.6 MeV mean shift | **median 57.7 MeV, q99 7.9 GeV, max 10.7 GeV** |
| candidates moved by > 5x the intended shift | — | **39 %** |

and the fits show it. At 300 k, resolution fixed, K(m) floated:

| variant | `m_Z` [MeV] | `Gamma_Z` [MeV] |
|---|---:|---:|
| both corrections on, **K(m) fixed** | +15.90 +- 5.79 | **+158.18 +- 12.16** |
| both on, K(m) floated | **-35.51 +- 6.81** | **-421.03 +- 12.33** |
| `a_res` off, Jensen exact on | -53.98 +- 6.88 | -427.46 +- 12.33 |

`base` and `noares` agreeing to 6 MeV on `Gamma_Z` says the runaway is the
JENSEN map, not the self-consistent sigma. And the fitted shape coefficients
come out at -1.06, -0.83, -0.61, -1.85, +0.31 -- an `exp(sum c_k P_k)` with
coefficients of order one is not a smooth K(m) correction, it is the fit using
the shape to compensate a deformed resolution model.

**The fix** (`rabbit-material` `b64f49f`): `corr_clip`, the corrections'
argument domain in units of `sigma_i`. Inside, bit-for-bit unchanged; outside,
they SATURATE -- the Jensen map continued with unit slope, so the residual
stays strictly monotone in the observable and the log-Jacobian is exactly zero
there. `corr_clip = 0` reproduces the J/psi behaviour, so every existing gate
is untouched. It is a scalar attribute of the term, so the whole scan runs off
ONE card (`fit.py --corr-clip`).

Running: `22163315` the six variants UNCLIPPED at full scale (the problem at
scale), `22163745` the clip scan 3/5/10/0, `22163752` the six variants at
`corr_clip = 5`.

---

### 2026-09-06 20:00 → 2026-09-07 07:00 — the reformulation, and phase 1 finished

**The corrections were reformulated** (rabbit `cd6c165`, `a3958f9`, `0b03d48`,
`a4cd32b`, `3324ed0`). `corr_form="fluctuation"`: both MASSCFTERM_SPEC
corrections as ONE deterministic per-candidate map of the resolution
fluctuation, applied inside the convolution,
`u_i(x) = sigma_i x + c_i x^2 + d_i` with the measure `p_x(x)(1 - a_i x)`,
which in Fourier space is one multiplicative factor on the resolution CF plus a
shift `d_i` of the residual. Derivation, the `(1 - a_i x)` measure term the task
sketch omitted (worth 27 MeV at the Z), and every gate: `STATE.md` sec. 0 and
`Documents/Resolution/NOTES.md` 2026-09-06.

Order of operations, and what each step cost:

| when | what | outcome |
|---|---|---|
| 20:00-20:30 | the refactor (`_chunk_resolution_parts`, cubic-spline `_dmat`) and the fluctuation form | all 7 pre-existing suites still pass; bit-identical at `upsample == 1` |
| 20:30-20:45 | `tests/test_fluctuation.py`, 7 checks | the density's mean is the closed form to 1e-10 |
| 20:42 | **GATE 1** on the real J/psi gun, 299 422 candidates | fluctuation vs residual **0.00087 / 0.00224 e-3** apart, both within 0.003 e-3 of the spec |
| 20:44-21:26 | the fluctuation cards, the 300 k ladder, `corr_census.py` | census: `E[Delta_i]` median -12.14 MeV against the residual form's 57.72 MeV displacement |
| 21:2x | `corr_coeff_max` | the `nojensen` arm put 19 of 300 000 densities negative; a bound on the COEFFICIENT (not the argument) at 0.08 fixes it and costs 0.049 MeV |
| 21:43-22:19 | the clip scan completed | **GATE 2**: `m_Z` swings 130 MeV over `corr_clip` 3/5/10/none |
| 22:19-23:00 | the 300 k fluctuation ladder | both corrections additive to 0.2 MeV |
| 22:48-23:52 | the J/psi v2 quadratic term, twice (the second time excluding the 16 bad tasks) | 16 755 046 candidates |
| 23:0x-23:40 | `fit_joint.py` and its three gates; the `--inject` closure re-measured | **12.14 % -> 0.0023 %** on `bfield_mode0` |
| 01:28-06:32 | the full-scale fluctuation fit on a preemptable H200 | **`m_Z` -11.06 +- 2.27, `Gamma_Z` -5.26 +- 4.16 MeV** |
| 01:39, 01:59 | the `--shape 7` and the 80-100 GeV window diagnostics | the K(m) basis is saturated; the window test is inconclusive |
| 06:38 | `cards/joint_v2.hdf5` rebuilt against the x16 quadratic term | 10.7 GB |
| 06:45 | **the Engaging SSH master expired** | needs `!eng-master` from a human |
| 06:52 | phase 2 restarted on CPU at 300 k + 300 k | running |
| 07:0x | phase 3's blocker identified | `extract_groups.py` needs `ioniurbanidx`, which `exportStepRecords=False` did not write; the per-group CF is in `cfmass_grp_*` instead and nothing reads it |

Dead ends worth not repeating:
* Sampling one candidate's density by making the plotting grid the `mobs`
  column gives every grid point its OWN `c_i`, `d_i` — a 3e-5 normalisation
  deficit that looks like a bug in the CF and is not (pitfall 10).
* The uncorrected residual form does not converge at 3.68 M any more than at
  300 k; job 22163315 spent 1:57 on its first variant and produced nothing.
* `--corr-clip` on a fluctuation-form card is meaningless and `fit.py` now
  refuses it.

---

### 2026-09-07 20:00-21:00 — the kernel programme, and a RETRACTION

**RETRACTED: "the FSR fold describes a sample radiating 1.66x more than the
reconstructed candidates".** The comparison behind it was wrong. I compared

    <u> = <-ln(m_post/m_pre)>   kernel npz, no mass cut          0.024032
                                selected candidates, 60-120      0.014447

and read the factor 1.66 as a defect of the kernel. It is not: the kernel is
built on the FULL gen sample precisely so that the fold can move a Born mass
anywhere, and the model's prediction for the SELECTED sample is the same gen
sample restricted to the window, which the truncation normalisation
(`norm_window` / `_norm_z`) is what implements. Measured on the same generator
merge (`genmerged_full.npz`, 29.27 M events):

| sample | `<u>` |
|---|---:|
| gen, inclusive | 0.027147 |
| gen, fiducial `pT > 5`, `\|eta\| < 2.4` (the kernel's cut) | 0.023456 |
| **gen, fiducial + `m_post` in [60, 120]** (the model's prediction) | **0.014234** |
| gen, fiducial `pT > 25` + window (the WRONG fiducial, for scale) | 0.010207 |
| **reconstructed, fully selected** | **0.014447** |

so the model and the data agree to **+2.1e-4 in `<u>`, 1.5 % relative**, not a
factor 1.66. The residual sign is that the data radiate slightly MORE than the
model, which pushes `m_Z` DOWN — the right sign for the -11 MeV — but 1.5 %,
not 66 %. The `pT > 5` fiducial is confirmed as the right one for this sample:
the selected candidates' sub-leading muon `pT` is 5.50 GeV at the 0.1 % point
and 9.22 at the 1 %, i.e. a ~5 GeV threshold, and the `pT > 25` fiducial's
`<u>` (0.0102) is nowhere near the data.

**The acceptance is also not the defect.** `A_sel(m_pre) = P(selected | m_pre)`
measured from the selected candidates against the generator's `m_pre` spectrum
is a top-hat (0.006 at 55.6 GeV, 0.70 at 61.6, plateau 0.95 over 85-118, 0.28
at 121.6, 0.13 at 124.6) — the 60-120 window acting through `m_post ~ m_pre`.
Divided by the loose gen acceptance in use it is **FLAT at 1.97 +- 0.05 across
61-118 GeV** and only falls outside the window, which is what the truncation
normalisation handles. So inside the window the acceptance shape in use is
right to a few per cent, and that residual slope is exactly what the floated
`K(m)` absorbs. (A degree-8 Bernstein cannot fit the top-hat: chi2/ndf =
30638/184. That is a statement about the parameterisation of a step, not about
the physics.)

New files: `zchannel/kern_from_selected.py` (the selected-sample kernel and
acceptance), `zchannel/data/kern_selected_band3.3e-4.npz` (3 682 662
candidates, 11 bands, 4529 atoms), `zchannel/data/acc_selected_d8.json`,
`zchannel/data/gen_selected.npz` (the selected candidates' own gen record, in
`fit_gen.py`'s format, so the generator-level chain can be run ON THEM).

**Kernel statistical precision**, on the selected sample: `<u> = 0.014447`,
`RMS(u) = 0.050139`, `N_eff = 2 993 821`, so `sigma(<u>) = 2.90e-5` = **2.64 MeV
on the peak position**. That is the floor a data-driven kernel could reach here
and it is NOT negligible against a 2.27 MeV statistical error.

**What IS there, and it has the right eta pattern.** `<u>` model vs data, both
with the same window applied, in the `|eta_lead|` bands the `m_Z` splits used:

| `\|eta_lead\|` | model (gen fid + window) | data (selected) | data - model | fitted `m_Z` (300 k) |
|---|---:|---:|---:|---:|
| 0 - 0.9 (barrel) | 0.014276 | 0.014476 | **+2.00e-4** | -35.79 +- 7.13 |
| 0.9 - 1.6 | 0.014235 | 0.014397 | +1.62e-4 | +0.08 +- 7.89 |
| 1.6 - 3.0 (endcap) | 0.014164 | 0.014160 | **-0.04e-4** | +4.07 +- 9.64 |
| inclusive | 0.014234 | 0.014367 | +1.33e-4 | -11.06 +- 2.27 (full stats) |

The barrel radiates 2.0e-4 more than the model says and the endcap agrees to
0.04e-4. `2e-4` of the mass is **18 MeV**, in the direction that pulls `m_Z`
DOWN, and the barrel-endcap `m_Z` difference is 40 +- 12 MeV. So the sign, the
size to a factor ~2, and the `eta` pattern all match. The cause is that the
selection acts on RECONSTRUCTED muons and the kernel's fiducial acts on
generated ones; the two differ in a way that correlates with the radiation.
That is the coordinator's hypothesis, at 1.5 % of `<u>` rather than 66 %.

**The decisive test: the kernel side CLOSES at generator level.** The
coordinator's step 1, run on the selected candidates' own gen record
(`zchannel/fit_gensel.py`, `data/fit_gensel_bands.log`): fit their GEN masses
with exactly the detector-level chain, (a) the pre-FSR mass against
lineshape (x) A (x) K and (b) the post-FSR mass against the full fold, with the
kernel/acceptance IN USE and with the pair rebuilt from these candidates.
`shape 5`, window 60-120, Born 50-130, no detector anywhere. MeV.

| `\|eta_lead\|` | pair | (a) pre-FSR, no fold | (b) post-FSR, full fold | (b) - (a) | detector-level `m_Z` |
|---|---|---:|---:|---:|---:|
| barrel | in use | -3.28 +- 1.90 | **+0.61 +- 2.05** | +3.89 | -35.79 +- 7.13 |
| barrel | selected | -1.51 +- 1.91 | +2.18 +- 2.07 | +3.69 | |
| transition | in use | +1.01 +- 2.31 | **+4.16 +- 2.51** | +3.15 | +0.08 +- 7.89 |
| transition | selected | +2.82 +- 2.32 | +5.79 +- 2.54 | +2.97 | |
| endcap | in use | -2.99 +- 2.42 | **-2.70 +- 2.63** | +0.29 | +4.07 +- 9.64 |
| endcap | selected | -1.26 +- 2.43 | -1.38 +- 2.70 | -0.12 | |
| inclusive | in use | -1.94 +- 1.25 | **+0.76 +- 1.36** | +2.69 | -11.06 +- 2.27 |
| inclusive | selected | -0.17 +- 1.26 | +2.30 +- 1.38 | +2.47 | |

**The fold is worth +2.7 MeV, not -11, and its `eta` spread is 3.6 MeV, not
40.** The chain reproduces the selected candidates' own gen masses to
+0.76 +- 1.36 MeV inclusively and to +0.6 / +4.2 / -2.7 per band. Rebuilding
the kernel AND the acceptance from those candidates moves `m_Z` by +1.5 MeV and
does not touch the `eta` pattern. `nm` 32768 -> 8192 moves `m_Z` by 0.01 MeV,
so the binning is not in this.

**RETRACTED: "what is left is the KERNEL".** It is not. The detector half
closes at +0.9 +- 2.1 MeV (kernel-free, delta lineshape) and the kernel half
closes at +0.8 +- 1.4 MeV (detector-free, gen masses). **The -11 MeV is in the
COMBINATION**, and the leading candidate is now identified.

### The per-candidate resolution is not independent of the mass

The likelihood is `prod_i p(m_i | sigma_i)` and the model computes it as
`int p(m') K_{sigma_i}(m_i - m') dm' / Z_i` — i.e. it assumes the Born
spectrum `p(m')` is the SAME for every candidate whatever its `sigma_i`.
Measured on the fitted sample (3 682 662 candidates, octiles of the absolute
`sigma_m`):

| `sigma_m` octile [GeV] | n | `<sigma>` | **`<m_gen>`** | `<sigma/m>` |
|---|---:|---:|---:|---:|
| 0.332 - 0.783 | 460 333 | 0.699 | **84.94** | 0.0083 |
| 0.783 - 0.914 | 460 333 | 0.849 | 88.16 | 0.0097 |
| 0.914 - 1.012 | 460 332 | 0.965 | 89.21 | 0.0109 |
| 1.012 - 1.103 | 460 333 | 1.056 | 90.12 | 0.0118 |
| 1.103 - 1.227 | 460 332 | 1.161 | 90.50 | 0.0129 |
| 1.227 - 1.403 | 460 333 | 1.308 | 90.84 | 0.0145 |
| 1.403 - 1.758 | 460 333 | 1.553 | 91.12 | 0.0171 |
| 1.758 - 11.9 | 460 333 | 2.597 | **91.28** | 0.0285 |

`rho(sigma, m_gen) = 0.168`, and the conditional mean of the TRUE mass runs
over **6.3 GeV** across the classes — the low-`sigma` candidates are a
genuinely low-mass-enriched population (`sigma` grows with `pT`, and `pT` with
the mass). Against this the same quantity in `sigma/m` classes moves only
1.1 GeV and `rho(sigma/m, m_gen) = 0.035`.

This mis-specification is invisible to BOTH closure tests that passed. The
generator-level fit has no `sigma` at all. The kernel-free residual fit has a
DELTA lineshape — every candidate's true mass is identically zero — so there is
no true-mass distribution left to correlate with `sigma`. It is exactly a
defect of the COMBINATION, which is where the -11 MeV was localised to.

It is also `eta`-dependent by construction (the endcap's `sigma` distribution
and its `sigma`-mass relation both differ from the barrel's), which is the
missing `eta` pattern.

**The test now running** (`sigslice.sh`, six absolute-`sigma` slices at 400 k
each, `K(m)` floated): inside a narrow slice the class-conditional Born
spectrum differs from the marginal by a SMOOTH function of `m`, which five
floated Legendre terms can absorb; inclusively one `K(m)` has to serve every
class at once and cannot. So the prediction is that the slices close and the
inclusive fit does not.

**The size of the mis-specification.** `p(m_gen | sigma class) / p(m_gen)`,
sextiles of the absolute `sigma_m`, normalised to 1 at the peak bin:

| `m_gen` | cls1 | cls2 | cls3 | cls4 | cls5 | cls6 |
|---|---:|---:|---:|---:|---:|---:|
| 60.5 | 5.18 | 0.81 | 0.32 | 0.25 | 0.18 | 0.15 |
| 75.5 | 2.67 | 1.51 | 0.70 | 0.59 | 0.53 | 0.43 |
| 90.5 | 1.07 | 1.06 | 1.00 | 0.97 | 0.96 | 0.96 |
| 105.5 | 0.16 | 0.68 | 0.52 | 1.13 | 1.58 | 1.73 |
| 117.5 | 0.00 | 0.23 | 0.52 | 0.72 | 1.78 | 2.48 |

(class edges in `sigma_m` [GeV]: 0.332, 0.827, 0.982, 1.103, 1.278, 1.601,
11.9.) The lowest-`sigma` class is a factor 5000 tilt across the window and is
essentially EMPTY above 110 GeV; the highest is a factor 16 the other way. The
model gives every one of them the same Born spectrum.

Two consequences, and both are testable:

1. **Inside a class the tilt is smooth.** Fitting `log` of the ratio with the
   same 5-term Legendre `K(m)` leaves an RMS residual of 0.03-0.07 for classes
   2-6 (1.0 for class 1, whose ratio hits zero). So a per-class fit with a
   floated `K(m)` should very nearly close.
2. **Inclusively it cannot be absorbed at all.** The class-averaged Born
   spectrum IS the marginal — `sum_c P(c) p(m'|c) = p(m')` — but the observed
   spectrum is `sum_c P(c) [p(m'|c) (x) K_c]`, and the model can only produce
   `sum_c P(c) [p(m') (x) K_c]`. Low true masses get NARROW kernels and high
   ones WIDE kernels; the model gives every mass the average mixture. That is
   an asymmetric smearing across the peak and there is no `K(m)` that repairs
   it, because `K(m)` multiplies the Born spectrum before the convolution and
   the defect is in the pairing of kernel width with mass.
