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
