# fullscale — STATE  (checkpoint 2026-09-06 21:00)

**Read this file top to bottom before touching anything.** The running log of
how each number was obtained is in `STATE_log.md` next to this file; this file
is what someone with no context needs.

Goal: the statistical precision and the MC closure of `m_Z` and `Gamma_Z` from
the unbinned CVH mass likelihood, alone (phase 1) and jointly with the J/psi
channel that fixes the momentum scale (phase 2), then with the full design
(phase 3).

Figures: `~/public_html/cvh/260906_fullscale/` (already has `index.php`).

---

## 0. THE STATISTICAL NUMBER, AND THE PHYSICS PROBLEM — NOW SOLVED

**Statistical precision, at the full 3.6 M candidates**, resolution fixed at
the MC truth, K(m) floated, Asimov (reference-point) information:

| | inverse Hessian | x1.109 sandwich |
|---|---:|---:|
| `sigma(m_Z)` | **1.98 MeV** | **2.19 MeV** |
| `sigma(Gamma_Z)` | **4.04 MeV** | **4.48 MeV** |

with K(m) FIXED it is 1.51 / 2.91 MeV, so floating the LO->MiNNLO shape costs
x1.31 / x1.39.

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
mass than a typical candidate. This bounds a per-candidate CONSTANT computed
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

## 2. WHAT IS RUNNING

### On submit82 — detached, logs in `fullscale/logs/`
* the five 300 k **residual-form** variant fits the previous session started;
  `base/noares/noshape/nojensen/noboth/jshift` have landed in `results/`,
  `clip3`/`clip5` (pids 509886/509887) are the last two. They are the
  "clipped, for the record" column and nothing depends on them.
* the five 300 k **fluctuation-form** fits (`fit_n300kfl_{base,noares,
  nojensen,noboth,noshape}.json`, logs alongside) — the phase-1 ladder at
  300 k, which is what tells us the reformulation is stable before the
  full-scale numbers land.
* `cf_inmaker.py pairs` on **J/psi v2, `--ntasks 600`** ->
  `runs/jpairs_v2_n600.npz`, the phase-2 J/psi leg (~7.2 M candidates). 600
  tasks, not 1642: the J/psi leg is not statistics-limited here (299 k gun
  candidates already give sigma(alpha) = 0.017e-3, i.e. 1.5 MeV at the Z; 7 M
  give 0.003e-3 = 0.3 MeV, an order below sigma(m_Z) = 2 MeV), and the cache
  and the card scale linearly with it.

### On Engaging (ORCD) — `eng 'timeout 30 squeue -u david_w'`

| job | what | status |
|---|---|---|
| **22161787** | `zfit`, 373-task **residual** card, base | running |
| **22163315** | `zvar`, six variants UNCLIPPED on `z_full380.hdf5` | running — this is the "unclipped runaway at full scale" column |
| ~~22163745~~ | `zclip`, `corr_clip` in {3,5,10,0} | **CANCELLED** — superseded by the reformulation; the clip dependence is covered at 300 k locally |
| ~~22163752~~ | `zvar` at `corr_clip = 5` | **CANCELLED**, same |
| ~~22161787~~ | the 373-task base fit | **CANCELLED** — redundant with 22163315's `base` at 380 tasks, and it was holding the GPU cap |
| **22167631** | `zvarfl`, the five variants on **`z_full380_fl.hdf5`** (fluctuation form) | **the phase-1 FINAL job**; `fullscale_variants_fl.sbatch` |

Engaging notes: `eng-master` may need a human Duo touch; wrap every remote
command in `timeout`; **H200, not L40S**; **`--chunk 32768`** (262144 OOMs).

---

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

### Phase 1 final (Z alone, 380 tasks, fluctuation form)
Build `cards/z_full380_fl.hdf5` and `cards/z_n300k_fl.hdf5`; run
`base / --ares off / --jensen off / both off / K(m) fixed`; quote
fitted - generator (`m_Z` 91.153509740726733, `Gamma_Z` 2.4932018986110700)
with statistical errors and the x1.109 sandwich; keep the residual-form
(clipped and unclipped) numbers alongside.

### Phase 2 (J/psi v2 + Z)
`./run_phase2.sh pairs quad` (it refuses a partial production), then the joint
card: the quadratic term on parmtypes 14+15 summed over both productions, a
J/psi `MassCFTerm` with a **delta kernel at the PDG mass and
`scale_param=None`** (the scale transfers through the field modes, not a free
alpha), and the Z term. Report `m_Z`/`Gamma_Z` closure, the pulls of the 92
globals, `rho(m_Z, field/material)`.

### Phase 3 (full design)
Per-group exponents on both legs -> `MaterialCFTerm` with the parmtype-15
parameters shared between the hit-chi2 term and both mass terms; hit-class
parameters from the mass terms; corrections truth-free from `Jpsi_covrefmom`;
Asimov/toy pulls; the group-leader table.

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
