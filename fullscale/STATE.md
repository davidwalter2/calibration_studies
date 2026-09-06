# fullscale — STATE  (checkpoint 2026-09-06 19:45)

**Read this file top to bottom before touching anything.** The running log of
how each number was obtained is in `STATE_log.md` next to this file; this file
is what someone with no context needs.

Goal: the statistical precision and the MC closure of `m_Z` and `Gamma_Z` from
the unbinned CVH mass likelihood, alone (phase 1) and jointly with the J/psi
channel that fixes the momentum scale (phase 2), then with the full design
(phase 3).

Figures: `~/public_html/cvh/260906_fullscale/` (already has `index.php`).

---

## 0. THE ONE RESULT SO FAR, AND THE ONE PROBLEM

**Statistical precision, at the full 3.6 M candidates**, resolution fixed at
the MC truth, K(m) floated, Asimov (reference-point) information:

| | inverse Hessian | x1.109 sandwich |
|---|---:|---:|
| `sigma(m_Z)` | **1.98 MeV** | **2.19 MeV** |
| `sigma(Gamma_Z)` | **4.04 MeV** | **4.48 MeV** |

with K(m) FIXED it is 1.51 / 2.91 MeV, so floating the LO->MiNNLO shape costs
x1.31 / x1.39. (`Gamma_Z` at 4 MeV is already at the interesting scale — the
Z-width sensitivity note targets ~2 MeV.)

**The problem, which is the phase-1 finding.** Both corrections of
`resolution/oddmoment/MASSCFTERM_SPEC.md` are expansions in the RESOLUTION
fluctuation, and what they are fed is `delta_i`, the deviation from the
reference mass. At the J/psi those coincide (+-0.35 GeV on 3.0969 GeV); at the
Z they do NOT — the window is +-27 sigma and the deviation out there is FSR
and the Breit-Wigner tail. Fed the full `delta`, the exact Jensen map moves the
residual by a **median 57.7 MeV and up to 10.7 GeV**, against the **20.6 MeV**
mean shift it exists to apply, and 39 % of candidates move by more than five
times that. The 300 k fit then runs away to `Gamma_Z = -421 MeV` with K(m)
coefficients of order one compensating.

The fix is committed: **`corr_clip`** (rabbit `b64f49f`), the corrections'
argument domain in units of `sigma_i`. Inside it, bit-for-bit unchanged;
outside, both corrections SATURATE instead of extrapolating (the Jensen map is
continued with unit slope, so the residual stays strictly monotone and the
log-Jacobian is exactly zero there). `corr_clip = 0` reproduces the J/psi
behaviour, so every gate the spec measured is untouched.

**What is not yet known: whether the clipped answer is stable in the clip.**
That is what jobs 22163745 / 22163752 and the local `clip3`/`clip5` fits are
for. If `m_Z` and `Gamma_Z` are flat over `corr_clip` 3–10 the treatment is
under control. If they are not, the corrections have to be reformulated to act
on the resolution fluctuation itself rather than on the residual, which is a
real piece of work and NOT a tuning exercise.

---

## 1. WHAT EXISTS

### The merged rabbit — this is not on any remote
`/work/submit/david_w/ZMass/rabbit-material`, branch **`material-resolution`**,
HEAD **`b64f49f`**. It is `material-resolution` (self-consistent resolution,
`MaterialCFTerm`, `ExternalParams`) with **`z-lineshape-kernel` merged in**
(`ZGammaLineshape`, `norm_window`, in-graph `upsample`), plus three commits
this session:

| commit | what |
|---|---|
| `4f762c0` | the merge itself. All five suites green as scripts; the additivity gate (each parent's features off reproduces the other parent) is 5.3e-14, mostly bit-identical |
| `58a3f82` | the floated smooth **K(m)** in the provider (`shape=`, `shape_window=`). Without it the LO->MiNNLO ratio is +76 MeV on `Gamma_Z` |
| `5658252` | the **exact Jensen map** in `MassCFTerm` (`jensen_s2=`, `jensen_mode=`). The branch shipped the hooks and no implementation |
| `b64f49f` | **`corr_clip`**, above |

Tests (run as SCRIPTS, not pytest — two of them use positional args that pytest
reads as fixtures, which is pre-existing on BASE and both parents):
```bash
cd calibration_studies/fullscale
for T in test_unbinned_mass test_material_cf test_global_term \
         test_unbinned_norm test_zgamma_kernel test_jensen test_zgamma_shape; do
  THREADS=16 ./run_tf.sh python3 -u ../../rabbit-material/tests/$T.py; done
```
All pass. `test_jensen.py` 7/7 and `test_zgamma_shape.py` 7/7 are new here.

### Caches and cards (`fullscale/runs`, `fullscale/cards`; both gitignored)

| file | content | how it was made |
|---|---|---|
| `runs/zpairs_dyv2.npz` | 3 663 056 Z candidates, **373 DY tasks** | `cf_inmaker.py pairs --files <dyv2> --ntasks 0 --mass-window 91.1876 60` |
| `runs/zpairs_dyv2_tail.npz` | 70 267, the 7 tasks that finished late | same, `--files @runs/missing_tasks_dyv2.txt` |
| `runs/zpairs_dyv2_full.npz` | **3 733 323, all 380 tasks** (4.62 GB) | `append_pairs.py --base ... --tail ...` |
| `runs/zpairs_dyv2_jac.npz` | 3 704 155, **377 tasks, WITH the (n,92) mass Jacobian** (5.71 GB) | same + `--jac-parmtypes 14 15`. **3 tasks still to append** — see §4 |
| `runs/quad_dyv2.npz` | DY quadratic term, **380 tasks**, 3 751 687 candidates | `globalfit/extract.py --no-mass --parmtypes 14 15 --max-chi2-ndof 3 --max-hess 1e8 --max-grad 1e6 --max-dEref-p 0.01` |
| `runs/quad_jpsiv1.npz` | J/psi **v1** quadratic, 15 965 797 candidates | same |
| `runs/xcheck_v1.npz`, `runs/xcheck_v2.npz` | the same 120 J/psi tasks through v1 and v2 | same, `--ntasks 120` |
| `cards/z_full380.hdf5` | **3 682 662 candidates after cuts** (3.63 GB) | `make_card.py --pairs runs/zpairs_dyv2_full.npz --shape 5 --fsr ... --acc ...` |
| `cards/z_full.hdf5` | the 373-task twin (3.56 GB) | same, `--pairs runs/zpairs_dyv2.npz` |
| `cards/z_n300k.hdf5` | a 300 000 random subsample, same model | same, `--maxn 300000` |
| `../zchannel/data/kern_loose_band3.3e-4.npz` | the FSR kernel: **11 bands, sigma_cap 3.3e-4, 14 315 atoms, 13.03 M gen events, pT>5, \|eta\|<2.4** | `fit_gen.py kernel --gen data/genmerged_full.npz --acc-pt 5 --acc-eta 2.4 --sigma-cap 3.3e-4 --u-fine 2e-5 --wclip 100 --bands 70 80 85 88 91 94 98 105 115 130` |
| `../zchannel/data/acc_loose_d8.json` | the acceptance, Bernstein 8, pT>5 | already existed and already matched |

### Scripts (all committed)

| file | what |
|---|---|
| `run_tf.sh` | runs a python script in the rabbit TF singularity image with the MERGED worktree first on PYTHONPATH. `RABBIT=`, `THREADS=`, `--ceph` |
| `make_card.py` | the Z card. Weights, both corrections, provider-side FSR+acceptance+K(m), documented cuts. REFUSES to build if the rabbit it imports lacks a piece of the model |
| `chunkfit.py` | value/grad/Hessian accumulated over candidate chunks, plus the sandwich meat by forward-mode autodiff |
| `fit.py` | the driver. `--fix`, `--ares/--jensen/--corr-clip` overrides, `--start-from`, truth comparison, projections |
| `validate_hessian.py` | the chunked Hessian against the monolithic one |
| `check_normz_sigma.py` | the one approximation the merge leaves in |
| `append_pairs.py` | merge a tail cache in; refuses unless disjoint on (run,lumi,event) |
| `report.py` | the JSONs -> the phase tables |
| `plot_inputs.py`, `plot_postfit.py` | the figures |
| `run_extract.sh`, `run_phase1.sh`, `run_phase2.sh`, `run_full380.sh` | the launchers |
| `make_joint_card.py` | phase 2, SCAFFOLD ONLY (a subagent was drafting it; `sum_quadratic` is finished) |
| `patches/{zgamma_shape,unbinned_jensen}.py` | idempotent patches, already applied to `rabbit-material` |
| `../engaging/fullscale_{gpu,variants,clipscan}.sbatch` | the GPU jobs |
| `stage_eng.sh` | push the merged rabbit + scripts + a card to Engaging |

---

## 2. WHAT IS RUNNING RIGHT NOW

### On submit82 — detached, `nohup`/`setsid`, logs in `fullscale/logs/`

| pid | label | command | output | done when |
|---|---|---|---|---|
| 334085 | `n300k_nojensen` | `fit.py --card cards/z_n300k.hdf5 --fix k_hit k_ms k_ioni k_rad --chunk 32768 --jensen off` | `results/fit_n300k_nojensen.json` | the json exists |
| 334080 | `n300k_noboth` | same `--ares off --jensen off` | `results/fit_n300k_noboth.json` | ditto |
| 334599 | `n300k_jshift` | same `--jensen shift` | `results/fit_n300k_jshift.json` | ditto |
| 509886 | `n300k_clip5` | same `--corr-clip 5` | `results/fit_n300k_clip5.json` | ditto |
| 509887 | `n300k_clip3` | same `--corr-clip 3` | `results/fit_n300k_clip3.json` | ditto |

**Caveat**: 334080/334085/334599 were started with `nohup ... &` from a tool
shell and share session 331982; `nohup` protects them from SIGHUP but not from
a kill of the process group. 509886/509887 are in their own sessions
(`setsid`). If any of the first three vanish, relaunch with the command in the
table — each takes ~45 min at 300 k.

### On Engaging (ORCD) — `eng 'timeout 30 squeue -u david_w'`

| job | what | output |
|---|---|---|
| **22161787** | `zfit`, the 373-task card, base variant, H200 | `~/orcd/pool/zmass/fitresults/fit_base.json`, log `~/orcd/pool/zmass/engaging/zfit_22161787.out` |
| **22163315** | `zvar`, the six variants UNCLIPPED on `z_full380.hdf5` | `fitresults/fit_f380_{base,noares,nojensen,noboth,jshift,noshape}.json` |
| **22163745** | `zclip`, `corr_clip` in {3,5,10,0} on the same card | `fitresults/fit_f380_clip{3,5,10,0}.json` |
| **22163752** | `zvar`, the six variants at `corr_clip = 5` | `fitresults/fit_f380c5_*.json` |

Bring them back with
`rsync -a engaging:orcd/pool/zmass/fitresults/ fullscale/results/` then
`python3 report.py --results 'results/fit_f380*.json'`.

Engaging notes: `eng-master` may need a human Duo touch; wrap every remote
command in `timeout`; **H200, not L40S**; `chunk 262144` OOMs on 141 GB (the
lineshape CF gather is a `(chunk, nt_int)` int64 tiled per pfor parameter) —
**use `--chunk 32768`**.

---

## 3. HOW TO REPRODUCE EACH STEP

```bash
FS=/work/submit/david_w/ZMass/calibration_studies/fullscale
Z=/work/submit/david_w/ZMass/calibration_studies/zchannel
RES=/work/submit/david_w/ZMass/calibration_studies/resolution
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate     # numpy/uproot side

# 1. pairs cache  (~25 min, ceph; submit82 has ceph, submit50/51 also)
python3 $RES/cf_inmaker.py pairs \
  --files /ceph/submit/data/user/d/david_w/ZMass/cvh/dymc_8p5M_260906_v2 \
  --ntasks 0 --cache $FS/runs/zpairs_dyv2_full.npz --mass-window 91.1876 60
# add --jac-parmtypes 14 15 for the phase-2 version (the (n,92) mass Jacobian)

# 2. quadratic term  (~2.5 min for DY, ~16 min for a J/psi production)
python3 $RES/globalfit/extract.py --files <production> --ntasks 0 \
  --parmtypes 14 15 --no-mass --max-chi2-ndof 3 --max-hess 1e8 --max-grad 1e6 \
  --max-dEref-p 0.01 -j 16 -o $FS/runs/quad_<tag>.npz
#   --no-mass is MANDATORY: these productions run exportStepRecords=False and
#   ship `radvgrid` alone, which the offline CF build rejects as a partial
#   radiative export.

# 3. card  (~13 min, mostly npz decompression)
THREADS=32 $FS/run_tf.sh python3 -u $FS/make_card.py \
  --pairs $FS/runs/zpairs_dyv2_full.npz --shape 5 \
  --fsr $Z/data/kern_loose_band3.3e-4.npz --acc $Z/data/acc_loose_d8.json \
  -o $FS/cards/z_full380.hdf5

# 4. fit  (locally ~45 min at 300 k; on an H200 ~20 min at 3.6 M)
THREADS=48 $FS/run_tf.sh python3 -u $FS/fit.py --card $FS/cards/z_full380.hdf5 \
  --fix k_hit k_ms k_ioni k_rad --chunk 32768 --corr-clip 5 \
  --label base -o $FS/results/fit_base.json

# 5. tables and figures
python3 $FS/report.py --results "$FS/results/fit_*.json"
python3 $FS/plot_inputs.py --pairs $FS/runs/zpairs_dyv2_full.npz
THREADS=8 $FS/run_tf.sh python3 -u $FS/plot_postfit.py \
  --card $FS/cards/z_full380.hdf5 --fit $FS/results/fit_base.json --nsub 4000
```

---

## 4. THE PLAN, AND THE DECISIONS ALREADY TAKEN

### Decisions
1. **Phases 2 and 3 run on J/psi v2**, `/ceph/.../cvh/jpsimc_20M_260906_v2/`
   (multi-stream, complete at 1642 tasks). Measured reason: on the SAME 120
   tasks, v2's parmtype-15 (material) information is **120x the trace and 666x
   the median diagonal** of v1's, while parmtype 14 (field) is bit-equal. v1's
   mean-loss-only quadratic term is very nearly blind to the material amounts.
   v1 is kept ONLY as that cross-check.
2. **The DY leg is all 380 tasks.** The phase-1 300 k ladder and the first
   H200 fit used the 373-task cache; the final numbers use the 380-task one.
3. **Cuts**: `m_obs` in [60,120] (the production selected on
   `60 < Jpsitrk_mass < 120`, the PRE-REFIT mass — the mismatch is **0.191 %
   and one-sided**), `chi2/ndof < 3`, `sigma_m/m < 0.10`. 3 682 662 of
   3 733 323 survive (98.6 %).
4. **Weights**: `genweight` clipped at 100x the modal, rescaled to mean 1.
   4.80 % negative, `N_eff/N = 0.8130`, so every error x1.109 in the sandwich.
5. **Corrections**: both on, `jensen_mode="exact"`, and now `corr_clip`
   (default 5 in `make_card.py`). `f_ang` is folded into `jensen_s2` per
   candidate on the Z (median 8e-5, i.e. negligible); on the J/psi v2 it comes
   from `Jpsi_covrefmom`, truth-free.
6. **K(m)**: 5 Legendre terms, floated, over the FIT window.
7. **The Hessian is chunk-accumulated.** Not optional: the monolithic
   construction is 115.6 GB at 300 k and TWO parameters.
8. **Resolution fixed at the MC truth** in phases 1-2 (`--fix k_hit k_ms
   k_ioni k_rad`); phase 3 replaces the knobs with the parmtype-15 amounts.
9. Alignment is fixed at the MC truth throughout this test.

### Immediately outstanding
* **Append the 3 late tasks to `runs/zpairs_dyv2_jac.npz`** exactly the way the
  7 were appended to `zpairs_dyv2.npz`:
  ```bash
  D=/ceph/submit/data/user/d/david_w/ZMass/cvh/dymc_8p5M_260906_v2
  grep -o "task_[0-9]*/globalcor" $FS/logs/zpairs_dyv2_jac.log | cut -d/ -f1 | sort -u > /tmp/used.txt
  ls -d $D/task_* | xargs -n1 basename | sort > /tmp/all.txt
  comm -13 /tmp/used.txt /tmp/all.txt | while read t; do ls $D/$t/globalcor_*.root; done > $FS/runs/missing_jac.txt
  python3 $RES/cf_inmaker.py pairs --files @$FS/runs/missing_jac.txt --ntasks 0 \
      --cache $FS/runs/zpairs_dyv2_jac_tail.npz --mass-window 91.1876 60 --jac-parmtypes 14 15
  python3 $FS/append_pairs.py --base $FS/runs/zpairs_dyv2_jac.npz \
      --tail $FS/runs/zpairs_dyv2_jac_tail.npz -o $FS/runs/zpairs_dyv2_jac_full.npz
  ```
* **Collect the four Engaging jobs** and run `report.py`.
* **Decide on `corr_clip`** from the scan (see §0).
* Finish `make_joint_card.py`.

### Phase 2 (J/psi v2 + Z)
`./run_phase2.sh pairs quad` (it refuses a partial production), then `card`,
then `fit`. The joint card: quadratic term on parmtypes 14+15 summed over both
productions, a J/psi `MassCFTerm` with a **delta kernel at the PDG mass and
`scale_param=None`** (the scale transfers through the field modes, not through
a free alpha), and the Z term. Report `m_Z`/`Gamma_Z` closure, the pulls of the
92 globals, and `rho(m_Z, field/material)`.

### Phase 3 (waits for the user's message)
Per-group exponents on both legs -> `MaterialCFTerm` with the parmtype-15
parameters shared between the hit-chi2 term and both mass terms; hit-class
parameters from the mass terms; corrections truth-free from `Jpsi_covrefmom`;
Asimov/toy pulls.

---

## 5. PITFALLS FOUND (each cost time; none is obvious)

1. **`extract.py`'s mass path is blocked on these productions** —
   `exportStepRecords=False` ships `radvgrid` alone and the guard rejects it.
   Use `--no-mass`; the mass side comes from `cf_inmaker.py pairs`, and the
   per-candidate mass Jacobian from its `--jac-parmtypes`.
2. **`chi2ndof` is not stored on the `--no-mass` path**, so the chi2 cut must
   be given to `extract.py` and cannot be deferred.
3. **`tf.gather` / `tensor_scatter_nd_update` gradients are `IndexedSlices`**
   and `tf.autodiff.ForwardAccumulator` cannot differentiate through one. The
   parameter map in `chunkfit.py` is a dense affine map for that reason.
4. **HDF5 datasets are stored FLAT** with an `original_shape` attribute; a raw
   `np.asarray(g[k])` is 1-D. Take arrays off the TERM.
5. **`nm=32768` with `fsr=` is a 16 GB constant** — the fold is a dense
   `(nm, n_born)` matrix. `make_card.py` defaults to `nm=8192` (1.0 GB,
   `dm = 9.8 MeV`, 1/256 of `Gamma_Z`).
6. **`np.savez_compressed` of a multi-GB cache takes minutes**, during which
   the file exists and is a truncated zip. Wait for the reader's `wrote` line,
   not for the file.
7. **git refuses to fetch into a checked-out branch** — `stage_eng.sh`
   detaches first.
8. The `_norm_z` truncation is evaluated at the STORED class sigma, not the
   self-consistent one. Measured cost: **0.016 MeV on `m_Z`**, 1 % of the
   statistical error (`check_normz_sigma.py`).
9. `rho(shape4, shape5) = -0.964` at detector level — the Legendre basis is
   orthogonal over the Born window, not over the observed spectrum.
