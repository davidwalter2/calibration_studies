# CLEANUP_STATE — resume file for the code cleanup of `calibration_studies`

Branch `resolution-energy-loss-corrections`.  Two jobs:

1. **ONE UNIT CONVENTION** for the material / hit-class terms — DONE.
2. **FINAL-STATE CODE**: drop superseded scripts, dead paths and historical
   commentary — IN PROGRESS.

Constraints: do NOT edit any `.md` but this one (a separate agent owns the
documentation); do NOT touch `CMSSW_*` (a third agent owns the C++).

## ENVIRONMENT NOTES

* `submit82` has NO readable ceph (`runs/perhit/*`, `runs/vtxres/{vtx,mass}.npz`
  and the `cards` trees are symlinks into it).  `ssh submit50` works and has
  ceph; use it for anything that reads those.
* The TF/rabbit environment is `hitlik/run_tf.sh`, `vtxres/run_tf.sh`,
  `matres/run_tf.sh` (singularity image + `rabbit-vmass` / `rabbit-material`).
* Nothing in `runs/` is tracked (`.gitignore`).

---

# JOB 1 — ONE UNIT CONVENTION (DONE)

## The convention

`rabbit.unbinned.MaterialCFTerm` applies `k_g = value * group_units[g]` and
`eps_c = value * hit_units[c]`.  The card may float a rescaled variable for
minimiser conditioning; the PHYSICAL quantity is always `k` (log material
amount) and `eps` (linear hit-variance scale).  Everything that LEAVES a term
— Fisher/Hessian/score matrices, fitted values and errors, priors, injected
amounts, tables — is now reported in physical units, with the factor READ from
the object that carries it.  No tool needs a unit flag any more.

## The helper (`resolution/matres/groups.py`, importable from every pipeline)

| function | returns |
|---|---|
| `card_group_units(ngroups, groups_file, whiten=True)` | the ONE definition of the card unit: `1/gprior_g` when whitened, else 1 |
| `units_for(param_names, group_params, group_units, hit_params, hit_units)` | `units` per parameter, in `param_names` order |
| `term_units(term)` | the same, read off a built `MaterialCFTerm` |
| `card_units(card_aux, param_names)` | the same, read off a card's `global_index_map` |
| `to_physical(v, u)` / `matrix_to_physical(M, u)` / `gradient_to_physical(g, u)` | `v*u`, `M/(u_i u_j)`, `g/u` |

`hitlik_term.build(..., group_units=None)` now TAKES the units (default =
physical, ones) and is the only place a term's units are set;
`make_hitlik_card.py` no longer monkey-patches `term.group_units` /
`term._gunits` after the fact.  All three card builders
(`hitlik/make_hitlik_card.py`, `vtxres/make_vtx_card.py`,
`matres/make_material_card.py`) get their units from `card_group_units`;
`make_material_card.py` additionally GATEs them against the quadratic
catalog's own `param_scales` whitening.

## Flags removed

* `hitlik/efficiency.py --prior-power`
* `hitlik/recovery.py --prior-sigma`
* `hitlik/final_table.py --inj-prior` (the prior is now the group's tier prior)
* `matres/report_fit.py --physical` (physical columns are always printed when
  the card declares units)

Callers updated: `hitlik/perhit/run_stage2.sh`, `vtxres/run_all.sh`.

## Scripts switched to physical reporting

`fisher_cmp.py`, `vtxres/fisher_vtx.py`, `perhit/fisher_joint.py` (H, J, G and
`grad0` written physical, with a `param_units` provenance key);
`efficiency.py`, `recovery.py` (section B), `final_table.py` (injection
block), `subspread.py`, `perhit/certify.py`, `perhit/saturation.py`,
`matres/report_fit.py`.

## TWO REAL BUGS THIS EXPOSED

1. **`perhit/fisher_joint.py` added a CARD-unit mass Hessian to a PHYSICAL
   residual Hessian** (`H + HM`), i.e. the mass term entered the joint `1/gprior^2`
   (400x for a 0.05-prior group) too strong.  Fixed by converting both to
   physical before the sum.  No published number came from that file
   (`eff_hitmass` / `eff_refmass` are not quoted in any STATE), so nothing in
   the notes changes.
2. **`perhit/saturation.py` used a prior of 1.0 for every material group**
   on matrices that are in physical `k`, i.e. a prior 1/gprior = 20x too loose,
   so `N_sat = n0 (sigma_free/sigma_prior)^2` came out 400x too SMALL.  Fixed
   to the parmtype-15 tier prior.  This DOES change published numbers — see
   "numbers that move" below.

## GATE — efficiency, sandwich/quoted, prior-free, with NO flags

`hitlik/efficiency.py` re-run on the CACHED H/J, no unit flags:

| file / cset | quantity | published | re-run |
|---|---|---|---|
| `runs/hitlik/fisherHJ20k.npz` `0123` | MATERIAL EFF marginal | 1.825 (1.62-2.12) | **1.825 (1.616-2.124)** |
| | MATERIAL EFF prior-free | 2.093 (1.52-3.71) | **2.093 (1.521-3.707)** |
| | sandwich/quoted CF / chi2 | 0.974 / 1.338 | **0.974 / 1.338** |
| | HIT EFF marginal / prior-free | 1.204 / 1.221 | **1.204 (1.069-1.607) / 1.221 (1.055-1.565)** |
| | HIT sandwich/quoted CF / chi2 | 1.052 / 1.222 | **1.052 / 1.222** |
| `runs/perhit/fisherHJ.npz` `hit` | EFF marginal / prior-free | 1.097 / 1.171 | **1.097 / 1.171** |
| | sandwich/quoted CF / chi2 | 0.529 / 0.563 | **0.529 / 0.563** |
| `runs/perhit/fisherHJ.npz` `ref` | EFF marginal / prior-free | 1.486 / 1.393 | **1.486 / 1.393** |
| `runs/perhit/fisherHJ_all.npz` `all` | EFF marginal / prior-free | 1.510 / 1.657 | **1.510 / 1.657** |
| | sandwich/quoted marginal CF / chi2 | 0.905 / 0.957 | **0.905 / 0.957** |
| | ... prior-free | 0.939 / 1.204 | **0.939 / 1.204** |
| `runs/vtxres/fisherHJ.npz` `vtx` | sandwich/quoted cf/gauss/gaussq | 0.885 / 1.040 / 1.099 | **0.885 / 1.040 / 1.099** |
| | MATERIAL EFF marginal / prior-free / claims | 2.659 (2.02-3.46) / 2.778 (1.85-15.4) / 1.006 | **2.659 (2.225-3.204) / 2.778 (1.851-15.406) / 1.006** |
| | MATERIAL prior-free sandwich/quoted CF / chi2 | 0.961 / 1.543 | **0.961 / 1.543** |
| | HIT EFF marginal / prior-free / claims | 1.215 (0.97-1.46) / 1.115 (0.50-1.30) / 0.968 | **1.215 (0.972-1.457) / 1.115 (0.501-1.304) / 0.968** |
| | HIT prior-free sandwich/quoted CF / chi2 | 1.061 / 1.177 | **1.061 / 1.177** |

The `runs/vtxres/fisherHJ{,_mj}.npz` and `runs/perhit/mass_HJ.npz` caches were
written in CARD units by the old code and were MIGRATED in place to physical
(`H/(u u^T)`, `J/(u u^T)`, `G/u`, plus a `param_units` key).  The card-unit
originals are kept beside them as `*.cardunits.bak.npz`.  The migration is
exactly `groups.matrix_to_physical` with `units_for(params, gnames,
card_group_units(ng, materialGroups50.txt))`.

## GATE — cards bit-identical

One small card per builder, built from the pristine `HEAD` worktree and from
the new tree with identical arguments, compared dataset by dataset with
`hdf5plugin` + `h5py`:

| builder | card | result |
|---|---|---|
| `hitlik/make_hitlik_card.py` | 300 tracks x 4 comps, `--whiten --inject material_tib_support:0.00243951` | **53 datasets bit-for-bit identical** |
| `vtxres/make_vtx_card.py` | 391 DY candidates, `--whiten --inject material_bpix_support6:0.00243951` | **52 datasets bit-for-bit identical** |
| `matres/make_material_card.py` | 400 gun candidates, `--whiten --quad-npz` | **62 datasets bit-for-bit identical** |

(only `auxiliary/global_index_map/provenance` differs, and only because it
records the `-o` path.)  `NLL(0)` and `max|grad|` agree to every printed digit
in all three.

## NUMBERS THAT MOVE (for the documentation agent)

Nothing about a ratio changes.  What changes is the UNIT of every absolute
sigma / value / error that used to be printed in card units — it is now
physical, i.e. multiplied by `1/gprior_g` (20x for a 0.05-prior group, 100x
for a 0.01-prior one):

* `vtxres/STATE.md` "WHAT THE VERTEX TERM MEASURES" and "VERTEX vs MASS vs
  JOINT" sigma tables, and the `certify` / `recovery` value+error columns.
* `hitlik/perhit/STATE.md` the `recovery.py` sigma column (e.g. 0.00183
  becomes 0.0366) and `final_table.py`'s injection block.
* `hitlik/STATE.md` is unaffected: it already quoted physical `k`.

**`perhit/saturation.py` N_sat, `runs/perhit/fisherHJ.npz`, `--cset hit`,
8000 tracks** (the prior-unit fix, bug 2 above):

| arm | published median N_sat (MATERIAL) | corrected |
|---|---|---|
| CF | 679 (247-4.5e3), max 4.06e4 | **6.37e5 (2.9e4-4.7e6), max 1.01e8** |
| chi2 | 1.05e3 (300-8.9e3), max 2.22e6 | **9.53e5 (3.6e4-7.3e6), max 5.56e9** |

The HIT-CLASS saturation points are unchanged (14.5 / 14.2, max 307 / 302):
their prior was already the physical 1.0.  **The recommendation that rests on
these numbers ("2 M tracks puts every material group past saturation") no
longer follows** — the median group needs ~6e5 tracks and the worst ~1e8.

## LEFT ALONE, DELIBERATELY

The card unit of a parmtype-15 material parameter is `value = k * gprior`
(`group_units = 1/gprior`), so one tier prior is `gprior**2` in card units.
That is the inverse of what `make_global_term.param_scales`' docstring claims
and of what its log line says ("1 card unit = 1 prior sigma"), and it does not
equalise curvatures the way a unit-prior whitening would.  It is CONSISTENT
end to end (`theta_card = theta_raw * s` for the quadratic gradient/Hessian,
`prior_sigmas * pscale`, `group_units = 1/pscale`), it is inside the fit, and
changing it would change every card — so only the misleading comments were
corrected, not the convention.

---

# JOB 2 — FINAL-STATE CODE (in progress)

## Rules applied

* Comments and docstrings say what the code does and why it is correct.  No
  dates, no "used to", no "the first attempt", no agent/person narrative, no
  references to another file's line numbers, and nothing about the withdrawn
  "prior units" bug report.
* A superseded script is deleted only after grepping the whole repo for its
  basename (`.py` imports, `.sh` callers, `.md` citations) AND checking it is
  not the producer of a number or figure still quoted in a STATE/SUMMARY file.
* Interfaces of surviving scripts stay as they are, except the unit flags of
  Job 1.
* Every surviving python script must import and pass `--help`.

## Done so far

**`hitlik/`, `hitlik/perhit/`, `vtxres/`, `matres/`, `pubhtml.py`,
`check_slide_overflow.py`, `md2html.js`** (commit `dde4b37`, `cb60314`):

| file | removed |
|---|---|
| `hitlik/tails.py` | 2 (the "bug the first version had", "it used to take a component index ... (2026-09-10)") |
| `hitlik/perhit/gates.py` | 2 (dated "measured 2026-09-10", "it cost an hour on 2026-09-10") |
| `hitlik/perhit/xcum_perhit.py` | 4 ("the WG question", "the task expects", prototype comparison) |
| `hitlik/plot_hitlik.py` | 1 ("`rows` used to be a component INDEX") |
| `hitlik/cost.py`, `hitlik/perhit/saturation.py`, `hitlik/perhit/plot_xcum.py`, `hitlik/perhit/run_all.sh`, `hitlik/perhit/fisher_joint.py`, `hitlik/perhit/run_prod.sh` | 1 each (dates, "the WG asked", "no longer needed") |
| `vtxres/gates.py`, `vtxres/run_prod.sh`, `matres/extract_groups.py` | 1 each |
| `matres/run_joint.sh` | 2 (dated recipe header, the removed `--physical` flag) |
| `pubhtml.py` | 2 ("the old template path ... is gone") |
| `check_slide_overflow.py` | 1 ("a first attempt at 2.5 %") |
| `hitlik/run_ladder.sh`, `vtxres/run_ladder.sh` | 1 each — and both were WRONG: `--whiten` makes the card float `k * prior_sigma`, not `k / prior_sigma` |
| `md2html.js` | header added (it had none) |

**Dead code removed:**

* `hitlik/fisher_cmp.py --upsample` — selected a `pass`; the term's upsample is
  a constructor argument and the Hessian always uses 1.
* `hitlik/recovery.py --whiten` — `store_true` with `default=True`, so it could
  never be unset; the card's own `group_units` decide anyway.
* `hitlik/hitlik_term._resolve_comps` — no caller; `load` / `load_perhit`
  resolve the spec inline.  Its documentation moved into `load_perhit`.

**Files deleted (commit `cb60314`):** the 32 tracked `resolution/runs_*.log`
(31 empty, one a traceback) from the CGF scheduling scans — no reference
anywhere.  `.gitignore` now carries `resolution/runs_*.log`.

**Verification:** every `.py` under `hitlik`, `hitlik/perhit`, `vtxres`,
`matres` imports and passes `--help` in the container (`matres/pick_models.py`
takes a bare path and has no argparse — expected); `bash -n` clean on every
`.sh`; end-to-end `efficiency.py`, `recovery.py`, `perhit/certify.py` and the
three card builders all run.

## `resolution/` top level — 109 scripts deleted (commit `d3b3f78`)

A survey resolved all 228 top-level scripts against every caller, every python
import, and the twelve final-state notes in `Documents/Resolution` (citations
in their `archive/`, the retired dev log, were NOT counted as evidence of
life).  81 are KEEP-CORE, 37 KEEP-REPRO, 109 deleted:

* **62 dated one-offs** with a later sibling or a rewritten replacement:
  `chain_*` / `drive_*` / `finish_*` / `rerun_*` babysitters; the four-step
  `chain_windownorm_260904` chain and the five `censor_*_260904` scripts (that
  test appears in no final note); `clampfix_*_260904`; the six-member
  `kms_solve` ladder; the six pair builders replaced by `masspairs_parallel.sh`;
  `check_exports_260906.py` (`smoke_exports_260906.sh` is what production runs).
* **9 hand-rolled minimisers**: `fit_ms_material.py`, `fit_hit_ms_joint.py`,
  `cf_ms_moliere.py`, `cf0_ecf.py`, `cgf_scoring_test.py`, `cgf_irls_exact.py`,
  `cgf_twocomp.py`, `cgf_mc_closure.py`, `cgf_infit_prep.py`.
* **17 superseded probes**: `attribute_skew_mass.py`, `mixture_test.py`,
  `scale_closure.py`, `tail_symmetry.py`, `jpsi_mass_closure.py`,
  `jpsi_bias_decompose.py`, `simhit_compare.py`, `phi_charge_parity.py`,
  `field_structure_test.py`, `census_probe.py`, `hit_residual_localize.py`,
  `hitres_cf_kmsscan.py`, `cf_pair_ditrack.py`, `cf_kernel_tt.py`,
  `fsr_kernel_study.py`, `plot_radiative_spectrum.py`, `fit_transmission.py`.
* **9 pass/fail gates** whose gated code is shipped: `cgf_cxx_validate.py`,
  `cgfshim.py`, `cgf_delta_validate.py`, `cf_delta_term_validate.py`,
  `cf_delta_term_impact.py`, `radterm_validate.py`, `hadrad_check.py`,
  `window_norm_validate.py`, `ioni_sign_probe.py`.

The one survey DELETE not acted on is `check_slide_overflow.py`: it has no
caller, but the cleanup brief names it as a file to clean, so it stays.

KEPT WITH REASON (marginal): `extract_parallel.sh` (no live caller, but the
generic `--extract` twin of the kept `masspairs_parallel.sh`, and every
track-resolution cache behind `CLOSURE_STATE.md` came through it); the
`*_260906` export-gate trio `check_variance_grads_260906.py` /
`fd_variance_260906.sh` / `fdinmaker_260906.sh` (dated one-offs, but
`PROCESS_NOISE_CGF.md` quotes their numbers); `run_transmission_scan.sh` (not
in `TRANSMISSION.md`'s Reproduce block, but it produces the six coherent dE/dx
shifts the quoted T_sys fit is made of).

## `hitclassbias/` + `oddmoment/` + `qmsmodel/` (commit `8a9acc8`)

17 deleted from `hitclassbias/`: `s8_variants.py`, `s9_who_moved.py`,
`s10_tail.py`, `s11_unconv.py` (pair on `(run,lumi,event)` with two muons per
event -- their own banner said retracted; `c1_conv.py`/`c2_pair.py`/
`conv_common.py` replace them), `s2_quality.py`, `s3_pixel.py`,
`s4_control.py`, `s5_joint.py` (binned splits on the fitted sigma, the trap
`c5_scaling.py` exists to avoid), `explore1.py`, `probe2.py`,
`probe_branches.py`, `list_branches.py`, `t0_consistency.py`,
`t1_locations.py`, `t1b_orient.py`, `t2b_diag.py`, `t7_scaling.py`.

Dead code: `oddmoment/aux_gen.py`'s post-check read locals of another function
(`main()` raised NameError on every run); `hitclassbias/t2_predict.py`'s
`stat()` was never called and was broken; two overwritten assignments.
32 comment blocks cleaned across 25 files.

## `fullscale/` (77 comment blocks across 36 files)

7 deleted: `test_jointhessp.py` (replaced by `gate_joint_hessian.py`),
`report.py`, `native_table.py` (both replaced by `certtable.py`),
`check_normz_sigma.py`, `corr_census.py`, `run_full380.sh`, `run_kernfix.sh`
(build cards that are not in the certified table).  `test_hessp.py` is KEPT:
`STATE.md` §9 publishes its four-random-tangent gate.

Dead code: `make_card.py`'s `if args.shape_prior and not args.shape: pass`;
`make_joint_card.py --clip-vg-other`, which clips the exact algebraic
remainder `vgf - sum_c v_c` at 0 and thereby breaks `vg_other + sum = vgf` for
40 % of J/psi candidates -- never passed by anything.

LEFT IN PLACE with reason: the `fit.py` / `fit_joint.py` / `chunkfit.py` /
`devobj.py` / `shardobj.py` / `minimize_driver.py` family drives
`scipy.optimize.minimize` directly, but `fullscale/STATE.md` retains them as
the reference implementation the rabbit path is checked against and as the
only source of the sandwich covariance; `gate_nanstep.py` calls scipy
deliberately, to see the abort rabbit's Fitter swallows.

## Figure paths and documentation pointers

The canonical figure root is `~/public_html/ZMass/cvh/<YYMMDD>_<tag>/`;
`~/public_html/cvh/` holds only symlinks into it.  `pubhtml.py` now defines
`FIGROOT` and `figdir(tag, date=None)`, and its index-template fallback looks
under the canonical root.  Repointed: `hitlik/plot_hitlik.py`,
`hitlik/tails.py`, `hitlik/perhit/run_stage2.sh` + `run_all.sh` +
`plot_xcum.py`, `vtxres/plot_vtx.py`, `cf_masslik_fit.py`,
`qmsmodel/qms_report.py`, `hitclassbias/{t8,s7,c3}_figs.py`,
`oddmoment/{figs,jensen_figs,mass_figs}.py`, and six `fullscale/` plotters.

`Documents/Resolution/NOTES.md` no longer exists; the bare citations to it are
repointed at the topical note that carries the material (IONISATION_MODEL,
MULTIPLE_SCATTERING, HIT_RESOLUTION, CLOSURE_STATE, PROCESS_NOISE_CGF) or at
`RESOLUTION.md`.  Citations to `NOTES_*.md` were left: those files still exist
under `Documents/Resolution/archive/`.

## Two wrong-signed figures removed

`~/public_html/ZMass/cvh/260908_hitclassbias/{pred_vs_meas,lever_subdet}.{pdf,png}`
were built from `t2_predict.py`, which propagates
`delta z = +sum_b s_b a_b mu_b`.  `c9_phipred.py` derives the minus the
exported influence functional carries (`F` is the RESIDUAL Jacobian) and is
the corrected implementation.  Regenerating from `t2_predict.py` would
reproduce the wrong sign, so the four files were DELETED from the figure
directory and `t2_predict.py` now states its convention explicitly.  Making
the two agree is a physics decision, not a cleanup one, and is left open.

## `production/` + `cleanprop/` + `cfcompress/` + `simprod/` + `globalfit/`

111 comment blocks across those five trees.  Several headers described the
WRONG job (`job_jpsimc_v2.sh` described the DY job, `resume_jpsimc_v2.sh`
named the DY output tag, `status_jpsimc_v2.sh` said "DY re-production",
`simprod/run_simprod_jpsigun.sh` described the muon gun and carried a
copy-pasted BOTH-CHARGES block naming a variable it does not have) -- those
are now right.  Kept, because the trap is still live: the `xrdcp`
returns-0-on-truncation warning, the corrupt split-99 repacked-input door
rules, the bash `GROUPS`-builtin trap, `skipBadFiles` silent empty output, the
`.complete`-sentinel rule, and the Geant4 stepper-precision block.

`make_global_term.param_scales`' docstring said "``theta_j * s_j`` is
physical".  It is the other way round: the code does `theta_card = theta_raw *
s`, so the physical value is `theta_card / s`, the card unit of a parmtype-15
material parameter is `1/gprior`, and one tier prior is `gprior**2` of card
value.  Corrected there, in `diagnose_quadratic.py` (docstring and the printed
unit label, which said "prior sigma") and in `solve_reference.py`'s log line.

15 scripts deleted from `production/`: the completed one-off input repair
(`repack_fix_260907/` minus the two `PRODUCTIONS.md` names as reusable,
`scan_pset_nulls.py` and `scan_fitfail.sh`), `check_table_race.py` (the
shared-Geant4-table race it diagnoses is fixed at `ca6058d96fc`) and
`watch_dy_recover.sh` (a watchdog for a slurm leg that was cancelled; its
three pid/stop/count `.gitignore` lines went with it).

Dead code removed: `globalfit/compare_fit.py --select` (parsed, compiled into
an index list, then never used), `cfcompress/gridtest.py`'s `if ...: pass`,
and a full sort computed and discarded in `production/profiling/parse_profile.py`.

Paths: the dead `/tmp/.../scratchpad/cfcompress` defaults in nine cfcompress
files now split by purpose -- multi-GB shuffled subsamples to
`resolution/runs/cfcompress`, the small committed artifacts to the in-repo
`cfcompress/results/` where they already live.  `PRODUCTION_NEXT.md §2` ->
`PRODUCTIONS.md §3` in five scripts, and three references to the deleted
`STATE.md`/`STATE_dy.md` -> `PRODUCTIONS.md §5`.  Every remaining
`~/public_html/cvh/` and `~/public_html/calibration_studies/` figure root in
`cleanprop/` repointed at `~/public_html/ZMass/cvh/`.

## Still to do

* Comment cleanup of the 119 SURVIVING top-level `resolution/` scripts --
  delegated, running.  `cf_track_resolution.py` alone carries ~45 dated or
  historical remarks.
* A final `--help` sweep over every surviving script once that lands.

## VERIFICATION

* `--help` / import sweep in the rabbit container over every surviving `.py`
  under `hitlik`, `hitlik/perhit`, `vtxres`, `matres`, `fullscale`,
  `production`, `globalfit`, `cfcompress`, `cleanprop`, `simprod`: clean.  The
  only four that do not answer `--help` are cmsRun CONFIGURATION files
  (`production/profiling/{runCvhProfile,repack_n}.py`,
  `simprod/step{1,2}_*.py`) -- they `import FWCore`, which exists only inside
  CMSSW.  `bash -n` clean on every `.sh`.
* END TO END, on the real production inputs, run from `submit50` because
  submit82 has no readable ceph:
  - `vtxres/make_vtx_card.py` on `runs/vtxres/vtx.npz`, 200 candidates:
    3232 group rows, NLL(0) = -632.094190, card written.
  - `hitlik/make_hitlik_card.py` on `runs/perhit/perhit.npz`, 150 tracks,
    `--comps hit`: 1984 rows, NLL(0) = 2698.468821, card written.
  - locally: `matres/make_material_card.py` on
    `runs/matres/gun_groups_probe.npz`; `efficiency.py` on four cached H/J
    files; `recovery.py` and `perhit/certify.py` on the stored rabbit fits.

## KEPT THOUGH THEY LOOK OBSOLETE

* `check_slide_overflow.py` -- no caller anywhere, but the brief names it.
* `extract_parallel.sh`, the `*_260906` export-gate trio,
  `run_transmission_scan.sh` -- see the marginal-call note above.
* `fullscale/{fit,fit_joint,chunkfit,devobj,shardobj,minimize_driver}.py` --
  they drive `scipy.optimize.minimize` directly rather than `Fitter.minimize`,
  but `fullscale/STATE.md` retains them as the reference implementation the
  rabbit path is checked against and as the only source of the sandwich
  covariance.  `gate_nanstep.py` calls scipy deliberately, to see the abort
  rabbit's Fitter swallows.
* `oddmoment/masslik_np.py` -- `fit_alpha` is a hand-rolled grid + parabolic
  minimiser, but the module is the executable spec of `MaterialCFTerm`
  (`MASSCFTERM_SPEC.md` names it, `fullscale/patches/unbinned_jensen.py` cites
  it) and the three `oddmoment/run_*fits.sh` drivers produce published numbers
  through it.
* `cfcompress/massnll_np.py` -- runs its own scipy BFGS, but over a numpy
  re-implementation, not a rabbit likelihood, because the TF stack is
  unreachable from the mfs venv; validated against the published numbers by
  `nll_impact.py --validate`.
* `production/{config,submit}_dymc8p5M.sh`, `resume_dy*.sh`, `status_dy.sh` --
  they drive a cancelled cross-check leg, but `PRODUCTIONS.md §7` is their
  re-run recipe.
* `cleanprop/make_slide_figs_260811.py` -- a dated one-off, but it produces
  `acceptance.png` / `closure.png` / `species.png`, which are checked-in slide
  assets.  It still reads a `/tmp/claude-*` scratchpad that happens to exist;
  that input path needs a durable home.
