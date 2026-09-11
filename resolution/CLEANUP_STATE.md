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
