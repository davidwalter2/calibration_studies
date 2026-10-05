# Correction-set registry (truth-assisted CVH corrections)

Every correction set gets the next free version number, continuing from Josh's v718 (the AN-21-131
corrections, never used here): **v719, v720, ...**. The number carries no meaning. What a set IS (mask,
priors, geometry, inputs, code) is recorded only here and in `corrections_registry.json` (same content,
machine readable). Rules:

* A new solve, or a new file derived from one (e.g. a subset of its parameters), gets max+1. Never reuse a number,
  never renumber, and never overwrite a registered file.
* In the solve directory the file keeps its original name, and `vNNN.npz` / `vNNN.root` symlinks point to it.
  Scripts written from now on refer to `vNNN` (e.g. `--cor v722=<solve>/v722.npz`), not to "corv718" / "cor6dofT".
* A new solve: `solve_local.sh <geom> v<NNN> <options>` (the version is the output name), then add a row to
  the registry table and to `corrections_registry.json`.
* Outputs made before 2026-10-05 use the old labels; see "Old labels" below.

## Kinds (solve options of `solve_truth.py`)

Common to all sets so far:
* Derivation: `runCvhTruthAssisted.py mode=derive`, gen-anchored (the five reference parameters frozen to gen).
* Inputs: UL16 MC, 30 J/psi tasks (`JPsiToMuMu_Pt8toInf-pythia8`, TkAlJpsiMuMu ALCARECO, 106X_mcRun2_asymptotic_v13)
  plus 15 DY tasks (`DYJetsToMuMu_H2ErratumFix_TuneCP5_13TeV-powhegMiNNLO-pythia8-photos`, MiniAODv2, ..._v17);
  180 `globalcor_truth_*.root`; SIM field 160812 + OAE.
* Free parameter types: alignment 0-5 and pixel classes 16-21 (`--free` default); Bz (6) and material frozen.
* Track cuts: `--min-valid-hits 9 --max-grad 1e5`.

| kind | solve options | meaning |
|---|---|---|
| v718-like | `--mask v718` | Josh's truth-assisted mask: only in-plane alignment (local x/y = types 0, 1 everywhere; in-plane rotation type 5 only in TID/TEC); types 2-4 frozen |
| 6dof | (none) | all six alignment dofs, default priors |
| 6dof-tight | `--prior 2=1e-2 --prior 3=5e-4 --prior 4=5e-4 --prior 5=5e-4` | all six alignment dofs, tight priors on local z [cm] and the rotations [rad] |
| class-only | (derived file) | the pixel-class parameters (types 16-21) of the parent set; everything else zero |

Known caveat for all of them: the J/psi inputs carry the official TkAlJpsiMuMu selection (mass window + good-ID).
On the J/psi gun this selection shifts the response by +0.067 / +0.126 / -0.006e-4 for |eta| < 0.8 / 0.8-1.6 /
1.6-2.4, and the corrections absorb that shift (anfig56 README, 10/04).

## Registry

| version | made | geometry | kind | parent / derivation | file (under /ceph/submit/data/user/d/david_w/ZMass/cvh/) | status |
|---|---|---|---|---|---|---|
| v719 | 2026-09-24 09:08 | realistic | 6dof | `truthassist_260924` (nano15 work area at 1d786ac50fe, not pinned; slurm 6536010) | `truthassist_260924/solve/truthcor.{npz,root}` | superseded by v723 |
| v720 | 2026-09-24 09:14 | realistic | v718-like | `truthassist_260924` | `truthassist_260924/solve/truthcor_v718mask.{npz,root}` | superseded by v722 |
| v721 | 2026-09-24 09:23 | realistic | 6dof-tight | `truthassist_260924` | `truthassist_260924/solve/truthcor_6dof_tightprior.{npz,root}` | superseded by v724 |
| v722 | 2026-10-03 03:58 | realistic | v718-like | `truthassist_261003` (pinned payload overlay_nano15_8fbd0f6af23, incl. the corFile alignment fix) | `truthassist_261003/solve/real_v718.{npz,root}` | in use |
| v723 | 2026-10-03 04:05 | realistic | 6dof | `truthassist_261003` | `truthassist_261003/solve/real_6dof.{npz,root}` | DO NOT USE (local-z rms 606 um, induces A ~ -1e-3 in the endcaps) |
| v724 | 2026-10-03 04:09 | realistic | 6dof-tight | `truthassist_261003` | `truthassist_261003/solve/real_6dof_tightprior.{npz,root}` | in use |
| v725 | 2026-10-03 04:57 | ideal | v718-like | `truthassist_261003` | `truthassist_261003/solve/ideal_v718.{npz,root}` | in use (ideal-geometry tests) |
| v726 | 2026-10-03 05:04 | ideal | 6dof-tight | `truthassist_261003` | `truthassist_261003/solve/ideal_6dof_tightprior.{npz,root}` | in use (ideal-geometry tests) |
| v727 | 2026-10-03 05:15 | realistic | class-only | of v722 | `truthassist_261003/solve/aux/real_v718_classonly.npz` | test product |
| v728 | 2026-10-03 05:15 | realistic | class-only | of v723 | `truthassist_261003/solve/aux/real_6dof_classonly.npz` | test product |

Solve logs: `truthassist_{260924,261003}/logs/solve_*.out` (the solve options are in the first line).

## Old labels (outputs made before 2026-10-05)

* `corv718` / `r_corv718`: v722 in realistic-geometry outputs, v725 in ideal-geometry outputs
  (`extract_tracks.py --cor corv718=<solve>/<geom>_v718.npz`).
* `cor6dofT` / `r_cor6dofT`: v724 (realistic), v726 (ideal).
* anfig56 in-fit corFile runs: `infit_v718_*` = v722, `infit_6dof_tightprior_*` = v724, `infit_6dof_*` = v723.
* 2026-09-24 closure plots (`truthcor*`): v719, v720, v721.
