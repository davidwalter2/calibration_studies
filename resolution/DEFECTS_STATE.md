# Four CVH defects reported from the track-likelihood session — state

Area under change: `/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2/src`,
branch `WmassNanoProd_15_0_19_patch2_dev` at **`7b54ce096b27`**, which is where
the area already stood on arrival (last reflog entry 2026-09-17 14:27, three
commits after the `f7fbef244c67` named in the brief; `f7fbef244c67` is an
ancestor of it and `my-cmssw/WmassNanoProd_15_0_19_patch2_dev` points at the
same `7b54ce096b27`). Nothing in this work moved HEAD: no fetch, pull,
checkout or reset was run; the one commit this work added, `3d4c926ff461`,
sits on top of it and touches only the files below.

| file | what changed |
|---|---|
| `TrackPropagation/Geant4e/src/Geant4ePropagator.cc` | (d) `res(1,4) = +S3` |
| `Analysis/HitAnalyzer/plugins/ResidualGlobalCorrectionMakerTwoTrackG4e.cc` | (a) reference EDM on the free indices; (b) step records cleared on the drain pass only; msmoli drain via the helper |
| `.../ResidualGlobalCorrectionMakerNTrackG4e.cc` | (a) same latent fix |
| `.../ResidualGlobalCorrectionMakerG4e.cc` | msmoli drain via the helper |
| `.../ResidualGlobalCorrectionMakerBase.h` | `MSMOLI_STRIDE`, `msmolistride`, `pushMsMoliStep` with its column check; stale "8 floats" comment corrected |
| `.../ResidualGlobalCorrectionMakerBase.cc` | `msmolistride` branch |
| `.../G4ePropagationExport.cc` | `msmolistride` / `ioniurbanstride` branches; stale `/8` and `/11` log divisors |
| `Analysis/HitAnalyzer/doc/resolution-cf-export.md` | stride table extended |

## Status per item — all four closed

| item | verified in dev2 | fixed | gated | affects a production? |
|---|---|---|---|---|
| (a) NaN reference EDM under `doVtxConstraint` | yes | yes, commit `3d4c926ff461` | all three gates | no repeat; a 2.9x CPU saving |
| (b) step records emptied under `doMassConstraint` | yes | yes, same commit | before/after both shown | only runs with BOTH switches on — none of ours |
| (c) `msmoliv` stride 13 | n/a (a format extension, not a defect here) | not ported, deliberately | every reader is stride-aware; `msmolistride` is written and cannot lie | no |
| (d) MS within-step correlation sign | yes | yes, same commit | derivation + subdivision invariance + impact | J/psi and Z: no repeat. K_S: redo |

Commit `3d4c926ff461`, pushed to `my-cmssw/WmassNanoProd_15_0_19_patch2_dev`.
Working tree in dev2 is clean. All A/B knobs were removed before the commit and
the knob-free build reproduces the knobbed build's fixed arm **bit-identically
on all 302 branches**, so the removal and the `pushMsMoliStep` refactor are
provably inert.

## The impact of (d), which is the one that decides a repeat

Momentum-scale translation uses the measured `d ln m / d ln p` of each decay:
0.994 (J/psi -> mu mu), 0.652 (K_S -> pi pi), 1.000 (Z -> mu mu).

| | J/psi gun, 1183 cand | K_S, 1794 cand | Z -> mu mu, 313 cand |
|---|---|---|---|
| mean `Delta m/m` | +3.17e-6 +- 3.23e-6 | -1.26e-5 +- 1.65e-5 | +3.96e-5 +- 3.76e-5 |
| 1-99 % trimmed | -5.9e-7 +- 1.0e-6 | +2.2e-6 +- 6.2e-6 | +1.3e-7 +- 4.9e-7 |
| median | 0 +- 1.3e-7 | +4.2e-7 +- 1.0e-6 | 0 +- 2.1e-8 |
| per-cand median \|`Delta m/m`\| | 4.9e-6 | 5.7e-5 | 4.2e-7 |
| per-cand p95 | 7.2e-5 | 6.0e-4 | 5.6e-6 |
| candidates moving > 1e-4 | 2.96 % | **35.8 %** | 0.96 % |
| momentum scale from the mean | +3.19e-6 +- 3.25e-6 | -1.93e-5 +- 2.53e-5 | +3.96e-5 +- 3.76e-5 |
| `sigma_m` relative, mean | +3.03e-4 +- 0.29e-4 | **+3.68e-3 +- 0.17e-3** | -3.9e-4 +- 2.9e-4 |
| `sigma_m` relative, median | +8.3e-5 | +1.48e-3 | +6.6e-7 |
| fitted q/p, mean relative | +1.7e-6 +- 1.3e-6 | -1.2e-4 +- 0.8e-4 | +1.0e-4 +- 1.0e-4 |
| `chisqval`, mean relative | -2.1e-4 +- 0.6e-4 | +2.0e-3 +- 0.7e-3 | -8.5e-4 +- 6.0e-4 |
| vertex z, mean | +0.28 +- 0.30 um | -0.25 +- 4.9 um | — |
| `Jpsi_covvtx`, mean relative | +4.3e-3 +- 2.4e-3 | +2.0e-3 +- 0.4e-3 | — |
| beam pull `Jpsi_bsz`, mean relative | — | — | -6.5e-4 +- 8.8e-4 |
| `Jpsi_bschi2`, mean relative | — | — | -1.4e-3 +- 0.7e-3 |
| `cfmass_ms`, mean relative | +6.8e-5 +- 0.1e-5 | +1.5e-4 +- 0.05e-4 | — |

Per-hit residuals normalised by the hit error, J/psi gun: the NON-BENDING
(local y) RMS moves +1.20 % (3.2875 -> 3.3270) while the BENDING (local x) RMS
moves -0.0017 % (6.38282 -> 6.38174) — a 700x asymmetry, the projection the
defect lives in and its control.

The (d) numbers are identical whether measured with the production 10-iteration
EDM or the fixed one (`+3.173e-6 +- 3.231e-6` against `+3.169e-6 +- 3.229e-6`),
so the two fixes are independent.

## The gates

**(d) step-subdivision invariance at the propagator**, `StepLengthLimit`
10/5/2.5/1.25 mm, 20 legs of a pT = 10 GeV muon, max `|Q(cap)/Q(10mm) - 1|`:

| | Q(4,4) lam-yt | Q(1,4) lam-yt | Q(3,3) phi-xt | Q(2,3) phi-xt |
|---|---|---|---|---|
| `res(1,4) = -S3` | **0.340** | **0.512** | 3e-5 | 7e-5 |
| `res(1,4) = +S3` | 3e-5 | 7e-5 | 3e-5 | 7e-5 |

Rebuilding the block from the propagator's own per-step log reproduces its
`dQMS(4,4)` to 2.3e-16 with the old sign and is 1.9e-2 off with the new one;
the old sign is low by a median 6.0 % (p05 26.5 %, worst 52 %) in a leg's
offset variance.

**(a)(i)** converged vs 10 iterations, (d) fixed in both, 1183 candidates:
`Jpsi_mass` median relative 8.3e-8, p95 3.3e-6, mean -9.8e-8 +- 7.4e-8
(-6.5e-6 +- 7.3e-6 in units of `sigma_m`); q/p median relative ~1e-6;
`chisqval` median relative 6.4e-6. NOT bit-identical, but two orders below the
1e-5 target — the 10-iteration results were converged.
**(a)(ii)** `<niter>` 10.000 -> 3.245, cap hit 100 % -> 2.4 %, 5.2 s/event ->
1.9 s/event on the same host.
**(a)(iii)** vertex constraint OFF: **292/292 branches BIT-IDENTICAL**.

**(b)** legacy clear, mass constraint on: **110/110 candidates EMPTY** in all
nine record branches. With the fix: 0/188 empty, and every record
bit-identical to the constraint-off run.

## Scripts

`resolution/msksign/`: `transport_sign.py` (the lever signs),
`subdivision.py` (the closed form), `run_subdiv.sh` + `reconstruct_qms.py`
(the propagator-level gate), `run_ab.sh` / `run_dy.sh` (the arms),
`compare_arms.py`, `massshift.py`, `hitpull.py`, `steprec_check.py`,
`iter_cpu.py`. Outputs in `resolution/runs/msksign_260918/`.

## Process lesson: never edit a script a job is reading

Editing `resolution/msksign/run_ab.sh` to add one arm, while four K_S `cmsRun`
jobs were running FROM that file, broke their epilogue: bash reads a script
incrementally by byte offset, so the insertion shifted the ground under the
running shells and they died on a syntax error inside the `case`. The cmsRun
processes themselves had already completed and the six output files were
verified valid, complete and correctly configured -- but that was luck. Had the
edit landed a few lines earlier the jobs would have run with the wrong
arguments, or not at all, with no obvious signature in the output.

**The rule: copy, edit, relaunch.** To add or change an arm while jobs are in
flight, copy the runner to a new name, edit the copy, and launch from the copy.
Never edit in place. The same rule as `scram b` against a running `cmsRun`, and
for the same reason: the artefact a running process reads must be immutable for
as long as it is running.

## Found in passing, pre-existing, NOT one of the four

`radstepv` is written at stride 12 by the makers and was being read at a
hard-coded 11 by eight offline sites: 91.5 % of candidates raise, 8.5 %
mis-parse silently (`Sum dE_rad` 4e7 MeV against a true 0.8 MeV). The two C++
shims refused strides 12 and 14 outright and their wrappers discarded the
return code, so the ionization/MS/radiative exponents came back as zeros.
Both are fixed in `calibration_studies` (uncommitted): a shared stride helper
in `resolution/prodfiles.py` and thirteen readers plus the two shims routed
through it.
