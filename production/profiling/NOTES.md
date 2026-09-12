# CVH two-track cost model (Z → μμ MiniAOD and J/ψ)

What a two-track CVH candidate costs, and which switches decide it. The numbers
here sized the 8 M-event Z leg and the 20 M-event J/ψ leg; the productions
themselves are in `production/PRODUCTIONS.md`.

## The one switch that matters: `CgfQoPMode`

`Geant4ePropagator_cfi.py` defaults `CgfQoPMode = 1` (the CGF Fisher-information
q/p weight). Under mode 1 the propagator runs `cvhcgf::inverseFisher` on
**every** propagate call — two 262144-point FFTs plus an O(nt × nsteps)
`blockExponent` sweep — and the two-track maker has no `setCgfOverride` hook, so
`CgfQoPRefresh=0` ("freeze after the first sweep") never takes effect. At ~88
propagate calls per Z candidate that is ~92 ms × 88 = **8.1 s of pure CGF per
candidate**. Confirmed directly: 6/6 gdb stack samples of a running Z job are
inside `cvhcgf::inverseFisher`.

**Mode 1 → 0 is ×140 on the bare fit and ×14 on the full production config.**
Everything else is second order.

Every J/ψ production driver pins it to 0 (`runCvhJpsiGenMC.py` forces
`CgfQoPMode=0` when unset). **Drivers that do not pin it inherit mode 1**: the
dimuon/MiniAOD driver did until this was found, and the custom NanoAOD path
still does — `PhysicsTools/NanoAOD/python/muons_cff.py` clones the same cfi
(`trackrefitdimuon = ResidualGlobalCorrectionMakerDiMuonG4e.clone()`) without
pinning the switch, so a NanoAOD Z production would inherit mode 1 too.

Physics, not cost: mode 0 is the legacy delta-truncated Gaussian weight and mode
1 the Fisher information; the cfi's own note says the CGF has NO measured
accuracy advantage against gen truth. The J/ψ calibration production runs mode
0, so mode 0 is also the *consistent* choice. **Do not silently mix the two
between the J/ψ and Z legs.**

## It is not Z kinematics

Same driver, same maker, same options, only the input tracks change
(`CgfQoPMode=0`, `doRes=True`, `exportStepRecords=False`,
`fillGradsFactored=True`, 50 field modes):

| channel | s/cand | median | niter | prop calls/cand | hits/track |
|---|---|---|---|---|---|
| Z, DY UL16 MiniAOD | 0.63 | 0.57 | 2.71 | 87.6 | 16.3 |
| J/ψ gun, generalTracks | 0.78 | 0.73 | 3.48 | 123.0 | 17.0 |
| J/ψ MC ALCARECO (split-1) | 0.80 | 0.79 | 3.31 | 114.9 | 17.3 |

The Z is *cheaper* per candidate: it converges in fewer Gauss-Newton iterations,
hence fewer propagations. Mean Geant4 steps per leg are Z 26.9 vs J/ψ gun 28.6 —
the G4e step is capped at 10 mm, so step count is geometry-driven, not
momentum-driven. And the J/ψ gun under the *inherited* mode 1 costs 15.7 s
CPU/event, i.e. worse than Z. The channel is irrelevant; the switch is
everything.

Nor is it a tail: at mode 1 the Z per-candidate distribution has mean 8.57 s vs
median 8.27 s, and the worst 5 % of candidates carry 10 % of the time. The fits
are healthy — 90/91 succeed, 0 propagation failures, 0 chi2 backtracks, 1.1 % at
the iteration cap.

## Full measurement

200–600 events each, single-threaded. CPU s/event is `(user − 32.01 s)/nevents`
(32.01 s is the measured one-event job init); s/cand drops the first
candidate-bearing event, which carries a fixed ~10.3 s of *lazy* Geant4 table
build.

| tag | CgfQoPMode | doRes | factored H | modes | CPU s/ev | s/cand | kB/cand |
|---|---|---|---|---|---|---|---|
| z_min | 1 (as found) | - | - | 50 | 3.789 | 8.570 | 0.9 |
| z_res | 1 | yes | - | 50 | 3.790 | 8.562 | 1.0 |
| z_grad | 1 | - | yes | 50 | 3.784 | 8.547 | 25.4 |
| z_full | 1 | yes | yes | 50 | 4.048 | 9.253 | 27.5 |
| z_min360 | 1 | - | - | 360 | 3.904 | 8.859 | 1.0 |
| z_min_m0 | 0 | - | - | 50 | 0.043 | 0.061 | 0.8 |
| z_res_m0 | 0 | yes | - | 50 | 0.073 | 0.062 | 1.0 |
| z_grad_m0 | 0 | - | yes | 50 | 0.077 | 0.064 | 25.4 |
| z_full_m0 | 0 | yes | yes | 50 | 0.307 | 0.631 | 28.1 |
| z_full_m0_tight | 0 | yes | yes | 50 | 0.311 | 0.631 | 27.5 |
| z_full_m0_steprec | 0 | yes | yes | 50 | 0.318 | 0.621 | 316.8 |
| z_min_m0_nofield | 0 (no per-step field modes) | - | - | 50 | 0.071 | 0.048 | 1.0 |
| z_min360_m0 | 0 | - | - | 360 | 0.163 | 0.273 | 1.0 |
| j_gun_full_m0 | 0 | yes | yes | 50 | 0.710 | 0.778 | 29.5 |
| j_alca_full_m0 | 0 | yes | yes | 50 | 0.761 | 0.800 | 30.7 |

Reading it:

* `doRes` and `fillGradsFactored` are each nearly free *alone* (0.062, 0.064)
  and cost ×10 *together* (0.631): the in-maker CF export is gated on
  `doRes_ && (fillGrads_ || fillGradsFactored_)`, so only the pair turns it on.
  That is the second-largest cost after the CGF, and it is a real deliverable.
* The per-step scalar-potential basis costs 0.013 s/cand at 50 modes (~20 % of
  the bare fit) and **0.21 s/cand at 360 modes** — run the 50-mode dump. At mode
  1 the same difference was invisible (8.57 vs 8.86) because the CGF swamped it.
* `tightG4eStepper` is free (0.631 vs 0.631): the G4e step is already capped at
  10 mm and a 45 GeV track's chord error never bites the tolerance.
* `exportStepRecords` costs nothing in CPU and **11× in output** (317 vs 28
  kB/cand). Keep it off.
* MiniAOD unpacking, `TrackProducerFromPatMuons` and
  `DiMuonTrackVertexCandidateProducer` are negligible: maker time is 0.293 of
  the 0.307 s CPU/event at mode 0 full.
* The `kB/cand` column here is ~28 against the production's ~34.7 because these
  runs had `fillJac=False`; `fillJac=True` adds 6.2 kB/candidate
  (`Mu{plus,minus}_jacRef` + `Jpsi_jacMass`).

These runs predate `globalMaterialModel`'s doRes families, so their global
parameter set is **81 824** (81 732 alignment + 50 parmtype-14 field modes + 42
parmtype-15 material groups), of which ~240 are touched per candidate; the
productions declare 126 452. parmtype 7 (per-module dBz / energy loss) is off
under `globalMaterialModel=True`.

## Projection that sized the Z leg

8 M DY MiniAOD events, 0.465 candidates/event → 3.7 M Z candidates:

| configuration | CPU s/ev | core-hours | output |
|---|---|---|---|
| **recommended**: `CgfQoPMode=0`, doRes, factored H, 50 modes, tight stepper | 0.311 | **~700** | 105 GB |
| same but 360 modes | ~0.41 | ~920 | 105 GB |
| as found (`CgfQoPMode=1` inherited) | 4.048 | **~9000** | 105 GB |

(+ ~32 s job init each; at 4000 events/job, 18 core-hours — negligible.) The
production came in at 0.323 s/event and 34.7 kB/candidate, i.e. +4 % on CPU and
the expected `fillJac` offset on volume.

## Reproducing a point

```bash
./run_profile.sh <tag> <outdir> <cmsRun args>        # one point
./launch_batch.sh batch2.txt runs260905 8            # a batch, <=8 concurrent
python parse_profile.py runs260905/z_full_m0 ...     # the table
./pmp_sample.sh <pattern> 25 out.txt && python pmp_fold.py out.txt   # poor-man's profiler
```

`runCvhProfile.py` is one driver for both input shapes — `inputType=miniaod`
(slimmedMuons → `TrackProducerFromPatMuons`, i.e. `runCvhDimuonMiniAOD.py`) and
`inputType=tracks` (an existing `reco::TrackCollection`: ALCARECO,
generalTracks) — both feeding the same `diMuonTrackVertexCandidates` +
`ResidualGlobalCorrectionMakerTwoTrackG4e`, so a per-candidate comparison is not
confounded by a different producer chain. It exposes the CVH switches via
`cvhSwitches`, so the estimator is pinned rather than inherited.

Two caveats when re-running:

* the J/ψ MC ALCARECO must be repacked to ROOT split level 1 first
  (`repack_split1.sh`) — the 10_6-written split-99 files mis-deserialise their
  `SiStripCluster`s in 15_X (ROOT #19773);
* this driver implements `tightG4eStepper` as a patch of `process.geopro`,
  which in this release is neither scheduled nor consumed, so the knob is inert
  here (it is bit-inert in the refit anyway — see `PRODUCTIONS.md` §5.2).
