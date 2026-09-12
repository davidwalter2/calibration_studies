# Clean propagation test

Predict the PDF of the propagated 5D helix state, and test that prediction
against Geant4 with nothing else in the way.

The idea (Josh Bendavid): take **one** particle with a **fixed** initial state,
run the full simulation many times, and try to predict the distribution of the
final state on a sensor surface. No hits, no track fit, no QED FSR, no
selection, no background. Geant4 is the right answer by definition, so a
disagreement is unambiguously in the transport-fluctuation model — which is
exactly what cannot be said of a closure on the mass fits.

The deliverable it points at is bigger than the validation: a propagator that
returns the full PDF of the propagated state rather than just a covariance
matrix.

## Pieces

| what | where |
|---|---|
| ground truth (N × the same particle, true state at each sensor) | `Analysis/HitAnalyzer/test/runCleanPropSim.py` + `plugins/SimHitStateNtuplizer.cc` |
| model (1 × deterministic, per-step physics + exact transport) | `Analysis/HitAnalyzer/test/runCleanPropModel.py` + `plugins/G4ePropagationExport.cc` |
| comparison in transform space | `../cf_propagation_test.py` |
| ray choice | `scan_ray.sh`, `rank_rays.py` |
| production | `run_cleanprop_sim.sh`, `run_campaign.sh`, `regen_models.sh`, `points_*.txt` |
| figures | `make_*_figs.py`, `path_averaged_closure.py`, `acceptance_bias.py` |

A `PSimHit` already carries the CMSSW 5D local parameterisation at the sensor
entry face (entry point, θ/φ at entry, |p|), so no custom Geant4 stepping action
is needed.

## Recipe

```bash
# 0. pick a ray that crosses every module through its FACE (see caveats)
./scan_ray.sh 500 && python rank_rays.py …

# 1. ground truth: 128 tasks x 8000 events, seeds are the ONLY difference
#    ./run_cleanprop_sim.sh [ntasks] [nev/task] [eta] [phi] [pt] [tag] [partId]
NPAR=128 ./run_cleanprop_sim.sh 128 8000 0.30 0.20 10 260804

# 2. target surfaces (modal crossed-module sequence + entry local z)
cd .. && python cf_propagation_test.py --targets \
    --sim <one sim file> --out cleanprop/targets_pt10_eta0.30_phi0.20.txt

# 3. model: one deterministic propagation through those same surfaces
cmsRun runCleanPropModel.py pt=10 eta=0.30 phi=0.20 \
    targets=cleanprop/targets_pt10_eta0.30_phi0.20.txt output=model.root

# 4. compare
python cf_propagation_test.py --compare --sim '<simdir>/simstates_*.root' \
    --model model.root --functionals qop locx --acceptance perplane
```

A whole (pt, eta, phi, species) scan is one command — `./run_campaign.sh
<points-file> [nev_total] [tag]`, with points-file lines
`<pt> <eta> <phi> <partId> <label>`. Every stage skips if its output exists, so
it is resumable. `regen_models.sh` rebuilds only the **model** side after a
propagator change: the SIM is Geant4 ground truth and does not age, and
regenerating 100–128 files per point would be hours of compute for no change.

Cost: the simulation runs at ~17 events/s/core, so 10⁶ events is ~16 min on 64
cores. The model side is one event.

Figures go to `~/public_html/ZMass/cvh/<YYMMDD>_cleanprop/` — pass `--outpath`,
since the script's built-in default is the older `~/public_html/cvh/…` path.

## Options that change the answer

| option | what it does |
|---|---|
| `--acceptance modal\|perplane` | **use `perplane`.** `modal` (the default, and what the earliest numbers used) keeps only rays whose whole (module, entry-face) sequence is the modal one, which drops 5.6 % of pT = 3 muons and 21–26 % of pT = 3 hadrons — **tail-first**, biasing the closure by +0.0022 (μ) to +0.0130 (p) at u = 1. `perplane` uses every ray that crossed a given plane's (detid, entry face) on that plane, valid because the model's prediction there depends only on the deterministic reference path. **Muons only**: for hadrons neither mode is a valid test |
| `--veto-eloss F` | drop rays whose worst single-plane momentum loss exceeds the per-plane median by more than F of p₀ — i.e. remove **hadronic inelastic** interactions, which the model side (ionisation + bremsstrahlung + transportation only) does not contain. At pT = 3 the muon loss tail is smooth while π and p carry a discrete population losing ~95 % of p₀ (0.85 % of protons), and those 0.85 % take std(z) from ~1 to 5.05; a cut of 0.2 separates them cleanly. It does **not** remove nuclear *elastic* scattering, which costs no momentum and which the model also lacks |
| `--kms` | log-scale on the MS log-CF exponent, exactly as `cf_track_resolution --kms`. Scanning it here measures the same quantity as the track-level `k_ms` with NO fit, NO hits and NO block pooling (exact per-step transport Jacobians): if the two agree the discrepancy is in the Molière FORMULA, if only the track-level one is non-zero it is in the fit machinery |
| `--functionals`, `--probes`, `--layers` | which linear functionals, which Weierstrass probes `u` in `⟨exp(−u z²)⟩` (small u weights the delta-ray tail, u ~ 1 the core), which layers |

## Caveats worth knowing before re-running

**The ray matters, a lot.** A ray that grazes a module edge produces events that
either miss that module or enter through its side; those are not on the plane
the reference was propagated to, and dropping them is exactly the selection
effect this test exists to remove. `scan_ray.sh` picks the ray empirically —
over an 18-point (η, φ) grid the clean fraction ranged from **14.6 % to 100 %**.
η = 0.30, φ = 0.20 gives 20 crossed modules and 99.63 % clean at 10⁶ events.

**Do not use η = 0.** The pixel barrel ladders have a module boundary at z = 0,
so a track at exactly η = 0 threads the gap in all three layers and produces
**no pixel hits at all**.

**Conditions must be pinned.** `auto:run2_design` with the `Ideal` XML label
yields a geometry with no pixel sensitive volumes in 15_X; both jobs pin
`106X_mcRun2_asymptotic_v17` + `XMLFILE_Geometry_2016_81YV1_Extended2016_mc`
(label `Extended`), matching the CVH refit drivers. The model side must use the
**ideal** tracker geometry (`useIdealGeometry=True`), because PSimHit local
coordinates are always in the ideal frame.

**Exact per-step weights.** The propagator has an opt-in
`setStepTransportLogging` that records the cumulative transport Jacobian at
every Geant4 step, so the noise of each individual step can be transported to
any target surface exactly (`A_s = Jacc_N Jacc_s⁻¹`) instead of through one
RMS-matched scalar per pooled block as the fit exports must use. The bookkeeping
is closed to machine precision against the propagator's own transported dQI
(ratio 1.0000 at every layer) — run that check before trusting any
disagreement.

**What each functional isolates.** Multiple scattering barely changes |p| (it
contributes ~1e-7 of the variance), so the q/p residual is essentially pure
ionisation straggling — the first real test of the muon Urban tail, which is
invisible at block level in the fit (hat values ~1e-6). Position and angle
residuals are Molière-dominated.
