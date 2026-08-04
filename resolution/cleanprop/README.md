# Clean propagation test

Predict the PDF of the propagated 5D helix state, and test that prediction
against Geant4 with nothing else in the way.

Josh Bendavid's suggestion (2026-08-04): take **one** particle with a **fixed**
initial state, run the full simulation many times, and try to predict the
distribution of the final state on a sensor surface. No hits, no track fit, no
QED FSR, no selection, no background. Geant4 is the right answer by
definition, so a disagreement is unambiguously in the transport-fluctuation
model — which is exactly what cannot be said of a closure on the mass fits.

The deliverable it points at is bigger than the validation: a propagator that
returns the full PDF of the propagated state rather than just a covariance
matrix.

## Pieces

| what | where |
|---|---|
| ground truth (N x the same muon, true state at each sensor) | `Analysis/HitAnalyzer/test/runCleanPropSim.py` + `plugins/SimHitStateNtuplizer.cc` |
| model (1 x deterministic, per-step physics + exact transport) | `Analysis/HitAnalyzer/test/runCleanPropModel.py` + `plugins/G4ePropagationExport.cc` |
| comparison in transform space | `../cf_propagation_test.py` |
| ray choice / production | `scan_ray.sh`, `run_cleanprop_sim.sh` |

A PSimHit already carries the CMSSW 5D local parameterization at the sensor
entry face (entry point, theta/phi at entry, |p|), so no custom Geant4
stepping action is needed.

## Recipe

```bash
# 0. pick a ray that crosses every module through its FACE (see caveats)
./scan_ray.sh 500

# 1. ground truth: 128 tasks x 8000 events, seeds are the ONLY difference
NPAR=128 ./run_cleanprop_sim.sh 128 8000 0.30 0.20 10 260804

# 2. target surfaces (modal crossed-module sequence + entry local z)
cd .. && python cf_propagation_test.py --targets \
    --sim <one sim file> --out cleanprop/targets_pt10_eta0.30_phi0.20.txt

# 3. model: one deterministic propagation through those same surfaces
cmsRun runCleanPropModel.py pt=10 eta=0.30 phi=0.20 \
    targets=cleanprop/targets_pt10_eta0.30_phi0.20.txt output=model.root

# 4. compare
python cf_propagation_test.py --compare --sim '<simdir>/simstates_*.root' \
    --model model.root --functionals qop locx
```

Cost: the simulation runs at ~17 events/s/core, so 10^6 events is ~16 min on
64 cores. The model side is one event.

## Caveats worth knowing before re-running

**The ray matters, a lot.** A ray that grazes a module edge produces events
that either miss that module or enter through its side; those are not on the
plane the reference was propagated to, and dropping them would be exactly the
selection effect this test exists to remove. `scan_ray.sh` picks the ray
empirically — over an 18-point (eta, phi) grid the clean fraction ranged from
**14.6% to 100%**. eta=0.30, phi=0.20 gives 20 crossed modules and 99.63% clean
at 10^6 events.

**Do not use eta = 0.** The pixel barrel ladders have a module boundary at
z = 0, so a track at exactly eta = 0 threads the gap in all three layers and
produces **no pixel hits at all**.

**Conditions must be pinned.** `auto:run2_design` with the `Ideal` XML label
yields a geometry with no pixel sensitive volumes in 15_X; both jobs pin
`106X_mcRun2_asymptotic_v17` + `XMLFILE_Geometry_2016_81YV1_Extended2016_mc`
(label `Extended`), matching the CVH refit drivers. The model side must use
the **ideal** tracker geometry (`useIdealGeometry=True`), because PSimHit local
coordinates are always in the ideal frame.

**Exact per-step weights.** The propagator gained an opt-in
`setStepTransportLogging` that records the cumulative transport Jacobian at
every Geant4 step, so the noise of each individual step can be transported to
any target surface exactly (A_s = Jacc_N Jacc_s^-1) instead of through one
RMS-matched scalar per pooled block as the fit exports must use. The
bookkeeping is closed to machine precision against the propagator's own
transported dQI (ratio 1.0000 at every layer) — run that check before trusting
any disagreement.

**What each functional isolates.** Multiple scattering barely changes |p|, so
the q/p residual is essentially pure ionization straggling (MS contributes
~1e-7 of its variance). It is therefore the first real test of the muon Urban
tail, which is invisible at block level in the fit (hat values ~1e-6).
Position and angle residuals are Moliere-dominated.
