#!/bin/bash
# STEP-SUBDIVISION INVARIANCE OF THE ACCUMULATED MS BLOCK, AT THE PROPAGATOR.
#
# `G4ePropagationExport` writes the propagator's own per-leg `dQMS` (5x5,
# curvilinear (qop, lambda, phi, xt, yt)). The scattering power DD of
# `PropagateErrorMSC` is LINEAR in the step length with no logarithmic term,
# so the material between two fixed target planes contributes the same Q no
# matter how finely Geant4e dices it: `dQMS` must be invariant under
# `StepLengthLimit`. It is, in the (phi, xt) projection; the (lambda, yt) one
# is the test of the res(1,4) sign.
set -uo pipefail
AREA=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2
CP=/work/submit/david_w/ZMass/calibration_studies/resolution/cleanprop
OUT=${1:-/work/submit/david_w/ZMass/calibration_studies/resolution/runs/msksign_260918/subdiv}
TARGETS=${TARGETS:-$CP/targets_pt10_eta0.30_phi0.20.txt}
PT=${PT:-10.0}; ETA=${ETA:-0.30}; PHI=${PHI:-0.20}; PARTID=${PARTID:-13}
mkdir -p "$OUT"
cd "$AREA/src"; source /cvmfs/cms.cern.ch/cmsset_default.sh; eval "$(scramv1 runtime -sh)"
cd "$OUT"
for sign in legacy fixed; do
  for L in 10.0 5.0 2.5 1.25; do
    tag="${sign}_L${L}"
    [[ -s "$OUT/model_$tag.root" ]] && { echo "[skip] $tag"; continue; }
    if [[ $sign == legacy ]]; then export CVH_MS_CORR_LEGACY=1; else unset CVH_MS_CORR_LEGACY; fi
    echo ">>> $tag"
    cmsRun "$AREA/src/Analysis/HitAnalyzer/test/runCleanPropModel.py" \
      pt=$PT eta=$ETA phi=$PHI partId=$PARTID targets="$TARGETS" \
      stepLength=$L output="$OUT/model_$tag.root" > "$OUT/log_$tag.txt" 2>&1
    echo "    rc=$? $(ls -la "$OUT/model_$tag.root" 2>/dev/null | awk '{print $5}')"
  done
done
