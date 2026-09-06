#!/bin/bash
# The SINGLE-TRACK maker on the SAME J/psi gun file the two-track smoke uses,
# so that "the same muon fitted alone and fitted as half of a pair" is a
# statement about the same tracks and the same parmtype-10 global indices
# (gate D of check_variance_grads_260906.py).
set -uo pipefail
AREA=${1:?usage: st_jpsigun_260906.sh <AREA> <OUT> [nev]}
OUT=${2:?}
NEV=${3:-60}
STAGE=/work/submit/david_w/ZMass/scratch_smoke_260906/inputs
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
mkdir -p "$OUT"; rm -f "$OUT"/*.root
( cd "$OUT" \
  && source /cvmfs/cms.cern.ch/cmsset_default.sh \
  && cd "$AREA/src" && eval "$(scramv1 runtime -sh)" && cd "$OUT" \
  && "$AREA/src/Analysis/HitAnalyzer/test/cmsswlock.sh" run \
     cmsRun "$AREA/src/Analysis/HitAnalyzer/test/runCvhResClosure.py" \
     input="$STAGE/jpsigun_task0000_step2.root" \
     nEvents=$NEV numberOfThreads=1 doRes=True \
     exportCfExponents=True exportStepRecords=False \
     fillGrads=True fitFromGenParms=False \
     exportMaterialNoise=True exportObjective=True \
     scalarPot3DInitFile=$INIT trackSrc=generalTracks \
     useIdealGeometry=True useDefaultField=True \
     globalTag=150X_mcRun2_asymptotic_v1 ) > "$OUT/cmsrun.log" 2>&1
echo "rc=$? $(ls -la "$OUT"/*.root 2>/dev/null | awk '{print $5,$9}')"
