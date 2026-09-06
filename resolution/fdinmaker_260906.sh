#!/bin/bash
# The IN-MAKER finite difference of the variance gradient, at FIXED
# linearization (`varianceFDGlobalIdx=-2`).  Two eps, so the residual can be
# shown to be O(s^2) truncation and nothing else.
set -uo pipefail
AREA=${1:?}; OUT=${2:?}; NEV=${3:-20}
STAGE=/work/submit/david_w/ZMass/scratch_smoke_260906/inputs
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
GUNFIELD="useIdealGeometry=True useDefaultField=True globalTag=150X_mcRun2_asymptotic_v1"
for eps in 1e-3 1e-4 1e-5; do
  d="$OUT/eps$eps"; mkdir -p "$d"; rm -f "$d"/*.root
  ( cd "$d" && source /cvmfs/cms.cern.ch/cmsset_default.sh \
    && cd "$AREA/src" && eval "$(scramv1 runtime -sh)" && cd "$d" \
    && "$AREA/src/Analysis/HitAnalyzer/test/cmsswlock.sh" run \
       cmsRun "$AREA/src/Analysis/HitAnalyzer/test/runCvhJpsiGenMC.py" \
       input="$STAGE/jpsigun_task0000_step2.root" \
       nEvents=$NEV numberOfThreads=1 doRes=True \
       exportCfExponents=True exportStepRecords=False \
       fillJac=True fillGrads=True fitFromGenParms=False CgfQoPMode=0 \
       scalarPot3DInitFile=$INIT trackSrc=generalTracks useLegacyPairLoop=True \
       doTrigger=False applyHltFilter=False $GUNFIELD \
       exportMaterialNoise=True exportVarianceGrads=True exportObjective=True \
       varianceGradFamilies=10,11,15 \
       varianceFDGlobalIdx=-2 varianceFDEps=$eps ) > "$d/cmsrun.log" 2>&1
  echo "eps=$eps rc=$? VARFD lines: $(grep -c '^VARFD ' "$d/cmsrun.log")"
done
