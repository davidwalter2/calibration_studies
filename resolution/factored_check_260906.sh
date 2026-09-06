#!/bin/bash
# Both Hessian exports in the same file, so B^T B can be compared against the
# packed dense triangle candidate by candidate -- the claim being that
# nRank = ndof + nvarcols keeps the FACTORED branch a complete description of
# `hess` once the log-det block is in it.
set -uo pipefail
AREA=${1:?}; OUT=${2:?}; NEV=${3:-25}
STAGE=/work/submit/david_w/ZMass/scratch_smoke_260906/inputs
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
GUNFIELD="useIdealGeometry=True useDefaultField=True globalTag=150X_mcRun2_asymptotic_v1"
run() {
  local tag="$1"; local d="$OUT/$1"; shift; mkdir -p "$d"; rm -f "$d"/*.root
  ( cd "$d" && source /cvmfs/cms.cern.ch/cmsset_default.sh \
    && cd "$AREA/src" && eval "$(scramv1 runtime -sh)" && cd "$d" \
    && "$AREA/src/Analysis/HitAnalyzer/test/cmsswlock.sh" run \
       cmsRun "$AREA/src/Analysis/HitAnalyzer/test/runCvhJpsiGenMC.py" \
       input="$STAGE/jpsigun_task0000_step2.root" \
       nEvents=$NEV numberOfThreads=1 doRes=True \
       exportCfExponents=True exportStepRecords=False \
       fillJac=True fillGrads=True fillGradsFactored=True \
       fitFromGenParms=False CgfQoPMode=0 \
       scalarPot3DInitFile=$INIT trackSrc=generalTracks useLegacyPairLoop=True \
       doTrigger=False applyHltFilter=False $GUNFIELD "$@" ) > "$d/cmsrun.log" 2>&1
  echo "$tag rc=$?"
}
run off
run var15  exportMaterialNoise=True exportVarianceGrads=True varianceGradFamilies=15
run varall exportMaterialNoise=True exportVarianceGrads=True varianceGradFamilies=8,9,10,11,15
