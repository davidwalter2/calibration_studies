#!/bin/bash
# Cost of the two-track log-det term, measured the only way that survives this
# login node (load swings 4x within minutes): USER CPU, arms INTERLEAVED, the
# same 60 events every time.
#
#   usage: timing_variance_260906.sh <AREA> <OUTROOT> [nev] [nrep]
set -uo pipefail
AREA=${1:?}; OUTROOT=${2:?}; NEV=${3:-60}; NREP=${4:-2}
STAGE=/work/submit/david_w/ZMass/scratch_smoke_260906/inputs
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
TESTDIR=$AREA/src/Analysis/HitAnalyzer/test
GUNFIELD="useIdealGeometry=True useDefaultField=True globalTag=150X_mcRun2_asymptotic_v1"

run_arm() {
  local out=$1; shift
  mkdir -p "$out"; rm -f "$out"/*.root
  ( cd "$out" \
    && source /cvmfs/cms.cern.ch/cmsset_default.sh \
    && cd "$AREA/src" && eval "$(scramv1 runtime -sh)" && cd "$out" \
    && /usr/bin/time -v "$AREA/src/Analysis/HitAnalyzer/test/cmsswlock.sh" run \
       cmsRun "$TESTDIR/runCvhJpsiGenMC.py" input="$STAGE/jpsigun_task0000_step2.root" \
       nEvents=$NEV numberOfThreads=1 doRes=True \
       exportCfExponents=True exportStepRecords=False \
       fillJac=True fillGrads=False fillGradsFactored=True \
       fitFromGenParms=False CgfQoPMode=0 \
       scalarPot3DInitFile=$INIT trackSrc=generalTracks useLegacyPairLoop=True \
       doTrigger=False applyHltFilter=False $GUNFIELD "$@" ) > "$out/cmsrun.log" 2>&1
  local u
  u=$(grep -m1 'User time (seconds)' "$out/cmsrun.log" | awk '{print $NF}')
  local n
  n=$(grep -c 'TrigReport' "$out/cmsrun.log" 2>/dev/null)
  local sz
  sz=$(stat -c %s "$out"/*.root 2>/dev/null | head -1)
  echo "$(basename "$out") user=$u bytes=$sz"
}

for rep in $(seq 1 "$NREP"); do
  run_arm "$OUTROOT/off_$rep"
  run_arm "$OUTROOT/var15_$rep"   exportMaterialNoise=True exportVarianceGrads=True varianceGradFamilies=15
  run_arm "$OUTROOT/varall_$rep"  exportMaterialNoise=True exportVarianceGrads=True varianceGradFamilies=8,9,10,11,15
done
