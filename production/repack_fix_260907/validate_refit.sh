#!/bin/bash
# Local dev2 CVH refit of N events of one repacked input, with the PRODUCTION
# switch set, as the acceptance gate on a repack.
#   pass: candidates/event ~ 0.997, fit failures ~ 0.3 %
#   fail: ~0.82 cand/event, i.e. the split-99 signature (ROOT #19773)
# usage: validate_refit.sh <input.root> <outdir> [nEvents]
set -uo pipefail
IN=${1:?input}; OUT=${2:?outdir}; NEV=${3:-2000}
AREA=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2
source /work/submit/david_w/ZMass/calibration_studies/production/condor_jpsimc_v2/config_jpsimc_v2.sh
mkdir -p "$OUT"; cd "$OUT"
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd "$AREA/src" && eval "$(scramv1 runtime -sh)"
cd "$OUT"
echo "area=$CMSSW_BASE  in=$IN  nEvents=$NEV  threads=$NTHREADS"
date -Is
# shellcheck disable=SC2086
cmsRun "$AREA/src/Analysis/HitAnalyzer/test/runCvhJpsiGenMC.py" \
    input="$IN" skipEvents=0 nEvents="$NEV" numberOfThreads="$NTHREADS" \
    $EXTRA > local.log 2>&1
echo "cmsRun rc=$? $(date -Is)"
grep -E 'fit summary' local.log | tail -5
ls -la globalcor_*.root 2>/dev/null
echo "VALIDATE DONE"
