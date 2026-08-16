#!/bin/bash
# Per-material-group influence-weight probe (closing measurement for
# NOTES_TRANSMISSION.md).
#
# For each of the 42 material groups, applies a COHERENT MEAN-ONLY log-scale
# shift eps to that group's energy loss (CVH_MATGROUP_PROBE / _EPS /
# _MEANONLY -- see MaterialGroupModel.cc). Paired per track against the
# nominal run this gives
#     w_g = d(p_fit at PCA) / d(applied loss in group g)
# i.e. the fit's influence weight for that group. MEAN-ONLY matters: the
# group k also scales the step's MS and ionisation VARIANCE in the production
# path (Geant4ePropagator.cc:1011), and moving Q would move the fit weights
# themselves, so the measured response would no longer be at fixed weights.
#
# usage: ./run_matgroup_scan.sh [nparallel] [ntasks] [nevents]
#   MATGROUPS="0 1 2 ..."  which groups (default 0..41)
#   EPS=0.693147        log-scale shift (default: x2)
set -euo pipefail
NPAR=${1:-100}
NTASKS=${2:-8}
NEVENTS=${3:--1}
EPS=${EPS:-0.693147}
MATGROUPS=${MATGROUPS:-$(seq 0 41)}

CFG=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/test/runCvhJpsiGenMC.py
export CMSSW_AREA=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev
RUN_ONE=/work/submit/david_w/ZMass/calibration_studies/slurm/run_one.sh
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
SIMROOT=/ceph/submit/data/user/d/david_w/ZMass/cvh/resolution_simprod_jpsigun_ul16
OUTROOT=/ceph/submit/data/user/d/david_w/ZMass/cvh
OUTTAG=${OUTTAG:-260814}

# identical to the transmission scan / the dedx-scan productions
COMMON="nEvents=${NEVENTS} numberOfThreads=1 doRes=True fillGrads=True \
fitFromGenParms=False scalarPot3DInitFile=$INIT trackSrc=generalTracks \
useLegacyPairLoop=True doTrigger=False applyHltFilter=False \
useIdealGeometry=True useDefaultField=True globalTag=150X_mcRun2_asymptotic_v1"

run_task() {
  local spec=$1
  local g=${spec%%:*}
  local idx=${spec##*:}
  local outdir="$OUTROOT/resolution_matgroup_${OUTTAG}_g$(printf '%02d' "$g")/task_$(printf '%04d' "$idx")"
  [[ -f "$outdir/.complete" ]] && { echo "[skip] g=$g task $idx"; return 0; }
  local input="$SIMROOT/task_$(printf '%04d' "$idx")/step2.root"
  [[ -f "$input" ]] || { echo "[err] missing $input"; return 1; }
  mkdir -p "$outdir"
  rm -f "$outdir"/globalcor_*.root "$outdir/.complete"
  export CVH_MATGROUP_PROBE=$g
  export CVH_MATGROUP_EPS=$EPS
  export CVH_MATGROUP_MEANONLY=1
  # shellcheck disable=SC2086
  if "$RUN_ONE" "$CFG" "$input" "$outdir" $COMMON > "$outdir/local.log" 2>&1; then
    touch "$outdir/.complete"; echo "[done] g=$g task $idx"
  else
    echo "[FAIL] g=$g task $idx (see $outdir/local.log)"
  fi
}
export -f run_task
export CFG RUN_ONE INIT SIMROOT OUTROOT OUTTAG COMMON EPS

JOBS=()
for g in $MATGROUPS; do
  for ((i=0; i<NTASKS; ++i)); do JOBS+=("$g:$i"); done
done
echo "${#JOBS[@]} tasks, eps=$EPS, ${NPAR}-way"
printf '%s\n' "${JOBS[@]}" | xargs -P "$NPAR" -I{} bash -c 'run_task {}'
echo "matgroup scan finished"
