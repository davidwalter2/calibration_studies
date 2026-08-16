#!/bin/bash
# Coherent energy-loss-reference transmission probe (CRITIQUE 260813 action 1.2).
#
# Re-runs the J/psi-gun CVH refit at several values of CVH_DEDX_SCALE, which is
# a COHERENT multiplicative shift of the reference mean dE/dx applied at
# table-build time (G4TablesForExtrapolatorForCVH), i.e. the same sign at every
# step of every track. The build carries a new per-leg branch Mu{plus,minus}_dEref
# = the total mean energy loss the reference trajectory actually applied, so the
# response can be normalised per track without any external model:
#
#     T_sys = d(p_fit at reference) / d(assumed total eloss)
#
# CVH_DEDX_SCALE touches ONLY the mean: the ionisation variance in Q comes from
# the Urban model in Geant4ePropagator::computeErrorIoni, not from the scaled
# table. So this is a pure reference re-centring, not a material rescale.
#
# usage: ./run_transmission_scan.sh [nparallel] [ntasks] [nevents]
set -euo pipefail
NPAR=${1:-16}
NTASKS=${2:-8}
NEVENTS=${3:--1}

SCALES=${SCALES:-"1.000 0.950 0.900 1.050 1.100"}

CFG=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/test/runCvhJpsiGenMC.py
export CMSSW_AREA=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev
RUN_ONE=/work/submit/david_w/ZMass/calibration_studies/slurm/run_one.sh
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
SIMROOT=/ceph/submit/data/user/d/david_w/ZMass/cvh/resolution_simprod_jpsigun_ul16
OUTROOT=/ceph/submit/data/user/d/david_w/ZMass/cvh
OUTTAG=${OUTTAG:-260813}

# EXACTLY the argument list of the existing dedx scan productions (from their
# local.log), so the only difference between this scan and those is the build
# and CVH_DEDX_SCALE.
COMMON="nEvents=${NEVENTS} numberOfThreads=1 doRes=True fillGrads=True \
fitFromGenParms=False scalarPot3DInitFile=$INIT trackSrc=generalTracks \
useLegacyPairLoop=True doTrigger=False applyHltFilter=False \
useIdealGeometry=True useDefaultField=True globalTag=150X_mcRun2_asymptotic_v1"

run_task() {
  local spec=$1
  local scale=${spec%%:*}
  local idx=${spec##*:}
  local stag
  stag=$(awk -v s="$scale" 'BEGIN{printf "%04d", s*1000+0.5}')
  local outdir="$OUTROOT/resolution_transmission_${OUTTAG}_s${stag}/task_$(printf '%04d' "$idx")"
  [[ -f "$outdir/.complete" ]] && { echo "[skip] s=$scale task $idx"; return 0; }
  local input="$SIMROOT/task_$(printf '%04d' "$idx")/step2.root"
  [[ -f "$input" ]] || { echo "[err] missing $input"; return 1; }
  mkdir -p "$outdir"
  rm -f "$outdir"/globalcor_*.root "$outdir/.complete"
  export CVH_DEDX_SCALE=$scale
  echo "[run ] s=$scale task $idx"
  # shellcheck disable=SC2086
  if "$RUN_ONE" "$CFG" "$input" "$outdir" $COMMON > "$outdir/local.log" 2>&1; then
    touch "$outdir/.complete"; echo "[done] s=$scale task $idx"
  else
    echo "[FAIL] s=$scale task $idx (see $outdir/local.log)"
  fi
}
export -f run_task
export CFG RUN_ONE INIT SIMROOT OUTROOT OUTTAG COMMON

JOBS=()
for s in $SCALES; do
  for ((i=0; i<NTASKS; ++i)); do JOBS+=("$s:$i"); done
done
printf '%s\n' "${JOBS[@]}" | xargs -P "$NPAR" -I{} bash -c 'run_task {}'
echo "transmission scan finished"
