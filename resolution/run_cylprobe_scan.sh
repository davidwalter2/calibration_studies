#!/bin/bash
# Nested-cylinder influence-weight probe (CVH_ELOSS_CYL_R/_Z/_EPS).
#
# For a family of nested cylinders (r < 120u, |z| < 300u) covering the tracker
# in order along any outgoing track, applies a coherent MEAN-ONLY log-scale
# shift to the energy loss INSIDE the cylinder. Paired per track against the
# nominal this gives the cumulative influence profile
#     Lambda(u) = sum_{inside u} w_i mu_i   (from the p_fit response)
#     M(u)      = sum_{inside u} mu_i       (from the dEref response)
# and w(u) = dLambda/dM. The largest cylinder must reproduce the global
# T_sys = 0.6127 -- that is the closure.
#
# The hook is in G4ErrorEnergyLossForCVH::AlongStepDoIt, which is the MEAN-loss
# path only, so Q is untouched by construction.
#
# usage: ./run_cylprobe_scan.sh [nparallel] [ntasks] [nevents]
set -euo pipefail
NPAR=${1:-96}
NTASKS=${2:-8}
NEVENTS=${3:--1}
EPS=${EPS:-0.693147}
# u x 1000; 1400 = everything (closure against the global T_sys)
UVALS=${UVALS:-"040 080 120 180 250 350 500 700 1000 1400"}
RMAX=120.0
ZMAX=300.0

CFG=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/test/runCvhJpsiGenMC.py
export CMSSW_AREA=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev
RUN_ONE=/work/submit/david_w/ZMass/calibration_studies/slurm/run_one.sh
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
SIMROOT=/ceph/submit/data/user/d/david_w/ZMass/cvh/resolution_simprod_jpsigun_ul16
OUTROOT=/ceph/submit/data/user/d/david_w/ZMass/cvh
OUTTAG=${OUTTAG:-260814cyl}

COMMON="nEvents=${NEVENTS} numberOfThreads=1 doRes=True fillGrads=True \
fitFromGenParms=False scalarPot3DInitFile=$INIT trackSrc=generalTracks \
useLegacyPairLoop=True doTrigger=False applyHltFilter=False \
useIdealGeometry=True useDefaultField=True globalTag=150X_mcRun2_asymptotic_v1"

run_task() {
  local spec=$1
  local u=${spec%%:*}
  local idx=${spec##*:}
  local outdir="$OUTROOT/resolution_cylprobe_${OUTTAG}_u${u}/task_$(printf '%04d' "$idx")"
  [[ -f "$outdir/.complete" ]] && { echo "[skip] u=$u task $idx"; return 0; }
  local input="$SIMROOT/task_$(printf '%04d' "$idx")/step2.root"
  [[ -f "$input" ]] || { echo "[err] missing $input"; return 1; }
  mkdir -p "$outdir"
  rm -f "$outdir"/globalcor_*.root "$outdir/.complete"
  export CVH_ELOSS_CYL_R=$(awk -v u="$u" -v r="$RMAX" 'BEGIN{printf "%.4f", r*u/1000.}')
  export CVH_ELOSS_CYL_Z=$(awk -v u="$u" -v z="$ZMAX" 'BEGIN{printf "%.4f", z*u/1000.}')
  export CVH_ELOSS_CYL_EPS=$EPS
  # shellcheck disable=SC2086
  if "$RUN_ONE" "$CFG" "$input" "$outdir" $COMMON > "$outdir/local.log" 2>&1; then
    touch "$outdir/.complete"; echo "[done] u=$u task $idx"
  else
    echo "[FAIL] u=$u task $idx (see $outdir/local.log)"
  fi
}
export -f run_task
export CFG RUN_ONE INIT SIMROOT OUTROOT OUTTAG COMMON EPS RMAX ZMAX

JOBS=()
for u in $UVALS; do
  for ((i=0; i<NTASKS; ++i)); do JOBS+=("$u:$i"); done
done
echo "${#JOBS[@]} tasks, eps=$EPS, ${NPAR}-way"
printf '%s\n' "${JOBS[@]}" | xargs -P "$NPAR" -I{} bash -c 'run_task {}'
echo "cylinder probe scan finished"
