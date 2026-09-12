#!/bin/bash
# Local production for the HIT-RESOLUTION pull study.
#
# What makes this configuration different from run_local_trackres.sh:
#   doSimHits=True        the PSimHit collections are read and matched, which
#                         is what fills dxrecsim / dyrecsim;
#   fitFromGenParms=True  the per-hit validation branches (dxrecsim, dxerr,
#                         hitUProj, cluster*) are only BOOKED in this mode --
#                         see the `if (fitFromGenParms_)` guard on the branch
#                         block in ResidualGlobalCorrectionMakerG4e.cc;
#   doRes=False           parmtype 8-11 are not needed here and cost time.
#
# The private guns are the only samples with PSimHits (the JPsiToMuMu and
# B->J/psi+X ALCARECOs have none), and they were simulated with IDEAL geometry,
# the DEFAULT (OAE) tracker field and GT 150X_mcRun2_asymptotic_v1 -- all three
# must be mirrored in the refit or the residual picks up a sim-vs-refit
# difference that has nothing to do with the hits.
#
# usage: ./run_local_hitres.sh <species> <filelist> <outtag> [nparallel] [from] [to]
#   env: EXTRA (extra cmsRun args, e.g. fitSimHitPositions=True), NEVENTS
set -euo pipefail
SPECIES=$1
FILELIST=$2
OUTTAG=$3
NPAR=${4:-40}
TASKS_FROM=${5:-0}
TASKS_TO=${6:-39}
CFG=${CFG:-/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/test/runCvhResClosure.py}
export CMSSW_AREA=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev
RUN_ONE=/work/submit/david_w/ZMass/calibration_studies/slurm/run_one.sh
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
OUTROOT=/ceph/submit/data/user/d/david_w/ZMass/cvh
COMMON="nEvents=${NEVENTS:--1} numberOfThreads=1 particle=$SPECIES doSimHits=True \
doRes=False fillGrads=False fitFromGenParms=True trackSrc=generalTracks \
useDefaultField=True useIdealGeometry=True globalTag=150X_mcRun2_asymptotic_v1 \
scalarPot3DInitFile=$INIT ${EXTRA:-}"

run_task() {
  local idx=$1
  local outdir="$OUTROOT/${OUTFAM:-hitres}_${OUTTAG}/task_$(printf "%04d" "$idx")"
  # every stream the task wrote, not `_0` by name: these runs pin
  # numberOfThreads=1, but a leftover from a multithreaded run must be wiped
  # whole, not down to streams 1..N-1.
  local outglob="$outdir/globalcor_resclosure_*.root"
  # Resume on a COMPLETION SENTINEL, never on the .root: cmsRun creates its
  # output at START, so a killed task leaves a non-empty TRUNCATED file and a
  # `-s` test would skip it forever (the trap documented in
  # run_local_trackres.sh).
  if [[ -f "$outdir/.complete" ]]; then echo "[skip] $OUTTAG task $idx"; return 0; fi
  sleep $(( (idx % NPAR) * ${STAGGER:-2} ))   # avoid the NSS/CVMFS thundering herd
  local input
  input=$(sed -n "$((idx + 1))p" "$FILELIST")
  [[ -n "$input" ]] || { echo "[err] empty filelist line for task $idx"; return 1; }
  mkdir -p "$outdir"
  rm -f $outglob
  echo "[run ] $OUTTAG task $idx"
  # shellcheck disable=SC2086
  if "$RUN_ONE" "$CFG" "$input" "$outdir" $COMMON > "$outdir/local.log" 2>&1; then
    touch "$outdir/.complete"; echo "[done] $OUTTAG task $idx"
  else
    echo "[FAIL] $OUTTAG task $idx (see $outdir/local.log)"
  fi
}
export -f run_task
export CFG RUN_ONE INIT FILELIST OUTROOT OUTTAG COMMON NPAR STAGGER OUTFAM
seq "$TASKS_FROM" "$TASKS_TO" | xargs -P "$NPAR" -I{} bash -c 'run_task {}'
echo "hitres $OUTTAG tasks $TASKS_FROM-$TASKS_TO finished"
