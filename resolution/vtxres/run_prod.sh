#!/bin/bash
# vtxres production: the two-track J/psi-gun refit with exportVtxResidual on.
#
# Same input, same fit settings as the mass-card production
# `resolution_trackres_jpsigun_ul16_260905d_m0` (so a joint vertex+mass fit
# shares candidates), plus:
#   exportVtxResidual=True        the new block
#   exportCfGroupExponents=True   the per-material-group split BOTH functionals
#                                 need for a MaterialCF card
#   exportStepRecords=False       the in-maker exponents replace them (the
#                                 260905d production had to carry 430 kB/cand
#                                 because its exponents were built offline)
# doVtxConstraint stays FALSE -- index 6 free, so the fit reports the DCA,
# which is what every production does.
#
# usage: ./run_prod.sh <nparallel> <task_from> <task_to>
set -uo pipefail
NPAR=${1:-40}
TASKS_FROM=${2:-0}
TASKS_TO=${3:-159}
CMSSW_AREA=${CMSSW_AREA:-/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2}
CFG=${CFG:-$CMSSW_AREA/src/Analysis/HitAnalyzer/test/runCvhJpsiGenMC.py}
RUN_ONE=/work/submit/david_w/ZMass/calibration_studies/slurm/run_one.sh
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
FILELIST=${FILELIST:-/work/submit/david_w/ZMass/calibration_studies/resolution/simprod/filelist_jpsigun_ul16.txt}
OUTROOT=${OUTROOT:-/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911}
OUTTAG=${OUTTAG:-prod}
export CMSSW_AREA
COMMON="nEvents=${NEVENTS:--1} numberOfThreads=1 doRes=True fillGrads=True fitFromGenParms=False \
scalarPot3DInitFile=$INIT trackSrc=generalTracks useLegacyPairLoop=True doTrigger=False \
applyHltFilter=False useIdealGeometry=True useDefaultField=True \
globalTag=150X_mcRun2_asymptotic_v1 CgfQoPMode=0 armijoSlack=1.0 \
exportVtxResidual=True exportCfGroupExponents=True exportStepRecords=False ${EXTRA:-}"

run_task() {
  local idx=$1
  local outdir="$OUTROOT/$OUTTAG/task_$(printf '%04d' "$idx")"
  # Resume on a COMPLETION SENTINEL, never on the .root: cmsRun creates its
  # output at START, so a killed task leaves a non-empty TRUNCATED file.
  if [[ -f "$outdir/.complete" ]]; then echo "[skip] task $idx"; return 0; fi
  sleep $(( (idx % ${NPAR:-12}) * ${STAGGER:-3} ))
  local input
  input=$(sed -n "$((idx + 1))p" "$FILELIST")
  [[ -n "$input" ]] || { echo "[err] empty filelist line for task $idx"; return 1; }
  mkdir -p "$outdir"
  rm -f "$outdir"/globalcor_*.root "$outdir/.complete"
  echo "[run ] task $idx -> $outdir"
  # shellcheck disable=SC2086
  if "$RUN_ONE" "$CFG" "$input" "$outdir" $COMMON > "$outdir/local.log" 2>&1; then
    touch "$outdir/.complete"; echo "[done] task $idx"
  else
    echo "[FAIL] task $idx (see $outdir/local.log)"
  fi
}
export -f run_task
export CFG RUN_ONE INIT FILELIST OUTROOT OUTTAG COMMON NPAR STAGGER
seq "$TASKS_FROM" "$TASKS_TO" | xargs -P "$NPAR" -I{} bash -c 'run_task {}'
echo "vtxres production block $TASKS_FROM-$TASKS_TO finished"
