#!/bin/bash
# Local (submit82) complement to submit_resolution_closure.sh: runs tasks
# 50-99 of each alpha variant on the local host while the slurm arrays
# (5731397/98/99, held to tasks 0-49) drain the queue. Writes into the SAME
# hash-tagged outdirs and task_<idx> layout as the slurm jobs, so the
# downstream fit_global_grads.py glob is source-agnostic.
#
# A task is skipped if its output file already exists and is non-empty
# (idempotent restarts; also protects against re-running a slurm-completed
# task if the split is ever changed).
#
# usage: ./run_local_closure.sh [nparallel] [task_from] [task_to]
#        (defaults: 48 50 99; slurm arrays were cancelled 2026-07-23 evening,
#        so the 0-49 half was run locally too)
# env overrides:
#   VARIANTS  space-separated subset of "alpha999 alpha997 alpha995"
#   OUTTAG    outdir date tag (default 260723; e.g. 260724_censor for the
#             gradchisqv-enabled rerun)
set -euo pipefail

NPAR=${1:-48}
TASKS_FROM=${2:-50}
TASKS_TO=${3:-99}
VARIANTS=${VARIANTS:-"alpha999 alpha997 alpha995"}
OUTTAG=${OUTTAG:-260723}
CFG=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/test/runCvhResClosure.py
export CMSSW_AREA=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev
RUN_ONE=/work/submit/david_w/ZMass/calibration_studies/slurm/run_one.sh
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
FILELIST=/work/submit/david_w/ZMass/calibration_studies/pixelhits/filelist_mc_all_chunk50.txt
OUTROOT=/ceph/submit/data/user/d/david_w/ZMass/cvh
HASH=fb01e7b7741
COMMON="nEvents=-1 numberOfThreads=1 doRes=True fillGrads=True scalarPot3DInitFile=$INIT ${EXTRA:-}"

run_task() {
  local variant=$1 alphaarg=$2 idx=$3
  local outdir="$OUTROOT/resolution_closure_${OUTTAG}_${variant}_${HASH}/task_$(printf '%04d' "$idx")"
  local outfile="$outdir/globalcor_resclosure_0.root"
  if [[ -s "$outfile" ]]; then
    echo "[skip] $variant task $idx (output exists)"
    return 0
  fi
  local input
  input=$(sed -n "$((idx + 1))p" "$FILELIST")
  [[ -n "$input" ]] || { echo "[err] empty filelist line for task $idx"; return 1; }
  mkdir -p "$outdir"
  echo "[run ] $variant task $idx -> $outdir"
  # shellcheck disable=SC2086
  "$RUN_ONE" "$CFG" "$input" "$outdir" $COMMON $alphaarg \
      > "$outdir/local.log" 2>&1 \
      && echo "[done] $variant task $idx" \
      || echo "[FAIL] $variant task $idx (see $outdir/local.log)"
}
export -f run_task
export CFG RUN_ONE INIT FILELIST OUTROOT HASH COMMON OUTTAG

{
  for idx in $(seq $TASKS_FROM $TASKS_TO); do
    for v in $VARIANTS; do
      case $v in
        alpha999) echo "alpha999|| $idx";;
        alpha997) echo "alpha997|ioniTruncationAlpha=0.997| $idx";;
        alpha995) echo "alpha995|ioniTruncationAlpha=0.995| $idx";;
        *) echo "unknown variant $v" >&2; exit 1;;
      esac
    done
  done
} | xargs -P "$NPAR" -I{} bash -c '
  IFS="|" read -r v a i <<< "{}"
  run_task "$v" "$a" "$i"
'
echo "all local tasks finished"
