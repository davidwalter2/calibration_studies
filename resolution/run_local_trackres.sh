#!/bin/bash
# Local production for the per-track resolution closure (influence-weight
# export, NOMINAL fit: fitFromGenParms=False). Same task layout as
# run_local_closure.sh so downstream globs stay uniform.
# usage: ./run_local_trackres.sh [nparallel] [task_from] [task_to]
set -euo pipefail
NPAR=${1:-12}
TASKS_FROM=${2:-0}
TASKS_TO=${3:-11}
CFG=${CFG:-/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/test/runCvhResClosure.py}
export CMSSW_AREA=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev
RUN_ONE=/work/submit/david_w/ZMass/calibration_studies/slurm/run_one.sh
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
FILELIST=${FILELIST:-/work/submit/david_w/ZMass/calibration_studies/pixelhits/filelist_mc_all_chunk50.txt}
OUTROOT=/ceph/submit/data/user/d/david_w/ZMass/cvh
OUTTAG=${OUTTAG:-260802}
COMMON="nEvents=${NEVENTS:--1} numberOfThreads=1 doRes=True fillGrads=True fitFromGenParms=False scalarPot3DInitFile=$INIT ${EXTRA:-}"

run_task() {
  local idx=$1
  local outdir="$OUTROOT/resolution_trackres_${OUTTAG}/task_$(printf '%04d' "$idx")"
  local outfile="$outdir/globalcor_resclosure_0.root"
  # Resume on a COMPLETION SENTINEL, never on the .root itself. cmsRun creates
  # its output at START, so a killed or crashed task leaves a NON-EMPTY but
  # TRUNCATED file, and a `-s` test then skips it forever -- the run looks
  # instantaneous and silently produces nothing. (Cost ~an hour on 2026-08-08;
  # same trap as the simprod step2 outputs and the extract_parallel merge.)
  #
  # Checked BEFORE the stagger sleep below: a fully-complete sample re-run for
  # resume would otherwise have all 160 tasks sleep up to STAGGER*NPAR seconds
  # each only to skip, ~8 minutes of pure waiting per pass (measured).
  if [[ -f "$outdir/.complete" ]]; then echo "[skip] task $idx"; return 0; fi
  # STAGGERED START. xargs -P fires all NPAR jobs at the same instant; at 60+
  # that wedged every one of them in futex_do_wait inside the NSS/sssd user
  # lookup (0 s CPU in 2 min, nothing open but the log and
  # /var/lib/sss/mc/passwd), while a SINGLE job runs at 90% CPU. Spreading the
  # starts over a few seconds each avoids the thundering herd on CMSSW/CVMFS
  # startup. Cost is bounded: STAGGER*NPAR seconds once per batch.
  sleep $(( (idx % ${NPAR:-12}) * ${STAGGER:-2} ))
  local input
  input=$(sed -n "$((idx + 1))p" "$FILELIST")
  [[ -n "$input" ]] || { echo "[err] empty filelist line for task $idx"; return 1; }
  mkdir -p "$outdir"
  rm -f "$outfile" "$outdir/.complete"      # drop any truncated leftover
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
echo "all local trackres tasks finished"
