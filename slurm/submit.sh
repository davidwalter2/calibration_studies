#!/bin/bash
# Submit a Slurm array job for cvh refit (or any per-file cmsRun job).
#
# usage:
#   ./submit.sh --config <cfg> --filelist <list> --outdir <dir>
#               [--max-running N] [--time HH:MM:SS] [--mem 4G]
#               [--partition submit] [--name cvh] [--dry-run]
#
# Notes:
#   * The cmsRun config must accept VarParsing 'input=<path>'. See
#     Analysis/HitAnalyzer/test/benchmark_io/bench_cmsrun_cfg.py for a
#     template (raw JPsi ALCARECO -> CVH refit).
#   * Per-task output lands in $OUTDIR/task_<idx>/; Slurm logs in $OUTDIR/logs/.
#   * --max-running caps concurrent array tasks (Ceph/NFS friendliness).
set -euo pipefail

CONFIG=""
FILELIST=""
OUTDIR=""
MAX_RUNNING=""
TIME="12:00:00"
MEM="4G"
PARTITION="submit"
NAME="cvh"
DRY_RUN=0

usage() {
  sed -n '2,15p' "$0"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)      CONFIG=$2; shift 2;;
    --filelist)    FILELIST=$2; shift 2;;
    --outdir)      OUTDIR=$2; shift 2;;
    --max-running) MAX_RUNNING=$2; shift 2;;
    --time)        TIME=$2; shift 2;;
    --mem)         MEM=$2; shift 2;;
    --partition)   PARTITION=$2; shift 2;;
    --name)        NAME=$2; shift 2;;
    --dry-run)     DRY_RUN=1; shift;;
    -h|--help)     usage;;
    *) echo "unknown arg: $1" >&2; usage;;
  esac
done

[[ -z "$CONFIG" || -z "$FILELIST" || -z "$OUTDIR" ]] && usage

CONFIG=$(readlink -f "$CONFIG")
FILELIST=$(readlink -f "$FILELIST")
OUTDIR=$(readlink -f -m "$OUTDIR")
SLURM_DIR=$(dirname "$(readlink -f "$0")")

[[ -f "$CONFIG" ]]   || { echo "config not found: $CONFIG"   >&2; exit 1; }
[[ -f "$FILELIST" ]] || { echo "filelist not found: $FILELIST" >&2; exit 1; }

# Count non-blank, non-comment lines.
NJOBS=$(grep -cvE '^\s*(#|$)' "$FILELIST")
[[ $NJOBS -eq 0 ]] && { echo "filelist has 0 jobs"; exit 1; }
LAST=$((NJOBS - 1))

ARRAY_SPEC="0-$LAST"
[[ -n "$MAX_RUNNING" ]] && ARRAY_SPEC="${ARRAY_SPEC}%${MAX_RUNNING}"

mkdir -p "$OUTDIR/logs"

cmd=(
  sbatch
  --job-name="$NAME"
  --partition="$PARTITION"
  --array="$ARRAY_SPEC"
  --time="$TIME"
  --mem="$MEM"
  --output="$OUTDIR/logs/${NAME}_%A_%a.out"
  --error="$OUTDIR/logs/${NAME}_%A_%a.err"
  --export="ALL,CONFIG=$CONFIG,FILELIST=$FILELIST,OUTDIR=$OUTDIR"
  "$SLURM_DIR/array.sbatch"
)

echo "submitting: array $ARRAY_SPEC ($NJOBS files)"
echo "  config   : $CONFIG"
echo "  filelist : $FILELIST"
echo "  outdir   : $OUTDIR"
echo "  partition: $PARTITION  time=$TIME  mem=$MEM"
[[ -n "$MAX_RUNNING" ]] && echo "  max-running: $MAX_RUNNING"

if (( DRY_RUN )); then
  printf '  cmd      :'
  printf ' %q' "${cmd[@]}"
  echo
  exit 0
fi

"${cmd[@]}"
