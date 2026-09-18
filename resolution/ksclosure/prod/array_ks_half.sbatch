#!/bin/bash
# Slurm array: HALF a 252-file chunk per task, so a task is ~5 min and fits the
# backfill window on a busy `submit` partition (measured 2026-09-18: a 12 min
# request backfills at once, a 15 min one waits behind the reservations).  Task
# 2n is the first 126 files of chunk n, task 2n+1 the last 126; both write into
# the SAME `task_NNNN/` with a distinct `outprefix`, and `ks_pairs.py` globs
# `globalcor_ks_*.root` there, so the chunking the truth join sees is unchanged.
#
# env (via --export): CONFIG, HALFDIR, OUTDIR, CMSSW_AREA, CMSRUN_ARGS
#SBATCH --job-name=cvhks
#SBATCH --partition=submit
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH --time=0:12:00
set -euo pipefail
: "${CONFIG:?}"; : "${HALFDIR:?}"; : "${OUTDIR:?}"; : "${CMSSW_AREA:?}"
: "${SLURM_ARRAY_TASK_ID:?}"

N=$((SLURM_ARRAY_TASK_ID / 2))
case $((SLURM_ARRAY_TASK_ID % 2)) in 0) H=a;; 1) H=b;; esac
IDX=$(printf '%04d' "$N")
CHUNK="$HALFDIR/half_${IDX}_${H}.txt"
[[ -s "$CHUNK" ]] || { echo "missing half list $CHUNK" >&2; exit 1; }
TASK_DIR="$OUTDIR/task_$IDX"
PREFIX="globalcor_ks_$H"
if [[ -f "$TASK_DIR/.complete_$H" ]]; then echo "already done"; exit 0; fi
mkdir -p "$TASK_DIR"
# ONLY this half's outputs -- the sibling's must survive
rm -f "$TASK_DIR"/${PREFIX}_*.root

cd "$CMSSW_AREA/src"
source /cvmfs/cms.cern.ch/cmsset_default.sh
eval $(scramv1 runtime -sh)
cd "$TASK_DIR"
echo ">>> $(date) host=$(hostname) half=$CHUNK"
EXTRA=()
if [[ -n "${CMSRUN_ARGS:-}" ]]; then read -r -a EXTRA <<< "$CMSRUN_ARGS"; fi
cmsRun "$CONFIG" inputFileList="$CHUNK" outprefix="$PREFIX" "${EXTRA[@]}"
touch "$TASK_DIR/.complete_$H"
# the chunk is complete when both halves are; a simultaneous finish can leave
# the seal to `seal_halves.sh`, which is idempotent
if [[ -f "$TASK_DIR/.complete_a" && -f "$TASK_DIR/.complete_b" ]]; then
  printf "streams=1 halves=2\n" > "$TASK_DIR/.complete"
  touch "$TASK_DIR/.done"
fi
echo ">>> $(date) finished"
