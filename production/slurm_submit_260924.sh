#!/bin/bash
# Submit (or resume) one 260924 closure production on slurm: every index of the
# production's first NTASKS_USED chunk lines that has no `.complete` sentinel
# and is not already queued under this production's job name.
#   usage: ./slurm_submit_260924.sh <condor_*24 dir> [--time H:MM:SS] [--throttle N] [--dry-run]
set -euo pipefail
P=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DIR=$(cd "$1" && pwd); shift
TIME=6:00:00; THR=150; DRY=0
while [[ $# -gt 0 ]]; do case $1 in --time) TIME=$2; shift 2;; --throttle) THR=$2; shift 2;; --dry-run) DRY=1; shift;; *) exit 1;; esac; done
source "$DIR"/config_*_v2.sh
NAME=cvh24_${TAG#*_*_}; NAME=${TAG}
mapfile -t Q < <(squeue -u "$USER" -h -r -n "$NAME" -o "%K" 2>/dev/null)
declare -A INQ=(); for q in ${Q[@]+"${Q[@]}"}; do INQ[$q]=1; done
MISS=()
for (( i=0; i<NTASKS_USED; i++ )); do
  [[ -f "$OUTBASE/task_$(printf '%04d' "$i")/.complete" ]] && continue
  [[ -n "${INQ[$i]:-}" ]] && continue
  MISS+=("$i")
done
echo "tag=$TAG tasks=$NTASKS_USED queued=${#Q[@]} to submit=${#MISS[@]}"
(( ${#MISS[@]} )) || exit 0
LIST=$(printf '%s\n' "${MISS[@]}" | awk 'NR==1{s=p=$1;next} $1==p+1{p=$1;next} {printf "%s,", (s==p?s:s"-"p); s=p=$1} END{print (s==p?s:s"-"p)}')
mkdir -p "$OUTBASE/logs"
CMD=(sbatch --parsable --job-name="$NAME" --time="$TIME" --array="${LIST}%${THR}"
     --output="$OUTBASE/logs/slurm_%a.out" --error="$OUTBASE/logs/slurm_%a.err"
     --export=ALL,DIR="$DIR" "$P/slurm_task_260924.sbatch")
(( DRY )) && { echo "${CMD[@]}"; exit 0; }
"${CMD[@]}"
