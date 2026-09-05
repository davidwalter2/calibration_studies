#!/bin/bash
# Resubmit ONLY the chunks of the 20M J/psi MC production that have no
# .complete sentinel.
#
# WHY NOT sacct: slurm/resubmit_failed.sh reconstructs the failed set from the
# job accounting, which needs the array job id, expires with the accounting
# retention, and cannot see a task that completed with rc=0 but wrote nothing.
# The sentinel is the ground truth: array_jpsimc.sbatch writes it only after
# checking that the output exists, is non-empty, and that the fit summary does
# not say attempted=0.
#
# The task index IS the chunk-list line number (offset included), so a resume
# is just "submit the missing indices" -- no re-chunking, no renumbering, and
# a resumed task reads exactly the same events as the one it replaces.
#
# usage: ./resume.sh [--dry-run] [--max-running N] [--list-only]
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shared with submit_jpsimc20M.sh so the two can never drift apart
source "$HERE/config_jpsimc20M.sh"
CHUNKLIST=$HERE/chunks_${TAG}.txt
DRY=0; LISTONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)     DRY=1; shift;;
    --list-only)   LISTONLY=1; shift;;
    --max-running) MAXRUNNING=$2; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

NTASK=$(grep -cvE '^\s*(#|$)' "$CHUNKLIST")
# Never resubmit something that is still queued or running: a second task
# writing the same directory would race the first one's output file.
mapfile -t INQ < <(squeue -u "$USER" -h -r -o "%j %K" 2>/dev/null | awk '$1 ~ /^jpsimc20M_a/ {
    split($1, p, "_a"); off = (p[2] + 0) * 1000;
    n = split($2, ids, ",");
    for (i = 1; i <= n; i++) {
      spec = ids[i]; sub(/%.*/, "", spec);
      if (spec ~ /-/) { split(spec, r, "-"); lo = r[1] + 0; hi = r[2] + 0;
                        for (j = lo; j <= hi; j++) print j + off }
      else if (spec ~ /^[0-9]+$/) { print (spec + 0) + off }
    }
  }')
declare -A BUSY=(); for i in "${INQ[@]:-}"; do [[ -n "$i" ]] && BUSY[$i]=1; done

MISSING=()
for (( i=0; i<NTASK; i++ )); do
  d="$OUTBASE/task_$(printf '%04d' "$i")"
  [[ -f "$d/.complete" ]] && continue
  [[ -n "${BUSY[$i]:-}" ]] && continue
  MISSING+=("$i")
done

echo "tasks=$NTASK complete=$(( NTASK - ${#MISSING[@]} - ${#BUSY[@]} )) in-queue=${#BUSY[@]} missing=${#MISSING[@]}"
(( ${#MISSING[@]} == 0 )) && { echo "nothing to resume"; exit 0; }
printf '  indices: %s%s\n' "$(printf '%s ' "${MISSING[@]:0:20}")" \
       "$( (( ${#MISSING[@]} > 20 )) && echo "... (+$(( ${#MISSING[@]} - 20 )) more)")"
(( LISTONLY )) && exit 0


# One array per contiguous run is overkill; a comma list of explicit indices is
# what sbatch --array takes, and MaxArraySize bounds the largest INDEX, so the
# list is chopped into groups whose max index is < 1000 by re-basing with the
# same IDXOFFSET mechanism the first submission uses.
mkdir -p "$OUTBASE/logs"
declare -A GROUPS=()
for i in "${MISSING[@]}"; do
  g=$(( i / 1000 ))
  GROUPS[$g]="${GROUPS[$g]:-}${GROUPS[$g]:+,}$(( i % 1000 ))"
done
for g in "${!GROUPS[@]}"; do
  OFF=$(( g * 1000 ))
  cmd=( sbatch --job-name="jpsimc20M_a${g}" --partition=submit
        --array="${GROUPS[$g]}%${MAXRUNNING}" --time=12:00:00 --mem=6G --cpus-per-task=1
        --output="$OUTBASE/logs/jpsimc20M_a${g}_%A_%a.out"
        --error="$OUTBASE/logs/jpsimc20M_a${g}_%A_%a.err"
        --export="ALL,CHUNKLIST=$CHUNKLIST,OUTBASE=$OUTBASE,IDXOFFSET=$OFF,EXTRA=$EXTRA,CMSSW_AREA=$CMSSW_AREA"
        "$HERE/array_jpsimc.sbatch" )
  n=$(awk -F, '{print NF}' <<< "${GROUPS[$g]}")
  echo "  resume array $g: $n tasks, offset=$OFF"
  if (( DRY )); then printf '    '; printf '%q ' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
done
