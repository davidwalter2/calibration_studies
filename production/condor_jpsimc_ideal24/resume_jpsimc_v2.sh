#!/bin/bash
# Re-drive the tasks of jpsimc_20M_260906_v2 that have no `.complete` sentinel.
#
# Same contract as production/resume.sh: the task index IS the chunk-list
# line, the configuration is sourced from the one shared file so a resumed task
# can never run a different physics configuration from the original
# submission, and completion is judged on the sentinel alone.
#
# usage: ./resume_jpsimc_v2.sh [--dry-run] [--only "3 7 11"] [--max N]
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/config_jpsimc_v2.sh"
DRY=0 ; ONLY="" ; MAX=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift;;
    --only)    ONLY=$2; shift 2;;
    --max)     MAX=$2; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

NTASK=$(grep -cvE '^\s*(#|$)' "$CHUNKLIST")
NTASK=${NTASKS_USED:-$NTASK}
# Indices already queued must not be queued twice: two jobs writing the same
# task dir would interleave their cmsRun outputs.
mapfile -t QUEUED < <(condor_q "$USER" -af CvhOutBase CvhTaskIdx 2>/dev/null \
                      | awk -v b="$OUTBASE" '$1 == b && $2 ~ /^[0-9]+$/ {print $2}' || true)
declare -A INQ=(); for q in ${QUEUED[@]+"${QUEUED[@]}"}; do [[ -n "$q" ]] && INQ[$q]=1; done

MISSING=()
for (( i=0; i<NTASK; i++ )); do
  [[ -n "$ONLY" ]] && ! grep -qw "$i" <<< "$ONLY" && continue
  [[ -f "$OUTBASE/task_$(printf '%04d' "$i")/.complete" ]] && continue
  [[ -n "${INQ[$i]:-}" ]] && continue
  MISSING+=("$i")
  (( MAX > 0 && ${#MISSING[@]} >= MAX )) && break
done

echo "tag=$TAG  tasks=$NTASK  already queued=${#QUEUED[@]}  to resubmit=${#MISSING[@]}"
[[ ${#MISSING[@]} -eq 0 ]] && exit 0
printf '%s\n' "${MISSING[@]}" > "$HERE/.resume_idx.txt"
echo "indices -> $HERE/.resume_idx.txt"
(( DRY )) && { head -20 "$HERE/.resume_idx.txt"; exit 0; }
exec "$HERE/submit_jpsimc_v2.sh" --idxfile "$HERE/.resume_idx.txt"
