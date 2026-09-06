#!/bin/bash
# Re-drive the dymc_8p5M_260905 tasks that DIED WITH SIGABRT (slurm exit 134)
# in the two-track maker's factored-Hessian export, from the FIXED area
# (CMSSW_15_0_19_patch2_dev2, branch cvh-exports-260906).
#
# THE BUG (2026-09-06). `ndof` is unsigned and equals nvalid + nvalidpixel - 10
# for the two-track fit. A Z candidate whose two MiniAOD legs carry only ten
# valid-hit-equivalents between them -- e.g. nhits=(8,1), the second leg a
# single stored hit -- lands on ndof == 0, so nrank = min(ndof, nparsfinal) = 0
# and the truncation-gap report reads `eigvals(nparsfinal)`, one past the end
# of the length-nparsfinal eigenvalue vector. Eigen's bounds assert aborts the
# PROCESS. Measured rate 0.033 aborts per 1000 Z candidates, which at ~9850
# candidates a chunk killed 37 of the first 133 finished tasks (28 %) -- and a
# crashed cmsRun output has NO KEYS, so the whole task is lost, not just the
# candidate. Fixed at cvh-exports-260906 fab515e.
#
# WHY A SEPARATE SCRIPT AND NOT resume_dy.sh: resume_dy.sh resubmits from
# $CMSSW_AREA in config_dymc8p5M.sh, which is the (unfixed, and deliberately
# untouched) production area every running task loads its .so from. This one
# points at dev2 and adds `exportHitResBlocks=False` so the recovered trees
# stay poolable with the rest of the set (see array_dymc_dev2.sbatch's header).
#
# IT DRIVES EVERY UNFINISHED CHUNK, not only the ones that have already
# crashed. On 2026-09-06 the 168 still-PENDING elements of the original array
# 6406978 were cancelled (`scancel --state=PENDING 6406978`; the 69 RUNNING
# ones were left to finish) precisely so that the remaining tail runs the FIXED
# build once instead of the unfixed one plus a recovery -- ~28 % of them would
# otherwise have burned up to 4 h of Geant4e each and produced a keyless file.
# So a chunk is submitted iff it has no `.complete` sentinel AND is not in
# squeue right now; whether it was never started, crashed, or was cancelled
# makes no difference and is deliberately NOT read from sacct (which expires
# with the accounting retention and cannot see a task that exited 0 having
# written nothing).
#
# The two guards that make that safe:
#   * the .complete SENTINEL is the only ground truth for "done" --
#     array_dymc*.sbatch writes it only after checking the output exists, is
#     non-empty, and that the fit summary does not say attempted=0. cmsRun
#     creates its output at START, so an `-s` test on the .root would accept a
#     truncated file forever.
#   * a LIVE squeue check, not a state snapshot: anything queued or running
#     under a `dymc8p5M_a*` name is skipped, because two tasks writing one
#     directory would race.
#
# usage: ./resume_dy_dev2.sh [--dry-run] [--list-only] [--max-running N] [--time T]
#
# NOTE the sbatch --array cap: a slurm array's largest INDEX must be < 1001
# here, which the IDXOFFSET re-basing handles; %MAXRUNNING caps concurrency.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/config_dymc8p5M.sh"

CHUNKLIST=$HERE/chunks_${TAG}.txt
FIXAREA=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2
# The ONE option the branch needs to stay poolable with the production set.
EXTRA="$EXTRA exportHitResBlocks=False"
# 4 h, not 12: the largest chunk measured 2.8 h and a 12 h declaration rarely
# backfills into a full cluster (submit_dymc8p5M.sh's header).
WALLTIME=4:00:00
DRY=0; LISTONLY=0
# Indices to leave alone, whitespace-separated. watch_dy_recover.sh uses it to
# stop re-driving a chunk that has failed the same way N times -- an unattended
# loop that resubmits a genuinely bad input forever is worse than a gap.
EXCLUDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)     DRY=1; shift;;
    --list-only)   LISTONLY=1; shift;;
    --max-running) MAXRUNNING=$2; shift 2;;
    --time)        WALLTIME=$2; shift 2;;
    --exclude)     EXCLUDE=$2; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done
declare -A SKIP=(); for i in $EXCLUDE; do SKIP[$i]=1; done

[[ -d "$FIXAREA/src" ]] || { echo "fixed area invalid: $FIXAREA" >&2; exit 1; }
NTASK=$(grep -cvE '^\s*(#|$)' "$CHUNKLIST")

# Everything of mine currently queued or running under the dymc8p5M_a* names,
# expanded from the array specs squeue prints.
mapfile -t INQ < <(squeue -u "$USER" -h -r -o "%j %K" 2>/dev/null | awk '$1 ~ /^dymc8p5M_a/ {
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

# A chunk is submitted iff it is NOT complete and NOT in flight. Crashed,
# cancelled and never-started are all the same case, and treating them the same
# is the point: the whole unfinished tail must run the fixed build.
DEAD=(); NEVER=0; TRIED=0
for (( i=0; i<NTASK; i++ )); do
  d="$OUTBASE/task_$(printf '%04d' "$i")"
  [[ -f "$d/.complete" ]] && continue
  [[ -n "${BUSY[$i]:-}" ]] && continue
  [[ -n "${SKIP[$i]:-}" ]] && continue
  if [[ -d "$d" ]]; then TRIED=$(( TRIED + 1 )); else NEVER=$(( NEVER + 1 )); fi
  DEAD+=("$i")
done

echo "tasks=$NTASK to-submit=${#DEAD[@]} (tried-and-unfinished=$TRIED never-started=$NEVER) in-queue=${#BUSY[@]} excluded=${#SKIP[@]}"
(( ${#DEAD[@]} == 0 )) && { echo "nothing to recover"; exit 0; }
printf '  indices: %s\n' "$(printf '%s ' "${DEAD[@]}")"
(( LISTONLY )) && exit 0

# Park the crashed remains (the unreadable .root and the local.log carrying the
# stack trace) rather than letting array_dymc_dev2.sbatch's `rm -f` eat them:
# the log is the only record that the task died of THIS bug. Never-started
# chunks have no directory and are skipped by the guard below.
for i in "${DEAD[@]}"; do
  d="$OUTBASE/task_$(printf '%04d' "$i")"
  if compgen -G "$d/*.root" > /dev/null || [[ -f "$d/local.log" ]]; then
    mkdir -p "$d/failed_260906"
    for f in "$d"/*.root "$d/local.log"; do
      [[ -e "$f" ]] && mv -f "$f" "$d/failed_260906/" || true
    done
  fi
done

mkdir -p "$OUTBASE/logs"
# NOT `GROUPS`: that is a bash BUILT-IN array (the caller's group ids).
# `declare -A GROUPS` fails with "cannot convert indexed to associative
# array" and the writes then land in the builtin, so `--array` came out as
# the user's gids (100999, 169571, 1000000...). Found 2026-09-06 -- this
# path had never actually been exercised.
declare -A CHUNKGRP=()
for i in "${DEAD[@]}"; do
  g=$(( i / 1000 ))
  CHUNKGRP[$g]="${CHUNKGRP[$g]:-}${CHUNKGRP[$g]:+,}$(( i % 1000 ))"
done
for g in "${!CHUNKGRP[@]}"; do
  OFF=$(( g * 1000 ))
  cmd=( sbatch --job-name="dymc8p5M_a${g}" --partition=submit
        --array="${CHUNKGRP[$g]}%${MAXRUNNING}" --time="$WALLTIME" --mem=6G --cpus-per-task=1
        --output="$OUTBASE/logs/dymc8p5M_a${g}_%A_%a.out"
        --error="$OUTBASE/logs/dymc8p5M_a${g}_%A_%a.err"
        --export="ALL,CHUNKLIST=$CHUNKLIST,OUTBASE=$OUTBASE,IDXOFFSET=$OFF,EXTRA=$EXTRA,CMSSW_AREA=$FIXAREA"
        "$HERE/array_dymc_dev2.sbatch" )
  n=$(awk -F, '{print NF}' <<< "${CHUNKGRP[$g]}")
  echo "  recovery array $g: $n tasks, offset=$OFF, area=$FIXAREA"
  if (( DRY )); then printf '    '; printf '%q ' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
done
