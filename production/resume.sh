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
TAG=jpsimc_20M_260905
OUTBASE=/ceph/submit/data/user/d/david_w/ZMass/cvh/$TAG
CHUNKLIST=$HERE/chunks_${TAG}.txt
CMSSW_AREA=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
MAXRUNNING=200
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

EXTRA="numberOfThreads=1 \
 doRes=True exportCfExponents=True exportStepRecords=False \
 fillJac=True fillGrads=False fillGradsFactored=True \
 fitFromGenParms=False \
 trackSrc=ALCARECOTkAlJpsiMuMu useLegacyPairLoop=True \
 doTrigger=True applyHltFilter=False doSimHits=False \
 useIdealGeometry=False useOpera3D=True globalTag=106X_mcRun2_asymptotic_v17 \
 doVtxConstraint=False doMassConstraint=False \
 CgfQoPMode=0 \
 propagationPtotLimit=0.2 maxMomentumStepFactor=2.0 stepBacktracking=True \
 scalarPot3DInitFile=$INIT"

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
