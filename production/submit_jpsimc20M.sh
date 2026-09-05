#!/bin/bash
# Submit the 20M-event UL16 J/psi MC CVH two-track production.
#
# WHAT THIS RUNS -- the "rung-B" real-geometry configuration:
#   * two-track J/psi fit, NO vertex constraint, NO mass constraint
#   * aligned MC geometry from the GT + the full 3D TOSCA grid field (160812)
#   * global-correction gradients with the LOW-RANK FACTORED Hessian
#     (H = B^T B) rather than the packed one -- ~9x smaller at this mode count
#   * doRes: the resolution families are registered and the in-maker mass-CF
#     exponents (cfmass_*) are written on the 64-point tau grid
#   * exportStepRecords=False: the RAW per-step records are 430 kB/candidate
#     (~5x the whole rest of the event) and their only consumer was the offline
#     exponent extractor, which the in-maker export replaces. Proven inert:
#     every other branch is bit-identical with and without it (see STATE.md).
#
# ARRAY SIZING: Slurm MaxArraySize here is 1001, and the production is ~1640
# tasks, so it goes out as several arrays with an index offset into the shared
# chunk list. Task index == chunk-list line, across all arrays -- which is what
# makes resume.sh able to re-drive any subset by index.
#
# usage: ./submit_jpsimc20M.sh [--dry-run] [--max-running N] [--only "0 1"]
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

TAG=jpsimc_20M_260905
CHUNKLIST=$HERE/chunks_${TAG}.txt
OUTBASE=/ceph/submit/data/user/d/david_w/ZMass/cvh/$TAG
CMSSW_AREA=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
MAXARRAY=1000          # tasks per array job (MaxArraySize is 1001 -> 0..1000)
MAXRUNNING=200         # concurrent tasks PER ARRAY
NAME=jpsimc20M
DRY=0
ONLY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)     DRY=1; shift;;
    --max-running) MAXRUNNING=$2; shift 2;;
    --only)        ONLY=$2; shift 2;;      # space-separated array numbers
    --chunklist)   CHUNKLIST=$2; shift 2;;
    --outbase)     OUTBASE=$2; shift 2;;
    -h|--help)     sed -n '2,25p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

[[ -f "$CHUNKLIST" ]] || { echo "chunk list not found: $CHUNKLIST" >&2; exit 1; }
[[ -d "$CMSSW_AREA/src" ]] || { echo "CMSSW area invalid: $CMSSW_AREA" >&2; exit 1; }
[[ -r "$INIT" ]] || { echo "scalar-potential init file missing: $INIT" >&2; exit 1; }

# The physics configuration. Every option is deliberate; see STATE.md for the
# table of what each one is and why.
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

NTASK=$(grep -cvE '^\s*(#|$)' "$CHUNKLIST")
NARRAY=$(( (NTASK + MAXARRAY - 1) / MAXARRAY ))
HASH=$(cd "$CMSSW_AREA/src" && git rev-parse --short HEAD)

mkdir -p "$OUTBASE/logs"
# Record what produced this output next to it -- the cmsRun provenance has the
# PSet, but not the chunking or the code revision the driver came from.
{ echo "tag       : $TAG"
  echo "submitted : $(date -Is) by $USER on $(hostname)"
  echo "cmssw     : $CMSSW_AREA @ $HASH"
  echo "chunklist : $CHUNKLIST ($NTASK tasks, $NARRAY arrays)"
  echo "extra     : $EXTRA"; } > "$OUTBASE/PROVENANCE.txt"

echo "tasks=$NTASK  arrays=$NARRAY  max-running=$MAXRUNNING/array  cmssw=$HASH"
echo "outbase=$OUTBASE"

for (( i=0; i<NARRAY; i++ )); do
  [[ -n "$ONLY" ]] && ! grep -qw "$i" <<< "$ONLY" && continue
  OFF=$(( i * MAXARRAY ))
  LAST=$(( NTASK - OFF - 1 )); (( LAST > MAXARRAY - 1 )) && LAST=$(( MAXARRAY - 1 ))
  SPEC="0-${LAST}%${MAXRUNNING}"
  cmd=( sbatch --job-name="${NAME}_a${i}" --partition=submit
        --array="$SPEC" --time=12:00:00 --mem=6G --cpus-per-task=1
        --output="$OUTBASE/logs/${NAME}_a${i}_%A_%a.out"
        --error="$OUTBASE/logs/${NAME}_a${i}_%A_%a.err"
        --export="ALL,CHUNKLIST=$CHUNKLIST,OUTBASE=$OUTBASE,IDXOFFSET=$OFF,EXTRA=$EXTRA,CMSSW_AREA=$CMSSW_AREA"
        "$HERE/array_jpsimc.sbatch" )
  echo "  array $i: offset=$OFF spec=$SPEC"
  if (( DRY )); then printf '    '; printf '%q ' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
done
