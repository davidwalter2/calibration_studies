#!/bin/bash
# Submit the 8.5M-event UL16 DY MiniAOD CVH two-track (Z -> mumu) production.
#
# WHAT THIS RUNS -- the Z leg of the CVH full-scale feasibility test, the
# mirror of submit_jpsimc20M.sh:
#   * two-track dimuon fit off MiniAOD (slimmedMuons -> TrackProducerFromPatMuons
#     -> diMuonTrackVertexCandidates in a 60-120 GeV window -> the same
#     ResidualGlobalCorrectionMakerTwoTrackG4e), NO vertex constraint, NO mass
#     constraint
#   * aligned MC geometry from the GT + THE FIELD THE SIM PROPAGATED THROUGH,
#     i.e. the unlabelled VolumeBasedMagneticField 160812 with
#     useParametrizedTrackerField=True -> OAE_1103l_071212 in the tracker
#   * global-correction gradients with the LOW-RANK FACTORED Hessian (H = B^T B)
#   * doRes: the resolution families are registered and the in-maker mass-CF
#     exponents (cfmass_*) are written on the 64-point tau grid. The export is
#     gated on doRes AND fillGradsFactored -- each is nearly free alone, the
#     pair is the whole 0.63 s/candidate.
#   * exportStepRecords=False: 317 vs 28 kB/candidate for records whose only
#     consumer the in-maker export replaces
#   * CgfQoPMode=0 pinned: mode 1 is 8.5 s/candidate here (profiling/NOTES.md)
#
# ARRAY SIZING: Slurm MaxArraySize here is 1001, so a production of more than
# 1000 tasks goes out as several arrays with an index offset into the shared
# chunk list. Task index == chunk-list line, across all arrays -- which is what
# makes resume_dy.sh able to re-drive any subset by index.
#
# WALL TIME: the default is 12 h, matching the J/psi arrays, against a largest
# chunk of 2.8 h -- deliberately generous. It is a `--time` OPTION rather than
# a constant because the over-request costs THROUGHPUT: with the cluster full
# and every job of mine at the same priority, slurm's backfill scheduler can
# only slot a job into a gap it declares it will fit in, and a 12 h declaration
# rarely fits. `--time 4:00:00` still leaves ~40 % headroom on the largest
# chunk and backfills far more readily.
#
# usage: ./submit_dymc8p5M.sh [--dry-run] [--max-running N] [--time T] [--only "0 1"]
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# shared with resume_dy.sh so the two can never drift apart
source "$HERE/config_dymc8p5M.sh"
CHUNKLIST=$HERE/chunks_${TAG}.txt
DRY=0
ONLY=""
WALLTIME=12:00:00

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)     DRY=1; shift;;
    --max-running) MAXRUNNING=$2; shift 2;;
    --time)        WALLTIME=$2; shift 2;;
    --only)        ONLY=$2; shift 2;;      # space-separated array numbers
    --chunklist)   CHUNKLIST=$2; shift 2;;
    --outbase)     OUTBASE=$2; shift 2;;
    -h|--help)     sed -n '2,30p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

[[ -f "$CHUNKLIST" ]] || { echo "chunk list not found: $CHUNKLIST" >&2; exit 1; }
[[ -d "$CMSSW_AREA/src" ]] || { echo "CMSSW area invalid: $CMSSW_AREA" >&2; exit 1; }
[[ -r "$INIT" ]] || { echo "scalar-potential init file missing: $INIT" >&2; exit 1; }

NTASK=$(grep -cvE '^\s*(#|$)' "$CHUNKLIST")
NARRAY=$(( (NTASK + MAXARRAY - 1) / MAXARRAY ))
HASH=$(cd "$CMSSW_AREA/src" && git rev-parse --short HEAD)

mkdir -p "$OUTBASE/logs"
# Record what produced this output next to it -- the cmsRun provenance has the
# PSet, but not the chunking or the code revision the driver came from.
{ echo "tag       : $TAG"
  echo "submitted : $(date -Is) by $USER on $(hostname)"
  echo "cmssw     : $CMSSW_AREA @ $HASH"
  echo "driver    : $CMSSW_AREA/src/Analysis/HitAnalyzer/test/runCvhDimuonMiniAOD.py"
  echo "chunklist : $CHUNKLIST ($NTASK tasks, $NARRAY arrays, walltime $WALLTIME)"
  echo "extra     : $EXTRA"; } > "$OUTBASE/PROVENANCE.txt"

echo "tasks=$NTASK  arrays=$NARRAY  max-running=$MAXRUNNING/array  time=$WALLTIME  cmssw=$HASH"
echo "outbase=$OUTBASE"

for (( i=0; i<NARRAY; i++ )); do
  [[ -n "$ONLY" ]] && ! grep -qw "$i" <<< "$ONLY" && continue
  OFF=$(( i * MAXARRAY ))
  LAST=$(( NTASK - OFF - 1 )); (( LAST > MAXARRAY - 1 )) && LAST=$(( MAXARRAY - 1 ))
  SPEC="0-${LAST}%${MAXRUNNING}"
  cmd=( sbatch --job-name="${NAME}_a${i}" --partition=submit
        --array="$SPEC" --time="$WALLTIME" --mem=6G --cpus-per-task=1
        --output="$OUTBASE/logs/${NAME}_a${i}_%A_%a.out"
        --error="$OUTBASE/logs/${NAME}_a${i}_%A_%a.err"
        --export="ALL,CHUNKLIST=$CHUNKLIST,OUTBASE=$OUTBASE,IDXOFFSET=$OFF,EXTRA=$EXTRA,CMSSW_AREA=$CMSSW_AREA"
        "$HERE/array_dymc.sbatch" )
  echo "  array $i: offset=$OFF spec=$SPEC"
  if (( DRY )); then printf '    '; printf '%q ' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
done
