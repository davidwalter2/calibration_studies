#!/bin/bash
# One array task = one CHUNK of one UL16 DY MiniAODv2 file -> one CVH
# two-track (Z -> mumu) refit output directory.
#
# WHY CHUNKS AND NOT WHOLE FILES: these MiniAOD files hold ~82k events each
# (max 88k), which at ~0.31 s CPU/event is ~7 h of Geant4e in a single task.
# Each file is therefore split into event ranges via the driver's
# skipEvents/nEvents, giving ~2 h tasks that fit comfortably inside the wall
# limit and fail in small pieces.
#
# WHY THE INPUT READABILITY IS ASSERTED: the submit login nodes' CephFS client
# intermittently returns EPERM for the whole /ceph/submit tree (submit82 is
# evicted right now). cmsRun with skipBadFiles=True then SILENTLY SKIPS the
# input and writes a valid, EMPTY output. Compute nodes see ceph fine, but a
# node can be evicted mid-flight, so the check stays.
#
# env (exported by submit_dymc8p5M.sh):
#   CHUNKLIST  absolute path to the chunk list, one "<path> <skip> <nev>" per line
#   OUTBASE    absolute output base directory
#   IDXOFFSET  added to SLURM_ARRAY_TASK_ID (MaxArraySize is 1001)
#   EXTRA      extra cmsRun VarParsing args (the physics configuration)
#SBATCH --job-name=dymc8p5M
#SBATCH --partition=submit
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=12:00:00
set -uo pipefail
: "${CHUNKLIST:?}" ; : "${OUTBASE:?}"
export CMSSW_AREA=${CMSSW_AREA:-/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev}
CFG=${CFG:-$CMSSW_AREA/src/Analysis/HitAnalyzer/test/runCvhDimuonMiniAOD.py}
RUN_ONE=/work/submit/david_w/ZMass/calibration_studies/slurm/run_one.sh

IDX=$(( SLURM_ARRAY_TASK_ID + ${IDXOFFSET:-0} ))
OUTDIR="$OUTBASE/task_$(printf '%04d' "$IDX")"

# Resume on the sentinel ONLY. cmsRun creates its output at START, so a killed
# task leaves a non-empty but TRUNCATED .root that an `-s` test would accept
# forever.
if [[ -f "$OUTDIR/.complete" ]]; then echo "[skip] task $IDX already complete"; exit 0; fi

LINE=$(sed -n "$((IDX + 1))p" "$CHUNKLIST")
[[ -n "$LINE" ]] || { echo "[FATAL] empty chunk line $((IDX+1)) of $CHUNKLIST" >&2; exit 2; }
read -r INPUT SKIP NEV _rest <<< "$LINE"
[[ -n "$INPUT" && -n "$SKIP" && -n "$NEV" ]] || { echo "[FATAL] malformed chunk line: $LINE" >&2; exit 2; }
[[ -r "$INPUT" ]] || { echo "[FATAL] input not readable on $(hostname): $INPUT" >&2; exit 3; }

mkdir -p "$OUTDIR"
rm -f "$OUTDIR"/*.root "$OUTDIR/.complete"

echo ">>> host=$(hostname) task=$IDX skip=$SKIP nev=$NEV $(date)"
echo ">>> input=$INPUT"
# shellcheck disable=SC2086
"$RUN_ONE" "$CFG" "$INPUT" "$OUTDIR" skipEvents="$SKIP" nEvents="$NEV" ${EXTRA:-} \
    > "$OUTDIR/local.log" 2>&1
rc=$?
if [[ $rc -ne 0 ]]; then echo "[FATAL] cmsRun rc=$rc (see $OUTDIR/local.log)" >&2; exit $rc; fi

shopt -s nullglob
out=("$OUTDIR"/*.root)
[[ ${#out[@]} -ge 1 && -s "${out[0]}" ]] || { echo "[FATAL] no/empty output in $OUTDIR" >&2; exit 4; }
# Guard against the silent-empty-input failure: the maker always prints a fit
# summary, and a run that read nothing prints attempted=0.
if grep -q "fit summary  attempted=0 " "$OUTDIR/local.log"; then
  echo "[FATAL] attempted=0 candidates -- input was skipped (ceph EPERM?)" >&2; exit 5
fi
touch "$OUTDIR/.complete"
echo "[done] task $IDX $(date)"
