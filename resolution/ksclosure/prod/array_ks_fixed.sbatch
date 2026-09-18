#!/bin/bash
# Slurm array: one task per 252-file chunk of the B -> J/psi + X v3 MC.
# Runs the CVH two-track K_S -> pi pi refit (vertex constraint ON, beam-spot
# constraint OFF, ideal geometry, default field).
#
# env (via --export): CONFIG, CHUNKDIR, OUTDIR, CMSSW_AREA, CMSRUN_ARGS
#SBATCH --job-name=cvhks
#SBATCH --partition=submit
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH --time=2:00:00
set -euo pipefail
: "${CONFIG:?}"; : "${CHUNKDIR:?}"; : "${OUTDIR:?}"; : "${CMSSW_AREA:?}"
: "${SLURM_ARRAY_TASK_ID:?}"

IDX=$(printf '%04d' "$SLURM_ARRAY_TASK_ID")
CHUNK="$CHUNKDIR/chunk_$IDX.txt"
[[ -s "$CHUNK" ]] || { echo "missing chunk $CHUNK" >&2; exit 1; }
TASK_DIR="$OUTDIR/task_$IDX"
if [[ -f "$TASK_DIR/.done" ]]; then echo "already done"; exit 0; fi
mkdir -p "$TASK_DIR"
rm -f "$TASK_DIR"/globalcor_ks_*.root

cd "$CMSSW_AREA/src"
source /cvmfs/cms.cern.ch/cmsset_default.sh
eval $(scramv1 runtime -sh)
cd "$TASK_DIR"
echo ">>> $(date) host=$(hostname) chunk=$CHUNK"
EXTRA=()
if [[ -n "${CMSRUN_ARGS:-}" ]]; then read -r -a EXTRA <<< "$CMSRUN_ARGS"; fi
cmsRun "$CONFIG" inputFileList="$CHUNK" "${EXTRA[@]}"
printf "streams=1\n" > "$TASK_DIR/.complete"
touch "$TASK_DIR/.done"
echo ">>> $(date) finished"
