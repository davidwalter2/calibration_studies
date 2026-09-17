#!/bin/bash
# Slurm array: Geant4 K_S -> pi+ pi- truth dump, one task per input chunk.
# FWLite only (no fit), so it is cheap; it runs on the same chunking as the
# CVH array so the two join chunk by chunk.
#SBATCH --job-name=kstruth
#SBATCH --partition=submit
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=4:00:00
set -euo pipefail
: "${CHUNKDIR:?}"; : "${OUTDIR:?}"; : "${SLURM_ARRAY_TASK_ID:?}"
IDX=$(printf '%04d' "$SLURM_ARRAY_TASK_ID")
CHUNK="$CHUNKDIR/chunk_$IDX.txt"
OUTNPZ="$OUTDIR/truth_$IDX.npz"
[[ -s "$OUTNPZ" ]] && { echo "already done"; exit 0; }
exec /work/submit/david_w/ZMass/calibration_studies/zchannel/fwlite.sh \
  /work/submit/david_w/ZMass/calibration_studies/resolution/ksclosure/ks_truth_dump.py \
  --out="$OUTNPZ" --filelist="$CHUNK"
