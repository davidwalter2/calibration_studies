#!/bin/bash
set -euo pipefail
OUT=/ceph/submit/data/user/d/david_w/ZMass/cvh/ks_btojpsix_260917_ideal
KS=/work/submit/david_w/ZMass/calibration_studies/resolution/ksclosure
FIRST=${1:-0}; LAST=${2:-499}; MAXRUN=${3:-150}
sbatch --array=${FIRST}-${LAST}%${MAXRUN} \
  --output="$OUT/logs_truth/slurm_%A_%a.out" --error="$OUT/logs_truth/slurm_%A_%a.err" \
  --export=ALL,CHUNKDIR=$OUT/chunks,OUTDIR=$OUT/truth \
  $KS/prod/array_truth.sbatch
