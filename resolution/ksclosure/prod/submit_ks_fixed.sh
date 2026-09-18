#!/bin/bash
# Submit the K_S CVH production array ON THE FIXED CODE (commit 3d4c926ff461).
# Same driver, same flags, same 500 chunk lists as ks_btojpsix_260917_ideal;
# only the CMSSW commit differs.
#   usage: ./submit_ks_fixed.sh [first] [last] [maxrunning]
set -euo pipefail
OUT=/ceph/submit/data/user/d/david_w/ZMass/cvh/ks_btojpsix_260918_fixed
KS=/work/submit/david_w/ZMass/calibration_studies/resolution/ksclosure
FIRST=${1:-0}; LAST=${2:-499}; MAXRUN=${3:-220}
mkdir -p "$OUT/logs"
sbatch --array=${FIRST}-${LAST}%${MAXRUN} \
  --output="$OUT/logs/slurm_%A_%a.out" --error="$OUT/logs/slurm_%A_%a.err" \
  --export=ALL,CONFIG=$KS/runCvhKs.py,CHUNKDIR=$OUT/chunks,OUTDIR=$OUT,CMSSW_AREA=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2,CMSRUN_ARGS="scalarPot3DInitFile=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt nEvents=-1" \
  $KS/prod/array_ks_fixed.sbatch
