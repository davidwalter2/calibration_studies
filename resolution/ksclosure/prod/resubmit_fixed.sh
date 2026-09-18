#!/bin/bash
# Resubmit the chunks of the FIXED-code production that have no output.
#   ./resubmit_fixed.sh [maxrunning]
set -euo pipefail
MAXRUN=${1:-150}
OUT=/ceph/submit/data/user/d/david_w/ZMass/cvh/ks_btojpsix_260918_fixed
KS=/work/submit/david_w/ZMass/calibration_studies/resolution/ksclosure
MISS=()
for i in $(seq 0 499); do
  I=$(printf '%04d' $i)
  [ -f "$OUT/task_$I/.complete" ] || MISS+=("$i")
done
if [ ${#MISS[@]} -eq 0 ]; then echo "nothing missing"; exit 0; fi
LIST=$(IFS=,; echo "${MISS[*]}")
echo "${#MISS[@]} missing: $LIST" | cut -c1-300
sbatch --array="${LIST}%${MAXRUN}" \
  --output="$OUT/logs/slurm_%A_%a.out" --error="$OUT/logs/slurm_%A_%a.err" \
  --export=ALL,CONFIG=$KS/runCvhKs.py,CHUNKDIR=$OUT/chunks,OUTDIR=$OUT,CMSSW_AREA=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2,CMSRUN_ARGS="scalarPot3DInitFile=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt nEvents=-1" \
  $KS/prod/array_ks.sbatch
