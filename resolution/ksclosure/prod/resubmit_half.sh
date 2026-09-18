#!/bin/bash
# Resubmit the half-chunk tasks with no sentinel, and seal the finished chunks.
#   ./resubmit_half.sh [maxrunning]
set -euo pipefail
MAXRUN=${1:-250}
OUT=/ceph/submit/data/user/d/david_w/ZMass/cvh/ks_btojpsix_260918_fixed
KS=/work/submit/david_w/ZMass/calibration_studies/resolution/ksclosure
"$KS/prod/seal_halves.sh" >/dev/null
MISS=()
for i in $(seq 0 999); do
  I=$(printf '%04d' $((i / 2)))
  case $((i % 2)) in 0) H=a;; 1) H=b;; esac
  [ -f "$OUT/task_$I/.complete_$H" ] || MISS+=("$i")
done
if [ ${#MISS[@]} -eq 0 ]; then echo "nothing missing"; exit 0; fi
LIST=$(IFS=,; echo "${MISS[*]}")
echo "${#MISS[@]} half-tasks missing" 
sbatch --array="${LIST}%${MAXRUN}" \
  --output="$OUT/logs/slurm_%A_%a.out" --error="$OUT/logs/slurm_%A_%a.err" \
  --export=ALL,CONFIG=$KS/runCvhKs.py,HALFDIR=$OUT/chunks_half,OUTDIR=$OUT,CMSSW_AREA=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2,CMSRUN_ARGS="scalarPot3DInitFile=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt nEvents=-1" \
  $KS/prod/array_ks_half.sbatch
