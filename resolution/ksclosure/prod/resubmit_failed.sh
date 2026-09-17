#!/bin/bash
# Resubmit the chunks that have no output.
#   ./resubmit_failed.sh cvh|truth [maxrunning]
# A task is redone when its sentinel is missing: `task_NNNN/.complete` for the
# refit, `truth/truth_NNNN.npz` for the truth dump.  Both runners are
# idempotent (they exit early on the sentinel), so a resubmission of a
# still-running task is harmless.
set -euo pipefail
WHAT=${1:?cvh|truth}
MAXRUN=${2:-100}
OUT=/ceph/submit/data/user/d/david_w/ZMass/cvh/ks_btojpsix_260917_ideal
KS=/work/submit/david_w/ZMass/calibration_studies/resolution/ksclosure
MISS=()
for i in $(seq 0 499); do
  I=$(printf '%04d' $i)
  if [ "$WHAT" = cvh ]; then
    [ -f "$OUT/task_$I/.complete" ] || MISS+=("$i")
  else
    [ -s "$OUT/truth/truth_$I.npz" ] || MISS+=("$i")
  fi
done
if [ ${#MISS[@]} -eq 0 ]; then echo "nothing missing"; exit 0; fi
LIST=$(IFS=,; echo "${MISS[*]}")
echo "${#MISS[@]} missing: $LIST" | cut -c1-300
if [ "$WHAT" = cvh ]; then
  sbatch --array="${LIST}%${MAXRUN}" \
    --output="$OUT/logs/slurm_%A_%a.out" --error="$OUT/logs/slurm_%A_%a.err" \
    --export=ALL,CONFIG=$KS/runCvhKs.py,CHUNKDIR=$OUT/chunks,OUTDIR=$OUT,CMSSW_AREA=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2,CMSRUN_ARGS="scalarPot3DInitFile=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt nEvents=-1" \
    $KS/prod/array_ks.sbatch
else
  sbatch --array="${LIST}%${MAXRUN}" \
    --output="$OUT/logs_truth/slurm_%A_%a.out" --error="$OUT/logs_truth/slurm_%A_%a.err" \
    --export=ALL,CHUNKDIR=$OUT/chunks,OUTDIR=$OUT/truth \
    $KS/prod/array_truth.sbatch
fi
