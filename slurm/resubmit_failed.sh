#!/bin/bash
# Identify the FAILED / TIMEOUT / OUT_OF_MEMORY array tasks of a job and
# resubmit them. Reads the original filelist by index, writes a new
# filelist with just the failed entries, and re-submits via submit.sh.
#
# usage: ./resubmit_failed.sh <ARRAY_JOB_ID> <ORIG_FILELIST> <ORIG_OUTDIR> [extra submit.sh args ...]
#
# example:
#   ./resubmit_failed.sh 5163621 \
#     filelists/jpsi_2016_scratch.txt \
#     /ceph/submit/data/user/d/david_w/ZMass/cvh/jpsi_stage2_2016_6257c5ad8b9 \
#     --time 30:00:00 --mem 4G --name cvh_jpsi16_retry \
#     --cmsrun-args "fillJac=1 fillGrads=1 doMassConstraint=1 useIdealGeometry=0 nEvents=-1 goldenJson=/work/submit/david_w/WRemnants/wremnants-data/data/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"
set -euo pipefail

ARRAY_JID=${1:?ARRAY_JOB_ID required}
ORIG_FILELIST=${2:?ORIG_FILELIST required}
ORIG_OUTDIR=${3:?ORIG_OUTDIR required}
shift 3

[[ -f "$ORIG_FILELIST" ]] || { echo "filelist not found: $ORIG_FILELIST" >&2; exit 1; }

# Pull array indices that ended in any non-COMPLETED state.
FAILED_IDXS=$(sacct -j "$ARRAY_JID" -X --format=JobID,State -n 2>/dev/null \
              | awk '$2 != "COMPLETED" && $2 != "RUNNING" && $2 != "PENDING" {print $1}' \
              | awk -F_ '{print $2}' | sort -un)

if [[ -z "$FAILED_IDXS" ]]; then
  echo "no failed indices in array $ARRAY_JID"
  exit 0
fi

NFAIL=$(echo "$FAILED_IDXS" | wc -l)
echo "$NFAIL failed indices in $ARRAY_JID:"
echo "$FAILED_IDXS" | tr '\n' ' '; echo

# Build per-index retry filelist preserving the order of FAILED_IDXS.
RETRY_LIST="$ORIG_OUTDIR/_retry_filelist_${ARRAY_JID}.txt"
> "$RETRY_LIST"
while read -r idx; do
  # sed line N (1-based) for array index N (0-based)
  line=$((idx + 1))
  sed -n "${line}p" "$ORIG_FILELIST" >> "$RETRY_LIST"
done <<< "$FAILED_IDXS"

echo "wrote retry filelist: $RETRY_LIST"
wc -l "$RETRY_LIST"
echo
echo "submitting retry..."
SLURM_DIR=$(dirname "$(readlink -f "$0")")
"$SLURM_DIR/submit.sh" \
  --filelist "$RETRY_LIST" \
  --outdir "${ORIG_OUTDIR}_retry" \
  "$@"
