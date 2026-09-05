#!/bin/bash
# Launch one batch file (lines "tag|args") with at most NPAR concurrent
# single-threaded cmsRun. Staggered starts: a thundering herd on CVMFS/NSS
# wedges every job in the user lookup (see run_local_trackres.sh).
set -uo pipefail
BATCH=$1; OUT=$2; NPAR=${3:-8}
P=/work/submit/david_w/ZMass/calibration_studies/production/profiling
run_one() {
  local line=$1 idx=$2
  local tag=${line%%|*} args=${line#*|}
  sleep $(( idx * 4 ))
  # shellcheck disable=SC2086
  "$P/run_profile.sh" "$tag" "$OUT" $args
  echo "[done] $tag"
}
export -f run_one; export P OUT
i=0
while IFS= read -r line; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  echo "$i|$line"; i=$((i+1))
done < "$BATCH" | xargs -P "$NPAR" -I{} bash -c 'l="{}"; run_one "${l#*|}" "${l%%|*}"'
echo "BATCH COMPLETE"
