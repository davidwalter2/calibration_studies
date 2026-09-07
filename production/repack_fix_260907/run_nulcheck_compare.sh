#!/bin/bash
# Compare the seven side-tree outputs (repacked, NUL-free inputs) against the
# ones already in the production (original, NUL-carrying inputs), candidate by
# candidate on a common (run,lumi,event) order.  Bit-identical => the NULs sat
# in psets cmsRun never parses and the existing outputs stay.
set -uo pipefail
B=/ceph/submit/data/user/d/david_w/ZMass/cvh/jpsimc_20M_260906_v2
S=/ceph/submit/data/user/d/david_w/ZMass/cvh/jpsimc_20M_260906_v2_nulcheck
OUT=/ceph/submit/data/user/d/david_w/ZMass/scratch_repack_260906/nulcheck_compare.txt
CMP=/work/submit/david_w/ZMass/calibration_studies/production/repack_fix_260907/compare_task_outputs.py
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
: > "$OUT"
RC=0
for i in 1342 1343 1344 1345 1374 1375 1376; do
  t=task_$(printf '%04d' "$i")
  echo "################ $t  $(date -Is)" >> "$OUT"
  python3 "$CMP" "$B/$t" "$S/$t" >> "$OUT" 2>&1 || RC=1
done
echo "COMPARE DONE rc=$RC $(date -Is)" >> "$OUT"
grep -E '^RESULT|^DIFFER|^################' "$OUT"
