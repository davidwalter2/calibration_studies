#!/bin/bash
# Sample-wide integrity check: the per-task CVH fit-failure rate.
# A file whose EVENT payload were silently corrupted the way the four bad
# inputs' provenance was would show up as an outlier here -- the split-99
# tasks did, at 17-18 % against a 0.0-0.1 % baseline.
set -uo pipefail
BASE=${1:-/ceph/submit/data/user/d/david_w/ZMass/cvh/jpsimc_20M_260906_v2}
OUT=${2:-/ceph/submit/data/user/d/david_w/ZMass/scratch_repack_260906/fitfail_per_task.txt}
: > "$OUT"
for d in "$BASE"/task_*/; do
  t=$(basename "$d")
  L=$d/local.log
  [[ -s $L ]] || { echo "$t NOLOG"; continue; }
  awk -v t="$t" '/fit summary/ {
        for (i=1;i<=NF;i++) {
          if ($i ~ /^attempted=/)  {split($i,a,"="); A+=a[2]}
          if ($i ~ /^succeeded=/)  {split($i,a,"="); S+=a[2]}
        } }
      END { if (A>0) printf "%s %d %d %.4f\n", t, A, S, 100.0*(A-S)/A;
            else     printf "%s 0 0 -1\n", t }' "$L"
done >> "$OUT"
echo "FITFAIL DONE $(date -Is)  $(wc -l < "$OUT") tasks"
sort -k4 -gr "$OUT" | head -25
