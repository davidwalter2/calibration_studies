#!/bin/bash
# Write `task_NNNN/.complete` for every chunk whose two halves are both there.
# Idempotent; covers the case where the two halves finish in the same instant
# and each sees the other's sentinel missing.
set -euo pipefail
OUT=${1:-/ceph/submit/data/user/d/david_w/ZMass/cvh/ks_btojpsix_260918_fixed}
n=0
for i in $(seq 0 499); do
  I=$(printf '%04d' $i); D="$OUT/task_$I"
  if [ -f "$D/.complete_a" ] && [ -f "$D/.complete_b" ] && [ ! -f "$D/.complete" ]; then
    printf "streams=1 halves=2\n" > "$D/.complete"; touch "$D/.done"; n=$((n+1))
  fi
done
echo "sealed $n chunks; complete $(ls $OUT/task_*/.complete 2>/dev/null | wc -l)/500"
