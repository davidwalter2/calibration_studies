#!/bin/bash
# PER-LEG |seed -> final dq/p| for both v2 productions, aligned to the pairs
# caches.  See `resolution/oddmoment/aux_seed.py` for what it extracts and why
# only the ABSOLUTE step is usable.
#
# *** NEEDS /ceph ***: run it from submit50 / submit51.  A sandboxed shell, and
# submit82 (cephx eviction), get permission denied.
set -euo pipefail
FS=/work/submit/david_w/ZMass/calibration_studies/fullscale
OM=/work/submit/david_w/ZMass/calibration_studies/resolution/oddmoment
CEPH=/ceph/submit/data/user/d/david_w/ZMass/cvh
OUT=$FS/runs
NPROC=${NPROC:-32}
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
cd "$OM"
for spec in \
    "dyv2:$CEPH/dymc_8p5M_260906_v2:$OUT/zpairs_dyv2_full.npz" \
    "jpsiv2:$CEPH/jpsimc_20M_260906_v2:$OUT/jpairs_v2_n600.npz" ; do
  tag=${spec%%:*}; rest=${spec#*:}; dir=${rest%%:*}; cache=${rest#*:}
  echo "=== $tag  $(date +%H:%M:%S)  $dir"
  python3 -u aux_seed.py --files "$dir" --cache "$cache" --nproc "$NPROC" \
      --out "$OUT/auxseed_$tag.npz" 2>&1 | tee "$FS/logs/auxseed_$tag.log"
done
echo "=== done $(date +%H:%M:%S)"
