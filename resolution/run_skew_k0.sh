#!/bin/bash
# per-charge skew closure on the Kokoulin-off current-fit caches.
# The ionization charge factor is no longer a flag: cf_skew_closure reads it
# off the cache (`ioni_charge_signed`, written by cf_track_resolution since
# 2026-09-03) and supplies q itself on older, unsigned caches.  The tags keep
# the `qsign` suffix so the output filenames match the 2026-09-03 tables.
set -uo pipefail
cd /work/submit/david_w/ZMass/calibration_studies/resolution
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CVH_IONI_KOKOULIN=0
mkdir -p runs/skew260903
for tag in "$@"; do c=runs/cf_trackres_$tag.npz; [ -s "$c" ] || { echo "[skip] $tag"; continue; }
  for q in 0 1 -1; do sfx=$([ $q = 0 ] && echo qsign || ([ $q = 1 ] && echo qsign_pos || echo qsign_neg))
    python3 cf_skew_closure.py --cache "$c" --charge $q --tag "${tag}_$sfx" --label "$tag q=$q signed" > runs/skew260903/${tag}_$sfx.log 2>&1 && echo "[ok] $tag $sfx $(date +%H:%M)" || echo "[FAIL] $tag $sfx"; done; done
