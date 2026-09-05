#!/bin/bash
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution; cd "$RES"
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for tag in mugun_ul16_260830 mugun_ul16_260830_m0 mugun_lowpt_260830 mugun_lowpt_260830_m0; do
  c=runs/cf_trackres_$tag.npz; [ -s "$c" ] || { echo "[skip] $tag (no cache yet)"; continue; }
  [ -f ~/public_html/cvh/$(date +%y%m%d)_skew/skew_closure_${tag}_qsign.txt ] && { echo "[skip] $tag done"; continue; }
  for q in 0 1 -1; do
    sfx=$([ $q = 0 ] && echo qsign || ([ $q = 1 ] && echo qsign_pos || echo qsign_neg))
    python3 cf_skew_closure.py --cache "$c" --charge $q --tag "${tag}_$sfx" --label "$tag q=$q signed" > runs/skew260902/${tag}_$sfx.log 2>&1 && echo "[ok] $tag $sfx" || echo "[FAIL] $tag $sfx"
  done
done
