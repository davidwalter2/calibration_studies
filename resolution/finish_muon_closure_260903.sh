#!/bin/bash
# Track-closure extractions restarted 2026-09-03 with the offline Kokoulin term
# OFF (>7x cheaper, ~1e-3 effect, consistent with the August caches which
# predate the flag). Legacy-Q arms (_m0) use the exact `var` normalisation;
# CGF arms need `--ioni-norm raw` (var is broken for CgfQoPMode=1). Serial.
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution; cd "$RES"
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
export CVH_IONI_KOKOULIN=0
CEPH=/ceph/submit/data/user/d/david_w/ZMass/cvh; LOG=$RES/runs/muclosure260830; mkdir -p "$LOG"
ex() { [ -s "runs/cf_trackres_$2.npz" ] && { echo "=== skip $2 (exists)"; return; }
  echo "=== extract $2 ($(date +%H:%M:%S)) ==="
  EXTRA_ARGS="$3" ./extract_parallel.sh "$CEPH/resolution_trackres_$1" "runs/cf_trackres_$2.npz" 160 > "$LOG/ext_$2.log" 2>&1 \
    && echo "    [ok] $2 ($(date +%H:%M:%S))" || echo "    [FAIL] $2"; }
ex mugun_ul16_260830_m0  mugun_ul16_260830_m0_k0  ""
ex mugun_ul16_260830     mugun_ul16_260830_raw_k0 "--ioni-norm raw"
ex mugun_lowpt_260830_m0 mugun_lowpt_260830_m0_k0 ""
ex mugun_lowpt_260830    mugun_lowpt_260830_raw_k0 "--ioni-norm raw"
echo "=== kms_solve ($(date +%H:%M:%S)) ==="
python3 kms_solve_260903.py > "$LOG/kms_solve_260903.txt" 2>&1 && cat "$LOG/kms_solve_260903.txt"
echo "=== ALL DONE ($(date +%H:%M:%S)) ==="
