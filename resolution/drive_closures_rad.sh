#!/bin/bash
# Waits for each _260903x extraction to land and runs its closures as soon as
# it does, so the closures overlap the remaining extractions; then the k_MS
# ladder. Skips a cache whose closure output already exists.
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution; cd "$RES"
LOG=$RES/runs/rad260903x
DAY=$(date +%y%m%d)
for spec in "mugun_lowpt_260903x_m0_k0 skew" "mugun_ul16_260903x_m0_k0 skew" \
            "mugun_lowpt_260903x_k0 skew" "mugun_ul16_260903x_k0 skew" \
            "mugun_lowpt_noms_260903x_m0_k0 noskew" \
            "mugun_lowpt_nomsrad_260903x_m0_k0 noskew"; do
  set -- $spec; t=$1; sk=$2
  until grep -q "\[ok\] $t " runs/chain_rad_260903x.log 2>/dev/null; do
    grep -q "\[FAIL\] $t" runs/chain_rad_260903x.log 2>/dev/null && { echo "[skip] $t extraction FAILED"; continue 2; }
    sleep 60
  done
  if ls "$HOME/public_html/cvh/"*_trackres/trackres_closure_${t}_krad1.txt >/dev/null 2>&1; then
    echo "[skip] $t (closure already done)"; continue
  fi
  echo "=== closures $t ($(date +%H:%M:%S))"
  ./run_closure_one.sh "$t" "$sk" >> "$LOG/clo_$t.log" 2>&1
  tail -1 "$LOG/clo_$t.log"
done
echo "=== all closures done, starting kms ladder ($(date +%H:%M:%S))"
./run_kms_260903x.sh
echo "=== DRIVER DONE ($(date +%H:%M:%S))"
