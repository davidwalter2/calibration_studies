#!/bin/bash
# btojpsix pairs -> jpsigun pairs -> candidate (mass) skew closures -> the fits.
# Sequential by design: each stage is 32-160 processes.
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution; cd "$RES"
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CVH_IONI_KOKOULIN=0
CEPH=/ceph/submit/data/user/d/david_w/ZMass/cvh
LOG=$RES/runs/rad260903x; mkdir -p "$LOG"

# 1. wait for the btojpsix pairs cache already running
until [ -s runs/cf_masspairs_btojpsix_v3_260903x_m0.npz ]; do
  pgrep -u "$USER" -f run_pairs_tt_shards >/dev/null || { echo "btojpsix pairs driver gone"; break; }
  sleep 60
done
echo "=== btojpsix pairs: $(ls -la runs/cf_masspairs_btojpsix_v3_260903x_m0.npz 2>/dev/null | awk '{print $5}') ($(date +%H:%M:%S))"

# 2. jpsigun pairs (160 tasks)
if [ ! -s runs/cf_masspairs_jpsigun_ul16_260903x_m0.npz ]; then
  echo "=== jpsigun pairs ($(date +%H:%M:%S))"
  PROD=$CEPH/resolution_trackres_jpsigun_ul16_260903x_m0 \
  OUT=$RES/runs/cf_masspairs_jpsigun_ul16_260903x_m0.npz \
  NPAR=64 KOK=0 ./run_pairs_tt_shards.sh > "$LOG/pairs_jpsigun_ul16.log" 2>&1
  echo "    -> $(ls -la runs/cf_masspairs_jpsigun_ul16_260903x_m0.npz 2>/dev/null | awk '{print $5}')"
fi

# 3. candidate-level odd-moment closures, both arms
echo "=== mass skew ($(date +%H:%M:%S))"
for t in jpsigun_ul16_260903x_m0 btojpsix_v3_260903x_m0; do
  c=runs/cf_masspairs_$t.npz; [ -s "$c" ] || { echo "[skip] $t"; continue; }
  for k in 1 0; do
    python3 cf_skew_closure.py --cache "$c" --mass --krad "$k" \
        --tag "masspairs_${t}_krad$k" --label "$t candidates krad=$k" \
        > "$LOG/mass_${t}_krad$k.log" 2>&1 \
      && echo "[ok] mass $t krad=$k $(date +%H:%M)" || echo "[FAIL] mass $t krad=$k" &
  done
done
wait

# 4. the fits
echo "=== mass fits ($(date +%H:%M:%S))"
./chain_masslikfit_rad_260903x.sh
python3 masslikfit_summary.py 'runs/masslikfit_*260903x*.npz' \
    -o "$HOME/public_html/cvh/$(date +%y%m%d)_masslikfit" > "$LOG/masslikfit_summary.txt" 2>&1
echo "=== PAIRS/MASS DRIVER DONE ($(date +%H:%M:%S))"
