#!/bin/bash
# closures for ONE trackres cache, both krad arms, launched as soon as its
# extraction lands (so the closures overlap the remaining extractions).
# usage: ./run_closure_one.sh <cache tag> [skew|noskew]
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution; cd "$RES"
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CVH_IONI_KOKOULIN=0
LOG=$RES/runs/rad260903x; mkdir -p "$LOG"
t=$1; SK=${2:-skew}; c="runs/cf_trackres_$t.npz"
[ -s "$c" ] || { echo "no $c"; exit 1; }
for k in 1 0; do
  python3 cf_track_resolution.py --closure --cache "$c" --krad "$k" \
      --postfix "_${t}_krad$k" > "$LOG/trk_${t}_krad$k.log" 2>&1 \
    && echo "[ok] trackres $t krad=$k $(date +%H:%M)" || echo "[FAIL] trackres $t krad=$k" &
done
if [ "$SK" = skew ]; then
 for q in 1 -1; do for k in 1 0; do
  sfx=$([ "$q" = 1 ] && echo pos || echo neg)
  python3 cf_skew_closure.py --cache "$c" --charge "$q" --krad "$k" \
      --tag "${t}_${sfx}_krad$k" --label "$t q=$q krad=$k" \
      > "$LOG/skew_${t}_${sfx}_krad$k.log" 2>&1 \
    && echo "[ok] skew $t $sfx krad=$k $(date +%H:%M)" || echo "[FAIL] skew $t $sfx krad=$k" &
 done; done
fi
wait
echo "=== closures for $t done ($(date +%H:%M:%S))"
