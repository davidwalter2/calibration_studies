#!/bin/bash
# Closures on the _260903x caches, WITH (krad=1) and WITHOUT (krad=0) the new
# radiative CF term.  krad is a LINEAR scale on the block, so krad=0 is the
# model exactly as it was before the term existed -- the two arms differ in
# one term and nothing else, from ONE extraction.
#
# stage 1  trackres : <e^{-u z^2}> closure, six caches x two arms
# stage 2  skew     : odd moment per charge, four caches x two arms
# stage 3  kms      : the A/B/C k_MS ladder (its own driver, 4-way parallel)
# stage 4  mass     : odd moment on the two candidate (pairs) caches
#
# usage: ./chain_closures_rad_260903x.sh [stage ...]   (default: all)
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution; cd "$RES"
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CVH_IONI_KOKOULIN=0
DAY=$(date +%y%m%d)
LOG=$RES/runs/rad260903x; mkdir -p "$LOG"
STAGES=${*:-"1 2 3 4"}

TRK=(mugun_lowpt_260903x_m0_k0 mugun_ul16_260903x_m0_k0
     mugun_lowpt_260903x_k0 mugun_ul16_260903x_k0
     mugun_lowpt_noms_260903x_m0_k0 mugun_lowpt_nomsrad_260903x_m0_k0)
SKW=(mugun_lowpt_260903x_m0_k0 mugun_ul16_260903x_m0_k0
     mugun_lowpt_260903x_k0 mugun_ul16_260903x_k0)

# ---------------------------------------------------------------- stage 1
run_trk() { # <cache tag> <krad>
  local t=$1 k=$2 c="runs/cf_trackres_$1.npz"
  [ -s "$c" ] || { echo "[skip] $t (no cache)"; return; }
  python3 cf_track_resolution.py --closure --cache "$c" --krad "$k" \
      --postfix "_${t}_krad$k" > "$LOG/trk_${t}_krad$k.log" 2>&1 \
    && echo "[ok] trackres $t krad=$k $(date +%H:%M)" || echo "[FAIL] trackres $t krad=$k"
}
# ---------------------------------------------------------------- stage 2
run_skw() { # <cache tag> <charge> <krad>
  local t=$1 q=$2 k=$3 c="runs/cf_trackres_$1.npz"
  [ -s "$c" ] || { echo "[skip] $t (no cache)"; return; }
  local sfx; sfx=$([ "$q" = 1 ] && echo pos || ([ "$q" = -1 ] && echo neg || echo both))
  python3 cf_skew_closure.py --cache "$c" --charge "$q" --krad "$k" \
      --tag "${t}_${sfx}_krad$k" --label "$t q=$q krad=$k" \
      > "$LOG/skew_${t}_${sfx}_krad$k.log" 2>&1 \
    && echo "[ok] skew $t $sfx krad=$k $(date +%H:%M)" || echo "[FAIL] skew $t $sfx krad=$k"
}
# ---------------------------------------------------------------- stage 4
run_mass() { # <pairs tag> <krad>
  local t=$1 k=$2 c="runs/cf_masspairs_$1.npz"
  [ -s "$c" ] || { echo "[skip] mass $t (no cache)"; return; }
  python3 cf_skew_closure.py --cache "$c" --mass --krad "$k" \
      --tag "masspairs_${t}_krad$k" --label "$t candidates krad=$k" \
      > "$LOG/mass_${t}_krad$k.log" 2>&1 \
    && echo "[ok] mass $t krad=$k $(date +%H:%M)" || echo "[FAIL] mass $t krad=$k"
}

for st in $STAGES; do
case $st in
1) echo "=== stage 1 trackres ($(date +%H:%M:%S)) ==="
   for t in "${TRK[@]}"; do for k in 1 0; do run_trk "$t" "$k" & done; done; wait ;;
2) echo "=== stage 2 skew ($(date +%H:%M:%S)) ==="
   # 8 at a time: each holds one full cache plus a (chunk x 448) complex block
   for t in "${SKW[@]}"; do
     for q in 1 -1; do for k in 1 0; do run_skw "$t" "$q" "$k" & done; done
     wait
   done ;;
3) echo "=== stage 3 kms ladder ($(date +%H:%M:%S)) ==="
   ./run_kms_260903x.sh ;;
4) echo "=== stage 4 mass ($(date +%H:%M:%S)) ==="
   for t in jpsigun_ul16_260903x_m0 btojpsix_v3_260903x_m0; do
     for k in 1 0; do run_mass "$t" "$k" & done
   done; wait ;;
esac
done
echo "=== closures done ($(date +%H:%M:%S)); figures in ~/public_html/cvh/${DAY}_{trackres,skew}/"
