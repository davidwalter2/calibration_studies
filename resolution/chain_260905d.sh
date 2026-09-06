#!/bin/bash
# Analysis chain on the _260905d (step-damping) productions. Character-for-
# character chain_260904f.sh with the tags swapped, so the two productions
# differ ONLY in the Gauss-Newton step control.
#
# NEVER run concurrently with a refit.
# stages: 1 kernel+pairs | 2 aux+masks | 3 mass fits | 4 candidate skew
#         5 single-track extraction + closures
# usage: ./chain_260905d.sh [stage ...]
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution; cd "$RES"
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
# shellcheck source=prodfiles.sh
source "$RES/prodfiles.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CVH_IONI_KOKOULIN=0
export PYTHONPATH="$RES:${PYTHONPATH:-}"
CEPH=/ceph/submit/data/user/d/david_w/ZMass/cvh
LOG=$RES/runs/stepdamp260905; mkdir -p "$LOG"
RUNTF=${RUNTF:-/work/submit/david_w/ZMass/calibration_studies/resolution/runs/stepdamp260905/slurm/run_tf_noceph.sh}
[ -x "$RUNTF" ] || RUNTF=/work/submit/david_w/ZMass/calibration_studies/env_tf/run_tf.sh
STAGES=${*:-"1 2 3 4 5"}

GUN=jpsigun_ul16_260905d_m0
MUL=mugun_lowpt_260905d_m0

step() { echo "=== $* ($(date +%H:%M:%S)) ==="; }

for st in $STAGES; do
case $st in

1) step "stage 1: ditrack kernel + pairs ($GUN)"
   G=$CEPH/resolution_trackres_$GUN
   # TASK-level, not stream-0-level: under numberOfThreads=N removing
   # globalcor_0.root alone leaves streams 1..N-1 of a TRUNCATED task for the
   # widened globs below to ingest.
   pf_clean_incomplete "$G" globalcor
   if [ ! -s "runs/cf_masskernel_$GUN.npz" ]; then
     python3 cf_masskernel_tt.py --files "$G/task_*/globalcor_*.root" --ntasks 160 \
         --kernel-cache runs/cf_masskernel_$GUN.npz --postfix "_$GUN" \
         > "$LOG/kernel_$GUN.log" 2>&1 && echo "  [ok] kernel" || echo "  [FAIL] $LOG/kernel_$GUN.log"
   else echo "  skip kernel"; fi
   if [ ! -s "runs/cf_masspairs_$GUN.npz" ]; then
     PROD=$G OUT=$RES/runs/cf_masspairs_$GUN.npz NPAR=120 KOK=0 \
       ./run_pairs_tt_shards.sh > "$LOG/pairs_$GUN.log" 2>&1
     echo "  -> $(ls -la runs/cf_masspairs_$GUN.npz 2>/dev/null | awk '{print $5}')"
   else echo "  skip pairs"; fi ;;

2) step "stage 2: aux columns + masks"
   [ -s "$LOG/aux_jpsigun_260905d.npz" ] || \
     python3 censoring_aux.py --files "$CEPH/resolution_trackres_$GUN/task_*/globalcor_*.root" \
        --cache runs/cf_masspairs_$GUN.npz --out "$LOG/aux_jpsigun_260905d.npz" --nproc 32 --ntasks 160 \
        > "$LOG/aux_jpsigun.log" 2>&1 && echo "  [ok] aux"
   python3 make_masks_260905d.py 2>&1 | tee "$LOG/masks.txt" ;;

3) step "stage 3: mass fits (gun)"
   fit() { nm=$1; P=$2; K=$3; shift 3
     [ -s "runs/masslikfit_${nm}.npz" ] && { echo "  skip $nm"; return; }
     [ -s "$P" ] || { echo "  skip $nm (no $P)"; return; }
     echo "  --- $nm ($(date +%H:%M:%S))"
     $RUNTF python3 cf_masslik_fit.py --pairs-cache "$P" --kernel-cache "$K" "$@" \
         --tag "$nm" --no-plots --out runs/masslikfit_${nm}.npz > "$LOG/fit_${nm}.log" 2>&1
     echo "      rc=$? -> $LOG/fit_${nm}.log"; }
   P=runs/cf_masspairs_$GUN.npz; K=runs/cf_masskernel_$GUN.npz; M=$LOG/mask_gun_260905d
   fit ${GUN}_r         "$P" "$K" --model r
   fit ${GUN}_fam_krad1 "$P" "$K" --model families --krad 1
   fit ${GUN}_r_q3      "$P" "$K" --model r --subset ${M}_q3.npz
   fit ${GUN}_fam_q3    "$P" "$K" --model families --krad 1 --subset ${M}_q3.npz
   fit ${GUN}_r_pt2     "$P" "$K" --model r --subset ${M}_pt2.npz
   fit ${GUN}_fam_pt2   "$P" "$K" --model families --krad 1 --subset ${M}_pt2.npz
   python3 masslikfit_summary.py 'runs/masslikfit_*260905d*.npz' \
       -o "$HOME/public_html/cvh/$(date +%y%m%d)_stepdamp" > "$LOG/masslikfit_summary.txt" 2>&1
   echo "  summary -> $LOG/masslikfit_summary.txt" ;;

4) step "stage 4: candidate (mass) odd-moment closure"
   c=runs/cf_masspairs_$GUN.npz; [ -s "$c" ] || { echo "[skip]"; break; }
   for k in 1 0; do
     python3 cf_skew_closure.py --cache "$c" --mass --krad "$k" \
         --tag "masspairs_${GUN}_krad$k" --label "$GUN candidates krad=$k" \
         > "$LOG/mass_${GUN}_krad$k.log" 2>&1 && echo "[ok] mass krad=$k" || echo "[FAIL] krad=$k" &
   done; wait ;;

5) step "stage 5: single-track extraction + closures"
   if [ ! -s "runs/cf_trackres_${MUL}_k0.npz" ]; then
     G=$CEPH/resolution_trackres_$MUL
     pf_clean_incomplete "$G" globalcor_resclosure
     ./extract_parallel.sh "$G" "runs/cf_trackres_${MUL}_k0.npz" 160 > "$LOG/ext_${MUL}.log" 2>&1 \
       && echo "  [ok] $(ls -la runs/cf_trackres_${MUL}_k0.npz | awk '{print $5}')" \
       || echo "  [FAIL] $LOG/ext_${MUL}.log"
   else echo "  skip extract"; fi
   c=runs/cf_trackres_${MUL}_k0.npz; [ -s "$c" ] || { echo "[skip] no cache"; break; }
   for k in 1 0; do
     python3 cf_track_resolution.py --closure --cache "$c" --krad "$k" \
         --postfix "_${MUL}_k0_krad$k" > "$LOG/trk_${MUL}_krad$k.log" 2>&1 \
       && echo "[ok] trackres krad=$k" || echo "[FAIL] trackres krad=$k" &
   done; wait
   for q in 1 -1; do for k in 1 0; do
     sfx=$([ "$q" = 1 ] && echo pos || echo neg)
     python3 cf_skew_closure.py --cache "$c" --charge "$q" --krad "$k" \
         --tag "${MUL}_k0_${sfx}_krad$k" --label "$MUL q=$q krad=$k" \
         > "$LOG/skew_${MUL}_${sfx}_krad$k.log" 2>&1 \
       && echo "[ok] skew $sfx krad=$k" || echo "[FAIL] skew $sfx krad=$k" &
   done; done; wait ;;
esac
done
echo "=== CHAIN DONE ($(date +%H:%M:%S)) ==="
