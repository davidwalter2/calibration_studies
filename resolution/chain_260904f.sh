#!/bin/bash
# Analysis chain on the _260904f (Gauss-Newton clamp-fix) productions.
# Mirrors chain_rad_260903x.sh / drive_pairs_mass_rad.sh / chain_masslikfit_
# rad_260903x.sh so the two productions differ ONLY in the clamp floor
# (2.0 GeV -> 0.25 GeV) and every downstream step is character-identical.
#
# NEVER runs concurrently with a refit (extract_parallel.sh takes one
# BLAS-pinned python per shard and the pair has put this user over `ulimit -u`
# before, which kills running cmsRun with what looks like a physics segfault).
#
# stages:
#   1  ditrack kernels + pairs caches (jpsigun 160 shards, v3 48 shards)
#   2  candidate aux columns (chi2, ptmin, frozen, niter) + quality masks
#   3  mass fits: model r + families(k_rad=1), UNCUT and chi2/ndof<3
#   4  candidate-level (mass) odd-moment closure
#   5  single-track extraction (160 shards, var norm) + closure + skew
#
# usage: ./chain_260904f.sh [stage ...]   (default: all)
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution; cd "$RES"
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
# shellcheck source=prodfiles.sh
source "$RES/prodfiles.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CVH_IONI_KOKOULIN=0
export PYTHONPATH="$RES:${PYTHONPATH:-}"
CEPH=/ceph/submit/data/user/d/david_w/ZMass/cvh
LOG=$RES/runs/clampfix260904; mkdir -p "$LOG"
RUNTF=/work/submit/david_w/ZMass/calibration_studies/env_tf/run_tf.sh
STAGES=${*:-"1 2 3 4 5"}

GUN=jpsigun_ul16_260904f_m0
V3=btojpsix_v3_260904f_m0
MUL=mugun_lowpt_260904f_m0

step() { echo "=== $* ($(date +%H:%M:%S)) ==="; }

for st in $STAGES; do
case $st in

1) step "stage 1: ditrack kernels + pairs"
   for s in "$GUN 160" "$V3 48"; do
     set -- $s; t=$1; n=$2
     G=$CEPH/resolution_trackres_$t
     # drop truncated outputs of any failed task so the globs see complete files only
     # TASK-level: removing stream 0 alone would leave streams 1..N-1 of a
     # truncated task for the widened globs below.
     pf_clean_incomplete "$G" globalcor
     if [ ! -s "runs/cf_masskernel_$t.npz" ]; then
       echo "--- kernel $t"
       python3 cf_masskernel_tt.py --files "$G/task_*/globalcor_*.root" --ntasks $n \
           --kernel-cache runs/cf_masskernel_$t.npz --postfix "_$t" \
           > "$LOG/kernel_$t.log" 2>&1 && echo "    [ok]" || echo "    [FAIL] $LOG/kernel_$t.log"
     else echo "--- skip kernel $t"; fi
     if [ ! -s "runs/cf_masspairs_$t.npz" ]; then
       echo "--- pairs $t"
       PROD=$G OUT=$RES/runs/cf_masspairs_$t.npz NPAR=$([ $n -gt 120 ] && echo 120 || echo $n) KOK=0 \
         ./run_pairs_tt_shards.sh > "$LOG/pairs_$t.log" 2>&1
       echo "    -> $(ls -la runs/cf_masspairs_$t.npz 2>/dev/null | awk '{print $5}')"
     else echo "--- skip pairs $t"; fi
   done ;;

2) step "stage 2: aux columns + masks"
   for s in "$GUN jpsigun 160" "$V3 btojpsix 48"; do
     set -- $s; t=$1; nm=$2; n=$3
     [ -s "$LOG/aux_${nm}_260904f.npz" ] && { echo "--- skip aux $nm"; continue; }
     python3 censoring_aux.py --files "$CEPH/resolution_trackres_$t/task_*/globalcor_*.root" \
        --cache runs/cf_masspairs_$t.npz --out "$LOG/aux_${nm}_260904f.npz" --nproc 32 --ntasks $n \
        > "$LOG/aux_$nm.log" 2>&1 && echo "    [ok] aux $nm" || echo "    [FAIL] aux $nm"
   done
   python3 make_masks_260904f.py 2>&1 | tee "$LOG/masks.txt" ;;

3) step "stage 3: mass fits"
   fit() { # <name> <pairs> <kernel> <extra...>
     nm=$1; P=$2; K=$3; shift 3
     [ -s "runs/masslikfit_${nm}.npz" ] && { echo "--- skip $nm"; return; }
     [ -s "$P" ] || { echo "--- skip $nm (no $P)"; return; }
     echo "--- $nm ($(date +%H:%M:%S))"
     $RUNTF python3 cf_masslik_fit.py --pairs-cache "$P" --kernel-cache "$K" "$@" \
         --tag "$nm" --no-plots --out runs/masslikfit_${nm}.npz \
         > "$LOG/fit_${nm}.log" 2>&1
     echo "    rc=$? -> $LOG/fit_${nm}.log"
   }
   for s in "$GUN gun" "$V3 v3"; do
     set -- $s; t=$1; sh=$2
     P=runs/cf_masspairs_$t.npz; K=runs/cf_masskernel_$t.npz
     M=$LOG/mask_${sh}_260904f
     fit ${t}_r          "$P" "$K" --model r
     fit ${t}_fam_krad1  "$P" "$K" --model families --krad 1
     fit ${t}_r_q3       "$P" "$K" --model r          --subset ${M}_q3.npz
     fit ${t}_fam_q3     "$P" "$K" --model families --krad 1 --subset ${M}_q3.npz
     fit ${t}_r_pt2      "$P" "$K" --model r          --subset ${M}_pt2.npz
     fit ${t}_fam_pt2    "$P" "$K" --model families --krad 1 --subset ${M}_pt2.npz
   done
   python3 masslikfit_summary.py 'runs/masslikfit_*260904f*.npz' \
       -o "$HOME/public_html/cvh/$(date +%y%m%d)_clampfix" > "$LOG/masslikfit_summary.txt" 2>&1
   echo "    summary -> $LOG/masslikfit_summary.txt" ;;

4) step "stage 4: candidate (mass) odd-moment closure"
   for t in $GUN $V3; do
     c=runs/cf_masspairs_$t.npz; [ -s "$c" ] || { echo "[skip] $t"; continue; }
     for k in 1 0; do
       python3 cf_skew_closure.py --cache "$c" --mass --krad "$k" \
           --tag "masspairs_${t}_krad$k" --label "$t candidates krad=$k" \
           > "$LOG/mass_${t}_krad$k.log" 2>&1 \
         && echo "[ok] mass $t krad=$k" || echo "[FAIL] mass $t krad=$k" &
     done
   done; wait ;;

5) step "stage 5: single-track extraction + closures"
   if [ ! -s "runs/cf_trackres_${MUL}_k0.npz" ]; then
     G=$CEPH/resolution_trackres_$MUL
     pf_clean_incomplete "$G" globalcor_resclosure
     echo "--- extract $MUL (160 shards)"
     ./extract_parallel.sh "$G" "runs/cf_trackres_${MUL}_k0.npz" 160 \
        > "$LOG/ext_${MUL}.log" 2>&1 \
       && echo "    [ok] $(ls -la runs/cf_trackres_${MUL}_k0.npz | awk '{print $5}')" \
       || echo "    [FAIL] $LOG/ext_${MUL}.log"
   else echo "--- skip extract $MUL"; fi
   c=runs/cf_trackres_${MUL}_k0.npz
   [ -s "$c" ] || { echo "[skip] no cache"; break; }
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
