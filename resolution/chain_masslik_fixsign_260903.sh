#!/bin/bash
# After the sharded pairs-tt builds (run_pairs_tt_shards.sh, Kokoulin off):
# unbinned (r,alpha) scan + odd-moment closure of the mass pulls, for
#   gun  sign-fixed            (no FSR)     vs published Aug-8 unfixed alpha=+0.316e-3
#   v3   before / after fix    (with FSR)
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution; cd "$RES"
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$RES:${PYTHONPATH:-}" CVH_IONI_KOKOULIN=0
LOG=$RES/runs/masssign260902; mkdir -p "$LOG"
step() { local n=$1; shift; echo "=== $n ($(date +%H:%M:%S)) ==="; "$@" > "$LOG/$(date +%m%d_%H%M%S)_$n.log" 2>&1 && echo "    [ok] $n ($(date +%H:%M:%S))" || echo "    [FAIL] $n rc=$? (see $LOG)"; }
waitmerge() { until grep -q "=== merged" "$1" 2>/dev/null; do sleep 120; done; }
do_sample() { # do_sample <shardlog> <pairs-cache> <kernel-cache> <postfix>
  waitmerge "$1"; echo "=== merged: $2 ($(date +%H:%M:%S)) ==="
  step "scan$4"  python3 cf_mass_likelihood.py --demo --pairs-cache "$2" --kernel-cache "$3" --alpha-min=-4e-4 --alpha-max=12e-4 --alpha-n=65 --postfix "$4"
  step "skew$4"  python3 cf_skew_closure.py --mass --cache "$2" --tag "masspairs$4" --label "J/psi mass pull $4"
}
do_sample runs/shards_gun_fixsign.log runs/cf_masspairs_jpsigun_ul16_260902_m0_fixsign.npz runs/cf_masskernel_jpsigun_ul16_260902_m0.npz _jpsigun_260902_fixsign &
do_sample runs/shards_v3_before.log  runs/cf_masspairs_btojpsix_v3_260902_m0.npz          runs/cf_masskernel_btojpsix_v3_260902_m0.npz _btojpsix_v3_260902_before &
do_sample runs/shards_v3_fixsign.log runs/cf_masspairs_btojpsix_v3_260902_m0_fixsign.npz  runs/cf_masskernel_btojpsix_v3_260902_m0.npz _btojpsix_v3_260902_fixsign &
wait; echo "=== ALL DONE ($(date +%H:%M:%S)) ==="
