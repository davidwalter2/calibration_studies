#!/bin/bash
# After the sign-fixed pairs-tt shards: merge -> unbinned (r,alpha) scan ->
# odd-moment closure of the mass pulls. Gun (no FSR) and B->J/psi+X v3 (FSR).
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution; cd "$RES"
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$RES:${PYTHONPATH:-}"
LOG=$RES/runs/masssign260902; mkdir -p "$LOG"
step() { echo "=== $1 ($(date +%H:%M:%S)) ==="; shift; "$@" > "$LOG/$(date +%H%M%S)_step.log" 2>&1 && echo "    [ok]" || echo "    [FAIL] rc=$? (see $LOG)"; }
scan() { # scan <pairs-cache> <kernel-cache> <postfix>
  step "scan $3" python3 cf_mass_likelihood.py --demo --pairs-cache "$1" --kernel-cache "$2" \
       --alpha-min=-4e-4 --alpha-max=12e-4 --alpha-n=65 --postfix "$3"
  step "skew-mass $3" python3 cf_skew_closure.py --mass --cache "$1" --tag "masspairs$3" --label "J/psi mass pull $3"
}
# ---- gun: wait for 160 parts, merge, scan
P=$RES/runs/parts_cf_masspairs_jpsigun_ul16_260902_m0_fixsign
until [ "$(ls $P/part_task_*.npz 2>/dev/null | wc -l)" -ge 160 ]; do sleep 300; done
echo "=== gun parts complete ($(date +%H:%M:%S)) ==="
step "merge gun" python3 merge_masspairs.py --out runs/cf_masspairs_jpsigun_ul16_260902_m0_fixsign.npz "$P"/part_task_*.npz
scan runs/cf_masspairs_jpsigun_ul16_260902_m0_fixsign.npz runs/cf_masskernel_jpsigun_ul16_260902_m0.npz _jpsigun_260902_fixsign
# ---- v3 before (auto sign, merged by run_pairs_tt_parallel.sh) and after
until [ -s runs/cf_masspairs_btojpsix_v3_260902_m0.npz ]; do sleep 300; done
scan runs/cf_masspairs_btojpsix_v3_260902_m0.npz runs/cf_masskernel_btojpsix_v3_260902_m0.npz _btojpsix_v3_260902_before
until [ -s runs/cf_masspairs_btojpsix_v3_260902_m0_fixsign.npz ]; do sleep 300; done
scan runs/cf_masspairs_btojpsix_v3_260902_m0_fixsign.npz runs/cf_masskernel_btojpsix_v3_260902_m0.npz _btojpsix_v3_260902_fixsign
echo "=== ALL DONE ($(date +%H:%M:%S)) ==="
