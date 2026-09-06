#!/bin/bash
# Unbinned mass likelihood on the 2026-09-02 J/psi-gun ditrack refit
# (resolution_trackres_jpsigun_ul16_260902_m0: current model, Q-matrix
# estimator). Reproduces the 260811 rung (published: alpha=(0.3159+-0.0168)e-3,
# r=0.9981+-0.0028, naive Gaussian 0.1451e-3 on the Aug-8 refit).
# Waits for all 160 tasks, then kernel -> pairs-tt -> demo scan -> replot.
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution; cd "$RES"
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
# shellcheck source=prodfiles.sh
source "$RES/prodfiles.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
TAG=jpsigun_ul16_260902_m0
G=/ceph/submit/data/user/d/david_w/ZMass/cvh/resolution_trackres_$TAG
LOG=$RES/runs/ditrack260902; mkdir -p "$LOG"
until [ "$(ls -d $G/task_*/.complete 2>/dev/null | wc -l)" -ge 160 ] || grep -q "refit done" "$RES/runs/ditrack260902_driver.log" 2>/dev/null; do sleep 120; done
echo "=== production: $(ls -d $G/task_*/.complete 2>/dev/null | wc -l)/160 complete ($(date +%H:%M:%S)) ==="
# Drop the payload of any failed task so the globs only see complete tasks.
# TASK-level: under numberOfThreads=N, removing stream 0 alone leaves streams
# 1..N-1 of a truncated task behind.
pf_clean_incomplete "$G" globalcor
step() { echo "=== $1 ($(date +%H:%M:%S)) ==="; shift; "$@" || echo "    [FAIL] rc=$?"; }
step kernel  python3 cf_mass_likelihood.py --kernel   --files "$G/task_*/globalcor_0.root" --ntasks 160 \
     --kernel-cache runs/cf_masskernel_$TAG.npz --postfix "_$TAG"
step pairs   python3 cf_mass_likelihood.py --pairs-tt --files "$G/task_*/globalcor_0.root" --ntasks 160 \
     --pairs-cache runs/cf_masspairs_$TAG.npz
step scan    python3 cf_mass_likelihood.py --demo --pairs-cache runs/cf_masspairs_$TAG.npz \
     --kernel-cache runs/cf_masskernel_$TAG.npz --alpha-min=-2e-4 --alpha-max=12e-4 --alpha-n=57 --postfix "_$TAG"
echo "=== ALL DONE ($(date +%H:%M:%S)) ==="
