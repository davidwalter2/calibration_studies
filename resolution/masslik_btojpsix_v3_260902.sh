#!/bin/bash
# Unbinned mass likelihood on the 2026-09-02 B->J/psi+X v3 ditrack refit
# (resolution_trackres_btojpsix_v3_260902_m0: current model, Q-matrix
# estimator, ladder rung B = grid field + aligned geometry).
# Reference: corrected-CF ladder 2026-08-05, rung B alpha = +0.191+-0.116 e-3.
# Waits for all 48 tasks, then kernel -> pairs-tt -> demo scan.
#
# KERNEL: cf_mass_likelihood.py --kernel rebuilds the gen dimuon mass by
# pairing the SINGLE-track tree's genParms/genCharge by (run,lumi,event);
# the TwoTrack maker writes neither branch (verified: KeyInFileError on
# 'genCharge' for any ditrack production) and exposes the paired gen mass
# as Jpsigen_mass instead.  cf_masskernel_tt.py builds the identical cache
# (same npz schema, same binning, same plot) straight from Jpsigen_mass, so
# kernel and candidate cache come from the SAME production as required.
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution; cd "$RES"
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
# shellcheck source=prodfiles.sh
source "$RES/prodfiles.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
TAG=btojpsix_v3_260902_m0
G=/ceph/submit/data/user/d/david_w/ZMass/cvh/resolution_trackres_$TAG
LOG=$RES/runs/ditrack260902; mkdir -p "$LOG"
until [ "$(ls -d $G/task_*/.complete 2>/dev/null | wc -l)" -ge 48 ] || grep -q "refit done" "$LOG/btojpsix_v3_driver.log" 2>/dev/null; do sleep 120; done
echo "=== production: $(ls -d $G/task_*/.complete 2>/dev/null | wc -l)/48 complete ($(date +%H:%M:%S)) ==="
# Drop the payload of any failed task so the globs only see complete tasks.
# TASK-level: under numberOfThreads=N, removing stream 0 alone leaves streams
# 1..N-1 of a truncated task behind.
pf_clean_incomplete "$G" globalcor
step() { echo "=== $1 ($(date +%H:%M:%S)) ==="; shift; "$@" || echo "    [FAIL] rc=$?"; }
step kernel  python3 cf_masskernel_tt.py --files "$G/task_*/globalcor_0.root" --ntasks 48 \
     --kernel-cache runs/cf_masskernel_$TAG.npz --postfix "_$TAG"
step pairs   python3 cf_mass_likelihood.py --pairs-tt --files "$G/task_*/globalcor_0.root" --ntasks 48 \
     --pairs-cache runs/cf_masspairs_$TAG.npz
step scan    python3 cf_mass_likelihood.py --demo --pairs-cache runs/cf_masspairs_$TAG.npz \
     --kernel-cache runs/cf_masskernel_$TAG.npz --alpha-min=-2e-4 --alpha-max=12e-4 --alpha-n=57 --postfix "_$TAG"
echo "=== ALL DONE ($(date +%H:%M:%S)) ==="
