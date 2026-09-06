#!/bin/bash
# PHASE 2 -- joint J/psi + Z, on J/psi **v2**.
#
# v2, not v1: it finishes at the same time as the v1 slurm arrays and it is the
# stronger leg. It carries `Jpsi_covrefmom` (so the Jensen s^2 is truth-free
# per candidate instead of carrying an MC-measured f_ang and its +-5 %), the
# two-track variance/log-det gradient on parmtype 15 (a factor 41 on the
# material Fisher information on the gun), `Mu*_maxfracloss`, and the per-group
# CF exponents phase 3 needs. v1 is kept only as a cross-check of the
# mean-loss-only quadratic term against v2's variance-block version.
#
# usage: ./run_phase2.sh [pairs|quad|card|fit]
set -euo pipefail
FS=/work/submit/david_w/ZMass/calibration_studies/fullscale
RES=/work/submit/david_w/ZMass/calibration_studies/resolution
Z=/work/submit/david_w/ZMass/calibration_studies/zchannel
CEPH=/ceph/submit/data/user/d/david_w/ZMass/cvh
JV2=$CEPH/jpsimc_20M_260906_v2
DYV2=$CEPH/dymc_8p5M_260906_v2
GRP=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/data/materialGroups50.txt
STAGES=${*:-pairs quad}

need_complete () {   # refuse to build a phase-2 input from a partial production
  local n; n=$(ls "$JV2"/task_*/.complete 2>/dev/null | wc -l)
  if [ "$n" -lt 1642 ]; then
    echo "jpsimc_20M_260906_v2 is $n/1642 complete -- phase 2 waits." >&2
    exit 1
  fi
}

for st in $STAGES; do
case $st in
pairs)
  need_complete
  source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
  python3 -u "$RES/cf_inmaker.py" pairs --files "$JV2" --ntasks 0 \
      --cache "$FS/runs/jpairs_v2.npz" --jac-parmtypes 14 15 \
      2>&1 | tee "$FS/logs/jpairs_v2.log"
  ;;
quad)
  need_complete
  source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
  python3 -u "$RES/globalfit/extract.py" --files "$JV2" --ntasks 0 \
      --parmtypes 14 15 --no-mass \
      --max-chi2-ndof 3 --max-hess 1e8 --max-grad 1e6 --max-dEref-p 0.01 \
      -j 16 -o "$FS/runs/quad_jpsiv2.npz" 2>&1 | tee "$FS/logs/quad_jpsiv2.log"
  ;;
card)
  THREADS=${THREADS:-32} "$FS/run_tf.sh" python3 -u "$FS/make_joint_card.py" \
      --jpsi-pairs "$FS/runs/jpairs_v2.npz" \
      --z-pairs    "$FS/runs/zpairs_dyv2_jac.npz" \
      --quad "$FS/runs/quad_jpsiv2.npz" "$FS/runs/quad_dyv2.npz" \
      --groups "$GRP" --whiten --shape 5 \
      --fsr "$Z/data/kern_loose_band3.3e-4.npz" --acc "$Z/data/acc_loose_d8.json" \
      -o "$FS/cards/joint_v2.hdf5" 2>&1 | tee "$FS/logs/card_joint_v2.log"
  ;;
fit)
  THREADS=${THREADS:-48} "$FS/run_tf.sh" python3 -u "$FS/fit.py" \
      --card "$FS/cards/joint_v2.hdf5" --fix k_hit k_ms k_ioni k_rad \
      --chunk 32768 --label joint_v2 -o "$FS/results/fit_joint_v2.json" \
      2>&1 | tee "$FS/logs/fit_joint_v2.log"
  ;;
esac
done
