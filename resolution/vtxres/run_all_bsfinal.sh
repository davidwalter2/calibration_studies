#!/bin/bash
# The FINAL beam-line chain: the same stages as `run_all_bs.sh`, pointed at the
# `dy_bs_final` / `dy_bsoff_final` productions (the whitened pair, the "+"-form
# leave-one-out construction, the cached sqrt(dV_b) and the two floating
# luminous-region widths) and at the dev2 build.
#
#   ./run_all_bsfinal.sh <stage> [args]        stages: see run_all_bs.sh
#   ./run_all_bsfinal.sh widths                the width-float readout
set -uo pipefail
HERE=/work/submit/david_w/ZMass/calibration_studies/resolution/vtxres
BL=/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/beamline
export BL
export R=$BL/runs_final
export ON=$BL/dy_bs_final
# THE ROWS-OFF REFERENCE is the FIRST pass's `dy_bsoff`: same six input
# files, same 4000 events, same rows-OFF configuration, and the FIT is
# untouched by this pass -- the gun gate is 264/264 branches bit-identical
# between the two builds with the rows off, so dev3's rows-OFF tree and dev2's
# are the same numbers.  `OFF=$BL/dy_bsoff_final` re-measures it from scratch
# when that leg lands.
export OFF=${OFF:-$BL/dy_bsoff}
export GRP=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2/src/Analysis/HitAnalyzer/data/materialGroups50.txt
export FIG=$HOME/public_html/ZMass/cvh/$(date +%y%m%d)_bsfinal
export NCAND=${NCAND:-8000}
mkdir -p "$R/cards" "$R/fits" "$FIG"
cd $HERE
case ${1:-} in
gates)
  # G2 / G4 compare against the weightless and the OLD-build legs, which are
  # the FIRST pass's productions and are not re-run here: the defect and the
  # weightless reduction were settled then and neither the whitening nor the
  # cache can move them.  This stage runs everything the new construction DOES
  # move: G1, G3, G5, G6, G7, G7a, G8.
  source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
  python3 -u gates_bs.py --files "$ON/task_*/globalcor_*.root" \
    --off "$OFF/task_*/globalcor_*.root" --max ${2:-40000} \
    2>&1 | tee logs_bs/gates_bsfinal.log ;;
widths)
  ./run_tf.sh python3 -u width_report.py --fits \
    $R/fits/bs_cf $R/fits/vtx_cf $R/fits/vtxbs_cf $R/fits/vtxbsm_cf \
    $R/fits/bsfree_cf $R/fits/vtxbsfree_cf $R/fits/vtxbsmfree_cf \
    2>&1 | tee logs_bs/widths.log ;;
*)
  exec ./run_all_bs.sh "$@" ;;
esac
