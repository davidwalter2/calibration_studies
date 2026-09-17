#!/bin/bash
# The BEAM3 chain: extract -> gates -> reference -> cards -> fits -> report.
#
# RUN IT ON A QUIET SUBMIT NODE (submit50/51/52).  The big submit8x machines
# are shared and can sit at load ~900 on 768 cores, which turns a 10-minute
# fit into an hour; `run_tf.sh` asks for OMP_NUM_THREADS but nothing enforces
# it.  Check `/proc/loadavg` before starting.
#
#   ./run_all_beam3.sh <stage> [args]
#   stages: extract gates genref cards fits report pulls plots
#   env: PROD (the production directory name), R (the runs directory),
#        MAXN, PRUNE, GEOM
set -uo pipefail
HERE=/work/submit/david_w/ZMass/calibration_studies/resolution/vtxres
BL=${BL:-/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/beamline}
PROD=${PROD:-dy_bs_final}
R=${R:-$BL/runs_beam3a}
GRP=${GRP:-/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2/src/Analysis/HitAnalyzer/data/materialGroups50.txt}
VENV=/work/submit/david_w/ZMass/mfs/.venv/bin/activate
MAXN=${MAXN:-8000}
PRUNE=${PRUNE:-0.001}
J=${J:-24}
TAG=${TAG:-dy}
FIG=${FIG:-$HOME/public_html/ZMass/cvh/$(date +%y%m%d)_beam3}
COMMON="--groups $GRP --maxn $MAXN --whiten --prune-frac $PRUNE --poi all \
  --hit-prior 1.0 --max-chi2-ndof 3.0 --m-ref 91.1876 --m-window 30 \
  --vtx-npz $R/dy_vtx.npz --bsx-npz $R/dy_bsx.npz --bsy-npz $R/dy_bsy.npz \
  --arm cf --same-candidates --beamwidth-prior 0"
MCOMMON="$COMMON --mass-npz $R/dy_mass.npz --alpha --mass-corrections \
  --corr-form fluctuation"
mkdir -p "$R/cards" "$R/fits" "$HERE/logs_bs"
cd "$HERE"
export R
case ${1:-} in
extract)
  source $VENV
  for F in bsx bsy vtx mass; do
    python3 -u extract_vtx.py --files "$BL/$PROD/task_*/globalcor_*.root" \
      --functional $F --groups $GRP -j $J --require-complete \
      --max-chi2-ndof 3 -o $R/dy_$F.npz 2>&1 | tee logs_bs/extract_b3_$F.log
  done ;;
gates)
  source $VENV
  python3 -u gate_beam3.py --dir $R 2>&1 | tee logs_bs/gate_beam3.log ;;
genref)
  source $VENV
  python3 -u beam3_gen.py --npz $R/dy_bsx.npz --nboot ${2:-200} \
    2>&1 | tee logs_bs/beam3_gen.log ;;
cards)
  for t in base b3; do
    EX=""; [ $t = b3 ] && EX="--beam3"
    ./run_tf.sh python3 -u make_vtx_card.py $COMMON  $EX -o $R/cards/${t}_vtxbs.hdf5  2>&1 | tee logs_bs/card_${t}_vtxbs.log
    ./run_tf.sh python3 -u make_vtx_card.py $MCOMMON $EX -o $R/cards/${t}_vtxbsm.hdf5 2>&1 | tee logs_bs/card_${t}_vtxbsm.log
  done
  ./run_tf.sh python3 -u gate_beam3_card.py --base $R/cards/base_vtxbs.hdf5 \
    --beam3 $R/cards/b3_vtxbs.hdf5 2>&1 | tee logs_bs/gate_beam3_card.log ;;
fits)
  for c in ${CARDS:-base_vtxbs b3_vtxbs base_vtxbsm b3_vtxbsm}; do
    [ -s $R/cards/$c.hdf5 ] || { echo "skip $c (no card)"; continue; }
    [ -s $R/fits/$c/fitresults.hdf5 ] && { echo "skip $c (done)"; continue; }
    echo "=== fit $c $(date +%H:%M:%S)"
    ./run_fit.sh $c > logs_bs/fit_b3_$c.log 2>&1 && echo "  ok" || echo "  FAIL"
  done ;;
report)
  ./run_tf.sh python3 -u beam3_report.py \
    --fits base=$R/fits/base_vtxbs b3=$R/fits/b3_vtxbs \
           basem=$R/fits/base_vtxbsm b3m=$R/fits/b3_vtxbsm \
    --genref $R/dy_bsx_genref.npz --delta basem,b3m \
    --delta-params alpha,hitres_ 2>&1 | tee logs_bs/beam3_report.log ;;
pulls)
  ./run_tf.sh python3 -u beam3_pulls.py --npz-dir $R --fit $R/fits/b3_vtxbs \
    --groups $GRP --maxn $MAXN 2>&1 | tee logs_bs/beam3_pulls.log ;;
plots)
  ./run_tf.sh python3 -u beam3_plots.py --npz-dir $R \
    --fits base=$R/fits/base_vtxbs b3=$R/fits/b3_vtxbs \
    --genref $R/dy_bsx_genref.npz --main b3 --tag $TAG \
    --outpath $FIG --closure --tilt --pulls 2>&1 | tee logs_bs/beam3_plots.log ;;
*) echo "usage: run_all_beam3.sh extract|gates|genref|cards|fits|report|pulls|plots"; exit 2 ;;
esac
