#!/bin/bash
# THE STANDARD-SELECTION STUDY (`STATE.md` section 14).
#
# Three cards from ONE extraction, so the only thing that differs between them
# is the likelihood:
#   a  no |z_v| cut, untruncated likelihood           -> the reference
#   b  |z_v| < 5 AND the truncated normalisation      -> the standard
#   c  |z_v| < 5 and NO truncated normalisation       -> the bias being
#                                                        demonstrated
# and the same three on the MASS term, which is where the cut's effect on the
# momentum scale and the material is read off.
#
#   ./run_sel.sh extract|cards|fits|report
set -uo pipefail
HERE=/work/submit/david_w/ZMass/calibration_studies/resolution/vtxres
SELROOT=${SELROOT:-/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/selection}
R=${R:-$SELROOT/runs}
PROD=${PROD:-/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/prod_vtxon}
GRP=${GRP:-/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2/src/Analysis/HitAnalyzer/data/materialGroups50.txt}
VENV=/work/submit/david_w/ZMass/mfs/.venv/bin/activate
J=${J:-24}
# per FILE; 161 files -> ~20 k candidates, twice the published 8 000 so the
# 0.1 % tail the cut removes is resolved at all
MAXC=${MAXC:-130}
mkdir -p $R/cards $R/fits $HERE/logs_sel
cd $HERE || exit 9

case ${1:-} in
extract)
  source $VENV
  # NO |z_v| cut here: the CARD applies it, so one extraction serves all
  # three arms and the three differ by nothing else.  The leg-hit and
  # finiteness parts of the standard selection ARE applied (they are the
  # maker default from `cvh-exports-clean-260911` on; an older production
  # gets them here).
  for F in vtx mass; do
    python3 -u extract_vtx.py --files "$PROD/task_*/globalcor_*.root" \
      --functional $F --groups $GRP -j $J \
      --max-abs-vtxz 0 \
      --max-chi2-ndof 3 --max-cands $MAXC -o $R/$F.npz \
      2>&1 | tee logs_sel/extract_$F.log
  done ;;
cards)
  for CH in vtx mass; do
    NPZ=$R/$CH.npz; ARG=--$CH-npz
    # the MASS channel floats the momentum scale as well, so the cut's effect
    # on `alpha` is measured and not only its effect on the widths
    EXTRA=""; [ $CH = mass ] && EXTRA="--alpha"
    # a: no cut, untruncated
    ./run_tf.sh python3 -u make_vtx_card.py --groups $GRP --maxn 0 --whiten \
      --prune-frac 0.001 --poi all --max-chi2-ndof 3 $ARG $NPZ --arm cf \
      --max-abs-vtxz 0 $EXTRA -o $R/cards/${CH}_a_nocut.hdf5 \
      2>&1 | tee logs_sel/card_${CH}_a_nocut.log
    # b: cut + truncated normalisation (the STANDARD)
    ./run_tf.sh python3 -u make_vtx_card.py --groups $GRP --maxn 0 --whiten \
      --prune-frac 0.001 --poi all --max-chi2-ndof 3 $ARG $NPZ --arm cf \
      $EXTRA -o $R/cards/${CH}_b_cut_norm.hdf5 \
      2>&1 | tee logs_sel/card_${CH}_b_cut_norm.log
    # c: cut, NO truncated normalisation -- the biased likelihood
    ./run_tf.sh python3 -u make_vtx_card.py --groups $GRP --maxn 0 --whiten \
      --prune-frac 0.001 --poi all --max-chi2-ndof 3 $ARG $NPZ --arm cf \
      --vtx-norm-window 0 $EXTRA -o $R/cards/${CH}_c_cut_nonorm.hdf5 \
      2>&1 | tee logs_sel/card_${CH}_c_cut_nonorm.log
  done ;;
fits)
  for c in vtx_a_nocut vtx_b_cut_norm vtx_c_cut_nonorm \
           mass_a_nocut mass_b_cut_norm mass_c_cut_nonorm; do
    [ -s $R/cards/$c.hdf5 ] || { echo "skip $c (no card)"; continue; }
    [ -s $R/fits/$c/fitresults.hdf5 ] && { echo "skip $c (done)"; continue; }
    echo "=== fit $c"; R=$R ./run_fit.sh $c 2>&1 | tee logs_sel/fit_$c.log
  done ;;
report)
  ./run_tf.sh python3 -u sel_report.py --fits $R/fits --groups $GRP \
    2>&1 | tee logs_sel/report.log ;;
*) echo "usage: run_sel.sh extract|cards|fits|report"; exit 2 ;;
esac
