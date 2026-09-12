#!/bin/bash
# The whole BEAM-LINE chain, one stage per argument.
#   ./run_all_bs.sh gates|extract|cards|fits|fisher|eff|plots|recovery|bkg|bill|cmp
# Everything large lands on ceph.
set -uo pipefail
HERE=/work/submit/david_w/ZMass/calibration_studies/resolution/vtxres
BL=${BL:-/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/beamline}
R=${R:-$BL/runs}
ON=${ON:-$BL/dy_bs}
OFF=${OFF:-$BL/dy_bsoff}
WIDE=${WIDE:-$BL/dy_bswide}
HALF=${HALF:-$BL/dy_bshalf}
OLD=${OLD:-$BL/dy_bsold}
GRP=${GRP:-/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev3/src/Analysis/HitAnalyzer/data/materialGroups50.txt}
VENV=/work/submit/david_w/ZMass/mfs/.venv/bin/activate
FIG=${FIG:-$HOME/public_html/ZMass/cvh/260913_beamline}
NCAND=${NCAND:-8000}
J=${J:-24}
mkdir -p $R/cards $R/fits $HERE/logs_bs
cd $HERE
export R GRP NCAND
export VNPZ=$R/dy_vtx.npz MNPZ=$R/dy_mass.npz XNPZ=$R/dy_bsx.npz YNPZ=$R/dy_bsy.npz

case ${1:-} in
gates)
  source $VENV
  python3 -u gates_bs.py --files "$ON/task_*/globalcor_*.root" \
    --off "$OFF/task_*/globalcor_*.root" \
    --wide "$WIDE/task_*/globalcor_*.root" \
    --half "$HALF/task_*/globalcor_*.root" \
    --old "$OLD/task_*/globalcor_*.root" \
    --max ${2:-20000} 2>&1 | tee logs_bs/gates_bs.log ;;
genvtx)
  source $VENV
  python3 -u bs_genvtx.py $ON 2>&1 | tee logs_bs/genvtx.log ;;
cmp)
  source $VENV
  python3 -u cmp_bson.py --on "$ON/task_*/globalcor_*.root" \
    --off "$OFF/task_*/globalcor_*.root" --max ${2:-40000} \
    2>&1 | tee logs_bs/cmp_bson.log ;;
extract)
  source $VENV
  for F in bsx bsy vtx mass; do
    python3 -u extract_vtx.py --files "$ON/task_*/globalcor_*.root" \
      --functional $F --groups $GRP -j $J --require-complete \
      --max-chi2-ndof 3 -o $R/dy_$F.npz 2>&1 | tee logs_bs/extract_$F.log
  done ;;
bkg)
  source $VENV
  python3 -u bkg_bs.py --bsx $XNPZ --bsy $YNPZ --tag dy_bs \
    2>&1 | tee logs_bs/bkg_bs.log
  python3 -u genbkg.py --npz $VNPZ --tag dy_bs_vtx --constraint on \
    --classes --cuts 2>&1 | tee logs_bs/genbkg_vtx.log ;;
plots)
  source $VENV
  # the figure directory is named explicitly (`260913_beamline`) rather than
  # by today's date: it is the study's directory and it is referenced by name
  # in the report and in STATE.
  python3 -u plot_vtx.py --npz $XNPZ $YNPZ $VNPZ $MNPZ --tags bsx bsy vtx mass \
    --maxn ${2:-20000} --outpath $FIG --densities --composition --sigma \
    2>&1 | tee logs_bs/plots.log ;;
cards)
  for c in bs_cf bs_gauss bs_gaussq vtx_cf vtx_gaussq mass_cf \
           vtxbs_cf vtxbs_gaussq vtxbsm_cf \
           inj_bs_cf inj_vtx_cf inj_vtxbs_cf inj_vtxbsm_cf \
           injhit_bs_cf injhit_vtx_cf injhit_vtxbs_cf; do
    echo "=== card $c"; ./run_ladder_bs.sh card_$c 2>&1 | tee logs_bs/card_$c.log
  done ;;
fits)
  for c in bs_cf bs_gauss bs_gaussq vtx_cf vtx_gaussq mass_cf \
           vtxbs_cf vtxbs_gaussq vtxbsm_cf \
           inj_bs_cf inj_vtx_cf inj_vtxbs_cf inj_vtxbsm_cf \
           injhit_bs_cf injhit_vtx_cf injhit_vtxbs_cf; do
    [ -s $R/cards/$c.hdf5 ] || { echo "skip $c (no card)"; continue; }
    [ -s $R/fits/$c/fitresults.hdf5 ] && { echo "skip $c (done)"; continue; }
    echo "=== fit $c"; ./run_fit.sh $c 2>&1 | tee logs_bs/fit_$c.log
  done ;;
fisher)
  ./run_tf.sh python3 -u fisher_vtx.py --vtx-npz $VNPZ --mass-npz $MNPZ \
    --bsx-npz $XNPZ --bsy-npz $YNPZ \
    --channels bs vtx vtxbs vtxbsmass --arms cf gauss gaussq \
    --maxn $NCAND --groups $GRP -o $R/fisherHJ.npz 2>&1 | tee logs_bs/fisher.log ;;
eff)
  for ch in bs vtx vtxbs vtxbsmass; do
    echo "=== channel $ch"
    ./run_tf.sh python3 -u ../hitlik/efficiency.py \
      --fisher $R/fisherHJ.npz --cset $ch --arms cf gauss gaussq \
      --ntrk $NCAND --groups $GRP -o $R/eff_$ch.npz 2>&1 | tee logs_bs/eff_$ch.log
  done ;;
recovery)
  ./run_tf.sh python3 -u ../hitlik/recovery.py \
    --pairs bs_cf=$R/fits/bs_cf:$R/fits/inj_bs_cf \
            vtx_cf=$R/fits/vtx_cf:$R/fits/inj_vtx_cf \
            vtxbs_cf=$R/fits/vtxbs_cf:$R/fits/inj_vtxbs_cf \
            vtxbsm_cf=$R/fits/vtxbsm_cf:$R/fits/inj_vtxbsm_cf \
    --card $R/cards/inj_bs_cf.hdf5 --param material_bpix_support6 \
    --groups $GRP 2>&1 | tee logs_bs/recovery.log ;;
recovery-hit)
  ./run_tf.sh python3 -u ../hitlik/recovery.py \
    --pairs bs_cf=$R/fits/bs_cf:$R/fits/injhit_bs_cf \
            vtx_cf=$R/fits/vtx_cf:$R/fits/injhit_vtx_cf \
            vtxbs_cf=$R/fits/vtxbs_cf:$R/fits/injhit_vtxbs_cf \
    --card $R/cards/injhit_bs_cf.hdf5 --param hitres_pix_x_q2 \
    --groups $GRP 2>&1 | tee logs_bs/recovery_hit.log ;;
certify)
  ./run_tf.sh python3 -u ../hitlik/perhit/certify.py \
    --fits $R/fits --params material_bpix_support6 hitres_pix_x_q2 \
    2>&1 | tee logs_bs/certify.log ;;
bill)
  source $VENV
  python3 -u cost_vtx.py --file $(ls $ON/task_0000/globalcor_*.root | head -1) \
    2>&1 | tee logs_bs/bill.log ;;
*) echo "usage: run_all_bs.sh gates|genvtx|cmp|extract|bkg|plots|cards|fits|fisher|eff|recovery|recovery-hit|certify|bill"; exit 2 ;;
esac
