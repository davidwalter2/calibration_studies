#!/bin/bash
# The whole vtxres chain, one stage per argument.
#   ./run_all.sh extract|gates|cards|fits|fisher|eff|xcum|plots|recovery
# Everything large lands on ceph; runs/vtxres is small (npz + cards + fits).
set -uo pipefail
HERE=/work/submit/david_w/ZMass/calibration_studies/resolution/vtxres
R=${R:-/work/submit/david_w/ZMass/calibration_studies/resolution/runs/vtxres}
PROD=${PROD:-/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/prod}
DY=${DY:-/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/dy}
GRP=${GRP:-/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2/src/Analysis/HitAnalyzer/data/materialGroups50.txt}
VENV=/work/submit/david_w/ZMass/mfs/.venv/bin/activate
NCAND=${NCAND:-8000}
NEXT=${NEXT:-60000}
J=${J:-24}
mkdir -p $R/cards $R/fits $HERE/logs
cd $HERE
export VNPZ=$R/vtx.npz MNPZ=$R/mass.npz R GRP NCAND

case ${1:-} in
extract)
  source $VENV
  for F in vtx mass; do
    python3 -u extract_vtx.py --files "$PROD/task_*/globalcor_*.root" \
      --functional $F --groups $GRP -j $J --require-complete \
      --max-chi2-ndof 3 --max-cands $((NEXT/160+1)) -o $R/$F.npz \
      2>&1 | tee logs/extract_$F.log
  done ;;
extract-dy)
  source $VENV
  for F in vtx mass; do
    python3 -u extract_vtx.py --files "$DY/task_*/globalcor_*.root" \
      --functional $F --groups $GRP -j $J --require-complete \
      --max-chi2-ndof 3 -o $R/dy_$F.npz 2>&1 | tee logs/extract_dy_$F.log
  done ;;
gates)
  source $VENV
  python3 -u gates.py --files "$PROD/task_00*/globalcor_*.root" --max ${2:-20000} \
    2>&1 | tee logs/gates_prod.log ;;
xcum)
  source $VENV
  python3 -u xcum_vtx.py --files "$PROD/task_00*/globalcor_*.root" --max ${2:-20000} \
    2>&1 | tee logs/xcum.log ;;
bill)
  source $VENV
  python3 -u cost_vtx.py --file $(ls $PROD/task_0000/globalcor_*.root | head -1) \
    2>&1 | tee logs/bill.log ;;
cards)
  for c in vtx_cf vtx_gauss vtx_gaussq mass_cf mass_gaussq joint_cf joint_gaussq \
           inj_vtx_cf inj_vtx_gaussq inj_mass_cf inj_joint_cf \
           injhit_vtx_cf injhit_vtx_gaussq; do
    echo "=== card $c"; ./run_ladder.sh card_$c 2>&1 | tee logs/card_$c.log
  done ;;
fits)
  for c in vtx_cf vtx_gauss vtx_gaussq mass_cf mass_gaussq joint_cf joint_gaussq \
           inj_vtx_cf inj_vtx_gaussq inj_mass_cf inj_joint_cf \
           injhit_vtx_cf injhit_vtx_gaussq; do
    [ -s $R/cards/$c.hdf5 ] || { echo "skip $c (no card)"; continue; }
    [ -s $R/fits/$c/fitresults.hdf5 ] && { echo "skip $c (done)"; continue; }
    echo "=== fit $c"; ./run_fit.sh $c 2>&1 | tee logs/fit_$c.log
  done ;;
fisher)
  ./run_tf.sh python3 -u fisher_vtx.py --vtx-npz $VNPZ --mass-npz $MNPZ \
    --groups $GRP --channels vtx mass joint --arms cf gauss gaussq \
    --maxn $NCAND -o $R/fisherHJ.npz 2>&1 | tee logs/fisher.log ;;
eff)
  for ch in vtx mass joint; do
    echo "=== channel $ch"
    ./run_tf.sh python3 -u /work/submit/david_w/ZMass/calibration_studies/resolution/hitlik/efficiency.py \
      --fisher $R/fisherHJ.npz --cset $ch --arms cf gauss gaussq \
      --ntrk $NCAND --groups $GRP -o $R/eff_$ch.npz 2>&1 | tee logs/eff_$ch.log
  done ;;
certify)
  ./run_tf.sh python3 -u /work/submit/david_w/ZMass/calibration_studies/resolution/hitlik/perhit/certify.py \
    --fits $R/fits --params material_bpix_support6 hitres_pix_x_q2 \
    2>&1 | tee logs/certify.log ;;
plots)
  source $VENV
  python3 -u plot_vtx.py --npz $VNPZ $MNPZ --tags vtx mass --maxn ${2:-20000} \
    2>&1 | tee logs/plots.log ;;
plots-dy)
  source $VENV
  python3 -u plot_vtx.py --npz $R/dy_vtx.npz --tags dyvtx --maxn ${2:-20000} \
    2>&1 | tee logs/plots_dy.log ;;
recovery)
  ./run_tf.sh python3 -u /work/submit/david_w/ZMass/calibration_studies/resolution/hitlik/recovery.py \
    --pairs vtx_cf=$R/fits/vtx_cf:$R/fits/inj_vtx_cf \
            vtx_gaussq=$R/fits/vtx_gaussq:$R/fits/inj_vtx_gaussq \
            mass_cf=$R/fits/mass_cf:$R/fits/inj_mass_cf \
            joint_cf=$R/fits/joint_cf:$R/fits/inj_joint_cf \
    --card $R/cards/inj_vtx_cf.hdf5 --param material_bpix_support6 \
    --groups $GRP 2>&1 | tee logs/recovery.log ;;
recovery-hit)
  ./run_tf.sh python3 -u /work/submit/david_w/ZMass/calibration_studies/resolution/hitlik/recovery.py \
    --pairs vtx_cf=$R/fits/vtx_cf:$R/fits/injhit_vtx_cf \
            vtx_gaussq=$R/fits/vtx_gaussq:$R/fits/injhit_vtx_gaussq \
    --card $R/cards/injhit_vtx_cf.hdf5 --param hitres_pix_x_q2 \
    --groups $GRP 2>&1 | tee logs/recovery_hit.log ;;
*) echo "usage: run_all.sh extract|extract-dy|gates|xcum|bill|cards|fits|fisher|eff|certify|plots|plots-dy|recovery|recovery-hit"; exit 2 ;;
esac
