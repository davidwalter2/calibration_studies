#!/bin/bash
# The card + fit ladder for the VERTEX-CONSTRAINT-RESIDUAL CF term.
#
#   run_ladder.sh card_<name>  [extra make_vtx_card.py args]
#   run_ladder.sh fit_<name>   [extra rabbit_fit.py args]
#
# Channels
#   vtx_*    the vertex term alone   (cf / gauss / gaussq)
#   mass_*   the MASS term alone, same candidates, same in-maker exponents
#   joint_*  BOTH over one parameter set, SAME candidates
#   inj_*    the same with an injection (material group or hit class)
set -e
HERE=/work/submit/david_w/ZMass/calibration_studies/resolution/vtxres
R=${R:-/work/submit/david_w/ZMass/calibration_studies/resolution/runs/vtxres}
VNPZ=${VNPZ:-$R/vtx.npz}
MNPZ=${MNPZ:-$R/mass.npz}
GRP=${GRP:-/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2/src/Analysis/HitAnalyzer/data/materialGroups50.txt}
NCAND=${NCAND:-8000}
# ln(1.05) = 0.0487902 physical k; --whiten means the CARD value is
# k / prior_sigma.  `material_bpix_support6` is the group the VERTEX term sees
# most (the innermost support the first propagation crosses).
INJ=${INJ:-material_bpix_support6:0.00243951}
# a 10 % hit-variance scale on an INNERMOST PIXEL class (linear mode: the card
# value IS eps)
HINJ=${HINJ:-hitres_pix_x_q2:0.10}
COMMON="--groups $GRP --maxn $NCAND --whiten --prune-frac ${PRUNE:-0.001} \
        --poi ${POI:-all} --hit-prior ${HITPRIOR:-1.0} --max-chi2-ndof ${MAXCHI2:-3.0}"
mkdir -p $R/cards $R/fits $HERE/logs
cd $HERE
step=$1; shift || true
case $step in
  card_vtx_cf)     ./run_tf.sh python3 -u make_vtx_card.py $COMMON --vtx-npz $VNPZ --arm cf     -o $R/cards/vtx_cf.hdf5 "$@" ;;
  card_vtx_gauss)  ./run_tf.sh python3 -u make_vtx_card.py $COMMON --vtx-npz $VNPZ --arm gauss  -o $R/cards/vtx_gauss.hdf5 "$@" ;;
  card_vtx_gaussq) ./run_tf.sh python3 -u make_vtx_card.py $COMMON --vtx-npz $VNPZ --arm gaussq -o $R/cards/vtx_gaussq.hdf5 "$@" ;;
  card_mass_cf)    ./run_tf.sh python3 -u make_vtx_card.py $COMMON --mass-npz $MNPZ --arm cf     -o $R/cards/mass_cf.hdf5 "$@" ;;
  card_mass_gaussq)./run_tf.sh python3 -u make_vtx_card.py $COMMON --mass-npz $MNPZ --arm gaussq -o $R/cards/mass_gaussq.hdf5 "$@" ;;
  card_joint_cf)   ./run_tf.sh python3 -u make_vtx_card.py $COMMON --vtx-npz $VNPZ --mass-npz $MNPZ --arm cf --same-candidates -o $R/cards/joint_cf.hdf5 "$@" ;;
  card_joint_gaussq) ./run_tf.sh python3 -u make_vtx_card.py $COMMON --vtx-npz $VNPZ --mass-npz $MNPZ --arm gaussq --same-candidates -o $R/cards/joint_gaussq.hdf5 "$@" ;;
  card_inj_vtx_cf) ./run_tf.sh python3 -u make_vtx_card.py $COMMON --vtx-npz $VNPZ --arm cf --inject $INJ -o $R/cards/inj_vtx_cf.hdf5 "$@" ;;
  card_inj_vtx_gaussq) ./run_tf.sh python3 -u make_vtx_card.py $COMMON --vtx-npz $VNPZ --arm gaussq --inject $INJ -o $R/cards/inj_vtx_gaussq.hdf5 "$@" ;;
  card_inj_mass_cf) ./run_tf.sh python3 -u make_vtx_card.py $COMMON --mass-npz $MNPZ --arm cf --inject $INJ -o $R/cards/inj_mass_cf.hdf5 "$@" ;;
  card_inj_joint_cf) ./run_tf.sh python3 -u make_vtx_card.py $COMMON --vtx-npz $VNPZ --mass-npz $MNPZ --arm cf --same-candidates --inject $INJ -o $R/cards/inj_joint_cf.hdf5 "$@" ;;
  card_injhit_vtx_cf) ./run_tf.sh python3 -u make_vtx_card.py $COMMON --vtx-npz $VNPZ --arm cf --inject $HINJ -o $R/cards/injhit_vtx_cf.hdf5 "$@" ;;
  card_injhit_vtx_gaussq) ./run_tf.sh python3 -u make_vtx_card.py $COMMON --vtx-npz $VNPZ --arm gaussq --inject $HINJ -o $R/cards/injhit_vtx_gaussq.hdf5 "$@" ;;
  fit_*)        name=${step#fit_}; ./run_fit.sh "$name" "$@" ;;
  *) echo "unknown step '$step'"; exit 2 ;;
esac
