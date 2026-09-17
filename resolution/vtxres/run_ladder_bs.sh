#!/bin/bash
# The card + fit ladder for the two BEAM-LINE residual CF terms, alongside the
# vertex and the mass ones.  Same machinery as `run_ladder.sh`; the beam
# channels are constraint residuals of the vertex kind, so they take the same
# delta-kernel path in `make_vtx_card.py`.
#
#   run_ladder_bs.sh card_<name> [extra make_vtx_card.py args]
#
# Channels
#   bs_*        the two beam terms alone   (cf / gauss / gaussq)
#   vtxbs_*     vertex + the two beam terms, ONE parameter set
#   vtxbsm_*    vertex + beam + the constrained MASS, ONE parameter set
#   inj*        the same with an injection
set -e
HERE=/work/submit/david_w/ZMass/calibration_studies/resolution/vtxres
R=${R:-/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/beamline/runs}
VNPZ=${VNPZ:-$R/dy_vtx.npz}
MNPZ=${MNPZ:-$R/dy_mass.npz}
XNPZ=${XNPZ:-$R/dy_bsx.npz}
YNPZ=${YNPZ:-$R/dy_bsy.npz}
GRP=${GRP:-/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2/src/Analysis/HitAnalyzer/data/materialGroups50.txt}
NCAND=${NCAND:-8000}
# ln(1.05) = 0.0487902 physical k; --whiten makes the card float k*prior_sigma
INJ=${INJ:-material_bpix_support6:0.00243951}
HINJ=${HINJ:-hitres_pix_x_q2:0.10}
# THE SAMPLE IS DY, NOT J/psi: the mass channel's reference mass and its
# background window have to be the Z's, or the |z| < 40 guard alone removes
# 97 % of the candidates (m0 - 3.0969 GeV over sigma_m is ~85).
MREF=${MREF:-91.1876}
MWIN=${MWIN:-30.0}
COMMON="--groups $GRP --maxn $NCAND --whiten --prune-frac ${PRUNE:-0.001} \
        --poi ${POI:-all} --hit-prior ${HITPRIOR:-1.0} --max-chi2-ndof ${MAXCHI2:-3.0} \
        --m-ref $MREF --m-window $MWIN"
mkdir -p $R/cards $R/fits $HERE/logs
cd $HERE
M=make_vtx_card.py
step=$1; shift || true
case $step in
  card_bs_cf)      ./run_tf.sh python3 -u $M $COMMON --bsx-npz $XNPZ --bsy-npz $YNPZ --arm cf     --same-candidates -o $R/cards/bs_cf.hdf5 "$@" ;;
  card_bs_gauss)   ./run_tf.sh python3 -u $M $COMMON --bsx-npz $XNPZ --bsy-npz $YNPZ --arm gauss  --same-candidates -o $R/cards/bs_gauss.hdf5 "$@" ;;
  card_bs_gaussq)  ./run_tf.sh python3 -u $M $COMMON --bsx-npz $XNPZ --bsy-npz $YNPZ --arm gaussq --same-candidates -o $R/cards/bs_gaussq.hdf5 "$@" ;;
  # THE LUMINOUS-REGION WIDTHS.  The default cards float `beamwidth_x/y` with
  # the RECORD's own prior (2 * BeamWidthError / BeamWidth ~ 0.054 on a
  # variance scale); the `*free*` cards float them with NO prior, which is the
  # honest reading when the record's error is five times smaller than the
  # record-vs-simulation mismatch.  Both are quoted.
  card_bsfree_cf)    ./run_tf.sh python3 -u $M $COMMON --bsx-npz $XNPZ --bsy-npz $YNPZ --arm cf --same-candidates --beamwidth-prior 0 -o $R/cards/bsfree_cf.hdf5 "$@" ;;
  card_vtxbsfree_cf) ./run_tf.sh python3 -u $M $COMMON --vtx-npz $VNPZ --bsx-npz $XNPZ --bsy-npz $YNPZ --arm cf --same-candidates --beamwidth-prior 0 -o $R/cards/vtxbsfree_cf.hdf5 "$@" ;;
  card_vtxbsmfree_cf) ./run_tf.sh python3 -u $M $COMMON --vtx-npz $VNPZ --bsx-npz $XNPZ --bsy-npz $YNPZ --mass-npz $MNPZ --arm cf --same-candidates --beamwidth-prior 0 -o $R/cards/vtxbsmfree_cf.hdf5 "$@" ;;
  card_nobw_bs_cf)   ./run_tf.sh python3 -u $M $COMMON --bsx-npz $XNPZ --bsy-npz $YNPZ --arm cf --same-candidates --no-beamwidth -o $R/cards/nobw_bs_cf.hdf5 "$@" ;;
  card_vtx_cf)     ./run_tf.sh python3 -u $M $COMMON --vtx-npz $VNPZ --arm cf     -o $R/cards/vtx_cf.hdf5 "$@" ;;
  card_vtx_gaussq) ./run_tf.sh python3 -u $M $COMMON --vtx-npz $VNPZ --arm gaussq -o $R/cards/vtx_gaussq.hdf5 "$@" ;;
  card_mass_cf)    ./run_tf.sh python3 -u $M $COMMON --mass-npz $MNPZ --arm cf    -o $R/cards/mass_cf.hdf5 "$@" ;;
  card_vtxbs_cf)   ./run_tf.sh python3 -u $M $COMMON --vtx-npz $VNPZ --bsx-npz $XNPZ --bsy-npz $YNPZ --arm cf --same-candidates -o $R/cards/vtxbs_cf.hdf5 "$@" ;;
  card_vtxbs_gaussq) ./run_tf.sh python3 -u $M $COMMON --vtx-npz $VNPZ --bsx-npz $XNPZ --bsy-npz $YNPZ --arm gaussq --same-candidates -o $R/cards/vtxbs_gaussq.hdf5 "$@" ;;
  card_vtxbsm_cf)  ./run_tf.sh python3 -u $M $COMMON --vtx-npz $VNPZ --bsx-npz $XNPZ --bsy-npz $YNPZ --mass-npz $MNPZ --arm cf --same-candidates -o $R/cards/vtxbsm_cf.hdf5 "$@" ;;
  card_inj_bs_cf)  ./run_tf.sh python3 -u $M $COMMON --bsx-npz $XNPZ --bsy-npz $YNPZ --arm cf --same-candidates --inject $INJ -o $R/cards/inj_bs_cf.hdf5 "$@" ;;
  card_inj_vtxbs_cf) ./run_tf.sh python3 -u $M $COMMON --vtx-npz $VNPZ --bsx-npz $XNPZ --bsy-npz $YNPZ --arm cf --same-candidates --inject $INJ -o $R/cards/inj_vtxbs_cf.hdf5 "$@" ;;
  card_inj_vtxbsm_cf) ./run_tf.sh python3 -u $M $COMMON --vtx-npz $VNPZ --bsx-npz $XNPZ --bsy-npz $YNPZ --mass-npz $MNPZ --arm cf --same-candidates --inject $INJ -o $R/cards/inj_vtxbsm_cf.hdf5 "$@" ;;
  card_injhit_bs_cf) ./run_tf.sh python3 -u $M $COMMON --bsx-npz $XNPZ --bsy-npz $YNPZ --arm cf --same-candidates --inject $HINJ -o $R/cards/injhit_bs_cf.hdf5 "$@" ;;
  card_injhit_vtxbs_cf) ./run_tf.sh python3 -u $M $COMMON --vtx-npz $VNPZ --bsx-npz $XNPZ --bsy-npz $YNPZ --arm cf --same-candidates --inject $HINJ -o $R/cards/injhit_vtxbs_cf.hdf5 "$@" ;;
  card_inj_vtx_cf) ./run_tf.sh python3 -u $M $COMMON --vtx-npz $VNPZ --arm cf --inject $INJ -o $R/cards/inj_vtx_cf.hdf5 "$@" ;;
  card_injhit_vtx_cf) ./run_tf.sh python3 -u $M $COMMON --vtx-npz $VNPZ --arm cf --inject $HINJ -o $R/cards/injhit_vtx_cf.hdf5 "$@" ;;
  fit_*)  name=${step#fit_}; R=$R ./run_fit.sh "$name" "$@" ;;
  *) echo "unknown step '$step'"; exit 2 ;;
esac
