#!/bin/bash
# The joint-fit ladder on the J/psi gun: quadratic-only, mass-only and joint,
# then the same three with a material injection through BOTH terms.
#
#   run_joint.sh <step> [extra args]
# steps: card_joint card_quad card_mass card_inj fit_<name> report
set -e
HERE=/work/submit/david_w/ZMass/calibration_studies/resolution/matres
RUNS=/work/submit/david_w/ZMass/calibration_studies/resolution/runs/matres
GRP=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/data/materialGroups50.txt
GNPZ=${GNPZ:-$RUNS/gun_groups_probe.npz}
QNPZ=$RUNS/gun_quad.npz
COMMON="--groups-npz $GNPZ --groups $GRP --whiten --max-chi2-ndof 3 \
        --freeze-zero-info --prune-frac ${PRUNE:-0.001} --poi bfield"
mkdir -p $RUNS/cards $RUNS/fits
cd $HERE
step=$1; shift || true

case $step in
  card_joint) ./run_tf.sh python3 -u make_material_card.py $COMMON --quad-npz $QNPZ \
                 -o $RUNS/cards/joint.hdf5 "$@" ;;
  card_quad)  ./run_tf.sh python3 -u make_material_card.py $COMMON --quad-npz $QNPZ \
                 --no-mass -o $RUNS/cards/quad.hdf5 "$@" ;;
  card_mass)  ./run_tf.sh python3 -u make_material_card.py $COMMON --no-quadratic \
                 -o $RUNS/cards/mass.hdf5 "$@" ;;
  card_*)     name=${step#card_}
              ./run_tf.sh python3 -u make_material_card.py $COMMON --quad-npz $QNPZ \
                 -o $RUNS/cards/$name.hdf5 "$@" ;;
  fit_*)      name=${step#fit_}
              # a card whose only parameters come from the external term needs
              # ExternalParams; one with an unbinned term needs UnbinnedParams;
              # a joint card declares everything through the unbinned term.
              MODELS=$(./run_tf.sh python3 pick_models.py $RUNS/cards/$name.hdf5 2>/dev/null | tail -1)
              echo "param models: $MODELS"
              ./run_tf.sh bash -c "PATH=/work/submit/david_w/ZMass/rabbit-material/bin:\$PATH \
                 rabbit_fit.py $RUNS/cards/$name.hdf5 -o $RUNS/fits/$name -t 0 --unblind \
                 $MODELS $*" ;;
  report)     ./run_tf.sh python3 -u report_fit.py "$@" ;;
  *)          echo "unknown step $step"; exit 1 ;;
esac
