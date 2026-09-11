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

# The full ladder, end to end:
#
#   RUNS=.../runs/matres
#   # per-group exponents, 24k ditrack candidates, 64-point tau grid
#   $RUNS/run_probe.sh                                    # 1812 s, 24 workers
#   # the quadratic term over the FULL production
#   $RUNS/run_quad.sh                                     # 51 s, 8 workers
#   ./run_joint.sh card_quad                              # quadratic-only card
#   ./run_joint.sh card_joint                             # quadratic + mass
#   ./run_joint.sh card_mass --field-prior 1.0            # mass only
#   ./run_joint.sh card_inj --inject material_tib_support:0.00243951   # 5 % more
#   ./run_joint.sh fit_quad --minimizerMethod trust-exact # 23 s
#   ./run_joint.sh fit_mass                               # 232 s
#   ./run_joint.sh fit_joint                              # 2467 s
#   ./run_joint.sh fit_inj                                # 449 s
#   ./run_tf.sh python3 cmp_scale.py --card $RUNS/cards/joint.hdf5 \
#       quad=$RUNS/fits/quad/fitresults.hdf5 \
#       mass=$RUNS/fits/mass/fitresults.hdf5 \
#       joint=$RUNS/fits/joint/fitresults.hdf5
#   ./run_joint.sh report --fit $RUNS/fits/inj/fitresults.hdf5 \
#       --card $RUNS/cards/inj.hdf5 --compare base=$RUNS/fits/joint/fitresults.hdf5
#
# MASSCFTERM_SPEC gate (self-consistent resolution):
#   C="--groups-npz $RUNS/gun_groups_probe.npz --groups $GRP --whiten \
#      --max-chi2-ndof 3 --freeze-zero-info --prune-frac 0.001 --no-quadratic \
#      --no-jac --with-alpha --poi none"
#   ./run_tf.sh python3 make_material_card.py $C --no-self-consistent-sigma \
#       -o $RUNS/cards/a2_naive.hdf5
#   ./run_tf.sh python3 make_material_card.py $C -o $RUNS/cards/a2_corr.hdf5
#   # ... then rabbit_fit.py on each, --paramModel UnbinnedParams
#   # naive -0.026120 +- 0.058680, corrected +0.121160 +- 0.058677,
#   # shift +0.147280 against the spec's +0.146 +- 0.01.
#   # --no-jac IS LOAD-BEARING: with the field modes free, bfield_mode0 (the
#   # same scale as alpha) takes +0.068e-3 of the shift and the gate reads
#   # +0.0121 instead.
