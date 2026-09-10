#!/bin/bash
# The card + fit ladder for the residual-vector CF likelihood.
#
#   run_ladder.sh card_<name>   [extra make_hitlik_card.py args]
#   run_ladder.sh fit_<name>    [extra rabbit_fit.py args]
#
# Cards (NTRK tracks of the 20-60 GeV mu gun, components $COMPS):
#   cf        the full non-Gaussian densities
#   gaussq    the same rows with the fit's own Q-matrix Gaussian (= the chi2)
#   gauss     the same rows with the variance-matched Gaussian
#   cf_c0     CF, q/p component only -- the existing single-functional term
#   joint     CF + the quadratic hit-chi2 term over the whole production
#   inj_*     the same with a 5 % material injection in tib_support
set -e
HERE=/work/submit/david_w/ZMass/calibration_studies/resolution/hitlik
R=/work/submit/david_w/ZMass/calibration_studies/resolution/runs/hitlik
NPZ=${NPZ:-$R/mugun20kv2.npz}
QNPZ=$R/mugun_quad.npz
NTRK=${NTRK:-6000}
COMPS=${COMPS:-0123}
# ln(1.05) = 0.0487902 physical k; the card unit is the group's prior sigma,
# so --whiten means the injected CARD value is k / prior_sigma.
INJ=${INJ:-material_tib_support:0.00243951}
# a 10 % hit-variance scale on the highest-share strip class (linear mode, so
# the card value IS eps)
HINJ=${HINJ:-hitres_str_N3_lo:0.10}
MASSNPZ=${MASSNPZ:-/work/submit/david_w/ZMass/calibration_studies/resolution/runs/matres/gun_groups_probe.npz}
COMMON="--npz $NPZ --quad-npz $QNPZ --max-tracks $NTRK --whiten \
        --prune-frac ${PRUNE:-0.001} --poi material --hit-prior ${HITPRIOR:-1.0}"
mkdir -p $R/cards $R/fits $HERE/logs
cd $HERE
step=$1; shift || true

case $step in
  card_cf)      ./run_tf.sh python3 -u make_hitlik_card.py $COMMON --arm cf \
                   --comps $COMPS --no-quadratic -o $R/cards/cf.hdf5 "$@" ;;
  card_gaussq)  ./run_tf.sh python3 -u make_hitlik_card.py $COMMON --arm gaussq \
                   --comps $COMPS --no-quadratic -o $R/cards/gaussq.hdf5 "$@" ;;
  card_gauss)   ./run_tf.sh python3 -u make_hitlik_card.py $COMMON --arm gauss \
                   --comps $COMPS --no-quadratic -o $R/cards/gauss.hdf5 "$@" ;;
  card_cf_c0)   ./run_tf.sh python3 -u make_hitlik_card.py $COMMON --arm cf \
                   --comps 0 --no-quadratic -o $R/cards/cf_c0.hdf5 "$@" ;;
  card_joint)   ./run_tf.sh python3 -u make_hitlik_card.py $COMMON --arm cf \
                   --comps $COMPS -o $R/cards/joint.hdf5 "$@" ;;
  card_quad)    ./run_tf.sh python3 -u make_hitlik_card.py $COMMON --arm cf \
                   --comps $COMPS --no-residual -o $R/cards/quad.hdf5 "$@" ;;
  card_inj_cf)  ./run_tf.sh python3 -u make_hitlik_card.py $COMMON --arm cf \
                   --comps $COMPS --no-quadratic --inject $INJ \
                   -o $R/cards/inj_cf.hdf5 "$@" ;;
  card_inj_gaussq) ./run_tf.sh python3 -u make_hitlik_card.py $COMMON --arm gaussq \
                   --comps $COMPS --no-quadratic --inject $INJ \
                   -o $R/cards/inj_gaussq.hdf5 "$@" ;;
  card_inj_joint) ./run_tf.sh python3 -u make_hitlik_card.py $COMMON --arm cf \
                   --comps $COMPS --inject $INJ \
                   -o $R/cards/inj_joint.hdf5 "$@" ;;
  # --- the residual term + the J/psi-gun MASS term, one set of amounts ---
  card_mass)    ./run_tf.sh python3 -u make_hitlik_card.py $COMMON --arm cf \
                   --comps $COMPS --no-quadratic --no-residual \
                   --mass-npz $MASSNPZ -o $R/cards/mass.hdf5 "$@" ;;
  card_resmass) ./run_tf.sh python3 -u make_hitlik_card.py $COMMON --arm cf \
                   --comps $COMPS --no-quadratic --mass-npz $MASSNPZ \
                   -o $R/cards/resmass.hdf5 "$@" ;;
  card_inj_mass) ./run_tf.sh python3 -u make_hitlik_card.py $COMMON --arm cf \
                   --comps $COMPS --no-quadratic --no-residual \
                   --mass-npz $MASSNPZ --inject $INJ \
                   -o $R/cards/inj_mass.hdf5 "$@" ;;
  card_inj_resmass) ./run_tf.sh python3 -u make_hitlik_card.py $COMMON --arm cf \
                   --comps $COMPS --no-quadratic --mass-npz $MASSNPZ \
                   --inject $INJ -o $R/cards/inj_resmass.hdf5 "$@" ;;
  card_inj_hit) ./run_tf.sh python3 -u make_hitlik_card.py $COMMON --arm cf \
                   --comps $COMPS --no-quadratic --inject $HINJ \
                   -o $R/cards/inj_hit.hdf5 "$@" ;;
  card_inj_hit_gaussq) ./run_tf.sh python3 -u make_hitlik_card.py $COMMON --arm gaussq \
                   --comps $COMPS --no-quadratic --inject $HINJ \
                   -o $R/cards/inj_hit_gaussq.hdf5 "$@" ;;
  card_*)       name=${step#card_}
                ./run_tf.sh python3 -u make_hitlik_card.py $COMMON --arm cf \
                   --comps $COMPS -o $R/cards/$name.hdf5 "$@" ;;
  fit_*)        name=${step#fit_}
                ./run_fit.sh $name "$@" ;;
  *) echo "unknown step $step"; exit 1 ;;
esac
