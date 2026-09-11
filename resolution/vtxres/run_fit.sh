#!/bin/bash
# run_fit.sh <cardname> [extra rabbit_fit args]
# Every fit goes through rabbit_fit.py and is certified by value AND NLL AND
# rabbit's EDM.  `pick_models.py` chooses tf-trust-krylov for frozen-parameter
# cards and trust-exact otherwise.
set -e
HERE=/work/submit/david_w/ZMass/calibration_studies/resolution/vtxres
R=${R:-/work/submit/david_w/ZMass/calibration_studies/resolution/runs/vtxres}
RABBIT=${RABBIT:-/work/submit/david_w/ZMass/rabbit-vmass}
name=$1; shift
cd $HERE
MODELS=$(./run_tf.sh python3 ../matres/pick_models.py $R/cards/$name.hdf5 2>/dev/null | tail -1)
echo "param models: $MODELS"
exec ./run_tf.sh bash -c "PATH=$RABBIT/bin:\$PATH rabbit_fit.py $R/cards/$name.hdf5 \
  -o $R/fits/$name -t 0 --unblind $MODELS $*"
