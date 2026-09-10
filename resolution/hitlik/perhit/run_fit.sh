#!/bin/bash
# Fit one per-hit card through rabbit_fit.py, EDM-certified by rabbit's own
# termination.  Native tf-trust-krylov (PR #153) for frozen-parameter cards.
#   run_fit.sh <name> [extra rabbit_fit args...]
set -e
HL=/work/submit/david_w/ZMass/calibration_studies/resolution/hitlik
R=/work/submit/david_w/ZMass/calibration_studies/resolution/runs/perhit
RABBIT=${RABBIT:-/work/submit/david_w/ZMass/rabbit-vmass}
name=$1; shift
cd $HL
MODELS=$(./run_tf.sh python3 ../matres/pick_models.py $R/cards/$name.hdf5 2>/dev/null | tail -1)
echo "param models: $MODELS"
exec ./run_tf.sh bash -c "PATH=$RABBIT/bin:\$PATH rabbit_fit.py $R/cards/$name.hdf5 \
  -o $R/fits/$name -t 0 --unblind $MODELS $*"
