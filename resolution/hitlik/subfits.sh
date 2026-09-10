#!/bin/bash
# The EMPIRICAL check of the estimator's actual spread: K DISJOINT subsamples
# of the same production, each fitted for real, for the CF arm and the
# Gaussian one.  The spread of theta_hat across the K fits, divided by
# sqrt(K), is the full-sample sigma with NO model assumption -- in particular
# without assuming H = J, which is exactly what the sandwich is testing.
set -e
HERE=/work/submit/david_w/ZMass/calibration_studies/resolution/hitlik
R=/work/submit/david_w/ZMass/calibration_studies/resolution/runs/hitlik
NPZ=${NPZ:-$R/mugun20kv2.npz}
QNPZ=$R/mugun_quad.npz
K=${K:-8}
NSUB=${NSUB:-2500}
COMPS=${COMPS:-0123}
GRP=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/data/materialGroups50.txt
cd $HERE
mkdir -p $R/cards $R/fits logs
for arm in ${ARMS:-cf gaussq}; do
  for k in $(seq 0 $((K-1))); do
    nm=sub_${arm}_$k
    off=$((k*NSUB))
    if [ ! -f $R/fits/$nm/fitresults.hdf5 ]; then
      echo "=== $nm (offset $off) $(date +%H:%M:%S)"
      ./run_tf.sh python3 -u make_hitlik_card.py --npz $NPZ --quad-npz $QNPZ \
        --max-tracks $NSUB --track-offset $off --whiten --prune-frac 0.001 \
        --poi material --hit-prior 1.0 --arm $arm --comps $COMPS \
        --no-quadratic -o $R/cards/$nm.hdf5 > logs/card_$nm.log 2>&1
      ./run_fit.sh $nm --minimizerMethod tf-trust-krylov > logs/fit_$nm.log 2>&1 \
        || echo "FIT $nm FAILED"
      grep -a edmval logs/fit_$nm.log | tail -1
    fi
  done
done
echo "SUBFITS DONE $(date +%H:%M:%S)"
