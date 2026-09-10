#!/bin/bash
# The quadratic (hit-chi2) term over the SAME mu-gun production, same cuts.
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /work/submit/david_w/ZMass/calibration_studies/resolution/globalfit
RUNS=/work/submit/david_w/ZMass/calibration_studies/resolution/runs/hitlik
PROD=${PROD:-resolution_trackres_mugun_ul16_260903x_m0}
exec python3 -u extract.py \
  --files "/ceph/submit/data/user/d/david_w/ZMass/cvh/$PROD/task_*/globalcor_resclosure_*.root" \
  --parmtypes 14 15 --no-mass \
  --max-chi2-ndof 3 --max-hess 1e8 --max-grad 1e6 \
  -j ${J:-12} -o $RUNS/${OUT:-mugun_quad.npz} "$@"
