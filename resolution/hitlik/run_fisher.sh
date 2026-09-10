#!/bin/bash
cd /work/submit/david_w/ZMass/calibration_studies/resolution/hitlik
R=/work/submit/david_w/ZMass/calibration_studies/resolution/runs/hitlik
OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} exec ./run_tf.sh python3 -u fisher_cmp.py \
  --npz $R/mugun20k.npz --arms cf gauss gaussq --comps 0 0123 \
  --chunk 8192 "$@"
