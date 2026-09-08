#!/bin/bash
# The full-scale PHASE-3 card (STATE sec. 10). ~36 GB, ~100 GB RAM, hours.
set -uo pipefail
FS=/work/submit/david_w/ZMass/calibration_studies/fullscale
Z=/work/submit/david_w/ZMass/calibration_studies/zchannel
GRP=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/data/materialGroups50.txt
export PYTHONPATH=/work/submit/david_w/ZMass/rabbit-vmass:/work/submit/david_w/WRemnants_dev/wums:$FS/../env_tf/pypath:$FS/../resolution
export TF_CPP_MIN_LOG_LEVEL=2 OMP_NUM_THREADS=8 TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=2
/opt/venv/bin/python3 -u $FS/make_joint_card.py --material \
  --jpsi-pairs $FS/runs/gpairs_v2_n50.npz \
  --z-pairs    $FS/runs/gzpairs_dyv2_n50.npz \
  --quad $FS/runs/quad_jpsiv2_ok.npz $FS/runs/quad_dyv2.npz \
  --groups $GRP --whiten --shape 5 \
  --fsr $Z/data/kern_loose_band3.3e-4.npz --acc $Z/data/acc_loose_d8.json \
  --chunk 16384 -o $FS/cards/joint_mat_v3.hdf5
