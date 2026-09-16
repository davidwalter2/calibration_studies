#!/bin/bash
# hitlik extraction. Must run on a node whose ceph client is alive (submit50/51).
#   run_extract.sh <output.npz> [extra extract_res5.py args]
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CVH_IONI_KOKOULIN=0
cd /work/submit/david_w/ZMass/calibration_studies/resolution/hitlik
OUT=$1; shift
PROD=${PROD:-resolution_trackres_mugun_ul16_260903x_m0}
GRP=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2/src/Analysis/HitAnalyzer/data/materialGroups50.txt
exec python3 -u extract_res5.py \
  --files "/ceph/submit/data/user/d/david_w/ZMass/cvh/$PROD/task_*/globalcor_resclosure_*.root" \
  --groups $GRP --max-chi2-ndof 3 --max-hess 1e8 --max-grad 1e6 \
  --decimate 4 --tmax 8 -o "$OUT" "$@"
