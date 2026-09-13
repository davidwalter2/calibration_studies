#!/bin/bash
# Run a python script in the rabbit TensorFlow environment with the rabbit
# worktree (`rabbit-vmass`, branch `vmass-conditioning`) first on PYTHONPATH.
# No /ceph bind, so it works
# on a node whose ceph client is evicted; pass --ceph as the first argument to
# add it back.
IMG=/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/bendavid/cmswmassdocker/wmassdevrolling:latest
BINDS="/work/submit,/home/submit,/scratch/submit,/tmp,/cvmfs"
if [ "${1:-}" = "--ceph" ]; then BINDS="$BINDS,/ceph/submit"; shift; fi
export APPTAINERENV_PYTHONPATH=/work/submit/david_w/ZMass/rabbit-vmass:\
/work/submit/david_w/WRemnants_dev/wums:\
/work/submit/david_w/ZMass/calibration_studies/env_tf/pypath:\
/work/submit/david_w/ZMass/calibration_studies/resolution
export APPTAINERENV_TF_CPP_MIN_LOG_LEVEL=2
export APPTAINERENV_OMP_NUM_THREADS=16
export APPTAINERENV_TF_NUM_INTRAOP_THREADS=16
export APPTAINERENV_TF_NUM_INTEROP_THREADS=2
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
exec singularity exec -B "$BINDS" \
     --pwd /work/submit/david_w/ZMass/calibration_studies/zchannel "$IMG" "$@"
