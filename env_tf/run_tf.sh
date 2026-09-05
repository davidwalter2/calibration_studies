#!/bin/bash
# Run a python script in the rabbit TensorFlow environment (wmassdev singularity
# image used by rabbit; see /work/submit/david_w/rabbit_260826_tfMinimizer/README.md).
# `wums` is not in the image and is added from the mfs venv via pypath/wums symlink.
IMG=/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/bendavid/cmswmassdocker/wmassdevrolling:latest
export APPTAINERENV_PYTHONPATH=/work/submit/david_w/ZMass/calibration_studies/env_tf/pypath:/work/submit/david_w/ZMass/calibration_studies/resolution
export APPTAINERENV_TF_CPP_MIN_LOG_LEVEL=2
export APPTAINERENV_XLA_FLAGS="--xla_cpu_multi_thread_eigen=true"
export APPTAINERENV_OMP_NUM_THREADS=32
# do not drop .pyc into the mfs venv that the wums shim points into
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
exec singularity exec -B /work/submit,/home/submit,/ceph/submit,/scratch/submit \
     --pwd /work/submit/david_w/ZMass/calibration_studies/resolution "$IMG" "$@"
