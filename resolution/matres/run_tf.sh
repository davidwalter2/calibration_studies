#!/bin/bash
# Run a python script in the rabbit TensorFlow environment, against the
# `material-resolution` rabbit worktree.  Same image and shims as
# calibration_studies/env_tf/run_tf.sh; /ceph/submit is bound only when it is
# actually mounted (submit82's ceph is evicted and an unmounted bind aborts
# the container).
IMG=/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/bendavid/cmswmassdocker/wmassdevrolling:latest
RABBIT=/work/submit/david_w/ZMass/rabbit-material
CS=/work/submit/david_w/ZMass/calibration_studies
export APPTAINERENV_PYTHONPATH=$RABBIT:/work/submit/david_w/WRemnants_dev/wums:$CS/resolution:$CS/resolution/matres
export APPTAINERENV_TF_CPP_MIN_LOG_LEVEL=2
export APPTAINERENV_XLA_FLAGS="--xla_cpu_multi_thread_eigen=true"
export APPTAINERENV_OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
# PATH is NOT overridden: the image's python lives on it.  Prepend the
# rabbit bin in the command instead (PATH=$RABBIT/bin:$PATH rabbit_fit.py ...).
export APPTAINERENV_RABBIT=$RABBIT
BINDS="-B /work/submit,/home/submit,/scratch/submit,/tmp"
mountpoint -q /ceph/submit 2>/dev/null && BINDS="$BINDS,/ceph/submit"
exec singularity exec $BINDS --pwd $CS/resolution/matres "$IMG" "$@"
