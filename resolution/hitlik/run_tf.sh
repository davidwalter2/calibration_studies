#!/bin/bash
# Run a hitlik script in the rabbit TF environment against rabbit-vmass.
IMG=/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/bendavid/cmswmassdocker/wmassdevrolling:latest
RABBIT=${RABBIT:-/work/submit/david_w/ZMass/rabbit-vmass}
CS=/work/submit/david_w/ZMass/calibration_studies
export APPTAINERENV_PYTHONPATH=$RABBIT:/work/submit/david_w/WRemnants_dev/wums:$CS/resolution:$CS/resolution/matres:$CS/resolution/hitlik:$CS/resolution/globalfit
export APPTAINERENV_TF_CPP_MIN_LOG_LEVEL=2
export APPTAINERENV_OMP_NUM_THREADS=${OMP_NUM_THREADS:-24}
export APPTAINERENV_TF_NUM_INTRAOP_THREADS=${OMP_NUM_THREADS:-24}
export APPTAINERENV_TF_NUM_INTEROP_THREADS=2
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
export APPTAINERENV_RABBIT=$RABBIT
export APPTAINERENV_PYTHONUNBUFFERED=1
unset APPTAINER_BIND SINGULARITY_BIND
BINDS="-B /work/submit,/home/submit,/scratch/submit,/tmp"
# bind ceph whenever it is actually readable on this host (runs/perhit/{cards,perhit.npz}
# are symlinks into it).  submit82 has the mount but no permission -> the ls guard.
ls /ceph/submit >/dev/null 2>&1 && BINDS="$BINDS,/ceph/submit"
exec singularity exec $BINDS --pwd $CS/resolution/hitlik "$IMG" "$@"
