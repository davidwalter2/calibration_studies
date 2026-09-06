#!/bin/bash
# Run a python script of `fullscale/` in the rabbit TF environment, with the
# MERGED rabbit worktree first on PYTHONPATH.
#
# The merged branch is `material-resolution` after `z-lineshape-kernel` was
# merged into it: the corrections and MaterialCFTerm of the first, the
# provider/norm_window/upsample of the second. Neither parent alone can build
# this card.
#
#   RABBIT=<path>   override the worktree (e.g. the merge sandbox)
#   THREADS=<n>     intra-op threads (default 32)
#   --ceph          add the /ceph bind (only the pairs extraction needs it)
IMG=/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/bendavid/cmswmassdocker/wmassdevrolling:latest
RABBIT=${RABBIT:-/work/submit/david_w/ZMass/rabbit-material}
THREADS=${THREADS:-32}
BINDS="/work/submit,/home/submit,/scratch/submit,/tmp,/cvmfs"
if [ "${1:-}" = "--ceph" ]; then BINDS="$BINDS,/ceph/submit"; shift; fi
export APPTAINERENV_PYTHONPATH=$RABBIT:\
/work/submit/david_w/WRemnants_dev/wums:\
/work/submit/david_w/ZMass/calibration_studies/env_tf/pypath:\
/work/submit/david_w/ZMass/calibration_studies/resolution
export APPTAINERENV_TF_CPP_MIN_LOG_LEVEL=2
export APPTAINERENV_OMP_NUM_THREADS=$THREADS
export APPTAINERENV_TF_NUM_INTRAOP_THREADS=$THREADS
export APPTAINERENV_TF_NUM_INTEROP_THREADS=2
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
exec singularity exec -B "$BINDS" \
     --pwd /work/submit/david_w/ZMass/calibration_studies/fullscale "$IMG" "$@"
