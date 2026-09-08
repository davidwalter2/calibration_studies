#!/bin/bash
# Run a command on submit50 (this session's node, submit82, has been evicted
# by cephx and cannot read /ceph, and its glibc does not match the mfs venv).
exec ssh -o BatchMode=yes -o StrictHostKeyChecking=no \
  -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm/%r@%h:%p -o ControlPersist=8h \
  submit50.mit.edu "source /work/submit/david_w/ZMass/mfs/.venv/bin/activate && \
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
           NUMEXPR_NUM_THREADS=1 CVH_IONI_KOKOULIN=0 && \
    cd /work/submit/david_w/ZMass/calibration_studies/resolution/hitclassbias && $*"
