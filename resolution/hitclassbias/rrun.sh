#!/bin/bash
# Run a command on submit50, which can read /ceph and whose glibc matches the
# mfs venv. Some submit nodes (submit82 among them) are evicted by cephx and
# cannot read /ceph at all, so the analysis must not be run on them directly.
exec ssh -o BatchMode=yes -o StrictHostKeyChecking=no \
  -o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm/%r@%h:%p -o ControlPersist=8h \
  submit50.mit.edu "source /work/submit/david_w/ZMass/mfs/.venv/bin/activate && \
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
           NUMEXPR_NUM_THREADS=1 CVH_IONI_KOKOULIN=0 && \
    cd /work/submit/david_w/ZMass/calibration_studies/resolution/hitclassbias && $*"
