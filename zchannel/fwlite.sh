#!/bin/bash
# Run a python script under the CMSSW 15_0 FWLite environment (read-only use of
# the release area).  The DY MiniAOD lives on /ceph/submit, so run this from a
# submit node that still has ceph mounted.
source /cvmfs/cms.cern.ch/cmsset_default.sh >/dev/null 2>&1
cd /work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src
eval $(scramv1 runtime -sh) 2>/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
exec python3 "$@"
