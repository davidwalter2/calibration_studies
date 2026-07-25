#!/bin/bash
set -e
cd /work/submit/david_w/ZMass/BtoJpsiX_MCprod/validation_build/CMSSW_10_6_20_patch1/src
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc7_amd64_gcc700
eval $(scramv1 runtime -sh)
cd /work/submit/david_w/ZMass/calibration_studies/pixelhits/lagun
cmsRun gun100k_$1_cfg.py > run100k_$1_c${GUN_CHUNK}.log 2>&1
