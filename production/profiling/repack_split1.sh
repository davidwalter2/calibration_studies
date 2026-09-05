#!/bin/bash
# Repack an ALCARECO to ROOT split-level 1 (splitLevel=0 +
# overrideInputFileSplitLevels) in CMSSW_10_6_26.  A 10_6-written,
# split-99 ALCARECO deserialises its SiStripClusters incorrectly in
# 15_X (ROOT #19773) -- see MEMORY project_cvh_15_0_ierr3_diagnosis.
set -uo pipefail
export REPACK_IN=$1 REPACK_OUT=$2 REPACK_N=${3:--1}
source /cvmfs/cms.cern.ch/cmsset_default.sh >/dev/null 2>&1
cd /work/submit/david_w/ZMass/CMSSW_10_6_26/src
eval "$(scramv1 runtime -sh)" 2>/dev/null
cd "$(dirname "$REPACK_OUT")"
cmsRun /work/submit/david_w/ZMass/calibration_studies/production/profiling/repack_n.py
