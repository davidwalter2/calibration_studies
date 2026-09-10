#!/bin/bash
# The per-hit (complement) residual production.
#
# Same input sample and the same selection as
# `resolution_trackres_mugun_ul16_260903x_m0` (the production the
# truth-referenced prototype ran on), so the two are directly comparable:
# 160 tasks of the UL16 20-60 GeV mu gun, both charges, |eta| < 2.4,
# generalTracks, ideal geometry, default field, CgfQoPMode=0,
# fitFromGenParms=False.  What is different is only the export:
#
#   exportPerHitResidual=True   the d = n_meas - 5 whitened complement
#                               components + 5 truth-referenced ones
#   perHitCfGroups=True         their per-material-group CF exponents
#   exportStepRecords=False     the 430 kB/track raw records are no longer
#                               needed -- the exponents are in the maker now
#   exportCfGroupExponents=False redundant: reference component 0 IS the q/p
#                               functional, and it carries its own group split
#
# usage: ./run_prod.sh [nparallel] [task_from] [task_to]
set -euo pipefail
NPAR=${1:-60}
FROM=${2:-0}
TO=${3:-159}
RES=/work/submit/david_w/ZMass/calibration_studies/resolution
export CFG=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2/src/Analysis/HitAnalyzer/test/runCvhResClosure.py
export CMSSW_AREA=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2
export FILELIST=$RES/simprod/filelist_mugun_ul16.txt
export OUTTAG=${OUTTAG:-mugun_ul16_260910_perhit}
export STAGGER=${STAGGER:-3}
export EXTRA="trackSrc=generalTracks useDefaultField=True useIdealGeometry=True \
globalTag=150X_mcRun2_asymptotic_v1 CgfQoPMode=0 \
exportPerHitResidual=True perHitCfGroups=True perHitRefComponents=True \
exportStepRecords=False exportCfExponents=True exportCfGroupExponents=False"
cd $RES
exec ./run_local_trackres.sh "$NPAR" "$FROM" "$TO"
