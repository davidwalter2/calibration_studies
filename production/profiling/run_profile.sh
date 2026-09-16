#!/bin/bash
# One profiling point: cmsRun under /usr/bin/time -v, into its own dir.
# usage: run_profile.sh <tag> <outdir> <extra cmsRun args...>
# NOTE: CMSSW's MessageLogger (and hence TimeReport / TimeEvent> /
# TimeModule>) writes to cerr, the maker's own summaries to cout -- merge
# both into run.log and keep /usr/bin/time in its own file.
set -uo pipefail
TAG=$1; OUT=$2; shift 2
CFG=/work/submit/david_w/ZMass/calibration_studies/production/profiling/runCvhProfile.py
AREA=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2
mkdir -p "$OUT/$TAG"; cd "$OUT/$TAG" || exit 1
rm -f .done
source /cvmfs/cms.cern.ch/cmsset_default.sh >/dev/null 2>&1
cd "$AREA/src" && eval "$(scramv1 runtime -sh)" 2>/dev/null
cd "$OUT/$TAG" || exit 1
echo "ARGS: $*" > args.txt
/usr/bin/time -v -o time.log cmsRun "$CFG" "$@" > run.log 2>&1
echo "rc=$?" >> args.txt
touch .done
