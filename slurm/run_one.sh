#!/bin/bash
# Inner per-task runner: runs cmsRun for a single input file inside cmssw-el7.
# Invoked with the el7 environment already entered (i.e. by array.sbatch via
# `cmssw-el7 --command-to-run`).
#
# usage: run_one.sh <cfg> <input> <outdir>
set -euo pipefail

CFG=$1
INPUT=$2
OUTDIR=$3
shift 3   # any further args (extra VarParsing knobs) get appended to cmsRun

# CMSSW area to source. submit.sh exports this; default keeps the script
# usable interactively without it.
: "${CMSSW_AREA:=/work/submit/david_w/ZMass/CMSSW_10_6_26}"

cd "${CMSSW_AREA}/src"
source /cvmfs/cms.cern.ch/cmsset_default.sh
# Only force the slc7 arch for legacy el7 areas (10_6, run inside
# cmssw-el7); el9-native areas (15_0+) let scram pick the host arch.
if compgen -G "${CMSSW_AREA}/lib/slc7_*" > /dev/null; then
  export SCRAM_ARCH=slc7_amd64_gcc700
fi
eval $(scramv1 runtime -sh)

mkdir -p "$OUTDIR"
cd "$OUTDIR"

echo ">>> $(date) host=$(hostname)"
echo ">>> cmssw=$CMSSW_AREA"
echo ">>> cfg=$CFG"
echo ">>> input=$INPUT"
echo ">>> outdir=$OUTDIR"
if [[ -n "${X509_USER_PROXY:-}" ]]; then
  TLEFT=$(voms-proxy-info -timeleft -file "$X509_USER_PROXY" 2>/dev/null || echo 0)
  echo ">>> proxy=$X509_USER_PROXY ($((TLEFT/3600))h ${TLEFT}s left)"
fi
echo ">>> extra cmsRun args: $*"
echo ">>> cmsRun starting"

exec cmsRun "$CFG" input="$INPUT" "$@"
