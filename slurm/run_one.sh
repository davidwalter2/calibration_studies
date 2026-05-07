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

cd /work/submit/david_w/ZMass/CMSSW_10_6_26/src
source /cvmfs/cms.cern.ch/cmsset_default.sh
eval $(scramv1 runtime -sh)

mkdir -p "$OUTDIR"
cd "$OUTDIR"

echo ">>> $(date) host=$(hostname)"
echo ">>> cfg=$CFG"
echo ">>> input=$INPUT"
echo ">>> outdir=$OUTDIR"
if [[ -n "${X509_USER_PROXY:-}" ]]; then
  TLEFT=$(voms-proxy-info -timeleft -file "$X509_USER_PROXY" 2>/dev/null || echo 0)
  echo ">>> proxy=$X509_USER_PROXY ($((TLEFT/3600))h ${TLEFT}s left)"
fi
echo ">>> cmsRun starting"

exec cmsRun "$CFG" input="$INPUT"
