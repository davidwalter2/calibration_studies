#!/bin/bash
# One split-99 -> split-1 repack, run INSIDE the el7 container from CMSSW_10_6_26_dev.
# The EDM splitLevel=0 of repack_generic_url.py maps to ROOT split level 1
# (EDM's level is offset +1 from ROOT's) -- see project_jpsi_mc_repack_split1.
# Writes to $REPACK_OUT.tmp.$$ and promotes only on cmsRun rc=0, so an
# interrupted repack can never leave a plausible-looking short file behind.
# usage: repack_worker.sh <in-path-or-url> <out-path>
set -uo pipefail
IN=${1:?in}; OUT=${2:?out}
TMP="$OUT.tmp.$$"
export REPACK_IN="$IN" REPACK_OUT="$TMP"
export X509_USER_PROXY=/home/submit/david_w/x509up_u$(id -u)
cd /work/submit/david_w/ZMass/CMSSW_10_6_26_dev/src
source /cvmfs/cms.cern.ch/cmsset_default.sh
eval $(scramv1 runtime -sh)
echo "[repack] IN=$IN"
echo "[repack] OUT=$OUT (tmp $TMP)"
date -Is
cmsRun /work/submit/david_w/ZMass/repack/repack_generic_url.py
RC=$?
date -Is
echo "[repack] cmsRun rc=$RC"
if [[ $RC -ne 0 ]]; then rm -f "$TMP"; echo "[repack] FAILED"; exit $RC; fi
[[ -s "$TMP" ]] || { echo "[repack] EMPTY OUTPUT"; rm -f "$TMP"; exit 9; }
mv "$TMP" "$OUT"
echo "[repack] OK $(stat -c %s "$OUT") B  $OUT"
