#!/bin/bash
# Two-phase launcher for the DYJetsToMuMu UL16 MiniAODv2 grid->ceph copy.
#
# Phase 1 pulls the 104-file / 8.5M-event priority subset so the Z ditrack
# feasibility test can start while the rest is still in flight; phase 2 then
# walks the complete 1731-file dataset and skips whatever phase 1 already
# published.  Both phases share one status file, so the whole thing is a single
# resumable job: re-running this script after any interruption continues where
# it stopped.
#
#   nohup ./run_dy_transfer.sh > logs/transfer_<date>.log 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export X509_USER_PROXY="${X509_USER_PROXY:-/tmp/x509up_u$(id -u)}"
source /cvmfs/cms.cern.ch/cmsset_default.sh >/dev/null 2>&1
export STATUS="${STATUS:-$HERE/transfer_status.tsv}"
export NSTREAM="${NSTREAM:-12}"
export SUBSTREAMS="${SUBSTREAMS:-4}"

t0=$SECONDS
echo "########## PHASE 1: 8.5M-event priority subset ##########"
"$HERE/transfer_dy.sh" "$HERE/dy_miniaod_subset.tsv"
[[ $? -eq 4 ]] && { echo "ABORT: destination storage unreachable"; exit 4; }

echo
echo "########## writing early filelists from the priority subset ##########"
python3 "$HERE/make_filelists.py" --manifest "$HERE/dy_miniaod_subset.tsv" || true

echo
echo "########## PHASE 2: complete dataset (1731 files) ##########"
"$HERE/transfer_dy.sh" "$HERE/dy_miniaod_full.tsv"
[[ $? -eq 4 ]] && { echo "ABORT: destination storage unreachable"; exit 4; }

echo
echo "########## final accounting ##########"
python3 "$HERE/make_filelists.py" --manifest "$HERE/dy_miniaod_full.tsv" \
        --out-prefix filelist_dy_miniaod_full_260905 || true
"$HERE/report.sh" || true
echo "TOTAL WALL: $((SECONDS - t0))s"
