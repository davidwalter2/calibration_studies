#!/bin/bash
# Kink-finder validation passes over the v3 inclusive B -> J/psi + X MC
# (2016postVFP ALCARECO), the first campaign that keeps the Geant4
# SimTracks + SimVertices needed for decay-in-flight truth.
#
# One pass per true species (the G4e propagator needs the right mass
# hypothesis for energy loss, so the species cannot be mixed in one job):
#   kaon  -> the B -> J/psi K calibration track
#   pi    -> same-topology cross-check, 7.4x smaller decay probability
#   mu    -> null (no decays, no nuclear interactions)
#
# Runs a local parallel pool on submit82 (768 cores); the per-chunk jobs
# are single-threaded because the Geant4e propagator is not thread safe.
#
# usage: ./run_v3_species.sh [kaon|pi|mu] <nfiles> [nworkers] [chunksize]
set -euo pipefail

PARTICLE=${1:?particle: kaon|pi|mu}
NFILES=${2:?number of input files}
NWORKERS=${3:-60}
CHUNK=${4:-50}
# optional: subdirectory tag and extra cmsRun arguments, e.g.
#   ./run_v3_species.sh kaon 10000 40 50 fakemu "fitAs=mu doMuons=True"
TAG=${5:-}
EXTRA=${6:-}

CMSSW_AREA=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev
CFG=$CMSSW_AREA/src/Analysis/HitAnalyzer/test/runCvhSingleTrackJpsiX.py
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
MCDIR=/ceph/submit/data/group/cms/store/mc/inclusive_btojpsix_2016postvfp_v3
DATE_TAG=$(date +%y%m%d)
OUTBASE=/ceph/submit/data/user/d/david_w/ZMass/cvh/kinkfinder_v3${TAG:+_$TAG}_${DATE_TAG}/${PARTICLE}

mkdir -p "$OUTBASE"/logs

# Stable, reproducible file selection: the production is still growing, so
# sort by name and take the first NFILES rather than whatever `find` returns.
# NB do not pipe into `head` here: under `set -o pipefail` the SIGPIPE that
# head sends back up the pipeline aborts the script.
FILELIST=$OUTBASE/inputfiles.txt
if [ ! -s "$FILELIST" ]; then
    find "$MCDIR" -maxdepth 1 -name '*.root' | sort > "$OUTBASE/allfiles.txt"
    head -n "$NFILES" "$OUTBASE/allfiles.txt" > "$FILELIST"
fi
NAVAIL=$(wc -l < "$FILELIST")
echo "[run] $PARTICLE: $NAVAIL files, chunks of $CHUNK, $NWORKERS workers -> $OUTBASE"

# Split into comma-joined chunks (the cfg splits input= on commas).
CHUNKLIST=$OUTBASE/chunks.txt
awk -v n="$CHUNK" '{ if (NR % n == 1) { if (NR > 1) print line; line = $0 } else line = line "," $0 } END { print line }' \
    "$FILELIST" > "$CHUNKLIST"
NCHUNK=$(wc -l < "$CHUNKLIST")
echo "[run] $NCHUNK chunks"

run_chunk() {
    local idx=$1 files=$2
    local wd="$OUTBASE/chunk_$(printf '%04d' "$idx")"
    # Resume on the .done marker, NOT on the output file: cmsRun creates the
    # ROOT file at start, so a job killed midway (e.g. the transient ceph
    # FileOpenError seen while the production is still writing) leaves a
    # non-empty but truncated file that must be redone, not skipped.
    if [ -f "$wd/.done" ]; then return 0; fi
    mkdir -p "$wd"
    rm -f "$wd/globalcor_jpsix_${PARTICLE}_0.root"
    if ( cd "$wd" && cmsRun "$CFG" \
        input="$files" particle="$PARTICLE" nEvents=-1 \
        scalarPot3DInitFile="$INIT" doKinkFinder=True doSimDecayTruth=True $EXTRA \
        > "$OUTBASE/logs/chunk_$(printf '%04d' "$idx").log" 2>&1 ); then
        touch "$wd/.done"
    else
        echo "[run] FAILED chunk $idx" >> "$OUTBASE/logs/failures.txt"
    fi
}
export -f run_chunk
export OUTBASE CFG INIT PARTICLE EXTRA

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd "$CMSSW_AREA/src"
eval "$(scramv1 runtime -sh)"

nl -ba "$CHUNKLIST" | xargs -P "$NWORKERS" -L1 bash -c 'run_chunk "$0" "$1"'

NOUT=$(find "$OUTBASE" -name "globalcor_jpsix_${PARTICLE}_0.root" | wc -l)
echo "[run] done: $NOUT / $NCHUNK chunk outputs in $OUTBASE"
[ -f "$OUTBASE/logs/failures.txt" ] && cat "$OUTBASE/logs/failures.txt" || true
