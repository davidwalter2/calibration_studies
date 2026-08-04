#!/bin/bash
# Scan (eta, phi) for a ray that crosses every module cleanly.
#
# The clean propagation test compares the true state to a deterministic
# reference ON A GIVEN PLANE, so every event must cross the same modules
# through their faces. A ray that grazes a module edge produces events that
# either miss that module or enter through its side, and dropping those is a
# selection effect -- precisely what this test exists to avoid. So we pick the
# ray empirically: the one whose (detid, entry local z) pattern is identical in
# the largest fraction of events.
#
# usage: ./scan_ray.sh [nevents] [outdir]
set -euo pipefail
NEV=${1:-500}
OUT=${2:-/ceph/submit/data/user/d/david_w/ZMass/cvh/cleanprop/rayscan}
CFG=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/test/runCleanPropSim.py
AREA=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev

mkdir -p "$OUT"
run_one() {
  local eta=$1 phi=$2
  local tag="eta${eta}_phi${phi}"
  local f="$OUT/scan_${tag}.root"
  [[ -s "$f" ]] && { echo "[skip] $tag"; return 0; }
  (
    cd "$AREA/src"
    source /cvmfs/cms.cern.ch/cmsset_default.sh
    eval "$(scramv1 runtime -sh)"
    cd "$OUT"
    cmsRun "$CFG" nEvents="$NEV" pt=10 eta="$eta" phi="$phi" seed=1 \
        output="$f" > "$OUT/scan_${tag}.log" 2>&1
  ) && echo "[done] $tag" || echo "[FAIL] $tag"
}
export -f run_one
export OUT CFG AREA NEV

for eta in 0.25 0.30 0.35 0.40 0.45 0.50; do
  for phi in 0.20 0.35 0.50; do
    echo "$eta $phi"
  done
done | xargs -P 18 -n 2 bash -c 'run_one "$0" "$1"'
echo "scan finished -> $OUT"
