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
# The cleanliness depends on pT (the trajectory bends) and on eta (which
# modules are crossed at all), so the scan must be redone for every (pt, eta)
# of the campaign -- a ray that is 99.6% clean for a 10 GeV muon says nothing
# about a 2 GeV one.
#
# usage: ./scan_ray.sh [nevents] [pt] [etalist] [philist] [partId] [outdir]
#   etalist/philist are space-separated, quoted:
#     ./scan_ray.sh 500 40 "0.30" "0.10 0.20 0.30 0.40 0.50"
#
# Rank the result with:  python rank_rays.py '<outdir>/scan_pt*<...>.root'
set -euo pipefail
NEV=${1:-500}
PT=${2:-10}
ETALIST=${3:-"0.25 0.30 0.35 0.40 0.45 0.50"}
PHILIST=${4:-"0.20 0.35 0.50"}
PARTID=${5:-13}
OUT=${6:-/ceph/submit/data/user/d/david_w/ZMass/cvh/cleanprop/rayscan}
NPAR=${NPAR:-18}

CFG=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/test/runCleanPropSim.py
AREA=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev

mkdir -p "$OUT"
run_one() {
  local eta=$1 phi=$2
  local tag="pt${PT}_eta${eta}_phi${phi}_pdg${PARTID}"
  local f="$OUT/scan_${tag}.root"
  [[ -s "$f" ]] && { echo "[skip] $tag"; return 0; }
  (
    cd "$AREA/src"
    source /cvmfs/cms.cern.ch/cmsset_default.sh
    eval "$(scramv1 runtime -sh)"
    cd "$OUT"
    cmsRun "$CFG" nEvents="$NEV" pt="$PT" eta="$eta" phi="$phi" seed=1 \
        partId="$PARTID" output="$f" > "$OUT/scan_${tag}.log" 2>&1
  ) && echo "[done] $tag" || echo "[FAIL] $tag"
}
export -f run_one
export OUT CFG AREA NEV PT PARTID

for eta in $ETALIST; do
  for phi in $PHILIST; do
    echo "$eta $phi"
  done
done | xargs -P "$NPAR" -n 2 bash -c 'run_one "$0" "$1"'
echo "scan finished -> $OUT"
