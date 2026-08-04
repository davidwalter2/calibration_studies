#!/bin/bash
# Production runner for the ground-truth side of the clean propagation test:
# the SAME fixed-state particle simulated many times, split across tasks that
# differ ONLY in the Geant4 seed.
#
# usage: ./run_cleanprop_sim.sh [ntasks] [nev_per_task] [eta] [phi] [pt] [tag]
#
# The tasks are statistically independent samples of the identical propagation
# kernel, so cf_propagation_test.py just globs them together.
set -euo pipefail
NTASK=${1:-64}
NEV=${2:-8000}
ETA=${3:-0.30}
PHI=${4:-0.35}
PT=${5:-10}
TAG=${6:-$(date +%y%m%d)}
NPAR=${NPAR:-64}

CFG=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/test/runCleanPropSim.py
AREA=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev
OUT=/ceph/submit/data/user/d/david_w/ZMass/cvh/cleanprop/sim_${TAG}_pt${PT}_eta${ETA}_phi${PHI}
mkdir -p "$OUT"

run_task() {
  local idx=$1
  local f="$OUT/simstates_$(printf '%04d' "$idx").root"
  [[ -s "$f" ]] && { echo "[skip] $idx"; return 0; }
  (
    cd "$AREA/src"
    source /cvmfs/cms.cern.ch/cmsset_default.sh
    eval "$(scramv1 runtime -sh)"
    cd "$OUT"
    cmsRun "$CFG" nEvents="$NEV" pt="$PT" eta="$ETA" phi="$PHI" seed="$((idx + 1))" \
        output="$f" > "$OUT/task_$(printf '%04d' "$idx").log" 2>&1
  ) && echo "[done] $idx" || echo "[FAIL] $idx"
}
export -f run_task
export OUT CFG AREA NEV PT ETA PHI

echo "output -> $OUT   ($NTASK tasks x $NEV events = $((NTASK * NEV)))"
seq 0 $((NTASK - 1)) | xargs -P "$NPAR" -I{} bash -c 'run_task {}'
echo "all sim tasks finished -> $OUT"
