#!/bin/bash
# Production runner for the ground-truth side of the clean propagation test:
# the SAME fixed-state particle simulated many times, split across tasks that
# differ ONLY in the Geant4 seed.
#
# usage: ./run_cleanprop_sim.sh [ntasks] [nev_per_task] [eta] [phi] [pt] [tag] [partId]
#
# The tasks are statistically independent samples of the identical propagation
# kernel, so cf_propagation_test.py just globs them together.
#
# partId is a PDG id: 13 = mu- (default), -321 = K-, -211 = pi-. The output
# directory carries the species so a scan over (pt, eta, species) never
# collides; muon output keeps the original naming for back-compatibility.
set -euo pipefail
NTASK=${1:-64}
NEV=${2:-8000}
ETA=${3:-0.30}
PHI=${4:-0.35}
PT=${5:-10}
TAG=${6:-$(date +%y%m%d)}
PARTID=${7:-13}
# NPAR: 100 concurrent jobs has run clean; 150 produced segfaults in ~64% of
# tasks (not reproducible in isolation with the same seed, i.e. a resource
# effect, not physics). Keep it at or below 100.
NPAR=${NPAR:-64}

CFG=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/test/runCleanPropSim.py
AREA=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev
OUT=/ceph/submit/data/user/d/david_w/ZMass/cvh/cleanprop/sim_${TAG}_pt${PT}_eta${ETA}_phi${PHI}
[[ "$PARTID" != "13" ]] && OUT="${OUT}_pdg${PARTID}"
mkdir -p "$OUT"

# Write to a temp name and rename only on success, so the presence of the
# final file MEANS complete. Previously cmsRun wrote the final name directly
# and the skip test was `-s`, so a crashed task left a short-but-non-empty
# file that a re-run then SKIPPED -- i.e. exactly the tasks that failed were
# the ones not retried. (Same trap as the partial targets file.)
run_task() {
  local idx=$1
  local f="$OUT/simstates_$(printf '%04d' "$idx").root"
  local tmp="$OUT/.tmp_simstates_$(printf '%04d' "$idx").root"
  [[ -s "$f" ]] && { echo "[skip] $idx"; return 0; }
  rm -f "$tmp"
  (
    cd "$AREA/src"
    source /cvmfs/cms.cern.ch/cmsset_default.sh
    eval "$(scramv1 runtime -sh)"
    cd "$OUT"
    cmsRun "$CFG" nEvents="$NEV" pt="$PT" eta="$ETA" phi="$PHI" seed="$((idx + 1))" \
        partId="$PARTID" \
        output="$tmp" > "$OUT/task_$(printf '%04d' "$idx").log" 2>&1
  ) && { mv -f "$tmp" "$f"; echo "[done] $idx"; } || { rm -f "$tmp"; echo "[FAIL] $idx"; }
}
export -f run_task
export OUT CFG AREA NEV PT ETA PHI PARTID CLEANPROP_LOOSE_STEPPER

echo "output -> $OUT   ($NTASK tasks x $NEV events = $((NTASK * NEV)))"
seq 0 $((NTASK - 1)) | xargs -P "$NPAR" -I{} bash -c 'run_task {}'
echo "all sim tasks finished -> $OUT"
