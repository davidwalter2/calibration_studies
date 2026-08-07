#!/bin/bash
# Rung-E private JPsi->mumu production in CMSSW_15_0 (same Geant4 as the
# CVH refit propagator), design GT / ideal geometry / DB grid field,
# PSimHits kept. Per task: GEN-SIM (step1) then DIGI..RECO (step2) with
# distinct random seeds and lumi blocks.
# usage: ./run_simprod.sh [nparallel] [task_from] [task_to] [nevents]
set -euo pipefail
NPAR=${1:-12}
FROM=${2:-0}
TO=${3:-11}
NEVT=${4:-4000}
CMSSW_AREA=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev
SIMPROD=/work/submit/david_w/ZMass/calibration_studies/resolution/simprod
OUTROOT=/ceph/submit/data/user/d/david_w/ZMass/cvh/resolution_simprod_260803

run_task() {
  local idx=$1
  local outdir="$OUTROOT/task_$(printf '%04d' "$idx")"
  [[ -s "$outdir/step2.root" ]] && { echo "[skip] task $idx"; return 0; }
  mkdir -p "$outdir"
  cd "$outdir"
  {
    cat "$SIMPROD/step1_gensim.py"
    echo ""
    echo "process.RandomNumberGeneratorService.generator.initialSeed = $((98700 + idx))"
    echo "process.RandomNumberGeneratorService.g4SimHits.initialSeed = $((13500 + idx))"
    echo "process.RandomNumberGeneratorService.VtxSmeared.initialSeed = $((77100 + idx))"
    echo "process.source.firstLuminosityBlock = cms.untracked.uint32($((idx + 1)))"
    echo "process.maxEvents.input = cms.untracked.int32($NEVT)"
  } > step1_task.py
  {
    cat "$SIMPROD/step2_digireco.py"
    echo ""
    echo "process.maxEvents.input = cms.untracked.int32(-1)"
  } > step2_task.py
  source /cvmfs/cms.cern.ch/cmsset_default.sh
  pushd "$CMSSW_AREA/src" > /dev/null
  eval "$(scramv1 runtime -sh)"
  popd > /dev/null
  echo "[run ] task $idx step1"
  cmsRun step1_task.py > step1.log 2>&1 || { echo "[FAIL] task $idx step1"; return 1; }
  echo "[run ] task $idx step2"
  # Write step2 to a temp name and rename on success. cmsRun creates its output
  # file at START, so a crashed/running task leaves a NON-EMPTY step2.root that
  # the `-s` skip test above would accept -- a resumed run would then silently
  # keep a truncated sample. (Same trap as the clean-prop sim outputs.)
  sed -i "s|fileName = cms.untracked.string('file:step2.root')|fileName = cms.untracked.string('file:step2.tmp.root')|" step2_task.py
  cmsRun step2_task.py > step2.log 2>&1 || { echo "[FAIL] task $idx step2"; rm -f step2.tmp.root; return 1; }
  mv -f step2.tmp.root step2.root
  rm -f step1.root   # keep only the RECO+simhits output
  echo "[done] task $idx"
}
export -f run_task
export CMSSW_AREA SIMPROD OUTROOT NEVT
seq "$FROM" "$TO" | xargs -P "$NPAR" -I{} bash -c 'run_task {}'
echo "all simprod tasks finished"
