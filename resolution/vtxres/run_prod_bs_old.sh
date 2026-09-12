#!/bin/bash
# The SAME beam-rows-ON configuration run against the OLD (double-emitting)
# build in `CMSSW_15_0_19_patch2_dev2`, which is the reference for gate G4:
# two identical rows with covariance S are one row with S/2, so the old build
# at nominal widths must equal the new build at beamWidthScale = 1/sqrt(2).
#
# NOTHING is written into dev2: the cfg is a sed-patched COPY of its driver
# (which has no `bsConstraint` VarParsing knob) placed in the job temp dir,
# and dev2 is only SOURCED, never built.
#
#   usage: ./run_prod_bs_old.sh <nparallel> <from> <to>
set -uo pipefail
NPAR=${1:-6}
TASKS_FROM=${2:-0}
TASKS_TO=${3:-5}
OLDAREA=${OLDAREA:-/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2}
TMP=${TMP:-/home/submit/david_w/.claude-work/jobs/28e0dfa8/tmp}
CFG=$TMP/runCvhDimuonMiniAOD_bson.py
mkdir -p "$TMP"
SRC=$OLDAREA/src/Analysis/HitAnalyzer/test/runCvhDimuonMiniAOD.py
grep -q 'exportVtxResidual=cms.bool(bool(opts.exportVtxResidual)),' "$SRC" || {
  echo "[err] anchor line not found in $SRC"; exit 2; }
sed 's|    exportVtxResidual=cms.bool(bool(opts.exportVtxResidual)),|    exportVtxResidual=cms.bool(bool(opts.exportVtxResidual)),\n    bsConstraint=cms.bool(True),|' \
  "$SRC" > "$CFG"
grep -c 'bsConstraint=cms.bool(True)' "$CFG"

CMSSW_AREA=$OLDAREA
RUN_ONE=/work/submit/david_w/ZMass/calibration_studies/slurm/run_one.sh
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
FILELIST=${FILELIST:-/work/submit/david_w/ZMass/calibration_studies/production/filelist_dymc_8p5M_260905.txt}
OUTROOT=${OUTROOT:-/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/beamline}
OUTTAG=${OUTTAG:-dy_bsold}
export CMSSW_AREA
COMMON="nEvents=${NEVENTS:-1200} numberOfThreads=1 doRes=True exportCfExponents=True \
exportStepRecords=False exportCfGroupExponents=True \
exportMaterialNoise=True exportVarianceGrads=True varianceGradFamilies=15 \
fillJac=True fillGrads=False fillGradsFactored=True \
fitFromGenParms=False doSimHits=False doGen=True requireGen=False \
genParticles=prunedGenParticles pileupInfo=slimmedAddPileupInfo \
doTrigger=False applyHltFilter=False massMin=60 massMax=120 \
useIdealGeometry=False useDefaultField=True globalTag=106X_mcRun2_asymptotic_v17 \
doMassConstraint=False CgfQoPMode=0 tightG4eStepper=True \
propagationPtotLimit=0.2 maxMomentumStepFactor=2.0 stepBacktracking=True \
scalarPot3DInitFile=$INIT exportVtxResidual=True ${EXTRA:-}"

run_task() {
  local idx=$1
  local outdir="$OUTROOT/$OUTTAG/task_$(printf '%04d' "$idx")"
  if [[ -f "$outdir/.complete" ]]; then echo "[skip] task $idx"; return 0; fi
  sleep $(( (idx % ${NPAR:-6}) * ${STAGGER:-4} ))
  local input; input=$(sed -n "$((idx + 1))p" "$FILELIST")
  [[ -n "$input" ]] || { echo "[err] empty filelist line $idx"; return 1; }
  mkdir -p "$outdir"; rm -f "$outdir"/globalcor_*.root "$outdir/.complete"
  echo "[run ] task $idx -> $outdir"
  # shellcheck disable=SC2086
  if "$RUN_ONE" "$CFG" "$input" "$outdir" $COMMON > "$outdir/local.log" 2>&1; then
    touch "$outdir/.complete"; echo "[done] task $idx"
  else
    echo "[FAIL] task $idx (see $outdir/local.log)"
  fi
}
export -f run_task
export CFG RUN_ONE INIT FILELIST OUTROOT OUTTAG COMMON NPAR STAGGER CMSSW_AREA
seq "$TASKS_FROM" "$TASKS_TO" | xargs -P "$NPAR" -I{} bash -c 'run_task {}'
echo "beamline OLD-build block $TASKS_FROM-$TASKS_TO ($OUTTAG) finished"
