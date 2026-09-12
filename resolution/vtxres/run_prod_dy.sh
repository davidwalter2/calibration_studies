#!/bin/bash
# The Z-like check: the SAME vertex export on DY MC (MiniAOD), where the muons
# are 25-60 GeV instead of 2-10 and the vertex residual should be HIT-dominated
# rather than MS-dominated.  Config verbatim from
# `production/condor_dymc_v2/config_dymc_v2.sh` plus `exportVtxResidual=True`.
# usage: ./run_prod_dy.sh <nparallel> <from> <to>
set -uo pipefail
NPAR=${1:-10}
TASKS_FROM=${2:-0}
TASKS_TO=${3:-19}
CMSSW_AREA=${CMSSW_AREA:-/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2}
CFG=${CFG:-$CMSSW_AREA/src/Analysis/HitAnalyzer/test/runCvhDimuonMiniAOD.py}
RUN_ONE=/work/submit/david_w/ZMass/calibration_studies/slurm/run_one.sh
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
FILELIST=${FILELIST:-/work/submit/david_w/ZMass/calibration_studies/production/filelist_dymc_8p5M_260905.txt}
OUTROOT=${OUTROOT:-/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911}
OUTTAG=${OUTTAG:-dy}
export CMSSW_AREA
COMMON="nEvents=${NEVENTS:-4000} numberOfThreads=1 doRes=True exportCfExponents=True \
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
  sleep $(( (idx % ${NPAR:-10}) * ${STAGGER:-4} ))
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
export CFG RUN_ONE INIT FILELIST OUTROOT OUTTAG COMMON NPAR STAGGER
seq "$TASKS_FROM" "$TASKS_TO" | xargs -P "$NPAR" -I{} bash -c 'run_task {}'
echo "vtxres DY block $TASKS_FROM-$TASKS_TO finished"
