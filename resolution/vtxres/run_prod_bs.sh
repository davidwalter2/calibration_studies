#!/bin/bash
# The BEAM-LINE leg: the same DY MiniAOD configuration as `run_prod_dy.sh`
# (which is `production/condor_dymc_v2`'s verbatim, plus exportVtxResidual)
# with the LUMINOUS-REGION rows on and the two transverse beam-line residuals
# exported.  `dy_vtxon` is the rows-OFF reference; this script's `dy_bs` is
# the rows-ON one, on the SAME six input files so every comparison is
# same-candidate.
#
#   usage: ./run_prod_bs.sh <nparallel> <from> <to>
#   env:   OUTTAG (dy_bs), EXTRA (extra cmsRun knobs), NEVENTS, CMSSW_AREA
#
# The GATE configurations, same inputs, set through EXTRA/OUTTAG:
#   OUTTAG=dy_bs                                    nominal widths
#   OUTTAG=dy_bsoff  BSOPTS='bsConstraint=False'    the rows-OFF reference,
#                       SAME build -- which `dy_vtxon` is not (it predates
#                       `Jpsi_covvtx`, which the leave-one-out gate needs)
#   OUTTAG=dy_bswide  EXTRA='beamWidthScale=1e6'    weightless rows == rows OFF
#   OUTTAG=dy_bshalf  EXTRA='beamWidthScale=0.7071067811865476'
#                       == the OLD double-emitted rows (two identical rows
#                          with covariance S are one row with S/2), which is
#                          how the double-emission defect is measured.
set -uo pipefail
NPAR=${1:-6}
TASKS_FROM=${2:-0}
TASKS_TO=${3:-5}
CMSSW_AREA=${CMSSW_AREA:-/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2}
CFG=${CFG:-$CMSSW_AREA/src/Analysis/HitAnalyzer/test/runCvhDimuonMiniAOD.py}
RUN_ONE=/work/submit/david_w/ZMass/calibration_studies/slurm/run_one.sh
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
FILELIST=${FILELIST:-/work/submit/david_w/ZMass/calibration_studies/production/filelist_dymc_8p5M_260905.txt}
OUTROOT=${OUTROOT:-/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/beamline}
OUTTAG=${OUTTAG:-dy_bs}
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
scalarPot3DInitFile=$INIT exportVtxResidual=True \
${BSOPTS:-bsConstraint=True exportBsResidual=True} ${EXTRA:-}"

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
export CFG RUN_ONE INIT FILELIST OUTROOT OUTTAG COMMON NPAR STAGGER BSOPTS
seq "$TASKS_FROM" "$TASKS_TO" | xargs -P "$NPAR" -I{} bash -c 'run_task {}'
echo "beamline block $TASKS_FROM-$TASKS_TO ($OUTTAG) finished"
