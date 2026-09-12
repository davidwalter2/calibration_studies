#!/bin/bash
# The gen-background leg of the vertex-residual study, and the gate on the
# minimum-size cut.
#
#   gate-gun   200 J/psi-gun events through the NEW build, same input and the
#              same settings as `prod_vtxon/task_0000`, so every surviving
#              candidate can be compared with that file byte for byte
#   gate-dy    400 DY MiniAOD events, ditto against `dy_vtxon/task_0000`
#   dy         the 6 DY files re-run with the gen-provenance export on
#              (`Mu*gen_idx`, `Mu*gen_motherPdgId`, `Jpsigen_sameDecay`)
#   dy-off     the same 6 files with the vertex constraint OFF.  That regime
#              is where the tail lives -- 146 outliers at |z_v| > 5 against 17
#              with the constraint on -- so it is where the gen composition of
#              the tail can actually be MEASURED rather than bounded.
#
# Nothing else changes: the maker's new `minNdof` / `minPairHits` are left at
# their defaults, which IS the cut under test, and the gate is what shows the
# candidates above it are unchanged.
#
# usage: ./run_bkg.sh {gate-gun|gate-dy|dy} [nparallel] [from] [to]
set -uo pipefail
WHAT=${1:-dy}
NPAR=${2:-6}
TASKS_FROM=${3:-0}
TASKS_TO=${4:-5}
CMSSW_AREA=${CMSSW_AREA:-/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2}
RUN_ONE=/work/submit/david_w/ZMass/calibration_studies/slurm/run_one.sh
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
OUTROOT=${OUTROOT:-/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/bkg}
export CMSSW_AREA

DY_CFG=$CMSSW_AREA/src/Analysis/HitAnalyzer/test/runCvhDimuonMiniAOD.py
GUN_CFG=$CMSSW_AREA/src/Analysis/HitAnalyzer/test/runCvhJpsiGenMC.py
DY_LIST=/work/submit/david_w/ZMass/calibration_studies/production/filelist_dymc_8p5M_260905.txt
GUN_LIST=/work/submit/david_w/ZMass/calibration_studies/resolution/simprod/filelist_jpsigun_ul16.txt

# verbatim from run_prod_dy.sh
DY_COMMON="numberOfThreads=1 doRes=True exportCfExponents=True \
exportStepRecords=False exportCfGroupExponents=True \
exportMaterialNoise=True exportVarianceGrads=True varianceGradFamilies=15 \
fillJac=True fillGrads=False fillGradsFactored=True \
fitFromGenParms=False doSimHits=False doGen=True requireGen=False \
genParticles=prunedGenParticles pileupInfo=slimmedAddPileupInfo \
doTrigger=False applyHltFilter=False massMin=60 massMax=120 \
useIdealGeometry=False useDefaultField=True globalTag=106X_mcRun2_asymptotic_v17 \
doMassConstraint=False CgfQoPMode=0 tightG4eStepper=True \
propagationPtotLimit=0.2 maxMomentumStepFactor=2.0 stepBacktracking=True \
scalarPot3DInitFile=$INIT exportVtxResidual=True"

# verbatim from run_prod.sh
GUN_COMMON="numberOfThreads=1 doRes=True fillGrads=True fitFromGenParms=False \
scalarPot3DInitFile=$INIT trackSrc=generalTracks useLegacyPairLoop=True doTrigger=False \
applyHltFilter=False useIdealGeometry=True useDefaultField=True \
globalTag=150X_mcRun2_asymptotic_v1 CgfQoPMode=0 armijoSlack=1.0 \
exportVtxResidual=True exportCfGroupExponents=True exportStepRecords=False"

run_task() {
  local idx=$1
  local outdir="$OUTROOT/$OUTTAG/task_$(printf '%04d' "$idx")"
  if [[ -f "$outdir/.complete" ]]; then echo "[skip] task $idx"; return 0; fi
  sleep $(( (idx % ${NPAR:-6}) * ${STAGGER:-4} ))
  local input; input=$(sed -n "$((idx + 1))p" "$LIST")
  [[ -n "$input" ]] || { echo "[err] empty filelist line $idx"; return 1; }
  mkdir -p "$outdir"; rm -f "$outdir"/globalcor_*.root "$outdir/.complete"
  echo "[run ] task $idx -> $outdir"
  # shellcheck disable=SC2086
  if "$RUN_ONE" "$CFG" "$input" "$outdir" $COMMON nEvents=$NEV > "$outdir/local.log" 2>&1; then
    touch "$outdir/.complete"; echo "[done] task $idx"
  else
    echo "[FAIL] task $idx (see $outdir/local.log)"
  fi
}
export -f run_task

case "$WHAT" in
  gate-gun) CFG=$GUN_CFG; LIST=$GUN_LIST; COMMON=$GUN_COMMON; OUTTAG=gate_gun; NEV=200; TASKS_FROM=0; TASKS_TO=0 ;;
  gate-dy)  CFG=$DY_CFG;  LIST=$DY_LIST;  COMMON=$DY_COMMON;  OUTTAG=gate_dy;  NEV=400; TASKS_FROM=0; TASKS_TO=0 ;;
  dy)       CFG=$DY_CFG;  LIST=$DY_LIST;  COMMON=$DY_COMMON;  OUTTAG=dy_vtxon_gen; NEV=4000 ;;
  dy-off)   CFG=$DY_CFG;  LIST=$DY_LIST;  COMMON="$DY_COMMON doVtxConstraint=False"; OUTTAG=dy_vtxoff_gen; NEV=4000 ;;
  *) echo "usage: $0 {gate-gun|gate-dy|dy|dy-off} [nparallel] [from] [to]"; exit 2 ;;
esac
export CFG LIST COMMON OUTTAG NEV RUN_ONE OUTROOT NPAR STAGGER
seq "$TASKS_FROM" "$TASKS_TO" | xargs -P "$NPAR" -I{} bash -c 'run_task {}'
echo "vtxres bkg $WHAT block $TASKS_FROM-$TASKS_TO finished"
