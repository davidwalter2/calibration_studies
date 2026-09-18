#!/bin/bash
# The Z leg of the (d) A/B: DY MiniAOD, where the beam-spot constraint is on by
# default, so the whitened beam pulls are exercised. Separate file from
# run_ab.sh because jobs are still reading that one.
set -uo pipefail
ARM=${1:?usage: run_dy.sh <dy_prod|dy_dfix> [nevents]}
NEV=${2:-700}
AREA=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2
OUT=/work/submit/david_w/ZMass/calibration_studies/resolution/runs/msksign_260918/$ARM
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
FL=/work/submit/david_w/ZMass/calibration_studies/resolution/simprod/filelist_dy_miniaod_260905.txt
mkdir -p "$OUT"; rm -f "$OUT"/*.root
cd "$AREA/src"; source /cvmfs/cms.cern.ch/cmsset_default.sh; eval "$(scramv1 runtime -sh)"; cd "$OUT"
case $ARM in
  dy_prod) export CVH_MS_CORR_LEGACY=1 ;;
  dy_dfix) : ;;
  *) echo "[FATAL] unknown arm $ARM" >&2; exit 2 ;;
esac
echo ">>> arm=$ARM nev=$NEV CVH_MS_CORR_LEGACY=${CVH_MS_CORR_LEGACY:-unset}"
cmsRun "$AREA/src/Analysis/HitAnalyzer/test/runCvhDimuonMiniAOD.py" \
  inputFileList=$FL nEvents=$NEV numberOfThreads=1 \
  doRes=True exportCfExponents=True exportStepRecords=False \
  fillJac=True fillGrads=False fillGradsFactored=True \
  fitFromGenParms=False doGen=True requireGen=False \
  genParticles=prunedGenParticles pileupInfo=slimmedAddPileupInfo \
  doTrigger=False applyHltFilter=False massMin=60 massMax=120 \
  useIdealGeometry=True useDefaultField=True globalTag=106X_mcRun2_asymptotic_v17 \
  doMassConstraint=False CgfQoPMode=0 tightG4eStepper=True \
  exportBsResidual=True exportVtxResidual=True \
  outprefix=globalcor scalarPot3DInitFile=$INIT
echo ">>> arm=$ARM rc=$?"
ls -la "$OUT"/*.root 2>/dev/null
