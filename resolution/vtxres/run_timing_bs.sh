#!/bin/bash
# The MAKER COST of the beam rows and the two beam functionals, measured
# back to back on ONE host with ONE input file and the same event count, so
# the two runs see the same machine and the same candidates.
#
#   ./run_timing_bs.sh [nevents]
#
# Three configurations, SEQUENTIALLY (never in parallel -- that would measure
# the contention, not the code):
#   off   bsConstraint=False                       (the reference)
#   rows  bsConstraint=True  exportBsResidual=False (the FIT cost alone)
#   full  bsConstraint=True  exportBsResidual=True  (fit + the two functionals)
set -uo pipefail
NEV=${1:-400}
CMSSW_AREA=${CMSSW_AREA:-/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev3}
CFG=$CMSSW_AREA/src/Analysis/HitAnalyzer/test/runCvhDimuonMiniAOD.py
RUN_ONE=/work/submit/david_w/ZMass/calibration_studies/slurm/run_one.sh
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
FILELIST=/work/submit/david_w/ZMass/calibration_studies/production/filelist_dymc_8p5M_260905.txt
OUT=${OUT:-/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/beamline/timing}
INPUT=$(sed -n 1p $FILELIST)
export CMSSW_AREA
COMMON="nEvents=$NEV numberOfThreads=1 doRes=True exportCfExponents=True \
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
for cfgname in off rows full; do
  case $cfgname in
    off)  X="bsConstraint=False" ;;
    rows) X="bsConstraint=True exportBsResidual=False" ;;
    full) X="bsConstraint=True exportBsResidual=True" ;;
  esac
  d=$OUT/$cfgname; mkdir -p $d; rm -f $d/globalcor_*.root
  t0=$(date +%s)
  # shellcheck disable=SC2086
  $RUN_ONE "$CFG" "$INPUT" "$d" $COMMON $X > $d/local.log 2>&1
  t1=$(date +%s)
  echo "$cfgname wall $((t1-t0)) s  bytes $(stat -c%s $d/globalcor_0.root 2>/dev/null)"
done
