#!/bin/bash
# THE COST OF THE sqrt(dV_b) CACHE, and what the beam functionals cost after
# it.  Same input file, same event count, ONE host, PINNED to one CPU, run
# SEQUENTIALLY -- never in parallel, which would measure the contention.
#
#   ./run_timing_cache.sh [nevents] [cpu]
#
# Five configurations:
#   old_off   dev3  bsConstraint=False                        the old baseline
#   old_full  dev3  bsConstraint=True exportBsResidual=True   the old +62.6 %
#   new_off   dev2  bsConstraint=False                        cached baseline
#   new_rows  dev2  bsConstraint=True exportBsResidual=False  the FIT cost
#   new_full  dev2  bsConstraint=True exportBsResidual=True   cached + beam
set -uo pipefail
NEV=${1:-400}
CPU=${2:-3}
DEV2=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2
DEV3=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev3
RUN_ONE=/work/submit/david_w/ZMass/calibration_studies/slurm/run_one.sh
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
FILELIST=/work/submit/david_w/ZMass/calibration_studies/production/filelist_dymc_8p5M_260905.txt
OUT=${OUT:-/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/beamline/timing_cache}
INPUT=$(sed -n 1p $FILELIST)
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
echo "host $(hostname)  cpu $CPU  nevents $NEV"
for cfgname in old_off old_full new_off new_rows new_full; do
  case $cfgname in
    old_off)  A=$DEV3; X="bsConstraint=False" ;;
    old_full) A=$DEV3; X="bsConstraint=True exportBsResidual=True" ;;
    new_off)  A=$DEV2; X="bsConstraint=False" ;;
    new_rows) A=$DEV2; X="bsConstraint=True exportBsResidual=False" ;;
    new_full) A=$DEV2; X="bsConstraint=True exportBsResidual=True" ;;
  esac
  d=$OUT/$cfgname; mkdir -p $d; rm -f $d/globalcor_*.root
  t0=$(date +%s)
  # shellcheck disable=SC2086
  CMSSW_AREA=$A taskset -c $CPU $RUN_ONE \
      "$A/src/Analysis/HitAnalyzer/test/runCvhDimuonMiniAOD.py" \
      "$INPUT" "$d" $COMMON $X > $d/local.log 2>&1
  t1=$(date +%s)
  echo "$cfgname wall $((t1-t0)) s  bytes $(stat -c%s $d/globalcor_0.root 2>/dev/null)"
done
echo TIMINGDONE
