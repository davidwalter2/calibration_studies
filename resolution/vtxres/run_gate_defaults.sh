#!/bin/bash
# GATE for the per-channel `bsConstraint` DEFAULTS (task 2 of the beam-line
# finish).  The code in dev2 after the fast-forward merge is IDENTICAL to
# dev3 @ 0cb6c291354 except for the cfi / driver defaults, so the gate is a
# same-input, same-host A/B between the two areas:
#
#   gun : dev3 (bsConstraint hardcoded False)  vs  dev2 (opt, DEFAULT False)
#         -> must be BIT-IDENTICAL: the J/psi gun sees no change.
#   dy  : dev3 (bsConstraint=True given EXPLICITLY)  vs  dev2 (NOT given --
#         the driver's new DEFAULT is True)
#         -> must reproduce the rows-ON study build.
#
# Run from a submit node (ceph + el9 arch).  usage: ./run_gate_defaults.sh
set -uo pipefail
DEV2=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2
DEV3=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev3
RUN_ONE=/work/submit/david_w/ZMass/calibration_studies/slurm/run_one.sh
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
OUT=${OUT:-/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/beamline/gate_defaults}

GUNLIST=/work/submit/david_w/ZMass/calibration_studies/resolution/simprod/filelist_jpsigun_ul16.txt
DYLIST=/work/submit/david_w/ZMass/calibration_studies/production/filelist_dymc_8p5M_260905.txt
GUNIN=$(sed -n 1p $GUNLIST)
DYIN=$(sed -n 1p $DYLIST)

GUN_COMMON="nEvents=200 numberOfThreads=1 doRes=True fillGrads=True fitFromGenParms=False \
scalarPot3DInitFile=$INIT trackSrc=generalTracks useLegacyPairLoop=True doTrigger=False \
applyHltFilter=False useIdealGeometry=True useDefaultField=True \
globalTag=150X_mcRun2_asymptotic_v1 CgfQoPMode=0 armijoSlack=1.0 \
exportVtxResidual=True exportCfGroupExponents=True exportStepRecords=False"

DY_COMMON="nEvents=400 numberOfThreads=1 doRes=True exportCfExponents=True \
exportStepRecords=False exportCfGroupExponents=True \
exportMaterialNoise=True exportVarianceGrads=True varianceGradFamilies=15 \
fillJac=True fillGrads=False fillGradsFactored=True \
fitFromGenParms=False doSimHits=False doGen=True requireGen=False \
genParticles=prunedGenParticles pileupInfo=slimmedAddPileupInfo \
doTrigger=False applyHltFilter=False massMin=60 massMax=120 \
useIdealGeometry=False useDefaultField=True globalTag=106X_mcRun2_asymptotic_v17 \
doMassConstraint=False CgfQoPMode=0 tightG4eStepper=True \
propagationPtotLimit=0.2 maxMomentumStepFactor=2.0 stepBacktracking=True \
scalarPot3DInitFile=$INIT exportVtxResidual=True exportBsResidual=True"

one () {  # one <area> <cfg> <input> <tag> <extra args...>
  local area=$1 cfg=$2 input=$3 tag=$4; shift 4
  local d=$OUT/$tag; mkdir -p "$d"; rm -f "$d"/globalcor_*.root
  CMSSW_AREA=$area $RUN_ONE "$area/src/Analysis/HitAnalyzer/test/$cfg" \
      "$input" "$d" "$@" > "$d/local.log" 2>&1 \
    && echo "[done] $tag" || echo "[FAIL] $tag -> $d/local.log"
}

# shellcheck disable=SC2086
one $DEV3 runCvhJpsiGenMC.py       "$GUNIN" gun_ref $GUN_COMMON
# shellcheck disable=SC2086
one $DEV2 runCvhJpsiGenMC.py       "$GUNIN" gun_new $GUN_COMMON
# shellcheck disable=SC2086
one $DEV3 runCvhDimuonMiniAOD.py   "$DYIN"  dy_ref  $DY_COMMON bsConstraint=True
# shellcheck disable=SC2086
one $DEV2 runCvhDimuonMiniAOD.py   "$DYIN"  dy_new  $DY_COMMON
echo "gate_defaults productions finished -> $OUT"
