#!/bin/bash
# CF-export smoke runs in the dev2 area.  usage: run_smoke.sh <arm> <case>
#   arm  = off (the CF model's knock-on switches off: cfKnockonJoint=False
#          cfQopExact=False, cf_knockon's CF_KNOCKON_JOINT=0 CF_QOP_EXACT=0) |
#          on (the default model + group split + nuclear elastic)
#   case = mu kaon jpsi   (the step records are on, for the branch-level check
#          of cvhcf_validate.py --compare-branches)
set -uo pipefail
ARM=$1; CASE=$2
BASE=/ceph/submit/data/user/d/david_w/ZMass/cvh/knockon_trackfit_260927
AREA=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2
T=$AREA/src/Analysis/HitAnalyzer/test
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
SIM=/ceph/submit/data/user/d/david_w/ZMass/cvh
ST="numberOfThreads=1 doRes=True fillGrads=True fitFromGenParms=False scalarPot3DInitFile=$INIT trackSrc=generalTracks useDefaultField=True useIdealGeometry=True globalTag=150X_mcRun2_asymptotic_v1 CgfQoPMode=0 localUpdate=True"
JP="numberOfThreads=1 doRes=True exportCfExponents=True exportCfGroupExponents=True exportMaterialNoise=True exportVarianceGrads=True varianceGradFamilies=15 fillJac=True fillGrads=False fillGradsFactored=True fitFromGenParms=False trackSrc=ALCARECOTkAlJpsiMuMu useLegacyPairLoop=True doTrigger=True applyHltFilter=False doSimHits=False useIdealGeometry=True useDefaultField=True globalTag=106X_mcRun2_asymptotic_v17 CgfQoPMode=0 propagationPtotLimit=0.2 maxMomentumStepFactor=2.0 stepBacktracking=True scalarPot3DInitFile=$INIT"
JPIN=/ceph/submit/data/group/cms/store/mc/RunIISummer20UL16RECO/JPsiToMuMu_Pt8toInf-pythia8/ALCARECO/TkAlJpsiMuMu-106X_mcRun2_asymptotic_v13-v2/2820000/02536347-8E42-C74C-9ED0-7EAF4290973A.root
case $CASE in
  mu)   CFG=$T/runCvhResClosure.py; ARGS="$ST particle=mu nEvents=15 input=$SIM/resolution_simprod_mugun_ul16/task_0000/step2.root exportStepRecords=True";;
  kaon) CFG=$T/runCvhResClosure.py; ARGS="$ST particle=kaon nEvents=25 input=$SIM/resolution_simprod_kaongun_ul16/task_0000/step2.root exportStepRecords=True";;
  jpsi) CFG=$T/runCvhJpsiGenMC.py; ARGS="$JP input=$JPIN nEvents=20 exportStepRecords=True doMassConstraint=False";;
  *) echo "unknown case $CASE"; exit 2;;
esac
case $ARM in
  off) ARGS="$ARGS cfKnockonJoint=False cfQopExact=False";;
  on)  ARGS="${ARGS/exportCfGroupExponents=True /} exportCfGroupExponents=True exportCfNucel=True";;
  *) echo "unknown arm $ARM"; exit 2;;
esac
OUT=$BASE/$ARM/$CASE
rm -rf "$OUT"; mkdir -p "$OUT"; cd "$AREA/src"
source /cvmfs/cms.cern.ch/cmsset_default.sh
eval $(scramv1 runtime -sh)
cd "$OUT"
echo ">>> $(date) $(hostname) HEAD=$(git -C $AREA/src rev-parse --short=12 HEAD) dirty=$(git -C $AREA/src status --short -uno | wc -l)" > run.log
echo ">>> cmsRun $CFG $ARGS" >> run.log
/usr/bin/time -v $T/cmsswlock.sh run cmsRun "$CFG" $ARGS >> run.log 2>&1
rc=$?
echo ">>> rc=$rc $(date)" >> run.log
exit $rc
