#!/bin/bash
# A/B arms for the three CVH defects of 2026-09-18:
#   (a) the reference EDM on the frozen vertex direction  -- CVH_EDM_LEGACY=1 restores the pre-fix NaN
#   (d) the within-step MS angle-offset correlation sign  -- CVH_MS_CORR_LEGACY=1 restores res(1,4) = -S3
# One binary, so an arm differs from another ONLY by the knob under test.
#
#   usage: run_ab.sh <arm> [nevents]
set -uo pipefail

ARM=${1:?usage: run_ab.sh <arm> [nevents]}
NEV=${2:-1300}

AREA=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2
OUTROOT=/work/submit/david_w/ZMass/calibration_studies/resolution/runs/msksign_260918
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
IN_GUN=/work/submit/david_w/ZMass/scratch_smoke_260906/inputs/jpsigun_task0000_step2.root
KSCHUNK=${KSCHUNK:-/ceph/submit/data/user/d/david_w/ZMass/cvh/ks_btojpsix_260917_ideal/chunks/chunk_0000.txt}
KSDRV=/work/submit/david_w/ZMass/calibration_studies/resolution/ksclosure/runCvhKs.py

GUNCOMMON="numberOfThreads=1 trackSrc=generalTracks useLegacyPairLoop=True
           doTrigger=False applyHltFilter=False fitFromGenParms=False CgfQoPMode=0
           useIdealGeometry=True useDefaultField=True globalTag=150X_mcRun2_asymptotic_v1
           doRes=True exportCfExponents=True fillJac=True fillGrads=True
           exportVtxResidual=True fillHitDiagnostics=True
           scalarPot3DInitFile=$INIT"

out=$OUTROOT/$ARM
mkdir -p "$out"
rm -f "$out"/*.root
cd "$AREA/src"
source /cvmfs/cms.cern.ch/cmsset_default.sh
eval "$(scramv1 runtime -sh)"
cd "$out"

case $ARM in
  # ---- J/psi gun, two-track ------------------------------------------------
  gun_prod)      export CVH_EDM_LEGACY=1 CVH_MS_CORR_LEGACY=1 ;;   # what the productions ran
  gun_dfix)      export CVH_EDM_LEGACY=1 ;;                        # (d) fixed only
  gun_dfix_afix) : ;;                                              # (a) and (d) fixed = final state
  gun_msleg_afix) export CVH_MS_CORR_LEGACY=1 ;;                   # (a) fixed, (d) LEGACY -- the (d) pair with gun_dfix_afix
  gun_novtx_leg) export CVH_EDM_LEGACY=1 ;;                        # vtx OFF, legacy EDM
  gun_novtx_fix) : ;;                                              # vtx OFF, fixed EDM  -> must match gun_novtx_leg
  gun_mcons_on)  : ;;                                              # mass constraint ON, step records
  gun_mcons_off) : ;;                                              # mass constraint OFF, step records
  legacy_all)    export CVH_EDM_LEGACY=1 CVH_MS_CORR_LEGACY=1 ;;   # 60-event regression vs the 260906 reference
  ks_prod|ks_prod1|ks_prod2)  export CVH_EDM_LEGACY=1 CVH_MS_CORR_LEGACY=1 ;;
  ks_dfix|ks_dfix1|ks_dfix2)  export CVH_EDM_LEGACY=1 ;;
  *) echo "[FATAL] unknown arm $ARM" >&2; exit 2 ;;
esac

echo ">>> arm=$ARM nev=$NEV CVH_EDM_LEGACY=${CVH_EDM_LEGACY:-unset} CVH_MS_CORR_LEGACY=${CVH_MS_CORR_LEGACY:-unset}"

case $ARM in
  gun_prod|gun_dfix|gun_dfix_afix|gun_msleg_afix)
    cmsRun "$AREA/src/Analysis/HitAnalyzer/test/runCvhJpsiGenMC.py" \
      input=$IN_GUN nEvents=$NEV $GUNCOMMON exportStepRecords=False ;;
  gun_novtx_leg|gun_novtx_fix)
    cmsRun "$AREA/src/Analysis/HitAnalyzer/test/runCvhJpsiGenMC.py" \
      input=$IN_GUN nEvents=$NEV $GUNCOMMON exportStepRecords=False doVtxConstraint=False ;;
  gun_mcons_on)
    cmsRun "$AREA/src/Analysis/HitAnalyzer/test/runCvhJpsiGenMC.py" \
      input=$IN_GUN nEvents=$NEV $GUNCOMMON exportStepRecords=True doMassConstraint=True ;;
  gun_mcons_off)
    cmsRun "$AREA/src/Analysis/HitAnalyzer/test/runCvhJpsiGenMC.py" \
      input=$IN_GUN nEvents=$NEV $GUNCOMMON exportStepRecords=True doMassConstraint=False ;;
  legacy_all)
    cmsRun "$AREA/src/Analysis/HitAnalyzer/test/runCvhJpsiGenMC.py" \
      input=$IN_GUN nEvents=60 numberOfThreads=1 doRes=True \
      exportCfExponents=True exportStepRecords=True \
      fillJac=True fillGrads=True fitFromGenParms=False CgfQoPMode=0 \
      scalarPot3DInitFile=$INIT trackSrc=generalTracks useLegacyPairLoop=True \
      doTrigger=False applyHltFilter=False \
      useIdealGeometry=True useDefaultField=True globalTag=150X_mcRun2_asymptotic_v1 ;;
  ks_prod*|ks_dfix*)
    cmsRun "$KSDRV" inputFileList=$KSCHUNK nEvents=$NEV \
      numberOfThreads=1 scalarPot3DInitFile=$INIT ;;
esac
rc=$?
echo ">>> arm=$ARM rc=$rc"
ls -la "$out"/*.root 2>/dev/null
exit $rc
