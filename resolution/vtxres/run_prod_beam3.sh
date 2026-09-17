#!/bin/bash
# THE BEAM3 STUDY's production: the same DY MiniAOD configuration as
# `run_prod_bs.sh` (which is `production/condor_dymc_v2`'s verbatim plus
# `exportVtxResidual`), luminous-region rows ON and both beam-line
# functionals exported, submitted as a SLURM ARRAY rather than an xargs
# pool because the sample has to be ~10x `dy_bs_final`.
#
# WHY THE SIZE.  The closure is on the luminous region's TILTS.  A single
# candidate measures `dxdz` to ~6.4e-4 (the beam pull's response to a 1e-4
# slope is rms 0.157), so `sigma(dxdz) = 6.4e-4/sqrt(N)`: 1e5 candidates
# give 2e-6, against a record-vs-simulation offset of 6e-6.  `dy_bs_final`
# (10 254 candidates) gives 6.3e-6 and cannot separate them.
#
# 80 files x 3500 events ~ 280 k events ~ 1.2e5 candidates, ~3.4 h/task.
#
#   usage: ./run_prod_beam3.sh [--dry-run]
#   env:   OUTTAG (dy_beam3), NEVENTS (3500), MAXRUN (40), EXTRA
set -uo pipefail
SLURM=/work/submit/david_w/ZMass/calibration_studies/slurm
CMSSW_AREA=${CMSSW_AREA:-/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2}
CFG=${CFG:-$CMSSW_AREA/src/Analysis/HitAnalyzer/test/runCvhDimuonMiniAOD.py}
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
FILELIST=${FILELIST:-/work/submit/david_w/ZMass/calibration_studies/production/filelist_dymc_beam3_260917.txt}
OUTROOT=${OUTROOT:-/ceph/submit/data/user/d/david_w/ZMass/cvh/runs_vtxres_260911/beamline}
OUTTAG=${OUTTAG:-dy_beam3_ideal}
# THE GEOMETRY.  `useIdealGeometry=True` is the default here: the Geant4
# detector in the MC is the IDEAL one, while the global tag's alignment
# payload (`TrackerAlignment_2016_ultralegacymc_v1`) is a realistic
# MISALIGNMENT scenario that nothing in these fits floats -- so an MC closure
# sample refit on the aligned geometry is comparing against a detector the
# simulation never had.  David, 2026-09-17: the MC closure samples move to the
# ideal geometry.  The luminous-region closure itself does not care (the beam
# parameters are measured against the GEN production vertex, which is the same
# in both), which is what makes the aligned leg a usable cross-check.
GEOM=${GEOM:-True}

COMMON="nEvents=${NEVENTS:-3500} numberOfThreads=1 doRes=True exportCfExponents=True \
exportStepRecords=False exportCfGroupExponents=True \
exportMaterialNoise=True exportVarianceGrads=True varianceGradFamilies=15 \
fillJac=True fillGrads=False fillGradsFactored=True \
fitFromGenParms=False doSimHits=False doGen=True requireGen=False \
genParticles=prunedGenParticles pileupInfo=slimmedAddPileupInfo \
doTrigger=False applyHltFilter=False massMin=60 massMax=120 \
useIdealGeometry=$GEOM useDefaultField=True globalTag=106X_mcRun2_asymptotic_v17 \
doMassConstraint=False CgfQoPMode=0 tightG4eStepper=True \
propagationPtotLimit=0.2 maxMomentumStepFactor=2.0 stepBacktracking=True \
scalarPot3DInitFile=$INIT exportVtxResidual=True \
bsConstraint=True exportBsResidual=True ${EXTRA:-}"

exec "$SLURM/submit.sh" \
  --config "$CFG" --filelist "$FILELIST" --outdir "$OUTROOT/$OUTTAG" \
  --cmssw-area "$CMSSW_AREA" --name beam3 \
  --time "${TIMELIM:-16:00:00}" --mem "${MEM:-5G}" --cpus 1 \
  --max-running "${MAXRUN:-40}" \
  --cmsrun-args "$COMMON" "$@"
