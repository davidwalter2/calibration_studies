#!/bin/bash
# B->J/psi+X v3 ditrack refit on the CURRENT dev build (2026-09-02),
# Q-matrix estimator (CgfQoPMode=0 explicit: the two-track maker has no CGF
# override hooks, so mode 1 only pays ~10x for computing a block it never
# substitutes -- see run_ditrack_jpsigun_260902.sh).
#
# Complements the J/psi-gun rerun of the same day: the gun has NO FSR (no
# shower machinery -> the mass kernel is a delta), this sample HAS Photos
# QED FSR, so the two together separate the FSR-kernel contribution to the
# unbinned mass likelihood from the pure resolution+scale closure.
#
# Input: /ceph/.../mc/inclusive_btojpsix_2016postvfp_v3 (the campaign WITH
# SimTracks/SimVertices), 48 tasks x 50 files, deterministic sorted-first
# selection of the 125,981 non-zero-length files (see simprod/
# filelist_btojpsix_v3_chunk50.txt; md5 of the 2400-path selection
# 543a99b4b51ee9c849dd89ad50b4a143).
#
# CONFIGURATION = LADDER RUNG B (NOTES.md 2026-08-05 corrected-CF ladder,
# +0.191 +- 0.116 e-3): grid field + ALIGNED geometry.
#   useOpera3D=True    the SIM used VolumeBasedMagneticFieldESProducerFromDB
#                      (raw conditions-DB grid, no OAE tracker parametrization)
#                      -> the full 3D TOSCA grid 160812 is the SIM-MATCHED
#                      field, which is what isolates the model.
#   useIdealGeometry=False   official UL16 sample -> aligned geometry from the
#                      GT, matching the reconstruction of the simulation.
#                      (Rung C is the same with useIdealGeometry=True; the two
#                      agreed to +0.177 vs +0.191, well inside 0.116.)
#   globalTag=106X_mcRun2_asymptotic_v17  the GT the MC was produced with
#                      (CMSSW_10_6_20_patch1 chain).
#   trackSrc/useLegacyPairLoop: driver defaults -- ALCARECOTkAlJpsiX tracks
#                      with the persisted ALCARECOTkAlJpsiXJpsiOnlyResonances
#                      candidates (the MC track collection also carries the
#                      other B daughters, so the all-pairs legacy loop is
#                      wrong here); passed explicitly for the record.
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution
LOG=$RES/runs/ditrack260902; mkdir -p "$LOG"; cd "$RES"
export CFG=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/test/runCvhJpsiGenMC.py
export FILELIST=$RES/simprod/filelist_btojpsix_v3_chunk50.txt
export OUTTAG=btojpsix_v3_260902_m0
export EXTRA="trackSrc=ALCARECOTkAlJpsiX useLegacyPairLoop=False doTrigger=True applyHltFilter=False useIdealGeometry=False useOpera3D=True globalTag=106X_mcRun2_asymptotic_v17 CgfQoPMode=0"
export STAGGER=3
NPAR=${NPAR:-48}
echo "=== btojpsix v3 ditrack, tasks 0-47, NPAR=$NPAR ($(date +%H:%M:%S)) ==="
./run_local_trackres.sh $NPAR 0 47 >> "$LOG/refit_btojpsix.log" 2>&1
echo "=== refit done: $(ls -d /ceph/submit/data/user/d/david_w/ZMass/cvh/resolution_trackres_$OUTTAG/task_*/.complete 2>/dev/null | wc -l)/48 ($(date +%H:%M:%S)) ==="
