#!/bin/bash
# J/psi gun ditrack refit on the CURRENT dev build (2026-09-02), Q-matrix
# estimator (CgfQoPMode=0 explicit: the two-track maker has no CGF override
# hooks, so mode 1 only pays 10x for computing a block it never substitutes;
# verified on a 40-event smoke, outputs bit-identical).
# Input: resolution_simprod_jpsigun_ul16 (160 x 2000 events, Aug 8 SIM).
# Reproduces the 260811 mass-likelihood rung with the current model
# (exact-delta, species/charge-aware reference dE/dx, dEdxlast fix, ...).
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution
LOG=$RES/runs/ditrack260902; mkdir -p "$LOG"; cd "$RES"
export CFG=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/test/runCvhJpsiGenMC.py
export FILELIST=$RES/simprod/filelist_jpsigun_ul16.txt
export OUTTAG=jpsigun_ul16_260902_m0
export EXTRA="trackSrc=generalTracks useLegacyPairLoop=True doTrigger=False applyHltFilter=False useIdealGeometry=True useDefaultField=True globalTag=150X_mcRun2_asymptotic_v1 CgfQoPMode=0"
export STAGGER=3
NPAR=${NPAR:-80}
for blk in 0 80; do end=$((blk+NPAR-1)); [ $end -gt 159 ] && end=159
  echo "=== block $blk-$end ($(date +%H:%M:%S)) ==="
  ./run_local_trackres.sh $NPAR $blk $end >> "$LOG/refit.log" 2>&1
done
echo "=== refit done: $(ls -d /ceph/submit/data/user/d/david_w/ZMass/cvh/resolution_trackres_$OUTTAG/task_*/.complete 2>/dev/null | wc -l)/160 ($(date +%H:%M:%S)) ==="
