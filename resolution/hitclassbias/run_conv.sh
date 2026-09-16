#!/bin/bash
# Convergence / seed-dependence diagnostic productions on the 20-60 GeV tight
# muon gun.
#
# Reproduces the `mugun_ul16_260903x_m0` production arm EXACTLY --
# same cfg (CMSSW_15_0_19_patch2_dev2, commit ca6058d96fc, identical to dev2),
# same filelist, same COMMON block, same E_MUGUN extras, CgfQoPMode=0 -- and
# changes ONLY the Gauss-Newton convergence knobs per variant:
#
#   base   nothing changed                      -> like-for-like bit check
#   tight  edmConvergence 1e-5 -> 1e-7, nIters 10 -> 20
#   damp   + gnDampAfter=1 gnDampFactor=0.5, nIters 30: the step is halved
#          from iteration 1 on, so the fit CANNOT land at the seed-proximal
#          point in two iterations and reaches the same minimum along a
#          different path. This is the substitute for "force >= 5 GN
#          iterations": there is no `minIters` cfi parameter and adding one
#          would need a rebuild.
#
# NOT USED: `fitFromGenParms=True`. Measured on `hitres2_mugun_ul16`, that
# switch FREEZES the reference block exactly at gen (max |refParms-genParms|
# = 0.0, refCov(0,0) = 0, niter = 1), so `z = (refParms[0]-genParms[0])/sigma`
# is identically zero and the observable does not exist. It removes the
# measurement, not the seed dependence.
#
# usage: ./run_conv.sh <variant> <ntasks> [nparallel]
set -uo pipefail
VARIANT=$1
NTASK=${2:-40}
NPAR=${3:-24}
RES=/work/submit/david_w/ZMass/calibration_studies/resolution
CFG=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2/src/Analysis/HitAnalyzer/test/runCvhResClosure.py
export CMSSW_AREA=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev2
RUN_ONE=/work/submit/david_w/ZMass/calibration_studies/slurm/run_one.sh
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
FILELIST=$RES/simprod/filelist_mugun_ul16.txt
OUTROOT=/ceph/submit/data/user/d/david_w/ZMass/cvh
OUTTAG=mugun_ul16_260909_conv_${VARIANT}

COMMON="nEvents=${NEVENTS:--1} numberOfThreads=1 doRes=True fillGrads=True \
fitFromGenParms=False scalarPot3DInitFile=$INIT \
trackSrc=generalTracks useDefaultField=True useIdealGeometry=True \
globalTag=150X_mcRun2_asymptotic_v1 CgfQoPMode=0"

case "$VARIANT" in
  base)  EXTRA="" ;;
  tight) EXTRA="edmConvergence=1e-7 nIters=20" ;;
  damp)  EXTRA="edmConvergence=1e-7 nIters=30 gnDampAfter=1 gnDampFactor=0.5" ;;
  *) echo "unknown variant $VARIANT"; exit 1 ;;
esac

run_task() {
  local idx=$1
  local outdir="$OUTROOT/resolution_trackres_${OUTTAG}/task_$(printf '%04d' "$idx")"
  [[ -f "$outdir/.complete" ]] && { echo "[skip] $OUTTAG $idx"; return 0; }
  sleep $(( (idx % NPAR) * 2 ))
  local input; input=$(sed -n "$((idx + 1))p" "$FILELIST")
  [[ -n "$input" ]] || { echo "[err] no filelist line $idx"; return 1; }
  mkdir -p "$outdir"
  rm -f "$outdir"/globalcor_resclosure_*.root "$outdir/.complete"
  echo "[run ] $OUTTAG $idx"
  # shellcheck disable=SC2086
  if "$RUN_ONE" "$CFG" "$input" "$outdir" $COMMON $EXTRA > "$outdir/local.log" 2>&1; then
    touch "$outdir/.complete"; echo "[done] $OUTTAG $idx"
  else
    echo "[FAIL] $OUTTAG $idx (see $outdir/local.log)"
  fi
}
export -f run_task
export CFG RUN_ONE FILELIST OUTROOT OUTTAG COMMON EXTRA NPAR CMSSW_AREA
seq 0 $((NTASK - 1)) | xargs -P "$NPAR" -I{} bash -c 'run_task {}'
echo "[$OUTTAG] $(ls -d $OUTROOT/resolution_trackres_$OUTTAG/task_*/.complete 2>/dev/null | wc -l)/$NTASK complete"
