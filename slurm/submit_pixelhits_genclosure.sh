#!/bin/bash
# Pixel edge / single-column hit study: free-fit (fitFromGenParms=False,
# no mass constraint) gen-closure A/B on the inclusive B->J/psi+X
# 2016postVFP MC ALCARECO, submitted as two Slurm arrays:
#   baseline  - legacy hit-quality veto (applyHitQuality, sizeX>=2, !edge)
#   keephits  - vetoed pixel hits re-included (keepPixelEdgeHits=True
#               pixelMinSizeX=1), no dedicated corrections yet
#
# The filelist is CHUNKED: each line is a comma-joined group of ~50 MC
# files (the files hold only ~10 events each); runCvhJpsiGenMC.py splits
# input= on commas.
#
# usage: ./submit_pixelhits_genclosure.sh [baseline|keephits|both] [extra submit.sh args]
set -euo pipefail

SLURM_DIR=$(dirname "$(readlink -f "$0")")
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
FILELIST=/work/submit/david_w/ZMass/calibration_studies/pixelhits/filelist_mc_all_chunk50.txt
CMSSW_AREA=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev
DATE_TAG=$(date +%y%m%d)
OUTBASE=/ceph/submit/data/user/d/david_w/ZMass/cvh/pixelhits_genclosure_${DATE_TAG}

MODE=${1:-both}

COMMON_ARGS="nEvents=-1 numberOfThreads=1 fitFromGenParms=False scalarPot3DInitFile=$INIT"

submit_one() {
  local tag=$1 extra=$2
  "$SLURM_DIR/submit.sh" \
    --config "$CMSSW_AREA/src/Analysis/HitAnalyzer/test/runCvhJpsiGenMC.py" \
    --filelist "$FILELIST" \
    --outdir "${OUTBASE}_${tag}" \
    --cmssw-area "$CMSSW_AREA" \
    --name "pixhit_${tag}" \
    --cpus 1 --mem 4G --time 04:00:00 \
    --cmsrun-args "$COMMON_ARGS $extra" \
    "${@:3}"
}

case "$MODE" in
  baseline) submit_one baseline "" "${@:2}";;
  keephits) submit_one keephits "keepPixelEdgeHits=True pixelMinSizeX=1" "${@:2}";;
  both)
    submit_one baseline "" "${@:2}"
    submit_one keephits "keepPixelEdgeHits=True pixelMinSizeX=1" "${@:2}"
    ;;
  *) echo "unknown mode: $MODE (use baseline|keephits|both)" >&2; exit 1;;
esac
