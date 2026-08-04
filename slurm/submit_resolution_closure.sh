#!/bin/bash
# Resolution gen-closure (doRes re-run of the 2022 attempt with current hit
# selection): single-track CVH refit of gen-matched muons in the inclusive
# B->J/psi+X 2016postVFP MC ALCARECO, fitFromGenParms=True, doRes=True,
# fillGrads=True. Solved downstream with
#   fit_global_grads.py --parmtypes 10 [8 9 11] --tie
# (closure criterion: tied type scales consistent with zero, INDEPENDENT of
# the ionization truncation alpha -- the alpha drift is the 2022 failure
# signature; see Documents/Resolution/NOTES.md).
#
# Submits one Slurm array per ionization-truncation-alpha value:
#   baseline alpha=0.999 (historical CVH), variants 0.997 / 0.995
#
# usage: ./submit_resolution_closure.sh [baseline|scan|both] [extra submit.sh args]
set -euo pipefail

SLURM_DIR=$(dirname "$(readlink -f "$0")")
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
FILELIST=/work/submit/david_w/ZMass/calibration_studies/pixelhits/filelist_mc_all_chunk50.txt
CMSSW_AREA=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev
DATE_TAG=$(date +%y%m%d)
OUTBASE=/ceph/submit/data/user/d/david_w/ZMass/cvh/resolution_closure_${DATE_TAG}

MODE=${1:-baseline}

COMMON_ARGS="nEvents=-1 numberOfThreads=1 doRes=True fillGrads=True scalarPot3DInitFile=$INIT"

submit_one() {
  local tag=$1 extra=$2
  "$SLURM_DIR/submit.sh" \
    --config "$CMSSW_AREA/src/Analysis/HitAnalyzer/test/runCvhResClosure.py" \
    --filelist "$FILELIST" \
    --outdir "${OUTBASE}_${tag}" \
    --cmssw-area "$CMSSW_AREA" \
    --name "resclos_${tag}" \
    --cpus 1 --mem 4G --time 04:00:00 \
    --cmsrun-args "$COMMON_ARGS $extra" \
    "${@:3}"
}

case "$MODE" in
  baseline) submit_one alpha999 "" "${@:2}";;
  scan)
    submit_one alpha997 "ioniTruncationAlpha=0.997" "${@:2}"
    submit_one alpha995 "ioniTruncationAlpha=0.995" "${@:2}"
    ;;
  both)
    submit_one alpha999 "" "${@:2}"
    submit_one alpha997 "ioniTruncationAlpha=0.997" "${@:2}"
    submit_one alpha995 "ioniTruncationAlpha=0.995" "${@:2}"
    ;;
  *) echo "unknown mode: $MODE (use baseline|scan|both)" >&2; exit 1;;
esac
