#!/bin/bash
#SBATCH --job-name=zvar
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#SBATCH --time=05:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
# Every phase-1 variant off ONE card, sequentially, on one GPU.
#
# `a_res` and `jensen_s2` are stored arrays whose switches are applied at FIT
# time, and "K(m) fixed" is `--fix shape1..shape5`, so each row of the table is
# the same candidates and the same numbers with exactly one thing changed --
# and one card, not six.
#
#   sbatch -A mit_general -G h200:1 -p mit_normal_gpu \
#     --export=ALL,CARD=$HOME/orcd/pool/zmass/cards/z_full380.hdf5,TAG=f380 \
#     fullscale_variants.sbatch
set -uo pipefail
ZMASS=$HOME/orcd/pool/zmass
source $ZMASS/engaging/setup_env_engaging.sh
export PYTHONPATH=$ZMASS/rabbit:$ZMASS/fullscale:$ZMASS/zchannel:${PYTHONPATH:-}
CARD=${CARD:?set CARD=...}
TAG=${TAG:-f380}
CH=${CH:-32768}
KFIX="k_hit k_ms k_ioni k_rad"
SHAPEFIX="shape1 shape2 shape3 shape4 shape5"
mkdir -p $ZMASS/fitresults
cd $ZMASS/fullscale
CLIPARG=""
if [ -n "${CLIP:-}" ]; then CLIPARG="--corr-clip $CLIP"; TAG="${TAG}c${CLIP}"; fi
run () {  # run <label> <extra args...>
  local L=$1; shift
  echo "############ $L  $(date +%H:%M:%S)"
  python3 -u fit.py --card "$CARD" --chunk $CH --fix $KFIX $CLIPARG "$@" \
      --label "${TAG}_$L" -o $ZMASS/fitresults/fit_${TAG}_$L.json \
    || echo "############ $L FAILED (exit $?)"
}
run base
run noares    --ares off
run nojensen  --jensen off
run noboth    --ares off --jensen off
run jshift    --jensen shift
run noshape   --fix $KFIX $SHAPEFIX
echo "############ done $(date +%H:%M:%S)"
