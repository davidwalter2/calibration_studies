#!/bin/bash
#SBATCH --job-name=zvarfl
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#SBATCH --time=05:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
# The phase-1 variant ladder off ONE FLUCTUATION-FORM card, on one GPU.
#
# Same idea as `fullscale_variants.sbatch`, minus `jshift`: in the fluctuation
# form the deterministic part of the Jensen map is s^2/2 (the exact form's) and
# there is no "mean shift" variant to run.  `--ares off` / `--jensen off` go
# through `term.set_corrections`, which REBUILDS the per-candidate c_i, d_i --
# flipping the flags in place would silently do nothing.
#
#   sbatch -A mit_general -G h200:1 -p mit_normal_gpu \
#     --export=ALL,CARD=$HOME/orcd/pool/zmass/cards/z_full380_fl.hdf5,TAG=f380fl \
#     fullscale_variants_fl.sbatch
set -uo pipefail
ZMASS=$HOME/orcd/pool/zmass
source $ZMASS/engaging/setup_env_engaging.sh
export PYTHONPATH=$ZMASS/rabbit:$ZMASS/fullscale:$ZMASS/zchannel:${PYTHONPATH:-}
CARD=${CARD:?set CARD=...}
TAG=${TAG:-f380fl}
CH=${CH:-32768}
KFIX="k_hit k_ms k_ioni k_rad"
SHAPEFIX="shape1 shape2 shape3 shape4 shape5"
mkdir -p $ZMASS/fitresults
cd $ZMASS/fullscale
run () {  # run <label> <extra args...>
  local L=$1; shift
  echo "############ $L  $(date +%H:%M:%S)"
  python3 -u fit.py --card "$CARD" --chunk $CH --fix $KFIX "$@" \
      --label "${TAG}_$L" -o $ZMASS/fitresults/fit_${TAG}_$L.json \
    || echo "############ $L FAILED (exit $?)"
}
run base
run noares    --ares off
run nojensen  --jensen off
run noboth    --ares off --jensen off
run noshape   --fix $KFIX $SHAPEFIX
echo "############ done $(date +%H:%M:%S)"
