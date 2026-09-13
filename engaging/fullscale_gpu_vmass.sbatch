#!/bin/bash
#SBATCH --job-name=zfitv
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#SBATCH --time=05:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
# Full-scale Z fit on a GPU through the NATIVE TF minimiser (rabbit-vmass,
# branch vmass-conditioning (it carries the native minimiser too):
# `--method tf-trust-krylov` implies
# `--engine device`, measured 16x the scipy trust-exact default on the full
# card -- 1206 s against 19 600 s, parameters agreeing to 9.8e-7 relative).
# It needs `rabbit.unbinned.ChunkTable`, which is ONLY on that branch, so the
# PYTHONPATH points at `$ZMASS/rabbit-vmass` and not `$ZMASS/rabbit`.
#
# Submit with:
#   sbatch -A mit_general -G h200:1 -p mit_normal_gpu \
#     --export=ALL,CARD=$HOME/orcd/pool/zmass/cards/z_full.hdf5,ARGS="--chunk 262144" \
#     fullscale_gpu.sbatch base
# H200, not L40S: the card is large and the pfor Hessian is the memory driver.
set -euo pipefail
ZMASS=$HOME/orcd/pool/zmass
source $ZMASS/engaging/setup_env_engaging.sh
export PYTHONPATH=$ZMASS/rabbit-vmass:$ZMASS/fullscale:$ZMASS/zchannel:${PYTHONPATH:-}
LABEL=${1:-base}; shift || true
CARD=${CARD:?set CARD=...}
mkdir -p $ZMASS/fitresults
cd $ZMASS/fullscale
python3 -u fit.py --card "$CARD" --fix k_hit k_ms k_ioni k_rad \
    --label "$LABEL" -o $ZMASS/fitresults/fit_$LABEL.json ${ARGS:-} "$@"
