#!/bin/bash
#SBATCH --job-name=znanstep
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#SBATCH --time=03:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
# `gate_nanstep.py` on a GPU: localise a `trust-exact` NaN to a point, a term,
# a resolution class and a candidate.
#
#   sbatch -A mit_general -p mit_preemptable -G h200:1 \
#     --export=ALL,CARD=$HOME/orcd/pool/zmass/cards/joint_ok_full.hdf5,\
# TAG=P2X,MODEL="ExternalParams bundle:global_params",\
# FREEZE="k_hit k_ms k_ioni k_rad ...",NOMIN=1 nanstep.sbatch
set -uo pipefail
ZMASS=$HOME/orcd/pool/zmass
source $ZMASS/engaging/setup_env_engaging.sh
export PYTHONPATH=$ZMASS/rabbit-vmass:$ZMASS/fullscale:$ZMASS/zchannel:$ZMASS/resolution/globalfit:${PYTHONPATH:-}
export MFS_DIR=$ZMASS/mfs
CARD=${CARD:?set CARD=...}
TAG=${TAG:-nanstep}
MODEL=${MODEL:-UnbinnedParams}
FREEZE=${FREEZE:-k_hit k_ms k_ioni k_rad}
cd $ZMASS/fullscale
echo "############ nanstep $TAG  $(date +%H:%M:%S)  card $CARD"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python3 -u gate_nanstep.py --card "$CARD" --model "$MODEL" \
    ${FREEZE:+--freeze $FREEZE} \
    $( [ "${NOMIN:-0}" = 1 ] && echo --no-min ) ${EXTRA:-}
echo "############ done $(date +%H:%M:%S) rc=$?"
