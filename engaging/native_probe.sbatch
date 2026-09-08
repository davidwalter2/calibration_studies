#!/bin/bash
#SBATCH --job-name=zprobe
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
# Host-vs-device cross-check and chunk-size scan of the unbinned objective.
#
#   sbatch -A mit_general -p mit_preemptable -G h200:1 \
#     --export=ALL,CARD=$HOME/orcd/pool/zmass/cards/z_n300k.hdf5,\
#CHUNKS="8192 32768 131072 300000" native_probe.sbatch
set -uo pipefail
ZMASS=$HOME/orcd/pool/zmass
source $ZMASS/engaging/setup_env_engaging.sh
export PYTHONPATH=$ZMASS/rabbit_native:$ZMASS/fullscale_native:$ZMASS/zchannel:${PYTHONPATH:-}
CARD=${CARD:?set CARD=...}
CHUNKS=${CHUNKS:-"8192 32768 131072"}
EXTRA=${EXTRA:-}
cd $ZMASS/fullscale_native
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "############ probe $(date +%H:%M:%S)  card $CARD  chunks $CHUNKS"
python3 -u probe_native.py --card "$CARD" --chunks $CHUNKS $EXTRA
echo "############ done $(date +%H:%M:%S)"
