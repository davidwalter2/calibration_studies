#!/bin/bash
#SBATCH --job-name=zjoint
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#SBATCH --time=05:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
# PHASE 2 -- the joint J/psi + Z + hit-chi2 fit, on one GPU.
#
#   sbatch -A mit_general -G h200:1 -p mit_normal_gpu \
#     --export=ALL,CARD=$HOME/orcd/pool/zmass/cards/joint_v2.hdf5,TAG=joint_v2 \
#     joint_gpu.sbatch
#
# `--hess-mode pfor`, not the driver's `hvp` default: `trust-exact` needs the
# FULL Hessian at every iteration, and over 99 free parameters pfor is ~4x
# faster than 99 sequential HVPs.  It is also ~99x the memory, which is why
# this asks for an H200 and 160 GB of host RAM; if it OOMs, fall back to
# `HESS=hvp` and expect ~4x the wall time.
#
# `--chunk` is NOT passed: the card carries a per-candidate sparse D whose
# per-chunk slices are built at write time, so re-chunking it crashes.
set -uo pipefail
ZMASS=$HOME/orcd/pool/zmass
source $ZMASS/engaging/setup_env_engaging.sh
export PYTHONPATH=$ZMASS/rabbit:$ZMASS/fullscale:$ZMASS/zchannel:${PYTHONPATH:-}
CARD=${CARD:?set CARD=...}
TAG=${TAG:-joint_v2}
HESS=${HESS:-pfor}
mkdir -p $ZMASS/fitresults
cd $ZMASS/fullscale
echo "############ $TAG  $(date +%H:%M:%S)  card $CARD  hess $HESS"
python3 -u fit_joint.py --card "$CARD" --fix k_hit k_ms k_ioni k_rad \
    --hess-mode "$HESS" --label "$TAG" \
    -o $ZMASS/fitresults/fit_${TAG}.json ${ARGS:-}
echo "############ done $(date +%H:%M:%S)"
