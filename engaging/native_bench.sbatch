#!/bin/bash
#SBATCH --job-name=zbench
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#SBATCH --time=05:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
# scipy-vs-native benchmark of the full-scale unbinned fit, on ONE GPU.
#
#   sbatch -A mit_general -p mit_preemptable -G h200:1 \
#     --export=ALL,CARD=$HOME/orcd/pool/zmass/cards/z_n300k.hdf5,TAG=n300k,\
#RUNS="host:trust-exact host:trust-krylov device:trust-krylov device:tf-trust-krylov device:tf-trust-ncg device:tf-trust-exact" \
#     native_bench.sbatch
#
# `mit_preemptable`, not `mit_normal_gpu`: measured today, every h200 job of
# ours started there inside 2.5 min while `mit_normal_gpu` was 5 h deep. It is
# PreemptMode=REQUEUE, which is exactly what the PR #155 snapshots are for --
# every run below writes one.
#
# RUNS is a space-separated list of `engine:method[:chunk]`. Each is a separate
# `fit.py` invocation on the SAME card from the SAME start, so the only thing
# that differs between two rows of the table is the minimiser and the engine.
set -uo pipefail
ZMASS=$HOME/orcd/pool/zmass
source $ZMASS/engaging/setup_env_engaging.sh
export PYTHONPATH=$ZMASS/rabbit_native:$ZMASS/fullscale_native:$ZMASS/zchannel:${PYTHONPATH:-}
CARD=${CARD:?set CARD=...}
TAG=${TAG:-bench}
FIX=${FIX:---fix k_hit k_ms k_ioni k_rad}
EXTRA=${EXTRA:-}
DRIVER=${DRIVER:-fit.py}
RUNS=${RUNS:-"host:trust-exact device:tf-trust-krylov"}
PERRUN=${PERRUN:-9000}
OUT=$ZMASS/fitresults/native
mkdir -p $OUT
cd $ZMASS/fullscale_native

echo "############ $TAG  $(date +%H:%M:%S)  card $CARD"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python3 -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))" 2>/dev/null | tail -1

for spec in $RUNS; do
    ENGINE=${spec%%:*}
    rest=${spec#*:}
    METHOD=${rest%%:*}
    CHUNK=${rest#*:}
    [ "$CHUNK" = "$METHOD" ] && CHUNK=0
    LBL="${TAG}_${ENGINE}_${METHOD}"
    [ "$CHUNK" != "0" ] && LBL="${LBL}_c${CHUNK}"
    echo
    echo "======== $LBL  $(date +%H:%M:%S)"
    timeout $PERRUN python3 -u $DRIVER \
        --card "$CARD" $FIX \
        --engine "$ENGINE" --method "$METHOD" \
        $( [ "$CHUNK" != "0" ] && echo --chunk $CHUNK ) \
        --gpu-monitor 1.0 --no-sandwich \
        --snapshot-file $OUT/${LBL}.snapshot.hdf5 --snapshot-interval 0.05 \
        --label "$LBL" -o $OUT/fit_${LBL}.json $EXTRA 2>&1
    echo "-------- $LBL done $(date +%H:%M:%S)"
done
echo "############ all done $(date +%H:%M:%S)"
