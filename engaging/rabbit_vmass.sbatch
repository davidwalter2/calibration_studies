#!/bin/bash
#SBATCH --job-name=zrabbitv
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#SBATCH --time=05:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
# THE MIGRATED PATH: an unbinned full-scale fit through `rabbit_fit.py`.
#
# The standalone drivers existed because an unbinned term did not fit in memory
# inside the Fitter. It now runs its candidate loop itself (rabbit.unbinned,
# chunk_mode="graph"), so this is an ordinary rabbit fit -- with rabbit's EDM,
# its termination convention, its snapshots and its output file.
#
#   sbatch -A mit_general -p mit_preemptable -G h200:1 \
#     --export=ALL,CARD=$HOME/orcd/pool/zmass/cards/z_n300k.hdf5,TAG=n300k,\
# FREEZE="k_hit k_ms k_ioni k_rad",METHOD=tf-trust-exact rabbit_native.sbatch
set -uo pipefail
ZMASS=$HOME/orcd/pool/zmass
source $ZMASS/engaging/setup_env_engaging.sh
export PYTHONPATH=$ZMASS/rabbit-vmass:$ZMASS/fullscale:$ZMASS/zchannel:${PYTHONPATH:-}
CARD=${CARD:?set CARD=...}
TAG=${TAG:-rabbit}
METHOD=${METHOD:-tf-trust-exact}
MODEL=${MODEL:-UnbinnedParams}
FREEZE=${FREEZE:-}
CHUNK=${CHUNK:-0}
EXTRA=${EXTRA:-}
GATEARGS=${GATEARGS:-}
OUT=$ZMASS/fitresults/native
mkdir -p $OUT
cd $ZMASS/fullscale

echo "############ $TAG  $(date +%H:%M:%S)  card $CARD  method $METHOD"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# the gate first: the Fitter must reproduce the standalone drivers before any
# number it produces is worth reading
if [ "${GATE:-1}" = "1" ]; then
    python3 -u test_rabbit_path.py --card "$CARD" --model "$MODEL" \
        $( [ "$CHUNK" != "0" ] && echo --chunk $CHUNK ) ${GATEARGS:-} || exit 1
fi

# `mit_preemptable` requeues rather than kills, and a requeued job starts the
# script again from the top -- so if a snapshot from the interrupted attempt is
# there, resume from it instead of re-descending from the prefit point. This is
# what makes a preemptable partition usable for a multi-hour fit.
SNAP=$OUT/rabbit_${TAG}.snapshot.hdf5
RESUME=""
if [ -f "$SNAP" ] && [ "${FRESH:-0}" != "1" ]; then
    echo "resuming from $SNAP"
    RESUME="--externalPostfit $SNAP"
fi

python3 -u $ZMASS/rabbit-vmass/bin/rabbit_fit.py "$CARD" \
    $RESUME \
    --paramModel "$MODEL" \
    --minimizerMethod "$METHOD" \
    ${FREEZE:+--freezeParameters $FREEZE} \
    $( [ "$CHUNK" != "0" ] && echo --unbinnedChunk $CHUNK ) \
    -t 0 --unblind --diagnostics \
    --snapshotFile $OUT/rabbit_${TAG}.snapshot.hdf5 --snapshotInterval 0.25 \
    --outpath $OUT --outname rabbit_${TAG}.hdf5 $EXTRA
echo "############ done $(date +%H:%M:%S)"
