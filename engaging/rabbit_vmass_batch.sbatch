#!/bin/bash
#SBATCH --job-name=zrabbitvb
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#SBATCH --time=11:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
# Several certified-table rows on ONE GPU, in sequence.
#
# A WARM row converges in 3-5 iterations (~10 min) but pays ~5 min of card load
# and TF start-up, and the GPU allowance is 4 on `mit_preemptable`. Running
# them one process per row therefore spends most of the allowance on start-up
# and blocks the long jobs (phase 2) behind them. This runs a list in one
# allocation instead. Each row is a separate `rabbit_fit.py` process, so a row
# that fails does not take the others with it.
#
#   sbatch ... --export=ALL,ROWS="SVs6 SVs7 SVetaT",FRESH=0 rabbit_vmass_batch.sbatch
set -uo pipefail
ZMASS=$HOME/orcd/pool/zmass
source $ZMASS/engaging/setup_env_engaging.sh
export PYTHONPATH=$ZMASS/rabbit-vmass:$ZMASS/fullscale:$ZMASS/zchannel:${PYTHONPATH:-}
declare -A CARDS=(
  [f380refS]=z_full380_fl      [Ss6]=z_full380_fl_s6   [Ss7]=z_full380_fl_s7
  [Sdc8]=z_F_dc8               [Stoy]=z_F_toy          [Stoydc]=z_F_toydc
  [Sw70110]=z_F_w70110
  [SMetaB]=z_M_etaB            [SMetaT]=z_M_etaT       [SMetaE]=z_M_etaE
  [SVfull]=z_V_full            [SVtoy]=z_V_toy
  [SVs6]=z_V_s6                [SVs7]=z_V_s7
  [SVetaB]=z_V_etaB            [SVetaT]=z_V_etaT       [SVetaE]=z_V_etaE
  [SVKetaB]=z_VK_etaB          [SVKetaT]=z_VK_etaT
  [f380refW]=z_full380_fl      [SVfullW]=z_V_full      [Sdc8W]=z_F_dc8
  # the extended K(m) ladder: every added Legendre term is still a
  # 13-15 sigma improvement at 5->6->7, so the basis is not saturated
  [Ss9]=z_full380_fl_s9        [SVs9]=z_V_s9
  [Ss12]=z_full380_fl_s12      [SVs12]=z_V_s12
  # the preconditioning control (sec. 0f.14 dropped --precondition
  # against the SINGULAR-subproblem cause; with scipy's subproblem
  # that cause is gone and the trust radius scaling is what is left)
  [f380refP]=z_full380_fl
)
FREEZE=${FREEZE:-k_hit k_ms k_ioni k_rad}
METHOD=${METHOD:-trust-exact}
OUT=$ZMASS/fitresults/native
mkdir -p $OUT
cd $ZMASS/fullscale
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
for TAG in ${ROWS:?set ROWS="tag tag ..."}; do
    CARD=$ZMASS/cards/${CARDS[$TAG]:?unknown row $TAG}.hdf5
    SNAP=$OUT/rabbit_${TAG}.snapshot.hdf5
    RESUME=""
    if [ -f "$SNAP" ] && [ "${FRESH:-0}" != "1" ]; then RESUME="--externalPostfit $SNAP"; fi
    echo "############ $TAG  $(date +%H:%M:%S)  card $CARD  method $METHOD ${RESUME:+(warm)}"
    python3 -u $ZMASS/rabbit-vmass/bin/rabbit_fit.py "$CARD" $RESUME \
        --paramModel ${MODEL:-UnbinnedParams} --minimizerMethod "$METHOD" \
        ${FREEZE:+--freezeParameters $FREEZE} \
        -t 0 --unblind --diagnostics \
        --snapshotFile $SNAP --snapshotInterval 0.25 \
        --outpath $OUT --outname rabbit_${TAG}.hdf5 ${EXTRA:-}
    echo "############ $TAG done $(date +%H:%M:%S) rc=$?"
done
