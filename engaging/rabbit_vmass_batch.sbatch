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
# RABBIT selects WHICH checkout the fit runs from, exactly as in
# `rabbit_vmass.sbatch`.  A row may override it with the `tag@checkout` form in
# ROWS, so one allocation can run the SAME card on two checkouts and settle
# whether a code change is a no-op (STATE 0f.72).  Until 2026-09-09 this script
# hard-coded `rabbit-vmass`, which is what made every card built after the
# `corr_a_max` commit unfittable here: the card carries the keyword and the
# staged checkout does not know it.
RABBIT=${RABBIT:-$ZMASS/rabbit-vmass}
PYBASE=$ZMASS/fullscale:$ZMASS/zchannel:${PYTHONPATH:-}
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
  # sigma/m split AT FIXED eta: does the v-form residual track the
  # resolution or the pseudorapidity? They are nearly collinear
  # inclusively, so this is the only clean separation.
  [SVetaBslo]=z_V_etaB_slo     [SVetaBshi]=z_V_etaB_shi
  [SVetaEslo]=z_V_etaE_slo     [SVetaEshi]=z_V_etaE_shi
  # THE SAFE BAND DEFINITION, max(|eta_p|,|eta_m|) instead of the |eta| of the
  # leg with the larger RECO pT: the latter makes the band edge a cut on the
  # residual (corr 0.0203 against 0.0025). STATE sec. 0f.63/0f.66.
  [VXetaB]=z_VX_etaB           [VXetaT]=z_VX_etaT      [VXetaE]=z_VX_etaE
  [VXetaBslo]=z_VX_etaB_slo    [VXetaBshi]=z_VX_etaB_shi
  # THE CODE/KEYWORD EQUIVALENCE TRIPLE (STATE 0f.72).  Same 300 k card, same
  # freeze, same method: `n300kV` on the staged `rabbit-vmass`, `n300kW` on the
  # checkout that understands `corr_a_max`, `n300kWK` on that checkout with a
  # card that CARRIES the keyword at its default 0.0.  All three must agree to
  # the last digit in NLL, EDM and every fitted value, or the new rows are not
  # comparable with the certified table.
  [n300kV]=z_n300k             [n300kW]=z_n300k        [n300kWK]=z_n300k_k
)
FREEZE=${FREEZE:-k_hit k_ms k_ioni k_rad}
METHOD=${METHOD:-trust-exact}
OUT=$ZMASS/fitresults/native
mkdir -p $OUT
cd $ZMASS/fullscale
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
for SPEC in ${ROWS:?set ROWS="tag tag ..."}; do
    TAG=${SPEC%%@*}
    if [ "$SPEC" = "$TAG" ]; then RB=$RABBIT; else RB=$ZMASS/${SPEC#*@}; fi
    export PYTHONPATH=$RB:$PYBASE
    CARD=$ZMASS/cards/${CARDS[$TAG]:?unknown row $TAG}.hdf5
    SNAP=$OUT/rabbit_${TAG}.snapshot.hdf5
    RESUME=""
    if [ -f "$SNAP" ] && [ "${FRESH:-0}" != "1" ]; then RESUME="--externalPostfit $SNAP"; fi
    echo "############ $TAG  $(date +%H:%M:%S)  card $CARD  method $METHOD ${RESUME:+(warm)}"
    echo "############   RABBIT=$RB"
    echo "############   FREEZE=${FREEZE:-<none>}  EXTRA=${EXTRA:-<none>}"
    python3 -u $RB/bin/rabbit_fit.py "$CARD" $RESUME \
        --paramModel ${MODEL:-UnbinnedParams} --minimizerMethod "$METHOD" \
        ${FREEZE:+--freezeParameters $FREEZE} \
        -t 0 --unblind --diagnostics \
        --snapshotFile $SNAP --snapshotInterval 0.25 \
        --outpath $OUT --outname rabbit_${TAG}.hdf5 ${EXTRA:-}
    # STATE 0f.71.7: this used to read `rc=$?` AFTER a `date` substitution in
    # the same line, so it reported the echo's status and every crashed stage
    # logged rc=0.  Capture it first.
    rc=$?
    echo "############ $TAG done $(date +%H:%M:%S) rc=$rc"
done
