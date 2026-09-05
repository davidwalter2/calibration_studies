#!/bin/bash
# The mass likelihood FIT on the _260903x pairs caches, which carry the
# radiative (brems + pair) family.
#
# Four fits per sample:
#   fam_krad1  families, k_rad fixed at 1        -- the physics model
#   fam_krad0  families, k_rad fixed at 0        -- the model of 2026-09-02,
#              i.e. the CONTROL: same sample, same code, one term removed
#   fam_kradf  families + k_rad floated (5 par)  -- what the data wants
#   r          one resolution scale multiplying ALL FOUR families
# Sequential: 32 TF threads each, and the node is shared.
set -u
cd "$(dirname "$0")"
RUN=/work/submit/david_w/ZMass/calibration_studies/env_tf/run_tf.sh
LOGD=${LOGD:-runs/rad260903x/masslikfit}
mkdir -p $LOGD

GUN_P=runs/cf_masspairs_jpsigun_ul16_260903x_m0.npz
GUN_K=runs/cf_masskernel_jpsigun_ul16_260903x_m0.npz
BTO_P=runs/cf_masspairs_btojpsix_v3_260903x_m0.npz
BTO_K=runs/cf_masskernel_btojpsix_v3_260903x_m0.npz

run () {  # run <name> <extra args...>
    nm=$1; shift
    [ -s "runs/masslikfit_${nm}.npz" ] && { echo "=== skip $nm (exists)"; return; }
    echo "=== $nm ($(date +%H:%M:%S)) ==="
    $RUN python3 cf_masslik_fit.py "$@" --tag "$nm" --no-plots \
        --out runs/masslikfit_${nm}.npz > $LOGD/${nm}.log 2>&1
    echo "   rc=$? -> $LOGD/${nm}.log"
}

for S in gun bto; do
  if [ $S = gun ]; then P=$GUN_P; K=$GUN_K; T=jpsigun_260903x; else P=$BTO_P; K=$BTO_K; T=btojpsix_v3_260903x; fi
  [ -s "$P" ] || { echo "=== skip $T (no $P)"; continue; }
  run ${T}_fam_krad1 --pairs-cache $P --kernel-cache $K --model families --krad 1
  run ${T}_fam_krad0 --pairs-cache $P --kernel-cache $K --model families --krad 0
  run ${T}_fam_kradf --pairs-cache $P --kernel-cache $K --model families --float-krad
  run ${T}_r         --pairs-cache $P --kernel-cache $K --model r
done
echo "ALL DONE ($(date +%H:%M:%S))"
