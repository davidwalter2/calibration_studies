#!/bin/bash
# Part 2: the same-sample comparison.  The part-1 window fits changed BOTH the
# likelihood and (implicitly) the weight of a handful of candidates whose
# sigma_m is comparable to the window itself -- sigma_m > 0.15 GeV is half the
# 2.95-3.25 window, so 1/P_i for those is a huge, parameter-dependent factor
# fitted on 56 (v3) / 1251 (gun) entries.  Here the SUBSET is fixed and only
# the likelihood changes.
set -u
cd "$(dirname "$0")"
RUN=/work/submit/david_w/ZMass/calibration_studies/env_tf/run_tf.sh
LOGD=runs/censoring260904/masslikfit; mkdir -p $LOGD
GUN_P=runs/cf_masspairs_jpsigun_ul16_260903x_m0.npz
GUN_K=runs/cf_masskernel_jpsigun_ul16_260903x_m0.npz
BTO_P=runs/cf_masspairs_btojpsix_v3_260903x_m0.npz
BTO_K=runs/cf_masskernel_btojpsix_v3_260903x_m0.npz
run () { nm=$1; shift
    [ -s "runs/masslikfit_${nm}.npz" ] && { echo "=== skip $nm"; return; }
    echo "=== $nm ($(date +%H:%M:%S)) ==="
    $RUN python3 cf_masslik_fit.py "$@" --tag "$nm" --no-plots \
        --out runs/masslikfit_${nm}.npz > $LOGD/${nm}.log 2>&1
    echo "   rc=$? -> $LOGD/${nm}.log"; }

M_V3=runs/censoring260904/mask_v3_winsane.npz
M_GUN=runs/censoring260904/mask_gun_sig015.npz
run btojpsix_v3_260904_fam_sane     --pairs-cache $BTO_P --kernel-cache $BTO_K \
    --model families --krad 1 --subset $M_V3
run btojpsix_v3_260904_fam_sane_win --pairs-cache $BTO_P --kernel-cache $BTO_K \
    --model families --krad 1 --subset $M_V3 --window-norm --window-lo 2.95 --window-hi 3.25
run btojpsix_v3_260904_r_sane       --pairs-cache $BTO_P --kernel-cache $BTO_K \
    --model r --subset $M_V3
run btojpsix_v3_260904_r_sane_win   --pairs-cache $BTO_P --kernel-cache $BTO_K \
    --model r --subset $M_V3 --window-norm --window-lo 2.95 --window-hi 3.25
run jpsigun_260904_fam_sane         --pairs-cache $GUN_P --kernel-cache $GUN_K \
    --model families --krad 1 --subset $M_GUN
run jpsigun_260904_fam_sane_win     --pairs-cache $GUN_P --kernel-cache $GUN_K \
    --model families --krad 1 --subset $M_GUN --window-norm --window-lo 2.7469 --window-hi 3.4469
echo "PART 2 DONE ($(date +%H:%M:%S))"
