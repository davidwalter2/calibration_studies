#!/bin/bash
# Part 3 -- the CONTROLLED-TRUNCATION closure, and the validation gates on the gun.
#
# The J/psi gun has NO reco-mass window anywhere in its chain (legacy pair loop
# over generalTracks + requireGen only).  So a hand-applied window on m_reco is
# a truncation of EXACTLY the form the L_i/[F(hi)-F(lo)] correction assumes: a
# sharp cut in the SAME variable the likelihood models.  Three fits per width:
#
#   sane            reference alpha on the untruncated population
#   cutNNN          the same fit after the cut, WITHOUT the correction  -> bias
#   cutNNN_win      the same fit after the cut, WITH the correction     -> recovery
#
# If the correction is right, cutNNN_win == sane and cutNNN - sane measures how
# many 1e-3 of alpha a censoring of that width costs.  (v3's real cut is on the
# PRE-refit mass, so for v3 the same correction is only an approximation; this
# is the exact case.)
set -u
cd "$(dirname "$0")"
RUN=/work/submit/david_w/ZMass/calibration_studies/env_tf/run_tf.sh
LOGD=runs/censoring260904/masslikfit; mkdir -p $LOGD
GUN_P=runs/cf_masspairs_jpsigun_ul16_260903x_m0.npz
GUN_K=runs/cf_masskernel_jpsigun_ul16_260903x_m0.npz
BTO_P=runs/cf_masspairs_btojpsix_v3_260903x_m0.npz
BTO_K=runs/cf_masskernel_btojpsix_v3_260903x_m0.npz
M=runs/censoring260904

# wait for any earlier chain to release the 32-thread slot
while pgrep -u "$USER" -f "chain_windownorm2_260904.sh" > /dev/null; do sleep 20; done

run () { nm=$1; shift
    [ -s "runs/masslikfit_${nm}.npz" ] && { echo "=== skip $nm"; return; }
    echo "=== $nm ($(date +%H:%M:%S)) ==="
    $RUN python3 cf_masslik_fit.py "$@" --tag "$nm" --no-plots \
        --out runs/masslikfit_${nm}.npz > $LOGD/${nm}.log 2>&1
    echo "   rc=$? -> $LOGD/${nm}.log"; }

# ---------------- gun: reference on the sane population -------------------
run gun_260904_sane_fam   --pairs-cache $GUN_P --kernel-cache $GUN_K \
    --model families --krad 1 --subset $M/mask_gun_sane.npz
run gun_260904_sane_r     --pairs-cache $GUN_P --kernel-cache $GUN_K \
    --model r --subset $M/mask_gun_sane.npz

# ---------------- gun: the v3-sized window, +-0.15 GeV = 4.6 sigma ---------
run gun_260904_cut015_fam     --pairs-cache $GUN_P --kernel-cache $GUN_K \
    --model families --krad 1 --subset $M/mask_gun_cut015.npz
run gun_260904_cut015_fam_win --pairs-cache $GUN_P --kernel-cache $GUN_K \
    --model families --krad 1 --subset $M/mask_gun_cut015.npz \
    --window-norm --window-lo 2.9469 --window-hi 3.2469
run gun_260904_cut015_r       --pairs-cache $GUN_P --kernel-cache $GUN_K \
    --model r --subset $M/mask_gun_cut015.npz
run gun_260904_cut015_r_win   --pairs-cache $GUN_P --kernel-cache $GUN_K \
    --model r --subset $M/mask_gun_cut015.npz \
    --window-norm --window-lo 2.9469 --window-hi 3.2469

# ---------------- gun: a HARD window, +-0.05 GeV = 1.5 sigma --------------
run gun_260904_cut005_fam     --pairs-cache $GUN_P --kernel-cache $GUN_K \
    --model families --krad 1 --subset $M/mask_gun_cut005.npz
run gun_260904_cut005_fam_win --pairs-cache $GUN_P --kernel-cache $GUN_K \
    --model families --krad 1 --subset $M/mask_gun_cut005.npz \
    --window-norm --window-lo 3.0469 --window-hi 3.1469
run gun_260904_cut005_r       --pairs-cache $GUN_P --kernel-cache $GUN_K \
    --model r --subset $M/mask_gun_cut005.npz
run gun_260904_cut005_r_win   --pairs-cache $GUN_P --kernel-cache $GUN_K \
    --model r --subset $M/mask_gun_cut005.npz \
    --window-norm --window-lo 3.0469 --window-hi 3.1469

# ---------------- v3: sensitivity to the ASSUMED window width -------------
# the real cut is on the PRE-refit mass, so the effective post-refit edge is
# smeared; this maps how much that ambiguity costs.
run v3_260904_sane_fam        --pairs-cache $BTO_P --kernel-cache $BTO_K \
    --model families --krad 1 --subset $M/mask_v3_sane.npz
run v3_260904_win290_330_fam --pairs-cache $BTO_P --kernel-cache $BTO_K \
    --model families --krad 1 --subset $M/mask_v3_post290_330.npz \
    --window-norm --window-lo 2.90 --window-hi 3.30
run v3_260904_win285_335_fam --pairs-cache $BTO_P --kernel-cache $BTO_K \
    --model families --krad 1 --subset $M/mask_v3_post285_335.npz \
    --window-norm --window-lo 2.85 --window-hi 3.35
run v3_260904_win295_325_fam --pairs-cache $BTO_P --kernel-cache $BTO_K \
    --model families --krad 1 --subset $M/mask_v3_post295_325.npz \
    --window-norm --window-lo 2.95 --window-hi 3.25
echo "PART 3 DONE ($(date +%H:%M:%S))"
