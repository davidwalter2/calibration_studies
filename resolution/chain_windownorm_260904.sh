#!/bin/bash
# 2026-09-04 CENSORING TEST -- the mass fit with and without the
# window-truncated likelihood.
#
#   gun : the maker has NO reco-mass window at all (legacy pair loop over
#         generalTracks, requireGen only), so --window-norm at +-0.35 GeV is a
#         NULL TEST: it must move nothing.
#   v3  : the candidates come from ALCARECOTkAlJpsiXJpsiOnlyCandidates, which
#         cuts the PRE-REFIT dimuon mass at 2.95 < m < 3.25 GeV.  That is a
#         real truncation (3-7 sigma_m), and this is its exact treatment.
set -u
cd "$(dirname "$0")"
RUN=/work/submit/david_w/ZMass/calibration_studies/env_tf/run_tf.sh
LOGD=${LOGD:-runs/censoring260904/masslikfit}
mkdir -p $LOGD

GUN_P=runs/cf_masspairs_jpsigun_ul16_260903x_m0.npz
GUN_K=runs/cf_masskernel_jpsigun_ul16_260903x_m0.npz
BTO_P=runs/cf_masspairs_btojpsix_v3_260903x_m0.npz
BTO_K=runs/cf_masskernel_btojpsix_v3_260903x_m0.npz

run () { nm=$1; shift
    [ -s "runs/masslikfit_${nm}.npz" ] && { echo "=== skip $nm (exists)"; return; }
    echo "=== $nm ($(date +%H:%M:%S)) ==="
    $RUN python3 cf_masslik_fit.py "$@" --tag "$nm" --no-plots \
        --out runs/masslikfit_${nm}.npz > $LOGD/${nm}.log 2>&1
    echo "   rc=$? -> $LOGD/${nm}.log"
}

# ---- B -> J/psi X : the sample that HAS a window --------------------------
run btojpsix_v3_260904_fam_win  --pairs-cache $BTO_P --kernel-cache $BTO_K \
    --model families --krad 1 --window-norm --window-lo 2.95 --window-hi 3.25
run btojpsix_v3_260904_r_win    --pairs-cache $BTO_P --kernel-cache $BTO_K \
    --model r --window-norm --window-lo 2.95 --window-hi 3.25
run btojpsix_v3_260904_fam_kradf_win --pairs-cache $BTO_P --kernel-cache $BTO_K \
    --model families --float-krad --window-norm --window-lo 2.95 --window-hi 3.25
# a WIDE window is the internal null: 10x the resolution, nothing truncated
run btojpsix_v3_260904_fam_winwide --pairs-cache $BTO_P --kernel-cache $BTO_K \
    --model families --krad 1 --window-norm --window-lo 2.0 --window-hi 4.2

# ---- J/psi gun : the null test -------------------------------------------
run jpsigun_260904_fam_win --pairs-cache $GUN_P --kernel-cache $GUN_K \
    --model families --krad 1 --window-norm --window-lo 2.7469 --window-hi 3.4469
run jpsigun_260904_r_win   --pairs-cache $GUN_P --kernel-cache $GUN_K \
    --model r --window-norm --window-lo 2.7469 --window-hi 3.4469
echo "ALL DONE ($(date +%H:%M:%S))"
