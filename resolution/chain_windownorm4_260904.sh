#!/bin/bash
# Part 4 -- THE FIT-PATHOLOGY population, which the census found carries more
# than half of the observed candidate-mass asymmetry on the J/psi gun.
#
#   gun: normchi2 > 3 on 12.3 % of candidates (q99 = 1.0e5), `frozen` (the
#        Gauss-Newton momentum-floor clamp at 2 GeV scaled the step to zero,
#        so the refit mass IS the seed mass) on 7.0 %, and 24 % of candidates
#        have a daughter below 2 GeV -- i.e. inside the clamp's reach.
#        <z e^{-0.05 z^2}>: 0.0361 (all) -> 0.0167 (normchi2<3) -> 0.0231
#        (not frozen).  The CF model describes the IDEAL LINEAR ESTIMATOR
#        given the noise record; for a clamped or non-converged candidate that
#        is not the estimator that produced the number, so those rows are
#        model-less, not censored.
#   v3 : normchi2 > 3 on 0.83 %, frozen on 0.006 % -- the same cuts are a null
#        test there (daughters are above the 3 GeV trigger threshold).
set -u
cd "$(dirname "$0")"
RUN=/work/submit/david_w/ZMass/calibration_studies/env_tf/run_tf.sh
LOGD=runs/censoring260904/masslikfit; mkdir -p $LOGD
GUN_P=runs/cf_masspairs_jpsigun_ul16_260903x_m0.npz
GUN_K=runs/cf_masskernel_jpsigun_ul16_260903x_m0.npz
BTO_P=runs/cf_masspairs_btojpsix_v3_260903x_m0.npz
BTO_K=runs/cf_masskernel_btojpsix_v3_260903x_m0.npz
M=runs/censoring260904

while pgrep -u "$USER" -f "chain_windownorm3_260904.sh" > /dev/null; do sleep 20; done
while pgrep -u "$USER" -f "prebuild_phik_260904.py" > /dev/null; do sleep 20; done

run () { nm=$1; shift
    [ -s "runs/masslikfit_${nm}.npz" ] && { echo "=== skip $nm"; return; }
    echo "=== $nm ($(date +%H:%M:%S)) ==="
    $RUN python3 cf_masslik_fit.py "$@" --tag "$nm" --no-plots \
        --out runs/masslikfit_${nm}.npz > $LOGD/${nm}.log 2>&1
    echo "   rc=$? -> $LOGD/${nm}.log"; }

for S in q3 q3nf nofrozen pt2 pt3 pt3q3; do
  run gun_260904_${S}_fam --pairs-cache $GUN_P --kernel-cache $GUN_K \
      --model families --krad 1 --subset $M/mask_gun_${S}.npz
  run gun_260904_${S}_r   --pairs-cache $GUN_P --kernel-cache $GUN_K \
      --model r --subset $M/mask_gun_${S}.npz
done
for S in q3 q3nf pt3q3; do
  run v3_260904_${S}_fam --pairs-cache $BTO_P --kernel-cache $BTO_K \
      --model families --krad 1 --subset $M/mask_v3_${S}.npz
done
echo "PART 4 DONE ($(date +%H:%M:%S))"
