#!/bin/bash
# The correlated two-leg kernel: the exact O(alpha) recoil sharing replaces
# D(x_+) D(x_-) at fixed z, with K(z) untouched.  `single` is the model; the
# matching-scale scan of the alternative composition is the systematic.
# numpy only.
set -eu
cd /work/submit/david_w/ZMass/calibration_studies/zchannel
P="python3 -u -W ignore"
HT=data/ht_pt25_1.0gev.npz
RUN=data/photos/gen_mcMix.npz
PAIRS="--pair e mu tau had"

# --- the model: the exact single-photon sharing applied to the whole loss
$P fsr_perleg.py corr --htable $HT -o data/kern_corr_data_1gev.npz \
   --acceptance data/acc_corr_data_1gev.json $PAIRS
$P fsr_perleg.py corr --htable $HT --run $RUN -o data/kern_corr_mc_1gev.npz \
   --acceptance data/acc_corr_mc_1gev.json

# --- matching-scale scan: exact hard emission above u_c x collinear-independent
#     soft remainder below it.  K = D_< (x) D_< (x) H_> exactly at every u_c.
for UC in 0.03 0.01 0.003; do
  T=$(echo $UC | tr -d '.')
  $P fsr_perleg.py corr --htable $HT --run $RUN --mode matched --u-c $UC \
     -o data/kern_corr_mc_uc$T.npz --acceptance data/acc_corr_mc_uc$T.json \
     --rho data/rho_uc${T}_mc.npz
done

