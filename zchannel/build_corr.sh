#!/bin/bash
# The correlated two-leg kernel: the exact O(alpha) recoil sharing replaces
# D(x_+) D(x_-) at fixed z, with K(z) untouched.  `single` is the model; the
# matching-scale scan of the alternative composition is the systematic.
#
# `--atoms` throughout: these are the LEGACY banded kernels the published
# atom rows of the per-leg sections were measured on.  The same kernels in
# the default cell-integrated table form are `build_machinery_table.sh`.
# numpy only.
set -eu
cd /work/submit/david_w/ZMass/calibration_studies/zchannel
P="python3 -u -W ignore"
HT=data/ht_pt25_1.0gev.npz
RUN=data/photos/gen_mcMix.npz
PAIRS="--pair e mu tau had"

# --- the model: the exact single-photon sharing applied to the whole loss
$P fsr_perleg.py corr --atoms --htable $HT -o data/kern_corr_data_1gev.npz \
   --acceptance data/acc_corr_data_1gev.json $PAIRS
$P fsr_perleg.py corr --atoms --htable $HT --run $RUN -o data/kern_corr_mc_1gev.npz \
   --acceptance data/acc_corr_mc_1gev.json

# --- matching-scale scan: exact hard emission above u_c x collinear-independent
#     soft remainder below it.  K = D_< (x) D_< (x) H_> exactly at every u_c.
for UC in 0.03 0.01 0.003; do
  T=$(echo $UC | tr -d '.')
  $P fsr_perleg.py corr --atoms --htable $HT --run $RUN --mode matched --u-c $UC \
     -o data/kern_corr_mc_uc$T.npz --acceptance data/acc_corr_mc_uc$T.json \
     --rho data/rho_uc${T}_mc.npz
done

# --- the multi-emission sharing: the exact O(alpha) angle carried by EVERY
#     photon of the kernel's Levy measure, exponentiated.  `coll` is its
#     collinear limit and must reproduce the per-leg product; `h` is the only
#     discretisation knob.
M() { t=$1; mode=$2; shift 2
  [ -s data/kern_corr_mc_$t.npz ] || $P fsr_perleg.py corr --atoms --htable $HT \
     --run $RUN --mode $mode -o data/kern_corr_mc_$t.npz \
     --acceptance data/acc_corr_mc_$t.json --rho data/rho_${t}_mc.npz "$@"; }
M multi  multi
M coll   coll
M multih25 multi --multi-h 2.5e-4
M multip32 multi --share-npanel 32 --share-ng 8
M multif4 multi --share-floor 1e-4

[ -s data/kern_corr_data_multi.npz ] || $P fsr_perleg.py corr --atoms --htable $HT \
   --mode multi -o data/kern_corr_data_multi.npz \
   --acceptance data/acc_corr_data_multi.json --rho data/rho_multi_data.npz \
   $PAIRS

# --- the same construction under the asymmetric 25/10 cut, on the pT_ref = 10
#     table that `build_selection.sh` writes
HT10=data/ht_ref10_1.0gev.npz
S() { t=$1; shift
  [ -s data/kern_corr_mc_$t.npz ] || $P fsr_perleg.py corr --atoms --htable $HT10 \
     --run $RUN -o data/kern_corr_mc_$t.npz \
     --acceptance data/acc_corr_mc_$t.json --rho data/rho_${t}_mc.npz "$@"
  [ -s data/kern_corr_data_$t.npz ] || $P fsr_perleg.py corr --atoms --htable $HT10 \
     -o data/kern_corr_data_$t.npz \
     --acceptance data/acc_corr_data_$t.json --rho data/rho_${t}_data.npz \
     $PAIRS "$@"; }
S 2525multi --pt-cuts 25 25 --mode multi
S 2510multi --pt-cuts 25 10 --mode multi
S 2510coll  --pt-cuts 25 10 --mode coll
M lin lin
