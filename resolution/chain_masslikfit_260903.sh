#!/bin/bash
# Step 1 of "replace the (alpha, r) scan by a fit": all cf_masslik_fit.py runs.
#
# Environment: the rabbit TensorFlow stack (wmassdev singularity) via
#   /work/submit/david_w/ZMass/calibration_studies/env_tf/run_tf.sh
# Runs are SEQUENTIAL on purpose: 32 TF threads each, and another agent is
# running CMSSW productions on this node.
set -u
cd "$(dirname "$0")"
RUN=/work/submit/david_w/ZMass/calibration_studies/env_tf/run_tf.sh
LOGD=${LOGD:-runs/260903_masslikfit}
mkdir -p $LOGD

GUN_P=runs/cf_masspairs_jpsigun_ul16_260902_m0_fixsign.npz
GUN_K=runs/cf_masskernel_jpsigun_ul16_260902_m0.npz
GUN_S=runs/masslik_demo_scan_jpsigun_260902_fixsign.npz

BTO_P=runs/cf_masspairs_btojpsix_v3_260902_m0_fixsign.npz
BTO_K=runs/cf_masskernel_btojpsix_v3_260902_m0.npz
BTO_S=runs/masslik_demo_scan_btojpsix_v3_260902_fixsign.npz

run () {  # run <name> <extra args...>
    nm=$1; shift
    echo "=== $nm ==="
    $RUN python3 cf_masslik_fit.py "$@" --tag "$nm" \
        --out runs/masslikfit_${nm}.npz > $LOGD/${nm}.log 2>&1
    echo "   rc=$? -> $LOGD/${nm}.log"
}

# ---- J/psi gun (no FSR tail) -------------------------------------------
run jpsigun_r        --pairs-cache $GUN_P --kernel-cache $GUN_K \
                     --model r --scan-check $GUN_S --profile alpha --fd-check
run jpsigun_fam      --pairs-cache $GUN_P --kernel-cache $GUN_K \
                     --model families --scan-check $GUN_S --no-plots
run jpsigun_fam_fbkg --pairs-cache $GUN_P --kernel-cache $GUN_K \
                     --model families --float-bkg --scan-check $GUN_S --no-plots
run jpsigun_r_f32    --pairs-cache $GUN_P --kernel-cache $GUN_K \
                     --model r --precision float32 --scan-check $GUN_S --no-plots

# ---- B -> J/psi + X v3 (FSR-bearing) -----------------------------------
run btojpsix_r        --pairs-cache $BTO_P --kernel-cache $BTO_K \
                      --model r --scan-check $BTO_S --profile alpha --fd-check
run btojpsix_fam      --pairs-cache $BTO_P --kernel-cache $BTO_K \
                      --model families --scan-check $BTO_S --no-plots
run btojpsix_fam_fbkg --pairs-cache $BTO_P --kernel-cache $BTO_K \
                      --model families --float-bkg --scan-check $BTO_S --no-plots
echo "ALL DONE"
