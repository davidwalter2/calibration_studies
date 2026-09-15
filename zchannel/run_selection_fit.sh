#!/bin/bash
# The fit benchmark of the ASYMMETRIC cuts and of the RESOLUTION in the
# acceptance.  Every row of a suite is the SAME run on the SAME events, so the
# differences inside a table are same-run differences; the four suites are four
# different selections and are NOT comparable row by row across tables.
# Runs on a node that can mount /ceph (the container's mount hook binds it).
set -u
Z=/work/submit/david_w/ZMass/calibration_studies/zchannel
NODE=${NODE:-submit50}
WHICH=${1:-2510}
RES=data/ptres_dyv2.npz
SEL=""

case "$WHICH" in
2525)   # the symmetric cut, on the pT_ref = 10 GeV table: the bridge to the
        # published 25/25 numbers, which were built on the pT_ref = 25 one
  SEL="--acc-pt 25"
  REF="data/kern_cond_2525.npz:data/acc_cond_2525.json"
  ALT=(
   "corr, mc K=data/kern_corr_mc_2525.npz:data/acc_corr_mc_2525.json"
   "corr, data K=data/kern_corr_data_2525.npz:data/acc_corr_data_2525.json"
   "corr, mc K, pT_ref 25 table=data/kern_corr_mc_1gev.npz:data/acc_corr_mc_1gev.json"
   "cond: true, per-leg record=data/kern_cond_true.npz:data/acc_cond_true.json"
  ) ;;
2510)   # the asymmetric cut: leading 25, trailing 10
  SEL="--acc-pt 25 --acc-pt-trail 10"
  REF="data/kern_cond_2510.npz:data/acc_cond_2510.json"
  ALT=(
   "corr, mc K=data/kern_corr_mc_2510.npz:data/acc_corr_mc_2510.json"
   "corr, data K=data/kern_corr_data_2510.npz:data/acc_corr_data_2510.json"
   "corr, mc K, symmetric 25/25 kernel=data/kern_corr_mc_2525.npz:data/acc_corr_mc_2525.json"
  ) ;;
res2525)  # the resolution toy, symmetric cut
  SEL="--acc-pt 25 --smear $RES"
  REF="data/kern_cond_2525sm.npz:data/acc_cond_2525sm.json"
  ALT=(
   "corr, mc K, smeared acceptance=data/kern_corr_mc_2525sm.npz:data/acc_corr_mc_2525sm.json"
   "corr, mc K, STEP acceptance=data/kern_corr_mc_2525.npz:data/acc_corr_mc_2525.json"
   "corr, data K, smeared acceptance=data/kern_corr_data_2525sm.npz:data/acc_corr_data_2525sm.json"
   "corr, data K, STEP acceptance=data/kern_corr_data_2525.npz:data/acc_corr_data_2525.json"
  ) ;;
res2510)  # the resolution toy, asymmetric cut
  SEL="--acc-pt 25 --acc-pt-trail 10 --smear $RES"
  REF="data/kern_cond_2510sm.npz:data/acc_cond_2510sm.json"
  ALT=(
   "corr, mc K, smeared acceptance=data/kern_corr_mc_2510sm.npz:data/acc_corr_mc_2510sm.json"
   "corr, mc K, STEP acceptance=data/kern_corr_mc_2510.npz:data/acc_corr_mc_2510.json"
   "corr, data K, smeared acceptance=data/kern_corr_data_2510sm.npz:data/acc_corr_data_2510sm.json"
   "corr, data K, STEP acceptance=data/kern_corr_data_2510.npz:data/acc_corr_data_2510.json"
   "corr, mc K, smeared, Gaussian core only=data/kern_corr_mc_2510sm_gauss.npz:data/acc_corr_mc_2510sm_gauss.json"
   "corr, mc K, smeared, fine b grid=data/kern_corr_mc_2510sm_fineb.npz:data/acc_corr_mc_2510sm_fineb.json"
   "corr, mc K, smeared, 8 eta bands=data/kern_corr_mc_2510sm_eta8.npz:data/acc_corr_mc_2510sm_eta8.json"
  ) ;;
res2510g) # the same toy with a GAUSSIAN smearing: does the tail of r matter?
  SEL="--acc-pt 25 --acc-pt-trail 10 --smear $RES --smear-mode gauss"
  REF="data/kern_cond_2510smg.npz:data/acc_cond_2510smg.json"
  ALT=(
   "corr, mc K, smeared (gauss) acceptance=data/kern_corr_mc_2510sm_gauss.npz:data/acc_corr_mc_2510sm_gauss.json"
   "corr, mc K, smeared (shape) acceptance=data/kern_corr_mc_2510sm.npz:data/acc_corr_mc_2510sm.json"
   "corr, mc K, STEP acceptance=data/kern_corr_mc_2510.npz:data/acc_corr_mc_2510.json"
  ) ;;
*) echo "unknown suite $WHICH" >&2; exit 1 ;;
esac

K=${REF%%:*}
A=${REF##*:}
OUT=data/fit_selection_$WHICH.json

ssh "$NODE" "cd $Z && ./run_tf_z.sh python3 -u fit_gen.py fit \
  --gen data/genmerged_full.npz --suite perleg $SEL \
  --kernel $K --acc $A --nm 8192 \
  --kernel-alt $(printf "'%s' " "${ALT[@]}") -o $OUT"
