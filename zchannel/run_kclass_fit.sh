#!/bin/bash
# The fit benchmark of the RESOLUTION CLASSES: the same events fitted with the
# class-conditional kernel and with the population one, all classes
# simultaneously with common m_Z, Gamma_Z and shape terms.  Every row of a
# suite is the SAME run on the SAME events, so the differences inside a table
# are same-run differences.
#
#   ./run_kclass_fit.sh <2525|2510> <nclass> [single]
#
# `kpr` is the model whose pass region carries the class -- the only correct
# construction for a class read off the OBSERVED muons; `kcl` is the same
# class built as a restriction of the h table, which makes class membership a
# pre-FSR property and is kept as the control that shows it is not one.
set -u
Z=/work/submit/david_w/ZMass/calibration_studies/zchannel
NODE=${NODE:-submit50}
CUTS=${1:-2510}
NG=${2:-5}
SINGLE=${3:-}
NM=${NM:-4096}
case "$CUTS" in
2525) SEL="--acc-pt 25" ;;
2510) SEL="--acc-pt 25 --acc-pt-trail 10" ;;
*) echo "unknown selection $CUTS" >&2; exit 1 ;;
esac
if [ "$SINGLE" = single ]; then SINGLE="--kclass-single"; OUT=data/fit_kclass_${CUTS}_n${NG}_single.json
else SINGLE=""; OUT=data/fit_kclass_${CUTS}_n${NG}.json; fi

# A spec WITHOUT `{c}` is the population model used in every class -- the
# misspecified one; a spec WITH `{c}` is the class-conditional model.
MODELS=(
 "MC-cond per class=data/kern_kcl_cond_${CUTS}_n${NG}_c{c}.npz:data/acc_kcl_cond_${CUTS}_n${NG}_c{c}.json"
 "MC-cond population=data/kern_cond_${CUTS}.npz:data/acc_cond_${CUTS}.json"
 "corr mc K per class=data/kern_kpr_mc_${CUTS}_n${NG}_c{c}.npz:data/acc_kpr_mc_${CUTS}_n${NG}_c{c}.json"
 "corr mc K population=data/kern_corr_mc_${CUTS}.npz:data/acc_corr_mc_${CUTS}.json"
 "corr data K per class=data/kern_kpr_data_${CUTS}_n${NG}_c{c}.npz:data/acc_kpr_data_${CUTS}_n${NG}_c{c}.json"
 "corr data K population=data/kern_corr_data_${CUTS}.npz:data/acc_corr_data_${CUTS}.json"
 "corr mc K per class, restricted table=data/kern_kcl_mc_${CUTS}_n${NG}_c{c}.npz:data/acc_kcl_mc_${CUTS}_n${NG}_c{c}.json"
)
ALT=(
 "corr, mc K=data/kern_corr_mc_${CUTS}.npz:data/acc_corr_mc_${CUTS}.json"
 "corr, data K=data/kern_corr_data_${CUTS}.npz:data/acc_corr_data_${CUTS}.json"
)

ssh "$NODE" "cd $Z && ./run_tf_z.sh python3 -u fit_gen.py fit \
  --gen data/genmerged_full.npz --suite kclass $SEL \
  --kernel data/kern_cond_${CUTS}.npz --acc data/acc_cond_${CUTS}.json \
  --nm $NM --kclass-json data/kcl10_2510.json --kclass-ngroup $NG \
  --kclass-mass-smear $SINGLE \
  --kernel-alt $(printf "'%s' " "${ALT[@]}") \
  --kclass-model $(printf "'%s' " "${MODELS[@]}") -o $OUT"
