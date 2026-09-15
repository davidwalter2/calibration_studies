#!/bin/bash
# The machinery-only benchmark with every row in the CELL-INTEGRATED table
# representation.  `fsr_table.py fit` runs `fit_gen`'s closure on an explicit
# row list, so an atom row and its table are the SAME events, the same window
# and the same five K(m) terms, and every difference below is a same-run
# difference.
set -u
Z=/work/submit/david_w/ZMass/calibration_studies/zchannel
NODE=${NODE:-submit81}
WHICH=${1:-main}
COND="data/kern_cond_2525.npz:data/acc_cond_2525.json"

if [ "$WHICH" = "main" ]; then
# the four numbers of the test, plus the two cross rows that separate the
# kernel representation from the acceptance representation on the model side
ROWS=(
 "cond:true atoms=$COND"
 "cond:true table=data/ktab_cond_2525.npz:data/atab_cond_2525.json"
 "model multi atoms=data/kern_corr_mc_multi.npz:data/acc_corr_mc_multi.json"
 "model multi table=data/ktab_corr_mc_multi.npz:data/atab_corr_mc_multi.json"
 "model multi table, atom A=data/ktab_corr_mc_multi.npz:data/acc_corr_mc_multi.json"
 "model multi atoms, table A=data/kern_corr_mc_multi.npz:data/atab_corr_mc_multi.json"
 "model multi table, sample K=data/ktab_corr_smp_multi.npz:data/atab_corr_smp_multi.json"
 "model single table, sample K=data/ktab_corr_smp_single.npz:data/atab_corr_smp_single.json"
 "model single table, mc K=data/ktab_corr_mc_single.npz:data/atab_corr_mc_single.json"
 "model multi table, sample K + rho=data/ktab_corr_smp_multirs.npz:data/atab_corr_smp_multirs.json"
 "cond:true atoms, per-leg record=data/kern_cond_true.npz:data/acc_cond_true.json"
 "MC-cond banded, Bernstein A=data/kern_fid_band3.3e-4.npz:data/acc_d8.json"
 "per-leg product, mc D=data/kern_perleg_mc_1gev.npz:data/acc_perleg_mc_1gev.json"
)
OUT=data/fit_machinery_table.json
elif [ "$WHICH" = "conv" ]; then
# the table knobs of both sides
ROWS=(
 "cond table, atom bands=data/ktab_cond_2525.npz:data/atab_cond_2525.json"
 "cond table, atom bands + tail=data/ktab_cond_2525x.npz:data/atab_cond_2525x.json"
 "cond table, 4 GeV bands=data/ktab_cond_2525w4.npz:data/atab_cond_2525w4.json"
 "cond table, 2 GeV bands=data/ktab_cond_2525w2.npz:data/atab_cond_2525w2.json"
 "cond table, 1 GeV bands=data/ktab_cond_2525w1.npz:data/atab_cond_2525w1.json"
 "cond table, 2 GeV, 1000 cells=data/ktab_cond_2525w2c1000.npz:data/atab_cond_2525w2c1000.json"
 "cond table, 2 GeV, 4000 cells=data/ktab_cond_2525w2c4000.npz:data/atab_cond_2525w2c4000.json"
 "cond table, 2 GeV, half A=data/ktab_cond_2525w2a.npz:data/atab_cond_2525w2a.json"
 "cond table, 2 GeV, half B=data/ktab_cond_2525w2b.npz:data/atab_cond_2525w2b.json"
 "model table, 2 GeV nodes=data/ktab_corr_mc_multi_dm2.npz:data/atab_corr_mc_multi_dm2.json"
 "model table, 1 GeV nodes=data/ktab_corr_mc_multi.npz:data/atab_corr_mc_multi.json"
 "model table, 0.5 GeV nodes=data/ktab_corr_mc_multi_dm05.npz:data/atab_corr_mc_multi_dm05.json"
 "model table, 1000 cells=data/ktab_corr_mc_multi_c1000.npz:data/atab_corr_mc_multi_c1000.json"
 "model table, 4000 cells=data/ktab_corr_mc_multi_c4000.npz:data/atab_corr_mc_multi_c4000.json"
 "model table, sample K, 0.5 GeV=data/ktab_corr_smp_multi_dm05.npz:data/atab_corr_smp_multi_dm05.json"
)
OUT=data/fit_machinery_conv.json
elif [ "$WHICH" = "conv2" ]; then
# the node spacing pushed one step further, and the cell-count scan repeated on
# one half of the sample: a knob that moves the fit by more than the
# half-sample noise is a representation effect, one that does not is the noise
ROWS=(
 "cond table, 2 GeV=data/ktab_cond_2525w2.npz:data/atab_cond_2525w2.json"
 "cond table, 2 GeV, half A=data/ktab_cond_2525w2a.npz:data/atab_cond_2525w2a.json"
 "cond table, 2 GeV, half A, 1000 cells=data/ktab_cond_2525w2c1000a.npz:data/atab_cond_2525w2c1000a.json"
 "cond table, 2 GeV, half A, 4000 cells=data/ktab_cond_2525w2c4000a.npz:data/atab_cond_2525w2c4000a.json"
 "model mc K, 0.5 GeV nodes=data/ktab_corr_mc_multi_dm05.npz:data/atab_corr_mc_multi_dm05.json"
 "model mc K, 0.25 GeV nodes=data/ktab_corr_mc_multi_dm025.npz:data/atab_corr_mc_multi_dm025.json"
 "model sample K, 1 GeV nodes=data/ktab_corr_smp_multi.npz:data/atab_corr_smp_multi.json"
 "model sample K, 0.5 GeV nodes=data/ktab_corr_smp_multi_dm05.npz:data/atab_corr_smp_multi_dm05.json"
 "model sample K, 0.25 GeV nodes=data/ktab_corr_smp_multi_dm025.npz:data/atab_corr_smp_multi_dm025.json"
 "model sample K, single, 0.5 GeV=data/ktab_corr_smp_single_dm05.npz:data/atab_corr_smp_single_dm05.json"
 "model mc K, single, 0.5 GeV=data/ktab_corr_mc_single_dm05.npz:data/atab_corr_mc_single_dm05.json"
 "model sample K + rho, 1 GeV=data/ktab_corr_smp_multirs.npz:data/atab_corr_smp_multirs.json"
)
OUT=data/fit_machinery_conv2.json
elif [ "$WHICH" = "noise" ]; then
# the two tables' own statistical floors, and the cell count pushed to 8000
ROWS=(
 "cond table, 2 GeV=data/ktab_cond_2525w2.npz:data/atab_cond_2525w2.json"
 "cond table, 2 GeV, 8000 cells=data/ktab_cond_2525w2c8000.npz:data/atab_cond_2525w2c8000.json"
 "cond table, 2 GeV, half A, 8000 cells=data/ktab_cond_2525w2c8000a.npz:data/atab_cond_2525w2c8000a.json"
 "model sample K, 0.5 GeV=data/ktab_corr_smp_multi_dm05.npz:data/atab_corr_smp_multi_dm05.json"
 "model sample K, 0.5 GeV, half A=data/ktab_corr_smp_multi_a.npz:data/atab_corr_smp_multi_a.json"
 "model sample K, 0.5 GeV, half B=data/ktab_corr_smp_multi_b.npz:data/atab_corr_smp_multi_b.json"
)
OUT=data/fit_machinery_noise.json
elif [ "$WHICH" = "incl" ]; then
# the standalone-vs-sample kernel floor, inclusive (no selection, no A(m)):
# the same two kernels the model can be built on
ROWS=(
 "inclusive mc standalone table=data/ktab_mc_dm10.npz"
 "inclusive sample table=data/ktab_incl_smp.npz"
 "inclusive mc standalone atoms=data/kern_cfg_mc_sc3.3e-4.npz"
 "inclusive sample atoms=data/kern_incl_sc3.3e-4.npz"
)
OUT=data/fit_machinery_incl.json
SEL=""
else
echo "unknown suite $WHICH" >&2; exit 1
fi

SEL=${SEL-"--acc-pt 25 --acc-eta 2.4"}
LOG=data/00_fit_$(basename "$OUT" .json | sed 's/^fit_machinery_//').log
# detached on the remote node: a dropped ssh must not take the fit with it.
# Poll $LOG (and $OUT) to follow it.
ssh -n "$NODE" "cd $Z && rm -f $OUT && setsid nohup ./run_tf_z.sh \
  python3 -u -W ignore fsr_table.py fit \
  --gen data/genmerged_full.npz --nm 8192 --shape 5 $SEL \
  $(printf -- "--row '%s' " "${ROWS[@]}") -o $OUT \
  > $LOG 2>&1 < /dev/null & echo started \$!"
echo "[run_machinery_table] $WHICH on $NODE -> $LOG, $OUT"
