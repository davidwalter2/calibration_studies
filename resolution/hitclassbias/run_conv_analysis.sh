#!/bin/bash
# The whole convergence-variant analysis in one pass, from the three finished
# productions to `out_conv.txt` and the figures. Run it THROUGH `rrun.sh`
# (submit82 cannot read /ceph and the mfs venv does not run there).
#
#   ./rrun.sh ./run_conv_analysis.sh
#
set -uo pipefail
OUT=out_conv.txt
: > "$OUT"
say() { echo "$@" | tee -a "$OUT"; }
run() { echo -e "\n\n############ $* ############" >> "$OUT"; "$@" >> "$OUT" 2>&1; }

for v in base tight damp; do
  say "=== extracting $v"
  python3 extract_conv.py --prod "resolution_trackres_mugun_ul16_260909_conv_$v" \
      --nproc 4 --out "data/conv_$v.npz" 2>&1 | tee -a "$OUT"
done
# the baseline cut to the same 40 tasks, for the bit check
[ -f data/conv_ref903x.npz ] || python3 extract_conv.py \
    --prod resolution_trackres_mugun_ul16_260903x_m0 --ntasks 40 --nproc 4 \
    --out data/conv_ref903x.npz 2>&1 | tee -a "$OUT"

run python3 c0_bitcheck.py --a data/conv_base.npz --b data/conv_ref903x.npz
run python3 c1_conv.py data/conv_base.npz data/conv_tight.npz data/conv_damp.npz
run python3 c2_pair.py --a data/conv_base.npz --b data/conv_tight.npz --na base --nb tight
run python3 c2_pair.py --a data/conv_base.npz --b data/conv_damp.npz  --na base --nb damp
run python3 c2_pair.py --a data/conv_tight.npz --b data/conv_damp.npz --na tight --nb damp
run python3 c4_secondorder.py data/conv_base.npz data/conv_damp.npz
run python3 c5_scaling.py data/conv_base.npz data/conv_damp.npz
run python3 c6_phi.py data/conv_base.npz data/conv_damp.npz
run python3 c7_eta.py data/conv_base.npz data/conv_damp.npz
echo "wrote $OUT"
