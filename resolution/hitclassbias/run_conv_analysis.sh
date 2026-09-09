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

# NTASKS pins the sample. The three productions were launched at 40 tasks and
# a SECOND, larger pass (NT=160) was started on the same output directories
# while this analysis was being written, so an unpinned extraction would keep
# growing the sample under the reader and no two runs of this script would
# compare the same tracks. 40 is what the numbers in NOTES/STATE quote; raise
# it deliberately when the 160-task pass has finished.
NTASKS=${NTASKS:-40}
for v in base tight damp; do
  say "=== extracting $v (first $NTASKS complete tasks)"
  python3 extract_conv.py --prod "resolution_trackres_mugun_ul16_260909_conv_$v" \
      --ntasks "$NTASKS" --nproc 2 --out "data/conv_$v.npz" 2>&1 | tee -a "$OUT"
  # a corrupt or half-flushed cache reads as EOFError deep inside the analysis;
  # fail HERE instead, where the message says which file
  python3 -c "import numpy,sys; d=numpy.load('data/conv_$v.npz'); \
      print('  cache OK:', len(d['z']), 'tracks')" 2>&1 | tee -a "$OUT" || exit 1
done
# the baseline cut to the same 40 tasks, for the bit check
[ -f data/conv_ref903x.npz ] || python3 extract_conv.py \
    --prod resolution_trackres_mugun_ul16_260903x_m0 --ntasks "$NTASKS" --nproc 2 \
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
run python3 c8_phiquad.py data/conv_base.npz
run python3 c3_figs.py
echo "wrote $OUT"
