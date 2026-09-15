#!/bin/bash
# The machinery-only benchmark: the MC's own per-leg radiator (the numerical
# convolution square root of the standalone Photos kernel), the per-leg kernels
# it feeds, and the MC's own conditional kernel read in the model's mass and
# selection variables.  Everything here is numpy only, ~4 min in total.
set -eu
cd /work/submit/david_w/ZMass/calibration_studies/zchannel
P=python3
PL=data/perleg_full.npz
RUN=data/photos/gen_mcMix.npz

# --- the effective leg of the tabulated kernel, at the kernel's own resolution
# and at half it, as the grid control
$P -u fsr_perleg.py legsqrt --run $RUN -o data/legD_mc.npz
$P -u fsr_perleg.py legsqrt --run $RUN --du 4e-5 -o data/legD_mc_du4.npz

# --- the selection-conditional kernels it feeds
$P -u fsr_perleg.py kernel --htable data/ht_pt25_1.0gev.npz \
   -o data/kern_perleg_mc_1gev.npz --acceptance data/acc_perleg_mc_1gev.json \
   --mc-leg data/legD_mc.npz
$P -u fsr_perleg.py kernel --htable data/ht_pt25_1.0gev.npz \
   -o data/kern_perleg_mc_du4.npz --acceptance data/acc_perleg_mc_du4.json \
   --mc-leg data/legD_mc_du4.npz
$P -u fsr_perleg.py kernel --htable data/ht_pt25_2.0gev.npz \
   -o data/kern_perleg_mc_2gev.npz --acceptance data/acc_perleg_mc_2gev.json \
   --mc-leg data/legD_mc.npz

# --- the MC's own conditional kernel in the model's variables
for M in true collmass collsel coll; do
  $P -u fsr_perleg.py condker --gen $PL --mode $M \
     -o data/kern_cond_${M}.npz --acceptance data/acc_cond_${M}.json
done

# --- the banded MC-conditional reference, from the fit's own sample
$P -u fit_gen.py kernel --gen data/genmerged_full.npz --acc-pt 25 --acc-eta 2.4 \
   --sigma-cap 3.3e-4 --bands 70 80 85 88 91 94 98 105 115 130 \
   -o data/kern_fid_band3.3e-4.npz
