#!/bin/bash
# The figures and the summary of the machinery benchmark in the table
# representation.  Needs the wums/mplhep environment (`run_tf_z.sh`).
set -u
Z=/work/submit/david_w/ZMass/calibration_studies/zchannel
NODE=${NODE:-submit52}
OUT=${OUT:-/home/submit/david_w/public_html/ZMass/cvh/260917_fsr_machinery}
ssh "$NODE" "cd $Z && ./run_tf_z.sh python3 -u -W ignore cmp_machinery.py \
  --outpath $OUT \
  --table 'MC, own selection-conditional=data/ktab_cond_2525w2.npz:black:-' \
  --table 'model, multi, sample K=data/ktab_corr_smp_multi_dm05.npz:C1:-' \
  --table 'model, multi, standalone K=data/ktab_corr_mc_multi_dm05.npz:C0:--' \
  --table 'model, single, sample K=data/ktab_corr_smp_single_dm05.npz:C2::' \
  --summary 'table=data/fit_machinery_table.json' \
  --summary 'conv=data/fit_machinery_conv.json' \
  --summary 'conv2=data/fit_machinery_conv2.json' \
  --summary 'noise=data/fit_machinery_noise.json' \
  --summary 'incl=data/fit_machinery_incl.json' \
  --fit 'tables=data/fit_machinery_table.json' \
  --fit 'convergence=data/fit_machinery_conv2.json' \
  --record data/genmerged_full.npz"
