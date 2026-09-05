#!/bin/bash
cd /work/submit/david_w/ZMass/calibration_studies/resolution
for tag in mugun_ul16_260830_raw_k0 mugun_lowpt_260830_m0_k0 mugun_lowpt_260830_raw_k0; do
  until grep -q "\[ok\] $tag" runs/muclosure260903_finish.log 2>/dev/null; do sleep 120; done
  ./run_skew_k0.sh $tag
done
echo "=== skew chain done ($(date +%H:%M:%S)) ==="
