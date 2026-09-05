#!/bin/bash
# MS-off control: extract the mugun_lowpt_{noms,nomsrad}_cf refits (legacy-Q,
# same config as mugun_lowpt_260830_m0) with the Kokoulin-off convention and run
# the per-charge closure. Discriminates a second-order MS estimator bias from an
# energy-loss/field/geometry origin of the flat charge-even momentum-high term.
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution; cd "$RES"
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
export CVH_IONI_KOKOULIN=0
CEPH=/ceph/submit/data/user/d/david_w/ZMass/cvh; LOG=$RES/runs/muclosure260830
for t in noms nomsrad; do
  ( echo "=== extract mugun_lowpt_${t}_k0 ($(date +%H:%M:%S)) ==="
    ./extract_parallel.sh "$CEPH/resolution_trackres_mugun_lowpt_${t}_cf" "runs/cf_trackres_mugun_lowpt_${t}_k0.npz" 80 > "$LOG/ext_mugun_lowpt_${t}_k0.log" 2>&1 \
      && echo "    [ok] mugun_lowpt_${t}_k0 ($(date +%H:%M:%S))" || echo "    [FAIL] mugun_lowpt_${t}_k0" ) &
done; wait
./run_skew_k0.sh mugun_lowpt_noms_k0 mugun_lowpt_nomsrad_k0
echo "=== ALL DONE ($(date +%H:%M:%S)) ==="
