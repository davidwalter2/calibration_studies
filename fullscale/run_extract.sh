#!/bin/bash
# Quadratic (hit-chi2) term extraction for both legs, field (50) + material (42).
#
#   --no-mass   MANDATORY on these productions: they run exportStepRecords=False
#               and so ship `radvgrid` alone, which the mass path rejects as a
#               `partial radiative export`. The mass side comes from
#               cf_inmaker.py pairs instead.
#   --max-dEref-p 0.01
#               the mean-loss quality requirement of NOTES sec. 7(a). `Mu*_dEref`
#               is in BOTH productions, so this is the quantity itself rather
#               than the daughter-pT proxy.
#   the sandwich J = sum_i G_i G_i^T is accumulated unconditionally.
set -euo pipefail
GF=/work/submit/david_w/ZMass/calibration_studies/resolution/globalfit
OUT=/work/submit/david_w/ZMass/calibration_studies/fullscale/runs
LOG=/work/submit/david_w/ZMass/calibration_studies/fullscale/logs
CEPH=/ceph/submit/data/user/d/david_w/ZMass/cvh
JOBS=${JOBS:-16}
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
cd "$GF"
for leg in "dyv2:$CEPH/dymc_8p5M_260906_v2" "jpsiv1:$CEPH/jpsimc_20M_260905"; do
  tag=${leg%%:*}; dir=${leg#*:}
  echo "=== $tag  $(date +%H:%M:%S) ==="
  python3 -u extract.py --files "$dir" --ntasks 0 --parmtypes 14 15 --no-mass \
      --max-chi2-ndof 3 --max-hess 1e8 --max-grad 1e6 --max-dEref-p 0.01 \
      -j "$JOBS" -o "$OUT/quad_$tag.npz" 2>&1 | tee "$LOG/quad_$tag.log"
done
echo "=== done $(date +%H:%M:%S) ==="
