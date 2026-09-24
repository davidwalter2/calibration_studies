#!/bin/bash
# The first-block-only diagnostic family in the same bins as run_nucel_fits.sh.
set -euo pipefail
KS=/work/submit/david_w/ZMass/calibration_studies/resolution/ksclosure
export N=$KS/nucel RUNS=$KS/runs/nucel THREADS=${THREADS:-16}
fitone() {
  PAIRS=$RUNS/kspairs_nucfirst_$1.npz FREEZE="k_hit k_ms k_ioni k_rad k_nucel" \
    $N/build_ks_nucel.sh nucfirst_$1 --add-nucel-family > $RUNS/build_nucfirst_$1.log 2>&1 \
    && echo "done nucfirst_$1" || echo "FAILED nucfirst_$1"
}
export -f fitone
printf "%s\n" fromb prompt r0_2 r2_4 r4_10 r10_60 plo pmid phi | xargs -P 9 -I{} bash -c 'fitone {}'
