#!/bin/bash
# The K_S closure refitted with the sigma-artefact slope taken from each of the
# candidate forms of `ares_angles.py`, plus the slope measured from MC truth.
# Everything else -- selection, window, Jensen term, minimiser -- is the
# nominal `build_ks.sh` configuration, so the spread between the runs is the
# `a_res` model uncertainty and nothing else.
set -euo pipefail
KS=/work/submit/david_w/ZMass/calibration_studies/resolution/ksclosure
RUNS=${RUNS:-$KS/runs}
export RUNS
CACHE=${CACHE:-$RUNS/kspairs_ares.npz}
FORMS=${FORMS:-"mom ang full_ms full_ms_nolam full_pop kin truth"}
for f in $FORMS; do
  echo "=== a_res = $f ==="
  PAIRS=$CACHE $KS/build_ks.sh "af_$f" --a-res-key "ares_$f" \
      > $RUNS/build_af_$f.log 2>&1 || echo "FAILED af_$f"
  grep -E "^  a_res|alpha" $RUNS/card_af_$f.log 2>/dev/null | head -3 || true
done
