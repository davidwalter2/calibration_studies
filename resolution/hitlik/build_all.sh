#!/bin/bash
# Build every card of the ladder, then fit them, logging each step.
set -e
HERE=/work/submit/david_w/ZMass/calibration_studies/resolution/hitlik
cd $HERE
for c in cf gaussq gauss cf_c0 joint inj_cf inj_gaussq inj_joint; do
  echo "=== card_$c $(date +%H:%M:%S)"
  ./run_ladder.sh card_$c > logs/card_$c.log 2>&1 || echo "CARD $c FAILED"
  tail -3 logs/card_$c.log
done
