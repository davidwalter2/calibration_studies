#!/bin/bash
# Fit every card of the ladder. EDM-certified: rabbit's own termination.
HERE=/work/submit/david_w/ZMass/calibration_studies/resolution/hitlik
cd $HERE
M=${M:-tf-trust-krylov}
for c in "$@"; do
  echo "=== fit_$c $(date +%H:%M:%S)"
  ./run_fit.sh $c --minimizerMethod $M > logs/fit_$c.log 2>&1 || echo "FIT $c FAILED"
  grep -iE "edm|converged|Minimization|elapsed" logs/fit_$c.log | tail -4
done
echo "ALL DONE $(date +%H:%M:%S)"
