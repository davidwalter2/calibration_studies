#!/bin/bash
# Launch all three convergence variants CONCURRENTLY at 10 parallel each
# (30 cmsRun processes, under the 32 cap). 40 tasks x 2000 events each.
# Detach with:  setsid nohup ./run_conv_all.sh > LOG 2>&1 < /dev/null &
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution
LOG=$RES/hitclassbias/logs
mkdir -p "$LOG"
NT=${NT:-40}
NP=${NP:-10}
for v in base tight damp; do
  "$RES/hitclassbias/run_conv.sh" "$v" "$NT" "$NP" > "$LOG/conv_$v.log" 2>&1 &
done
wait
echo "ALL VARIANTS DONE $(date)"
