#!/bin/bash
# Waits for run_all_260904f.sh to report ALL DONE, then runs the whole
# downstream analysis chain. Nothing here starts while a refit is running
# (the ALL DONE marker is written after the last block returns), which is the
# standing rule -- extract_parallel.sh + a live refit has put this user over
# `ulimit -u` before and killed cmsRun with what looks like a physics crash.
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution; cd "$RES"
PLOG=$RES/runs/prod260904f_driver.log
LOG=$RES/runs/clampfix260904; mkdir -p "$LOG"

echo "=== waiting for the refits ($(date +%H:%M:%S)) ==="
until grep -q "ALL DONE" "$PLOG" 2>/dev/null; do sleep 60; done
echo "=== refits done ($(date +%H:%M:%S)) ==="
sleep 30
./chain_260904f.sh 1 2 3 4 >> "$LOG/chain_1234.log" 2>&1
echo "=== stages 1-4 done ($(date +%H:%M:%S)) rc=$? ==="
./chain_260904f.sh 5 >> "$LOG/chain_5.log" 2>&1
echo "=== stage 5 done ($(date +%H:%M:%S)) rc=$? ==="
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CVH_IONI_KOKOULIN=0
python3 clampfix_summary_260904.py > "$LOG/summary_tables.txt" 2>&1
python3 clampfix_figs_260904.py > "$LOG/figs.log" 2>&1
echo "=== DRIVER ALL DONE ($(date +%H:%M:%S)) ==="
