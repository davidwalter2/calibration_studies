#!/bin/bash
# The four A/B/C ladder solves in PARALLEL (one process per sample): serially
# they are ~30 min each and the four are completely independent.
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution; cd "$RES"
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CVH_IONI_KOKOULIN=0
OUT=${OUT:-$RES/runs/rad260903x}; mkdir -p "$OUT"
for s in lowpt_B lowpt_C ul16_B ul16_C; do
  python3 kms_solve_260903x.py --sample "$s" > "$OUT/kms_$s.txt" 2>&1 &
done
wait
cat "$OUT"/kms_{lowpt_B,lowpt_C,ul16_B,ul16_C}.txt > "$OUT/kms_solve_260903x.txt"
echo "=== kms ladder done ($(date +%H:%M:%S)) -> $OUT/kms_solve_260903x.txt"
