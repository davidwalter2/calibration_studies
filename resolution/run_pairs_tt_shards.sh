#!/bin/bash
# Sharded cf_mass_likelihood.py --pairs-tt (one process per task file) + merge.
# env: PROD OUT [NPAR=48] [KOK=0|1]
# The ionization block's mass-functional sign is no longer an option:
# it is -1 for every block and both charges (cf_mass_likelihood.IONI_SGN,
# fixed 2026-09-03), and the cache records it as `ioni_sign_fixed`.
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution; cd "$RES"
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$RES:${PYTHONPATH:-}"
# shellcheck source=prodfiles.sh
source "$RES/prodfiles.sh"
export CVH_IONI_KOKOULIN=${KOK:-0}     # offline Kokoulin term: >7x cost, ~1e-3 effect; OFF by decision 2026-09-03
PROD=${PROD:?} OUT=${OUT:?} NPAR=${NPAR:-48}
PARTS=$RES/runs/parts_$(basename "${OUT%.npz}"); mkdir -p "$PARTS"
# ONE PART PER TASK, not per file. A numberOfThreads=N task is N stream files
# with the SAME task directory, so a per-file loop named every part
# `part_task_NNNN.npz` N times over: the first won, the other N-1 hit the
# `[ -s "$out" ]` resume guard and were skipped -- 1/N of the candidates, with
# a race for the part file thrown in. `--files "$d"` (a directory) reads every
# stream of that one task.
run_part() { local d=$1; local tag; tag=$(basename "$d"); local out="$PARTS/part_$tag.npz"
  [ -s "$out" ] && { echo "[skip] $tag"; return 0; }
  if python3 cf_mass_likelihood.py --pairs-tt --files "$d" --ntasks 1 --pairs-cache "$out" > "$PARTS/part_$tag.log" 2>&1
  then echo "[done] $tag $(date +%H:%M)"; else echo "[FAIL] $tag"; rm -f "$out"; fi; }
export -f run_part; export PARTS
echo "=== $OUT: KOK=$CVH_IONI_KOKOULIN NPAR=$NPAR ($(date +%H:%M:%S)) ==="
pf_task_dirs "$PROD" | xargs -P "$NPAR" -I{} bash -c 'run_part {}'
echo "=== parts done ($(date +%H:%M:%S)); merging ==="
python3 merge_masspairs.py --out "$OUT" "$PARTS"/part_task_*.npz && echo "=== merged $OUT ($(date +%H:%M:%S)) ==="
