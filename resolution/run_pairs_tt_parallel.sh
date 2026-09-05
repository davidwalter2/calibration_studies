#!/bin/bash
# Parallel cf_mass_likelihood.py --pairs-tt over a TwoTrack production.
#
# --pairs-tt is a per-file loop with no cross-file state (the TwoTrack tree
# is already per candidate), but it runs serially at ~12 min per 2.7k-
# candidate task file -> ~10 h for a 48-task production, which dominates
# the whole chain.  One invocation per task file + merge_masspairs.py gives
# the same cache in the wall time of a single file.
#
# The script to run is pinned via SCRIPT= so every part is built with ONE
# version of cf_mass_likelihood.py (it can be edited by other work while
# the parts run); PYTHONPATH keeps the `cf_track_resolution` import working
# when the pinned copy lives outside the repo.
#
# usage: PROD=<globalcor dir> OUT=<cache path> [NPAR=48] [SCRIPT=<path>] \
#          ./run_pairs_tt_parallel.sh
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution; cd "$RES"
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$RES:${PYTHONPATH:-}"
PROD=${PROD:?set PROD to the production directory}
OUT=${OUT:?set OUT to the merged cache path}
NPAR=${NPAR:-48}
SCRIPT=${SCRIPT:-$RES/cf_mass_likelihood.py}
PARTS=${PARTS:-$RES/runs/parts_$(basename "${OUT%.npz}")}
mkdir -p "$PARTS"

run_part() {
  local f=$1
  local tag; tag=$(basename "$(dirname "$f")")
  local out="$PARTS/part_$tag.npz"
  [ -s "$out" ] && { echo "[skip] $tag"; return 0; }
  if python3 "$SCRIPT" --pairs-tt --files "$f" --ntasks 1 --pairs-cache "$out" \
       > "$PARTS/part_$tag.log" 2>&1; then
    echo "[done] $tag $(python3 -c "import numpy as np;print(len(np.load('$out')['z']))" 2>/dev/null)"
  else
    echo "[FAIL] $tag (see $PARTS/part_$tag.log)"; rm -f "$out"
  fi
}
export -f run_part
export PARTS SCRIPT

ls -d "$PROD"/task_*/globalcor_0.root 2>/dev/null | sort | xargs -P "$NPAR" -I{} bash -c 'run_part {}'
echo "=== parts done ($(date +%H:%M:%S)); merging ==="
python3 merge_masspairs.py --out "$OUT" "$PARTS/part_task_*.npz"
