#!/bin/bash
# Parallel --pairs-tt build with --ioni-sign neg (the corrected ionization
# skew sign), one part per task file + merge_masspairs.py.
#
# WHY A SEPARATE SCRIPT rather than an EXTRA= hook in run_pairs_tt_parallel.sh:
# that script is running RIGHT NOW for another chain and re-reading it mid-run
# is how a production silently changes underneath itself.  Same structure, own
# parts directory, own pinned interpreter copy.
#
# usage: PROD=<globalcor dir> OUT=<cache path> [NPAR=8] ./run_pairs_tt_fixsign_260902.sh
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution; cd "$RES"
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$RES:${PYTHONPATH:-}"
PROD=${PROD:?set PROD}
OUT=${OUT:?set OUT}
NPAR=${NPAR:-8}
SCRIPT=${SCRIPT:-/tmp/claude-125124/-work-submit-david-w-ZMass/44cdba6b-f250-445b-baa6-7a2aaa730d3a/scratchpad/cf_mass_likelihood_pinned_fixsign.py}
PARTS=${PARTS:-$RES/runs/parts_$(basename "${OUT%.npz}")}
mkdir -p "$PARTS"

run_part() {
  local f=$1
  local tag; tag=$(basename "$(dirname "$f")")
  local out="$PARTS/part_$tag.npz"
  [ -s "$out" ] && { echo "[skip] $tag"; return 0; }
  if python3 "$SCRIPT" --pairs-tt --ioni-sign neg --files "$f" --ntasks 1 \
       --pairs-cache "$out" > "$PARTS/part_$tag.log" 2>&1; then
    echo "[done] $tag"
  else
    echo "[FAIL] $tag (see $PARTS/part_$tag.log)"; rm -f "$out"
  fi
}
export -f run_part
export PARTS SCRIPT

ls -d "$PROD"/task_*/globalcor_0.root 2>/dev/null | sort \
  | xargs -P "$NPAR" -I{} bash -c 'run_part {}'
echo "=== parts done ($(date +%H:%M:%S)); merging ==="
python3 merge_masspairs.py --out "$OUT" "$PARTS/part_task_*.npz"
