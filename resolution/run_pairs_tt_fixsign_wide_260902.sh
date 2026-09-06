#!/bin/bash
# Sign-fixed (--ioni-sign neg) per-file pairs-tt shards for the 2026-09-02 J/psi
# gun ditrack, files task_0014..0159 (0000-0013 were already in flight from
# run_pairs_tt_fixsign_260902.sh at 14-way, whose driver was detached so the
# two cannot race on a part file). Per file ~3-4 h single core, so wide.
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution; cd "$RES"
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$RES:${PYTHONPATH:-}"
# shellcheck source=prodfiles.sh
source "$RES/prodfiles.sh"
PROD=/ceph/submit/data/user/d/david_w/ZMass/cvh/resolution_trackres_jpsigun_ul16_260902_m0
PARTS=$RES/runs/parts_cf_masspairs_jpsigun_ul16_260902_m0_fixsign
SCRIPT=/tmp/claude-125124/-work-submit-david-w-ZMass/44cdba6b-f250-445b-baa6-7a2aaa730d3a/scratchpad/cf_mass_likelihood_pinned_fixsign.py
# one part per TASK (all of its streams), not per stream file
run_part() { local d=$1; local tag; tag=$(basename "$d"); local out="$PARTS/part_$tag.npz"
  [ -s "$out" ] && { echo "[skip] $tag"; return 0; }
  if python3 "$SCRIPT" --pairs-tt --ioni-sign neg --files "$d" --ntasks 1 --pairs-cache "$out" > "$PARTS/part_$tag.log" 2>&1; then echo "[done] $tag $(date +%H:%M)"; else echo "[FAIL] $tag"; rm -f "$out"; fi; }
export -f run_part; export PARTS SCRIPT
pf_task_dirs "$PROD" | awk 'NR>14' | xargs -P 146 -I{} bash -c 'run_part {}'
echo "=== wide shards done ($(date +%H:%M:%S)) ==="
