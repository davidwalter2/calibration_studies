#!/bin/bash
# Produce the six layered-toy configurations (pT = 3, 40 x NSUB = 1, 4, 16).
#
# SAFE TO RUN IN PARALLEL: every configuration owns a private area under
# $CMSSW_BASE/toyscan_pt/<tag> (geometry + planes + drivers), so the shared
# gen_toy_config.py outputs in src/Analysis/HitAnalyzer are never written.
# See the module docstring of toy_pt_scan.py.
set -u
cd "$(dirname "$0")"
NEV=${NEV:-400000}
TAGS=${TAGS:-"pt3_K1 pt3_K4 pt3_K16 pt40_K1 pt40_K4 pt40_K16"}
LOGD=/tmp/claude-125124/-work-submit-david-w-ZMass/40621a07-b09b-46d9-b7b3-4189252bcf3e/scratchpad/tp
mkdir -p "$LOGD"

one() {
  local t=$1
  echo "[$t] start $(date +%T)"
  python -u toy_pt_scan.py setup "$t" --fresh   >> "$LOGD/$t.drv" 2>&1 || { echo "[$t] SETUP FAILED"; return 1; }
  python -u toy_pt_scan.py model "$t"           >> "$LOGD/$t.drv" 2>&1 || { echo "[$t] MODEL FAILED"; return 1; }
  python -u toy_pt_scan.py sim   "$t" --events "$NEV" >> "$LOGD/$t.drv" 2>&1 || { echo "[$t] SIM FAILED"; return 1; }
  echo "[$t] done $(date +%T)"
}

for t in $TAGS; do one "$t" & done
wait
echo "ALL DONE $(date +%T)"
