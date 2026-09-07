#!/bin/bash
# Cheap global integrity scan of the 410 production inputs for the
# NUL-in-ParameterSets corruption (see scan_pset_nulls.py).
set -uo pipefail
FL=${1:-/work/submit/david_w/ZMass/calibration_studies/production/filelist_jpsimc_20M_260905.txt}
OUT=${2:-/ceph/submit/data/user/d/david_w/ZMass/scratch_repack_260906/scan_psets_410.txt}
SC=/work/submit/david_w/ZMass/calibration_studies/production/repack_fix_260907/scan_pset_nulls.py
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
awk 'NF{print $1}' "$FL" | xargs -P 12 -n 4 python3 "$SC" > "$OUT" 2>&1
echo "SCAN DONE $(date -Is)"
echo "files: $(wc -l < "$OUT")"
echo "not OK:"; grep -v '^OK' "$OUT" || echo "  (none)"
