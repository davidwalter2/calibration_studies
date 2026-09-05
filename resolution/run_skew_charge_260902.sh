#!/bin/bash
# The CHARGE-SIGN arm of the odd-moment closure.
#
# `ioniurbanv` column 10 is `us.cs = E/p^3`, POSITIVE for every track, and maps
# dE -> d(q/p) for q = +1 only.  The in-fit CGF block multiplies it by the
# charge sign; `cf_track_resolution.ioni_step_exponent` does not.  So the
# cached model carries the mu+ skew for BOTH charges, while the data -- a
# 50/50 mu+/mu- gun -- cancels it.  This runs the closure four ways per sample:
#
#   nosign  both     the published arm (the model as it stands)
#   sign    both     with Im S_ioni -> q Im S_ioni
#   sign    q=+1     the charge where the cached sign is already right
#   sign    q=-1     the charge where it was wrong
#
# plus the two unsigned per-charge arms, which are what make the omission
# visible: the DATA must show equal and opposite skew in the two charges while
# the unsigned MODEL shows the same skew in both.
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution
cd "$RES"
source "${VENV:-/work/submit/david_w/ZMass/mfs/.venv}/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=${OUT:-$HOME/public_html/cvh/$(date +%y%m%d)_skew}
mkdir -p "$OUT"
LOG=$RES/runs/skew$(date +%y%m%d); mkdir -p "$LOG"

run() {  # run <tag> <cache> <ptfrom> <label> [extra...]
  local tag=$1 cache=$2 ptf=$3 lab=$4; shift 4
  [ -f "$cache" ] || { echo "skip $tag (no $cache)"; return; }
  echo "=== $tag ($(date +%H:%M:%S))"
  python cf_skew_closure.py --cache "$cache" --ptfrom "$ptf" --tag "$tag" \
      --label "$lab" --outpath "$OUT" "$@" > "$LOG/$tag.log" 2>&1 \
    && echo "    [ok] $tag ($(date +%H:%M:%S))" || echo "    [FAIL] $tag"
}

for S in ul16:"pT 20-60" lowpt:"pT 2-20"; do
  s=${S%%:*}; ptlab=${S#*:}
  C=runs/cf_trackres_mugun_${s}_fix.npz
  P=runs/cf_trackres_mugun_${s}_sel.npz
  run "mugun_${s}_qsign"     "$C" "$P" "mu gun $ptlab [q-signed]"        &
  run "mugun_${s}_qsign_pos" "$C" "$P" "mu gun $ptlab [q-signed, mu+]"   --charge 1 &
  run "mugun_${s}_qsign_neg" "$C" "$P" "mu gun $ptlab [q-signed, mu-]"   --charge -1 &
  run "mugun_${s}_nosign_pos" "$C" "$P" "mu gun $ptlab [unsigned, mu+]"  --charge 1 &
  run "mugun_${s}_nosign_neg" "$C" "$P" "mu gun $ptlab [unsigned, mu-]"  --charge -1 &
done
wait
echo "outputs in $OUT"
