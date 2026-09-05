#!/bin/bash
# ODD-moment closure of the CF track-resolution model (cf_skew_closure.py).
#
# The published closure statistic <e^{-u z^2}> is EVEN in z and cannot see the
# asymmetry that the mean-vs-mode question is about.  This runs the odd twin
# <z e^{-u z^2}> on every muon-gun cache that exists, and the mode/mean
# comparison alongside it.
#
# The August caches (`_fix`) carry no momentum variable; their `_sel` siblings
# are bit-identical extensions of the same tracks that DO carry genpt, so the
# momentum binning is taken from there (checked track-by-track in `load`).
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution
cd "$RES"
source "${VENV:-/work/submit/david_w/ZMass/mfs/.venv}/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=${OUT:-$HOME/public_html/cvh/$(date +%y%m%d)_skew}
mkdir -p "$OUT"
LOG=$RES/runs/skew$(date +%y%m%d); mkdir -p "$LOG"

python cf_skew_closure.py --validate --outpath "$OUT" > "$LOG/validate.log" 2>&1

run() {  # run <tag> <cache> <ptfrom|-> <label>
  local tag=$1 cache=$2 ptf=$3 lab=$4
  [ -f "$cache" ] || { echo "skip $tag (no $cache)"; return; }
  local extra=""
  [ "$ptf" != "-" ] && { [ -f "$ptf" ] || { echo "skip $tag (no $ptf)"; return; }; extra="--ptfrom $ptf"; }
  echo "=== $tag ($(date +%H:%M:%S))"
  python cf_skew_closure.py --cache "$cache" $extra --tag "$tag" \
      --label "$lab" --outpath "$OUT" "${@:5}" > "$LOG/$tag.log" 2>&1 \
    && echo "    [ok] $tag ($(date +%H:%M:%S))" || echo "    [FAIL] $tag"
}

# A: the August 8 model AND fit -- the caches the published closure used
run mugun_ul16_fix  runs/cf_trackres_mugun_ul16_fix.npz \
    runs/cf_trackres_mugun_ul16_sel.npz  "mu gun pT 20-60 [Aug-8]" &
run mugun_lowpt_fix runs/cf_trackres_mugun_lowpt_fix.npz \
    runs/cf_trackres_mugun_lowpt_sel.npz "mu gun pT 2-20 [Aug-8]" &
wait

# C/B: the 2026-08-30 campaign, if it has landed (these carry genpt themselves)
for t in mugun_ul16_260830 mugun_lowpt_260830 \
         mugun_ul16_260830_m0 mugun_lowpt_260830_m0 \
         mugun_ul16_260830_m0_sub; do
  case $t in *_m0*) arm="legacy truncated Q";; *) arm="CGF Fisher weight";; esac
  case $t in *lowpt*) pt="pT 2-20";; *) pt="pT 20-60";; esac
  run "$t" "runs/cf_trackres_$t.npz" - "mu gun $pt [260830, $arm]"
done
echo "outputs in $OUT"
