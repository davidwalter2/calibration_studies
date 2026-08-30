#!/bin/bash
# The LEGACY-Q arm of the muon track-level closure: identical to
# rerun_muon_closure_260830.sh except `CgfQoPMode=0`.
#
# WHAT THIS ISOLATES, AND WHAT IT DOES NOT. Three configurations exist:
#
#   A  August 7-8 refits (published slide 4)   old model, old fit
#   B  this script                             new model, current reference,
#                                              LEGACY truncated-Q weight
#   C  rerun_muon_closure_260830.sh            new model, current reference,
#                                              CGF Fisher weight
#
# B vs C isolates the CGF weight EXACTLY -- one switch, everything else equal.
# A vs B does NOT isolate the model, because the fit's ENERGY-LOSS REFERENCE
# also moved between 08-08 and now (c269543 exact delta + Kokoulin variance +
# charge-aware meanLoss, e232c20 corrections default-ON, 2845c92 dEdxlast from
# a controlled-length step, f4d0e64 hadron radiative loss). A vs B is
# "everything except the weight", not "the model".
#
# COST: the legacy Q is 13.7x cheaper per fitted track than the CGF
# (NOTES_CGFFIT s89), so this arm is nearly free next to the one it compares
# against -- which is the reason to run it rather than reason about it.
#
# 24-way, not 40: the CGF arm is already running 40-way and 60+ simultaneous
# starts wedge in the NSS/sssd lookup. 24-way staggered was measured at ~98%
# CPU/job.
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution
CEPH=/ceph/submit/data/user/d/david_w/ZMass/cvh
LOG=${LOG:-$RES/runs/muclosure260830}
mkdir -p "$LOG"
cd "$RES"

VENV=${VENV:-/work/submit/david_w/ZMass/mfs/.venv}
# shellcheck disable=SC1091
source "$VENV/bin/activate" || { echo "FATAL: no venv at $VENV"; exit 1; }

# Same as the CGF arm plus the one switch. IoniTruncationAlpha is left at its
# 0.999 default on purpose: that is the value the published numbers were
# produced with, and it is live again only under CgfQoPMode=0.
export EXTRA="trackSrc=generalTracks useDefaultField=True useIdealGeometry=True globalTag=150X_mcRun2_asymptotic_v1 CgfQoPMode=0"
export STAGGER=3

refit() {
  local src=$1 tag=$2
  echo "=== refit $tag (mode 0) ($(date +%H:%M:%S)) ==="
  export FILELIST=$RES/simprod/filelist_$src.txt
  export OUTTAG=$tag
  for blk in 0 24 48 72 96 120 144; do
    local end=$((blk+23)); [ $end -gt 159 ] && end=159
    ./run_local_trackres.sh 24 $blk $end >> "$LOG/refit_$tag.log" 2>&1
  done
  unset FILELIST OUTTAG
  echo "    done: $(ls -d $CEPH/resolution_trackres_$tag/task_*/.complete 2>/dev/null | wc -l)/160 tasks"
}

refit mugun_ul16  mugun_ul16_260830_m0
refit mugun_lowpt mugun_lowpt_260830_m0

echo "=== extract ($(date +%H:%M:%S)) ==="
for t in mugun_ul16_260830_m0 mugun_lowpt_260830_m0; do
  ( ./extract_parallel.sh "$CEPH/resolution_trackres_$t" "runs/cf_trackres_${t}.npz" 160 \
      > "$LOG/ext_$t.log" 2>&1 && echo "    [ok] $t" || echo "    [FAIL] $t" ) &
done
wait
echo "=== mode-0 arm finished $(date +%H:%M:%S) ==="
