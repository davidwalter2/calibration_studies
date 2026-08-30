#!/bin/bash
# Re-run the MUON track-level closure (the right panel of slide 4 of
# 260811_trackres_masslik_david.pdf) with the current CF construction.
#
# WHY EVERYTHING IS STALE. The published muon numbers (k_ms +0.0129 at
# pT 20-60, +0.0100 at pT 2-20) were produced from refits of 2026-08-07/08 and
# npz caches of 2026-08-08. FIVE things have changed since, on BOTH sides of
# the comparison:
#
#   MODEL   1ca2915 (08-16)  exact delta-ray spectrum + Kokoulin correction
#           7f80706 (08-18)  MS and ionization defaults follow the C++
#           e05c02f (08-18)  the closure's a-vector basis fix
#   FIT     9a7c692 (08-24)  the q/p process-noise weight is now the block's
#                            Fisher information (CgfQoPMode=1 default), so the
#                            fitted tracks AND the exported influence weights
#                            w_b both differ
#           0b7670f (08-29)  legacy Q restored as an option (default unchanged)
#
# The nuclear-elastic channel (e5eda56 etc., 08-17) also landed but is a
# hadron effect and does not touch these two samples.
#
# So this is the same argument as the header of rerun_all_closures.sh, one
# model generation later -- except that this time the FIT moved too, which is
# why stage 1 (refit) cannot be skipped.
#
# NEW OUTPUT TAGS, DELIBERATELY. run_local_trackres.sh resumes on a `.complete`
# sentinel, so reusing OUTTAG=mugun_ul16 would silently keep the August refits
# and produce a "new" result that is entirely old. The August samples are also
# the only copy of the published numbers' input -- do not overwrite them.
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution
CEPH=/ceph/submit/data/user/d/david_w/ZMass/cvh
LOG=${LOG:-$RES/runs/muclosure260830}
mkdir -p "$LOG"
cd "$RES"

VENV=${VENV:-/work/submit/david_w/ZMass/mfs/.venv}
# shellcheck disable=SC1091
source "$VENV/bin/activate" || { echo "FATAL: no venv at $VENV"; exit 1; }
python3 -c "import numpy, matplotlib, uproot" \
  || { echo "FATAL: venv missing numpy/matplotlib/uproot"; exit 1; }

# The refit arguments are copied verbatim from the August runs' local.log so
# the ONLY difference between old and new is the code.
export EXTRA="trackSrc=generalTracks useDefaultField=True useIdealGeometry=True globalTag=150X_mcRun2_asymptotic_v1"
export STAGGER=3

refit() {   # refit <simprod-tag> <out-tag>
  local src=$1 tag=$2
  echo "=== refit $tag ($(date +%H:%M:%S)) ==="
  ls $CEPH/resolution_simprod_$src/task_*/step2.root | sort > simprod/filelist_$src.txt
  export FILELIST=$RES/simprod/filelist_$src.txt
  export OUTTAG=$tag
  # 40-way in blocks: 60+ simultaneous starts wedge in the NSS/sssd lookup.
  for blk in 0 40 80 120; do
    local end=$((blk+39)); [ $end -gt 159 ] && end=159
    ./run_local_trackres.sh 40 $blk $end >> "$LOG/refit_$tag.log" 2>&1
  done
  unset FILELIST OUTTAG
  echo "    done: $(ls -d $CEPH/resolution_trackres_$tag/task_*/.complete 2>/dev/null | wc -l)/160 tasks"
}

refit mugun_ul16  mugun_ul16_260830
refit mugun_lowpt mugun_lowpt_260830

echo "=== extract ($(date +%H:%M:%S)) ==="
for t in mugun_ul16_260830 mugun_lowpt_260830; do
  ( ./extract_parallel.sh "$CEPH/resolution_trackres_$t" "runs/cf_trackres_${t}.npz" 160 \
      > "$LOG/ext_$t.log" 2>&1 && echo "    [ok] $t" || echo "    [FAIL] $t" ) &
done
wait

echo "=== k_ms ($(date +%H:%M:%S)) ==="
python3 kms_solve_260830.py | tee "$LOG/kms.txt"
echo "=== finished $(date +%H:%M:%S) ==="
