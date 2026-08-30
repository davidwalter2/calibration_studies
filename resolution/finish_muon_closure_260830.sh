#!/bin/bash
# Finish the muon closure after the 2026-08-30 thread-limit incident.
#
# WHAT HAPPENED, so it is not repeated. The mode-0 arm is 13.7x cheaper than
# the CGF arm, so it finished its refits first and started extracting while the
# CGF refits were still running. extract_parallel launches NSHARD python3
# processes, and numpy's BLAS defaults to one thread per core: 160 shards x 2
# samples x ~68 threads put this user at 32 373 threads against `ulimit -u`
# 32 768. Past that nothing can create a thread, and EVERY cmsRun started
# afterwards died with an immediate segmentation violation before its first log
# line -- a trivial EmptySource job too, on a machine with 1.2 TB free. It cost
# 40 CGF tasks in one sample and 69 in the next, and it looked exactly like a
# physics crash in our own code.
#
# Two fixes: extract_parallel now pins the BLAS/OpenMP thread count to 1, and
# THIS script runs the stages strictly one at a time. Refits and extractions
# must not overlap.
#
# Resume is safe: run_local_trackres skips on the `.complete` sentinel, so the
# 120 surviving CGF tasks are not redone.
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution
CEPH=/ceph/submit/data/user/d/david_w/ZMass/cvh
LOG=${LOG:-$RES/runs/muclosure260830}
mkdir -p "$LOG"
cd "$RES"

VENV=${VENV:-/work/submit/david_w/ZMass/mfs/.venv}
# shellcheck disable=SC1091
source "$VENV/bin/activate" || { echo "FATAL: no venv at $VENV"; exit 1; }

guard() {   # refuse to start a stage if the thread budget is already tight
  local used lim
  used=$(ps -u "$USER" -L --no-headers 2>/dev/null | wc -l)
  lim=$(ulimit -u)
  echo "    [guard] threads $used / $lim"
  if [ "$used" -gt $(( lim * 6 / 10 )) ]; then
    echo "    [guard] ABORT: over 60 % of the per-user thread limit before the stage even starts"
    exit 1
  fi
}

export EXTRA="trackSrc=generalTracks useDefaultField=True useIdealGeometry=True globalTag=150X_mcRun2_asymptotic_v1"
export STAGGER=3

refit() {   # refit <simprod-tag> <out-tag>
  local src=$1 tag=$2
  echo "=== refit $tag ($(date +%H:%M:%S)) ==="
  guard
  export FILELIST=$RES/simprod/filelist_$src.txt
  export OUTTAG=$tag
  for blk in 0 40 80 120; do
    local end=$((blk+39)); [ $end -gt 159 ] && end=159
    ./run_local_trackres.sh 40 $blk $end >> "$LOG/refit_$tag.log" 2>&1
  done
  unset FILELIST OUTTAG
  echo "    done: $(ls -d $CEPH/resolution_trackres_$tag/task_*/.complete 2>/dev/null | wc -l)/160 tasks"
}

extract() {   # extract <tag>, ONE sample at a time
  local t=$1
  echo "=== extract $t ($(date +%H:%M:%S)) ==="
  guard
  if ./extract_parallel.sh "$CEPH/resolution_trackres_$t" "runs/cf_trackres_${t}.npz" 160 \
       > "$LOG/ext_$t.log" 2>&1; then
    echo "    [ok] $t"
  else
    echo "    [FAIL] $t (see $LOG/ext_$t.log)"
  fi
}

# CGF arm: 120/160 survived on the high-pT sample, 0/160 on the low-pT one.
refit mugun_ul16  mugun_ul16_260830
refit mugun_lowpt mugun_lowpt_260830

# Then, and only then, the four extractions -- serially.
extract mugun_ul16_260830
extract mugun_lowpt_260830
extract mugun_ul16_260830_m0
extract mugun_lowpt_260830_m0

echo "=== k_ms ($(date +%H:%M:%S)) ==="
python3 kms_solve_260830.py | tee "$LOG/kms.txt"
echo "=== finished $(date +%H:%M:%S) ==="
