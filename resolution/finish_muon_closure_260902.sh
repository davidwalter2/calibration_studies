#!/bin/bash
# Finish the 2026-08-30 muon closure (second resume, 2026-09-02).
#
# WHAT HAPPENED THIS TIME. The 2026-08-30 17:38 crash of every running
# low-pT CGF task (tasks 120-145, segfault/bus error at the same second) was
# NOT the thread limit: pluginAnalysisHitAnalyzerAuto.so was relinked at
# 17:38:25 (MuonProcessDisabler.cc recompiled; the four maker objects are
# from 11:11, before the campaign) and a shared library replaced under a
# running cmsRun kills it. CMSSW's crash handler then attached gdb, which
# hung for 2.8 days in ptrace_stop and blocked finish_muon_closure_260830.sh.
# The maker binary is unchanged, so the 26 reruns are consistent with the
# other 134 tasks.
#
# RULE: never `scram b` in CMSSW_15_0_19_patch2_dev while a production is
# running from it (the "shared build lock" only protects builds from each
# other, not jobs from builds).
#
# Stages: (1) refit tasks 120-145 of mugun_lowpt_260830 in the background;
# (2) extract the three complete samples serially (BLAS pinned to 1 thread
# in extract_parallel, so overlapping with the refits is safe now);
# (3) wait for the refits, extract mugun_lowpt_260830; (4) kms_solve.
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution
CEPH=/ceph/submit/data/user/d/david_w/ZMass/cvh
LOG=${LOG:-$RES/runs/muclosure260830}
mkdir -p "$LOG"
cd "$RES"
VENV=${VENV:-/work/submit/david_w/ZMass/mfs/.venv}
# shellcheck disable=SC1091
source "$VENV/bin/activate" || { echo "FATAL: no venv at $VENV"; exit 1; }
NSHARD=${NSHARD:-160}

guard() {
  local used lim
  used=$(ps -u "$USER" -L --no-headers 2>/dev/null | wc -l); lim=$(ulimit -u)
  echo "    [guard] threads $used / $lim"
  [ "$used" -gt $(( lim * 6 / 10 )) ] && { echo "    [guard] ABORT"; exit 1; }
}
ndone() { ls -d "$CEPH/resolution_trackres_$1"/task_*/.complete 2>/dev/null | wc -l; }

export EXTRA="trackSrc=generalTracks useDefaultField=True useIdealGeometry=True globalTag=150X_mcRun2_asymptotic_v1"
export STAGGER=3

echo "=== refit mugun_lowpt_260830 tasks 120-145 ($(date +%H:%M:%S)) === done so far: $(ndone mugun_lowpt_260830)/160"
guard
( FILELIST=$RES/simprod/filelist_mugun_lowpt.txt OUTTAG=mugun_lowpt_260830 \
  ./run_local_trackres.sh 26 120 145 >> "$LOG/refit_mugun_lowpt_260830.log" 2>&1 ) &
REFIT_PID=$!

extract() {
  local t=$1
  echo "=== extract $t ($(date +%H:%M:%S)) === tasks complete: $(ndone "$t")/160"
  guard
  if ./extract_parallel.sh "$CEPH/resolution_trackres_$t" "runs/cf_trackres_${t}.npz" "$NSHARD" \
       > "$LOG/ext_$t.log" 2>&1; then echo "    [ok] $t ($(date +%H:%M:%S))"
  else echo "    [FAIL] $t (see $LOG/ext_$t.log)"; fi
}

for t in mugun_ul16_260830 mugun_ul16_260830_m0 mugun_lowpt_260830_m0; do
  [ -f "runs/cf_trackres_${t}.npz" ] && { echo "=== skip extract $t (npz exists)"; continue; }
  extract "$t"
done

echo "=== waiting for refits ($(date +%H:%M:%S)) ==="
wait $REFIT_PID
echo "    refits finished: $(ndone mugun_lowpt_260830)/160 complete ($(date +%H:%M:%S))"
if [ "$(ndone mugun_lowpt_260830)" -ne 160 ]; then
  echo "    WARNING: incomplete sample; extracting only the complete tasks"
  for d in "$CEPH"/resolution_trackres_mugun_lowpt_260830/task_*; do
    [ -f "$d/.complete" ] || { echo "    dropping partial $d"; rm -f "$d/globalcor_resclosure_0.root"; }
  done
fi
extract mugun_lowpt_260830

echo "=== kms_solve ($(date +%H:%M:%S)) ==="
python3 kms_solve_260830.py > "$LOG/kms_solve_260902.txt" 2>&1 && cat "$LOG/kms_solve_260902.txt"
echo "=== ALL DONE ($(date +%H:%M:%S)) ==="
