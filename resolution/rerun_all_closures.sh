#!/bin/bash
# Re-run every track-level closure with the CORRECTED multiple-scattering model.
#
# WHY EVERYTHING MUST BE RE-EXTRACTED. cf_track_resolution --extract stores the
# per-track log-CF exponent Sms, which is computed through
# cf_ms_exact.moliere_params. Four corrections landed 2026-08-07/08:
#   * Geant4's screening radius factor (1+exp(-Z^2/1000))     ~4.5%
#   * per-element chi_c^2 / chi_a^2 (msmoliv stride 8 -> 10)   0.3%
#   * nuclear form factor |F| -> |F|^2                         0.07%
#   * within-step lateral displacement (sub-step quadrature)   8.8% of the
#     position variance -- cf_propagation_test only, but the model is shared
# so every cached npz predates the model it is meant to test.
#
# WHAT IS RUN
#   1. refit of the low-pT muon control (pT 2-20, same range as the hadrons) --
#      the only sample with no refit yet, and the one that breaks the
#      species/momentum confound at FIT level;
#   2. re-extraction of all five samples;
#   3. k_ms for each.
#
# Staged to respect memory (~2 GB per extraction shard, ~2.5 GB per cmsRun) on
# a 768-core / 1.4 TB machine.
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution
CEPH=/ceph/submit/data/user/d/david_w/ZMass/cvh
LOG=${LOG:-/tmp/claude-125124/-work-submit-david-w-ZMass/40621a07-b09b-46d9-b7b3-4189252bcf3e/scratchpad}
cd "$RES"

echo "=== [1/3] refit the low-pT muon control ($(date +%H:%M:%S)) ==="
ls $CEPH/resolution_simprod_mugun_lowpt/task_*/step2.root | sort > simprod/filelist_mugun_lowpt.txt
# 60-way with a STAGGERED RAMP. Launching 160 at once wedged every job in
# futex_do_wait inside the NSS/sssd user lookup: 40 min elapsed, 1 s CPU each,
# nothing open but the log and /var/lib/sss/mc/passwd. NSS was responsive again
# the moment they were killed, so it is a launch-RATE problem, not capacity.
#
# The variables MUST be exported. A `VAR=x cmd` prefix applies only to that one
# command, so with the loop below they would be set in this shell and never
# reach run_local_trackres.sh -- which silently falls back to its DEFAULT
# filelist and OUTTAG, i.e. refits the wrong sample. (Cost me one relaunch.)
export FILELIST=$RES/simprod/filelist_mugun_lowpt.txt
export OUTTAG=mugun_lowpt
export EXTRA="trackSrc=generalTracks useDefaultField=True useIdealGeometry=True globalTag=150X_mcRun2_asymptotic_v1"
# 40-way with STAGGER=3 (validated: 24-way staggered ran at ~98% CPU/job,
# whereas 60 simultaneous wedged at 0 s CPU). Completed tasks are skipped via
# the .complete sentinel, so this resumes rather than redoing work.
export STAGGER=3
for blk in 0 40 80 120; do
  end=$((blk+39)); [ $end -gt 159 ] && end=159
  ./run_local_trackres.sh 40 $blk $end >> $LOG/refit_lowpt.log 2>&1
done
unset STAGGER
unset FILELIST OUTTAG EXTRA
echo "    done: $(grep -c '^\[done\]' $LOG/refit_lowpt.log) tasks, $(grep -c FAIL $LOG/refit_lowpt.log) fail"

echo "=== [2/3] re-extract all samples with the corrected model ($(date +%H:%M:%S)) ==="
# two samples at a time: 2 x 160 shards x ~2 GB stays well inside memory
run_pair() {
  for t in "$@"; do
    ( ./extract_parallel.sh "$CEPH/resolution_trackres_$t" "runs/cf_trackres_${t}_fix.npz" 160 \
        > "$LOG/ext_${t}_fix.log" 2>&1 && echo "    [ok] $t" || echo "    [FAIL] $t" ) &
  done
  wait
}
run_pair mugun_ul16 kaon_ul16
run_pair pion_ul16 proton_ul16
run_pair mugun_lowpt

echo "=== [3/3] k_ms for every sample ($(date +%H:%M:%S)) ==="
python3 - <<'PY'
import numpy as np, importlib.util, sys, os
spec=importlib.util.spec_from_file_location("cft","cf_track_resolution.py")
m=importlib.util.module_from_spec(spec); sys.modules["cft"]=m; spec.loader.exec_module(m)
class A: khit=0.; kms=0.; kioni=0.
print(f"{'sample':16} {'ntracks':>8}  " + "  ".join(f"u={u:<5g}" for u in (0.05,0.2,1.0,2.0)) + "    mean")
for tag in ("mugun_ul16","mugun_lowpt","kaon_ul16","pion_ul16","proton_ul16"):
    f=f"runs/cf_trackres_{tag}_fix.npz"
    if not os.path.exists(f):
        print(f"{tag:16} (missing)"); continue
    d=np.load(f); z=d["z"]; n=len(z); ks=[]
    for u in (0.05,0.2,1.0,2.0):
        Ed=np.exp(-u*z**2).mean(); lo,hi=-0.60,0.60
        for _ in range(16):
            mid=0.5*(lo+hi); a=A(); a.kms=mid
            if Ed-m.weier(m.model_phi(d,a),u).mean()>0: hi=mid
            else: lo=mid
        ks.append(0.5*(lo+hi))
    print(f"{tag:16} {n:8d}  " + "  ".join(f"{k:+.4f}" for k in ks) + f"   {np.mean(ks):+.4f}")
PY
echo "=== all closures finished $(date +%H:%M:%S) ==="
