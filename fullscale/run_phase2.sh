#!/bin/bash
# PHASE 2 -- joint J/psi + Z, on J/psi **v2**.
#
# v2, not v1: it finishes at the same time as the v1 slurm arrays and it is the
# stronger leg. It carries `Jpsi_covrefmom` (so the Jensen s^2 is truth-free
# per candidate instead of carrying an MC-measured f_ang and its +-5 %), the
# two-track variance/log-det gradient on parmtype 15 (a factor 41 on the
# material Fisher information on the gun), `Mu*_maxfracloss`, and the per-group
# CF exponents phase 3 needs. v1 is kept only as a cross-check of the
# mean-loss-only quadratic term against v2's variance-block version.
#
# usage: ./run_phase2.sh [pairs|quad|card|fit]
set -euo pipefail
FS=/work/submit/david_w/ZMass/calibration_studies/fullscale
RES=/work/submit/david_w/ZMass/calibration_studies/resolution
Z=/work/submit/david_w/ZMass/calibration_studies/zchannel
CEPH=/ceph/submit/data/user/d/david_w/ZMass/cvh
JV2=$CEPH/jpsimc_20M_260906_v2
# EXCLUSIONS: NONE.  `./run_phase2.sh xlist` writes the file list from whatever
# carries a `.complete` sentinel; as of 2026-09-07 15:30 that is all 1645 tasks
# (6580 files) of a production that finished at 21 750 740 events.
#
# HISTORY, so nobody re-derives it.  Three separate defects were excluded and
# then repaired:
#   1219-1222, 1334-1337, 1409-1412  the "re-staged" grid copies -- 0.82
#       candidates/event against 0.997 and chi2/ndof median 3.5e6, i.e.
#       worthless, so the exclusion was necessary and not conservative;
#   1552-1555                        input exited rc=91 after 27 s, no output;
#   1313                             silently EMPTY -- four 15.5 kB streams WITH
#       a `.complete` sentinel, because its input lacked a StreamerInfo, so the
#       sentinel check could not catch it and it had to be named.
# All were re-produced from properly repacked inputs and validated at 0.9967
# candidates/event, 0.0073 % failures, chi2/ndof median 0.953 against 0.954 for
# the control.  Repairing 1313 also recovered its input's missing tail as three
# NEW tasks, 1642-1644.
JV2LIST=$FS/runs/jpsiv2_tasks_ok.txt
JV2_BAD=""
DYV2=$CEPH/dymc_8p5M_260906_v2
GRP=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/data/materialGroups50.txt
STAGES=${*:-pairs quad}

# `prodfiles` skips any task without its `.complete` sentinel, so an extraction
# started before the last tasks land is SAFE -- it simply uses fewer of them.
# The quadratic term is a sum over its own candidates and the mass term a
# product over its own; they need not be the same set.  What is not safe is
# quoting a number as "the full production" when it is not, so the count is
# logged.
report_complete () {
  local n; n=$(ls "$JV2"/task_*/.complete 2>/dev/null | wc -l)
  echo "jpsimc_20M_260906_v2: $n/1642 tasks complete" >&2
}

for st in $STAGES; do
case $st in
xlist)
  python3 - "$JV2" "$JV2LIST" $JV2_BAD <<'PY'
import glob, os, sys
d, out, bad = sys.argv[1], sys.argv[2], {int(x) for x in sys.argv[3:]}
fs, n = [], 0
for t in sorted(glob.glob(os.path.join(d, "task_*"))):
    i = int(os.path.basename(t).split("_")[1])
    if i in bad or not os.path.exists(os.path.join(t, ".complete")):
        continue
    g = sorted(glob.glob(os.path.join(t, "globalcor_*.root")))
    if g:
        n += 1
        fs += g
open(out, "w").write("\n".join(fs) + "\n")
print(f"{n} tasks, {len(fs)} files -> {out}")
PY
  ;;
pairs)
  report_complete
  source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
  # NTASKS=600 by default: ~7.9 M candidates and a 12 GB cache.  The J/psi leg
  # is NOT statistics-limited here -- 299 k gun candidates already give
  # sigma(alpha) = 0.017e-3 = 1.5 MeV at the Z, and 7.9 M give 0.003e-3 =
  # 0.3 MeV, an order below sigma(m_Z) = 2 MeV -- while the cache, the card and
  # the fit all scale linearly with it.
  NT=${NTASKS:-600}
  # tasks 0-599 contain none of the 16 excluded indices, so --ntasks 600 on the
  # directory is already clean; use "@$JV2LIST" if NTASKS is ever raised past 1219
  python3 -u "$RES/cf_inmaker.py" pairs --files "$JV2" --ntasks "$NT" \
      --cache "$FS/runs/jpairs_v2_n${NT}.npz" --jac-parmtypes 14 15 \
      2>&1 | tee "$FS/logs/jpairs_v2_n${NT}.log"
  ;;
quad)
  report_complete
  source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
  python3 -u "$RES/globalfit/extract.py" --files "@$JV2LIST" --ntasks 0 \
      --parmtypes 14 15 --no-mass \
      --max-chi2-ndof 3 --max-hess 1e8 --max-grad 1e6 --max-dEref-p 0.01 \
      -j 16 -o "$FS/runs/quad_jpsiv2_ok.npz" 2>&1 \
      | tee "$FS/logs/quad_jpsiv2_ok.log"
  ;;
card)
  THREADS=${THREADS:-32} "$FS/run_tf.sh" python3 -u "$FS/make_joint_card.py" \
      --jpsi-pairs "$FS/runs/jpairs_v2_n${NTASKS:-600}.npz" \
      --jpsi-maxn  "${JPSI_MAXN:-3000000}" \
      --z-pairs    "$FS/runs/zpairs_dyv2_jac_full.npz" \
      --quad "$FS/runs/quad_jpsiv2_ok.npz" "$FS/runs/quad_dyv2.npz" \
      --groups "$GRP" --whiten --shape 5 \
      --fsr "$Z/data/kern_loose_band3.3e-4.npz" --acc "$Z/data/acc_loose_d8.json" \
      -o "$FS/cards/joint_v2.hdf5" 2>&1 | tee "$FS/logs/card_joint_v2.log"
  ;;
fit)
  THREADS=${THREADS:-48} "$FS/run_tf.sh" python3 -u "$FS/fit.py" \
      --card "$FS/cards/joint_v2.hdf5" --fix k_hit k_ms k_ioni k_rad \
      --chunk 32768 --label joint_v2 -o "$FS/results/fit_joint_v2.json" \
      2>&1 | tee "$FS/logs/fit_joint_v2.log"
  ;;
esac
done
