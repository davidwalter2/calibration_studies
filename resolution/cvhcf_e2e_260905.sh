#!/bin/bash
# END-TO-END gate for the in-maker resolution-CF exponents (2026-09-05).
#
# Question: does the unbinned mass likelihood give the SAME answer when its
# per-candidate exponents come from the maker instead of from the offline
# extractor?  Everything else -- the same refit output file, the same FSR
# kernel, the same fit configuration -- is held fixed, so the only thing that
# can move alpha is the evaluator.
#
# The offline cache is DECIMATED onto the maker's 64-point grid before the
# comparison (`cf_inmaker.py decimate`).  That is not a convenience: the
# maker's grid is a SUBSET of the offline one, so decimating removes the grid
# change from the comparison and leaves the evaluator alone.  The grid change
# itself is a separate, already-measured effect (cfcompress/gridtest.py,
# -1.6e-8 in alpha) and the FULL-GRID fit is run too, so both are on the table.
#
# usage: ./cvhcf_e2e_260905.sh [stage ...]     stages: refit pairs fit compare
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution; cd "$RES"
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CVH_IONI_KOKOULIN=0
export PYTHONPATH="$RES:${PYTHONPATH:-}"

TAG=cvhcf_e2e_260905
OUT=${OUT:-/tmp/claude-125124/-work-submit-david-w-ZMass/9cb79a9f-18ea-4214-84d2-2de84e8651c2/scratchpad/inmaker_cf/e2e}
LOG=$RES/runs/cvhcf260905; mkdir -p "$LOG" "$OUT"
RUNTF=/work/submit/david_w/ZMass/calibration_studies/env_tf/run_tf.sh
export CMSSW_AREA=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev
RUN_ONE=/work/submit/david_w/ZMass/calibration_studies/slurm/run_one.sh
INIT=/work/submit/david_w/ZMass/mfs/data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt
IN=/ceph/submit/data/user/d/david_w/ZMass/cvh/resolution_simprod_jpsigun_ul16/task_0000/step2.root
NEV=${NEV:-2000}
STAGES=${*:-"refit pairs fit compare"}

step() { echo "=== $* ($(date +%H:%M:%S)) ==="; }

for st in $STAGES; do
case $st in

refit) step "refit: $NEV events, two-track, exportStepRecords=True (both routes need it)"
  d=$OUT/tt; mkdir -p "$d"; rm -f "$d/globalcor_0.root" "$d/.complete"
  $RUN_ONE /work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/test/runCvhJpsiGenMC.py \
      "$IN" "$d" nEvents=$NEV numberOfThreads=1 doRes=True fillGrads=True fitFromGenParms=False \
      scalarPot3DInitFile=$INIT trackSrc=generalTracks useLegacyPairLoop=True doTrigger=False \
      applyHltFilter=False useIdealGeometry=True useDefaultField=True \
      globalTag=150X_mcRun2_asymptotic_v1 CgfQoPMode=0 \
      > "$d/local.log" 2>&1 && touch "$d/.complete"
  echo "  rc=$?  $(ls -la $d/globalcor_0.root 2>/dev/null | awk '{print $5}') bytes" ;;

pairs) step "pairs: offline extractor, in-maker reader, and the decimated offline cache"
  F=$OUT/tt/globalcor_0.root
  python3 cf_masskernel_tt.py --files "$F" --ntasks 1 \
      --kernel-cache "$LOG/kernel_$TAG.npz" --postfix "_$TAG" > "$LOG/kernel.log" 2>&1
  echo "  kernel rc=$?"
  python3 cf_mass_likelihood.py --pairs-tt --files "$F" --ntasks 1 \
      --pairs-cache "$LOG/pairs_py448_$TAG.npz" > "$LOG/pairs_py.log" 2>&1
  echo "  python 448 rc=$?"
  python3 cf_inmaker.py decimate --cache "$LOG/pairs_py448_$TAG.npz" \
      --out "$LOG/pairs_py64_$TAG.npz" > "$LOG/decimate.log" 2>&1
  echo "  decimate rc=$?"
  python3 cf_inmaker.py pairs --files "$F" --ntasks 1 \
      --cache "$LOG/pairs_cf64_$TAG.npz" > "$LOG/pairs_cf.log" 2>&1
  echo "  in-maker 64 rc=$?" ;;

fit) step "fit: the same likelihood on each cache"
  K=$LOG/kernel_$TAG.npz
  for arm in py448 py64 cf64; do
    P=$LOG/pairs_${arm}_$TAG.npz
    [ -s "$P" ] || { echo "  skip $arm (no $P)"; continue; }
    for model in "r" "families"; do
      nm=${TAG}_${arm}_${model}
      $RUNTF python3 cf_masslik_fit.py --pairs-cache "$P" --kernel-cache "$K" \
          --model $model $( [ "$model" = families ] && echo --krad 1 ) \
          --tag "$nm" --no-plots --out "$LOG/fit_${nm}.npz" > "$LOG/fit_${nm}.log" 2>&1
      echo "  $nm rc=$?"
    done
  done ;;

compare) step "compare"
  python3 cf_inmaker.py compare --cache "$LOG/pairs_py64_$TAG.npz" \
      --other "$LOG/pairs_cf64_$TAG.npz" 2>&1 | tee "$LOG/compare_caches.txt"
  python3 - <<'EOF' 2>&1 | tee "$LOG/compare_fits.txt"
import glob, json, os
import numpy as np
LOG = os.path.join(os.environ.get("RESDIR", "/work/submit/david_w/ZMass/calibration_studies/resolution"),
                   "runs/cvhcf260905")
rows = {}
for fn in sorted(glob.glob(f"{LOG}/fit_cvhcf_e2e_260905_*.npz")):
    d = np.load(fn, allow_pickle=True)
    key = os.path.basename(fn).replace("fit_cvhcf_e2e_260905_", "").replace(".npz", "")
    meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
    nm = meta.get("parnames", [])
    rows[key] = (np.asarray(d["fit_x"]), np.asarray(d["fit_err"]),
                 float(d["fit_nll"]), [str(x) for x in nm])
for k, (x, e, nll, nm) in rows.items():
    print(f"{k:<20} NLL={nll:.6f}  " + "  ".join(
        f"{n}={v:+.6f}+-{s:.6f}" for n, v, s in zip(nm or list(range(len(x))), x, e)))
for model in ("r", "families"):
    a, b = f"py64_{model}", f"cf64_{model}"
    if a in rows and b in rows:
        xa, ea = rows[a][0], rows[a][1]
        xb = rows[b][0]
        print(f"\n{model}: in-maker minus offline (same 64-pt grid)")
        for n, va, vb, s in zip(rows[a][3] or range(len(xa)), xa, xb, ea):
            print(f"   {str(n):<10} {vb - va:+.3e}   ({(vb - va) / s:+.4f} sigma)")
    a2 = f"py448_{model}"
    if a2 in rows and b in rows:
        xa, ea = rows[a2][0], rows[a2][1]
        xb = rows[b][0]
        print(f"{model}: in-maker(64) minus offline(448)  [evaluator + grid]")
        for n, va, vb, s in zip(rows[a2][3] or range(len(xa)), xa, xb, ea):
            print(f"   {str(n):<10} {vb - va:+.3e}   ({(vb - va) / s:+.4f} sigma)")
EOF
  ;;
esac
done
echo "=== DONE ($(date +%H:%M:%S)) ==="
