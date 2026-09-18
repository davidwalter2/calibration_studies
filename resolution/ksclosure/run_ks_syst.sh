#!/bin/bash
# Systematic variations of the K_S closure, all on the SAME cache unless the
# variation is in the truth match itself.
#
#   ./run_ks_syst.sh [prod dir]
set -euo pipefail
PROD=${1:-/ceph/submit/data/user/d/david_w/ZMass/cvh/ks_btojpsix_260917_ideal}
KS=/work/submit/david_w/ZMass/calibration_studies/resolution/ksclosure
RUNS=${RUNS:-$KS/runs}
export RUNS
export THREADS=${THREADS:-8}

run() { tag=$1; shift; PAIRS=$1; shift
  echo "=== $tag ==="
  PAIRS=$PAIRS $KS/build_ks.sh "$tag" "$@" > $RUNS/build_$tag.log 2>&1 \
    || echo "FAILED $tag"; }

# 1. the unmodelled tail: cut the residual at ~3 sigma (the likelihood
#    normalises over the window it cuts to, so this costs no bias of its own)
run s_resid3 $RUNS/kspairs_all.npz --max-resid 0.020
run s_resid5 $RUNS/kspairs_all.npz --max-resid 0.035
# 2. absorb the flat plateau with a floating uniform background
run s_bkg    $RUNS/kspairs_all.npz --background uniform --fbkg 0.005 --float-bkg
# 3. the fit-quality cut
run s_chi2   $RUNS/kspairs_all.npz --max-chi2-ndof 1.5
# 4. the truth-match window: a tight and a loose cache
( source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
  OMP_NUM_THREADS=1 python3 -u $KS/ks_pairs.py --prod "$PROD" \
      --max-dlam 0.05 --max-dphi 0.05 --max-dp 0.20 --max-dvtx 1.0 \
      --cache $RUNS/kspairs_matchtight.npz > $RUNS/pairs_matchtight.log 2>&1
  OMP_NUM_THREADS=1 python3 -u $KS/ks_pairs.py --prod "$PROD" \
      --max-dlam 0.60 --max-dphi 0.60 --max-dp 0.90 --max-dvtx 5.0 \
      --cache $RUNS/kspairs_matchloose.npz > $RUNS/pairs_matchloose.log 2>&1 )
run s_mtight $RUNS/kspairs_matchtight.npz
run s_mloose $RUNS/kspairs_matchloose.npz
