#!/bin/bash
# The whole K_S closure on the FIXED-code production, beside the old one.
#
#   ./run_ks_fixed.sh
#
# Writes every card, log, cache and fit under `runs/fixed/`, so the 260917
# results in `runs/` are untouched and `ks_compare_closure.py` can put the two
# side by side.
set -euo pipefail
KS=/work/submit/david_w/ZMass/calibration_studies/resolution/ksclosure
PROD=${PROD:-/ceph/submit/data/user/d/david_w/ZMass/cvh/ks_btojpsix_260918_fixed}
export RUNS=$KS/runs/fixed
mkdir -p "$RUNS/results"
export THREADS=${THREADS:-16}
export FIGTAG=ksclosure_fixed
VENV=/work/submit/david_w/ZMass/mfs/.venv/bin/activate

echo "############ 1. the ladder, the populations and the bins ############"
$KS/run_ks_closure.sh "$PROD"

echo "############ 2. the systematics ############"
$KS/run_ks_syst.sh "$PROD"

echo "############ 3. the sigma-artefact slope from truth ############"
( source $VENV
  OMP_NUM_THREADS=1 python3 -u $KS/ks_cov_extract.py --prod "$PROD" \
      --out $RUNS/kscov_all.npz 2>&1 | tail -5
  OMP_NUM_THREADS=1 python3 -u $KS/ares_angles.py --cov $RUNS/kscov_all.npz \
      --pairs $RUNS/kspairs_all.npz --masses pion \
      --write $RUNS/kspairs_ares.npz --tag ares_angles_fixed ) \
  2>&1 | tee $RUNS/ares_angles.log
CACHE=$RUNS/kspairs_ares.npz FORMS="mom truth" $KS/run_ares_forms.sh

echo "############ 4. the tables and the figures ############"
( source $VENV
  python3 $KS/ks_table.py --runs $RUNS --results $RUNS/results \
      --order all_naive all_ares all af_mom af_truth fromb fromb0 prompt \
              r0_2 r2_4 r4_10 r10_60 plo pmid phi \
              s_resid3 s_resid5 s_bkg s_chi2 s_mtight s_mloose
  echo
  python3 $KS/ks_compare_closure.py --old $KS/runs --new $RUNS
  echo
  OMP_NUM_THREADS=1 python3 $KS/ks_fix_plots.py \
      --old $KS/runs/kspairs_all.npz --new $RUNS/kspairs_all.npz \
      --old-runs $KS/runs --new-runs $RUNS --tag ksclosure_fixed )
