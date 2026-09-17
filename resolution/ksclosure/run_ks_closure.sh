#!/bin/bash
# The whole K_S closure, from the production directory to the table.
#
#   ./run_ks_closure.sh [prod dir]
#
# Re-runnable: the cache is rebuilt from whatever (task, truth) chunk pairs are
# complete, and every fit is redone on it.
set -euo pipefail
PROD=${1:-/ceph/submit/data/user/d/david_w/ZMass/cvh/ks_btojpsix_260917_ideal}
KS=/work/submit/david_w/ZMass/calibration_studies/resolution/ksclosure
RUNS=$KS/runs
mkdir -p "$RUNS" "$RUNS/results"
export THREADS=${THREADS:-16}

if [ "${SKIPCACHE:-0}" != "1" ]; then
  echo "=== pairs cache ==="
  ( source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
    OMP_NUM_THREADS=1 python3 -u $KS/ks_pairs.py --prod "$PROD" \
        --cache $RUNS/kspairs_all.npz ) 2>&1 | tee $RUNS/pairs.log
fi

echo "=== subsets ==="
( source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
  sub() { OMP_NUM_THREADS=1 python3 -u $KS/ks_subset.py --cache $RUNS/kspairs_all.npz \
            --out $RUNS/kspairs_$1.npz --cut "$2"; }
  sub fromb   'ks_fromb > 0'
  sub fromb0  'ks_fromb == 511'
  sub prompt  'ks_fromb == 0'
  sub r0_2    'ks_rdec < 2'
  sub r2_4    '(ks_rdec >= 2) & (ks_rdec < 4)'
  sub r4_10   '(ks_rdec >= 4) & (ks_rdec < 10)'
  sub r10_60  'ks_rdec >= 10'
  sub plo     'np.minimum(ks_pgenp, ks_pgenm) < 0.8'
  sub pmid    '(np.minimum(ks_pgenp, ks_pgenm) >= 0.8) & (np.minimum(ks_pgenp, ks_pgenm) < 1.5)'
  sub phi     'np.minimum(ks_pgenp, ks_pgenm) >= 1.5'
) 2>&1 | tee $RUNS/subsets.log

# THE CORRECTION LADDER on the inclusive sample, exactly as the J/psi closure
# reports it: naive -> + the sigma-artefact (self-consistent resolution) ->
# + the exact Jensen second-order term.  Each step is a separate card, so the
# three are additive on the same candidates.
echo "=== fit all_naive ==="
PAIRS=$RUNS/kspairs_all.npz $KS/build_ks.sh all_naive --ares off --jensen off \
    > $RUNS/build_all_naive.log 2>&1 || echo "FAILED all_naive"
echo "=== fit all_ares ==="
PAIRS=$RUNS/kspairs_all.npz $KS/build_ks.sh all_ares --jensen off \
    > $RUNS/build_all_ares.log 2>&1 || echo "FAILED all_ares"

for tag in all fromb fromb0 prompt r0_2 r2_4 r4_10 r10_60 plo pmid phi; do
  [ -s "$RUNS/kspairs_$tag.npz" ] || continue
  echo "=== fit $tag ==="
  PAIRS=$RUNS/kspairs_$tag.npz $KS/build_ks.sh "$tag" \
      > $RUNS/build_$tag.log 2>&1 || echo "FAILED $tag"
done

echo "=== table ==="
( source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
  python3 $KS/ks_table.py --order all_naive all_ares all fromb fromb0 prompt r0_2 r2_4 r4_10 r10_60 plo pmid phi )

echo "=== figures ==="
( source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
  OMP_NUM_THREADS=1 python3 $KS/ks_plots.py --cache $RUNS/kspairs_all.npz --tag ksclosure )
