#!/bin/bash
# The K_S mass term with and without the nuclear-elastic family: the inclusive
# fit (closed-form and measured a_res), recoil on/off, and the radius and
# production-point bins -- same candidates, same card recipe, same minimiser.
#   ./run_nucel_fits.sh            (after ks_nucel_cf.py wrote runs/nucel/kspairs_nucel.npz)
set -euo pipefail
KS=/work/submit/david_w/ZMass/calibration_studies/resolution/ksclosure
N=$KS/nucel
export RUNS=$KS/runs/nucel
export THREADS=${THREADS:-16}
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
if [ "${SKIPVAR:-0}" != "1" ]; then
python3 $N/make_variants.py --cache $RUNS/kspairs_nucel.npz \
    --ares $KS/runs/fixed/kspairs_ares.npz --outdir $RUNS
for v in base nuc; do
  sub() { OMP_NUM_THREADS=1 python3 $KS/ks_subset.py --cache $RUNS/kspairs_${v}_all.npz \
            --out $RUNS/kspairs_${v}_$1.npz --cut "$2"; }
  sub fromb   'ks_fromb > 0'
  sub prompt  'ks_fromb == 0'
  sub r0_2    'ks_rdec < 2'
  sub r2_4    '(ks_rdec >= 2) & (ks_rdec < 4)'
  sub r4_10   '(ks_rdec >= 4) & (ks_rdec < 10)'
  sub r10_60  'ks_rdec >= 10'
  sub plo     'np.minimum(ks_pgenp, ks_pgenm) < 0.8'
  sub pmid    '(np.minimum(ks_pgenp, ks_pgenm) >= 0.8) & (np.minimum(ks_pgenp, ks_pgenm) < 1.5)'
  sub phi     'np.minimum(ks_pgenp, ks_pgenm) >= 1.5'
done
fi
# one fit per line: tag cache freeze extra-args; run NPAR at a time
F0="k_hit k_ms k_ioni k_rad"
F1="k_hit k_ms k_ioni k_rad k_nucel"
LIST=$RUNS/fitlist.txt
: > $LIST
for bin in all fromb prompt r0_2 r2_4 r4_10 r10_60 plo pmid phi; do
  echo "base_$bin|$RUNS/kspairs_base_$bin.npz|$F0|" >> $LIST
  echo "nuc_$bin|$RUNS/kspairs_nuc_$bin.npz|$F1|--add-nucel-family" >> $LIST
done
echo "nucnr_all|$RUNS/kspairs_nucnr_all.npz|$F1|--add-nucel-family" >> $LIST
echo "base_all_atruth|$RUNS/kspairs_base_all.npz|$F0|--a-res-key ares_truth" >> $LIST
echo "nuc_all_atruth|$RUNS/kspairs_nuc_all.npz|$F1|--a-res-key ares_truth --add-nucel-family" >> $LIST
echo "base_all_naive|$RUNS/kspairs_base_all.npz|$F0|--ares off --jensen off" >> $LIST
echo "nuc_all_naive|$RUNS/kspairs_nuc_all.npz|$F1|--ares off --jensen off --add-nucel-family" >> $LIST
export N
fitone() {
  IFS='|' read -r tag cache frz extra <<< "$1"
  PAIRS=$cache FREEZE="$frz" $N/build_ks_nucel.sh "$tag" $extra > $RUNS/build_$tag.log 2>&1 \
    && echo "done $tag" || echo "FAILED $tag"
}
export -f fitone
xargs -P ${NPAR:-6} -d '\n' -I{} bash -c 'fitone "{}"' < $LIST
python3 $KS/ks_table.py --runs $RUNS --results $RUNS/results
