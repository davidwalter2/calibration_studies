#!/bin/bash
# The DATA-side per-hit chain, end to end.  Every step is idempotent and named,
# so a resumed session runs only what is missing:
#
#   ./run_all.sh extract      tree -> runs/perhit/perhit.npz
#   ./run_all.sh quad         the quadratic hit-chi2 term over the same tracks
#   ./run_all.sh gates        gates 1-5 from the npz + the trees
#   ./run_all.sh cards        the arm and injection ladder
#   ./run_all.sh fits         rabbit_fit.py on all of them (EDM certified)
#   ./run_all.sh fisher       H and J per arm (the sandwich inputs)
#   ./run_all.sh efficiency   the sandwich table
#   ./run_all.sh xcum         the cross-cumulant tables (the WG question)
#   ./run_all.sh cost         time + export bill
#   ./run_all.sh plots        figures into ~/public_html/cvh/<YYMMDD>_perhit
#   ./run_all.sh subfits      8 disjoint subsample fits per arm
set -uo pipefail
HERE=/work/submit/david_w/ZMass/calibration_studies/resolution/hitlik/perhit
HL=/work/submit/david_w/ZMass/calibration_studies/resolution/hitlik
export R=${R:-/work/submit/david_w/ZMass/calibration_studies/resolution/runs/perhit}
export NPZ=${NPZ:-$R/perhit.npz}
export QNPZ=${QNPZ:-$R/perhit_quad.npz}
export COMPS=${COMPS:-hit}
export NTRK=${NTRK:-0}
export PROD=${PROD:-resolution_trackres_mugun_ul16_260910_perhit}
GRP=/work/submit/david_w/ZMass/CMSSW_15_0_19_patch2_dev/src/Analysis/HitAnalyzer/data/materialGroups50.txt
CEPH=/ceph/submit/data/user/d/david_w/ZMass/cvh
mkdir -p $R/cards $R/fits $R/figs $HERE/logs
step=${1:-help}; shift || true

case $step in
  extract)
    source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    cd $HERE
    exec python3 -u extract_perhit.py \
      --files "$CEPH/$PROD/task_*/globalcor_resclosure_*.root" \
      --groups $GRP --max-chi2-ndof 3 --max-hess 1e8 --max-grad 1e6 \
      --max-cands ${MAXCANDS:-125} --no-compress --require-complete \
      -j ${J:-24} -o $NPZ "$@"
    ;;
  quad)
    cd $HL
    RUNS=$R OUT=perhit_quad.npz PROD=$PROD J=${J:-16} exec ./run_quad.sh "$@"
    ;;
  gates)
    source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
    cd $HERE
    exec python3 -u gates.py --require-complete \
      --files "$CEPH/$PROD/task_*/globalcor_resclosure_*.root" "$@"
    ;;
  xcum)
    source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
    cd $HERE
    exec python3 -u xcum_perhit.py --npz $NPZ "$@"
    ;;
  cards)
    cd $HL
    # the arms, then the injections, then the two joints.
    #  cf/gaussq/gauss     the per-hit complement components
    #  ref                 the 5 truth-referenced ones (the prototype's term)
    #  all                 BOTH, same tracks -- the complementarity test
    #  resmass             per-hit + the J/psi-gun MASS term (disjoint samples)
    for c in cf gaussq gauss; do
      echo "=== card ph_$c $(date +%H:%M:%S)"
      COMPS=hit R=$R NPZ=$NPZ QNPZ=$QNPZ NTRK=$NTRK \
        ./run_ladder.sh card_$c > $HERE/logs/card_ph_$c.log 2>&1 \
        && mv $R/cards/$c.hdf5 $R/cards/ph_$c.hdf5 || echo "FAILED ph_$c"
      tail -2 $HERE/logs/card_ph_$c.log
    done
    for spec in ref all; do
      echo "=== card ph_cf_$spec $(date +%H:%M:%S)"
      COMPS=$spec R=$R NPZ=$NPZ QNPZ=$QNPZ NTRK=$NTRK \
        ./run_ladder.sh card_cf > $HERE/logs/card_ph_cf_$spec.log 2>&1 \
        && mv $R/cards/cf.hdf5 $R/cards/ph_cf_$spec.hdf5 || echo "FAILED $spec"
      tail -2 $HERE/logs/card_ph_cf_$spec.log
    done
    for c in inj_cf inj_gaussq inj_hit inj_hit_gaussq; do
      echo "=== card ph_$c $(date +%H:%M:%S)"
      COMPS=hit R=$R NPZ=$NPZ QNPZ=$QNPZ NTRK=$NTRK \
        ./run_ladder.sh card_$c > $HERE/logs/card_ph_$c.log 2>&1 \
        && mv $R/cards/$c.hdf5 $R/cards/ph_$c.hdf5 || echo "FAILED ph_$c"
      tail -2 $HERE/logs/card_ph_$c.log
    done
    for c in mass resmass inj_mass inj_resmass; do
      echo "=== card ph_$c $(date +%H:%M:%S)"
      COMPS=hit R=$R NPZ=$NPZ QNPZ=$QNPZ NTRK=$NTRK \
        ./run_ladder.sh card_$c > $HERE/logs/card_ph_$c.log 2>&1 \
        && mv $R/cards/$c.hdf5 $R/cards/ph_$c.hdf5 || echo "FAILED ph_$c"
      tail -2 $HERE/logs/card_ph_$c.log
    done
    # the joint of the two arms on the SAME tracks, and its injection
    for nm in cf_all; do :; done
    echo "=== card ph_inj_cf_all $(date +%H:%M:%S)"
    COMPS=all R=$R NPZ=$NPZ QNPZ=$QNPZ NTRK=$NTRK \
      ./run_ladder.sh card_inj_cf > $HERE/logs/card_ph_inj_cf_all.log 2>&1 \
      && mv $R/cards/inj_cf.hdf5 $R/cards/ph_inj_cf_all.hdf5 || echo FAILED
    echo "=== card ph_inj_cf_ref $(date +%H:%M:%S)"
    COMPS=ref R=$R NPZ=$NPZ QNPZ=$QNPZ NTRK=$NTRK \
      ./run_ladder.sh card_inj_cf > $HERE/logs/card_ph_inj_cf_ref.log 2>&1 \
      && mv $R/cards/inj_cf.hdf5 $R/cards/ph_inj_cf_ref.hdf5 || echo FAILED
    echo "CARDS DONE $(date +%H:%M:%S)"
    ;;
  fits)
    cd $HL
    R=$R M=${M:-tf-trust-krylov} exec ./fit_all.sh "$@"
    ;;
  fisher)
    cd $HL
    OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} exec ./run_tf.sh python3 -u fisher_cmp.py \
      --npz $NPZ --arms cf gauss gaussq --comps hit ref "$@"
    ;;
  cost)
    cd $HL
    exec ./run_tf.sh python3 -u cost.py --npz $NPZ --comps ${COMPS} "$@"
    ;;
  plots)
    cd $HL
    exec ./run_tf.sh python3 -u plot_hitlik.py --npz $NPZ --comps ${COMPS} "$@"
    ;;
  subfits)
    cd $HL
    R=$R NPZ=$NPZ QNPZ=$QNPZ COMPS=hit exec ./subfits.sh "$@"
    ;;
  *) sed -n '2,20p' $0; exit 1 ;;
esac
