#!/bin/bash
# Everything after the production, in dependency order.  Each stage is
# skipped if its output exists, so this is restartable.
#   ./run_stage2.sh [stage ...]     (default: all)
set -uo pipefail
HERE=/work/submit/david_w/ZMass/calibration_studies/resolution/hitlik/perhit
HL=/work/submit/david_w/ZMass/calibration_studies/resolution/hitlik
R=/work/submit/david_w/ZMass/calibration_studies/resolution/runs/perhit
NPZ=$R/perhit.npz
FIG=$HOME/public_html/cvh/260910_perhit
NTRKF=${NTRKF:-10000}     # tracks for the Fisher/sandwich step
NTRKC=${NTRKC:-8000}      # tracks for the card ladder (as hitlik's NTRK=6000)
mkdir -p $R/cards $R/fits $HERE/logs $FIG
cd $HERE
stages=${*:-extract gates xcum quad cards fits fisher efficiency saturation cost plots recovery finaltable}

run() { echo "=== $1  $(date +%H:%M:%S)"; shift; "$@" ; echo "    rc=$?  $(date +%H:%M:%S)"; }

for st in $stages; do
case $st in
  extract)
    [ -s $NPZ ] && { echo "=== extract SKIP (exists)"; continue; }
    run extract ./run_all.sh extract > logs/extract.log 2>&1
    tail -3 logs/extract.log ;;
  gates)
    run gates ./run_all.sh gates --max-tracks 20000 > logs/gates.log 2>&1
    tail -40 logs/gates.log ;;
  xcum)
    run xcum ./run_all.sh xcum -o $R/xcum.npz > logs/xcum.log 2>&1
    cat logs/xcum.log ;;
  quad)
    [ -s $R/perhit_quad.npz ] && { echo "=== quad SKIP"; continue; }
    run quad ./run_all.sh quad > logs/quad.log 2>&1
    tail -3 logs/quad.log ;;
  cards)
    run cards env NTRK=$NTRKC ./run_all.sh cards > logs/cards.log 2>&1
    grep -aE "^===|rows;|residual term|mass term|FAILED" logs/cards.log ;;
  fits)
    cd $HL
    for c in ph_cf ph_gaussq ph_gauss ph_cf_ref ph_cf_all \
             ph_inj_cf ph_inj_gaussq ph_inj_hit ph_inj_hit_gaussq \
             ph_inj_cf_ref ph_inj_cf_all ph_mass ph_resmass \
             ph_inj_mass ph_inj_resmass; do
      [ -s $R/fits/$c/fitresults.hdf5 ] && { echo "  skip $c"; continue; }
      echo "=== fit $c $(date +%H:%M:%S)"
      R=$R ./run_fit.sh $c --minimizerMethod tf-trust-krylov \
        > $HERE/logs/fit_$c.log 2>&1 || echo "  FIT $c FAILED"
      grep -a edmval $HERE/logs/fit_$c.log | tail -1
    done
    cd $HERE ;;
  fisher)
    [ -s $R/fisherHJ.npz ] && { echo "=== fisher SKIP"; continue; }
    cd $HL
    OMP_NUM_THREADS=24 ./run_tf.sh python3 -u fisher_cmp.py --npz $NPZ \
      --arms cf gauss gaussq --comps hit ref --max-tracks $NTRKF \
      --nbatch 200 --chunk 8192 -o $R/fisherHJ.npz \
      > $HERE/logs/fisher.log 2>&1
    tail -20 $HERE/logs/fisher.log
    cd $HERE ;;
  efficiency)
    cd $HL
    for cs in hit ref; do
      ./run_tf.sh python3 -u efficiency.py --fisher $R/fisherHJ.npz \
        --cset $cs --arms cf gauss gaussq --ref cf --ntrk $NTRKF \
        --scale-to 320000 -o $R/eff_$cs.npz > $HERE/logs/eff_$cs.log 2>&1
      echo "--- efficiency, cset $cs"; cat $HERE/logs/eff_$cs.log
    done
    cd $HERE ;;
  saturation)
    source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
    for cs in hit ref; do
      python3 -u saturation.py --fisher $R/fisherHJ.npz --cset $cs \
        --arms cf gaussq --ntrk $NTRKF > logs/sat_$cs.log 2>&1
      echo "--- saturation, cset $cs"; cat logs/sat_$cs.log
    done ;;
  cost)
    cd $HL
    for cs in hit ref all; do
      ./run_tf.sh python3 -u cost.py --npz $NPZ --comps $cs --max-tracks 2000 \
        > $HERE/logs/cost_$cs.log 2>&1
      echo "--- cost, comps $cs"; cat $HERE/logs/cost_$cs.log
    done
    cd $HERE ;;
  plots)
    cd $HL
    for g in cls relpos ckind; do
      ./run_tf.sh python3 -u plot_hitlik.py --npz $NPZ --comps all \
        --arms cf gaussq --densities --group-by $g --min-rows 500 \
        --max-tracks 4000 --outpath $FIG > $HERE/logs/plots_$g.log 2>&1
      grep -a "wrote\|component" $HERE/logs/plots_$g.log | tail -6
    done
    ./run_tf.sh python3 -u plot_hitlik.py --npz $NPZ --comps hit \
      --efficiency $R/eff_hit.npz --outpath $FIG \
      > $HERE/logs/plots_eff.log 2>&1
    grep -a wrote $HERE/logs/plots_eff.log
    cd $HERE ;;
  recovery)
    cd $HL
    P=""
    for n in cf gaussq cf_ref cf_all mass resmass; do
      P="$P $n=$R/fits/ph_$n:$R/fits/ph_inj_$n"
    done
    ./run_tf.sh python3 -u recovery.py --pairs $P \
      --param material_tib_support --prior-sigma 1.0 --truth 0.00243951 \
      > $HERE/logs/recovery_mat.log 2>&1
    cat $HERE/logs/recovery_mat.log
    ./run_tf.sh python3 -u recovery.py --pairs \
      cf=$R/fits/ph_cf:$R/fits/ph_inj_hit \
      gaussq=$R/fits/ph_gaussq:$R/fits/ph_inj_hit_gaussq \
      --param hitres_str_N3_lo --prior-sigma 1.0 \
      > $HERE/logs/recovery_hit.log 2>&1
    cat $HERE/logs/recovery_hit.log
    cd $HERE ;;
  finaltable)
    cd $HL
    ./run_tf.sh python3 -u final_table.py --efficiency $R/eff_hit.npz \
      --fits $R/fits --cf ph_cf --gauss ph_gaussq \
      --inj-cf ph_inj_cf --inj-gauss ph_inj_gaussq \
      --hit-inj-cf ph_inj_hit --hit-inj-gauss ph_inj_hit_gaussq \
      --ntrk-fisher $NTRKF --ntrk-fit $NTRKC \
      > $HERE/logs/final_hit.log 2>&1
    cat $HERE/logs/final_hit.log
    ./run_tf.sh python3 -u final_table.py --efficiency $R/eff_ref.npz \
      --fits $R/fits --cf ph_cf_ref --gauss ph_cf_ref \
      --inj-cf ph_inj_cf_ref --inj-gauss ph_inj_cf_ref \
      --hit-inj-cf ph_inj_hit --hit-inj-gauss ph_inj_hit_gaussq \
      --ntrk-fisher $NTRKF --ntrk-fit $NTRKC \
      > $HERE/logs/final_ref.log 2>&1
    cat $HERE/logs/final_ref.log
    cd $HERE ;;
  *) echo "unknown stage $st" ;;
esac
done
echo "STAGE2 DONE $(date +%H:%M:%S)"
