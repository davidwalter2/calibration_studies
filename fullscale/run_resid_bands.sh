#!/bin/bash
# Fit the kernel-free per-band cards LOCALLY: one free parameter (`alpha`),
# so this never needs the GPU queue. Through `rabbit_fit.py`, as every fit in
# this campaign is, then `fit.py --start-from` at rabbit's minimum for the
# sandwich -- which is also the independent check that rabbit's point is a
# minimum (a seeded fit that walks away is a fit that had not converged).
set -uo pipefail
FS=/work/submit/david_w/ZMass/calibration_studies/fullscale
RABBIT=/work/submit/david_w/ZMass/rabbit-vmass
export PYTHONPATH=$RABBIT:/work/submit/david_w/WRemnants_dev/wums:$FS/../env_tf/pypath:$FS/../resolution
export TF_CPP_MIN_LOG_LEVEL=2 OMP_NUM_THREADS=${THREADS:-6} TF_NUM_INTRAOP_THREADS=${THREADS:-6}
P=/opt/venv/bin/python3
mkdir -p $FS/results/resid
for tag in "$@"; do
  card=$FS/cards/z_$tag.hdf5
  [ -f "$card" ] || { echo "[$tag] no card"; continue; }
  echo "############ $tag  $(date +%H:%M:%S)"
  $P -u $RABBIT/bin/rabbit_fit.py "$card" \
      --paramModel UnbinnedParams --minimizerMethod trust-exact \
      --freezeParameters k_hit k_ms k_ioni k_rad \
      -t 0 --unblind --diagnostics \
      --outpath $FS/results/resid --outname rabbit_$tag.hdf5 \
      > $FS/logs/fit_$tag.log 2>&1
  echo "   rabbit rc=$?  $(grep -oE 'edmval: [0-9.e+-]+' $FS/logs/fit_$tag.log | tail -1)"
  $P -u $FS/rabbit_to_json.py $FS/results/resid/rabbit_$tag.hdf5 \
      -o $FS/results/resid/${tag}_start.json >/dev/null 2>&1
  $P -u $FS/fit.py --card "$card" --fix k_hit k_ms k_ioni k_rad --label "$tag" \
      --start-from $FS/results/resid/${tag}_start.json \
      --method trust-exact --gtol 0 -o $FS/results/resid/fit_$tag.json \
      >> $FS/logs/fit_$tag.log 2>&1
  echo "   sandwich rc=$?"
done
