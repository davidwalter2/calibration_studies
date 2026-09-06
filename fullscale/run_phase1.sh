#!/bin/bash
# PHASE 1 -- Z alone, DY v2, resolution FIXED at the MC truth.
#
# ONE card serves every variant: `a_res` and `jensen_s2` are stored arrays and
# the ON/OFF switches are applied at FIT time (`fit.py --ares/--jensen`), and
# "K(m) fixed" is `--fix shape1..shape5`, so every row of the table is the same
# candidates and the same numbers with one thing changed.
#
# usage:  ./run_phase1.sh [stage ...]      stages: cards fits validate
set -euo pipefail
FS=/work/submit/david_w/ZMass/calibration_studies/fullscale
Z=/work/submit/david_w/ZMass/calibration_studies/zchannel
RUN=$FS/run_tf.sh
PAIRS=$FS/runs/zpairs_dyv2.npz
FSRK=$Z/data/kern_loose_band3.3e-4.npz
ACC=$Z/data/acc_loose_d8.json
CARDS=$FS/cards; LOGS=$FS/logs; RES=$FS/results
mkdir -p "$CARDS" "$LOGS" "$RES"
CHUNK=${CHUNK:-262144}
KFIX="k_hit k_ms k_ioni k_rad"
SHAPEFIX="shape1 shape2 shape3 shape4 shape5"
STAGES=${*:-cards fits}

card () {   # card <tag> <extra make_card args...>
  local tag=$1; shift
  [ -f "$CARDS/z_$tag.hdf5" ] && { echo "[card] $tag exists"; return; }
  echo "[card] $tag  $(date +%H:%M:%S)"
  $RUN python3 -u "$FS/make_card.py" --pairs "$PAIRS" --fsr "$FSRK" --acc "$ACC" \
      -o "$CARDS/z_$tag.hdf5" "$@" 2>&1 | tee "$LOGS/card_$tag.log"
}

fit () {    # fit <tag> <card tag> <extra fit args...>
  local tag=$1 ct=$2; shift 2
  echo "[fit] $tag  $(date +%H:%M:%S)"
  THREADS=${THREADS:-48} $RUN python3 -u "$FS/fit.py" --card "$CARDS/z_$ct.hdf5" \
      --chunk "$CHUNK" --fix $KFIX "$@" --label "$tag" \
      -o "$RES/fit_$tag.json" 2>&1 | tee "$LOGS/fit_$tag.log"
}

for st in $STAGES; do
case $st in
cards)
  # the scaling ladder for the chunked Hessian, then the full card
  card n300k --maxn 300000
  card n1M   --maxn 1000000
  card full
  ;;
validate)
  # the chunked Hessian against the monolithic one, and pfor against hvp
  THREADS=32 $RUN python3 -u "$FS/validate_hessian.py" \
      --card "$CARDS/z_n300k.hdf5" --fix $KFIX 2>&1 | tee "$LOGS/validate_hessian.log"
  for t in n300k n1M full; do
    fit "scale_$t" "$t" --no-fit --no-sandwich
  done
  ;;
fits)
  fit base       full
  fit noares     full --ares off
  fit nojensen   full --jensen off
  fit noboth     full --ares off --jensen off
  fit noshape    full --fix $KFIX $SHAPEFIX
  fit jshift     full --jensen shift
  ;;
esac
done
echo "[run_phase1] done $(date +%H:%M:%S)"
