#!/bin/bash
# K_S -> pi pi closure: cache -> card -> fit -> result.
#
# usage: ./build_ks.sh <tag> [extra make_card args...]
#   tag = a name for this variant (all / fromb / r-bins / ...)
# env: PAIRS (cache path, default runs/kspairs_<tag>.npz), CARDARGS
set -euo pipefail
TAG=${1:?tag}; shift || true
KS=/work/submit/david_w/ZMass/calibration_studies/resolution/ksclosure
FS=/work/submit/david_w/ZMass/calibration_studies/fullscale
RUNS=$KS/runs
PAIRS=${PAIRS:-$RUNS/kspairs_all.npz}
CARD=$RUNS/ks_${TAG}.hdf5
mkdir -p "$RUNS" "$RUNS/results"

# M(K_S) = 497.611 +- 0.013 MeV (PDG).  The kernel is a DELTA at the
# per-candidate SIMULATED mass (residual mode models m_reco - m_gen), so the
# 2.6e-5 relative uncertainty of the PDG value does not enter the MC closure;
# --mref only sets the lever arm of alpha (delta = mobs - m_ref*alpha*1e-3).
MREF=0.497611

echo "=== card $CARD ==="
RABBIT=/work/submit/david_w/ZMass/rabbit-vmass THREADS=${THREADS:-16} $FS/run_tf.sh \
  python3 -u $FS/make_card.py \
    --pairs "$PAIRS" \
    --residual-mode --mref $MREF --max-resid ${MAXRESID:-0.06} \
    --window ${WINLO:-0.40} ${WINHI:-0.60} \
    --max-sigma-rel ${MAXSIGREL:-0.06} \
    --ares on --jensen exact --corr-form fluctuation \
    --floor-scale 1e-7 --chunk 32768 \
    --name ks --channel ks \
    -o "$CARD" "$@" 2>&1 | tee "$RUNS/card_${TAG}.log"

echo "=== fit ==="
RABBIT=/work/submit/david_w/ZMass/rabbit-vmass THREADS=${THREADS:-16} $FS/run_tf.sh \
  python3 -u /work/submit/david_w/ZMass/rabbit-vmass/bin/rabbit_fit.py "$CARD" \
    --paramModel UnbinnedParams --minimizerMethod trust-exact \
    --freezeParameters k_hit k_ms k_ioni k_rad \
    -t 0 --unblind --diagnostics \
    --outpath "$RUNS/results" --outname "rabbit_${TAG}.hdf5" \
    2>&1 | tee "$RUNS/fit_${TAG}.log"

echo "=== result ==="
RABBIT=/work/submit/david_w/ZMass/rabbit-vmass THREADS=${THREADS:-16} $FS/run_tf.sh \
  python3 -u $FS/rabbit_to_json.py "$RUNS/results/rabbit_${TAG}.hdf5" \
    -o "$RUNS/results/${TAG}.json" 2>&1 | tail -20
grep -oE 'edmval: [0-9.eE+-]+' "$RUNS/fit_${TAG}.log" | tail -2 || true
