#!/bin/bash
# alpha on the toys: naive (fit sigma) vs truth sigma_bar vs the truth-free
# correction.  The toys are drawn from the REAL per-candidate CF models of the
# gun cache with a known self-consistency coefficient injected, so the only
# thing that differs between arms is how the likelihood treats sigma.
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution
cd "$RES"
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P=runs/cf_masspairs_jpsigun_ul16_260905d_m0.npz
K=runs/cf_masskernel_jpsigun_ul16_260905d_m0.npz
O=oddmoment/out/toyfits.txt
run() { python3 oddmoment/masslik_np.py fit --pairs "$P" --kernel "$K" --r 1.0 --krad 1.0 "$@" --out "$O"; }
case "${1:-all}" in
  a) run --override oddmoment/out/toy_a0.0.npz   --label "toy a=0.000 NAIVE" ;;
  b) run --override oddmoment/out/toy_a0.011.npz --label "toy a=0.011 NAIVE" ;;
  c) run --override oddmoment/out/toy_a0.05.npz  --label "toy a=0.050 NAIVE" ;;
  d) run --override oddmoment/out/toy_a0.011.npz --sigma-source bar --sbar oddmoment/out/toy_a0.011.npz --label "toy a=0.011 TRUE sigma_bar" ;;
  e) run --override oddmoment/out/toy_a0.011.npz --sigma-source corrected --a-const 0.011 --label "toy a=0.011 CORRECTED" ;;
  f) run --override oddmoment/out/toy_a0.05.npz  --sigma-source corrected --a-const 0.05  --label "toy a=0.050 CORRECTED" ;;
  g) run --override oddmoment/out/toy_a0.05.npz  --sigma-source bar --sbar oddmoment/out/toy_a0.05.npz --label "toy a=0.050 TRUE sigma_bar" ;;
esac
