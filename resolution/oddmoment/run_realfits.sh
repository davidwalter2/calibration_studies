#!/bin/bash
# alpha on the REAL candidates: naive (fit sigma) vs truth-referenced sigma_bar
# vs the truth-free correction.  r is held at the published single-scale value
# of each sample (rho(alpha, r) = -0.04, so alpha is effectively decoupled).
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution
cd "$RES"
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
G=runs/cf_masspairs_jpsigun_ul16_260905d_m0.npz
GK=runs/cf_masskernel_jpsigun_ul16_260905d_m0.npz
V=runs/cf_masspairs_btojpsix_v3_260903x_m0.npz
VK=runs/cf_masskernel_btojpsix_v3_260903x_m0.npz
SB=oddmoment/out/masspull_jpsigun_260905d.npz
O=oddmoment/out/realfits.txt
case "${1:-}" in
  g_naive) python3 oddmoment/masslik_np.py fit --pairs $G --kernel $GK --r 0.98582 --label "GUN 260905d NAIVE" --out $O ;;
  g_bar)   python3 oddmoment/masslik_np.py fit --pairs $G --kernel $GK --r 0.98582 --sigma-source bar --sbar $SB --label "GUN 260905d TRUTH sigma_bar" --out $O ;;
  g_corr)  python3 oddmoment/masslik_np.py fit --pairs $G --kernel $GK --r 0.98582 --sigma-source corrected --label "GUN 260905d CORRECTED a=(1+vgf)sigma/m" --out $O ;;
  g_corr88) python3 oddmoment/masslik_np.py fit --pairs $G --kernel $GK --r 0.98582 --sigma-source corrected --a-scale 0.88 --label "GUN 260905d CORRECTED a x0.88 (measured/closed-form)" --out $O ;;
  v_naive) python3 oddmoment/masslik_np.py fit --pairs $V --kernel $VK --r 0.98180 --label "V3 260903x NAIVE" --out $O ;;
  v_corr)  python3 oddmoment/masslik_np.py fit --pairs $V --kernel $VK --r 0.98180 --sigma-source corrected --label "V3 260903x CORRECTED a=(1+vgf)sigma/m" --out $O ;;
esac
