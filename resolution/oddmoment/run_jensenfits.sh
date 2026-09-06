#!/bin/bash
# The GENUINE residual the corrected likelihood exposes, against the
# second-order (Jensen) term of the mass functional.
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution
cd "$RES"
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
G=runs/cf_masspairs_jpsigun_ul16_260905d_m0.npz
GK=runs/cf_masskernel_jpsigun_ul16_260905d_m0.npz
V=runs/cf_masspairs_btojpsix_v3_260903x_m0.npz
VK=runs/cf_masskernel_btojpsix_v3_260903x_m0.npz
O=oddmoment/out/jensenfits.txt
g() { python3 oddmoment/masslik_np.py fit --pairs $G --kernel $GK --r 0.98582 --out $O "$@"; }
v() { python3 oddmoment/masslik_np.py fit --pairs $V --kernel $VK --r 0.98180 --out $O "$@"; }
case "${1:-}" in
  g_cj)   g --sigma-source corrected --jensen 1 --label "GUN corrected + JENSEN" ;;
  g_j)    g --jensen 1 --label "GUN naive + JENSEN (additivity check)" ;;
  v_cj)   v --sigma-source corrected --jensen 1 --label "V3 corrected + JENSEN" ;;
  v_bar)  v --sigma-source corrected --a-scale 0 --label "V3 naive control (a=0)" ;;
  g_q0)   g --sigma-source corrected --srel-bin 0/5 --label "GUN corrected srel bin 0/5" ;;
  g_q1)   g --sigma-source corrected --srel-bin 1/5 --label "GUN corrected srel bin 1/5" ;;
  g_q2)   g --sigma-source corrected --srel-bin 2/5 --label "GUN corrected srel bin 2/5" ;;
  g_q3)   g --sigma-source corrected --srel-bin 3/5 --label "GUN corrected srel bin 3/5" ;;
  g_q4)   g --sigma-source corrected --srel-bin 4/5 --label "GUN corrected srel bin 4/5" ;;
esac
# (appended) differential test on the TRUTH-FREE corrected resolution
case "${1:-}" in
  g_c0) g --sigma-source corrected --binon csrel --srel-bin 0/5 --label "GUN corr csrel bin 0/5" ;;
  g_c1) g --sigma-source corrected --binon csrel --srel-bin 1/5 --label "GUN corr csrel bin 1/5" ;;
  g_c2) g --sigma-source corrected --binon csrel --srel-bin 2/5 --label "GUN corr csrel bin 2/5" ;;
  g_c3) g --sigma-source corrected --binon csrel --srel-bin 3/5 --label "GUN corr csrel bin 3/5" ;;
  g_c4) g --sigma-source corrected --binon csrel --srel-bin 4/5 --label "GUN corr csrel bin 4/5" ;;
  g_cj0) g --sigma-source corrected --jensen 1 --binon csrel --srel-bin 0/5 --label "GUN corr+JENSEN csrel bin 0/5" ;;
  g_cj4) g --sigma-source corrected --jensen 1 --binon csrel --srel-bin 4/5 --label "GUN corr+JENSEN csrel bin 4/5" ;;
esac
# (appended) the EXACT second-order form, against the mean-shift approximation
case "${1:-}" in
  g_ex)  g --sigma-source corrected --jensen 1 --jensen-mode exact --label "GUN corrected + JENSEN-EXACT" ;;
  v_ex)  v --sigma-source corrected --jensen 1 --jensen-mode exact --label "V3 corrected + JENSEN-EXACT" ;;
  g_ex1) g --sigma-source corrected --jensen 1 --jensen-mode exact --binon csrel --srel-bin 1/5 --label "GUN corr+JEXACT csrel bin 1/5" ;;
  g_ex4) g --sigma-source corrected --jensen 1 --jensen-mode exact --binon csrel --srel-bin 4/5 --label "GUN corr+JEXACT csrel bin 4/5" ;;
  g_sh1) g --sigma-source corrected --jensen 1 --binon csrel --srel-bin 1/5 --label "GUN corr+JSHIFT csrel bin 1/5" ;;
esac
