#!/bin/bash
# The certified-table suite, every row through `rabbit_fit.py` with SCIPY
# `trust-exact` (the explicit-Hessian More-Sorensen with LAPACK's accelerated
# lambda bound).  Both TF ports fail their subproblem on the full-statistics
# card -- GLTR loses Lanczos orthogonality and the TF More-Sorensen port hits
# its lambda-search maxiter -- so neither reaches the known minimum there.
#
#   ./submit_scipy_suite.sh [tag ...]     (default: all)
set -uo pipefail
Z=$HOME/orcd/pool/zmass
FREEZE="k_hit k_ms k_ioni k_rad"
declare -A CARDS=(
  [f380refS]=z_full380_fl      [Ss6]=z_full380_fl_s6   [Ss7]=z_full380_fl_s7
  [Sdc8]=z_F_dc8               [Stoy]=z_F_toy          [Stoydc]=z_F_toydc
  [Sw70110]=z_F_w70110
  [SMetaB]=z_M_etaB            [SMetaT]=z_M_etaT       [SMetaE]=z_M_etaE
  [SVfull]=z_V_full            [SVtoy]=z_V_toy
  [SVs6]=z_V_s6                [SVs7]=z_V_s7
  [SVetaB]=z_V_etaB            [SVetaT]=z_V_etaT       [SVetaE]=z_V_etaE
  [SVKetaB]=z_VK_etaB          [SVKetaT]=z_VK_etaT
)
TAGS=${*:-${!CARDS[@]}}
for t in $TAGS; do
  c=${CARDS[$t]:?unknown tag $t}
  sbatch -A mit_general -p ${PART:-mit_preemptable} -G h200:1 ${SBOPT:-} \
    --export=ALL,CARD=$Z/cards/${c}.hdf5,TAG=$t,METHOD=trust-exact,FREEZE="$FREEZE",GATE=0,FRESH=1 \
    rabbit_vmass.sbatch | sed "s/^/[$t $c] /"
done
