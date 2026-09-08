#!/bin/bash
# PHASE 2: the joint J/psi + Z + hit-chi2 fit, through `rabbit_fit.py`.
#
# `cards/joint_ok_full.hdf5` is the FINAL card: the hit-chi2 external quadratic
# from ALL 1645 J/psi v2 tasks (`quad_jpsiv2_ok.npz`, 16 955 312) plus all 380
# DY tasks (3 751 687) = 20 706 999 candidates; a J/psi MassCFTerm of 3 000 000
# (delta kernel at the PDG mass, no free alpha) and a Z MassCFTerm of 3 682 662.
#
# FROZEN, and named because they must be: the four resolution knobs at the MC
# truth, and the four material groups the hit-chi2 Hessian carries NO
# information on (sec. 0f.17) -- `material_pp1_cables` is touched by no
# candidate on either mass leg either, `thermal_screen` and `support_tube` by
# under 0.1 %.  Freezing them is the alternative to letting the frozen-diagonal
# fix make an unmeasured parameter look measured.
#
#   ./submit_phase2.sh [tag ...]      tags: P2smoke P2X P2K
set -uo pipefail
Z=$HOME/orcd/pool/zmass
# The four the hit-chi2 Hessian is exactly blind to. `material_beampipe` is
# the CONSERVATIVE one: the phase-3 card's occupancy census measures it at
# 99.995 % on both mass legs, so the mass terms could constrain it and this
# list does not let them. FREEZE3=1 drops it, as the cross-check.
FREEZE="k_hit k_ms k_ioni k_rad material_thermal_screen material_support_tube material_pp1_cables"
[ "${FREEZE3:-0}" = 1 ] || FREEZE="$FREEZE material_beampipe"
MODEL="ExternalParams bundle:global_params"
sub () {  # tag card method partition extra-sbatch
  local FREEZE="$FREEZE"
  [ "${FREEZE3:-0}" = 1 ] && FREEZE="${FREEZE% material_beampipe}"
  sbatch -A mit_general -p "$4" -G h200:1 ${5:-} \
    --export=ALL,CARD=$Z/cards/$2.hdf5,TAG=$1,METHOD=$3,MODEL="$MODEL",FREEZE="$FREEZE",GATE=0,FRESH=1 \
    rabbit_vmass.sbatch | sed "s/^/[$1 $2 $3] /"
}
for t in ${*:-P2smoke P2X P2K}; do
  case $t in
    P2smoke) sub P2smoke joint_ok_n500k trust-exact  mit_preemptable "--time=05:45:00" ;;
    P2X)     sub P2X     joint_ok_full  trust-exact  mit_preemptable "--time=1-12:00:00" ;;
    P2K3)    sub P2K3    joint_ok_full  trust-krylov mit_preemptable "--time=1-12:00:00" ;;
    # PHASE 3: both mass terms are MaterialCFTerms over the SAME parmtype-15
    # amounts the hit-chi2 term and the sparse D carry, plus 18 hitres_<class>.
    # Its census names exactly three groups to freeze, and NOT beampipe.
    P3K)     FREEZE3=1 sub P3K joint_mat_v3 trust-krylov mit_preemptable "--time=1-12:00:00" ;;
    P3X)     FREEZE3=1 sub P3X joint_mat_v3 trust-exact  mit_preemptable "--time=1-12:00:00" ;;
    P2K)     sub P2K     joint_ok_full  trust-krylov mit_preemptable "--time=1-12:00:00" ;;
    *) echo "unknown tag $t" ;;
  esac
done
