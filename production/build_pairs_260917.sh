#!/bin/bash
# Pairs (mass-term) caches for the four 2026-09-17 ideal-geometry closure
# productions, built exactly as the v2 caches they replace.
#
#   Z leg    -- `--mass-window 91.1876 60` is MANDATORY: the cut is on the GEN
#               mass and the default window is the J/psi one, which selects
#               ZERO Z candidates.  60 gives m_gen in [31, 151], loose enough to
#               be inactive once the card applies the real m_obs in [60, 120],
#               and below the 190.2 that would let the -99 no-gen-match sentinel
#               through.
#   both     -- `--jac-parmtypes 14 15` stores the per-candidate dm/dtheta on
#               the 50 field modes + 42 material groups; that is the block a
#               joint fit needs and the one the v2 `_jac_full` cache carries.
#   ordering -- a sequential pass with no `--maxn`, so the cache is a
#               SUBSEQUENCE of the tree in order and `oddmoment/aux_gen.py` can
#               align to it row by row.
#
# The caches are several GB each and /work is the tight filesystem, so they are
# written on ceph and symlinked under fullscale/runs/.
#
#   usage: ./build_pairs_260917.sh [zideal|zctl|jideal|jctl ...]   (default: all)
set -euo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution
RUNS=/work/submit/david_w/ZMass/calibration_studies/fullscale/runs
LOGS=/work/submit/david_w/ZMass/calibration_studies/fullscale/logs
CEPH=/ceph/submit/data/user/d/david_w/ZMass/cvh
OUT=$CEPH/pairs_260917
mkdir -p "$OUT" "$LOGS"
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate

build () {   # build <cache name> <production dir> <ntasks> [extra args...]
  local name=$1 dir=$2 nt=$3; shift 3
  local dest="$OUT/$name.npz"
  echo "=== $name  $(date +%H:%M:%S)  $dir  ntasks=$nt"
  python3 -u "$RES/cf_inmaker.py" pairs --files "$dir" --ntasks "$nt" \
      --cache "$dest" --jac-parmtypes 14 15 "$@" 2>&1 | tee "$LOGS/$name.log"
  ln -sfn "$dest" "$RUNS/$name.npz"
  ls -lL "$RUNS/$name.npz"
}

for what in "${@:-zideal zctl jideal jctl}"; do
case $what in
  zideal) build zpairs_dyideal_full   "$CEPH/dymc_8p5M_260917_ideal"     0   --mass-window 91.1876 60 ;;
  zctl)   build zpairs_dyalignctl     "$CEPH/dymc_8p5M_260917_alignctl"  0   --mass-window 91.1876 60 ;;
  jideal) build jpairs_ideal_n600     "$CEPH/jpsimc_20M_260917_ideal"    600 ;;
  jctl)   build jpairs_alignctl_n60   "$CEPH/jpsimc_20M_260917_alignctl" 60  ;;
  *) echo "unknown target: $what" >&2; exit 1;;
esac
done
echo "=== done $(date +%H:%M:%S)"
