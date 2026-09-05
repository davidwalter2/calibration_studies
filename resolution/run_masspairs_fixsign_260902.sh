#!/bin/bash
# The --ioni-sign neg (fixed) candidate-mass caches.
#
# `build_pairs_tt` signs each ionization block with `sign(sum resinfv)`.  That
# sign follows the noise-eigenvector convention, not physics: measured over
# 142 267 blocks it is 50.0 % +1 / 50.0 % -1, both signs appear in 100 % of
# candidates, and the w^3-weighted signed survival averages +0.0006 +- 0.0105
# -- i.e. the mass CF carries essentially NO net ionization skew.  The correct
# weight is -1 for every block (dm = -p^2 cs (dm/dp) dE < 0 for both charges).
#
# Two builds, deliberately NOT overwriting anything:
#   runs/cf_masspairs_jpsigun_fixsign.npz               <- Aug-8 production,
#       the matched "after" for runs/cf_masspairs_jpsigun.npz ("before")
#   runs/cf_masspairs_jpsigun_ul16_260902_m0_fixsign.npz <- NEW production;
#       the unfixed ..._m0.npz of the same production is built by another
#       chain and is NOT touched here.
set -uo pipefail
RES=/work/submit/david_w/ZMass/calibration_studies/resolution; cd "$RES"
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CEPH=/ceph/submit/data/user/d/david_w/ZMass/cvh
LOG=$RES/runs/masssign260902; mkdir -p "$LOG"

build() {  # build <outname> <proddir>
  local out=$1 prod=$2
  echo "=== $out ($(date +%H:%M:%S))"
  python3 cf_mass_likelihood.py --pairs-tt --ioni-sign neg \
      --files "$CEPH/$prod/task_*/globalcor_0.root" --ntasks 160 \
      --pairs-cache "runs/$out" > "$LOG/${out%.npz}.log" 2>&1 \
    && echo "    [ok] $out ($(date +%H:%M:%S))" || echo "    [FAIL] $out"
}

build cf_masspairs_jpsigun_fixsign.npz                resolution_trackres_jpsigun_ul16 &
build cf_masspairs_jpsigun_ul16_260902_m0_fixsign.npz resolution_trackres_jpsigun_ul16_260902_m0 &
wait
echo "=== builds done ($(date +%H:%M:%S)) ==="
