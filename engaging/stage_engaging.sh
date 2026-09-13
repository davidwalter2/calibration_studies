#!/bin/bash
# ---------------------------------------------------------------------------
# Push code + caches from submit to MIT Engaging.  RUN THIS ON SUBMIT.
#
#   ./stage_engaging.sh [code|caches|all]      (default: all)
#
# Nothing on Engaging is shared with submit's /work or /ceph, and Engaging has
# neither /cvmfs nor xrdcp -- so every input has to be pushed from here.
# The `eng` master connection (see the `engaging` skill) carries all of it.
# ---------------------------------------------------------------------------
set -euo pipefail

ZM=/work/submit/david_w/ZMass
RES=$ZM/calibration_studies/resolution
DEST=engaging:orcd/pool/zmass
STAGE=${1:-all}
TMP=${TMPDIR:-/tmp}/eng_stage.$$
mkdir -p "$TMP"; trap 'rm -rf "$TMP"' EXIT

eng-master   # 8 h multiplexed master; may need a human Duo touch (see README)

if [ "$STAGE" = code ] || [ "$STAGE" = all ]; then
  echo "=== rabbit (branch vmass-conditioning) via git bundle"
  # A bundle keeps the transfer to one 6 MB file and does not depend on
  # Engaging being able to reach github.com.
  git -C "$ZM/rabbit-vmass" bundle create "$TMP/rabbit.bundle" vmass-conditioning
  rsync -a "$TMP/rabbit.bundle" $DEST/
  eng 'bash -lc "
    cd ~/orcd/pool/zmass
    if [ -d rabbit/.git ]; then
      git -C rabbit fetch -f ../rabbit.bundle \
          vmass-conditioning:vmass-conditioning
    else
      git clone -b vmass-conditioning rabbit.bundle rabbit
      git -C rabbit remote set-url origin https://github.com/WMass/rabbit
    fi
    git -C rabbit checkout -q vmass-conditioning && git -C rabbit log --oneline -1
  "'

  echo "=== resolution scripts"
  # Sync every top-level .py (2.4 MB): cf_masslik_fit imports cf_mass_likelihood
  # -> cf_track_resolution -> cf_ms_exact / cf_ioni_exact / hitres_classes /
  # cf_delta_ray / cf_brems_exact / pubhtml / ratiopanel, and those chains keep
  # growing.  Cheaper to ship the lot than to track the closure by hand.
  rsync -a --include='*.py' --exclude='*' "$RES/" $DEST/resolution/

  # wums is NOT staged: it comes from PyPI (0.2.0) inside the env, because
  # rabbit needs wums.sparse_hist which submit's pinned 0.1.12 does not have.

  # `resolution/globalfit/` is NOT covered by the line above: the rsync filter
  # takes top-level .py only. `make_joint_card.py --material` imports
  # `make_global_term` from it, and without it the phase-3 card build dies at
  # `ModuleNotFoundError: No module named 'make_global_term'` (job 22328636).
  rsync -a --include='*.py' --exclude='*' \
      "$RES/globalfit/" $DEST/resolution/globalfit/

  # ... and `make_global_term.param_scales` (what `--whiten` calls) needs the
  # harmonic basis and the 50-mode coefficient table out of the `mfs` checkout,
  # which is not on Engaging. Two files, so stage those rather than the tree;
  # `MFS_DIR` in the batch script points the module at them.
  echo "=== mfs (harmonic basis + the 50-mode coefficients, for --whiten)"
  rsync -a "$ZM/mfs/harmonic_basis.py" $DEST/mfs/
  rsync -a --relative \
      "$ZM/mfs/./data/fitresults/polyfit3d_full_coeffs_lmax18_custom50.txt" \
      $DEST/mfs/


  echo "=== batch scripts"
  rsync -a "$ZM/calibration_studies/engaging/" $DEST/engaging/
fi

if [ "$STAGE" = caches ] || [ "$STAGE" = all ]; then
  echo "=== mass-likelihood caches (3.5 GB, 4 parallel streams ~ 130 MB/s)"
  # Single-stream rsync tops out near 50 MB/s; splitting the file list across
  # processes roughly doubles-to-triples it.  rsync skips identical files, so
  # re-running is cheap.
  cd "$RES/runs"
  for f in cf_masspairs_jpsigun_ul16_260903x_m0.npz \
           cf_masspairs_btojpsix_v3_260903x_m0.npz \
           cf_masskernel_jpsigun_ul16_260903x_m0.npz \
           cf_masskernel_btojpsix_v3_260903x_m0.npz; do
    rsync -a "$f" $DEST/resolution/runs/ &
  done
  wait
fi

echo "=== staged.  Remote listing:"
eng 'du -sh ~/orcd/pool/zmass/* 2>/dev/null'
