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
  echo "=== rabbit (branch unbinned-mass-term) via git bundle"
  # A bundle keeps the transfer to one 6 MB file and does not depend on
  # Engaging being able to reach github.com.
  git -C "$ZM/rabbit" bundle create "$TMP/rabbit.bundle" \
      unbinned-mass-term global-term-card main
  rsync -a "$TMP/rabbit.bundle" $DEST/
  eng 'bash -lc "
    cd ~/orcd/pool/zmass
    if [ -d rabbit/.git ]; then
      git -C rabbit fetch -f ../rabbit.bundle \
          unbinned-mass-term:unbinned-mass-term global-term-card:global-term-card main:main
    else
      git clone -b unbinned-mass-term rabbit.bundle rabbit
      git -C rabbit remote set-url origin https://github.com/WMass/rabbit
    fi
    git -C rabbit checkout -q unbinned-mass-term && git -C rabbit log --oneline -1
  "'

  echo "=== resolution scripts"
  # Sync every top-level .py (2.4 MB): cf_masslik_fit imports cf_mass_likelihood
  # -> cf_track_resolution -> cf_ms_exact / cf_ioni_exact / hitres_classes /
  # cf_delta_ray / cf_brems_exact / pubhtml / ratiopanel, and those chains keep
  # growing.  Cheaper to ship the lot than to track the closure by hand.
  rsync -a --include='*.py' --exclude='*' "$RES/" $DEST/resolution/

  # wums is NOT staged: it comes from PyPI (0.2.0) inside the env, because
  # rabbit needs wums.sparse_hist which submit's pinned 0.1.12 does not have.

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
