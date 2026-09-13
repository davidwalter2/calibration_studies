#!/bin/bash
# Push the full-scale fit to MIT Engaging (the merged rabbit + fullscale/ + a card).
#
# The chunked Hessian removes the memory wall but not the arithmetic: on submit
# a full-scale Hessian is tens of minutes, on an H200 it is seconds
# (engaging/README.md measures 136x on NLL+grad+Hessian). Engaging has no
# /cvmfs and no /ceph, so everything travels by rsync over the `eng` master.
#
#   ./stage_eng.sh code            merged rabbit bundle + fullscale scripts
#   ./stage_eng.sh card <path>     one datacard
set -euo pipefail
ZM=/work/submit/david_w/ZMass
FS=$ZM/calibration_studies/fullscale
DEST=engaging:orcd/pool/zmass
TMP=${TMPDIR:-/tmp}/fs_stage.$$
mkdir -p "$TMP"; trap 'rm -rf "$TMP"' EXIT
eng-master

case "${1:-code}" in
code)
  echo "=== rabbit (branch vmass-conditioning) via git bundle"
  git -C "$ZM/rabbit-vmass" bundle create "$TMP/rabbit_vmass.bundle" vmass-conditioning
  rsync -a "$TMP/rabbit_vmass.bundle" $DEST/
  # git refuses to fetch into a CHECKED-OUT branch, so detach first
  eng 'bash -lc "
    cd ~/orcd/pool/zmass
    if [ -d rabbit/.git ]; then
      git -C rabbit checkout -q --detach
      git -C rabbit fetch -f ../rabbit_vmass.bundle \
          vmass-conditioning:vmass-conditioning
    else
      git clone -b vmass-conditioning rabbit_vmass.bundle rabbit
    fi
    git -C rabbit checkout -q vmass-conditioning && git -C rabbit log --oneline -1
  "'
  echo "=== fullscale scripts"
  rsync -a --include='*.py' --include='*.sh' --exclude='*' "$FS/" $DEST/fullscale/
  rsync -a --include='*.py' --exclude='*' "$ZM/calibration_studies/zchannel/" \
        $DEST/zchannel/
  ;;
card)
  CARD=${2:?usage: ./stage_eng.sh card <path>}
  echo "=== $(basename "$CARD") ($(du -h "$CARD" | cut -f1))"
  time rsync -a --info=progress2 "$CARD" $DEST/cards/
  ;;
*) echo "usage: $0 [code|card <path>]"; exit 1;;
esac
