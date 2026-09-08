#!/bin/bash
# Stage the NATIVE-MINIMIZER sandbox to Engaging, in directories of its own.
#
# `stage_eng.sh` and `stage_engaging.sh` both push into the SHARED
# `~/orcd/pool/zmass/{rabbit,fullscale}`, and `stage_engaging.sh` additionally
# does `git checkout unbinned-mass-term` there. Either would change the code
# under a running fit of the analysis agent. This one touches neither: it
# writes only `rabbit_native/` and `fullscale_native/`.
#
#   ./stage_native.sh code      rabbit-native bundle + the fullscale drivers
#   ./stage_native.sh card <p>  one datacard into the SHARED cards/ (append only)
#   ./stage_native.sh           code
set -euo pipefail
ZM=/work/submit/david_w/ZMass
FS=$ZM/calibration_studies/fullscale
BR=material-resolution-native
REMOTE=engaging:orcd/pool/zmass

what=${1:-code}
case "$what" in
code)
    TMP=$(mktemp -d)
    trap 'rm -rf "$TMP"' EXIT
    git -C $ZM/rabbit bundle create "$TMP/rabbit_native.bundle" $BR
    eng-master
    rsync -a --info=progress2 "$TMP/rabbit_native.bundle" $REMOTE/
    eng "bash -lc '
      set -e
      cd ~/orcd/pool/zmass
      if [ ! -d rabbit_native ]; then
          git clone -q -b $BR rabbit_native.bundle rabbit_native
      else
          git -C rabbit_native checkout -q --detach
          git -C rabbit_native fetch -f ../rabbit_native.bundle $BR:$BR
          git -C rabbit_native checkout -q $BR
          git -C rabbit_native reset -q --hard $BR
      fi
      git -C rabbit_native log --oneline -1
    '"
    rsync -a --include='*.py' --include='*.sh' --exclude='*' \
          $FS/ $REMOTE/fullscale_native/
    rsync -a --include='*.py' --exclude='*' \
          $ZM/calibration_studies/zchannel/ $REMOTE/zchannel/
    rsync -a $ZM/calibration_studies/engaging/ $REMOTE/engaging/
    echo "staged: rabbit_native ($BR), fullscale_native, engaging"
    ;;
card)
    CARD=${2:?usage: ./stage_native.sh card <path>}
    eng-master
    rsync -a --info=progress2 "$CARD" $REMOTE/cards/
    ;;
*)
    echo "usage: $0 [code|card <path>]" >&2
    exit 1
    ;;
esac
