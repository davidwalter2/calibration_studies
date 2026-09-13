#!/bin/bash
# Stage the WORKING branch (`vmass-conditioning`) to Engaging, into
# `~/orcd/pool/zmass/rabbit-vmass` -- the checkout the live job scripts read
# (`rabbit_vmass*.sbatch`, `fullscale_gpu_vmass.sbatch`, `build_phase3.sbatch`,
# `nanstep.sbatch`).
#
# `stage_eng.sh` and `stage_engaging.sh` push into the SHARED
# `~/orcd/pool/zmass/{rabbit,fullscale}` and would change the code under a
# running fit there. This one touches neither: it writes only `rabbit-vmass/`
# and `fullscale_native/`.
#
#   ./stage_native.sh code      rabbit-vmass bundle + the fullscale drivers
#   ./stage_native.sh card <p>  one datacard into the SHARED cards/ (append only)
#   ./stage_native.sh           code
set -euo pipefail
ZM=/work/submit/david_w/ZMass
FS=$ZM/calibration_studies/fullscale
BR=vmass-conditioning
REMOTE=engaging:orcd/pool/zmass

what=${1:-code}
case "$what" in
code)
    TMP=$(mktemp -d)
    trap 'rm -rf "$TMP"' EXIT
    git -C $ZM/rabbit-vmass bundle create "$TMP/rabbit_vmass.bundle" $BR
    eng-master
    rsync -a --info=progress2 "$TMP/rabbit_vmass.bundle" $REMOTE/
    eng "bash -lc '
      set -e
      cd ~/orcd/pool/zmass
      if [ ! -d rabbit-vmass ]; then
          git clone -q -b $BR rabbit_vmass.bundle rabbit-vmass
      else
          git -C rabbit-vmass checkout -q --detach
          git -C rabbit-vmass fetch -f ../rabbit_vmass.bundle $BR:$BR
          git -C rabbit-vmass checkout -q $BR
          git -C rabbit-vmass reset -q --hard $BR
      fi
      git -C rabbit-vmass log --oneline -1
    '"
    rsync -a --include='*.py' --include='*.sh' --exclude='*' \
          $FS/ $REMOTE/fullscale_native/
    rsync -a --include='*.py' --exclude='*' \
          $ZM/calibration_studies/zchannel/ $REMOTE/zchannel/
    rsync -a $ZM/calibration_studies/engaging/ $REMOTE/engaging/
    echo "staged: rabbit-vmass ($BR), fullscale_native, engaging"
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
