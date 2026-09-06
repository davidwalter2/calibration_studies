#!/bin/bash
#SBATCH --job-name=zclip
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#SBATCH --time=05:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
# The corrections' domain, scanned off ONE card.
#
# `corr_clip` is a scalar attribute of the term, so the scan needs no rebuild:
# each arm is the same candidates and the same arrays with the corrections'
# argument bounded at a different multiple of sigma. If the answer is flat over
# 3-10 sigma the treatment is under control; if it is not, the corrections are
# doing something they were never meant to do out in the FSR tail.
set -uo pipefail
ZMASS=$HOME/orcd/pool/zmass
source $ZMASS/engaging/setup_env_engaging.sh
export PYTHONPATH=$ZMASS/rabbit:$ZMASS/fullscale:$ZMASS/zchannel:${PYTHONPATH:-}
CARD=${CARD:?set CARD=...}
TAG=${TAG:-f380}
CH=${CH:-32768}
KFIX="k_hit k_ms k_ioni k_rad"
mkdir -p $ZMASS/fitresults
cd $ZMASS/fullscale
for c in ${CLIPS:-2 3 5 10 0}; do
  echo "############ corr_clip $c  $(date +%H:%M:%S)"
  python3 -u fit.py --card "$CARD" --chunk $CH --fix $KFIX --corr-clip $c \
      --label "${TAG}_clip$c" -o $ZMASS/fitresults/fit_${TAG}_clip$c.json \
    || echo "############ clip $c FAILED (exit $?)"
done
echo "############ done $(date +%H:%M:%S)"
