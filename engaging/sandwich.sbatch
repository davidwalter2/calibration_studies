#!/bin/bash
#SBATCH --job-name=zsand
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#SBATCH --time=05:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
# The SANDWICH stage of a certified row.
#
# `rabbit_fit.py` finds the minimum and reports the EDM; it does NOT compute
# the sandwich covariance, and the sandwich is not optional -- the MiNNLO
# weights alone put the error up by 9-18 % and the factor is NOT the flat
# x1.109 (measured: 1.088 to 1.177 across the rows of this table).  So:
#
#   rabbit_to_json.py rabbit_$TAG.hdf5 -> $TAG_start.json
#   fit.py --start-from that --method trust-exact --gtol 0
#
# `fit.py` from rabbit's minimum is ALSO the independent cross-check: if it is
# genuinely at a minimum the seeded fit takes ~no steps and reproduces the NLL.
# A row where it walks away is a row whose rabbit fit did not converge.
set -uo pipefail
ZMASS=$HOME/orcd/pool/zmass
source $ZMASS/engaging/setup_env_engaging.sh
export PYTHONPATH=$ZMASS/rabbit-vmass:$ZMASS/fullscale:$ZMASS/zchannel:${PYTHONPATH:-}
CARD=${CARD:?set CARD=...}
TAG=${TAG:?set TAG=...}
RES=${RES:-$ZMASS/fitresults/native/rabbit_${TAG}.hdf5}
OUT=$ZMASS/fitresults
FIX=${FIX:-k_hit k_ms k_ioni k_rad}
cd $ZMASS/fullscale
echo "############ sandwich $TAG  $(date +%H:%M:%S)  card $CARD  from $RES"
python3 -u rabbit_to_json.py "$RES" -o $OUT/${TAG}_start.json --print || exit 1
python3 -u fit.py --card "$CARD" --fix $FIX --label "$TAG" \
    --start-from $OUT/${TAG}_start.json --method trust-exact --gtol 0 \
    ${ARGS:-} -o $OUT/fit_${TAG}.json
echo "############ done $(date +%H:%M:%S)"
