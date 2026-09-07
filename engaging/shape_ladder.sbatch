#!/bin/bash
#SBATCH --job-name=zshape
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#SBATCH --time=05:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
# The K(m) ladder at FULL statistics -- the one loose end of phase 1.
#
# At 300 k, 5 -> 7 Legendre terms moves m_Z by -1.3 MeV (inside its error) and
# Gamma_Z by +42 MeV, where at generator level 5 -> 6 moved 0.4 MeV.  Until this
# ladder exists at 3.68 M the Gamma_Z closure is a tens-of-MeV statement.
#
#   sbatch -A mit_general -G h200:1 -p mit_preemptable shape_ladder.sbatch
set -uo pipefail
ZMASS=$HOME/orcd/pool/zmass
source $ZMASS/engaging/setup_env_engaging.sh
export PYTHONPATH=$ZMASS/rabbit:$ZMASS/fullscale:$ZMASS/zchannel:${PYTHONPATH:-}
cd $ZMASS/fullscale
mkdir -p $ZMASS/fitresults
for N in 6 7; do
  C=$ZMASS/cards/z_full380_fl_s$N.hdf5
  [ -f "$C" ] || { echo "############ s$N: no card $C"; continue; }
  echo "############ s$N  $(date +%H:%M:%S)"
  python3 -u fit.py --card "$C" --fix k_hit k_ms k_ioni k_rad --chunk 32768 \
      --label f380fl_s$N -o $ZMASS/fitresults/fit_f380fl_s$N.json \
    || echo "############ s$N FAILED (exit $?)"
done
echo "############ done $(date +%H:%M:%S)"
