#!/bin/bash
# One sandwich job per certified row.  ./submit_sandwich.sh [tag ...]
set -uo pipefail
Z=$HOME/orcd/pool/zmass
declare -A CARDS=(
  [f380refS]=z_full380_fl      [Ss6]=z_full380_fl_s6   [Ss7]=z_full380_fl_s7
  [Sdc8]=z_F_dc8               [Stoy]=z_F_toy          [Stoydc]=z_F_toydc
  [Sw70110]=z_F_w70110
  [SMetaB]=z_M_etaB            [SMetaT]=z_M_etaT       [SMetaE]=z_M_etaE
  [SVfull]=z_V_full            [SVtoy]=z_V_toy
  [SVs6]=z_V_s6                [SVs7]=z_V_s7
  [SVetaB]=z_V_etaB            [SVetaT]=z_V_etaT       [SVetaE]=z_V_etaE
  [SVKetaB]=z_VK_etaB          [SVKetaT]=z_VK_etaT
)
for t in ${*:-${!CARDS[@]}}; do
  c=${CARDS[$t]:?unknown tag $t}
  [ -f $Z/fitresults/native/rabbit_${t}.hdf5 ] || { echo "[$t] no rabbit result yet"; continue; }
  sbatch -A mit_general -p mit_preemptable -G h200:1 \
    --export=ALL,CARD=$Z/cards/${c}.hdf5,TAG=$t sandwich.sbatch | sed "s/^/[$t $c] /"
done
