#!/bin/bash
#SBATCH --job-name=zcard3
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#SBATCH --time=11:45:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
# Build the full-scale PHASE-3 card ON ENGAGING.
#
# It is built here and not on submit because it is ~36 GB and submit's /work
# quota cannot hold it: two attempts died with `OSError: [Errno 122] Disk quota
# exceeded` partway through the HDF5 write. Engaging's pool has 869 GB free,
# and the fit runs here anyway, so the card never has to cross the wire.
set -uo pipefail
ZMASS=$HOME/orcd/pool/zmass
source $ZMASS/engaging/setup_env_engaging.sh
export PYTHONPATH=$ZMASS/rabbit-vmass:$ZMASS/fullscale:$ZMASS/zchannel:${PYTHONPATH:-}
cd $ZMASS/fullscale
echo "############ phase-3 card  $(date +%H:%M:%S)"
python3 -u make_joint_card.py --material \
  --jpsi-pairs $ZMASS/runs/gpairs_v2_n50.npz \
  --z-pairs    $ZMASS/runs/gzpairs_dyv2_n50.npz \
  --quad $ZMASS/runs/quad_jpsiv2_ok.npz $ZMASS/runs/quad_dyv2.npz \
  --groups $ZMASS/data/materialGroups50.txt --whiten --shape 5 \
  --fsr $ZMASS/zchannel/data/kern_loose_band3.3e-4.npz \
  --acc $ZMASS/zchannel/data/acc_loose_d8.json \
  --chunk 16384 -o $ZMASS/cards/joint_mat_v3.hdf5
echo "############ done $(date +%H:%M:%S) rc=$?"
