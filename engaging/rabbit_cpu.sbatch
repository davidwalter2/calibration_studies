#!/bin/bash
#SBATCH -J rabbit_cpu
#SBATCH -A mit_general
#SBATCH -p mit_normal
#SBATCH -c 64
#SBATCH --mem 360G
#SBATCH -t 12:00:00
#SBATCH -o /home/david_w/orcd/pool/zmass/logs/%x_%j.out
#SBATCH -e /home/david_w/orcd/pool/zmass/logs/%x_%j.err
# ---------------------------------------------------------------------------
# rabbit_fit.py on host memory.  The guaranteed-to-finish path for the unbinned
# mass card: rabbit builds the Hessian with tf.vectorized_map (pfor) over the
# parameters, and the whole pfor batch is live at once, so an L40S's 46 GB is
# not enough for a 300k-candidate MassCFTerm -- it OOMs at chunk 32768 and, six
# minutes later in the minimisation, still OOMs at chunk 8192.  Either give it
# an H200 (141 GB) or run it here, where `mit_normal` allows 386 GB of host RAM.
#
#   sbatch rabbit_cpu.sbatch [card.hdf5]
# ---------------------------------------------------------------------------
set -u
ZMASS=$HOME/orcd/pool/zmass
CARD=${1:-$ZMASS/cards/jpsigun_260903x_families.hdf5}
source $ZMASS/engaging/setup_env_engaging.sh
export CUDA_VISIBLE_DEVICES=""          # force the CPU device even if one leaks in
mkdir -p $ZMASS/fitresults

echo "=== node $(hostname)  job $SLURM_JOB_ID  cores ${SLURM_CPUS_PER_TASK}  $(date)"
echo "=== card $CARD"
/usr/bin/time -v python $ZMASS/rabbit/bin/rabbit_fit.py "$CARD" \
    -o $ZMASS/fitresults --postfix $(basename "${CARD%.hdf5}")_cpu \
    -t 0 --unblind --paramModel UnbinnedParams \
    2>&1 | grep -vE '^\s+(Voluntary|Involuntary|Swaps|File system|Socket|Signals|Page size|Average)'
echo "=== done $(date)"
