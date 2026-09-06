#!/bin/bash
#SBATCH -J extract_cpu
#SBATCH -A mit_general
#SBATCH -p mit_normal
#SBATCH -c 8
#SBATCH --mem 96G
#SBATCH -t 06:00:00
#SBATCH -o /home/david_w/orcd/pool/zmass/logs/%x_%j.out
#SBATCH -e /home/david_w/orcd/pool/zmass/logs/%x_%j.err
# ---------------------------------------------------------------------------
# Can the extraction-style (non-GPU) half of the workflow move to Engaging too?
#
#   sbatch extract_cpu.sbatch [nshards]      (default 4)
#
# `cf_track_resolution.py --extract` is pure numpy + uproot over CVH
# `globalcor_resclosure_*.root` files, embarrassingly parallel over tasks and
# serial within one.  So the figure of merit is *wall per file*, and the node
# just multiplies it: mit_normal caps a user at 96 CPU, so 96 shards at a time.
#
# ONE BLAS/OpenMP thread per shard.  numpy defaults to one thread per core, so
# N shards on a 96-core node ask for ~9000 threads from one process group; on
# submit that has previously hit `ulimit -u` and killed unrelated cmsRun jobs.
# ---------------------------------------------------------------------------
set -u
NSHARD=${1:-4}
ZMASS=$HOME/orcd/pool/zmass
source $ZMASS/engaging/setup_env_engaging.sh    # also exports CVH_IONI_KOKOULIN=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd $ZMASS/resolution

IN=$ZMASS/trackres/mugun_ul16_260903x_m0
OUTD=$ZMASS/trackres/out; mkdir -p $OUTD

echo "=== node $(hostname)  job $SLURM_JOB_ID  cores ${SLURM_CPUS_PER_TASK}  $(date)"
echo "=== $(ls -d $IN/task_* | wc -l) input files, $NSHARD shards"
free -g | head -2

t0=$(date +%s)
i=0
for d in $(ls -d $IN/task_* | head -$NSHARD); do
  (
    s=$(date +%s)
    python cf_track_resolution.py --extract \
        --files "$d" \
        --ntasks 1 --cache $OUTD/shard_$(basename $d).npz \
        > $ZMASS/logs/extract_$(basename $d).log 2>&1
    rc=$?
    echo "shard $(basename $d): rc=$rc wall=$(( $(date +%s) - s ))s size=$(stat -c%s $OUTD/shard_$(basename $d).npz 2>/dev/null || echo 0)"
  ) &
  i=$((i+1))
done
wait
echo "=== all $NSHARD shards in $(( $(date +%s) - t0 ))s"
ls -la $OUTD/
echo "=== done $(date)"
