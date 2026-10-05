#!/bin/bash
# Gate A3 of the conditional-Kalman recursion: one file slice of the tail toy
# with arm R (rbgrid.py) next to the estimator arms and the Gaussian marginal,
# on the replicas of the full run (seeds follow the global model index).
#
#   sbatch --array=0-39 --export=ALL,NFILE=4,REPL=60,OUT=<dir> rb_tail_toy.sbatch
#SBATCH --job-name=rbtoy
#SBATCH --partition=submit
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=24:00:00
set -eo pipefail
: "${NFILE:?}" "${REPL:?}" "${OUT:?}"
source /work/submit/david_w/ZMass/mfs/.venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /work/submit/david_w/ZMass/calibration_studies/resolution/trklik
a=$(( SLURM_ARRAY_TASK_ID * NFILE ))
b=$(( a + NFILE ))
mkdir -p "$OUT"
python -u solver/tail_toy.py '/ceph/submit/data/user/d/david_w/ZMass/cvh/trklik/gun_fm_260918/task_*/globalcor_0.root' \
    --models-per-file 16 --replicas "$REPL" --jobs "$SLURM_CPUS_PER_TASK" --skip-A --arm-R \
    --R-h "${RH:-0.3}" --R-rbox "${RBOX:-20}" --R-trim "${RTRIM:-25}" --R-rmax "${RMAX:-3}" ${RMM:+--R-mm} \
    --file-start "$a" --file-stop "$b" --out "$OUT/part_$(printf %03d "$SLURM_ARRAY_TASK_ID").npz" \
    > "$OUT/part_$(printf %03d "$SLURM_ARRAY_TASK_ID").log" 2>&1
