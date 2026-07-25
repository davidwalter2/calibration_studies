#!/bin/bash
#SBATCH --job-name=lagun
#SBATCH --partition=submit
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=/work/submit/david_w/ZMass/calibration_studies/pixelhits/lagun/slurm_%A_%a.out
set -euo pipefail
VARIANTS=(nominal shifted)
V=${VARIANTS[$((SLURM_ARRAY_TASK_ID % 2))]}
export GUN_CHUNK=$((1 + SLURM_ARRAY_TASK_ID / 2))
export APPTAINERENV_GUN_CHUNK=$GUN_CHUNK
APPTAINER_BIND="/tmp,/home/submit,/work/submit,/scratch/submit,/ceph/submit,/cvmfs,/etc/grid-security,/run" \
  cmssw-el7 --command-to-run /work/submit/david_w/ZMass/calibration_studies/pixelhits/lagun/gun_chunk_run.sh ${V}
