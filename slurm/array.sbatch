#!/bin/bash
# Slurm array job: one task per ALCARECO file. Tasks pick their input from
# $FILELIST using $SLURM_ARRAY_TASK_ID, then exec cmsRun inside cmssw-el7.
#
# Required env (passed via --export by submit.sh):
#   CONFIG    absolute path to a parameterizable cmsRun cfg (uses VarParsing
#             'analysis' with `input=<file>`)
#   FILELIST  absolute path to a text file, one absolute input file per line
#   OUTDIR    absolute base output directory; per-task output goes to
#             $OUTDIR/task_<idx>/
#
#SBATCH --job-name=cvh
#SBATCH --partition=submit
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00

set -euo pipefail
: "${CONFIG:?CONFIG must be set}"
: "${FILELIST:?FILELIST must be set}"
: "${OUTDIR:?OUTDIR must be set}"
: "${SLURM_ARRAY_TASK_ID:?must be run as a Slurm array task}"

# Pick the input for this array index (1-based sed line, 0-based array id).
LINE=$((SLURM_ARRAY_TASK_ID + 1))
INFILE=$(sed -n "${LINE}p" "$FILELIST" | sed 's/[[:space:]]*$//')
if [[ -z "$INFILE" || "$INFILE" == \#* ]]; then
  echo "no input at line ${LINE} of ${FILELIST} (or commented)" >&2
  exit 1
fi

TASK_DIR="$OUTDIR/task_$(printf '%04d' "$SLURM_ARRAY_TASK_ID")"
mkdir -p "$TASK_DIR"

# Point the inner runner at this task's working dir.
RUN_ONE=$(dirname "$(readlink -f "$0")")/run_one.sh

APPTAINER_BIND="/tmp,/home/submit,/work/submit,/scratch/submit,/ceph/submit,/cvmfs,/etc/grid-security,/run" \
  cmssw-el7 --command-to-run "$RUN_ONE" "$CONFIG" "$INFILE" "$TASK_DIR"
