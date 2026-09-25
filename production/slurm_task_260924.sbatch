#!/bin/bash
# slurm carrier for the 2026-09-24 closure productions: runs the condor job
# script (job_{dymc,jpsimc}_v2.sh) of one condor_*24/ directory UNCHANGED, in a
# node-local sandbox laid out exactly as HTCondor lays it out (chunk list,
# pinned overlay tarball, coefficient file and proxy in the cwd), so every task
# runs the same pinned binary, reads its input through the same door/redirector
# and stages out through the same size-verified xrdcp + `.complete` sentinel.
# Why: on 2026-09-24 the flock collector t3serv009:11000 was down (no job of any
# user on the submit06 schedd had started since 09-23 21:48).
#
# env (via --export): DIR = the condor_*24 directory
#SBATCH --partition=submit
#SBATCH --cpus-per-task=4
#SBATCH --mem=5000M
set -uo pipefail
: "${DIR:?}"
source "$DIR"/config_*_v2.sh
IDX=$SLURM_ARRAY_TASK_ID
TASK=task_$(printf '%04d' "$IDX")
[[ -f "$OUTBASE/$TASK/.complete" ]] && { echo "[skip] $TASK complete"; exit 0; }
JOBSH=$(ls "$DIR"/job_*_v2.sh)
PAYLOAD="$OUTBASE/payload/overlay_${TAG}.tgz"
SB=/tmp/david_w_cvh24_${SLURM_JOB_ID}_${IDX}
rm -rf "$SB"; mkdir -p "$SB"; trap 'cd /; rm -rf "$SB"' EXIT
cp "$CHUNKLIST" "$PAYLOAD" "$INIT" "$SB"/ || { echo "[FATAL] sandbox copy failed on $(hostname)" >&2; exit 12; }
cp /home/submit/david_w/x509up_u$(id -u) "$SB/x509up"; chmod 600 "$SB/x509up"
cd "$SB"
export _CONDOR_SCRATCH_DIR=$SB X509_USER_PROXY=$SB/x509up GLIDEIN_Site=slurm_submit
export OUTBASE OUTROOT OUTHOST NTHREADS RELEASE CFGREL EXTRA
export REDIR=${REDIR:-root://cms-xrd-global.cern.ch/} INDOORS=${INDOORS:-submit50.mit.edu}
export CHUNKS=$(basename "$CHUNKLIST") PAYLOAD=$(basename "$PAYLOAD") INITBASE=$(basename "$INIT") SCRAM_ARCH_USE=$SCRAMARCH
bash "$JOBSH" "$IDX"
