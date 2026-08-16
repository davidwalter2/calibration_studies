#!/bin/bash
# Submit a Slurm array job for cvh refit (or any per-file cmsRun job).
#
# usage:
#   ./submit.sh --config <cfg> --filelist <list> --outdir <dir>
#               [--max-running N] [--time HH:MM:SS] [--mem 4G] [--cpus N]
#               [--partition submit] [--name cvh] [--dry-run]
#               [--cmssw-area /path] [--cmsrun-args "k=v k=v"]
#
# Notes:
#   * The cmsRun config must accept VarParsing 'input=<path>'. See
#     Analysis/HitAnalyzer/test/runCvhJpsi.py for the production driver.
#   * Per-task output lands in $OUTDIR/task_<idx>/; Slurm logs in $OUTDIR/logs/.
#   * Default --cmssw-area is the dedicated prod tree (separate from dev).
#     The prod-tree HEAD short hash is auto-appended to OUTDIR for
#     reproducibility; advance prod via "git pull && scram b" in-place.
#   * --max-running caps concurrent array tasks (Ceph/NFS friendliness).
set -euo pipefail

CONFIG=""
FILELIST=""
OUTDIR=""
MAX_RUNNING=""
TIME="12:00:00"
MEM="4G"
CPUS="1"
PARTITION="submit"
NAME="cvh"
DRY_RUN=0
CMSRUN_ARGS=""
# Default to the dedicated prod CMSSW area (separate from dev tree). To
# advance prod to a newer commit:
#   cd /work/submit/david_w/ZMass/CMSSW_10_6_26_prod/src
#   git pull && scram b -j 8        # incremental, no re-clone
# OUTDIR auto-suffix below tags every submission with the prod HEAD hash,
# so reproducibility is preserved without per-commit area names.
# Override with --cmssw-area for debugging.
CMSSW_AREA="/work/submit/david_w/ZMass/CMSSW_10_6_26_prod"

usage() {
  sed -n '2,15p' "$0"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)      CONFIG=$2; shift 2;;
    --filelist)    FILELIST=$2; shift 2;;
    --outdir)      OUTDIR=$2; shift 2;;
    --max-running) MAX_RUNNING=$2; shift 2;;
    --time)        TIME=$2; shift 2;;
    --mem)         MEM=$2; shift 2;;
    --cpus)        CPUS=$2; shift 2;;
    --partition)   PARTITION=$2; shift 2;;
    --name)        NAME=$2; shift 2;;
    --cmsrun-args) CMSRUN_ARGS=$2; shift 2;;
    --cmssw-area)  CMSSW_AREA=$2; shift 2;;
    --dry-run)     DRY_RUN=1; shift;;
    -h|--help)     usage;;
    *) echo "unknown arg: $1" >&2; usage;;
  esac
done

[[ -z "$CONFIG" || -z "$FILELIST" || -z "$OUTDIR" ]] && usage

CONFIG=$(readlink -f "$CONFIG")
FILELIST=$(readlink -f "$FILELIST")
OUTDIR=$(readlink -f -m "$OUTDIR")
SLURM_DIR=$(dirname "$(readlink -f "$0")")

[[ -f "$CONFIG" ]]   || { echo "config not found: $CONFIG"   >&2; exit 1; }
[[ -f "$FILELIST" ]] || { echo "filelist not found: $FILELIST" >&2; exit 1; }

# Stage X509 grid proxy to a path visible from worker nodes (worker /tmp
# is private; /work/submit is shared). Required when the filelist contains
# root:// URLs to /store/data; harmless otherwise.
SRC_PROXY=${X509_USER_PROXY:-/tmp/x509up_u$(id -u)}
PROXY_DST=/work/submit/${USER}/.x509up_slurm.proxy
if [[ -r "$SRC_PROXY" ]]; then
  install -m 600 "$SRC_PROXY" "$PROXY_DST"
  TLEFT=$(voms-proxy-info -timeleft -file "$PROXY_DST" 2>/dev/null || echo 0)
  if [[ "$TLEFT" -lt 86400 ]]; then
    printf 'WARNING: proxy has only %dh%dm left; refresh with:\n  voms-proxy-init -voms cms -valid 192:00\n' \
      $((TLEFT/3600)) $(((TLEFT%3600)/60)) >&2
  else
    printf 'proxy: %s (%dh left)\n' "$PROXY_DST" $((TLEFT/3600))
  fi
else
  echo "WARNING: no readable X509 proxy at $SRC_PROXY (xrootd reads will fail)" >&2
  PROXY_DST=""
fi

# Count non-blank, non-comment lines.
NJOBS=$(grep -cvE '^\s*(#|$)' "$FILELIST")
[[ $NJOBS -eq 0 ]] && { echo "filelist has 0 jobs"; exit 1; }
LAST=$((NJOBS - 1))

ARRAY_SPEC="0-$LAST"
[[ -n "$MAX_RUNNING" ]] && ARRAY_SPEC="${ARRAY_SPEC}%${MAX_RUNNING}"

mkdir -p "$OUTDIR/logs"

CMSSW_AREA=$(readlink -f "$CMSSW_AREA")
[[ -d "$CMSSW_AREA/src" ]] || { echo "CMSSW_AREA invalid: $CMSSW_AREA" >&2; exit 1; }

# Tag OUTDIR with the prod-area commit hash so outputs are traceable to
# the build that made them. Idempotent: skipped if the hash is already
# the suffix.
HASH=$(cd "$CMSSW_AREA/src" && git rev-parse --short HEAD 2>/dev/null || true)
if [[ -n "$HASH" && "$OUTDIR" != *"_${HASH}" ]]; then
  OUTDIR="${OUTDIR}_${HASH}"
fi

EXPORT="ALL,CONFIG=$CONFIG,FILELIST=$FILELIST,OUTDIR=$OUTDIR,SLURM_DIR=$SLURM_DIR,CMSSW_AREA=$CMSSW_AREA"
[[ -n "$PROXY_DST"   ]] && EXPORT="${EXPORT},X509_USER_PROXY=$PROXY_DST"
[[ -n "$CMSRUN_ARGS" ]] && EXPORT="${EXPORT},CMSRUN_ARGS=$CMSRUN_ARGS"

cmd=(
  sbatch
  --job-name="$NAME"
  --partition="$PARTITION"
  --array="$ARRAY_SPEC"
  --time="$TIME"
  --mem="$MEM"
  --cpus-per-task="$CPUS"
  --output="$OUTDIR/logs/${NAME}_%A_%a.out"
  --error="$OUTDIR/logs/${NAME}_%A_%a.err"
  --export="$EXPORT"
  "$SLURM_DIR/array.sbatch"
)

echo "submitting: array $ARRAY_SPEC ($NJOBS files)"
echo "  config   : $CONFIG"
echo "  filelist : $FILELIST"
echo "  outdir   : $OUTDIR"
echo "  cmssw    : $CMSSW_AREA"
echo "  partition: $PARTITION  time=$TIME  mem=$MEM  cpus=$CPUS"
[[ -n "$MAX_RUNNING" ]] && echo "  max-running: $MAX_RUNNING"

if (( DRY_RUN )); then
  printf '  cmd      :'
  printf ' %q' "${cmd[@]}"
  echo
  exit 0
fi

"${cmd[@]}"
