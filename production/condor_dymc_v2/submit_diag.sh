#!/bin/bash
# Diagnostic re-run of specific dymc_8p5M_260906_v2 task indices.
#
# Same payload (the PINNED overlay tarball of the production, so this is the
# UNFIXED build by construction), same cmsRun command line, same 4 threads and
# 5000 MB request -- only the wrapper differs (job_dymc_v2_diag.sh: memory
# sampler, always stage out local.log + memtrace.txt, tail -400 on failure).
#
# It writes to its OWN output tree so it can never collide with the production
# task dirs, and advertises its own +CvhOutBase so status/resume of the
# production do not see it.
#
# usage: ./submit_diag.sh "0 2 3 260 293 294" [REQMEM_MB] [TAGSUFFIX]
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/config_dymc_v2.sh"
IDXLIST=${1:?usage: ./submit_diag.sh "0 2 3" [REQMEM] [SUFFIX]}
REQMEM=${2:-$REQMEM}
SUF=${3:-mem$REQMEM}
PRODBASE=/ceph/submit/data/user/d/david_w/ZMass/cvh/$TAG
OUTBASE=/ceph/submit/data/user/d/david_w/ZMass/cvh/diag_${TAG}_${SUF}
OUTROOT=${OUTBASE#/ceph/submit}
PAYLOAD=$PRODBASE/payload/overlay_${TAG}.tgz
[[ -s "$PAYLOAD" ]] || { echo "pinned payload missing: $PAYLOAD" >&2; exit 1; }
mkdir -p "$OUTBASE/logs"
tr ' ' '\n' <<< "$IDXLIST" | grep -E '^[0-9]+$' > "$HERE/.diag_idx.txt"
while read -r i; do mkdir -p "$OUTBASE/task_$(printf '%04d' "$i")"; done < "$HERE/.diag_idx.txt"
PROXY=/home/submit/david_w/x509up_u$(id -u)
echo "diag outbase=$OUTBASE  mem=${REQMEM}MB  n=$(wc -l < "$HERE/.diag_idx.txt")  payload=$(basename "$PAYLOAD")"
condor_submit \
  -append "JOBSH        = $HERE/job_dymc_v2_diag.sh" \
  -append "IDXFILE      = $HERE/.diag_idx.txt" \
  -append "CHUNKSFILE   = $CHUNKLIST" \
  -append "CHUNKSBASE   = $(basename "$CHUNKLIST")" \
  -append "PAYLOADFILE  = $PAYLOAD" \
  -append "PAYLOADBASE  = $(basename "$PAYLOAD")" \
  -append "INITFILE     = $INIT" \
  -append "INITBASE     = $(basename "$INIT")" \
  -append "OUTBASE      = $OUTBASE" \
  -append "OUTROOT      = $OUTROOT" \
  -append "OUTHOST      = $OUTHOST" \
  -append "REDIR        = $REDIR" \
  -append "LOGDIR       = $OUTBASE/logs" \
  -append "NTHREADS     = $NTHREADS" \
  -append "REQMEM       = $REQMEM" \
  -append "REQDISK      = $REQDISK" \
  -append "RELEASE      = $RELEASE" \
  -append "SCRAMARCH    = $SCRAMARCH" \
  -append "CFGREL       = $CFGREL" \
  -append "PROXY        = $PROXY" \
  -append "OWNER        = $USER" \
  -append "EXTRA        = $EXTRA" \
  -append "DESIRED_SITES = $DESIRED_SITES" \
  -append "REQUIREMENTS = $REQUIREMENTS" \
  "$HERE/dymc_v2_diag.sub"
