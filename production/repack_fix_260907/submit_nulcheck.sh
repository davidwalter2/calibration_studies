#!/bin/bash
# Re-run the 7 chunks of the two NUL-carrying-but-runnable inputs into a SIDE
# output tree, so their outputs can be compared candidate by candidate against
# the ones already in the production before anything is replaced.
#
#   0B395A0D-...  tasks 1342-1345      290E1F42-...  tasks 1374-1376
#
# THE BINARY MUST MATCH THE ONE THAT MADE THE EXISTING OUTPUTS, or a difference
# could come from the code rather than from the input.  Those tasks ran in the
# original cluster 3803264, i.e. with `overlay_jpsimc_20M_260906_v2.tgz`
# (fab515e) -- NOT the later `_xrdfix` overlay (ca6058d).  So this pins the
# ORIGINAL payload explicitly; that is the whole point of the exercise and it
# is why recover_xrdfix.sh is not reused here.
#
# It also uses job_jpsimc_v2.sh, not the _diag twin, for the same reason: the
# existing outputs were made by the plain wrapper.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CDIR=/work/submit/david_w/ZMass/calibration_studies/production/condor_jpsimc_v2
source "$CDIR/config_jpsimc_v2.sh"

SIDE=/ceph/submit/data/user/d/david_w/ZMass/cvh/jpsimc_20M_260906_v2_nulcheck
IDX="1342 1343 1344 1345 1374 1375 1376"
PAYLOAD=/ceph/submit/data/user/d/david_w/ZMass/cvh/jpsimc_20M_260906_v2/payload/overlay_jpsimc_20M_260906_v2.tgz
[[ -s $PAYLOAD ]] || { echo "original payload missing: $PAYLOAD" >&2; exit 1; }

mkdir -p "$SIDE/logs"
printf '%s\n' $IDX > "$HERE/.nulcheck_idx.txt"
for i in $IDX; do mkdir -p "$SIDE/task_$(printf '%04d' "$i")"; done

PROXY=/home/submit/david_w/x509up_u$(id -u)
(( $(voms-proxy-info -timeleft -file "$PROXY" 2>/dev/null || echo 0) > 7200 )) \
  || { echo "proxy under 2 h" >&2; exit 1; }

{ echo "nulcheck  : $(date -Is) by $USER"
  echo "purpose   : re-run 1342-1345 / 1374-1376 from repacked (NUL-free) inputs"
  echo "payload   : $(basename "$PAYLOAD")  -- the ORIGINAL, to match the existing outputs"
  echo "chunklist : $CHUNKLIST"; } >> "$SIDE/PROVENANCE.txt"

condor_submit \
  -append "JOBSH        = $CDIR/job_jpsimc_v2.sh" \
  -append "IDXFILE      = $HERE/.nulcheck_idx.txt" \
  -append "CHUNKSFILE   = $CHUNKLIST" \
  -append "CHUNKSBASE   = $(basename "$CHUNKLIST")" \
  -append "PAYLOADFILE  = $PAYLOAD" \
  -append "PAYLOADBASE  = $(basename "$PAYLOAD")" \
  -append "INITFILE     = $INIT" \
  -append "INITBASE     = $(basename "$INIT")" \
  -append "OUTBASE      = $SIDE" \
  -append "OUTROOT      = ${SIDE#/ceph/submit}" \
  -append "OUTHOST      = $OUTHOST" \
  -append "INDOORS      = $INDOORS" \
  -append "LOGDIR       = $SIDE/logs" \
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
  "$CDIR/jpsimc_v2.sub"
