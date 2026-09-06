#!/bin/bash
# Re-run the tasks that died on the XrdAdaptor null dereference, from a build
# that has the fix (see the commit "XrdAdaptor: a failed transport query is a
# null pointer, not an empty string").
#
# WHY THIS IS NOT `resume_*_v2.sh`
# -------------------------------
# `resume` deliberately reuses the payload tarball PINNED in the production's
# own output tree, so that a task resumed months later runs the same binary as
# the original submission. That is the right default and must not be broken:
# the ~1000 jobs still in the queue re-fetch that same pinned tarball whenever
# condor restarts them, so OVERWRITING it would silently mix two binaries into
# one sample.
#
# So the fixed build goes to a SECOND pinned tarball next to the first, and only
# the tasks named here get it. The two differ in:
#     libUtilitiesXrdAdaptor.so   (the crash fix -- I/O only, no physics)
#     libTrackPropagationGeant4e.so + pluginAnalysisHitAnalyzerAuto.so
#                                 (locking only: one shared mutex for the two
#                                  shared dE/dx table builds + atomic pointers)
# Both are value-neutral, and the gate for that claim is
# `resolution/smoke_exports_260906.sh <area> <out>` reproducing
# scratch_smoke_260906/v6_default bit for bit -- run it before using this.
#
# It also runs the DIAGNOSTIC wrapper (job_jpsimc_v2_diag.sh): same cmsRun command
# line, plus a 15 s memory sampler and an unconditional stage-out of local.log,
# so a repeat failure leaves evidence instead of a truncated .err.
#
# usage: ./recover_xrdfix.sh [--dry-run] [--only "0 2 3"]
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/config_jpsimc_v2.sh"
DRY=0 ; ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift;;
    --only)    ONLY=$2; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

PAYLOAD="$OUTBASE/payload/overlay_${TAG}_xrdfix.tgz"
HASH=$(cd "$CMSSW_AREA/src" && git rev-parse --short HEAD)
if [[ ! -s "$PAYLOAD" ]]; then
  echo "building fixed payload from $CMSSW_AREA @ $HASH ..."
  mkdir -p "$OUTBASE/payload"
  tar czf "$PAYLOAD.tmp" -C "$CMSSW_AREA" --exclude='.git' --exclude='*.o' \
      lib biglib cfipython python src
  mv "$PAYLOAD.tmp" "$PAYLOAD"
fi
echo "payload: $PAYLOAD ($(stat -c %s "$PAYLOAD") B) cmssw=$HASH"

# Which indices: no sentinel, and not already in the queue (two jobs writing the
# same task dir would interleave their outputs).
NTASK=$(grep -cvE '^\s*(#|$)' "$CHUNKLIST")
mapfile -t QUEUED < <(condor_q "$USER" -af CvhOutBase CvhTaskIdx 2>/dev/null \
                      | awk -v b="$OUTBASE" '$1 == b && $2 ~ /^[0-9]+$/ {print $2}' || true)
declare -A INQ=(); for q in ${QUEUED[@]+"${QUEUED[@]}"}; do [[ -n "$q" ]] && INQ[$q]=1; done
MISSING=()
for (( i=0; i<NTASK; i++ )); do
  [[ -n "$ONLY" ]] && ! grep -qw "$i" <<< "$ONLY" && continue
  [[ -f "$OUTBASE/task_$(printf '%04d' "$i")/.complete" ]] && continue
  [[ -n "${INQ[$i]:-}" ]] && continue
  MISSING+=("$i")
done
echo "tag=$TAG  already queued=${#QUEUED[@]}  to recover=${#MISSING[@]}"
[[ ${#MISSING[@]} -eq 0 ]] && exit 0
printf '%s\n' "${MISSING[@]}" > "$HERE/.recover_idx.txt"
(( DRY )) && { cat "$HERE/.recover_idx.txt" | tr '\n' ' '; echo; exit 0; }

PROXY=/home/submit/david_w/x509up_u$(id -u)
(( $(voms-proxy-info -timeleft -file "$PROXY" 2>/dev/null || echo 0) > 7200 )) \
  || { echo "proxy at $PROXY under 2 h" >&2; exit 1; }
while read -r i; do mkdir -p "$OUTBASE/task_$(printf '%04d' "$i")"; done < "$HERE/.recover_idx.txt"

{ echo "recovered : $(date -Is) by $USER"
  echo "cmssw     : $CMSSW_AREA @ $HASH  (payload $(basename "$PAYLOAD"))"
  echo "indices   : $(tr '\n' ' ' < "$HERE/.recover_idx.txt")"
  echo "reason    : XrdAdaptor tracerouteRedirections null deref (SIGSEGV)"; } \
  >> "$OUTBASE/PROVENANCE.txt"

condor_submit \
  -append "JOBSH        = $HERE/job_jpsimc_v2_diag.sh" \
  -append "IDXFILE      = $HERE/.recover_idx.txt" \
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
  "$HERE/jpsimc_v2.sub"
