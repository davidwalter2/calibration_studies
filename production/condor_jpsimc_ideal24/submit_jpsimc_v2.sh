#!/bin/bash
# Submit the J/psi ALCARECO CVH two-track re-production to HTCondor (CMS global
# pool).  The condor twin of production/submit_jpsimc20M.sh: same 1642 chunks
# (including the 12 lines repointed at the re-staged copies), same task index
# -> same event range, same `.complete` sentinel, same PROVENANCE.txt written
# next to the output.
#
# WHAT IT DOES BEYOND RENDERING A .sub
#   * builds (or reuses) the CMSSW OVERLAY TARBALL and pins it INSIDE the
#     production's own output tree.  A resumed task must run the same binary
#     as the original submission; a payload sitting in a scratch directory that
#     someone rebuilds next week would silently produce a mixed sample, which
#     is the exact failure config_jpsimc_v2.sh exists to prevent.
#   * refreshes the shared proxy at a path every execute node can be handed
#     (condor delivers it into the job sandbox; /tmp is per-node).
#
# usage: ./submit_jpsimc_v2.sh [--dry-run] [--only "0 1 2"] [--idxfile FILE]
#                            [--threads N] [--mem MB] [--maxidle N] [--rebuild-payload]
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/config_jpsimc_v2.sh"

DRY=0 ; ONLY="" ; IDXFILE="" ; MAXIDLE=0 ; REBUILD=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift;;
    --only)    ONLY=$2; shift 2;;
    --idxfile) IDXFILE=$2; shift 2;;
    --threads) NTHREADS=$2; shift 2;;
    --mem)     REQMEM=$2; shift 2;;
    --maxidle) MAXIDLE=$2; shift 2;;
    --outbase) OUTBASE=$2; shift 2;;
    --rebuild-payload) REBUILD=1; shift;;
    -h|--help) sed -n '2,20p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

[[ -f "$CHUNKLIST" ]] || { echo "chunk list not found: $CHUNKLIST" >&2; exit 1; }
[[ -d "$CMSSW_AREA/src" ]] || { echo "CMSSW area invalid: $CMSSW_AREA" >&2; exit 1; }
[[ -r "$CFG" ]] || { echo "driver not found: $CFG" >&2; exit 1; }
[[ -r "$INIT" ]] || { echo "scalar-potential init file missing: $INIT" >&2; exit 1; }

mkdir -p "$OUTBASE/logs" "$OUTBASE/payload"

# ---- payload: cvmfs supplies the base release, this supplies the diff -------
# lib + biglib + cfipython + python + src.  NOT .SCRAM/config: those bake the
# absolute path of the build area into RuntimeCache.json and MakeData/*.mk, so
# an untar-in-place area reports the ORIGINAL CMSSW_BASE and silently loads the
# wrong libraries.  The worker runs `scram project` to get a clean skeleton and
# untars this on top.
PAYLOAD="$OUTBASE/payload/overlay_${TAG}.tgz"
if [[ ! -s "$PAYLOAD" || $REBUILD -eq 1 ]]; then
  echo "building payload from $CMSSW_AREA ..."
  tar czf "$PAYLOAD.tmp" -C "$CMSSW_AREA" --exclude='.git' --exclude='*.o' \
      lib biglib cfipython python src
  mv "$PAYLOAD.tmp" "$PAYLOAD"
fi
echo "payload: $PAYLOAD ($(stat -c %s "$PAYLOAD") B, built $(stat -c %y "$PAYLOAD" | cut -d. -f1))"

# ---- proxy ----------------------------------------------------------------
PROXY=/home/submit/david_w/x509up_u$(id -u)
SRC=${X509_USER_PROXY:-/tmp/x509up_u$(id -u)}
if [[ -r "$SRC" ]] && [[ $(voms-proxy-info -timeleft -file "$SRC" 2>/dev/null || echo 0) -gt \
                          $(voms-proxy-info -timeleft -file "$PROXY" 2>/dev/null || echo 0) ]]; then
  install -m 600 "$SRC" "$PROXY"
fi
TLEFT=$(voms-proxy-info -timeleft -file "$PROXY" 2>/dev/null || echo 0)
(( TLEFT > 7200 )) || { echo "proxy at $PROXY has ${TLEFT}s left; run voms-proxy-init -voms cms -valid 192:00" >&2; exit 1; }
echo "proxy : $PROXY ($((TLEFT/3600)) h left)"

# ---- the index list -------------------------------------------------------
NTASK_ALL=$(grep -cvE '^\s*(#|$)' "$CHUNKLIST")
# This production runs the FIRST $NTASKS_USED lines of the shared chunk list,
# so task_NNNN is the same event range as in the production it reproduces.
NTASK=${NTASKS_USED:-$NTASK_ALL}
(( NTASK <= NTASK_ALL )) || { echo "NTASKS_USED=$NTASK > $NTASK_ALL lines in $CHUNKLIST" >&2; exit 1; }
if   [[ -n "$IDXFILE" ]]; then cp "$IDXFILE" "$HERE/.submit_idx.txt"
elif [[ -n "$ONLY"    ]]; then tr ' ' '\n' <<< "$ONLY" | grep -E '^[0-9]+$' > "$HERE/.submit_idx.txt"
else                           seq 0 $((NTASK - 1)) > "$HERE/.submit_idx.txt"
fi
NSUB=$(wc -l < "$HERE/.submit_idx.txt")
HASH=$(cd "$CMSSW_AREA/src" && git rev-parse --short HEAD)

# Pre-create every task directory over POSIX ceph. The worker also asks the
# xrootd door to make it (`xrdfs mkdir -p`, and `xrdcp -p`), but doing it here
# turns 380 concurrent namespace creations at stage-out time into one cheap
# serial pass, and it fails LOUDLY here if the quota or the mount is wrong
# instead of after two hours of Geant4e.
while read -r i; do mkdir -p "$OUTBASE/task_$(printf '%04d' "$i")"; done < "$HERE/.submit_idx.txt"

{ echo "tag       : $TAG"
  echo "submitted : $(date -Is) by $USER on $(hostname) via HTCondor (CMS global pool)"
  echo "cmssw     : $CMSSW_AREA @ $HASH  (payload $(basename "$PAYLOAD"))"
  echo "driver    : $CFG"
  echo "chunklist : $CHUNKLIST ($NTASK tasks; $NSUB submitted now)"
  echo "input     : streamed from doors [$INDOORS] (NOT a redirector -- repacked copies)"
  echo "output    : $OUTHOST$OUTROOT  (= $OUTBASE)"
  echo "threads   : $NTHREADS   request_memory=${REQMEM}MB request_disk=${REQDISK}KB"
  echo "sites     : $DESIRED_SITES"
  echo "extra     : numberOfThreads=$NTHREADS $EXTRA"; } > "$OUTBASE/PROVENANCE.txt"

echo "tag=$TAG tasks=$NTASK submitting=$NSUB threads=$NTHREADS mem=${REQMEM}MB cmssw=$HASH"
echo "outbase=$OUTBASE"

APPEND=(
  -append "JOBSH        = $HERE/job_jpsimc_v2.sh"
  -append "IDXFILE      = $HERE/.submit_idx.txt"
  -append "CHUNKSFILE   = $CHUNKLIST"
  -append "CHUNKSBASE   = $(basename "$CHUNKLIST")"
  -append "PAYLOADFILE  = $PAYLOAD"
  -append "PAYLOADBASE  = $(basename "$PAYLOAD")"
  -append "INITFILE     = $INIT"
  -append "INITBASE     = $(basename "$INIT")"
  -append "OUTBASE      = $OUTBASE"
  -append "OUTROOT      = $OUTROOT"
  -append "OUTHOST      = $OUTHOST"
  -append "INDOORS      = $INDOORS"
  -append "LOGDIR       = $OUTBASE/logs"
  -append "NTHREADS     = $NTHREADS"
  -append "REQMEM       = $REQMEM"
  -append "REQDISK      = $REQDISK"
  -append "RELEASE      = $RELEASE"
  -append "SCRAMARCH    = $SCRAMARCH"
  -append "CFGREL       = $CFGREL"
  -append "PROXY        = $PROXY"
  -append "OWNER        = $USER"
  -append "EXTRA        = $EXTRA"
  -append "DESIRED_SITES = $DESIRED_SITES"
  -append "REQUIREMENTS = $REQUIREMENTS"
)
(( MAXIDLE > 0 )) && APPEND+=( -append "max_idle = $MAXIDLE" )

if (( DRY )); then
  printf 'condor_submit '; printf '%q ' "${APPEND[@]}" "$HERE/jpsimc_v2.sub"; echo
  echo "--- first indices ---"; head -5 "$HERE/.submit_idx.txt"; exit 0
fi
condor_submit "${APPEND[@]}" "$HERE/jpsimc_v2.sub"
