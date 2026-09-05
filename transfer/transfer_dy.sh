#!/bin/bash
# Resumable, verified, parallel grid->ceph copy driver.
#
#   ./transfer_dy.sh [manifest.tsv]
#
# Every file is fetched with xrdcp into <dest>.part, checked against the DAS
# size and adler32, and only then renamed to its final name -- so the driver is
# safely interruptible and re-runnable: a second invocation skips whatever is
# already published and picks up the rest.  Progress is appended to
# transfer_status.tsv (lfn, bytes, adler_ok, wall_s, rate_MBps, source, utc).
#
# Environment knobs:
#   NSTREAM     concurrent files            (default 12, keep <= 16)
#   SUBSTREAMS  TCP substreams per xrdcp    (default 4)
#   DESTBASE    ceph prefix the LFN hangs off (default /ceph/submit/data/group/cms)
#   FALLBACKS   endpoints to try after the manifest source
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${1:-$HERE/dy_miniaod_full.tsv}"
export DESTBASE="${DESTBASE:-/ceph/submit/data/group/cms}"
export STATUS="${STATUS:-$HERE/transfer_status.tsv}"
export FALLBACKS="${FALLBACKS:-root://cmsxrootd.fnal.gov/ root://cms-xrd-global.cern.ch/}"
export SUBSTREAMS="${SUBSTREAMS:-4}"
export ATTEMPTS="${ATTEMPTS:-2}"
export X509_USER_PROXY="${X509_USER_PROXY:-/tmp/x509up_u$(id -u)}"
NSTREAM="${NSTREAM:-12}"

[[ -r "$MANIFEST" ]] || { echo "no manifest: $MANIFEST" >&2; exit 1; }
[[ -r "$X509_USER_PROXY" ]] || { echo "no proxy: $X509_USER_PROXY" >&2; exit 1; }
command -v xrdcp >/dev/null || { echo "xrdcp not on PATH: source cmsset_default.sh" >&2; exit 1; }

# preflight: a dead storage mount must stop the run, not fail 1731 files
probe="$DESTBASE/store"
if [[ ! -d "$probe" || ! -w "$probe" ]]; then
  echo "destination not writable: $probe (storage down? wrong node?)" >&2; exit 4
fi

touch "$STATUS" "$STATUS.lock"
# xrootd client: keep a single transfer from hanging the whole pool
export XRD_REQUESTTIMEOUT=${XRD_REQUESTTIMEOUT:-1800}
export XRD_CONNECTIONWINDOW=${XRD_CONNECTIONWINDOW:-60}
export XRD_STREAMTIMEOUT=${XRD_STREAMTIMEOUT:-120}

ntot=$(grep -vc '^#' "$MANIFEST")
bytes=$(grep -v '^#' "$MANIFEST" | awk -F'\t' '{s+=$2} END{printf "%.3f", s/1e12}')
echo "=== transfer start $(date -u +%FT%TZ)"
echo "    manifest : $MANIFEST  ($ntot files, $bytes TB)"
echo "    dest     : $DESTBASE/store/..."
echo "    streams  : $NSTREAM concurrent x $SUBSTREAMS substreams"
echo "    status   : $STATUS"

grep -v '^#' "$MANIFEST" \
  | xargs -d '\n' -P "$NSTREAM" -I{} "$HERE/transfer_one.sh" "{}"
rc=$?

nok=$(awk -F'\t' '$3=="ok"' "$STATUS" 2>/dev/null | cut -f1 | sort -u | wc -l)
nfail=$(awk -F'\t' '$3=="FAIL"' "$STATUS" 2>/dev/null | cut -f1 | sort -u | wc -l)
echo "=== transfer end   $(date -u +%FT%TZ)  verified=$nok failed=$nfail xargs_rc=$rc"
echo "    re-run the same command to retry anything missing (already-published files are skipped)"
exit $rc
