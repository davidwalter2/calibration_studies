#!/bin/bash
# Copy + verify + atomically publish one file of a transfer manifest.
#
# Called by transfer_dy.sh via xargs; the argument is one manifest TSV line.
# Contract: the final destination path only ever appears once the bytes on
# disk have been independently checked against the DAS size AND adler32, so a
# consumer may treat "file exists at the final name" as "file is good".
set -uo pipefail

LINE="$1"
IFS=$'\t' read -r LFN SIZE NEVENTS ADLER SRC <<< "$LINE"
[[ -n "${LFN:-}" ]] || exit 2

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERIFY="$HERE/verify_file.py"
DESTBASE="${DESTBASE:-/ceph/submit/data/group/cms}"
STATUS="${STATUS:?STATUS not set}"
FALLBACKS="${FALLBACKS:-root://cmsxrootd.fnal.gov/ root://cms-xrd-global.cern.ch/}"
SUBSTREAMS="${SUBSTREAMS:-4}"
ATTEMPTS="${ATTEMPTS:-2}"
TIMEOUT="${XRDTIMEOUT:-3600}"

DEST="$DESTBASE$LFN"
PART="$DEST.part.$$"
mkdir -p "$(dirname "$DEST")" 2>/dev/null
# Storage-level guard: if the destination tree is unreachable (e.g. the ceph
# client gets evicted / cephx auth fails, which denies the whole mount), bail
# out with a distinct code instead of recording a per-file FAIL -- otherwise a
# single storage outage burns through the manifest and marks every file failed.
if [[ ! -d "$(dirname "$DEST")" || ! -w "$(dirname "$DEST")" ]]; then
  echo "[HALT] $LFN destination unreachable: $(dirname "$DEST") -- storage down?"
  exit 3
fi

record() {  # lfn bytes adler_ok wall_s rate_MBps source
  flock "$STATUS.lock" -c "printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
     '$LFN' '$1' '$2' '$3' '$4' '$5' \"\$(date -u +%FT%TZ)\" >> '$STATUS'"
}

# --- already published? -------------------------------------------------
if [[ -f "$DEST" ]]; then
  if [[ "$(stat -c %s "$DEST")" == "$SIZE" ]] && grep -qF "$LFN"$'\t' "$STATUS" 2>/dev/null; then
    echo "[skip] $LFN (verified previously)"; exit 0
  fi
  if python3 "$VERIFY" "$DEST" "$SIZE" "$ADLER" >/dev/null 2>&1; then
    echo "[skip] $LFN (revalidated on disk)"; record "$SIZE" ok 0 0 "pre-existing"; exit 0
  fi
  echo "[bad ] $LFN existing copy failed verification -- refetching"
  rm -f "$DEST"
fi

# --- fetch, trying the manifest source then the fallback doors ----------
t0=$SECONDS
for ep in "$SRC" $FALLBACKS; do
  for try in $(seq 1 "$ATTEMPTS"); do
    rm -f "$PART"
    # --cksum makes xrdcp check adler32 on the fly; we still verify on disk,
    # because a truncated transfer can exit 0 (see project note on xrdcp --retry).
    timeout "$TIMEOUT" xrdcp -f --nopbar --silent \
        --streams "$SUBSTREAMS" --cksum "adler32:$ADLER" \
        "${ep%/}/$LFN" "$PART" >/dev/null 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
      echo "[retry] $LFN xrdcp rc=$rc from $ep (try $try)"; continue
    fi
    if ! python3 "$VERIFY" "$PART" "$SIZE" "$ADLER" >/dev/null 2>&1; then
      echo "[retry] $LFN verification FAILED from $ep (try $try)"; continue
    fi
    # verified -> publish atomically (same POSIX filesystem, so mv is a rename)
    mv -f "$PART" "$DEST" || { echo "[FAIL] $LFN mv failed"; rm -f "$PART"; exit 1; }
    chmod 664 "$DEST" 2>/dev/null
    wall=$((SECONDS - t0))
    rate=$(awk -v s="$SIZE" -v w="$wall" 'BEGIN{printf "%.1f", (w>0? s/w/1e6 : 0)}')
    echo "[ ok ] $LFN ${wall}s ${rate}MB/s $ep"
    record "$SIZE" ok "$wall" "$rate" "$ep"
    exit 0
  done
  echo "[fall] $LFN exhausted $ep"
done

rm -f "$PART"
echo "[FAIL] $LFN all sources exhausted"
record 0 FAIL $((SECONDS - t0)) 0 none
exit 1
