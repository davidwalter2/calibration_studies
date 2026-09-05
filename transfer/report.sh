#!/bin/bash
# Terse progress/completion report for the DY grid->ceph copy.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATUS="${STATUS:-$HERE/transfer_status.tsv}"
MAN="${1:-$HERE/dy_miniaod_full.tsv}"
DESTBASE="${DESTBASE:-/ceph/submit/data/group/cms}"

awk -F'\t' -v man="$MAN" -v dest="$DESTBASE" '
  FNR==NR { if ($0 !~ /^#/) { want[$1]=$2; tb+=$2; tn++ } ; next }
  $3=="ok"   { if (!( $1 in seen )) { seen[$1]=1; ok++; okb+=$2; w+=$4 } }
  $3=="FAIL" { fail[$1]=1 }
  END {
    for (f in fail) if (!(f in seen)) nf++
    printf "files   : %d / %d verified   (%d still to do, %d hard-failed)\n", ok, tn, tn-ok, nf+0
    printf "bytes   : %.3f / %.3f TB   (%.1f%%)\n", okb/1e12, tb/1e12, 100.0*okb/tb
    printf "cpu-wall: %.1f h of transfer time across all streams\n", w/3600.0
  }' "$MAN" "$STATUS"

if [[ -f "$STATUS" ]]; then
  first=$(awk -F'\t' 'NR==1{print $7}' "$STATUS")
  last=$(tail -1 "$STATUS" | cut -f7)
  s=$(date -d "$first" +%s 2>/dev/null); e=$(date -d "$last" +%s 2>/dev/null)
  if [[ -n "$s" && -n "$e" && $e -gt $s ]]; then
    b=$(awk -F'\t' '$3=="ok" && !seen[$1]++ {s+=$2} END{print s+0}' "$STATUS")
    awk -v b="$b" -v d="$((e - s))" 'BEGIN{
      printf "elapsed : %.2f h wall, aggregate %.0f MB/s\n", d/3600.0, b/d/1e6}'
  fi
  echo "failures:"; awk -F'\t' '$3=="FAIL"{print "  "$1}' "$STATUS" | sort -u | head -20
fi
