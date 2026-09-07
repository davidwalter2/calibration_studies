#!/bin/bash
# Round 2 repoint.
#
# FDB8C946 is not just unreadable, it is TRUNCATED: the central original holds
# 51 478 events and our repack held 19 797, so `mkchunks.py` -- which reads
# Events->GetEntries() on the local file -- tiled only 38 % of it.  The
# production has therefore been missing 31 681 events from this file on top of
# the 19 797 that task_1313 silently skipped.
#
# The fix must NOT renumber anything: task index == chunk-list line number, and
# other people's caches are keyed on it.  So line 1314 (task_1313) keeps its
# `0 19797` range and the missing tail is APPENDED as three new tasks at the
# end of the list, 1642-1644.  Every existing index is untouched.
#
# 0B395A0D and 290E1F42 tile their originals exactly (56 789 and 46 077), so
# they only need the path and size swapped.
set -euo pipefail
CHUNKS=/work/submit/david_w/ZMass/calibration_studies/production/condor_jpsimc_v2/chunks_jpsimc_20M_260906_v2.txt
NEWDIR=/ceph/submit/data/user/d/david_w/ZMass/restaged/jpsimc_20M_260906_repack
DRY=${1:-}
FILES="FDB8C946-2D24-0849-AFBB-4CF6EB0AD8E9 0B395A0D-B510-814C-ADFB-E201F55CC6AB 290E1F42-3F99-7B4F-8793-9EFB57F8B51E"
for f in $FILES; do [[ -s $NEWDIR/$f.root ]] || { echo "missing repack $f" >&2; exit 1; }; done

[[ -f $CHUNKS.bak_260907r2 ]] || cp -p "$CHUNKS" "$CHUNKS.bak_260907r2"
SIZES=$(for f in $FILES; do echo "$f $(stat -c %s "$NEWDIR/$f.root")"; done)
awk -v newdir="$NEWDIR" -v sizes="$SIZES" 'BEGIN{
    n=split(sizes,a,"\n"); for(i=1;i<=n;i++){split(a[i],b," "); if(b[1]!="") S[b[1]]=b[2]}
  }
  { p=$1; k=p; sub(/.*\//,"",k); sub(/\.root$/,"",k);
    if (k in S) { $1=newdir"/"k".root"; $4=S[k]; NCH++ }
    print }
  END{ printf("repointed %d lines\n", NCH) > "/dev/stderr" }' "$CHUNKS" > "$CHUNKS.new"

# the FDB8C946 tail: 51478 - 19797 = 31681 events as three tasks
FD=$NEWDIR/FDB8C946-2D24-0849-AFBB-4CF6EB0AD8E9.root
FDSZ=$(stat -c %s "$FD")
if ! grep -q "FDB8C946-2D24-0849-AFBB-4CF6EB0AD8E9.root 19797 " "$CHUNKS.new"; then
  { echo "$FD 19797 10561 $FDSZ"
    echo "$FD 30358 10560 $FDSZ"
    echo "$FD 40918 10560 $FDSZ"; } >> "$CHUNKS.new"
  echo "appended the 31 681-event tail as tasks 1642-1644" >&2
fi
if [[ -n $DRY ]]; then diff "$CHUNKS" "$CHUNKS.new" | head -40; rm -f "$CHUNKS.new"; exit 0; fi
mv "$CHUNKS.new" "$CHUNKS"
echo "chunk list now $(wc -l < "$CHUNKS") lines"
