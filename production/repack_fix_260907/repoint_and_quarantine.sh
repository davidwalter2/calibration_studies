#!/bin/bash
# Point the 16 chunk lines of the four repaired files at the new split-1
# repacks and quarantine the 12 outputs that were made from the split-99
# staged copies.
#
# The chunk list's 4th field is the file SIZE, and the job wrapper refuses a
# door that serves anything else -- so the size MUST be updated together with
# the path or every re-run dies with "WRONG REPLICA" (exit 8).
#
# The 12 bad outputs are MOVED, not deleted: they are the evidence for the
# 0.82-candidates/event measurement.  They go to task_XXXX/bad_split99_260906/,
# one level down, where the downstream `task_*/globalcor_*.root` globs (all of
# them single-level) cannot reach them, and where their absence at the top
# level makes recover_xrdfix.sh see the task as incomplete.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CHUNKS=/work/submit/david_w/ZMass/calibration_studies/production/condor_jpsimc_v2/chunks_jpsimc_20M_260906_v2.txt
NEWDIR=/ceph/submit/data/user/d/david_w/ZMass/restaged/jpsimc_20M_260906_repack
OUTBASE=/ceph/submit/data/user/d/david_w/ZMass/cvh/jpsimc_20M_260906_v2
DRY=${1:-}

FILES="BDA060EF-B8F8-7349-9277-363C3AB7EA76 0909778B-8728-764F-B41B-1C9DCE5C849E \
       4B9D2D77-92ED-8440-8697-DD4AC560E61C 03249796-312B-514A-990A-00EC873E70E2"
for f in $FILES; do [[ -s $NEWDIR/$f.root ]] || { echo "missing repack $f" >&2; exit 1; }; done

# --- 1. chunk list ---------------------------------------------------------
[[ -f $CHUNKS.bak_260906b ]] || cp -p "$CHUNKS" "$CHUNKS.bak_260906b"
SIZES=$(for f in $FILES; do echo "$f $(stat -c %s "$NEWDIR/$f.root")"; done)
awk -v newdir="$NEWDIR" -v sizes="$SIZES" 'BEGIN{
    n=split(sizes,a,"\n"); for(i=1;i<=n;i++){split(a[i],b," "); if(b[1]!="") S[b[1]]=b[2]}
  }
  { p=$1; k=p; sub(/.*\//,"",k); sub(/\.root$/,"",k);
    if (k in S) { $1=newdir"/"k".root"; $4=S[k]; NCH++ }
    print }
  END{ printf("repointed %d lines\n", NCH) > "/dev/stderr" }' "$CHUNKS" > "$CHUNKS.new"
if [[ -n $DRY ]]; then diff "$CHUNKS" "$CHUNKS.new" | head -40; rm -f "$CHUNKS.new"; exit 0; fi
[[ $(wc -l < "$CHUNKS.new") -eq $(wc -l < "$CHUNKS") ]] || { echo "line count changed!" >&2; exit 1; }
mv "$CHUNKS.new" "$CHUNKS"

# --- 2. quarantine the 12 outputs made from the split-99 copies -------------
for i in 1219 1220 1221 1222 1334 1335 1336 1337 1409 1410 1411 1412; do
  d=$OUTBASE/task_$(printf '%04d' $i)
  q=$d/bad_split99_260906
  [[ -d $d ]] || continue
  compgen -G "$d/globalcor_*.root" > /dev/null || { echo "$d already quarantined"; continue; }
  mkdir -p "$q"
  mv "$d"/globalcor_*.root "$d"/.complete "$q"/ 2>/dev/null || true
  [[ -f $d/local.log ]] && mv "$d/local.log" "$q"/
  echo "quarantined $d -> $q"
done
echo "done"
