#!/bin/bash
# Progress of the HTCondor DY re-production (dymc_8p5M_260906_v2).
#
# The counting rule is the slurm production's: completion comes from the
# `.complete` SENTINEL, never from the .root, because cmsRun creates its output
# at START and a killed job leaves a non-empty but truncated file that an `-s`
# test would accept forever.
#
# usage: ./status_dymc_v2.sh [--slow]
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/config_dymc_v2.sh"
SLOW=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --slow) SLOW=1; shift;;
    --outbase) OUTBASE=$2; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

NTASK=$(grep -cvE '^\s*(#|$)' "$CHUNKLIST" 2>/dev/null || echo 0)
NEVTOT=$(awk '{s+=$3} END{print s+0}' "$CHUNKLIST" 2>/dev/null)
DONE=$(ls -d "$OUTBASE"/task_*/.complete 2>/dev/null | wc -l)
STARTED=$(ls -d "$OUTBASE"/task_*/ 2>/dev/null | wc -l)

read -r RUN IDLE HELD < <(condor_q -totals -af:h 2>/dev/null >/dev/null; \
  condor_q "$USER" -json 2>/dev/null | python3 -c '
import json,sys
try: j=json.load(sys.stdin)
except Exception: j=[]
r=sum(1 for x in j if x.get("JobStatus")==2)
i=sum(1 for x in j if x.get("JobStatus")==1)
h=sum(1 for x in j if x.get("JobStatus")==5)
print(r,i,h)')
RUN=${RUN:-0}; IDLE=${IDLE:-0}; HELD=${HELD:-0}
STALE=$(( STARTED - DONE - RUN )); (( STALE < 0 )) && STALE=0

printf 'tag        : %s\n' "$TAG"
printf 'outbase    : %s\n' "$OUTBASE"
printf 'threads    : %s   request_memory: %s MB\n' "$NTHREADS" "$REQMEM"
printf 'tasks      : %d total\n' "$NTASK"
printf 'complete   : %d (%.1f%%)\n' "$DONE" "$(awk -v d=$DONE -v n=$NTASK 'BEGIN{print n?100*d/n:0}')"
printf 'condor     : %d running, %d idle, %d held\n' "$RUN" "$IDLE" "$HELD"
printf 'started-not-complete (failed or in flight): %d\n' "$STALE"
printf 'events     : %d done of %d planned\n' \
  "$(awk -v d=$DONE -v n=$NTASK -v e=$NEVTOT 'BEGIN{print n?int(e*d/n):0}')" "$NEVTOT"

if (( DONE > 1 && DONE < NTASK )); then
  read -r T0 T1 < <(stat -c %Y "$OUTBASE"/task_*/.complete 2>/dev/null \
                    | sort -n | awk 'NR==1{f=$1} {l=$1} END{print f, l}')
  if [[ -n "${T0:-}" && -n "${T1:-}" && $T1 -gt $T0 ]]; then
    awk -v d=$DONE -v n=$NTASK -v t0=$T0 -v t1=$T1 'BEGIN{
      rate=(d-1)/(t1-t0); left=n-d;
      if (rate>0) printf "rate       : %.1f tasks/h -> %.1f h left (finish ~%s)\n",
        rate*3600, left/rate/3600, strftime("%Y-%m-%d %H:%M", t1+left/rate);
    }'
  fi
fi

if (( SLOW )); then
  echo "--- slow scan ---"
  du -sh "$OUTBASE" 2>/dev/null
  grep -h "fit summary" "$OUTBASE"/task_*/local.log 2>/dev/null \
    | sed 's/.*attempted=\([0-9]*\).*succeeded=\([0-9]*\).*/\1 \2/' \
    | awk '{a+=$1; s+=$2} END{if(a) printf "candidates : %d attempted, %d succeeded (%.3f%% failed)\n", a, s, 100*(a-s)/a}'
fi

BAD=$(grep -l "FATAL" "$OUTBASE"/logs/*.err 2>/dev/null | wc -l)
(( BAD > 0 )) && { echo "--- $BAD condor .err files contain FATAL; top 3: ---"
                   grep -h "FATAL" "$OUTBASE"/logs/*.err 2>/dev/null \
                     | sed 's/[0-9]\{4,\}/N/g' | sort | uniq -c | sort -rn | head -3; }
exit 0
