#!/bin/bash
# Progress of the 20M J/psi MC CVH production.
#
# Completion is counted from the .complete SENTINEL, never from the .root:
# cmsRun creates its output at START, so a killed task leaves a non-empty but
# TRUNCATED file that an `-s` test would accept forever.
#
# usage: ./status.sh [--outbase DIR] [--chunks FILE] [--slow]
#   --slow  also add up the output bytes and the candidate counts (walks every
#           task dir; minutes on ceph, so it is not the default)
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TAG=jpsimc_20M_260905
OUTBASE=/ceph/submit/data/user/d/david_w/ZMass/cvh/$TAG
CHUNKS=$HERE/chunks_${TAG}.txt
SLOW=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --outbase) OUTBASE=$2; shift 2;;
    --chunks)  CHUNKS=$2; shift 2;;
    --slow)    SLOW=1; shift;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

NTASK=$(grep -cvE '^\s*(#|$)' "$CHUNKS" 2>/dev/null || echo 0)
NEVTOT=$(awk '{s+=$3} END{print s+0}' "$CHUNKS" 2>/dev/null)

DONE=$(ls -d "$OUTBASE"/task_*/.complete 2>/dev/null | wc -l)
STARTED=$(ls -d "$OUTBASE"/task_*/ 2>/dev/null | wc -l)
NAMES=jpsimc20M_a0,jpsimc20M_a1,jpsimc20M_a2
RUN=$(squeue -u "$USER" -h -r -n "$NAMES" -t R 2>/dev/null | wc -l)
PEND=$(squeue -u "$USER" -h -r -n "$NAMES" -t PD 2>/dev/null | wc -l)
# A task dir with no sentinel and no queue entry is a failure that already
# finished; that is the number resume.sh will pick up.
STALE=$(( STARTED - DONE - RUN ))
(( STALE < 0 )) && STALE=0

printf 'tag        : %s\n' "$TAG"
printf 'outbase    : %s\n' "$OUTBASE"
printf 'tasks      : %d total\n' "$NTASK"
printf 'complete   : %d (%.1f%%)\n' "$DONE" "$(awk -v d=$DONE -v n=$NTASK 'BEGIN{print n?100*d/n:0}')"
printf 'running    : %d   pending: %d\n' "$RUN" "$PEND"
printf 'started-not-complete (failed or in flight): %d\n' "$STALE"
printf 'events     : %d done of %d planned\n' \
  "$(awk -v d=$DONE -v n=$NTASK -v e=$NEVTOT 'BEGIN{print n?int(e*d/n):0}')" "$NEVTOT"

if (( DONE > 0 && DONE < NTASK )); then
  # Rate from the sentinel mtimes: first to newest. Only meaningful once a
  # few hundred have landed, so it is reported as a range, not a promise.
  # oldest and newest sentinel mtime, in one pass
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

# Failing tasks are worth seeing early: a systematic failure (bad conditions,
# ceph eviction) looks exactly like a slow start otherwise.
BAD=$(grep -l "FATAL" "$OUTBASE"/logs/*.err 2>/dev/null | wc -l)
(( BAD > 0 )) && { echo "--- $BAD slurm .err files contain FATAL; most recent 3: ---"
                   grep -h "FATAL" $(ls -t "$OUTBASE"/logs/*.err 2>/dev/null | head -20) 2>/dev/null | sort | uniq -c | sort -rn | head -3; }
exit 0
