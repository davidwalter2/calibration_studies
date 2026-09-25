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
NTASK=${NTASKS_USED:-$NTASK}
NEVTOT=$(head -n "$NTASK" "$CHUNKLIST" 2>/dev/null | awk '{s+=$3} END{print s+0}')
DONE=$(ls -d "$OUTBASE"/task_*/.complete 2>/dev/null | wc -l)
STARTED=$(ls -d "$OUTBASE"/task_*/ 2>/dev/null | wc -l)

# Count THIS production's jobs only. `condor_q $USER` would mix in every other
# cluster -- with the DY and J/psi v2 legs in the queue at the same time that
# is not a cosmetic error, it is two productions' numbers added together. The
# submit file advertises +CvhOutBase for exactly this.
read -r RUN IDLE HELD < <(condor_q -constraint "CvhOutBase == \"$OUTBASE\"" \
  -af JobStatus 2>/dev/null | awk '{ if ($1==2) r++; else if ($1==1) i++; else if ($1==5) h++ }
                                   END { printf "%d %d %d\n", r+0, i+0, h+0 }')
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
  # STREAM FILES: one `globalcor_<stream>.root` per EDM stream. Anything but a
  # single mode of $NTHREADS means some tasks lost streams, which is invisible
  # downstream because every stream file is individually valid and carries its
  # own copy of the runtree.
  for d in "$OUTBASE"/task_*/; do
    [[ -f "$d/.complete" ]] || continue
    printf '%d %d\n' "$(ls "$d"globalcor_*.root 2>/dev/null | wc -l)" \
                      "$(find "$d" -name 'globalcor_*.root' -empty 2>/dev/null | wc -l)"
  done | sort | uniq -c | awk -v want="$NTHREADS" '{
      printf "streams/task : %d task(s) with %s file(s)%s%s\n", $1, $2,
             ($2 == want ? "" : "  <-- EXPECTED " want),
             ($3 > 0 ? "  -- " $3 " EMPTY" : "") }'

  grep -h "fit summary" "$OUTBASE"/task_*/local.log 2>/dev/null \
    | sed 's/.*attempted=\([0-9]*\).*succeeded=\([0-9]*\).*/\1 \2/' \
    | awk '{a+=$1; s+=$2} END{if(a) printf "candidates : %d attempted, %d succeeded (%.3f%% failed)\n", a, s, 100*(a-s)/a}'
fi

BAD=$(grep -l "FATAL" "$OUTBASE"/logs/*.err 2>/dev/null | wc -l)
(( BAD > 0 )) && { echo "--- $BAD condor .err files contain FATAL; top 3: ---"
                   grep -h "FATAL" "$OUTBASE"/logs/*.err 2>/dev/null \
                     | sed 's/[0-9]\{4,\}/N/g' | sort | uniq -c | sort -rn | head -3; }
exit 0
