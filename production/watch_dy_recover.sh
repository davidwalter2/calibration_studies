#!/bin/bash
# Keep re-driving dymc_8p5M_260905 tasks killed by the `ndof == 0` abort until
# the ORIGINAL (unfixed) array 6406978 has drained.
#
# WHY A WATCHDOG AND NOT ONE RESUME.  69 elements of the original array 6406978
# were already RUNNING the unfixed .so when the fix landed and were left to
# finish (its 168 still-PENDING elements were cancelled and resubmitted from the
# fixed area instead, so only those 69 can still hit the bug). ~28 % of them
# will die, one at a time over the next hours, and a single
# `resume_dy_dev2.sh` recovers only what has already died. This loops it. It
# also re-drives anything the FIXED tail loses for an unrelated reason (a node
# eviction, a wall-clock overrun).
#
# It cannot double-submit: `resume_dy_dev2.sh` skips any index that is queued or
# running under a `dymc8p5M_a*` job name and any index with a `.complete`
# sentinel, both checked live against squeue and the filesystem.
#
# IT ALSO GIVES UP.  An unattended loop that resubmits a genuinely bad chunk
# forever is worse than a gap, so each index gets at most MAXATTEMPTS (default
# 3) submissions from this watcher; beyond that it is held out via
# `--exclude` and named in the log. The count file is
# `.watch_dy_recover.counts` -- delete it to forgive everything.
#
# usage:   nohup ./watch_dy_recover.sh > ~/dy_recover_watch.log 2>&1 &
#          ./watch_dy_recover.sh --stop        # ask the running one to exit
# options: INTERVAL (default 1800 s), MAXHOURS (default 48), MAXATTEMPTS (3)
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PIDFILE=$HERE/.watch_dy_recover.pid
STOPFILE=$HERE/.watch_dy_recover.stop
COUNTFILE=$HERE/.watch_dy_recover.counts
INTERVAL=${INTERVAL:-1800}
MAXHOURS=${MAXHOURS:-48}
MAXATTEMPTS=${MAXATTEMPTS:-3}

if [[ "${1:-}" == "--stop" ]]; then
  : > "$STOPFILE"
  echo "stop requested; the watcher exits within $INTERVAL s"
  [[ -f "$PIDFILE" ]] && echo "  (pid $(cat "$PIDFILE"))"
  exit 0
fi

rm -f "$STOPFILE"
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

DEADLINE=$(( $(date +%s) + MAXHOURS * 3600 ))
echo "[watch] started $(date -Is) pid=$$ interval=${INTERVAL}s deadline=$(date -Is -d "@$DEADLINE")"

while :; do
  [[ -f "$STOPFILE" ]] && { echo "[watch] stop file present, exiting $(date -Is)"; rm -f "$STOPFILE"; break; }
  (( $(date +%s) > DEADLINE )) && { echo "[watch] deadline reached, exiting $(date -Is)"; break; }

  # Anything left in flight? When nothing is queued or running under
  # dymc8p5M_a* AND nothing is left to recover, the job is done.
  nq=$(squeue -u "$USER" -h -r -o "%j" 2>/dev/null | grep -c '^dymc8p5M_a' || true)

  # Indices already at the attempt cap are held out of this pass.
  HELD=$(awk -v m="$MAXATTEMPTS" '$2 >= m {print $1}' "$COUNTFILE" 2>/dev/null | tr '\n' ' ')

  # Dry pass first, so the attempt counters follow what is ACTUALLY submitted.
  listing=$("$HERE/resume_dy_dev2.sh" --list-only --exclude "$HELD" 2>&1)
  idxs=$(sed -n 's/^ *indices: *//p' <<< "$listing")
  for i in $idxs; do
    n=$(awk -v k="$i" '$1 == k {print $2}' "$COUNTFILE" 2>/dev/null)
    printf '%s %s\n' "$i" "$(( ${n:-0} + 1 ))" >> "$COUNTFILE.new"
  done
  if [[ -f "$COUNTFILE.new" ]]; then
    # keep the highest count per index, then replace
    { cat "$COUNTFILE" 2>/dev/null; cat "$COUNTFILE.new"; } \
      | sort -k1,1n -k2,2nr | awk '!seen[$1]++' > "$COUNTFILE.tmp"
    mv -f "$COUNTFILE.tmp" "$COUNTFILE"; rm -f "$COUNTFILE.new"
  fi

  out=$("$HERE/resume_dy_dev2.sh" --exclude "$HELD" 2>&1)
  echo "[watch] $(date -Is) inqueue=$nq held=[${HELD% }]"
  sed 's/^/    /' <<< "$out"

  if [[ -n "${HELD// }" ]]; then
    echo "[watch] GIVEN UP after $MAXATTEMPTS attempts: ${HELD% }"
  fi

  if [[ $nq -eq 0 ]] && grep -q "nothing to recover" <<< "$out"; then
    echo "[watch] nothing in flight and nothing left to recover, exiting $(date -Is)"
    break
  fi
  sleep "$INTERVAL"
done
