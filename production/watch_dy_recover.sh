#!/bin/bash
# Keep re-driving dymc_8p5M_260905 tasks killed by the `ndof == 0` abort until
# the ORIGINAL (unfixed) array 6406978 has drained.
#
# WHY A WATCHDOG AND NOT ONE RESUME.  The original array still had ~240 chunks
# queued when the fix landed, every one of them running the unfixed .so out of
# CMSSW_15_0_19_patch2_dev, and ~28 % of them will die the same way. They fail
# one at a time over the next day and a half, so a single `resume_dy_dev2.sh`
# recovers only what had already died. This loops it.
#
# It cannot double-submit: `resume_dy_dev2.sh` skips any index that is queued or
# running under a `dymc8p5M_a*` job name, and any index with a `.complete`
# sentinel. It only ever picks up a directory that EXISTS (the task was tried)
# and has no sentinel.
#
# usage:   nohup ./watch_dy_recover.sh > ~/dy_recover_watch.log 2>&1 &
#          ./watch_dy_recover.sh --stop        # ask the running one to exit
# options: INTERVAL (default 1800 s), MAXHOURS (default 48)
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PIDFILE=$HERE/.watch_dy_recover.pid
STOPFILE=$HERE/.watch_dy_recover.stop
INTERVAL=${INTERVAL:-1800}
MAXHOURS=${MAXHOURS:-48}

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

  # Anything of the ORIGINAL array left? When nothing is queued or running under
  # dymc8p5M_a* AND nothing is left to recover, the job is done.
  nq=$(squeue -u "$USER" -h -r -o "%j" 2>/dev/null | grep -c '^dymc8p5M_a' || true)
  out=$("$HERE/resume_dy_dev2.sh" 2>&1)
  echo "[watch] $(date -Is) inqueue=$nq"
  sed 's/^/    /' <<< "$out"

  if [[ $nq -eq 0 ]] && grep -q "nothing to recover" <<< "$out"; then
    echo "[watch] array drained and nothing left to recover, exiting $(date -Is)"
    break
  fi
  sleep "$INTERVAL"
done
