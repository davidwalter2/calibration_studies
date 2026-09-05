#!/bin/bash
# Poor-man's profiler: sample stacks of a running cmsRun with gdb.
# No perf on the submit nodes, and callgrind is ~100x too slow for a fit that
# is already seconds per candidate, so sample instead.
# usage: pmp_sample.sh <pattern-in-cmdline> <nsamples> <outfile> [sleep_s]
# NOTE: resolve the pid from /proc/<pid>/cmdline of an actual `cmsRun` exe --
# `pgrep -f <pat>` also matches the shell running this very command.
set -uo pipefail
PAT=$1; N=$2; OUT=$3; SLP=${4:-2}
PID=""
for p in $(pgrep -u "$USER" -x cmsRun); do
  if tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q -- "$PAT"; then PID=$p; break; fi
done
[[ -n "$PID" ]] || { echo "no cmsRun matching '$PAT'"; exit 1; }
echo "sampling pid $PID"
: > "$OUT"
for i in $(seq 1 "$N"); do
  kill -0 "$PID" 2>/dev/null || break
  gdb -q -p "$PID" -batch -ex "set auto-solib-add off" -ex "set pagination off" -ex "bt 40" 2>/dev/null >> "$OUT" || true
  echo "=== SAMPLE $i ===" >> "$OUT"
  sleep "$SLP"
done
echo "done $OUT"
