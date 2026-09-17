#!/bin/bash
# Stamp `.complete` in the task directories of a finished SLURM ARRAY.
#
# `slurm/array.sbatch` does not write the marker `run_prod_bs.sh` does, so an
# extraction that globs a production while it is still running can read a
# half-written ROOT file.  `sacct` knows which array tasks reached COMPLETED;
# this turns that into the marker `extract_vtx.py --require-complete` wants.
#
#   usage: ./mark_complete.sh <jobid> <outdir>
set -uo pipefail
JOB=${1:?jobid}
OUT=${2:?outdir}
n=0
while read -r id state; do
  [[ "$state" == "COMPLETED" ]] || continue
  idx=${id##*_}
  [[ "$idx" =~ ^[0-9]+$ ]] || continue
  d="$OUT/task_$(printf '%04d' "$idx")"
  if [[ -d "$d" ]] && compgen -G "$d/globalcor_*.root" > /dev/null; then
    touch "$d/.complete"; n=$((n+1))
  fi
done < <(sacct -j "$JOB" -X -n -P -o JobID,State | tr '|' ' ')
echo "$n task(s) marked complete under $OUT"
ls -d "$OUT"/task_*/ 2>/dev/null | wc -l | xargs echo "task dirs:"
