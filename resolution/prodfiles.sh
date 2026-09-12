#!/bin/bash
# Task-aware input listing for CVH productions, for shell drivers.
#
# The shell twin of `prodfiles.py`, and a THIN one on purpose: every function
# here delegates to that module, so there is exactly one definition of "which
# files of this production may be read" and the shell drivers cannot drift
# from the python readers. Only the stdlib is used, so any python3 will do --
# no venv is required to list files.
#
#   source /work/submit/.../calibration_studies/resolution/prodfiles.sh
#
#   pf_files      SPEC [MAXTASKS]   # every stream file of the usable tasks
#   pf_task_dirs  SPEC [MAXTASKS]   # the usable task directories
#   pf_runtrees   SPEC [MAXTASKS]   # ONE file per task (the runtree copy)
#   pf_stream_files TASKDIR [STEM]  # that task's streams, in stream order
#   pf_task_complete TASKDIR [STEM] # exit 0 if usable, 1 otherwise
#   pf_clean_incomplete BASE [STEM] # delete the payload of unusable tasks
#   pf_stats      SPEC [MAXTASKS]   # ntasks_used / nfiles / ... as k=v lines
#
# SPEC is what the python `--files` takes: a glob (`.../task_*/globalcor_0.root`
# -- the stream index is widened), a production directory, or an `@list.txt`.
# MAXTASKS caps TASKS, not files; 0 or unset means all of them.
#
# WHY A TASK AND NOT A FILE. Under `numberOfThreads=4` a task is
# `globalcor_0.root .. globalcor_3.root`. `ls task_*/globalcor_0.root` then
# takes a quarter of the statistics, and an incomplete-task cleanup that
# removes stream 0 only leaves streams 1-3 of a TRUNCATED task for the next
# widened glob to swallow. Both are decided per task here.

PF_PY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/prodfiles.py"
PF_PYTHON=${PF_PYTHON:-python3}

pf_files()      { "$PF_PYTHON" "$PF_PY" "$1" --ntasks "${2:-0}"; }
pf_task_dirs()  { "$PF_PYTHON" "$PF_PY" "$1" --ntasks "${2:-0}" --tasks; }
pf_runtrees()   { "$PF_PYTHON" "$PF_PY" "$1" --ntasks "${2:-0}" --runtrees; }
pf_stats()      { "$PF_PYTHON" "$PF_PY" "$1" --ntasks "${2:-0}" --stats; }

pf_stream_files() {
  "$PF_PYTHON" -c '
import os, sys
sys.path.insert(0, os.path.dirname(sys.argv[1]))
import prodfiles as P
print("\n".join(P.stream_files(sys.argv[2], sys.argv[3])))' \
    "$PF_PY" "$1" "${2:-globalcor}"
}

pf_task_complete() {
  "$PF_PYTHON" -c '
import os, sys
sys.path.insert(0, os.path.dirname(sys.argv[1]))
import prodfiles as P
sys.exit(0 if P.task_complete(sys.argv[2], sys.argv[3]) else 1)' \
    "$PF_PY" "$1" "${2:-globalcor}"
}

# Delete the ROOT payload of every unusable task under BASE (no `.complete`,
# an empty stream file, or fewer streams than the sentinel declares). It takes
# out EVERY stream, which is the whole point: removing stream 0 alone is what
# leaves a truncated task readable by a widened glob.
#
# A `.complete` that survives its own payload is removed with it. The sentinel
# is the only thing `resume_*.sh` consults, so a task whose sentinel landed
# before a stream did (or whose stream came out empty) would otherwise be
# skipped by every future resume and be missing for good; without the sentinel
# it is simply re-driven. The task DIRECTORY stays, so the production
# bookkeeping still sees the index.
pf_clean_incomplete() {
  "$PF_PYTHON" -c '
import os, sys
sys.path.insert(0, os.path.dirname(sys.argv[1]))
import prodfiles as P
base, stem = sys.argv[2], sys.argv[3]
n = nf = 0
for d in P.task_dirs(base, stem):
    fs = P.stream_files(d, stem)
    if not fs or P.task_complete(d, stem):
        continue
    print(f"dropping incomplete {d} ({P.task_reason(d, stem)}): "
          f"{len(fs)} stream file(s)")
    for f in fs:
        os.remove(f)
        nf += 1
    sen = os.path.join(d, ".complete")
    if os.path.exists(sen):
        os.remove(sen)
        print(f"    and its .complete, so a resume re-drives the task")
    n += 1
print(f"[pf_clean_incomplete] {n} task(s), {nf} file(s) removed under {base}")' \
    "$PF_PY" "$1" "${2:-globalcor}"
}
