#!/usr/bin/env python3
"""Task-aware input listing for CVH productions, single- OR multi-stream.

WHY THIS EXISTS. Both single- and multi-stream CVH productions live on disk.
Under ``numberOfThreads=1`` a task is ONE file, ``task_XXXX/globalcor_0.root``,
and the plain reader idiom is::

    files = sorted(glob.glob(args.files))[:args.ntasks]

with ``args.files`` naming ``globalcor_0.root`` literally. Under
``numberOfThreads=4`` a task is FOUR files,
``globalcor_0.root .. globalcor_3.root``. An event never splits across streams
and the candidate content is bit-identical to a single-thread run after
sorting on (run, lumi, event), so the four files simply CONCATENATE -- but a
reader that names stream 0 silently takes a quarter of the statistics and says
nothing.

TWO THINGS GO WRONG IF ONLY THE GLOB IS WIDENED, hence this module rather than
a `sed`:

1. ``--ntasks`` in the plain idiom caps the FILE list. Widening the glob
   without changing the cap makes ``--ntasks 160`` cover forty tasks. Here the
   cap counts TASKS (directories) and every stream of a kept task is returned.
2. An incomplete-task cleanup that deletes stream 0 only leaves streams
   1..N-1 of a TRUNCATED task on disk for a widened glob to ingest.
   Completeness is therefore decided per TASK here: a task is usable iff its
   ``.complete`` sentinel exists, every stream file it wrote is non-empty, and
   -- when the sentinel says ``streams=N`` -- N of them are present. A task
   that fails any of these is skipped WHOLE and counted.

THE RUNTREE. Every stream file carries a byte-identical copy of the 13 MB
parameter map (``runtree``). It must be read from ONE file per task and never
concatenated: `runtree_file()` returns the first existing stream file of a
task, without assuming that stream 0 is the one that exists.

API (all paths absolute, all lists in a stable, reproducible order)::

    task_dirs(base)                 -> [task dir, ...]           sorted
    stream_files(task_dir)          -> [file, ...]               stream order
    task_complete(task_dir)         -> bool
    task_reason(task_dir)           -> ""  or why it was skipped
    runtree_file(task_dir)          -> one file, or None
    iter_files(base, max_tasks)     -> yields files, task-major
    resolve(spec, max_tasks)        -> list of files  (DROP-IN for the old
                                       `sorted(glob(spec))[:ntasks]`)

``resolve`` is what the readers call. ``spec`` may be

* a glob naming a stream file -- ``.../task_*/globalcor_0.root``. The stream
  index is WIDENED (``globalcor_*.root``) and the cap counts tasks, so the
  existing command lines keep working and now see the whole production;
* a glob already naming every stream -- ``.../task_*/globalcor_*.root``;
* a production directory -- ``/ceph/.../jpsimc_20M_260906_v2``;
* a ``.txt`` file (or ``@file``) holding one input path per line, which is how
  the sharded wrappers hand a worker an explicit subset without a symlink farm.

On a single-stream production this returns exactly the plain
`sorted(glob(...))[:ntasks]` list, so existing caches reproduce bit-identically.
"""
import glob as _glob
import os
import re

__all__ = [
    "task_dirs", "stream_files", "task_complete", "task_reason",
    "runtree_file", "single_file", "iter_files", "resolve", "group_by_task",
    "last_stats",
]

# `globalcor_0.root`, `globalcor_resclosure_12.root`, `globalcor_*.root`
_STREAM_RE = re.compile(r"^(?P<stem>.+?)_(?P<idx>\d+|\*)\.root$")
_SENTINEL = ".complete"
_TASK_GLOB = "task_*"

# Filled by the last `resolve` / `iter_files` call; a caller that wants to log
# what was dropped reads this instead of threading counters through.
_STATS = {}


def last_stats():
    """Counters from the most recent resolve()/iter_files() call."""
    return dict(_STATS)


# ---------------------------------------------------------------------------
def split_stream(path):
    """('globalcor', 3) for `.../globalcor_3.root`; (basename, None) if not a
    stream-numbered name."""
    b = os.path.basename(path)
    m = _STREAM_RE.match(b)
    if not m:
        return b, None
    idx = m.group("idx")
    return m.group("stem"), (None if idx == "*" else int(idx))


def stream_pattern(stem):
    """The per-task glob that matches every stream of `stem`."""
    return f"{stem}_*.root" if stem else "*.root"


def _stem_of_pattern(pattern):
    """The stream stem implied by a user glob, or None if it names no stream.

    `.../task_*/globalcor_0.root` -> 'globalcor'   (widened to every stream)
    `.../task_*/globalcor_*.root` -> 'globalcor'
    `.../task_*/*.root`           -> None          (taken literally)
    """
    stem, _ = split_stream(pattern)
    return stem if stem != os.path.basename(pattern) else None


# ---------------------------------------------------------------------------
def stream_files(task_dir, stem=None, basename=None):
    """This task's stream files, ordered by stream index (numerically).

    Lexical order is wrong from ten streams on (`_10` before `_2`); the index
    is parsed so the order is the stream order at any thread count. Names that
    carry no index sort last, by name.
    """
    pat = basename or stream_pattern(stem)
    fs = _glob.glob(os.path.join(task_dir, pat))
    if stem and not basename:
        # `globalcor_*.root` also matches `globalcor_resclosure_0.root`, so the
        # stem is required to match EXACTLY: a directory holding both makers'
        # output must not have the two trees concatenated into one list.
        fs = [f for f in fs if split_stream(f)[0] == stem]
    return sorted(fs, key=lambda p: (split_stream(p)[1] is None,
                                     split_stream(p)[1] or 0,
                                     os.path.basename(p)))


def declared_streams(task_dir):
    """`streams=N` from the `.complete` sentinel, or None if it does not say.

    Multi-stream makers write it; older sentinels are empty files. When it is
    there it is the authority on how many stream files the task SHOULD have,
    which catches a task whose sentinel landed before a straggler stream was
    copied.
    """
    p = os.path.join(task_dir, _SENTINEL)
    try:
        with open(p, "r") as fh:
            txt = fh.read(4096)
    except OSError:
        return None
    m = re.search(r"^\s*streams\s*=\s*(\d+)\s*$", txt, re.M)
    return int(m.group(1)) if m else None


def task_reason(task_dir, stem=None, basename=None, require_complete=True):
    """"" if the task is usable, else a short reason why it is not."""
    fs = stream_files(task_dir, stem, basename)
    if not fs:
        return "no-stream-files"
    if require_complete and not os.path.exists(os.path.join(task_dir, _SENTINEL)):
        return "no-.complete"
    if any(os.path.getsize(f) == 0 for f in fs):
        return "empty-stream-file"
    n = declared_streams(task_dir) if require_complete else None
    if n is not None and len(fs) != n:
        return f"missing-streams({len(fs)}/{n})"
    return ""


def task_complete(task_dir, stem=None, basename=None, require_complete=True):
    """True iff every stream file this task wrote is present and non-empty."""
    return not task_reason(task_dir, stem, basename, require_complete)


def runtree_file(task_dir, stem=None, basename=None):
    """ONE file of this task, for the `runtree` parameter map.

    Every stream carries a byte-identical copy, so any of them will do -- but
    stream 0 is not guaranteed to be the one on disk (a task may have been
    produced at a different thread count, or its stream 0 cleaned away), so
    this returns the FIRST EXISTING stream rather than assuming an index.
    """
    fs = stream_files(task_dir, stem, basename)
    return fs[0] if fs else None


def single_file(task_dir, stem=None, basename=None, warn=None):
    """The ONE stream file of a task that was produced single-threaded.

    For a diagnostic that compares two trees file by file, and whose producer
    pins `numberOfThreads=1`, one file IS the task. This returns it without
    naming stream 0 (a task produced at another thread count would have no
    `globalcor_0.root`), and shouts if the task turns out to have several
    streams -- which is the point: such a diagnostic would silently be reading
    1/N of the candidates, and it should say so rather than quietly halve its
    own conclusion.
    """
    fs = stream_files(task_dir, stem, basename)
    if not fs:
        return os.path.join(task_dir, f"{stem or 'globalcor'}_0.root")
    if len(fs) > 1:
        msg = (f"prodfiles: {task_dir} has {len(fs)} stream files; this reader "
               f"takes only {os.path.basename(fs[0])}, i.e. 1/{len(fs)} of its "
               f"candidates. Re-run it against a single-threaded output, or "
               f"teach it prodfiles.resolve()")
        if warn is None:
            import warnings
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        else:
            warn(msg)
    return fs[0]


# ---------------------------------------------------------------------------
def task_dirs(base, stem=None, basename=None):
    """The task directories under `base`, sorted.

    `base` may be a production directory (holding `task_*/`), a glob of task
    directories, or a flat directory of stream files -- the last is treated as
    a single anonymous task, which is what makes ad-hoc smoke outputs and
    staged shard directories work unchanged.
    """
    base = os.path.abspath(os.path.expanduser(base))
    if os.path.isdir(base):
        ds = sorted(d for d in _glob.glob(os.path.join(base, _TASK_GLOB))
                    if os.path.isdir(d))
        if ds:
            return ds
        return [base] if stream_files(base, stem, basename) else []
    return sorted(d for d in _glob.glob(base) if os.path.isdir(d))


def group_by_task(files):
    """{task dir: [files]} in first-seen task order, streams in stream order."""
    out = {}
    for f in files:
        out.setdefault(os.path.dirname(os.path.abspath(f)), []).append(f)
    for d in out:
        out[d] = sorted(out[d], key=lambda p: (split_stream(p)[1] is None,
                                               split_stream(p)[1] or 0,
                                               os.path.basename(p)))
    return out


# ---------------------------------------------------------------------------
def autostem(dirs, logger=None):
    """The single stream stem present under `dirs`.

    A directory spec (`--files /ceph/.../task_0007`) does not say WHICH kind of
    output to read, and a production carries exactly one: `globalcor` for the
    two-track maker, `globalcor_resclosure` for the single-track one. Reading
    both from one directory would concatenate two different trees, so a
    directory holding more than one stem is an error, not a guess.
    """
    stems = set()
    for d in dirs:
        for f in _glob.glob(os.path.join(d, "*.root")):
            stems.add(split_stream(f)[0])
    if len(stems) > 1:
        raise SystemExit(
            "prodfiles: more than one output stem under these task "
            f"directories ({', '.join(sorted(stems))}) -- name one with a glob "
            "such as .../task_*/<stem>_*.root instead of a directory")
    if not stems:
        return "globalcor"
    stem = stems.pop()
    if logger is not None:
        logger(f"prodfiles: stem {stem}_*.root")
    return stem


def _select(dirs, stem, basename, max_tasks, require_complete, logger):
    """Common task filter: keep usable tasks, cap on TASKS, count the rest."""
    if require_complete == "auto":
        # A directory tree that has no sentinels at all is not a production
        # with truncated tasks -- it is a hand-made smoke output or a staged
        # shard directory, and requiring a sentinel there would return nothing
        # at all. Requiring it is the default the moment ONE task has one.
        require_complete = any(
            os.path.exists(os.path.join(d, _SENTINEL)) for d in dirs)
        if dirs and not require_complete and logger is not None:
            logger(f"prodfiles: no {_SENTINEL} anywhere under "
                   f"{os.path.dirname(dirs[0]) or dirs[0]} -- taking every "
                   f"task (this is normal for a smoke or staged directory, "
                   f"and WRONG for a production still being written)")

    files, kept, skipped, reasons = [], 0, 0, {}
    for d in dirs:
        why = task_reason(d, stem, basename, require_complete)
        if why:
            skipped += 1
            reasons[why] = reasons.get(why, 0) + 1
            continue
        if max_tasks and kept >= max_tasks:
            break
        files.extend(stream_files(d, stem, basename))
        kept += 1

    _STATS.clear()
    _STATS.update(ntasks_found=len(dirs), ntasks_used=kept,
                  ntasks_skipped=skipped, nfiles=len(files),
                  streams_per_task=(len(files) / kept if kept else 0),
                  require_complete=bool(require_complete), reasons=reasons)
    if logger is not None:
        msg = (f"prodfiles: {kept} tasks / {len(files)} files"
               + (f" ({len(files)/kept:.3g} streams per task)" if kept else ""))
        if skipped:
            msg += (f"; skipped {skipped} unusable task(s): "
                    + ", ".join(f"{k} x{v}" for k, v in sorted(reasons.items())))
        logger(msg)
    return files


def iter_files(base, max_tasks=None, stem=None, basename=None,
               require_complete="auto", logger=None):
    """Every stream file of the first `max_tasks` usable tasks under `base`."""
    dirs = task_dirs(base, stem, basename)
    if stem is None and basename is None:
        stem = autostem(dirs, logger)
    return _select(dirs, stem, basename, max_tasks, require_complete, logger)


def _read_list(path):
    with open(path) as fh:
        return [ln.strip() for ln in fh
                if ln.strip() and not ln.lstrip().startswith("#")]


def resolve(spec, max_tasks=None, require_complete="auto", logger=None,
            widen=True):
    """Drop-in for `sorted(glob.glob(spec))[:ntasks]`, but task-aware.

    `max_tasks` caps TASKS, not files; 0 / None means no cap. See the module
    docstring for what `spec` may be. `widen=False` keeps a pattern that names
    one stream literally -- no caller passes it; it is the escape hatch for a
    deliberate single-stream read.
    """
    if isinstance(spec, (list, tuple)):
        out = []
        for s in spec:
            out.extend(resolve(s, None, require_complete, logger, widen))
        if max_tasks:
            groups = group_by_task(out)
            out = [f for d in list(groups)[:max_tasks] for f in groups[d]]
        return out

    spec = os.path.expanduser(str(spec))

    # (a) an explicit list of inputs -- what the sharded wrappers pass
    if spec.startswith("@") or (spec.endswith(".txt") and os.path.isfile(spec)):
        listed = _read_list(spec[1:] if spec.startswith("@") else spec)
        groups = group_by_task(listed)
        if require_complete == "auto":
            require_complete = any(
                os.path.exists(os.path.join(d, _SENTINEL)) for d in groups)
        files, kept, skipped = [], 0, 0
        for d, fs in groups.items():
            if require_complete and not os.path.exists(os.path.join(d, _SENTINEL)):
                skipped += 1
                continue
            if any(os.path.getsize(f) == 0 for f in fs if os.path.exists(f)):
                skipped += 1
                continue
            if max_tasks and kept >= max_tasks:
                break
            files.extend(fs)
            kept += 1
        _STATS.clear()
        _STATS.update(ntasks_found=len(groups), ntasks_used=kept,
                      ntasks_skipped=skipped, nfiles=len(files),
                      require_complete=bool(require_complete), reasons={})
        if logger is not None:
            logger(f"prodfiles: {kept} tasks / {len(files)} files from list {spec}"
                   + (f"; skipped {skipped} unusable" if skipped else ""))
        return files

    # (b) a production / task directory
    if os.path.isdir(spec):
        return iter_files(spec, max_tasks, logger=logger,
                          require_complete=require_complete)

    # (c) a glob naming stream files
    stem = _stem_of_pattern(spec) if widen else None
    basename = None if stem else os.path.basename(spec)
    parent = os.path.dirname(spec)
    dirs = {os.path.dirname(os.path.abspath(f)) for f in _glob.glob(spec)}
    if parent:
        # Also take the task directories the PARENT glob names, not only the
        # ones where the pattern matched: a task produced at another thread
        # count, or one whose stream 0 a cleanup removed, has no
        # `globalcor_0.root` but is otherwise perfectly good input. Directories
        # holding nothing are dropped by `task_reason` below and counted.
        dirs |= {os.path.abspath(d) for d in _glob.glob(parent)
                 if os.path.isdir(d)}
    return _select(sorted(dirs), stem, basename, max_tasks, require_complete,
                   logger)


# ---------------------------------------------------------------------------
def _cli():
    import argparse
    import sys
    p = argparse.ArgumentParser(
        description="list the usable input files of a CVH production")
    p.add_argument("spec", help="glob, directory, or @list.txt")
    p.add_argument("--ntasks", type=int, default=0, help="cap on TASKS (0 = all)")
    p.add_argument("--tasks", action="store_true", help="print task dirs")
    p.add_argument("--runtrees", action="store_true",
                   help="print one file per task (the runtree copy)")
    p.add_argument("--stats", action="store_true", help="print counters only")
    p.add_argument("--require-complete", choices=["auto", "yes", "no"],
                   default="auto")
    a = p.parse_args()
    rc = {"auto": "auto", "yes": True, "no": False}[a.require_complete]
    log = (lambda m: print(m, file=sys.stderr))
    files = resolve(a.spec, a.ntasks, require_complete=rc, logger=log)
    if a.stats:
        for k, v in sorted(last_stats().items()):
            print(f"{k}={v}")
        return
    if a.tasks:
        seen = []
        for f in files:
            d = os.path.dirname(f)
            if d not in seen:
                seen.append(d)
        print("\n".join(seen))
        return
    if a.runtrees:
        seen = {}
        for f in files:
            seen.setdefault(os.path.dirname(f), f)
        print("\n".join(seen.values()))
        return
    print("\n".join(files))


if __name__ == "__main__":
    _cli()
