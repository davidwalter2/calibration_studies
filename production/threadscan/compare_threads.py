#!/usr/bin/env python3
"""Correctness gate for the multithreaded CVH refit.

`root_bitcompare.py` hashes whole branches and so requires the two files to be
in the same ORDER.  A multithreaded run is not: `numberOfStreams` follows
`numberOfThreads`, the maker writes one file per stream
(`<outprefix>_<stream>.root`), and TBB hands events to whichever stream is
free.  The comparison therefore has to be per candidate, on a key, after a
permutation.

THE KEY IS `(run, lumi, event)` WITH A **STABLE** SORT, and that is exact, not
an approximation: an event is never split across streams, so all candidates of
one event are written consecutively by ONE stream in the maker's own order.  A
stable sort of the concatenated streams therefore reproduces the single-thread
sequence exactly, including the intra-event order that no branch identifies.
Both sides are sorted with the same key, so a non-monotonic input file is
handled too.

usage: compare_threads.py <ref_dir> <test_dir> [--tol 0]
       compare_threads.py .../t1 .../t8
"""
import argparse
import glob
import os
import sys

import numpy as np
import uproot


def load(dirpath, tree="tree", per_stream=False):
    """Concatenate every stream file of a run, in stream order."""
    files = sorted(glob.glob(os.path.join(dirpath, "globalcor_*.root")),
                   key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
    if not files:
        raise SystemExit(f"no globalcor_*.root in {dirpath}")
    cols, nent = {}, 0
    for fn in files:
        with uproot.open(fn) as f:
            t = f[tree]
            n = t.num_entries
            arrs = {b: t[b].array(library="np") for b in t.keys()}
            for b, a in arrs.items():
                cols.setdefault(b, []).append(a)
            nent += n
    out = {}
    for b, chunks in cols.items():
        try:
            out[b] = np.concatenate(chunks)
        except Exception:
            out[b] = np.concatenate([np.asarray(c, dtype=object) for c in chunks])
    return out, nent, files


def order(cols):
    # A tree without the event key (the `runtree`, which carries the global
    # parameter map and not candidates) has nothing to permute: every stream
    # writes the same rows in the same order, so compare it as-is.
    if not {"run", "lumi", "event"} <= set(cols):
        n = len(next(iter(cols.values())))
        return np.arange(n)
    key = np.stack([np.asarray(cols["run"], dtype=np.int64),
                    np.asarray(cols["lumi"], dtype=np.int64),
                    np.asarray(cols["event"], dtype=np.int64)])
    # lexsort is stable and takes the LAST row as primary key
    return np.lexsort((np.arange(key.shape[1]), key[2], key[1], key[0]))


def diff_entry(a, b):
    """Return (n_differing_elements, max_abs, max_rel) for one entry pair."""
    a = np.atleast_1d(np.asarray(a))
    b = np.atleast_1d(np.asarray(b))
    if a.shape != b.shape:
        return (max(a.size, b.size), np.inf, np.inf)
    if a.dtype.kind in "fc":
        d = ~(a == b) & ~(np.isnan(a) & np.isnan(b))
        if not d.any():
            return (0, 0.0, 0.0)
        da = np.abs(a[d].astype(np.float64) - b[d].astype(np.float64))
        scale = np.maximum(np.abs(a[d].astype(np.float64)),
                           np.abs(b[d].astype(np.float64)))
        with np.errstate(divide="ignore", invalid="ignore"):
            dr = np.where(scale > 0, da / scale, 0.0)
        return (int(d.sum()), float(np.max(da)), float(np.nanmax(dr)))
    d = a != b
    if not d.any():
        return (0, 0.0, 0.0)
    return (int(d.sum()), float(np.max(np.abs(a[d].astype(np.float64)
                                              - b[d].astype(np.float64)))), np.inf)


def compare_runtrees(refdir, testdir):
    """Every stream file's runtree must equal the reference run's runtree."""
    import hashlib

    def sig(fn):
        with uproot.open(fn) as f:
            t = f["runtree"]
            h = hashlib.sha256()
            for b in sorted(t.keys()):
                a = t[b].array(library="np")
                if a.dtype == object:
                    for v in a:
                        h.update(np.ascontiguousarray(np.asarray(v)).view(np.uint8).tobytes())
                else:
                    h.update(np.ascontiguousarray(a).view(np.uint8).tobytes())
            return h.hexdigest(), t.num_entries

    ref = sorted(glob.glob(os.path.join(refdir, "globalcor_*.root")))[0]
    rh, rn = sig(ref)
    ok = True
    for fn in sorted(glob.glob(os.path.join(testdir, "globalcor_*.root"))):
        th, tn = sig(fn)
        tag = "same" if (th, tn) == (rh, rn) else "DIFFERS"
        print(f"runtree {os.path.basename(fn)}: {tn} entries  {tag}")
        ok &= (th, tn) == (rh, rn)
    print(f"runtree: reference {rn} entries; per-stream copies "
          f"{'all identical' if ok else 'NOT identical'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--tree", default="tree")
    ap.add_argument("--also-runtree", action="store_true",
                    help="additionally compare the runtree (global parameter "
                         "map). With N streams the runtree is written N times; "
                         "this checks the concatenation, so N copies of an "
                         "identical map fail on the entry count -- which is "
                         "the point: a reader must take ONE stream's runtree.")
    ap.add_argument("--maxreport", type=int, default=40)
    args = ap.parse_args()

    if args.also_runtree:
        rc = compare_runtrees(args.ref, args.test)
    else:
        rc = 0
    A, na, fa = load(args.ref, args.tree)
    B, nb, fb = load(args.test, args.tree)
    print(f"ref  {args.ref}: {len(fa)} stream file(s), {na} entries")
    print(f"test {args.test}: {len(fb)} stream file(s), {nb} entries")
    if na != nb:
        print(f"FAIL: entry count differs ({na} vs {nb})")
        # still report which events are missing
        ka = set(zip(A['run'], A['lumi'], A['event']))
        kb = set(zip(B['run'], B['lumi'], B['event']))
        print(f"  events only in ref : {len(ka - kb)}  only in test: {len(kb - ka)}")
        return 1

    ia, ib = order(A), order(B)
    ka = np.stack([A['run'][ia], A['lumi'][ia], A['event'][ia]])
    kb = np.stack([B['run'][ib], B['lumi'][ib], B['event'][ib]])
    if not np.array_equal(ka, kb):
        n = int((ka != kb).any(axis=0).sum())
        print(f"FAIL: (run,lumi,event) key sequence differs in {n} places")
        return 1
    print(f"key aligned on {na} candidates; comparing {len(set(A) & set(B))} branches")

    names = sorted(set(A) & set(B))
    only = (set(A) ^ set(B))
    if only:
        print(f"BRANCH SET DIFFERS: {sorted(only)}")

    bad = []
    for b in names:
        a, c = A[b][ia], B[b][ib]
        if a.dtype != object and c.dtype != object:
            nd, ma, mr = diff_entry(a, c)
        else:
            nd = ma = mr = 0
            for x, y in zip(a, c):
                n1, m1, r1 = diff_entry(x, y)
                nd += n1
                ma = max(ma, m1)
                mr = max(mr, r1)
        if nd:
            bad.append((b, nd, ma, mr))
    if not bad:
        print(f"PASS: all {len(names)} branches bit-identical over {na} candidates")
        return rc
    print(f"FAIL: {len(bad)} of {len(names)} branches differ")
    print(f"{'branch':<28} {'n_elem_diff':>12} {'max_abs':>14} {'max_rel':>12}")
    for b, nd, ma, mr in sorted(bad, key=lambda t: -t[3] if np.isfinite(t[3]) else -1e99)[:args.maxreport]:
        print(f"{b:<28} {nd:>12} {ma:>14.6g} {mr:>12.4g}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
