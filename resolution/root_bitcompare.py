#!/usr/bin/env python3
"""Branch-by-branch BIT comparison of two ROOT files.

ROOT file bytes are not comparable directly (headers carry a timestamp and a
UUID, and compression is not guaranteed reproducible), so the comparison is on
the DECODED branch contents: every branch of every tree is read and its raw
buffer hashed. Any difference in any value, of any type, shows up.

    python root_bitcompare.py a.root b.root [--trees tree runtree]

Exit code 0 iff every tree, every branch and every entry is bit-identical.
"""
import argparse
import hashlib
import re
import sys

import numpy as np
import uproot


def _hash(obj, h):
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            for v in obj:
                _hash(np.asarray(v), h)
        else:
            h.update(np.ascontiguousarray(obj).view(np.uint8).tobytes())
    else:
        try:
            arr = np.asarray(obj)
        except Exception:
            h.update(repr(obj).encode())
            return
        _hash(arr, h)


def branch_hashes(fn, trees):
    out = {}
    with uproot.open(fn) as f:
        # recurse into TDirectories: the cleanprop export writes
        # propExport/legs, so a top-level-only scan compares nothing and
        # "passes" on a single branch.
        keys = [k.split(";")[0] for k in f.keys(recursive=True)]
        for tn in sorted(set(keys)):
            if trees and tn not in trees:
                continue
            try:
                t = f[tn]
                names = t.keys()
            except Exception:
                continue
            for b in sorted(names):
                try:
                    a = t[b].array(library="np")
                except Exception as e:
                    out[(tn, b)] = f"UNREADABLE:{type(e).__name__}"
                    continue
                h = hashlib.sha256()
                _hash(a, h)
                out[(tn, b)] = h.hexdigest()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--trees", nargs="*", default=None)
    ap.add_argument("--exclude", nargs="*", default=None, metavar="REGEX",
                    help="branch-name regexes to drop from BOTH sides before "
                         "comparing. The use case is proving that a storage "
                         "switch is inert: a run with exportStepRecords=False "
                         "is missing exactly the step-record branches, so "
                         "without this every such comparison reports "
                         "DIFFERENT on the schema alone and never says "
                         "anything about the branches that were kept. "
                         "Matched with re.search on the branch name (not the "
                         "tree name), so '^ioniurban' and 'ioniurban' both "
                         "work. Excluded branches are listed in the output "
                         "so a pass can never hide what was skipped.")
    args = ap.parse_args()

    ha = branch_hashes(args.a, args.trees)
    hb = branch_hashes(args.b, args.trees)

    excluded = []
    if args.exclude:
        pats = [re.compile(x) for x in args.exclude]
        def drop(k):
            return any(p.search(k[1]) for p in pats)
        excluded = sorted({k for k in set(ha) | set(hb) if drop(k)})
        ha = {k: v for k, v in ha.items() if not drop(k)}
        hb = {k: v for k, v in hb.items() if not drop(k)}

    only_a = sorted(set(ha) - set(hb))
    only_b = sorted(set(hb) - set(ha))
    common = sorted(set(ha) & set(hb))
    diff = [k for k in common if ha[k] != hb[k]]

    print(f"A: {args.a}")
    print(f"B: {args.b}")
    print(f"branches: {len(common)} common, {len(only_a)} only in A, "
          f"{len(only_b)} only in B")
    if excluded:
        print(f"EXCLUDED by --exclude: {len(excluded)}")
        for k in excluded:
            print(f"  EXCLUDED {k[0]}/{k[1]}")
    for k in only_a:
        print(f"  ONLY IN A: {k}")
    for k in only_b:
        print(f"  ONLY IN B: {k}")
    if diff:
        print(f"DIFFERING BRANCHES: {len(diff)}")
        for k in diff:
            print(f"  DIFF {k[0]}/{k[1]}")
    else:
        print("all common branches BIT-IDENTICAL")
    ok = not diff and not only_a and not only_b
    print("RESULT: " + ("IDENTICAL" if ok else "DIFFERENT"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
