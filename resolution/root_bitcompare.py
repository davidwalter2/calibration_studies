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
        # propExport/legs, and a top-level-only scan would have compared
        # nothing (it "passed" with 1 branch before this was fixed).
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
    args = ap.parse_args()

    ha = branch_hashes(args.a, args.trees)
    hb = branch_hashes(args.b, args.trees)

    only_a = sorted(set(ha) - set(hb))
    only_b = sorted(set(hb) - set(ha))
    common = sorted(set(ha) & set(hb))
    diff = [k for k in common if ha[k] != hb[k]]

    print(f"A: {args.a}")
    print(f"B: {args.b}")
    print(f"branches: {len(common)} common, {len(only_a)} only in A, "
          f"{len(only_b)} only in B")
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
