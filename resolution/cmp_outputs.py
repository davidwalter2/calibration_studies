#!/usr/bin/env python3
"""BIT-IDENTITY of two outputs -- two npz, or two rabbit hdf5 cards.

The gate a refactor has to pass: the same inputs and the same settings through
the old code and the new one must produce the same numbers, not merely
compatible ones.  Compares every array key by key and reports the first
difference rather than a summary, because "close" is not the claim.

usage:
  python3 cmp_outputs.py a.npz b.npz
  python3 cmp_outputs.py a.hdf5 b.hdf5 [--skip-keys ...]
"""
import argparse
import sys

import numpy as np


def load(path):
    if path.endswith((".hdf5", ".h5")):
        import h5py
        out, fh = {}, h5py.File(path, "r")

        def walk(name, obj):
            if isinstance(obj, h5py.Dataset):
                out[name] = obj[()]
        fh.visititems(walk)
        return out, fh
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}, d


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--skip-keys", nargs="*", default=[],
                   help="keys whose difference is expected and harmless "
                        "(a provenance string carrying the argv, say)")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()

    A, _ha = load(a.a)
    B, _hb = load(a.b)
    ka, kb = set(A), set(B)
    bad = []
    if ka != kb:
        bad.append(f"key sets differ: only in A {sorted(ka - kb)}, "
                   f"only in B {sorted(kb - ka)}")
    nsame = ndiff = 0
    for k in sorted(ka & kb):
        if k in a.skip_keys:
            continue
        if same(A[k], B[k]):
            nsame += 1
        else:
            ndiff += 1
            x, y = np.asarray(A[k]), np.asarray(B[k])
            det = f"shape {x.shape} vs {y.shape}, dtype {x.dtype} vs {y.dtype}"
            if x.shape == y.shape and x.dtype.kind in "fiu":
                with np.errstate(invalid="ignore"):
                    det += f", max|a-b| = {np.nanmax(np.abs(x.astype(float) - y.astype(float))):.6g}"
            bad.append(f"{k}: {det}")
    print(f"{a.a}\n{a.b}")
    print(f"  {nsame} keys IDENTICAL, {ndiff} differ, "
          f"{len(a.skip_keys)} skipped")
    for m in bad[:20]:
        print("  !! " + m)
    if bad:
        sys.exit(1)
    print("  BIT-IDENTICAL")


if __name__ == "__main__":
    main()
