#!/usr/bin/env python3
"""Append a tail cache to a pairs cache, without redoing the whole pass.

A production that finishes after the cache was built leaves the cache short by
whole tasks. Rebuilding is hours; the tasks that were missed are cheap. This
concatenates, checks that the two really are disjoint (on `run`/`lumi`/`event`,
which is why those columns are cached), and refuses if they are not.

usage: append_pairs.py --base a.npz --tail b.npz -o merged.npz
"""
import argparse
import os

import numpy as np


def key_of(d):
    r = np.asarray(d["run"], np.int64)
    l = np.asarray(d["lumi"], np.int64)
    e = np.asarray(d["event"], np.int64)
    # event needs 27 bits and lumi 17 on this sample; use a structured view
    # rather than a hand-rolled shift, so a wider sample cannot silently alias
    return np.rec.fromarrays([r, l, e], names="r,l,e")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--tail", required=True, nargs="+")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--allow-overlap", action="store_true",
                   help="drop the overlapping rows instead of refusing")
    a = p.parse_args()

    parts = [np.load(a.base, allow_pickle=True)]
    for t in a.tail:
        parts.append(np.load(t, allow_pickle=True))
    keys = set(parts[0].files)
    for i, d in enumerate(parts[1:], 1):
        if set(d.files) != keys:
            raise SystemExit(
                f"{a.tail[i-1]} has a different column set: "
                f"only in base {sorted(keys - set(d.files))}, "
                f"only in tail {sorted(set(d.files) - keys)}")
        if not np.array_equal(parts[0]["tgrid"], d["tgrid"]):
            raise SystemExit(f"{a.tail[i-1]} was built on a different tau grid")
        if str(parts[0]["cf_model"]) != str(d["cf_model"]):
            raise SystemExit(f"{a.tail[i-1]} was built with a different CF model")

    have = key_of(parts[0])
    keep = [np.ones(len(have), bool)]
    for i, d in enumerate(parts[1:], 1):
        k = key_of(d)
        dup = np.isin(k, have)
        n = int(dup.sum())
        if n and not a.allow_overlap:
            raise SystemExit(
                f"{a.tail[i-1]} shares {n} of its {len(k)} candidates with the "
                f"base cache -- the tail is not disjoint. Re-derive the task "
                f"list, or pass --allow-overlap to drop them.")
        keep.append(~dup)
        have = np.concatenate([have, k[~dup]])
        print(f"  {os.path.basename(a.tail[i-1])}: {len(k)} candidates, "
              f"{n} already present, {int((~dup).sum())} appended")

    out = {}
    for k in sorted(keys):
        v0 = parts[0][k]
        if k in ("tgrid", "cf_source", "cf_model", "rad_model",
                 "ioni_sign_fixed", "ioni_charge_signed", "hitclsnames",
                 "jac_globalidx", "jac_parmtype", "jac_subidx"):
            out[k] = v0
            continue
        out[k] = np.concatenate([p[k][m] for p, m in zip(parts, keep)], axis=0)
    n0, n1 = len(parts[0]["z"]), len(out["z"])
    np.savez_compressed(a.output, **out)
    print(f"wrote {a.output}: {n0} + {n1 - n0} = {n1} candidates "
          f"({os.path.getsize(a.output)/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
