#!/usr/bin/env python3
"""Read, add or strip a key in an unbinned term's stored `config` JSON.

WHY THIS EXISTS.  A card records each unbinned term's structural description as
a JSON blob, and `read_unbinned_terms_from_h5` splats it straight into the term
constructor (`**cfg`).  So a card built by a NEWER rabbit than the one the fit
runs on dies at load with

    TypeError: MassCFTerm.__init__() got an unexpected keyword argument '...'

This is the failure mode of a card written by a checkout carrying a keyword
such as `corr_a_max` and then submitted against a staged checkout that does
not have it.

Two uses:

  --show                 print the config keys/values of every term
  --set corr_a_max=0.0   add or overwrite a key (writes IN PLACE unless -o)
  --strip corr_a_max     remove a key

`-o OUT` copies the card first, which is what the code-equivalence triple
needs: the SAME candidates, differing only by the presence of the keyword.
"""
import argparse
import json
import os
import shutil

import h5py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("card")
    ap.add_argument("-o", "--output", help="copy the card first, edit the copy")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=JSONVALUE")
    ap.add_argument("--strip", action="append", default=[], metavar="KEY")
    ap.add_argument("--term", default=None, help="only this term (default: all)")
    a = ap.parse_args()

    path = a.card
    if a.output:
        if os.path.exists(a.output):
            raise SystemExit(f"refusing to overwrite {a.output}")
        print(f"copying {a.card} -> {a.output}")
        shutil.copyfile(a.card, a.output)
        path = a.output

    mode = "r" if (a.show and not a.set and not a.strip) else "r+"
    with h5py.File(path, mode) as f:
        g = f.get("unbinned_terms")
        if g is None:
            raise SystemExit("card has no unbinned_terms group")
        for name in g:
            if a.term and name != a.term:
                continue
            dset = g[name]["config"]
            raw = dset[...]
            cfg = json.loads(raw[0].decode() if isinstance(raw[0], bytes) else str(raw[0]))
            if a.show:
                print(f"--- term '{name}'")
                for k, v in cfg.items():
                    if isinstance(v, (int, float, str, type(None), bool)):
                        print(f"    {k} = {v!r}")
                    else:
                        print(f"    {k} : <{type(v).__name__}>")
            changed = False
            for kv in a.set:
                k, _, v = kv.partition("=")
                cfg[k] = json.loads(v)
                print(f"[{name}] set {k} = {cfg[k]!r}")
                changed = True
            for k in a.strip:
                if k in cfg:
                    print(f"[{name}] strip {k} (was {cfg[k]!r})")
                    cfg.pop(k)
                    changed = True
                else:
                    print(f"[{name}] {k} not present")
            if changed:
                # the dataset is a fixed-length variable string array; rewrite it
                dt = dset.dtype
                del g[name]["config"]
                g[name].create_dataset("config", data=[json.dumps(cfg)], dtype=dt)
    print("done")


if __name__ == "__main__":
    main()
