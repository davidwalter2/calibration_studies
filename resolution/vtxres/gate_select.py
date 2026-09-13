#!/usr/bin/env python3
"""Dump what `fullscale/make_card.select` SELECTS, so two code versions can be
compared on the selection alone.

The J/psi leg of the joint card does not go through `make_card.build`: it calls
`make_card.select` and assembles its own term (`make_joint_card.build_jpsi`).
The selection IS therefore the whole of what this refactor could have changed
there, and comparing the index array is a sharper gate than comparing a card --
it is the same test with none of the 10-minute term assembly.

usage:
  python3 gate_select.py --base <tree with fullscale/> --pairs <cache.npz> \
      --channel jpsi --mref 3.0969 --window 2.9469 3.2469 -o out.npz
"""
import argparse
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="the tree whose fullscale/make_card.py to use")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="-- then any make_card.py arguments")
    a = ap.parse_args()

    for p in (os.path.join(a.base, "fullscale"),
              os.path.join(a.base, "resolution"),
              os.path.join(a.base, "resolution", "globalfit")):
        sys.path.insert(0, p)
    import make_card

    argv = ["--pairs", a.pairs, "-o", "/dev/null"] + \
           [x for x in a.rest if x != "--"]
    args = make_card.parse_args(argv)
    d = np.load(a.pairs, allow_pickle=True)
    idx, m_all, sig_all, w_all, srel = make_card.select(d, args, print)
    out = {"idx": np.asarray(idx, np.int64),
           "m": np.asarray(m_all, np.float64)[idx],
           "sigma": np.asarray(sig_all, np.float64)[idx],
           "srel": np.asarray(srel, np.float64)[idx],
           "w": (np.asarray(w_all, np.float64)[idx]
                 if w_all is not None else np.zeros(0)),
           "make_card": np.array(os.path.abspath(make_card.__file__))}
    np.savez_compressed(a.output, **out)
    print(f"wrote {a.output}: {len(idx)} selected")


if __name__ == "__main__":
    main()
