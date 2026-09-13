#!/usr/bin/env python3
"""The gen-SIGNAL mask of an `extract_vtx.py` npz, written once for the cards.

`genbkg.classify` labels every candidate `signal` / `dup` / `unmatched` /
`otherdecay` / `nonres` from the maker's gen provenance.  A study of what a
SELECTION does to the terms has to be made on gen signal alone, otherwise the
cut's effect on the fitted parameters is a mixture of what it does to the
physics and what it does to the background it also removes -- and on DY the
chi2 cut takes 0.65 of the background as well as 1.0 % of the signal.

The mask is written to its own npz and applied by `make_vtx_card.py
--keep-mask`, so every arm of a comparison sits on exactly the same truth and
the mask is computed once rather than once per card.

usage:
  python3 gensig_mask.py --npz <dy_vtx.npz> -o <dy_vtx_gensig.npz>
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.dirname(_HERE)
for _p in (_HERE, _RES, os.path.join(_RES, "matres")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import genbkg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--classes", nargs="+", default=["signal"],
                    help="the gen classes to KEEP (default: signal)")
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args()

    d = genbkg.load_npz(a.npz, max_chi2_ndof=0.0)
    n = len(np.asarray(d["vtxz"]))
    cls, names, _ = genbkg.classify(d)
    names = list(names)
    want = [names.index(c) for c in a.classes]
    keep = np.isin(cls, want)
    print(f"{a.npz}: {n} candidates")
    for i, c in enumerate(names):
        k = int((cls == i).sum())
        print(f"  {c:12s} {k:8d}  ({100.0*k/max(n,1):6.3f} %)"
              + ("   KEPT" if i in want else ""))
    np.savez_compressed(a.output, keep=keep,
                        label=np.array("gen " + "+".join(a.classes)),
                        source=np.array(os.path.abspath(a.npz)))
    print(f"wrote {a.output}: {int(keep.sum())}/{n} kept "
          f"({100.0*keep.mean():.3f} %)")


if __name__ == "__main__":
    main()
