#!/usr/bin/env python3
"""THE STANDARD SELECTION, class by class, on the gen-classified DY sample.

What the two cuts of the standard selection do to each gen class of
`genbkg.py` -- `signal`, `dup`, `unmatched`, `otherdecay`, `nonres` -- applied
IN THE ORDER THEY ARE APPLIED, so the table reads as a cut flow and not as a
set of marginal efficiencies.

usage:
  python3 sel_dytable.py --npz <dy_vtxon_gen_vtx_all.npz> [--chi2 3]
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

import selection  # noqa: E402
import genbkg  # noqa: E402

from wums import logging  # noqa: E402

logger = logging.child_logger(__name__)


def wilson(k, n):
    """Binomial fraction with a symmetric Wilson error, as a string."""
    if n == 0:
        return "0"
    p = k / n
    s = np.sqrt(max(p * (1 - p), 0.0) / n)
    return f"{100*p:6.2f} +- {100*s:.2f} %"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--chi2", type=float, default=3.0,
                    help="the extraction's chi2/ndof cut, applied FIRST so "
                         "the flow matches what a card sees; 0 = off")
    selection.add_args(ap)
    a = ap.parse_args()

    d = genbkg.load_npz(a.npz, max_chi2_ndof=0.0)
    n0 = len(np.asarray(d["vtxz"]))
    cls, names, _aux = genbkg.classify(d)
    logger.info(f"{n0} candidates, classes {list(names)}")

    cfg = selection.from_args(a)
    steps = [("all", np.ones(n0, bool))]
    keep = np.ones(n0, bool)
    if a.chi2 > 0:
        keep = keep & (np.asarray(d["chi2ndof"], float) < a.chi2)
        steps.append((f"chi2/ndof < {a.chi2:g}", keep.copy()))
    # the standard selection, cut by cut, in the order `selection.standard`
    # applies them
    for label, kw in (("Jpsi_vtxok + finite sigma",
                       dict(max_abs_vtxz=0, min_leg_hits=0)),
                      (f"min leg hits >= {cfg['min_leg_hits']}",
                       dict(max_abs_vtxz=0)),
                      (f"|z_v| < {cfg['max_abs_vtxz']:g}", {})):
        m, _s = selection.standard(d, a, n=n0, **kw)
        keep = keep & m
        steps.append((label, keep.copy()))

    print(f"\n{'cut':28s} {'total':>8s} " +
          " ".join(f"{c:>10s}" for c in names))
    prev = None
    for label, m in steps:
        row = f"{label:28s} {int(m.sum()):8d} "
        row += " ".join(f"{int((m & (cls == i)).sum()):10d}"
                        for i in range(len(names)))
        print(row)
        prev = m
    print()
    # efficiency of the WHOLE standard selection per class, against the
    # sample the chi2 cut leaves
    base = steps[1][1] if a.chi2 > 0 else steps[0][1]
    fin = steps[-1][1]
    print(f"{'class':12s} {'before':>8s} {'after':>8s} {'efficiency':>20s}")
    for i, c in enumerate(names):
        nb = int((base & (cls == i)).sum())
        na = int((fin & (cls == i)).sum())
        print(f"{c:12s} {nb:8d} {na:8d} {wilson(na, nb):>20s}")
    nb = int(base.sum())
    na = int(fin.sum())
    print(f"{'ALL':12s} {nb:8d} {na:8d} {wilson(na, nb):>20s}")
    sig = names.index("signal") if "signal" in list(names) else 0
    pb = int((base & (cls == sig)).sum()) / max(nb, 1)
    pa = int((fin & (cls == sig)).sum()) / max(na, 1)
    print(f"\npurity {100*pb:.2f} % -> {100*pa:.2f} %")


if __name__ == "__main__":
    main()
