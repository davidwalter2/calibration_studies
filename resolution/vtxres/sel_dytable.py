#!/usr/bin/env python3
"""THE STANDARD SELECTION, class by class, on the gen-classified DY sample.

What the two cuts of the standard selection do to each gen class of
`genbkg.py` -- `signal`, `dup`, `unmatched`, `otherdecay`, `nonres` -- applied
IN THE ORDER THEY ARE APPLIED, so the table reads as a cut flow and not as a
set of marginal efficiencies.

The chi2/ndof cut is IN the flow, in its place: it is part of the standard
selection (`resolution/selection.py`), applied after the leg-hit minimum and
before `|z_v| < 5`.  `--max-chi2-ndof 0` takes it out again.

usage:
  python3 sel_dytable.py --npz <dy_vtxon_gen_vtx_all.npz> [--max-chi2-ndof 0]
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
    selection.add_args(ap)
    a = ap.parse_args()

    d = genbkg.load_npz(a.npz, max_chi2_ndof=0.0)
    n0 = len(np.asarray(d["vtxz"]))
    cls, names, _aux = genbkg.classify(d)
    logger.info(f"{n0} candidates, classes {list(names)}")

    cfg = selection.from_args(a)
    steps = [("all", np.ones(n0, bool))]
    keep = np.ones(n0, bool)
    # the standard selection, cut by cut, IN THE ORDER `selection.standard`
    # applies them -- so the table reads as a cut flow and not as a set of
    # marginal efficiencies.  `chi2/ndof` is one of its cuts now, in its place.
    flow = [("Jpsi_vtxok + finite sigma",
             dict(min_leg_hits=0, max_chi2_ndof=0, max_abs_vtxz=0)),
            (f"min leg hits >= {cfg['min_leg_hits']}",
             dict(max_chi2_ndof=0, max_abs_vtxz=0)),
            (f"chi2/ndof < {cfg['max_chi2_ndof']:g}", dict(max_abs_vtxz=0)),
            (f"|z_v| < {cfg['max_abs_vtxz']:g}", {})]
    if cfg["max_chi2_ndof"] <= 0:
        flow = [f for f in flow if not f[0].startswith("chi2/ndof")]
    for label, kw in flow:
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
    # efficiency of the WHOLE standard selection per class, against everything
    base = steps[0][1]
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
