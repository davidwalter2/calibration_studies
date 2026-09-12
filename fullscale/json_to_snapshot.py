#!/usr/bin/env python3
"""Seed a `rabbit_fit.py` run from a point some earlier fit reached.

`rabbit_fit.py --externalPostfit <snapshot>` restarts from a snapshot, and a
snapshot is the smallest thing `Fitter.load_fitresult` accepts: the parameter
names and their values. This writes one from any json that carries `params`
and `fitted` -- a `fit.py` result, or `native_dump.py`'s dump of a rabbit
result -- with the card's own defaults for whatever the json does not name.

The point of it is throughput, not correctness: on the full-statistics card a
COLD scipy `trust-exact` fit is ~38 iterations and ~5.5 h, and most rows of the
certified table already have a stored point within a few sigma of their
minimum. The acceptance test is on where a fit ARRIVES -- value
AND NLL AND EDM -- not on where it started, so a warm start cannot flatter a
result; it can only save iterations. The one row that must stay COLD is the
control, which is testing that the minimiser finds -11.0643 on its own.

    python3 json_to_snapshot.py --card cards/z_V_full.hdf5 \\
        --from results/eng/fit_V_full.json -o results/native/rabbit_SVfull.snapshot.hdf5
"""
import argparse
import json
import os

import h5py

try:  # rabbit writes with a blosc filter; the plugin has to be imported
    import hdf5plugin  # noqa: F401
except ImportError:
    pass
import numpy as np


def card_params(card):
    """(names, defaults) over every unbinned term and the auxiliary bundle.

    Reads only the tiny name/default datasets -- h5py does not touch the
    multi-GB candidate arrays, so this is instant on a 10 GB card.
    """
    names, defaults = [], []
    with h5py.File(card, "r") as f:
        for grp in ("unbinned_terms", "auxiliary"):
            if grp not in f:
                continue
            for term in f[grp]:
                g = f[grp][term]
                key = "params" if "params" in g else None
                if key is None:
                    continue
                nm = [n.decode() if isinstance(n, bytes) else str(n)
                      for n in np.asarray(g[key])]
                dk = "param_defaults" if "param_defaults" in g else "defaults"
                dv = (np.asarray(g[dk], dtype=np.float64).ravel()
                      if dk in g else np.zeros(len(nm)))
                for n, d in zip(nm, dv):
                    if n not in names:
                        names.append(n)
                        defaults.append(float(d))
    return names, np.array(defaults, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True)
    ap.add_argument("--from", dest="src", required=True,
                    help="a json with 'params' and 'fitted'")
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args()

    names, x = card_params(a.card)
    d = json.load(open(a.src))
    src = dict(zip(d["params"], d["fitted"]))
    seeded = [n for n in names if n in src]
    for i, n in enumerate(names):
        if n in src:
            x[i] = float(src[n])
    missing = sorted(set(src) - set(names))

    os.makedirs(os.path.dirname(os.path.abspath(a.output)) or ".", exist_ok=True)
    from rabbit.snapshot import write_snapshot
    write_snapshot(a.output, names, x, meta={"source": os.path.abspath(a.src)})
    print(f"{len(names)} parameters -> {a.output}")
    print(f"  seeded {len(seeded)}: {seeded}")
    if missing:
        print(f"  in the json but not in the card (ignored): {missing}")
    w = max(len(s) for s in names) + 1
    for n, v in zip(names, x):
        print(f"  {n:>{w}s} {v:+16.9f}{'' if n in src else '   (card default)'}")


if __name__ == "__main__":
    main()
