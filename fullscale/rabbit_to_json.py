#!/usr/bin/env python3
"""Turn a `rabbit_fit.py` result into the json `fit.py --start-from` reads.

The division of labour is: `rabbit_fit.py` finds the
minimum (with the EDM, the termination convention and the snapshots), and the
standalone driver is still the only thing that computes the **sandwich**
covariance -- which is not optional here, since the MiNNLO weights alone cost
a flat x1.109 on every error. So:

    rabbit_fit.py ... --outname rabbit_f380.hdf5          # the minimum
    python3 rabbit_to_json.py results/rabbit_f380.hdf5 -o results/f380_start.json
    python3 fit.py --card ... --no-fit --start-from results/f380_start.json \
        --engine device                                    # Hessian + sandwich

`--no-fit` evaluates at the seeded point, so the sandwich and the covariance
are the ones at rabbit's minimum, not at a point some other minimiser stopped
at. `--params` restricts what is carried over, which matters when the two
parameter sets differ (rabbit's vector holds the frozen ones too).

There is nothing in `io_tools` that does this: it returns the fit result as
boost histograms, and `--start-from` wants names and values.
"""
import argparse
import json
import os
import sys

import numpy as np


def read_rabbit(path, key="results"):
    """``(names, values, errors)`` from a rabbit fit output file."""
    from rabbit import io_tools

    fitresult = io_tools.get_fitresult(path)
    h = fitresult["parms"].get()
    names = [str(s) for s in h.axes[0]]
    values = np.asarray(h.values(), dtype=np.float64)
    var = h.variances()
    errors = (
        np.sqrt(np.asarray(var, dtype=np.float64))
        if var is not None
        else np.full(values.shape, np.nan)
    )
    extra = {}
    for k in ("edmval", "nllvalreduced", "ndfsat"):
        if k in fitresult:
            try:
                extra[k] = float(np.asarray(fitresult[k]))
            except Exception:
                pass
    return names, values, errors, extra


def main():
    p = argparse.ArgumentParser()
    p.add_argument("result", help="the rabbit_fit.py output hdf5")
    p.add_argument("-o", "--output", required=True)
    p.add_argument(
        "--params",
        nargs="*",
        default=None,
        help="only carry these parameters over (default: all)",
    )
    p.add_argument("--print", action="store_true", help="also print the values")
    args = p.parse_args()

    names, values, errors, extra = read_rabbit(args.result)
    if args.params:
        keep = [i for i, n in enumerate(names) if n in set(args.params)]
        missing = set(args.params) - set(names)
        if missing:
            raise SystemExit(f"not in the result: {sorted(missing)}")
        names = [names[i] for i in keep]
        values = values[keep]
        errors = errors[keep]

    out = {
        "params": names,
        "fitted": values.tolist(),
        "err": errors.tolist(),
        "source": os.path.abspath(args.result),
        **extra,
    }
    d = os.path.dirname(os.path.abspath(args.output))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=1)
    print(f"{len(names)} parameters -> {args.output}"
          + (f"   (EDM {extra['edmval']:.4g})" if "edmval" in extra else ""))
    if args.print:
        w = max(len(s) for s in names) + 1
        for n, v, e in zip(names, values, errors):
            print(f"  {n:>{w}s} {v:+16.9f} +- {e:.9f}")


if __name__ == "__main__":
    main()
