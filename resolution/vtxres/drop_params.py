#!/usr/bin/env python3
"""Write a Fisher npz with some parameters removed from H, J and G.

WHY.  `efficiency.py` inverts `H + P` over the WHOLE parameter set, so a
direction the term carries no information on -- a hit class whose curvature at
MC truth is NEGATIVE, held up only by its prior -- makes `(H+P)^-1` large along
that direction and every OTHER parameter's MARGINAL error inherits it.  On the
J/psi gun that direction is `hitres_str_N5_hi`: it carries 0.27 % of
`sigma_v^2` per candidate, `H` has an eigenvalue of -3 to -9 along it in every
arm, and how close `H + P` then comes to singular is a property of the sample,
not of the physics.  Removing the parameter is the only treatment that is
IDENTICAL for every arm and every regime; it is not a scale, a bound or a clip
on anything measured.

usage:
  drop_params.py -i fisherHJ.npz -o fisherHJ_nodeg.npz --drop hitres_str_N5_hi
"""
import argparse
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--drop", nargs="+", required=True)
    a = p.parse_args()
    d = np.load(a.input, allow_pickle=True)
    params = [str(s) for s in d["params"]]
    keep = np.array([q not in a.drop for q in params])
    missing = [q for q in a.drop if q not in params]
    if missing:
        raise SystemExit(f"not in the parameter set: {missing}")
    out = {"params": np.array([q for q, k in zip(params, keep) if k],
                              dtype=object)}
    for k in d.files:
        if k == "params":
            continue
        v = d[k]
        if k.startswith(("H_", "J_")) and v.ndim == 2 and v.shape[0] == len(params):
            out[k] = np.asarray(v, float)[np.ix_(keep, keep)]
        elif k.startswith("G_") and v.ndim == 2 and v.shape[1] == len(params):
            out[k] = np.asarray(v, float)[:, keep]
        elif k == "param_units" and v.shape == (len(params),):
            out[k] = np.asarray(v)[keep]
        else:
            out[k] = v
    np.savez_compressed(a.output, **out)
    print(f"dropped {a.drop}: {len(params)} -> {int(keep.sum())} parameters "
          f"-> {a.output}")


if __name__ == "__main__":
    main()
