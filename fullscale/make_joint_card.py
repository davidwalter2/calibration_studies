#!/usr/bin/env python3
"""PHASE 2 -- the joint J/psi + Z card over ONE set of calibration parameters.

The Z channel cannot measure `m_Z`: `m_Z` and a momentum scale are exactly
degenerate in a single-resonance fit. The J/psi fixes the scale, and in this
card it does so THROUGH THE PHYSICS rather than through a free `alpha`: the
J/psi term carries a delta kernel at the PDG mass and no scale parameter at
all, so the only way its candidates can move is through the field modes and the
material amounts -- exactly the parameters the Z term also depends on. That is
what makes the transfer a calibration rather than a fitted offset.

Three terms over one parameter vector:

* the **quadratic hit-chi2 term** on parmtypes 14 (50 field modes) and 15 (42
  material groups), summed over the J/psi and the DY productions
  (`globalfit/extract.py --no-mass`), written the way
  `globalfit/make_global_term.py` writes it -- same `chi2 -> NLL` factor of
  1/2, same whitening, same `D_card = -dm/dtheta` sign convention;
* the **J/psi mass term**: delta kernel at `MJPSI`, `scale_param=None`, both
  corrections, and the sparse `D` on the 92 parameters;
* the **Z mass term** of `make_card.py`: the `ZGammaLineshape` provider with
  the banded FSR fold, the acceptance, a floated 5-term `K(m)`, `m_Z` and
  `Gamma_Z` as POIs, both corrections, and the same sparse `D`.

The resolution scales `k_*` stay FIXED at the MC truth in phase 2; phase 3
replaces them with the parmtype-15 amounts themselves (`MaterialCFTerm`).

The `f_ang` caveat: `jpsimc_20M_260905` predates `Jpsi_covrefmom`, so the
J/psi leg's Jensen `s^2` cannot be made truth-free from the candidate itself.
`--jpsi-fang` supplies the MC-measured value (gun 0.086 / data 0.106); the DY
leg uses its own per-candidate `Jpsi_fang`.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.join(os.path.dirname(HERE), "resolution")
_GF = os.path.join(_RES, "globalfit")
for _p in (HERE, _RES, _GF):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MJPSI = 3.0969


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jpsi-pairs", required=True)
    p.add_argument("--z-pairs", required=True)
    p.add_argument("--quad", nargs="+", required=True,
                   help="one or more globalfit/extract.py --no-mass outputs; "
                        "they are SUMMED (the parameter map is bit-identical "
                        "across the two productions, verified)")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--groups", default=None, help="materialGroups tier file")
    p.add_argument("--coeffs", default=None, help="mode dump for --whiten")
    p.add_argument("--whiten", action="store_true", default=True)
    p.add_argument("--no-whiten", dest="whiten", action="store_false")
    p.add_argument("--fsr", default=None)
    p.add_argument("--acc", default=None)
    p.add_argument("--shape", type=int, default=5)
    p.add_argument("--jpsi-maxn", type=int, default=0)
    p.add_argument("--z-maxn", type=int, default=0)
    p.add_argument("--jpsi-window", type=float, default=0.35,
                   help="|m_obs - MJPSI| halfwidth [GeV]")
    p.add_argument("--jpsi-fang", type=float, default=0.096,
                   help="the angular share of the mass variance on the J/psi "
                        "leg, which v1 cannot supply per candidate. The MC "
                        "measurements are 0.086 (gun) and 0.106 (data); the "
                        "midpoint is the default and the spread is a "
                        "systematic to scan.")
    p.add_argument("--field-prior", type=float, default=0.0)
    p.add_argument("--material-prior-scale", type=float, default=1.0)
    p.add_argument("--max-chi2-ndof", type=float, default=3.0)
    p.add_argument("--max-sigma-rel", type=float, default=0.10)
    p.add_argument("--chunk", type=int, default=32768)
    p.add_argument("--fit-upsample-z", type=int, default=4)
    p.add_argument("--fit-upsample-jpsi", type=int, default=1,
                   help="the J/psi window is +-0.35 GeV, i.e. a handful of "
                        "oscillation periods across the tau grid, so it does "
                        "not need the Z's 4x")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args(argv)


def sum_quadratic(paths, log=print):
    """G, K and the sandwich meat, summed over productions."""
    G = K = J = None
    cat = None
    tot = 0
    for f in paths:
        d = np.load(f)
        if cat is None:
            cat = {k: d[k] for k in ("fitidx", "parmtype", "subidx", "g2f",
                                     "nglobal")}
        else:
            for k in ("fitidx", "parmtype", "subidx"):
                if not np.array_equal(cat[k], d[k]):
                    raise SystemExit(
                        f"{f}: the parameter map differs from the first "
                        f"extraction on `{k}`; the two cannot be summed")
        G = d["grad"] if G is None else G + d["grad"]
        K = d["hess"] if K is None else K + d["hess"]
        if "jsand" in d.files:
            J = d["jsand"] if J is None else J + d["jsand"]
        tot += int(d["ncand_quadratic"])
        log(f"  {os.path.basename(f)}: {int(d['ncand_quadratic'])} candidates, "
            f"{int(d['ncand_chi2cut'])} cut")
    log(f"  quadratic term: {tot} candidates, {len(G)} parameters")
    return G, K, J, cat, tot


def main():
    args = parse_args()
    print(f"[make_joint_card] {args.output}")
    G, K, J, cat, nquad = sum_quadratic(args.quad)
    print("  (the rest of the assembly is phase 2 and is written against the "
          "phase-1 pieces: make_card.build for the Z term, the same builder "
          "with a delta kernel for the J/psi term, and "
          "globalfit/make_global_term for the quadratic block.)")
    raise SystemExit(
        "make_joint_card.py is the phase-2 entry point and is not complete; "
        "phase 1 is Z-alone. What is already usable here is `sum_quadratic`, "
        "which checks that two extractions share a parameter map before "
        "adding them.")


if __name__ == "__main__":
    main()
