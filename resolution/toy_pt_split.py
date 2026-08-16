#!/usr/bin/env python3
"""Independent-subsample check for the toy pT scan.

The sims are seeded deterministically (`RandomNumberGeneratorService.generator.
initialSeed = 1`), and a 400k job's first 100k events are BIT-IDENTICAL to a
100k job's (verified event-for-event).  The 400k result is therefore a strict
superset of the published 100k one, NOT an independent measurement of it, and
quoting "they agree" would be circular.

This splits each 400k sample into the first 100k (= the published sample) and
the disjoint last 300k, and quotes the closure on each.  The two ARE
independent, so their difference is a genuine consistency test with a
well-defined error.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "cleanprop"))
import fisher_norm as fn
import geom_closure as gc
import toy_pt_scan as tp

UHEAD, IHEAD = fn.UHEAD, fn.IHEAD


def sub(sim, lo, hi):
    out = {}
    for k, v in sim.items():
        out[k] = v[lo:hi] if isinstance(v, np.ndarray) and v.ndim == 2 else v
    out["valid"] = sim["valid"][lo:hi]
    out["ntot"] = hi - lo
    return out


def main():
    tags = sys.argv[1:] or ["pt3_K1", "pt3_K4", "pt3_K16"]
    print("First 100k (== the published sample, bit-identical) vs the disjoint "
          "last 300k.\nFisher normalization, mean over planes.\n")
    print(f"  {'config':<12s} {'func':<5s} {'sample':<10s} " +
          "".join(f"{'u=' + str(u):>20s}" for u in UHEAD))
    for tag in tags:
        legs, sim = tp.legs_of(tag), tp.sim_of(tag)
        n = sim["valid"].shape[0]
        for func in ("qop", "locx"):
            sc = fn.plane_scales(legs, func, tag=tag)
            got = {}
            for name, lo, hi in (("first100k", 0, 100000),
                                 ("last300k", 100000, n),
                                 ("all", 0, n)):
                r = gc.closure_rows(legs, sub(sim, lo, hi), func, sc["sF"])
                got[name] = (r[0].mean(axis=0)[IHEAD], r[4][IHEAD])
                print(f"  {tag:<12s} {func:<5s} {name:<10s} " +
                      "".join(f"{m:+.5f}+-{e:.5f}".rjust(20)
                              for m, e in zip(*got[name])))
            d = got["last300k"][0] - got["first100k"][0]
            e = np.hypot(got["last300k"][1], got["first100k"][1])
            print(f"  {'':<12s} {'':<5s} {'diff':<10s} " +
                  "".join(f"{di:+.5f}({di/ei:+.1f}s)".rjust(20)
                          for di, ei in zip(d, e)))
    print()


if __name__ == "__main__":
    main()
