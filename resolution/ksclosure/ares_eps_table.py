#!/usr/bin/env python3
"""The K_S momentum scale for each sigma-artefact form, as one table.

`build_ks.sh af_<form>` fits the SAME candidates with the SAME everything
except `a_res`, so the spread across the rows is the `a_res` model uncertainty
and nothing else.  `eps = alpha / <f>_{1/sigma^2}` is the momentum scale
(`ks_table.py` section 6); the last column is what a form COSTS relative to
the one shipped.
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M_PI = 0.13957039
EDM_TOL = 1e-3


def lever(cache):
    with np.load(cache, allow_pickle=True) as d:
        f = np.asarray(d["ks_fmom"], dtype=np.float64)
        w = 1.0 / np.asarray(d["sigma"], dtype=np.float64) ** 2
        return float((w * f).sum() / w.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"))
    ap.add_argument("--cache", default=None)
    ap.add_argument("--forms", nargs="+",
                    default=["ang", "ang_simple", "full_ms_momonly", "mom",
                             "truth", "full_ms_nolam", "full_ms", "full_pop",
                             "kin"])
    ap.add_argument("--ref", default="mom")
    a = ap.parse_args()
    cache = a.cache or os.path.join(a.runs, "kspairs_ares.npz")
    fmean = lever(cache)
    print(f"<f>_1/sigma^2 = {fmean:.4f}   (eps = alpha / <f>)\n")
    got = {}
    for f in a.forms:
        fn = os.path.join(a.runs, "results", f"af_{f}.json")
        if not os.path.exists(fn):
            continue
        with open(fn) as fh:
            j = json.load(fh)
        got[f] = (j["fitted"][0], j["err"][0], j["edmval"])
    if a.ref not in got:
        raise SystemExit(f"the reference form {a.ref} has no result")
    ref = got[a.ref][0] / fmean
    ares = dict(np.load(cache))
    hdr = (f"{'a_res form':22s} {'median a_res':>13s} {'alpha [1e-3]':>14s} "
           f"{'eps [1e-3]':>12s} {'eps - eps(%s)' % a.ref:>16s} {'EDM':>10s}")
    print(hdr)
    print("-" * len(hdr))
    for f in a.forms:
        if f not in got:
            continue
        al, er, edm = got[f]
        eps = al / fmean
        med = float(np.median(ares["ares_" + f])) if "ares_" + f in ares else np.nan
        flag = "" if edm < EDM_TOL else "  NOT CONVERGED"
        print(f"{f:22s} {med:13.5f} {al:+9.4f} +-{er:.4f} {eps:+12.4f} "
              f"{eps-ref:+16.4f} {edm:10.1e}{flag}")
    print(f"\nstatistical error on eps: {got[a.ref][1]/fmean:.4f} x 1e-3")


if __name__ == "__main__":
    main()
