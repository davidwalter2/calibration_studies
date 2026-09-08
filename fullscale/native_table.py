#!/usr/bin/env python3
"""The scipy-vs-native table, straight out of the `fit.py` json files.

Each row is one `(engine, method)` on the SAME card from the SAME start, so
the only thing that differs between two rows is who evaluates the objective
and who chooses the steps. The parameter columns are what makes it a
comparison rather than a benchmark: a faster minimiser that lands somewhere
else has not done the job.

    python3 native_table.py results/native/fit_n300k_*.json --ref host_trust-exact
"""
import argparse
import json
import os

import numpy as np


def load(paths):
    out = []
    for p in paths:
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception as ex:
            print(f"  (skipping {p}: {ex})")
            continue
        if "fitted" not in d:
            continue
        d["_path"] = p
        d["_key"] = f"{d.get('engine','?')}_{d.get('method','?')}"
        out.append(d)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("json", nargs="+")
    p.add_argument("--ref", default=None, help="engine_method of the reference row")
    p.add_argument("--params", nargs="*", default=None)
    args = p.parse_args()

    rows = load(args.json)
    if not rows:
        raise SystemExit("no fit json with a 'fitted' block")
    ref = None
    if args.ref:
        ref = next((r for r in rows if r["_key"] == args.ref), None)
    if ref is None:
        ref = next((r for r in rows if r["_key"].startswith("host_trust-exact")), rows[0])
    print(f"card {os.path.basename(ref['card'])}, n = {ref['n']}, "
          f"{len(ref['params'])} free: {ref['params']}")
    print(f"reference row: {ref['_key']}\n")

    hdr = (
        f"{'engine':>7s} {'method':>16s} {'it':>4s} {'wall':>9s} {'x':>6s} "
        f"{'t/it':>8s} {'grad':>5s} {'hvp':>6s} {'GPU%':>6s} {'p90':>6s} "
        f"{'GPUmem':>8s} {'RSS':>7s} {'dNLL':>10s} {'max drel':>10s}"
    )
    print(hdr)
    print("-" * len(hdr))
    tref = ref.get("t_fit", float("nan"))
    for r in sorted(rows, key=lambda r: (r["_key"] != ref["_key"], r["_key"])):
        m = r.get("minimizer", {})
        g = m.get("gpu", {})
        nc = m.get("ncall", {})
        t = r.get("t_fit", float("nan"))
        x = np.asarray(r["fitted"])
        xr = np.asarray(ref["fitted"])
        drel = (
            np.max(np.abs(x - xr) / np.maximum(np.abs(xr), 1e-12))
            if x.shape == xr.shape
            else float("nan")
        )
        dnll = r["nll"] - ref["nll"]
        print(
            f"{r.get('engine','?'):>7s} {r.get('method','?'):>16s} "
            f"{r.get('nit',-1):4d} {t:8.1f}s {tref/max(t,1e-9):5.1f}x "
            f"{t/max(r.get('nit',1),1):7.2f}s "
            f"{nc.get('grad', 0):5d} {nc.get('hessp', 0):6d} "
            f"{g.get('util_mean', float('nan')):6.1f} "
            f"{g.get('util_p90', float('nan')):6.1f} "
            f"{r.get('minimizer',{}).get('gpu_peak_gb', float('nan')):7.2f}G "
            f"{m.get('rss_gb', r.get('rss_gb', float('nan'))):6.1f}G "
            f"{dnll:+10.2e} {drel:10.2e}"
        )

    print("\nfitted parameters")
    names = ref["params"]
    keep = names if args.params is None else [n for n in names if n in args.params]
    w = max(len(s) for s in keep) + 1
    print(f"  {'row':>24s} " + " ".join(f"{n:>14s}" for n in keep))
    for r in rows:
        v = dict(zip(r["params"], r["fitted"]))
        print(f"  {r['_key']:>24s} " + " ".join(f"{v[n]:14.8f}" for n in keep))
    print(f"  {'(truth)':>24s} " + " ".join(
        f"{ref.get('truth', {}).get(n, float('nan')):14.8f}" for n in keep))


if __name__ == "__main__":
    main()
