#!/usr/bin/env python3
"""Dump every `rabbit_fit.py` result in a directory as one table.

Value AND NLL AND EDM for each, plus the free-parameter values,
straight out of the rabbit output file. No fit is run.
"""
import argparse
import glob
import json
import os

import numpy as np


def read(path):
    from rabbit import io_tools

    fr = io_tools.get_fitresult(path)
    h = fr["parms"].get()
    names = [str(s) for s in h.axes[0]]
    vals = np.asarray(h.values(), dtype=np.float64)
    var = h.variances()
    errs = np.sqrt(np.asarray(var, dtype=np.float64)) if var is not None else np.full(vals.shape, np.nan)
    out = {"path": path, "params": names, "fitted": vals.tolist(), "err": errs.tolist()}
    for k in ("edmval", "nllvalreduced", "nllvalfull", "ndfsat"):
        if k in fr:
            try:
                out[k] = float(np.asarray(fr[k]))
            except Exception:
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("-o", "--outdir", default=None,
                    help="also write one json per result here")
    a = ap.parse_args()
    files = []
    for p in a.paths:
        files += sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p]
    files = [f for f in files if f.endswith(".hdf5") and "snapshot" not in f]
    hdr = f"{'result':26s} {'m_Z':>18s} {'Gamma_Z':>18s} {'NLL':>20s} {'EDM':>11s}"
    print(hdr)
    print("-" * len(hdr))
    for f in files:
        try:
            d = read(f)
        except Exception as ex:
            print(f"{os.path.basename(f):26s}  (unreadable: {ex})")
            continue
        p, v, e = d["params"], d["fitted"], d["err"]
        g = lambda q: (v[p.index(q)], e[p.index(q)]) if q in p else (np.nan, np.nan)
        m, gz = g("m_Z"), g("Gamma_Z")
        nll = d.get("nllvalreduced", float("nan"))
        edm = d.get("edmval", float("nan"))
        print(f"{os.path.basename(f)[:-5]:26s} {m[0]:+9.3f} +-{m[1]:6.3f} "
              f"{gz[0]:+9.3f} +-{gz[1]:6.3f} {nll:20.6f} {edm:11.3e}")
        if a.outdir:
            os.makedirs(a.outdir, exist_ok=True)
            o = os.path.join(a.outdir, os.path.basename(f)[:-5] + ".json")
            json.dump(d, open(o, "w"), indent=1)


if __name__ == "__main__":
    main()
