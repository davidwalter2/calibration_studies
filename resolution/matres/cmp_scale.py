#!/usr/bin/env python3
"""Side-by-side of named parameters across several rabbit fits.

Written for the one comparison the whole architecture is about: what the
quadratic hit-chi2 term, the mass term and the joint fit each say about the
momentum scale (`bfield_mode0`, the l=1 m=0 uniform-Bz mode) and about the
material amounts.

    python cmp_scale.py --card cards/joint.hdf5 \
        quad=fits/quad/fitresults.hdf5 joint=fits/joint/fitresults.hdf5 \
        --params bfield_mode0 material_tib_support ...
"""
import argparse
import sys

import numpy as np

# 1 whitened parmtype-14 unit = 1 T of RMS |dB| in the tracker; the uniform-Bz
# mode 0 then has dB/B = 1/3.8114e3 per unit (globalfit/STATE.md).
DBB_PER_TESLA = 1.0 / 3.8114


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("fits", nargs="+", metavar="LABEL=PATH")
    p.add_argument("--card", default=None)
    p.add_argument("--params", nargs="*", default=None)
    p.add_argument("--top-material", type=int, default=10)
    a = p.parse_args()

    from rabbit import io_tools

    res = {}
    for spec in a.fits:
        lab, path = spec.split("=", 1)
        fr = io_tools.get_fitresult(path)
        h = fr["parms"].get()
        nm = [str(s) for s in np.array(h.axes["parms"])]
        res[lab] = (nm, np.asarray(h.values(), np.float64),
                    np.sqrt(np.asarray(h.variances(), np.float64)))
    scale = None
    if a.card:
        import h5py

        with h5py.File(a.card, "r") as f:
            g = f["auxiliary/global_index_map"]
            pr = [s.decode() if isinstance(s, bytes) else str(s)
                  for s in g["params"][...]]
            scale = dict(zip(pr, np.asarray(g["scale"][...])))

    names = a.params
    if not names:
        base = res[list(res)[0]][0]
        mats = [n for n in base if n.startswith("material_")]
        _, v0, e0 = res[list(res)[0]]
        order = sorted(mats, key=lambda n: e0[base.index(n)])
        names = ["bfield_mode0"] + order[: a.top_material]

    labs = list(res)
    print(f"{'parameter':<28} " + "  ".join(
        f"{l+' value':>14} {l+' err':>12}" for l in labs))
    for n in names:
        line = f"{n:<28} "
        for l in labs:
            nm, v, e = res[l]
            if n in nm:
                i = nm.index(n)
                line += f"  {v[i]:+14.7g} {e[i]:12.6g}"
            else:
                line += f"  {'-':>14} {'-':>12}"
        print(line)
    print()
    print(f"{'parameter':<28} " + "  ".join(f"{'err/'+l:>12}" for l in labs[1:])
          + "     physical error")
    for n in names:
        nm0, v0, e0 = res[labs[0]]
        if n not in nm0:
            continue
        i0 = nm0.index(n)
        line = f"{n:<28} "
        for l in labs[1:]:
            nm, v, e = res[l]
            line += (f"  {e[nm.index(n)]/max(e0[i0],1e-300):12.4f}"
                     if n in nm else f"  {'-':>12}")
        if scale:
            s = scale.get(n, 1.0)
            if n.startswith("bfield_mode"):
                # card unit IS Tesla of RMS |dB|; mode 0 is the uniform Bz
                extra = "  ".join(
                    f"{l}: {e[nm.index(n)]:.4g} T"
                    + (f" = dB/B {e[nm.index(n)]*DBB_PER_TESLA*1e3:.3g}e-3"
                       if n == "bfield_mode0" else "")
                    for l, (nm, v, e) in res.items() if n in nm)
            else:
                extra = "  ".join(
                    f"{l}: sigma(k)={e[nm.index(n)]/s:.4g}"
                    for l, (nm, v, e) in res.items() if n in nm)
            line += "     " + extra
        print(line)


if __name__ == "__main__":
    main()
